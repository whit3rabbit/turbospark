//! The macOS Core ML boundary. Every `unsafe` block in this crate lives in
//! this file. The rules it follows (see `AGENTS.md`):
//!
//! - Every Objective-C call that can raise runs inside `objc2::exception::catch`,
//!   so an NSException becomes `CoreMlError::Exception` instead of aborting.
//! - Every API newer than macOS 13 is gated by `supports` first. objc2 looks
//!   selectors up at runtime; an unchecked call on an older OS raises.
//! - Every prediction runs in its own autorelease pool. Without one, the
//!   autoreleased feature providers and arrays from a decode loop pile up
//!   for the life of the calling thread (same reason as
//!   `gpu::autorelease_pool`).
//! - Buffers are bounds-checked against their declared strides before any
//!   copy (`tensor::strided_extent`).

use std::collections::HashMap;
use std::fs;
use std::panic::AssertUnwindSafe;
use std::path::{Path, PathBuf};
use std::ptr::NonNull;
use std::sync::{mpsc, Mutex};

use block2::RcBlock;
use objc2::rc::{autoreleasepool, Retained};
use objc2::runtime::{AnyObject, ProtocolObject};
use objc2::{available, AllocAnyThread};
use objc2_core_ml::{
    MLComputeUnits, MLDictionaryFeatureProvider, MLFeatureDescription, MLFeatureProvider,
    MLFeatureType, MLFeatureValue, MLModel, MLModelConfiguration, MLMultiArray,
    MLMultiArrayDataType, MLMultiArrayShapeConstraint, MLMultiArrayShapeConstraintType, MLState,
};
use objc2_foundation::{NSArray, NSDictionary, NSError, NSNumber, NSProcessInfo, NSString, NSURL};

use crate::cache::{cache_key, compiled_path_for, copy_dir, is_cache_entry_for};
use crate::error::{CoreMlError, Result};
use crate::tensor::{
    element_count, gather_strided, scatter_strided, strided_extent, DType, Tensor,
};
use crate::{check_inputs, Capability, ComputeUnits, FeatureKind, FeatureSpec, LoadOptions};

/// Runs `f`, converting an Objective-C exception into an error.
fn guarded<R>(what: &str, f: impl FnOnce() -> R) -> Result<R> {
    objc2::exception::catch(AssertUnwindSafe(f)).map_err(|e| {
        let detail = e
            .map(|e| format!("{e:?}"))
            .unwrap_or_else(|| "nil exception".into());
        CoreMlError::Exception(format!("{what}: {detail}"))
    })
}

fn ns_error(e: &NSError) -> String {
    format!("{} ({} {})", e.localizedDescription(), e.domain(), e.code())
}

pub fn supports(cap: Capability) -> bool {
    match cap {
        Capability::MultiFunction | Capability::State => available!(macos = 15.0),
        Capability::AsyncCompile => available!(macos = 14.4),
    }
}

/// `(major, minor, patch)` of the running macOS.
pub fn os_version() -> (u32, u32, u32) {
    let v = NSProcessInfo::processInfo().operatingSystemVersion();
    (
        v.majorVersion as u32,
        v.minorVersion as u32,
        v.patchVersion as u32,
    )
}

/// The OS build string ("Version 26.6.2 (Build 25G83)"), used in the
/// compile-cache key.
fn os_tag() -> String {
    NSProcessInfo::processInfo()
        .operatingSystemVersionString()
        .to_string()
}

fn ml_dtype(t: MLMultiArrayDataType) -> Option<DType> {
    if t == MLMultiArrayDataType::Float32 {
        Some(DType::F32)
    } else if t == MLMultiArrayDataType::Float16 {
        Some(DType::F16)
    } else if t == MLMultiArrayDataType::Int32 {
        Some(DType::I32)
    } else {
        None
    }
}

fn to_ml_dtype(d: DType) -> MLMultiArrayDataType {
    match d {
        DType::F32 => MLMultiArrayDataType::Float32,
        DType::F16 => MLMultiArrayDataType::Float16,
        DType::I32 => MLMultiArrayDataType::Int32,
    }
}

fn ml_units(u: ComputeUnits) -> MLComputeUnits {
    match u {
        ComputeUnits::CpuOnly => MLComputeUnits::CPUOnly,
        ComputeUnits::CpuAndGpu => MLComputeUnits::CPUAndGPU,
        ComputeUnits::CpuAndNeuralEngine => MLComputeUnits::CPUAndNeuralEngine,
        ComputeUnits::All => MLComputeUnits::All,
    }
}

fn numbers(a: &NSArray<NSNumber>) -> Vec<isize> {
    a.iter().map(|n| n.integerValue()).collect()
}

fn to_usizes(name: &str, v: &[isize]) -> Result<Vec<usize>> {
    v.iter()
        .map(|&x| {
            usize::try_from(x).map_err(|_| {
                CoreMlError::Prediction(format!("feature {name}: negative dimension {x}"))
            })
        })
        .collect()
}

fn feature_spec(d: &MLFeatureDescription) -> Result<FeatureSpec> {
    // SAFETY: plain property reads on a live description object.
    let (name, ty, optional) = unsafe { (d.name().to_string(), d.r#type(), d.isOptional()) };
    let kind = if ty == MLFeatureType::MultiArray {
        // SAFETY: the constraint exists for every multi-array feature.
        let c = unsafe { d.multiArrayConstraint() }
            .ok_or_else(|| CoreMlError::Load(format!("{name}: multi-array without constraint")))?;
        // SAFETY: property reads on a live constraint object.
        let (dtype, shape, sc) = unsafe {
            (
                ml_dtype(c.dataType()),
                numbers(&c.shape()),
                c.shapeConstraint(),
            )
        };
        FeatureKind::MultiArray {
            dtype,
            shape: fixed_shape(&name, &shape, &sc)?,
        }
    } else if ty == MLFeatureType::State {
        // SAFETY: the constraint exists for every state feature (macOS 15+,
        // and state features only appear in models that need it).
        let c = unsafe { d.stateConstraint() }
            .ok_or_else(|| CoreMlError::Load(format!("{name}: state without constraint")))?;
        // SAFETY: property reads on a live constraint object.
        let (dtype, shape) = unsafe { (ml_dtype(c.dataType()), numbers(&c.bufferShape())) };
        FeatureKind::State {
            dtype,
            shape: to_usizes(&name, &shape)?,
        }
    } else {
        FeatureKind::Other(ty.0)
    };
    Ok(FeatureSpec {
        name,
        kind,
        optional,
    })
}

/// The shape a multi-array feature accepts, when it accepts exactly one.
/// ML programs often describe a fixed shape as a one-entry enumeration or
/// as ranges with equal bounds rather than as "unspecified", so all three
/// forms are recognized. Anything genuinely flexible returns `None` and is
/// left to Core ML to validate.
fn fixed_shape(
    name: &str,
    default: &[isize],
    sc: &MLMultiArrayShapeConstraint,
) -> Result<Option<Vec<usize>>> {
    // SAFETY: property reads on a live shape constraint.
    let ty = unsafe { sc.r#type() };
    if ty == MLMultiArrayShapeConstraintType::Unspecified {
        return Ok(Some(to_usizes(name, default)?));
    }
    if ty == MLMultiArrayShapeConstraintType::Enumerated {
        // SAFETY: as above.
        let shapes = unsafe { sc.enumeratedShapes() };
        if shapes.len() == 1 {
            let only = numbers(&shapes.to_vec()[0]);
            return Ok(Some(to_usizes(name, &only)?));
        }
        return Ok(None);
    }
    if ty == MLMultiArrayShapeConstraintType::Range {
        // SAFETY: as above.
        let ranges = unsafe { sc.sizeRangeForDimension() };
        let mut dims = Vec::with_capacity(ranges.len());
        for v in ranges.iter() {
            match v.get_range() {
                Some(r) if r.length == 0 => dims.push(r.location),
                _ => return Ok(None),
            }
        }
        return Ok(Some(dims));
    }
    Ok(None)
}

fn specs(map: &NSDictionary<NSString, MLFeatureDescription>) -> Result<Vec<FeatureSpec>> {
    let (_, values) = map.to_vecs();
    let mut out = values
        .iter()
        .map(|d| feature_spec(d))
        .collect::<Result<Vec<_>>>()?;
    out.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(out)
}

/// Builds a contiguous `MLMultiArray` holding `t`.
fn to_multiarray(name: &str, t: &Tensor) -> Result<Retained<MLMultiArray>> {
    let shape: Vec<Retained<NSNumber>> = t
        .shape()
        .iter()
        .map(|&d| NSNumber::numberWithInteger(d as isize))
        .collect();
    let shape = NSArray::from_retained_slice(&shape);
    // SAFETY: initWithShape:dataType:error: with a valid shape array; the
    // returned array is uninitialized and fully written below.
    let arr = guarded("MLMultiArray init", || unsafe {
        MLMultiArray::initWithShape_dataType_error(
            MLMultiArray::alloc(),
            &shape,
            to_ml_dtype(t.dtype()),
        )
    })?
    .map_err(|e| CoreMlError::Prediction(format!("{name}: {}", ns_error(&e))))?;
    write_multiarray(name, &arr, t)?;
    Ok(arr)
}

/// Scatters `t` into `arr`, honoring whatever strides Core ML hands back.
fn write_multiarray(name: &str, arr: &MLMultiArray, t: &Tensor) -> Result<()> {
    let elem = t.dtype().size();
    let src = t.to_bytes();
    let shape = t.shape().to_vec();
    let mut failure: Option<CoreMlError> = None;
    {
        let failure_ref = &mut failure;
        let failure_cell = std::cell::RefCell::new(failure_ref);
        let block = RcBlock::new(
            |ptr: NonNull<std::ffi::c_void>, size: isize, strides: NonNull<NSArray<NSNumber>>| {
                // SAFETY: Core ML passes a live strides array for the
                // duration of the block.
                let strides = numbers(unsafe { strides.as_ref() });
                let res = (|| -> Result<()> {
                    let extent = strided_extent(&shape, &strides)?;
                    let size = usize::try_from(size).unwrap_or(0);
                    if extent * elem > size {
                        return Err(CoreMlError::Prediction(format!(
                            "{name}: buffer of {size} bytes is smaller than its strides imply"
                        )));
                    }
                    let strides: Vec<usize> = strides.iter().map(|&s| s as usize).collect();
                    // SAFETY: `ptr` is valid for `size` bytes inside the
                    // block, and the extent check above keeps every write
                    // in range.
                    let dst = unsafe { std::slice::from_raw_parts_mut(ptr.as_ptr().cast(), size) };
                    scatter_strided(dst, &src, &shape, &strides, elem);
                    Ok(())
                })();
                if let Err(e) = res {
                    **failure_cell.borrow_mut() = Some(e);
                }
            },
        );
        // SAFETY: getMutableBytesWithHandler: runs the block synchronously.
        guarded("MLMultiArray write", || unsafe {
            arr.getMutableBytesWithHandler(&block)
        })?;
    }
    match failure {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

/// Gathers `arr` into a dense tensor.
fn from_multiarray(name: &str, arr: &MLMultiArray) -> Result<Tensor> {
    // SAFETY: property reads on a live array.
    let (dt, shape, strides) = unsafe {
        (
            arr.dataType(),
            numbers(&arr.shape()),
            numbers(&arr.strides()),
        )
    };
    let dtype = ml_dtype(dt).ok_or_else(|| CoreMlError::DType {
        name: name.into(),
        expected: "float32, float16 or int32".into(),
        got: format!("MLMultiArrayDataType {}", dt.0),
    })?;
    let shape = to_usizes(name, &shape)?;
    let extent = strided_extent(&shape, &strides)?;
    let ustrides: Vec<usize> = strides.iter().map(|&s| s as usize).collect();
    let elem = dtype.size();
    let mut out: Option<Result<Vec<u8>>> = None;
    {
        let out_cell = std::cell::RefCell::new(&mut out);
        let block = RcBlock::new(|ptr: NonNull<std::ffi::c_void>, size: isize| {
            let size = usize::try_from(size).unwrap_or(0);
            let res = if extent * elem > size {
                Err(CoreMlError::Prediction(format!(
                    "{name}: buffer of {size} bytes is smaller than its strides imply"
                )))
            } else {
                // SAFETY: `ptr` is valid for `size` bytes inside the block
                // and the extent check bounds every read.
                let src = unsafe { std::slice::from_raw_parts(ptr.as_ptr().cast::<u8>(), size) };
                Ok(gather_strided(src, &shape, &ustrides, elem))
            };
            **out_cell.borrow_mut() = Some(res);
        });
        if element_count(&shape) > 0 {
            // SAFETY: getBytesWithHandler: (macOS 12.3+) runs synchronously.
            guarded("MLMultiArray read", || unsafe {
                arr.getBytesWithHandler(&block)
            })?;
        }
    }
    let bytes = match out {
        Some(r) => r?,
        None if element_count(&shape) == 0 => Vec::new(),
        None => {
            return Err(CoreMlError::Prediction(format!(
                "{name}: Core ML did not expose the output buffer"
            )))
        }
    };
    Tensor::from_bytes(dtype, shape, &bytes)
}

fn provider(inputs: &[(&str, &Tensor)]) -> Result<Retained<MLDictionaryFeatureProvider>> {
    let mut keys = Vec::with_capacity(inputs.len());
    let mut values: Vec<Retained<MLFeatureValue>> = Vec::with_capacity(inputs.len());
    for (name, t) in inputs {
        let arr = to_multiarray(name, t)?;
        keys.push(NSString::from_str(name));
        // SAFETY: wraps a live multi-array.
        values.push(unsafe { MLFeatureValue::featureValueWithMultiArray(&arr) });
    }
    let key_refs: Vec<&NSString> = keys.iter().map(|k| &**k).collect();
    let value_refs: Vec<&AnyObject> = values.iter().map(|v| v.as_ref()).collect();
    let dict = NSDictionary::<NSString, AnyObject>::from_slices(&key_refs, &value_refs);
    // SAFETY: every value is an MLFeatureValue, which is what
    // initWithDictionary:error: requires of the AnyObject values.
    guarded("MLDictionaryFeatureProvider init", || unsafe {
        MLDictionaryFeatureProvider::initWithDictionary_error(
            MLDictionaryFeatureProvider::alloc(),
            &dict,
        )
    })?
    .map_err(|e| CoreMlError::Prediction(ns_error(&e)))
}

fn read_outputs(
    out: &ProtocolObject<dyn MLFeatureProvider>,
    specs: &[FeatureSpec],
) -> Result<HashMap<String, Tensor>> {
    let mut map = HashMap::new();
    for spec in specs {
        let FeatureKind::MultiArray { .. } = spec.kind else {
            continue;
        };
        let key = NSString::from_str(&spec.name);
        // SAFETY: featureValueForName: on a live provider.
        let Some(value) = (unsafe { out.featureValueForName(&key) }) else {
            if spec.optional {
                continue;
            }
            return Err(CoreMlError::MissingFeature(spec.name.clone()));
        };
        // SAFETY: multiArrayValue on a live feature value.
        let arr = unsafe { value.multiArrayValue() }.ok_or_else(|| {
            CoreMlError::Prediction(format!("{} is not a multi-array", spec.name))
        })?;
        map.insert(spec.name.clone(), from_multiarray(&spec.name, &arr)?);
    }
    Ok(map)
}

/// `Retained<MLModel>` is not `Send` because objc2 cannot know the class is
/// thread-safe. MLModel is not bound to a thread, and every use goes
/// through the `Mutex` in `CoreMlModel`, so moving it is sound.
struct SendModel(Retained<MLModel>);
// SAFETY: see the type comment.
unsafe impl Send for SendModel {}

/// A loaded Core ML model (one function of a package, when it has several).
pub struct CoreMlModel {
    model: Mutex<SendModel>,
    inputs: Vec<FeatureSpec>,
    outputs: Vec<FeatureSpec>,
    states: Vec<FeatureSpec>,
    path: PathBuf,
    function: Option<String>,
}

impl std::fmt::Debug for CoreMlModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CoreMlModel")
            .field("path", &self.path)
            .field("function", &self.function)
            .field("inputs", &self.inputs)
            .field("outputs", &self.outputs)
            .field("states", &self.states)
            .finish()
    }
}

impl CoreMlModel {
    /// Loads a compiled `.mlmodelc` directory. Use [`CoreMlModel::open`]
    /// for a package that may need compiling first.
    pub fn load(path: &Path, opts: &LoadOptions) -> Result<Self> {
        if path.extension().and_then(|e| e.to_str()) != Some("mlmodelc") {
            return Err(CoreMlError::Load(format!(
                "{} is not a compiled .mlmodelc; compile it with compile_if_needed",
                path.display()
            )));
        }
        if !path.is_dir() {
            return Err(CoreMlError::Load(format!(
                "{} does not exist",
                path.display()
            )));
        }
        if opts.function.is_some() && !supports(Capability::MultiFunction) {
            return Err(Capability::MultiFunction.too_old());
        }
        let url = NSURL::from_directory_path(path)
            .ok_or_else(|| CoreMlError::Load(format!("bad path {}", path.display())))?;
        autoreleasepool(|_| {
            // SAFETY: a fresh configuration object; setFunctionName: is only
            // called after the macOS 15 check above.
            let config = unsafe { MLModelConfiguration::new() };
            unsafe { config.setComputeUnits(ml_units(opts.compute_units)) };
            if let Some(f) = &opts.function {
                let f = NSString::from_str(f);
                unsafe { config.setFunctionName(Some(&f)) };
            }
            // SAFETY: modelWithContentsOfURL:configuration:error: with a
            // valid file URL and configuration.
            let model = guarded("MLModel load", || unsafe {
                MLModel::modelWithContentsOfURL_configuration_error(&url, &config)
            })?
            .map_err(|e| CoreMlError::Load(format!("{}: {}", path.display(), ns_error(&e))))?;
            // SAFETY: property reads on the freshly loaded model.
            let desc = unsafe { model.modelDescription() };
            let inputs = specs(&*unsafe { desc.inputDescriptionsByName() })?;
            let outputs = specs(&*unsafe { desc.outputDescriptionsByName() })?;
            let states = if supports(Capability::State) {
                specs(&*unsafe { desc.stateDescriptionsByName() })?
            } else {
                Vec::new()
            };
            Ok(Self {
                model: Mutex::new(SendModel(model)),
                inputs,
                outputs,
                states,
                path: path.to_path_buf(),
                function: opts.function.clone(),
            })
        })
    }

    /// Loads `path`, compiling a `.mlpackage` or `.mlmodel` into
    /// `cache_root` first (once per content key).
    pub fn open(path: &Path, cache_root: &Path, opts: &LoadOptions) -> Result<Self> {
        let compiled = compile_if_needed(path, cache_root)?;
        Self::load(&compiled, opts)
    }

    pub fn inputs(&self) -> &[FeatureSpec] {
        &self.inputs
    }

    pub fn outputs(&self) -> &[FeatureSpec] {
        &self.outputs
    }

    pub fn states(&self) -> &[FeatureSpec] {
        &self.states
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn function(&self) -> Option<&str> {
        self.function.as_deref()
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, SendModel> {
        // A poisoned lock only means another prediction panicked in Rust
        // code around the call; the model object itself is still valid.
        self.model.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// Runs one stateless prediction and returns every multi-array output.
    pub fn predict(&self, inputs: &[(&str, &Tensor)]) -> Result<HashMap<String, Tensor>> {
        check_inputs(&self.inputs, inputs)?;
        let guard = self.lock();
        autoreleasepool(|_| {
            let p = provider(inputs)?;
            let p = ProtocolObject::from_ref(&*p);
            // SAFETY: predictionFromFeatures:error: with a provider built
            // from inputs that passed check_inputs.
            let out = guarded("MLModel prediction", || unsafe {
                guard.0.predictionFromFeatures_error(p)
            })?
            .map_err(|e| CoreMlError::Prediction(ns_error(&e)))?;
            read_outputs(&out, &self.outputs)
        })
    }

    /// Creates a fresh state for a stateful model.
    pub fn new_state(&self) -> Result<CoreMlState> {
        if !supports(Capability::State) {
            return Err(Capability::State.too_old());
        }
        if self.states.is_empty() {
            return Err(CoreMlError::Prediction(format!(
                "{} declares no state",
                self.path.display()
            )));
        }
        let guard = self.lock();
        // SAFETY: newState on a stateful model, after the macOS 15 check.
        let state = guarded("MLModel newState", || unsafe { guard.0.newState() })?;
        Ok(CoreMlState {
            state: SendState(state),
            owner: self.owner_id(),
            specs: self.states.clone(),
        })
    }

    fn owner_id(&self) -> usize {
        self as *const Self as usize
    }

    /// Runs one prediction that reads and updates `state`.
    pub fn predict_with_state(
        &self,
        state: &mut CoreMlState,
        inputs: &[(&str, &Tensor)],
    ) -> Result<HashMap<String, Tensor>> {
        if state.owner != self.owner_id() {
            return Err(CoreMlError::Prediction(
                "state was created by a different model".into(),
            ));
        }
        check_inputs(&self.inputs, inputs)?;
        let guard = self.lock();
        autoreleasepool(|_| {
            let p = provider(inputs)?;
            let p = ProtocolObject::from_ref(&*p);
            // SAFETY: stateful prediction after the macOS 15 check made in
            // new_state, with a state this model created.
            let out = guarded("MLModel stateful prediction", || unsafe {
                guard
                    .0
                    .predictionFromFeatures_usingState_error(p, &state.state.0)
            })?
            .map_err(|e| CoreMlError::Prediction(ns_error(&e)))?;
            read_outputs(&out, &self.outputs)
        })
    }
}

struct SendState(Retained<MLState>);
// SAFETY: MLState is a buffer container with no thread affinity; the
// wrapper is not Sync, so it is never used from two threads at once.
unsafe impl Send for SendState {}

/// The mutable state of a stateful model (for example a KV cache). Not
/// `Sync`: one prediction at a time.
pub struct CoreMlState {
    state: SendState,
    owner: usize,
    specs: Vec<FeatureSpec>,
}

impl std::fmt::Debug for CoreMlState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CoreMlState")
            .field("specs", &self.specs)
            .finish()
    }
}

impl CoreMlState {
    pub fn specs(&self) -> &[FeatureSpec] {
        &self.specs
    }

    /// Copies every state buffer of `src` into this state. The two may come
    /// from different models (for example the prefill and decode functions
    /// of one package, which declare the same cache), but every buffer must
    /// match in name, shape, strides and dtype.
    pub fn copy_from(&mut self, src: &CoreMlState) -> Result<()> {
        fn names(s: &CoreMlState) -> Vec<String> {
            let mut v: Vec<String> = s.specs.iter().map(|f| f.name.clone()).collect();
            v.sort_unstable();
            v
        }
        if names(self) != names(src) {
            return Err(CoreMlError::Prediction(format!(
                "state buffers differ: {:?} vs {:?}",
                names(src),
                names(self)
            )));
        }
        for spec in &self.specs {
            let key = NSString::from_str(&spec.name);
            let mut failure: Option<CoreMlError> = None;
            {
                let cell = std::cell::RefCell::new(&mut failure);
                let outer = RcBlock::new(|src_arr: NonNull<MLMultiArray>| {
                    // SAFETY: both arrays are live for the duration of their
                    // (synchronous, nested) blocks.
                    let src_arr = unsafe { src_arr.as_ref() };
                    let inner = RcBlock::new(|dst_arr: NonNull<MLMultiArray>| {
                        let dst_arr = unsafe { dst_arr.as_ref() };
                        let r = copy_array(&spec.name, src_arr, dst_arr);
                        if let Err(e) = r {
                            **cell.borrow_mut() = Some(e);
                        }
                    });
                    // SAFETY: synchronous accessor on a live state object.
                    let r = guarded("MLState write", || unsafe {
                        self.state
                            .0
                            .getMultiArrayForStateNamed_handler(&key, &inner)
                    });
                    if let Err(e) = r {
                        **cell.borrow_mut() = Some(e);
                    }
                });
                // SAFETY: synchronous accessor on a live state object.
                guarded("MLState read", || unsafe {
                    src.state.0.getMultiArrayForStateNamed_handler(&key, &outer)
                })?;
            }
            if let Some(e) = failure {
                return Err(e);
            }
        }
        Ok(())
    }

    /// Copies one state buffer out. Meant for tests and diagnostics.
    pub fn read(&self, name: &str) -> Result<Tensor> {
        if !self.specs.iter().any(|s| s.name == name) {
            return Err(CoreMlError::MissingFeature(name.into()));
        }
        let key = NSString::from_str(name);
        let mut out: Option<Result<Tensor>> = None;
        {
            let cell = std::cell::RefCell::new(&mut out);
            let block = RcBlock::new(|arr: NonNull<MLMultiArray>| {
                // SAFETY: the array is live for the duration of the block.
                let r = from_multiarray(name, unsafe { arr.as_ref() });
                **cell.borrow_mut() = Some(r);
            });
            // SAFETY: synchronous accessor on a live state object.
            guarded("MLState read", || unsafe {
                self.state
                    .0
                    .getMultiArrayForStateNamed_handler(&key, &block)
            })?;
        }
        out.unwrap_or_else(|| Err(CoreMlError::MissingFeature(name.into())))
    }
}

/// Byte copy between two state arrays of identical layout.
fn copy_array(name: &str, src: &MLMultiArray, dst: &MLMultiArray) -> Result<()> {
    // SAFETY: property reads on live arrays.
    let (ss, sd, st, dd, ds, dt) = unsafe {
        (
            numbers(&src.shape()),
            numbers(&src.strides()),
            src.dataType(),
            numbers(&dst.shape()),
            numbers(&dst.strides()),
            dst.dataType(),
        )
    };
    if ss != dd || sd != ds || st != dt {
        return Err(CoreMlError::Prediction(format!(
            "state {name}: shape {ss:?}/{dd:?} or strides {sd:?}/{ds:?} or dtype differ"
        )));
    }
    let elem = ml_dtype(st)
        .map(DType::size)
        .ok_or_else(|| CoreMlError::Prediction(format!("state {name}: unsupported dtype")))?;
    let shape = to_usizes(name, &ss)?;
    let bytes = strided_extent(&shape, &sd)? * elem;
    // Read the source's raw buffer, then write it into the destination.
    // Same layout on both sides, so the strided bytes copy one to one.
    let mut raw: Option<Result<Vec<u8>>> = None;
    {
        let cell = std::cell::RefCell::new(&mut raw);
        let block = RcBlock::new(|ptr: NonNull<std::ffi::c_void>, size: isize| {
            let size = usize::try_from(size).unwrap_or(0);
            **cell.borrow_mut() = Some(if size < bytes {
                Err(CoreMlError::Prediction(format!(
                    "state {name}: source buffer is short"
                )))
            } else {
                // SAFETY: `ptr` is valid for `size >= bytes` bytes inside the block.
                Ok(
                    unsafe { std::slice::from_raw_parts(ptr.as_ptr().cast::<u8>(), bytes) }
                        .to_vec(),
                )
            });
        });
        // SAFETY: synchronous accessor on a live array.
        guarded("MLMultiArray read", || unsafe {
            src.getBytesWithHandler(&block)
        })?;
    }
    let raw = raw.unwrap_or_else(|| {
        Err(CoreMlError::Prediction(format!(
            "state {name}: source buffer not exposed"
        )))
    })?;
    let mut failure: Option<CoreMlError> = None;
    {
        let cell = std::cell::RefCell::new(&mut failure);
        let block = RcBlock::new(
            |ptr: NonNull<std::ffi::c_void>, size: isize, strides: NonNull<NSArray<NSNumber>>| {
                // SAFETY: Core ML passes a live strides array for the block.
                let now = numbers(unsafe { strides.as_ref() });
                let size = usize::try_from(size).unwrap_or(0);
                if now != sd || size < bytes {
                    **cell.borrow_mut() = Some(CoreMlError::Prediction(format!(
                        "state {name}: destination layout changed under the copy"
                    )));
                    return;
                }
                // SAFETY: `ptr` is valid for `size >= bytes` bytes and the
                // layout matches the source's.
                let dst =
                    unsafe { std::slice::from_raw_parts_mut(ptr.as_ptr().cast::<u8>(), bytes) };
                dst.copy_from_slice(&raw);
            },
        );
        // SAFETY: synchronous accessor on a live array.
        guarded("MLMultiArray write", || unsafe {
            dst.getMutableBytesWithHandler(&block)
        })?;
    }
    match failure {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

/// Returns a compiled `.mlmodelc` for `path`.
///
/// A `.mlmodelc` is returned unchanged. A `.mlpackage` or `.mlmodel` is
/// compiled once into `cache_root/<stem>-<key>.mlmodelc` (see
/// [`crate::cache_key`]); later calls with unchanged content reuse it.
/// Older entries for the same stem are pruned after a successful compile,
/// so a re-downloaded multi-gigabyte package does not leave its stale
/// compile behind.
pub fn compile_if_needed(path: &Path, cache_root: &Path) -> Result<PathBuf> {
    match path.extension().and_then(|e| e.to_str()) {
        Some("mlmodelc") => return Ok(path.to_path_buf()),
        Some("mlpackage") | Some("mlmodel") => {}
        _ => {
            return Err(CoreMlError::Compile(format!(
                "{} is not a .mlpackage, .mlmodel or .mlmodelc",
                path.display()
            )))
        }
    }
    let key = cache_key(path, &os_tag())?;
    let dest = compiled_path_for(path, cache_root, &key);
    if dest.is_dir() {
        return Ok(dest);
    }
    fs::create_dir_all(cache_root)?;
    let stem = path
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "model".into());
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let staging = cache_root.join(format!(
        ".staging-{stem}-{}-{nanos}.mlmodelc",
        std::process::id()
    ));
    let url = NSURL::from_file_path(path)
        .ok_or_else(|| CoreMlError::Compile(format!("bad path {}", path.display())))?;

    let compiled_tmp = compile_to_temp(&url)?;
    // The compiler writes to a temporary directory it may clean up, so move
    // the result into the cache now. A cross-volume install falls back to
    // a copy.
    if fs::rename(&compiled_tmp, &staging).is_err() {
        copy_dir(&compiled_tmp, &staging)
            .map_err(|e| CoreMlError::Compile(format!("copying compiled model: {e}")))?;
        let _ = fs::remove_dir_all(&compiled_tmp);
    }
    match fs::rename(&staging, &dest) {
        Ok(()) => {}
        Err(_) if dest.is_dir() => {
            // Another process finished the same compile first.
            let _ = fs::remove_dir_all(&staging);
        }
        Err(e) => {
            let _ = fs::remove_dir_all(&staging);
            return Err(CoreMlError::Compile(format!(
                "installing compiled model: {e}"
            )));
        }
    }
    prune_stale(cache_root, &stem, &dest);
    Ok(dest)
}

fn prune_stale(cache_root: &Path, stem: &str, keep: &Path) {
    let Ok(entries) = fs::read_dir(cache_root) else {
        return;
    };
    for entry in entries.flatten() {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        let path = entry.path();
        if path != keep && is_cache_entry_for(&name, stem) && path.is_dir() {
            let _ = fs::remove_dir_all(&path);
        }
    }
}

fn compile_to_temp(url: &NSURL) -> Result<PathBuf> {
    if supports(Capability::AsyncCompile) {
        let (tx, rx) = mpsc::channel::<Result<PathBuf>>();
        let tx = Mutex::new(Some(tx));
        let block = RcBlock::new(move |out: *mut NSURL, err: *mut NSError| {
            // SAFETY: Core ML passes either a valid URL or a valid error,
            // both live for the duration of the handler.
            let res = unsafe {
                match (out.as_ref(), err.as_ref()) {
                    (Some(u), _) => u.to_file_path().ok_or_else(|| {
                        CoreMlError::Compile("compiled URL is not a file path".into())
                    }),
                    (None, Some(e)) => Err(CoreMlError::Compile(ns_error(e))),
                    (None, None) => Err(CoreMlError::Compile("no result and no error".into())),
                }
            };
            if let Some(tx) = tx.lock().ok().and_then(|mut g| g.take()) {
                let _ = tx.send(res);
            }
        });
        // SAFETY: gated on macOS 14.4 above; the handler is retained by
        // Core ML until it runs.
        guarded("MLModel compile", || unsafe {
            MLModel::compileModelAtURL_completionHandler(url, &block)
        })?;
        rx.recv()
            .map_err(|_| CoreMlError::Compile("compile handler was dropped".into()))?
    } else {
        // SAFETY: the synchronous compile, deprecated in 14.4 but the only
        // option before it.
        #[allow(deprecated)]
        let r = guarded("MLModel compile", || unsafe {
            MLModel::compileModelAtURL_error(url)
        })?;
        let u = r.map_err(|e| CoreMlError::Compile(ns_error(&e)))?;
        u.to_file_path()
            .ok_or_else(|| CoreMlError::Compile("compiled URL is not a file path".into()))
    }
}
