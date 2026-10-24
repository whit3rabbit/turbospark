//! The macOS Core ML boundary. Every `unsafe` block in this crate lives in
//! this file. The rules it follows (see `AGENTS.md`):
//!
//! - NSExceptions are NOT caught. Catching needs objc2's `exception`
//!   feature, whose Objective-C helper adds `__objc_personality_v0` to the
//!   link. The macOS app already links three personality routines (two Rust
//!   static libraries plus C++), compact unwind encodes at most three, and a
//!   fourth fails the app link. So every call is made safe up front instead:
//!   inputs are checked against the model description, and OS-gated APIs
//!   behind `supports`. An exception that still escapes aborts.
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
use std::path::{Path, PathBuf};
use std::ptr::NonNull;
use std::sync::{mpsc, Mutex};

use block2::RcBlock;
use objc2::rc::{autoreleasepool, Retained};
use objc2::runtime::{AnyClass, AnyObject, NSObjectProtocol, ProtocolObject};
use objc2::{available, AllocAnyThread};
use objc2_core_ml::{
    MLComputeDeviceProtocol, MLComputePlan, MLComputeUnits, MLDictionaryFeatureProvider,
    MLFeatureDescription, MLFeatureProvider, MLFeatureType, MLFeatureValue, MLModel,
    MLModelConfiguration, MLModelStructure, MLModelStructureProgramBlock, MLMultiArray,
    MLMultiArrayDataType, MLMultiArrayShapeConstraint, MLMultiArrayShapeConstraintType, MLState,
};
use objc2_foundation::{NSArray, NSDictionary, NSError, NSNumber, NSProcessInfo, NSString, NSURL};

use crate::cache::{cache_key, compiled_path_for, copy_dir, is_cache_entry_for};
use crate::error::{CoreMlError, Result};
use crate::tensor::{
    element_count, gather_strided, scatter_strided, strided_extent, DType, Tensor,
};
use crate::{
    check_inputs, Capability, ComputeDevice, ComputePlan, ComputeUnits, FeatureKind, FeatureSpec,
    LoadOptions, OpPlacement,
};

/// Marks an Objective-C call site. It cannot catch an NSException (see the
/// module docs); it names the call so a future catch has one place to go.
fn guarded<R>(_what: &str, f: impl FnOnce() -> R) -> Result<R> {
    Ok(f())
}

fn ns_error(e: &NSError) -> String {
    format!("{} ({} {})", e.localizedDescription(), e.domain(), e.code())
}

pub fn supports(cap: Capability) -> bool {
    match cap {
        Capability::MultiFunction | Capability::State => available!(macos = 15.0),
        Capability::AsyncCompile | Capability::ComputePlan => available!(macos = 14.4),
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

/// A fresh configuration for `opts`. Callers check
/// `Capability::MultiFunction` before passing a function name.
fn configuration(opts: &LoadOptions) -> Retained<MLModelConfiguration> {
    // SAFETY: a fresh configuration object; setFunctionName: is only
    // reached after the caller's macOS 15 check.
    let config = unsafe { MLModelConfiguration::new() };
    unsafe { config.setComputeUnits(ml_units(opts.compute_units)) };
    if let Some(f) = &opts.function {
        let f = NSString::from_str(f);
        unsafe { config.setFunctionName(Some(&f)) };
    }
    config
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
            let config = configuration(opts);
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

/// Reads Core ML's compute plan for a compiled `.mlmodelc`: the device each
/// ML program operation is planned on under `opts`, and Core ML's cost
/// estimate for it. It loads the model on its own, separately from any
/// `CoreMlModel`, and may trigger the same device compile a load does.
pub fn compute_plan(path: &Path, opts: &LoadOptions) -> Result<ComputePlan> {
    if path.extension().and_then(|e| e.to_str()) != Some("mlmodelc") || !path.is_dir() {
        return Err(CoreMlError::Load(format!(
            "{} is not a compiled .mlmodelc directory",
            path.display()
        )));
    }
    if !supports(Capability::ComputePlan) {
        return Err(Capability::ComputePlan.too_old());
    }
    if opts.function.is_some() && !supports(Capability::MultiFunction) {
        return Err(Capability::MultiFunction.too_old());
    }
    let url = NSURL::from_directory_path(path)
        .ok_or_else(|| CoreMlError::Load(format!("bad path {}", path.display())))?;
    let (tx, rx) = mpsc::channel::<Result<ComputePlan>>();
    let tx = Mutex::new(Some(tx));
    autoreleasepool(|_| {
        let config = configuration(opts);
        // The plan is walked inside the handler, on whatever queue Core ML
        // calls it from, so no Objective-C object crosses a thread: only
        // the plain-Rust result is sent back.
        let block = RcBlock::new(move |plan: *mut MLComputePlan, err: *mut NSError| {
            // SAFETY: Core ML passes either a valid plan or a valid error,
            // both live for the duration of the handler.
            let res = unsafe {
                match (plan.as_ref(), err.as_ref()) {
                    (Some(p), _) => autoreleasepool(|_| walk_plan(p)),
                    (None, Some(e)) => Err(CoreMlError::Load(ns_error(e))),
                    (None, None) => Err(CoreMlError::Load("no compute plan and no error".into())),
                }
            };
            if let Some(tx) = tx.lock().ok().and_then(|mut g| g.take()) {
                let _ = tx.send(res);
            }
        });
        // SAFETY: gated on macOS 14.4 above (15 for a function name); the
        // handler is retained by Core ML until it runs.
        guarded("MLComputePlan load", || unsafe {
            MLComputePlan::loadContentsOfURL_configuration_completionHandler(&url, &config, &block)
        })
    })?;
    rx.recv()
        .map_err(|_| CoreMlError::Load("compute plan handler was dropped".into()))?
}

fn walk_plan(plan: &MLComputePlan) -> Result<ComputePlan> {
    let mut out = ComputePlan::default();
    // SAFETY: property read on a live plan.
    let structure = unsafe { plan.modelStructure() };
    walk_structure(plan, &structure, "", &mut out)?;
    Ok(out)
}

fn walk_structure(
    plan: &MLComputePlan,
    s: &MLModelStructure,
    prefix: &str,
    out: &mut ComputePlan,
) -> Result<()> {
    // SAFETY: property reads on a live structure, on the handler's thread
    // only (Core ML marks them not thread-safe).
    if let Some(program) = unsafe { s.program() } {
        let (names, functions) = unsafe { program.functions() }.to_vecs();
        for (name, f) in names.iter().zip(functions.iter()) {
            // SAFETY: as above.
            let block = unsafe { f.block() };
            walk_block(plan, &block, &format!("{prefix}{name}"), out);
        }
        return Ok(());
    }
    // SAFETY: as above.
    if let Some(pipeline) = unsafe { s.pipeline() } {
        let names = unsafe { pipeline.subModelNames() };
        let subs = unsafe { pipeline.subModels() };
        for (name, sub) in names.iter().zip(subs.iter()) {
            walk_structure(plan, &sub, &format!("{prefix}{name}/"), out)?;
        }
        return Ok(());
    }
    Err(CoreMlError::Load(
        "compute plans are read for ML programs and pipelines only, not neural networks".into(),
    ))
}

fn walk_block(
    plan: &MLComputePlan,
    block: &MLModelStructureProgramBlock,
    function: &str,
    out: &mut ComputePlan,
) {
    // SAFETY: property reads on live structure objects, and plan lookups
    // for operations taken from this plan's own structure.
    for op in unsafe { block.operations() }.iter() {
        let operator = unsafe { op.operatorName() }.to_string();
        let output = unsafe { op.outputs() }
            .iter()
            .next()
            .map(|o| unsafe { o.name() }.to_string());
        let usage = unsafe { plan.computeDeviceUsageForMLProgramOperation(&op) };
        let (preferred, supported) = match usage {
            Some(u) => {
                let preferred = unsafe { u.preferredComputeDevice() };
                let supported = unsafe { u.supportedComputeDevices() };
                (
                    Some(device(&preferred)),
                    supported.iter().map(|d| device(&d)).collect(),
                )
            }
            None => (None, Vec::new()),
        };
        let cost =
            unsafe { plan.estimatedCostOfMLProgramOperation(&op) }.map(|c| unsafe { c.weight() });
        out.ops.push(OpPlacement {
            function: function.to_string(),
            operator,
            output,
            preferred,
            supported,
            cost,
        });
        // Control-flow operations (cond, while_loop) carry nested blocks.
        for inner in unsafe { op.blocks() }.iter() {
            walk_block(plan, &inner, function, out);
        }
    }
}

/// Classifies a compute device by class. The classes are looked up by name
/// so the GPU one does not need objc2-metal.
fn device(d: &ProtocolObject<dyn MLComputeDeviceProtocol>) -> ComputeDevice {
    for (name, dev) in [
        (c"MLNeuralEngineComputeDevice", ComputeDevice::NeuralEngine),
        (c"MLGPUComputeDevice", ComputeDevice::Gpu),
        (c"MLCPUComputeDevice", ComputeDevice::Cpu),
    ] {
        if AnyClass::get(name).is_some_and(|c| d.isKindOfClass(c)) {
            return dev;
        }
    }
    let obj: &AnyObject = d.as_ref();
    ComputeDevice::Other(obj.class().name().to_string_lossy().into_owned())
}

// ---- MEASUREMENT ONLY: IOSurface-backed planes (feature `ane-split-probe`) ----

#[cfg(feature = "ane-split-probe")]
pub use shared_plane::SharedF16Plane;

#[cfg(feature = "ane-split-probe")]
mod shared_plane {
    //! F16 planes that Metal and Core ML both read and write in place, for
    //! the qwen38 Neural Engine split prototype (`docs/MOE_PRIOR_ART.md`).
    //! Each plane is an IOSurface wrapped as a `CVPixelBuffer` and an
    //! `MLMultiArray`; the caller wraps [`SharedF16Plane::base_ptr`] in a
    //! no-copy Metal buffer and must drop that buffer first. Inputs and
    //! outputs bound this way are never copied by Core ML, which is what a
    //! plain `MLMultiArray` would cost per prediction on the Neural Engine.

    use std::ffi::c_void;

    use objc2::encode::{Encoding, RefEncode};
    use objc2::rc::{autoreleasepool, Retained};
    use objc2::runtime::{AnyClass, AnyObject, ProtocolObject};
    use objc2::{msg_send, AllocAnyThread};
    use objc2_core_ml::{
        MLDictionaryFeatureProvider, MLFeatureProvider, MLFeatureValue, MLMultiArray,
    };
    use objc2_foundation::{NSArray, NSDictionary, NSError, NSNumber, NSString};

    use super::{guarded, ns_error, CoreMlModel};
    use crate::error::{CoreMlError, Result};
    use crate::tensor::DType;
    use crate::{FeatureKind, FeatureSpec};

    type CFTypeRef = *const c_void;

    /// `CVBufferRef`'s pointee, so the message encoding matches
    /// `^{__CVBuffer=}` (objc2 verifies it under debug assertions).
    #[repr(C)]
    struct CvBuffer {
        _private: [u8; 0],
    }
    // SAFETY: the encoding CoreVideo declares for CVBufferRef.
    unsafe impl RefEncode for CvBuffer {
        const ENCODING_REF: Encoding = Encoding::Pointer(&Encoding::Struct("__CVBuffer", &[]));
    }

    #[link(name = "IOSurface", kind = "framework")]
    extern "C" {
        static kIOSurfaceWidth: CFTypeRef;
        static kIOSurfaceHeight: CFTypeRef;
        static kIOSurfaceBytesPerElement: CFTypeRef;
        static kIOSurfaceBytesPerRow: CFTypeRef;
        static kIOSurfacePixelFormat: CFTypeRef;
        fn IOSurfaceCreate(properties: CFTypeRef) -> *mut c_void;
        fn IOSurfaceGetBaseAddress(surface: *mut c_void) -> *mut c_void;
        fn IOSurfaceGetAllocSize(surface: *mut c_void) -> usize;
        fn IOSurfaceGetBytesPerRow(surface: *mut c_void) -> usize;
        fn IOSurfaceAlignProperty(property: CFTypeRef, value: usize) -> usize;
    }

    #[link(name = "CoreVideo", kind = "framework")]
    extern "C" {
        fn CVPixelBufferCreateWithIOSurface(
            allocator: CFTypeRef,
            surface: *mut c_void,
            attributes: CFTypeRef,
            out: *mut *mut CvBuffer,
        ) -> i32;
    }

    #[link(name = "CoreFoundation", kind = "framework")]
    extern "C" {
        fn CFRelease(cf: CFTypeRef);
    }

    /// `kCVPixelFormatType_OneComponent16Half`, the one format
    /// `initWithPixelBuffer:shape:` accepts.
    const ONE_COMPONENT_16_HALF: u32 = u32::from_be_bytes(*b"L00h");

    /// A `[1, 1, rows, cols]` Float16 plane over one IOSurface.
    pub struct SharedF16Plane {
        array: Retained<MLMultiArray>,
        pixel: *mut CvBuffer,
        surface: *mut c_void,
        rows: usize,
        cols: usize,
        stride: usize,
    }

    // SAFETY: IOSurface and CVPixelBuffer are thread-safe CoreFoundation
    // objects, and the multi-array is only read through `&self` by one
    // prediction at a time (the model's mutex). Not Sync.
    unsafe impl Send for SharedF16Plane {}

    impl std::fmt::Debug for SharedF16Plane {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("SharedF16Plane")
                .field("rows", &self.rows)
                .field("cols", &self.cols)
                .field("stride", &self.stride)
                .finish()
        }
    }

    fn key(k: CFTypeRef) -> &'static NSString {
        // SAFETY: the kIOSurface* keys are immortal CFStrings, toll-free
        // bridged to NSString.
        unsafe { &*(k as *const NSString) }
    }

    impl SharedF16Plane {
        pub fn new(rows: usize, cols: usize) -> Result<Self> {
            if rows == 0 || cols == 0 {
                return Err(CoreMlError::Prediction("empty shared plane".into()));
            }
            autoreleasepool(|_| {
                // SAFETY: plain C calls with a valid key and value; the keys
                // are framework statics.
                let bpr = unsafe { IOSurfaceAlignProperty(kIOSurfaceBytesPerRow, cols * 2) };
                let keys = unsafe {
                    [
                        key(kIOSurfaceWidth),
                        key(kIOSurfaceHeight),
                        key(kIOSurfaceBytesPerElement),
                        key(kIOSurfaceBytesPerRow),
                        key(kIOSurfacePixelFormat),
                    ]
                };
                let values = [
                    NSNumber::numberWithUnsignedInteger(cols),
                    NSNumber::numberWithUnsignedInteger(rows),
                    NSNumber::numberWithUnsignedInteger(2),
                    NSNumber::numberWithUnsignedInteger(bpr),
                    NSNumber::numberWithUnsignedInteger(ONE_COMPONENT_16_HALF as usize),
                ];
                let value_refs: Vec<&NSNumber> = values.iter().map(|v| &**v).collect();
                let props = NSDictionary::<NSString, NSNumber>::from_slices(&keys, &value_refs);
                // SAFETY: an NSDictionary is a CFDictionary (toll-free
                // bridged); IOSurfaceCreate follows the Create rule.
                let surface = unsafe { IOSurfaceCreate(Retained::as_ptr(&props).cast()) };
                if surface.is_null() {
                    return Err(CoreMlError::Prediction(format!(
                        "IOSurfaceCreate {cols} x {rows} failed"
                    )));
                }
                let mut pixel: *mut CvBuffer = std::ptr::null_mut();
                // SAFETY: a live surface; the out pointer follows the Create
                // rule.
                let rc = unsafe {
                    CVPixelBufferCreateWithIOSurface(
                        std::ptr::null(),
                        surface,
                        std::ptr::null(),
                        &mut pixel,
                    )
                };
                if rc != 0 || pixel.is_null() {
                    // SAFETY: releasing our +1 on the surface.
                    unsafe { CFRelease(surface) };
                    return Err(CoreMlError::Prediction(format!(
                        "CVPixelBufferCreateWithIOSurface returned {rc}"
                    )));
                }
                let shape: Vec<Retained<NSNumber>> = [1, 1, rows, cols]
                    .iter()
                    .map(|&d| NSNumber::numberWithInteger(d as isize))
                    .collect();
                let shape = NSArray::from_retained_slice(&shape);
                // SAFETY: initWithPixelBuffer:shape: (macOS 12) with a
                // OneComponent16Half buffer whose width is the last
                // dimension and height the product of the others.
                let array: Option<Retained<MLMultiArray>> =
                    guarded("MLMultiArray initWithPixelBuffer", || unsafe {
                        msg_send![MLMultiArray::alloc(), initWithPixelBuffer: pixel, shape: &*shape]
                    })?;
                // SAFETY: plain getters on a live surface.
                let stride = unsafe { IOSurfaceGetBytesPerRow(surface) } / 2;
                let plane = Self {
                    array: match array {
                        Some(a) => a,
                        None => {
                            // SAFETY: releasing our +1s.
                            unsafe {
                                CFRelease(pixel as CFTypeRef);
                                CFRelease(surface);
                            }
                            return Err(CoreMlError::Prediction(
                                "MLMultiArray initWithPixelBuffer returned nil".into(),
                            ));
                        }
                    },
                    pixel,
                    surface,
                    rows,
                    cols,
                    stride,
                };
                Ok(plane)
            })
        }

        /// Base of the surface's memory. Page-aligned; valid while `self`
        /// lives.
        pub fn base_ptr(&self) -> *const u8 {
            // SAFETY: a getter on a live surface.
            unsafe { IOSurfaceGetBaseAddress(self.surface) as *const u8 }
        }

        /// Bytes allocated behind [`Self::base_ptr`] (page-rounded).
        pub fn alloc_size(&self) -> usize {
            // SAFETY: a getter on a live surface.
            unsafe { IOSurfaceGetAllocSize(self.surface) }
        }

        pub fn rows(&self) -> usize {
            self.rows
        }

        pub fn cols(&self) -> usize {
            self.cols
        }

        /// Halfs between the starts of consecutive rows.
        pub fn stride(&self) -> usize {
            self.stride
        }

        fn as_feature(&self) -> Retained<MLFeatureValue> {
            // SAFETY: wraps a live multi-array.
            unsafe { MLFeatureValue::featureValueWithMultiArray(&self.array) }
        }
    }

    impl Drop for SharedF16Plane {
        fn drop(&mut self) {
            // SAFETY: releasing the +1s taken in `new`. The multi-array
            // keeps its own reference to the pixel buffer until it drops.
            unsafe {
                CFRelease(self.pixel as CFTypeRef);
                CFRelease(self.surface);
            }
        }
    }

    fn check_plane(specs: &[FeatureSpec], name: &str, plane: &SharedF16Plane) -> Result<()> {
        let spec = specs
            .iter()
            .find(|s| s.name == name)
            .ok_or_else(|| CoreMlError::MissingFeature(name.to_string()))?;
        let want = [1, 1, plane.rows, plane.cols];
        match &spec.kind {
            FeatureKind::MultiArray {
                dtype: Some(DType::F16),
                shape: Some(s),
            } if s.as_slice() == want => Ok(()),
            other => Err(CoreMlError::Shape {
                name: format!("{name} ({other:?})"),
                expected: want.to_vec(),
                got: Vec::new(),
            }),
        }
    }

    impl CoreMlModel {
        /// One stateless prediction reading `inputs` and writing `outputs`
        /// in place (`MLPredictionOptions.outputBackings`). Every declared
        /// input and output must be bound, and every plane must match its
        /// feature's fixed F16 shape. Refuses a result Core ML did not
        /// write into the backing, which would otherwise read as stale data.
        pub fn predict_shared(
            &self,
            inputs: &[(&str, &SharedF16Plane)],
            outputs: &[(&str, &SharedF16Plane)],
        ) -> Result<()> {
            for (name, plane) in inputs {
                check_plane(&self.inputs, name, plane)?;
            }
            for (name, plane) in outputs {
                check_plane(&self.outputs, name, plane)?;
            }
            if inputs.len() != self.inputs.len() || outputs.len() != self.outputs.len() {
                return Err(CoreMlError::MissingFeature(
                    "predict_shared binds every input and output".into(),
                ));
            }
            let options_class = AnyClass::get(c"MLPredictionOptions")
                .ok_or_else(|| CoreMlError::Prediction("no MLPredictionOptions".into()))?;
            let guard = self.lock();
            autoreleasepool(|_| {
                let keys: Vec<Retained<NSString>> =
                    inputs.iter().map(|(n, _)| NSString::from_str(n)).collect();
                let values: Vec<Retained<MLFeatureValue>> =
                    inputs.iter().map(|(_, p)| p.as_feature()).collect();
                let key_refs: Vec<&NSString> = keys.iter().map(|k| &**k).collect();
                let value_refs: Vec<&AnyObject> = values.iter().map(|v| v.as_ref()).collect();
                let dict = NSDictionary::<NSString, AnyObject>::from_slices(&key_refs, &value_refs);
                // SAFETY: every value is an MLFeatureValue.
                let provider = guarded("MLDictionaryFeatureProvider init", || unsafe {
                    MLDictionaryFeatureProvider::initWithDictionary_error(
                        MLDictionaryFeatureProvider::alloc(),
                        &dict,
                    )
                })?
                .map_err(|e| CoreMlError::Prediction(ns_error(&e)))?;
                let provider = ProtocolObject::<dyn MLFeatureProvider>::from_ref(&*provider);

                let out_keys: Vec<Retained<NSString>> =
                    outputs.iter().map(|(n, _)| NSString::from_str(n)).collect();
                let out_key_refs: Vec<&NSString> = out_keys.iter().map(|k| &**k).collect();
                let out_refs: Vec<&AnyObject> =
                    outputs.iter().map(|(_, p)| p.array.as_ref()).collect();
                let backings =
                    NSDictionary::<NSString, AnyObject>::from_slices(&out_key_refs, &out_refs);
                // SAFETY: `new` on the MLPredictionOptions class, then its
                // documented outputBackings setter (macOS 11) with a
                // name -> MLMultiArray dictionary.
                let options: Retained<AnyObject> = unsafe { msg_send![options_class, new] };
                let () = unsafe { msg_send![&*options, setOutputBackings: &*backings] };
                // SAFETY: predictionFromFeatures:options:error: with a
                // provider and backings that passed the shape checks.
                let result: std::result::Result<
                    Retained<ProtocolObject<dyn MLFeatureProvider>>,
                    Retained<NSError>,
                > = guarded("MLModel shared prediction", || unsafe {
                    msg_send![&*guard.0, predictionFromFeatures: provider, options: &*options, error: _]
                })?;
                let result = result.map_err(|e| CoreMlError::Prediction(ns_error(&e)))?;
                for ((name, plane), key) in outputs.iter().zip(&out_keys) {
                    // SAFETY: featureValueForName: and multiArrayValue on a
                    // live provider.
                    let same = unsafe { result.featureValueForName(key) }
                        .and_then(|v| unsafe { v.multiArrayValue() })
                        .is_some_and(|a| Retained::as_ptr(&a) == Retained::as_ptr(&plane.array));
                    if !same {
                        return Err(CoreMlError::Prediction(format!(
                            "{name}: Core ML did not write into the output backing"
                        )));
                    }
                }
                Ok(())
            })
        }
    }
}
