//! Metal device/queue/pipeline-cache context (`MetalContext`).

use std::cell::RefCell;
use std::collections::HashMap;

use metal::{
    CommandQueue, ComputePipelineState, Device, FunctionConstantValues, Library, MTLResourceOptions,
};

use super::error::GpuError;
use super::pass::PassEncoder;
use crate::dispatch_profile::{self, PassProfile};

/// Owns the Metal device and command queue, and caches one
/// [`ComputePipelineState`] per (library source, function name) pair so
/// repeated dispatches of the same kernel skip recompilation.
pub struct MetalContext {
    device: Device,
    queue: CommandQueue,
    libraries: HashMap<usize, Library>,
    functions: HashMap<CacheKey, SpecializationBucket<metal::Function>>,
    pipelines: HashMap<CacheKey, SpecializationBucket<ComputePipelineState>>,
    buffer_allocations: std::sync::atomic::AtomicU64,
}

/// Every specialization of one cached function, keyed by its caller-supplied
/// constants fingerprint (see [`find`] for why this is a Vec, not a map).
type SpecializationBucket<T> = Vec<(Box<[u8]>, T)>;

/// A shader source plus a function name, keyed by the source's ADDRESS,
/// not its text.
///
/// Every caller passes the same `&'static str` from `include_str!` for a
/// given shader file (the documented contract on
/// [`MetalContext::pipeline`]), so the pointer identifies the file. Hashing
/// the text instead would rehash tens of kilobytes of MSL on every one of
/// the ~900 dispatches a single decoded token encodes.
type CacheKey = (usize, &'static str);

fn cache_key(source: &'static str, function_name: &'static str) -> CacheKey {
    (source.as_ptr() as usize, function_name)
}

/// Looks a specialization up by its constant fingerprint. The bucket holds
/// one entry per distinct fingerprint for that function, which is a handful
/// at most, so a linear scan beats hashing the bytes and never allocates on
/// the hit path.
fn find<'a, T>(bucket: Option<&'a SpecializationBucket<T>>, constants_key: &[u8]) -> Option<&'a T> {
    bucket?
        .iter()
        .find(|(key, _)| &**key == constants_key)
        .map(|(_, value)| value)
}

impl MetalContext {
    pub fn new() -> Result<Self, GpuError> {
        let device = Device::system_default().ok_or(GpuError::NoDevice)?;
        let queue = device.new_command_queue();
        Ok(Self {
            device,
            queue,
            libraries: HashMap::new(),
            functions: HashMap::new(),
            pipelines: HashMap::new(),
            buffer_allocations: std::sync::atomic::AtomicU64::new(0),
        })
    }

    fn function(
        &mut self,
        source: &'static str,
        function_name: &'static str,
        constants: &FunctionConstantValues,
        constants_key: &[u8],
    ) -> Result<metal::Function, GpuError> {
        let key = cache_key(source, function_name);
        if let Some(function) = find(self.functions.get(&key), constants_key) {
            return Ok(function.clone());
        }
        let library = self.library(source)?;
        let function = library
            .get_function(function_name, Some(constants.clone()))
            .map_err(|_| GpuError::FunctionNotFound(function_name.to_string()))?;
        self.functions
            .entry(key)
            .or_default()
            .push((constants_key.into(), function.clone()));
        Ok(function)
    }

    /// Builds (cached per function) an argument encoder for the argument
    /// buffer bound at `buffer_index` of `function_name` -- how the MoE
    /// kernels receive their `RoutedBlobs` pointer array, matching the
    /// Swift original's `makeArgumentEncoder(bufferIndex:)`.
    pub fn argument_encoder(
        &mut self,
        source: &'static str,
        function_name: &'static str,
        constants: &FunctionConstantValues,
        constants_key: &[u8],
        buffer_index: u64,
    ) -> Result<metal::ArgumentEncoder, GpuError> {
        let function = self.function(source, function_name, constants, constants_key)?;
        Ok(function.new_argument_encoder(buffer_index))
    }

    /// Running count of Metal buffers this context has allocated
    /// (`new_buffer_with_data` + `new_output_buffer`). The decode hot path
    /// is required to allocate none: steady-state tests assert this stays
    /// flat across generated tokens.
    pub fn buffer_allocation_count(&self) -> u64 {
        self.buffer_allocations
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Returns reference to the active Metal device.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Compiles (once) and returns the pipeline state for `function_name`
    /// inside the MSL `source`. `source` is used as the library cache key,
    /// so pass the same `&'static str` (e.g. `include_str!(...)`) every
    /// call for a given shader file. `constants` specializes every
    /// function-constant index the shader declares -- Metal requires all of
    /// them to be set before a pipeline state can be built, even indices
    /// the shader's own `is_function_constant_defined` guard means are
    /// conditionally unread (see each kernel module's own constants
    /// helper, e.g. `rms_norm::unused_function_constants`).
    ///
    /// `FunctionConstantValues` offers no introspection, so callers must
    /// also pass `constants_key`: a byte fingerprint of every value that
    /// went into `constants` (empty when the constants are the same fixed
    /// set every call). Two calls with the same function but different
    /// constant values then cache as distinct pipelines -- the Swift
    /// original keys its PSO cache the same way (name + sorted constants).
    ///
    /// Returns an owned (cheaply-cloned, Objective-C reference-counted)
    /// handle so callers don't hold a borrow of the context across the
    /// dispatch call that follows.
    pub fn pipeline(
        &mut self,
        source: &'static str,
        function_name: &'static str,
        constants: &FunctionConstantValues,
        constants_key: &[u8],
    ) -> Result<ComputePipelineState, GpuError> {
        let key = cache_key(source, function_name);
        if let Some(pipeline) = find(self.pipelines.get(&key), constants_key) {
            return Ok(pipeline.clone());
        }
        let function = self.function(source, function_name, constants, constants_key)?;
        let pipeline = self
            .device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(GpuError::PipelineCreate)?;
        dispatch_profile::register_pipeline(&pipeline, function_name);
        self.pipelines
            .entry(key)
            .or_default()
            .push((constants_key.into(), pipeline.clone()));
        Ok(pipeline)
    }

    fn library(&mut self, source: &'static str) -> Result<Library, GpuError> {
        if let Some(lib) = self.libraries.get(&(source.as_ptr() as usize)) {
            return Ok(lib.clone());
        }
        let options = metal::CompileOptions::new();
        let library = self
            .device
            .new_library_with_source(source, &options)
            .map_err(GpuError::LibraryCompile)?;
        self.libraries
            .insert(source.as_ptr() as usize, library.clone());
        Ok(library)
    }

    /// Returns reference to the command queue.
    pub fn queue(&self) -> &CommandQueue {
        &self.queue
    }

    /// Opens one command buffer + one serial compute encoder to batch many
    /// kernel dispatches into a single submission (the Swift original
    /// encodes a whole layer, or more, per command buffer instead of one
    /// kernel per buffer with a synchronous wait each). Serial dispatch
    /// order within the encoder guarantees each dispatch sees the previous
    /// one's writes.
    pub fn begin_pass(&self) -> PassEncoder {
        self.begin_pass_labeled("pass")
    }

    /// [`Self::begin_pass`] with a name for this command buffer's role in
    /// the decode step (`cb1`, `routed`, ...). The label goes onto the
    /// `MTLCommandBuffer` (so a GPU capture or Instruments trace shows it
    /// instead of an anonymous buffer) and groups the rows of
    /// `MFERENCE_DISPATCH_PROFILE=1`'s per-dispatch report.
    pub fn begin_pass_labeled(&self, label: &'static str) -> PassEncoder {
        let command_buffer = self.queue.new_command_buffer().to_owned();
        command_buffer.set_label(label);
        let encoder = command_buffer.new_compute_command_encoder().to_owned();
        PassEncoder::new(
            command_buffer,
            RefCell::new(encoder),
            dispatch_profile::enabled()
                .then(|| PassProfile::new(&self.device, label).map(RefCell::new))
                .flatten(),
        )
    }

    /// Allocates a shared Metal GPU buffer initialized with `data` bytes.
    pub fn new_buffer_with_data<T>(&self, data: &[T]) -> metal::Buffer {
        self.buffer_allocations
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let byte_len = std::mem::size_of_val(data) as u64;
        self.device.new_buffer_with_data(
            data.as_ptr().cast(),
            byte_len,
            MTLResourceOptions::StorageModeShared,
        )
    }

    /// Allocates an uninitialized shared Metal GPU output buffer of `byte_len` bytes.
    pub fn new_output_buffer(&self, byte_len: u64) -> metal::Buffer {
        self.buffer_allocations
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.device
            .new_buffer(byte_len, MTLResourceOptions::StorageModeShared)
    }
}
