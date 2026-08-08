//! Metal device/queue/pipeline-cache context. Ported from the shape of
//! `Infrastructure/Metal/MetalContext.swift`: shaders are compiled from MSL
//! source at runtime (not precompiled into a `.metallib`), matching how
//! Mference itself builds its pipelines, and compute pipeline states are
//! cached by function name so a kernel used every decode step compiles once.

use std::cell::RefCell;
use std::collections::HashMap;

use metal::{
    CommandQueue, ComputePipelineState, Device, FunctionConstantValues, Library,
    MTLResourceOptions, MTLSize,
};

use crate::dispatch_profile::{self, PassProfile};

#[derive(Debug)]
pub enum GpuError {
    NoDevice,
    LibraryCompile(String),
    FunctionNotFound(String),
    PipelineCreate(String),
    BufferCreate(String),
}

impl std::fmt::Display for GpuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GpuError::NoDevice => write!(f, "no Metal device available"),
            GpuError::LibraryCompile(detail) => write!(f, "MSL compile failed: {detail}"),
            GpuError::FunctionNotFound(name) => write!(f, "kernel function not found: {name}"),
            GpuError::PipelineCreate(detail) => {
                write!(f, "pipeline state creation failed: {detail}")
            }
            GpuError::BufferCreate(detail) => {
                write!(f, "buffer creation failed: {detail}")
            }
        }
    }
}

impl std::error::Error for GpuError {}

/// Runs `f` inside an Objective-C autorelease pool.
///
/// Not optional bookkeeping -- a correctness requirement for any loop that
/// encodes work repeatedly. `MTLCommandQueue.commandBuffer` and
/// `MTLCommandBuffer.computeCommandEncoder` return AUTORELEASED objects:
/// the `metal` crate's `to_owned()` adds our own retain, and dropping the
/// wrapper drops it, but the pool's retain lives until the pool drains.
/// Swift drains one per run-loop turn; a plain Rust binary has exactly one
/// pool, around `main`, so without an inner pool every command buffer and
/// encoder the process ever created stays alive until exit.
///
/// Measured cost of getting this wrong on the real Gemma 4 install: ~6 KiB
/// per command buffer, 31 command buffers per token, ~180 KiB per decoded
/// token, growing without bound. `crates/bench/tests/memory_oracle.rs`'s
/// steady-state guard is what catches a regression here.
pub fn autorelease_pool<R>(f: impl FnOnce() -> R) -> R {
    metal::objc::rc::autoreleasepool(f)
}

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
    /// buffer bound at `buffer_index` of `function_name` — how the MoE
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

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Compiles (once) and returns the pipeline state for `function_name`
    /// inside the MSL `source`. `source` is used as the library cache key,
    /// so pass the same `&'static str` (e.g. `include_str!(...)`) every
    /// call for a given shader file. `constants` specializes every
    /// function-constant index the shader declares — Metal requires all of
    /// them to be set before a pipeline state can be built, even indices
    /// the shader's own `is_function_constant_defined` guard means are
    /// conditionally unread (see each kernel module's own constants
    /// helper, e.g. `rms_norm::unused_function_constants`).
    ///
    /// `FunctionConstantValues` offers no introspection, so callers must
    /// also pass `constants_key`: a byte fingerprint of every value that
    /// went into `constants` (empty when the constants are the same fixed
    /// set every call). Two calls with the same function but different
    /// constant values then cache as distinct pipelines — the Swift
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
        PassEncoder {
            command_buffer,
            encoder: RefCell::new(encoder),
            profile: dispatch_profile::enabled()
                .then(|| PassProfile::new(&self.device, label).map(RefCell::new))
                .flatten(),
            ended: std::cell::Cell::new(false),
        }
    }

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

    pub fn new_output_buffer(&self, byte_len: u64) -> metal::Buffer {
        self.buffer_allocations
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.device
            .new_buffer(byte_len, MTLResourceOptions::StorageModeShared)
    }
}

/// An open command buffer + serial compute encoder; see
/// [`MetalContext::begin_pass`]. Every `encode_*` call appends one
/// dispatch; nothing runs until [`PassEncoder::commit_and_wait`] (or
/// [`PassEncoder::commit`]).
pub struct PassEncoder {
    command_buffer: metal::CommandBuffer,
    /// Rebound per dispatch under `MFERENCE_DISPATCH_PROFILE=1` (this
    /// device can only sample counters at encoder boundaries), hence the
    /// cell; exactly one encoder for the whole pass otherwise.
    encoder: RefCell<metal::ComputeCommandEncoder>,
    profile: Option<RefCell<PassProfile>>,
    /// Whether `endEncoding` has been sent. Read by the `Drop` impl below,
    /// which is what stops an error path aborting the process.
    ended: std::cell::Cell<bool>,
}

impl PassEncoder {
    /// Renames this command buffer mid-pass, for the case where one
    /// buffer's role changes partway (the final head rides whatever
    /// buffer the last layer left open). Affects labelling only.
    pub fn relabel(&self, label: &'static str) {
        self.command_buffer.set_label(label);
        if let Some(profile) = &self.profile {
            profile.borrow_mut().relabel(label);
        }
    }

    /// Under profiling, closes the running encoder and opens a fresh
    /// timestamped one for the dispatch about to be encoded. A no-op
    /// otherwise, which is the production path.
    fn begin_dispatch(&self, pipeline: &ComputePipelineState) {
        let Some(profile) = &self.profile else {
            return;
        };
        profile.borrow_mut().begin_dispatch(
            &self.command_buffer,
            &mut self.encoder.borrow_mut(),
            pipeline,
        );
    }

    /// Appends one `dispatchThreadgroups` call. Buffer bindings are
    /// `(buffer, argument index, byte offset)`.
    pub fn encode_threadgroups(
        &self,
        pipeline: &ComputePipelineState,
        buffers: &[(&metal::Buffer, u64, u64)],
        bytes: &[(&[u8], u64)],
        threadgroups: u64,
        threads_per_group: u64,
    ) {
        self.begin_dispatch(pipeline);
        let encoder = self.encoder.borrow();
        encoder.set_compute_pipeline_state(pipeline);
        for &(buffer, index, offset) in buffers {
            encoder.set_buffer(index, Some(buffer), offset);
        }
        for &(data, index) in bytes {
            encoder.set_bytes(index, data.len() as u64, data.as_ptr().cast());
        }
        encoder.dispatch_thread_groups(
            MTLSize::new(threadgroups, 1, 1),
            MTLSize::new(threads_per_group, 1, 1),
        );
    }

    /// [`Self::encode_threadgroups`] with 3D threadgroup and threadgroup-size
    /// counts. The GDN kernels need it: their grids are `(head, row)` and
    /// their threadgroups `(32, 4)` or `(128, 1)`, which the 1D form cannot
    /// express.
    pub fn encode_threadgroups_3d(
        &self,
        pipeline: &ComputePipelineState,
        buffers: &[(&metal::Buffer, u64, u64)],
        bytes: &[(&[u8], u64)],
        threadgroups: (u64, u64, u64),
        threads_per_group: (u64, u64, u64),
    ) {
        self.begin_dispatch(pipeline);
        let encoder = self.encoder.borrow();
        encoder.set_compute_pipeline_state(pipeline);
        for &(buffer, index, offset) in buffers {
            encoder.set_buffer(index, Some(buffer), offset);
        }
        for &(data, index) in bytes {
            encoder.set_bytes(index, data.len() as u64, data.as_ptr().cast());
        }
        encoder.dispatch_thread_groups(
            MTLSize::new(threadgroups.0, threadgroups.1, threadgroups.2),
            MTLSize::new(
                threads_per_group.0,
                threads_per_group.1,
                threads_per_group.2,
            ),
        );
    }

    /// Appends one `dispatchThreads` call (non-threadgroup-quantized grid).
    pub fn encode_threads_3d(
        &self,
        pipeline: &ComputePipelineState,
        buffers: &[(&metal::Buffer, u64, u64)],
        bytes: &[(&[u8], u64)],
        grid: (u64, u64, u64),
        threadgroup: (u64, u64, u64),
    ) {
        self.begin_dispatch(pipeline);
        let encoder = self.encoder.borrow();
        encoder.set_compute_pipeline_state(pipeline);
        for &(buffer, index, offset) in buffers {
            encoder.set_buffer(index, Some(buffer), offset);
        }
        for &(data, index) in bytes {
            encoder.set_bytes(index, data.len() as u64, data.as_ptr().cast());
        }
        encoder.dispatch_threads(
            MTLSize::new(grid.0, grid.1, grid.2),
            MTLSize::new(threadgroup.0, threadgroup.1, threadgroup.2),
        );
    }

    /// Declares a buffer referenced only indirectly (through an argument
    /// buffer's pointer array) as read by the pass — Metal's
    /// `useResource(_:usage:.read)`, required for the MoE expert blobs.
    pub fn use_read_buffer(&self, buffer: &metal::Buffer) {
        self.encoder
            .borrow()
            .use_resource(buffer, metal::MTLResourceUsage::Read);
        // Resource state is per encoder, and profiling opens one per
        // dispatch, so it has to re-declare this on each of them.
        if let Some(profile) = &self.profile {
            profile.borrow_mut().note_used_read(buffer);
        }
    }

    /// Ends encoding, commits, and blocks until the GPU finishes. Shared-
    /// storage outputs are CPU-readable after this returns.
    pub fn commit_and_wait(self) {
        self.commit().wait();
    }

    /// [`Self::commit_and_wait`] that also reports the buffer's GPU-side
    /// busy interval in seconds. See [`CommittedPass::wait_with_gpu_time`].
    pub fn commit_and_wait_with_gpu_time(self) -> f64 {
        self.commit().wait_with_gpu_time()
    }

    /// Ends encoding and commits without waiting, handing back something
    /// the caller can wait on later. Buffers committed to one queue execute
    /// in commit order, so a caller can queue follow-on GPU work and then
    /// wait on an earlier buffer to be woken as soon as *its* results are
    /// ready, with the later work still running.
    pub fn commit(mut self) -> CommittedPass {
        self.encoder.borrow().end_encoding();
        self.ended.set(true);
        self.command_buffer.commit();
        // Timestamps can only be resolved once the buffer has completed,
        // and a pass may be committed and never waited on (the shared and
        // hit-expert buffers are), so profiling waits here rather than
        // losing those dispatches. This is the serialization the module
        // doc warns about: profiled runs are not throughput runs.
        if let Some(profile) = self.profile.take() {
            self.command_buffer.wait_until_completed();
            profile.into_inner().resolve();
        }
        CommittedPass {
            command_buffer: self.command_buffer.clone(),
        }
    }
}

/// Ends encoding on a pass that is dropped without being committed, which
/// is what every `?` inside an encode sequence does.
///
/// Without this, Metal aborts the process from `-[_MTLCommandEncoder
/// dealloc]` with "Command encoder released without endEncoding", and that
/// assertion is all anyone sees: the real error is still travelling up the
/// stack when the encoder is deallocated, so it never reaches a caller that
/// could print it. Found while opening the first mixed-block-type GGUF
/// install, where a one-line dispatch refusal presented as a Metal crash.
impl Drop for PassEncoder {
    fn drop(&mut self) {
        if !self.ended.get() {
            self.encoder.borrow().end_encoding();
            self.ended.set(true);
        }
    }
}

/// A committed, not-yet-waited-on command buffer. Dropping it without
/// waiting is fine: the queue keeps the buffer alive until it completes.
pub struct CommittedPass {
    command_buffer: metal::CommandBuffer,
}

impl CommittedPass {
    /// Blocks until this buffer finishes. Its shared-storage outputs are
    /// CPU-readable after this returns; buffers committed after it may
    /// still be running.
    pub fn wait(self) {
        self.command_buffer.wait_until_completed();
    }

    /// [`Self::wait`] that also reports the buffer's GPU-side busy
    /// interval (`GPUEndTime - GPUStartTime`) in seconds, the per-buffer
    /// attribution the wall-clock wait cannot give (a wait on a buffer
    /// queued behind others pays for all of them). The accessors are only
    /// valid after completion, hence wait-then-read; metal-rs 0.33 does
    /// not bind them, hence the raw `msg_send`. Returns 0.0 if the device
    /// reports nothing.
    // The allow is for objc's `sel_impl!`, whose expansion carries a
    // `cfg(feature = "cargo-clippy")` this crate does not declare.
    #[allow(unexpected_cfgs)]
    pub fn wait_with_gpu_time(self) -> f64 {
        use metal::objc::{msg_send, sel, sel_impl};
        self.command_buffer.wait_until_completed();
        // SAFETY: `GPUStartTime`/`GPUEndTime` are documented
        // `CFTimeInterval` (f64) accessors on `MTLCommandBuffer`, called
        // on a live, completed buffer.
        #[allow(unsafe_code)]
        let (start, end): (f64, f64) = unsafe {
            let cb: &metal::CommandBufferRef = &self.command_buffer;
            (msg_send![cb, GPUStartTime], msg_send![cb, GPUEndTime])
        };
        (end - start).max(0.0)
    }
}

/// Host-writes `bytes` into a shared-storage buffer at `offset` (a plain
/// memcpy into unified memory). Callers sequence this against GPU work
/// themselves: write before committing the pass that reads it.
pub fn write_buffer_bytes(buffer: &metal::Buffer, offset: usize, bytes: &[u8]) {
    assert!(offset + bytes.len() <= buffer.length() as usize);
    // SAFETY: bounds asserted above against a live shared-storage
    // `MTLBuffer`'s allocation; the source and destination never overlap
    // (one is a Rust slice, the other a Metal allocation).
    #[allow(unsafe_code)]
    unsafe {
        std::ptr::copy_nonoverlapping(
            bytes.as_ptr(),
            (buffer.contents() as *mut u8).add(offset),
            bytes.len(),
        );
    }
}

/// Host-reads `count` halfs from a shared-storage buffer starting at
/// `byte_offset`. Only valid after the pass that wrote them completed.
pub fn read_buffer_f16(buffer: &metal::Buffer, byte_offset: usize, count: usize) -> Vec<half::f16> {
    assert!(byte_offset + count * 2 <= buffer.length() as usize);
    // SAFETY: bounds asserted above; u16 has no invalid bit patterns and
    // the 2-byte alignment holds for any offset this crate binds halfs at.
    #[allow(unsafe_code)]
    let bits = unsafe {
        std::slice::from_raw_parts(
            (buffer.contents() as *const u8).add(byte_offset) as *const u16,
            count,
        )
    };
    bits.iter().map(|&b| half::f16::from_bits(b)).collect()
}

/// One threadgroup per row/head, `threads_per_group` threads each — the
/// dispatch shape every kernel in `rmsnorm.metal` assumes.
pub fn dispatch_one_threadgroup_per_row(
    context: &MetalContext,
    pipeline: &ComputePipelineState,
    buffers: &[(&metal::Buffer, u64)],
    bytes: &[(&[u8], u64)],
    rows: u64,
    threads_per_group: u64,
) {
    let command_buffer = context.queue().new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    for &(buffer, index) in buffers {
        encoder.set_buffer(index, Some(buffer), 0);
    }
    for &(data, index) in bytes {
        encoder.set_bytes(index, data.len() as u64, data.as_ptr().cast());
    }
    encoder.dispatch_thread_groups(
        MTLSize::new(rows, 1, 1),
        MTLSize::new(threads_per_group, 1, 1),
    );
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();
}

/// Like [`dispatch_one_threadgroup_per_row`], but each buffer binding
/// carries an explicit byte offset — the shape used to bind tensors as
/// offsets into the one shared resident-weights `MTLBuffer`
/// (`ResidentGpuWeights`) instead of staging copies. Offsets need only
/// match the kernel argument's element alignment (2 for `half`/`bfloat`,
/// 1 for `uint8_t`), which the `.gturbo` layout guarantees.
pub fn dispatch_one_threadgroup_per_row_offsets(
    context: &MetalContext,
    pipeline: &ComputePipelineState,
    buffers: &[(&metal::Buffer, u64, u64)],
    bytes: &[(&[u8], u64)],
    rows: u64,
    threads_per_group: u64,
) {
    let command_buffer = context.queue().new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    for &(buffer, index, offset) in buffers {
        encoder.set_buffer(index, Some(buffer), offset);
    }
    for &(data, index) in bytes {
        encoder.set_bytes(index, data.len() as u64, data.as_ptr().cast());
    }
    encoder.dispatch_thread_groups(
        MTLSize::new(rows, 1, 1),
        MTLSize::new(threads_per_group, 1, 1),
    );
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();
}

/// One thread per `(x, y, z)` grid cell, via `dispatchThreads` (not
/// threadgroup-quantized) — the dispatch shape `rope.metal`'s kernels
/// assume, addressed by `[[thread_position_in_grid]]`.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_threads_3d(
    context: &MetalContext,
    pipeline: &ComputePipelineState,
    buffers: &[(&metal::Buffer, u64)],
    bytes: &[(&[u8], u64)],
    grid: (u64, u64, u64),
    threadgroup: (u64, u64, u64),
) {
    let command_buffer = context.queue().new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    for &(buffer, index) in buffers {
        encoder.set_buffer(index, Some(buffer), 0);
    }
    for &(data, index) in bytes {
        encoder.set_bytes(index, data.len() as u64, data.as_ptr().cast());
    }
    encoder.dispatch_threads(
        MTLSize::new(grid.0, grid.1, grid.2),
        MTLSize::new(threadgroup.0, threadgroup.1, threadgroup.2),
    );
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();
}
