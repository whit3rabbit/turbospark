//! Metal device/queue/pipeline-cache context. Ported from the shape of
//! `Infrastructure/Metal/MetalContext.swift`: shaders are compiled from MSL
//! source at runtime (not precompiled into a `.metallib`), matching how
//! Mference itself builds its pipelines, and compute pipeline states are
//! cached by function name so a kernel used every decode step compiles once.

use std::collections::HashMap;

use metal::{
    CommandQueue, ComputePipelineState, Device, FunctionConstantValues, Library,
    MTLResourceOptions, MTLSize,
};

#[derive(Debug)]
pub enum GpuError {
    NoDevice,
    LibraryCompile(String),
    FunctionNotFound(String),
    PipelineCreate(String),
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
        }
    }
}

impl std::error::Error for GpuError {}

/// Owns the Metal device and command queue, and caches one
/// [`ComputePipelineState`] per (library source, function name) pair so
/// repeated dispatches of the same kernel skip recompilation.
pub struct MetalContext {
    device: Device,
    queue: CommandQueue,
    libraries: HashMap<&'static str, Library>,
    pipelines: HashMap<(&'static str, &'static str), ComputePipelineState>,
}

impl MetalContext {
    pub fn new() -> Result<Self, GpuError> {
        let device = Device::system_default().ok_or(GpuError::NoDevice)?;
        let queue = device.new_command_queue();
        Ok(Self {
            device,
            queue,
            libraries: HashMap::new(),
            pipelines: HashMap::new(),
        })
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
    /// helper, e.g. `rms_norm::unused_function_constants`). Returns an
    /// owned (cheaply-cloned, Objective-C reference-counted) handle so
    /// callers don't hold a borrow of the context across the dispatch call
    /// that follows.
    pub fn pipeline(
        &mut self,
        source: &'static str,
        function_name: &'static str,
        constants: &FunctionConstantValues,
    ) -> Result<ComputePipelineState, GpuError> {
        let key = (source, function_name);
        if !self.pipelines.contains_key(&key) {
            let library = self.library(source)?;
            let function = library
                .get_function(function_name, Some(constants.clone()))
                .map_err(|_| GpuError::FunctionNotFound(function_name.to_string()))?;
            let pipeline = self
                .device
                .new_compute_pipeline_state_with_function(&function)
                .map_err(GpuError::PipelineCreate)?;
            self.pipelines.insert(key, pipeline);
        }
        Ok(self.pipelines[&key].clone())
    }

    fn library(&mut self, source: &'static str) -> Result<Library, GpuError> {
        if let Some(lib) = self.libraries.get(source) {
            return Ok(lib.clone());
        }
        let options = metal::CompileOptions::new();
        let library = self
            .device
            .new_library_with_source(source, &options)
            .map_err(GpuError::LibraryCompile)?;
        self.libraries.insert(source, library.clone());
        Ok(library)
    }

    pub fn queue(&self) -> &CommandQueue {
        &self.queue
    }

    pub fn new_buffer_with_data<T>(&self, data: &[T]) -> metal::Buffer {
        let byte_len = std::mem::size_of_val(data) as u64;
        self.device.new_buffer_with_data(
            data.as_ptr().cast(),
            byte_len,
            MTLResourceOptions::StorageModeShared,
        )
    }

    pub fn new_output_buffer(&self, byte_len: u64) -> metal::Buffer {
        self.device
            .new_buffer(byte_len, MTLResourceOptions::StorageModeShared)
    }
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
