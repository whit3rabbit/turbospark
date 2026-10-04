//! Metal device/queue/pipeline-cache context (`MetalContext`).

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use metal::{
    CommandQueue, ComputePipelineDescriptor, ComputePipelineState, Device, FunctionConstantValues,
    Library, MTLPipelineOption, MTLResourceOptions,
};

use super::error::GpuError;
use super::pass::PassEncoder;
use super::pipeline_cache::{MetalCompilationStats, PipelineDiskCache};
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
    compiler_identities: HashMap<usize, String>,
    disk_cache: Option<PipelineDiskCache>,
    precise_math: bool,
    use_precompiled: bool,
    stats: MetalCompilationStats,
    buffer_allocations: AtomicU64,
    buffer_allocation_ns: AtomicU64,
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
        let disk_cache = PipelineDiskCache::from_environment(&device);
        let precise_math = std::env::var_os("TURBOSPARK_METAL_PRECISE_MATH").is_some();
        let use_precompiled = std::env::var("TURBOSPARK_METAL_PRECOMPILED").as_deref() == Ok("1");
        Ok(Self::with_options(
            device,
            disk_cache,
            precise_math,
            use_precompiled,
        ))
    }

    fn with_options(
        device: Device,
        disk_cache: Option<PipelineDiskCache>,
        precise_math: bool,
        use_precompiled: bool,
    ) -> Self {
        let queue = device.new_command_queue();
        let stats = MetalCompilationStats {
            cache_enabled: disk_cache.is_some(),
            ..MetalCompilationStats::default()
        };
        Self {
            device,
            queue,
            libraries: HashMap::new(),
            functions: HashMap::new(),
            pipelines: HashMap::new(),
            compiler_identities: HashMap::new(),
            disk_cache,
            precise_math,
            use_precompiled,
            stats,
            buffer_allocations: AtomicU64::new(0),
            buffer_allocation_ns: AtomicU64::new(0),
        }
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
            self.stats.in_memory_function_hits += 1;
            return Ok(function.clone());
        }
        let library = self.library(source)?;
        let started = Instant::now();
        let function = library
            .get_function(function_name, Some(constants.clone()))
            .map_err(|_| GpuError::FunctionNotFound(function_name.to_string()))?;
        self.stats.function_specializations += 1;
        self.stats.function_specialization_ms += started.elapsed().as_secs_f64() * 1000.0;
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
        self.buffer_allocations.load(Ordering::Relaxed)
    }

    /// Snapshot of host startup compilation work, including verified disk hits.
    pub fn compilation_stats(&self) -> MetalCompilationStats {
        MetalCompilationStats {
            buffer_allocations: self.buffer_allocation_count(),
            buffer_allocation_ms: self.buffer_allocation_ns.load(Ordering::Relaxed) as f64 / 1e6,
            ..self.stats
        }
    }

    /// Diagnostic snapshot; this never allocates a GPU buffer or runs a kernel.
    pub fn pipeline_inventory(&self) -> Vec<(String, Vec<u8>)> {
        let mut entries: Vec<_> = self
            .pipelines
            .iter()
            .flat_map(|((_, name), bucket)| {
                bucket
                    .iter()
                    .map(move |(key, _)| (name.to_string(), key.to_vec()))
            })
            .collect();
        entries.sort();
        entries
    }

    /// Publish newly encountered specializations at a startup or first-inference
    /// boundary. Cache write failures remain diagnostic and never fail inference.
    pub fn flush_pipeline_cache(&mut self) {
        if let Some(cache) = &mut self.disk_cache {
            super::error::autorelease_pool(|| cache.flush(&mut self.stats));
        }
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
            self.stats.in_memory_pipeline_hits += 1;
            return Ok(pipeline.clone());
        }
        let function = self.function(source, function_name, constants, constants_key)?;
        let archive_key = self.disk_cache.as_mut().map(|cache| {
            cache.entry_key(
                source,
                function_name,
                constants_key,
                &self.compiler_identities[&(source.as_ptr() as usize)],
            )
        });
        let archive = archive_key.as_ref().and_then(|key| {
            self.disk_cache
                .as_ref()
                .and_then(|cache| cache.load(&self.device, key, &mut self.stats))
        });
        let descriptor = ComputePipelineDescriptor::new();
        descriptor.set_compute_function(Some(&function));
        let started = Instant::now();
        let archived_pipeline = archive.as_ref().and_then(|archive| {
            descriptor.set_binary_archives(&[archive]);
            // Fail-on-miss makes archive statistics actual reuse evidence. The
            // reflection request keeps metal-rs's returned reflection non-null.
            self.device
                .new_compute_pipeline_state_with_reflection(
                    &descriptor,
                    MTLPipelineOption::FailOnBinaryArchiveMiss | MTLPipelineOption::ArgumentInfo,
                )
                .ok()
                .map(|(pipeline, _)| pipeline)
        });
        let archive_hit = archived_pipeline.is_some();
        if archive_key.is_some() {
            if archive_hit {
                self.stats.archive_hits += 1;
            } else {
                self.stats.archive_misses += 1;
            }
        }
        let pipeline = match archived_pipeline {
            Some(pipeline) => pipeline,
            None => self
                .device
                .new_compute_pipeline_state_with_function(&function)
                .map_err(GpuError::PipelineCreate)?,
        };
        self.stats.pipeline_creations += 1;
        self.stats.pipeline_create_ms += started.elapsed().as_secs_f64() * 1000.0;
        if !archive_hit {
            if let (Some(cache), Some(archive_key)) = (&mut self.disk_cache, archive_key) {
                descriptor.set_binary_archives(&[]);
                cache.collect(&self.device, archive_key, &descriptor, &mut self.stats);
            }
        }
        dispatch_profile::register_pipeline(&pipeline, function_name);
        self.pipelines
            .entry(key)
            .or_default()
            .push((constants_key.into(), pipeline.clone()));
        Ok(pipeline)
    }

    fn library(&mut self, source: &'static str) -> Result<Library, GpuError> {
        if let Some(lib) = self.libraries.get(&(source.as_ptr() as usize)) {
            self.stats.in_memory_library_hits += 1;
            return Ok(lib.clone());
        }
        let options = metal::CompileOptions::new();
        // Diagnostic only: image parity can opt out of Metal's relaxed math
        // mode to separate compiler arithmetic from model/layout drift. Keep
        // the platform default unless the caller explicitly requests this
        // experiment.
        if self.precise_math {
            options.set_fast_math_enabled(false);
        }
        let precompiled = if self.use_precompiled && !self.precise_math {
            let started = Instant::now();
            let library = crate::precompiled::load(&self.device, source);
            self.stats.library_load_ms += started.elapsed().as_secs_f64() * 1000.0;
            library
        } else {
            None
        };
        let compiler_identity;
        let library = if let Some(library) = precompiled {
            self.stats.library_loads += 1;
            compiler_identity = crate::precompiled::compiler_identity(source)
                .expect("loaded bundled library has a compiler identity")
                .to_string();
            library
        } else {
            let started = Instant::now();
            let library = self
                .device
                .new_library_with_source(source, &options)
                .map_err(GpuError::LibraryCompile)?;
            self.stats.library_compiles += 1;
            self.stats.library_compile_ms += started.elapsed().as_secs_f64() * 1000.0;
            compiler_identity = format!(
                "source-v1|metal{}|fast-math={}",
                options.language_version() as u64,
                options.is_fast_math_enabled()
            );
            library
        };
        if self.disk_cache.is_some() {
            self.compiler_identities
                .insert(source.as_ptr() as usize, compiler_identity);
        }
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
    /// `TURBOSPARK_DISPATCH_PROFILE=1`'s per-dispatch report.
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
        let started = Instant::now();
        self.buffer_allocations.fetch_add(1, Ordering::Relaxed);
        let byte_len = std::mem::size_of_val(data) as u64;
        let buffer = self.device.new_buffer_with_data(
            data.as_ptr().cast(),
            byte_len,
            MTLResourceOptions::StorageModeShared,
        );
        self.buffer_allocation_ns
            .fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
        buffer
    }

    /// Allocates an uninitialized shared Metal GPU output buffer of `byte_len` bytes.
    pub fn new_output_buffer(&self, byte_len: u64) -> metal::Buffer {
        let started = Instant::now();
        self.buffer_allocations.fetch_add(1, Ordering::Relaxed);
        let buffer = self
            .device
            .new_buffer(byte_len, MTLResourceOptions::StorageModeShared);
        self.buffer_allocation_ns
            .fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
        buffer
    }
}

impl Drop for MetalContext {
    fn drop(&mut self) {
        self.flush_pipeline_cache();
    }
}

#[cfg(test)]
#[path = "device_cache_tests.rs"]
mod cache_tests;
