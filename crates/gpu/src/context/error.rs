//! GPU error types and autorelease pool helper.

/// Errors occurring during Metal device, pipeline, or buffer operations.
#[derive(Debug)]
pub enum GpuError {
    /// Metal device is not available on this host.
    NoDevice,
    /// MSL shader library compilation error with details.
    LibraryCompile(String),
    /// Shader function name was not found in compiled MSL library.
    FunctionNotFound(String),
    /// Compute pipeline state creation error with details.
    PipelineCreate(String),
    /// Buffer allocation error with details.
    BufferCreate(String),
    /// An argument buffer was encoded with a different reflected ABI than
    /// the one used to allocate it.
    ArgumentBufferMismatch(String),
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
            GpuError::ArgumentBufferMismatch(detail) => {
                write!(f, "argument buffer layout mismatch: {detail}")
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
