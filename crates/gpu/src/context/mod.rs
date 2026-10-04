//! Metal device/queue/pipeline-cache context. In-memory caches avoid repeated
//! compilation within a runner; optional packaged libraries and device-built
//! archives also avoid compilation across runner lifetimes.

mod buffer_io;
mod device;
mod dispatch;
mod error;
mod pass;
mod pipeline_cache;

pub use buffer_io::{read_buffer_bytes, read_buffer_f16, read_buffer_f16_into, write_buffer_bytes};
pub use device::MetalContext;
pub use dispatch::{
    dispatch_one_threadgroup_per_row, dispatch_one_threadgroup_per_row_offsets, dispatch_threads_3d,
};
pub use error::{autorelease_pool, GpuError};
pub use pass::{CommittedPass, PassEncoder};
pub use pipeline_cache::MetalCompilationStats;
