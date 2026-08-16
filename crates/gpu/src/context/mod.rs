//! Metal device/queue/pipeline-cache context. Ported from the shape of
//! `Infrastructure/Metal/MetalContext.swift`: shaders are compiled from MSL
//! source at runtime (not precompiled into a `.metallib`), matching how
//! Mference itself builds its pipelines, and compute pipeline states are
//! cached by function name so a kernel used every decode step compiles once.

mod buffer_io;
mod device;
mod dispatch;
mod error;
mod pass;

pub use buffer_io::{read_buffer_bytes, read_buffer_f16, read_buffer_f16_into, write_buffer_bytes};
pub use device::MetalContext;
pub use dispatch::{
    dispatch_one_threadgroup_per_row, dispatch_one_threadgroup_per_row_offsets, dispatch_threads_3d,
};
pub use error::{autorelease_pool, GpuError};
pub use pass::{CommittedPass, PassEncoder};
