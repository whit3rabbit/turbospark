//! Standalone single-pass Metal kernel dispatch helpers.

use metal::{ComputePipelineState, MTLSize};

use super::device::MetalContext;
use super::error::autorelease_pool;
use super::pass::warn_on_command_buffer_error;

/// One threadgroup per row/head, `threads_per_group` threads each -- the
/// dispatch shape every kernel in `rmsnorm.metal` assumes.
pub fn dispatch_one_threadgroup_per_row(
    context: &MetalContext,
    pipeline: &ComputePipelineState,
    buffers: &[(&metal::Buffer, u64)],
    bytes: &[(&[u8], u64)],
    rows: u64,
    threads_per_group: u64,
) {
    autorelease_pool(|| {
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
        warn_on_command_buffer_error(command_buffer);
    });
}

/// Like [`dispatch_one_threadgroup_per_row`], but each buffer binding
/// carries an explicit byte offset -- the shape used to bind tensors as
/// offsets into the one shared resident-weights `MTLBuffer`
/// (`ResidentGpuWeights`) instead of staging copies. Offsets need only
/// match the kernel argument's element alignment for most callers (2 for
/// `half`/`bfloat`, 1 for `uint8_t`), which the `.gturbo` layout
/// guarantees. **The INT4 GEMV is the one caller that needs MORE**:
/// `dequant_int4.metal`'s vectorized read takes `x` as `half4`-aligned
/// (8 bytes, `lane*8` elements), so `dequant_int4_gemv.rs`'s encoder
/// asserts that offset explicitly rather than relying on this doc alone.
pub fn dispatch_one_threadgroup_per_row_offsets(
    context: &MetalContext,
    pipeline: &ComputePipelineState,
    buffers: &[(&metal::Buffer, u64, u64)],
    bytes: &[(&[u8], u64)],
    rows: u64,
    threads_per_group: u64,
) {
    autorelease_pool(|| {
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
        warn_on_command_buffer_error(command_buffer);
    });
}

/// One thread per `(x, y, z)` grid cell, via `dispatchThreads` (not
/// threadgroup-quantized) -- the dispatch shape `rope.metal`'s kernels
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
    autorelease_pool(|| {
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
        warn_on_command_buffer_error(command_buffer);
    });
}
