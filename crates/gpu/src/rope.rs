//! Host-side dispatch for the `rope_proportional_neox` kernel in
//! `shaders/rope.metal` (vendored verbatim from `Metal/Primitives/rope.metal`).
//! This is Gemma 4's proportional-RoPE convention: NeoX pairing `(i,
//! head_dim/2 + i)` across the full head, frequency divisor `head_dim` —
//! the same convention `mrefrust_compute::rope_neox` implements, so this
//! dispatch is parity-tested directly against it.
//!
//! `rope.metal` also ships `rope_default_neox` (full-head NeoX, no partial
//! rotation) and `rope_neox_subdim` (Qwen's rotary-dim-windowed variant,
//! frequency divisor = rotary_dim rather than head_dim); neither has a
//! matching `mrefrust_compute` reference yet, so only the kernel that does
//! is wired here.

use half::f16;
use metal::{FunctionConstantValues, MTLDataType};

use crate::bytes::{f32_bytes, half_slice_to_le_bytes, read_half_buffer, u32_bytes};
use crate::context::{dispatch_threads_3d, GpuError, MetalContext};

const SOURCE: &str = include_str!("shaders/rope.metal");

/// `rope.metal`'s kernels declare function constants `FC_ROPE_HEAD_DIM`
/// (50, uint), `FC_ROPE_NUM_HEADS` (51, uint), `FC_ROPE_ROTATED_PAIRS` (52,
/// uint), and `FC_ROPE_USE_FC` (53, bool). Setting `FC_ROPE_USE_FC = false`
/// keeps every kernel on its normal runtime-argument path.
fn unused_function_constants() -> FunctionConstantValues {
    let values = FunctionConstantValues::new();
    let zero: u32 = 0;
    let use_fc = false;
    values.set_constant_value_at_index((&zero as *const u32).cast(), MTLDataType::UInt, 50);
    values.set_constant_value_at_index((&zero as *const u32).cast(), MTLDataType::UInt, 51);
    values.set_constant_value_at_index((&zero as *const u32).cast(), MTLDataType::UInt, 52);
    values.set_constant_value_at_index((&use_fc as *const bool).cast(), MTLDataType::Bool, 53);
    values
}

/// Applies Gemma 4's proportional NeoX RoPE in place, dispatched on the
/// GPU via `rope_proportional_neox`. `data` is `[num_tokens, num_heads,
/// head_dim]`; every token uses the same `position` (matching the single-
/// position decode-step call shape the kernel is built for).
#[allow(clippy::too_many_arguments)]
pub fn rope_proportional_neox(
    context: &mut MetalContext,
    data: &[f16],
    position: u32,
    num_tokens: u32,
    num_heads: u32,
    head_dim: u32,
    rotated_pairs: u32,
    theta: f32,
) -> Result<Vec<f16>, GpuError> {
    let data_bytes = half_slice_to_le_bytes(data);
    let buffer = context.new_buffer_with_data(&data_bytes);

    let pipeline = context.pipeline(
        SOURCE,
        "rope_proportional_neox",
        &unused_function_constants(),
    )?;
    dispatch_threads_3d(
        context,
        &pipeline,
        &[(&buffer, 0)],
        &[
            (u32_bytes(&position), 1),
            (u32_bytes(&head_dim), 2),
            (u32_bytes(&num_heads), 3),
            (f32_bytes(&theta), 4),
            (u32_bytes(&rotated_pairs), 5),
        ],
        (
            rotated_pairs.max(1) as u64,
            num_heads.max(1) as u64,
            num_tokens.max(1) as u64,
        ),
        (1, 1, 1),
    );

    Ok(read_half_buffer(&buffer, data.len()))
}
