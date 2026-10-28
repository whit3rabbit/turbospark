//! `lfm3`'s gated short-convolution block (mask 6).
//!
//! `in_proj` GEMV to `[B | C | xx]`, then one kernel that forms `B * xx`,
//! convolves it causally over the last `taps` rows (the tail plus this
//! row) with NO activation, multiplies by `C` and shifts the tail, then the
//! `out_proj` GEMV. The tail is the layer's whole recurrent state.

use model_io::{ArchConfig, ResidentIndex};

use crate::families::lfm3::layer_tensor;
use crate::families::lfm3::state::RealLfm3State;
use crate::real_forward_dispatch::encode_gemv_any;
use crate::real_forward_types::{DecodeScratch, RealForwardError};
use crate::real_forward_utils::norm_view;

/// Reads `scratch.normed` and writes the block's output, pre-
/// `post_operator_norm`, to `scratch.o`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_conv_block(
    context: &mut gpu::MetalContext,
    pass: &gpu::PassEncoder,
    weights: &gpu::ResidentGpuWeights,
    index: &ResidentIndex,
    arch: &ArchConfig,
    lfm3: &RealLfm3State,
    scratch: &DecodeScratch,
    layer: usize,
) -> Result<(), RealForwardError> {
    let hidden = arch.hidden_size as usize;
    let name = |suffix: &str| layer_tensor(layer, &format!("conv.{suffix}"));

    encode_gemv_any(
        context,
        pass,
        weights,
        index,
        &name("in_proj.weight"),
        3 * hidden,
        hidden,
        (&scratch.normed, 0),
        (&lfm3.in_proj, 0),
    )?;
    let weight = norm_view(
        weights,
        index,
        &name("conv.weight"),
        hidden * lfm3.taps as usize,
    )?;
    gpu::encode_short_conv_decode(
        context,
        pass,
        (lfm3.conv.tail_buffer(layer), 0),
        (&lfm3.in_proj, 0),
        weight,
        (&lfm3.conv_out, 0),
        hidden as u32,
        lfm3.taps,
    )
    .map_err(RealForwardError::Gpu)?;
    encode_gemv_any(
        context,
        pass,
        weights,
        index,
        &name("out_proj.weight"),
        hidden,
        hidden,
        (&lfm3.conv_out, 0),
        (&scratch.o, 0),
    )
}
