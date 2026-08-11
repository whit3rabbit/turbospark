//! The DENSE FFN half of the `llama` architecture (ROADMAP M4): what a
//! Mistral or a Llama 2/3.x layer runs where a Mixtral layer runs a router
//! and two routed experts.
//!
//! ```text
//! m = rms_norm(x, post_attention_layernorm)   // encoded by the caller
//! h = down_proj( act(gate_proj @ m, up_proj @ m) )
//! x = x + h
//! ```
//!
//! NO NEW KERNEL. Every dispatch here already existed for Gemma's shared
//! expert, which is the same gated FFN with a trailing norm this architecture
//! does not have; the block types are resolved per tensor by
//! `encode_gemv_any`, so a mixed `Q4_K_M` needs nothing extra.
//!
//! **The width is `intermediate_size`, never `moe_intermediate_size`.** They
//! are the same number on Mixtral (one `feed_forward_length` in the header,
//! copied to both) and would be silently interchangeable there, which is
//! exactly why the dense case has to name the right one: a dense checkpoint
//! sets only the first, and taking the second would read a zero width and
//! encode nothing at all.
//!
//! **A dense layer needs no host round trip.** The MoE path commits `cb1` and
//! waits mid-layer because the top-k selection has to be read back on the
//! host before the experts can be bound. Nothing here is data-dependent, so
//! the whole token stays in one command buffer and the layer loop never
//! syncs.

use model_io::ResidentIndex;

use crate::families::llama::{layer_tensor, RealLlamaState};
use crate::real_forward_dispatch::encode_gemv_any;
use crate::real_forward_types::{DecodeScratch, RealForwardError};

/// Encodes one dense layer's FFN plus its residual add into `pass`.
///
/// `x` is read as the residual and written back with the FFN output added;
/// `llama.moe_x` holds the post-attention norm the caller produced, and is
/// the FFN's input. Both projections read it, so it must not be the
/// destination of anything encoded in between.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_llama_layer_dense(
    context: &mut gpu::MetalContext,
    pass: &gpu::PassEncoder,
    weights: &gpu::ResidentGpuWeights,
    index: &ResidentIndex,
    scratch: &DecodeScratch,
    llama: &RealLlamaState,
    layer: usize,
    hidden: usize,
    inter: usize,
    use_silu: bool,
) -> Result<(), RealForwardError> {
    let gpu_err = RealForwardError::Gpu;

    for (suffix, out) in [
        ("mlp.gate_proj.weight", &scratch.ffn_gate),
        ("mlp.up_proj.weight", &scratch.ffn_up),
    ] {
        encode_gemv_any(
            context,
            pass,
            weights,
            index,
            &layer_tensor(layer, suffix),
            inter,
            hidden,
            (&llama.moe_x, 0),
            (out, 0),
        )?;
    }

    // SiLU for every real `llama` checkpoint; the branch exists because
    // `hidden_activation` is a manifest field and a GELU variant of this
    // architecture would differ in nothing else.
    let act = if use_silu {
        gpu::encode_silu_mul
    } else {
        gpu::encode_gelu_mul
    };
    act(
        context,
        pass,
        (&scratch.ffn_gate, 0),
        (&scratch.ffn_up, 0),
        (&scratch.ffn_act, 0),
        inter as u32,
    )
    .map_err(gpu_err)?;

    encode_gemv_any(
        context,
        pass,
        weights,
        index,
        &layer_tensor(layer, "mlp.down_proj.weight"),
        hidden,
        inter,
        (&scratch.ffn_act, 0),
        (&llama.h2, 0),
    )?;

    // RAW residual add, matching the MoE half: this architecture normalizes
    // neither the attention output nor the FFN output on the way back into
    // the stream.
    gpu::encode_residual_add(
        context,
        pass,
        (&scratch.x, 0),
        (&llama.h2, 0),
        hidden as u32,
    )
    .map_err(gpu_err)?;
    Ok(())
}
