//! The DENSE FFN half of the Qwen hybrid flow (ROADMAP's 1-bit entry, step
//! 4): what a `qwen3_5` layer runs where a Qwen 3.6 layer runs a router, a
//! gated shared expert and eight routed ones.
//!
//! ```text
//! m = rms_norm(x, post_attention_layernorm)   // encoded by the caller
//! h = down_proj( act(gate_proj @ m, up_proj @ m) )
//! x = x + h
//! ```
//!
//! NO NEW KERNEL, and the sibling to read beside this one is
//! `families/llama/dense.rs`: the same three dispatches, for the same reason
//! (ROADMAP M4). What differs is only which buffers they read and write --
//! Qwen's post-attention norm lands in `qwen.moe_x` because that one norm
//! feeds everything below it, and the FFN output goes to `qwen.h2`, the
//! buffer phase 2 writes on the MoE half.
//!
//! **The width is `intermediate_size`, never `moe_intermediate_size`**, which
//! reads as an obvious statement and is the one place this family is more
//! dangerous than `llama`: `qwen_gdn_moe_35b_a3b()` sets `intermediate_size` to the
//! SHARED EXPERT's width and `qwen_gdn_dense_27b()` sets it to the dense FFN's, so the
//! same field means two things across the two installs this file's flow
//! serves. A dense install sets `moe_intermediate_size` to 0, so taking that
//! one encodes nothing at all.
//!
//! **A dense layer needs no mid-layer commit.** The MoE half commits `cb1` and
//! waits inside the layer because the top-k has to reach the host before the
//! experts can be bound; nothing here is data-dependent, so the whole token
//! stays in one command buffer.

use model_io::ResidentIndex;

use crate::families::qwen::{prefixed_layer_tensor, RealQwenState};
use crate::real_forward_dispatch::encode_gemv_any;
use crate::real_forward_types::{DecodeScratch, RealForwardError};

/// Encodes one dense layer's FFN plus its residual add into `pass`.
///
/// `residual` is read as the residual and written back with the FFN output
/// added; `qwen.moe_x` holds the post-attention norm the caller produced and
/// is the FFN's input, so it must not be the destination of anything encoded
/// in between.
///
/// `residual` is a PARAMETER rather than `scratch.x` for one caller only: the
/// MTP head runs this same FFN over its own hidden stream
/// (`docs/MTP_SPECULATIVE.md`), after the trunk's token is complete. The
/// trunk still passes `scratch.x` and is byte-identical for it.
///
/// `x_off` is the residual's row offset within `residual`, following the
/// `families/llama/dense.rs` / `crates/runtime/CLAUDE.md` Gotcha 21
/// precedent: the chunked-prefill driver packs several tokens into one
/// buffer at their own offsets, where every other caller (the sequential
/// trunk, the MTP head) sits at row 0 and passes `0` explicitly.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_qwen_layer_dense(
    context: &mut gpu::MetalContext,
    pass: &gpu::PassEncoder,
    weights: &gpu::ResidentGpuWeights,
    index: &ResidentIndex,
    scratch: &DecodeScratch,
    qwen: &RealQwenState,
    residual: &gpu::MetalBuffer,
    x_off: u64,
    prefix: &str,
    layer: usize,
    hidden: usize,
    inter: usize,
    use_silu: bool,
) -> Result<(), RealForwardError> {
    let gpu_err = RealForwardError::Gpu;

    // Per NAME: on a folded checkpoint the gate/up pair reads the caller's
    // transformed post-attention norm (they share its width and sign vector);
    // `down_proj`'s input is the FFN activation at `inter` width and gets its
    // own transform below, at its own call site.
    let gate_name = prefixed_layer_tensor(prefix, layer, "mlp.gate_proj.weight");
    let ffn_input = match qwen.hadamard.as_ref() {
        Some(h) if h.is_folded(&gate_name) => (&h.moe_x_h, 0),
        _ => (&qwen.moe_x, 0),
    };

    // THE FUSED TAIL (TURBOSPARK_QWEN_FUSED_FFN, default on): gate+up+act in
    // one dispatch and the down projection with the residual add folded into
    // its store, 5 dispatches -> 2. Only for INT4 weights whose gate and down
    // inputs are not hadamard-folded (a folded input or transform slot has no
    // fused form and keeps the per-tensor chain below); bit-exact either way,
    // which the gpu parity tests and the frozen digests pin.
    let down_name = prefixed_layer_tensor(prefix, layer, "mlp.down_proj.weight");
    let gate_folded = matches!(qwen.hadamard.as_ref(), Some(h) if h.is_folded(&gate_name));
    let down_folded = matches!(qwen.hadamard.as_ref(), Some(h) if h.is_folded(&down_name));
    // The fused kernels are GEMV arithmetic: under `TURBOSPARK_INT4_MMA8` a
    // served shape must reach the shared tile instead, on both sides.
    let mma8 = crate::real_forward_dispatch::int4_mma8_takes;
    if crate::families::qwen::fused_ffn()
        && !mma8(inter, hidden)
        && !mma8(hidden, inter)
        && !gate_folded
        && !down_folded
        && crate::real_forward_utils::entry(index, &gate_name)
            .map(|e| e.dtype == 4)
            .unwrap_or(false)
        && crate::real_forward_utils::entry(
            index,
            &prefixed_layer_tensor(prefix, layer, "mlp.up_proj.weight"),
        )
        .map(|e| e.dtype == 4)
        .unwrap_or(false)
        && crate::real_forward_utils::entry(index, &down_name)
            .map(|e| e.dtype == 4)
            .unwrap_or(false)
    {
        foundation::engaged!("qwen_fused_ffn");
        let gate_m =
            crate::real_forward_utils::resident_matrix(weights, index, &gate_name, inter, hidden)?;
        let up_m = crate::real_forward_utils::resident_matrix(
            weights,
            index,
            &prefixed_layer_tensor(prefix, layer, "mlp.up_proj.weight"),
            inter,
            hidden,
        )?;
        let down_m =
            crate::real_forward_utils::resident_matrix(weights, index, &down_name, hidden, inter)?;
        gpu::encode_gate_up_act_gemv(
            context,
            pass,
            &gate_m,
            &up_m,
            ffn_input,
            (&scratch.ffn_act, 0),
            !use_silu,
        )
        .map_err(gpu_err)?;
        crate::lab_hooks::encode_dispatch_probe(context, pass, (&scratch.ffn_act, 0), inter)?;
        // RAW residual add, matching the MoE half and the unfused chain's
        // epilogue: the FFN output is rounded to half BEFORE it meets the
        // stream, which is the double round the fused kernel reproduces.
        gpu::encode_down_epi_gemv(
            context,
            pass,
            &down_m,
            (&scratch.ffn_act, 0),
            (residual, x_off),
            (residual, x_off),
            gpu::DownEpilogue::Residual,
        )
        .map_err(gpu_err)?;
        return Ok(());
    }

    for (suffix, out) in [
        ("mlp.gate_proj.weight", &scratch.ffn_gate),
        ("mlp.up_proj.weight", &scratch.ffn_up),
    ] {
        encode_gemv_any(
            context,
            pass,
            weights,
            index,
            &prefixed_layer_tensor(prefix, layer, suffix),
            inter,
            hidden,
            ffn_input,
            (out, 0),
        )?;
    }

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
    crate::lab_hooks::encode_dispatch_probe(context, pass, (&scratch.ffn_act, 0), inter)?;

    let down_input = match qwen.hadamard.as_ref() {
        Some(h) if h.is_folded(&down_name) => {
            h.transform(
                context,
                pass,
                (&scratch.ffn_act, 0),
                (&h.ffn_act_h, 0),
                1,
                inter as u32,
                true,
            )?;
            (&h.ffn_act_h, 0)
        }
        _ => (&scratch.ffn_act, 0),
    };
    encode_gemv_any(
        context,
        pass,
        weights,
        index,
        &down_name,
        hidden,
        inter,
        down_input,
        (&qwen.h2, 0),
    )?;

    // RAW residual add, matching the MoE half: this family normalizes neither
    // the attention output nor the FFN output on the way back into the stream
    // (`ffn_sandwich_norms: false`), and doing so is what took the Qwen 3.6
    // reference perplexity from 6.25 to 255,409 once already.
    gpu::encode_residual_add(
        context,
        pass,
        (residual, x_off),
        (&qwen.h2, 0),
        hidden as u32,
    )
    .map_err(gpu_err)?;
    Ok(())
}
