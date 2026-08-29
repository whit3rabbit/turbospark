//! The two per-layer attention blocks of the Qwen 3.6 decode flow.

use model_io::{ArchConfig, ResidentIndex};

use crate::families::qwen::{layer_tensor, prefixed_layer_tensor, RealQwenState, RMS_EPS};
use crate::real_forward::RealForwardError;
use crate::real_forward_dispatch::encode_gemv_any;
use crate::real_forward_types::DecodeScratch;
use crate::real_forward_utils::{entry, norm_view, resident_matrix};

/// Mask-2 layer: gated DeltaNet. No position, no RoPE, no KV -- the whole
/// history is the layer's FP32 recurrent state plus its `K-1` row conv
/// tail, both owned by [`gpu::GdnStateManager`] and advanced in place.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_linear_block(
    context: &mut gpu::MetalContext,
    pass: &gpu::PassEncoder,
    weights: &gpu::ResidentGpuWeights,
    index: &ResidentIndex,
    arch: &ArchConfig,
    qwen: &RealQwenState,
    scratch: &DecodeScratch,
    layer: usize,
) -> Result<(), RealForwardError> {
    let gpu_err = RealForwardError::Gpu;
    let hidden = arch.hidden_size as usize;
    let shape = qwen.shape;
    let qkv_dim = shape.qkv_dim() as usize;
    let value_dim = shape.value_dim() as usize;
    let v_heads = shape.num_v_heads as usize;

    let name = |suffix: &str| layer_tensor(layer, &format!("linear_attn.{suffix}"));
    let in_proj = [
        ("in_proj_qkv.weight", qkv_dim, &qwen.gdn_qkv_raw),
        ("in_proj_z.weight", value_dim, &qwen.gdn_z),
        ("in_proj_a.weight", v_heads, &qwen.gdn_a),
        ("in_proj_b.weight", v_heads, &qwen.gdn_b),
    ];
    if entry(index, &name(in_proj[0].0))?.dtype == 4 {
        let projection = |suffix: &str, rows: usize| {
            resident_matrix(weights, index, &name(suffix), rows, hidden)
        };
        let (qkv_w, z_w, a_w, b_w) = (
            projection(in_proj[0].0, qkv_dim)?,
            projection(in_proj[1].0, value_dim)?,
            projection(in_proj[2].0, v_heads)?,
            projection(in_proj[3].0, v_heads)?,
        );
        gpu::encode_gdn_in_proj(
            context,
            pass,
            &qkv_w,
            &z_w,
            &a_w,
            &b_w,
            (&scratch.normed, 0),
            (&qwen.gdn_qkv_raw, 0),
            (&qwen.gdn_z, 0),
            (&qwen.gdn_a, 0),
            (&qwen.gdn_b, 0),
        )
        .map_err(gpu_err)?;
    } else {
        for (suffix, rows, out) in in_proj {
            encode_gemv_any(
                context,
                pass,
                weights,
                index,
                &name(suffix),
                rows,
                hidden,
                (&scratch.normed, 0),
                (out, 0),
            )?;
        }
    }

    let conv_w = norm_view(
        weights,
        index,
        &name("conv1d.weight"),
        qkv_dim * shape.conv_kernel_size as usize,
    )?;
    gpu::encode_gdn_conv_decode(
        context,
        pass,
        shape,
        (qwen.gdn.conv_tail_buffer(layer), 0),
        (&qwen.gdn_qkv_raw, 0),
        conv_w,
        (&qwen.gdn_conv_out, 0),
    )
    .map_err(gpu_err)?;
    gpu::encode_gdn_qk_norm(context, pass, shape, (&qwen.gdn_conv_out, 0), 1).map_err(gpu_err)?;

    // A_log and dt_bias carry NO `.weight` suffix in the checkpoint.
    let a_log = norm_view(weights, index, &name("A_log"), v_heads)?;
    let dt_bias = norm_view(weights, index, &name("dt_bias"), v_heads)?;
    gpu::encode_gdn_delta_decode(
        context,
        pass,
        shape,
        (&qwen.gdn_conv_out, 0),
        (&qwen.gdn_a, 0),
        (&qwen.gdn_b, 0),
        a_log,
        dt_bias,
        qwen.gdn.state_buffer(layer),
        (&qwen.gdn_y, 0),
    )
    .map_err(gpu_err)?;

    let gated_norm = norm_view(
        weights,
        index,
        &name("norm.weight"),
        shape.value_head_dim as usize,
    )?;
    gpu::encode_gdn_gated_norm(
        context,
        pass,
        shape,
        (&qwen.gdn_y, 0),
        (&qwen.gdn_z, 0),
        gated_norm,
        (&qwen.gdn_out, 0),
        1,
    )
    .map_err(gpu_err)?;
    encode_gemv_any(
        context,
        pass,
        weights,
        index,
        &name("out_proj.weight"),
        hidden,
        value_dim,
        (&qwen.gdn_out, 0),
        (&scratch.o, 0),
    )
}

/// Which convention this block's per-head `q_norm`/`k_norm` are stored in.
///
/// **This is a property of the TENSOR, not of the family** (AGENTS.md Gotcha
/// 50). The `qwen3_5` trunk and its multi-token-prediction head both carry
/// tensors named `self_attn.q_norm.weight` and `self_attn.k_norm.weight`, at
/// the same shape, resolved by the same code below -- and the trunk's are
/// plain (`x * w`) while the head's store an OFFSET FROM UNITY (`x * (1 + w)`),
/// because mlx-vlm's converter bakes the `+1` into its published trunk weights
/// and leaves the head's centered. Reading the head's plainly is not a subtle
/// error: it put the true next-next token at median rank 248,308 of 248,320
/// (`docs/MTP.md`).
///
/// An enum rather than a bool so the call site states which question it is
/// answering; a bare `true` two frames from the dispatch says nothing.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum QkNormConvention {
    /// `x * w`. The trunk's, and every other family's.
    Plain,
    /// `x * (1 + w)`. The MTP head's.
    Centered,
}

/// Which position, or positions, this block's RoPE rotates by.
///
/// **The `position` argument keeps its other two jobs either way**: it is the
/// KV slot index and the `position + 1` attention span, and neither moves for
/// an image. Only the ANGLE differs, which is what lets vision reach this
/// family without touching the cache at all.
///
/// A parameter rather than a field on `RealQwenState`, following
/// [`QkNormConvention`]'s precedent one call site over: the MTP head runs
/// through this same function and has no prompt of its own to index.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum RopePosition {
    /// Rotate by the `position` argument. Every caller before ROADMAP M-V5,
    /// and every caller on a text-only prompt.
    Sequential,
    /// Rotate by this `(t, h, w)`, which an image prompt's own position table
    /// supplies (`vision::PromptVision::rope_position`).
    ///
    /// **`t == h == w` here still takes the EXISTING kernel**, and that is the
    /// whole dispatch rule. It is a property of the DATA rather than a
    /// classification of the token: `get_rope_index` gives every text token of
    /// a mixed prompt the same number in all three slots, so the divergence
    /// test IS the "is this an image pad" test, with nothing extra to plumb
    /// and nothing to get out of step.
    Triple(i32, i32, i32),
}

/// Mask-1 layer: gated full attention.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_full_attention_block(
    context: &mut gpu::MetalContext,
    pass: &gpu::PassEncoder,
    weights: &gpu::ResidentGpuWeights,
    index: &ResidentIndex,
    arch: &ArchConfig,
    qwen: &RealQwenState,
    scratch: &DecodeScratch,
    kv: &gpu::KvCacheManager,
    prefix: &str,
    layer: usize,
    position: usize,
    qk_norms: QkNormConvention,
    rope_position: RopePosition,
) -> Result<(), RealForwardError> {
    let gpu_err = RealForwardError::Gpu;
    let hidden = arch.hidden_size as usize;
    let num_heads = arch.num_heads as u32;
    let num_kv = arch.num_full_kv_heads as u32;
    let head_dim = arch.full_head_dim as u32;
    let q_dim = (num_heads * head_dim) as usize;
    let kv_dim = (num_kv * head_dim) as usize;
    let name = |suffix: &str| prefixed_layer_tensor(prefix, layer, &format!("self_attn.{suffix}"));

    let (k_buf, k_off) = kv.k_slot(layer, position);
    let (v_buf, v_off) = kv.v_slot(layer, position);
    encode_gemv_any(
        context,
        pass,
        weights,
        index,
        &name("q_proj.weight"),
        2 * q_dim,
        hidden,
        (&scratch.normed, 0),
        (&qwen.q_packed, 0),
    )?;
    gpu::encode_split_q_gate(
        context,
        pass,
        (&qwen.q_packed, 0),
        (&scratch.q, 0),
        (&qwen.attn_gate, 0),
        num_heads,
        head_dim,
    )
    .map_err(gpu_err)?;
    for (suffix, out) in [
        ("k_proj.weight", (k_buf, k_off as u64)),
        ("v_proj.weight", (v_buf, v_off as u64)),
    ] {
        encode_gemv_any(
            context,
            pass,
            weights,
            index,
            &name(suffix),
            kv_dim,
            hidden,
            (&scratch.normed, 0),
            out,
        )?;
    }

    // Per TENSOR, never per family: see `QkNormConvention`.
    let encode_qk_norm = match qk_norms {
        QkNormConvention::Plain => gpu::encode_rms_norm_bf16w_perhead,
        QkNormConvention::Centered => gpu::encode_rms_norm_bf16w_perhead_centered,
    };
    for (suffix, data, heads) in [
        ("q_norm.weight", (&scratch.q, 0u64), num_heads),
        ("k_norm.weight", (k_buf, k_off as u64), num_kv),
    ] {
        let weight = norm_view(weights, index, &name(suffix), head_dim as usize)?;
        encode_qk_norm(context, pass, data, weight, data, heads, head_dim, RMS_EPS)
            .map_err(gpu_err)?;
    }

    let theta = arch.full_rope_theta as f32;
    // The mRoPE kernel fires only where the three components actually differ,
    // which on a real prompt is image-pad positions and nothing else. A text
    // token of a MIXED prompt takes this same `encode_rope_neox_subdim` call
    // it always took (`docs/VISION_PHASE0.md` item 2 proves `t == h == w`
    // there), so the pre-vision engine's bytes survive an image appearing
    // earlier in the same prompt -- not merely a prompt with no image in it.
    //
    // The two kernels share `apply_neox_pair`, so the boundary between these
    // arms is exact rather than approximate; `rope_mrope_interleaved` at
    // `t == h == w` produces the identical bits.
    //
    // **THE DEGENERATE ARM IS THEREFORE A NO-OP TODAY, AND THE MUTATION THAT
    // SAYS SO SURVIVES ON PURPOSE.** Deleting it -- so every `Triple` reaches
    // the new kernel, degenerate or not -- leaves all nine cases in
    // `tests/vision_inject_synthetic.rs` green, the frozen digest of a run
    // with three genuinely divergent image positions included. That is the
    // PREDICTED result and it is the end-to-end confirmation of what
    // `at_t_equals_h_equals_w_it_is_bit_identical_to_rope_neox_subdim` asserts
    // one crate down, reached here through the real trunk rather than a
    // fixture. The arm is kept for two reasons that are not numerical: it
    // keeps a text-only mixed prompt on the dispatch path the pre-vision
    // engine used, so a future change to the mRoPE kernel cannot reach a text
    // token at all, and it means the claim rests on which function is called
    // rather than on the shader compiler continuing to agree.
    let mrope = match rope_position {
        RopePosition::Sequential => None,
        RopePosition::Triple(t, h, w) if t == h && h == w => None,
        RopePosition::Triple(t, h, w) => Some((t as u32, h as u32, w as u32)),
    };
    let scalar = match rope_position {
        RopePosition::Sequential => position as u32,
        RopePosition::Triple(t, _, _) => t as u32,
    };
    let section = arch.vision.mrope_section;
    for (data, heads) in [
        ((&scratch.q, 0u64), num_heads),
        ((k_buf, k_off as u64), num_kv),
    ] {
        match mrope {
            Some(positions) => gpu::encode_rope_mrope_interleaved(
                context,
                pass,
                data,
                positions,
                heads,
                head_dim,
                qwen.rotary_dim,
                (section[0] as u32, section[1] as u32, section[2] as u32),
                theta,
            )
            .map_err(gpu_err)?,
            None => gpu::encode_rope_neox_subdim(
                context,
                pass,
                data,
                scalar,
                heads,
                head_dim,
                qwen.rotary_dim,
                theta,
            )
            .map_err(gpu_err)?,
        }
    }

    gpu::encode_attention_decode(
        context,
        pass,
        (&scratch.q, 0),
        k_buf,
        v_buf,
        &scratch.attn,
        (&scratch.attn_out, 0),
        head_dim,
        num_heads,
        num_kv,
        (position + 1) as u32,
        0,
        0,
        arch.attention_scale as f32,
        None,
    )
    .map_err(gpu_err)?;
    gpu::encode_sigmoid_gate_mul(
        context,
        pass,
        (&scratch.attn_out, 0),
        (&qwen.attn_gate, 0),
        q_dim as u32,
    )
    .map_err(gpu_err)?;
    encode_gemv_any(
        context,
        pass,
        weights,
        index,
        &name("o_proj.weight"),
        hidden,
        q_dim,
        (&scratch.attn_out, 0),
        (&scratch.o, 0),
    )
}
