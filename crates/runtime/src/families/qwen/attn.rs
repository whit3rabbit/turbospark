//! The two per-layer attention blocks of the Qwen 3.6 decode flow.

use model_io::{ArchConfig, ResidentIndex};

use crate::families::qwen::{layer_tensor, RealQwenState, RMS_EPS};
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
    layer: usize,
    position: usize,
) -> Result<(), RealForwardError> {
    let gpu_err = RealForwardError::Gpu;
    let hidden = arch.hidden_size as usize;
    let num_heads = arch.num_heads as u32;
    let num_kv = arch.num_full_kv_heads as u32;
    let head_dim = arch.full_head_dim as u32;
    let q_dim = (num_heads * head_dim) as usize;
    let kv_dim = (num_kv * head_dim) as usize;
    let name = |suffix: &str| layer_tensor(layer, &format!("self_attn.{suffix}"));

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

    for (suffix, data, heads) in [
        ("q_norm.weight", (&scratch.q, 0u64), num_heads),
        ("k_norm.weight", (k_buf, k_off as u64), num_kv),
    ] {
        let weight = norm_view(weights, index, &name(suffix), head_dim as usize)?;
        gpu::encode_rms_norm_bf16w_perhead(
            context, pass, data, weight, data, heads, head_dim, RMS_EPS,
        )
        .map_err(gpu_err)?;
    }

    let theta = arch.full_rope_theta as f32;
    for (data, heads) in [
        ((&scratch.q, 0u64), num_heads),
        ((k_buf, k_off as u64), num_kv),
    ] {
        gpu::encode_rope_neox_subdim(
            context,
            pass,
            data,
            position as u32,
            heads,
            head_dim,
            qwen.rotary_dim,
            theta,
        )
        .map_err(gpu_err)?;
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
