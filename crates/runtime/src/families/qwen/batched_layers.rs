use model_io::{ArchConfig, ResidentIndex};

use super::batched::BatchedScratch;
use crate::families::qwen::{
    layer_tensor, prefixed_layer_tensor, RealQwenState, RMS_EPS, TRUNK_PREFIX,
};
use crate::real_forward_dispatch::encode_gemm_any;
use crate::real_forward_types::{DecodeScratch, RealForwardError};
use crate::real_forward_utils::norm_view;

/// Mask-1 layer at M rows.
///
/// `k_proj` and `v_proj` write STRAIGHT into the KV cache, exactly as the
/// per-token block does -- the only difference is that one dispatch fills M
/// adjacent slots instead of one. That works because the batched kernel's
/// output is token-major with a row stride of `kv_dim` halfs, which is the
/// cache's own per-token stride; the assertion below is what keeps that a
/// checked fact rather than a coincidence.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_full_attention_block_batched(
    context: &mut gpu::MetalContext,
    pass: &gpu::PassEncoder,
    weights: &gpu::ResidentGpuWeights,
    index: &ResidentIndex,
    arch: &ArchConfig,
    qwen: &RealQwenState,
    scratch: &DecodeScratch,
    batched: &BatchedScratch,
    kv: &gpu::KvCacheManager,
    layer: usize,
    start_position: usize,
    batch: usize,
) -> Result<(), RealForwardError> {
    let gpu_err = RealForwardError::Gpu;
    let hidden = arch.hidden_size as usize;
    let num_heads = arch.num_heads as u32;
    let num_kv = arch.num_full_kv_heads as u32;
    let head_dim = arch.full_head_dim as u32;
    let q_dim = (num_heads * head_dim) as usize;
    let kv_dim = (num_kv * head_dim) as usize;
    let name =
        |suffix: &str| prefixed_layer_tensor(TRUNK_PREFIX, layer, &format!("self_attn.{suffix}"));

    // `--kv-bits` is not wired to this M-row GEMM path (`crate::kv_write`'s
    // module doc): it writes K/V straight into the cache via a batched
    // projection and bypasses `kv_write_target`/`encode_kv_commit` entirely.
    // The stride check two lines below would ALSO catch a quantized layer
    // (its packed row is narrower than `kv_dim` FP16 halfs), but that
    // message reads as a shape bug rather than as this feature, so refuse
    // by name first.
    if kv.layer_quant(layer).is_some() {
        return Err(RealForwardError::Unsupported(
            "--kv-bits is not yet supported with TURBOSPARK_BATCHED_GEMV".to_string(),
        ));
    }
    if kv.k_stride(layer) != kv_dim * 2 {
        return Err(RealForwardError::Unsupported(format!(
            "layer {layer}: KV stride {} is not {kv_dim} halfs, so a batched projection \
             cannot write M adjacent slots in one dispatch",
            kv.k_stride(layer)
        )));
    }
    let (k_buf, k_off) = kv.k_slot(layer, start_position);
    let (v_buf, v_off) = kv.v_slot(layer, start_position);

    encode_gemm_any(
        context,
        pass,
        weights,
        index,
        &name("q_proj.weight"),
        2 * q_dim,
        hidden,
        (&batched.normed, 0),
        (&batched.q_packed, 0),
        batch,
    )?;
    for (suffix, out) in [
        ("k_proj.weight", (k_buf, k_off as u64)),
        ("v_proj.weight", (v_buf, v_off as u64)),
    ] {
        encode_gemm_any(
            context,
            pass,
            weights,
            index,
            &name(suffix),
            kv_dim,
            hidden,
            (&batched.normed, 0),
            out,
            batch,
        )?;
    }

    let q_norm = norm_view(weights, index, &name("q_norm.weight"), head_dim as usize)?;
    let k_norm = norm_view(weights, index, &name("k_norm.weight"), head_dim as usize)?;
    let theta = arch.full_rope_theta as f32;
    for m in 0..batch {
        let position = start_position + m;
        gpu::encode_split_q_gate(
            context,
            pass,
            (&batched.q_packed, (m * 2 * q_dim) as u64 * 2),
            (&batched.q, (m * q_dim) as u64 * 2),
            (&batched.attn_gate, (m * q_dim) as u64 * 2),
            num_heads,
            head_dim,
        )
        .map_err(gpu_err)?;

        let q_row = (&batched.q, (m * q_dim) as u64 * 2);
        let k_row = (k_buf, (k_off + m * kv_dim * 2) as u64);
        // The TRUNK's q/k norms are plain; only its MTP head's are centered.
        for (data, weight, heads) in [(q_row, q_norm, num_heads), (k_row, k_norm, num_kv)] {
            gpu::encode_rms_norm_bf16w_perhead(
                context, pass, data, weight, data, heads, head_dim, RMS_EPS,
            )
            .map_err(gpu_err)?;
        }
        for (data, heads) in [(q_row, num_heads), (k_row, num_kv)] {
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
    }

    // Attention loops, and the span is what makes the loop causal: query m
    // sees `[0, start_position + m]` even though rows above it are already
    // written. `scratch.attn` is reused across the M queries because command
    // buffers on one queue execute in commit order (`crates/gpu` Gotcha 8) --
    // it is a GPU-only intermediate, so the reuse serializes the queries and
    // cannot corrupt them. Attention is 0.6% of this family's decode compute;
    // widening it is measured to be not worth a kernel.
    for m in 0..batch {
        gpu::encode_attention_decode(
            context,
            pass,
            (&batched.q, (m * q_dim) as u64 * 2),
            k_buf,
            v_buf,
            &scratch.attn,
            (&batched.attn_out, (m * q_dim) as u64 * 2),
            head_dim,
            num_heads,
            num_kv,
            (start_position + m + 1) as u32,
            0,
            0,
            arch.attention_scale as f32,
            None,
        )
        .map_err(gpu_err)?;
        gpu::encode_sigmoid_gate_mul(
            context,
            pass,
            (&batched.attn_out, (m * q_dim) as u64 * 2),
            (&batched.attn_gate, (m * q_dim) as u64 * 2),
            q_dim as u32,
        )
        .map_err(gpu_err)?;
    }

    encode_gemm_any(
        context,
        pass,
        weights,
        index,
        &name("o_proj.weight"),
        hidden,
        q_dim,
        (&batched.attn_out, 0),
        (&batched.o, 0),
        batch,
    )
}

/// Mask-2 layer at M rows: batched input projections, then the recurrent
/// chain in TOKEN ORDER.
///
/// The recurrence is why this is not simply "batch everything": token m's
/// delta-rule state is token m-1's output, so the chain is sequential by
/// definition. What batches is the input projection either side of it.
///
/// The multi-row conv, delta and norm kernels already existed
/// (`gdn_conv_mix_prefill`, `gdn_delta_step_prefill`, and the `rows`
/// argument on both norms) and were dispatched by nothing but
/// `gdn_parity.rs`, exactly as `dequant_int4_gemm_simd` was. They advance
/// the layer's state across all M rows in one dispatch, so the "loop" here
/// is inside the kernel rather than in this file.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_linear_block_batched(
    context: &mut gpu::MetalContext,
    pass: &gpu::PassEncoder,
    weights: &gpu::ResidentGpuWeights,
    index: &ResidentIndex,
    arch: &ArchConfig,
    qwen: &RealQwenState,
    batched: &BatchedScratch,
    layer: usize,
    batch: usize,
    tape_slot: Option<usize>,
) -> Result<(), RealForwardError> {
    let gpu_err = RealForwardError::Gpu;
    let hidden = arch.hidden_size as usize;
    let shape = qwen.shape;
    let qkv_dim = shape.qkv_dim() as usize;
    let value_dim = shape.value_dim() as usize;
    let v_heads = shape.num_v_heads as usize;
    let name = |suffix: &str| layer_tensor(layer, &format!("linear_attn.{suffix}"));

    // The FUSED input projection (`gdn_in_proj_gemv_simd`) has no batched
    // form, so this takes the four-separate-projection branch that already
    // exists for the non-INT4 path. Four batched GEMMs beat one fused GEMV
    // per token at every M this pass runs at.
    for (suffix, rows, out) in [
        ("in_proj_qkv.weight", qkv_dim, &batched.gdn_qkv_raw),
        ("in_proj_z.weight", value_dim, &batched.gdn_z),
        ("in_proj_a.weight", v_heads, &batched.gdn_a),
        ("in_proj_b.weight", v_heads, &batched.gdn_b),
    ] {
        encode_gemm_any(
            context,
            pass,
            weights,
            index,
            &name(suffix),
            rows,
            hidden,
            (&batched.normed, 0),
            (out, 0),
            batch,
        )?;
    }

    // THE TAPE, when this scratch records one: copy the three row inputs
    // the state advance is a function of, per linear-layer ordinal. The
    // conv tail update and the delta rule below never rewrite them, so the
    // copies can sit anywhere after the projections; they sit here so the
    // pass keeps one ordering for every layer. `z` is not copied -- see
    // `BatchedVerifyTape`'s note. The stride below is THIS pass's `batch`,
    // which is exactly what `replay_linear_state_batched`'s `recorded_batch`
    // names: change either site and the other reads every slot past 0 from
    // the wrong offset.
    if let Some(tape) = batched.verify_tape.as_ref() {
        let slot = tape_slot.ok_or_else(|| {
            RealForwardError::Unsupported(
                "a tape-carrying scratch must be driven with a tape slot".to_string(),
            )
        })?;
        // The general strided-row copy; dflash-named because the drafter's
        // aux capture was its first caller.
        let off = |dim: usize| (slot * batch * dim * 2) as u64;
        let (rows, w_qkv, w_heads) = (batch as u32, qkv_dim as u32, v_heads as u32);
        gpu::encode_dflash_copy_rows(
            context,
            pass,
            (&batched.gdn_qkv_raw, 0),
            (&tape.qkv_raw, off(qkv_dim)),
            rows,
            w_qkv,
            w_qkv,
        )
        .map_err(gpu_err)?;
        gpu::encode_dflash_copy_rows(
            context,
            pass,
            (&batched.gdn_a, 0),
            (&tape.a, off(v_heads)),
            rows,
            w_heads,
            w_heads,
        )
        .map_err(gpu_err)?;
        gpu::encode_dflash_copy_rows(
            context,
            pass,
            (&batched.gdn_b, 0),
            (&tape.b, off(v_heads)),
            rows,
            w_heads,
            w_heads,
        )
        .map_err(gpu_err)?;
    }

    let conv_w = norm_view(
        weights,
        index,
        &name("conv1d.weight"),
        qkv_dim * shape.conv_kernel_size as usize,
    )?;
    gpu::encode_gdn_conv_prefill(
        context,
        pass,
        shape,
        (qwen.gdn.conv_tail_buffer(layer), 0),
        (&batched.gdn_qkv_raw, 0),
        conv_w,
        (&batched.gdn_conv_out, 0),
        batch as u32,
        // Undilated, matching `attn.rs`'s decode-path call.
        1,
    )
    .map_err(gpu_err)?;
    // The prefill conv reads the tail and does NOT advance it; the decode
    // sibling does both. Skipping this leaves the layer's history one block
    // behind, which is fluent and wrong.
    gpu::encode_gdn_conv_tail_update(
        context,
        pass,
        shape,
        (qwen.gdn.conv_tail_buffer(layer), 0),
        (&batched.gdn_qkv_raw, 0),
        batch as u32,
        1,
    )
    .map_err(gpu_err)?;
    gpu::encode_gdn_qk_norm(
        context,
        pass,
        shape,
        (&batched.gdn_conv_out, 0),
        batch as u32,
    )
    .map_err(gpu_err)?;

    // A_log and dt_bias carry NO `.weight` suffix in the checkpoint.
    let a_log = norm_view(weights, index, &name("A_log"), v_heads)?;
    let dt_bias = norm_view(weights, index, &name("dt_bias"), v_heads)?;
    gpu::encode_gdn_delta_prefill(
        context,
        pass,
        shape,
        (&batched.gdn_conv_out, 0),
        (&batched.gdn_a, 0),
        (&batched.gdn_b, 0),
        a_log,
        dt_bias,
        qwen.gdn.state_buffer(layer),
        (&batched.gdn_y, 0),
        batch as u32,
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
        (&batched.gdn_y, 0),
        (&batched.gdn_z, 0),
        gated_norm,
        (&batched.gdn_out, 0),
        batch as u32,
    )
    .map_err(gpu_err)?;
    encode_gemm_any(
        context,
        pass,
        weights,
        index,
        &name("out_proj.weight"),
        hidden,
        value_dim,
        (&batched.gdn_out, 0),
        (&batched.o, 0),
        batch,
    )
}

/// Replays the first `keep_rows` rows of the last verify's linear-layer
/// state advance over the verify tape, restoring the recurrent state to
/// what the same rows of that pass produced -- bit for bit, because these
/// are the same kernels over the same recorded inputs from the same
/// snapshot the pass started at. The caller has already written the
/// snapshot back with `GdnStateManager::restore`; this advances each
/// linear layer's conv tail and state across the kept rows and nothing
/// else. The gated norm and the out projection are NOT replayed: their
/// outputs feed the residual stream, which the next pass recomputes from
/// embeddings anyway.
///
/// This is the whole of what a retaining rollback costs on the GDN half.
/// The expensive alternative it replaces -- a full shortened verify pass
/// over the kept rows -- is `speculative.rs`'s old rollback shape, whose
/// cost is the term `docs/DFLASH2.md` names as the throughput driver.
#[allow(clippy::too_many_arguments)]
pub(crate) fn replay_linear_state_batched(
    context: &mut gpu::MetalContext,
    pass: &gpu::PassEncoder,
    weights: &gpu::ResidentGpuWeights,
    index: &ResidentIndex,
    arch: &ArchConfig,
    qwen: &RealQwenState,
    batched: &BatchedScratch,
    keep_rows: usize,
    recorded_batch: usize,
) -> Result<(), RealForwardError> {
    let gpu_err = RealForwardError::Gpu;
    let shape = qwen.shape;
    let qkv_dim = shape.qkv_dim() as usize;
    let v_heads = shape.num_v_heads as usize;
    let conv_kernel = shape.conv_kernel_size as usize;
    let tape = batched.verify_tape.as_ref().ok_or_else(|| {
        RealForwardError::Unsupported(
            "replay needs a verify tape; this scratch was built with record_tape = false"
                .to_string(),
        )
    })?;
    let rows = keep_rows as u32;
    // The tape's per-slot stride is the batch the RECORDING pass actually
    // fed (`recorded_batch`, the verify tape's own `rows`), not the
    // scratch's fixed max capacity (`batched.batch`). The two agree only
    // when a round runs at the scratch's full size; `speculative.rs`
    // legitimately shrinks the round near the end of `max_new_tokens` or
    // `max_context`, and using the wrong (larger) stride there would read
    // every slot past 0 from the wrong byte offset -- silent recurrent-state
    // corruption rather than an error.
    let slot_stride_qkv = recorded_batch * qkv_dim * 2;
    let slot_stride_heads = recorded_batch * v_heads * 2;
    let mut slot = 0usize;
    for layer in 0..arch.num_layers as usize {
        if !arch.layer_is_linear(layer) {
            continue;
        }
        let name = |suffix: &str| layer_tensor(layer, &format!("linear_attn.{suffix}"));
        let qkv_off = (slot * slot_stride_qkv) as u64;
        let heads_off = (slot * slot_stride_heads) as u64;
        let conv_w = norm_view(
            weights,
            index,
            &name("conv1d.weight"),
            qkv_dim * conv_kernel,
        )?;
        gpu::encode_gdn_conv_prefill(
            context,
            pass,
            shape,
            (qwen.gdn.conv_tail_buffer(layer), 0),
            (&tape.qkv_raw, qkv_off),
            conv_w,
            (&batched.gdn_conv_out, 0),
            rows,
            // Undilated, matching both verify call sites.
            1,
        )
        .map_err(gpu_err)?;
        gpu::encode_gdn_conv_tail_update(
            context,
            pass,
            shape,
            (qwen.gdn.conv_tail_buffer(layer), 0),
            (&tape.qkv_raw, qkv_off),
            rows,
            1,
        )
        .map_err(gpu_err)?;
        gpu::encode_gdn_qk_norm(context, pass, shape, (&batched.gdn_conv_out, 0), rows)
            .map_err(gpu_err)?;
        // A_log and dt_bias carry NO `.weight` suffix in the checkpoint,
        // exactly as the verify path's own comment records.
        let a_log = norm_view(weights, index, &name("A_log"), v_heads)?;
        let dt_bias = norm_view(weights, index, &name("dt_bias"), v_heads)?;
        gpu::encode_gdn_delta_prefill(
            context,
            pass,
            shape,
            (&batched.gdn_conv_out, 0),
            (&tape.a, heads_off),
            (&tape.b, heads_off),
            a_log,
            dt_bias,
            qwen.gdn.state_buffer(layer),
            (&batched.gdn_y, 0),
            rows,
        )
        .map_err(gpu_err)?;
        slot += 1;
    }
    Ok(())
}

/// The dense FFN at M rows: batched gate/up, `silu_mul` per token, batched
/// down, then the raw residual add per token.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_dense_ffn_batched(
    context: &mut gpu::MetalContext,
    pass: &gpu::PassEncoder,
    weights: &gpu::ResidentGpuWeights,
    index: &ResidentIndex,
    scratch: &DecodeScratch,
    batched: &BatchedScratch,
    layer: usize,
    hidden: usize,
    inter: usize,
    use_silu: bool,
    batch: usize,
) -> Result<(), RealForwardError> {
    let gpu_err = RealForwardError::Gpu;
    for (suffix, out) in [
        ("mlp.gate_proj.weight", &batched.ffn_gate),
        ("mlp.up_proj.weight", &batched.ffn_up),
    ] {
        encode_gemm_any(
            context,
            pass,
            weights,
            index,
            &prefixed_layer_tensor(TRUNK_PREFIX, layer, suffix),
            inter,
            hidden,
            (&batched.moe_x, 0),
            (out, 0),
            batch,
        )?;
    }

    let act = if use_silu {
        gpu::encode_silu_mul
    } else {
        gpu::encode_gelu_mul
    };
    for m in 0..batch {
        let row = (m * inter) as u64 * 2;
        act(
            context,
            pass,
            (&batched.ffn_gate, row),
            (&batched.ffn_up, row),
            (&batched.ffn_act, row),
            inter as u32,
        )
        .map_err(gpu_err)?;
    }

    encode_gemm_any(
        context,
        pass,
        weights,
        index,
        &prefixed_layer_tensor(TRUNK_PREFIX, layer, "mlp.down_proj.weight"),
        hidden,
        inter,
        (&batched.ffn_act, 0),
        (&batched.h2, 0),
        batch,
    )?;

    for m in 0..batch {
        let row = (m * hidden) as u64 * 2;
        gpu::encode_residual_add(
            context,
            pass,
            (&scratch.x, row),
            (&batched.h2, row),
            hidden as u32,
        )
        .map_err(gpu_err)?;
    }
    Ok(())
}
