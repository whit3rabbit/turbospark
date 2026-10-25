use model_io::{ArchConfig, ResidentIndex};

use super::batched::BatchedScratch;
use crate::families::qwen::{
    layer_tensor, prefixed_layer_tensor, QkNormConvention, RealQwenState, RMS_EPS,
};
use crate::real_forward_dispatch::encode_gemm_rows;
use crate::real_forward_types::{DecodeScratch, RealForwardError};
use crate::real_forward_utils::norm_view;

/// Whether the batched layers run their per-row ops (attention's split, norm,
/// RoPE and gate, the dense FFN's activation and residual add) as one dispatch
/// per tensor over all M rows (the default) or as one per row.
/// Both are bit-identical; `TURBOSPARK_QWEN_BATCHED_ROW_OPS=0` selects the
/// per-row form. Resolved once, never per layer or per token.
pub(crate) fn row_batched_ops() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("TURBOSPARK_QWEN_BATCHED_ROW_OPS").as_deref() != Ok("0"))
}

/// Mask-1 layer at M rows.
///
/// `k_proj` and `v_proj` write STRAIGHT into the KV cache, exactly as the
/// per-token block does -- the only difference is that one dispatch fills M
/// adjacent slots instead of one. That works because the batched kernel's
/// output is token-major with a row stride of `kv_dim` halfs, which is the
/// cache's own per-token stride; the assertion below is what keeps that a
/// checked fact rather than a coincidence.
///
/// `prefix`, `layer` and `qk_norms` name the block as the single-row
/// `attn::encode_full_attention_block` does: the trunk's layers, or the MTP
/// head's one block (`MTP_PREFIX`, layer 0, its own KV, centered q/k norms)
/// when the chunked driver primes the head over a micro-batch.
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
    prefix: &str,
    layer: usize,
    qk_norms: QkNormConvention,
    start_position: usize,
    batch: usize,
) -> Result<(), RealForwardError> {
    let slots = full_attention_kv_slots(arch, kv, layer, start_position)?;
    encode_full_attention_in_proj_batched(
        context, pass, weights, index, arch, batched, slots, prefix, layer, batch, true,
    )?;
    encode_full_attention_core_batched(
        context,
        pass,
        weights,
        index,
        arch,
        qwen,
        scratch,
        batched,
        slots,
        prefix,
        layer,
        qk_norms,
        start_position,
        batch,
    )?;
    encode_full_attention_o_proj_batched(
        context, pass, weights, index, arch, batched, prefix, layer, batch,
    )
}

/// The K and V cache slots a mask-1 layer's batched projections write, as
/// `(k_buf, k_off, v_buf, v_off)` with byte offsets.
pub(crate) type KvSlots<'a> = (&'a gpu::MetalBuffer, usize, &'a gpu::MetalBuffer, usize);

/// The checks that keep [`encode_full_attention_block_batched`]'s K/V
/// projections writing M adjacent cache slots, then the slots themselves.
/// Split out (with the stages below) for the opt-in ANE prefill split
/// (`ane_split.rs`).
pub(crate) fn full_attention_kv_slots<'a>(
    arch: &ArchConfig,
    kv: &'a gpu::KvCacheManager,
    layer: usize,
    start_position: usize,
) -> Result<KvSlots<'a>, RealForwardError> {
    let kv_dim = (arch.num_full_kv_heads * arch.full_head_dim) as usize;
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
    Ok((k_buf, k_off, v_buf, v_off))
}

/// The mask-1 input projections: q (with its gate, `2 * q_dim` rows) unless
/// `with_q` is false, then K and V straight into the cache slots.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_full_attention_in_proj_batched(
    context: &mut gpu::MetalContext,
    pass: &gpu::PassEncoder,
    weights: &gpu::ResidentGpuWeights,
    index: &ResidentIndex,
    arch: &ArchConfig,
    batched: &BatchedScratch,
    slots: KvSlots<'_>,
    prefix: &str,
    layer: usize,
    batch: usize,
    with_q: bool,
) -> Result<(), RealForwardError> {
    let hidden = arch.hidden_size as usize;
    let q_dim = (arch.num_heads * arch.full_head_dim) as usize;
    let kv_dim = (arch.num_full_kv_heads * arch.full_head_dim) as usize;
    let name = |suffix: &str| prefixed_layer_tensor(prefix, layer, &format!("self_attn.{suffix}"));
    let (k_buf, k_off, v_buf, v_off) = slots;
    if with_q {
        encode_gemm_rows(
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
            batched.steel_gemm,
        )?;
    }
    for (suffix, out) in [
        ("k_proj.weight", (k_buf, k_off as u64)),
        ("v_proj.weight", (v_buf, v_off as u64)),
    ] {
        encode_gemm_rows(
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
            batched.steel_gemm,
        )?;
    }
    Ok(())
}

/// The mask-1 layer between its input projections and `o_proj`: q/gate
/// split, q/k norms, RoPE, attention and the output gate into
/// `batched.attn_out`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_full_attention_core_batched(
    context: &mut gpu::MetalContext,
    pass: &gpu::PassEncoder,
    weights: &gpu::ResidentGpuWeights,
    index: &ResidentIndex,
    arch: &ArchConfig,
    qwen: &RealQwenState,
    scratch: &DecodeScratch,
    batched: &BatchedScratch,
    slots: KvSlots<'_>,
    prefix: &str,
    layer: usize,
    qk_norms: QkNormConvention,
    start_position: usize,
    batch: usize,
) -> Result<(), RealForwardError> {
    let gpu_err = RealForwardError::Gpu;
    let num_heads = arch.num_heads as u32;
    let num_kv = arch.num_full_kv_heads as u32;
    let head_dim = arch.full_head_dim as u32;
    let q_dim = (num_heads * head_dim) as usize;
    let kv_dim = (num_kv * head_dim) as usize;
    let name = |suffix: &str| prefixed_layer_tensor(prefix, layer, &format!("self_attn.{suffix}"));
    let (k_buf, k_off, v_buf, _) = slots;

    let q_norm = norm_view(weights, index, &name("q_norm.weight"), head_dim as usize)?;
    let k_norm = norm_view(weights, index, &name("k_norm.weight"), head_dim as usize)?;
    // Per TENSOR, never per family: see `QkNormConvention`.
    let encode_qk_norm = match qk_norms {
        QkNormConvention::Plain => gpu::encode_rms_norm_bf16w_perhead,
        QkNormConvention::Centered => gpu::encode_rms_norm_bf16w_perhead_centered,
    };
    let theta = arch.full_rope_theta as f32;
    // The M rows sit contiguously (`q_packed`/`q`/`attn_gate` at a row stride
    // of a whole number of heads, K at `kv_dim` per slot, checked above), and
    // split, norm and gate are per head or per element, so M rows are just M
    // times the heads in one dispatch with the same arithmetic per element.
    // RoPE takes its row's position from the grid. One dispatch per tensor
    // instead of five per row.
    let rows = batch as u32;
    if row_batched_ops() {
        gpu::encode_split_q_gate(
            context,
            pass,
            (&batched.q_packed, 0),
            (&batched.q, 0),
            (&batched.attn_gate, 0),
            num_heads * rows,
            head_dim,
        )
        .map_err(gpu_err)?;
        encode_qk_norm(
            context,
            pass,
            (&batched.q, 0),
            q_norm,
            (&batched.q, 0),
            num_heads * rows,
            head_dim,
            RMS_EPS,
        )
        .map_err(gpu_err)?;
        let k_rows = (k_buf, k_off as u64);
        encode_qk_norm(
            context,
            pass,
            k_rows,
            k_norm,
            k_rows,
            num_kv * rows,
            head_dim,
            RMS_EPS,
        )
        .map_err(gpu_err)?;
        for (data, heads) in [((&batched.q, 0), num_heads), (k_rows, num_kv)] {
            gpu::encode_rope_neox_subdim_rows(
                context,
                pass,
                data,
                start_position as u32,
                rows,
                heads,
                head_dim,
                qwen.rotary_dim,
                theta,
            )
            .map_err(gpu_err)?;
        }
    } else {
        // The per-row form, kept one env var away (`TURBOSPARK_QWEN_BATCHED_ROW_OPS=0`)
        // as the kill switch and the A side of a paired measurement.
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
            for (data, weight, heads) in [(q_row, q_norm, num_heads), (k_row, k_norm, num_kv)] {
                encode_qk_norm(context, pass, data, weight, data, heads, head_dim, RMS_EPS)
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
    }

    // Attention loops, and the span is what makes the loop causal: query m
    // sees `[0, start_position + m]` even though rows above it are already
    // written. `scratch.attn` is reused across the M queries because command
    // buffers on one queue execute in commit order (`crates/gpu` Gotcha 8) --
    // it is a GPU-only intermediate, so the reuse serializes the queries and
    // cannot corrupt them. Attention is 0.6% of this family's decode compute
    // at short context, where widening it measured not worth a kernel; its
    // long-context cost is what the GQA-shared partial behind
    // `encode_attention_decode` addresses (`real_forward_types.rs`).
    //
    // The chunked-prefill scratch can instead run all M queries through the
    // multi-row GQA partial: one dispatch for the batch where the loop pays
    // two per row, each too small to fill the GPU. Not bit-exact against the
    // loop (its chunk boundaries are shared by the batch), so the verify
    // scratch, which must match sequential decode, has no `rows_attention`
    // and keeps the loop. A single row keeps the loop too, where the
    // multi-row kernel measured 0.89-1.07x. It follows the scratch's one-row
    // kernel choice, so `TURBOSPARK_ATTN_DECODE_KERNEL=barrier` still
    // restores the loop.
    let rows_attention = batched.rows_attention.as_ref().filter(|_| {
        batch > 1 && scratch.attn.decode_kernel() == gpu::DecodeAttentionKernel::GqaShared
    });
    if let Some(rows) = rows_attention {
        gpu::encode_attention_decode_gqa_rows(
            context,
            pass,
            (&batched.q, 0),
            k_buf,
            v_buf,
            rows,
            (&batched.attn_out, 0),
            head_dim,
            num_heads,
            num_kv,
            start_position as u32,
            batch as u32,
            0,
            0,
            arch.attention_scale as f32,
        )
        .map_err(gpu_err)?;
    }
    for m in 0..batch {
        if rows_attention.is_none() {
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
        }
    }
    // Elementwise, so one dispatch over all M rows once every row's attention
    // output is written (the loop above reuses `scratch.attn`, not `attn_out`).
    if row_batched_ops() {
        gpu::encode_sigmoid_gate_mul(
            context,
            pass,
            (&batched.attn_out, 0),
            (&batched.attn_gate, 0),
            (q_dim * batch) as u32,
        )
        .map_err(gpu_err)?;
    } else {
        for m in 0..batch {
            gpu::encode_sigmoid_gate_mul(
                context,
                pass,
                (&batched.attn_out, (m * q_dim) as u64 * 2),
                (&batched.attn_gate, (m * q_dim) as u64 * 2),
                q_dim as u32,
            )
            .map_err(gpu_err)?;
        }
    }
    Ok(())
}

/// The mask-1 layer's `o_proj`, `batched.attn_out` into `batched.o`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_full_attention_o_proj_batched(
    context: &mut gpu::MetalContext,
    pass: &gpu::PassEncoder,
    weights: &gpu::ResidentGpuWeights,
    index: &ResidentIndex,
    arch: &ArchConfig,
    batched: &BatchedScratch,
    prefix: &str,
    layer: usize,
    batch: usize,
) -> Result<(), RealForwardError> {
    let hidden = arch.hidden_size as usize;
    let q_dim = (arch.num_heads * arch.full_head_dim) as usize;
    let name = |suffix: &str| prefixed_layer_tensor(prefix, layer, &format!("self_attn.{suffix}"));
    encode_gemm_rows(
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
        batched.steel_gemm,
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
    encode_linear_in_proj_batched(
        context, pass, weights, index, arch, qwen, batched, layer, batch, true,
    )?;
    encode_linear_core_batched(
        context, pass, weights, index, qwen, batched, layer, batch, tape_slot,
    )?;
    encode_linear_out_proj_batched(
        context, pass, weights, index, arch, qwen, batched, layer, batch,
    )
}

/// The mask-2 input projections: qkv and z unless `with_qkvz` is false,
/// then a and b.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_linear_in_proj_batched(
    context: &mut gpu::MetalContext,
    pass: &gpu::PassEncoder,
    weights: &gpu::ResidentGpuWeights,
    index: &ResidentIndex,
    arch: &ArchConfig,
    qwen: &RealQwenState,
    batched: &BatchedScratch,
    layer: usize,
    batch: usize,
    with_qkvz: bool,
) -> Result<(), RealForwardError> {
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
    let all = [
        ("in_proj_qkv.weight", qkv_dim, &batched.gdn_qkv_raw),
        ("in_proj_z.weight", value_dim, &batched.gdn_z),
        ("in_proj_a.weight", v_heads, &batched.gdn_a),
        ("in_proj_b.weight", v_heads, &batched.gdn_b),
    ];
    for &(suffix, rows, out) in &all[if with_qkvz { 0 } else { 2 }..] {
        encode_gemm_rows(
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
            batched.steel_gemm,
        )?;
    }
    Ok(())
}

/// The mask-2 layer between its input projections and `out_proj`: the tape
/// copy, conv, q/k norm, delta rule and gated norm into `batched.gdn_out`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_linear_core_batched(
    context: &mut gpu::MetalContext,
    pass: &gpu::PassEncoder,
    weights: &gpu::ResidentGpuWeights,
    index: &ResidentIndex,
    qwen: &RealQwenState,
    batched: &BatchedScratch,
    layer: usize,
    batch: usize,
    tape_slot: Option<usize>,
) -> Result<(), RealForwardError> {
    let gpu_err = RealForwardError::Gpu;
    let shape = qwen.shape;
    let qkv_dim = shape.qkv_dim() as usize;
    let v_heads = shape.num_v_heads as usize;
    let name = |suffix: &str| layer_tensor(layer, &format!("linear_attn.{suffix}"));

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
    // THE LAZY SPLIT (TURBOSPARK_GDN_LAZY_TAPE, default on): with a rank-1
    // tape, the two in-place state writes below are DEFERRED. The conv tail
    // update does not run here at all -- the round-end fold re-runs it over
    // exactly the accepted rows -- and the delta kernel runs in its
    // WRITE_STATE=false build, which leaves the live state at the block
    // start and records each row's delta instead. y is identical either
    // way: nothing after the register load reads the state back.
    let rank1 = batched
        .verify_tape
        .as_ref()
        .and_then(|tape| tape.rank1.as_ref());
    if rank1.is_none() {
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
    }
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
    if let (Some(_tape_present), Some(rank1)) = (batched.verify_tape.as_ref(), rank1) {
        let slot = tape_slot.ok_or_else(|| {
            RealForwardError::Unsupported(
                "a lazy tape-carrying scratch must be driven with a tape slot".to_string(),
            )
        })?;
        // The post-norm k slice, packed per row out of conv_out (whose row
        // stride is the full qkv width) into the rank-1 tape. This is the k
        // the delta kernel below consumes, captured before the scratch moves
        // on; the fold reads it back as the exact halves this pass used.
        let k_width = (shape.num_k_heads * shape.key_head_dim) as usize;
        let k_off = (slot * batch * k_width * 2) as u64;
        gpu::encode_copy_rows_fp16_strided(
            context,
            pass,
            (
                &batched.gdn_conv_out,
                (k_width * 2) as u64, // the k slice starts after q
            ),
            (&rank1.k, k_off),
            batch as u32,
            k_width as u32,
            qkv_dim as u32,
            k_width as u32,
        )
        .map_err(gpu_err)?;
        let delta_off = (slot * batch * shape.value_dim() as usize * 4) as u64;
        gpu::encode_gdn_delta_prefill_lazy(
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
            (&rank1.delta, delta_off),
            batch as u32,
        )
        .map_err(gpu_err)?;
    } else {
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
    }

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
    .map_err(gpu_err)
}

/// The mask-2 layer's `out_proj`, `batched.gdn_out` into `batched.o`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_linear_out_proj_batched(
    context: &mut gpu::MetalContext,
    pass: &gpu::PassEncoder,
    weights: &gpu::ResidentGpuWeights,
    index: &ResidentIndex,
    arch: &ArchConfig,
    qwen: &RealQwenState,
    batched: &BatchedScratch,
    layer: usize,
    batch: usize,
) -> Result<(), RealForwardError> {
    let hidden = arch.hidden_size as usize;
    let value_dim = qwen.shape.value_dim() as usize;
    let name = |suffix: &str| layer_tensor(layer, &format!("linear_attn.{suffix}"));
    encode_gemm_rows(
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
        batched.steel_gemm,
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

/// Commits the first `keep_rows` rows of the LAZY verify tape into the live
/// recurrent state: the round-end half of `TURBOSPARK_GDN_LAZY_TAPE`. Per
/// linear layer it runs the same two kernels the eager path's state writes
/// effectively summarize -- `gdn_conv_tail_update` over the kept taped rows
/// (the very kernel [`replay_linear_state_batched`] dispatches) and
/// `gdn_state_fold`, which replays the recorded per-row rank-1 update into
/// the registers and writes the result once. The live state was never moved
/// by the verify, so starting from the block start and applying `keep_rows`
/// updates lands bit for bit where the eager verify's own write-back, or the
/// eager restore-plus-replay, left it.
///
/// `recorded_batch` is the batch the RECORDING pass fed, the same stride
/// rule [`replay_linear_state_batched`] documents: it is the tape's own row
/// count, not this scratch's capacity, and reading slots past 0 with the
/// wrong stride is silent recurrent-state corruption.
#[allow(clippy::too_many_arguments)]
pub(crate) fn commit_linear_state_batched(
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
    let k_width = (shape.num_k_heads * shape.key_head_dim) as usize;
    let delta_width = shape.value_dim() as usize;
    let tape = batched.verify_tape.as_ref().ok_or_else(|| {
        RealForwardError::Unsupported(
            "the lazy commit needs a verify tape; this scratch was built with record_tape = false"
                .to_string(),
        )
    })?;
    let rank1 = tape.rank1.as_ref().ok_or_else(|| {
        RealForwardError::Unsupported(
            "the lazy commit needs a rank-1 tape; this scratch was built with \
             TURBOSPARK_GDN_LAZY_TAPE=0"
                .to_string(),
        )
    })?;
    let rows = keep_rows as u32;
    // Byte strides per linear-layer slot, from the batch the recording pass
    // actually fed (see the stride warning on `replay_linear_state_batched`).
    let slot_stride_qkv = recorded_batch * qkv_dim * 2;
    let slot_stride_heads = recorded_batch * v_heads * 2;
    let slot_stride_k = recorded_batch * k_width * 2;
    let slot_stride_delta = recorded_batch * delta_width * 4;
    let mut slot = 0usize;
    for layer in 0..arch.num_layers as usize {
        if !arch.layer_is_linear(layer) {
            continue;
        }
        let name = |suffix: &str| layer_tensor(layer, &format!("linear_attn.{suffix}"));
        let qkv_off = (slot * slot_stride_qkv) as u64;
        let heads_off = (slot * slot_stride_heads) as u64;
        let k_off = (slot * slot_stride_k) as u64;
        let delta_off = (slot * slot_stride_delta) as u64;
        gpu::encode_gdn_conv_tail_update(
            context,
            pass,
            shape,
            (qwen.gdn.conv_tail_buffer(layer), 0),
            (&tape.qkv_raw, qkv_off),
            rows,
            // Undilated, matching the verify and replay call sites.
            1,
        )
        .map_err(gpu_err)?;
        // A_log and dt_bias carry NO `.weight` suffix in the checkpoint,
        // exactly as the verify and replay paths' comments record.
        let a_log = norm_view(weights, index, &name("A_log"), v_heads)?;
        let dt_bias = norm_view(weights, index, &name("dt_bias"), v_heads)?;
        gpu::encode_gdn_state_fold(
            context,
            pass,
            shape,
            (&rank1.k, k_off),
            (&rank1.delta, delta_off),
            (&tape.a, heads_off),
            a_log,
            dt_bias,
            qwen.gdn.state_buffer(layer),
            rows,
        )
        .map_err(gpu_err)?;
        slot += 1;
    }
    Ok(())
}

/// The dense FFN at M rows: batched gate/up, `silu_mul` over the rows, batched
/// down, then the raw residual add over the rows. `prefix` is `TRUNK_PREFIX`,
/// or `MTP_PREFIX` for the head, whose FFN is the trunk's dense width.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_dense_ffn_batched(
    context: &mut gpu::MetalContext,
    pass: &gpu::PassEncoder,
    weights: &gpu::ResidentGpuWeights,
    index: &ResidentIndex,
    scratch: &DecodeScratch,
    batched: &BatchedScratch,
    prefix: &str,
    layer: usize,
    hidden: usize,
    inter: usize,
    use_silu: bool,
    batch: usize,
) -> Result<(), RealForwardError> {
    for (suffix, out) in [
        ("mlp.gate_proj.weight", &batched.ffn_gate),
        ("mlp.up_proj.weight", &batched.ffn_up),
    ] {
        encode_gemm_rows(
            context,
            pass,
            weights,
            index,
            &prefixed_layer_tensor(prefix, layer, suffix),
            inter,
            hidden,
            (&batched.moe_x, 0),
            (out, 0),
            batch,
            batched.steel_gemm,
        )?;
    }
    encode_dense_ffn_tail_batched(
        context, pass, weights, index, scratch, batched, prefix, layer, hidden, inter, use_silu,
        batch,
    )
}

/// [`encode_dense_ffn_batched`] after gate and up: `silu_mul` per token,
/// batched down, then the raw residual add per token. Split out so the ANE
/// prefill split can put gate and up in their own command buffers.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_dense_ffn_tail_batched(
    context: &mut gpu::MetalContext,
    pass: &gpu::PassEncoder,
    weights: &gpu::ResidentGpuWeights,
    index: &ResidentIndex,
    scratch: &DecodeScratch,
    batched: &BatchedScratch,
    prefix: &str,
    layer: usize,
    hidden: usize,
    inter: usize,
    use_silu: bool,
    batch: usize,
) -> Result<(), RealForwardError> {
    encode_dense_ffn_act_batched(context, pass, batched, inter, use_silu, batch)?;
    encode_gemm_rows(
        context,
        pass,
        weights,
        index,
        &prefixed_layer_tensor(prefix, layer, "mlp.down_proj.weight"),
        hidden,
        inter,
        (&batched.ffn_act, 0),
        (&batched.h2, 0),
        batch,
        batched.steel_gemm,
    )?;
    encode_dense_ffn_residual_batched(context, pass, scratch, batched, hidden, batch)
}

/// The FFN activation per token: `ffn_act = act(ffn_gate) * ffn_up`.
pub(crate) fn encode_dense_ffn_act_batched(
    context: &mut gpu::MetalContext,
    pass: &gpu::PassEncoder,
    batched: &BatchedScratch,
    inter: usize,
    use_silu: bool,
    batch: usize,
) -> Result<(), RealForwardError> {
    let gpu_err = RealForwardError::Gpu;
    let act = if use_silu {
        gpu::encode_silu_mul
    } else {
        gpu::encode_gelu_mul
    };
    // Elementwise over M contiguous rows, so one dispatch of M * inter is the
    // same arithmetic per element (`row_batched_ops` selects the per-row form).
    if row_batched_ops() {
        act(
            context,
            pass,
            (&batched.ffn_gate, 0),
            (&batched.ffn_up, 0),
            (&batched.ffn_act, 0),
            (inter * batch) as u32,
        )
        .map_err(gpu_err)?;
    } else {
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
    }
    Ok(())
}

/// The FFN's raw residual add per token, `batched.h2` into `scratch.x`.
pub(crate) fn encode_dense_ffn_residual_batched(
    context: &mut gpu::MetalContext,
    pass: &gpu::PassEncoder,
    scratch: &DecodeScratch,
    batched: &BatchedScratch,
    hidden: usize,
    batch: usize,
) -> Result<(), RealForwardError> {
    let gpu_err = RealForwardError::Gpu;
    if row_batched_ops() {
        gpu::encode_residual_add(
            context,
            pass,
            (&scratch.x, 0),
            (&batched.h2, 0),
            (hidden * batch) as u32,
        )
        .map_err(gpu_err)?;
    } else {
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
    }
    Ok(())
}
