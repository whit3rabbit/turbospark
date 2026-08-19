//! The M-row trunk forward: `docs/MTP_SPECULATIVE.md` step 4's verify pass.
//!
//! This is the FIRST batched math path on a decode flow in this repo, and
//! the distinction matters because one already exists that is not one:
//! `prefill_chunk_real_gemma4` batches COMMAND BUFFERS and still calls
//! `encode_attention_decode` once per token, which is why its win is the
//! per-layer blocking wait rather than the kernels. Here the GEMVs really do
//! become GEMMs.
//!
//! **WHAT IS BATCHED AND WHAT IS NOT COMES FROM THE MEASURED COMPUTE SPLIT**
//! (`docs/MTP_SPECULATIVE.md`: GEMV 93.1%, norms and elementwise 4.1%, GDN
//! recurrence 2.2%, attention 0.6%, un-amortizable floor 6.4%). Every GEMV is
//! batched; norms, RoPE, attention and the recurrent step loop per token.
//! That is not a first-cut simplification to be tightened later -- it is what
//! the composite those numbers feed already assumes, and widening attention
//! at decode context would buy 0.6% of a pass.
//!
//! Three refusals, all BY NAME rather than by falling back:
//!
//! 1. **Dense only.** A batched routed-expert pair does not exist, and the
//!    union of M tokens' experts is larger than what the slot cache already
//!    loads (AGENTS.md Gotcha 54), so there is nothing to fall back TO.
//! 2. **INT4 only**, enforced one level down in `encode_gemm_any`. The 1-bit
//!    and 2-bit checkpoints of this same architecture have no batched kernel.
//! 3. **No KV wrap.** M consecutive positions occupy M ADJACENT slots only
//!    while `position % capacity` does not roll over inside the block.
//!
//! The first two would otherwise be silent: a sequential fallback is
//! numerically identical, so it would pass every losslessness test while
//! making the measurement describe the wrong engine.

use foundation::LogitValue;
use model_io::{ArchConfig, ResidentIndex};

use crate::families::qwen::{
    layer_tensor, prefixed_layer_tensor, RealQwenState, RMS_EPS, TRUNK_PREFIX,
};
use crate::real_forward::RealForwardRunner;
use crate::real_forward_dispatch::{encode_embed_any, encode_gemm_any};
use crate::real_forward_types::{DecodeScratch, RealForwardError};
use crate::real_forward_utils::norm_view;

/// The M-row siblings of the buffers `DecodeScratch` and `RealQwenState`
/// hold one row of.
///
/// **Allocated only when a draft depth was asked for**, beside [`MtpState`]
/// and for its reason: `MFERENCE_MTP_DRAFT` unset must allocate nothing, so
/// that `qwen38_memory_oracle`'s frozen row keeps describing the engine that
/// shipped before this module existed. At the block sizes that pay this is
/// ~10 MiB, almost all of it `logits` (`batch * vocab` halfs, and this
/// family's vocab is 248,320).
///
/// `x` is NOT here: `DecodeScratch::x` already holds `MAX_PREFILL_BATCH`
/// rows, because the residual stream crosses layers and the chunk driver
/// needed the same thing first.
pub(crate) struct BatchedScratch {
    /// Rows this was sized for. A verify of block M needs `M + 1`.
    pub(crate) batch: usize,
    normed: gpu::MetalBuffer,
    q_packed: gpu::MetalBuffer,
    q: gpu::MetalBuffer,
    attn_gate: gpu::MetalBuffer,
    attn_out: gpu::MetalBuffer,
    o: gpu::MetalBuffer,
    gdn_qkv_raw: gpu::MetalBuffer,
    gdn_conv_out: gpu::MetalBuffer,
    gdn_z: gpu::MetalBuffer,
    gdn_a: gpu::MetalBuffer,
    gdn_b: gpu::MetalBuffer,
    gdn_y: gpu::MetalBuffer,
    gdn_out: gpu::MetalBuffer,
    moe_x: gpu::MetalBuffer,
    ffn_gate: gpu::MetalBuffer,
    ffn_up: gpu::MetalBuffer,
    ffn_act: gpu::MetalBuffer,
    h2: gpu::MetalBuffer,
    logits: gpu::MetalBuffer,
}

impl BatchedScratch {
    pub(crate) fn new(
        context: &gpu::MetalContext,
        arch: &ArchConfig,
        qwen_shape: gpu::GdnShape,
        batch: usize,
    ) -> Self {
        let hidden = arch.hidden_size as u64;
        let q_dim = (arch.num_heads * arch.full_head_dim) as u64;
        let inter = arch.intermediate_size as u64;
        let vocab = arch.vocab_size as u64;
        let qkv_dim = qwen_shape.qkv_dim() as u64;
        let value_dim = qwen_shape.value_dim() as u64;
        let v_heads = qwen_shape.num_v_heads as u64;
        let b = batch as u64;
        let halfs = |n: u64| context.new_output_buffer(n.max(1) * b * 2);
        Self {
            batch,
            normed: halfs(hidden),
            q_packed: halfs(2 * q_dim),
            q: halfs(q_dim),
            attn_gate: halfs(q_dim),
            attn_out: halfs(q_dim),
            o: halfs(hidden),
            gdn_qkv_raw: halfs(qkv_dim),
            gdn_conv_out: halfs(qkv_dim),
            gdn_z: halfs(value_dim),
            gdn_a: halfs(v_heads),
            gdn_b: halfs(v_heads),
            gdn_y: halfs(value_dim),
            gdn_out: halfs(value_dim),
            moe_x: halfs(hidden),
            ffn_gate: halfs(inter),
            ffn_up: halfs(inter),
            ffn_act: halfs(inter),
            h2: halfs(hidden),
            logits: halfs(vocab),
        }
    }
}

impl RealForwardRunner {
    /// Runs `tokens` through the trunk in ONE pass, writing `tokens.len() *
    /// vocab` logits.
    ///
    /// Token `m` occupies `start_position + m` and attends over
    /// `[0, start_position + m]`, so the causal structure is identical to
    /// running the same tokens sequentially -- and that is not arranged here,
    /// it falls out of `encode_attention_decode` taking its span from the
    /// position ARGUMENT rather than from the cache cursor. All M KV rows are
    /// written before any query reads, and a query simply does not look at
    /// the rows above its own span.
    ///
    /// The KV cursor advances by `tokens.len()`, exactly as the same tokens
    /// run one at a time would leave it, so `rollback` needs no batched
    /// variant.
    pub fn produce_batched(
        &mut self,
        tokens: &[i32],
        start_position: usize,
        logits: &mut [LogitValue],
    ) -> Result<(), RealForwardError> {
        let arch = self.arch.clone();
        let hidden = arch.hidden_size as usize;
        let inter = arch.intermediate_size as usize;
        let vocab = arch.vocab_size as usize;
        let use_silu = arch.hidden_activation.contains("silu");
        let batch = tokens.len();
        let gpu_err = RealForwardError::Gpu;

        if arch.num_experts != 0 {
            return Err(RealForwardError::Unsupported(
                "batched forward is dense-only: the routed-expert pair has no batched \
                 kernel, and the union of M tokens' experts exceeds what the slot cache \
                 already loads (AGENTS.md Gotcha 54), so there is nothing to batch"
                    .to_string(),
            ));
        }
        if batch == 0 {
            return Err(RealForwardError::Unsupported(
                "batched forward needs at least one token".to_string(),
            ));
        }
        // A DFlash2 install carries no MTP head but needs the SAME verify
        // pass, so its state owns a BatchedScratch of its own and either
        // drafter's scratch serves.
        let batched =
            match (&self.real_mtp, &self.real_dflash) {
                (Some(m), _) => &m.batched,
                (None, Some(d)) => &d.batched,
                (None, None) => return Err(RealForwardError::Unsupported(
                    "no batched scratch; set MFERENCE_MTP_DRAFT or MFERENCE_DFLASH_DRAFT before \
                     opening the model"
                        .to_string(),
                )),
            };
        if batch > batched.batch {
            return Err(RealForwardError::Unsupported(format!(
                "batched forward of {batch} rows against scratch sized for {}",
                batched.batch
            )));
        }
        if start_position != self.kv.position() {
            return Err(RealForwardError::Unsupported(format!(
                "non-sequential batched start {start_position}; KV cache is at {}",
                self.kv.position()
            )));
        }
        if logits.len() != batch * vocab {
            return Err(RealForwardError::Unsupported(format!(
                "batched logits buffer is {}, expected {batch} x {vocab}",
                logits.len()
            )));
        }
        for &t in tokens {
            if (t as usize) >= vocab {
                return Err(RealForwardError::Unsupported(format!(
                    "token id {t} outside vocab {vocab}"
                )));
            }
        }
        // M consecutive positions are M ADJACENT slots only while
        // `position % capacity` does not roll over inside the block. This
        // family has no sliding window so its full layers are linear, but
        // "linear" still wraps at `max_context`; a batched projection writing
        // across that boundary would scatter into row 0 with no symptom
        // beyond wrong attention.
        for layer in 0..arch.num_layers as usize {
            if arch.layer_is_linear(layer) {
                continue;
            }
            let capacity = self.kv.capacity(layer);
            if start_position % capacity + batch > capacity {
                return Err(RealForwardError::Unsupported(format!(
                    "batched forward of {batch} rows at position {start_position} wraps \
                     layer {layer}'s KV capacity {capacity}; the batched projection writes \
                     M adjacent slots and cannot straddle the boundary"
                )));
            }
        }

        let embed_name = "language_model.model.embed_tokens.weight";
        let pass = self.context.begin_pass_labeled("batched verify");
        let (context, weights, index, scratch, qwen, kv, dflash) = (
            &mut self.context,
            &self.weights,
            &self.index,
            &self.scratch,
            self.real_qwen
                .as_ref()
                .ok_or_else(|| RealForwardError::Unsupported("not a Qwen install".to_string()))?,
            &mut self.kv,
            self.real_dflash.as_ref(),
        );

        // An embedding lookup has no trip count to amortize, so it loops for
        // the same reason the norms below do.
        for (m, &token) in tokens.iter().enumerate() {
            encode_embed_any(
                context,
                &pass,
                weights,
                index,
                embed_name,
                (&scratch.x, (m * hidden) as u64 * 2),
                token as u32,
                hidden as u32,
                1.0,
            )?;
        }

        for layer in 0..arch.num_layers as usize {
            let input_norm = norm_view(
                weights,
                index,
                &layer_tensor(layer, "input_layernorm.weight"),
                hidden,
            )?;
            for m in 0..batch {
                gpu::encode_rms_norm_bf16w(
                    context,
                    &pass,
                    (&scratch.x, (m * hidden) as u64 * 2),
                    input_norm,
                    (&batched.normed, (m * hidden) as u64 * 2),
                    hidden as u32,
                    RMS_EPS,
                )
                .map_err(gpu_err)?;
            }

            if arch.layer_is_linear(layer) {
                encode_linear_block_batched(
                    context, &pass, weights, index, &arch, qwen, batched, layer, batch,
                )?;
            } else {
                encode_full_attention_block_batched(
                    context,
                    &pass,
                    weights,
                    index,
                    &arch,
                    qwen,
                    scratch,
                    batched,
                    kv,
                    layer,
                    start_position,
                    batch,
                )?;
            }

            let post_attn = norm_view(
                weights,
                index,
                &layer_tensor(layer, "post_attention_layernorm.weight"),
                hidden,
            )?;
            for m in 0..batch {
                let row = (m * hidden) as u64 * 2;
                // RAW residual add: this family has no sandwich norms, and
                // normalizing here took the Qwen 3.6 reference perplexity
                // from 6.25 to 255,409 once already (crate Gotcha 11).
                gpu::encode_residual_add(
                    context,
                    &pass,
                    (&scratch.x, row),
                    (&batched.o, row),
                    hidden as u32,
                )
                .map_err(gpu_err)?;
                gpu::encode_rms_norm_bf16w(
                    context,
                    &pass,
                    (&scratch.x, row),
                    post_attn,
                    (&batched.moe_x, row),
                    hidden as u32,
                    RMS_EPS,
                )
                .map_err(gpu_err)?;
            }

            encode_dense_ffn_batched(
                context, &pass, weights, index, scratch, batched, layer, hidden, inter, use_silu,
                batch,
            )?;

            // THE DFLASH2 AUX CAPTURE at M rows: the residual rows this
            // layer just produced, into the fc input's layout. Same point
            // as the per-token hook (`families/qwen/mod.rs`), same zero
            // dispatches when no drafter is open.
            if let Some(d) = dflash {
                if let Some(aux) = d.aux_slot(layer) {
                    gpu::encode_dflash_copy_rows(
                        context,
                        &pass,
                        (&scratch.x, 0),
                        (&d.capture, (aux * hidden) as u64 * 2),
                        batch as u32,
                        hidden as u32,
                        (d.shape.aux_count * hidden) as u32,
                    )
                    .map_err(gpu_err)?;
                }
            }
        }

        let final_norm = norm_view(weights, index, "language_model.model.norm.weight", hidden)?;
        for m in 0..batch {
            gpu::encode_rms_norm_bf16w(
                context,
                &pass,
                (&scratch.x, (m * hidden) as u64 * 2),
                final_norm,
                (&batched.normed, (m * hidden) as u64 * 2),
                hidden as u32,
                RMS_EPS,
            )
            .map_err(gpu_err)?;
        }
        let head_name = if arch.tie_word_embeddings {
            embed_name.to_string()
        } else {
            "language_model.lm_head.weight".to_string()
        };
        encode_gemm_any(
            context,
            &pass,
            weights,
            index,
            &head_name,
            vocab,
            hidden,
            (&batched.normed, 0),
            (&batched.logits, 0),
            batch,
        )?;

        pass.commit_and_wait();
        for _ in 0..batch {
            kv.advance();
        }
        // THE LAST ROW'S RESIDUAL HAS TO LAND AT ROW 0, because that is where
        // a SEQUENTIAL run of the same tokens leaves it and because
        // `mtp_draft_step` reads `h_t` from `scratch.x` at offset 0.
        //
        // Without this the head drafts off the FIRST token of the block
        // instead of the last, and the failure is invisible in every place
        // one would look: the trunk is untouched, so the committed stream
        // stays byte-identical to a non-speculative run and the losslessness
        // gate passes. What moves is the DRAFTER's quality -- measured on the
        // real install, accept length fell from 1.84 to 1.10 per round and
        // rollbacks went from 0 to 62 of 123 rounds, which reads as a verdict
        // about MTP rather than as a bug in the pass.
        //
        // A host copy rather than a blit: `commit_and_wait` has already run,
        // it is `hidden` halfs (10 KiB here), and it needs no kernel.
        if batch > 1 {
            let row = hidden * 2;
            let last = gpu::read_buffer_bytes(&scratch.x, (batch - 1) * row, row);
            gpu::write_buffer_bytes(&scratch.x, 0, &last);
        }
        gpu::read_buffer_f16_into(&batched.logits, 0, logits);
        // The dflash capture's bookkeeping, last so it cannot alias the
        // scratch borrow above: `batch` rows whose positions start at
        // `start_position`, ready for the next round's context write over
        // the accepted prefix.
        if let Some(d) = self.real_dflash.as_mut() {
            d.note_capture(start_position, batch);
        }
        Ok(())
    }
}

/// Mask-1 layer at M rows.
///
/// `k_proj` and `v_proj` write STRAIGHT into the KV cache, exactly as the
/// per-token block does -- the only difference is that one dispatch fills M
/// adjacent slots instead of one. That works because the batched kernel's
/// output is token-major with a row stride of `kv_dim` halfs, which is the
/// cache's own per-token stride; the assertion below is what keeps that a
/// checked fact rather than a coincidence.
#[allow(clippy::too_many_arguments)]
fn encode_full_attention_block_batched(
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

    if kv.stride(layer) != kv_dim * 2 {
        return Err(RealForwardError::Unsupported(format!(
            "layer {layer}: KV stride {} is not {kv_dim} halfs, so a batched projection \
             cannot write M adjacent slots in one dispatch",
            kv.stride(layer)
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
fn encode_linear_block_batched(
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

/// The dense FFN at M rows: batched gate/up, `silu_mul` per token, batched
/// down, then the raw residual add per token.
#[allow(clippy::too_many_arguments)]
fn encode_dense_ffn_batched(
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
