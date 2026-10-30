//! Chunked prefill for `lfm3`: every layer of a micro-batch of up to
//! [`LFM3_PREFILL_ROWS`] tokens in ONE command buffer, with each projection
//! run once for all rows (`encode_gemm_rows`) instead of once per token.
//!
//! The sequential path reads every weight once per prompt token, which is
//! why a 2.9K-token prompt took about 13 s on the 1.7B install. Here a
//! weight is read once per micro-batch.
//!
//! Exactness, in two arms:
//! - `TURBOSPARK_PREFILL_STEEL=0` and `TURBOSPARK_PREFILL_ROWS_ATTN=0`: the
//!   scalar batched GEMM, the one-row attention kernel per row, and per-row
//!   ops that each reproduce the sequential kernel's arithmetic. Bit-exact
//!   against `produce` (`tests/real_forward_lfm3.rs` pins it).
//! - The default: Steel INT4 GEMMs from 32 rows and the multi-row GQA
//!   attention on the FULL layers. Neither is bit-exact against the scalar
//!   kernels (`docs/SPEEDUP_PLAYBOOK.md`), so this arm is held to a KL probe.
//!
//! Sliding layers keep the one-row kernel per row in both arms: the
//! multi-row kernel takes one `kv_start` for the whole batch, and each
//! sliding row has its own window start.
//!
//! K and V are projected into staging rows, normed and rotated there, then
//! copied into the cache in runs of contiguous slots. A sliding layer's ring
//! can wrap inside a micro-batch, so a projection cannot write it directly.
//! The ring holds `window + max chunk` rows, so a micro-batch no wider than
//! `ring - window` never evicts a position its own rows still read; the
//! width is capped to that.
//!
//! Only INT4-affine projections have the batched kernel this driver needs,
//! so the GGUF Q8_0 install keeps the sequential path
//! ([`RealForwardRunner::lfm3_chunked_refusal`]).

use std::time::Instant;

use foundation::LogitValue;

use super::state::{Lfm3Batched, RMS_EPS};
use super::{attn::rope_params_for, layer_tensor};
use crate::kv_write::encode_attention_any;
use crate::real_forward::RealForwardRunner;
use crate::real_forward_dispatch::{encode_embed_any, encode_gemm_rows, encode_gemv_any};
use crate::real_forward_types::{RealForwardError, MAX_PREFILL_ROWS};
use crate::real_forward_utils::{entry, norm_view};

/// The widest micro-batch: every row of `DecodeScratch::x`.
pub(crate) const LFM3_PREFILL_ROWS: usize = MAX_PREFILL_ROWS;

/// `TURBOSPARK_LFM3_CHUNKED_PREFILL`, default on, read once. `0` runs the
/// pre-driver per-token loop behind the same chunk entry point, so both arms
/// of an A/B carry the chunked label (the flint precedent).
pub(crate) fn lfm3_chunked_prefill() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("TURBOSPARK_LFM3_CHUNKED_PREFILL").as_deref() != Ok("0"))
}

/// The projections this driver runs as M-row GEMMs, per layer.
fn projections(conv: bool) -> &'static [&'static str] {
    const CONV: &[&str] = &[
        "conv.in_proj.weight",
        "conv.out_proj.weight",
        "feed_forward.w1.weight",
        "feed_forward.w2.weight",
        "feed_forward.w3.weight",
    ];
    const ATTN: &[&str] = &[
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
        "self_attn.gate_proj.weight",
        "feed_forward.w1.weight",
        "feed_forward.w2.weight",
        "feed_forward.w3.weight",
    ];
    if conv {
        CONV
    } else {
        ATTN
    }
}

impl RealForwardRunner {
    /// Why this install cannot take the chunked driver, or `None`. One
    /// function answers both `supports_chunked_prefill` and the driver's own
    /// refusal, so they cannot disagree.
    pub(crate) fn lfm3_chunked_refusal(&self) -> Option<String> {
        for layer in 0..self.arch.num_layers as usize {
            for suffix in projections(self.arch.layer_is_short_conv(layer)) {
                let name = layer_tensor(layer, suffix);
                match entry(&self.index, &name) {
                    Ok(e) if e.dtype == 4 => {}
                    Ok(e) => {
                        return Some(format!(
                            "{name} is dtype {}; the lfm3 chunked driver needs INT4-affine \
                             projections, the only ones with a batched GEMM it runs",
                            e.dtype
                        ))
                    }
                    Err(e) => return Some(e.to_string()),
                }
            }
            if !self.arch.layer_is_short_conv(layer) && self.kv.layer_quant(layer).is_some() {
                return Some(
                    "--kv-bits is not wired to the lfm3 chunked driver; it writes FP16 K/V rows"
                        .to_string(),
                );
            }
        }
        None
    }

    /// Runs a whole prefill chunk, writing the logits for the position after
    /// its last token.
    pub(crate) fn prefill_chunk_real_lfm3(
        &mut self,
        tokens: &[i32],
        start_position: usize,
        logits: &mut [LogitValue],
    ) -> Result<(), RealForwardError> {
        if tokens.is_empty() {
            return Err(RealForwardError::Unsupported(
                "prefill_chunk called with an empty chunk".to_string(),
            ));
        }
        if let Some(reason) = self.lfm3_chunked_refusal() {
            return Err(RealForwardError::Unsupported(reason));
        }
        if !lfm3_chunked_prefill() {
            return self.prefill_chunk_lfm3_sequential(tokens, start_position, logits);
        }
        // A per-run diagnostic, so refused here rather than folded into the
        // install predicate above.
        if self.ffn_hist.is_some() {
            return Err(RealForwardError::Unsupported(
                "TURBOSPARK_FFN_HIST is not wired to the lfm3 chunked driver; run with \
                 TURBOSPARK_LFM3_CHUNKED_PREFILL=0"
                    .to_string(),
            ));
        }
        let width = self.lfm3_prefill_width();
        if self
            .real_lfm3
            .as_ref()
            .is_some_and(|s| s.batched.as_ref().is_none_or(|b| b.rows != width))
        {
            let arch = self.arch.clone();
            let batched = Lfm3Batched::new(&mut self.context, &arch, width);
            self.real_lfm3.as_mut().expect("checked above").batched = Some(batched);
        }
        foundation::engaged!("lfm3_chunked_prefill");
        let mut offset = 0usize;
        while offset < tokens.len() {
            let take = (tokens.len() - offset).min(width);
            let last = offset + take == tokens.len();
            let started = Instant::now();
            let result = self.prefill_micro_batch_lfm3(
                &tokens[offset..offset + take],
                start_position + offset,
                last,
                logits,
            );
            self.phases.calls += take as u64;
            self.phases.total_nanos += started.elapsed().as_nanos() as u64;
            result?;
            offset += take;
        }
        Ok(())
    }

    /// The pre-driver behaviour: one `produce` per token, the head only on
    /// the last. The OFF arm of the switch.
    fn prefill_chunk_lfm3_sequential(
        &mut self,
        tokens: &[i32],
        start_position: usize,
        logits: &mut [LogitValue],
    ) -> Result<(), RealForwardError> {
        let last = tokens.len() - 1;
        for (i, &token) in tokens.iter().enumerate() {
            let position = start_position + i;
            if i == last {
                self.produce_real_lfm3(token, position, logits)?;
            } else {
                self.skip_head = true;
                let result = self.produce_real_lfm3(token, position, logits);
                self.skip_head = false;
                result?;
            }
        }
        Ok(())
    }

    /// [`LFM3_PREFILL_ROWS`], capped so that no sliding layer's ring evicts a
    /// position a row of the same micro-batch still reads.
    fn lfm3_prefill_width(&self) -> usize {
        let window = self.arch.sliding_window.max(0) as usize;
        (0..self.arch.num_layers as usize)
            .map(|layer| self.kv.ring_capacity(layer))
            .filter(|&ring| ring > 0)
            .map(|ring| ring.saturating_sub(window).max(1))
            .fold(LFM3_PREFILL_ROWS, usize::min)
    }

    fn prefill_micro_batch_lfm3(
        &mut self,
        tokens: &[i32],
        start_position: usize,
        want_head: bool,
        logits: &mut [LogitValue],
    ) -> Result<(), RealForwardError> {
        let arch = self.arch.clone();
        let hidden = arch.hidden_size as usize;
        let inter = arch.intermediate_size as usize;
        let vocab = arch.vocab_size as usize;
        let gpu_err = RealForwardError::Gpu;
        let m = tokens.len();
        let steel = self.prefill_steel;
        let rows_attention = self.prefill_rows_attention;

        if start_position != self.kv.position() {
            return Err(RealForwardError::Unsupported(format!(
                "non-sequential chunk start {start_position}; KV cache is at {}",
                self.kv.position()
            )));
        }
        if let Some(&bad) = tokens.iter().find(|&&t| (t as usize) >= vocab) {
            return Err(RealForwardError::Unsupported(format!(
                "token id {bad} outside vocab {vocab}"
            )));
        }

        let embed_name = "language_model.model.embed_tokens.weight";
        let pass = self.context.begin_pass_labeled("lfm3 chunk cb");
        for (t, &token) in tokens.iter().enumerate() {
            encode_embed_any(
                &mut self.context,
                &pass,
                &self.weights,
                &self.index,
                embed_name,
                (&self.scratch.x, (t * hidden * 2) as u64),
                token as u32,
                hidden as u32,
                vocab,
                1.0,
            )?;
        }

        let (context, weights, index, scratch, kv, lfm3) = (
            &mut self.context,
            &self.weights,
            &self.index,
            &self.scratch,
            &self.kv,
            self.real_lfm3.as_ref().expect("real lfm3 state present"),
        );
        let b = lfm3
            .batched
            .as_ref()
            .expect("allocated by the chunk driver");
        // An M-row norm is the per-head kernel with M "heads" of width
        // `hidden`: the whole-vector kernel's helper, loop and launch width.
        let norm_rows = |context: &mut gpu::MetalContext,
                         x: &gpu::MetalBuffer,
                         name: &str,
                         out: &gpu::MetalBuffer|
         -> Result<(), RealForwardError> {
            let w = norm_view(weights, index, name, hidden)?;
            gpu::encode_rms_norm_bf16w_perhead(
                context,
                &pass,
                (x, 0),
                w,
                (out, 0),
                m as u32,
                hidden as u32,
                RMS_EPS,
            )
            .map_err(gpu_err)
        };
        let gemm = |context: &mut gpu::MetalContext,
                    name: &str,
                    rows: usize,
                    cols: usize,
                    x: &gpu::MetalBuffer,
                    y: &gpu::MetalBuffer|
         -> Result<(), RealForwardError> {
            encode_gemm_rows(
                context,
                &pass,
                weights,
                index,
                name,
                rows,
                cols,
                (x, 0),
                (y, 0),
                m,
                steel,
            )
        };

        for layer in 0..arch.num_layers as usize {
            norm_rows(
                context,
                &scratch.x,
                &layer_tensor(layer, "operator_norm.weight"),
                &b.normed,
            )?;
            if arch.layer_is_short_conv(layer) {
                let name = |suffix: &str| layer_tensor(layer, &format!("conv.{suffix}"));
                gemm(
                    context,
                    &name("in_proj.weight"),
                    3 * hidden,
                    hidden,
                    &b.normed,
                    &b.in_proj,
                )?;
                let taps = norm_view(
                    weights,
                    index,
                    &name("conv.weight"),
                    hidden * lfm3.taps as usize,
                )?;
                gpu::encode_short_conv_prefill(
                    context,
                    &pass,
                    (lfm3.conv.tail_buffer(layer), 0),
                    (&b.in_proj, 0),
                    taps,
                    (&b.conv_out, 0),
                    hidden as u32,
                    lfm3.taps,
                    m as u32,
                )
                .map_err(gpu_err)?;
                gemm(
                    context,
                    &name("out_proj.weight"),
                    hidden,
                    hidden,
                    &b.conv_out,
                    &b.o,
                )?;
            } else {
                encode_attention_rows(
                    context,
                    &pass,
                    weights,
                    index,
                    &arch,
                    lfm3,
                    b,
                    scratch,
                    kv,
                    layer,
                    start_position,
                    m,
                    steel,
                    rows_attention,
                )?;
            }
            norm_rows(
                context,
                &b.o,
                &layer_tensor(layer, "post_operator_norm.weight"),
                &b.normed,
            )?;
            gpu::encode_residual_add(
                context,
                &pass,
                (&scratch.x, 0),
                (&b.normed, 0),
                (m * hidden) as u32,
            )
            .map_err(gpu_err)?;

            norm_rows(
                context,
                &scratch.x,
                &layer_tensor(layer, "ffn_norm.weight"),
                &b.ffn_normed,
            )?;
            gemm(
                context,
                &layer_tensor(layer, "feed_forward.w1.weight"),
                inter,
                hidden,
                &b.ffn_normed,
                &b.ffn_gate,
            )?;
            gemm(
                context,
                &layer_tensor(layer, "feed_forward.w3.weight"),
                inter,
                hidden,
                &b.ffn_normed,
                &b.ffn_up,
            )?;
            gpu::encode_silu_mul(
                context,
                &pass,
                (&b.ffn_gate, 0),
                (&b.ffn_up, 0),
                (&b.ffn_act, 0),
                (m * inter) as u32,
            )
            .map_err(gpu_err)?;
            gemm(
                context,
                &layer_tensor(layer, "feed_forward.w2.weight"),
                hidden,
                inter,
                &b.ffn_act,
                &b.ffn_out,
            )?;
            norm_rows(
                context,
                &b.ffn_out,
                &layer_tensor(layer, "post_ffn_norm.weight"),
                &b.ffn_normed,
            )?;
            gpu::encode_residual_add(
                context,
                &pass,
                (&scratch.x, 0),
                (&b.ffn_normed, 0),
                (m * hidden) as u32,
            )
            .map_err(gpu_err)?;
        }

        if want_head {
            pass.relabel("lfm3 chunk cb (head)");
            let last_off = ((m - 1) * hidden * 2) as u64;
            let final_norm = norm_view(
                weights,
                index,
                "language_model.model.embedding_norm.weight",
                hidden,
            )?;
            gpu::encode_rms_norm_bf16w(
                context,
                &pass,
                (&scratch.x, last_off),
                final_norm,
                (&scratch.normed, 0),
                hidden as u32,
                RMS_EPS,
            )
            .map_err(gpu_err)?;
            encode_gemv_any(
                context,
                &pass,
                weights,
                index,
                embed_name,
                vocab,
                hidden,
                (&scratch.normed, 0),
                (&scratch.logits, 0),
            )?;
        } else {
            pass.relabel("lfm3 chunk cb (no head)");
        }
        let t_wait = Instant::now();
        self.phases.final_cb_gpu_nanos += (pass.commit_and_wait_with_gpu_time() * 1e9) as u64;
        self.phases.final_wait_nanos += t_wait.elapsed().as_nanos() as u64;
        self.kv.advance_by(m);

        if !want_head {
            return Ok(());
        }
        // Raw logits; the sampler normalizes (AGENTS.md Gotcha 16).
        if vocab != logits.len() {
            return Err(RealForwardError::Unsupported(format!(
                "vocab mismatch: model has {}, caller expected {}",
                vocab,
                logits.len()
            )));
        }
        gpu::read_buffer_f16_into(&self.scratch.logits, 0, logits);
        Ok(())
    }
}

/// One attention layer over `m` rows: `b.normed` in, `b.o` out (pre
/// `post_operator_norm`). The same steps as `attn::encode_attention_block`,
/// each over all rows at once except the sliding layers' attention.
#[allow(clippy::too_many_arguments)]
fn encode_attention_rows(
    context: &mut gpu::MetalContext,
    pass: &gpu::PassEncoder,
    weights: &gpu::ResidentGpuWeights,
    index: &model_io::ResidentIndex,
    arch: &model_io::ArchConfig,
    lfm3: &super::RealLfm3State,
    b: &Lfm3Batched,
    scratch: &crate::real_forward_types::DecodeScratch,
    kv: &gpu::KvCacheManager,
    layer: usize,
    start_position: usize,
    m: usize,
    steel: bool,
    rows_attention: bool,
) -> Result<(), RealForwardError> {
    let gpu_err = RealForwardError::Gpu;
    let hidden = arch.hidden_size as usize;
    let num_heads = arch.num_heads as u32;
    let num_kv = arch.num_kv_heads as u32;
    let head_dim = arch.full_head_dim as u32;
    let q_dim = (num_heads * head_dim) as usize;
    let kv_dim = (num_kv * head_dim) as usize;
    let name = |suffix: &str| layer_tensor(layer, &format!("self_attn.{suffix}"));

    for (suffix, rows, out) in [
        ("q_proj.weight", q_dim, &b.q),
        ("k_proj.weight", kv_dim, &b.k_stage),
        ("v_proj.weight", kv_dim, &b.v_stage),
        ("gate_proj.weight", num_heads as usize, &b.attn_gate),
    ] {
        encode_gemm_rows(
            context,
            pass,
            weights,
            index,
            &name(suffix),
            rows,
            hidden,
            (&b.normed, 0),
            (out, 0),
            m,
            steel,
        )?;
    }

    // Per-head RMSNorm on q and k over every row: M rows of H heads are M*H
    // heads of the same width.
    for (data, heads, suffix) in [
        (&b.q, num_heads, "q_norm.weight"),
        (&b.k_stage, num_kv, "k_norm.weight"),
    ] {
        let w = norm_view(weights, index, &name(suffix), head_dim as usize)?;
        gpu::encode_rms_norm_bf16w_perhead(
            context,
            pass,
            (data, 0),
            w,
            (data, 0),
            heads * m as u32,
            head_dim,
            RMS_EPS,
        )
        .map_err(gpu_err)?;
    }

    let is_full = arch.layer_is_full(layer);
    let (pairs, theta) = rope_params_for(
        is_full,
        lfm3.full_rotated_pairs,
        head_dim,
        arch.full_rope_theta as f32,
        arch.rope_theta as f32,
    );
    for (data, heads) in [(&b.q, num_heads), (&b.k_stage, num_kv)] {
        gpu::encode_rope_proportional_neox_rows(
            context,
            pass,
            (data, 0),
            start_position as u32,
            m as u32,
            heads,
            head_dim,
            pairs,
            theta,
        )
        .map_err(gpu_err)?;
    }

    commit_kv_rows(context, pass, kv, b, layer, start_position, m, kv_dim)?;

    let scale = arch.attention_scale as f32;
    let rows_kernel = b
        .rows_attention
        .as_ref()
        .filter(|_| is_full && rows_attention && m > 1);
    if let Some(rows) = rows_kernel {
        foundation::engaged!("prefill_rows_attn");
        let (k_buf, _) = kv.k_slot(layer, start_position);
        let (v_buf, _) = kv.v_slot(layer, start_position);
        gpu::encode_attention_decode_gqa_rows(
            context,
            pass,
            (&b.q, 0),
            k_buf,
            v_buf,
            rows,
            (&b.attn_out, 0),
            head_dim,
            num_heads,
            num_kv,
            start_position as u32,
            m as u32,
            0,
            0,
            scale,
        )
        .map_err(gpu_err)?;
    } else {
        // The sequential flow's one-row attention per row, with its window
        // start and ring arguments computed the same way (`attn.rs`).
        for r in 0..m {
            let position = start_position + r;
            let seq_len = (position + 1) as u32;
            let (kv_start, active_ring) = if is_full {
                (0, 0)
            } else {
                let ring = kv.ring_capacity(layer) as u32;
                (
                    seq_len.saturating_sub(arch.sliding_window as u32),
                    if ring > 0 && seq_len > ring { ring } else { 0 },
                )
            };
            let off = (r * q_dim * 2) as u64;
            encode_attention_any(
                context,
                pass,
                (&b.q, off),
                kv,
                scratch,
                (&b.attn_out, off),
                layer,
                position,
                head_dim,
                num_heads,
                num_kv,
                seq_len,
                kv_start,
                active_ring,
                scale,
                None,
            )?;
        }
    }

    // One sigmoid per (row, head): the kernel reads `gate[i / head_dim]`,
    // and `[M, heads]` gate rows line up with `[M, heads * head_dim]`.
    gpu::encode_sigmoid_head_gate_mul(
        context,
        pass,
        (&b.attn_out, 0),
        (&b.attn_gate, 0),
        head_dim,
        (m * q_dim) as u32,
    )
    .map_err(gpu_err)?;

    encode_gemm_rows(
        context,
        pass,
        weights,
        index,
        &name("o_proj.weight"),
        hidden,
        q_dim,
        (&b.attn_out, 0),
        (&b.o, 0),
        m,
        steel,
    )
}

/// Copies the staged K and V rows into their cache slots, one copy per run
/// of physically contiguous slots: one run on a full layer, at most two
/// where a sliding layer's ring wraps.
#[allow(clippy::too_many_arguments)]
fn commit_kv_rows(
    context: &mut gpu::MetalContext,
    pass: &gpu::PassEncoder,
    kv: &gpu::KvCacheManager,
    b: &Lfm3Batched,
    layer: usize,
    start_position: usize,
    m: usize,
    kv_dim: usize,
) -> Result<(), RealForwardError> {
    let stride = kv.k_stride(layer);
    if stride != kv_dim * 2 {
        return Err(RealForwardError::Unsupported(format!(
            "layer {layer}: KV stride {stride} is not {kv_dim} halfs"
        )));
    }
    let mut run_start = 0usize;
    while run_start < m {
        let (_, first) = kv.k_slot(layer, start_position + run_start);
        let mut run_len = 1usize;
        while run_start + run_len < m
            && kv.k_slot(layer, start_position + run_start + run_len).1 == first + run_len * stride
        {
            run_len += 1;
        }
        let src_off = (run_start * kv_dim * 2) as u64;
        for (stage, slot) in [
            (&b.k_stage, kv.k_slot(layer, start_position + run_start)),
            (&b.v_stage, kv.v_slot(layer, start_position + run_start)),
        ] {
            gpu::encode_copy_rows_fp16_strided(
                context,
                pass,
                (stage, src_off),
                (slot.0, slot.1 as u64),
                run_len as u32,
                kv_dim as u32,
                kv_dim as u32,
                kv_dim as u32,
            )
            .map_err(RealForwardError::Gpu)?;
        }
        run_start += run_len;
    }
    Ok(())
}
