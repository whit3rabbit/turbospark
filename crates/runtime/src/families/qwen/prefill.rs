//! Chunked prefill for the DENSE half of the qwen linear-attention flow
//! (`qwenGdnDense`, `qwen38-27b.gturbo`): the sixth
//! [`crate::producer::ChunkedPrefillRunner`] implementation, and the same
//! "step 1" shape as `families/llama/prefill.rs` and
//! `families/museglimmer/prefill.rs` -- loop the EXISTING per-token kernels
//! inside a micro-batch, batching command buffers rather than GEMVs.
//!
//! **THE DEFAULT ARM IS A DIFFERENT DESIGN THAN `families/qwen/batched.rs`.**
//! That file implements `docs/BATCHED_PREFILL.md` steps 2-6 (GEMVs become
//! GEMMs) for the MTP/DFlash2 verify pass, sized for tiny block depths and
//! allocated only when a drafter is open. The default arm needs neither: it
//! calls `attn::encode_linear_block` / `attn::encode_full_attention_block`
//! and `dense::encode_qwen_layer_dense` exactly as the sequential flow
//! (`produce.rs`) does, once per token, inside one command buffer per
//! micro-batch instead of one per token. **No new kernel, no new buffer.**
//!
//! **`MFERENCE_BATCHED_GEMV` IS THE SECOND ARM, AND IT IS THAT FILE'S
//! MACHINERY REUSED RATHER THAN A SECOND COPY OF IT** (step 6, wired
//! 2026-08-29). `batched_layers.rs`'s three encoders already take the
//! parameters this driver has, already read and write `scratch.x` at the
//! `t * hidden * 2` row convention this driver uses, and already carry the
//! GDN recurrence's multi-row kernels -- so the arm is a branch in the layer
//! loop plus a lazily-allocated `BatchedScratch`, with NO new dispatch code
//! and no new kernel. The one constant that makes it fit exactly is that
//! `MAX_PREFILL_BATCH` IS `gpu::MAX_BATCH_ROWS` (both 16), so a full
//! micro-batch is one GEMM dispatch and never needs sub-batching.
//!
//! **THE TWO ARMS ARE NOT BYTE-IDENTICAL ON A REAL INSTALL, AND THAT IS A
//! FLOOR RATHER THAN A DEFECT.** `produce_batched` and `produce` differ by
//! 6.2e-8 to 1.5e-5 nats with the argmax agreeing on every row, against a
//! measured dense batched-vs-cached shape floor of 7.4e-6 for this same
//! architecture on MLX (`crates/bench/tests/batched_forward_probe.rs`,
//! commit `e8deb6c`) -- every engine's batched and cached passes disagree by
//! about this much. So the batched arm's gate is the QUALITY gate, not the
//! md5 identity every other chunked driver was verified with; what must stay
//! byte-identical is the DEFAULT arm, which is what the caller who sets no
//! env var gets. The synthetic fixture cannot see the floor at all (it reads
//! 0 differing logits at every span,
//! `tests/real_forward_qwen35_batched_onset.rs`), so a fixture-level
//! byte-identity case pins the WIRING -- rows, offsets, ordering -- and must
//! not be read as evidence about the arithmetic.
//!
//! Three properties make the default arm safe.
//!
//! `attn.rs`'s two block encoders read `scratch.normed` / write `scratch.o`
//! at offset 0 and never touch `scratch.x` directly, so they need no change
//! (the same property that let `llama`'s and `muse_glimmer`'s attention
//! blocks land unmodified, `crates/runtime/CLAUDE.md` Gotcha 14).
//!
//! The GDN recurrent state (`qwen.gdn.state_buffer(layer)` /
//! `conv_tail_buffer(layer)`) is a persistent per-layer buffer that
//! `encode_linear_block`'s decode-shaped kernels advance in place, with no
//! position argument -- it does not know whether it is being called from a
//! chunk or from sequential decode, only that it is called once per token.
//! Calling it once per token, strictly in increasing `t` order, within one
//! layer's inner loop, before moving to the next layer, reproduces
//! sequential decode's math exactly. That ordering is what
//! `crates/runtime/CLAUDE.md` Gotcha 4 requires ("`reset()` must rewind the
//! GDN state, not just the KV cache") -- satisfied here by construction
//! rather than by a new mechanism, and it is also what makes cross-chunk
//! continuity free: the state buffer is the same one sequential decode
//! reads and writes, so a prompt spanning several micro-batches (or several
//! `prefill_chunk` calls) carries it forward automatically.
//!
//! `qwen.moe_x`, `qwen.h2`, and the GDN scratch fields on `RealQwenState`
//! are single-row GPU-only intermediates, safe to reuse per token within
//! one command buffer because a serial compute encoder runs dispatches in
//! commit order (`crates/gpu/CLAUDE.md` Gotcha 8) -- exactly the reasoning
//! `families/llama/prefill.rs` already established for `llama.moe_x`/`h2`.
//!
//! KV writes stay per-token via the existing `k_slot`/`v_slot` calls inside
//! `encode_full_attention_block`, so there is no batched-projection KV-wrap
//! hazard to guard against (that hazard belongs to the M-row GEMM path in
//! `batched_layers.rs`, which this driver does not use).
//!
//! Two refusals are BY NAME rather than silent, per this repo's convention:
//!
//! - **Vision.** The image injection in `produce.rs` is this family's only
//!   embedding call site today (Gotcha 27). This first cut stays text-only;
//!   `RealForwardRunner::supports_chunked_prefill` also excludes an install
//!   with a live `prompt_vision` map so callers route around this driver
//!   entirely rather than reaching the refusal.
//! - **An open drafter.** `produce.rs`'s dense branch fires the DFlash2
//!   aux-capture hook on every forward pass, which `dflash_prime_from_capture`
//!   later reads; this driver does not encode that hook, so it refuses
//!   rather than silently prefilling a prompt whose aux cache the drafter
//!   would read back empty.

use std::time::Instant;

use foundation::LogitValue;

use super::layer_tensor;
use super::prefill_layers::{
    encode_qwen_dense_layer_batched_prefill, encode_qwen_dense_layer_per_token_prefill,
};
use crate::real_forward::RealForwardRunner;
use crate::real_forward_dispatch::encode_embed_any;
use crate::real_forward_types::{RealForwardError, MAX_PREFILL_BATCH};
use crate::real_forward_utils::norm_view;

impl RealForwardRunner {
    /// Runs a whole prefill chunk through the dense qwen flow, writing the
    /// logits for the position after its last token. Call only once
    /// `RealQwenState::dense` is known true; a MoE install is refused at
    /// [`crate::producer::ChunkedPrefillRunner::prefill_chunk`], by name,
    /// before this is reached.
    pub(crate) fn prefill_chunk_real_qwen_dense(
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
        if self.prompt_vision.is_some() {
            return Err(RealForwardError::Unsupported(
                "the qwen chunked prefill driver is text-only: this install has an image \
                 prompt attached, and the image injection's only embedding call site is the \
                 sequential flow"
                    .to_string(),
            ));
        }
        if self.real_mtp.is_some() || self.real_dflash.is_some() {
            return Err(RealForwardError::Unsupported(
                "the qwen chunked prefill driver does not encode the drafter's aux capture; \
                 open without MFERENCE_MTP_DRAFT / MFERENCE_DFLASH_DRAFT to use it, or prefill \
                 sequentially"
                    .to_string(),
            ));
        }
        if self.batched_gemv_prefill {
            // INT4-AFFINE ONLY, and refused UP FRONT rather than at the first
            // dispatch. `encode_gemm_any` is the per-tensor backstop and
            // refuses by name on its own, but it would do so mid-encode, so a
            // caller who set the seam on the 1-bit or 2-bit checkpoint of
            // THIS SAME ARCHITECTURE (which have no batched kernel at all)
            // learns it before any KV row is written. One probe cannot see a
            // mixed-width install; that is what the backstop is for.
            let probe = layer_tensor(0, "mlp.gate_proj.weight");
            let dtype = crate::real_forward_utils::entry(&self.index, &probe)?.dtype;
            if dtype != 4 {
                return Err(RealForwardError::Unsupported(format!(
                    "MFERENCE_BATCHED_GEMV needs INT4-affine (dtype 4) weights and {probe} is \
                     dtype {dtype}: the M-row GEMM has no kernel at this width, and looping the \
                     per-token GEMVs anyway would measure the unbatched engine under the \
                     batched arm's label"
                )));
            }
            // The ONE place this arm's M-row scratch is allocated, and it is
            // here rather than at open so a run that never sets the seam
            // never pays for it. Idempotent, and ahead of the loop so every
            // `batched_prefill()` below it is infallible.
            let arch = self.arch.clone();
            let context = &mut self.context;
            self.real_qwen
                .as_mut()
                .ok_or_else(|| RealForwardError::Unsupported("not a Qwen install".to_string()))?
                .ensure_batched_prefill(context, &arch)?;
        }
        let mut offset = 0usize;
        while offset < tokens.len() {
            let take = (tokens.len() - offset).min(MAX_PREFILL_BATCH);
            let last = offset + take == tokens.len();
            self.prefill_micro_batch_qwen_dense(
                &tokens[offset..offset + take],
                start_position + offset,
                last,
                logits,
            )?;
            offset += take;
        }
        Ok(())
    }

    fn prefill_micro_batch_qwen_dense(
        &mut self,
        tokens: &[i32],
        start_position: usize,
        want_head: bool,
        logits: &mut [LogitValue],
    ) -> Result<(), RealForwardError> {
        let started = Instant::now();
        let result =
            self.prefill_micro_batch_qwen_dense_inner(tokens, start_position, want_head, logits);
        // One forward pass per TOKEN, matching every other phase divisor in
        // this port (`crates/runtime/CLAUDE.md` Gotcha 6).
        self.phases.calls += tokens.len() as u64;
        self.phases.total_nanos += started.elapsed().as_nanos() as u64;
        result
    }

    fn prefill_micro_batch_qwen_dense_inner(
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
        let use_silu = arch.hidden_activation.contains("silu");
        let m = tokens.len();

        if !self.real_qwen.as_ref().is_some_and(|s| s.dense) {
            return Err(RealForwardError::Unsupported(
                "prefill_chunk_real_qwen_dense called on a non-dense (MoE) qwen install"
                    .to_string(),
            ));
        }
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

        let batched_gemv = self.batched_gemv_prefill;
        // A BACKSTOP THAT BELONGS TO THE BATCHED ARM ALONE, and only that arm
        // can trip it. `encode_full_attention_block_batched`'s k/v projections
        // write M ADJACENT KV slots in one dispatch, so M consecutive
        // positions have to be M consecutive rows -- true only while
        // `position % capacity` does not roll over inside the micro-batch.
        // This family has no sliding window, so its full layers address the
        // cache linearly, but "linear" still wraps at `max_context`, and a
        // projection straddling that boundary scatters into row 0 with no
        // symptom beyond wrong attention. `produce_batched` carries the
        // identical check for the identical reason.
        //
        // It should be unreachable during prefill -- a prompt longer than
        // `max_context` is refused upstream, so `start_position + m` never
        // exceeds capacity and the modulo is the identity -- but "should be
        // unreachable" is what this repo asks be asserted rather than
        // assumed, and the check is one comparison per layer.
        if batched_gemv {
            for layer in 0..arch.num_layers as usize {
                if arch.layer_is_linear(layer) {
                    continue;
                }
                let capacity = self.kv.capacity(layer);
                if start_position % capacity + m > capacity {
                    return Err(RealForwardError::Unsupported(format!(
                        "MFERENCE_BATCHED_GEMV: a micro-batch of {m} rows at position \
                         {start_position} wraps layer {layer}'s KV capacity {capacity}; the \
                         batched projection writes M adjacent slots and cannot straddle the \
                         boundary"
                    )));
                }
            }
        }

        let embed_name = "language_model.model.embed_tokens.weight";

        let pass = self.context.begin_pass_labeled("qwen dense chunk cb");
        // No `sqrt(hidden)` embedding scale, matching the sequential flow:
        // `RealQwenState::build` refuses an install that declares
        // `embeddingScaledBySqrtHidden`.
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
                1.0,
            )?;
        }

        let (context, weights, index, scratch, kv, qwen, resid_capture, steering) = (
            &mut self.context,
            &self.weights,
            &self.index,
            &self.scratch,
            &mut self.kv,
            self.real_qwen.as_ref().expect("checked dense above"),
            self.resid_capture.as_ref(),
            self.steering.as_ref(),
        );

        for layer in 0..arch.num_layers as usize {
            let input_norm = norm_view(
                weights,
                index,
                &layer_tensor(layer, "input_layernorm.weight"),
                hidden,
            )?;
            let post_attn_norm = norm_view(
                weights,
                index,
                &layer_tensor(layer, "post_attention_layernorm.weight"),
                hidden,
            )?;

            if batched_gemv {
                encode_qwen_dense_layer_batched_prefill(
                    context,
                    &pass,
                    weights,
                    index,
                    scratch,
                    kv,
                    qwen,
                    &arch,
                    resid_capture,
                    steering,
                    input_norm,
                    post_attn_norm,
                    layer,
                    hidden,
                    inter,
                    use_silu,
                    start_position,
                    m,
                )?;
            } else {
                encode_qwen_dense_layer_per_token_prefill(
                    context,
                    &pass,
                    weights,
                    index,
                    scratch,
                    kv,
                    qwen,
                    &arch,
                    resid_capture,
                    steering,
                    input_norm,
                    post_attn_norm,
                    layer,
                    hidden,
                    inter,
                    use_silu,
                    start_position,
                    m,
                )?;
            }
        }

        if want_head {
            super::prefill_layers::encode_qwen_dense_chunk_head(
                context, &pass, weights, index, scratch, &arch, embed_name, hidden, vocab, m,
            )?;
            // No softcap: `RealQwenState::build` refuses an install that
            // declares one, matching the sequential flow.
        } else {
            pass.relabel("qwen dense chunk cb (no head)");
        }
        self.phases.final_cb_gpu_nanos += (pass.commit_and_wait_with_gpu_time() * 1e9) as u64;
        self.kv.advance_by(m);

        let skip_head = !want_head;
        let last_position = start_position + m - 1;
        if let Some(capture) = self.resid_capture.as_mut() {
            capture.record_pass(last_position, skip_head);
        }

        if !want_head {
            return Ok(());
        }
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
