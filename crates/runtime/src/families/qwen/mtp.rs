//! The multi-token-prediction head's draft step (`docs/MTP_SPECULATIVE.md`,
//! step 2), for the dense `qwen3_5` family.
//!
//! **No new Metal kernel and no new dispatch shape.** The head is one FULL
//! attention block whose every tensor is a trunk full-attention layer's shape
//! field for field (`crates/repack/tests/mtp_head_network.rs` establishes
//! that off the real header, and it is the finding the whole cost estimate
//! forks on), so a draft step is `attn.rs` and `dense.rs` called under
//! `mtp.layers.0.*` names. What the head adds beyond the block is `fc`, a
//! plain GEMV over a `[2 * hidden]` buffer, and three RMS norms.
//!
//! ```text
//! x   = fc([ norm_e(embed(next)), norm_h(h_t) ])   // 2H -> H
//! x   = x + attn(input_layernorm(x))               // the head's OWN KV
//! x   = x + ffn(post_attention_layernorm(x))
//! out = lm_head(mtp.norm(x))                       // the TRUNK's lm_head
//! ```
//!
//! **The head has neither an embedding table nor an output head**; both are
//! shared with the trunk, which is what makes it 849 MB rather than several
//! GB and ~1.5% of a forward pass once quantized to INT4.
//!
//! # Why this costs two buffers rather than a parallel scratch set
//!
//! Reusing `scratch.x`, `scratch.normed`, `scratch.o`, `ffn_*`, `qwen.h2`
//! and `qwen.moe_x` is safe because **`scratch.x` is re-initialised from the
//! embedding at the top of every token**, so clobbering it after the trunk's
//! logits have been read cannot affect anything downstream. The draft step
//! therefore runs strictly between the trunk's readback and the next token's
//! embedding, and the only state it needs of its own is its KV and the
//! concatenation buffer `fc` reads.
//!
//! # Why its KV is its own one-layer cache
//!
//! Widening the trunk's `KvCacheManager` is not an option:
//! its sizing is driven by `ArchConfig` and every family's memory-oracle peak
//! is frozen against it (`crates/runtime/CLAUDE.md` Gotcha 15). `MtpState`
//! builds a SEPARATE manager from a cloned `ArchConfig` at `num_layers: 1`
//! and `full_attention_layer_mask: vec![1]`, which reuses that whole
//! constructor and allocates exactly one full-attention layer -- about 4 KiB
//! per token on this architecture (4 KV heads x 256 x 2 bytes, K and V), so
//! 16 MiB at a 4,096 context.
//!
//! It is allocated ONLY when a draft depth is asked for. With
//! `MFERENCE_MTP_DRAFT` unset the flow allocates nothing and encodes nothing,
//! so the unset path is identical to the one that shipped before this module
//! existed in bytes AND in footprint -- which is what lets the frozen oracle
//! row stand rather than needing a new one.

use foundation::LogitValue;

pub(crate) use super::mtp_state::MtpState;
use super::mtp_state::{
    dump_dir, FC, FINAL_NORM, PRE_FC_NORM_EMBEDDING, PRE_FC_NORM_HIDDEN, TRUNK_FINAL_NORM,
};
use crate::families::qwen::{
    dense, encode_full_attention_block, prefixed_layer_tensor, QkNormConvention, MTP_PREFIX,
    RMS_EPS,
};
use crate::real_forward::{RealForwardError, RealForwardRunner};
use crate::real_forward_dispatch::{encode_embed_any, encode_gemv_any};
use crate::real_forward_utils::norm_view;

impl RealForwardRunner {
    /// Encodes ONE draft step and returns its logits.
    ///
    /// **`position` is where the HIDDEN STATE came from, not where
    /// `next_token` sits.** The head's input pair is `(h_i, emb(t_{i+1}))`
    /// and it predicts `t_{i+2}`, so a call at `position = i` takes the token
    /// occupying `i + 1` and drafts the one occupying `i + 2`. That is the
    /// published MTP shift (a module reads the trunk's hidden at `i` beside
    /// the token that FOLLOWS it), and it is what makes the head's rows land
    /// contiguously at 0, 1, 2, ... with no unwritten row: `h_0` exists, so
    /// row 0 does too.
    ///
    /// `h_t` is read from `scratch.x`, which still holds the trunk's final
    /// hidden state for `position` -- **so this must be called after the
    /// trunk's logits are read and before the next `produce`**, which is the
    /// window the module header's buffer argument is about.
    ///
    /// Depth beyond one is the CALLER's loop: it feeds the drafted token back
    /// as `next_token` at `position + 1`, where `scratch.x` now holds the
    /// HEAD's own residual stream standing in for `h_{i+1}`. That chaining
    /// approximation is inherent to drafting more than one token from a
    /// single module, and it is one of the two things the accept length
    /// measures. Keeping the loop at the call site is also what lets the
    /// probe stop early on a rejection.
    pub fn mtp_draft_step(
        &mut self,
        next_token: i32,
        position: usize,
        logits: &mut [LogitValue],
    ) -> Result<(), RealForwardError> {
        self.mtp_step(next_token, position, Some(logits))
    }

    /// A step taken for its KV ROW alone, skipping the full-vocab head.
    ///
    /// This is how the head is primed over a prompt. Without it a draft at
    /// decode position `P` attends over `P` rows the head never wrote, which
    /// is not an error and does not look like one -- it just quietly costs
    /// accept length, which is the number the whole exercise is trying to
    /// measure. The trunk's own `produce_prefill` skips the head for the same
    /// reason and the saving is the same one: a full-vocab GEMV per prompt
    /// token, and nobody reads the result.
    pub fn mtp_prime_step(
        &mut self,
        next_token: i32,
        position: usize,
    ) -> Result<(), RealForwardError> {
        self.mtp_step(next_token, position, None)
    }

    /// Drops head rows at `[position, cursor)`, so the head can follow the
    /// trunk back after a rejected draft.
    ///
    /// The trunk's [`Self::rollback`] deliberately does NOT reach this. A
    /// speculative round rewinds the trunk to where the block STARTED and
    /// then replays the accepted prefix, but the head has already written
    /// correct rows for that prefix and cannot recompute them (the trunk's
    /// `produce` has overwritten the `scratch.x` each one needs). So the head
    /// rewinds to the accepted end rather than to the checkpoint, which is a
    /// different target, and only the caller knows it.
    pub fn mtp_rewind_to(&mut self, position: usize) -> Result<(), RealForwardError> {
        let mtp = self.real_mtp.as_mut().ok_or_else(|| {
            RealForwardError::Unsupported("no MTP head state to rewind".to_string())
        })?;
        let cursor = mtp.kv.position();
        if position > cursor {
            return Err(RealForwardError::Unsupported(format!(
                "MTP rewind target {position} is ahead of the head's cursor {cursor}"
            )));
        }
        mtp.kv.rewind_by(cursor - position);
        Ok(())
    }

    /// The head's KV cursor; see [`MtpState::kv_position`].
    pub fn mtp_kv_position(&self) -> usize {
        self.real_mtp.as_ref().map_or(0, |m| m.kv_position())
    }

    fn mtp_step(
        &mut self,
        next_token: i32,
        position: usize,
        logits: Option<&mut [LogitValue]>,
    ) -> Result<(), RealForwardError> {
        let arch = self.arch.clone();
        let hidden = arch.hidden_size as usize;
        let inter = arch.intermediate_size as usize;
        let vocab = arch.vocab_size as usize;
        let use_silu = arch.hidden_activation.contains("silu");
        let gpu_err = RealForwardError::Gpu;

        if self.real_mtp.is_none() {
            return Err(RealForwardError::Unsupported(
                "no MTP head state; set MFERENCE_MTP_DRAFT=<depth> before opening the model"
                    .to_string(),
            ));
        }
        if (next_token as usize) >= vocab {
            return Err(RealForwardError::Unsupported(format!(
                "draft token id {next_token} outside vocab {vocab}"
            )));
        }
        if let Some(out) = logits.as_ref() {
            if vocab != out.len() {
                return Err(RealForwardError::Unsupported(format!(
                    "vocab mismatch: model has {vocab}, caller expected {}",
                    out.len()
                )));
            }
        }
        // The head's KV covers [0, cursor) and this block will attend over
        // [0, position]. Off by either sign the step still runs, still
        // returns finite logits and still looks exactly like a working
        // drafter; see `MtpState::kv_position`.
        let cursor = self.real_mtp.as_ref().expect("checked above").kv_position();
        if position != cursor {
            return Err(RealForwardError::Unsupported(format!(
                "MTP step at position {position} but the head's KV covers [0, {cursor}): \
                 prime the head over the prompt with mtp_prime_step, and rewind it with \
                 mtp_rewind_to after a rejected draft (docs/MTP_SPECULATIVE.md)"
            )));
        }

        let embed_name = "language_model.model.embed_tokens.weight";
        let head_name = if arch.tie_word_embeddings {
            embed_name.to_string()
        } else {
            "language_model.lm_head.weight".to_string()
        };

        let pass = self.context.begin_pass_labeled("mtp draft");
        let (context, weights, index, scratch, qwen, mtp) = (
            &mut self.context,
            &self.weights,
            &self.index,
            &self.scratch,
            self.real_qwen
                .as_ref()
                .ok_or_else(|| RealForwardError::Unsupported("not a Qwen install".to_string()))?,
            self.real_mtp.as_ref().expect("checked above"),
        );

        // 1. `fc`'s input: the next token's embedding and the trunk's hidden
        //    state, each through its OWN norm, concatenated. The embedding
        //    lands in `scratch.normed` first because a norm reduces before it
        //    writes and reading and writing one buffer in one dispatch is a
        //    property of the kernel rather than of this call site.
        // THE HIDDEN HALF IS THE TRUNK'S POST-FINAL-NORM STATE, NOT ITS
        // RESIDUAL STREAM. `scratch.x` carries the residual; the head wants
        // what the trunk's OWN lm_head consumes, i.e. `model.norm` applied
        // first. The reference is explicit about this -- mlx-vlm's
        // `Qwen3_5Model.__call__` returns `self.norm(h)` and that one value
        // is both what `hidden_states[-1]` hands the drafter and what
        // `lm_head` reads. Feeding the residual instead is not a scale
        // error that `pre_fc_norm_hidden` would absorb, because
        // `model.norm` carries a LEARNED per-channel weight: it is a
        // different direction, which is why it produced finite,
        // plausible-looking logits that were ANTI-aligned with the trunk
        // (docs/MTP_SPECULATIVE.md step 3).
        //
        // It goes FIRST because it borrows `scratch.normed` as its
        // temporary and the embedding below overwrites that buffer.
        let trunk_norm = norm_view(weights, index, TRUNK_FINAL_NORM, hidden)?;
        gpu::encode_rms_norm_bf16w(
            context,
            &pass,
            (&scratch.x, 0),
            trunk_norm,
            (&scratch.normed, 0),
            hidden as u32,
            RMS_EPS,
        )
        .map_err(gpu_err)?;
        let w = norm_view(weights, index, PRE_FC_NORM_HIDDEN, hidden)?;
        gpu::encode_rms_norm_bf16w_centered(
            context,
            &pass,
            (&scratch.normed, 0),
            w,
            (&mtp.concat, hidden as u64 * 2),
            hidden as u32,
            RMS_EPS,
        )
        .map_err(gpu_err)?;

        encode_embed_any(
            context,
            &pass,
            weights,
            index,
            embed_name,
            (&scratch.normed, 0),
            next_token as u32,
            hidden as u32,
            1.0,
        )?;
        let w = norm_view(weights, index, PRE_FC_NORM_EMBEDDING, hidden)?;
        gpu::encode_rms_norm_bf16w_centered(
            context,
            &pass,
            (&scratch.normed, 0),
            w,
            (&mtp.concat, 0),
            hidden as u32,
            RMS_EPS,
        )
        .map_err(gpu_err)?;
        // `fc` is [hidden, 2 * hidden] and its output IS the head's residual
        // stream, so it overwrites `scratch.x`. Safe only because the trunk's
        // logits for this token were read before the call.
        encode_gemv_any(
            context,
            &pass,
            weights,
            index,
            FC,
            hidden,
            2 * hidden,
            (&mtp.concat, 0),
            (&scratch.x, 0),
        )?;

        // 2. The block, which is a trunk full-attention layer under a
        //    different prefix and against the head's own KV.
        let input_norm = norm_view(
            weights,
            index,
            &prefixed_layer_tensor(MTP_PREFIX, 0, "input_layernorm.weight"),
            hidden,
        )?;
        gpu::encode_rms_norm_bf16w_centered(
            context,
            &pass,
            (&scratch.x, 0),
            input_norm,
            (&scratch.normed, 0),
            hidden as u32,
            RMS_EPS,
        )
        .map_err(gpu_err)?;
        // CENTERED, unlike the trunk's tensors of the same names one call
        // site over. The head's `q_norm`/`k_norm` store an offset from unity
        // exactly as its five whole-vector norms do; MTPLX's
        // `_RMSNORM_SUFFIXES` lists all seven, and the raw means here read
        // 0.780 and 0.797 against its "healthy >= 1.74" threshold.
        encode_full_attention_block(
            context,
            &pass,
            weights,
            index,
            &arch,
            qwen,
            scratch,
            &mtp.kv,
            MTP_PREFIX,
            0,
            position,
            QkNormConvention::Centered,
        )?;
        // RAW residual add. This family has `ffn_sandwich_norms: false`, and
        // normalizing here is the mutation that took the Qwen 3.6 reference
        // perplexity from 6.25 to 255,409 once already (Gotcha 11).
        gpu::encode_residual_add(
            context,
            &pass,
            (&scratch.x, 0),
            (&scratch.o, 0),
            hidden as u32,
        )
        .map_err(gpu_err)?;

        let post_attn = norm_view(
            weights,
            index,
            &prefixed_layer_tensor(MTP_PREFIX, 0, "post_attention_layernorm.weight"),
            hidden,
        )?;
        gpu::encode_rms_norm_bf16w_centered(
            context,
            &pass,
            (&scratch.x, 0),
            post_attn,
            (&qwen.moe_x, 0),
            hidden as u32,
            RMS_EPS,
        )
        .map_err(gpu_err)?;
        // The head's FFN is the trunk's DENSE width, and `residual` is the
        // head's own `x` -- the parameter `dense.rs` grew for this one caller.
        dense::encode_qwen_layer_dense(
            context, &pass, weights, index, scratch, qwen, &scratch.x, MTP_PREFIX, 0, hidden,
            inter, use_silu,
        )?;

        // 3. The head's own final norm, then the TRUNK's lm_head: the head
        //    has no output projection of its own. Both are skipped on a
        //    priming step, which wants only the KV row this block just wrote.
        if logits.is_some() {
            let final_norm = norm_view(weights, index, FINAL_NORM, hidden)?;
            gpu::encode_rms_norm_bf16w_centered(
                context,
                &pass,
                (&scratch.x, 0),
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
                &head_name,
                vocab,
                hidden,
                (&scratch.normed, 0),
                (&scratch.logits, 0),
            )?;
        }

        // The wait is NOT conditional: a priming step exists for its KV row,
        // and the row is not written until the GPU has run this block.
        pass.commit_and_wait();
        self.real_mtp.as_mut().expect("checked above").kv.advance();
        if let Some(out) = logits {
            gpu::read_buffer_f16_into(&self.scratch.logits, 0, out);
        }
        if let Some(dir) = dump_dir() {
            self.dump_mtp_stage(&dir, next_token, position, hidden, vocab);
        }
        Ok(())
    }

    /// Writes the head's surviving intermediates for `scripts/mtp_bisect.py`.
    ///
    /// **Which tensors these are is decided by what OUTLIVES the pass, not by
    /// what would be nicest to have.** One command buffer runs the whole step,
    /// so anything overwritten downstream is gone by the time the host can
    /// read it: `fc`'s raw output is clobbered when the attention residual
    /// adds into `scratch.x`. That one is recoverable offline -- the script
    /// recomputes it from `concat` and the `fc` weights -- so nothing is lost
    /// and the step keeps its single-commit shape. What survives is enough to
    /// bisect: `concat` is the exact input, `moe_x` is the post-attention
    /// norm (so it brackets `fc` AND attention), `x` is the block output
    /// after the FFN residual, and `normed` is after the head's own norm.
    fn dump_mtp_stage(
        &self,
        dir: &std::path::Path,
        token: i32,
        position: usize,
        hidden: usize,
        vocab: usize,
    ) {
        let _ = std::fs::create_dir_all(dir);
        let qwen = match self.real_qwen.as_ref() {
            Some(q) => q,
            None => return,
        };
        let mtp = match self.real_mtp.as_ref() {
            Some(m) => m,
            None => return,
        };
        let write = |name: &str, buf: &gpu::MetalBuffer, len: usize| {
            let mut host = vec![LogitValue::from_f32(0.0); len];
            gpu::read_buffer_f16_into(buf, 0, &mut host);
            let bytes: Vec<u8> = host
                .iter()
                .flat_map(|v| v.to_bits().to_le_bytes())
                .collect();
            let _ = std::fs::write(dir.join(name), bytes);
        };
        write("concat.f16", &mtp.concat, 2 * hidden);
        write("moe_x.f16", &qwen.moe_x, hidden);
        write("block_out.f16", &self.scratch.x, hidden);
        write("post_norm.f16", &self.scratch.normed, hidden);
        write("logits.f16", &self.scratch.logits, vocab);
        let _ = std::fs::write(
            dir.join("meta.json"),
            format!(
                "{{\"token\": {token}, \"position\": {position}, \
                 \"hidden\": {hidden}, \"vocab\": {vocab}}}\n"
            ),
        );
    }

    /// The configured draft depth, 0 when drafting is off.
    /// Why speculative decoding cannot run on this runner, or `None` if it
    /// can. See [`MtpState::speculation_blocker`]; this is the reachable form,
    /// answering for the install actually open.
    pub fn speculation_blocker(&self) -> Option<String> {
        MtpState::speculation_blocker(&self.index, &self.arch, self.real_mtp.is_some())
    }

    pub fn mtp_draft_depth(&self) -> usize {
        self.real_mtp.as_ref().map_or(0, |m| m.depth)
    }
}
