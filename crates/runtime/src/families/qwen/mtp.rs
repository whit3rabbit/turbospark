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
use model_io::{ArchConfig, ResidentIndex};

use crate::families::qwen::{
    dense, encode_full_attention_block, prefixed_layer_tensor, MTP_PREFIX, RMS_EPS,
};
use crate::real_forward::{RealForwardError, RealForwardRunner};
use crate::real_forward_dispatch::{encode_embed_any, encode_gemv_any};
use crate::real_forward_utils::{entry, norm_view};

/// `mtp.fc.weight` and the three norms that are not inside the block.
const FC: &str = "mtp.fc.weight";
const PRE_FC_NORM_EMBEDDING: &str = "mtp.pre_fc_norm_embedding.weight";
const PRE_FC_NORM_HIDDEN: &str = "mtp.pre_fc_norm_hidden.weight";
const FINAL_NORM: &str = "mtp.norm.weight";

/// Every tensor a draft step binds, probed at build so a half-ingested head
/// fails at open rather than at the first draft.
const REQUIRED: [&str; 15] = [
    FC,
    PRE_FC_NORM_EMBEDDING,
    PRE_FC_NORM_HIDDEN,
    FINAL_NORM,
    "mtp.layers.0.input_layernorm.weight",
    "mtp.layers.0.post_attention_layernorm.weight",
    "mtp.layers.0.self_attn.q_proj.weight",
    "mtp.layers.0.self_attn.k_proj.weight",
    "mtp.layers.0.self_attn.v_proj.weight",
    "mtp.layers.0.self_attn.o_proj.weight",
    "mtp.layers.0.self_attn.q_norm.weight",
    "mtp.layers.0.self_attn.k_norm.weight",
    "mtp.layers.0.mlp.gate_proj.weight",
    "mtp.layers.0.mlp.up_proj.weight",
    "mtp.layers.0.mlp.down_proj.weight",
];

/// The head's per-open state: its own KV and the buffer `fc` reads.
pub(crate) struct MtpState {
    /// One full-attention layer. See the module header for why this is not
    /// the trunk's cache widened by one.
    pub(crate) kv: gpu::KvCacheManager,
    /// `[2 * hidden]` halfs: the normalized next-token embedding in the low
    /// half and the normalized trunk hidden state in the high half. `fc` is
    /// the only tensor in the model whose input is `2 * hidden` wide, so
    /// nothing else can be read through this buffer by accident.
    pub(crate) concat: gpu::MetalBuffer,
    /// How many tokens a round proposes, from `MFERENCE_MTP_DRAFT`.
    pub(crate) depth: usize,
}

impl MtpState {
    /// Returns `None` when no draft depth was asked for, so an install that
    /// HAS a head still allocates nothing until someone wants drafts.
    ///
    /// **The two conditions are deliberately different in kind.** Whether the
    /// install has a head is read off the resident index and cannot drift
    /// from the bytes (there is no manifest field, by design); whether to
    /// build the state is an operator's request. An install without a head
    /// and a depth asked for is an ERROR rather than a silent no-op -- the
    /// caller asked for something this install cannot do.
    pub(crate) fn build(
        context: &mut gpu::MetalContext,
        index: &ResidentIndex,
        arch: &ArchConfig,
        max_context: usize,
        depth: usize,
    ) -> Result<Option<Self>, RealForwardError> {
        if depth == 0 {
            return Ok(None);
        }
        if !index.entries.contains_key(FC) {
            return Err(RealForwardError::Unsupported(format!(
                "MFERENCE_MTP_DRAFT={depth} asks for speculative drafting, but this install \
                 carries no multi-token-prediction head ({FC} is not in the resident index). \
                 The mlx-community conversion drops `mtp.*`; stream an install that adds the \
                 official checkpoint's last shard (docs/MTP_SPECULATIVE.md)."
            )));
        }
        for name in REQUIRED {
            entry(index, name)?;
        }

        // A ONE-LAYER, FULL-ATTENTION clone. Every other field is the
        // trunk's, so the head's block is sized exactly as the layers it was
        // trained beside -- which is the same statement `mtp_head_network.rs`
        // makes about the published shapes, arriving here as a construction
        // rather than as an assertion.
        let mut head_arch = arch.clone();
        head_arch.num_layers = 1;
        head_arch.full_attention_layer_mask = vec![1];

        // `fp16_ring_enabled: false` and no sliding window: this block is
        // full attention, so a ring would be a linear buffer with extra
        // arithmetic. The chunk budget is 1 because a draft step is one
        // token and never a prefill chunk.
        let kv = gpu::KvCacheManager::new(
            context.device(),
            &head_arch,
            max_context,
            false,
            None,
            1,
            None,
        )
        .map_err(RealForwardError::Gpu)?;

        let hidden = arch.hidden_size as u64;
        Ok(Some(Self {
            kv,
            concat: context.new_output_buffer(2 * hidden * 2),
            depth,
        }))
    }

    /// Rewinds the head to empty context, beside the trunk's own reset.
    pub(crate) fn reset(&mut self) {
        self.kv.reset();
    }

    /// How many head positions have been written: the head's KV covers
    /// `[0, kv_position())`, so the next step must be taken AT that position.
    ///
    /// This is a real invariant rather than bookkeeping, because
    /// `encode_full_attention_block` derives its attention span from the
    /// `position` ARGUMENT (`position + 1`) and never from this cursor. A step
    /// taken PAST the cursor attends over rows nobody has written; a step
    /// taken BEHIND it silently re-drafts history. Neither is an error on its
    /// own, which is why [`RealForwardRunner::mtp_step`] checks it.
    pub(crate) fn kv_position(&self) -> usize {
        self.kv.position()
    }
}

/// Reads `MFERENCE_MTP_DRAFT`. Unset, unparsable or 0 is off, matching the
/// `MFERENCE_PREFILL_CHUNK` seam next door rather than inventing a third
/// convention for the same shape of switch.
pub(crate) fn draft_depth_from_env() -> usize {
    std::env::var("MFERENCE_MTP_DRAFT")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(0)
}

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
        for (src, name, dst_offset) in [
            ((&scratch.normed, 0u64), PRE_FC_NORM_EMBEDDING, 0u64),
            ((&scratch.x, 0u64), PRE_FC_NORM_HIDDEN, hidden as u64 * 2),
        ] {
            let w = norm_view(weights, index, name, hidden)?;
            gpu::encode_rms_norm_bf16w(
                context,
                &pass,
                src,
                w,
                (&mtp.concat, dst_offset),
                hidden as u32,
                RMS_EPS,
            )
            .map_err(gpu_err)?;
        }
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
        gpu::encode_rms_norm_bf16w(
            context,
            &pass,
            (&scratch.x, 0),
            input_norm,
            (&scratch.normed, 0),
            hidden as u32,
            RMS_EPS,
        )
        .map_err(gpu_err)?;
        encode_full_attention_block(
            context, &pass, weights, index, &arch, qwen, scratch, &mtp.kv, MTP_PREFIX, 0, position,
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
        gpu::encode_rms_norm_bf16w(
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
            gpu::encode_rms_norm_bf16w(
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
        Ok(())
    }

    /// The configured draft depth, 0 when drafting is off.
    pub fn mtp_draft_depth(&self) -> usize {
        self.real_mtp.as_ref().map_or(0, |m| m.depth)
    }
}
