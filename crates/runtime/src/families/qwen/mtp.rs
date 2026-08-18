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
    dense, encode_full_attention_block, prefixed_layer_tensor, QkNormConvention, MTP_PREFIX,
    RMS_EPS, TRUNK_PREFIX,
};
use crate::real_forward::{RealForwardError, RealForwardRunner};
use crate::real_forward_dispatch::{encode_embed_any, encode_gemv_any};
use crate::real_forward_utils::{entry, norm_view};

/// `mtp.fc.weight` and the three norms that are not inside the block.
const FC: &str = "mtp.fc.weight";
const PRE_FC_NORM_EMBEDDING: &str = "mtp.pre_fc_norm_embedding.weight";
const PRE_FC_NORM_HIDDEN: &str = "mtp.pre_fc_norm_hidden.weight";
const FINAL_NORM: &str = "mtp.norm.weight";
/// The TRUNK's final norm. The head's hidden input is what this produces,
/// not the residual stream underneath it; see `mtp_step`.
const TRUNK_FINAL_NORM: &str = "language_model.model.norm.weight";

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
    /// The M-row buffers the batched verify pass runs on (step 4).
    ///
    /// It lives HERE rather than beside `DecodeScratch` for the same reason
    /// the rest of this state does: `MFERENCE_MTP_DRAFT` unset must allocate
    /// nothing at all, which is what lets `qwen38_memory_oracle`'s frozen row
    /// keep describing the pre-MTP engine. Sized for `depth + 1` rows,
    /// because a round verifies the confirmed token plus `depth` proposals.
    pub(crate) batched: super::batched::BatchedScratch,
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
        policy: MtpDraftPolicy,
        gdn_shape: gpu::GdnShape,
    ) -> Result<Option<Self>, RealForwardError> {
        // The two conditions answer different questions and the ORDER used to
        // hide the detection entirely: `depth == 0` returned before anything
        // looked at the index, so the presence check below existed only to
        // word an error, and an install that HAD a head decoded sequentially
        // and silently unless an operator happened to set the env var.
        let depth = match policy {
            MtpDraftPolicy::Off => return Ok(None),
            MtpDraftPolicy::Auto => {
                if !install_has_mtp_head(index) {
                    return Ok(None);
                }
                MtpDraftPolicy::AUTO_DEPTH
            }
            MtpDraftPolicy::Fixed(depth) => {
                if !install_has_mtp_head(index) {
                    // Still an ERROR under an explicit request, and only under
                    // one: the caller named something this install cannot do.
                    return Err(RealForwardError::Unsupported(format!(
                        "MFERENCE_MTP_DRAFT={depth} asks for speculative drafting, but this \
                         install carries no multi-token-prediction head ({FC} is not in the \
                         resident index). The mlx-community conversion drops `mtp.*`; stream an \
                         install that adds the official checkpoint's last shard (docs/MTP.md)."
                    )));
                }
                depth
            }
        };
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
        // `depth + 1`: a round verifies the confirmed token plus `depth`
        // proposals, so the widest pass is one row wider than the block.
        let batched =
            super::batched::BatchedScratch::new(context, arch, gdn_shape, depth.saturating_add(1));
        Ok(Some(Self {
            kv,
            concat: context.new_output_buffer(2 * hidden * 2),
            depth,
            batched,
        }))
    }

    /// Rewinds the head to empty context, beside the trunk's own reset.
    pub(crate) fn reset(&mut self) {
        self.kv.reset();
    }

    /// The dtype tag [`crate::real_forward_dispatch::encode_gemm_any`] accepts.
    /// A LITERAL, mirroring that function's own arm for the reason its comment
    /// gives: spelling it as a named constant there becomes a catch-all
    /// binding rather than a comparison.
    const BATCHED_GEMM_DTYPE: u8 = 4;

    /// Why a speculative round could not run on this install, or `None` if it
    /// can. Checked at OPEN, so a caller learns before generating rather than
    /// part-way through a round.
    ///
    /// **A head is necessary and not sufficient, and that gap is a latent bug
    /// this function exists to close.** Drafting needs `mtp.fc.weight`;
    /// VERIFYING needs `produce_batched`, whose refusals are narrower --
    /// dense only, because the routed pair has no batched kernel, and INT4
    /// only, because `encode_gemm_any` has no other arm. So a 1-bit, 2-bit or
    /// MoE checkpoint that happened to carry a head would pass a
    /// head-presence check, enable speculation, and then fail at the first
    /// verify with the generation already under way.
    ///
    /// Note this says nothing about whether a model DECODES: every one of
    /// those installs decodes normally, and the batched pass is used by
    /// speculation alone.
    pub(crate) fn speculation_blocker(
        index: &ResidentIndex,
        arch: &ArchConfig,
        has_head: bool,
    ) -> Option<String> {
        // THE ARCHITECTURAL CHECKS COME FIRST, and the order is a choice about
        // which reason is more useful. A MoE or sub-4-bit install cannot
        // speculate whatever head it acquires, so naming the head as the
        // obstacle would send a reader looking for a checkpoint that does not
        // help. It also makes these two arms reachable on the fixtures that
        // exist, which a head-first order does not.
        if arch.num_experts != 0 {
            return Some(format!(
                "the batched verify is dense-only and this install routes to \
                 {} experts; the routed pair has no batched kernel",
                arch.num_experts
            ));
        }
        // The batched GEMM has one arm. Read off a tensor the verify really
        // dispatches rather than off the manifest, so a hand-edited manifest
        // cannot talk its way past it.
        //
        // THE FIRST FULL-ATTENTION LAYER, not layer 0. This architecture is
        // three linear layers to one full, so layer 0 has no `self_attn.*` at
        // all -- probing it reports "not in the resident index" on a perfectly
        // good install, which is a wrong answer dressed as a cautious one.
        let full = (0..arch.num_layers as usize).find(|&l| !arch.layer_is_linear(l));
        let Some(full) = full else {
            return Some(
                "the batched verify needs a full-attention layer to probe and this \
                 install declares none"
                    .to_string(),
            );
        };
        let probe = prefixed_layer_tensor(TRUNK_PREFIX, full, "self_attn.q_proj.weight");
        match index.entries.get(&probe) {
            None => Some(format!(
                "cannot tell whether the batched verify can run: {probe} is not in \
                 the resident index"
            )),
            Some(e) if e.dtype != Self::BATCHED_GEMM_DTYPE => Some(format!(
                "the batched verify is INT4-only and this install's {probe} is dtype {} \
                 (the 1-bit and 2-bit checkpoints of this architecture have no batched \
                 kernel); the model decodes normally, only speculation is unavailable",
                e.dtype
            )),
            Some(_) => {
                if !has_head {
                    return Some(format!(
                        "this install carries no multi-token-prediction head \
                         ({FC} is not in the resident index)"
                    ));
                }
                None
            }
        }
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

/// Reads `MFERENCE_MTP_DUMP`: a directory to write one draft step's
/// surviving intermediates into, for `scripts/mtp_bisect.py`. Off by
/// default, and it OVERWRITES on every step, so a caller that wants a
/// specific position takes exactly one step with it set.
pub(crate) fn dump_dir() -> Option<std::path::PathBuf> {
    std::env::var_os("MFERENCE_MTP_DUMP").map(std::path::PathBuf::from)
}

/// What a caller asked for, which is NOT the same question as whether the
/// install can serve it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MtpDraftPolicy {
    /// Build a head iff the install carries one. An install without one is
    /// not an error: nobody asked for anything this install cannot do.
    Auto,
    /// Never build one, whatever the install carries. What every MEASURING
    /// caller passes, for AGENTS.md Gotcha 35's reason -- a harness that
    /// sensed the environment would let a frozen footprint row acquire the
    /// head's allocation without saying so.
    Off,
    /// Build one at this depth, and ERROR if the install has no head.
    Fixed(usize),
}

impl MtpDraftPolicy {
    /// The depth `Auto` resolves to. A round of block B takes `B + 1` draft
    /// steps and verifies `B + 1` rows, so this is the largest block the
    /// resolved state can serve; block 2 is the measured optimum
    /// (`docs/MTP.md`), and a caller wanting more asks for it explicitly.
    pub const AUTO_DEPTH: usize = 2;

    /// Reads `MFERENCE_MTP_DRAFT`.
    pub fn from_env() -> Self {
        Self::from_env_value(std::env::var("MFERENCE_MTP_DRAFT").ok().as_deref())
    }

    /// The mapping [`Self::from_env`] applies, as a pure function of the
    /// string, so it can be tested without a process-global write that would
    /// race every other test in the binary.
    ///
    /// **UNSET is `Auto`, and that one line is what turned this feature from
    /// opt-in into detected.** An explicit 0 is still off, and any other
    /// parsable number is still an explicit depth, so nothing that used to
    /// work reads differently. An UNPARSABLE value is `Auto` rather than off,
    /// matching the unset case: a typo should not silently disable a feature
    /// the install can serve.
    pub fn from_env_value(raw: Option<&str>) -> Self {
        match raw.and_then(|v| v.trim().parse::<usize>().ok()) {
            None => MtpDraftPolicy::Auto,
            Some(0) => MtpDraftPolicy::Off,
            Some(n) => MtpDraftPolicy::Fixed(n),
        }
    }
}

/// Whether this install carries a multi-token-prediction head.
///
/// **This is read off the RESIDENT INDEX and there is no manifest field, by
/// design** (`crates/repack/CLAUDE.md`): the answer is the bytes, so nothing
/// can claim a head the install does not have, and a hand-edited manifest
/// cannot lie about it. It is also the whole of the detection story -- the
/// head's other tensors are checked in [`MtpState::build`], where a partial
/// head is an error rather than a reason to decline.
pub fn install_has_mtp_head(index: &ResidentIndex) -> bool {
    index.entries.contains_key(FC)
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
