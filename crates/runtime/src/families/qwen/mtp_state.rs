use model_io::{ArchConfig, ResidentIndex};

use crate::families::qwen::{prefixed_layer_tensor, MOE_SPECULATION_BLOCKER_MARKER, TRUNK_PREFIX};
use crate::real_forward::RealForwardError;
use crate::real_forward_utils::entry;

/// `mtp.fc.weight` and the three norms that are not inside the block.
pub(crate) const FC: &str = "mtp.fc.weight";
pub(crate) const PRE_FC_NORM_EMBEDDING: &str = "mtp.pre_fc_norm_embedding.weight";
pub(crate) const PRE_FC_NORM_HIDDEN: &str = "mtp.pre_fc_norm_hidden.weight";
pub(crate) const FINAL_NORM: &str = "mtp.norm.weight";
/// The TRUNK's final norm. The head's hidden input is what this produces,
/// not the residual stream underneath it; see `mtp_step`.
pub(crate) const TRUNK_FINAL_NORM: &str = "language_model.model.norm.weight";

/// Every tensor a draft step binds, probed at build so a half-ingested head
/// fails at open rather than at the first draft.
pub(crate) const REQUIRED: [&str; 15] = [
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
    /// How many tokens a round proposes, from `TURBOSPARK_MTP_DRAFT`.
    pub(crate) depth: usize,
    /// The M-row buffers the batched verify pass runs on (step 4).
    ///
    /// It lives HERE rather than beside `DecodeScratch` for the same reason
    /// the rest of this state does: `TURBOSPARK_MTP_DRAFT` unset must allocate
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
                        "TURBOSPARK_MTP_DRAFT={depth} asks for speculative drafting, but this \
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
        // The tape records: the MTP path's partial rounds replay over it
        // exactly as the DFlash2 path does.
        let batched = super::batched::BatchedScratch::new(
            context,
            arch,
            gdn_shape,
            depth.saturating_add(1),
            true,
        )
        .map_err(RealForwardError::Gpu)?;
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
    /// VERIFYING needs `produce_batched`, which is INT4 only, because
    /// `encode_gemm_any` has no other arm. So a 1-bit or 2-bit checkpoint
    /// that happened to carry a head would pass a head-presence check, enable
    /// speculation, and then fail at the first verify with the generation
    /// already under way.
    ///
    /// **THE MoE ARM IS THE OTHER KIND OF REFUSAL AND THE TWO ARE NOT
    /// INTERCHANGEABLE.** `produce_batched` was dense-only until ROADMAP
    /// Phase 3 (`5640c3f`) landed the routed half, so that arm no longer
    /// reports a capability the engine lacks: it reports that nothing can
    /// DRIVE the verify, since no published MoE conversion of this
    /// architecture carries an ingestible drafter. Lifting it is a checkpoint
    /// question rather than a kernel one, which is why the message says so.
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
        // which reason is more useful. Neither a MoE nor a sub-4-bit install
        // can speculate on any head it acquires from the official checkpoint,
        // so naming THIS install's missing head would send a reader after a
        // shard that does not help. It also makes these two arms reachable on
        // the fixtures that exist, which a head-first order does not.
        //
        // THE MoE ARM IS A POLICY AND SAYS SO. It used to read "the routed
        // pair has no batched kernel", which was true until `5640c3f` and
        // sends a reader hunting a kernel that now exists and is gated
        // (`moe_batch.rs`). What is missing is a DRAFTER: every mlx
        // conversion of this architecture's MoE half drops `mtp.*` (read off
        // the published indexes), and Ornith's own head is itself MoE where
        // `REQUIRED` names the dense FFN tensors a `qwen3_5` head has.
        if arch.num_experts != 0 {
            return Some(format!(
                "{MOE_SPECULATION_BLOCKER_MARKER}: this install routes to {} experts, \
                 and no published MoE conversion of this architecture ships a drafter \
                 this port can ingest (every mlx conversion drops mtp.*). The batched \
                 routed verify itself runs, so this is a checkpoint gap and not a \
                 missing kernel",
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
                    // CARRIES THE POINTER `MtpState::build`'s open-time error
                    // carries, and the reason it has to is a consequence of
                    // the headless arm in `draft_policies`: since 2026-08-21 a
                    // named block on a DENSE headless install is refused HERE
                    // rather than at the open, so this is the only string such
                    // a caller sees, and it had quietly lost the half that
                    // says which artifact fixes it. The advice is correct on a
                    // dense install and a wild goose chase on a MoE one, which
                    // is why the MoE arm above returns first.
                    return Some(format!(
                        "this install carries no multi-token-prediction head \
                         ({FC} is not in the resident index); the mlx conversion drops \
                         mtp.*, so stream an install that adds the official checkpoint's \
                         last shard (docs/MTP.md)"
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

/// Reads `TURBOSPARK_MTP_DUMP`: a directory to write one draft step's
/// surviving intermediates into, for `scripts/mtp_bisect.py`. Off by
/// default, and it OVERWRITES on every step, so a caller that wants a
/// specific position takes exactly one step with it set.
pub(crate) fn dump_dir() -> Option<std::path::PathBuf> {
    std::env::var_os("TURBOSPARK_MTP_DUMP").map(std::path::PathBuf::from)
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

    /// Reads `TURBOSPARK_MTP_DRAFT`.
    pub fn from_env() -> Self {
        let val = std::env::var("TURBOSPARK_MTP_DRAFT").ok();
        Self::from_env_value(val.as_deref())
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
