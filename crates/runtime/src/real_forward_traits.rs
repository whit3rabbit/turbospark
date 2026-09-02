use foundation::{LogitValue, TokenId};

use super::real_forward::{RealForwardRunner, RollbackPoint};
use crate::producer::{ChunkedPrefillRunner, LogitProducer, SpeculativeProducer};

impl LogitProducer for RealForwardRunner {
    fn reset(&mut self) {
        // PARK the live session rather than clobbering it, whenever there is
        // a pool to park it IN and something worth parking. This is the path
        // that actually fixes multi-conversation stomping
        // (`crates/server/CLAUDE.md`'s `--session-slots` Gotcha): every
        // caller that finds no match on the live session -- including the
        // speculative loop, which never calls `try_reuse_prefix` at all --
        // reaches `reset()`, and it used to destroy whatever conversation
        // was live with no chance to continue it later. Guarded on
        // `session_pool.capacity() > 0`, which is false at the default
        // `--session-slots 1`, so this is a true no-op there.
        self.session_slot_evicted = false;
        if self.session_pool.capacity() > 0 && self.kv.position() > 0 {
            let promoted = self.session_pool.take_lru();
            // The slot about to be overwritten (`kv.reset()` below) is an
            // EVICTION, reportable via `session_slot_evicted`, only if it
            // held a real conversation. A freshly-allocated slot that has
            // never served a request (`position() == 0`, every parked slot
            // at open) costs nothing to overwrite.
            self.session_slot_evicted = promoted.kv.position() > 0;
            let outgoing = self.swap_live_session(promoted);
            self.session_pool.park(outgoing);
        }
        // Cleared HERE, beside the cache it describes, so the two can never
        // disagree about what the state holds.
        self.kv_prefix.clear();
        self.kv.reset();
        if let Some(qwen) = self.real_qwen.as_mut() {
            qwen.reset();
        }
        // The head keeps its OWN KV, so the trunk's reset leaves it holding
        // the previous generation's context -- the same shape of leak
        // Gotcha 4 records for the GDN recurrent state, one cache over.
        if let Some(mtp) = self.real_mtp.as_mut() {
            mtp.reset();
        }
        // And the drafter its own, one cache over again.
        if let Some(dflash) = self.real_dflash.as_mut() {
            dflash.reset();
        }
        // Re-arm the residual capture for the next generation, so a caller
        // that opens once and walks a corpus gets one snapshot per prompt
        // rather than one per process.
        if let Some(capture) = self.resid_capture.as_mut() {
            capture.note_generation_start();
        }
        // **`prompt_vision` IS DELIBERATELY NOT CLEARED HERE, and M-V5 had it
        // the other way round.** `reset` is called at the START of a
        // generation by `run_raw_completion` itself, and a caller sets the
        // injection map just BEFORE that call -- so clearing here destroyed
        // the map for the very prompt it belonged to, and every `--image` run
        // prefilled placeholder embeddings and hallucinated. Measured on the
        // real install: the model described a page it had not been shown, with
        // the right prompt length and no error anywhere.
        //
        // Nothing caught it because every test that existed drove `produce`
        // directly. `an_injected_map_survives_the_generation_loops_own_reset`
        // is the guard, and it goes through `run_raw_completion`.
        //
        // What this field's lifetime rests on instead is the CALLER consuming
        // it: `clear_prompt_vision` after each turn, which is what the CLI's
        // page loop does. A caller who forgets gets NO injection on the next
        // turn rather than the previous page's -- the safe direction, since a
        // model handed placeholder embeddings answers vaguely instead of
        // describing the wrong picture confidently.
    }

    fn produce(
        &mut self,
        token: i32,
        position: usize,
        logits: &mut [LogitValue],
    ) -> Result<(), String> {
        let result = gpu::autorelease_pool(|| self.produce_inner(token, position, logits))
            .map_err(|e| e.to_string());
        // Recorded only on SUCCESS, and after the call: a failed forward may
        // have left the KV half-written, and a record naming a position the
        // pass did not complete is the one lie this whole mechanism exists to
        // avoid. `produce_prefill` routes through here, so this is the single
        // recording point for both.
        if result.is_ok() {
            self.kv_prefix.record(&[token], position);
        } else {
            self.kv_prefix.taint();
        }
        result
    }

    /// See `crate::kv_prefix`. Answers 0 unless the host opted in with
    /// [`RealForwardRunner::set_prefix_reuse`], so every existing caller
    /// keeps resetting and prefilling from position 0.
    ///
    /// COMMITS by moving the KV cursor back to the agreed prefix, and the
    /// two refusals around that are the whole safety argument:
    ///
    /// - a family with RECURRENT state (the gated-DeltaNet layers on the
    ///   qwen flows) folds history into a fixed-size accumulator through a
    ///   non-invertible update, so there is no going back without the ~60 MiB
    ///   snapshot `RollbackPoint` carries -- and nobody took one last turn.
    ///   `rewind_by` would leave that state describing tokens the KV no
    ///   longer holds, which is fluent wrong output rather than an error.
    /// - a SLIDING-WINDOW ring can only go back as far as its slack
    ///   (`max_safe_rewind`); past that the rows were overwritten by this
    ///   generation and attending over them is the same silent failure.
    ///
    /// Both refuse by returning 0, which costs a full prefill and nothing
    /// else.
    fn try_reuse_prefix(&mut self, prompt_ids: &[TokenId]) -> usize {
        if !self.prefix_reuse_enabled {
            return 0;
        }
        // Swap in whichever PARKED session best continues this prompt,
        // before scoring the live one. A no-op at `session_pool.capacity()
        // == 0` (the default), which is the whole byte-identity guarantee
        // for that case. See `Self::select_session`.
        let swapped = self.select_session(prompt_ids);
        let keep = self.kv_prefix.common_prefix(prompt_ids);
        if keep == 0 {
            return 0;
        }
        let cursor = self.kv.position();
        if keep > cursor {
            return 0;
        }
        let back = cursor - keep;
        if back > 0 {
            if self.real_qwen.is_some() {
                return 0;
            }
            if back > self.kv.max_safe_rewind() {
                return 0;
            }
            // A session pool changes what a small `keep` MEANS, but only for
            // the session that `select_session` did NOT already vet.
            //
            // When it found nothing better than what was already live
            // (`swapped == false`), a shallow `keep` is exactly as
            // suspicious as it always was pre-pooling, and now more
            // dangerous: without a pool, a shallow match against an
            // unrelated prompt is harmless (the discarded tail was never
            // going to be read again). WITH a pool, that same shallow match
            // is DESTRUCTIVE -- it silently overwrites the live session's
            // real content in place instead of PARKING it, because parking
            // only happens inside `reset()`, which this rewind path exists
            // specifically to avoid calling. Measured on a real install:
            // two conversations sharing nothing but a chat template's
            // opening tokens (`<bos><start_of_turn>user\n`, 6 tokens on
            // Gemma 4) still score a nonzero `keep` against each other, and
            // a session live on that shared prefix gets its real content
            // clobbered by the unrelated turn before the pool ever gets a
            // chance to preserve it.
            //
            // When `select_session` DID swap, the pool has already done the
            // vetting: this candidate beat every other live-or-parked
            // option FOR THIS PROMPT, so a small `keep` here is not a
            // coincidence to distrust, it is this codebase's own
            // established norm for what real reuse looks like (the CLI's
            // real chat REPL measures 13/33 and 29/49 -- 39% and 59% --
            // as WORKING reuse, both of which a `back > keep` bar would
            // refuse). Refusing the swap's own verdict here would not save
            // this session; it would force ANOTHER reset() that re-parks
            // the very session `select_session` just picked, evicts
            // whatever else was parked to make room for a THIRD, and ends
            // up reusing nothing at all -- measured: exactly this cascade,
            // needlessly evicting an unrelated third conversation's session
            // to serve a request that already had its own real match in
            // hand.
            if self.session_pool.capacity() > 0 && !swapped && back > keep {
                return 0;
            }
            self.kv.rewind_by(back);
            self.kv_prefix.rewind_to(keep);
        }
        keep
    }

    fn produce_prefill(
        &mut self,
        token: i32,
        position: usize,
        scratch: &mut [LogitValue],
    ) -> Result<(), String> {
        self.skip_head = true;
        let result = self.produce(token, position, scratch);
        self.skip_head = false;
        result
    }

    fn session_slot_evicted(&self) -> bool {
        self.session_slot_evicted
    }
}

/// The MTP head as a drafter and the M-row pass as the verify
/// (`docs/MTP.md`). Every method here is a thin forward to an inherent one
/// that already existed and was reachable only from
/// `crates/bench/tests/mtp_accept_length_probe.rs`; the trait is what lets
/// `run_raw_completion_speculative` drive them.
///
/// The three refusals the pieces carry are NOT re-stated here, deliberately.
/// `mtp_draft_step` errors on an install with no head or a step off the
/// drafter's cursor, and `produce_batched` refuses a non-dense, non-INT4 or
/// KV-wrapping block by name. Restating them would be a second copy of a
/// condition that has to agree with the first, and the failure mode of
/// disagreeing is a refusal message that names the wrong cause.
impl SpeculativeProducer for RealForwardRunner {
    type Checkpoint = RollbackPoint;

    // WHICH DRAFTER answers is a property of the open state, and at most
    // one is ever open in practice: the CLI refuses a request that names
    // both. If both somehow are, the BLOCK drafter wins here, which keeps
    // `drafts_block_passes` and the priming/rewind pair in one branch.
    fn prime_drafter(&mut self, next: i32, position: usize) -> Result<(), String> {
        if self.real_dflash.is_some() {
            // The DFlash2 context write needs only the captured states for
            // `position`; `next` is an MTP-head input it has no use for.
            self.dflash_prime_from_capture(position)
                .map_err(|e| e.to_string())
        } else {
            self.mtp_prime_step(next, position)
                .map_err(|e| e.to_string())
        }
    }

    fn draft_step(
        &mut self,
        token: i32,
        position: usize,
        logits: &mut [LogitValue],
    ) -> Result<(), String> {
        self.mtp_draft_step(token, position, logits)
            .map_err(|e| e.to_string())
    }

    fn drafts_block_passes(&self) -> bool {
        self.real_dflash.is_some()
    }

    fn draft_block(
        &mut self,
        anchor: i32,
        base: usize,
        block: usize,
        proposals: &mut Vec<i32>,
    ) -> Result<(), String> {
        let want = self.real_dflash.as_ref().map_or(0, |d| d.block);
        // A SHORTER round is the normal end of a run, not a caller error:
        // `run_raw_completion_speculative` shrinks `round_block` against
        // the generation budget and the context window, so every capped
        // run passes through 1..want-1 on its way to 0. The drafter always
        // runs its own `want + 1` rows; the loop reads the first `block`
        // proposals and the rows above them are rewritten before anything
        // reads them again (`families/qwen/dflash.rs`'s cursor rules). A
        // LONGER one really is a caller error -- there is no forward to
        // take those proposals from.
        if block == 0 || block > want {
            return Err(format!(
                "the open DFlash2 drafter proposes at most {want} tokens, and the loop asked for \
                 {block}; open it at or above the block the loop is running"
            ));
        }
        self.dflash_draft_block(anchor, base, proposals)
            .map_err(|e| e.to_string())?;
        proposals.truncate(block);
        Ok(())
    }

    fn rewind_drafter(&mut self, position: usize) -> Result<(), String> {
        if self.real_dflash.is_some() {
            self.dflash_rewind_to(position).map_err(|e| e.to_string())
        } else {
            self.mtp_rewind_to(position).map_err(|e| e.to_string())
        }
    }

    fn checkpoint(&mut self) -> RollbackPoint {
        RealForwardRunner::checkpoint(self)
    }

    fn rollback(&mut self, point: &RollbackPoint) {
        RealForwardRunner::rollback(self, point)
    }

    fn verify(
        &mut self,
        feed: &[i32],
        base: usize,
        logits: &mut [LogitValue],
    ) -> Result<(), String> {
        let result = self
            .produce_batched(feed, base, logits)
            .map_err(|e| e.to_string());
        // TAINTED rather than recorded. A verify feeds a whole block and the
        // caller then rolls back whatever the target rejected, so the record
        // would have to track an accept count this method is not told. The
        // only caller is `run_raw_completion_speculative`, which does not
        // offer prefix reuse anyway, so refusing costs nothing real and is
        // the safe direction: a wrong record answers the next turn from
        // another conversation's state, silently.
        self.kv_prefix.taint();
        result
    }
}

impl ChunkedPrefillRunner for RealForwardRunner {
    /// Gemma 4, BOTH halves of `llama` (Mistral, Llama 2/3.x, Mixtral,
    /// `qwen3moe`), `muse_glimmer` and `gpt-oss` today, and the refusal is
    /// BY NAME rather than a silent fallback to the sequential path. A
    /// caller that asked for chunked prefill and quietly got the
    /// token-at-a-time loop would measure the old engine and report it as
    /// the new one, which is the failure mode this whole phase exists to
    /// avoid. [`Self::supports_chunked_prefill`] is the SAME predicate this
    /// refusal uses, so a caller deciding whether to route here at all and
    /// this method's own hard refusal can never disagree.
    fn prefill_chunk(
        &mut self,
        tokens: &[i32],
        start_position: usize,
        logits: &mut [LogitValue],
    ) -> Result<(), String> {
        let result = self.prefill_chunk_inner(tokens, start_position, logits);
        // The server routes long prompts through here rather than through
        // `produce`, so without this the case prefix reuse exists for would
        // record nothing and never fire.
        if result.is_ok() {
            self.kv_prefix.record(tokens, start_position);
        } else {
            self.kv_prefix.taint();
        }
        result
    }
}

impl RealForwardRunner {
    /// Swap in whichever PARKED session best continues `prompt_ids`, if any
    /// beats the LIVE session's own match. See `crate::session_pool`.
    ///
    /// Only ever SWAPS toward a genuine match (`SessionPool::best_match`
    /// already filters out an all-zero pool); it never manufactures a fresh
    /// session for an unmatched prompt; that is `reset()`'s job below, which
    /// is reached next when this leaves `keep == 0`.
    ///
    /// Returns whether it swapped. The caller (`try_reuse_prefix`) needs
    /// this: a swap means the pool has already vetted this candidate as the
    /// best available match FOR THIS PROMPT, which licenses trusting even a
    /// partial (well under 50%) reuse of it -- exactly this codebase's own
    /// established norm for what real reuse looks like (the CLI's real chat
    /// REPL measures 13/33 and 29/49, i.e. 39% and 59%, as working reuse).
    /// No swap means the live session was never compared favorably against
    /// anything else, so a shallow match against it gets no such benefit of
    /// the doubt.
    fn select_session(&mut self, prompt_ids: &[TokenId]) -> bool {
        if self.session_pool.capacity() == 0 {
            return false;
        }
        let live_score = self.kv_prefix.common_prefix(prompt_ids);
        let Some((idx, parked_score)) = self.session_pool.best_match(prompt_ids) else {
            return false;
        };
        // A tie favors the LIVE session: swapping to an equally-good parked
        // one would pay the swap for no benefit, and would needlessly evict
        // whatever slot lands in the outgoing live session's place.
        if parked_score <= live_score {
            return false;
        }
        let incoming = self.session_pool.take(idx);
        let outgoing = self.swap_live_session(incoming);
        self.session_pool.park(outgoing);
        true
    }

    /// Moves `incoming`'s `kv`/`gdn`/`kv_prefix` onto the live fields via
    /// `std::mem::replace` (an O(1) struct move, never a memcpy) and returns
    /// what was there, for the caller to park.
    ///
    /// A live `GdnStateManager` swap rather than the `GdnSnapshot`/
    /// `snapshot()`/`restore()` pair `RollbackPoint` uses for speculative
    /// rollback: that pair is a real host memcpy (tens to ~150 MiB
    /// depending on family, `crates/gpu/src/gdn_state.rs`'s own doc), which
    /// would reintroduce for GDN state exactly the per-switch cost a
    /// swap-based pool exists to avoid paying for KV. A second LIVE
    /// `GdnStateManager` costs one extra allocation at open and zero cost
    /// per switch.
    fn swap_live_session(
        &mut self,
        mut incoming: crate::session_pool::SessionSlot,
    ) -> crate::session_pool::SessionSlot {
        let kv = std::mem::replace(&mut self.kv, incoming.kv);
        let gdn = self.real_qwen.as_mut().and_then(|qwen| {
            incoming
                .gdn
                .take()
                .map(|fresh| std::mem::replace(&mut qwen.gdn, fresh))
        });
        let kv_prefix = std::mem::replace(&mut self.kv_prefix, incoming.kv_prefix);
        crate::session_pool::SessionSlot {
            kv,
            gdn,
            kv_prefix,
            // Restamped by `SessionPool::park`, which every caller of this
            // method calls immediately with the returned slot.
            last_used: 0,
        }
    }

    /// The family dispatch, split out so [`ChunkedPrefillRunner::prefill_chunk`]
    /// above is just "run it, then record what was fed".
    fn prefill_chunk_inner(
        &mut self,
        tokens: &[i32],
        start_position: usize,
        logits: &mut [LogitValue],
    ) -> Result<(), String> {
        if self.real.is_some() {
            return gpu::autorelease_pool(|| {
                self.prefill_chunk_real_gemma4(tokens, start_position, logits)
            })
            .map_err(|e| e.to_string());
        }
        if self.real_llama.as_ref().is_some_and(|s| s.dense) {
            return gpu::autorelease_pool(|| {
                self.prefill_chunk_real_llama_dense(tokens, start_position, logits)
            })
            .map_err(|e| e.to_string());
        }
        if self.real_llama.is_some() {
            return gpu::autorelease_pool(|| {
                self.prefill_chunk_real_llama_moe(tokens, start_position, logits)
            })
            .map_err(|e| e.to_string());
        }
        if self.real_muse.is_some() {
            return gpu::autorelease_pool(|| {
                self.prefill_chunk_real_muse(tokens, start_position, logits)
            })
            .map_err(|e| e.to_string());
        }
        if self.real_gpt_oss.is_some() {
            return gpu::autorelease_pool(|| {
                self.prefill_chunk_real_gpt_oss(tokens, start_position, logits)
            })
            .map_err(|e| e.to_string());
        }
        if self.real_qwen.as_ref().is_some_and(|s| s.dense) {
            return gpu::autorelease_pool(|| {
                self.prefill_chunk_real_qwen_dense(tokens, start_position, logits)
            })
            .map_err(|e| e.to_string());
        }
        Err(format!(
            "chunked prefill is wired for the real Gemma 4 flow, both halves of the \
             llama flow (Mistral, Llama 2/3.x, Mixtral, qwen3moe), muse_glimmer, \
             gpt-oss, and the dense qwen flow (qwenGdnDense) only; this install is {:?}",
            self.arch.family
        ))
    }
}
