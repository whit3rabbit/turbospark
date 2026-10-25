use crate::real_forward::RealForwardRunner;

/// `TURBOSPARK_GDN_LAZY_TAPE`, default on, read once. On: the batched verify
/// leaves the gated-DeltaNet state (recurrence and conv tail) at the block
/// start and records a rank-1 tape instead of advancing in place, and every
/// round ends by folding exactly the rows it committed
/// ([`RealForwardRunner::commit_pending`], or the fold inside
/// [`RealForwardRunner::rollback_retaining`]). Off: today's eager behavior
/// -- the verify advances the state in place, `checkpoint` snapshots it, a
/// partial round restores and replays. This is the A/B arm and the kill
/// switch; both modes must produce byte-identical committed streams, which
/// the dflash/MTP suites assert under both settings.
pub(crate) fn gdn_lazy_tape() -> bool {
    static LAZY: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *LAZY.get_or_init(|| std::env::var("TURBOSPARK_GDN_LAZY_TAPE").as_deref() != Ok("0"))
}

/// An opaque restore point from [`RealForwardRunner::checkpoint`].
pub struct RollbackPoint {
    pub(crate) position: usize,
    pub(crate) gdn: Option<gpu::GdnSnapshot>,
    /// True for a ROUND checkpoint ([`RealForwardRunner::checkpoint_round`])
    /// that skipped its snapshot because the live GDN state provably EQUALED
    /// this point. Rolling back to it restores nothing, restores the snapshot
    /// the first eager advance deferred, or -- if a fold moved the state past
    /// it -- refuses loudly.
    pub(crate) unsnapshotted: bool,
}

impl RollbackPoint {
    /// The KV position this point restores to.
    pub fn position(&self) -> usize {
        self.position
    }
}

/// What the last batched verify (`produce_batched`) left behind for
/// [`RealForwardRunner::rollback_retaining`]: where the pass started and
/// how many rows it fed. The metadata half only -- the recorded row inputs
/// themselves sit in the owning `BatchedScratch`'s tape buffers. Cleared by
/// every other trunk pass and by every rollback, so a stale record cannot
/// answer a call it no longer describes.
#[derive(Clone, Copy, Debug)]
pub(crate) struct VerifyTape {
    pub(crate) start_position: usize,
    pub(crate) rows: usize,
}

impl RealForwardRunner {
    /// Captures everything a forward pass mutates that is not addressed by
    /// position, so a later [`Self::rollback`] can undo an arbitrary run of
    /// tokens. The two halves need opposite treatment and that asymmetry is
    /// the whole content of this pair:
    ///
    /// - the KV cache keeps one row per position, so undoing it is moving a
    ///   cursor and nothing is copied here;
    /// - a gated-DeltaNet layer folds its whole history into a fixed-size
    ///   accumulator through a non-invertible update, so the only way back
    ///   is a copy taken beforehand (~60 MiB on Qwen 3.6, a memcpy).
    ///
    /// Take it BEFORE the tokens you may want to drop, and only when the
    /// previous pass has completed -- `produce` waits on its own command
    /// buffer before returning, so any point between calls is safe.
    pub fn checkpoint(&self) -> RollbackPoint {
        assert!(
            self.real_qwen4.is_none(),
            "checkpoint/rollback is not implemented for qwen4_exp; recurrent/hybrid state \
             (gdn, qsa, ngram_context, ple_conv_tail) cannot be rolled back (families/qwen4/state.rs)"
        );
        RollbackPoint {
            position: self.kv.position(),
            unsnapshotted: false,
            gdn: self
                .real_qwen
                .as_ref()
                .map(|qwen| qwen.gdn.snapshot(self.context.queue())),
        }
    }

    /// [`Self::checkpoint`] for the speculative loop, which takes one before
    /// every verify. Under `TURBOSPARK_GDN_LAZY_TAPE` a CLEAN state (a fold,
    /// restore or reset just ran, no eager advance since) IS the state this
    /// point names, and the lazy verify that follows never moves it, so the
    /// ~150 MiB snapshot blit would copy a value the live buffers still hold.
    /// That is every speculative round after the first.
    ///
    /// The skip is only safe because both ways the state can later leave the
    /// point are accounted for: an eager advance takes the skipped snapshot
    /// first ([`Self::note_eager_advance`]), and a fold or restore past the
    /// point taints it ([`Self::retire_round_checkpoint`]). Only ONE round
    /// checkpoint is tracked; taking a newer one replaces it, which is the
    /// loop's usage (one outstanding point per round). The generic
    /// [`Self::checkpoint`] never skips, so callers holding several points
    /// (`HiddenTrunk` prefix snapshots) keep full generality.
    pub fn checkpoint_round(&mut self) -> RollbackPoint {
        assert!(
            self.real_qwen4.is_none(),
            "checkpoint/rollback is not implemented for qwen4_exp"
        );
        let skip = gdn_lazy_tape() && self.gdn_state_clean && self.real_qwen.is_some();
        if !skip {
            self.unsnapshotted_checkpoint_alive = false;
            self.unsnapshotted_checkpoint_tainted = false;
            self.deferred_gdn_snapshot = None;
            return self.checkpoint();
        }
        self.unsnapshotted_checkpoint_alive = true;
        self.unsnapshotted_checkpoint_tainted = false;
        self.deferred_gdn_snapshot = None;
        RollbackPoint {
            position: self.kv.position(),
            unsnapshotted: true,
            gdn: None,
        }
    }

    /// Every EAGER GDN state advance (`produce`, a prefill pass, a hidden-row
    /// step) calls this BEFORE it encodes. The state stops being a committed
    /// boundary, and an alive snapshot-less round checkpoint takes the
    /// snapshot it skipped now, while the live state still equals it.
    pub(crate) fn note_eager_advance(&mut self) {
        self.gdn_state_clean = false;
        if self.unsnapshotted_checkpoint_alive {
            self.deferred_gdn_snapshot = self
                .real_qwen
                .as_ref()
                .map(|qwen| qwen.gdn.snapshot(self.context.queue()));
            self.unsnapshotted_checkpoint_alive = false;
        }
    }

    /// A fold or restore is about to move (or just moved) the state past an
    /// alive snapshot-less round checkpoint without an eager advance, so no
    /// snapshot can be taken for it any more. Taint it: a later rollback to
    /// it panics instead of restoring nothing.
    fn retire_round_checkpoint(&mut self) {
        if self.unsnapshotted_checkpoint_alive || self.deferred_gdn_snapshot.is_some() {
            self.unsnapshotted_checkpoint_alive = false;
            self.unsnapshotted_checkpoint_tainted = true;
            self.deferred_gdn_snapshot = None;
        }
    }

    /// Returns the runner to a [`Self::checkpoint`]. Afterwards the next
    /// `produce` must be called at `point.position`, and any tokens to keep
    /// are replayed through it.
    ///
    /// Panics if the KV rewind is not safe, which on a sliding-window model
    /// means more tokens than the ring's slack; see
    /// `gpu::KvCacheManager::max_safe_rewind`. That is deliberate: the
    /// alternative is attending over rows this generation has already
    /// overwritten, which produces plausible text rather than an error.
    pub fn rollback(&mut self, point: &RollbackPoint) {
        assert!(
            self.real_qwen4.is_none(),
            "rollback is not implemented for qwen4_exp; recurrent/hybrid state \
             (gdn, qsa, ngram_context, ple_conv_tail) cannot be rolled back (families/qwen4/state.rs)"
        );
        assert!(
            point.position <= self.kv.position(),
            "rollback target is ahead of the cursor"
        );
        if point.unsnapshotted {
            if self.unsnapshotted_checkpoint_tainted {
                panic!(
                    "a round checkpoint that skipped its snapshot was rolled back to after a \
                     fold committed the state past it; the state it names no longer exists \
                     anywhere, so restoring it is impossible rather than approximate"
                );
            }
            // An eager advance since the checkpoint took the skipped snapshot
            // first; with none, the live state provably still equals the point.
            if let (Some(qwen), Some(snapshot)) =
                (self.real_qwen.as_mut(), self.deferred_gdn_snapshot.as_ref())
            {
                qwen.gdn.restore(snapshot, self.context.queue());
            }
        } else {
            // A different point's restore moves the state under an alive
            // round checkpoint; it can no longer be answered.
            self.retire_round_checkpoint();
            if let (Some(qwen), Some(snapshot)) = (self.real_qwen.as_mut(), point.gdn.as_ref()) {
                qwen.gdn.restore(snapshot, self.context.queue());
            }
        }
        // The state is back at the point's boundary -- exactly the
        // cleanliness a following round checkpoint's snapshot-skip relies on.
        self.gdn_state_clean = true;
        if point.unsnapshotted {
            // If no eager advance happened yet, keep alive true so that any future
            // eager advance will still take the deferred snapshot. If a deferred
            // snapshot was already captured, it remains valid for subsequent rollbacks.
            if self.deferred_gdn_snapshot.is_none() {
                self.unsnapshotted_checkpoint_alive = true;
            }
            self.unsnapshotted_checkpoint_tainted = false;
        }
        self.kv.rewind_by(self.kv.position() - point.position);
        // The record describes the KV, so it rewinds with it. Without this a
        // speculative round that rolled back would leave the record claiming
        // the rejected tokens, and the next turn would match a prefix the
        // cache no longer holds.
        self.kv_prefix.rewind_to(point.position);
        // The verify tape described a state this just undid. A retaining
        // rollback replays from the snapshot directly; nothing else should
        // be able to reach for a tape whose pass no longer happened.
        self.batched_tape = None;
        self.batched_tape_row0 = None;
    }

    /// Returns true if `point` is currently safe to roll back to.
    /// An unsnapshotted round checkpoint becomes invalid once a commit folds
    /// the state past it.
    pub fn is_valid_rollback_point(&self, point: &RollbackPoint) -> bool {
        if point.position > self.kv.position() {
            return false;
        }
        if point.unsnapshotted && self.unsnapshotted_checkpoint_tainted {
            return false;
        }
        true
    }

    /// Attempts to rollback to `point`. If the point is stale (e.g. a round
    /// checkpoint that skipped its snapshot and has since been committed past),
    /// returns an error instead of panicking.
    pub fn try_rollback(&mut self, point: &RollbackPoint) -> Result<(), String> {
        if point.position > self.kv.position() {
            return Err("rollback target is ahead of the cursor".to_string());
        }
        if point.unsnapshotted && self.unsnapshotted_checkpoint_tainted {
            return Err(
                "a round checkpoint that skipped its snapshot was rolled back to after a \
                 fold committed the state past it; the state it names no longer exists"
                    .to_string(),
            );
        }
        self.rollback(point);
        Ok(())
    }

    /// Restores the target to `point` while KEEPING the first `keep_rows`
    /// rows the last verify fed: their KV rows stay in place (only the
    /// rejected tail's rows are rewound away), and the recurrent state is
    /// restored to the block start and replayed across the kept rows over
    /// the tape the verify recorded. The verify's own logits for row
    /// `keep_rows - 1` stay valid, so the caller does NOT re-verify -- which
    /// is the point, and the difference from [`Self::rollback`] followed by
    /// a shortened verify pass: on the real drafter's workloads a partial
    /// round is the COMMON case (17-98% of rounds by workload and block,
    /// `docs/DFLASH2.md`), and the replay is a few small kernel dispatches
    /// where the re-verify was a whole forward pass.
    ///
    /// `keep_rows` is the count of KEPT feed rows: the confirmed token plus
    /// every accepted proposal. The bonus token is not fed yet and is not
    /// part of the count.
    ///
    /// Errors, rather than degrading, when there is no tape for this
    /// checkpoint: the caller's alternative is a full re-verify, and
    /// silently doing the expensive thing is how a broken fast path hides.
    ///
    /// Every fallible check runs before anything is mutated, and the
    /// replay -- the one fallible step that remains -- runs BEFORE the KV
    /// cursor moves: a refusal leaves the runner untouched, and a replay
    /// failure leaves cursor and tape intact with the recurrent state at
    /// the checkpoint the call was restoring anyway, so a retry re-restores
    /// from the same snapshot and replays the same recorded inputs.
    pub fn rollback_retaining(
        &mut self,
        point: &RollbackPoint,
        keep_rows: usize,
    ) -> Result<(), String> {
        assert!(
            self.real_qwen4.is_none(),
            "retaining rollback is not implemented for qwen4_exp; recurrent/hybrid state \
             (gdn, qsa, ngram_context, ple_conv_tail) cannot be rolled back (families/qwen4/state.rs)"
        );
        let tape = *self.batched_tape.as_ref().ok_or_else(|| {
            "retaining rollback needs a verify tape; the last trunk pass was not a batched verify"
                .to_string()
        })?;
        assert!(
            point.position <= self.kv.position(),
            "rollback target is ahead of the cursor"
        );
        // The rewind below subtracts the rejected tail from the cursor, so
        // the tape's row count has to BE the rows the KV gained since the
        // pass started. Every trunk advance site clears the tape
        // (`real_forward_traits.rs`), so this holds by construction; the
        // assert is what turns a future pass that forgets to clear into a
        // loud stop rather than a rewind by a count nobody recorded.
        assert_eq!(
            self.kv.position(),
            tape.start_position + tape.rows,
            "the verify tape's row count does not match the KV cursor"
        );
        if tape.start_position != point.position {
            return Err(format!(
                "the verify tape is from position {}, but the checkpoint is at {}",
                tape.start_position, point.position
            ));
        }
        if keep_rows == 0 || keep_rows > tape.rows {
            return Err(format!(
                "retaining rollback of {keep_rows} rows against a tape of {} rows",
                tape.rows
            ));
        }
        // A snapshot-less round checkpoint names the live state only while
        // nothing moved it: a fold past it (tainted) or an eager advance
        // since (deferred snapshot present) means the verify did not start
        // from the point's state, and folding from here would be wrong.
        if point.unsnapshotted
            && (self.unsnapshotted_checkpoint_tainted || self.deferred_gdn_snapshot.is_some())
        {
            return Err(
                "the state moved after this snapshot-less round checkpoint was taken; the \
                 verify did not start from the state the retaining rollback assumes"
                    .to_string(),
            );
        }
        let lazy = gdn_lazy_tape();
        let snapshot = if lazy {
            // No snapshot exists to need: the verify this tape recorded never
            // moved the state, so the live buffers still hold the block start
            // and the fold below advances them by exactly the kept rows.
            None
        } else {
            Some(point.gdn.as_ref().ok_or_else(|| {
                "retaining rollback needs a GDN snapshot; this install has no recurrent state"
                    .to_string()
            })?)
        };
        // `keep_rows == 1` restores row 0 from the stash `produce_batched`
        // left (see the copy below), and that stash exists exactly when the
        // clobber it undoes happened: a verify of more than one row. A
        // one-row tape never clobbered row 0, so there the stash is
        // legitimately absent; at more rows its absence means the pairing
        // record is gone and the rebuild below would silently leave the
        // wrong residual for the drafter to read.
        if keep_rows == 1 && tape.rows > 1 && self.batched_tape_row0.is_none() {
            return Err(
                "the verify tape is missing its row-0 stash, so the kept row's residual \
                 cannot be restored"
                    .to_string(),
            );
        }
        // Recurrent state first, then the advance over the tape -- both
        // before the KV cursor moves, because the fold is the one fallible
        // step left and it reads neither the cursor nor the cache.
        if let Some(snapshot) = snapshot {
            let qwen = self
                .real_qwen
                .as_mut()
                .ok_or_else(|| "retaining rollback needs a GDN install".to_string())?;
            qwen.gdn.restore(snapshot, self.context.queue());
            // Restored to the block start: the cleanliness a following
            // checkpoint's snapshot-skip relies on.
            self.gdn_state_clean = true;
        }
        if lazy {
            // The fold replaces BOTH the restore and the replay: the state is
            // already at the block start, and applying the recorded rank-1
            // updates for the kept rows lands it exactly where the eager
            // restore-plus-replay did (bit for bit -- the fold is the same
            // element-wise sequence the eager kernel's registers ran, and
            // the conv-tail pass is the same kernel the replay dispatched).
            // On a fold failure the tape is intact and the state is still at
            // the block start, so a retry re-folds the same recorded inputs.
            self.fold_gdn_tape(keep_rows, tape)?;
            // The fold's output IS the committed boundary for the kept rows,
            // and it moved the state past the round checkpoint: the point is
            // consumed, so a stray rollback to it must refuse, not no-op.
            self.gdn_state_clean = true;
            self.retire_round_checkpoint();
        } else {
            let qwen = self
                .real_qwen
                .as_ref()
                .ok_or_else(|| "retaining rollback needs a GDN install".to_string())?;
            let batched = match (&self.real_mtp, &self.real_dflash) {
                (Some(m), _) => &m.batched,
                (None, Some(d)) => &d.batched,
                (None, None) => {
                    return Err(
                        "retaining rollback needs a drafter's batched scratch; none is open"
                            .to_string(),
                    )
                }
            };
            let (context, weights, index, arch) =
                (&mut self.context, &self.weights, &self.index, &self.arch);
            gpu::autorelease_pool(|| {
                let pass = context.begin_pass_labeled("gdn tape replay");
                crate::families::qwen::replay_linear_state_batched(
                    context, &pass, weights, index, arch, qwen, batched, keep_rows, tape.rows,
                )?;
                pass.commit_and_wait();
                Ok(())
            })
            .map_err(|e: crate::real_forward_types::RealForwardError| e.to_string())?;
        }
        // KV: the kept rows are the correct ones -- this pass wrote them at
        // the positions they occupy -- so only the rejected tail rewinds.
        self.kv.rewind_by(tape.rows - keep_rows);
        self.kv_prefix.rewind_to(point.position + keep_rows);
        // The last kept row's residual goes to scratch row 0, which is where
        // the step-wise drafter reads `h_t` -- the same contract the pass
        // itself serves at batch > 1 (`produce_batched`'s trailing copy),
        // kept exact here because the re-verify that used to refresh row 0
        // no longer runs.
        //
        // `keep_rows == 1` cannot use the "copy row (keep_rows - 1) into row
        // 0" rebuild every other keep_rows relies on: row (keep_rows - 1) IS
        // row 0, and `produce_batched`'s own trailing copy already
        // overwrote it with the LAST proposed row's residual (on the
        // assumption the whole block would be kept) before this function
        // ever ran. Reading and writing offset 0 in that case is a no-op
        // over the wrong value, not a restore -- so this case restores from
        // the pre-clobber stash `produce_batched` left instead.
        let hidden = self.arch.hidden_size as usize;
        let row_bytes = hidden * 2;
        let scratch = &self.scratch;
        if keep_rows == 1 {
            if let Some(saved) = self.batched_tape_row0.as_deref() {
                gpu::write_buffer_bytes(&scratch.x, 0, saved);
            }
        } else {
            let last = gpu::read_buffer_bytes(&scratch.x, (keep_rows - 1) * row_bytes, row_bytes);
            gpu::write_buffer_bytes(&scratch.x, 0, &last);
        }
        self.batched_tape = None;
        self.batched_tape_row0 = None;
        Ok(())
    }

    /// The largest number of tokens [`Self::rollback`] can undo. Bounds the
    /// speculative block size on a sliding-window model; unbounded (in
    /// practice `usize::MAX`) on one whose layers all keep full history.
    pub fn max_rollback(&self) -> usize {
        self.kv.max_safe_rewind()
    }

    /// The guard every non-speculative trunk pass runs at entry: a LAZY tape
    /// still present means a verify happened and its committed rows were
    /// never folded, and the pass about to run would discard the record and
    /// advance the state from the block start -- fluent wrong output rather
    /// than an error. The eager mode never defers, so under it this is
    /// always silent (and the tape is genuinely discardable, the behavior
    /// the clear sites have always had). Production cannot hit this: the
    /// speculative loop folds at every round end. A caller that can is a
    /// probe driving `verify` directly, and the loud refusal is the point.
    pub fn refuse_uncommitted_lazy_tape(&self) -> Result<(), String> {
        if gdn_lazy_tape() && self.batched_tape.is_some() {
            return Err(
                "an uncommitted lazy GDN tape is pending; the round must commit_pending or \
                 roll back before another trunk pass runs"
                    .to_string(),
            );
        }
        Ok(())
    }

    /// Discards any pending lazy verify tape, rewinding the KV cache to the
    /// tape's start position and leaving the runner clean for another trunk
    /// pass. This is the cleanup hook for speculative rounds cut short by an
    /// error (such as a sampling failure) between `verify` and
    /// `commit_pending`.
    pub fn discard_pending(&mut self) {
        if let Some(tape) = self.batched_tape.take() {
            self.batched_tape_row0 = None;
            if self.kv.position() >= tape.start_position {
                self.kv.rewind_by(self.kv.position() - tape.start_position);
                self.kv_prefix.rewind_to(tape.start_position);
            }
            self.gdn_state_clean = true;
        }
    }

    /// Blocks until any in-flight asynchronous GDN fold pass has completed on
    /// the GPU. Called before CPU host writes to GDN buffers (such as reset).
    pub fn wait_pending_gdn_fold(&mut self) {
        if let Some(pass) = self.pending_gdn_fold.take() {
            pass.wait();
        }
    }

    /// Commits the lazy verify tape's first `keep_rows` rows into the
    /// recurrent state. The round-end half of `TURBOSPARK_GDN_LAZY_TAPE`
    /// (default on): the verify left the state at the block start and
    /// recorded rank-1 updates, so the FULL-acceptance round -- the one case
    /// the eager path served for free and [`Self::rollback_retaining`] never
    /// saw -- folds here instead. The speculative loop calls this every round
    /// with the committed row count; it is a no-op on the eager path and
    /// whenever the tape is already gone (a partial round's
    /// [`Self::rollback_retaining`] folded its kept rows and cleared it).
    ///
    /// Errors, rather than degrading, when the tape does not cover
    /// `keep_rows`: a silent shortfall would leave the state behind the
    /// committed stream, which is fluent wrong output on every later token.
    pub fn commit_pending(&mut self, keep_rows: usize) -> Result<(), String> {
        if !gdn_lazy_tape() {
            return Ok(());
        }
        let Some(tape) = self.batched_tape else {
            return Ok(());
        };
        foundation::engaged!("gdn_lazy_tape");
        assert!(
            self.real_qwen4.is_none(),
            "the lazy GDN tape is not implemented for qwen4_exp"
        );
        if keep_rows == 0 || keep_rows > tape.rows {
            return Err(format!(
                "lazy GDN commit of {keep_rows} rows against a tape of {} rows",
                tape.rows
            ));
        }
        self.fold_gdn_tape(keep_rows, tape)?;
        // The fold's output IS the committed boundary for these rows, and it
        // moved the state past the round checkpoint (see
        // `retire_round_checkpoint`).
        self.gdn_state_clean = true;
        self.retire_round_checkpoint();
        // Nothing of the record survives the fold: the row-0 stash described
        // the same pass, and the next trunk pass must not reach for either.
        self.batched_tape = None;
        self.batched_tape_row0 = None;
        Ok(())
    }

    /// Runs the fold for the first `keep_rows` rows of `tape`: per linear
    /// layer, the recorded conv-tail advance over the taped rows, then the
    /// rank-1 state fold. Committed asynchronously on the GPU so CPU drafting
    /// is not blocked waiting for the fold on every round.
    fn fold_gdn_tape(&mut self, keep_rows: usize, tape: VerifyTape) -> Result<(), String> {
        let qwen = self
            .real_qwen
            .as_ref()
            .ok_or_else(|| "the lazy GDN tape needs a GDN install".to_string())?;
        let batched = match (&self.real_mtp, &self.real_dflash) {
            (Some(m), _) => &m.batched,
            (None, Some(d)) => &d.batched,
            (None, None) => {
                return Err(
                    "the lazy GDN tape needs a drafter's batched scratch; none is open".to_string(),
                )
            }
        };
        let (context, weights, index, arch) =
            (&mut self.context, &self.weights, &self.index, &self.arch);
        gpu::autorelease_pool(|| {
            let pass = context.begin_pass_labeled("gdn tape commit");
            crate::families::qwen::commit_linear_state_batched(
                context, &pass, weights, index, arch, qwen, batched, keep_rows, tape.rows,
            )?;
            let committed = pass.commit();
            self.pending_gdn_fold = Some(committed);
            Ok(())
        })
        .map_err(|e: crate::real_forward_types::RealForwardError| e.to_string())
    }
}
