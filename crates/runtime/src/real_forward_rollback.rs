use crate::real_forward::RealForwardRunner;

/// An opaque restore point from [`RealForwardRunner::checkpoint`].
pub struct RollbackPoint {
    pub(crate) position: usize,
    pub(crate) gdn: Option<gpu::GdnSnapshot>,
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
        RollbackPoint {
            position: self.kv.position(),
            gdn: self.real_qwen.as_ref().map(|qwen| qwen.gdn.snapshot()),
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
            point.position <= self.kv.position(),
            "rollback target is ahead of the cursor"
        );
        self.kv.rewind_by(self.kv.position() - point.position);
        // The record describes the KV, so it rewinds with it. Without this a
        // speculative round that rolled back would leave the record claiming
        // the rejected tokens, and the next turn would match a prefix the
        // cache no longer holds.
        self.kv_prefix.rewind_to(point.position);
        if let (Some(qwen), Some(snapshot)) = (self.real_qwen.as_mut(), point.gdn.as_ref()) {
            qwen.gdn.restore(snapshot);
        }
        // The verify tape described a state this just undid. A retaining
        // rollback replays from the snapshot directly; nothing else should
        // be able to reach for a tape whose pass no longer happened.
        self.batched_tape = None;
        self.batched_tape_row0 = None;
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
        let snapshot = point.gdn.as_ref().ok_or_else(|| {
            "retaining rollback needs a GDN snapshot; this install has no recurrent state"
                .to_string()
        })?;
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
        // Recurrent state first, then the replay over the tape -- both
        // before the KV cursor moves, because the replay is the one
        // fallible step left and it reads neither the cursor nor the cache.
        {
            let qwen = self
                .real_qwen
                .as_mut()
                .ok_or_else(|| "retaining rollback needs a GDN install".to_string())?;
            qwen.gdn.restore(snapshot);
        }
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
        let (context, weights, index, scratch, arch) = (
            &mut self.context,
            &self.weights,
            &self.index,
            &self.scratch,
            &self.arch,
        );
        gpu::autorelease_pool(|| {
            let pass = context.begin_pass_labeled("gdn tape replay");
            crate::families::qwen::replay_linear_state_batched(
                context, &pass, weights, index, arch, qwen, batched, keep_rows, tape.rows,
            )?;
            pass.commit_and_wait();
            Ok(())
        })
        .map_err(|e: crate::real_forward_types::RealForwardError| e.to_string())?;
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
}
