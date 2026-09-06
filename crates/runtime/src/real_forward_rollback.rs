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
    pub fn rollback_retaining(
        &mut self,
        point: &RollbackPoint,
        keep_rows: usize,
    ) -> Result<(), String> {
        let tape = self.batched_tape.ok_or_else(|| {
            "retaining rollback needs a verify tape; the last trunk pass was not a batched verify"
                .to_string()
        })?;
        assert!(
            point.position <= self.kv.position(),
            "rollback target is ahead of the cursor"
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
        // KV: the kept rows are the correct ones -- this pass wrote them at
        // the positions they occupy -- so only the rejected tail rewinds.
        self.kv.rewind_by(tape.rows - keep_rows);
        self.kv_prefix.rewind_to(point.position + keep_rows);
        // Recurrent state: back to the block start, then forward again over
        // only the kept rows' recorded inputs.
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
        let arch = self.arch.clone();
        let (context, weights, index, scratch) =
            (&mut self.context, &self.weights, &self.index, &self.scratch);
        gpu::autorelease_pool(|| {
            let pass = context.begin_pass_labeled("gdn tape replay");
            crate::families::qwen::replay_linear_state_batched(
                context, &pass, weights, index, &arch, qwen, batched, keep_rows,
            )?;
            pass.commit_and_wait();
            Ok(())
        })
        .map_err(|e: crate::real_forward_types::RealForwardError| e.to_string())?;
        // The last kept row's residual goes to scratch row 0, which is where
        // the step-wise drafter reads `h_t` -- the same contract the pass
        // itself serves at batch > 1 (`produce_batched`'s trailing copy),
        // kept exact here because the re-verify that used to refresh row 0
        // no longer runs.
        let hidden = self.arch.hidden_size as usize;
        let row_bytes = hidden * 2;
        let last = gpu::read_buffer_bytes(&scratch.x, (keep_rows - 1) * row_bytes, row_bytes);
        gpu::write_buffer_bytes(&scratch.x, 0, &last);
        self.batched_tape = None;
        Ok(())
    }

    /// The largest number of tokens [`Self::rollback`] can undo. Bounds the
    /// speculative block size on a sliding-window model; unbounded (in
    /// practice `usize::MAX`) on one whose layers all keep full history.
    pub fn max_rollback(&self) -> usize {
        self.kv.max_safe_rewind()
    }
}
