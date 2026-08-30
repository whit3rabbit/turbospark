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
    }

    /// The largest number of tokens [`Self::rollback`] can undo. Bounds the
    /// speculative block size on a sliding-window model; unbounded (in
    /// practice `usize::MAX`) on one whose layers all keep full history.
    pub fn max_rollback(&self) -> usize {
        self.kv.max_safe_rewind()
    }
}
