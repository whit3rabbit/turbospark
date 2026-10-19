//! Adaptive draft depth for the step-wise (MTP) speculative loop.
//!
//! A fixed block is the measured optimum on prose (`docs/MTP.md`: block 2
//! pays 1.66x, block 8 pays 1.17x) and the wrong answer on content the head
//! predicts almost perfectly, where a deeper round commits more tokens per
//! verify. [`AdaptiveDepth`] moves the per-round block between a floor and a
//! cap from the acceptance it just saw.
//!
//! The shape is a hysteresis controller, after ggml-org/llama.cpp#27210
//! (`draft-mtp-adaptive`): a streak of fully accepted rounds climbs one step,
//! and accumulated miss pressure drops one step. The idea is theirs. The
//! code and the constants are not: that PR's floor of 3 and its 3->4 barrier
//! were tuned on another engine, where verify and rollback cost differently.
//!
//! **THE CONSTANTS BELOW ARE UNMEASURED HYPOTHESES ON THIS ENGINE.** The
//! floor is [`DEFAULT_SPECULATION_BLOCK`] because block 2 is the measured
//! optimum here, and the barrier sits one step above it for the same reason.
//! Nothing in this module is a performance claim until the paired gate in
//! `crates/bench` has been run; an adaptive run that does not beat fixed
//! block 2 on code, or that loses on prose, ships nothing.

use crate::speculative::DEFAULT_SPECULATION_BLOCK;

/// Per-round draft depth controller. Pure state, no GPU, no I/O.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdaptiveDepth {
    floor: usize,
    cap: usize,
    cur: usize,
    /// Consecutive rounds that accepted every proposal.
    climb: usize,
    /// Sum of `offered - accepted` since the last full accept or depth change.
    pressure: usize,
}

impl AdaptiveDepth {
    /// Starts at the floor. `cap` below `floor` collapses to a fixed depth at
    /// the cap, and both are clamped to at least 1 so a degenerate request can
    /// never ask for a round that proposes nothing.
    pub fn new(floor: usize, cap: usize) -> Self {
        let cap = cap.max(1);
        let floor = floor.max(1).min(cap);
        Self {
            floor,
            cap,
            cur: floor,
            climb: 0,
            pressure: 0,
        }
    }

    /// The floor [`Self::new`] should get unless a caller measured otherwise.
    pub const DEFAULT_FLOOR: usize = DEFAULT_SPECULATION_BLOCK;

    /// The depth the next round should propose.
    pub fn depth(&self) -> usize {
        self.cur
    }

    /// Consecutive full-accept rounds needed to climb one step from `depth`.
    ///
    /// **Strict at every step, and that is the second iteration.** The first
    /// one climbed after 2 clean rounds from depth 5 up and measured 0.61x of
    /// fixed 2 on code, because it spent most rounds at depth 4 to 7 where
    /// per-position acceptance had decayed to 0.72-0.50 (`docs/MTP_SPECULATIVE.md`).
    /// The break-even arithmetic says why: a round at depth B pays for B draft
    /// steps and B+1 verify rows, so depth beyond 2 only wins while the whole
    /// accept chain stays near 1.0. A long clean streak is the evidence of
    /// that, and a short one is not.
    const fn climb_threshold(depth: usize) -> usize {
        match depth {
            0..=1 => 2,
            2 | 3 => 8,
            4 => 6,
            _ => 4,
        }
    }

    /// Accumulated `offered - accepted` that drops one step from `depth`.
    /// Two missed tokens, whatever the depth: a single total miss at any
    /// depth above the floor drops, and so do two separate one-token misses
    /// with no clean round between them. Falling back is cheap (the floor is
    /// the measured optimum) and staying deep on a decaying chain is not.
    const fn drop_pressure(_depth: usize) -> usize {
        2
    }

    /// Feeds one verified round. `offered` is how many proposals the round
    /// made and `accepted` how many the target kept.
    ///
    /// A round truncated by the generation budget or the context window is
    /// not an acceptance signal and the caller must not feed it: a short
    /// round that fully accepts would otherwise count toward a climb it did
    /// not earn. `offered == 0` is ignored here for the same reason.
    pub fn observe(&mut self, offered: usize, accepted: usize) {
        if offered == 0 {
            return;
        }
        let accepted = accepted.min(offered);
        if accepted == offered {
            self.pressure = 0;
            if self.cur < self.cap {
                self.climb += 1;
                if self.climb >= Self::climb_threshold(self.cur) {
                    self.cur += 1;
                    self.climb = 0;
                }
            }
        } else {
            self.climb = 0;
            // At the floor there is nowhere to fall, so no pressure builds
            // and a later deep run is not charged for old misses.
            if self.cur > self.floor {
                self.pressure += offered - accepted;
                if self.pressure >= Self::drop_pressure(self.cur) {
                    self.cur -= 1;
                    self.pressure = 0;
                }
            }
        }
    }
}

#[cfg(test)]
#[path = "speculative_adaptive_tests.rs"]
mod tests;
