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

use std::sync::OnceLock;

use crate::speculative::DEFAULT_SPECULATION_BLOCK;

/// The controller's tuning: how long a clean streak must be to climb from a
/// depth band, and how many missed proposals drop one step.
///
/// [`Default`] is the shipped tuning ([`AdaptiveDepth::climb_threshold`] and
/// [`AdaptiveDepth::drop_pressure`]). `TURBOSPARK_ADAPTIVE_DEPTH=LOW,MID,HIGH,DROP`
/// overrides it for a process, read once, so a paired A/B or an offline
/// replay (`tslab depth-replay`) can try other values without a rebuild.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AdaptiveParams {
    /// Clean rounds needed to climb from depth 2 or 3.
    pub climb_low: usize,
    /// Clean rounds needed to climb from depth 4.
    pub climb_mid: usize,
    /// Clean rounds needed to climb from depth 5 and above.
    pub climb_high: usize,
    /// Accumulated missed proposals that drop one step.
    pub drop: usize,
}

impl Default for AdaptiveParams {
    fn default() -> Self {
        Self {
            climb_low: AdaptiveDepth::climb_threshold(2),
            climb_mid: AdaptiveDepth::climb_threshold(4),
            climb_high: AdaptiveDepth::climb_threshold(5),
            drop: AdaptiveDepth::drop_pressure(2),
        }
    }
}

impl AdaptiveParams {
    /// Clean rounds to climb one step from `depth`. Depth 0 or 1 keeps the
    /// shipped 2, since only a floor below 2 ever sits there.
    pub fn climb(&self, depth: usize) -> usize {
        match depth {
            0..=1 => AdaptiveDepth::climb_threshold(depth),
            2 | 3 => self.climb_low,
            4 => self.climb_mid,
            _ => self.climb_high,
        }
    }

    /// `LOW,MID,HIGH,DROP`, every field at least 1.
    pub fn parse(s: &str) -> Option<Self> {
        let v: Vec<usize> = s
            .split(',')
            .map(|x| x.trim().parse().ok().filter(|&n: &usize| n >= 1))
            .collect::<Option<_>>()?;
        match v.as_slice() {
            [low, mid, high, drop] => Some(Self {
                climb_low: *low,
                climb_mid: *mid,
                climb_high: *high,
                drop: *drop,
            }),
            _ => None,
        }
    }

    /// The process's tuning: `TURBOSPARK_ADAPTIVE_DEPTH` when it parses,
    /// else the default. Read once; an unparseable value says so once and
    /// falls back rather than silently running some other tuning.
    pub fn from_env() -> Self {
        static PARAMS: OnceLock<AdaptiveParams> = OnceLock::new();
        *PARAMS.get_or_init(|| match std::env::var("TURBOSPARK_ADAPTIVE_DEPTH") {
            Ok(v) => Self::parse(&v).unwrap_or_else(|| {
                eprintln!(
                    "TURBOSPARK_ADAPTIVE_DEPTH={v:?} is not LOW,MID,HIGH,DROP (each >= 1); \
                     using the default tuning"
                );
                Self::default()
            }),
            Err(_) => Self::default(),
        })
    }
}

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
    params: AdaptiveParams,
}

impl AdaptiveDepth {
    /// Starts at the floor. `cap` below `floor` collapses to a fixed depth at
    /// the cap, and both are clamped to at least 1 so a degenerate request can
    /// never ask for a round that proposes nothing.
    pub fn new(floor: usize, cap: usize) -> Self {
        Self::with_params(floor, cap, AdaptiveParams::from_env())
    }

    /// [`Self::new`] with explicit tuning instead of the process's.
    pub fn with_params(floor: usize, cap: usize, params: AdaptiveParams) -> Self {
        let cap = cap.max(1);
        let floor = floor.max(1).min(cap);
        Self {
            floor,
            cap,
            cur: floor,
            climb: 0,
            pressure: 0,
            params,
        }
    }

    /// The floor [`Self::new`] should get unless a caller measured otherwise.
    pub const DEFAULT_FLOOR: usize = DEFAULT_SPECULATION_BLOCK;

    /// `cap` held to the widest verify one pass of a row tile serves. A
    /// round at block B verifies B + 1 rows, so with a tile of `rows` rows
    /// the block past `rows - 1` pays a second tile pass for its last row.
    /// `None` (no tile in the verify path) leaves the cap alone.
    ///
    /// Under the INT4 MMA8 tile (8 rows) that second pass doubled the verify
    /// phase at block 8 while blocks 3 to 7 verified in about the same time,
    /// and the tuned controller at cap 7 decoded the same rounds a quarter
    /// faster than at cap 8 on a quote-back prompt, within noise on code
    /// (`docs/KAPITSA_KERNELS.md`, "Measured: the hybrid replay"). On prose
    /// the controller never nears the cap, so the clamp changes nothing
    /// there. A fixed block is the caller's choice and is not clamped.
    pub fn tile_cap(cap: usize, tile_rows: Option<usize>) -> usize {
        match tile_rows {
            Some(rows) => cap.min(rows.saturating_sub(1).max(1)),
            None => cap,
        }
    }

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
                if self.climb >= self.params.climb(self.cur) {
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
                if self.pressure >= self.params.drop {
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
