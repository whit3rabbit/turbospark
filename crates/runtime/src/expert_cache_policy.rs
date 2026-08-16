//! How many routed-expert slots to cache per layer, and how `Auto` resolves.
//!
//! **This is a pure RAM-for-throughput trade and there is no third option on
//! this axis.** `docs/DECODE_BUDGET.md` measures a decoded token on the real
//! Gemma 4 install across an 8/16/32 slot sweep: GPU busy time is FLAT at
//! 10.26 / 10.24 / 9.99 ms while wall clock moves 25.07 -> 19.54, because the
//! whole variable is the exposed expert `pread` (7.62 -> 3.13) and the GPU
//! idles 52% -> 40% of the token waiting on it. Buying cache is the only
//! thing that fills that gap. The two alternatives are both measured dead
//! ends: expert prefetch (Swift benched it, 7% cross-layer predictor hits)
//! and shrinking the experts so a miss costs less (ROADMAP Phase S, where
//! 20% smaller experts decoded SLOWER at 2.0-2.4x the energy).
//!
//! Deliberately portable -- no `gpu`, no `#[cfg(target_os = "macos")]`, and
//! no clock or OS probe of its own. [`resolve`] takes the machine's memory as
//! a parameter, so every case below is a unit test that runs on any platform
//! rather than something only observable on a Mac with an install on disk.

use foundation::runtime_config::{ALLOWED_CACHE_SLOTS, DEFAULT_CACHE_SLOTS};

/// Held back from the budget for everything that is not the slot cache: the
/// KV cache (537 MiB for a dense 7B at 4,096 and over a GiB at 8,192), the
/// Metal command-buffer churn, the host-side sampler scratch, and whatever
/// else the user is running. A fixed figure rather than a fraction because
/// the terms it covers do not scale with installed memory.
pub const HEADROOM_RESERVE_BYTES: u64 = 4 * 1024 * 1024 * 1024;

/// The share of what is left after the reserve and the install's own resident
/// bytes that the slot cache may take. A quarter, so that three other things
/// the size of this engine can run beside it without the machine swapping.
pub const HEADROOM_FRACTION: f64 = 0.25;

/// Routed-expert cache sizing, as it reaches [`crate::RealForwardRunner`].
///
/// The mirror of `turbospark_invocation::ExpertCacheSlots`, spelled again
/// here for the reason `PowerProfile` is: this crate does not depend on the
/// pure argument parser, and `crates/cli` maps between the two.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ExpertCacheSlots {
    /// Exactly this many slots per layer, whatever the machine looks like.
    /// What every measuring harness passes -- see the note on [`resolve`].
    Fixed(usize),
    /// Size against the machine at open. Never resolves below
    /// [`DEFAULT_CACHE_SLOTS`].
    #[default]
    Auto,
}

impl ExpertCacheSlots {
    /// Resolve to a concrete per-layer slot count.
    ///
    /// `bytes_per_slot` is the whole-model cost of ONE additional slot,
    /// i.e. the per-layer expert stride summed over layers -- so the slot
    /// cache is `n * bytes_per_slot` and not `n * layers * stride` again.
    /// `resident_bytes` is the install's own mapped weight region, which
    /// `newBufferWithBytesNoCopy` pins for a streamed MoE install (AGENTS.md
    /// Gotcha 19), so it is genuinely unavailable rather than merely mapped.
    ///
    /// **`Auto` may only ever climb.** The floor is [`DEFAULT_CACHE_SLOTS`],
    /// which is what shipped before this existed, so no machine can be made
    /// slower by the feature being on: a 13 GB install on a 16 GB machine has
    /// no headroom by this arithmetic and gets exactly the 16 slots it always
    /// got, including the same allocation failure if 16 will not fit. An
    /// earlier draft floored at the bottom of `ALLOWED_CACHE_SLOTS` instead
    /// and would have resolved that machine DOWN to 8.
    ///
    /// Nothing that measures should ever pass `Auto`, and nothing in
    /// `crates/bench` does: the protocol pins `PROTOCOL_EXPERT_CACHE_SLOTS`
    /// and both memory oracles and all the quality gates go through it. That
    /// is the same discipline AGENTS.md Gotcha 35 states for the power
    /// profile -- when a knob has an environment-sensing default, the harness
    /// that measures the knob is exactly the caller that must not sense.
    /// A frozen footprint row taken at whatever the machine felt like that
    /// morning is not a row.
    pub fn resolve(self, physical_bytes: u64, resident_bytes: u64, bytes_per_slot: u64) -> usize {
        match self {
            Self::Fixed(n) => n,
            Self::Auto => auto_slots(physical_bytes, resident_bytes, bytes_per_slot),
        }
    }
}

/// The `Auto` arm of [`ExpertCacheSlots::resolve`], split out so the policy
/// can be tested without naming the enum in every case.
fn auto_slots(physical_bytes: u64, resident_bytes: u64, bytes_per_slot: u64) -> usize {
    let floor = DEFAULT_CACHE_SLOTS as usize;
    // A dense install has no routed experts, so every slot count describes
    // the same empty cache and the arithmetic below would divide by zero.
    if bytes_per_slot == 0 {
        return floor;
    }
    let free = physical_bytes.saturating_sub(resident_bytes.saturating_add(HEADROOM_RESERVE_BYTES));
    let budget = (free as f64 * HEADROOM_FRACTION) as u64;

    ALLOWED_CACHE_SLOTS
        .iter()
        .map(|&n| n as usize)
        .filter(|&n| n >= floor)
        .filter(|&n| (n as u64).saturating_mul(bytes_per_slot) <= budget)
        .max()
        .unwrap_or(floor)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One slot of Gemma 4 26B-A4B: 30 layers at ~3.2 MiB of expert stride.
    /// The install whose sweep in `docs/DECODE_BUDGET.md` this policy exists
    /// to act on.
    const GEMMA4_BYTES_PER_SLOT: u64 = 30 * 3_355_443;
    const GEMMA4_RESIDENT: u64 = 13 * 1024 * 1024 * 1024;
    const GIB: u64 = 1024 * 1024 * 1024;

    #[test]
    fn a_machine_with_headroom_climbs_to_the_top_of_the_allowed_set() {
        // This machine: 36 GB, a 13 GB install. (36 - 13 - 4) * 0.25 =
        // 4.75 GB against 3.0 GiB of slot cache at 32.
        assert_eq!(
            auto_slots(36 * GIB, GEMMA4_RESIDENT, GEMMA4_BYTES_PER_SLOT),
            32
        );
    }

    /// **The case that makes this file DISCRIMINATE `HEADROOM_FRACTION` and
    /// `HEADROOM_RESERVE_BYTES` at all**, and it had to be worked out rather
    /// than guessed at. Every other case here sits far from a boundary: at
    /// 36 GiB the budget clears 32 slots by 1.75 GiB and at 16 GiB it clears
    /// nothing, so doubling the fraction or zeroing the reserve moves neither
    /// answer and both constants could be silently wrong. A 27 GiB machine
    /// lands BETWEEN the two rungs -- budget 2.5 GiB against 2.25 for 24 slots
    /// and 3.0 for 32 -- so it is the one input where either constant changes
    /// the result. Same discipline AGENTS.md Gotcha 48 states for sub-4-bit
    /// packing: assert that the fixture can see the property before trusting
    /// what it says about it.
    #[test]
    fn the_budget_constants_decide_the_rung_at_the_boundary() {
        // (27 - 13 - 4) * 0.25 = 2.5 GiB. 24 slots want 2.25, 32 want 3.0.
        assert_eq!(
            auto_slots(27 * GIB, GEMMA4_RESIDENT, GEMMA4_BYTES_PER_SLOT),
            24
        );
        // Just under the 32-slot rung, and just over it: `P - 17 GiB` has to
        // reach 12 GiB for 3.0 GiB of budget at a quarter.
        assert_eq!(
            auto_slots(28 * GIB, GEMMA4_RESIDENT, GEMMA4_BYTES_PER_SLOT),
            24
        );
        assert_eq!(
            auto_slots(30 * GIB, GEMMA4_RESIDENT, GEMMA4_BYTES_PER_SLOT),
            32
        );
    }

    /// The case that made the floor necessary. Budgeting from headroom alone
    /// resolves this machine to 8 slots -- SLOWER than the 16 it gets today,
    /// which would make the feature a regression for exactly the users least
    /// able to absorb one.
    #[test]
    fn a_machine_with_no_headroom_stays_at_the_default_rather_than_dropping() {
        for physical in [8 * GIB, 16 * GIB, 18 * GIB] {
            assert_eq!(
                auto_slots(physical, GEMMA4_RESIDENT, GEMMA4_BYTES_PER_SLOT),
                DEFAULT_CACHE_SLOTS as usize,
                "physical {physical}"
            );
        }
    }

    /// The rule the floor encodes, asserted directly rather than left to the
    /// two cases above to imply: over any machine, any install and any expert
    /// granularity, `Auto` is incapable of returning less than what shipped.
    #[test]
    fn auto_can_never_resolve_below_the_shipped_default() {
        for physical_gib in [1u64, 2, 4, 8, 16, 24, 32, 36, 64, 96, 128, 512] {
            for resident_gib in [0u64, 1, 4, 13, 25, 60, 400] {
                for bytes_per_slot in [0, 1, 1024, GEMMA4_BYTES_PER_SLOT, 109 * 1024 * 1024] {
                    let got = auto_slots(physical_gib * GIB, resident_gib * GIB, bytes_per_slot);
                    assert!(
                        got >= DEFAULT_CACHE_SLOTS as usize,
                        "{physical_gib} GiB / {resident_gib} GiB / {bytes_per_slot} B \
                         resolved to {got}, below the {DEFAULT_CACHE_SLOTS} floor"
                    );
                }
            }
        }
    }

    /// Whatever it returns is a value the rest of the engine accepts:
    /// `RuntimeConfig`'s setters PANIC outside the allowed set rather than
    /// clamping (AGENTS.md Gotcha 2), so a resolver that invented 20 would
    /// abort the process somewhere else entirely.
    #[test]
    fn every_resolved_value_is_in_the_allowed_set() {
        for physical_gib in [1u64, 8, 16, 36, 64, 128, 512] {
            for bytes_per_slot in [0, 1, GEMMA4_BYTES_PER_SLOT, 109 * 1024 * 1024] {
                let got = auto_slots(physical_gib * GIB, GEMMA4_RESIDENT, bytes_per_slot);
                assert!(
                    ALLOWED_CACHE_SLOTS.contains(&(got as u32)),
                    "{physical_gib} GiB / {bytes_per_slot} B resolved to {got}, \
                     outside {ALLOWED_CACHE_SLOTS:?}"
                );
            }
        }
    }

    /// A coarse-grained MoE cannot buy its way up: Mixtral 8x7B is 8 experts
    /// of ~109 MiB over 32 layers, so ONE extra slot is 3.4 GiB and even a
    /// 128 GB machine stays at the floor. This is AGENTS.md Gotcha 36's
    /// arithmetic reaching the policy -- what decides the slot cache is
    /// expert GRANULARITY, never model size.
    #[test]
    fn a_coarse_grained_moe_stays_at_the_floor_on_any_machine() {
        let mixtral_bytes_per_slot = 32 * 109 * 1024 * 1024;
        for physical_gib in [16u64, 36, 64, 128] {
            assert_eq!(
                auto_slots(physical_gib * GIB, 27 * GIB, mixtral_bytes_per_slot),
                DEFAULT_CACHE_SLOTS as usize,
                "physical {physical_gib} GiB"
            );
        }
    }

    /// A dense install has no routed experts at all, so there is no cache to
    /// size and no stride to divide by.
    #[test]
    fn a_dense_install_resolves_to_the_default_without_dividing_by_zero() {
        assert_eq!(
            auto_slots(36 * GIB, 4 * GIB, 0),
            DEFAULT_CACHE_SLOTS as usize
        );
    }

    /// `Fixed` is the identity, including for values `Auto` would never
    /// choose. That is what lets a harness pin 8 for a constrained-cache arm.
    #[test]
    fn fixed_is_returned_untouched_and_ignores_the_machine() {
        for n in [8usize, 16, 24, 32] {
            assert_eq!(
                ExpertCacheSlots::Fixed(n).resolve(GIB, 400 * GIB, GEMMA4_BYTES_PER_SLOT),
                n
            );
        }
    }

    /// The default is `Auto`, which is the whole behavioural change: a
    /// caller that says nothing now gets sizing rather than a literal 16.
    #[test]
    fn the_default_is_auto() {
        assert_eq!(ExpertCacheSlots::default(), ExpertCacheSlots::Auto);
    }
}
