use super::*;

const GIB: u64 = 1024 * 1024 * 1024;

/// The four tiers that form a total order. `Custom` is deliberately absent:
/// it is `Relaxed` plus a cap, so it sits at `Relaxed`'s place on the reserve
/// and fraction axes and below it on nothing else.
const ORDERED: [LoadGuard; 4] = [
    LoadGuard::Off,
    LoadGuard::Relaxed,
    LoadGuard::Balanced,
    LoadGuard::Strict,
];

/// **THE TEST THIS MODULE EXISTS TO NOT BREAK.** Every frozen memory-oracle
/// peak and every `measured` block in `models.json` describes an engine
/// budgeting at these three numbers. If the default moves, none of those rows
/// is wrong in a way anything reports -- they simply stop describing the
/// engine, which is the failure mode `assert_agrees_with_catalog` exists to
/// make loud and cannot see here.
#[test]
fn relaxed_is_exactly_todays_arithmetic() {
    let b = LoadGuard::Relaxed.budget();
    assert_eq!(b.reserve_bytes, crate::HEADROOM_RESERVE_BYTES);
    assert_eq!(b.budget_fraction, crate::CONTEXT_BUDGET_FRACTION);
    // `catalog::recommend::fit`'s TIGHT_FRACTION, which that crate cannot
    // import from here without a dependency edge it does not need. Stated as
    // a literal on both sides on purpose; the two are pinned against each
    // other by `fit_tests::the_default_guard_reproduces_the_frozen_thresholds`.
    assert_eq!(b.tight_fraction, 0.9);
    assert!(b.refuses);
    assert_eq!(b.hard_cap, None);
}

#[test]
fn the_default_is_relaxed_and_imposes_no_floor() {
    assert_eq!(LoadGuard::default(), LoadGuard::Relaxed);
    let policy = LoadPolicy::default();
    assert_eq!(policy.guard, LoadGuard::Relaxed);
    // Zero is what silence means: a caller that never asked for a minimum
    // has not asked to be refused.
    assert_eq!(policy.min_auto_context, 0);
    assert_eq!(LoadPolicy::new(LoadGuard::Strict).min_auto_context, 0);
}

/// The property a user relies on when moving the setting one notch. Nothing
/// about three independent fields enforces it, so it is swept rather than
/// asserted on the constants.
#[test]
fn the_tiers_are_ordered_from_off_down_to_strict() {
    for machine in [8 * GIB, 16 * GIB, 36 * GIB, 128 * GIB] {
        for committed in [0, GIB, 13 * GIB] {
            let avail: Vec<u64> = ORDERED
                .iter()
                .map(|g| g.available(machine, committed))
                .collect();
            let spend: Vec<f64> = ORDERED
                .iter()
                .zip(&avail)
                .map(|(g, &a)| a as f64 * g.budget().budget_fraction)
                .collect();
            for i in 1..ORDERED.len() {
                assert!(
                    avail[i - 1] >= avail[i],
                    "{:?} left less available than {:?} at {machine}/{committed}",
                    ORDERED[i - 1],
                    ORDERED[i]
                );
                assert!(
                    spend[i - 1] >= spend[i],
                    "{:?} would spend less than {:?} at {machine}/{committed}",
                    ORDERED[i - 1],
                    ORDERED[i]
                );
            }
        }
    }
}

/// A sweep over equal inputs proves nothing about a policy whose whole job is
/// to differ, so assert the tiers are DISTINGUISHABLE before believing the
/// ordering above. Without this, collapsing every arm of `budget()` onto
/// `Relaxed`'s numbers passes the ordering test on every input.
#[test]
fn the_tiers_are_distinguishable_on_a_real_machine() {
    let spends: Vec<u64> = ORDERED
        .iter()
        .map(|g| (g.available(36 * GIB, 13 * GIB) as f64 * g.budget().budget_fraction) as u64)
        .collect();
    for i in 1..spends.len() {
        assert!(
            spends[i - 1] > spends[i],
            "{:?} and {:?} spend the same on a 36 GiB machine",
            ORDERED[i - 1],
            ORDERED[i]
        );
    }
}

/// `Off` is not "a very large budget". A budget alone still refuses a request
/// past it, and this tier's whole purpose is to not do that.
#[test]
fn off_alone_declines_to_refuse_and_reserves_nothing() {
    let b = LoadGuard::Off.budget();
    assert!(!b.refuses);
    assert_eq!(b.reserve_bytes, 0);
    assert_eq!(b.budget_fraction, 1.0);
    assert_eq!(LoadGuard::Off.available(36 * GIB, 13 * GIB), 23 * GIB);
    for guard in [LoadGuard::Relaxed, LoadGuard::Balanced, LoadGuard::Strict] {
        assert!(guard.budget().refuses, "{guard:?} should refuse");
    }
}

#[test]
fn custom_carries_relaxeds_fractions_and_its_own_ceiling() {
    let custom = LoadGuard::Custom {
        max_counted_bytes: 3 * GIB,
    };
    let b = custom.budget();
    let relaxed = LoadGuard::Relaxed.budget();
    assert_eq!(b.reserve_bytes, relaxed.reserve_bytes);
    assert_eq!(b.budget_fraction, relaxed.budget_fraction);
    assert_eq!(b.tight_fraction, relaxed.tight_fraction);
    assert!(b.refuses);
    // The one thing that differs, and the reason the tier exists.
    assert_eq!(b.hard_cap, Some(3 * GIB));
    assert_eq!(relaxed.hard_cap, None);
}

/// A reserve larger than the machine is a real configuration (`Strict` on an
/// 8 GiB Mac reserves 12), and it must read as no budget rather than as a
/// wrapped one -- which would be the largest budget expressible.
#[test]
fn a_reserve_larger_than_the_machine_saturates_to_no_budget() {
    assert_eq!(LoadGuard::Strict.available(8 * GIB, 0), 0);
    assert_eq!(LoadGuard::Relaxed.available(8 * GIB, 13 * GIB), 0);
    assert_eq!(LoadGuard::Off.available(0, 0), 0);
}

#[test]
fn tiers_round_trip_through_their_spelling() {
    for guard in ORDERED {
        assert_eq!(LoadGuard::parse(guard.as_str()), Some(guard));
    }
    assert_eq!(LoadGuard::parse("Strict"), None);
    assert_eq!(LoadGuard::parse("aggressive"), None);
    // `custom` renders but does not parse: the word alone does not carry the
    // byte count, so each front end parses a number beside its own units.
    assert_eq!(
        LoadGuard::Custom {
            max_counted_bytes: 1
        }
        .as_str(),
        "custom"
    );
    assert_eq!(LoadGuard::parse("custom"), None);
}
