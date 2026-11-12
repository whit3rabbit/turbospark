use super::*;

/// Feeds `n` rounds that accept everything offered at the current depth.
fn full_accepts(c: &mut AdaptiveDepth, n: usize) {
    for _ in 0..n {
        let d = c.depth();
        c.observe(d, d);
    }
}

#[test]
fn starts_at_the_floor() {
    assert_eq!(AdaptiveDepth::new(2, 8).depth(), 2);
}

#[test]
fn degenerate_bounds_collapse_without_a_zero_depth() {
    // cap below floor: the cap wins, so a fixed request stays fixed.
    assert_eq!(AdaptiveDepth::new(4, 2).depth(), 2);
    // zeros never produce a round that proposes nothing.
    assert_eq!(AdaptiveDepth::new(0, 0).depth(), 1);
    let mut c = AdaptiveDepth::new(3, 3);
    full_accepts(&mut c, 50);
    assert_eq!(c.depth(), 3, "floor == cap never moves");
}

#[test]
fn the_tile_cap_holds_the_block_to_one_verify_pass() {
    // An 8-row tile: block 7 verifies 8 rows in one pass, block 8 needs two.
    assert_eq!(AdaptiveDepth::tile_cap(8, Some(8)), 7);
    assert_eq!(
        AdaptiveDepth::tile_cap(4, Some(8)),
        4,
        "a cap under the tile is untouched"
    );
    assert_eq!(AdaptiveDepth::tile_cap(8, None), 8, "no tile, no clamp");
    assert_eq!(
        AdaptiveDepth::tile_cap(8, Some(1)),
        1,
        "a one-row tile still leaves a round"
    );
    let mut c = AdaptiveDepth::new(2, AdaptiveDepth::tile_cap(8, Some(8)));
    full_accepts(&mut c, 200);
    assert_eq!(
        c.depth(),
        7,
        "the controller never climbs past the clamped cap"
    );
}

#[test]
fn climbs_after_exactly_the_threshold_streak() {
    let mut c = AdaptiveDepth::new(2, 8);
    let need = AdaptiveDepth::climb_threshold(2);
    full_accepts(&mut c, need - 1);
    assert_eq!(c.depth(), 2, "one short of the streak must not climb");
    full_accepts(&mut c, 1);
    assert_eq!(c.depth(), 3);
}

#[test]
fn no_later_step_climbs_easier_than_the_first() {
    // The barrier is the whole point: prose should stay pinned at the floor,
    // and a deeper step must never be cheaper to reach than the first.
    let from_floor = AdaptiveDepth::climb_threshold(2);
    assert!(
        from_floor >= 8,
        "the first climb must need a long clean run"
    );
    for depth in 3..=8 {
        assert!(
            from_floor >= AdaptiveDepth::climb_threshold(depth),
            "depth {depth} climbs easier than the barrier"
        );
    }
}

#[test]
fn a_miss_resets_the_climb_streak() {
    let mut c = AdaptiveDepth::new(2, 8);
    let need = AdaptiveDepth::climb_threshold(2);
    full_accepts(&mut c, need - 1);
    c.observe(2, 1);
    full_accepts(&mut c, need - 1);
    assert_eq!(c.depth(), 2, "the streak must restart after a miss");
    full_accepts(&mut c, 1);
    assert_eq!(c.depth(), 3);
}

#[test]
fn never_exceeds_the_cap() {
    let mut c = AdaptiveDepth::new(2, 5);
    full_accepts(&mut c, 500);
    assert_eq!(c.depth(), 5);
}

#[test]
fn pressure_drops_one_step_and_resets() {
    let mut c = AdaptiveDepth::new(2, 8);
    full_accepts(&mut c, 500);
    assert_eq!(c.depth(), 8);
    // One-token misses, so the budget is counted in rounds: it must not drop
    // one round early, and must drop on the round that reaches it.
    let budget = AdaptiveDepth::drop_pressure(8);
    for _ in 0..budget - 1 {
        c.observe(8, 7);
        assert_eq!(c.depth(), 8, "dropped before the budget was reached");
    }
    c.observe(8, 7);
    assert_eq!(c.depth(), 7);
    // Pressure restarted: a single one-token miss at 7 must not drop again.
    assert!(AdaptiveDepth::drop_pressure(7) > 1);
    c.observe(7, 6);
    assert_eq!(c.depth(), 7);
}

#[test]
fn a_total_miss_above_the_floor_drops_at_once() {
    // The strict half of the retune: one bad round is enough.
    let mut c = AdaptiveDepth::new(2, 8);
    full_accepts(&mut c, 500);
    c.observe(8, 0);
    assert_eq!(c.depth(), 7);
}

#[test]
fn a_full_accept_clears_accumulated_pressure() {
    let mut c = AdaptiveDepth::new(2, 8);
    full_accepts(&mut c, 500);
    // One round short of the budget, cleared by a clean round, then again:
    // without the reset the second batch would carry the first one's
    // pressure across the line.
    let short = AdaptiveDepth::drop_pressure(8) - 1;
    assert!(
        short >= 1,
        "the budget must be more than one token for this"
    );
    for _ in 0..short {
        c.observe(8, 7);
    }
    c.observe(8, 8);
    for _ in 0..short {
        c.observe(8, 7);
    }
    assert_eq!(c.depth(), 8);
}

#[test]
fn no_pressure_builds_at_the_floor() {
    let mut c = AdaptiveDepth::new(2, 8);
    for _ in 0..1000 {
        c.observe(2, 0);
    }
    assert_eq!(c.depth(), 2, "cannot fall below the floor");
    // Old misses must not be charged to the first deep round after a climb.
    full_accepts(&mut c, AdaptiveDepth::climb_threshold(2));
    assert_eq!(c.depth(), 3);
    c.observe(3, 2);
    assert_eq!(c.depth(), 3, "one stale-pressure miss dropped the depth");
}

#[test]
fn an_empty_round_is_not_a_signal() {
    let mut c = AdaptiveDepth::new(2, 8);
    let need = AdaptiveDepth::climb_threshold(2);
    full_accepts(&mut c, need - 1);
    c.observe(0, 0);
    assert_eq!(c.depth(), 2, "an empty round must not complete a streak");
    full_accepts(&mut c, 1);
    assert_eq!(c.depth(), 3);
}

#[test]
fn accepted_beyond_offered_is_clamped_not_trusted() {
    let mut c = AdaptiveDepth::new(2, 8);
    c.observe(2, 99);
    assert_eq!(c.climb, 1, "treated as a full accept, not as a miss");
}

#[test]
fn a_prose_like_stream_stays_at_the_floor() {
    // The acceptance the doc measured on prose (single-step ~0.9, with a
    // miss every few rounds) must not walk the controller upward.
    let mut c = AdaptiveDepth::new(2, 12);
    for round in 0..600 {
        if round % 4 == 3 {
            c.observe(2, 1);
        } else {
            c.observe(2, 2);
        }
    }
    assert_eq!(c.depth(), 2);
}

#[test]
fn a_perfect_stream_reaches_the_cap_and_a_collapse_returns_to_the_floor() {
    let mut c = AdaptiveDepth::new(2, 10);
    full_accepts(&mut c, 500);
    assert_eq!(c.depth(), 10);
    for _ in 0..500 {
        let d = c.depth();
        c.observe(d, 0);
    }
    assert_eq!(c.depth(), 2);
}

#[test]
fn a_stream_that_misses_every_eighth_round_never_leaves_the_floor() {
    // Seven clean rounds then a miss is ~94% round-level acceptance, which
    // looks good and is exactly the code-like stream the first iteration
    // climbed on and lost 39 percent with. The streak never reaches 8.
    let mut c = AdaptiveDepth::new(2, 8);
    for round in 0..800 {
        if round % 8 == 7 {
            c.observe(2, 1);
        } else {
            c.observe(2, 2);
        }
    }
    assert_eq!(c.depth(), 2);
}

#[test]
fn default_params_are_the_shipped_tuning() {
    let p = AdaptiveParams::default();
    for depth in 0..=12 {
        assert_eq!(
            p.climb(depth),
            AdaptiveDepth::climb_threshold(depth),
            "depth {depth}"
        );
    }
    assert_eq!(p.drop, AdaptiveDepth::drop_pressure(8));
}

#[test]
fn params_parse_and_refuse() {
    assert_eq!(
        AdaptiveParams::parse("8,6,4,2"),
        Some(AdaptiveParams::default())
    );
    assert_eq!(
        AdaptiveParams::parse(" 3, 2,1 ,1"),
        Some(AdaptiveParams {
            climb_low: 3,
            climb_mid: 2,
            climb_high: 1,
            drop: 1
        })
    );
    for bad in ["", "8,6,4", "8,6,4,2,1", "0,6,4,2", "a,6,4,2", "-1,6,4,2"] {
        assert_eq!(AdaptiveParams::parse(bad), None, "{bad:?} accepted");
    }
}

#[test]
fn custom_params_change_when_the_controller_climbs() {
    let quick = AdaptiveParams {
        climb_low: 2,
        climb_mid: 2,
        climb_high: 2,
        drop: 1,
    };
    let mut c = AdaptiveDepth::with_params(2, 8, quick);
    full_accepts(&mut c, 2);
    assert_eq!(c.depth(), 3, "a streak of 2 climbs under the quick tuning");
    // One missed token drops at once with drop = 1.
    c.observe(3, 2);
    assert_eq!(c.depth(), 2);
}
