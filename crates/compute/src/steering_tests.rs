use super::*;

fn direction(n: usize) -> Vec<f32> {
    (0..n).map(|i| ((i as f32) * 0.37).sin()).collect()
}

fn row(n: usize) -> Vec<f32> {
    (0..n).map(|i| ((i as f32) * 0.11).cos() * 3.0).collect()
}

fn l2(v: &[f32]) -> f32 {
    v.iter().map(|a| a * a).sum::<f32>().sqrt()
}

/// A row carrying a LARGE component along `d`, which the plain [`row`]
/// fixture does not.
///
/// It exists because the discrimination test below MEASURED that [`row`]
/// cannot do the job: it reads a rescale factor of 1.006, so `Renorm` and
/// `Ablate` agree on it to within 0.6% and comparing them there proves
/// almost nothing. This is not an inflated fixture either -- on the real
/// `qwen38-27b` the extracted direction is 19-28% of the residual stream
/// at the late layers (`docs/OBLITERATION.md`), which is the regime the
/// mode exists for.
fn row_carrying(n: usize) -> Vec<f32> {
    let d = direction(n);
    row(n)
        .iter()
        .zip(d.iter())
        .map(|(x, dv)| x + 4.0 * dv)
        .collect()
}

/// The defining property of ablation: after it, the stream carries none
/// of the direction. This is the one assertion that would survive a
/// rewrite of everything above it.
#[test]
fn ablation_at_alpha_one_removes_the_whole_component() {
    let d = direction(64);
    let inv = inv_norm(&d);
    let mut x = row(64);
    let before = unit_coefficient(&x, &d, inv);
    assert!(before.abs() > 0.1, "fixture must carry the direction");

    steer_in_place(&mut x, &d, SteeringMode::Ablate, 1.0, inv, 0.0, 0.0);

    let after = unit_coefficient(&x, &d, inv);
    assert!(
        after.abs() < 1e-4,
        "component survived ablation: {before} -> {after}"
    );
}

/// Ablation is a projection, so a second application must do nothing.
/// A sign error in `scale` passes the test above on the first pass and
/// fails here, because it would drive the coefficient to `-before`
/// rather than to zero and then oscillate.
#[test]
fn ablation_is_idempotent() {
    let d = direction(48);
    let inv = inv_norm(&d);
    let mut x = row(48);
    steer_in_place(&mut x, &d, SteeringMode::Ablate, 1.0, inv, 0.0, 0.0);
    let once = x.clone();
    steer_in_place(&mut x, &d, SteeringMode::Ablate, 1.0, inv, 0.0, 0.0);
    for (a, b) in once.iter().zip(x.iter()) {
        assert!((a - b).abs() < 1e-5, "second ablation moved the row");
    }
}

/// Clamp's defining property, and the one that separates it from `Add`:
/// it drives the coefficient to `target` REGARDLESS of what it was, where
/// `Add` moves it by a fixed amount. A fixture whose starting coefficient
/// happened to be near zero could not tell the two apart.
#[test]
fn clamp_reaches_the_target_from_either_side() {
    let d = direction(96);
    let inv = inv_norm(&d);
    for target in [2.5f32, -4.0] {
        let mut x = row(96);
        let before = unit_coefficient(&x, &d, inv);
        assert!(
            (before - target).abs() > 0.5,
            "fixture must start away from the target"
        );
        steer_in_place(&mut x, &d, SteeringMode::Clamp, 1.0, inv, target, 0.0);
        let after = unit_coefficient(&x, &d, inv);
        assert!(
            (after - target).abs() < 1e-3,
            "clamp missed: wanted {target}, got {after}"
        );
    }
}

/// `Add` reads the direction's MAGNITUDE where the other two do not.
/// Scaling `d` by two doubles what `Add` writes and leaves `Ablate`
/// alone, which is the asymmetry the file format forces.
#[test]
fn only_add_reads_the_direction_magnitude() {
    let d = direction(32);
    let scaled: Vec<f32> = d.iter().map(|v| v * 2.0).collect();

    let mut a = row(32);
    let mut b = row(32);
    steer_in_place(
        &mut a,
        &d,
        SteeringMode::Ablate,
        1.0,
        inv_norm(&d),
        0.0,
        0.0,
    );
    steer_in_place(
        &mut b,
        &scaled,
        SteeringMode::Ablate,
        1.0,
        inv_norm(&scaled),
        0.0,
        0.0,
    );
    for (x, y) in a.iter().zip(b.iter()) {
        assert!((x - y).abs() < 1e-4, "ablate moved with |d|");
    }

    let base = row(32);
    let mut c = base.clone();
    let mut e = base.clone();
    steer_in_place(&mut c, &d, SteeringMode::Add, 1.0, inv_norm(&d), 0.0, 0.0);
    steer_in_place(
        &mut e,
        &scaled,
        SteeringMode::Add,
        1.0,
        inv_norm(&scaled),
        0.0,
        0.0,
    );
    let moved_once = c[1] - base[1];
    let moved_twice = e[1] - base[1];
    assert!(
        (moved_twice - 2.0 * moved_once).abs() < 1e-4,
        "add did not scale with |d|"
    );
}

/// THE defining property of the norm-preserving mode -- and only half of
/// its characterization, because a NO-OP preserves the norm too. Pair it
/// with the ablation test below; neither alone pins the edit.
///
/// **IT SWEEPS ALPHA ON PURPOSE.** [`renorm_gamma`]'s coefficient is
/// `alpha * (2 - alpha)`, which is equal to `alpha` at exactly 1.0 -- so
/// a version of this written only at full strength cannot tell the
/// correct expression from a plain `alpha`. Full strength is also the
/// natural value to reach for here, since making it usable is the whole
/// point of the mode, which is what makes the blind spot likely rather
/// than theoretical. Same shape as the `ablate`-at-1 / `clamp`-at-0
/// degenerate point that let a shader mode-code swap pass its parity
/// case, one crate over.
#[test]
fn renorm_preserves_the_row_norm_at_every_strength() {
    let d = direction(96);
    let inv = inv_norm(&d);
    for alpha in [0.3f32, 0.6, 1.0] {
        // `row_carrying` and not `row`, for the same reason the
        // discrimination test below needs it: on a row that barely
        // carries the direction, ablation removes almost no norm, so
        // "the norm was preserved" is true of an implementation that
        // does nothing at all.
        let mut x = row_carrying(96);
        let before = l2(&x);
        steer_in_place(&mut x, &d, SteeringMode::Renorm, alpha, inv, 0.0, 0.0);
        let after = l2(&x);
        assert!(
            (after - before).abs() / before < 1.0e-5,
            "alpha {alpha}: norm moved {before} -> {after}"
        );
    }
}

/// The other half. Preserving the norm while leaving the component in
/// place is the IDENTITY, which passes the test above and is a different
/// edit entirely. Rescaling cannot reintroduce a component that was
/// projected out -- the row is scaled, not rotated -- so this must hold
/// at exactly the same tolerance plain ablation does.
#[test]
fn renorm_at_alpha_one_still_removes_the_whole_component() {
    let d = direction(64);
    let inv = inv_norm(&d);
    let mut x = row(64);
    let before = unit_coefficient(&x, &d, inv);
    assert!(before.abs() > 0.1, "fixture must carry the direction");

    steer_in_place(&mut x, &d, SteeringMode::Renorm, 1.0, inv, 0.0, 0.0);

    let after = unit_coefficient(&x, &d, inv);
    assert!(
        after.abs() < 1e-4,
        "component survived renorm: {before} -> {after}"
    );
}

/// `Renorm` and `Ablate` must be different functions, and **the fixture
/// has to be shown capable of seeing that before the comparison means
/// anything**: as the coefficient goes to zero there is no norm to
/// restore, `gamma` goes to 1, and the two edits converge. A tidy fixture
/// that barely carries the direction would pass this while proving
/// nothing (AGENTS.md Gotcha 48).
#[test]
fn renorm_and_ablate_are_different_edits() {
    let d = direction(64);
    let inv = inv_norm(&d);
    let probe = row_carrying(64);
    let xx: f32 = probe.iter().map(|v| v * v).sum();
    let gamma = renorm_gamma(xx, unit_coefficient(&probe, &d, inv), 1.0);
    assert!(
        gamma > 1.01,
        "fixture cannot discriminate: gamma is {gamma}, so renorm and ablate coincide on it"
    );

    let mut ablated = row_carrying(64);
    let mut renormed = row_carrying(64);
    steer_in_place(&mut ablated, &d, SteeringMode::Ablate, 1.0, inv, 0.0, 0.0);
    steer_in_place(&mut renormed, &d, SteeringMode::Renorm, 1.0, inv, 0.0, 0.0);
    assert!(
        ablated
            .iter()
            .zip(renormed.iter())
            .any(|(a, b)| (a - b).abs() > 1e-4),
        "renorm reproduced ablate's row"
    );
}

/// A row lying entirely ALONG `d` has nothing left after ablation, so
/// there is no norm to restore and `sqrt(xx / 0)` is the natural wrong
/// answer. It must come out as the identity: an infinity here reaches the
/// residual stream as NaN, and NaN is the one wrong value no downstream
/// rank or top-k instrument reports as wrong.
#[test]
fn renorm_on_a_row_along_the_direction_stays_finite() {
    let d = direction(48);
    let inv = inv_norm(&d);
    let mut x: Vec<f32> = d.iter().map(|v| v * 2.5).collect();
    let c = steer_in_place(&mut x, &d, SteeringMode::Renorm, 1.0, inv, 0.0, 0.0);
    assert!(c.is_finite(), "reported a non-finite coefficient");
    assert!(
        x.iter().all(|v| v.is_finite()),
        "wrote a non-finite row: {x:?}"
    );
}

/// `alpha == 0` must be the exact identity in every mode that scales by
/// it. This is the null control the GPU probe leans on: with the kernel
/// dispatched and `alpha` zero, the result must be bit-identical to not
/// dispatching it at all, which is what catches a wrong reduction, a
/// wrong offset and a wrong row stride in one comparison.
///
/// `Renorm` is in the loop and its exactness is not an accident of
/// tolerance: at `alpha == 0` the projection removes nothing, so
/// `denom == xx`, `xx / xx == 1.0` and `sqrt(1.0) == 1.0`, all exact.
#[test]
fn alpha_zero_is_the_identity() {
    let d = direction(80);
    let inv = inv_norm(&d);
    for mode in [
        SteeringMode::Ablate,
        SteeringMode::Add,
        SteeringMode::Renorm,
    ] {
        let before = row(80);
        let mut x = before.clone();
        steer_in_place(&mut x, &d, mode, 0.0, inv, 0.0, 0.0);
        assert_eq!(x, before, "{mode:?} at alpha=0 moved the row");
    }
}

/// A zero direction names no subspace. Every mode must leave the row
/// finite and unmoved rather than propagating the NaN that `1 / 0` would
/// produce -- the one wrong value no downstream instrument reports as
/// wrong.
#[test]
fn a_zero_direction_is_inert_rather_than_nan() {
    let d = vec![0.0f32; 40];
    let inv = inv_norm(&d);
    assert_eq!(inv, 0.0);
    for mode in [
        SteeringMode::Ablate,
        SteeringMode::Add,
        SteeringMode::Clamp,
        SteeringMode::Renorm,
    ] {
        let before = row(40);
        let mut x = before.clone();
        let c = steer_in_place(&mut x, &d, mode, 1.0, inv, 1.0, 0.0);
        assert!(c.is_finite(), "{mode:?} reported a non-finite coefficient");
        assert!(
            x.iter().all(|v| v.is_finite()),
            "{mode:?} wrote a non-finite row"
        );
    }
}

/// The gate blocks the edit and still reports, so a gated run remains
/// measurable.
#[test]
fn the_gate_blocks_the_edit_but_not_the_report() {
    let d = direction(64);
    let inv = inv_norm(&d);
    let before = row(64);
    let mut x = before.clone();
    let c = steer_in_place(&mut x, &d, SteeringMode::Ablate, 1.0, inv, 0.0, 1.0e6);
    assert_eq!(x, before, "gated edit still fired");
    assert!(c.abs() > 0.0, "gated edit reported no coefficient");
}
