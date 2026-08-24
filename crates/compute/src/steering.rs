//! FP32 reference for the directional-steering edit on a residual stream
//! row. PORT-LOCAL: the Swift engine has no steering surface, so there is no
//! upstream kernel to diff against and this module is the only definition of
//! what `steer_direction_fp16` computes.
//!
//! # The edit
//!
//! Given a residual row `x` and a direction `d` of the same length, all four
//! modes are one reduction pass plus one strided write. Writing `c = d . x`
//! and `c_hat = c / ||d||` for the coefficient along the unit direction
//! `d_hat`:
//!
//! ```text
//! Ablate:  x -= alpha * c_hat * d_hat
//! Add:     x += alpha * d
//! Clamp:   x += (target - c_hat) * d_hat
//! Renorm:  Ablate, then scale the whole row back to its original ||x||
//! ```
//!
//! # `Renorm` needs no second pass, and that is not an optimization
//!
//! It looks like it needs two extra reductions -- `||x||` before the edit and
//! `||x'||` after -- and it needs neither.
//!
//! The FIRST fuses into the loop that already computes `c`: `x[i]` is in
//! register there, so `||x||^2` costs one extra multiply-add per element and
//! no extra memory traffic at all.
//!
//! The SECOND is analytic, because ablation is an orthogonal projection. With
//! `x' = x - alpha * c_hat * d_hat` and `d_hat . x = c_hat`:
//!
//! ```text
//! ||x'||^2 = ||x||^2 - 2*alpha*c_hat^2 + alpha^2*c_hat^2
//!          = ||x||^2 - alpha*(2 - alpha)*c_hat^2
//! ```
//!
//! At `alpha == 1` that is Pythagoras. So the post-edit norm is available
//! from two scalars the reduction already produced, and the kernel stays at
//! ONE reduction pass and ONE write pass -- which is the cost that matters,
//! since this dispatches once per steered layer per token and the shader is
//! dispatch-bound rather than bandwidth-bound.
//!
//! **Do not "simplify" this into a second reduction over the written row.**
//! It would double the loop count for a quantity that is already known.
//!
//! `Ablate` at `alpha == 1.0` is exactly `x' = x - d_hat d_hat^T x`, the
//! operation that the weight edit `W' = W - d_hat d_hat^T W` precomputes: the
//! two are the same function, so applying it here rather than at repack time
//! gives up no fidelity. It gains some, on this engine: every install here is
//! quantized, so a weight edit means dequantize, edit, requantize, and
//! `crates/bench/tests/quality_sensitivity.rs` measured that damaging 0.0122%
//! of routed-expert bytes moves perplexity +10.5%. This runs in FP32 on the
//! accumulator and writes no weight byte.
//!
//! # Why `||d||` is a parameter and not computed here
//!
//! Every function below takes `inv_norm = 1 / ||d||` rather than deriving it.
//! It is a property of the DIRECTION, constant across every layer, every
//! token and every generation, so recomputing it inside a per-token kernel
//! would be a reduction over `d` on every dispatch to learn something the
//! loader already knows. The reference takes it by the same route so that the
//! two sides compute the same expression rather than agreeing by luck.
//!
//! Passing `inv_norm` rather than pre-normalizing `d` is what lets one buffer
//! serve all four modes: `Add` reads `d`'s magnitude (a llama.cpp control
//! vector carries its strength IN the vector) while `Ablate` and `Clamp` are
//! defined against the unit direction.
//!
//! # The FP16 hazard, which is real and not theoretical
//!
//! The GPU stores the residual stream as FP16, ceiling 65,504. [`Ablate`]
//! removes a component of `x` and cannot overflow; [`Renorm`] cannot either,
//! because it restores a magnitude the row already had (see
//! `SteeringMode::can_grow`). [`Add`] and [`Clamp`] can,
//! and this engine has met that exact failure before: the DFlash2 drafter's
//! residual peaks at 113,920 and needed a power-of-two scale to fit
//! (AGENTS.md Gotcha 60). The overflow arrives as `inf`, then as NaN, and NaN
//! reads as a PERFECT score on any rank or top-k instrument (Gotcha 59). This
//! reference is FP32 throughout and so cannot reproduce it; the finiteness
//! assertion belongs to the caller, at the point a measurement is taken.
//!
//! [`Ablate`]: foundation::SteeringMode::Ablate
//! [`Add`]: foundation::SteeringMode::Add
//! [`Clamp`]: foundation::SteeringMode::Clamp
//! [`Renorm`]: foundation::SteeringMode::Renorm

use foundation::SteeringMode;

/// `1 / ||d||` for a direction, the `inv_norm` every function here takes.
///
/// Returns `0.0` for a zero direction rather than an infinity, which makes
/// every mode below the identity on such a direction. That is the honest
/// degenerate answer: a direction with no magnitude names no subspace, so
/// there is nothing to project out, add, or clamp to. It also keeps the
/// failure finite, which matters because the alternative propagates a NaN
/// into the residual stream and NaN is the one wrong value no downstream
/// instrument reports as wrong.
pub fn inv_norm(d: &[f32]) -> f32 {
    let sum_sq: f32 = d.iter().map(|v| v * v).sum();
    if sum_sq > 0.0 {
        1.0 / sum_sq.sqrt()
    } else {
        0.0
    }
}

/// The raw dot product `c = d . x`.
///
/// Summed in the slice's own order. The GPU reduces in a different order (a
/// two-stage `simd_sum` over a strided read), and FP addition is not
/// associative, so the parity test compares within a tolerance rather than
/// bitwise -- unlike [`crate::rms_norm`]'s, whose contract is the same shape
/// but whose test fixtures are small enough to sum exactly.
pub fn direction_coefficient(x: &[f32], d: &[f32]) -> f32 {
    assert_eq!(x.len(), d.len(), "x and d must match length");
    x.iter().zip(d.iter()).map(|(xv, dv)| xv * dv).sum()
}

/// The coefficient along the UNIT direction, `c_hat = (d . x) / ||d||`.
///
/// This is the number worth reporting rather than the raw dot product: it is
/// comparable across directions and across layers, where `c` scales with
/// whatever magnitude the extraction happened to produce. It is what the
/// kernel writes to its coefficient buffer, and it is the whole measurement
/// story here -- it says how much of the direction the stream carried at each
/// layer, which is the "compare edited against unedited" signal without a
/// second model to compare against.
pub fn unit_coefficient(x: &[f32], d: &[f32], inv_norm: f32) -> f32 {
    direction_coefficient(x, d) * inv_norm
}

/// Applies one steering edit to `x` in place, returning the unit coefficient
/// [`unit_coefficient`] would have reported for the row BEFORE the edit.
///
/// The pre-edit coefficient is the useful one and is also the only one
/// available: `Ablate` drives the post-edit coefficient to `(1 - alpha) *
/// c_hat` and `Clamp` drives it to `target` by construction, so reporting
/// after the write would measure the parameters rather than the model.
///
/// `gate_threshold` is a magnitude below which the edit does not fire.
/// A non-positive threshold fires always. **The gate is evaluated here, on
/// the coefficient, and never by a caller reading it back**: on the GPU a
/// host-side gate would cost a command-buffer synchronization per layer per
/// token, which is the cost structure the whole decode path is built to
/// avoid. The coefficient is still reported when the gate blocks, so a gated
/// run can still be measured.
pub fn steer_in_place(
    x: &mut [f32],
    d: &[f32],
    mode: SteeringMode,
    alpha: f32,
    inv_norm: f32,
    target: f32,
    gate_threshold: f32,
) -> f32 {
    assert_eq!(x.len(), d.len(), "x and d must match length");

    // BOTH reductions in one pass, which is what the kernel does and is why
    // they are written together here rather than as two calls: `x[i]` is
    // already in register there, so `||x||^2` costs one extra multiply-add
    // per element and no extra memory traffic. `Renorm` is the only mode
    // that reads `xx`; computing it unconditionally keeps the two sides the
    // same expression instead of two that agree by luck.
    let mut c = 0.0f32;
    let mut xx = 0.0f32;
    for (xv, dv) in x.iter().zip(d.iter()) {
        c += xv * dv;
        xx += xv * xv;
    }
    let c_hat = c * inv_norm;

    if gate_threshold > 0.0 && c_hat.abs() < gate_threshold {
        return c_hat;
    }

    // Every mode reduces to one scalar multiplying `d`, which is why one
    // kernel serves all four. `Ablate`, `Renorm` and `Clamp` carry a second
    // factor of `inv_norm` because they are defined against the unit
    // direction and are written here in terms of the stored one:
    // `c_hat * d_hat` is `c * inv_norm * d * inv_norm`.
    let scale = match mode {
        SteeringMode::Ablate | SteeringMode::Renorm => -alpha * c * inv_norm * inv_norm,
        SteeringMode::Add => alpha,
        SteeringMode::Clamp => (target - c_hat) * inv_norm,
    };

    // **`gamma` IS EXACTLY 1.0 IN EVERY MODE BUT `Renorm`, AND `1.0 * v == v`
    // EXACTLY IN IEEE-754** for every finite `v`, signed zeros included. That
    // identity is what lets one write loop serve all four modes while leaving
    // the three older ones unchanged to the last bit -- no branch, no second
    // loop, no new pipeline. It is a claim about arithmetic rather than about
    // tolerance, and the real-model null control (`docs/OBLITERATION.md`)
    // checks it rather than assuming it.
    let gamma = match mode {
        SteeringMode::Renorm => renorm_gamma(xx, c_hat, alpha),
        _ => 1.0,
    };

    for (xv, dv) in x.iter_mut().zip(d.iter()) {
        *xv = gamma * (*xv + scale * dv);
    }
    c_hat
}

/// The factor [`SteeringMode::Renorm`] multiplies the projected row by to
/// restore the norm ablation removed, given `xx = ||x||^2` and the pre-edit
/// unit coefficient.
///
/// `||x'||^2 = xx - alpha * (2 - alpha) * c_hat^2` exactly (see the module
/// docs), so this is `sqrt(xx / ||x'||^2)` and needs no second reduction.
///
/// **The coefficient is `alpha * (2 - alpha)` and NOT `alpha`, and the two
/// are equal at `alpha == 1`.** Any test of this function written only at
/// full strength is therefore blind to the difference -- the same degenerate
/// point that let a mutation swapping `ablate` and `clamp` pass the GPU
/// parity case, one file over. Test it at a fractional alpha.
///
/// Returns `1.0` -- the identity, not an infinity -- when there is no norm
/// left to restore. That happens when the row lies entirely along `d`, and it
/// is [`inv_norm`]'s decision on a zero direction for the same reason: an
/// infinity here becomes a NaN in the stream, and NaN reads as a PERFECT
/// score on any rank instrument (AGENTS.md Gotcha 59). There is deliberately
/// no epsilon band around zero, which would be a fabricated threshold
/// (Gotcha 38): at any magnitude this runs at, `denom` cannot reach a
/// denormal without `c_hat^2` matching `xx` to ~43 significant FP32 digits.
pub fn renorm_gamma(xx: f32, c_hat: f32, alpha: f32) -> f32 {
    let denom = xx - alpha * (2.0 - alpha) * c_hat * c_hat;
    if denom <= 0.0 {
        return 1.0;
    }
    let gamma = (xx / denom).sqrt();
    if gamma.is_finite() {
        gamma
    } else {
        1.0
    }
}

#[cfg(test)]
mod tests {
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
}
