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
#[path = "steering_tests.rs"]
mod steering_tests;
