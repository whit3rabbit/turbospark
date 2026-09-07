//! Error-measurement helpers and the tolerance table, modeled on MLX
//! (`test_fast.py:80`). Ported from `Support/Tolerance/RelError.swift` and
//! `Support/Tolerance/Tolerances.swift`.

/// A `f32::max`-style fold step that is NaN-sticky: if either side is
/// already NaN, the result is NaN, rather than `f32::max`'s own behaviour of
/// returning the other (non-NaN) argument.
///
/// This is why the three functions below fold with `worst` instead of a bare
/// `.max()`. An all-NaN GPU parity output would otherwise fold down to
/// `0.0`, which passes `err < tol` and reads as a perfect result
/// (AGENTS.md Gotcha 59: NaN reads as a perfect score on an unguarded
/// comparison, one level down at the instrument that every parity test in
/// `crates/gpu` reads through). `+inf` is unaffected: `inf.max(x)` is
/// already `inf` for any finite `x`, so it propagates through the ordinary
/// `f32::max` path with no special case needed.
#[inline]
fn worst(acc: f32, d: f32) -> f32 {
    if acc.is_nan() || d.is_nan() {
        f32::NAN
    } else {
        acc.max(d)
    }
}

/// Standard relative error:
/// `max_i |actual[i] - reference[i]| / max(max_i |reference[i]|, 1e-6)`.
///
/// NaN-sticky in `actual` through [`max_abs_diff`]: `NaN / x` is `NaN` for
/// any `x`, so a NaN `max_abs_diff` reaches the caller as `NaN` regardless
/// of `ref_norm`.
pub fn rel_error(actual: &[f32], reference: &[f32]) -> f32 {
    assert_eq!(actual.len(), reference.len(), "length mismatch");
    let max_abs_diff = max_abs_diff(actual, reference);
    let ref_norm = reference.iter().fold(0f32, |acc, r| acc.max(r.abs()));
    max_abs_diff / ref_norm.max(1e-6)
}

/// `max_i |a[i] - b[i]|`. NaN-sticky: a NaN anywhere in either slice makes
/// the result NaN rather than the largest finite difference elsewhere.
pub fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "length mismatch");
    a.iter()
        .zip(b.iter())
        .fold(0f32, |acc, (x, y)| worst(acc, (x - y).abs()))
}

/// Relative error with an absolute floor on the denominator, for reference
/// values small enough that a pure relative bound would exceed the
/// meaningful FP16 noise floor.
///
/// NaN-sticky: a NaN `actual[i]` makes `diff` (and so the folded result)
/// NaN, even though `denom` alone would read as finite (`r.abs().max(floor)`
/// is not NaN-sticky, since `denom` on its own is not the value under test).
pub fn bounded_rel_error(actual: &[f32], reference: &[f32], abs_floor: f32) -> f32 {
    assert_eq!(actual.len(), reference.len(), "length mismatch");
    actual
        .iter()
        .zip(reference.iter())
        .fold(0f32, |acc, (a, r)| {
            let diff = (a - r).abs();
            let denom = r.abs().max(abs_floor);
            worst(acc, diff / denom)
        })
}

/// Relative-error tolerance bars against an FP32 reference, sized to the
/// accumulation depth and dtype of the kernel under test.
pub struct Tolerance;

impl Tolerance {
    /// FP32 identity / pass-through. Bit-exact-ish.
    pub const IDENTITY: f32 = 1e-5;
    /// FP16 single-reduction kernels (RMSNorm, one GEMV, short-axis softmax).
    pub const FP16_REDUCTION: f32 = 5e-3;
    /// FP16 chained reductions: multi-stage MoE, attention's softmax+matmul
    /// composition, every block-level fusion.
    pub const FP16_CHAINED_REDUCTION: f32 = 1e-2;
    /// Quantization-aware comparisons; callers usually override with a
    /// mathematically derived bound (e.g. `|w - w_hat| <= |scales|`).
    pub const QUANT_INT4: f32 = 1.5e-3;
    pub const QUANT_INT8: f32 = 1e-3;
}
