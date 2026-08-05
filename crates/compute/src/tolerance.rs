//! Error-measurement helpers and the tolerance table, modeled on MLX
//! (`test_fast.py:80`). Ported from `Support/Tolerance/RelError.swift` and
//! `Support/Tolerance/Tolerances.swift`.

/// Standard relative error:
/// `max_i |actual[i] - reference[i]| / max(max_i |reference[i]|, 1e-6)`.
pub fn rel_error(actual: &[f32], reference: &[f32]) -> f32 {
    assert_eq!(actual.len(), reference.len(), "length mismatch");
    let max_abs_diff = max_abs_diff(actual, reference);
    let ref_norm = reference.iter().fold(0f32, |acc, r| acc.max(r.abs()));
    max_abs_diff / ref_norm.max(1e-6)
}

/// `max_i |a[i] - b[i]|`.
pub fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "length mismatch");
    a.iter()
        .zip(b.iter())
        .fold(0f32, |acc, (x, y)| acc.max((x - y).abs()))
}

/// Relative error with an absolute floor on the denominator, for reference
/// values small enough that a pure relative bound would exceed the
/// meaningful FP16 noise floor.
pub fn bounded_rel_error(actual: &[f32], reference: &[f32], abs_floor: f32) -> f32 {
    assert_eq!(actual.len(), reference.len(), "length mismatch");
    actual
        .iter()
        .zip(reference.iter())
        .fold(0f32, |worst, (a, r)| {
            let diff = (a - r).abs();
            let denom = r.abs().max(abs_floor);
            worst.max(diff / denom)
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
