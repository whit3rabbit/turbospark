//! L2 normalization and Matryoshka truncation.
//!
//! `l2_normalize` is `torch.nn.functional.normalize(p=2, eps=1e-12)`, the
//! sentence-transformers `Normalize` module: `x / max(||x||, eps)`. The
//! clamp is on the norm, not added to it, so a non-degenerate vector is
//! divided by its exact norm.

/// `F.normalize` default epsilon.
pub const L2_EPS: f32 = 1e-12;

/// Euclidean norm, accumulated in f64 so long vectors do not lose bits.
pub fn l2_norm(x: &[f32]) -> f32 {
    x.iter()
        .map(|&v| f64::from(v) * f64::from(v))
        .sum::<f64>()
        .sqrt() as f32
}

/// Normalizes `x` in place to unit L2 norm (`x / max(||x||, eps)`).
pub fn l2_normalize_eps(x: &mut [f32], eps: f32) {
    let denom = l2_norm(x).max(eps);
    for v in x.iter_mut() {
        *v /= denom;
    }
}

/// [`l2_normalize_eps`] with [`L2_EPS`].
pub fn l2_normalize(x: &mut [f32]) {
    l2_normalize_eps(x, L2_EPS);
}

/// Matryoshka truncation: keeps the first `dim` values and renormalizes,
/// which is how EmbeddingGemma's 512/256/128 outputs are produced. `dim`
/// larger than the vector leaves it unchanged apart from the renormalize.
pub fn truncate_and_normalize(x: &[f32], dim: usize) -> Vec<f32> {
    let mut out = x[..dim.min(x.len())].to_vec();
    l2_normalize(&mut out);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_to_unit_length() {
        let mut v = vec![3.0, 4.0];
        l2_normalize(&mut v);
        assert_eq!(v, vec![0.6, 0.8]);
    }

    #[test]
    fn zero_vector_stays_zero() {
        let mut v = vec![0.0, 0.0];
        l2_normalize(&mut v);
        assert_eq!(v, vec![0.0, 0.0]);
    }

    #[test]
    fn truncation_renormalizes() {
        let v = truncate_and_normalize(&[3.0, 4.0, 12.0], 2);
        assert_eq!(v, vec![0.6, 0.8]);
        assert!((l2_norm(&truncate_and_normalize(&[1.0, 2.0], 9)) - 1.0).abs() < 1e-6);
    }
}
