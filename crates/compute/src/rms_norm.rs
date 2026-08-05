//! FP32 RMSNorm reference. Ported from
//! `Support/Reference/Primitives/RmsNorm.swift`.

/// `y[i] = x[i] * weight[i] / sqrt(mean(x^2) + eps)`.
pub fn rms_norm(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    assert_eq!(x.len(), weight.len(), "x and weight must match length");
    let d = x.len();
    let sum_sq: f32 = x.iter().map(|v| v * v).sum();
    let inv_rms = 1.0 / (sum_sq / d as f32 + eps).sqrt();
    x.iter()
        .zip(weight.iter())
        .map(|(xv, wv)| xv * wv * inv_rms)
        .collect()
}
