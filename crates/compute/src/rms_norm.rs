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

/// `y[i] = x[i] * (1 + weight[i]) / sqrt(mean(x^2) + eps)` -- the CENTERED
/// form, whose checkpoint weight is an OFFSET from unity rather than the
/// scale itself.
///
/// The contract for `rmsnorm_bf16w_centered`, and PORT-LOCAL: the Swift
/// engine has no architecture that uses this convention, so there is no
/// upstream kernel to diff against and this function is the only definition
/// of what that one computes.
///
/// **Which convention a checkpoint uses is a property of the MODEL and not
/// of the family, and one model can use both.** `muse_glimmer` is the first
/// family here to need this: its four per-layer norms are
/// `mlx_vlm.models.muse_glimmer.language.CenteredRMSNorm` (`x * (1 + w)`)
/// while its FINAL norm is a plain `nn.RMSNorm` (`x * w`), so the two live
/// side by side in one forward pass. Gemma's `+1` was a real candidate and
/// was REFUTED by measurement (its GGUF and MLX installs' resident cores are
/// bit-identical), so [`rms_norm`] stays the plain form and this is an
/// addition rather than a switch.
///
/// **The `1 +` is applied in FP32 on the accumulator, never baked into the
/// stored weight.** Baking it at repack time is the obvious alternative and
/// it is lossy in a way that matters: these weights are centred at zero, and
/// BF16 has 8 mantissa bits, so near 1.0 its absolute resolution is 2^-8 =
/// 0.0039. A weight of 0.01 -- an ordinary value for a centred norm -- would
/// come back with roughly 39% of its own magnitude destroyed, on the SCALE of
/// a normalization, which is a quality regression no coherence smoke can see.
pub fn rms_norm_centered(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    assert_eq!(x.len(), weight.len(), "x and weight must match length");
    let d = x.len();
    let sum_sq: f32 = x.iter().map(|v| v * v).sum();
    let inv_rms = 1.0 / (sum_sq / d as f32 + eps).sqrt();
    x.iter()
        .zip(weight.iter())
        .map(|(xv, wv)| xv * inv_rms * (1.0 + wv))
        .collect()
}
