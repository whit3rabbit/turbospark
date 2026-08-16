/// Pearson correlation of two equal-length vectors.
///
/// Lives here rather than in a test because settling `FUSED_GATE_FIRST` needs
/// it (comparing a GGUF-dequantized expert against the same expert in the
/// MLX-derived install: two different quantizations of the same trained
/// weights, so they agree in direction and not to the bit), and because a
/// future quant kernel is more likely to be judged by correlation than by an
/// element-wise tolerance. Returns 0.0 when either side is constant, which is
/// the answer that makes a caller's threshold behave: an undefined
/// correlation must not read as a match.
pub fn pearson(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "correlation needs equal lengths");
    assert!(!a.is_empty());
    let n = a.len() as f64;
    let mean_a = a.iter().map(|&v| v as f64).sum::<f64>() / n;
    let mean_b = b.iter().map(|&v| v as f64).sum::<f64>() / n;
    let (mut cov, mut va, mut vb) = (0f64, 0f64, 0f64);
    for (&x, &y) in a.iter().zip(b.iter()) {
        let (dx, dy) = (x as f64 - mean_a, y as f64 - mean_b);
        cov += dx * dy;
        va += dx * dx;
        vb += dy * dy;
    }
    if va == 0.0 || vb == 0.0 {
        return 0.0;
    }
    (cov / (va.sqrt() * vb.sqrt())) as f32
}
