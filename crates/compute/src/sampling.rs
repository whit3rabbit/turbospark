//! FP32 reference for `softmax(softcap * tanh(x / softcap))`. Ported from
//! `Support/Reference/Sampling/LogitSoftcapSoftmax.swift`. Shares its spec
//! with the `selection` crate's shaping front-end.

/// Softcap is `c * tanh(x / c)`; default `c = 30.0` for Gemma 4.
pub fn logit_softcap_softmax(x: &[f32], softcap: f32) -> Vec<f32> {
    let inv_c = 1.0 / softcap;
    let mut y: Vec<f32> = x.iter().map(|&xv| (xv * inv_c).tanh() * softcap).collect();

    let mx = y.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    for v in y.iter_mut() {
        *v = (*v - mx).exp();
    }
    let sum: f32 = y.iter().sum();
    let inv_sum = 1.0 / sum;
    for v in y.iter_mut() {
        *v *= inv_sum;
    }
    y
}
