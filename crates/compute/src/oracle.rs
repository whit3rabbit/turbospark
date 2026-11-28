//! FP64 references and the pass rule for kernel oracles (KERN-13,
//! `docs/designs/speedup-kern.md` Step 2).
//!
//! A parity test that compares a new kernel with the incumbent only checks
//! that the two agree, and a bit-equality bench rejects every reordering. An
//! oracle compares both with a reference computed in FP64 from the same
//! inputs, and accepts a candidate whose error is within
//! [`oracle_tolerance`] of the incumbent's own. Inputs are the values the GPU
//! actually reads (already rounded to `half` or `bfloat`), widened to f64, so
//! the reference carries no storage rounding of its own.

use crate::gdn::GdnDims;

/// Error of `got` against an FP64 reference `want`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ErrorStats {
    /// Largest `|got - want|`.
    pub max_abs: f64,
    /// Root mean square of `got - want`.
    pub rms: f64,
    /// Largest `|want|`, the scale `max_abs` is read against.
    pub max_ref: f64,
}

/// `got` (f32, as read back from the GPU) against `want`. Panics on a length
/// mismatch or an empty input: either is a harness bug, not an error value.
pub fn error_stats(got: &[f32], want: &[f64]) -> ErrorStats {
    assert_eq!(got.len(), want.len(), "oracle compares equal lengths");
    assert!(!want.is_empty(), "oracle needs at least one value");
    let mut max_abs = 0.0f64;
    let mut sum_sq = 0.0f64;
    let mut max_ref = 0.0f64;
    for (&g, &w) in got.iter().zip(want) {
        let e = (g as f64 - w).abs();
        // NaN in the candidate must fail, never compare as small.
        max_abs = if e.is_nan() {
            f64::INFINITY
        } else {
            max_abs.max(e)
        };
        sum_sq += e * e;
        max_ref = max_ref.max(w.abs());
    }
    ErrorStats {
        max_abs,
        rms: (sum_sq / want.len() as f64).sqrt(),
        max_ref,
    }
}

/// The design's pass bound: `max(floor, 3 x the incumbent's error)`. The
/// floor keeps an incumbent that happens to be exact on a fixture from
/// making every honest reordering fail.
pub fn oracle_tolerance(floor: f64, incumbent_max_abs: f64) -> f64 {
    floor.max(3.0 * incumbent_max_abs)
}

/// One gated-delta decode step in FP64, the math of `gdn_delta_step_decode`
/// in `gdn.metal` (see `crate::gdn` for the recurrence). `conv` is one
/// `[q: Hk*Dk][k: Hk*Dk][v: Hv*Dv]` row whose q and k are already normed and
/// scaled; `a`, `b`, `a_log`, `dt_bias` are `[Hv]`. Advances `state`
/// (`[Hv, Dv, Dk]`) and returns `y` (`[Hv * Dv]`), unrounded.
pub fn gdn_delta_step_f64(
    d: GdnDims,
    conv: &[f64],
    a: &[f64],
    b: &[f64],
    a_log: &[f64],
    dt_bias: &[f64],
    state: &mut [f64],
) -> Vec<f64> {
    let (hk, hv) = (d.num_k_heads, d.num_v_heads);
    let (dk, dv) = (d.key_head_dim, d.value_head_dim);
    assert_eq!(conv.len(), d.qkv_dim());
    assert_eq!(a.len(), hv);
    assert_eq!(b.len(), hv);
    assert_eq!(a_log.len(), hv);
    assert_eq!(dt_bias.len(), hv);
    assert_eq!(state.len(), hv * dv * dk);
    assert!(hk > 0 && hv % hk == 0);
    let softplus = |x: f64| if x > 20.0 { x } else { x.exp().ln_1p() };
    let mut y = vec![0.0f64; hv * dv];
    for head in 0..hv {
        let hk_index = head / (hv / hk);
        let q = &conv[hk_index * dk..][..dk];
        let k = &conv[hk * dk + hk_index * dk..][..dk];
        let v = &conv[2 * hk * dk + head * dv..][..dv];
        let g = (-a_log[head].exp() * softplus(a[head] + dt_bias[head])).exp();
        let beta = 1.0 / (1.0 + (-b[head]).exp());
        for row in 0..dv {
            let s = &mut state[(head * dv + row) * dk..][..dk];
            let mut kv = 0.0;
            for (si, &ki) in s.iter_mut().zip(k) {
                *si *= g;
                kv += *si * ki;
            }
            let delta = (v[row] - kv) * beta;
            let mut out = 0.0;
            for ((si, &ki), &qi) in s.iter_mut().zip(k).zip(q) {
                *si += ki * delta;
                out += *si * qi;
            }
            y[head * dv + row] = out;
        }
    }
    y
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dims(hk: usize, hv: usize, dk: usize, dv: usize) -> GdnDims {
        GdnDims {
            num_k_heads: hk,
            num_v_heads: hv,
            key_head_dim: dk,
            value_head_dim: dv,
            conv_kernel_size: 4,
        }
    }

    #[test]
    fn one_head_step_matches_the_hand_computed_recurrence() {
        // Hk = Hv = 1, Dk = 2, Dv = 1. q = (1, 0), k = (0.5, 0.5), v = 2.
        let d = dims(1, 1, 2, 1);
        let conv = [1.0, 0.0, 0.5, 0.5, 2.0];
        // a_log = ln(ln 2) and softplus(a + dt) = 1 make g = exp(-ln 2) = 0.5;
        // b = 0 makes beta = 0.5.
        let sp_inv_1 = (1f64.exp() - 1.0).ln();
        let a_log = [2f64.ln().ln()];
        let mut state = vec![0.4, 0.8];
        let y = gdn_delta_step_f64(d, &conv, &[sp_inv_1], &[0.0], &a_log, &[0.0], &mut state);
        // S = (0.2, 0.4); kv = 0.3; delta = (2 - 0.3) * 0.5 = 0.85;
        // S += 0.85 * k = (0.625, 0.825); y = S . q = 0.625.
        assert!((state[0] - 0.625).abs() < 1e-12, "{state:?}");
        assert!((state[1] - 0.825).abs() < 1e-12, "{state:?}");
        assert!((y[0] - 0.625).abs() < 1e-12, "{y:?}");
    }

    #[test]
    fn value_heads_share_their_key_head() {
        // Hv = 2 over Hk = 1: both value heads read the same q and k.
        let d = dims(1, 2, 1, 1);
        let conv = [1.0, 1.0, 3.0, 5.0];
        let mut state = vec![0.0, 0.0];
        let big = 50.0; // beta = sigmoid(50) = 1 to f64 precision
        let y = gdn_delta_step_f64(
            d,
            &conv,
            &[0.0; 2],
            &[big; 2],
            &[-50.0; 2],
            &[0.0; 2],
            &mut state,
        );
        // g = exp(-exp(-50) * ln 2) = 1 to f64 precision; S = v; y = v.
        assert!(
            (y[0] - 3.0).abs() < 1e-9 && (y[1] - 5.0).abs() < 1e-9,
            "{y:?}"
        );
    }

    #[test]
    fn error_stats_and_tolerance() {
        let s = error_stats(&[1.0, 2.5, -1.0], &[1.0, 2.0, -1.5]);
        assert_eq!(s.max_abs, 0.5);
        assert!((s.rms - (0.5f64 / 3.0).sqrt()).abs() < 1e-12);
        assert_eq!(s.max_ref, 2.0);
        assert_eq!(oracle_tolerance(1e-3, 1e-4), 1e-3);
        assert_eq!(oracle_tolerance(1e-3, 1e-2), 3e-2);
    }

    #[test]
    fn a_nan_candidate_never_passes() {
        let s = error_stats(&[f32::NAN, 1.0], &[0.0, 1.0]);
        assert!(s.max_abs.is_infinite());
    }
}
