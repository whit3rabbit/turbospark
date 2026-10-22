//! Cost curves. A batched kernel's time over M rows is close to affine,
//! `t(M) = a + b*M`: `a` is the dispatch and weight-read floor, `b` the
//! marginal cost of a row. That is the shape the speculative-verify c(M)
//! tables in `docs/SPECULATIVE_DECODING.md` describe, and the break-even M
//! between two arms falls out of two such lines.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Affine {
    /// Fixed cost, ms.
    pub a: f64,
    /// Per-row cost, ms.
    pub b: f64,
    /// Largest relative residual over the fitted points.
    pub max_rel_residual: f64,
}

impl Affine {
    pub fn at(&self, m: f64) -> f64 {
        self.a + self.b * m
    }
}

/// Least-squares line through `(m, ms)` points. `None` under two distinct M.
pub fn fit_affine(points: &[(f64, f64)]) -> Option<Affine> {
    let n = points.len() as f64;
    if points.len() < 2 {
        return None;
    }
    let mx = points.iter().map(|p| p.0).sum::<f64>() / n;
    let my = points.iter().map(|p| p.1).sum::<f64>() / n;
    let sxx: f64 = points.iter().map(|p| (p.0 - mx).powi(2)).sum();
    if sxx == 0.0 {
        return None;
    }
    let sxy: f64 = points.iter().map(|p| (p.0 - mx) * (p.1 - my)).sum();
    let b = sxy / sxx;
    let a = my - b * mx;
    let max_rel_residual = points
        .iter()
        .map(|&(m, t)| ((a + b * m) - t).abs() / t.abs().max(f64::MIN_POSITIVE))
        .fold(0.0, f64::max);
    Some(Affine {
        a,
        b,
        max_rel_residual,
    })
}

/// The M above which `candidate` is cheaper than `reference`, if the lines
/// cross at a positive M. `Some(0.0)` when it is cheaper everywhere.
pub fn break_even(reference: &Affine, candidate: &Affine) -> Option<f64> {
    let db = reference.b - candidate.b;
    let da = candidate.a - reference.a;
    if db < 0.0 {
        // The candidate's rows cost more: it can only win BELOW a crossing,
        // never above one.
        return None;
    }
    if db == 0.0 {
        // Parallel lines: cheaper everywhere or nowhere.
        return (da < 0.0).then_some(0.0);
    }
    Some((da / db).max(0.0))
}

/// c(M): the cost of M rows in one batched pass relative to M separate
/// one-row passes.
pub fn c_of_m(batched_ms: f64, one_row_ms: f64, m: usize) -> f64 {
    batched_ms / (one_row_ms * m as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recovers_a_line_and_its_crossing() {
        let gemv = fit_affine(&[(1.0, 1.0), (2.0, 2.0), (4.0, 4.0), (8.0, 8.0)]).unwrap();
        assert!((gemv.a).abs() < 1e-12 && (gemv.b - 1.0).abs() < 1e-12);
        let gemm = fit_affine(&[(1.0, 3.1), (2.0, 3.2), (4.0, 3.4), (8.0, 3.8)]).unwrap();
        assert!((gemm.a - 3.0).abs() < 1e-9 && (gemm.b - 0.1).abs() < 1e-9);
        let m = break_even(&gemv, &gemm).unwrap();
        assert!((m - 3.0 / 0.9).abs() < 1e-9, "{m}");
        assert_eq!(
            break_even(&gemm, &gemv),
            None,
            "gemv never beats gemm past the crossing"
        );
        assert!((c_of_m(3.8, 1.0, 8) - 0.475).abs() < 1e-12);
    }

    #[test]
    fn degenerate_inputs() {
        assert!(fit_affine(&[(1.0, 1.0)]).is_none());
        assert!(fit_affine(&[(2.0, 1.0), (2.0, 3.0)]).is_none());
    }
}
