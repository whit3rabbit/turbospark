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

/// One matmul the Neural Engine gate decides on: its size, its traffic, and
/// what each device measured (`kernel::ane_matmul`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GateCase {
    /// Multiply-accumulates, `rows * cols * tokens`.
    pub macs: f64,
    /// MACs per estimated byte moved, conversion traffic included.
    pub intensity: f64,
    pub steel_ms: f64,
    pub ane_ms: f64,
    pub steel_j: Option<f64>,
    pub ane_j: Option<f64>,
}

/// What the gate fit minimizes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GateObjective {
    Speed,
    Energy,
    /// Energy-delay product.
    Edp,
}

/// A per-matmul routing rule of the `GGML_ANE_MIN_MACS` and
/// `GGML_ANE_MIN_INTENSITY` kind: send a matmul to the ANE only at or above
/// both thresholds.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GateFit {
    pub objective: GateObjective,
    pub min_macs: f64,
    pub min_intensity: f64,
    /// Cases the rule routes to the ANE, and how many of those lose there.
    pub selected: usize,
    pub regressions: usize,
    /// The weighted sum the search minimized (ms, J or J*ms): gains count
    /// once, regressions twice.
    pub score: f64,
    /// Every case on Steel, every case on its cheaper device, and the rule.
    pub steel_total: f64,
    pub oracle_total: f64,
    pub routed_total: f64,
}

/// Fits the two thresholds over the observed values (and 0, and one past
/// the largest) so that the selected cases' summed delta against Steel is
/// smallest, with a regression weighing twice a gain. `None` without cases,
/// or when the objective needs joules a case lacks.
pub fn ane_gate_fit(cases: &[GateCase], objective: GateObjective) -> Option<GateFit> {
    if cases.is_empty() {
        return None;
    }
    let cost = |c: &GateCase, ane: bool| -> Option<f64> {
        let (ms, j) = if ane {
            (c.ane_ms, c.ane_j)
        } else {
            (c.steel_ms, c.steel_j)
        };
        match objective {
            GateObjective::Speed => Some(ms),
            GateObjective::Energy => j,
            GateObjective::Edp => j.map(|j| j * ms),
        }
    };
    let deltas: Vec<f64> = cases
        .iter()
        .map(|c| Some(cost(c, true)? - cost(c, false)?))
        .collect::<Option<_>>()?;
    let candidates = |f: fn(&GateCase) -> f64| -> Vec<f64> {
        let mut v: Vec<f64> = cases.iter().map(f).collect();
        v.push(0.0);
        let max = v.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        v.push(max * (1.0 + 1e-9) + f64::MIN_POSITIVE);
        v.sort_by(|a, b| a.total_cmp(b));
        v.dedup();
        v
    };
    let mut best: Option<GateFit> = None;
    for &min_macs in &candidates(|c| c.macs) {
        for &min_intensity in &candidates(|c| c.intensity) {
            let selected: Vec<usize> = (0..cases.len())
                .filter(|&i| cases[i].macs >= min_macs && cases[i].intensity >= min_intensity)
                .collect();
            let score: f64 = selected
                .iter()
                .map(|&i| {
                    if deltas[i] < 0.0 {
                        deltas[i]
                    } else {
                        2.0 * deltas[i]
                    }
                })
                .sum();
            // `<=`: thresholds rise through the search, so a tie keeps the
            // stricter rule. A case the ANE does not beat (delta 0) adds
            // nothing to the score and must not be routed to it.
            if best.as_ref().is_none_or(|b| score <= b.score) {
                let steel_total: f64 = cases.iter().filter_map(|c| cost(c, false)).sum();
                best = Some(GateFit {
                    objective,
                    min_macs,
                    min_intensity,
                    selected: selected.len(),
                    regressions: selected.iter().filter(|&&i| deltas[i] > 0.0).count(),
                    score,
                    steel_total,
                    oracle_total: steel_total + deltas.iter().filter(|d| **d < 0.0).sum::<f64>(),
                    routed_total: steel_total + selected.iter().map(|&i| deltas[i]).sum::<f64>(),
                });
            }
        }
    }
    best
}

#[cfg(test)]
mod gate_tests {
    use super::*;

    fn case(macs: f64, intensity: f64, steel_ms: f64, ane_ms: f64) -> GateCase {
        GateCase {
            macs,
            intensity,
            steel_ms,
            ane_ms,
            steel_j: Some(steel_ms * 0.05),
            ane_j: Some(ane_ms * 0.01),
        }
    }

    #[test]
    fn picks_the_thresholds_that_keep_only_the_winners() {
        // Small matmuls lose on the ANE, big ones win; one big low-intensity
        // case loses, so the rule needs both thresholds.
        let cases = [
            case(1e8, 10.0, 0.1, 0.9),
            case(1e9, 30.0, 1.0, 1.5),
            case(1e10, 60.0, 10.0, 6.0),
            case(2e10, 80.0, 20.0, 12.0),
            case(3e10, 20.0, 30.0, 35.0),
        ];
        let fit = ane_gate_fit(&cases, GateObjective::Speed).unwrap();
        assert_eq!(fit.selected, 2, "{fit:?}");
        assert_eq!(fit.regressions, 0);
        // Whatever thresholds the search lands on, the rule must route the
        // two winners and nothing else: the low-intensity big case stays out.
        let routed: Vec<usize> = (0..cases.len())
            .filter(|&i| cases[i].macs >= fit.min_macs && cases[i].intensity >= fit.min_intensity)
            .collect();
        assert_eq!(routed, vec![2, 3], "{fit:?}");
        assert!((fit.score - (-12.0)).abs() < 1e-9);
        assert!((fit.steel_total - 61.1).abs() < 1e-9);
        assert!((fit.routed_total - 49.1).abs() < 1e-9);
        assert!((fit.oracle_total - 49.1).abs() < 1e-9);
        // Energy (0.01 J/ms on the ANE against 0.05 on Steel) favors the ANE
        // everywhere but the smallest case, where its slowness outweighs it.
        let e = ane_gate_fit(&cases, GateObjective::Energy).unwrap();
        assert_eq!(e.selected, 4, "{e:?}");
        assert_eq!(e.regressions, 0);
        assert!(cases[0].macs < e.min_macs || cases[0].intensity < e.min_intensity);
    }

    #[test]
    fn a_regression_costs_twice_a_gain() {
        // A small loser costing 0.6 ms below a big winner saving 1 ms, at
        // one intensity: a plain sum would route both (-0.4), the doubled
        // penalty routes only the winner (-1.0 against -1.0 + 1.2).
        let cases = [case(1e9, 50.0, 2.0, 2.6), case(2e9, 50.0, 2.0, 1.0)];
        let fit = ane_gate_fit(&cases, GateObjective::Speed).unwrap();
        assert_eq!(fit.selected, 1, "{fit:?}");
        assert_eq!(fit.regressions, 0);
        assert!(
            fit.min_macs > 1e9 && fit.min_macs <= 2e9,
            "{}",
            fit.min_macs
        );
        assert!((fit.score + 1.0).abs() < 1e-9);
        // Without joules the energy objective cannot be fitted.
        let dry = [GateCase {
            steel_j: None,
            ane_j: None,
            ..cases[0]
        }];
        assert!(ane_gate_fit(&dry, GateObjective::Energy).is_none());
        assert!(ane_gate_fit(&dry, GateObjective::Speed).is_some());
        assert!(ane_gate_fit(&[], GateObjective::Speed).is_none());
    }

    #[test]
    fn a_tie_keeps_the_stricter_rule() {
        // The small case is a dead heat, the big one a win: routing both or
        // only the winner scores the same, so the tie must not route the
        // dead heat.
        let cases = [case(1e9, 50.0, 2.0, 2.0), case(2e9, 50.0, 2.0, 1.0)];
        let fit = ane_gate_fit(&cases, GateObjective::Speed).unwrap();
        assert_eq!(fit.selected, 1, "{fit:?}");
        assert!(fit.min_macs > 1e9, "{fit:?}");
    }
}
