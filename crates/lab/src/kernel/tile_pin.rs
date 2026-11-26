//! The no-harm-per-bucket rule for pinning a kernel layout from a sweep
//! (evaluator item LAB-27).
//!
//! A layout that wins on average and loses at one point is not pinned: for a
//! bucket of the swept variable, a candidate is accepted only if it is no
//! slower than the baseline at **every** swept point of the bucket and faster
//! over the bucket by more than a minimum gain. Among accepted candidates the
//! one with the smallest worst-case regret wins. Nothing accepted means the
//! baseline stays, which is the safe answer for a noisy sweep.
//!
//! A pin is a reading of one chip, keyed by it, and of medians without a
//! noise estimate: it proposes a setting, it is not a verdict, and it says
//! nothing about arithmetic (a layout that changes reduction order still needs
//! its exactness gate, so the record carries what the hashes showed).

use serde_json::{json, Value};

use crate::bench::BenchResult;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PinRule {
    /// A candidate point above `baseline * (1 + harm_tol)` is harm.
    pub harm_tol: f64,
    /// A candidate must beat the baseline's geometric mean by this much.
    pub min_gain: f64,
}

impl Default for PinRule {
    fn default() -> Self {
        Self {
            harm_tol: 0.0,
            min_gain: 0.02,
        }
    }
}

/// An inclusive range of the swept variable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Bucket {
    pub lo: u64,
    pub hi: u64,
}

impl Bucket {
    pub fn holds(&self, x: f64) -> bool {
        x >= self.lo as f64 && x <= self.hi as f64
    }
}

/// One arm's `(x, ms)` points.
#[derive(Debug, Clone)]
pub struct Curve {
    pub label: String,
    pub points: Vec<(f64, f64)>,
}

impl Curve {
    fn at(&self, x: f64) -> Option<f64> {
        self.points
            .iter()
            .find(|p| p.0 == x)
            .map(|p| p.1)
            .filter(|ms| ms.is_finite() && *ms > 0.0)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Judged {
    pub arm: String,
    pub accepted: bool,
    /// Largest `candidate / baseline` over the bucket.
    pub worst_ratio: Option<f64>,
    /// Geometric mean of the same ratios.
    pub geomean_ratio: Option<f64>,
    pub why: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BucketPin {
    pub bucket: Bucket,
    /// The baseline's swept x inside the bucket.
    pub xs: Vec<f64>,
    /// `None`: keep the baseline.
    pub winner: Option<String>,
    pub candidates: Vec<Judged>,
}

/// Judge one bucket. `None` when the baseline has no point in it.
pub fn judge_bucket(
    bucket: Bucket,
    baseline: &Curve,
    candidates: &[Curve],
    rule: PinRule,
) -> Option<BucketPin> {
    let xs: Vec<f64> = baseline
        .points
        .iter()
        .map(|p| p.0)
        .filter(|&x| bucket.holds(x))
        .collect();
    if xs.is_empty() {
        return None;
    }
    let judge = |c: &Curve| -> Judged {
        let refuse = |why: String, worst, geo| Judged {
            arm: c.label.clone(),
            accepted: false,
            worst_ratio: worst,
            geomean_ratio: geo,
            why,
        };
        let mut ratios = Vec::with_capacity(xs.len());
        for &x in &xs {
            let Some(base) = baseline.at(x) else {
                return refuse(format!("baseline has no usable time at {x}"), None, None);
            };
            // Unmeasured is not "no harm".
            let Some(ms) = c.at(x) else {
                return refuse(format!("not measured at {x}"), None, None);
            };
            ratios.push((x, ms / base));
        }
        let geo = (ratios.iter().map(|r| r.1.ln()).sum::<f64>() / ratios.len() as f64).exp();
        let (worst_x, worst) = ratios
            .iter()
            .copied()
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .expect("xs is not empty");
        if worst > 1.0 + rule.harm_tol {
            return refuse(
                format!("slower than the baseline at {worst_x} (x{worst:.3})"),
                Some(worst),
                Some(geo),
            );
        }
        if geo >= 1.0 - rule.min_gain {
            return refuse(
                format!(
                    "gain under {:.0}% (geomean x{geo:.3})",
                    rule.min_gain * 100.0
                ),
                Some(worst),
                Some(geo),
            );
        }
        Judged {
            arm: c.label.clone(),
            accepted: true,
            worst_ratio: Some(worst),
            geomean_ratio: Some(geo),
            why: "no point slower and the bucket gains".into(),
        }
    };
    let judged: Vec<Judged> = candidates.iter().map(judge).collect();
    let accepted: Vec<&Curve> = candidates
        .iter()
        .zip(&judged)
        .filter(|(_, j)| j.accepted)
        .map(|(c, _)| c)
        .collect();
    // Min-max regret among the accepted: the slowest-against-best point of
    // each, smallest wins; the geomean breaks a tie.
    let regret = |c: &Curve| {
        xs.iter()
            .map(|&x| {
                let best = accepted
                    .iter()
                    .filter_map(|a| a.at(x))
                    .fold(f64::INFINITY, f64::min);
                c.at(x).expect("accepted means measured") / best - 1.0
            })
            .fold(0.0, f64::max)
    };
    let geo = |label: &str| {
        judged
            .iter()
            .find(|j| j.arm == label)
            .and_then(|j| j.geomean_ratio)
            .unwrap_or(f64::INFINITY)
    };
    let winner = accepted
        .iter()
        .min_by(|a, b| {
            regret(a)
                .total_cmp(&regret(b))
                .then(geo(&a.label).total_cmp(&geo(&b.label)))
        })
        .map(|c| c.label.clone());
    Some(BucketPin {
        bucket,
        xs,
        winner,
        candidates: judged,
    })
}

fn curve_of(r: &BenchResult, label: &str) -> Option<Curve> {
    let a = r.arm(label)?;
    (!a.points.is_empty()).then(|| Curve {
        label: label.into(),
        points: a.points.iter().map(|p| (p.x, p.ms)).collect(),
    })
}

/// Output bits of `arm` equal to the baseline's at every x both hashed;
/// `None` when nothing was hashed.
fn bit_equal(r: &BenchResult, base: &Curve, arm: &str, xs: &[f64]) -> Option<bool> {
    let hash = |label: &str, x: f64| {
        r.arm(label)?
            .points
            .iter()
            .find(|p| p.x == x)?
            .out_fnv
            .clone()
    };
    // The baseline may be spliced from two arms, so its hashes come from
    // whichever arm `base_label` names at that x.
    let pairs: Vec<bool> = xs
        .iter()
        .filter_map(|&x| Some(hash(&base.label, x)? == hash(arm, x)?))
        .collect();
    (!pairs.is_empty()).then(|| pairs.iter().all(|&e| e))
}

/// A baseline arm per bucket: the route the runtime takes there today.
pub struct Plan<'a> {
    pub buckets: Vec<(Bucket, &'a str)>,
    pub candidates: Vec<&'a str>,
}

/// Apply the rule to a finished sweep and return the record the bench puts in
/// its result: the rule, the chip it was read on, and one row per bucket.
pub fn pin_result(r: &BenchResult, plan: &Plan, rule: PinRule, chip: Option<String>) -> Value {
    let mut rows = Vec::new();
    for &(bucket, base_label) in &plan.buckets {
        let Some(base) = curve_of(r, base_label) else {
            continue;
        };
        let candidates: Vec<Curve> = plan
            .candidates
            .iter()
            .filter(|&&c| c != base_label)
            .filter_map(|c| curve_of(r, c))
            .collect();
        let Some(pin) = judge_bucket(bucket, &base, &candidates, rule) else {
            continue;
        };
        let exact = pin
            .winner
            .as_deref()
            .and_then(|w| bit_equal(r, &base, w, &pin.xs));
        rows.push(json!({
            "lo": bucket.lo, "hi": bucket.hi, "xs": pin.xs, "baseline": base_label,
            "winner": pin.winner, "winner_bit_equal": exact,
            "candidates": pin.candidates.iter().map(|j| json!({
                "arm": j.arm, "accepted": j.accepted, "worst_ratio": j.worst_ratio,
                "geomean_ratio": j.geomean_ratio, "why": j.why,
            })).collect::<Vec<_>>(),
        }));
    }
    json!({
        "rule": {"harm_tol": rule.harm_tol, "min_gain": rule.min_gain},
        "chip": chip,
        "x": r.x.name,
        "buckets": rows,
    })
}

/// Put the pin of this sweep into the result: `subject.tile_pin` and one
/// printed note. The ledger's `kernel_curve` record carries both, with the
/// host's eligibility beside them: a pin from a DIRTY record is not one.
pub fn attach(r: &mut BenchResult, plan: &Plan) {
    let pin = pin_result(
        r,
        plan,
        PinRule::default(),
        turbospark_bench::memory::chip_brand_string(),
    );
    let rows: Vec<String> = pin["buckets"]
        .as_array()
        .into_iter()
        .flatten()
        .map(|b| {
            let winner = b["winner"].as_str().unwrap_or("keep baseline");
            let exact = match b["winner_bit_equal"].as_bool() {
                Some(true) => " (bit-equal)",
                Some(false) => " (DIFFERS in bits)",
                None => "",
            };
            format!("{}..{}: {winner}{exact}", b["lo"], b["hi"])
        })
        .collect();
    if !rows.is_empty() {
        r.notes.push(format!(
            "tile pin by {} (no-harm-per-bucket, this chip only, medians without a noise estimate): {}",
            r.x.name,
            rows.join("; ")
        ));
    }
    r.subject.insert("tile_pin".into(), pin);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn curve(label: &str, ms: &[f64]) -> Curve {
        Curve {
            label: label.into(),
            points: ms
                .iter()
                .enumerate()
                .map(|(i, &t)| ((i + 1) as f64, t))
                .collect(),
        }
    }

    const ALL: Bucket = Bucket { lo: 1, hi: 4 };

    fn judge(cands: &[Curve]) -> BucketPin {
        judge_bucket(ALL, &curve("base", &[10.0; 4]), cands, PinRule::default()).unwrap()
    }

    #[test]
    fn one_slower_point_refuses_a_candidate_that_wins_on_average() {
        let p = judge(&[curve("mean_winner", &[5.0, 5.0, 5.0, 10.5])]);
        assert_eq!(p.winner, None);
        assert!(p.candidates[0].why.contains("slower"), "{:?}", p.candidates);
        assert!(p.candidates[0].geomean_ratio.unwrap() < 0.7);
    }

    #[test]
    fn no_harm_and_a_real_gain_is_pinned() {
        let p = judge(&[curve("good", &[9.0, 10.0, 8.0, 9.0])]);
        assert_eq!(p.winner.as_deref(), Some("good"));
        // Equal to the baseline is not a gain.
        let p = judge(&[curve("same", &[10.0; 4])]);
        assert_eq!(p.winner, None);
        assert!(p.candidates[0].why.contains("gain under"));
    }

    #[test]
    fn the_smallest_worst_regret_wins_among_the_accepted() {
        // "spiky" has the better mean but is 50% off the best at x = 4.
        let p = judge(&[
            curve("spiky", &[4.0, 4.0, 4.0, 9.0]),
            curve("flat", &[6.0, 6.0, 6.0, 6.0]),
        ]);
        assert!(p.candidates.iter().all(|c| c.accepted));
        // Regrets: spiky 9/6 - 1 = 0.5, flat 6/4 - 1 = 0.5; the tie goes to
        // the better geomean.
        assert_eq!(p.winner.as_deref(), Some("spiky"));
        let p = judge(&[
            curve("spiky", &[4.0, 4.0, 4.0, 9.9]),
            curve("flat", &[6.0, 6.0, 6.0, 6.0]),
        ]);
        assert_eq!(p.winner.as_deref(), Some("flat"));
    }

    #[test]
    fn an_unmeasured_point_is_not_no_harm() {
        let p = judge(&[curve("short", &[5.0, 5.0, 5.0])]);
        assert_eq!(p.winner, None);
        assert!(p.candidates[0].why.contains("not measured"));
        let p = judge(&[curve("nan", &[5.0, f64::NAN, 5.0, 5.0])]);
        assert_eq!(p.winner, None);
    }

    #[test]
    fn buckets_are_judged_apart() {
        let base = curve("base", &[10.0; 4]);
        let c = [curve("c", &[12.0, 12.0, 5.0, 5.0])];
        let low = judge_bucket(Bucket { lo: 1, hi: 2 }, &base, &c, PinRule::default()).unwrap();
        let high = judge_bucket(Bucket { lo: 3, hi: 4 }, &base, &c, PinRule::default()).unwrap();
        assert_eq!(low.winner, None);
        assert_eq!(high.winner.as_deref(), Some("c"));
        assert!(judge_bucket(Bucket { lo: 9, hi: 12 }, &base, &c, PinRule::default()).is_none());
    }
}
