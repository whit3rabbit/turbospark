//! Evaluator item L8: price a kernel mode before a model run, and check the
//! price after.
//!
//! `tslab project` turns kernel curves, per-forward counts, a measured base
//! and an optional dispatch price into a predicted ratio, with the fraction of
//! the base the benched kernels explain (`coverage`). `tslab calibrate` joins
//! that record to the A/B verdict that followed and ledgers the miss, so a
//! prediction that keeps disagreeing with the model shows up as a bias per
//! bench instead of as a surprise.
//!
//! Why it exists: a per-dispatch time is not a round time. The 8-row tile
//! timed 2.7-5.6x faster than the scalar GEMM on the small projections and a
//! dispatch profile agreed, yet the clean wall-clock sweep read the verify
//! slower (`docs/KAPITSA_KERNELS.md`, lever 9). The same tile's one-row bench
//! (1.22x the GEMV) understated the model's plain-decode loss (0.77-0.81x).
//! A projection says up front how much of the base it explains and where the
//! unexplained part would have to hide.
//!
//! The math is the sum over the install's shapes, no fit and no free
//! parameter:
//!
//! ```text
//! delta_ms = sum_s count_s * (mode_ms(s, x) - default_ms(s, x))
//!            + dispatches * us_per_dispatch / 1000
//! ratio    = base_ms / (base_ms + delta_ms)          (throughput)
//! coverage = sum_s count_s * default_ms(s, x) / base_ms
//! ```
//!
//! Portable: JSON in, JSON and tables out.

use std::collections::BTreeMap;

use serde::Serialize;
use serde_json::{json, Value};

use crate::args::Args;
use crate::bench::BenchResult;
use crate::knobs::{lane_of, Compare};
use crate::ledger::Ledger;
use crate::paths::{experiment_dir, lab_dir};
use crate::schema::CmdOutput;
use crate::stats::{median, MIN_EFFECT_FLOOR};

/// Bumped when the formula changes meaning. Stored in every record.
pub const FORMULA_VERSION: u32 = 1;

/// Where the base time of the projection came from.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Base {
    /// Milliseconds the benched kernels sit inside: a decode token, a
    /// prefill token, or a speculative round (or one phase of it).
    pub ms: f64,
    /// A ledger id or a note saying where a typed-in number came from.
    pub source: String,
}

#[derive(Debug, Clone)]
pub struct Params {
    pub x: usize,
    pub default_arm: String,
    pub mode_arm: String,
    /// Only these case labels, when given.
    pub shapes: Option<Vec<String>>,
    /// A shape with fewer output rows is not served by the mode (the 8-row
    /// tile's routing threshold): left on the default, so it costs nothing
    /// extra and stays inside the base.
    pub min_rows: Option<usize>,
    pub base: Base,
    /// Dispatches the mode adds (positive) or removes (negative) per base unit.
    pub dispatches: i64,
    pub us_per_dispatch: Option<f64>,
    pub floor: f64,
    /// A stand-in block's in-place cost, ms per base unit, to read the benched
    /// kernels against (`blocks.<name>.ms_per_token` of a `standin` record).
    pub standin: Option<(String, f64)>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CaseRow {
    pub case: String,
    pub count: u64,
    pub default_ms: f64,
    pub mode_ms: f64,
    /// `count * (mode_ms - default_ms)`.
    pub delta_ms: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Skipped {
    pub case: String,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct Projection {
    pub formula: u32,
    pub x: usize,
    pub default_arm: String,
    pub mode_arm: String,
    pub base: Base,
    pub cases: Vec<CaseRow>,
    pub skipped: Vec<Skipped>,
    pub kernel_delta_ms: f64,
    pub dispatch_delta_ms: f64,
    pub delta_ms: f64,
    /// Throughput ratio, mode over default.
    pub ratio: f64,
    /// `ratio - 1`: what a higher-is-better metric (tok/s) should read.
    pub improvement_throughput: f64,
    /// `-delta / base`: what a lower-is-better time metric should read.
    pub improvement_time: f64,
    /// Benched default kernel time over the base.
    pub coverage: f64,
    pub floor: f64,
    pub clears_floor: bool,
    /// Benched default kernel time over the stand-in block's cost, when given.
    pub benched_vs_standin: Option<f64>,
    pub notes: Vec<String>,
}

fn arm_ms(r: &BenchResult, arm: &str, x: usize) -> Result<f64, String> {
    let a = r
        .arm(arm)
        .ok_or_else(|| format!("no arm {arm:?} (has {})", labels(r)))?;
    if let Some(why) = &a.unsupported {
        return Err(format!("arm {arm:?} unsupported: {why}"));
    }
    a.points
        .iter()
        .find(|p| p.x == x as f64)
        .map(|p| p.ms)
        .ok_or_else(|| format!("arm {arm:?} has no point at x={x}"))
}

fn labels(r: &BenchResult) -> String {
    r.arms
        .iter()
        .map(|a| a.label.as_str())
        .collect::<Vec<_>>()
        .join(", ")
}

/// Price the mode. `default` and `mode` are the bench results for the two
/// builds (the same list when both arms ran in one experiment).
pub fn project(
    p: &Params,
    default: &[BenchResult],
    mode: &[BenchResult],
) -> Result<Projection, String> {
    if !(p.base.ms.is_finite() && p.base.ms > 0.0) {
        return Err(format!("base time {} ms is not positive", p.base.ms));
    }
    let mut cases = Vec::new();
    let mut skipped = Vec::new();
    for d in default {
        if p.shapes
            .as_ref()
            .is_some_and(|s| !s.iter().any(|c| c == &d.case))
        {
            continue;
        }
        let skip = |why: String| Skipped {
            case: d.case.clone(),
            reason: why,
        };
        let rows = d
            .subject
            .get("rows")
            .and_then(Value::as_u64)
            .map(|r| r as usize);
        if let (Some(min), Some(rows)) = (p.min_rows, rows) {
            if rows < min {
                skipped.push(skip(format!(
                    "{rows} rows is under the mode's routing threshold {min}: stays on the default"
                )));
                continue;
            }
        }
        let count = d.subject.get("count").and_then(Value::as_u64).unwrap_or(0);
        if count == 0 {
            skipped.push(skip(
                "no per-forward count (a config fallback has none): not priced".into(),
            ));
            continue;
        }
        let Some(m) = mode.iter().find(|m| m.case == d.case) else {
            skipped.push(skip("case missing from the mode's run".into()));
            continue;
        };
        match (arm_ms(d, &p.default_arm, p.x), arm_ms(m, &p.mode_arm, p.x)) {
            (Ok(dm), Ok(mm)) => cases.push(CaseRow {
                case: d.case.clone(),
                count,
                default_ms: dm,
                mode_ms: mm,
                delta_ms: count as f64 * (mm - dm),
            }),
            (Err(e), _) | (_, Err(e)) => skipped.push(skip(e)),
        }
    }
    if cases.is_empty() {
        let why: Vec<String> = skipped
            .iter()
            .map(|s| format!("{}: {}", s.case, s.reason))
            .collect();
        return Err(format!(
            "nothing to price at x={}: {}",
            p.x,
            if why.is_empty() {
                "no case matched --shapes".to_string()
            } else {
                why.join("; ")
            }
        ));
    }
    let kernel_delta_ms: f64 = cases.iter().map(|c| c.delta_ms).sum();
    let benched_default_ms: f64 = cases.iter().map(|c| c.count as f64 * c.default_ms).sum();
    let dispatch_delta_ms =
        match (p.dispatches, p.us_per_dispatch) {
            (0, _) => 0.0,
            (n, Some(us)) => n as f64 * us / 1000.0,
            (_, None) => return Err(
                "--dispatches needs --us-per-dispatch or --probe-from (a dispatch_probe record)"
                    .into(),
            ),
        };
    let delta_ms = kernel_delta_ms + dispatch_delta_ms;
    let after = p.base.ms + delta_ms;
    if after <= 0.0 {
        return Err(format!(
            "base {} ms plus delta {delta_ms} ms is not positive: the inputs disagree",
            p.base.ms
        ));
    }
    let ratio = p.base.ms / after;
    let coverage = benched_default_ms / p.base.ms;
    let mut notes = Vec::new();
    if coverage > 1.0 {
        notes.push(format!(
            "the benched kernels cost {:.0}% of the base: the counts or the base are wrong",
            coverage * 100.0
        ));
    } else if coverage < 0.5 && delta_ms.abs() > 0.0 {
        notes.push(format!(
            "the benched kernels explain {:.0}% of the base: a mode that changes anything else (dispatch structure, scratch, other kernels) is not priced",
            coverage * 100.0
        ));
    }
    if !skipped.is_empty() {
        notes.push(format!(
            "{} case(s) not priced; their cost is inside the base",
            skipped.len()
        ));
    }
    let benched_vs_standin = p
        .standin
        .as_ref()
        .filter(|(_, ms)| *ms > 0.0)
        .map(|(_, ms)| benched_default_ms / ms);
    if let (Some((block, _)), Some(r)) = (&p.standin, benched_vs_standin) {
        if !(0.7..=1.3).contains(&r) {
            notes.push(format!(
                "the benched kernels are {:.0}% of the {block} stand-in's in-place cost: the bench does not explain the block",
                r * 100.0
            ));
        }
    }
    Ok(Projection {
        formula: FORMULA_VERSION,
        x: p.x,
        default_arm: p.default_arm.clone(),
        mode_arm: p.mode_arm.clone(),
        base: p.base.clone(),
        cases,
        skipped,
        kernel_delta_ms,
        dispatch_delta_ms,
        delta_ms,
        ratio,
        improvement_throughput: ratio - 1.0,
        improvement_time: -delta_ms / p.base.ms,
        coverage,
        floor: p.floor,
        clears_floor: (ratio - 1.0).abs() >= p.floor,
        benched_vs_standin,
        notes,
    })
}

/// The prediction a verdict on `higher_is_better` is judged against.
pub fn predicted_for(projection: &Value, higher_is_better: bool) -> Option<f64> {
    let key = if higher_is_better {
        "improvement_throughput"
    } else {
        "improvement_time"
    };
    projection[key].as_f64()
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Calibration {
    pub predicted: f64,
    pub measured: f64,
    /// `measured - predicted`.
    pub error: f64,
    pub floor: f64,
    /// The prediction and the measurement disagree on the sign of an effect
    /// either side calls real (past the floor), or by more than the floor.
    pub miss: bool,
    pub sign_flip: bool,
    pub reason: String,
}

pub fn calibrate(predicted: f64, measured: f64, floor: f64) -> Calibration {
    let error = measured - predicted;
    let real = |v: f64| v.abs() >= floor;
    let sign_flip = (real(predicted) || real(measured)) && predicted.signum() != measured.signum();
    let big = error.abs() > floor;
    let reason = if sign_flip {
        "the sign of the effect is wrong".to_string()
    } else if big {
        format!(
            "off by {:+.1} points (more than the {:.1}-point floor)",
            error * 100.0,
            floor * 100.0
        )
    } else {
        "within the floor".to_string()
    };
    Calibration {
        predicted,
        measured,
        error,
        floor,
        miss: sign_flip || big,
        sign_flip,
        reason,
    }
}

// --- ledger readers and the commands -------------------------------------

/// `EXP` or `EXP:ID`: an experiment, and optionally one record in it.
fn split_ref(s: &str) -> (&str, Option<&str>) {
    match s.split_once(':') {
        Some((e, id)) if !e.is_empty() && !id.is_empty() => (e, Some(id)),
        _ => (s, None),
    }
}

fn open(experiment: &str) -> Result<Ledger, String> {
    Ledger::open(&experiment_dir(&lab_dir(), experiment)?)
}

/// The newest record of `kind` in `experiment` (or the one with `id`).
fn newest(experiment: &str, kind: &str, id: Option<&str>) -> Result<Value, String> {
    let recs = open(experiment)?.of_kind(kind)?;
    let pick = match id {
        Some(id) => recs.into_iter().rev().find(|r| r["id"] == id),
        None => recs.into_iter().next_back(),
    };
    pick.ok_or_else(|| match id {
        Some(id) => format!("{experiment}: no {kind} record {id}"),
        None => format!("{experiment}: no {kind} record"),
    })
}

/// Kernel results of `bench` in an experiment: the latest eligible record per
/// case (else the latest), with the ids and the count of dirty ones used.
fn curves(experiment: &str, bench: &str) -> Result<(Vec<BenchResult>, Vec<String>, usize), String> {
    let recs = open(experiment)?.of_kind("kernel_curve")?;
    let mut by_case: BTreeMap<String, (bool, String, BenchResult)> = BTreeMap::new();
    for r in recs {
        let Ok(res) = serde_json::from_value::<BenchResult>(r["result"].clone()) else {
            continue;
        };
        if res.bench != bench {
            continue;
        }
        let eligible = r["eligible"] != false;
        let id = r["id"].as_str().unwrap_or("?").to_string();
        let keep = match by_case.get(&res.case) {
            Some((was_eligible, _, _)) => eligible || !*was_eligible,
            None => true,
        };
        if keep {
            by_case.insert(res.case.clone(), (eligible, id, res));
        }
    }
    if by_case.is_empty() {
        return Err(format!(
            "{experiment}: no kernel_curve record for bench {bench} (run tslab kernel --bench {bench} --experiment {experiment})"
        ));
    }
    let dirty = by_case.values().filter(|(e, _, _)| !e).count();
    let ids = by_case.values().map(|(_, id, _)| id.clone()).collect();
    Ok((
        by_case.into_values().map(|(_, _, r)| r).collect(),
        ids,
        dirty,
    ))
}

/// A decode or prefill base: `1000 / tok_s` of an arm of the newest
/// `profile_summary` in the experiment.
fn base_from_profile(spec: &str, metric: &str) -> Result<Base, String> {
    let (experiment, arm) = split_ref(spec);
    let rec = newest(experiment, "profile_summary", None)?;
    let summary = rec["summary"]
        .as_object()
        .ok_or_else(|| format!("{experiment}: profile_summary has no summary"))?;
    let arm = match arm {
        Some(a) => a.to_string(),
        None if summary.contains_key("base") => "base".to_string(),
        None if summary.len() == 1 => summary.keys().next().cloned().unwrap_or_default(),
        None => {
            return Err(format!(
                "{experiment}: pick an arm with --base-from {experiment}:ARM (has {})",
                summary.keys().cloned().collect::<Vec<_>>().join(", ")
            ))
        }
    };
    let tok_s = summary
        .get(&arm)
        .and_then(|a| a[metric].as_f64())
        .filter(|v| *v > 0.0)
        .ok_or_else(|| format!("{experiment}: arm {arm} has no {metric}"))?;
    Ok(Base {
        ms: 1000.0 / tok_s,
        source: format!(
            "{experiment} profile_summary {} arm {arm} {metric} {tok_s:.3}",
            rec["id"].as_str().unwrap_or("?")
        ),
    })
}

/// A speculative round's cost from `round_cost` records at one verify width.
fn base_from_rounds(
    experiment: &str,
    width: u64,
    flags: &str,
    kv: Option<u64>,
    phase: Option<&str>,
) -> Result<Base, String> {
    let recs = open(experiment)?.of_kind("round_cost")?;
    let mut vals = Vec::new();
    for r in &recs {
        if r["eligible"] != true || r["flags"] != flags || r["width"].as_u64() != Some(width) {
            continue;
        }
        if kv.is_some_and(|k| r["kv_bucket"].as_u64() != Some(k)) {
            continue;
        }
        let v = match phase {
            Some(p) => r["phases"][p].as_f64(),
            None => r["ms_per_round"].as_f64(),
        };
        if let Some(v) = v {
            vals.push(v);
        }
    }
    if vals.is_empty() {
        return Err(format!(
            "{experiment}: no eligible round_cost at width {width} with flags {flags:?} (arms need TURBOSPARK_SPEC_STATS=1)"
        ));
    }
    let m = median(&vals);
    let kept: Vec<f64> = vals.iter().copied().filter(|v| *v <= 3.0 * m).collect();
    Ok(Base {
        ms: median(&kept),
        source: format!(
            "{experiment} round_cost width {width} flags {flags} {} ({} records)",
            phase.unwrap_or("whole round"),
            kept.len()
        ),
    })
}

fn us_from_probe(experiment: &str) -> Result<(f64, String), String> {
    let r = newest(experiment, "dispatch_probe", None)?;
    let us = r["us_per_chained_dispatch"]
        .as_f64()
        .ok_or_else(|| format!("{experiment}: dispatch_probe has no us_per_chained_dispatch"))?;
    Ok((us, r["id"].as_str().unwrap_or("?").to_string()))
}

fn floor_from_null(experiment: &str) -> Result<(f64, String), String> {
    let r = newest(experiment, "null_calibration", None)?;
    let f = r["floor"].as_f64().unwrap_or(0.0).max(MIN_EFFECT_FLOOR);
    Ok((f, r["id"].as_str().unwrap_or("?").to_string()))
}

fn standin_from(experiment: &str, block: &str) -> Result<(f64, String), String> {
    let r = newest(experiment, "standin", None)?;
    let ms = r["blocks"][block]["ms_per_token"]
        .as_f64()
        .ok_or_else(|| format!("{experiment}: standin has no block {block}"))?;
    Ok((ms, r["id"].as_str().unwrap_or("?").to_string()))
}

fn pct(v: f64) -> String {
    format!("{:+.1}%", v * 100.0)
}

/// `tslab project`.
pub fn project_cmd(mut args: Args) -> Result<CmdOutput, String> {
    let knob = args.take("--knob")?;
    let lane = knob
        .as_deref()
        .map(|k| {
            lane_of(k).ok_or_else(|| {
                let names: Vec<&str> = crate::knobs::LANES.iter().map(|l| l.knob).collect();
                format!(
                    "--knob {k:?} is not a kernel mode (one of {})",
                    names.join(", ")
                )
            })
        })
        .transpose()?;
    let bench = match (args.take("--bench")?, lane) {
        (Some(b), _) => b,
        (None, Some(l)) => l.bench.to_string(),
        (None, None) => {
            return Err("give --knob NAME or --bench NAME --default ARM --mode ARM".into())
        }
    };
    let from_lane = |f: fn(&Compare) -> String| lane.map(|l| f(&l.compare));
    let default_arm = args
        .take("--default")?
        .or_else(|| {
            from_lane(|c| match c {
                Compare::Arms { baseline, .. } => baseline.to_string(),
                Compare::Process { arm, .. } => arm.to_string(),
            })
        })
        .ok_or("--default ARM is required without --knob")?;
    let mode_arm = args
        .take("--mode")?
        .or_else(|| {
            from_lane(|c| match c {
                Compare::Arms { lane, .. } => lane.to_string(),
                Compare::Process { arm, .. } => arm.to_string(),
            })
        })
        .ok_or("--mode ARM is required without --knob")?;
    let x: usize = args.parsed("--x", 1)?;
    let shapes = args
        .take("--shapes")?
        .map(|s| s.split(',').map(|v| v.trim().to_string()).collect());
    let min_rows: Option<usize> = args
        .take("--min-rows")?
        .map(|v| v.parse())
        .transpose()
        .map_err(|_| "--min-rows N".to_string())?;
    let curves_from = args.take("--curves")?;
    let default_from = args.take("--default-from")?;
    let mode_from = args.take("--mode-from")?;
    let base_ms = args.take("--base-ms")?;
    let base_from = args.take("--base-from")?;
    let base_metric = args
        .take("--base-metric")?
        .unwrap_or_else(|| "decode_tok_s".into());
    let round_from = args.take("--round-from")?;
    let width: u64 = args.parsed("--width", 0)?;
    let flags = args.take("--flags")?.unwrap_or_else(|| "default".into());
    let kv: Option<u64> = args
        .take("--kv")?
        .map(|v| v.parse())
        .transpose()
        .map_err(|_| "--kv N".to_string())?;
    let phase = args.take("--phase")?;
    let dispatches: i64 = args.parsed("--dispatches", 0)?;
    let us_flag: Option<f64> = args
        .take("--us-per-dispatch")?
        .map(|v| v.parse())
        .transpose()
        .map_err(|_| "--us-per-dispatch X".to_string())?;
    let probe_from = args.take("--probe-from")?;
    let floor_from = args.take("--floor-from")?;
    let share_from = args.take("--share-from")?;
    let block = args.take("--block")?;
    let experiment = args
        .take("--experiment")?
        .unwrap_or_else(|| "projection".into());
    args.finish()?;

    let mut inputs: Vec<Value> = Vec::new();
    let mut dirty = 0usize;
    let (default_res, mode_res) = match (&curves_from, &default_from, &mode_from) {
        (Some(c), None, None) => {
            if default_arm == mode_arm {
                return Err(format!(
                    "{default_arm} against {default_arm} in one experiment prices nothing: a switch the gpu crate reads once per process is compared across two \
                     `tslab kernel` runs, so give --default-from A --mode-from B"
                ));
            }
            let (r, ids, d) = curves(c, &bench)?;
            inputs.push(json!({"curves": c, "ids": ids}));
            dirty += d;
            (r.clone(), r)
        }
        (None, Some(a), Some(b)) => {
            let (ra, ia, da) = curves(a, &bench)?;
            let (rb, ib, db) = curves(b, &bench)?;
            inputs.push(json!({"default_curves": a, "ids": ia}));
            inputs.push(json!({"mode_curves": b, "ids": ib}));
            dirty += da + db;
            (ra, rb)
        }
        _ => return Err("give --curves EXP (both arms in one experiment) or --default-from A --mode-from B (one arm in two processes)".into()),
    };

    let base = match (&base_ms, &base_from, &round_from) {
        (Some(ms), None, None) => Base {
            ms: ms.parse().map_err(|_| format!("--base-ms {ms:?}"))?,
            source: "typed on the command line".into(),
        },
        (None, Some(spec), None) => base_from_profile(spec, &base_metric)?,
        (None, None, Some(e)) => {
            if width == 0 {
                return Err("--round-from needs --width W (the verify width, block + 1)".into());
            }
            base_from_rounds(e, width, &flags, kv, phase.as_deref())?
        }
        _ => return Err(
            "give exactly one of --base-ms X, --base-from EXP[:ARM], --round-from EXP --width W"
                .into(),
        ),
    };
    inputs.push(json!({"base": base.source}));

    let us = match (us_flag, &probe_from) {
        (Some(_), Some(_)) => return Err("give --us-per-dispatch or --probe-from, not both".into()),
        (Some(us), None) => {
            inputs.push(json!({"us_per_dispatch": us, "source": "typed on the command line"}));
            Some(us)
        }
        (None, Some(e)) => {
            let (us, id) = us_from_probe(e)?;
            inputs.push(json!({"dispatch_probe": id, "us_per_dispatch": us}));
            Some(us)
        }
        (None, None) => None,
    };
    let floor = match &floor_from {
        Some(e) => {
            let (f, id) = floor_from_null(e)?;
            inputs.push(json!({"null_calibration": id, "floor": f}));
            f
        }
        None => MIN_EFFECT_FLOOR,
    };
    let standin = match (&share_from, &block) {
        (Some(e), Some(b)) => {
            let (ms, id) = standin_from(e, b)?;
            inputs.push(json!({"standin": id, "block": b, "ms_per_token": ms}));
            Some((b.clone(), ms))
        }
        (None, None) => None,
        _ => return Err("--share-from EXP and --block gdn|mlp|attn go together".into()),
    };

    let params = Params {
        x,
        default_arm,
        mode_arm,
        shapes,
        min_rows,
        base,
        dispatches,
        us_per_dispatch: us,
        floor,
        standin,
    };
    let p = project(&params, &default_res, &mode_res)?;

    crate::say!(
        "projection{}: {} -> {} at x={} over {} case(s)",
        knob.as_deref()
            .map(|k| format!(" for {k}"))
            .unwrap_or_default(),
        p.default_arm,
        p.mode_arm,
        p.x,
        p.cases.len()
    );
    for c in &p.cases {
        crate::say!(
            "  {:<14} x{:<4} {:>9.4} -> {:>9.4} ms   {:+.3} ms",
            c.case,
            c.count,
            c.default_ms,
            c.mode_ms,
            c.delta_ms
        );
    }
    for s in &p.skipped {
        crate::say!("  {:<14} not priced: {}", s.case, s.reason);
    }
    crate::say!(
        "  kernels {:+.3} ms, dispatches {:+.3} ms, base {:.3} ms ({})",
        p.kernel_delta_ms,
        p.dispatch_delta_ms,
        p.base.ms,
        p.base.source
    );
    crate::say!(
        "  ratio {:.3}x ({} tok/s, {} time), coverage {:.0}%, floor {:.1}%: {}",
        p.ratio,
        pct(p.improvement_throughput),
        pct(p.improvement_time),
        p.coverage * 100.0,
        p.floor * 100.0,
        if p.clears_floor {
            "clears the floor"
        } else {
            "UNDER THE FLOOR: a model A/B cannot see it"
        }
    );
    for n in &p.notes {
        crate::say!("  note: {n}");
    }
    if dirty > 0 {
        crate::say!("  note: {dirty} input curve(s) came from DIRTY runs");
    }

    let lab = lab_dir();
    let id = Ledger::open(&experiment_dir(&lab, &experiment)?)?.append(
        "projection",
        json!({"knob": knob, "bench": bench, "inputs": inputs, "dirty_inputs": dirty,
               "projection": p}),
    )?;
    let mut out = CmdOutput::ok(json!({"id": id, "projection": p}));
    if dirty > 0 {
        out = out.warn(format!("{dirty} input curve(s) came from dirty runs"));
    }
    Ok(out)
}

/// A verdict's signed median improvement, its metric, and where it was read.
fn measured_from(spec: &str) -> Result<(f64, String, String), String> {
    let (experiment, id) = split_ref(spec);
    let recs = open(experiment)?.read_all()?;
    let usable = |r: &&Value| {
        matches!(
            r["kind"].as_str(),
            Some("verdict" | "confirm" | "experiment")
        ) && r["verdict"]["median"].is_number()
    };
    let rec = match id {
        Some(id) => recs.iter().rev().find(|r| r["id"] == id && usable(r)),
        None => recs.iter().rev().find(usable),
    }
    .ok_or_else(|| {
        format!("{experiment}: no verdict, confirm or experiment record with a median")
    })?;
    let metric = rec["metric"].as_str().unwrap_or("decode_tok_s").to_string();
    Ok((
        rec["verdict"]["median"].as_f64().unwrap_or(0.0),
        metric,
        format!("{experiment}:{}", rec["id"].as_str().unwrap_or("?")),
    ))
}

fn higher_is_better(metric: &str) -> Result<bool, String> {
    let m: crate::workload::Metric =
        serde_json::from_value(json!(metric)).map_err(|_| format!("unknown metric {metric:?}"))?;
    Ok(m.direction() == crate::stats::Direction::HigherIsBetter)
}

/// `tslab calibrate`.
pub fn calibrate_cmd(mut args: Args) -> Result<CmdOutput, String> {
    if args.flag("--all") {
        args.finish()?;
        return calibrations_table();
    }
    let proj_ref = args.required("--projection")?;
    let verdict_ref = args.take("--verdict")?;
    let measured_flag = args.take("--measured")?;
    let source = args.take("--source")?;
    let metric_flag = args.take("--metric")?;
    let invert = args.flag("--invert");
    let experiment = args
        .take("--experiment")?
        .unwrap_or_else(|| "calibration".into());
    args.finish()?;

    let (pe, pid) = split_ref(&proj_ref);
    let rec = newest(pe, "projection", pid)?;
    let proj = &rec["projection"];
    let floor = proj["floor"].as_f64().unwrap_or(MIN_EFFECT_FLOOR);
    let (measured, metric, verdict_src) = match (&verdict_ref, &measured_flag) {
        (Some(v), None) => measured_from(v)?,
        (None, Some(m)) => (
            m.parse::<f64>().map_err(|_| format!("--measured {m:?}"))?,
            metric_flag.clone().unwrap_or_else(|| "decode_tok_s".into()),
            source
                .clone()
                .ok_or("--measured needs --source TEXT saying where the number came from (a doc row, a run id)")?,
        ),
        _ => return Err("give --verdict EXP[:ID] or --measured X --source TEXT".into()),
    };
    // A `confirm` record carries no metric, so `--metric` names it.
    let metric = metric_flag.clone().unwrap_or(metric);
    let measured = if invert { -measured } else { measured };
    let hib = higher_is_better(&metric)?;
    let predicted = predicted_for(proj, hib).ok_or(
        "the projection record lacks improvement fields (a formula older than this build?)",
    )?;
    let c = calibrate(predicted, measured, floor);

    crate::say!(
        "calibration: predicted {} measured {} error {:+.1} points ({}) -> {}",
        pct(c.predicted),
        pct(c.measured),
        c.error * 100.0,
        metric,
        if c.miss {
            format!("MISS, {}", c.reason)
        } else {
            c.reason.clone()
        }
    );
    let lab = lab_dir();
    let id = Ledger::open(&experiment_dir(&lab, &experiment)?)?.append(
        "calibration",
        json!({"projection": format!("{pe}:{}", rec["id"].as_str().unwrap_or("?")),
               "verdict": verdict_src, "metric": metric, "inverted": invert,
               "knob": rec["knob"], "bench": rec["bench"],
               "coverage": proj["coverage"], "x": proj["x"],
               "default_arm": proj["default_arm"], "mode_arm": proj["mode_arm"],
               "calibration": c}),
    )?;
    Ok(CmdOutput::ok(json!({"id": id, "calibration": c})))
}

/// `tslab calibrate --all`: every calibration in the lab dir, and the mean
/// signed error per bench arm pair (the bias to subtract next time).
fn calibrations_table() -> Result<CmdOutput, String> {
    let lab = lab_dir();
    let mut rows: Vec<Value> = Vec::new();
    let Ok(entries) = std::fs::read_dir(&lab) else {
        return Err(format!("{}: no lab directory", lab.display()));
    };
    let mut dirs: Vec<_> = entries.flatten().filter(|e| e.path().is_dir()).collect();
    dirs.sort_by_key(|e| e.file_name());
    for e in dirs {
        let name = e.file_name().to_string_lossy().to_string();
        let Ok(l) = Ledger::open(&e.path()) else {
            continue;
        };
        for mut r in l.of_kind("calibration").unwrap_or_default() {
            r["experiment"] = json!(name);
            rows.push(r);
        }
    }
    if rows.is_empty() {
        crate::say!("no calibration records yet (tslab calibrate --projection ... --verdict ...)");
        return Ok(CmdOutput::ok(json!({"calibrations": [], "bias": {}})));
    }
    let mut bias: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    for r in &rows {
        let c = &r["calibration"];
        crate::say!(
            "{:<24} {:<14} {:<22} predicted {:>7} measured {:>7} {}",
            r["experiment"].as_str().unwrap_or("?"),
            r["bench"].as_str().unwrap_or("?"),
            format!(
                "{}->{} x{}",
                r["default_arm"].as_str().unwrap_or("?"),
                r["mode_arm"].as_str().unwrap_or("?"),
                r["x"]
            ),
            pct(c["predicted"].as_f64().unwrap_or(0.0)),
            pct(c["measured"].as_f64().unwrap_or(0.0)),
            if c["miss"] == true { "MISS" } else { "ok" }
        );
        if let Some(e) = c["error"].as_f64() {
            bias.entry(format!(
                "{} {}->{}",
                r["bench"].as_str().unwrap_or("?"),
                r["default_arm"].as_str().unwrap_or("?"),
                r["mode_arm"].as_str().unwrap_or("?")
            ))
            .or_default()
            .push(e);
        }
    }
    crate::say!("mean signed error (measured - predicted) per bench and arm pair:");
    let mut summary = serde_json::Map::new();
    for (k, v) in &bias {
        let mean = v.iter().sum::<f64>() / v.len() as f64;
        crate::say!(
            "  {k:<44} {:+.1} points over {} calibration(s)",
            mean * 100.0,
            v.len()
        );
        summary.insert(k.clone(), json!({"mean_error": mean, "n": v.len()}));
    }
    Ok(CmdOutput::ok(
        json!({"calibrations": rows, "bias": summary}),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bench::{ArmCurve, Axis, Point};

    fn result(case: &str, count: u64, arms: &[(&str, &[(f64, f64)])]) -> BenchResult {
        let mut subject = BTreeMap::new();
        subject.insert("count".to_string(), json!(count));
        BenchResult {
            bench: "b".into(),
            case: case.into(),
            subject,
            x: Axis::new("m", "rows", "log2"),
            layout: "l".into(),
            arms: arms
                .iter()
                .map(|(label, pts)| ArmCurve {
                    label: (*label).into(),
                    points: pts.iter().map(|(x, ms)| Point::new(*x, *ms)).collect(),
                    fit: None,
                    unsupported: None,
                })
                .collect(),
            break_even: Vec::new(),
            notes: Vec::new(),
        }
    }

    fn params(base_ms: f64) -> Params {
        Params {
            x: 1,
            default_arm: "gemv".into(),
            mode_arm: "tile".into(),
            shapes: None,
            min_rows: None,
            base: Base {
                ms: base_ms,
                source: "test".into(),
            },
            dispatches: 0,
            us_per_dispatch: None,
            floor: 0.02,
            standin: None,
        }
    }

    #[test]
    fn a_tile_that_is_slower_over_most_of_the_base_projects_a_loss() {
        // Kernels cost 40 ms of a 50 ms token; the mode is 1.25x slower on
        // all of them: +10 ms, so 50 / 60 of the throughput.
        let r = vec![
            result(
                "big",
                10,
                &[("gemv", &[(1.0, 3.0)]), ("tile", &[(1.0, 3.75)])],
            ),
            result(
                "small",
                5,
                &[("gemv", &[(1.0, 2.0)]), ("tile", &[(1.0, 2.5)])],
            ),
        ];
        let p = project(&params(50.0), &r, &r).unwrap();
        assert!((p.kernel_delta_ms - 10.0).abs() < 1e-9, "{p:?}");
        assert!((p.ratio - 50.0 / 60.0).abs() < 1e-9);
        assert!((p.coverage - 40.0 / 50.0).abs() < 1e-9);
        assert!(p.clears_floor);
        assert!((p.improvement_throughput + (1.0 - 50.0 / 60.0)).abs() < 1e-9);
        assert!((p.improvement_time + 0.2).abs() < 1e-9, "10 ms of 50");
    }

    #[test]
    fn dispatches_price_linearly_and_need_a_price() {
        let r = vec![result(
            "s",
            1,
            &[("gemv", &[(1.0, 1.0)]), ("tile", &[(1.0, 1.0)])],
        )];
        let mut p = params(10.0);
        p.dispatches = -100;
        assert!(project(&p, &r, &r).is_err(), "no price, no projection");
        p.us_per_dispatch = Some(2.0);
        let one = project(&p, &r, &r).unwrap();
        assert!((one.dispatch_delta_ms + 0.2).abs() < 1e-9);
        p.dispatches = -200;
        let two = project(&p, &r, &r).unwrap();
        assert!((two.dispatch_delta_ms - 2.0 * one.dispatch_delta_ms).abs() < 1e-9);
        assert!(two.ratio > one.ratio && one.ratio > 1.0);
    }

    #[test]
    fn a_case_without_a_count_or_a_point_is_named_not_priced() {
        let r = vec![
            result("ok", 2, &[("gemv", &[(1.0, 1.0)]), ("tile", &[(1.0, 2.0)])]),
            result(
                "nocount",
                0,
                &[("gemv", &[(1.0, 1.0)]), ("tile", &[(1.0, 1.0)])],
            ),
            result(
                "nox",
                3,
                &[("gemv", &[(8.0, 1.0)]), ("tile", &[(8.0, 1.0)])],
            ),
        ];
        let p = project(&params(10.0), &r, &r).unwrap();
        assert_eq!(p.cases.len(), 1);
        assert_eq!(p.skipped.len(), 2);
        assert!(p.skipped.iter().any(|s| s.case == "nocount"));
        assert!(p
            .skipped
            .iter()
            .any(|s| s.reason.contains("no point at x=1")));
        let none = vec![result("nocount", 0, &[("gemv", &[(1.0, 1.0)])])];
        let err = project(&params(10.0), &none, &none).unwrap_err();
        assert!(err.contains("nocount"), "{err}");
    }

    #[test]
    fn shapes_under_the_routing_threshold_stay_on_the_default() {
        let mut big = result(
            "big",
            4,
            &[("gemv", &[(1.0, 1.0)]), ("tile", &[(1.0, 1.25)])],
        );
        big.subject.insert("rows".into(), json!(17408));
        let mut small = result(
            "small",
            96,
            &[("gemv", &[(1.0, 0.05)]), ("tile", &[(1.0, 0.2)])],
        );
        small.subject.insert("rows".into(), json!(48));
        let r = vec![big, small];
        let mut p = params(50.0);
        let all = project(&p, &r, &r).unwrap();
        p.min_rows = Some(4096);
        let routed = project(&p, &r, &r).unwrap();
        assert_eq!(all.cases.len(), 2);
        assert_eq!(routed.cases.len(), 1);
        assert_eq!(routed.skipped.len(), 1);
        assert!(routed.skipped[0].reason.contains("routing threshold"));
        // 4 x 0.25 ms for the served shape; the small one adds nothing.
        assert!((routed.kernel_delta_ms - 1.0).abs() < 1e-9, "{routed:?}");
        assert!(all.kernel_delta_ms > routed.kernel_delta_ms);
    }

    #[test]
    fn two_processes_compare_the_same_arm_across_builds() {
        let baseline = vec![result("g", 48, &[("lazy", &[(8.0, 0.10)])])];
        let baked = vec![result("g", 48, &[("lazy", &[(8.0, 0.04)])])];
        let mut p = params(85.0);
        p.x = 8;
        p.default_arm = "lazy".into();
        p.mode_arm = "lazy".into();
        let pr = project(&p, &baseline, &baked).unwrap();
        // 48 * 0.06 = 2.88 ms saved of 85.
        assert!((pr.kernel_delta_ms + 2.88).abs() < 1e-9);
        assert!(pr.ratio > 1.03 && pr.clears_floor);
    }

    #[test]
    fn an_effect_under_the_floor_says_so_and_low_coverage_is_flagged() {
        let r = vec![result(
            "s",
            1,
            &[("gemv", &[(1.0, 1.0)]), ("tile", &[(1.0, 1.1)])],
        )];
        let p = project(&params(100.0), &r, &r).unwrap();
        assert!(!p.clears_floor, "0.1 ms of 100 ms");
        assert!(
            p.notes.iter().any(|n| n.contains("explain")),
            "{:?}",
            p.notes
        );
    }

    #[test]
    fn a_standin_that_disagrees_with_the_bench_is_flagged() {
        let r = vec![result(
            "s",
            10,
            &[("gemv", &[(1.0, 1.0)]), ("tile", &[(1.0, 1.0)])],
        )];
        let mut p = params(50.0);
        p.standin = Some(("gdn".into(), 25.0));
        let pr = project(&p, &r, &r).unwrap();
        assert_eq!(pr.benched_vs_standin, Some(10.0 / 25.0));
        assert!(
            pr.notes.iter().any(|n| n.contains("stand-in")),
            "{:?}",
            pr.notes
        );
    }

    #[test]
    fn calibration_flags_a_flipped_sign_and_a_large_error_but_not_noise() {
        // The lever 9 shape: a predicted gain that measured as a loss.
        let c = calibrate(0.045, -0.04, 0.02);
        assert!(c.miss && c.sign_flip);
        // Right sign, 5 points off, floor 2.
        let c = calibrate(0.10, 0.05, 0.02);
        assert!(c.miss && !c.sign_flip);
        // Both under the floor, opposite signs: that is noise, not a miss.
        let c = calibrate(0.005, -0.004, 0.02);
        assert!(!c.miss, "{c:?}");
        let c = calibrate(0.041, 0.0407, 0.02);
        assert!(!c.miss && c.reason == "within the floor");
    }

    #[test]
    fn the_prediction_follows_the_metric_direction() {
        let p = json!({"improvement_throughput": 0.25, "improvement_time": 0.2});
        assert_eq!(predicted_for(&p, true), Some(0.25));
        assert_eq!(predicted_for(&p, false), Some(0.2));
        assert!(higher_is_better("decode_tok_s").unwrap());
        assert!(!higher_is_better("followup_ttft_s").unwrap());
        assert!(higher_is_better("nope").is_err());
    }

    #[test]
    fn refs_split_on_the_first_colon_only() {
        assert_eq!(split_ref("exp"), ("exp", None));
        assert_eq!(split_ref("exp:123-4"), ("exp", Some("123-4")));
        assert_eq!(split_ref(":x"), (":x", None));
    }
}
