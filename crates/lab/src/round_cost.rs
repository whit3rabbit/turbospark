//! Evaluator item L7 (`docs/MLX_SERVE_KERNELS.md`): speculative round costs
//! measured in the runs themselves, keyed by (install, verify width, KV
//! bucket, flags), so `tslab depth-replay` prices a round from this host's
//! ledger instead of two constants read off a doc table.
//!
//! The runtime's `[spec-stats] ms/round` line is a per-call average over
//! that call's rounds, by phase. A call is one prompt of a run; the first
//! call of every run is the worker's warm pass and is dropped. Adaptive
//! arms move the width every round, so they carry no single width and are
//! skipped. Across records, readings over three times the median are
//! dropped as spikes, the rule mlx-serve's planner uses.
//!
//! Portable: JSON in, JSON and tables out.

use std::collections::BTreeMap;

use serde_json::{json, Value};

/// The power of two at or above `tokens`: KV buckets of 1K, 2K, 4K, ...
pub fn kv_bucket(tokens: u64) -> u64 {
    tokens.max(1).next_power_of_two().max(1024)
}

/// Which round-shaping flags an arm set, as one stable label.
pub fn flags_label(env: &Value) -> String {
    let keys = ["TURBOSPARK_INT4_MMA8", "TURBOSPARK_GDN_LAZY_TAPE"];
    let set: Vec<String> = keys
        .iter()
        .filter_map(|k| env[*k].as_str().map(|v| format!("{k}={v}")))
        .collect();
    if set.is_empty() {
        "default".into()
    } else {
        // `+`, not `,`: `--costs` lists are comma separated.
        set.join("+")
    }
}

/// The `round_cost` payloads one speculative run yields: one per measured
/// call. `run` is the orchestrator's ledger payload (arm env, speculation,
/// spec_stats) and `record` the worker's record (prompts, manifest hash).
pub fn from_run(run: &Value, record: &Value) -> Vec<Value> {
    let spec = &run["speculation"];
    let Some(block) = spec["block"].as_u64() else {
        return Vec::new();
    };
    if spec["adaptive"] == true || run["warmup"] == true || run["status"] != "ok" {
        return Vec::new();
    }
    let calls = run["spec_stats"].as_array().cloned().unwrap_or_default();
    let prompts = record["prompts"].as_array().cloned().unwrap_or_default();
    let mut out = Vec::new();
    for (i, call) in calls.iter().skip(1).enumerate() {
        let Some(phases) = call["ms_per_round"].as_object() else {
            continue;
        };
        let total: f64 = phases.values().filter_map(Value::as_f64).sum();
        if total <= 0.0 {
            continue;
        }
        let p = prompts.get(i).cloned().unwrap_or(Value::Null);
        let context =
            p["prompt_tokens"].as_u64().unwrap_or(0) + p["decode_tokens"].as_u64().unwrap_or(0);
        out.push(json!({
            "model": run["model"], "manifest_sha256": record["manifest_sha256"],
            "family": record["family"], "drafter": spec["drafter"],
            "block": block, "width": block + 1, "kv_bucket": kv_bucket(context),
            "flags": flags_label(&run["arm_env"]), "ms_per_round": total,
            "phases": phases, "rounds": call["rounds"], "n": 1,
            "committed_per_round": call["committed_per_round"],
            "run_id": run["run_id"], "eligible": run["eligible"],
        }));
    }
    out
}

/// Measured ms per round by depth (block), from `round_cost` records of one
/// flags label and, when given, one KV bucket. Eligible records only;
/// spikes over 3x the per-depth median dropped.
pub fn by_depth(records: &[Value], flags: &str, kv: Option<u64>) -> BTreeMap<usize, (f64, usize)> {
    let mut groups: BTreeMap<usize, Vec<f64>> = BTreeMap::new();
    for r in records {
        if r["eligible"] != true || r["flags"] != flags {
            continue;
        }
        if kv.is_some_and(|k| r["kv_bucket"].as_u64() != Some(k)) {
            continue;
        }
        if let (Some(b), Some(ms)) = (r["block"].as_u64(), r["ms_per_round"].as_f64()) {
            groups.entry(b as usize).or_default().push(ms);
        }
    }
    groups
        .into_iter()
        .filter_map(|(d, v)| {
            let m = crate::stats::median(&v);
            let kept: Vec<f64> = v.into_iter().filter(|x| *x <= 3.0 * m).collect();
            (!kept.is_empty()).then(|| (d, (crate::stats::median(&kept), kept.len())))
        })
        .collect()
}

/// Depths 1..=`max_depth` priced from measured depths: measured ones as
/// read, the rest on the least-squares line through them (modeled).
/// Needs two measured depths.
pub fn fill(
    measured: &BTreeMap<usize, (f64, usize)>,
    max_depth: usize,
) -> Result<BTreeMap<usize, f64>, String> {
    let pts: Vec<(f64, f64)> = measured
        .iter()
        .map(|(d, (ms, _))| (*d as f64, *ms))
        .collect();
    let fit = crate::fit::fit_affine(&pts).ok_or_else(|| {
        format!(
            "round costs measured at {} depth(s); the replay needs two (run speculative arms at two blocks with TURBOSPARK_SPEC_STATS=1)",
            pts.len()
        )
    })?;
    Ok((1..=max_depth)
        .filter_map(|d| {
            let ms = measured.get(&d).map_or_else(|| fit.at(d as f64), |m| m.0);
            (ms > 0.0).then_some((d, ms))
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(block: u64, verify: f64, eligible: bool) -> Value {
        json!({"speculation": {"drafter": "mtp", "block": block, "adaptive": false},
               "warmup": false, "status": "ok", "eligible": eligible, "run_id": "r",
               "arm_env": {"TURBOSPARK_SPEC_STATS": "1"},
               "spec_stats": [
                   {"ms_per_round": {"draft": 1.0, "verify": 1.0}},
                   {"rounds": 50.0, "ms_per_round": {"draft": 10.0, "verify": verify}}]})
    }

    #[test]
    fn a_run_yields_one_cost_per_measured_call() {
        let rec = json!({"prompts": [{"prompt_tokens": 1500, "decode_tokens": 64}],
                         "manifest_sha256": "abc"});
        let c = from_run(&run(2, 50.0, true), &rec);
        assert_eq!(c.len(), 1, "the warm call is dropped");
        assert_eq!(c[0]["width"], 3);
        assert_eq!(c[0]["kv_bucket"], 2048);
        assert_eq!(c[0]["ms_per_round"], 60.0);
        assert_eq!(c[0]["flags"], "default");
        let mut adaptive = run(4, 50.0, true);
        adaptive["speculation"]["adaptive"] = json!(true);
        assert!(from_run(&adaptive, &rec).is_empty());
    }

    #[test]
    fn depths_fill_from_two_measured_and_spikes_drop() {
        let rec = json!({"prompts": [{"prompt_tokens": 100, "decode_tokens": 64}]});
        let mut rows: Vec<Value> = Vec::new();
        for (b, v) in [(2, 57.0), (2, 58.0), (2, 500.0), (4, 120.0)] {
            rows.extend(from_run(&run(b, v, true), &rec));
        }
        rows.extend(from_run(&run(4, 1.0, false), &rec));
        let m = by_depth(&rows, "default", None);
        assert_eq!(m[&2].1, 2, "the 3x spike is dropped");
        assert_eq!(m[&4], (130.0, 1), "the dirty run is not counted");
        let filled = fill(&m, 8).unwrap();
        assert_eq!(filled[&4], 130.0);
        assert!((filled[&3] - (filled[&2] + filled[&4]) / 2.0).abs() < 1e-9);
        assert!(fill(&by_depth(&rows[..1], "default", None), 8).is_err());
    }
}
