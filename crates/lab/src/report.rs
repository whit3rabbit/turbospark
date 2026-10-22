//! Human-readable summaries of orchestrator outcomes. The ledger holds the
//! data; this only prints it.

use serde_json::{json, Value};

use crate::orchestrate::Outcome;
use crate::phases::BUCKET_NAMES;
use crate::stats::{median, VerdictReport};

/// Per-token bucket ms for one phase (`prefill_phases` or `decode_phases`),
/// summed over the run's prompts and divided by their total forward passes.
pub fn aggregate(record: &Value, phase: &str) -> Option<Value> {
    let prompts = record["prompts"].as_array()?;
    let mut calls = 0f64;
    let mut ms = vec![0f64; BUCKET_NAMES.len()];
    let (mut cb1, mut routed, mut fin) = (0f64, 0f64, 0f64);
    let (mut req, mut hits, mut bytes_req) = (0f64, 0f64, 0f64);
    let mut bytes_phys: Option<f64> = Some(0.0);
    for p in prompts {
        let ph = &p[phase];
        if ph.is_null() {
            continue;
        }
        let c = ph["calls"].as_f64()?;
        calls += c;
        for (i, b) in ph["buckets"].as_array()?.iter().enumerate().take(ms.len()) {
            ms[i] += b["ms"].as_f64()?;
        }
        cb1 += ph["gpu_busy_ms_per_token"]["cb1"].as_f64()? * c;
        routed += ph["gpu_busy_ms_per_token"]["routed"].as_f64()? * c;
        fin += ph["gpu_busy_ms_per_token"]["final_cb"].as_f64()? * c;
        req += ph["expert_requests"].as_f64()?;
        hits += ph["expert_hits"].as_f64()?;
        bytes_req += ph["expert_bytes_requested"].as_f64()?;
        bytes_phys = match (bytes_phys, ph["expert_bytes_physical"].as_f64()) {
            (Some(a), Some(b)) => Some(a + b),
            _ => None,
        };
    }
    if calls == 0.0 {
        return None;
    }
    let mib = |b: f64| b / (1024.0 * 1024.0) / calls;
    Some(json!({
        "calls": calls,
        "ms_per_token": BUCKET_NAMES.iter().zip(&ms).map(|(n, v)| (n.to_string(), json!(v / calls))).collect::<serde_json::Map<_, _>>(),
        "total_ms_per_token": ms.iter().sum::<f64>() / calls,
        "gpu_busy": {"cb1": cb1 / calls, "routed": routed / calls, "final": fin / calls},
        "expert_hit_rate": (req > 0.0).then(|| hits / req),
        "expert_mib_per_token": {"requested": mib(bytes_req), "physical": bytes_phys.map(mib)},
    }))
}

fn med(values: impl Iterator<Item = Option<f64>>) -> Option<(f64, f64, f64)> {
    let v: Vec<f64> = values.flatten().collect();
    if v.is_empty() {
        return None;
    }
    let lo = v.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = v.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    Some((median(&v), lo, hi))
}

/// The profile table: per arm, clean measured runs only.
pub fn print_profile(arm_names: &[String], outcomes: &[Outcome]) -> Value {
    let mut summary = serde_json::Map::new();
    for name in arm_names {
        let all: Vec<&Outcome> = outcomes
            .iter()
            .filter(|o| &o.arm == name && !o.warmup)
            .collect();
        let clean: Vec<&Outcome> = all.iter().copied().filter(|o| o.eligible).collect();
        println!();
        let refused = outcomes
            .iter()
            .find(|o| &o.arm == name && o.status == "refused")
            .and_then(|o| o.record["refusal"].as_str().map(str::to_string));
        if let Some(r) = refused {
            println!("{name}: REFUSED: {r}");
            summary.insert(name.clone(), json!({"refused": r}));
            continue;
        }
        let first = clean.first().or(all.first());
        println!(
            "{name}: {} clean of {} measured runs; slots {} residency {}",
            clean.len(),
            all.len(),
            first.map_or(json!(null), |o| o.record["resolved_slots"].clone()),
            first.map_or(json!(null), |o| o.record["resolved_residency"].clone()),
        );
        if clean.is_empty() {
            for o in &all {
                println!("  dirty run {}: {}", o.run_id, o.reasons.join("; "));
            }
            summary.insert(name.clone(), json!({"clean_runs": 0}));
            continue;
        }
        let tok_s = med(clean
            .iter()
            .map(|o| o.record["totals"]["prefill_tok_s"].as_f64()));
        let pre_s = med(clean
            .iter()
            .map(|o| o.record["totals"]["prefill_s"].as_f64()));
        let dec = med(clean
            .iter()
            .map(|o| o.record["totals"]["decode_tok_s"].as_f64()));
        let peak = med(clean.iter().map(|o| {
            o.record["peak_footprint_bytes"]
                .as_f64()
                .map(|b| b / (1u64 << 30) as f64)
        }));
        if let (Some(t), Some(s)) = (tok_s, pre_s) {
            println!(
                "  prefill   {:.1} tok/s [{:.1}-{:.1}]   {:.2} s [{:.2}-{:.2}]",
                t.0, t.1, t.2, s.0, s.1, s.2
            );
        }
        if let Some(d) = dec {
            println!("  decode    {:.2} tok/s [{:.2}-{:.2}]", d.0, d.1, d.2);
        }
        if let Some(p) = peak {
            println!("  peak      {:.2} GiB phys_footprint", p.0);
        }
        let mut arm_json = json!({
            "clean_runs": clean.len(),
            "prefill_tok_s": tok_s.map(|t| t.0),
            "prefill_s": pre_s.map(|t| t.0),
            "decode_tok_s": dec.map(|t| t.0),
            "peak_gib": peak.map(|t| t.0),
        });
        for phase in ["prefill_phases", "decode_phases"] {
            let aggs: Vec<Value> = clean
                .iter()
                .filter_map(|o| aggregate(&o.record, phase))
                .collect();
            if aggs.is_empty() {
                continue;
            }
            println!(
                "  {} (median ms/token over runs):",
                phase.trim_end_matches("_phases")
            );
            let mut phase_json = serde_json::Map::new();
            for b in BUCKET_NAMES {
                let m = med(aggs.iter().map(|a| a["ms_per_token"][b].as_f64()));
                if let Some((m, lo, hi)) = m {
                    let total = med(aggs.iter().map(|a| a["total_ms_per_token"].as_f64()))
                        .map_or(1.0, |t| t.0);
                    println!(
                        "    {b:<24} {m:>8.3}  [{lo:.3}-{hi:.3}]  {:>5.1}%",
                        100.0 * m / total
                    );
                    phase_json.insert(b.to_string(), json!(m));
                }
            }
            let busy_cb1 = med(aggs.iter().map(|a| a["gpu_busy"]["cb1"].as_f64()));
            let busy_routed = med(aggs.iter().map(|a| a["gpu_busy"]["routed"].as_f64()));
            if let (Some(c), Some(r)) = (busy_cb1, busy_routed) {
                println!("    gpu busy: cb1 {:.3}, routed {:.3} ms/token", c.0, r.0);
            }
            if let Some(h) = med(aggs.iter().map(|a| a["expert_hit_rate"].as_f64())) {
                let rq = med(aggs
                    .iter()
                    .map(|a| a["expert_mib_per_token"]["requested"].as_f64()));
                let ph = med(aggs
                    .iter()
                    .map(|a| a["expert_mib_per_token"]["physical"].as_f64()));
                println!(
                    "    experts: hit {:.1}%, requested {:.1} MiB/token, physical {}",
                    h.0 * 100.0,
                    rq.map_or(0.0, |v| v.0),
                    ph.map_or("unmeasured".to_string(), |v| format!(
                        "{:.1} MiB/token",
                        v.0
                    ))
                );
            }
            arm_json[phase] = Value::Object(phase_json);
        }
        summary.insert(name.clone(), arm_json);
    }
    Value::Object(summary)
}

pub fn print_verdict(label: &str, v: &VerdictReport) {
    println!();
    println!(
        "{label}: {:?} on {} clean pairs; median {:+.2}%, 95% CI [{:+.2}%, {:+.2}%], {} better / {} worse, effect floor {:.2}%",
        v.verdict,
        v.pairs,
        v.median * 100.0,
        v.ci_low * 100.0,
        v.ci_high * 100.0,
        v.positives,
        v.negatives,
        v.min_effect * 100.0
    );
    for r in &v.reasons {
        println!("  {r}");
    }
}
