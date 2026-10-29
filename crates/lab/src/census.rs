//! `tslab expert-census`: how an install's router actually reuses experts.
//!
//! The reopen premise for routing hot experts elsewhere ("experts used 17+
//! times in a batch", `docs/SPEEDUP_PLAYBOOK.md`, "Not pursued") and the
//! streaming cache's sizing question ("does a batch's expert union fit the
//! slots?", `docs/EXPERT_RESIDENCY.md`) are the same measurement: the
//! runtime's bounded batch capture (`TURBOSPARK_ROUTER_HIST` plus
//! `TURBOSPARK_ROUTER_BATCHES=1`, `crates/runtime/src/router_hist.rs`).
//! This module reads that JSON and reduces it to a ledger record. The
//! analysis is pure and portable; the command that runs the capture is
//! macOS-only.
//!
//! Only complete rows count. A row is one layer of one batch: its
//! `routes` are the top-k ids per token, in token order. "Within a batch"
//! means within one such row, which is the unit the runtime can execute at
//! once. Thresholds are inclusive: `17` means used 17 times or more.

use serde_json::{json, Value};

/// Reuse thresholds, inclusive. 16 and 17 bracket the archived eligibility
/// floor so an off-by-one shows.
pub const REUSE_THRESHOLDS: [usize; 4] = [8, 16, 17, 32];
/// Expert-cache slot counts the union-fit table reports.
pub const SLOT_COUNTS: [usize; 4] = [16, 32, 64, 96];
/// Union caps for the routed sub-batch replay (`MAX_PREFILL_EXPERT_BINDINGS`
/// is 32).
pub const UNION_CAPS: [usize; 3] = [16, 32, 64];
/// Coverage levels for the concentration table.
pub const COVERAGE_LEVELS: [f64; 3] = [0.5, 0.8, 0.95];

/// One complete capture row.
#[derive(Debug, Clone, PartialEq)]
pub struct Row {
    pub phase: String,
    pub layer: usize,
    pub token_count: usize,
    pub routes: Vec<Vec<usize>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Capture {
    pub num_layers: usize,
    pub num_experts: usize,
    pub top_k: usize,
    pub truncated: bool,
    /// Complete rows only; incomplete ones are counted in `dropped`.
    pub rows: Vec<Row>,
    pub dropped: usize,
}

fn usize_of(v: &Value, what: &str) -> Result<usize, String> {
    v.as_u64()
        .map(|n| n as usize)
        .ok_or_else(|| format!("{what}: expected a non-negative integer, got {v}"))
}

/// Reads a `TURBOSPARK_ROUTER_HIST` file that was written with
/// `TURBOSPARK_ROUTER_BATCHES=1`. Validates what the replay script
/// validates: version 1, distinct in-range top-k ids per token, completed
/// rows carrying every token.
pub fn parse_capture(v: &Value) -> Result<Capture, String> {
    let num_layers = usize_of(&v["num_layers"], "num_layers")?;
    let num_experts = usize_of(&v["num_experts"], "num_experts")?;
    let top_k = usize_of(&v["top_k"], "top_k")?;
    if num_experts == 0 || top_k == 0 || top_k > num_experts {
        return Err(format!(
            "capture has {num_experts} experts and top-k {top_k}"
        ));
    }
    let detail = &v["batches"];
    if detail["version"].as_u64() != Some(1) {
        return Err(
            "capture has no version-1 batches: run with TURBOSPARK_ROUTER_BATCHES=1".into(),
        );
    }
    let truncated = detail["truncated"]
        .as_bool()
        .ok_or("batches.truncated must be a boolean")?;
    let mut rows = Vec::new();
    let mut dropped = 0;
    for (i, row) in detail["rows"]
        .as_array()
        .ok_or("batches.rows must be an array")?
        .iter()
        .enumerate()
    {
        let layer = usize_of(&row["layer"], "layer")?;
        if layer >= num_layers {
            return Err(format!(
                "row {i}: layer {layer} outside {num_layers} layers"
            ));
        }
        let token_count = usize_of(&row["token_count"], "token_count")?;
        let completed = row["completed"]
            .as_bool()
            .ok_or_else(|| format!("row {i}: completed must be a boolean"))?;
        let routes: Vec<Vec<usize>> = row["routes"]
            .as_array()
            .ok_or_else(|| format!("row {i}: routes must be an array"))?
            .iter()
            .map(|r| {
                r.as_array()
                    .ok_or_else(|| format!("row {i}: a route must be an array"))?
                    .iter()
                    .map(|e| usize_of(e, "expert id"))
                    .collect::<Result<Vec<_>, _>>()
            })
            .collect::<Result<_, _>>()?;
        for route in &routes {
            let mut seen = route.clone();
            seen.sort_unstable();
            seen.dedup();
            if route.len() != top_k || seen.len() != top_k || seen.iter().any(|&e| e >= num_experts)
            {
                return Err(format!(
                    "row {i}: a route must hold {top_k} distinct experts below {num_experts}"
                ));
            }
        }
        if !completed || routes.len() != token_count {
            dropped += 1;
            continue;
        }
        rows.push(Row {
            phase: row["phase"].as_str().unwrap_or("").to_string(),
            layer,
            token_count,
            routes,
        });
    }
    Ok(Capture {
        num_layers,
        num_experts,
        top_k,
        truncated,
        rows,
        dropped,
    })
}

/// Token counts of the greedy union-bounded sub-batches, in order: the
/// replay of `moe_prefill_pipeline::next_routed_sub_batch`. A sub-batch
/// takes tokens while the union of their experts fits `cap`. `Err` when one
/// token alone routes more experts than `cap`.
pub fn routed_sub_batches(routes: &[Vec<usize>], cap: usize) -> Result<Vec<usize>, String> {
    let mut sizes = Vec::new();
    let mut start = 0;
    while start < routes.len() {
        let mut union: Vec<usize> = routes[start].clone();
        union.sort_unstable();
        union.dedup();
        if union.len() > cap {
            return Err(format!(
                "one token routes {} experts against a cap of {cap}",
                union.len()
            ));
        }
        let mut n = 1;
        while start + n < routes.len() {
            let mut next = union.clone();
            next.extend(routes[start + n].iter().copied());
            next.sort_unstable();
            next.dedup();
            if next.len() > cap {
                break;
            }
            union = next;
            n += 1;
        }
        sizes.push(n);
        start += n;
    }
    Ok(sizes)
}

fn median(v: &mut [f64]) -> Option<f64> {
    if v.is_empty() {
        return None;
    }
    v.sort_by(|a, b| a.total_cmp(b));
    Some(v[v.len() / 2])
}

/// Distinct experts in a row.
fn union_size(routes: &[Vec<usize>]) -> usize {
    let mut all: Vec<usize> = routes.iter().flatten().copied().collect();
    all.sort_unstable();
    all.dedup();
    all.len()
}

/// Experts needed to cover `level` of the selections in `counts` (sorted
/// descending by the caller).
fn experts_for(sorted_desc: &[u64], total: u64, level: f64) -> usize {
    let target = (total as f64 * level).ceil() as u64;
    let mut acc = 0;
    for (i, c) in sorted_desc.iter().enumerate() {
        acc += c;
        if acc >= target {
            return i + 1;
        }
    }
    sorted_desc.len()
}

/// The census of one phase's complete rows.
fn phase_summary(c: &Capture, phase: &str) -> Value {
    let rows: Vec<&Row> = c.rows.iter().filter(|r| r.phase == phase).collect();
    if rows.is_empty() {
        return json!({"rows": 0});
    }
    let tokens: usize = rows.iter().map(|r| r.token_count).sum();
    let selections = tokens * c.top_k;

    // Concentration per layer: how many experts carry half, 80%, 95% of
    // that layer's selections in this phase.
    let mut per_layer: Vec<Vec<u64>> = vec![vec![0; c.num_experts]; c.num_layers];
    for r in &rows {
        for e in r.routes.iter().flatten() {
            per_layer[r.layer][*e] += 1;
        }
    }
    let mut concentration = serde_json::Map::new();
    for level in COVERAGE_LEVELS {
        let mut needs: Vec<f64> = per_layer
            .iter()
            .filter(|l| l.iter().any(|&n| n > 0))
            .map(|l| {
                let total: u64 = l.iter().sum();
                let mut sorted = l.clone();
                sorted.sort_unstable_by(|a, b| b.cmp(a));
                experts_for(&sorted, total, level) as f64
            })
            .collect();
        let max = needs.iter().copied().fold(0.0, f64::max);
        let med = median(&mut needs).unwrap_or(0.0);
        concentration.insert(
            format!("{}", (level * 100.0).round() as usize),
            json!({"experts_median_over_layers": med, "experts_max_over_layers": max,
                   "share_of_experts_median": med / c.num_experts as f64}),
        );
    }

    // Reuse within a row, inclusive thresholds.
    let mut reuse = serde_json::Map::new();
    let mut largest_group = 0usize;
    let mut row_counts: Vec<Vec<usize>> = Vec::with_capacity(rows.len());
    for r in &rows {
        let mut counts = vec![0usize; c.num_experts];
        for e in r.routes.iter().flatten() {
            counts[*e] += 1;
        }
        largest_group = largest_group.max(counts.iter().copied().max().unwrap_or(0));
        row_counts.push(counts);
    }
    for t in REUSE_THRESHOLDS {
        let covered: usize = row_counts
            .iter()
            .map(|counts| counts.iter().filter(|&&n| n >= t).sum::<usize>())
            .sum();
        let rows_with_any = row_counts
            .iter()
            .filter(|counts| counts.iter().any(|&n| n >= t))
            .count();
        let experts_per_row: f64 = row_counts
            .iter()
            .map(|counts| counts.iter().filter(|&&n| n >= t).count() as f64)
            .sum::<f64>()
            / rows.len() as f64;
        reuse.insert(
            t.to_string(),
            json!({"selection_coverage": covered as f64 / selections.max(1) as f64,
                   "rows_with_any": rows_with_any as f64 / rows.len() as f64,
                   "experts_per_row_mean": experts_per_row}),
        );
    }

    // Full-width rows: the batch the runtime actually executes.
    let width = rows.iter().map(|r| r.token_count).max().unwrap_or(0);
    let full: Vec<&Row> = rows
        .iter()
        .copied()
        .filter(|r| r.token_count == width)
        .collect();
    let mut unions: Vec<f64> = full.iter().map(|r| union_size(&r.routes) as f64).collect();
    let union_min = unions.iter().copied().fold(f64::INFINITY, f64::min);
    let union_max = unions.iter().copied().fold(0.0, f64::max);
    let union_med = median(&mut unions).unwrap_or(0.0);
    let mut slots_fit = serde_json::Map::new();
    for n in SLOT_COUNTS {
        let fit = full.iter().filter(|r| union_size(&r.routes) <= n).count();
        slots_fit.insert(n.to_string(), json!(fit as f64 / full.len().max(1) as f64));
    }
    let mut sub_batches = serde_json::Map::new();
    for cap in UNION_CAPS {
        let mut counts: Vec<f64> = Vec::new();
        let mut per_sub: Vec<f64> = Vec::new();
        let mut refused = None;
        for r in &full {
            match routed_sub_batches(&r.routes, cap) {
                Ok(sizes) => {
                    counts.push(sizes.len() as f64);
                    per_sub.push(r.token_count as f64 / sizes.len() as f64);
                }
                Err(e) => {
                    refused = Some(e);
                    break;
                }
            }
        }
        sub_batches.insert(
            cap.to_string(),
            match refused {
                Some(e) => json!({"refused": e}),
                None => json!({
                    "per_row_median": median(&mut counts).unwrap_or(0.0),
                    "tokens_per_sub_batch_mean":
                        per_sub.iter().sum::<f64>() / per_sub.len().max(1) as f64,
                }),
            },
        );
    }

    json!({
        "rows": rows.len(),
        "tokens": tokens,
        "selections": selections,
        "width": width,
        "full_rows": full.len(),
        "concentration": concentration,
        "reuse": reuse,
        "largest_group": largest_group,
        "union": {"min": union_min, "median": union_med, "max": union_max},
        "slots_fit": slots_fit,
        "sub_batches": sub_batches,
    })
}

/// The census record payload's `summary`.
pub fn summarize(c: &Capture) -> Value {
    let mut phases = serde_json::Map::new();
    let mut names: Vec<&str> = c.rows.iter().map(|r| r.phase.as_str()).collect();
    names.sort_unstable();
    names.dedup();
    for p in &names {
        phases.insert(p.to_string(), phase_summary(c, p));
    }
    // The capture covers the Qwen MoE chunked driver and single-token
    // calls (`crates/runtime/src/router_hist/batches.rs`). A family whose
    // prefill driver records nothing shows decode rows only, and that is a
    // coverage gap to name, not a routing finding.
    let mut notes = Vec::new();
    if !names.contains(&"prefill") {
        notes.push(
            "no prefill rows captured: this install's prefill driver does not record router \
             batches (only the Qwen MoE chunked driver and single-token calls do), so the \
             prefill reuse and union questions are unanswered here"
                .to_string(),
        );
    }
    if c.truncated {
        notes.push(
            "capture truncated at TURBOSPARK_ROUTER_BATCH_LIMIT: later rows are missing".into(),
        );
    }
    json!({
        "num_layers": c.num_layers,
        "num_experts": c.num_experts,
        "top_k": c.top_k,
        "truncated": c.truncated,
        "complete_rows": c.rows.len(),
        "dropped_rows": c.dropped,
        "thresholds": REUSE_THRESHOLDS,
        "slot_counts": SLOT_COUNTS,
        "union_caps": UNION_CAPS,
        "phases": phases,
        "notes": notes,
    })
}

/// The human table for one summary.
#[cfg(target_os = "macos")]
pub fn print_summary(summary: &Value) {
    let f = |v: &Value| v.as_f64().unwrap_or(f64::NAN);
    crate::say!(
        "experts {} top-k {} layers {}; {} complete rows{}",
        summary["num_experts"],
        summary["top_k"],
        summary["num_layers"],
        summary["complete_rows"],
        if summary["truncated"] == true {
            " (capture TRUNCATED)"
        } else {
            ""
        }
    );
    for n in summary["notes"].as_array().into_iter().flatten() {
        crate::say!("  NOTE: {}", n.as_str().unwrap_or(""));
    }
    let Some(phases) = summary["phases"].as_object() else {
        return;
    };
    for (phase, p) in phases {
        crate::say!(
            "{phase}: {} rows of width {}, {} selections, largest in-row group {}",
            p["rows"],
            p["width"],
            p["selections"],
            p["largest_group"]
        );
        if let Some(r) = p["reuse"].as_object() {
            let cells: Vec<String> = r
                .iter()
                .map(|(t, v)| format!("{t}+: {:.1}%", 100.0 * f(&v["selection_coverage"])))
                .collect();
            crate::say!("  selections in experts used {}", cells.join(", "));
        }
        if let Some(c) = p["concentration"].as_object() {
            let cells: Vec<String> = c
                .iter()
                .map(|(l, v)| format!("{l}%: {}", v["experts_median_over_layers"]))
                .collect();
            crate::say!("  experts per layer covering {}", cells.join(", "));
        }
        crate::say!(
            "  union per full row min {} median {} max {}",
            p["union"]["min"],
            p["union"]["median"],
            p["union"]["max"]
        );
        if let Some(s) = p["slots_fit"].as_object() {
            let cells: Vec<String> = s
                .iter()
                .map(|(n, v)| format!("{n}: {:.0}%", 100.0 * f(v)))
                .collect();
            crate::say!("  full rows whose union fits N slots {}", cells.join(", "));
        }
        if let Some(s) = p["sub_batches"].as_object() {
            let cells: Vec<String> = s
                .iter()
                .map(|(cap, v)| match v["refused"].as_str() {
                    Some(e) => format!("cap {cap}: {e}"),
                    None => format!(
                        "cap {cap}: {} sub-batches of {:.1} tokens",
                        v["per_row_median"],
                        f(&v["tokens_per_sub_batch_mean"])
                    ),
                })
                .collect();
            crate::say!("  routed sub-batches per row {}", cells.join("; "));
        }
    }
}

#[cfg(target_os = "macos")]
pub use cmd::census_cmd;

#[cfg(target_os = "macos")]
mod cmd {
    use std::collections::BTreeMap;

    use serde_json::json;

    use super::{parse_capture, print_summary, summarize};
    use crate::args::Args;
    use crate::lock::MachineLock;
    use crate::orchestrate::Session;
    use crate::paths::lab_dir;
    use crate::worker::Mode;
    use crate::workload::{Arm, OpenParams};

    /// `tslab expert-census --model M [workload] [host] [--experiment NAME]`:
    /// one capture run, then the census record.
    pub fn census_cmd(mut args: Args) -> Result<(), String> {
        let model = args.required("--model")?;
        let experiment = args
            .take("--experiment")?
            .unwrap_or_else(|| "expert-census".into());
        let workload = crate::cli::workload_from(&mut args)?;
        let rules = crate::cli::host_rules_from(&mut args)?;
        let timeout = crate::cli::timeout_from(&mut args)?;
        args.finish()?;
        if workload.audio.is_some() {
            return Err("expert-census measures text installs".into());
        }
        let lab = lab_dir();
        let _lock = MachineLock::acquire(&lab, &format!("tslab expert-census {experiment}"))?;
        let session = Session::new(&lab, &experiment, &model, rules, timeout)?;
        let capture_path = session
            .exp_dir
            .join("runs")
            .join(format!("{}.routes.json", crate::ledger::new_id()));
        let mut env = BTreeMap::new();
        env.insert(
            "TURBOSPARK_ROUTER_HIST".to_string(),
            capture_path.display().to_string(),
        );
        env.insert("TURBOSPARK_ROUTER_BATCHES".to_string(), "1".to_string());
        let arm = Arm {
            name: "census".into(),
            env,
            open: OpenParams::default(),
            speculation: None,
            expect_engaged: Vec::new(),
            contention: None,
        };
        eprintln!(
            "tslab expert-census: one capture run of {model}; ledger {}",
            session.ledger.path().display()
        );
        let outcome = session.run_once(&arm, &workload, Mode::Timing, 0, false)?;
        if outcome.status != "ok" {
            return Err(format!(
                "capture run {}: {}{}",
                outcome.run_id,
                outcome.status,
                outcome.record["refusal"]
                    .as_str()
                    .map(|r| format!(": {r}"))
                    .unwrap_or_default()
            ));
        }
        let text = std::fs::read_to_string(&capture_path).map_err(|e| {
            format!(
                "{}: {e} (the runtime writes the capture when the runner drops; a dense install \
                 writes none)",
                capture_path.display()
            )
        })?;
        let value: serde_json::Value =
            serde_json::from_str(&text).map_err(|e| format!("{}: {e}", capture_path.display()))?;
        let capture = parse_capture(&value)?;
        let summary = summarize(&capture);
        print_summary(&summary);
        session.ledger.append(
            "expert_census",
            json!({
                "model": model,
                "workload": workload,
                "run_id": outcome.run_id,
                "family": outcome.record["family"],
                "resolved_slots": outcome.record["resolved_slots"],
                "capture": capture_path.display().to_string(),
                "exe_sha256": outcome.record["provenance"]["exe_sha256"],
                "summary": summary,
                "provenance": crate::provenance::collect(),
            }),
        )?;
        Ok(())
    }
}
