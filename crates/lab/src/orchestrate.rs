//! Runs workers and decides what their records mean.
//!
//! Every run is a fresh `tslab-worker` process with a CLEARED environment
//! plus a small passthrough plus the arm's own variables. That makes knobs
//! cached in a `OnceLock` behave exactly like knobs captured at open, and it
//! means no variable from the calling shell can leak into an arm. The host
//! is sampled before, during and after each run, and the run is recorded in
//! the ledger whether or not it was clean.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use serde_json::{json, Value};

use crate::host::{eligibility, HostMonitor, HostRules, HostSample};
use crate::knobs;
use crate::ledger::{new_id, Ledger};
use crate::stats::{self, VerdictReport};
use crate::worker::{Mode, WorkerRequest, EXIT_REFUSED};
use crate::workload::{Arm, Metric, Workload};

/// Non-TURBOSPARK variables the worker inherits.
pub const PASSTHROUGH: &[&str] = &["PATH", "HOME", "TMPDIR", "USER", "LANG", "LC_ALL"];
/// TURBOSPARK variables the worker inherits because they locate things
/// (models, the lab) rather than change behavior.
pub const PASSTHROUGH_TURBOSPARK: &[&str] = &["TURBOSPARK_HOME", "TURBOSPARK_LAB_DIR"];

const SAMPLE_EVERY: Duration = Duration::from_secs(2);

/// The worker binary that sits next to the running `tslab`.
pub fn sibling_worker() -> Result<PathBuf, String> {
    let exe = std::env::current_exe().map_err(|e| e.to_string())?;
    let worker = exe.with_file_name("tslab-worker");
    if worker.is_file() {
        Ok(worker)
    } else {
        Err(format!(
            "{} not found; build both binaries: cargo build --release -p turbospark-lab",
            worker.display()
        ))
    }
}

pub struct Session {
    pub exp_dir: PathBuf,
    pub ledger: Ledger,
    pub worker: PathBuf,
    pub rules: HostRules,
    pub timeout: Duration,
    pub model: String,
    pub verbose: bool,
}

/// One finished run as the orchestrator sees it.
#[derive(Debug, Clone)]
pub struct Outcome {
    pub run_id: String,
    pub arm: String,
    pub round: usize,
    pub warmup: bool,
    pub record: Value,
    pub status: String,
    pub eligible: bool,
    pub reasons: Vec<String>,
}

impl Outcome {
    pub fn metric(&self, m: Metric) -> Option<f64> {
        m.read(&self.record)
    }

    /// The page-cache classes of the run's prompts, in order.
    pub fn cache_classes(&self) -> Vec<Value> {
        self.record["prompts"]
            .as_array()
            .map(|ps| ps.iter().map(|p| p["cache_class"].clone()).collect())
            .unwrap_or_default()
    }
}

fn summarize_host(samples: &[HostSample]) -> Value {
    let max = |f: fn(&HostSample) -> f64| samples.iter().map(f).fold(f64::NAN, f64::max);
    json!({
        "samples": samples.len(),
        "load1_max": max(|s| s.load1),
        "foreign_cpu_max": max(|s| s.foreign_cpu_percent),
        "swap_used_mib_max": max(|s| s.swap_used_mib.unwrap_or(f64::NAN)),
        "busiest": samples
            .iter()
            .max_by(|a, b| a.foreign_cpu_percent.total_cmp(&b.foreign_cpu_percent))
            .map(|s| s.foreign_top.iter().take(3).map(|p| json!({"cpu": p.cpu_percent, "command": p.command})).collect::<Vec<_>>()),
    })
}

/// The runtime's `[spec-stats]` stderr lines (`TURBOSPARK_SPEC_STATS=1`) as
/// data: one entry per speculative call, warm pass included, in order. The
/// runtime prints them and keeps no accessor, so they are parsed here.
pub fn parse_spec_stats(stderr: &str) -> Value {
    let num = |s: &str| s.trim().parse::<f64>().ok();
    let mut calls: Vec<serde_json::Map<String, Value>> = Vec::new();
    for line in stderr.lines() {
        let Some(rest) = line.strip_prefix("[spec-stats] ") else {
            continue;
        };
        if rest.starts_with("rounds=") {
            let mut m = serde_json::Map::new();
            for kv in rest.split_whitespace() {
                if let Some((k, v)) = kv.split_once('=') {
                    if let Some(x) = num(v) {
                        m.insert(k.replace('/', "_per_"), json!(x));
                    }
                }
            }
            calls.push(m);
        } else if let (Some(cur), Some(curve)) = (
            calls.last_mut(),
            rest.strip_prefix("per-position acceptance: "),
        ) {
            cur.insert(
                "per_position_acceptance".into(),
                json!(curve.split_whitespace().map(num).collect::<Vec<_>>()),
            );
        } else if let (Some(cur), Some(ms)) = (calls.last_mut(), rest.strip_prefix("ms/round: ")) {
            let mut m = serde_json::Map::new();
            for kv in ms.split_whitespace() {
                if let Some((k, v)) = kv.split_once('=') {
                    if let Some(x) = num(v) {
                        m.insert(k.to_string(), json!(x));
                    }
                }
            }
            cur.insert("ms_per_round".into(), Value::Object(m));
        }
    }
    json!(calls)
}

impl Session {
    pub fn new(
        lab: &Path,
        experiment: &str,
        model: &str,
        rules: HostRules,
        timeout: Duration,
    ) -> Result<Self, String> {
        let exp_dir = crate::paths::experiment_dir(lab, experiment)?;
        std::fs::create_dir_all(exp_dir.join("runs"))
            .map_err(|e| format!("{}: {e}", exp_dir.display()))?;
        Ok(Self {
            ledger: Ledger::open(&exp_dir)?,
            exp_dir,
            // Checked at spawn: run-experiment swaps in a worker it builds.
            worker: sibling_worker().unwrap_or_default(),
            rules,
            timeout,
            model: model.to_string(),
            verbose: true,
        })
    }

    /// Use `worker` instead of the sibling binary: `run-experiment` measures
    /// a worker built from the experiment's own tree.
    pub fn with_worker(mut self, worker: PathBuf) -> Self {
        self.worker = worker;
        self
    }

    /// Spawn one worker for `arm`, watch it, record it.
    pub fn run_once(
        &self,
        arm: &Arm,
        workload: &Workload,
        mode: Mode,
        round: usize,
        warmup: bool,
    ) -> Result<Outcome, String> {
        if !self.worker.is_file() {
            return Err(sibling_worker()
                .err()
                .unwrap_or_else(|| format!("worker {} not found", self.worker.display())));
        }
        let run_id = new_id();
        let runs = self.exp_dir.join("runs");
        let request_path = runs.join(format!("{run_id}.request.json"));
        let record_path = runs.join(format!("{run_id}.json"));
        let stderr_path = runs.join(format!("{run_id}.stderr"));
        let logits_path =
            (mode == Mode::Quality).then(|| runs.join(format!("{run_id}.logits.f16")));
        let request = WorkerRequest {
            run_id: run_id.clone(),
            model: self.model.clone(),
            arm: arm.clone(),
            workload: workload.clone(),
            mode,
            logits_path: logits_path.clone(),
        };
        crate::atomic::write_new_json(
            &request_path,
            &serde_json::to_value(&request).map_err(|e| e.to_string())?,
        )?;
        let stderr = std::fs::File::create(&stderr_path)
            .map_err(|e| format!("{}: {e}", stderr_path.display()))?;

        let mut monitor = HostMonitor::new();
        let mut samples = vec![monitor.sample(None)];
        let mut cmd = Command::new(&self.worker);
        cmd.arg("--request")
            .arg(&request_path)
            .arg("--out")
            .arg(&record_path)
            .env_clear()
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::from(stderr));
        for key in PASSTHROUGH.iter().chain(PASSTHROUGH_TURBOSPARK) {
            if let Some(v) = std::env::var_os(key) {
                cmd.env(key, v);
            }
        }
        cmd.envs(&arm.env);
        let started = Instant::now();
        let mut child = cmd
            .spawn()
            .map_err(|e| format!("{}: {e}", self.worker.display()))?;
        let pid = child.id();
        let mut timed_out = false;
        let mut last_sample = Instant::now();
        let exit = loop {
            if let Some(status) = child.try_wait().map_err(|e| e.to_string())? {
                break status;
            }
            if started.elapsed() > self.timeout {
                let _ = child.kill();
                timed_out = true;
                break child.wait().map_err(|e| e.to_string())?;
            }
            // Poll exit often, sample the host on the slower cadence: each
            // sample runs `ps` and two `pmset` calls.
            std::thread::sleep(Duration::from_millis(250));
            if last_sample.elapsed() >= SAMPLE_EVERY {
                samples.push(monitor.sample(Some(pid)));
                last_sample = Instant::now();
            }
        };
        let wall_s = started.elapsed().as_secs_f64();
        samples.push(monitor.sample(None));

        let mut record: Value = std::fs::read_to_string(&record_path)
            .ok()
            .and_then(|t| serde_json::from_str(&t).ok())
            .unwrap_or(Value::Null);
        let spec_stats = std::fs::read_to_string(&stderr_path)
            .map(|t| parse_spec_stats(&t))
            .unwrap_or(Value::Null);
        if spec_stats.as_array().is_some_and(|a| !a.is_empty()) {
            if let Value::Object(m) = &mut record {
                m.insert("spec_stats".into(), spec_stats.clone());
            }
        }
        let status = if timed_out {
            "timeout".to_string()
        } else if exit.success() {
            record["status"].as_str().unwrap_or("missing").to_string()
        } else if exit.code() == Some(EXIT_REFUSED) {
            "refused".to_string()
        } else {
            "crash".to_string()
        };

        let (host_ok, mut reasons) = eligibility(&samples, &self.rules);
        if status != "ok" {
            reasons.insert(0, format!("worker status {status}"));
        }
        for env in knobs::unquotable_envs() {
            if arm.env.contains_key(env) {
                reasons.push(format!("{env} perturbs timing; run is not quotable"));
            }
        }
        let eligible = host_ok && reasons.is_empty();

        self.ledger.append(
            "run",
            json!({
                "run_id": run_id,
                "arm": arm.name,
                "arm_env": arm.env,
                "arm_open": arm.open,
                "round": round,
                "warmup": warmup,
                "mode": mode,
                "status": status,
                "exit_code": exit.code(),
                "wall_s": wall_s,
                "eligible": eligible,
                "reasons": reasons,
                "host": summarize_host(&samples),
                "record": record_path.display().to_string(),
                "refusal": record["refusal"],
                "family": record["family"],
                "totals": record["totals"],
                "peak_footprint_bytes": record["peak_footprint_bytes"],
                "exe_sha256": record["provenance"]["exe_sha256"],
                "speculation": arm.speculation,
                "spec_stats": spec_stats,
            }),
        )?;
        if self.verbose {
            eprintln!(
                "  round {round}{} {:<16} {status:<8} {}{}",
                if warmup { " (warmup)" } else { "" },
                arm.name,
                record["totals"]["prefill_tok_s"]
                    .as_f64()
                    .map(|v| format!("prefill {v:.1} tok/s"))
                    .unwrap_or_default(),
                if eligible {
                    String::new()
                } else {
                    format!("  DIRTY: {}", reasons.join("; "))
                }
            );
        }
        Ok(Outcome {
            run_id,
            arm: arm.name.clone(),
            round,
            warmup,
            record,
            status,
            eligible,
            reasons,
        })
    }

    /// `rounds` measured rounds over `arms` after one discarded warmup round,
    /// arm order rotated each round so no arm always runs first. Stops an arm
    /// for good once it is refused.
    pub fn run_rounds(
        &self,
        arms: &[Arm],
        workload: &Workload,
        rounds: usize,
    ) -> Result<Vec<Outcome>, String> {
        let mut out = Vec::new();
        let mut refused: std::collections::HashSet<String> = Default::default();
        for round in 0..=rounds {
            let n = arms.len();
            let mut order: Vec<usize> = (0..n).map(|i| (i + round) % n).collect();
            if round % 2 == 1 {
                order.reverse();
            }
            for i in order {
                let arm = &arms[i];
                if refused.contains(&arm.name) {
                    continue;
                }
                let o = self.run_once(arm, workload, Mode::Timing, round, round == 0)?;
                if o.status == "refused" {
                    eprintln!(
                        "  {} refused: {}",
                        arm.name,
                        o.record["refusal"]
                            .as_str()
                            .unwrap_or("(no reason recorded)")
                    );
                    refused.insert(arm.name.clone());
                }
                out.push(o);
            }
        }
        Ok(out)
    }
}

/// A paired A/B: `min_pairs` clean pairs, extended up to `max_pairs` only to
/// replace dirty ones (never to chase significance, which would inflate the
/// false-win rate). A pair is clean when both runs are eligible and every
/// prompt has the same page-cache class in both.
#[allow(clippy::too_many_arguments)]
pub fn run_ab(
    session: &Session,
    baseline: &Arm,
    candidate: &Arm,
    workload: &Workload,
    metric: Metric,
    min_effect: f64,
    min_pairs: usize,
    max_pairs: usize,
    deadline: Option<Instant>,
) -> Result<(VerdictReport, Vec<(Outcome, Outcome)>), String> {
    // Discarded warmup of each arm: Metal shader caches and the page cache.
    for arm in [baseline, candidate] {
        let o = session.run_once(arm, workload, Mode::Timing, 0, true)?;
        if o.status == "refused" {
            return Err(format!(
                "arm {} refused: {}",
                arm.name,
                o.record["refusal"].as_str().unwrap_or("(no reason)")
            ));
        }
    }
    let mut pairs = Vec::new();
    let mut clean = Vec::new();
    let mut round = 1;
    while clean.len() < min_pairs && pairs.len() < max_pairs {
        if deadline.is_some_and(|d| Instant::now() >= d) {
            eprintln!("  time budget spent after {} pairs", pairs.len());
            break;
        }
        let (first, second) = if round % 2 == 1 {
            (baseline, candidate)
        } else {
            (candidate, baseline)
        };
        let o1 = session.run_once(first, workload, Mode::Timing, round, false)?;
        let o2 = session.run_once(second, workload, Mode::Timing, round, false)?;
        let (a, b) = if round % 2 == 1 { (o1, o2) } else { (o2, o1) };
        let same_cache = a.cache_classes() == b.cache_classes();
        if a.eligible && b.eligible && same_cache {
            if let (Some(va), Some(vb)) = (a.metric(metric), b.metric(metric)) {
                if let Some(imp) = stats::improvement(va, vb, metric.direction()) {
                    clean.push(imp);
                }
            }
        } else if !same_cache {
            eprintln!("  round {round}: page-cache class differs between arms; pair excluded");
        }
        pairs.push((a, b));
        round += 1;
    }
    Ok((stats::decide(&clean, min_effect), pairs))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spec_stats_lines_parse_per_call() {
        let err = "noise\n[spec-stats] rounds=10 accepted/round=1.40 committed/round=2.40 rollbacks=6\n\
                   [spec-stats] per-position acceptance: 0.76 0.53 -\n\
                   [spec-stats] rounds per depth: 2:10\n\
                   [spec-stats] ms/round: draft=11.10 verify=62.40 accept=1.00 rollback=0.06 commit=0.08 rewind=0.00 bonus=0.49\n\
                   [spec-stats] rounds=3 accepted/round=1.00 committed/round=2.00 rollbacks=1\n";
        let v = parse_spec_stats(err);
        let calls = v.as_array().unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0]["committed_per_round"], 2.4);
        assert_eq!(calls[0]["rollbacks"], 6.0);
        assert_eq!(calls[0]["ms_per_round"]["verify"], 62.4);
        assert_eq!(calls[0]["per_position_acceptance"][2], Value::Null);
        assert!(calls[1].get("ms_per_round").is_none());
    }
}
