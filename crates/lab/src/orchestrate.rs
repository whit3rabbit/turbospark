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

/// Variables the orchestrator itself sets for every run, outside any arm:
/// engagement lines (`foundation::engaged!`) print only, change no lane, and
/// the worker's environment check accepts exactly these values.
pub const ORCHESTRATOR_ENV: &[(&str, &str)] = &[("TURBOSPARK_ENGAGEMENT", "1")];

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
        "contention": samples.iter().find_map(|s| s.contention),
        "busiest": samples
            .iter()
            .max_by(|a, b| a.foreign_cpu_percent.total_cmp(&b.foreign_cpu_percent))
            .map(|s| s.foreign_top.iter().take(3).map(|p| json!({"cpu": p.cpu_percent, "command": p.command})).collect::<Vec<_>>()),
    })
}

/// The runtime's `[ane-prefill]` stderr lines (`TURBOSPARK_ANE_PREFILL_STATS=1`,
/// `families/qwen/ane_split.rs` `print_stats`) as data: one entry per split
/// micro-batch, warm pass included, plus a summary of the host handoff. The
/// handoff is `wake_ms` (GPU done to the host seeing it) plus `resume_ms`
/// (host signal to the next segment's first GPU work), the two costs a
/// device-side event would remove; its share of `wall_ms` is the reopen
/// test in `docs/SPEEDUP_PLAYBOOK.md`, "Not pursued". `Null` without lines.
pub fn parse_ane_stats(stderr: &str) -> Value {
    let mut batches: Vec<serde_json::Map<String, Value>> = Vec::new();
    for line in stderr.lines() {
        let Some(rest) = line.strip_prefix("[ane-prefill] ") else {
            continue;
        };
        if !rest.starts_with("mode=") {
            continue;
        }
        let mut m = serde_json::Map::new();
        for kv in rest.split_whitespace() {
            if let Some((k, v)) = kv.split_once('=') {
                m.insert(
                    k.to_string(),
                    v.parse::<f64>().map_or_else(|_| json!(v), |x| json!(x)),
                );
            }
        }
        batches.push(m);
    }
    if batches.is_empty() {
        return Value::Null;
    }
    let f = |m: &serde_json::Map<String, Value>, k: &str| {
        m.get(k).and_then(Value::as_f64).unwrap_or(0.0)
    };
    let wall: f64 = batches.iter().map(|m| f(m, "wall_ms")).sum();
    let handoff: f64 = batches
        .iter()
        .map(|m| f(m, "wake_ms") + f(m, "resume_ms"))
        .sum();
    let points: f64 = batches.iter().map(|m| f(m, "points")).sum();
    let gpu_waits: f64 = batches.iter().map(|m| f(m, "gpu_waits_ane_ms")).sum();
    let ane_waits: f64 = batches.iter().map(|m| f(m, "ane_waits_gpu_ms")).sum();
    let mut per_point: Vec<f64> = batches
        .iter()
        .filter(|m| f(m, "points") > 0.0)
        .map(|m| (f(m, "wake_ms") + f(m, "resume_ms")) / f(m, "points"))
        .collect();
    per_point.sort_by(|a, b| a.total_cmp(b));
    let pct = |p: f64| {
        per_point
            .get(((per_point.len() as f64 - 1.0) * p).round() as usize)
            .copied()
    };
    json!({
        "micro_batches": batches,
        "summary": {
            "micro_batches": per_point.len(),
            "points": points,
            "wall_ms": wall,
            "handoff_ms": handoff,
            "handoff_share": if wall > 0.0 { handoff / wall } else { 0.0 },
            "handoff_ms_per_point_mean": if points > 0.0 { handoff / points } else { 0.0 },
            "handoff_ms_per_point_p50": pct(0.5),
            "handoff_ms_per_point_p95": pct(0.95),
            "gpu_waits_ane_ms": gpu_waits,
            "ane_waits_gpu_ms": ane_waits,
        }
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
        // Before anything is built or written: a release worktree is about
        // 5 GB and a measuring session's records add more, and a full disk
        // fails in the middle of a run with a message about something else.
        crate::host::check_free_disk(lab, rules.min_free_gb)?;
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
        // Controlled contention: a sibling `tslab-worker --hog N` started
        // before the measured worker and stopped (stdin closed) after it. A
        // child of this process, so the host monitor's ppid walk never counts
        // it as foreign load; the sample records it instead.
        let mut hog = match &arm.contention {
            Some(c) => {
                let child = Command::new(&self.worker)
                    .arg("--hog")
                    .arg(c.hog_threads.to_string())
                    .stdin(Stdio::piped())
                    .stdout(Stdio::null())
                    .stderr(Stdio::null())
                    .spawn()
                    .map_err(|e| format!("hog {}: {e}", self.worker.display()))?;
                monitor.set_hog(child.id(), c.hog_threads);
                Some(child)
            }
            None => None,
        };
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
        // Every run prints its engagement lines; an arm whose variables
        // promise a lane the log does not show is not the arm it claims.
        cmd.envs(ORCHESTRATOR_ENV.iter().copied());
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
        if let Some(mut h) = hog.take() {
            drop(h.stdin.take());
            let _ = h.wait();
        }

        let mut record: Value = std::fs::read_to_string(&record_path)
            .ok()
            .and_then(|t| serde_json::from_str(&t).ok())
            .unwrap_or(Value::Null);
        let stderr_text = std::fs::read_to_string(&stderr_path).unwrap_or_default();
        let spec_stats = parse_spec_stats(&stderr_text);
        let ane_stats = parse_ane_stats(&stderr_text);
        let engaged = knobs::engaged_lines(&stderr_text);
        if let Value::Object(m) = &mut record {
            if spec_stats.as_array().is_some_and(|a| !a.is_empty()) {
                m.insert("spec_stats".into(), spec_stats.clone());
            }
            if !ane_stats.is_null() {
                m.insert("ane_stats".into(), ane_stats.clone());
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
        if let Some(c) = &arm.contention {
            reasons.push(format!(
                "controlled contention hog:{}, not quotable",
                c.hog_threads
            ));
        }
        // Only a run that got far enough to run its lanes can be held to
        // them: a refused or crashed run already carries its reason.
        if status == "ok" {
            for v in knobs::engagement_violations(&arm.env, &arm.expect_engaged, &engaged) {
                reasons.push(format!("not engaged: {v}"));
            }
        }
        let eligible = host_ok && reasons.is_empty();

        let run_payload = json!({
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
            "audio": record["audio"],
            "peak_footprint_bytes": record["peak_footprint_bytes"],
            "exe_sha256": record["provenance"]["exe_sha256"],
            "speculation": arm.speculation,
            "spec_stats": spec_stats,
            "ane_stats": ane_stats,
            "controlled_contention": arm.contention,
            "engaged": engaged,
        });
        self.ledger.append("run", run_payload.clone())?;
        // L7: every measured speculative call is also a round-cost reading.
        let mut with_model = run_payload;
        with_model["model"] = json!(self.model);
        for cost in crate::round_cost::from_run(&with_model, &record) {
            self.ledger.append("round_cost", cost)?;
        }
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

/// What a paired A/B produced.
pub struct AbRun {
    pub verdict: VerdictReport,
    pub pairs: Vec<(Outcome, Outcome)>,
    /// Why the run stopped before it had `min_pairs` clean pairs for a reason
    /// other than the pair budget: the host never went quiet. `None` when it
    /// ran to its natural end. An early stop is not evidence about the
    /// candidate, and callers report it as such.
    pub stopped: Option<String>,
}

/// How many runs each reason made dirty, most common first. This is what to
/// fix before re-running: the verdict only says the pairs were dirty.
pub fn dirty_causes(pairs: &[(Outcome, Outcome)]) -> Vec<(String, usize)> {
    let mut counts: std::collections::BTreeMap<String, usize> = Default::default();
    for (a, b) in pairs {
        for o in [a, b].into_iter().filter(|o| !o.eligible) {
            for reason in &o.reasons {
                *counts.entry(reason.clone()).or_default() += 1;
            }
        }
    }
    let mut v: Vec<(String, usize)> = counts.into_iter().collect();
    v.sort_by(|x, y| y.1.cmp(&x.1).then_with(|| x.0.cmp(&y.0)));
    v
}

/// One pair: the two arms back to back in an order that alternates by
/// round, so slow drift cancels within the pair. Returns the outcomes in
/// (baseline, candidate) order and the pair's improvement when it is clean:
/// both runs eligible and every prompt in the same page-cache class.
fn run_pair(
    session: &Session,
    baseline: &Arm,
    candidate: &Arm,
    workload: &Workload,
    metric: Metric,
    round: usize,
) -> Result<((Outcome, Outcome), Option<f64>), String> {
    let (first, second) = if round % 2 == 1 {
        (baseline, candidate)
    } else {
        (candidate, baseline)
    };
    let o1 = session.run_once(first, workload, Mode::Timing, round, false)?;
    let o2 = session.run_once(second, workload, Mode::Timing, round, false)?;
    let (a, b) = if round % 2 == 1 { (o1, o2) } else { (o2, o1) };
    let same_cache = a.cache_classes() == b.cache_classes();
    let mut clean = None;
    if a.eligible && b.eligible && same_cache {
        if let (Some(va), Some(vb)) = (a.metric(metric), b.metric(metric)) {
            clean = stats::improvement(va, vb, metric.direction());
        }
    } else if !same_cache {
        eprintln!("  round {round}: page-cache class differs between arms; pair excluded");
    }
    Ok(((a, b), clean))
}

/// Wait for a quiet host before a pair, or say why the run stops.
fn quiet_or_stop(session: &Session, deadline: Option<Instant>, round: usize) -> Option<String> {
    if session.rules.quiet_wait_s <= 0.0 {
        return None;
    }
    let quiet = crate::host::wait_until_quiet(&session.rules, deadline);
    if !quiet.quiet {
        let why = format!(
            "the host did not go quiet within {:.0}s ({} checks) before pair {round}: {}",
            quiet.waited_s,
            quiet.checks,
            quiet.reasons.join("; ")
        );
        eprintln!("  {why}");
        return Some(why);
    }
    if quiet.checks > 1 {
        eprintln!(
            "  host went quiet after {:.0}s ({} checks); starting pair {round}",
            quiet.waited_s, quiet.checks
        );
    }
    None
}

fn print_dirty_causes(pairs: &[(Outcome, Outcome)]) {
    let causes = dirty_causes(pairs);
    if !causes.is_empty() {
        let dirty_runs: usize = pairs
            .iter()
            .map(|(a, b)| usize::from(!a.eligible) + usize::from(!b.eligible))
            .sum();
        eprintln!(
            "  {dirty_runs} of {} runs were dirty. Causes, by runs affected:",
            pairs.len() * 2
        );
        for (reason, n) in causes.iter().take(6) {
            eprintln!("    {n:>3}  {reason}");
        }
    }
}

/// A paired A/B: `min_pairs` clean pairs, extended up to `max_pairs` only to
/// replace dirty ones (never to chase significance, which would inflate the
/// false-win rate). A pair is clean when both runs are eligible and every
/// prompt has the same page-cache class in both.
///
/// Before each pair the host must be quiet (`rules.quiet_wait_s`, 0 to skip):
/// a pair started on a busy host is a dirty pair that also spends the pair
/// budget and the time. If the host does not settle in that time the run STOPS
/// with the offenders named, instead of measuring through the dirt.
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
) -> Result<AbRun, String> {
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
    let mut stopped = None;
    let mut round = 1;
    while clean.len() < min_pairs && pairs.len() < max_pairs {
        if deadline.is_some_and(|d| Instant::now() >= d) {
            eprintln!("  time budget spent after {} pairs", pairs.len());
            break;
        }
        if let Some(why) = quiet_or_stop(session, deadline, round) {
            stopped = Some(why);
            break;
        }
        let (pair, imp) = run_pair(session, baseline, candidate, workload, metric, round)?;
        if let Some(imp) = imp {
            clean.push(imp);
        }
        pairs.push(pair);
        round += 1;
    }
    print_dirty_causes(&pairs);
    Ok(AbRun {
        verdict: stats::decide(&clean, min_effect),
        pairs,
        stopped,
    })
}

/// One candidate's share of a sweep.
pub struct SweepArm {
    pub candidate: Arm,
    pub verdict: VerdictReport,
    pub pairs: Vec<(Outcome, Outcome)>,
    /// Why the candidate was dropped before it had a pair (a refusal).
    pub refused: Option<String>,
}

/// What a sweep produced: every candidate judged against the one baseline.
pub struct SweepRun {
    pub arms: Vec<SweepArm>,
    pub stopped: Option<String>,
}

/// Several candidates against one baseline, each through the same pair
/// discipline as [`run_ab`]: the baseline is re-run next to every candidate
/// so no pair shares a run, and `stats::decide` judges each candidate's own
/// clean pairs under the same floor. Candidates are interleaved within a
/// round in a rotated order, so thermal drift falls on all of them alike.
/// A refused candidate is recorded and dropped; the others continue.
#[allow(clippy::too_many_arguments)]
pub fn run_sweep(
    session: &Session,
    baseline: &Arm,
    candidates: &[Arm],
    workload: &Workload,
    metric: Metric,
    min_effect: f64,
    min_pairs: usize,
    max_pairs: usize,
    deadline: Option<Instant>,
) -> Result<SweepRun, String> {
    let o = session.run_once(baseline, workload, Mode::Timing, 0, true)?;
    if o.status == "refused" {
        return Err(format!(
            "baseline {} refused: {}",
            baseline.name,
            o.record["refusal"].as_str().unwrap_or("(no reason)")
        ));
    }
    let mut arms: Vec<SweepArm> = Vec::with_capacity(candidates.len());
    let mut clean: Vec<Vec<f64>> = vec![Vec::new(); candidates.len()];
    for c in candidates {
        let o = session.run_once(c, workload, Mode::Timing, 0, true)?;
        let refused = (o.status == "refused").then(|| {
            let why = o.record["refusal"]
                .as_str()
                .unwrap_or("(no reason)")
                .to_string();
            eprintln!("  {} refused: {why}", c.name);
            why
        });
        arms.push(SweepArm {
            candidate: c.clone(),
            verdict: stats::decide(&[], min_effect),
            pairs: Vec::new(),
            refused,
        });
    }
    let needs_more = |i: usize, arms: &[SweepArm], clean: &[Vec<f64>]| {
        arms[i].refused.is_none() && clean[i].len() < min_pairs && arms[i].pairs.len() < max_pairs
    };
    let mut stopped = None;
    let mut round = 1;
    loop {
        let active: Vec<usize> = (0..candidates.len())
            .filter(|&i| needs_more(i, &arms, &clean))
            .collect();
        if active.is_empty() {
            break;
        }
        if deadline.is_some_and(|d| Instant::now() >= d) {
            eprintln!("  time budget spent after round {}", round - 1);
            break;
        }
        if let Some(why) = quiet_or_stop(session, deadline, round) {
            stopped = Some(why);
            break;
        }
        // Rotate which candidate goes first each round.
        let n = active.len();
        let order: Vec<usize> = (0..n).map(|k| active[(k + round - 1) % n]).collect();
        for i in order {
            let (pair, imp) = run_pair(session, baseline, &candidates[i], workload, metric, round)?;
            if let Some(imp) = imp {
                clean[i].push(imp);
            }
            arms[i].pairs.push(pair);
        }
        round += 1;
    }
    for (i, arm) in arms.iter_mut().enumerate() {
        arm.verdict = stats::decide(&clean[i], min_effect);
        if !arm.pairs.is_empty() {
            eprintln!("  {}:", arm.candidate.name);
            print_dirty_causes(&arm.pairs);
        }
    }
    Ok(SweepRun { arms, stopped })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn outcome(eligible: bool, reasons: &[&str]) -> Outcome {
        Outcome {
            run_id: "r".into(),
            arm: "a".into(),
            round: 1,
            warmup: false,
            record: Value::Null,
            status: "ok".into(),
            eligible,
            reasons: reasons.iter().map(|s| s.to_string()).collect(),
        }
    }

    #[test]
    fn dirty_causes_count_runs_by_reason_most_common_first() {
        let pairs = vec![
            (
                outcome(
                    false,
                    &["mediaanalysisd over 90%", "a compile (rustc) was running"],
                ),
                outcome(false, &["mediaanalysisd over 90%"]),
            ),
            (
                outcome(true, &[]),
                outcome(false, &["a compile (rustc) was running"]),
            ),
        ];
        assert_eq!(
            dirty_causes(&pairs),
            vec![
                ("a compile (rustc) was running".to_string(), 2),
                ("mediaanalysisd over 90%".to_string(), 2),
            ],
            "ties break alphabetically, and clean runs contribute nothing"
        );
        // A clean run that still carries a stray reason is not counted.
        assert!(dirty_causes(&[(outcome(true, &["stale"]), outcome(true, &[]))]).is_empty());
    }

    #[test]
    fn ane_stats_lines_parse_per_micro_batch_with_a_handoff_summary() {
        let err = "noise\n\
            [ane-prefill] mode=Split start=0 m=2048 slot=2048 points=4 encode_ms=10.0 wall_ms=200.0 \
            gpu_busy_ms=150.0 steel_rows_ms=90.0 ane_ms=80.0 wake_ms=1.0 gpu_waits_ane_ms=0.5 \
            ane_waits_gpu_ms=2.0 resume_ms=1.0 gateup_n=4 gateup_rows_ms=90.0\n\
            [ane-prefill] mode=Split start=2048 m=1024 slot=2048 points=2 encode_ms=5.0 wall_ms=100.0 \
            gpu_busy_ms=70.0 steel_rows_ms=40.0 ane_ms=30.0 wake_ms=3.0 gpu_waits_ane_ms=0.0 \
            ane_waits_gpu_ms=0.0 resume_ms=3.0\n";
        let v = parse_ane_stats(err);
        assert_eq!(v["micro_batches"].as_array().unwrap().len(), 2);
        assert_eq!(v["micro_batches"][0]["mode"], "Split");
        assert_eq!(v["micro_batches"][0]["gateup_n"], 4.0);
        let s = &v["summary"];
        assert_eq!(s["points"], 6.0);
        assert_eq!(s["wall_ms"], 300.0);
        // (1 + 1) + (3 + 3) = 8 ms of handoff over 300 ms of wall.
        assert_eq!(s["handoff_ms"], 8.0);
        assert!((s["handoff_share"].as_f64().unwrap() - 8.0 / 300.0).abs() < 1e-12);
        // Per point: 0.5 ms in the first micro-batch, 3.0 in the second.
        assert_eq!(s["handoff_ms_per_point_p50"], 3.0);
        assert_eq!(s["handoff_ms_per_point_p95"], 3.0);
        assert!((s["handoff_ms_per_point_mean"].as_f64().unwrap() - 8.0 / 6.0).abs() < 1e-12);
        assert!(parse_ane_stats("nothing here\n").is_null());
    }

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
