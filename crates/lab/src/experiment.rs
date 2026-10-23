//! `tslab gate`, `tslab run-experiment`, `tslab confirm`: the evaluator the
//! autoresearch loop (`program.md`) calls.
//!
//! `run-experiment` is the whole judgment for one hypothesis, in order:
//! integrity (the diff may not touch the evaluator), a frozen copy of the
//! spec, a worker built from the experiment's own tree, the spec's required
//! tests, the quality gate against the baseline arm, the paired A/B, the
//! memory check, and a decision. It never edits code and never flips a
//! default; it only writes the ledger.
//!
//! A KEEP needs a second, held-out look: `tslab confirm` runs the same arms
//! on the spec's `confirm_workload`, once per spec. A change tuned to the
//! main prompts fails there.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

use serde_json::{json, Value};

use crate::args::Args;
use crate::gates::{effective_bounds, judge_quality, GateClass, PromptComparison, QualityResult};
use crate::integrity;
use crate::lock::MachineLock;
use crate::orchestrate::{run_ab, Session};
use crate::paths::lab_dir;
use crate::spec::ExperimentSpec;
use crate::stats::Verdict;
use crate::worker::Mode;
use crate::workload::{PromptSource, Workload};

/// The arm probes' prompt lengths (`qwen38_prefill_arm_probe`).
pub const QUALITY_LENGTHS: [usize; 6] = [64, 200, 500, 1000, 2000, 3000];
/// Greedy steps decoded after each quality prompt, for the token-equality
/// half of the bit-exact gate.
pub const QUALITY_DECODE: usize = 32;

fn load_spec(path: &str) -> Result<(ExperimentSpec, String, String), String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
    let spec = ExperimentSpec::from_json(&text)?;
    let sha = model_io::hash_data(text.as_bytes());
    Ok((spec, text, sha))
}

/// The quality workload: the probe corpus at every length that fits.
pub fn quality_workload(spec: &ExperimentSpec) -> Workload {
    let max = spec.workload.max_context as usize;
    Workload {
        prompts: QUALITY_LENGTHS
            .iter()
            .filter(|&&n| n + QUALITY_DECODE < max)
            .map(|&n| PromptSource::Corpus(n))
            .collect(),
        decode_tokens: QUALITY_DECODE,
        prefill_chunk: spec.workload.prefill_chunk,
        max_context: spec.workload.max_context,
        speculation: spec.workload.speculation.clone(),
    }
}

fn read_logits(path: &Path, vocab_rows: usize) -> Result<Vec<Vec<f64>>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if vocab_rows == 0 || bytes.len() % (2 * vocab_rows) != 0 {
        return Err(format!(
            "{}: size {} is not {vocab_rows} rows of f16",
            path.display(),
            bytes.len()
        ));
    }
    let vocab = bytes.len() / 2 / vocab_rows;
    Ok(bytes
        .chunks_exact(2 * vocab)
        .map(|row| {
            row.chunks_exact(2)
                .map(|b| half::f16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32() as f64)
                .collect()
        })
        .collect())
}

/// Baseline and candidate prefill the probe corpus once each; the candidate
/// is held to the spec's exactness class against the baseline.
pub fn quality_gate(session: &Session, spec: &ExperimentSpec) -> Result<QualityResult, String> {
    let workload = quality_workload(spec);
    let n = workload.prompts.len();
    let a = session.run_once(&spec.baseline, &workload, Mode::Quality, 0, false)?;
    let b = session.run_once(&spec.candidate, &workload, Mode::Quality, 0, false)?;
    for o in [&a, &b] {
        if o.status != "ok" {
            return Err(format!(
                "quality run for {} was {}: {}",
                o.arm,
                o.status,
                o.record["refusal"].as_str().unwrap_or("(see its stderr)")
            ));
        }
    }
    let logits = |o: &crate::orchestrate::Outcome| {
        session
            .exp_dir
            .join("runs")
            .join(format!("{}.logits.f16", o.run_id))
    };
    let pa = a.record["prompts"].as_array().cloned().unwrap_or_default();
    let pb = b.record["prompts"].as_array().cloned().unwrap_or_default();
    // A speculative arm runs the runtime's loop, which keeps its prefill
    // logits to itself. Losslessness is then a question about generated
    // tokens only, so such arms are judged bit-exact on those.
    let speculative = spec.baseline.effective_speculation(&workload).is_some()
        || spec.candidate.effective_speculation(&workload).is_some();
    if speculative {
        if spec.exactness != GateClass::BitExact {
            return Err(
                "speculative arms are gated on generated tokens; declare \"exactness\": \"bit_exact\""
                    .into(),
            );
        }
        let mut tokens_equal = true;
        let prompts = (0..n)
            .map(|i| {
                let same = pa[i]["decode_digest"] == pb[i]["decode_digest"];
                tokens_equal &= same;
                PromptComparison {
                    prompt_tokens: pa[i]["prompt_tokens"].as_u64().unwrap_or(0) as usize,
                    kl: 0.0,
                    max_abs_dlogit: 0.0,
                    top1_agree: same,
                    top5_overlap: 0,
                    logits_bits_equal: same,
                }
            })
            .collect();
        return Ok(judge_quality(
            GateClass::BitExact,
            GateClass::BitExact.default_bounds(),
            prompts,
            tokens_equal,
        ));
    }
    let (la, lb) = (read_logits(&logits(&a), n)?, read_logits(&logits(&b), n)?);
    let mut prompts = Vec::new();
    let mut tokens_equal = true;
    for i in 0..n {
        let c = turbospark_bench::logit_compare::compare(&la[i], &lb[i]);
        tokens_equal &= pa[i]["decode_digest"] == pb[i]["decode_digest"];
        prompts.push(PromptComparison {
            prompt_tokens: pa[i]["prompt_tokens"].as_u64().unwrap_or(0) as usize,
            kl: c.kl,
            max_abs_dlogit: c.max_abs_dlogit,
            top1_agree: c.top1_agree,
            top5_overlap: c.top5_overlap,
            logits_bits_equal: pa[i]["prefill_logits_digest"] == pb[i]["prefill_logits_digest"],
        });
    }
    let bounds = match spec.exactness {
        GateClass::BitExact => spec.exactness.default_bounds(),
        class => effective_bounds(class, spec.quality_bounds)?,
    };
    Ok(judge_quality(spec.exactness, bounds, prompts, tokens_equal))
}

/// Run each `cargo test` argument string in `repo`. Each must pass AND run
/// at least one test: a filter that matches nothing passes vacuously.
pub fn required_tests(repo: &Path, tests: &[String], log_dir: &Path) -> Result<Value, String> {
    let mut results = Vec::new();
    for (i, t) in tests.iter().enumerate() {
        let log = log_dir.join(format!("required-test-{i}.log"));
        let out = Command::new("cargo")
            .current_dir(repo)
            .arg("test")
            .arg("--release")
            .args(t.split_whitespace())
            .output()
            .map_err(|e| format!("cargo test {t}: {e}"))?;
        let text = format!(
            "{}\n{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
        let _ = std::fs::write(&log, &text);
        let passed: u64 = text
            .lines()
            .filter_map(|l| l.strip_prefix("test result: ok. "))
            .filter_map(|l| l.split_whitespace().next()?.parse::<u64>().ok())
            .sum();
        let ok = out.status.success() && passed > 0;
        results
            .push(json!({"args": t, "ok": ok, "passed": passed, "log": log.display().to_string()}));
        if !ok {
            return Err(format!(
                "required test `cargo test --release {t}` {} (log {})",
                if out.status.success() {
                    "ran no tests"
                } else {
                    "failed"
                },
                log.display()
            ));
        }
    }
    Ok(json!(results))
}

/// Build `tslab-worker` from the experiment tree, so the candidate code is
/// what gets measured.
fn build_worker(repo: &Path, spec: &ExperimentSpec, log_dir: &Path) -> Result<PathBuf, String> {
    let ane = [&spec.baseline, &spec.candidate]
        .iter()
        .any(|a| a.env.contains_key("TURBOSPARK_ANE_PREFILL"));
    let mut cmd = Command::new("cargo");
    cmd.current_dir(repo).args([
        "build",
        "--release",
        "-p",
        "turbospark-lab",
        "--bin",
        "tslab-worker",
    ]);
    if ane {
        cmd.args(["--features", "ane-prefill"]);
    }
    let out = cmd.output().map_err(|e| format!("cargo build: {e}"))?;
    let _ = std::fs::write(
        log_dir.join("worker-build.log"),
        String::from_utf8_lossy(&out.stderr).as_bytes(),
    );
    if !out.status.success() {
        return Err(format!(
            "worker build failed (log {})",
            log_dir.join("worker-build.log").display()
        ));
    }
    let target = std::env::var_os("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| repo.join("target"));
    Ok(target.join("release").join("tslab-worker"))
}

struct Prepared {
    spec: ExperimentSpec,
    sha: String,
    repo: PathBuf,
    session: Session,
    trusted: bool,
}

fn prepare(args: &mut Args, purpose: &str) -> Result<(Prepared, MachineLock), String> {
    let spec_path = args.required("--spec")?;
    let dev = args.flag("--dev-untrusted");
    let worker_override = args.take("--worker")?;
    let base = args.take("--base")?;
    let rules = crate::cli::host_rules_from(args)?;
    let (spec, text, sha) = load_spec(&spec_path)?;
    let lab = lab_dir();
    let trusted = match integrity::trusted_exe(&lab) {
        Ok(_) => true,
        Err(e) if dev => {
            eprintln!("warning: {e}");
            false
        }
        Err(e) => return Err(e),
    };
    let repo = integrity::repo_root(&std::env::current_dir().map_err(|e| e.to_string())?)?;
    let base = match base {
        Some(b) => b,
        None => integrity::default_base(&repo)?,
    };
    let changes = integrity::changes_since(&repo, &base)?;
    let bad = integrity::violations(&changes);
    if !bad.is_empty() {
        return Err(format!(
            "the experiment's diff from {base} touches evaluator inputs:\n  {}",
            bad.join("\n  ")
        ));
    }
    let lock = MachineLock::acquire(&lab, &format!("tslab {purpose} {}", spec.name))?;
    let timeout = Duration::from_secs(60 * u64::from(spec.time_budget_min));
    let mut session = Session::new(&lab, &spec.name, &spec.model, rules, timeout)?;
    let frozen = session.exp_dir.join("specs").join(format!("{sha}.json"));
    if !frozen.exists() {
        crate::atomic::write_new(&frozen, text.as_bytes())?;
    }
    let worker = match worker_override {
        Some(w) => PathBuf::from(w),
        None => build_worker(&repo, &spec, &session.exp_dir)?,
    };
    session = session.with_worker(worker);
    session.ledger.append(
        "experiment_start",
        json!({"purpose": purpose, "spec_sha256": sha, "spec": spec, "base": base,
               "changes": changes.iter().map(|c| format!("{} {}", c.status, c.path)).collect::<Vec<_>>(),
               "trusted": trusted,
               "tslab_exe_sha256": std::env::current_exe().ok().as_deref().and_then(crate::provenance::sha256_file),
               "worker": session.worker.display().to_string(),
               "worker_sha256": crate::provenance::sha256_file(&session.worker)}),
    )?;
    Ok((
        Prepared {
            spec,
            sha,
            repo,
            session,
            trusted,
        },
        lock,
    ))
}

fn print_quality(q: &QualityResult) {
    println!(
        "quality ({:?}): {}  mean KL {:.3e}, top-1 {:.2}, bits equal {}, tokens equal {}",
        q.class,
        if q.passed { "PASS" } else { "FAIL" },
        q.mean_kl,
        q.top1_fraction,
        q.all_bits_equal,
        q.tokens_equal
    );
    for r in &q.reasons {
        println!("  {r}");
    }
}

pub fn gate_cmd(mut args: Args) -> Result<(), String> {
    let (p, _lock) = prepare(&mut args, "gate")?;
    args.finish()?;
    let tests = required_tests(&p.repo, &p.spec.required_tests, &p.session.exp_dir)?;
    let q = quality_gate(&p.session, &p.spec)?;
    print_quality(&q);
    p.session.ledger.append(
        "gate",
        json!({"spec_sha256": p.sha, "required_tests": tests, "quality": q}),
    )?;
    if q.passed {
        Ok(())
    } else {
        Err("quality gate failed".into())
    }
}

/// The decision values the ledger records.
pub const KEEP_PENDING_CONFIRM: &str = "keep_pending_confirm";
pub const DISCARD: &str = "discard";

pub fn run_experiment_cmd(mut args: Args) -> Result<(), String> {
    let started = Instant::now();
    let (p, _lock) = prepare(&mut args, "run-experiment")?;
    args.finish()?;
    let knob = p.spec.require_knob()?.to_string();
    let deadline = started + Duration::from_secs(60 * u64::from(p.spec.time_budget_min));
    let ledger = &p.session.ledger;

    let finish = |decision: &str, reason: String, extra: Value| -> Result<(), String> {
        let mut rec = json!({"spec_sha256": p.sha, "knob_env": knob, "decision": decision,
                             "reason": reason, "trusted": p.trusted,
                             "elapsed_s": started.elapsed().as_secs_f64()});
        if let (Value::Object(m), Value::Object(e)) = (&mut rec, extra) {
            m.extend(e);
        }
        let id = ledger.append("experiment", rec)?;
        println!("DECISION: {decision} ({reason}); ledger id {id}");
        Ok(())
    };

    let tests = match required_tests(&p.repo, &p.spec.required_tests, &p.session.exp_dir) {
        Ok(t) => t,
        Err(e) => return finish(DISCARD, format!("required tests: {e}"), json!({})),
    };
    let q = match quality_gate(&p.session, &p.spec) {
        Ok(q) => q,
        Err(e) => {
            return finish(
                DISCARD,
                format!("quality gate could not run: {e}"),
                json!({"required_tests": tests}),
            )
        }
    };
    print_quality(&q);
    if !q.passed {
        return finish(
            DISCARD,
            format!("quality gate failed: {}", q.reasons.join("; ")),
            json!({"required_tests": tests, "quality": q}),
        );
    }
    let (floor, memory_noise) =
        crate::cli::null_floor(&p.session, &p.spec.workload, p.spec.primary_metric)?;
    let (verdict, pairs) = run_ab(
        &p.session,
        &p.spec.baseline,
        &p.spec.candidate,
        &p.spec.workload,
        p.spec.primary_metric,
        p.spec.min_effect.max(floor),
        p.spec.min_pairs,
        p.spec.max_pairs,
        Some(deadline),
    )?;
    crate::report::print_verdict(
        &format!("{} vs {}", p.spec.baseline.name, p.spec.candidate.name),
        &verdict,
    );
    let memory = crate::cli::memory_check(&pairs, p.spec.memory_allowance_bytes, memory_noise);
    println!("  memory: {memory}");
    let run_ids: Vec<(String, String)> = pairs
        .iter()
        .map(|(a, b)| (a.run_id.clone(), b.run_id.clone()))
        .collect();
    let extra = json!({"required_tests": tests, "quality": q, "verdict": verdict,
                       "null_floor": floor, "memory": memory, "pairs": run_ids});
    let (decision, reason) = if !p.trusted {
        (
            DISCARD,
            "evaluator was not the trusted install (--dev-untrusted)".to_string(),
        )
    } else if verdict.verdict != Verdict::Win {
        (DISCARD, format!("verdict {:?}", verdict.verdict))
    } else if memory["passed"] != true {
        (
            DISCARD,
            "peak footprint grew past the allowance".to_string(),
        )
    } else {
        (
            KEEP_PENDING_CONFIRM,
            "WIN with gates passed; run tslab confirm".to_string(),
        )
    };
    finish(decision, reason, extra)
}

pub fn confirm_cmd(mut args: Args) -> Result<(), String> {
    let (p, _lock) = prepare(&mut args, "confirm")?;
    args.finish()?;
    let ledger = &p.session.ledger;
    let experiments = ledger.of_kind("experiment")?;
    let pending = experiments
        .iter()
        .rev()
        .find(|r| r["spec_sha256"] == p.sha.as_str())
        .is_some_and(|r| r["decision"] == KEEP_PENDING_CONFIRM);
    if !pending {
        return Err("no keep_pending_confirm run-experiment record for this exact spec".into());
    }
    if ledger
        .of_kind("confirm")?
        .iter()
        .any(|r| r["spec_sha256"] == p.sha.as_str())
    {
        return Err(
            "this spec already used its one confirm pass; a new hypothesis needs a new spec".into(),
        );
    }
    let workload = p
        .spec
        .confirm_workload
        .clone()
        .ok_or("the spec has no confirm_workload")?;
    let (verdict, _pairs) = run_ab(
        &p.session,
        &p.spec.baseline,
        &p.spec.candidate,
        &workload,
        p.spec.primary_metric,
        p.spec.min_effect,
        p.spec.min_pairs,
        p.spec.max_pairs,
        None,
    )?;
    crate::report::print_verdict("confirm (held-out prompts)", &verdict);
    // The held-out look must not contradict the main one: a LOSS or a
    // negative median there means the gain was specific to the main prompts.
    let confirmed = p.trusted && verdict.verdict != Verdict::Loss && verdict.median > 0.0;
    let decision = if confirmed { "keep" } else { DISCARD };
    let id = ledger.append(
        "confirm",
        json!({"spec_sha256": p.sha, "verdict": verdict, "decision": decision}),
    )?;
    println!("CONFIRM: {decision}; ledger id {id}");
    Ok(())
}
