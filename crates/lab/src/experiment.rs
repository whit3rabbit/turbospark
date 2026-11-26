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
use crate::gates::{
    effective_bounds, judge_decode, judge_quality, DecodeResult, GateClass, PromptComparison,
    QualityBounds, QualityResult, DECODE_TRACE_TOKENS,
};
use crate::gates::{judge_divergence, judge_text_classes, Divergence};
use crate::integrity;
use crate::lock::MachineLock;
use crate::logit_trace::TraceReader;
use crate::orchestrate::{run_ab, AbRun, Outcome, Session};
use crate::paths::lab_dir;
use crate::schema::CmdOutput;
use crate::spec::{ExperimentSpec, SecondaryGuard};
use crate::stats::{decide_noninferior, NonInferiority, NonInferiorityReport, Verdict};
use crate::worker::Mode;
use crate::workload::{Arm, PromptSource, Workload};

/// The arm probes' prompt lengths (`qwen38_prefill_arm_probe`).
pub const QUALITY_LENGTHS: [usize; 6] = [64, 200, 500, 1000, 2000, 3000];
/// Corpus prompt lengths of the forced decode trace.
pub const DECODE_TRACE_LENGTHS: [usize; 3] = [64, 500, 2000];
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
pub fn quality_workload(w: &Workload) -> Workload {
    let max = w.max_context as usize;
    Workload {
        prompts: QUALITY_LENGTHS
            .iter()
            .filter(|&&n| n + QUALITY_DECODE < max)
            .map(|&n| PromptSource::Corpus(n))
            .collect(),
        decode_tokens: QUALITY_DECODE,
        prefill_chunk: w.prefill_chunk,
        max_context: w.max_context,
        speculation: w.speculation.clone(),
        audio: None,
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

/// What the quality gate compares: two arms over one workload, held to one
/// exactness class. `tslab gate` takes it from a spec, `tslab sweep --gate`
/// from a sweep.
pub struct GateArms<'a> {
    pub baseline: &'a Arm,
    pub candidate: &'a Arm,
    pub workload: &'a Workload,
    pub exactness: GateClass,
    pub quality_bounds: Option<QualityBounds>,
}

impl<'a> GateArms<'a> {
    pub fn from_spec(spec: &'a ExperimentSpec) -> Self {
        Self {
            baseline: &spec.baseline,
            candidate: &spec.candidate,
            workload: &spec.workload,
            exactness: spec.exactness,
            quality_bounds: spec.quality_bounds,
        }
    }
}

/// Baseline and candidate prefill the probe corpus once each; the candidate
/// is held to the spec's exactness class against the baseline.
pub fn quality_gate(session: &Session, spec: &ExperimentSpec) -> Result<QualityResult, String> {
    quality_gate_arms(session, &GateArms::from_spec(spec))
}

/// [`quality_gate`] over any two arms.
pub fn quality_gate_arms(session: &Session, spec: &GateArms) -> Result<QualityResult, String> {
    if spec.workload.audio.is_some() {
        return audio_quality_gate(session, spec);
    }
    // A workload that decodes gets the forced-decode gate too: the prefill
    // comparison below never sees the decode path (LAB-01).
    let result = prefill_quality_gate(session, spec)?;
    if spec.workload.decode_tokens == 0 || runs_the_runtime_loop(spec, spec.workload) {
        return Ok(result);
    }
    Ok(result.with_decode(decode_trace_gate(session, spec)?))
}

/// A speculative arm, and any chat, runs the runtime's loop, which keeps its
/// logits to itself. Losslessness is then a question about generated tokens.
fn runs_the_runtime_loop(spec: &GateArms, workload: &Workload) -> bool {
    spec.baseline.effective_speculation(workload).is_some()
        || spec.candidate.effective_speculation(workload).is_some()
        || workload.prompts.iter().any(PromptSource::is_chat)
}

/// The chat-templated prompts of the forced trace, its second text class.
pub const DECODE_TRACE_CHAT: [&str; 2] = ["short-explanation", "medium-review"];

/// The text class a trace prompt belongs to.
fn text_class(p: &PromptSource) -> &'static str {
    match p {
        PromptSource::Corpus(_) | PromptSource::TextFile(_) => "documents",
        _ => "chat",
    }
}

/// The prompts and decode length of the forced trace.
pub fn decode_trace_workload(w: &Workload) -> Workload {
    let max = w.max_context as usize;
    // Three of the quality corpus lengths: short, medium and long fill. A
    // context too small for the full trace decodes what fits.
    let steps = DECODE_TRACE_TOKENS.min(max.saturating_sub(DECODE_TRACE_LENGTHS[0] + 1));
    Workload {
        prompts: DECODE_TRACE_LENGTHS
            .iter()
            .filter(|&&n| n + steps < max)
            .map(|&n| PromptSource::Corpus(n))
            .chain(
                // A chat prompt is a few hundred tokens at most.
                DECODE_TRACE_CHAT
                    .iter()
                    .filter(|_| 512 + steps < max)
                    .map(|&c| PromptSource::Protocol(c.into())),
            )
            .collect(),
        decode_tokens: steps,
        ..quality_workload(w)
    }
}

/// Compare a reference trace with an arm's trace of the same forced ids,
/// row by row, grouped by prompt.
pub fn compare_traces(
    reference: &Path,
    arm: &Path,
    prompt_tokens: &[usize],
    forced: &[Vec<i32>],
) -> Result<Vec<(usize, Vec<crate::kl_stats::Position>)>, String> {
    let (mut a, mut b) = (TraceReader::open(reference)?, TraceReader::open(arm)?);
    let rows: usize = forced.iter().map(Vec::len).sum();
    if a.rows != rows as u64 || b.rows != rows as u64 || a.vocab != b.vocab {
        return Err(format!(
            "decode traces hold {} and {} rows (vocab {} and {}), {rows} were forced",
            a.rows, b.rows, a.vocab, b.vocab
        ));
    }
    let mut out = Vec::new();
    for (ids, &n) in forced.iter().zip(prompt_tokens) {
        let mut positions = Vec::with_capacity(ids.len());
        for &id in ids {
            let (x, y) = a
                .next_row()?
                .zip(b.next_row()?)
                .ok_or("decode trace ended early")?;
            positions.push(crate::kl_stats::position(&x, &y, id as usize)?);
        }
        out.push((n, positions));
    }
    Ok(out)
}

/// One forced-decode run: its trace file, the ids it decoded and each
/// prompt's token count.
pub struct Traced {
    pub trace: PathBuf,
    pub ids: Vec<Vec<i32>>,
    pub prompt_tokens: Vec<usize>,
}

/// Run `arm` over `workload` in the forced mode. `forced` is `None` for the
/// reference, whose own greedy ids become the sequence.
pub fn traced_run(
    session: &Session,
    arm: &Arm,
    workload: &Workload,
    forced: Option<Vec<Vec<i32>>>,
) -> Result<Traced, String> {
    let o = session.run_forced(arm, workload, Mode::ForcedDecode, 0, false, forced)?;
    if o.status != "ok" {
        return Err(format!(
            "decode trace for {} was {}: {}",
            o.arm,
            o.status,
            o.record["refusal"].as_str().unwrap_or("(see its stderr)")
        ));
    }
    let prompts = o.record["prompts"].as_array().cloned().unwrap_or_default();
    if prompts.len() != workload.prompts.len() {
        return Err(format!("decode trace for {}: prompts missing", o.arm));
    }
    Ok(Traced {
        trace: session
            .exp_dir
            .join("runs")
            .join(format!("{}.trace.f16", o.run_id)),
        ids: prompts
            .iter()
            .map(|p| serde_json::from_value(p["decode_ids"].clone()).map_err(|e| e.to_string()))
            .collect::<Result<_, _>>()?,
        prompt_tokens: prompts
            .iter()
            .map(|p| p["prompt_tokens"].as_u64().unwrap_or(0) as usize)
            .collect(),
    })
}

/// The teacher-forced decode gate: the baseline records its greedy ids and
/// per-step logits, the candidate replays the ids, and every position is
/// compared. The traces are large and removed once judged.
pub fn decode_trace_gate(session: &Session, spec: &GateArms) -> Result<DecodeResult, String> {
    if runs_the_runtime_loop(spec, spec.workload) {
        return Err("the decode trace takes plain prompts, no speculation or chat".into());
    }
    let workload = decode_trace_workload(spec.workload);
    let a = traced_run(session, spec.baseline, &workload, None)?;
    let b = traced_run(session, spec.candidate, &workload, Some(a.ids.clone()))?;
    let compared = compare_traces(&a.trace, &b.trace, &a.prompt_tokens, &a.ids);
    for t in [&a.trace, &b.trace] {
        let _ = std::fs::remove_file(t);
    }
    let bounds = match spec.exactness {
        GateClass::BitExact | GateClass::TextExact | GateClass::TokenDivergence => {
            spec.exactness.default_bounds()
        }
        class => effective_bounds(class, spec.quality_bounds)?,
    };
    let compared = compared?;
    let labels: Vec<&str> = workload.prompts.iter().map(text_class).collect();
    let judged = judge_decode(spec.exactness, bounds, &compared);
    Ok(judge_text_classes(
        judged,
        spec.exactness,
        bounds,
        &labels,
        &compared,
    ))
}

/// The final-position prefill comparison over the quality corpus.
pub fn prefill_quality_gate(session: &Session, spec: &GateArms) -> Result<QualityResult, String> {
    let workload = quality_workload(spec.workload);
    let n = workload.prompts.len();
    let a = session.run_once(spec.baseline, &workload, Mode::Quality, 0, false)?;
    let b = session.run_once(spec.candidate, &workload, Mode::Quality, 0, false)?;
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
    if spec.exactness == GateClass::TokenDivergence {
        return divergence_gate(session, spec, &workload);
    }
    // Runtime-loop workloads are judged bit-exact on generated tokens.
    if runs_the_runtime_loop(spec, &workload) {
        if spec.exactness != GateClass::BitExact {
            return Err(
                "speculative arms and chats are gated on generated tokens; declare \
                 \"exactness\": \"bit_exact\""
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
    let result = judge_quality(spec.exactness, bounds, prompts, tokens_equal);
    Ok(match position_comparison(session, spec, &a, &b, bounds)? {
        Some(positions) => result.with_positions(positions),
        None => result,
    })
}

/// The chunk-end prefill logits of the two quality runs (LAB-02). `None` when
/// the workload prefills token by token or one chunk covers every prompt.
fn position_comparison(
    session: &Session,
    spec: &GateArms,
    a: &Outcome,
    b: &Outcome,
    bounds: QualityBounds,
) -> Result<Option<DecodeResult>, String> {
    let path = |o: &Outcome| {
        session
            .exp_dir
            .join("runs")
            .join(format!("{}.logits.positions", o.run_id))
    };
    let targets: Vec<Vec<i32>> = a.record["prompts"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|p| serde_json::from_value(p["position_targets"].clone()).ok())
        .collect();
    let rows: usize = targets.iter().map(Vec::len).sum();
    let result = if rows == 0 {
        None
    } else {
        let labels = vec![0usize; targets.len()];
        let compared = compare_traces(&path(a), &path(b), &labels, &targets)?;
        Some(judge_decode(spec.exactness, bounds, &compared))
    };
    for o in [a, b] {
        let _ = std::fs::remove_file(path(o));
    }
    Ok(result)
}

/// Verify-width changes: both arms under the probe, compared where their
/// outputs part, against the model's own run-to-run floor.
fn divergence_gate(
    session: &Session,
    spec: &GateArms,
    workload: &Workload,
) -> Result<QualityResult, String> {
    let run = |arm: &Arm| -> Result<Vec<Value>, String> {
        let o = session.run_probe(arm, workload, None)?;
        if o.status != "ok" {
            return Err(format!(
                "divergence run for {} was {}: {}",
                o.arm,
                o.status,
                o.record["refusal"].as_str().unwrap_or("(see its stderr)")
            ));
        }
        Ok(o.record["prompts"].as_array().cloned().unwrap_or_default())
    };
    let (a, b) = (run(spec.baseline)?, run(spec.candidate)?);
    let ids = |p: &Value| -> Vec<i64> {
        serde_json::from_value(p["decode_ids"].clone()).unwrap_or_default()
    };
    let divergences: Vec<Divergence> = a
        .iter()
        .zip(&b)
        .map(|(x, y)| {
            let first = crate::checks::first_divergence(&ids(x), &ids(y));
            Divergence {
                first,
                baseline_margin: first.and_then(|i| x["spec_probe"]["margins"][i].as_f64()),
            }
        })
        .collect();
    let floor = crate::checks::noise_floor(&lab_dir(), &session.model);
    let (passed, reasons) = judge_divergence(&divergences, floor);
    let mut result = judge_quality(
        GateClass::BitExact,
        GateClass::BitExact.default_bounds(),
        Vec::new(),
        divergences.iter().all(|d| d.first.is_none()),
    );
    result.class = GateClass::TokenDivergence;
    result.all_bits_equal = false;
    result.passed = passed && a.len() == b.len();
    result.reasons = reasons;
    result
        .reasons
        .push(format!("floor {floor:.2}; divergences {divergences:?}"));
    Ok(result)
}

/// An audio spec runs its job once per arm and compares outputs: the
/// waveform or transcript bytes for `bit_exact`, the transcript text for
/// `text_exact`. No other class has an audio bound yet, so none is accepted.
fn audio_quality_gate(session: &Session, spec: &GateArms) -> Result<QualityResult, String> {
    if !matches!(spec.exactness, GateClass::BitExact | GateClass::TextExact) {
        return Err(
            "an audio spec declares bit_exact or text_exact; no KL bound exists for audio".into(),
        );
    }
    let a = session.run_once(spec.baseline, spec.workload, Mode::Timing, 0, false)?;
    let b = session.run_once(spec.candidate, spec.workload, Mode::Timing, 0, false)?;
    for o in [&a, &b] {
        if o.status != "ok" {
            return Err(format!(
                "audio quality run for {} was {}: {}",
                o.arm,
                o.status,
                o.record["refusal"].as_str().unwrap_or("(see its stderr)")
            ));
        }
    }
    let (da, db) = (&a.record["audio"], &b.record["audio"]);
    let bytes_equal = da["output_digest"].is_string() && da["output_digest"] == db["output_digest"];
    // A music job has no text: its "text" is its bytes.
    let text_equal = if da["text_digest"].is_string() {
        da["text_digest"] == db["text_digest"]
    } else {
        bytes_equal
    };
    let cmp = PromptComparison {
        prompt_tokens: 0,
        kl: 0.0,
        max_abs_dlogit: 0.0,
        top1_agree: text_equal,
        top5_overlap: 0,
        logits_bits_equal: bytes_equal,
    };
    Ok(judge_quality(
        spec.exactness,
        spec.exactness.default_bounds(),
        vec![cmp],
        text_equal,
    ))
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
    // The declared class must not be looser than the install's own.
    let model_dir = catalog::resolve_model_arg(&spec.model);
    if model_dir.join("manifest.json").is_file() {
        let arch = repack::peek_manifest_arch(&model_dir)?;
        spec.exactness.check_declared(arch.num_experts > 0)?;
    }
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
    crate::say!(
        "quality ({:?}): {}  mean KL {:.3e}, top-1 {:.2}, bits equal {}, tokens equal {}",
        q.class,
        if q.passed { "PASS" } else { "FAIL" },
        q.mean_kl,
        q.top1_fraction,
        q.all_bits_equal,
        q.tokens_equal
    );
    for r in &q.reasons {
        crate::say!("  {r}");
    }
    if let Some(d) = &q.decode {
        let s = &d.overall;
        crate::say!(
            "  decode trace: {} positions, mean KL {:.3e} (p99 {:.3e}), {} flips ({} sure), \
             PPL {:.3} to {:.3}, NLL delta {:+.4} +/- {:.4}",
            s.tokens,
            s.mean_kl,
            s.p99_kl,
            s.flips,
            s.large_margin_flips,
            s.ref_ppl,
            s.arm_ppl,
            s.nll_delta,
            s.nll_delta_se
        );
        for (class, c) in &d.classes {
            crate::say!(
                "    {class}: {} positions, mean KL {:.3e}, {} flips ({} sure), NLL delta {:+.4}",
                c.tokens,
                c.mean_kl,
                c.flips,
                c.large_margin_flips,
                c.nll_delta
            );
        }
    }
}

pub fn gate_cmd(mut args: Args) -> Result<CmdOutput, String> {
    let force_trace = args.flag("--decode-trace");
    // `--noise-arms`: also run each arm against itself, so the candidate's
    // KL is read beside the run-to-run spread (LAB-03).
    let noise_arms = args.flag("--noise-arms");
    let (mut p, _lock) = prepare(&mut args, "gate")?;
    args.finish()?;
    if force_trace && p.spec.workload.decode_tokens == 0 {
        p.spec.workload.decode_tokens = DECODE_TRACE_TOKENS;
    }
    let tests = required_tests(&p.repo, &p.spec.required_tests, &p.session.exp_dir)?;
    let q = quality_gate(&p.session, &p.spec)?;
    print_quality(&q);
    let noise = noise_arms
        .then(|| self_comparisons(&p.session, &p.spec))
        .transpose()?;
    let floor = print_floor(&p.spec, &q);
    let rec = json!({"spec_sha256": p.sha, "required_tests": tests, "quality": q,
                     "noise_arms": noise, "floor": floor});
    p.session.ledger.append("gate", rec.clone())?;
    if q.passed {
        Ok(CmdOutput::ok(rec))
    } else {
        // Exit 1 as before, but keep the gate's reasons in the envelope.
        eprintln!("tslab gate: quality gate failed");
        Ok(CmdOutput::with_code(1, rec))
    }
}

/// Baseline against baseline and candidate against candidate: the spread a
/// result between the two arms has to be read beside.
fn self_comparisons(session: &Session, spec: &ExperimentSpec) -> Result<Value, String> {
    let mut out = serde_json::Map::new();
    for arm in [&spec.baseline, &spec.candidate] {
        let again = Arm {
            name: format!("{}-again", arm.name),
            ..arm.clone()
        };
        let q = quality_gate_arms(
            session,
            &GateArms {
                baseline: arm,
                candidate: &again,
                ..GateArms::from_spec(spec)
            },
        )?;
        let kl = q.decode.as_ref().map(|d| d.overall.mean_kl);
        crate::say!(
            "  noise {} against itself: prefill mean KL {:.3e}, bits equal {}{}",
            arm.name,
            q.mean_kl,
            q.all_bits_equal,
            kl.map_or(String::new(), |k| format!(", decode mean KL {k:.3e}"))
        );
        out.insert(arm.name.clone(), json!(q));
    }
    Ok(Value::Object(out))
}

/// "limit L, floor F, margin M" when the family has a recorded floor
/// (LAB-04). The floor explains the margin; it never moves the limit.
fn print_floor(spec: &ExperimentSpec, q: &QualityResult) -> Option<Value> {
    let dir = catalog::resolve_model_arg(&spec.model);
    let family = repack::peek_manifest_arch(&dir).ok()?.family;
    let floor = crate::checks::latest_floor(&lab_dir(), family.as_str())?;
    let f = floor["mean_kl"].as_f64()?;
    crate::say!(
        "  limit {:.1e}, family floor {f:.3e}, margin {:.3e}{}",
        q.bounds.mean_kl,
        q.bounds.mean_kl - f,
        if floor["bits_identical"] == false {
            " (this family's baseline is not run-to-run exact)"
        } else {
            ""
        }
    );
    Some(json!({"family": family.as_str(), "mean_kl": f,
                "bits_identical": floor["bits_identical"], "record": floor["id"]}))
}

/// The decision values the ledger records.
pub const KEEP_PENDING_CONFIRM: &str = "keep_pending_confirm";
pub const DISCARD: &str = "discard";
/// No evidence either way: too few clean pairs, usually a busy host. NOT a
/// reason to revert the change; the idea is untested, not refuted.
pub const INCONCLUSIVE: &str = "inconclusive";

/// Process exit codes of `run-experiment` and `confirm`, so a script can
/// branch on the outcome. 0 is a keep, 1 is an error that reached no
/// decision, and 2 is a usage error (`tslab` with no arguments).
pub const EXIT_DISCARD: i32 = 3;
pub const EXIT_INCONCLUSIVE: i32 = 4;

/// The exit code for a recorded decision (`keep` is `confirm`'s).
pub fn exit_code_for(decision: &str) -> i32 {
    match decision {
        KEEP_PENDING_CONFIRM | "keep" => 0,
        DISCARD => EXIT_DISCARD,
        INCONCLUSIVE => EXIT_INCONCLUSIVE,
        _ => 1,
    }
}

/// What `run-experiment` records once the A/B is done.
///
/// `Inconclusive` is its own decision, not a discard: `program.md` tells the
/// loop to revert on a discard, and reverting a change because the host was
/// busy throws away work no measurement has judged.
pub fn decide(
    trusted: bool,
    verdict: Verdict,
    memory_passed: bool,
    clean_pairs: usize,
    min_pairs: usize,
    stopped: Option<&str>,
) -> (&'static str, String) {
    if !trusted {
        return (
            DISCARD,
            "evaluator was not the trusted install (--dev-untrusted)".to_string(),
        );
    }
    match verdict {
        Verdict::Inconclusive => {
            let mut reason = format!(
                "only {clean_pairs} clean pair(s), need {min_pairs}; the idea is untested, \
                 not refuted. Do NOT revert. Fix the condition and re-run"
            );
            if let Some(why) = stopped {
                reason.push_str(&format!(" ({why})"));
            }
            (INCONCLUSIVE, reason)
        }
        Verdict::Win if !memory_passed => (
            DISCARD,
            "peak footprint grew past the allowance".to_string(),
        ),
        Verdict::Win => (
            KEEP_PENDING_CONFIRM,
            "WIN with gates passed; run tslab confirm".to_string(),
        ),
        other => (DISCARD, format!("verdict {other:?}")),
    }
}

/// The decision for a spec that asks "not worse than `limit`" (LAB-05).
pub fn decide_noninferiority(
    trusted: bool,
    report: &NonInferiorityReport,
    memory_passed: bool,
) -> (&'static str, String) {
    if !trusted {
        return (
            DISCARD,
            "evaluator was not the trusted install (--dev-untrusted)".to_string(),
        );
    }
    let span = format!(
        "one-sided bounds {:+.2}% to {:+.2}% against a {:.2}% limit",
        report.lower * 100.0,
        report.upper * 100.0,
        report.limit * 100.0
    );
    match report.result {
        NonInferiority::NotWorse if !memory_passed => (
            DISCARD,
            "peak footprint grew past the allowance".to_string(),
        ),
        NonInferiority::NotWorse => (
            KEEP_PENDING_CONFIRM,
            format!("not worse: {span}; run tslab confirm"),
        ),
        NonInferiority::Worse => (DISCARD, format!("worse: {span}")),
        // More pairs "until it passes" is not allowed: a larger fixed budget
        // is a new spec.
        NonInferiority::Inconclusive => (
            INCONCLUSIVE,
            format!(
                "{} clean pairs, {span}; declare a larger budget",
                report.pairs
            ),
        ),
    }
}

/// The second metric's median improvement over the clean pairs, and whether
/// it stayed within the allowed regression (LAB-33). No reading fails.
pub fn secondary_guard(pairs: &[(Outcome, Outcome)], guard: SecondaryGuard) -> Value {
    let improvements: Vec<f64> = pairs
        .iter()
        .filter(|(a, b)| a.eligible && b.eligible)
        .filter_map(|(a, b)| {
            crate::stats::improvement(
                a.metric(guard.metric)?,
                b.metric(guard.metric)?,
                guard.metric.direction(),
            )
        })
        .collect();
    let median = crate::stats::median(&improvements);
    json!({
        "metric": guard.metric,
        "max_regression": guard.max_regression,
        "pairs": improvements.len(),
        "median": median.is_finite().then_some(median),
        "passed": !improvements.is_empty() && median >= -guard.max_regression,
    })
}

/// A guard that failed turns a KEEP into a DISCARD; it never rescues one.
pub fn apply_guard(
    decision: (&'static str, String),
    guard: Option<&Value>,
) -> (&'static str, String) {
    match guard {
        Some(g) if decision.0 == KEEP_PENDING_CONFIRM && g["passed"] != true => (
            DISCARD,
            format!(
                "secondary guard failed: {g}; the primary result was: {}",
                decision.1
            ),
        ),
        _ => decision,
    }
}

pub fn run_experiment_cmd(mut args: Args) -> Result<CmdOutput, String> {
    let started = Instant::now();
    let (p, _lock) = prepare(&mut args, "run-experiment")?;
    args.finish()?;
    let knob = p.spec.require_knob()?.to_string();
    let deadline = started + Duration::from_secs(60 * u64::from(p.spec.time_budget_min));
    let ledger = &p.session.ledger;

    let finish = |decision: &str, reason: String, extra: Value| -> Result<CmdOutput, String> {
        let mut rec = json!({"spec_sha256": p.sha, "knob_env": knob, "decision": decision,
                             "reason": reason, "trusted": p.trusted,
                             "elapsed_s": started.elapsed().as_secs_f64()});
        if let (Value::Object(m), Value::Object(e)) = (&mut rec, extra) {
            m.extend(e);
        }
        let id = ledger.append("experiment", rec.clone())?;
        crate::say!("DECISION: {decision} ({reason}); ledger id {id}");
        Ok(CmdOutput::with_code(exit_code_for(decision), rec))
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
    let AbRun {
        verdict,
        pairs,
        stopped,
    } = run_ab(
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
    crate::say!("  memory: {memory}");
    let run_ids: Vec<(String, String)> = pairs
        .iter()
        .map(|(a, b)| (a.run_id.clone(), b.run_id.clone()))
        .collect();
    let noninferiority = p
        .spec
        .noninferiority_limit
        .map(|limit| decide_noninferior(&verdict.improvements, limit));
    let guard = p.spec.secondary_guard.map(|g| secondary_guard(&pairs, g));
    if let Some(g) = &guard {
        crate::say!("  secondary guard: {g}");
    }
    let extra = json!({"required_tests": tests, "quality": q, "verdict": verdict,
                       "null_floor": floor, "memory": memory, "pairs": run_ids,
                       "stopped": stopped, "noninferiority": noninferiority,
                       "secondary_guard": guard, "also_changes": p.spec.also_changes});
    let decided = match &noninferiority {
        Some(report) => decide_noninferiority(p.trusted, report, memory["passed"] == true),
        None => decide(
            p.trusted,
            verdict.verdict,
            memory["passed"] == true,
            verdict.improvements.len(),
            p.spec.min_pairs,
            stopped.as_deref(),
        ),
    };
    let (decision, reason) = apply_guard(decided, guard.as_ref());
    finish(decision, reason, extra)
}

pub fn confirm_cmd(mut args: Args) -> Result<CmdOutput, String> {
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
    let AbRun { verdict, pairs, .. } = run_ab(
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
    // A non-inferiority spec was never claiming a gain: its held-out median
    // only has to stay inside the limit it declared.
    let held = match p.spec.noninferiority_limit {
        Some(limit) => verdict.median >= -limit,
        None => verdict.median > 0.0,
    };
    let guard = p.spec.secondary_guard.map(|g| secondary_guard(&pairs, g));
    let confirmed = p.trusted
        && verdict.verdict != Verdict::Loss
        && held
        && guard.as_ref().is_none_or(|g| g["passed"] == true);
    let decision = if confirmed { "keep" } else { DISCARD };
    let rec = json!({"spec_sha256": p.sha, "verdict": verdict, "decision": decision,
                     "secondary_guard": guard});
    let id = ledger.append("confirm", rec.clone())?;
    crate::say!("CONFIRM: {decision}; ledger id {id}");
    Ok(CmdOutput::with_code(exit_code_for(decision), rec))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn d(verdict: Verdict, memory: bool, stopped: Option<&str>) -> (&'static str, String) {
        decide(true, verdict, memory, 0, 3, stopped)
    }

    /// The bug this exists for: `program.md` reverts on a discard, and a busy
    /// host produced a "discard" for an idea nothing had measured.
    #[test]
    fn an_inconclusive_verdict_is_not_a_discard() {
        let (decision, reason) = d(Verdict::Inconclusive, true, None);
        assert_eq!(decision, INCONCLUSIVE);
        assert!(reason.contains("Do NOT revert"), "{reason}");
        assert!(reason.contains("only 0 clean pair(s), need 3"), "{reason}");
    }

    #[test]
    fn an_early_stop_is_named_in_the_inconclusive_reason() {
        let (decision, reason) = d(
            Verdict::Inconclusive,
            true,
            Some("the host did not go quiet"),
        );
        assert_eq!(decision, INCONCLUSIVE);
        assert!(reason.contains("the host did not go quiet"), "{reason}");
    }

    #[test]
    fn noise_and_loss_are_discards() {
        assert_eq!(d(Verdict::Noise, true, None).0, DISCARD);
        assert_eq!(d(Verdict::Loss, true, None).0, DISCARD);
    }

    #[test]
    fn only_a_win_that_passes_memory_is_kept() {
        assert_eq!(d(Verdict::Win, true, None).0, KEEP_PENDING_CONFIRM);
        let (decision, reason) = d(Verdict::Win, false, None);
        assert_eq!(decision, DISCARD);
        assert!(reason.contains("peak footprint"), "{reason}");
    }

    /// An untrusted evaluator decides nothing, whatever the numbers say.
    #[test]
    fn an_untrusted_evaluator_never_keeps_or_calls_inconclusive() {
        for v in [Verdict::Win, Verdict::Inconclusive, Verdict::Noise] {
            assert_eq!(decide(false, v, true, 9, 3, None).0, DISCARD);
        }
    }

    #[test]
    fn a_noninferiority_spec_keeps_only_when_the_loss_is_ruled_out() {
        let report = |result| NonInferiorityReport {
            result,
            pairs: 10,
            limit: 0.02,
            lower: -0.01,
            upper: 0.01,
        };
        let d = |r, mem| decide_noninferiority(true, &report(r), mem).0;
        assert_eq!(d(NonInferiority::NotWorse, true), KEEP_PENDING_CONFIRM);
        assert_eq!(d(NonInferiority::NotWorse, false), DISCARD);
        assert_eq!(d(NonInferiority::Worse, true), DISCARD);
        assert_eq!(d(NonInferiority::Inconclusive, true), INCONCLUSIVE);
        let untrusted = decide_noninferiority(false, &report(NonInferiority::NotWorse), true);
        assert_eq!(untrusted.0, DISCARD);
    }

    #[test]
    fn a_failed_secondary_guard_discards_a_keep_and_only_a_keep() {
        let run = |prefill: f64, eligible: bool| Outcome {
            run_id: "r".into(),
            arm: "a".into(),
            round: 1,
            warmup: false,
            record: json!({"totals": {"prefill_tok_s": prefill}}),
            status: "ok".into(),
            eligible,
            reasons: Vec::new(),
        };
        let guard = SecondaryGuard {
            metric: crate::workload::Metric::PrefillTokS,
            max_regression: 0.02,
        };
        let slower = secondary_guard(&[(run(100.0, true), run(90.0, true))], guard);
        assert_eq!(slower["passed"], false, "{slower}");
        let fine = secondary_guard(&[(run(100.0, true), run(99.0, true))], guard);
        assert_eq!(fine["passed"], true, "{fine}");
        // Dirty pairs are not readings, and no reading fails.
        let dirty = secondary_guard(&[(run(100.0, false), run(120.0, true))], guard);
        assert_eq!(dirty["passed"], false, "{dirty}");
        let keep = (KEEP_PENDING_CONFIRM, "WIN".to_string());
        assert_eq!(apply_guard(keep.clone(), Some(&slower)).0, DISCARD);
        assert_eq!(
            apply_guard(keep.clone(), Some(&fine)).0,
            KEEP_PENDING_CONFIRM
        );
        assert_eq!(apply_guard(keep, None).0, KEEP_PENDING_CONFIRM);
        let inconclusive = (INCONCLUSIVE, "few pairs".to_string());
        assert_eq!(apply_guard(inconclusive, Some(&slower)).0, INCONCLUSIVE);
    }

    #[test]
    fn each_decision_has_its_own_exit_code() {
        assert_eq!(exit_code_for(KEEP_PENDING_CONFIRM), 0);
        assert_eq!(exit_code_for("keep"), 0);
        assert_eq!(exit_code_for(DISCARD), 3);
        assert_eq!(exit_code_for(INCONCLUSIVE), 4);
        assert_eq!(exit_code_for("something else"), 1);
        // 1 is an error and 2 is usage, so no decision may collide with them.
        for decision in [KEEP_PENDING_CONFIRM, "keep", DISCARD, INCONCLUSIVE] {
            assert!(![1, 2].contains(&exit_code_for(decision)), "{decision}");
        }
    }
}
