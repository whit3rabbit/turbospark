//! `tslab-worker`: measures one arm once, in its own process, and writes one
//! JSON record. It decides nothing; the orchestrator does.
//!
//! It drives prefill and decode itself rather than through
//! `run_raw_completion*`, because the phase counters are cumulative and the
//! only way to separate a prefill breakdown from a decode breakdown is to
//! snapshot them at the boundary. The prefill walk is the chunked driver's
//! own (`foundation::prefill_chunk_spans` plus `prefill_chunk_with_status`),
//! and decode is greedy for a FIXED number of steps that ignores end-of-turn,
//! so two arms always do the same decode work.
//!
//! Exit codes: 0 measured, 3 the arm was refused or failed cleanly (the
//! reason is in the record), anything else a crash.

use std::path::{Path, PathBuf};
use std::time::Instant;

use foundation::{prefill_chunk_spans, LogitValue};
use runtime::{ChunkedPrefillRunner, LogitProducer, RealForwardRunner};
use serde::{Deserialize, Serialize};
use tokenizer::{Message, MfTokenizer, Role};
use turbospark_bench::memory::{task_fault_counters, AppMemorySampler};
use turbospark_bench::protocol::PROTOCOL_CASES;

use crate::phases::{self, CacheClass, PhaseReport};
use crate::provenance::{self, Provenance};
use crate::workload::{Arm, PromptSource, Residency, Slots, Workload};

pub const EXIT_REFUSED: i32 = 3;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Mode {
    /// Time prefill and decode for every prompt.
    Timing,
    /// Prefill each prompt and store its final-position logits for the
    /// quality gate. Timing is recorded but not quotable.
    Quality,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerRequest {
    pub run_id: String,
    pub model: String,
    pub arm: Arm,
    pub workload: Workload,
    pub mode: Mode,
    /// Where the quality mode writes its f16 logits (one row per prompt).
    #[serde(default)]
    pub logits_path: Option<PathBuf>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptRecord {
    pub source: PromptSource,
    pub prompt_tokens: usize,
    pub prefill_s: f64,
    pub decode_tokens: usize,
    pub decode_s: f64,
    /// SHA-256 over the generated token ids (little-endian i32).
    pub decode_digest: String,
    /// SHA-256 over the final prefill logits' f16 bits.
    pub prefill_logits_digest: String,
    pub prefill_phases: PhaseReport,
    pub decode_phases: Option<PhaseReport>,
    pub cache_class: CacheClass,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Totals {
    pub prompt_tokens: usize,
    pub prefill_s: f64,
    pub prefill_tok_s: f64,
    pub decode_tokens: usize,
    pub decode_s: f64,
    pub decode_tok_s: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunRecord {
    pub run_id: String,
    pub arm: String,
    pub mode: Mode,
    /// `ok`, or `refused` with `refusal` set.
    pub status: String,
    pub refusal: Option<String>,
    pub family: Option<String>,
    pub model_dir: Option<String>,
    pub manifest_sha256: Option<String>,
    pub open_s: Option<f64>,
    pub warm_s: Option<f64>,
    pub resolved_slots: Option<usize>,
    pub resolved_residency: Option<String>,
    pub chunked: Option<bool>,
    pub prompts: Vec<PromptRecord>,
    pub totals: Option<Totals>,
    pub peak_footprint_bytes: Option<u64>,
    pub faults: Option<(u64, u64)>,
    /// `TURBOSPARK_DISPATCH_PROFILE=1`'s text report. Present only when that
    /// knob was on, which also makes the run unquotable.
    pub dispatch_profile: Option<String>,
    pub provenance: Provenance,
}

fn refused(req: &WorkerRequest, reason: String) -> RunRecord {
    RunRecord {
        run_id: req.run_id.clone(),
        arm: req.arm.name.clone(),
        mode: req.mode,
        status: "refused".into(),
        refusal: Some(reason),
        family: None,
        model_dir: None,
        manifest_sha256: None,
        open_s: None,
        warm_s: None,
        resolved_slots: None,
        resolved_residency: None,
        chunked: None,
        prompts: Vec::new(),
        totals: None,
        peak_footprint_bytes: None,
        faults: None,
        dispatch_profile: None,
        provenance: provenance::collect(),
    }
}

/// Render a prompt source to token ids.
pub fn prompt_ids(tokenizer: &MfTokenizer, source: &PromptSource) -> Result<Vec<i32>, String> {
    let templated = |messages: &[Message]| -> Result<Vec<i32>, String> {
        let rendered = tokenizer
            .apply_chat_template(messages)
            .map_err(|e| format!("chat template: {e}"))?;
        // The dialect template renders its own <bos>.
        Ok(tokenizer.encode(&rendered, false))
    };
    match source {
        PromptSource::Protocol(id) => {
            let case = PROTOCOL_CASES
                .iter()
                .find(|c| c.id == id)
                .ok_or_else(|| format!("unknown protocol case {id:?}"))?;
            templated(&[Message::new(Role::User, case.content)])
        }
        PromptSource::MessagesFile(path) => {
            let text =
                std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
            let rows: Vec<serde_json::Value> =
                serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
            let mut messages = Vec::new();
            for row in rows {
                let role = match row["role"].as_str() {
                    Some("system") => Role::System,
                    Some("user") => Role::User,
                    Some("assistant") => Role::Assistant,
                    other => return Err(format!("{}: unsupported role {other:?}", path.display())),
                };
                let content = row["content"]
                    .as_str()
                    .ok_or_else(|| format!("{}: content must be a string here", path.display()))?;
                messages.push(Message::new(role, content));
            }
            templated(&messages)
        }
        PromptSource::TextFile(path) => {
            let text =
                std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
            Ok(tokenizer.encode(&text, false))
        }
    }
}

fn f16_bits(logits: &[LogitValue]) -> Vec<u8> {
    logits
        .iter()
        .flat_map(|v| v.to_bits().to_le_bytes())
        .collect()
}

fn argmax(logits: &[LogitValue]) -> Result<i32, String> {
    let mut best = (0usize, f32::NEG_INFINITY);
    for (i, v) in logits.iter().enumerate() {
        let x = v.to_f32();
        if x.is_nan() {
            return Err(format!("NaN logit at vocab index {i}"));
        }
        if x > best.1 {
            best = (i, x);
        }
    }
    Ok(best.0 as i32)
}

/// Prefill `ids` from position 0, chunked when `chunk` is set.
fn prefill(
    runner: &mut RealForwardRunner,
    ids: &[i32],
    chunk: Option<usize>,
    logits: &mut [LogitValue],
    sampler: &mut AppMemorySampler,
) -> Result<(), String> {
    match chunk {
        Some(chunk) => {
            let spans = prefill_chunk_spans(ids.len(), 0, chunk);
            let last = spans.len().saturating_sub(1);
            for (i, span) in spans.iter().enumerate() {
                let tokens = &ids[span.token_offset..span.token_offset + span.token_count];
                runner.prefill_chunk_with_status(tokens, span.start_position, logits, i == last)?;
                sampler.sample();
            }
        }
        None => {
            for (pos, &t) in ids.iter().enumerate() {
                if pos + 1 == ids.len() {
                    runner.produce(t, pos, logits)?;
                } else {
                    runner.produce_prefill(t, pos, logits)?;
                }
            }
            sampler.sample();
        }
    }
    Ok(())
}

fn open(
    req: &WorkerRequest,
    dir: &Path,
) -> Result<(RealForwardRunner, MfTokenizer, String), String> {
    let arch = repack::peek_manifest_arch(dir)?;
    let family = arch.family.as_str().to_string();
    let tokenizer = MfTokenizer::load_from_dir(dir)
        .map_err(|e| format!("tokenizer from {}: {e}", dir.display()))?;
    let slots = match req.arm.open.slots {
        Slots::Auto => runtime::ExpertCacheSlots::Auto,
        Slots::Fixed(n) => runtime::ExpertCacheSlots::Fixed(n),
    };
    let residency = match req.arm.open.residency {
        Residency::Auto => runtime::ExpertResidency::Auto,
        Residency::Streamed => runtime::ExpertResidency::Streamed,
        Residency::Mapped => runtime::ExpertResidency::Mapped,
    };
    let runner = RealForwardRunner::open_with_residency(
        dir,
        arch,
        req.workload.max_context as usize,
        slots,
        runtime::DraftPolicies::off(),
        runtime::SteeringPolicy::off(),
        1,
        runtime::KvQuant::Off,
        residency,
    )
    .map_err(|e| e.to_string())?;
    Ok((runner, tokenizer, family))
}

/// Run the request. `Err` is a refusal or clean failure (exit 3); a panic is
/// a crash.
pub fn run(req: &WorkerRequest) -> RunRecord {
    match run_inner(req) {
        Ok(r) => r,
        Err(reason) => refused(req, reason),
    }
}

fn run_inner(req: &WorkerRequest) -> Result<RunRecord, String> {
    // The environment the orchestrator built must be the arm's, exactly.
    // A stray inherited variable would silently make this a different arm.
    // Location variables are passed through for alias resolution and are
    // not knobs.
    let mut seen = provenance::turbospark_env();
    for location in crate::orchestrate::PASSTHROUGH_TURBOSPARK {
        seen.remove(*location);
    }
    if seen != req.arm.env {
        return Err(format!(
            "worker environment {seen:?} does not match arm {} {:?}",
            req.arm.name, req.arm.env
        ));
    }
    let dir = catalog::resolve_model_arg(&req.model);
    let mut sampler = AppMemorySampler::new();
    sampler.sample();
    let manifest_sha256 = provenance::sha256_file(&dir.join("manifest.json"));

    let t_open = Instant::now();
    let (mut runner, tokenizer, family) = open(req, &dir)?;
    let open_s = t_open.elapsed().as_secs_f64();
    sampler.sample();

    if req.workload.prefill_chunk.is_some() && !runner.supports_chunked_prefill() {
        return Err(format!(
            "{family} install does not support chunked prefill here; use \"prefill_chunk\": null \
             rather than measuring the sequential path under a chunked label"
        ));
    }
    let vocab = runner.vocab_size();
    let mut logits = vec![LogitValue::from_f32(0.0); vocab];
    let all_ids: Vec<Vec<i32>> = req
        .workload
        .prompts
        .iter()
        .map(|p| prompt_ids(&tokenizer, p))
        .collect::<Result<_, _>>()?;
    for (ids, p) in all_ids.iter().zip(&req.workload.prompts) {
        if ids.is_empty() {
            return Err(format!("{p:?} tokenized to nothing"));
        }
        if ids.len() + req.workload.decode_tokens + 1 > req.workload.max_context as usize {
            return Err(format!(
                "{p:?} is {} tokens; with {} decode steps it does not fit max_context {}",
                ids.len(),
                req.workload.decode_tokens,
                req.workload.max_context
            ));
        }
    }

    // In-process warm pass: absorbs Metal pipeline JIT and first-touch
    // allocation so the first measured prompt does not pay for them.
    let t_warm = Instant::now();
    {
        let warm: Vec<i32> = all_ids[0].iter().copied().take(64).collect();
        runner.reset();
        prefill(
            &mut runner,
            &warm,
            req.workload.prefill_chunk,
            &mut logits,
            &mut sampler,
        )?;
        for pos in warm.len()..warm.len() + 4 {
            let t = argmax(&logits)?;
            runner.produce(t, pos, &mut logits)?;
        }
    }
    let warm_s = t_warm.elapsed().as_secs_f64();

    let mut prompts = Vec::new();
    let mut logits_out: Vec<u8> = Vec::new();
    for (ids, source) in all_ids.iter().zip(&req.workload.prompts) {
        runner.reset();
        let p0 = runner.phase_counters();
        let t0 = Instant::now();
        prefill(
            &mut runner,
            ids,
            req.workload.prefill_chunk,
            &mut logits,
            &mut sampler,
        )?;
        let prefill_s = t0.elapsed().as_secs_f64();
        let p1 = runner.phase_counters();
        let bits = f16_bits(&logits);
        let prefill_logits_digest = model_io::hash_data(&bits);
        if req.mode == Mode::Quality {
            logits_out.extend_from_slice(&bits);
        }

        let mut generated: Vec<i32> = Vec::with_capacity(req.workload.decode_tokens);
        let t1 = Instant::now();
        for step in 0..req.workload.decode_tokens {
            let t = argmax(&logits)?;
            generated.push(t);
            runner.produce(t, ids.len() + step, &mut logits)?;
            if step % 8 == 0 {
                sampler.sample();
            }
        }
        let decode_s = t1.elapsed().as_secs_f64();
        let p2 = runner.phase_counters();
        sampler.sample();

        let prefill_phases = phases::report(&phases::delta(&p0, &p1));
        let decode_phases =
            (req.workload.decode_tokens > 0).then(|| phases::report(&phases::delta(&p1, &p2)));
        let cache_class = phases::cache_class(&prefill_phases);
        let token_bytes: Vec<u8> = generated.iter().flat_map(|t| t.to_le_bytes()).collect();
        prompts.push(PromptRecord {
            source: source.clone(),
            prompt_tokens: ids.len(),
            prefill_s,
            decode_tokens: generated.len(),
            decode_s,
            decode_digest: model_io::hash_data(&token_bytes),
            prefill_logits_digest,
            prefill_phases,
            decode_phases,
            cache_class,
        });
    }

    if let Some(path) = &req.logits_path {
        if req.mode == Mode::Quality {
            crate::atomic::write_new(path, &logits_out)?;
        }
    }

    let prompt_tokens: usize = prompts.iter().map(|p| p.prompt_tokens).sum();
    let prefill_s: f64 = prompts.iter().map(|p| p.prefill_s).sum();
    let decode_tokens: usize = prompts.iter().map(|p| p.decode_tokens).sum();
    let decode_s: f64 = prompts.iter().map(|p| p.decode_s).sum();
    let calls = runner.phase_counters().calls;
    let dispatch_profile = (std::env::var("TURBOSPARK_DISPATCH_PROFILE").as_deref() == Ok("1"))
        .then(|| runtime::dispatch_profile_report(calls))
        .flatten();
    Ok(RunRecord {
        run_id: req.run_id.clone(),
        arm: req.arm.name.clone(),
        mode: req.mode,
        status: "ok".into(),
        refusal: None,
        family: Some(family),
        model_dir: Some(dir.display().to_string()),
        manifest_sha256,
        open_s: Some(open_s),
        warm_s: Some(warm_s),
        resolved_slots: Some(runner.expert_cache_slots()),
        resolved_residency: Some(format!("{:?}", runner.resolved_expert_residency())),
        chunked: Some(req.workload.prefill_chunk.is_some()),
        totals: Some(Totals {
            prompt_tokens,
            prefill_s,
            prefill_tok_s: prompt_tokens as f64 / prefill_s.max(f64::MIN_POSITIVE),
            decode_tokens,
            decode_s,
            decode_tok_s: (decode_tokens > 0).then(|| decode_tokens as f64 / decode_s),
        }),
        prompts,
        peak_footprint_bytes: sampler.peak_bytes(),
        faults: task_fault_counters(),
        dispatch_profile,
        provenance: provenance::collect(),
    })
}

/// The `tslab-worker` binary body: `--request <json> --out <json>`.
pub fn main_entry() -> i32 {
    let args: Vec<String> = std::env::args().collect();
    let get = |flag: &str| {
        args.iter()
            .position(|a| a == flag)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };
    let (Some(request), Some(out)) = (get("--request"), get("--out")) else {
        eprintln!("usage: tslab-worker --request <request.json> --out <record.json>");
        return 2;
    };
    let req: WorkerRequest = match std::fs::read_to_string(&request)
        .map_err(|e| e.to_string())
        .and_then(|t| serde_json::from_str(&t).map_err(|e| e.to_string()))
    {
        Ok(r) => r,
        Err(e) => {
            eprintln!("tslab-worker: {request}: {e}");
            return 2;
        }
    };
    let record = gpu::autorelease_pool(|| run(&req));
    let code = if record.status == "ok" {
        0
    } else {
        EXIT_REFUSED
    };
    let value = match serde_json::to_value(&record) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("tslab-worker: serialize: {e}");
            return 1;
        }
    };
    if let Err(e) = crate::atomic::write_new_json(Path::new(&out), &value) {
        eprintln!("tslab-worker: {e}");
        return 1;
    }
    code
}
