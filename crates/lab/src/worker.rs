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
use crate::workload::{Arm, Drafter, PromptSource, Residency, Slots, Speculation, Workload};

pub const EXIT_REFUSED: i32 = 3;

/// Printed to stderr once the warm pass is done and measured work begins.
pub const MEASURING_MARKER: &str = "[tslab-worker] measuring";

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
    /// Per turn, for a chat prompt. The fields above then total the chat
    /// (`prompt_tokens` counting only the tokens prefilled, not the reused
    /// ones), and `decode_digest` covers every turn's tokens in order.
    #[serde(default)]
    pub turns: Option<Vec<TurnRecord>>,
}

/// One turn of a chat prompt.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurnRecord {
    pub prompt_tokens: usize,
    /// Leading prompt tokens the runtime continued from instead of
    /// prefilling (`RawDecodeResult::reused_prefix_tokens`).
    pub reused_tokens: usize,
    /// From the generation call to the first token's callback.
    pub ttft_s: f64,
    pub prefill_s: f64,
    pub decode_tokens: usize,
    pub decode_s: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Totals {
    pub prompt_tokens: usize,
    pub prefill_s: f64,
    pub prefill_tok_s: f64,
    pub decode_tokens: usize,
    pub decode_s: f64,
    pub decode_tok_s: Option<f64>,
    /// Mean time to first token over every chat turn after the first
    /// (`Metric::FollowupTtftS`). `None` without a chat follow-up.
    #[serde(default)]
    pub followup_ttft_s: Option<f64>,
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
            templated(&parse_messages(&rows, path)?)
        }
        // The opening turn, which is what the warm pass and the fit check
        // see; the follow-ups are rendered as the chat runs.
        PromptSource::Chat(path) => templated(&ChatFile::load(path, tokenizer)?.opening),
        PromptSource::TextFile(path) => {
            let text =
                std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
            Ok(tokenizer.encode(&text, false))
        }
        PromptSource::Corpus(_) => Err("corpus prompts are resolved per workload".into()),
    }
}

fn parse_messages(rows: &[serde_json::Value], path: &Path) -> Result<Vec<Message>, String> {
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
    Ok(messages)
}

/// A `chat:` prompt file:
///
/// ```json
/// {"opening": [{"role": "user", "content": "What does it solve?"}],
///  "document": "../../docs/MTP.md", "document_tokens": 6000,
///  "follow_ups": ["Summarize it.", "What would you change?"]}
/// ```
///
/// `opening` is a messages-file conversation. `document`, optional and
/// relative to the chat file, is cut to `document_tokens` (default 6000)
/// tokens and put before the first user message's content, which is how a
/// long chat is built without pasting the text into the file.
#[derive(Debug, Clone)]
pub struct ChatFile {
    pub opening: Vec<Message>,
    pub follow_ups: Vec<String>,
    /// Whether the model reasons before it replies, so a reply carries a
    /// `</think>` and the client sends back only what follows it. `false` for a
    /// template with no thinking channel, where the whole reply is the content.
    /// The chat file's `"reasoning"` key; absent means `true`.
    pub reasoning: bool,
}

impl ChatFile {
    pub fn load(path: &Path, tokenizer: &MfTokenizer) -> Result<Self, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let v: serde_json::Value =
            serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
        let rows = v["opening"]
            .as_array()
            .ok_or_else(|| format!("{}: \"opening\" must be a message array", path.display()))?;
        let mut opening = parse_messages(rows, path)?;
        let follow_ups = v["follow_ups"]
            .as_array()
            .ok_or_else(|| format!("{}: \"follow_ups\" must be an array", path.display()))?
            .iter()
            .map(|f| {
                f.as_str()
                    .map(str::to_string)
                    .ok_or_else(|| format!("{}: a follow-up must be a string", path.display()))
            })
            .collect::<Result<Vec<_>, _>>()?;
        if let Some(doc) = v["document"].as_str() {
            let doc_path = path.parent().unwrap_or(Path::new(".")).join(doc);
            let cap = v["document_tokens"].as_u64().unwrap_or(6000) as usize;
            let doc_text = std::fs::read_to_string(&doc_path)
                .map_err(|e| format!("{}: {e}", doc_path.display()))?;
            let ids = tokenizer.encode(&doc_text, false);
            let cut = tokenizer.decode(&ids[..ids.len().min(cap)], false);
            let first = opening
                .iter_mut()
                .find(|m| m.role == Role::User)
                .ok_or_else(|| format!("{}: a document needs a user message", path.display()))?;
            let content = first.content.clone().unwrap_or_default();
            *first = Message::new(Role::User, format!("{cut}\n\n{content}"));
        }
        let reasoning = match v.get("reasoning") {
            None => true,
            Some(r) => r.as_bool().ok_or_else(|| {
                format!("{}: \"reasoning\" must be true or false", path.display())
            })?,
        };
        Ok(Self {
            opening,
            follow_ups,
            reasoning,
        })
    }
}

/// A reply as a client sends it back. With `reasoning` (a thinking template)
/// the client drops the reasoning: whatever follows the last `</think>`, or
/// nothing when the turn ended inside it. Without `reasoning` there is no
/// `</think>` to find and the whole reply is the content; dropping it would
/// send every assistant turn back empty.
pub fn visible_content(text: &str, reasoning: bool) -> String {
    if !reasoning {
        return text.trim().to_string();
    }
    text.rsplit_once("</think>")
        .map(|(_, after)| after.trim().to_string())
        .unwrap_or_default()
}

/// Every prompt of a workload as token ids. Corpus prompts are sliced
/// together so the k-th one gets the probes' k-th offset.
pub fn workload_ids(
    tokenizer: &MfTokenizer,
    prompts: &[PromptSource],
) -> Result<Vec<Vec<i32>>, String> {
    let corpus_lengths: Vec<usize> = prompts
        .iter()
        .filter_map(|p| match p {
            PromptSource::Corpus(n) => Some(*n),
            _ => None,
        })
        .collect();
    let mut corpus_slices = if corpus_lengths.is_empty() {
        Vec::new()
    } else {
        let text = turbospark_bench::quality_corpus::corpus_text(provenance::source_root())?;
        let ids = tokenizer.encode(&text, false);
        turbospark_bench::quality_corpus::slices(&ids, &corpus_lengths)?
    }
    .into_iter();
    prompts
        .iter()
        .map(|p| match p {
            PromptSource::Corpus(_) => corpus_slices
                .next()
                .ok_or_else(|| "corpus slice count mismatch".to_string()),
            other => prompt_ids(tokenizer, other),
        })
        .collect()
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

fn greedy_config(max_new: usize) -> Result<runtime::GenerationConfig, String> {
    Ok(runtime::GenerationConfig {
        // Temperature exactly 0: the speculative loop refuses anything else
        // as greedy, and acceptance is exact only there.
        shaping: selection::ShapingConfig::new(0.0, 1, None, 1.0, Some(0))
            .map_err(|e| e.to_string())?,
        max_new_tokens: max_new as u32,
        stop_strings: Vec::new(),
        extra_stop_tokens: Vec::new(),
        rate: Default::default(),
    })
}

/// A chat prompt as a conversation, through the runtime's own loops (the
/// speculative one for a speculative arm, else the chunked or sequential
/// one), greedy, with prefix reuse ON so each follow-up continues from what
/// the runtime can keep. End-of-turn stops a reply, as it would for a client.
/// Returns the turns and every generated id in order.
#[allow(clippy::too_many_arguments)]
fn chat_once(
    runner: &mut RealForwardRunner,
    tokenizer: &MfTokenizer,
    chat: &ChatFile,
    sp: Option<&Speculation>,
    chunk: Option<usize>,
    max_context: u32,
    max_new: usize,
    sampler: &mut AppMemorySampler,
) -> Result<(Vec<TurnRecord>, Vec<i32>), String> {
    let config = greedy_config(max_new)?;
    let vocab = runner.vocab_size();
    runner.reset();
    runner.set_prefix_reuse(true);
    let mut history = chat.opening.clone();
    let mut turns = Vec::new();
    let mut all = Vec::new();
    let outcome = (|| {
        for turn in 0..=chat.follow_ups.len() {
            if turn > 0 {
                history.push(Message::new(Role::User, chat.follow_ups[turn - 1].clone()));
            }
            let rendered = tokenizer
                .apply_chat_template(&history)
                .map_err(|e| format!("chat template: {e}"))?;
            let ids = tokenizer.encode(&rendered, false);
            if ids.len() + max_new + 1 > max_context as usize {
                return Err(format!(
                    "chat turn {} is {} tokens; with {max_new} decode steps it does not fit \
                     max_context {max_context}",
                    turn + 1,
                    ids.len()
                ));
            }
            let started = Instant::now();
            let mut first: Option<f64> = None;
            let mut generated = Vec::new();
            let mut text = String::new();
            let progress = |event: runtime::RawDecodeProgress| match event {
                runtime::RawDecodeProgress::Token {
                    id, delta, index, ..
                } => {
                    if first.is_none() {
                        first = Some(started.elapsed().as_secs_f64());
                    }
                    generated.push(id);
                    text.push_str(&delta);
                    if index % 8 == 0 {
                        sampler.sample();
                    }
                }
                runtime::RawDecodeProgress::Tail(t) => text.push_str(&t),
                _ => {}
            };
            let result = match (sp, chunk) {
                (Some(sp), _) if sp.adaptive => runtime::run_raw_completion_speculative_adaptive(
                    runner,
                    tokenizer,
                    &ids,
                    &config,
                    max_context,
                    vocab,
                    sp.block,
                    runtime::AdaptiveDepth::DEFAULT_FLOOR.min(sp.block),
                    progress,
                ),
                (Some(sp), _) => runtime::run_raw_completion_speculative(
                    runner,
                    tokenizer,
                    &ids,
                    &config,
                    max_context,
                    vocab,
                    sp.block,
                    progress,
                ),
                (None, Some(chunk)) => runtime::run_raw_completion_chunked(
                    runner,
                    tokenizer,
                    &ids,
                    &config,
                    max_context,
                    vocab,
                    chunk,
                    progress,
                ),
                (None, None) => runtime::run_raw_completion(
                    runner,
                    tokenizer,
                    &ids,
                    &config,
                    max_context,
                    vocab,
                    progress,
                ),
            }
            .map_err(|e| e.to_string())?;
            turns.push(TurnRecord {
                prompt_tokens: ids.len(),
                reused_tokens: result.reused_prefix_tokens,
                ttft_s: first.ok_or_else(|| format!("chat turn {} produced no token", turn + 1))?,
                prefill_s: result.prefill_seconds,
                decode_tokens: result.new_tokens,
                decode_s: result.decode_seconds,
            });
            all.extend_from_slice(&generated);
            history.push(Message::new(
                Role::Assistant,
                visible_content(&text, chat.reasoning),
            ));
        }
        Ok(())
    })();
    // Leave the runner as every other prompt expects it: no reuse, nothing
    // carried into the next prompt.
    runner.set_prefix_reuse(false);
    runner.reset();
    outcome.map(|()| (turns, all))
}

/// One prompt through the runtime's speculative loop, greedy, at most
/// `max_new` tokens. Returns the result and the generated ids in order.
fn speculative_once(
    runner: &mut RealForwardRunner,
    tokenizer: &MfTokenizer,
    ids: &[i32],
    sp: &Speculation,
    max_context: u32,
    max_new: usize,
    sampler: &mut AppMemorySampler,
) -> Result<(runtime::RawDecodeResult, Vec<i32>), String> {
    let config = greedy_config(max_new)?;
    let vocab = runner.vocab_size();
    let mut generated = Vec::with_capacity(max_new);
    let progress = |event: runtime::RawDecodeProgress| {
        if let runtime::RawDecodeProgress::Token { index, id, .. } = event {
            generated.push(id);
            if index % 8 == 0 {
                sampler.sample();
            }
        }
    };
    let result = if sp.adaptive {
        runtime::run_raw_completion_speculative_adaptive(
            runner,
            tokenizer,
            ids,
            &config,
            max_context,
            vocab,
            sp.block,
            runtime::AdaptiveDepth::DEFAULT_FLOOR.min(sp.block),
            progress,
        )
    } else {
        runtime::run_raw_completion_speculative(
            runner,
            tokenizer,
            ids,
            &config,
            max_context,
            vocab,
            sp.block,
            progress,
        )
    }
    .map_err(|e| e.to_string())?;
    if result.reused_prefix_tokens != 0 {
        return Err(format!(
            "the runtime reused {} prompt tokens from a previous turn; a measured prompt must prefill in full",
            result.reused_prefix_tokens
        ));
    }
    Ok((result, generated))
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
    let policies = match &req.arm.effective_speculation(&req.workload) {
        None => runtime::DraftPolicies::off(),
        Some(sp) => {
            let asked = match sp.drafter {
                Drafter::Auto => runtime::SpeculativeDrafter::Auto,
                Drafter::Mtp => runtime::SpeculativeDrafter::Mtp,
                Drafter::Dflash => runtime::SpeculativeDrafter::Dflash,
            };
            let choice = runtime::resolve_drafter(asked, dir);
            if let Some(note) = &choice.note {
                return Err(format!("drafter not enabled: {note}"));
            }
            let depth = runtime::Speculation::Block(sp.block as u32);
            runtime::draft_policies(&choice, depth)
        }
    };
    let runner = RealForwardRunner::open_with_residency(
        dir,
        arch,
        req.workload.max_context as usize,
        slots,
        policies,
        runtime::SteeringPolicy::off(),
        1,
        runtime::KvQuant::Off,
        residency,
    )
    .map_err(|e| e.to_string())?;
    Ok((runner, tokenizer, family))
}

/// The environment the orchestrator built must be the arm's, exactly. A
/// stray inherited variable would silently make this a different arm.
/// Location variables pass through for alias resolution, and the
/// orchestrator's own (`ORCHESTRATOR_ENV`) are accepted at their one value
/// unless the arm sets them itself.
pub fn env_matches_arm(
    mut seen: std::collections::BTreeMap<String, String>,
    arm: &Arm,
) -> Result<(), String> {
    for location in crate::orchestrate::PASSTHROUGH_TURBOSPARK {
        seen.remove(*location);
    }
    for (k, v) in crate::orchestrate::ORCHESTRATOR_ENV {
        if !arm.env.contains_key(*k) && seen.get(*k).map(String::as_str) == Some(*v) {
            seen.remove(*k);
        }
    }
    if seen != arm.env {
        return Err(format!(
            "worker environment {seen:?} does not match arm {} {:?}",
            arm.name, arm.env
        ));
    }
    Ok(())
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
    env_matches_arm(provenance::turbospark_env(), &req.arm)?;
    // Without the feature the runtime never reads TURBOSPARK_ANE_PREFILL, so
    // such an arm would measure Steel under an ANE label.
    if !cfg!(feature = "ane-prefill")
        && req
            .arm
            .env
            .get("TURBOSPARK_ANE_PREFILL")
            .is_some_and(|v| !matches!(v.as_str(), "0" | "off" | ""))
    {
        return Err(
            "this tslab-worker was built without the ane-prefill feature; build it with \
             cargo build --release -p turbospark-lab --features ane-prefill"
                .into(),
        );
    }
    let dir = catalog::resolve_model_arg(&req.model);
    let mut sampler = AppMemorySampler::new();
    sampler.sample();
    let manifest_sha256 = provenance::sha256_file(&dir.join("manifest.json"));

    let t_open = Instant::now();
    let (mut runner, tokenizer, family) = open(req, &dir)?;
    let open_s = t_open.elapsed().as_secs_f64();
    sampler.sample();

    // The speculative loop owns its prefill, so the chunk size does not
    // apply to it.
    let speculation = req.arm.effective_speculation(&req.workload);
    if speculation.is_none()
        && req.workload.prefill_chunk.is_some()
        && !runner.supports_chunked_prefill()
    {
        return Err(format!(
            "{family} install does not support chunked prefill here; use `--chunk off` (in a spec: \
             \"prefill_chunk\": null) rather than measuring the sequential path under a chunked \
             label. An experiment that adds a chunk driver behind a switch should give the OFF arm \
             a chunk entry point that runs the old per-token loop, so both arms share the label"
        ));
    }
    let vocab = runner.vocab_size();
    let mut logits = vec![LogitValue::from_f32(0.0); vocab];
    let all_ids = workload_ids(&tokenizer, &req.workload.prompts)?;
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
    if let Some(sp) = &speculation {
        let warm: Vec<i32> = all_ids[0].iter().copied().take(64).collect();
        runner.reset();
        speculative_once(
            &mut runner,
            &tokenizer,
            &warm,
            sp,
            req.workload.max_context,
            8,
            &mut sampler,
        )?;
    } else {
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
    // `tslab trace` attaches its recorder when it reads this line, so the
    // trace covers measured work, not the model load or the warm pass.
    eprintln!("{MEASURING_MARKER}");

    let mut prompts = Vec::new();
    let mut logits_out: Vec<u8> = Vec::new();
    for (ids, source) in all_ids.iter().zip(&req.workload.prompts) {
        if let PromptSource::Chat(path) = source {
            let chat = ChatFile::load(path, &tokenizer)?;
            let p0 = runner.phase_counters();
            let (turns, generated) = chat_once(
                &mut runner,
                &tokenizer,
                &chat,
                speculation.as_ref(),
                req.workload.prefill_chunk,
                req.workload.max_context,
                req.workload.decode_tokens,
                &mut sampler,
            )?;
            let p1 = runner.phase_counters();
            let all = phases::report(&phases::delta(&p0, &p1));
            let token_bytes: Vec<u8> = generated.iter().flat_map(|t| t.to_le_bytes()).collect();
            prompts.push(PromptRecord {
                source: source.clone(),
                // Tokens actually PREFILLED, so `prefill_tok_s` stays a
                // prefill rate: a reused prefix is not work the prefill did.
                // Each turn keeps its full prompt length and reuse.
                prompt_tokens: turns
                    .iter()
                    .map(|t| t.prompt_tokens - t.reused_tokens)
                    .sum(),
                prefill_s: turns.iter().map(|t| t.prefill_s).sum(),
                decode_tokens: turns.iter().map(|t| t.decode_tokens).sum(),
                decode_s: turns.iter().map(|t| t.decode_s).sum(),
                decode_digest: model_io::hash_data(&token_bytes),
                prefill_logits_digest: String::new(),
                cache_class: phases::cache_class(&all),
                prefill_phases: all,
                decode_phases: None,
                turns: Some(turns),
            });
            continue;
        }
        if let Some(sp) = &speculation {
            runner.reset();
            let p0 = runner.phase_counters();
            let (result, generated) = speculative_once(
                &mut runner,
                &tokenizer,
                ids,
                sp,
                req.workload.max_context,
                req.workload.decode_tokens,
                &mut sampler,
            )?;
            let p1 = runner.phase_counters();
            let all = phases::report(&phases::delta(&p0, &p1));
            let token_bytes: Vec<u8> = generated.iter().flat_map(|t| t.to_le_bytes()).collect();
            prompts.push(PromptRecord {
                source: source.clone(),
                prompt_tokens: result.prompt_tokens,
                prefill_s: result.prefill_seconds,
                decode_tokens: result.new_tokens,
                decode_s: result.decode_seconds,
                decode_digest: model_io::hash_data(&token_bytes),
                // The speculative loop does not hand back its prefill logits.
                prefill_logits_digest: String::new(),
                cache_class: phases::cache_class(&all),
                prefill_phases: all,
                decode_phases: None,
                turns: None,
            });
            continue;
        }
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
            turns: None,
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
    let followups: Vec<f64> = prompts
        .iter()
        .filter_map(|p| p.turns.as_ref())
        .flat_map(|t| t.iter().skip(1).map(|t| t.ttft_s))
        .collect();
    let followup_ttft_s =
        (!followups.is_empty()).then(|| followups.iter().sum::<f64>() / followups.len() as f64);
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
            followup_ttft_s,
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn arm(env: &[(&str, &str)]) -> Arm {
        Arm {
            name: "a".into(),
            env: env
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect(),
            open: Default::default(),
            speculation: None,
            expect_engaged: Vec::new(),
        }
    }

    fn seen(env: &[(&str, &str)]) -> BTreeMap<String, String> {
        env.iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    /// The bug this exists for: the orchestrator's own engagement variable
    /// made every worker refuse its arm.
    #[test]
    fn the_orchestrators_variables_are_not_a_stray() {
        let base = [("TURBOSPARK_ENGAGEMENT", "1"), ("TURBOSPARK_HOME", "/x")];
        assert!(env_matches_arm(seen(&base), &arm(&[])).is_ok());
        let with_knob = [
            ("TURBOSPARK_ENGAGEMENT", "1"),
            ("TURBOSPARK_INT4_MMA8", "1"),
        ];
        assert!(env_matches_arm(seen(&with_knob), &arm(&[("TURBOSPARK_INT4_MMA8", "1")])).is_ok());
        // A stray knob, or the variable at another value, still refuses.
        assert!(env_matches_arm(seen(&with_knob), &arm(&[])).is_err());
        assert!(env_matches_arm(seen(&[("TURBOSPARK_ENGAGEMENT", "0")]), &arm(&[])).is_err());
    }
}
