//! The JSON shapes that cross the boundary.
//!
//! **Options and results travel as JSON strings rather than as C structs, and
//! that is the single decision that keeps `turbospark.h` small.** It removes
//! about fifteen struct definitions and every layout, alignment and
//! versioning question with them, and it means adding a knob is a field here
//! rather than an ABI break. The cost is a serde round trip on calls that
//! happen once per session or once per second; the per-token path carries no
//! JSON at all and is a raw pointer and a length.
//!
//! Field names are `camelCase`, so a Swift `Codable` needs no
//! `CodingKeys` and the two sides cannot drift over a spelling.

use serde::{Deserialize, Serialize};

/// Reads a window or slot count that may be a number, `"auto"`, `null`, or
/// absent. `None` means automatic.
///
/// All three spellings of automatic are accepted because a Swift optional
/// bridges to an omitted key, a `null`, or the string `"auto"` depending on
/// how the caller writes its encoder, and making those mean different things
/// would be a trap invisible from the header. Any OTHER string is an error
/// rather than a silent fallback to automatic: `"atuo"` should be heard
/// about, not quietly honoured as the default.
pub fn sized(value: &Option<serde_json::Value>, name: &str) -> Result<Option<u32>, String> {
    match value {
        None | Some(serde_json::Value::Null) => Ok(None),
        Some(serde_json::Value::String(s)) if s.eq_ignore_ascii_case("auto") => Ok(None),
        Some(serde_json::Value::Number(n)) => n
            .as_u64()
            .filter(|v| *v <= u32::MAX as u64)
            .map(|v| Some(v as u32))
            .ok_or_else(|| format!("{name} must be a non-negative integer, got {n}")),
        Some(other) => Err(format!("{name} must be a number or \"auto\", got {other}")),
    }
}

/// Reads a load guard that may be a tier NAME, a byte ceiling, `null`, or
/// absent.
///
/// Absent and `null` mean `relaxed` for [`sized`]'s reason -- a Swift optional
/// bridges to either -- and, more importantly here, because `relaxed` is what
/// this binding did before the option existed. An unrecognized STRING is an
/// error rather than a silent fallback: quietly honouring `"strcit"` as the
/// default is exactly the trap `sized` refuses.
pub fn load_guard(value: &Option<serde_json::Value>) -> Result<model_io::LoadGuard, String> {
    match value {
        None | Some(serde_json::Value::Null) => Ok(model_io::LoadGuard::default()),
        Some(serde_json::Value::String(s)) => model_io::LoadGuard::parse(&s.to_ascii_lowercase())
            .ok_or_else(|| {
                format!("loadGuard must be off, relaxed, balanced, strict or a byte count, got {s}")
            }),
        Some(serde_json::Value::Number(n)) => n
            .as_u64()
            .filter(|v| *v > 0)
            .map(|v| model_io::LoadGuard::Custom {
                max_counted_bytes: v,
            })
            .ok_or_else(|| format!("loadGuard as a byte ceiling must be positive, got {n}")),
        Some(other) => Err(format!(
            "loadGuard must be a tier name or a byte count, got {other}"
        )),
    }
}

/// Arguments to `ts_recommend_json`. One field today, and a JSON blob rather
/// than a second `uint32_t` for the reason this crate takes every other
/// options bag as JSON: a knob added here is a field rather than an ABI
/// break, and the header stays readable.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct RecommendOptions {
    /// Same spellings as `OpenOptions::load_guard`, and it MUST be the same
    /// value the host will open with; see `models::recommend_json`.
    pub load_guard: Option<serde_json::Value>,
}

/// Arguments to `ts_session_open`. Every field is optional; `{}` is valid and
/// means "everything automatic", which is what the two user-facing binaries
/// default to.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct OpenOptions {
    pub max_context: Option<serde_json::Value>,
    pub expert_cache_slots: Option<serde_json::Value>,
    /// `performance` | `balanced` | `efficiency`. Absent means ASK THE OS,
    /// which is what a user-facing binary should do (Low Power Mode selects
    /// `efficiency`) and what a measurement harness must not.
    pub power_profile: Option<String>,
    /// `off` | `relaxed` | `balanced` | `strict`, or a NUMBER, which is an
    /// absolute ceiling in bytes on what the engine may allocate. Absent
    /// means `relaxed`, which is what this binding did before the option
    /// existed -- and what every frozen footprint row describes.
    pub load_guard: Option<serde_json::Value>,
    /// Refuse to open when an AUTOMATIC window resolves below this. Says
    /// nothing about an explicit `maxContext`; see `model_io::LoadPolicy`.
    pub min_auto_context: Option<u32>,
    pub max_tokens_per_sec: Option<f64>,
    /// `"off"` | `"auto"` | a block size, as a number or a string. Absent
    /// means `auto`, which is what both other front ends default to.
    ///
    /// Resolved at OPEN and not per turn, because that is where the
    /// drafter's state is allocated and there is one engine per session --
    /// the same reason `turbospark-server` makes it a process-level flag.
    /// A named block that cannot be served is an ERROR from
    /// `ts_session_open`; `auto` that cannot be served opens fine and
    /// reports the reason in `sessionInfo.speculation`.
    pub speculation: Option<serde_json::Value>,
    /// `auto` | `mtp` | `dflash`. Absent means `auto`, which ENABLES an MTP
    /// head and only REPORTS a DFlash2 one (`docs/DFLASH2.md`).
    pub speculative_drafter: Option<String>,
    /// Path to a control vector (.gguf, llama.cpp layout) to steer with.
    pub steering: Option<String>,
    /// `ablate` | `add` | `clamp` | `renorm`. Default is `ablate` or whatever
    /// the vector declares.
    pub steering_mode: Option<String>,
    /// Multiplier on the edit strength (default 1.0; 0.0 is identity).
    pub steering_scale: Option<f64>,
    /// Layer range to steer, `START:END` inclusive 0-based (default all).
    pub steering_layers: Option<String>,
    /// Coefficient for `clamp` mode (default 0.0).
    pub steering_target: Option<f64>,
    /// Minimum activation magnitude to fire the edit (default 0.0).
    pub steering_gate: Option<f64>,
}

/// What a session resolved about directional steering, once, at open.
#[derive(Debug, Clone, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SteeringInfo {
    /// True when a control vector is active on this session.
    pub active: bool,
    /// `ablate` | `add` | `clamp` | `renorm`, present only when active.
    pub mode: Option<String>,
    /// Active scale multiplier, present only when active.
    pub scale: Option<f64>,
    /// Human-readable one-line description, or null when inactive.
    pub summary: Option<String>,
}

/// What a session resolved about speculative decoding, once, at open.
///
/// Reported for the reason the resolved slot count is: a caller who asked
/// for `auto` named no number, and an install carrying a drafter that
/// decodes one token at a time with nothing said is the failure the feature
/// was built to end.
#[derive(Debug, Clone, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SpeculationInfo {
    /// How many tokens a round proposes, or null when this session does not
    /// speculate at all. Null is the "is it on" test; there is no separate
    /// flag to disagree with it.
    pub block: Option<usize>,
    /// `mtp` | `dflash`, present only when `block` is. Two drafters serve
    /// one family with different shapes and different measured optima, so a
    /// throughput figure is unreadable without knowing which one ran.
    pub drafter: Option<String>,
    /// Why speculation is off, when the caller might have expected it on.
    /// Null when they asked for `off` and got it, and null when it is on.
    pub reason: Option<String>,
}

/// Arguments to `ts_generate`.
///
/// The sampling defaults are the CLI's, so a GUI that sends `{}` gets what
/// `turbospark-check` gives with no flags.
#[derive(Debug, Clone, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct GenerateOptions {
    pub max_new_tokens: u32,
    pub temperature: f64,
    pub top_k: u32,
    pub top_p: f64,
    pub repetition_penalty: f64,
    pub seed: Option<u64>,
    pub stop: Vec<String>,
    pub stop_tokens: Vec<u32>,
    /// `off` | `low` | `medium` | `high` | `xhigh`. The ACCEPTED SET IS THE
    /// CHECKPOINT'S, not this crate's: a level the template rejects comes
    /// back as an error naming the level, because a per-family allowlist
    /// here would be a second, staler copy of a set the checkpoint already
    /// states.
    pub reasoning: String,
}

impl Default for GenerateOptions {
    fn default() -> Self {
        Self {
            max_new_tokens: 512,
            temperature: 0.2,
            top_k: 64,
            top_p: 0.95,
            repetition_penalty: 1.0,
            seed: None,
            stop: Vec::new(),
            stop_tokens: Vec::new(),
            reasoning: "off".to_string(),
        }
    }
}

/// One chat message, in the shape `--messages-file` accepts.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct WireMessage {
    pub role: String,
    #[serde(default)]
    pub content: String,
}

/// The result of fitting a conversation into a context window budget.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct WindowFitOutcome {
    pub retained: Vec<WireMessage>,
    pub measured_tokens: u64,
    pub removed_turn_count: usize,
    pub has_room_for_generation: bool,
}

/// Special token identifiers for tokenizer introspection.
#[derive(Debug, Clone, Default, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct SpecialTokensInfo {
    pub bos_id: Option<i32>,
    pub eos_id: Option<i32>,
    pub pad_id: Option<i32>,
    pub end_of_turn_id: Option<i32>,
    pub stop_token_ids: Vec<i32>,
    pub think_start_id: Option<i32>,
    pub think_end_id: Option<i32>,
}

/// What `ts_generate` writes to its result out-parameter.
///
/// Carries the accumulated `content` and `reasoning` as well as streaming
/// them, so a caller that only wants the finished turn can ignore the
/// callback entirely.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GenerateResult {
    pub prompt_tokens: usize,
    pub new_tokens: usize,
    pub prefill_seconds: f64,
    pub decode_seconds: f64,
    /// `endOfTurn` | `toolCalls` | `eos` | `stopString` | `maxTokens` |
    /// `cancelled`.
    pub stop_reason: String,
    /// Decode tokens per second, or null when no decoding happened (a run
    /// cancelled during prefill). Null rather than zero so a caller cannot
    /// plot a rate that was never measured.
    pub tokens_per_second: Option<f64>,
    pub content: String,
    pub reasoning: String,
    /// `normal` | `warn` | `critical`: the WORST memory pressure seen while
    /// this turn decoded.
    ///
    /// **`normal` when nothing was watching, which is the default.** The
    /// in-loop probe follows the power profile's stepping, so a session on
    /// `performance` reports the absence of a reading rather than a reading
    /// of "fine". This field exists to catch a SPIKE between polls;
    /// `ts_system_info_json` is what a status panel should read.
    pub peak_memory_pressure: String,
}

/// What `ts_session_info_json` returns: everything a status panel needs that
/// does not change during a session.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionInfo {
    pub model_path: String,
    pub family: String,
    /// The RESOLVED window, never what the caller asked for. Under `auto`
    /// the request carries no number, and the KV cache has already been
    /// allocated at this one.
    pub max_context: u32,
    /// The checkpoint's own trained context, or null for an install written
    /// before that was recorded. `maxContext` above it is legal and only
    /// degrades quality, which is why `pastTrainedContext` is reported
    /// rather than refused.
    pub trained_context: Option<u32>,
    pub past_trained_context: bool,
    /// The RESOLVED slot count. Worth 44.2 tok/s against 51.2 on one
    /// install, so no throughput or footprint figure is readable without it.
    pub expert_cache_slots: usize,
    pub vocab_size: usize,
    pub dialect: String,
    /// `level` | `toggleOnly` | `none`. A GUI should disable its reasoning
    /// picker on `none` and grey out the LEVELS on `toggleOnly`, where
    /// thinking turns on but the level is dropped.
    pub reasoning_support: String,
    pub steering: SteeringInfo,
    /// What speculative decoding resolved to. **`block` being non-null is a
    /// statement about this SESSION and not about the next turn**:
    /// acceptance is exact only at temperature 0, so a sampled turn decodes
    /// sequentially whatever this says.
    pub speculation: SpeculationInfo,
    pub special_tokens: SpecialTokensInfo,
}

/// What `ts_session_phases_json` returns: `MFERENCE_PHASES=1`'s breakdown.
///
/// **Cumulative over every forward pass the runner has served, PREFILL
/// INCLUDED**, so a per-token number here is an average over the session's
/// whole context range rather than a number at the current context. And the
/// buckets cover the inside of `produce` only: the sampler and the
/// detokenizer run after it returns and appear in none of them.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PhaseReport {
    pub calls: u64,
    pub total_ms_per_call: f64,
    pub gpu_wait_ms: f64,
    pub final_wait_ms: f64,
    pub router_ms: f64,
    pub expert_io_ms: f64,
    pub bind_ms: f64,
    pub pipeline_wait_ms: f64,
    pub cb1_gpu_ms: f64,
    pub routed_cb_gpu_ms: f64,
    pub final_cb_gpu_ms: f64,
    pub expert_requests: u64,
    pub expert_hits: u64,
    /// Null when nothing has been requested yet, rather than a 0% hit rate
    /// on no data.
    pub expert_hit_rate: Option<f64>,
}
