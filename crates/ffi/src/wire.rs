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
    pub max_tokens_per_sec: Option<f64>,
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
            reasoning: "off".to_string(),
        }
    }
}

/// One chat message, in the shape `--messages-file` accepts.
#[derive(Debug, Clone, Deserialize)]
pub struct WireMessage {
    pub role: String,
    #[serde(default)]
    pub content: String,
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
