//! The validated invocation value and its constituent field types.
//!
//! Every field here already holds either the caller's validated value or its
//! documented default; nothing is left as raw, unvalidated text.

use foundation::runtime_config::DEFAULT_CHUNK_SIZE;

/// Documented default generated-token limit.
pub const DEFAULT_MAX_NEW: u32 = 1024;
/// Documented default context-size limit.
pub const DEFAULT_MAX_CONTEXT: u32 = 4096;
/// Documented default sampling temperature.
pub const DEFAULT_TEMPERATURE: f64 = 0.2;
/// Documented default rank-based candidate count. Zero means the rank-based
/// shaping is disabled.
pub const DEFAULT_TOP_K: u32 = 64;
/// Maximum accepted rank-based candidate count (inclusive).
pub const MAX_TOP_K: u32 = 256;
/// Documented default cumulative-probability threshold.
pub const DEFAULT_TOP_P: f64 = 0.95;
/// Documented default repetition penalty factor (identity, no attenuation).
pub const DEFAULT_REPETITION_PENALTY: f64 = 1.0;

/// The selected input mode. A validated invocation always carries exactly
/// one of these.
#[derive(Debug, Clone, PartialEq)]
pub enum Mode {
    /// Single-turn completion from a raw prompt string.
    Prompt(String),
    /// Conversation input read from a messages-file path.
    MessagesFile(String),
    /// Interactive chat mode. Carries no value of its own.
    Chat,
}

/// Read-ahead advisory mode for prompt processing. The named-mode set and
/// spellings are destination-selected; only the documented default
/// (disabled) and the existence of a small fixed set are contractual.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ReadAheadMode {
    /// Read-ahead disabled. Documented default.
    #[default]
    Off,
    /// Normal read-ahead.
    Normal,
    /// Aggressive read-ahead.
    Aggressive,
}

impl ReadAheadMode {
    /// Parse a read-ahead mode from its documented spelling. Returns `None`
    /// for any text outside the fixed set.
    pub fn parse(text: &str) -> Option<Self> {
        match text {
            "off" => Some(Self::Off),
            "normal" => Some(Self::Normal),
            "aggressive" => Some(Self::Aggressive),
            _ => None,
        }
    }
}

/// Power profile selection. Declared here rather than reused from the
/// runtime crate because this crate is pure and depends only on
/// `foundation`; the two spellings are pinned against each other by
/// `crates/cli`'s mapping and by `parse_outcomes.rs`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PowerProfile {
    /// No rate cap and no thermal stepping.
    Performance,
    /// Uncapped, steps down under thermal pressure.
    Balanced,
    /// Capped near reading speed, steps down further under pressure.
    Efficiency,
}

impl PowerProfile {
    /// Parse a profile from its documented spelling. Returns `None` for
    /// any text outside the fixed set.
    pub fn parse(text: &str) -> Option<Self> {
        match text {
            "performance" => Some(Self::Performance),
            "balanced" => Some(Self::Balanced),
            "efficiency" => Some(Self::Efficiency),
            _ => None,
        }
    }

    /// The inverse of [`Self::parse`].
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Performance => "performance",
            Self::Balanced => "balanced",
            Self::Efficiency => "efficiency",
        }
    }
}

/// Prompt-processing chunk-size tuning: a fixed token count drawn from the
/// foundation-published allowed set, or automatic sizing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrefillChunk {
    /// A fixed chunk size drawn from the foundation-published allowed set.
    Fixed(u32),
    /// Automatic sizing, selected with the "auto" keyword.
    Auto,
}

impl Default for PrefillChunk {
    fn default() -> Self {
        PrefillChunk::Fixed(DEFAULT_CHUNK_SIZE)
    }
}

/// A fully populated, validated invocation.
#[derive(Debug, Clone, PartialEq)]
pub struct InvocationRequest {
    /// The required model-path value, accepted as an opaque string.
    pub model: String,
    /// The single selected input mode.
    pub mode: Mode,
    /// The combined leading (system) message, chat mode only.
    pub system: Option<String>,
    /// The generated-token limit.
    pub max_new: u32,
    /// The context-size limit.
    pub max_context: u32,
    /// The sampling temperature.
    pub temperature: f64,
    /// The rank-based candidate count; zero means disabled.
    pub top_k: u32,
    /// The cumulative-probability threshold.
    pub top_p: f64,
    /// The repetition penalty factor.
    pub repetition_penalty: f64,
    /// The optional determinism seed.
    pub seed: Option<u64>,
    /// The accumulated stop-string list, in supplied order.
    pub stop: Vec<String>,
    /// The read-ahead advisory mode.
    pub rdadvise: ReadAheadMode,
    /// The routed-cache slot count.
    pub expert_cache_slots: u32,
    /// The prompt-processing chunk-size tuning.
    pub prefill_chunk: PrefillChunk,
    /// The power profile, or `None` for automatic (which resolves against
    /// the OS's Low Power Mode when the session opens).
    pub power_profile: Option<PowerProfile>,
    /// The explicit decode rate cap in tokens per second, or `None` to
    /// take whatever the resolved profile carries.
    pub max_tokens_per_sec: Option<f64>,
    /// Whether incidental output is suppressed.
    pub quiet: bool,
}
