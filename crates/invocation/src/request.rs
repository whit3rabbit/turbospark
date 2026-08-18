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

/// How hard the model is asked to think before it answers.
///
/// Declared here rather than reused from the tokenizer crate for the reason
/// [`PowerProfile`] is: this crate is pure and depends only on `foundation`.
/// `crates/cli`'s `map_reasoning_effort` pins the two spellings against each
/// other, which is the third such mapping (see this crate's Gotcha 2).
///
/// **THE SET IS A UNION AND NO CHECKPOINT ACCEPTS ALL OF IT.** Qwen 3.8 takes
/// `xhigh`/`medium`/`low` and rejects `high`; Harmony and Muse Glimmer take
/// `high`/`medium`/`low`. This crate may not look at the install, so it
/// cannot know which -- the checkpoint's own template validates and names its
/// own set in the error. Rejecting the union here would refuse a spelling
/// some future checkpoint accepts; accepting anything at all would let a typo
/// through to a Jinja error.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ReasoningEffort {
    /// Thinking disabled. The default, and what every release before this
    /// flag rendered.
    #[default]
    Off,
    /// Brief, focused thinking.
    Low,
    /// The middle setting.
    Medium,
    /// Harmony's and Muse Glimmer's top setting.
    High,
    /// Qwen 3.8's top setting.
    XHigh,
}

impl ReasoningEffort {
    /// Parse a level from its documented spelling. Returns `None` for any
    /// text outside the fixed set.
    pub fn parse(text: &str) -> Option<Self> {
        match text {
            "off" => Some(Self::Off),
            "low" => Some(Self::Low),
            "medium" => Some(Self::Medium),
            "high" => Some(Self::High),
            "xhigh" => Some(Self::XHigh),
            _ => None,
        }
    }

    /// The inverse of [`Self::parse`].
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::XHigh => "xhigh",
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

/// Routed-expert cache sizing: a fixed per-layer slot count drawn from the
/// foundation-published allowed set, or automatic sizing.
///
/// Unlike [`PrefillChunk`], whose default is a fixed value, the default here
/// is [`Self::Auto`]. The slot cache is a pure RAM-for-throughput trade --
/// `docs/DECODE_BUDGET.md` measures the GPU idle 40-52% of a decoded token
/// waiting on the expert `pread` it exists to avoid -- and a machine with
/// headroom should spend it. What makes that safe as a DEFAULT is that
/// resolution can only ever climb: see the runtime-side resolver, which
/// floors at `DEFAULT_CACHE_SLOTS`, so no machine gets a smaller cache than
/// it had before this existed.
///
/// Resolution needs the install's per-layer expert stride and the machine's
/// memory, neither of which this pure crate may look at, so `Auto` crosses
/// into `crates/runtime` unresolved. That is the same division
/// [`PowerProfile`] takes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ExpertCacheSlots {
    /// A fixed slot count drawn from the foundation-published allowed set.
    Fixed(u32),
    /// Automatic sizing, selected with the "auto" keyword and the default.
    #[default]
    Auto,
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
    /// The routed-cache slot count, or `Auto` to size it at open.
    pub expert_cache_slots: ExpertCacheSlots,
    /// The prompt-processing chunk-size tuning.
    pub prefill_chunk: PrefillChunk,
    /// The power profile, or `None` for automatic (which resolves against
    /// the OS's Low Power Mode when the session opens).
    pub power_profile: Option<PowerProfile>,
    /// The explicit decode rate cap in tokens per second, or `None` to
    /// take whatever the resolved profile carries.
    pub max_tokens_per_sec: Option<f64>,
    /// How hard the model is asked to think, rendered into the prompt by the
    /// checkpoint's own chat template.
    pub reasoning: ReasoningEffort,
    /// Whether incidental output is suppressed.
    pub quiet: bool,
}
