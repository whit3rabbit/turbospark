//! The validated invocation value and its constituent field types.
//!
//! Every field here already holds either the caller's validated value or its
//! documented default; nothing is left as raw, unvalidated text.

use foundation::runtime_config::DEFAULT_CHUNK_SIZE;

/// Documented default generated-token limit.
pub const DEFAULT_MAX_NEW: u32 = 1024;
/// Documented default context-size limit.
///
/// Re-exported from `foundation` rather than restated: `crates/runtime`'s
/// context policy resolves `auto` against the same number, and two copies
/// would let the parser's documented default and the resolver's fallback
/// drift apart silently.
pub const DEFAULT_MAX_CONTEXT: u32 = foundation::runtime_config::DEFAULT_MAX_CONTEXT;
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

/// Context-window sizing: a fixed token count, or automatic sizing against
/// the checkpoint and the machine.
///
/// The THIRD flag here to take an `auto` keyword and the second to default
/// to it, following [`ExpertCacheSlots`]. Resolution needs the install's
/// trained context, its per-layer KV strides and the machine's memory --
/// none of which this pure crate may look at -- so `Auto` crosses into
/// `crates/runtime` unresolved, the same division [`PowerProfile`] takes.
///
/// Unlike the slot count, this axis is NOT throughput-only: it decides how
/// much KV is allocated and how long a prompt is admitted. What makes an
/// environment-sensing default safe anyway is the resolver's rule that an
/// install declaring no trained context resolves to [`DEFAULT_MAX_CONTEXT`]
/// -- which is every install written before that field existed, so nothing
/// already on disk moves.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MaxContext {
    /// Exactly this many tokens, whatever the checkpoint and the machine
    /// look like. Refused only when the machine cannot hold it.
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
    /// The context-size limit, or `Auto` to size it at open.
    pub max_context: MaxContext,
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
    /// Whether incidental output is suppressed.
    pub quiet: bool,
}
