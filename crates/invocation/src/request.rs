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

/// Speculative decoding: whether to draft ahead with the checkpoint's own
/// multi-token-prediction head, and what to do when it cannot be served.
///
/// The three values are not a spectrum, they are three different answers to
/// "what happens if this install cannot speculate":
///
/// - [`Speculation::Auto`] WARNS on stderr and decodes sequentially. The
///   default, because most installs carry no head and a hard error would make
///   the common case a failure.
/// - [`Speculation::Off`] never drafts and never warns.
/// - [`Speculation::Block`] HARD FAILS. A caller who named a block size is
///   measuring or benchmarking, and the failure mode this avoids is the one
///   `crates/runtime`'s own gotchas keep naming: a run that quietly did not
///   speculate, reported success, and got recorded as the speculative number.
///
/// **The warning is not decoration.** Silence here would be the same bug the
/// engine had until the head was detected at all -- an install with a
/// drafter decoding one token at a time and nothing saying so.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Speculation {
    /// Draft if the install can, warn and continue if it cannot.
    #[default]
    Auto,
    /// Never draft.
    Off,
    /// Draft this many tokens per round, or fail naming the reason.
    Block(u32),
}

/// The blocks a caller may name. The upper bound is the batched verify's
/// register-bound row cap (a round of block B verifies `B + 1` rows against
/// `MAX_BATCH_ROWS = 16`); the measured optimum is 2 and everything above 4
/// loses on this engine (`docs/MTP.md`), so the range is deliberately wider
/// than the useful part rather than pretending to be a recommendation.
pub const ALLOWED_SPECULATION_BLOCKS: std::ops::RangeInclusive<u32> = 1..=15;

/// Which drafter `--speculative` drives. The two drafters are ALTERNATIVES
/// (`docs/MTP_SPECULATIVE.md`, `docs/DFLASH2.md`): the checkpoint's own MTP
/// head drafts a token at a time, the DFlash2 block-diffusion drafter
/// proposes a whole block in one pass. A third value is not a spectrum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SpeculativeDrafter {
    /// Whichever drafter the INSTALL carries, read off its resident index.
    ///
    /// The default, because the alternative is a default that is wrong for
    /// half the installs: pinned at `Mtp`, a DFlash2-carrying install
    /// reported "this install carries no multi-token-prediction head" and
    /// decoded sequentially with a working drafter on disk. An install
    /// carrying BOTH resolves to `Mtp`, so nothing that worked before this
    /// existed changes.
    #[default]
    Auto,
    /// The checkpoint's own multi-token-prediction head (`mtp.*` tensors).
    Mtp,
    /// The DFlash2 block-diffusion drafter (`dflash.*` tensors).
    Dflash,
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
    /// Speculative decoding policy; see [`Speculation`].
    pub speculation: Speculation,
    /// Which drafter that policy drives; see [`SpeculativeDrafter`].
    pub speculative_drafter: SpeculativeDrafter,
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
