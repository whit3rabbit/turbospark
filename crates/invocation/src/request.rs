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

impl PrefillChunk {
    /// The concrete chunk size a caller consuming this value should use.
    /// `Auto` defers to `foundation`'s automatic chunk-size resolver, which
    /// today has no input length to resolve against
    /// (`foundation::InputLength::Unknown`) and so returns the same fixed
    /// default `Fixed` carries -- but it is the one place an adaptive
    /// `Known` arm would land, rather than each consumer re-deciding it.
    pub fn resolved(&self) -> u32 {
        match self {
            PrefillChunk::Fixed(n) => *n,
            PrefillChunk::Auto => {
                foundation::resolve_automatic_chunk_size(foundation::InputLength::Unknown)
            }
        }
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

/// Routed expert residency mode: whether routed experts are streamed through
/// a per-layer slot cache or mapped directly in host memory.
///
/// Mirrors `model_io::ExpertResidency`, kept here so `turbospark-invocation`
/// stays pure and depends only on `foundation`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ExpertResidency {
    /// Automatic selection, defaulting to streamed today until eviction
    /// characteristics under pressure are calibrated.
    #[default]
    Auto,
    /// Traditional streamed mode using pinned slot cache buffers.
    Streamed,
    /// Mapped mode reading experts directly out of mapped memory.
    Mapped,
}

impl ExpertResidency {
    /// Parse an expert residency mode from its documented spelling.
    pub fn parse(text: &str) -> Option<Self> {
        match text.trim().to_ascii_lowercase().as_str() {
            "auto" => Some(Self::Auto),
            "streamed" => Some(Self::Streamed),
            "mapped" => Some(Self::Mapped),
            _ => None,
        }
    }

    /// The inverse of [`Self::parse`].
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Streamed => "streamed",
            Self::Mapped => "mapped",
        }
    }
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

/// The blocks a caller may name, re-exported from `foundation` so this
/// parser and `turbospark-server`'s own read ONE range (AGENTS.md Gotcha 2).
pub use foundation::runtime_config::ALLOWED_SPECULATION_BLOCKS;
pub use foundation::SteeringMode;

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
    ///
    /// **DETECTING A DRAFTER IS NOT ENABLING IT, and the two drafters differ
    /// on exactly that.** An MTP head is switched on (1.44-1.66x); a DFlash2
    /// drafter is REPORTED and left off, because through the shipped loop it
    /// reads 0.88x throughput at +17.4% J/token on prose against 1.47x on
    /// code, and a default that makes the common workload slower has to be
    /// asked for. `crates/cli`'s `resolve_drafter` owns that split -- this
    /// crate is pure and may not read an install.
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

/// How much of the machine may be committed to loading a model.
///
/// The mirror of `model_io::LoadGuard`, spelled again here for the reason
/// [`MaxContext`] and [`ExpertCacheSlots`] are: this crate is pure and may
/// not read a machine or an install, so it cannot depend on the crate that
/// owns the arithmetic. `crates/cli` maps between the two.
///
/// **[`Self::Relaxed`] is the default and is what shipped before this flag
/// existed.** The other tiers are a user choice and none of them is the
/// baseline any measurement in this repo was taken under.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LoadGuard {
    /// No memory precautions; nothing is refused for size.
    Off,
    /// The shipped behaviour.
    #[default]
    Relaxed,
    /// A larger reserve and a smaller share of what is left.
    Balanced,
    /// Larger still.
    Strict,
    /// `Relaxed`'s shares plus an absolute ceiling, in bytes, on what the
    /// engine may ALLOCATE. Not on the install's size: exceeding memory with
    /// the mapped install is the streaming this engine is built around.
    Custom(u64),
}

impl LoadGuard {
    /// The flag spelling. `None` for anything else, including a byte count,
    /// which the parser tries next -- so an unrecognized word and an
    /// unparsable number produce one diagnostic rather than two.
    pub fn parse(text: &str) -> Option<Self> {
        match text {
            "off" => Some(Self::Off),
            "relaxed" => Some(Self::Relaxed),
            "balanced" => Some(Self::Balanced),
            "strict" => Some(Self::Strict),
            _ => None,
        }
    }
}

/// TurboQuant KV-cache quantization selection. The mirror of
/// `model_io::KvQuant`, spelled again here for the reason `MaxContext` and
/// `ExpertCacheSlots` are: this crate is pure and may not depend on
/// `model_io`. `crates/cli`'s mapping pins the two spellings against each
/// other, and both accept exactly `off|2|3|3.5|4`.
///
/// **Unlike `MaxContext`/`ExpertCacheSlots`, this is NOT a sensing default.**
/// `Off` is the default and is exactly what every release before this flag
/// existed produced, so no frozen memory-oracle or quality-gate row moves for
/// a caller who does not opt in (`docs/TRUBOQUANT.md`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum KvBits {
    /// FP16 everywhere. The default.
    #[default]
    Off,
    /// TurboQuant at `k_bits`/`v_bits`. Only reached through
    /// [`KvBits::parse`], which is the four widths `model_io::KvQuant::parse`
    /// accepts -- `"3.5"` splits into K3/V4, mlx-vlm's own convention for its
    /// one fractional width.
    TurboQuant {
        /// Bits per key-row coordinate.
        k_bits: u8,
        /// Bits per value-row coordinate.
        v_bits: u8,
    },
}

impl KvBits {
    /// Parse a selection from its documented spelling. Returns `None` for
    /// any text outside the fixed set.
    pub fn parse(text: &str) -> Option<Self> {
        match text {
            "off" => Some(Self::Off),
            "2" => Some(Self::TurboQuant {
                k_bits: 2,
                v_bits: 2,
            }),
            "3" => Some(Self::TurboQuant {
                k_bits: 3,
                v_bits: 3,
            }),
            "3.5" => Some(Self::TurboQuant {
                k_bits: 3,
                v_bits: 4,
            }),
            "4" => Some(Self::TurboQuant {
                k_bits: 4,
                v_bits: 4,
            }),
            _ => None,
        }
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
    /// The context-size limit, or `Auto` to size it at open.
    pub max_context: MaxContext,
    /// How much of the machine may be committed to loading the model.
    pub load_guard: LoadGuard,
    /// The fewest tokens an AUTOMATIC context resolution may land on, or 0
    /// for no floor. Deliberately says nothing about an explicit
    /// `--max-context`: a caller naming a number has decided how to spend
    /// their own machine.
    pub min_auto_context: u32,
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
    /// Image paths, in supplied order (ROADMAP M-V7).
    ///
    /// ORDER-PRESERVING and not a set: the nth path pairs with the nth
    /// marker run the template renders, so reordering them silently pairs
    /// each picture with the wrong span.
    ///
    /// Opaque strings, like `steering` below: this crate is pure and reads
    /// no file, so decoding and preprocessing is the front end's job.
    pub images: Vec<String>,
    /// Run the prompt once PER image rather than once with all of them.
    ///
    /// The bulk-OCR shape: one open runner, one generation per page, KV
    /// reset between. Without it every `--image` lands in ONE turn, which is
    /// what the checkpoint's template does natively and what the flag reads
    /// like.
    pub image_batch: bool,
    /// The read-ahead advisory mode.
    pub rdadvise: ReadAheadMode,
    /// The routed-cache slot count, or `Auto` to size it at open.
    pub expert_cache_slots: ExpertCacheSlots,
    /// The routed-expert residency mode; see [`ExpertResidency`].
    pub expert_residency: ExpertResidency,
    /// Speculative decoding policy; see [`Speculation`].
    pub speculation: Speculation,
    /// Which drafter that policy drives; see [`SpeculativeDrafter`].
    pub speculative_drafter: SpeculativeDrafter,
    /// Path to a control vector to steer with, if any. An opaque string:
    /// this crate is pure and reads no file, so resolving and parsing it is
    /// the front end's job (`docs/OBLITERATION.md`).
    pub steering: Option<String>,
    /// Which edit to apply. `None` defers to whatever the vector file
    /// declares, and to `Ablate` if it declares nothing.
    pub steering_mode: Option<SteeringMode>,
    /// Strength. `None` means 1.0.
    pub steering_scale: Option<f32>,
    /// Inclusive, 0-based layer range to restrict the vector to.
    pub steering_layers: Option<(u32, u32)>,
    /// What `SteeringMode::Clamp` pins the coefficient to.
    pub steering_target: f32,
    /// Coefficient magnitude below which the edit does not fire.
    pub steering_gate: f32,
    /// The prompt-processing chunk-size tuning.
    pub prefill_chunk: PrefillChunk,
    /// The power profile, or `None` for automatic (which resolves against
    /// the OS's Low Power Mode when the session opens).
    pub power_profile: Option<PowerProfile>,
    /// The explicit decode rate cap in tokens per second, or `None` to
    /// take whatever the resolved profile carries.
    pub max_tokens_per_sec: Option<f64>,
    /// Path to a standalone vision-tower sidecar install to attach to a
    /// text-only trunk (vision memory sidecar, Part A4), or `None` to use
    /// the trunk's own tower (if any).
    ///
    /// An opaque string, like `steering` and `images` above: this crate is
    /// pure and reads no directory, so resolving and attaching it is the
    /// front end's job. **`"auto"` is not special here and is read as a
    /// literal path** -- catalog-based sidecar resolution is a later part,
    /// not yet built, so this crate deliberately does not reserve the
    /// keyword ahead of it.
    pub vision_sidecar: Option<String>,
    /// How hard the model is asked to think, rendered into the prompt by the
    /// checkpoint's own chat template.
    pub reasoning: ReasoningEffort,
    /// TurboQuant KV-cache quantization selection; see [`KvBits`].
    pub kv_bits: KvBits,
    /// Whether incidental output is suppressed.
    pub quiet: bool,
}
