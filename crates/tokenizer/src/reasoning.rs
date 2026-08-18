//! Reasoning-effort selection, and what a checkpoint's own template can do
//! with it.
//!
//! **THE LEVEL LIVES IN THE TEMPLATE, NOT IN THE WEIGHTS.** A reasoning model
//! reasons harder because its rendered system preamble tells it to, so the
//! whole feature here is two context keys and a boolean.
//! `mlx-community/Qwen3.8-27B-4bit`'s `chat_template.jinja` is the worked
//! example:
//!
//! ```jinja
//! {%- if enable_thinking is undefined or enable_thinking is true %}
//!     {%- set resolved_reasoning_effort = reasoning_effort|default('xhigh') %}
//! ```
//!
//! Read that gate before believing any claim about a default. `xhigh` is the
//! default only for a caller that leaves BOTH keys undefined, which is what
//! transformers and mlx-lm do and is where the model card's claim comes from.
//! This port used to pass `enable_thinking: false` unconditionally, so it took
//! neither the default nor a level: it got no thinking at all, and the
//! generation prompt ended in a pre-closed `<think>\n\n</think>\n\n`.
//!
//! **A LEVEL THEREFORE IMPLIES `enable_thinking: true`.** The effort key is
//! read INSIDE that gate on every template that has both, so a caller asking
//! for `low` while thinking stays off would set a key nothing reads -- a flag
//! that does nothing, reports nothing, and looks like it worked. [`Off`] is
//! the one value that disables thinking, and it is the default, so every
//! existing caller renders the bytes it always did.
//!
//! [`Off`]: ReasoningEffort::Off

use crate::dialect::MfTokenizer;

/// How hard the model is asked to think before it answers.
///
/// The names are the UNION of the vocabularies the shipped checkpoints use,
/// not a set this port invented, and no checkpoint accepts all five: Qwen 3.8
/// takes `xhigh`/`medium`/`low` and RAISES on `high`, Harmony and Muse
/// Glimmer take `high`/`medium`/`low`. Validation is deliberately left to the
/// template, which knows its own set and says so by name -- Qwen 3.8's
/// message is "Unexpected reasoning effort high. Supported types are xhigh
/// (default), medium, and low", which reaches the caller through
/// [`TokenizerError::InvalidChatTemplate`]. Inventing a per-family allowlist
/// here would be a second, staler copy of that knowledge.
///
/// [`TokenizerError::InvalidChatTemplate`]: crate::TokenizerError::InvalidChatTemplate
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ReasoningEffort {
    /// No thinking: `enable_thinking` renders false and no effort key is set
    /// at all. The default, and byte-identical to what every caller rendered
    /// before this type existed.
    #[default]
    Off,
    /// Brief, focused thinking.
    Low,
    /// The middle setting, spelled the same way by every checkpoint that has
    /// one.
    Medium,
    /// Harmony's and Muse Glimmer's top setting. **Qwen 3.8 REJECTS this
    /// spelling**; its top setting is [`Self::XHigh`].
    High,
    /// Qwen 3.8's top setting, and its own default when nothing is passed.
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

    /// The string to put in the template's effort key, or `None` when no key
    /// should be inserted at all.
    ///
    /// Inserting nothing for [`Self::Off`] is not the same as inserting
    /// `"off"`: a template resolves its key with `|default('xhigh')`, so an
    /// absent key takes the checkpoint's default and an unrecognized value
    /// raises. Neither is wanted when thinking is disabled outright.
    pub fn level(self) -> Option<&'static str> {
        match self {
            Self::Off => None,
            other => Some(other.as_str()),
        }
    }

    /// What `enable_thinking` renders as. See the module header for why this
    /// is not an independent axis.
    pub fn enable_thinking(self) -> bool {
        self != Self::Off
    }
}

/// What a checkpoint's installed template can actually do with a requested
/// level.
///
/// **This exists so that a level cannot be a silent no-op**, which is the
/// failure this repo has paid for repeatedly (AGENTS.md Gotchas 41, 44, 49):
/// a knob that renders no differently gives fluent output with nothing wrong
/// in it, and nothing anywhere says the request was dropped. The three
/// shipped shapes are genuinely different and a caller can act on the
/// difference -- refuse, warn, or proceed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReasoningSupport {
    /// The template names none of the three keys, so a level changes no byte
    /// of the prompt. `gemma4`'s dialect fallback and every template-less
    /// synthetic install land here.
    None,
    /// The template reads `enable_thinking` and no effort key: thinking can
    /// be turned on and off, and the LEVEL is ignored. `Qwen3.5`-era
    /// checkpoints (`bonsai27b`, `ternary27b`) are this shape.
    ToggleOnly,
    /// The template reads an effort key (`reasoning_effort`, as Qwen 3.8 and
    /// Harmony spell it, or `reasoning_strength`, as Muse Glimmer does), so
    /// every level is expressible.
    Level,
}

/// The two spellings the shipped checkpoints use for the same idea. Both are
/// set, because a template reads the one it knows and ignores the other, and
/// deciding between them per family would be a table that rots the first time
/// a checkpoint arrives with the other spelling.
pub(crate) const EFFORT_KEYS: [&str; 2] = ["reasoning_effort", "reasoning_strength"];

/// The key that gates every effort key on every template seen here.
pub(crate) const THINKING_KEY: &str = "enable_thinking";

impl MfTokenizer {
    /// What this checkpoint's installed template can do with a reasoning
    /// level, read off the template SOURCE.
    ///
    /// A substring scan, deliberately: a Jinja variable is read by name, so a
    /// template that never writes the identifier provably cannot honour it.
    /// The error direction is toward PERMITTING (a template mentioning the
    /// name in a comment reads as support), which is the safe one -- a false
    /// `Level` costs a rendered prompt that ignores the key, while a false
    /// `None` would refuse a request the checkpoint could have served.
    pub fn reasoning_support(&self) -> ReasoningSupport {
        let Some(source) = self.chat_template_source.as_deref() else {
            return ReasoningSupport::None;
        };
        if EFFORT_KEYS.iter().any(|key| source.contains(key)) {
            ReasoningSupport::Level
        } else if source.contains(THINKING_KEY) {
            ReasoningSupport::ToggleOnly
        } else {
            ReasoningSupport::None
        }
    }
}
