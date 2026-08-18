//! Single declaration table for every recognized option spelling.
//!
//! The token scan (`parser`) and the usage renderer (`usage`) both read this
//! table, so a new option cannot be added to one without appearing in the
//! other. `--help` is handled separately by the scan as an immediate
//! short-circuit, but is still listed here so it appears in the usage text.

/// Declaration of one recognized option spelling.
#[derive(Debug, Clone, Copy)]
pub struct OptionDecl {
    /// The exact recognized spelling, for example `--model`.
    pub flag: &'static str,
    /// Whether this option consumes the following token as its value.
    pub takes_value: bool,
    /// Whether this is the one required option.
    pub is_required: bool,
    /// Whether this is one of the three mode-selecting options.
    pub is_mode_selecting: bool,
    /// A short, non-contractual description of the default or the accepted
    /// value shape, used to render usage text.
    pub usage_hint: &'static str,
}

/// The complete, ordered set of recognized option spellings. The count is
/// asserted in `tests/usage_and_status.rs`, which is the fifth of the five
/// places adding a flag touches (see the crate's `CLAUDE.md` Gotcha 1).
pub const OPTIONS: &[OptionDecl] = &[
    OptionDecl {
        flag: "--model",
        takes_value: true,
        is_required: true,
        is_mode_selecting: false,
        usage_hint: "path to the model (required)",
    },
    OptionDecl {
        flag: "--prompt",
        takes_value: true,
        is_required: false,
        is_mode_selecting: true,
        usage_hint: "single-turn completion prompt (mode-selecting)",
    },
    OptionDecl {
        flag: "--messages-file",
        takes_value: true,
        is_required: false,
        is_mode_selecting: true,
        usage_hint: "conversation-file path (mode-selecting)",
    },
    OptionDecl {
        flag: "--chat",
        takes_value: false,
        is_required: false,
        is_mode_selecting: true,
        usage_hint: "interactive chat mode (mode-selecting)",
    },
    OptionDecl {
        flag: "--system",
        takes_value: true,
        is_required: false,
        is_mode_selecting: false,
        usage_hint: "leading message, repeatable, chat mode only (default: none)",
    },
    OptionDecl {
        flag: "--max-new",
        takes_value: true,
        is_required: false,
        is_mode_selecting: false,
        usage_hint: "generated-token limit, positive integer (default 1024)",
    },
    OptionDecl {
        flag: "--max-context",
        takes_value: true,
        is_required: false,
        is_mode_selecting: false,
        usage_hint: "context-size limit, positive integer, or auto (default auto: the checkpoint's trained context, capped by what memory holds, and 4096 when the install declares none)",
    },
    OptionDecl {
        flag: "--temperature",
        takes_value: true,
        is_required: false,
        is_mode_selecting: false,
        usage_hint: "sampling temperature, zero or more (default 0.2)",
    },
    OptionDecl {
        flag: "--top-k",
        takes_value: true,
        is_required: false,
        is_mode_selecting: false,
        usage_hint: "rank-based candidate count, 0 disables (default 64, allowed 0-256)",
    },
    OptionDecl {
        flag: "--top-p",
        takes_value: true,
        is_required: false,
        is_mode_selecting: false,
        usage_hint: "cumulative-probability threshold, (0, 1] (default 0.95)",
    },
    OptionDecl {
        flag: "--repetition-penalty",
        takes_value: true,
        is_required: false,
        is_mode_selecting: false,
        usage_hint: "repetition penalty factor, greater than 0 (default 1.0)",
    },
    OptionDecl {
        flag: "--seed",
        takes_value: true,
        is_required: false,
        is_mode_selecting: false,
        usage_hint: "determinism seed, non-negative integer (default: unset)",
    },
    OptionDecl {
        flag: "--stop",
        takes_value: true,
        is_required: false,
        is_mode_selecting: false,
        usage_hint: "stop string, repeatable, accumulates in order (default: none)",
    },
    OptionDecl {
        flag: "--rdadvise",
        takes_value: true,
        is_required: false,
        is_mode_selecting: false,
        usage_hint: "read-ahead mode: off, normal, aggressive (default off)",
    },
    OptionDecl {
        flag: "--expert-cache-slots",
        takes_value: true,
        is_required: false,
        is_mode_selecting: false,
        usage_hint: "routed-cache slot count, allowed 8/16/24/32, or auto (default auto, which never resolves below 16)",
    },
    OptionDecl {
        flag: "--prefill-chunk",
        takes_value: true,
        is_required: false,
        is_mode_selecting: false,
        usage_hint: "prompt-processing chunk size, or auto (default 128)",
    },
    OptionDecl {
        flag: "--power-profile",
        takes_value: true,
        is_required: false,
        is_mode_selecting: false,
        usage_hint: "power profile: performance, balanced, efficiency (default performance, or efficiency under Low Power Mode)",
    },
    OptionDecl {
        flag: "--reasoning",
        takes_value: true,
        is_required: false,
        is_mode_selecting: false,
        usage_hint: "reasoning effort: off, low, medium, high, xhigh (default off; the accepted set is the checkpoint's, not this one)",
    },
    OptionDecl {
        flag: "--max-tokens-per-sec",
        takes_value: true,
        is_required: false,
        is_mode_selecting: false,
        usage_hint: "decode rate cap, greater than 0 (default: uncapped, or the efficiency profile's reading speed)",
    },
    OptionDecl {
        flag: "--quiet",
        takes_value: false,
        is_required: false,
        is_mode_selecting: false,
        usage_hint: "suppress incidental output, plain toggle (default off)",
    },
    OptionDecl {
        flag: "--help",
        takes_value: false,
        is_required: false,
        is_mode_selecting: false,
        usage_hint: "print usage text and exit",
    },
    OptionDecl {
        flag: "--version",
        takes_value: false,
        is_required: false,
        is_mode_selecting: false,
        usage_hint: "print the version and exit",
    },
];
