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
        flag: "--load-guard",
        takes_value: true,
        is_required: false,
        is_mode_selecting: false,
        usage_hint: "how much memory may be committed: off, relaxed, balanced, strict, or a byte ceiling on what the engine allocates (default relaxed, which is what shipped before this flag existed)",
    },
    OptionDecl {
        flag: "--min-auto-context",
        takes_value: true,
        is_required: false,
        is_mode_selecting: false,
        usage_hint: "refuse to open when `--max-context auto` resolves below this many tokens; 0 imposes no floor and does not constrain an explicit --max-context (default 0)",
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
        flag: "--image",
        takes_value: true,
        is_required: false,
        is_mode_selecting: false,
        usage_hint: "image path, repeatable, all images land in ONE turn (default: none)",
    },
    OptionDecl {
        flag: "--image-batch",
        takes_value: false,
        is_required: false,
        is_mode_selecting: false,
        usage_hint: "run the prompt once PER --image instead of once with all of them",
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
        usage_hint: "routed-cache slot count, allowed 8/16/24/32/48/64/96/128, or auto (default auto, which never resolves below 16)",
    },
    OptionDecl {
        flag: "--speculative",
        takes_value: true,
        is_required: false,
        is_mode_selecting: false,
        usage_hint: "speculative decoding: auto, off, or a block size 1-15 \
                     (default auto; a named block FAILS if the model cannot serve it)",
    },
    OptionDecl {
        flag: "--speculative-drafter",
        takes_value: true,
        is_required: false,
        is_mode_selecting: false,
        usage_hint: "drafter --speculative drives: auto, mtp or dflash (default auto, \
                     which enables an mtp head but only REPORTS a dflash one -- \
                     dflash is 0.88x on prose, so name it to run it)",
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
        flag: "--steering",
        takes_value: true,
        is_required: false,
        is_mode_selecting: false,
        usage_hint: "path to a control vector (.gguf, llama.cpp layout) to steer with \
                     (default none; see docs/OBLITERATION.md)",
    },
    OptionDecl {
        flag: "--steering-mode",
        takes_value: true,
        is_required: false,
        is_mode_selecting: false,
        usage_hint: "steering edit: ablate, add, clamp, renorm (default ablate, or \
                     whatever the vector file declares)",
    },
    OptionDecl {
        flag: "--steering-scale",
        takes_value: true,
        is_required: false,
        is_mode_selecting: false,
        usage_hint: "steering strength (default 1.0; 0.0 is the exact identity, and \
                     large values on add/clamp can overflow the FP16 residual stream)",
    },
    OptionDecl {
        flag: "--steering-layers",
        takes_value: true,
        is_required: false,
        is_mode_selecting: false,
        usage_hint: "layer range to steer, START:END inclusive, 0-based (default every \
                     layer the vector covers)",
    },
    OptionDecl {
        flag: "--steering-target",
        takes_value: true,
        is_required: false,
        is_mode_selecting: false,
        usage_hint: "coefficient --steering-mode clamp pins the stream to (default 0.0; \
                     ignored by ablate and add)",
    },
    OptionDecl {
        flag: "--steering-gate",
        takes_value: true,
        is_required: false,
        is_mode_selecting: false,
        usage_hint: "only steer where the direction's coefficient reaches this magnitude \
                     (default 0.0, meaning always)",
    },
    OptionDecl {
        flag: "--vision-sidecar",
        takes_value: true,
        is_required: false,
        is_mode_selecting: false,
        usage_hint: "path to a standalone vision-tower sidecar install to attach to a \
                     text-only trunk (default: none; the trunk's own tower, if any, is used)",
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
