//! The frozen community benchmark protocol (real-generation-v1), vendored
//! from the Swift repo's `docs/COMMUNITY_BENCHMARKS.md` and
//! `docs/benchmark-prompts/real-generation-v1/*.json` so Rust and Swift
//! measure the same workload: same prompts, same seeds, same sampling
//! settings, same `[stop=...]` footer for `grep -h '^\[stop='`.
//!
//! The prompt texts under `crates/bench/prompts/real-generation-v1/` are
//! the byte-exact `content` string of the single user message in each
//! source JSON. Source JSON SHA-256, for provenance:
//!   short-explanation.json c57da2677143657be55e03797c1aabe8e012d61c1a71bf9345acdd575117d1e1
//!   medium-review.json     23add7976db2069c9927affd13c2f5120508702a1c4915ea88b7de565bdd4b33
//!   long-synthesis.json    b12e4a71493ad151d7a85b91e878ba784b116ef65de9fcb1b1b1c05198b537b8

use runtime::StopReason;

/// Test case specification for the community benchmark protocol.
pub struct ProtocolCase {
    /// Benchmark case identifier string.
    pub id: &'static str,
    /// Random seed for generation sampling.
    pub seed: u64,
    /// User prompt text content string.
    pub content: &'static str,
}

/// The three standard benchmark protocol cases (short-explanation, medium-review, long-synthesis).
pub const PROTOCOL_CASES: [ProtocolCase; 3] = [
    ProtocolCase {
        id: "short-explanation",
        seed: 20260721,
        content: include_str!("../prompts/real-generation-v1/short-explanation.txt"),
    },
    ProtocolCase {
        id: "medium-review",
        seed: 20260722,
        content: include_str!("../prompts/real-generation-v1/medium-review.txt"),
    },
    ProtocolCase {
        id: "long-synthesis",
        seed: 20260723,
        content: include_str!("../prompts/real-generation-v1/long-synthesis.txt"),
    },
];

/// Benchmark protocol sampling temperature setting.
pub const PROTOCOL_TEMPERATURE: f64 = 0.2;
/// Benchmark protocol top-k sampling limit.
pub const PROTOCOL_TOP_K: u32 = 64;
/// Benchmark protocol top-p nucleus sampling limit.
pub const PROTOCOL_TOP_P: f64 = 0.95;
/// Benchmark protocol maximum generated tokens cap.
pub const PROTOCOL_MAX_NEW: u32 = 1024;
/// Benchmark protocol maximum context window token capacity.
pub const PROTOCOL_MAX_CONTEXT: u32 = 4096;

/// The routed-expert cache size both engines default to, and the one every
/// published number in `docs/BENCHMARKS.md` and the memory oracle's
/// per-chip rows was measured at. A Swift comparison has to match it, and
/// the oracle ceiling only means anything against it (a slot costs ~3.2 MB
/// of pinned host memory per layer on the 26B).
pub const PROTOCOL_EXPERT_CACHE_SLOTS: usize = 16;

/// The Swift `String(describing:)` spellings of the stop reasons, for
/// byte-parity with the community protocol's grep.
pub fn swift_reason_name(reason: StopReason) -> &'static str {
    match reason {
        StopReason::EndOfTurn => "endOfTurn",
        StopReason::ToolCalls => "toolCalls",
        StopReason::Eos => "eos",
        StopReason::StopString => "stopString",
        StopReason::MaxTokens => "maxTokens",
    }
}

/// The Swift CLI's run-summary footer, byte-identical in shape:
/// `[stop=<reason> prefill=<N>tok/<S>s new=<M>tok decode=<S>s tok/s=<X>]`.
pub fn swift_footer(
    reason: StopReason,
    prompt_tokens: usize,
    prefill_seconds: f64,
    new_tokens: usize,
    decode_seconds: f64,
) -> String {
    let tokens_per_second = if decode_seconds > 0.0 {
        new_tokens as f64 / decode_seconds
    } else {
        0.0
    };
    format!(
        "[stop={} prefill={}tok/{:.2}s new={}tok decode={:.2}s tok/s={:.3}]",
        swift_reason_name(reason),
        prompt_tokens,
        prefill_seconds,
        new_tokens,
        decode_seconds,
        tokens_per_second
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cases_carry_the_frozen_seeds_in_order() {
        assert_eq!(PROTOCOL_CASES[0].seed, 20260721);
        assert_eq!(PROTOCOL_CASES[1].seed, 20260722);
        assert_eq!(PROTOCOL_CASES[2].seed, 20260723);
        assert!(PROTOCOL_CASES.iter().all(|c| !c.content.is_empty()));
    }

    #[test]
    fn footer_matches_the_swift_shape() {
        let footer = swift_footer(StopReason::EndOfTurn, 62, 7.7449, 493, 21.3899);
        assert_eq!(
            footer,
            "[stop=endOfTurn prefill=62tok/7.74s new=493tok decode=21.39s tok/s=23.048]"
        );
    }
}
