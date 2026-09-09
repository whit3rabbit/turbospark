#![cfg(target_os = "macos")]
//! The quality gate for the `spark2_5` GGUF install (`XHToken/
//! Spark-X2.5-4B-GGUF`).
//!
//! Not run by default:
//!
//!   TURBOSPARK_SPARK_INSTALL_DIR=~/models/spark25.gturbo \
//!     cargo test -p turbospark-bench --test spark_quality_gate --release -- --ignored --nocapture
//!
//! **AN ASSISTANT PREFIX IS REQUIRED HERE, and `</think>` is the whole
//! reason** (crate Gotcha 13). The template's generation prompt ends
//! `<｜start▁of▁sentence｜><|Bot|><think>` -- the think frame is FORCED OPEN,
//! so the next thing the model emits is scratchpad, and a reference answer
//! spliced in raw would be scored as reasoning. `</think>` closes the frame
//! and puts the answer in the visible slot, which is where a perplexity over
//! an answer belongs. The prefix is scored as PROMPT, never as a target
//! (`run_quality_gate_with_assistant_prefix` guarantees it).
//!
//! NOTE the corpus and the digest prompt are the frozen protocol's, chosen
//! for Gemma 4 and reused verbatim, so this measures Spark-X2.5 on a
//! Gemma-shaped text. Compare each row only against its own past.

mod quality_common;

/// Closes the forced-open think frame; see the module header.
const SPARK_ASSISTANT_PREFIX: &str = "</think>";

/// Per-chip rows, MOST SPECIFIC SUBSTRING FIRST.
///
/// No Swift row at any chip and there will not be one: the Swift original
/// has no `spark2_5` support at all, so every row here is this port
/// measuring itself.
///
/// Frozen from the first real run (2026-09-08), after two fresh processes
/// reproduced both digests.
///
/// **12.6162 IS HEALTHY HERE even though muse's row reads 6.29**, and the
/// difference is POSITION, not quality: the absolute perplexities are not a
/// ranking across families because each template puts the reference answer
/// somewhere else, and Spark's answer follows a `</think>` that closes a
/// forced-open think frame -- a different context shape from muse's
/// ` to=user<|message|>` header. What the number IS evidence for: a
/// two-digit-or-worse reading beside degenerate generations would say the
/// answer landed inside the think frame the prefix exists to escape, and
/// the greedy generations beside this run are coherent prose.
const BASELINES: &[quality_common::ChipQuality] = &[quality_common::ChipQuality {
    brand_substr: "Apple M4 Max",
    perplexity: 12.6162,
    greedy_digest: "eaefc4e7aaf84b78c7014e5315e21ae02cf195b938c85526361e1d022b849efd",
    sampled_digest: "d621d1f7c504fe21fa3148ea825d2d5a3451c990aa3ac0c67ee087200f526b5a",
    source: "this port, 2026-09-08, Apple M4 Max, AC, 16 slots -- the 8-slot digest equals the 16-slot one (dense; Gotcha 9)",
}];

fn install_dir() -> Option<std::path::PathBuf> {
    std::env::var_os("TURBOSPARK_SPARK_INSTALL_DIR").map(std::path::PathBuf::from)
}

#[test]
#[ignore = "needs a real spark2_5 .gturbo install (TURBOSPARK_SPARK_INSTALL_DIR)"]
fn real_spark_install_quality_holds() {
    let Some(dir) = install_dir() else {
        eprintln!(
            "spark_quality_gate: TURBOSPARK_SPARK_INSTALL_DIR is not set; skipping. \
             Point it at a streamed Spark-X2.5-4B .gturbo install to run the gate."
        );
        return;
    };
    quality_common::run_quality_gate_with_assistant_prefix(&dir, BASELINES, SPARK_ASSISTANT_PREFIX);
}
