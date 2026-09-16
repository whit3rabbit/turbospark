#![cfg(target_os = "macos")]
//! The quality gate for `prism-ml/Bonsai-27B-mlx-1bit`, against a REAL
//! `qwen35` `.gturbo` install. Same body and same measurements as
//! `quality_gate.rs` (see `quality_common`); only the install and the rows
//! differ.
//!
//! A SEPARATE TARGET, not a second `#[test]`: one real model per process,
//! the same rule the memory oracles are split under.
//!
//! Not run by default (needs a ~5 GB install; use --release):
//!
//!   TURBOSPARK_QWEN35_INSTALL_DIR=~/models/bonsai27b.gturbo \
//!     cargo test -p turbospark-bench --test bonsai_quality_gate --release -- --ignored --nocapture
//!
//! Build the install with `tests/qwen35_checkpoint_network.rs` in
//! `turbospark-repack`, which streams `prism-ml/Bonsai-27B-mlx-1bit` and
//! drops the tokenizer sidecars in beside it.
//!
//! **THE LAST PIECE OF THE ORIGINAL `qwen35` GAP.** The family shipped with
//! Bonsai-27B carrying no gate and no oracle; qwen38 and ternary have since
//! landed both, and the KL row (the strongest instrument, against MLX) has
//! been frozen since 2026-08-14 -- but a self-referential sentinel for THIS
//! install's own past has never existed. This file is it. Read with the
//! cross-engine row beside it: the KL says the kernels match MLX on the
//! same bytes; this gate says the numbers stay where they were.
//!
//! **NO ASSISTANT PREFIX**, for the ternary/qwen38 measured reason: this
//! family's template renders with `enable_thinking: false` and its
//! non-thinking branch emits a CLOSED, EMPTY `<think>` block, so the
//! reference answer already lands in the answer position. That rendering
//! was verified on ternary and qwen38, same family and same sidecar
//! template shape -- but read the rendered prompt before freezing a row
//! here anyway (`crates/tokenizer`'s `installed_template` target prints
//! it): a structured assistant slot scored in the wrong position reads in
//! the tens of thousands, which is a loud failure, not a quiet one.
//!
//! NOTE the corpus and the digest prompt are the frozen protocol's, chosen
//! for Gemma 4 and reused verbatim. Compare each row only against its own
//! past; 1-bit perplexity is a QAT property as much as a width property.

mod quality_common;

/// Per-chip rows for Bonsai-27B, MOST SPECIFIC SUBSTRING FIRST. Frozen
/// 2026-09-16 from the first run on this machine (two fresh processes
/// agreeing on the perplexity to the last digit and both digests to the
/// last hex character), then re-run to ASSERT.
///
/// **8.3554 IS A CROSS-INSTRUMENT MATCH, not just a freeze.** The frozen
/// KL section in `docs/BENCHMARKS.md` records this same install's
/// reference-answer perplexity, measured through a different driver
/// (`logit_dump` + `kld_mlx_affine.py`) on a different day, as 8.3554 --
/// identical here to the last digit. Numerics are thermal-invariant, so
/// the hot-machine first run was valid for the freeze where the oracle's
/// throughput rows were not.
///
/// The constrained arm's digest is byte-identical to the 16-slot one
/// (dense model, no expert cache; the arms differ only in order), which is
/// the assertion that matters there.
const BASELINES: &[quality_common::ChipQuality] = &[quality_common::ChipQuality {
    brand_substr: "Apple M4 Max",
    perplexity: 8.3554,
    greedy_digest: "68beca3645872afc637fd15f0aeea554dafb995e0d135b417ea020b082219fde",
    sampled_digest: "8ae2e262c2972a41c9992eac3b17fd34fd9ecf1d80db1a054455fbd1a0489ff9",
    source: "this port, 2026-09-16, Apple M4 Max, 16 slots",
}];

fn install_dir() -> Option<std::path::PathBuf> {
    std::env::var_os("TURBOSPARK_QWEN35_INSTALL_DIR").map(std::path::PathBuf::from)
}

#[test]
#[ignore = "needs a real Bonsai-27B .gturbo install via TURBOSPARK_QWEN35_INSTALL_DIR"]
fn real_bonsai_install_quality_holds() {
    let Some(dir) = install_dir() else {
        eprintln!(
            "bonsai_quality_gate: TURBOSPARK_QWEN35_INSTALL_DIR is not set; skipping. \
             Point it at a streamed Bonsai-27B .gturbo install to run the gate."
        );
        return;
    };
    quality_common::run_quality_gate(&dir, BASELINES);
}
