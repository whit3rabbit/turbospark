#![cfg(target_os = "macos")]
//! The quality gate for ROADMAP Phase S's 3-bit install of Gemma 4 26B-A4B
//! (`unsloth/gemma-4-26B-A4B-it-UD-Q3_K_M`, IQ3_XXS gate/up over IQ4_NL down).
//! `quality_common` has the reasoning behind every part of it.
//!
//! A SEPARATE TARGET with its OWN rows rather than an env var pointed at
//! `quality_gate.rs`, and that is the whole reason this file exists. The
//! existing rows are keyed on the CHIP and freeze the MLX INT4 install's
//! goldens; pointing an install var at a different artifact would assert
//! those digests against a different set of weights and fail for a reason
//! that is not a regression. The same trap was already recorded for the two
//! GGUF installs, which is why neither of them ever got a Phase Q run.
//!
//! WHAT THIS GATE CAN AND CANNOT SAY. It compares this install against ITS
//! OWN past, which is what freezes the IQ kernels against future change. It
//! does NOT say the 3-bit checkpoint is as good as the 4-bit one: the two
//! perplexities are on the same corpus and the same scoring, so they ARE
//! comparable, but the comparison belongs in `docs/BENCHMARKS.md` next to the
//! pre-ingest llama.cpp survey (38.0997 against the incumbent's 37.4176,
//! +1.82%) rather than in an assertion here. What licenses reading this
//! install's number against llama.cpp's is that they are the same bytes.
//!
//! Not run by default (needs the ~10 GB install
//! `gguf_iq_install_network.rs` writes; use --release):
//!
//!   TURBOSPARK_GEMMA4_IQ_INSTALL_DIR=~/models/gemma4-iq3.gturbo \
//!     cargo test -p turbospark-bench --test iq3_quality_gate --release -- --ignored --nocapture

mod quality_common;

/// Per-chip rows, MOST SPECIFIC SUBSTRING FIRST -- the lookup takes the first
/// `contains` hit.
///
/// A row cannot be invented: a golden digest is worth something only because
/// it was observed on a build known to generate coherent text. On a new chip,
/// run the gate once, read the printed values, and paste them in with the
/// date and power source. With no row the gate still runs and still asserts
/// everything that needs no baseline -- determinism across repeats,
/// byte-identical output at 8 and 16 expert-cache slots, and the
/// constrained-working-set throughput floor.
const BASELINES: &[quality_common::ChipQuality] = &[
    // M4 Max 36GB (the development machine; see CLAUDE.local.md).
    //
    // Frozen 2026-08-08 on AC, on the first install of the candidate, after
    // the three checks that say the number means something:
    //
    // 1. Greedy and sampled generations are coherent to 300-400 tokens.
    // 2. `gguf_nondeterminism_probe` reads ONE distinct output at 8, 16 and
    //    32 slots, and the SAME digest at all three, so nothing here depends
    //    on cache state (AGENTS.md Gotcha 27).
    // 3. `scripts/kld_llamacpp.py` against llama.cpp on the IDENTICAL GGUF
    //    bytes reads 0.00440 mean nats at 97.5% top-1, between a shape floor
    //    of 0.00051 and a backend floor of 0.03741. The port sits 8.5x closer
    //    to llama.cpp than ggml's own Metal and CPU paths sit to each other,
    //    which is what licenses reading this perplexity as the checkpoint's
    //    rather than as a kernel artifact.
    //
    // 38.3753 against llama.cpp's 38.0997 on the same bytes is +0.72%, inside
    // PERPLEXITY_REL_TOLERANCE and inside llama.cpp's own 1.2% Metal/CPU
    // spread on this file. Against the MLX INT4 install's 37.4176 it is
    // +2.56%, which is the QUANTIZATION and belongs in docs/BENCHMARKS.md.
    quality_common::ChipQuality {
        brand_substr: "Apple M4 Max",
        perplexity: 38.3753,
        greedy_digest: "46b97f42ed0894f51405be2c4ede85b38d9cc8d1494b19a4ece176530ba3802d",
        sampled_digest: "a8eafcb2a0ee17198d694e26de5ed83e0bcec47d7da88aed6c551783e510ef15",
        source: "this port, 2026-08-08, Apple M4 Max, AC, 16 slots, IQ3_XXS/IQ4_NL install",
    },
];

fn install_dir() -> Option<std::path::PathBuf> {
    std::env::var_os("TURBOSPARK_GEMMA4_IQ_INSTALL_DIR").map(std::path::PathBuf::from)
}

#[test]
#[ignore = "needs the real ~10 GB 3-bit Gemma 4 .gturbo install (TURBOSPARK_GEMMA4_IQ_INSTALL_DIR)"]
fn real_gemma4_iq3_install_quality_holds() {
    let Some(dir) = install_dir() else {
        eprintln!(
            "iq3_quality_gate: TURBOSPARK_GEMMA4_IQ_INSTALL_DIR is not set; skipping. \
             Point it at the install `gguf_iq_install_network.rs` writes to run the gate."
        );
        return;
    };
    quality_common::run_quality_gate(&dir, BASELINES);
}
