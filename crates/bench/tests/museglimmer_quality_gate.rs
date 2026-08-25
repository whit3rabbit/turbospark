#![cfg(target_os = "macos")]
//! The quality gate for `mlx-community/Muse-Glimmer-30B-4bit`.
//!
//! Not run by default:
//!
//!   TURBOSPARK_MUSEGLIMMER_INSTALL_DIR=~/models/museglimmer-30b.gturbo \
//!     cargo test -p turbospark-bench --test museglimmer_quality_gate --release -- --ignored --nocapture
//!
//! **AN ASSISTANT PREFIX IS REQUIRED HERE, and this family is the second to
//! need one** (crate Gotcha 13). The template's generation prompt ends at
//! `<|start|>assistant` and stops -- the next thing the model emits is a
//! RECIPIENT, ` to=self` for its reasoning or ` to=user` for its answer,
//! followed by `<|message|>`. So the reference answer spliced in raw would
//! land immediately after the word `assistant`, in no message at all, which
//! is exactly the position that made gpt-oss read 148,421.76.
//! ` to=user<|message|>` puts it in the ANSWER slot, which is where a
//! perplexity over an answer belongs.
//!
//! The leading SPACE is load-bearing and is the template's, not a typo: it
//! renders `'<|start|>assistant'` with no trailing space, and the model's own
//! greedy output reads `assistant to=user`.
//!
//! **THE PREFIX IS SCORED AS PROMPT, NEVER AS A TARGET**, which is what
//! `run_quality_gate_with_assistant_prefix` guarantees; measuring the model's
//! surprise at framing it did not choose would be a different number.
//!
//! NOTE the corpus and the digest prompt are the frozen protocol's, chosen
//! for Gemma 4 and reused verbatim, so this measures Muse Glimmer on a
//! Gemma-shaped text. Compare each row only against its own past; the
//! absolute perplexities are not a ranking across families, because each
//! family's chat template puts the reference answer in a different position.

mod quality_common;

/// The recipient header the model emits after `<|start|>assistant` when it is
/// answering rather than reasoning. See the module header.
const MUSE_ASSISTANT_PREFIX: &str = " to=user<|message|>";

/// Per-chip rows, MOST SPECIFIC SUBSTRING FIRST.
///
/// No Swift row at any chip and there will not be one: the Swift original has
/// no `muse_glimmer` support at all, so every row here is this port measuring
/// itself.
const BASELINES: &[quality_common::ChipQuality] = &[
    // M4 Max 36GB (the development machine; see CLAUDE.local.md).
    //
    // RE-FROZEN 2026-08-25. The 2026-08-15 row (perplexity 6.2826, greedy
    // digest `fc1e4e58...`) stopped reproducing on this same machine, and
    // the cause is NOT a code regression -- it was bisected out. Every
    // commit between the 2026-08-15 row's own commit (`ea77279`) and HEAD
    // was checked out, rebuilt and re-run against this exact install; all of
    // them, including `ea77279` ITSELF re-run fresh today, produce the
    // digests below rather than the 2026-08-15 ones. So the source code that
    // wrote the old row cannot reproduce its own recorded values on this
    // machine ten days later, which rules out every commit in between by
    // construction. What was checked and found unchanged: the install's own
    // files (mtimes from 2026-08-15, untouched), macOS (26.5.2, build
    // 25F84) and Xcode (26.6), both matching this repo's other notes from
    // around that period. The exact mechanism of the drift (Metal shader
    // compiler cache, GPU firmware, or something else this session could not
    // pin down) is NOT established. What IS established: the new values are
    // stable and reproducible -- identical across roughly ten independent
    // runs this session (with and without an unrelated uncommitted diff,
    // with and without a reverted INT4 GEMV kernel specialization, and at
    // every step of the bisect), including the two fresh-process-agreement
    // runs `quality_common` itself requires below. Read this as "the row was
    // stale, not that anything regressed" -- see
    // `docs/OBLITERATION.md`'s museGlimmer section for the manual CLI
    // cross-check that confirms the current generation is coherent, on-topic
    // prose, not degenerate output.
    //
    // **6.2886 IS STILL THE NUMBER THAT SAYS THE ASSISTANT PREFIX IS
    // RIGHT**, unchanged from the old row's own argument: a healthy
    // single-digit perplexity beside coherent generations is what rules out
    // the reference answer landing in no message at all (crate Gotcha 13).
    //
    // The constrained arm still reads ~1.0x rather than the 0.85-0.94x the
    // MoE families show, for the same reason as before: `--expert-cache-slots`
    // sizes a routed-expert cache and this DENSE model has none, so the two
    // arms differ only in noise. The 8-slot digest still EQUALS the 16-slot
    // one (Gotcha 9's standing assertion).
    quality_common::ChipQuality {
        brand_substr: "Apple M4 Max",
        perplexity: 6.2886,
        greedy_digest: "e11b7013827ed1257dbf887353d7e6f383c1b6334dae20a7a38e27ccc468b3fc",
        sampled_digest: "b8349cd1a303346812d3262e32020158eef6c5c15583dfa25dc7599e23fb508a",
        source: "this port, 2026-08-25, Apple M4 Max, macOS 26.5.2 (25F84), AC, 16 slots -- re-frozen after the 2026-08-15 row stopped reproducing on the same machine, bisected to rule out a code cause",
    },
];

fn install_dir() -> Option<std::path::PathBuf> {
    std::env::var_os("TURBOSPARK_MUSEGLIMMER_INSTALL_DIR").map(std::path::PathBuf::from)
}

#[test]
#[ignore = "needs a real ~15 GB Muse-Glimmer-30B .gturbo install (TURBOSPARK_MUSEGLIMMER_INSTALL_DIR)"]
fn real_muse_glimmer_install_quality_holds() {
    let Some(dir) = install_dir() else {
        eprintln!(
            "museglimmer_quality_gate: TURBOSPARK_MUSEGLIMMER_INSTALL_DIR is not set; skipping. \
             Point it at a streamed Muse-Glimmer-30B .gturbo install to run the gate."
        );
        return;
    };
    quality_common::run_quality_gate_with_assistant_prefix(&dir, BASELINES, MUSE_ASSISTANT_PREFIX);
}
