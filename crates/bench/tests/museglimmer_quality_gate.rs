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
    // Frozen 2026-08-15 on AC, release, 16 expert-cache slots, from TWO fresh
    // processes that agreed on the perplexity to the last digit and on both
    // digests to the last hex character. Only the tok/s moved between them
    // (16.288 then 19.886 at 16 slots), and no tok/s is frozen here.
    //
    // **6.2826 IS THE NUMBER THAT SAYS THE ASSISTANT PREFIX IS RIGHT**, and
    // that is most of what this row is for on this family. The generation
    // prompt ends at `<|start|>assistant` and the model's next emission is a
    // RECIPIENT; splicing the reference answer in raw would land it in no
    // message at all, which is the position that made gpt-oss read
    // 148,421.76 (crate Gotcha 13). A healthy single-digit number beside
    // coherent generations is what rules that out.
    //
    // The constrained arm reads 1.00x and 0.99x rather than the 0.85-0.94x
    // the MoE families show, which is the mechanism rather than a surprise:
    // `--expert-cache-slots` sizes a routed-expert cache and a DENSE model
    // has no routed experts, so the two arms differ only in noise. The 8-slot
    // digest EQUALS the 16-slot one, which is Gotcha 9's standing assertion
    // and is trivially satisfied here for the same reason.
    quality_common::ChipQuality {
        brand_substr: "Apple M4 Max",
        perplexity: 6.2826,
        greedy_digest: "fc1e4e58a6fd99757d3bc02cba8b0adb54a65d3dcfa0d34a08609b96ee453811",
        sampled_digest: "24fe355decf2f389be48f1d9e93946b1374cfb145809b55ae50b4124359b0bd4",
        source: "this port, 2026-08-15, Apple M4 Max, AC, 16 slots",
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
