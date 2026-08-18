#![cfg(target_os = "macos")]
//! The memory oracle for `gpt-oss-20b`, against a REAL `gptOss` `.gturbo`
//! install (ROADMAP M5). Same body and the same four assertions as
//! `memory_oracle.rs` (see `oracle_common`); the install, the rows, the
//! context window AND the generation budget differ.
//!
//! A SEPARATE TARGET, not a second `#[test]`: the footprint assertion is a
//! whole-session peak and no two families share a ceiling.
//!
//! Not run by default (needs a ~12 GB install; use --release or the tok/s
//! numbers are meaningless):
//!
//!   TURBOSPARK_GPTOSS_INSTALL_DIR=~/models/gptoss-20b.gturbo \
//!     cargo test -p turbospark-bench --test gptoss_memory_oracle --release -- --ignored --nocapture
//!
//! Build the install with `tests/gguf_gptoss_install_network.rs` in
//! `turbospark-repack`, which streams the published
//! `ggml-org/gpt-oss-20b-GGUF` MXFP4 file and drops the tokenizer sidecars
//! (including `chat_template.jinja`, which this family cannot render without)
//! in beside it.
//!
//! **THIS ROW MOVES TWO PARAMETERS AT ONCE AND BOTH ARE FORCED.** Every other
//! family's row is the protocol at 4,096 context and 1,024 new tokens; the
//! dense `llama` row already moved the first (`mistral_memory_oracle.rs`).
//! This one moves both, so it is comparable to NO other row here without
//! reading them.
//!
//! THE BUDGET IS THE INTERESTING ONE, because it is a property of the MODEL
//! rather than of its tokenizer. Harmony puts the model's reasoning in an
//! `analysis` channel BEFORE its answer, so the three protocol cases need
//! 818 / 1,780 / 1,211 tokens to reach `<|return|>` where every other family
//! finishes inside 1,024. At the shared budget two of three stop on
//! `maxTokens` and the validity gate refuses them -- correctly, since a
//! truncated run is not comparable to a completed one, and for a reason that
//! is not a defect.
//!
//! MEASURED TWICE, AND THE FIRST MEASUREMENT WAS WRONG IN A WAY WORTH
//! RECORDING. Each case was first run to completion greedily, giving
//! 818 / 1,780 / 1,211, and 2,048 looked like comfortable margin. Under the
//! protocol's OWN settings (T=0.2, top-k 64, top-p 0.95, the frozen per-case
//! seed) `medium-review` needs 2,153 -- sampling lengthens the reasoning
//! channel, and a budget derived from a greedy probe understates it. 3,072 is
//! set from the sampled number with ~40% margin. A reasoning model's answer
//! length is a DISTRIBUTION, not a constant, so this is margin rather than a
//! bound; if a future run ever trips the validity gate here, raise the budget
//! rather than reading it as a regression.
//!
//! THE WINDOW FOLLOWS FROM THE BUDGET. `long-synthesis` is 2,839 tokens under
//! this checkpoint's o200k vocab, and `2839 + 3072` does not fit 4,096.
//! Raising the SHARED constants was not an option in either direction: KV is
//! sized at open, so a wider window moves every already-frozen peak, and a
//! larger budget lets every other family generate further.

mod oracle_common;

/// Per-chip rows for `gpt-oss-20b`, MOST SPECIFIC SUBSTRING FIRST
/// (`memory_oracle.rs` explains the lookup order).
///
/// No Swift row at any chip and there will not be one: the Swift original has
/// no GGUF intake at all, so every row here is this port measuring itself.
const BASELINES: &[oracle_common::ChipBaseline] = &[
    // M4 Max 36GB (the development machine; see CLAUDE.local.md).
    //
    // Frozen protocol, release, on AC, 16 expert-cache slots, first session
    // after the real checkpoint was streamed (2026-08-12). TWO readings, the
    // second on an otherwise idle machine:
    //   peak footprint     5,417.2 then 5,421.1 MiB
    //   short-explanation  29.123 then 31.343 tok/s  (818 new tokens)
    //   medium-review      27.192 then 26.710        (2,153)
    //   long-synthesis     23.371 then 23.472        (1,108)
    // All three cases stopped endOfTurn on the second run; the first was at
    // the 2,048 budget and truncated `medium-review`, which is where the
    // 3,072 above comes from.
    //
    // THE FIRST READING WAS TAKEN WITH A RELEASE BUILD RUNNING BESIDE IT and
    // is 7% low on the shortest case as a result. It is kept because the
    // floor should come from the SLOWER of two readings (crate Gotcha 12) and
    // because the peak was unmoved by the contention -- memory reproduces
    // here to a few MiB where throughput does not.
    //
    // **THE PEAK IS THE HIGHEST OF ANY FAMILY HERE AND IT IS ARITHMETIC.**
    // The slot cache is `slots x layers x expert_stride`: 12.64 MiB per
    // expert x 24 layers x 16 slots is 4,854 MiB of slot capacity on its own,
    // against Qwen3-30B-A3B's 2,094 and Gemma 4's ~1,500. So this install
    // sits far ABOVE the 1.6-2.2 GiB band the README quotes, and the parity
    // matrix must not imply otherwise. It is still STREAMING and that is the
    // whole point of the family: the expert table is 9.5 GiB and only 4.7 of
    // it is ever resident, where Mixtral's 8 experts of 108.9 MiB wanted
    // 54.5 GiB and could not run here at all (AGENTS.md Gotcha 36).
    //
    // A SECOND READING WORTH RECORDING: 4,854 MiB of that peak is slot cache
    // and the measured total is only ~470 MiB more, which does NOT leave room
    // for the 1.8 GiB resident core. So the resident mapping is largely
    // uncounted here, as it is on the dense install (AGENTS.md Gotcha 40) and
    // unlike what Gotcha 19 says of the MoE installs it was written for. Not
    // chased further; recorded so the next person does not derive a
    // contradiction from the two gotchas.
    //
    // Ceiling 5,700 is the measured peak + ~5%, the same rule the other rows
    // use. Floor 16.0 is 0.68 of the slower long-synthesis reading (23.371),
    // slightly looser than the ~0.73 the `qwen3moe` and Mistral rows take,
    // because this row rests on one session and AGENTS.md Gotcha 22 is
    // explicit that cross-session absolutes here have repeatedly failed to
    // reproduce. Tighten it after a run on another day, not before.
    oracle_common::ChipBaseline {
        brand_substr: "Apple M4 Max",
        footprint_ceiling_mib: 5700,
        tok_s_floor: 16.0,
        source: "this port, 2026-08-12, Apple M4 Max, AC, 16 slots -- Swift has no GGUF intake",
    },
];

/// Unknown chip: memory sizing does not depend on the chip, so hold the one
/// measured ceiling. Throughput is reported but not asserted.
const UNKNOWN_CHIP_FOOTPRINT_CEILING_MIB: u64 = 5700;

/// See the module header: both of these are forced, and both have to be read
/// with the numbers.
///
/// Kept local rather than read from
/// `turbospark_bench::real_model::protocol_parameters` -- they carry this
/// row's provenance -- with the `const` block in the test body asserting the
/// two agree. `turbospark-bench --model` resolves the same pair from the
/// install's family, which is what lets a power capture of all three cases
/// be compared against this row at all.
const GPTOSS_MAX_CONTEXT: u32 = 8192;
const GPTOSS_MAX_NEW: u32 = 3072;

fn install_dir() -> Option<std::path::PathBuf> {
    std::env::var_os("TURBOSPARK_GPTOSS_INSTALL_DIR").map(std::path::PathBuf::from)
}

#[test]
#[ignore = "needs a real ~12 GB gpt-oss .gturbo install (TURBOSPARK_GPTOSS_INSTALL_DIR)"]
fn real_gpt_oss_install_peak_footprint_and_throughput_hold() {
    let Some(dir) = install_dir() else {
        eprintln!(
            "gptoss_memory_oracle: TURBOSPARK_GPTOSS_INSTALL_DIR is not set; skipping. \
             Point it at a streamed gpt-oss-20b .gturbo install to run the oracle."
        );
        return;
    };
    // Compile-time, so it fails the BUILD rather than only when this
    // `#[ignore]`d target runs with a 12 GB install present.
    const {
        let resolved =
            turbospark_bench::real_model::protocol_parameters(model_io::ModelFamily::GptOss);
        assert!(
            resolved.max_context == GPTOSS_MAX_CONTEXT && resolved.max_new == GPTOSS_MAX_NEW,
            "this row's window/budget and turbospark-bench's resolved pair have drifted apart"
        );
    }
    oracle_common::run_oracle_with_budget(
        &dir,
        BASELINES,
        UNKNOWN_CHIP_FOOTPRINT_CEILING_MIB,
        GPTOSS_MAX_CONTEXT,
        GPTOSS_MAX_NEW,
    );
}

/// The catalog half of this row, checked offline on every `cargo test`.
///
/// NOT `#[ignore]`d and needs no install: it asserts that the ceiling and
/// floor above still agree with the `measured` block in `models.json` they
/// were calibrated from. See `oracle_common::assert_agrees_with_catalog`.
#[test]
fn the_baselines_agree_with_the_catalogs_measured_rows() {
    oracle_common::assert_agrees_with_catalog(
        "gptoss-20b",
        BASELINES,
        GPTOSS_MAX_CONTEXT,
        turbospark_bench::protocol::PROTOCOL_EXPERT_CACHE_SLOTS as u32,
    );
}
