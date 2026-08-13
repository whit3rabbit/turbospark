#![cfg(target_os = "macos")]
//! The memory oracle for Mistral-7B-Instruct-v0.3, against a REAL DENSE
//! `llama` `.gturbo` install (ROADMAP M4). Same body, same frozen protocol
//! and the same four assertions as its three siblings (see `oracle_common`);
//! the install, the baseline rows and THE KV WINDOW differ.
//!
//! A SEPARATE TARGET, not a second `#[test]`, for the same reason the others
//! are: the footprint assertion is a whole-session peak, so two models in one
//! process measure against each other's high-water mark.
//!
//! Not run by default (needs a ~4.1 GB install; use --release or the tok/s
//! numbers are meaningless):
//!
//!   TURBOSPARK_MISTRAL_INSTALL_DIR=~/models/mistral7b-dense.gturbo \
//!     cargo test -p turbospark-bench --test mistral_memory_oracle --release -- --ignored --nocapture
//!
//! Build the install with `tests/gguf_mixtral_install_network.rs` in
//! `turbospark-repack` (`repacks_the_real_mistral`), which streams the
//! published `bartowski/Mistral-7B-Instruct-v0.3-GGUF` Q4_K_M and drops the
//! tokenizer sidecars in beside it.
//!
//! **THIS ROW RUNS AT 8,192 CONTEXT AND ITS SIBLINGS RUN AT 4,096.** That is
//! not a knob someone reached for: the protocol freezes the PROSE, and its
//! token count belongs to the checkpoint's tokenizer. `long-synthesis` is
//! 3,444 tokens under Mistral's 32k vocab against 2,842 under
//! Qwen3-30B-A3B's 152k, and `3444 + 1024 > 4096`, so at the shared default
//! the case does not run at all and the `endOfTurn` gate cannot be
//! satisfied on two of three cases. Raising `PROTOCOL_MAX_CONTEXT` itself
//! would resize KV for every family and move every already-frozen peak, so
//! the window is per-target. 8,192 is the next power of two above what the
//! protocol needs, with room for tokenizer drift, and is far inside the
//! model's own 32k.
//!
//! **THE WINDOW IS MOST OF WHAT THIS PARTICULAR CEILING ASSERTS**, which is
//! the one thing to understand before reusing the number. A dense install's
//! resident weights are NOT counted (AGENTS.md Gotcha 40: 4.07 GiB of
//! weights against a 684 MiB peak at 4,096 context, on two independent
//! counters), and there is no expert slot cache at all, so KV is the term
//! that is left: 32 layers x 8 KV heads x 128 head_dim x 2 x 2 bytes x
//! 8,192 = 1,024 MiB. Do NOT compare this row's ceiling against a sibling's
//! without noting the window; the difference between 684 and this is the
//! window, not the model.

mod oracle_common;

use turbospark_bench::protocol::PROTOCOL_MAX_CONTEXT;
use turbospark_bench::real_model::protocol_parameters;

/// The KV window this family's protocol run needs. See the module header.
///
/// Kept as a local constant rather than read from
/// [`protocol_parameters`] -- it carries this row's provenance, and the
/// `const` block in the test body asserts the two agree. `turbospark-bench
/// --model` resolves the same number from the install's family, so the
/// binary and this row measure one workload; that agreement is a build
/// failure to break, not a runtime one.
const MISTRAL_MAX_CONTEXT: u32 = 8192;

/// Per-chip rows for Mistral-7B-Instruct-v0.3 Q4_K_M, MOST SPECIFIC
/// SUBSTRING FIRST (`memory_oracle.rs` explains the lookup order).
///
/// No Swift row at any chip and there will not be one: the Swift original
/// has no dense `llama` support, so every row here is this port measuring
/// itself.
const BASELINES: &[oracle_common::ChipBaseline] = &[
    // M4 Max 36GB (the development machine; see CLAUDE.local.md).
    //
    // Frozen protocol, release, on AC, 16 expert-cache slots (inert here --
    // a dense install has no routed experts to cache), 8,192 context, first
    // session after the real checkpoint was streamed (2026-08-10), TWO
    // back-to-back readings:
    //   peak footprint     1,202.6 / 1,201.0 MiB
    //   short-explanation  30.447 / 27.919 tok/s
    //   medium-review      26.304 / 24.452 tok/s
    //   long-synthesis     18.673 / 16.338 tok/s
    // All three cases stopped endOfTurn both times; replay grew +0.00 MiB
    //   both times.
    //
    // THE SECOND READING IS WHY THE FLOOR IS 12 AND NOT 14. The peak
    // reproduces to 1.6 MiB, but throughput moved 12.5% on the slowest case
    // between two runs minutes apart on one machine -- the ordinary
    // cross-run spread this repo keeps warning about (AGENTS.md Gotcha 22).
    // A floor derived from a single reading would have sat 14% under it and
    // flaked on machine state rather than on a regression.
    //
    // **THE PEAK IS ALMOST ENTIRELY KV, AND THE WEIGHTS ARE ABSENT FROM IT.**
    // 32 layers x 8 KV heads x 128 head_dim x 2 (K and V) x 2 bytes x 8,192
    // context is 1,024 MiB, which leaves ~178 MiB for everything else. The
    // install's resident region is 4,371,570,688 bytes and NONE of it shows
    // up: measured at 683.9 MiB peak at 4,096 context by the bench sampler
    // and 684.2 MiB by `/usr/bin/time -l`, agreeing to under a MiB, with a
    // maximum RSS of 46.8 MiB (AGENTS.md Gotcha 40). Halving the window
    // roughly halves the number, which is what says KV is the term.
    //
    // So this ceiling is NOT comparable to the three MoE rows, twice over:
    // it is at a different window, and it is dominated by a term that is
    // negligible in theirs while their dominant term (the expert slot cache,
    // `slots x layers x expert_stride`) is zero here. The parity matrix has
    // to say so per row.
    //
    // Ceiling 1,300 is the measured peak + ~8%, in the same spirit as the
    // other rows: loose enough that allocator jitter cannot flake it, tight
    // enough that a doubling cannot hide. There is no expert-slot warming
    // here to widen the spread, which is why the two peaks agree so closely.
    oracle_common::ChipBaseline {
        brand_substr: "Apple M4 Max",
        footprint_ceiling_mib: 1300,
        // 0.73 of the SLOWER of the two readings of the slowest case, which
        // is the same margin the qwen3moe row takes and lands on the same
        // number (12.0 against its 16.036).
        tok_s_floor: 12.0,
        source: "this port, 2026-08-10, Apple M4 Max, AC, 8192 context",
    },
];

/// Ceiling for an unlisted chip. Memory sizing does not depend on the chip,
/// and on this family it is nearly all KV, which is a pure function of the
/// architecture and the window.
const UNKNOWN_CHIP_CEILING_MIB: u64 = 1300;

#[test]
#[ignore = "needs a real Mistral 7B dense install via TURBOSPARK_MISTRAL_INSTALL_DIR"]
fn real_mistral_install_peak_footprint_and_throughput_hold() {
    let Some(dir) = std::env::var_os("TURBOSPARK_MISTRAL_INSTALL_DIR") else {
        eprintln!("skipping: TURBOSPARK_MISTRAL_INSTALL_DIR is not set");
        return;
    };
    // Both operands are constants, so this is a compile-time invariant rather
    // than a runtime one: a `const` block fails the BUILD if the override
    // stops being an override, where a plain `assert!` would only fire on the
    // rare occasions this `#[ignore]`d target is run with the install present.
    const {
        assert!(
            MISTRAL_MAX_CONTEXT > PROTOCOL_MAX_CONTEXT,
            "this target exists because the shared window is too small for this \
             checkpoint's tokenizer; if that stops being true, delete the override \
             rather than leaving a silent divergence"
        );
        // The binary resolves its own window per family. If the two ever
        // disagree, this row and `turbospark-bench --model` are measuring
        // different workloads while reporting one number.
        assert!(
            protocol_parameters(model_io::ModelFamily::Llama).max_context == MISTRAL_MAX_CONTEXT,
            "this row's window and turbospark-bench's resolved window have drifted apart"
        );
    }
    oracle_common::run_oracle_at_context(
        std::path::Path::new(&dir),
        BASELINES,
        UNKNOWN_CHIP_CEILING_MIB,
        MISTRAL_MAX_CONTEXT,
    );
}
