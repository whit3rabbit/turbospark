#![cfg(target_os = "macos")]
//! The memory oracle for TinyLlama-1.1B-Chat (Q6_K GGUF intake), against a
//! REAL dense `llama`-flow `.gturbo` install. Same body, same frozen
//! protocol and the same assertions as the mistral sibling (see
//! `oracle_common`); the install and the baseline rows differ.
//!
//! A SEPARATE TARGET, not a second `#[test]`: one real model per process.
//!
//! Not run by default (needs a ~1 GB install; use --release or the tok/s
//! numbers are meaningless):
//!
//!   TURBOSPARK_DENSE_LLAMA_INSTALL_DIR=~/models/tinyllama-dense.gturbo \
//!     cargo test -p turbospark-bench --test tinyllama_memory_oracle --release -- --ignored --nocapture
//!
//! Build the install with `tests/gguf_mixtral_install_network.rs` in
//! `turbospark-repack` (the `repacks_a_real_dense_llama_gguf` case), which
//! streams the published TinyLlama Q6_K file and drops the tokenizer
//! sidecars in beside it. The env var is the stream test's own, so one
//! spelling serves the pull and the gates.
//!
//! **THE WINDOW IS THE LLAMA FLOW'S OWN 8,192**, asserted equal to
//! `protocol_parameters`' resolution exactly as the mistral row asserts its
//! own -- and per AGENTS.md Gotcha 40 that makes this row MOSTLY an
//! assertion about the window: a 1.1B install's resident weights are
//! absent from `phys_footprint`, so what the ceiling pins is KV (22 layers
//! x 4 KV heads x 64 head dim x 2 x 2 bytes x 8,192 ~= 352 MiB) plus fixed
//! state. TinyLlama's trained context is 2,048, so `long-synthesis` runs
//! past trained RoPE and its TEXT is not meaningful; the oracle measures
//! memory and throughput, which are what the row is for. The checkpoint
//! that caught Gotcha 39 earns its row the same way it earned the gotcha.

mod oracle_common;

use turbospark_bench::protocol::{PROTOCOL_MAX_CONTEXT, PROTOCOL_MAX_NEW};
use turbospark_bench::real_model::protocol_parameters;

/// The dense `llama` flow's resolved window, pinned the way the mistral row
/// pins its own: the const block below fails the BUILD if the shared
/// resolution ever moves, so this row and `turbospark-bench` cannot drift
/// into measuring different workloads while reporting one number.
const TINYLLAMA_MAX_CONTEXT: u32 = 8192;

/// Per-chip rows, MOST SPECIFIC SUBSTRING FIRST. Frozen 2026-09-16 from
/// the first runs on this machine (release, 8,192 context): peak 349.1 MiB
/// against the ~352 MiB the KV arithmetic predicts -- the row is indeed
/// mostly the window, exactly as Gotcha 40 says for a dense install --
/// and short-explanation decode at 185.3 tok/s. The floor is 0.73x that
/// reading, the margin the other rows take against allocator and machine
/// jitter.
const BASELINES: &[oracle_common::ChipBaseline] = &[oracle_common::ChipBaseline {
    brand_substr: "Apple M4 Max",
    footprint_ceiling_mib: 380,
    tok_s_floor: 135.0,
    source: "this port, 2026-09-16, Apple M4 Max, 8192 context, short case only",
}];

/// Superseded by the frozen row above; kept as the documented first-run
/// placeholder so the freeze history reads in one file.
const UNKNOWN_CHIP_CEILING_MIB: u64 = 600;

#[test]
#[ignore = "needs a real TinyLlama dense install via TURBOSPARK_DENSE_LLAMA_INSTALL_DIR"]
fn real_tinyllama_install_peak_footprint_and_throughput_hold() {
    let Some(dir) = std::env::var_os("TURBOSPARK_DENSE_LLAMA_INSTALL_DIR") else {
        eprintln!("skipping: TURBOSPARK_DENSE_LLAMA_INSTALL_DIR is not set");
        return;
    };
    const {
        assert!(
            TINYLLAMA_MAX_CONTEXT > PROTOCOL_MAX_CONTEXT,
            "this target exists because the shared window is too small for the \
             dense-llama protocol; if that stops being true, delete the override"
        );
        assert!(
            protocol_parameters(model_io::ModelFamily::Llama).max_context == TINYLLAMA_MAX_CONTEXT,
            "this row's window and turbospark-bench's resolved window have drifted apart"
        );
    }
    oracle_common::run_oracle_over_cases(
        std::path::Path::new(&dir),
        BASELINES,
        UNKNOWN_CHIP_CEILING_MIB,
        TINYLLAMA_MAX_CONTEXT,
        PROTOCOL_MAX_NEW,
        SHORT_CASE_ONLY,
    );
}

/// SHORT-EXPLANATION ONLY, and that is a measured answer rather than a
/// truncation of the protocol. The shared oracle's validity gate requires
/// every measured case to stop `endOfTurn`, because a run that dies on
/// maxTokens is not comparable to published rows -- and this checkpoint
/// rambles to the budget on both longer cases (medium-review measured
/// MaxTokens on its first run, and `long-synthesis` at 2,940+ prompt
/// tokens is far past the 2,048 trained context besides). A 1.1B 2023
/// model's refusal to stop is a property of the model, not of the engine;
/// the row this oracle freezes asserts the short case's memory and
/// throughput, which is what a 1.1B KV-window row is for. Same shape as
/// `qwen4_exp`'s case subset, for a stop-reason reason instead of a
/// window one.
const SHORT_CASE_ONLY: &[turbospark_bench::protocol::ProtocolCase] =
    std::slice::from_ref(&turbospark_bench::protocol::PROTOCOL_CASES[0]);
