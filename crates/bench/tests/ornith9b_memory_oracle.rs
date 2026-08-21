#![cfg(target_os = "macos")]
//! The memory oracle for `ornith-ai/Ornith-1.5-9B`, against a REAL `qwen35`
//! `.gturbo` install streamed from the published Q8_0 GGUF.
//!
//! A SEPARATE TARGET, not a second `#[test]`: the footprint assertion is a
//! whole-session peak, so one real model per process.
//!
//!   TURBOSPARK_ORNITH9B_INSTALL_DIR=~/models/ornith9b.gturbo \
//!     cargo test -p turbospark-bench --test ornith9b_memory_oracle --release -- --ignored --nocapture
//!
//! **THIS ROW VERIFIES A PROTOCOL PARAMETER THAT WAS MARKED UNVERIFIED.**
//! `real_model::protocol_parameters` puts `qwen3_5` in the shared
//! 4,096/1,024 group on the TOKENIZER's evidence -- the checkpoint declares
//! Qwen 3.6's 248,320-wide vocab under the same ChatML dialect, so the
//! protocol's ~2.9k-token long case should fit -- and its own comment says
//! the first memory-oracle run is what confirms it. It does: all three cases
//! reach endOfTurn, and `long-synthesis` is 2,940 prompt tokens plus 682
//! generated, i.e. 3,622 of 4,096 with the generation 33% under budget.

mod oracle_common;

const BASELINES: &[oracle_common::ChipBaseline] = &[
    // M4 Max 36GB (the development machine; see CLAUDE.local.md).
    //
    // Frozen 2026-08-20 on AC, release, 16 expert-cache slots, 4,096 context.
    //
    // **THE 8.9 GB OF RESIDENT WEIGHTS ARE ABSENT FROM THIS NUMBER**, which
    // is the one thing to understand before reusing it. AGENTS.md Gotcha 40
    // says to re-derive that per install shape rather than quote it, and this
    // is a fourth shape: a DENSE GGUF install, where Mistral 7B was dense
    // GGUF at half the size and qwen38 dense SAFETENSORS at nearly double.
    // It holds again.
    //
    // The accounting closes on the terms that are left, from shapes rather
    // than fitted to the measurement. This model is 32 layers with
    // `full_attention_interval: 4`, so only 8 of them hold KV:
    //   KV    8 full layers x 4 kv heads x 256 head_dim x 2 x 2 B
    //         = 32 KiB/token x 4,096                         = 128.0 MiB
    //   GDN   24 linear layers x 32 v heads x 128 x 128 x 4 B = 144.0 MiB
    //         (delta-rule S; fixed, does NOT grow with context)
    //   conv  24 x 8,192 x 4 taps x 4 B                       =   3.0 MiB
    //   ----------------------------------------------------------------
    //   sum                                                     275.0 MiB
    // against 436.5 measured, leaving ~160 MiB of process baseline and host
    // scratch -- unremarkable at a 248,320-wide vocab.
    //
    // NOTE the GDN term is LARGER than the KV term here, which inverts the
    // usual reading and is a property of the layer mask: three quarters of
    // this model's layers are recurrent, and a recurrent layer's state does
    // not grow with the window while a KV layer's is nothing but the window.
    //
    // Ceiling 520 is the measured peak + ~19%. The three cases' peaks span
    // 1.6 MiB (434.9 / 435.2 / 436.5), so there is almost nothing for the
    // margin to absorb -- a dense install has no expert-slot warming to
    // spread it (the 77 MiB band the gemma4 row documents) -- but a doubling
    // still cannot hide under it.
    oracle_common::ChipBaseline {
        brand_substr: "Apple M4 Max",
        footprint_ceiling_mib: 520,
        // 0.73 of the slowest case's reading (long-synthesis, 24.033), the
        // same margin the mistral, qwen3moe and qwen38 rows take.
        tok_s_floor: 17.5,
        source: "this port, 2026-08-20, Apple M4 Max, AC, 4096 context",
    },
];

/// Ceiling for an unlisted chip. Memory sizing does not depend on the chip,
/// and on this install it is KV plus a fixed recurrent state, both pure
/// functions of the architecture and the window.
const UNKNOWN_CHIP_CEILING_MIB: u64 = 520;

#[test]
#[ignore = "needs a real Ornith-1.5-9B install via TURBOSPARK_ORNITH9B_INSTALL_DIR"]
fn real_ornith9b_install_peak_footprint_and_throughput_hold() {
    let Some(dir) = std::env::var_os("TURBOSPARK_ORNITH9B_INSTALL_DIR") else {
        eprintln!("skipping: TURBOSPARK_ORNITH9B_INSTALL_DIR is not set");
        return;
    };
    oracle_common::run_oracle(
        std::path::Path::new(&dir),
        BASELINES,
        UNKNOWN_CHIP_CEILING_MIB,
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
        "ornith9b",
        BASELINES,
        turbospark_bench::protocol::PROTOCOL_MAX_CONTEXT,
        turbospark_bench::protocol::PROTOCOL_EXPERT_CACHE_SLOTS as u32,
    );
}
