#![cfg(target_os = "macos")]
//! The memory oracle for Phi-3.5-mini-instruct (dense `llama` flow, LongRoPE),
//! against a REAL `.gturbo` install streamed from
//! `bartowski/Phi-3.5-mini-instruct-GGUF` Q4_K_M. Same body and assertions as
//! its siblings (see `oracle_common`).
//!
//! Not run by default (needs the 2.2 GiB install; use --release or the tok/s
//! numbers are meaningless):
//!
//!   TURBOSPARK_PHI3_INSTALL_DIR=~/.turbospark/models/text/phi35-q4km.gturbo \
//!     cargo test -p turbospark-bench --test phi3_memory_oracle --release -- --ignored --nocapture
//!
//! **TWO OF THE THREE PROTOCOL CASES.** `medium-review` stops on `maxTokens` at
//! the shared 1,024-token budget on BOTH readings (the model keeps writing), and
//! the validity gate correctly refuses a truncated run, so this target drives
//! `short-explanation` and `long-synthesis`, which end on `endOfTurn`: the
//! `lfm3` precedent. A budget raise would be a per-target choice that needs its
//! own frozen row; none was measured.
//!
//! **8,192 CONTEXT, AND THE PEAK IS KV.** Phi-3.5 is MHA (32 kv heads over
//! head_dim 96, 32 layers): 32 x 32 x 96 x 2 x 2 bytes x 8,192 = 3,072 MiB of
//! the 3,246 MiB peak. The 2.2 GiB of weights are not counted (Gotcha 40). The
//! prose is 3,413 tokens under this 32k vocabulary, so the shared 4,096 window
//! could not hold `long-synthesis` plus its answer.

mod oracle_common;

use turbospark_bench::protocol::PROTOCOL_MAX_CONTEXT;
use turbospark_bench::real_model::protocol_parameters;

const PHI3_MAX_CONTEXT: u32 = 8192;

/// The two cases that complete: short-explanation and long-synthesis.
const CASES: [turbospark_bench::protocol::ProtocolCase; 2] = [
    turbospark_bench::protocol::PROTOCOL_CASES[0],
    turbospark_bench::protocol::PROTOCOL_CASES[2],
];

/// Apple M4 Max 36GB, release, AC power, 8,192 context, 16 expert-cache slots
/// (inert: dense), two back-to-back readings on 2026-11-20 (the machine was NOT
/// idle: other processes held the load average near 5 to 6, which is part of
/// why the floor takes the usual margin):
///   peak footprint     3,230.1 / 3,246.3 MiB
///   short-explanation  51.603 / 51.712 tok/s
///   long-synthesis     24.425 / 24.392 tok/s
/// Both stopped endOfTurn; replay grew +0.00 MiB. medium-review: maxTokens.
///
/// Ceiling 3,500 is the higher peak + ~8%. Floor 17.8 is 0.73 of the slower
/// reading of the slowest case, the same margin the other rows take.
const BASELINES: &[oracle_common::ChipBaseline] = &[oracle_common::ChipBaseline {
    brand_substr: "Apple M4 Max",
    footprint_ceiling_mib: 3500,
    tok_s_floor: 17.8,
    source: "this port, 2026-11-20, Apple M4 Max, AC, 8192 context, two cases",
}];

/// Memory sizing does not depend on the chip, and here it is nearly all KV.
const UNKNOWN_CHIP_CEILING_MIB: u64 = 3500;

#[test]
#[ignore = "needs a real Phi-3.5 install via TURBOSPARK_PHI3_INSTALL_DIR"]
fn real_phi3_install_peak_footprint_and_throughput_hold() {
    let Some(dir) = std::env::var_os("TURBOSPARK_PHI3_INSTALL_DIR") else {
        eprintln!("skipping: TURBOSPARK_PHI3_INSTALL_DIR is not set");
        return;
    };
    // `Llama` resolves 8,192 for the dense half; this target must not drift
    // from what `turbospark-bench --model` resolves for the same install.
    const {
        assert!(
            PHI3_MAX_CONTEXT > PROTOCOL_MAX_CONTEXT,
            "this target exists because the shared window is too small"
        );
        assert!(
            protocol_parameters(model_io::ModelFamily::Llama).max_context == PHI3_MAX_CONTEXT,
            "this row's window and turbospark-bench's resolved window have drifted apart"
        );
    }
    oracle_common::run_oracle_over_cases(
        std::path::Path::new(&dir),
        BASELINES,
        UNKNOWN_CHIP_CEILING_MIB,
        PHI3_MAX_CONTEXT,
        turbospark_bench::protocol::PROTOCOL_MAX_NEW,
        &CASES,
    );
}

/// The catalog half, offline on every `cargo test`: the ceiling and floor above
/// still agree with the `measured` block in `models.json` they came from.
#[test]
fn the_baselines_agree_with_the_catalogs_measured_rows() {
    oracle_common::assert_agrees_with_catalog(
        "phi35-mini",
        BASELINES,
        PHI3_MAX_CONTEXT,
        turbospark_bench::protocol::PROTOCOL_EXPERT_CACHE_SLOTS as u32,
    );
}
