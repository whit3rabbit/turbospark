#![cfg(target_os = "macos")]
//! The memory oracle: runs the frozen community-protocol benchmark
//! against a REAL Gemma 4 `.gturbo` install and asserts this port stays
//! at or under the published Swift baselines.
//!
//! Not run by default (needs a ~14.6 GB install and takes minutes; use
//! --release or the tok/s numbers are meaningless):
//!
//!   MREFRUST_GEMMA4_INSTALL_DIR=~/models/gemma4.gturbo \
//!     cargo test -p mrefrust-bench --test memory_oracle --release -- --ignored --nocapture
//!
//! What is asserted:
//! - Peak `phys_footprint` (the exact counter the Swift baselines report,
//!   sampled at the same cadence) <= the documented Swift ceiling + ~5%
//!   headroom (the Swift docs' own repeat-run variance). Asserted always;
//!   memory sizing does not depend on the chip.
//! - Decode tok/s >= the documented Swift floor, per case, ONLY when the
//!   chip brand matches a baseline row. A tok/s failure here means "the
//!   port decodes slower than the Swift baseline on this hardware" (a
//!   true finding -- e.g. the missing MTLSharedEvent CB overlap, see
//!   DEVIATIONS.md), not a broken test.
//! - Every measured case must stop with `endOfTurn`, the frozen
//!   protocol's validity gate.

use mrefrust_bench::memory::{chip_brand_string, AppMemorySampler};
use mrefrust_bench::protocol::{swift_footer, PROTOCOL_CASES};
use mrefrust_bench::real_model::{open_model_runner, run_protocol_case};
use runtime::StopReason;

struct ChipBaseline {
    brand_substr: &'static str,
    footprint_ceiling_mib: u64,
    tok_s_floor: f64,
}

/// Swift docs/BENCHMARKS.md Gemma 4 rows, most specific substring first
/// ("Apple M2" also matches M2 Pro/Max, whose real floors are strictly
/// higher -- acceptable for a floor). Ceilings are the documented peak
/// footprint + ~5%, the Swift docs' own cross-run variance band; more
/// headroom would mask a regression on the order of one KV layer.
const BASELINES: &[ChipBaseline] = &[
    // M5 Pro 24GB: peak footprint 2,126-2,142 MiB, decode 31.01-35.17 tok/s.
    ChipBaseline {
        brand_substr: "Apple M5 Pro",
        footprint_ceiling_mib: 2250,
        tok_s_floor: 31.0,
    },
    // M2 8GB: peak footprint 1,776-1,971 MiB, decode 5.10-6.30 tok/s.
    ChipBaseline {
        brand_substr: "Apple M2",
        footprint_ceiling_mib: 2070,
        tok_s_floor: 5.1,
    },
];

/// Unknown chip: memory parity is chip-independent, so hold the loosest
/// documented ceiling; throughput is only reported.
const UNKNOWN_CHIP_FOOTPRINT_CEILING_MIB: u64 = 2250;

fn install_dir() -> Option<std::path::PathBuf> {
    std::env::var_os("MREFRUST_GEMMA4_INSTALL_DIR").map(std::path::PathBuf::from)
}

#[test]
#[ignore = "needs a real ~14.6 GB Gemma 4 .gturbo install (MREFRUST_GEMMA4_INSTALL_DIR)"]
fn real_install_peak_footprint_and_throughput_meet_swift_baselines() {
    let Some(dir) = install_dir() else {
        eprintln!(
            "memory_oracle: MREFRUST_GEMMA4_INSTALL_DIR is not set; skipping. \
             Point it at a repacked Gemma 4 .gturbo install to run the oracle."
        );
        return;
    };

    let brand = chip_brand_string();
    let baseline = brand
        .as_deref()
        .and_then(|b| BASELINES.iter().find(|row| b.contains(row.brand_substr)));
    match baseline {
        Some(row) => eprintln!(
            "memory_oracle: chip {:?} -> ceiling {} MiB, tok/s floor {}",
            brand, row.footprint_ceiling_mib, row.tok_s_floor
        ),
        None => eprintln!(
            "memory_oracle: chip {brand:?} not in the baseline table -> ceiling \
             {UNKNOWN_CHIP_FOOTPRINT_CEILING_MIB} MiB, tok/s reported but not asserted"
        ),
    }

    let (mut runner, tokenizer) = open_model_runner(&dir).expect("real install should open");
    let mut sampler = AppMemorySampler::new();

    let mut measured = Vec::new();
    for case in &PROTOCOL_CASES {
        // Frozen protocol: one discarded warmup, then the measured run.
        run_protocol_case(&mut runner, &tokenizer, case, &mut sampler)
            .unwrap_or_else(|e| panic!("{} warmup failed: {e}", case.id));
        let result = run_protocol_case(&mut runner, &tokenizer, case, &mut sampler)
            .unwrap_or_else(|e| panic!("{} failed: {e}", case.id));
        eprintln!(
            "{:<18} {}  peak so far {:.1} MiB",
            result.case_id,
            swift_footer(
                result.reason,
                result.prompt_tokens,
                result.prefill_seconds,
                result.new_tokens,
                result.decode_seconds
            ),
            sampler.peak_bytes().unwrap_or(0) as f64 / 1_048_576.0
        );
        measured.push(result);
    }

    // Protocol validity gate: a run that dies on maxTokens or a stray stop
    // is not comparable to the published rows.
    for result in &measured {
        assert_eq!(
            result.reason,
            StopReason::EndOfTurn,
            "{}: measured run must stop with endOfTurn (got {:?})",
            result.case_id,
            result.reason
        );
    }

    // The memory oracle proper.
    let peak = sampler.peak_bytes().expect("footprint sampling worked");
    let peak_mib = peak / 1_048_576;
    let ceiling_mib = baseline.map_or(UNKNOWN_CHIP_FOOTPRINT_CEILING_MIB, |row| {
        row.footprint_ceiling_mib
    });
    eprintln!("memory_oracle: session peak {peak_mib} MiB, ceiling {ceiling_mib} MiB");
    assert!(
        peak_mib <= ceiling_mib,
        "peak phys_footprint {peak_mib} MiB exceeds the Swift baseline ceiling \
         {ceiling_mib} MiB: this port uses more memory than the Swift engine"
    );

    // The throughput floor, when this chip has a published row.
    if let Some(row) = baseline {
        for result in &measured {
            let tok_s = result.tokens_per_second();
            assert!(
                tok_s >= row.tok_s_floor,
                "{}: {tok_s:.3} tok/s is under the Swift floor {} for {}",
                result.case_id,
                row.tok_s_floor,
                row.brand_substr
            );
        }
    } else {
        for result in &measured {
            eprintln!(
                "memory_oracle: {} decode {:.3} tok/s (not asserted, unknown chip)",
                result.case_id,
                result.tokens_per_second()
            );
        }
    }
}
