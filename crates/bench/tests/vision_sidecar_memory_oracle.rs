//! Multi-page vision memory oracle THROUGH A SIDECAR-ATTACHED TRUNK (ROADMAP
//! P1 item 1): the same four-round, largest-first, content-asserted walk as
//! `vision_memory_oracle.rs`, over a TEXT-ONLY `qwen38-27b.gturbo` trunk with
//! the standalone `qwen38-vision-tower.gturbo-vision` sidecar attached.
//!
//! # Why a separate target rather than a second `#[test]`
//!
//! `oracle_common`'s ONE MODEL PER PROCESS rule: the footprint assertion is
//! against a WHOLE-SESSION peak, and two tests in one binary run on parallel
//! threads in one process, so each would be measured against the other's
//! high-water mark. Separate integration-test targets are separate binaries,
//! which cargo runs one after another.
//!
//! # What this arm can see that the combined arm cannot
//!
//! The sidecar's tower weights are opened by `VisionTower::open_with_sidecar`
//! -- a separate resident-index read, mmap and dtype backstop from the
//! trunk's -- so a sidecar that bound the wrong weights, or leaked its own
//! resident mapping across pages, passes the synthetic byte-identity fixture
//! (same tower bytes by construction) and fails HERE, on measured peaks and
//! transcribed content. The memory STORY should be the combined install's
//! (same tower, same scratch, same trunk): the ceiling below starts from the
//! combined arm's measured numbers and is re-derived from this target's own
//! first runs.
//!
//! # No catalog tie, deliberately
//!
//! `the_baselines_agree_with_the_catalogs_measured_rows` in the combined
//! oracle ties ITS row to `models.json`'s `qwen38-27b-vision` measured block.
//! A sidecar-attached trunk is not a catalog install (it is two installs and
//! an attach), so this target keeps its baselines LOCAL and reports tok/s
//! rather than asserting a catalog-agreed floor on chips it has not run on.
//!
//! # Setup
//!
//! The same three fixture pages and the same trunk/sidecar pair as the
//! parity gate's sidecar arm:
//!
//! ```sh
//! TURBOSPARK_QWEN38_TRUNK_INSTALL_DIR=~/models/qwen38-27b.gturbo \
//! TURBOSPARK_VISION_SIDECAR_DIR=~/.turbospark/models/qwen38-vision-tower.gturbo-vision \
//!   cargo test -p turbospark-bench --test vision_sidecar_memory_oracle --release -- \
//!   --ignored --nocapture
//! ```
//!
//! `TURBOSPARK_VISION_ORACLE_PAGES_DIR` overrides the pages directory, same
//! default as the combined oracle.
#![cfg(target_os = "macos")]

use std::path::PathBuf;

use turbospark_bench::memory::{chip_brand_string, AppMemorySampler};
use turbospark_bench::protocol::PROTOCOL_EXPERT_CACHE_SLOTS;
use turbospark_bench::real_model_open::open_model_runner_with_context_and_vision_sidecar;
use turbospark_vision_io::VisionSpecialIds;

mod oracle_common;
mod vision_oracle_rounds;
use vision_oracle_rounds::{params_from, run_rounds, RoundResult};

/// Measured on the real trunk+sidecar pair, Apple M4 Max, BATTERY (86%,
/// recorded per AGENTS.md Gotcha 22), 2026-09-10, context 4,096, 16
/// expert-cache slots (inert on this dense trunk): peaks read 855.6 /
/// 856.5 / 871.3 / 872.3 MiB across the four rounds -- within a few MiB of
/// the combined install's 854.9 / 855.9 / 870.6 / 785.3 (same tower, same
/// scratch, same trunk), which is the memory story this target exists to
/// hold: the sidecar's own resident mapping adds nothing the peak counter
/// sees. Flat-peak growth was +16.7 MiB (round 3 above round 0), inside
/// the slack. Ceiling 950 MiB -- the highest reading plus ~9%, the same
/// derivation and value as the combined arm's.
const CEILING_MIB: u64 = 950;

/// Growth allowed between the first round's peak and the final (repeated
/// largest-page) round's, mirroring the combined arm's 64 MiB: catch a
/// leaked PAGE (hundreds of MiB), tolerate allocator jitter. Measured
/// growth: +16.7 MiB.
const STEADY_STATE_SLACK_MIB: u64 = 64;

/// Same window as the combined arm: KV dominates a dense install's counted
/// footprint (AGENTS.md Gotcha 40), so the two targets are only comparable
/// at the same window and slot count.
const VISION_ORACLE_MAX_CONTEXT: u32 = 4096;

/// Local per-chip rows; see the header for why this target does not tie to
/// `models.json`.
const BASELINES: &[oracle_common::ChipBaseline] = &[oracle_common::ChipBaseline {
    brand_substr: "Apple M4 Max",
    footprint_ceiling_mib: CEILING_MIB,
    // The COMBINED arm's floor, kept here rather than a sidecar-specific
    // 0.73-of-slowest derivation on purpose: the first sidecar reading was
    // on BATTERY (decode 15.377-18.801 tok/s), and a floor frozen from one
    // battery reading carries that session's thermal state. 9.0 is the
    // number the combined arm froze on AC; an AC sidecar run that wants a
    // tighter floor re-derives it then.
    tok_s_floor: 9.0,
    source: "this port, 2026-09-10, Apple M4 Max, battery (86%), 4096 context, one reading of \
             four rounds; floor inherited from the combined arm's AC row",
}];

fn env_dir(key: &str) -> Option<PathBuf> {
    let raw = std::env::var(key).ok()?;
    if raw.trim().is_empty() {
        return None;
    }
    Some(PathBuf::from(shellexpand(&raw)))
}

fn shellexpand(raw: &str) -> String {
    match raw.strip_prefix("~/") {
        Some(rest) => format!("{}/{rest}", std::env::var("HOME").unwrap_or_default()),
        None => raw.to_string(),
    }
}

#[test]
#[ignore = "needs a real text-only trunk (TURBOSPARK_QWEN38_TRUNK_INSTALL_DIR) and the \
            standalone sidecar (TURBOSPARK_VISION_SIDECAR_DIR)"]
fn peak_footprint_is_flat_across_pages_through_a_sidecar_attached_trunk() {
    let Some(trunk) = env_dir("TURBOSPARK_QWEN38_TRUNK_INSTALL_DIR") else {
        eprintln!("vision_sidecar_memory_oracle: TURBOSPARK_QWEN38_TRUNK_INSTALL_DIR is not set; skipping.");
        return;
    };
    let Some(sidecar) = env_dir("TURBOSPARK_VISION_SIDECAR_DIR") else {
        eprintln!(
            "vision_sidecar_memory_oracle: TURBOSPARK_VISION_SIDECAR_DIR is not set; skipping."
        );
        return;
    };
    let pages_dir = env_dir("TURBOSPARK_VISION_ORACLE_PAGES_DIR")
        .unwrap_or_else(|| PathBuf::from(shellexpand("~/models/vision-probe-qwen38/imgs/oracle")));

    // ASSERT THE FIXTURE DISCRIMINATES before opening: a trunk that already
    // carries its own tower would re-measure the combined install's path
    // and prove nothing about the sidecar's.
    let trunk_arch = repack::peek_manifest_arch(&trunk).expect("trunk manifest peeks");
    assert!(
        !trunk_arch.vision.is_active(),
        "the trunk must be TEXT-ONLY; point the var at qwen38-27b.gturbo, not the \
         combined vision install"
    );

    let (mut runner, tokenizer) = open_model_runner_with_context_and_vision_sidecar(
        &trunk,
        PROTOCOL_EXPERT_CACHE_SLOTS,
        VISION_ORACLE_MAX_CONTEXT,
        &sidecar,
    )
    .unwrap_or_else(|e| panic!("the trunk+sidecar pair should open: {e}"));
    assert!(
        runner.has_vision_tower(),
        "the sidecar attach must activate the vision capability"
    );

    // The vision config on the runner IS the sidecar's now: params and the
    // special ids are read POST-ATTACH, which is the whole difference from
    // the combined arm's peeked arch.
    let params = params_from(runner.vision_config());
    let vision = runner.vision_config();
    let special = VisionSpecialIds {
        vision_start: vision.vision_start_token_id as i32,
        image_pad: vision.image_token_id as i32,
    };

    eprintln!(
        "vision_sidecar_memory_oracle: context={VISION_ORACLE_MAX_CONTEXT}, \
         expert_cache_slots={PROTOCOL_EXPERT_CACHE_SLOTS} (inert on this dense trunk), \
         ceiling={CEILING_MIB} MiB, steady_state_slack={STEADY_STATE_SLACK_MIB} MiB"
    );

    let mut sampler = AppMemorySampler::new();
    let rounds: Vec<RoundResult> = run_rounds(
        &mut runner,
        &tokenizer,
        &params,
        special,
        &pages_dir,
        VISION_ORACLE_MAX_CONTEXT,
        "vision_sidecar_memory_oracle",
        &mut sampler,
    );

    for r in &rounds {
        assert!(
            r.peak_mib <= CEILING_MIB as f64,
            "{}: peak {:.1} MiB exceeds the {CEILING_MIB} MiB ceiling",
            r.label,
            r.peak_mib
        );
    }

    let min_tok_s = rounds
        .iter()
        .map(|r| r.decode_tok_s)
        .fold(f64::INFINITY, f64::min);
    let max_tok_s = rounds
        .iter()
        .map(|r| r.decode_tok_s)
        .fold(f64::NEG_INFINITY, f64::max);
    eprintln!(
        "vision_sidecar_memory_oracle: decode tok/s across {} rounds: {min_tok_s:.3} min, \
         {max_tok_s:.3} max",
        rounds.len()
    );

    let brand = chip_brand_string();
    let baseline = brand
        .as_deref()
        .and_then(|b| BASELINES.iter().find(|row| b.contains(row.brand_substr)));
    if let Some(row) = baseline {
        assert!(
            min_tok_s >= row.tok_s_floor,
            "decode fell to {min_tok_s:.3} tok/s, below the {} tok/s floor from {} \
             (chip {brand:?})",
            row.tok_s_floor,
            row.source
        );
    } else {
        eprintln!(
            "vision_sidecar_memory_oracle: chip {brand:?} not in the baseline table -> tok/s \
             reported but not asserted"
        );
    }

    // THE FLAT-PEAK CLAIM, on the sidecar's own residency: the final round
    // repeats the FIRST (largest) page, so growth here would be the sidecar
    // tower's mapping or scratch accumulating across pages.
    let first_peak = rounds[0].peak_mib;
    let last_peak = rounds[rounds.len() - 1].peak_mib;
    let growth = last_peak - first_peak;
    assert!(
        growth <= STEADY_STATE_SLACK_MIB as f64,
        "peak grew {growth:.1} MiB from the first large-page round ({first_peak:.1} MiB) to \
         the repeated large-page round ({last_peak:.1} MiB), past the {STEADY_STATE_SLACK_MIB} \
         MiB slack: the SIDECAR tower's resources may be accumulating rather than being \
         dropped per page"
    );

    // Engagement: the rounds above must have run the SIDECAR's tower, which
    // a text-only trunk cannot satisfy any other way, asserted after the
    // measurement so a failure reads as "the wrong tower answered".
    assert_eq!(
        runner.vision_is_sidecar(),
        Some(true),
        "the rounds just measured must have come from the SIDECAR's tower"
    );
}
