//! Multi-page vision memory oracle (ROADMAP M-V9), against the real
//! `qwen38-27b-vision.gturbo` install.
//!
//! # What this asserts, and why it needed a new target
//!
//! `crates/cli`'s `--image-batch` already walks pages over ONE open runner,
//! with the tower's scratch (`VisionScratch`) allocated per page and dropped
//! with the embedding -- that is the whole constant-memory story the vision
//! design exists for. Nothing asserted it. This target is the assertion:
//! peak `phys_footprint` over a run of pages of DIFFERENT sizes must not grow
//! past what the first (largest) page establishes.
//!
//! A separate integration-test target for `oracle_common`'s own reason: the
//! footprint assertion is a WHOLE-SESSION peak, so a second model opened in
//! the same process would be measured against the first one's high-water
//! mark. This target opens exactly one install.
//!
//! # Two traps this design answers directly
//!
//! **Same-size pages would prove nothing.** Three identical pages read a
//! flat peak whether or not `VisionScratch` is actually dropped between
//! them -- a leaked constant-size scratch and a correctly-freed one look
//! identical on that input. The four rounds below vary size by roughly 7x
//! (small/medium/large) and run the LARGEST page FIRST, so the peak the
//! ceiling is checked against is established by the biggest scratch
//! allocation up front; every following round (including a final repeat of
//! the largest page, the real steady-state proof) can only fail to grow.
//!
//! **A flat peak cannot prove the pages were actually read.** That is
//! exactly the M-V5 bug shape (`docs/VISION.md`, "The map survives
//! `reset()`"): every length and count agreed while the model answered
//! fluently about a page it had never seen. Each round therefore also
//! transcribes its page and asserts the output contains a substring unique
//! to THAT page -- the trailing random float
//! `scripts/make_vision_test_page.py` draws right after its shared word
//! pool. The `NNNNN | ...` line-number PREFIX these pages carry is a pixel
//! position, identical across every page at a given font size, so it
//! cannot discriminate; the trailing float can, because it comes from a
//! per-page-seeded draw.
//!
//! # This is a DENSE install, so the ceiling is mostly the context window
//!
//! `qwen38-27b-vision.gturbo` carries no expert slot cache (AGENTS.md Gotcha
//! 40: a dense install's resident weights are not counted by
//! `phys_footprint` at all), so KV dominates whatever this measures. The
//! window and slot count are printed on every run for that reason
//! (`oracle_common`'s convention, AGENTS.md Gotcha 58) -- this ceiling is not
//! comparable to any other family's without them.
//!
//! # Catalog agreement
//!
//! `qwen38-27b-vision` has its own `models.json` row (a THIRD entry beside
//! `qwen38-27b`, deliberately: adding tower bytes to the row backing that
//! family's frozen oracle and quality-gate rows would force a re-freeze for a
//! component neither gate exercises -- see `CLAUDE.local.md`).
//! `the_baselines_agree_with_the_catalogs_measured_rows` ties `BASELINES`
//! below to that row's `measured` block, offline, exactly as every other
//! family's oracle does.
//!
//! # Setup
//!
//! Three fixture pages, largest first, each a distinct size and a distinct
//! `--seed` (same seed at different sizes would make a smaller page's
//! content a truncated PREFIX of a larger one's, which could not tell a
//! stale embedding from a fresh one):
//!
//! ```sh
//! uv run --python 3.12 --with pillow -- \
//!   scripts/make_vision_test_page.py ~/models/vision-probe-qwen38/imgs/oracle/large.png \
//!   --size 1536 1536 --seed 41
//! uv run --python 3.12 --with pillow -- \
//!   scripts/make_vision_test_page.py ~/models/vision-probe-qwen38/imgs/oracle/medium.png \
//!   --size 1024 1280 --seed 42
//! uv run --python 3.12 --with pillow -- \
//!   scripts/make_vision_test_page.py ~/models/vision-probe-qwen38/imgs/oracle/small.png \
//!   --size 512 640 --seed 43
//! ```
//!
//! Then:
//!
//! ```sh
//! TURBOSPARK_QWEN38_VISION_INSTALL_DIR=~/models/qwen38-27b-vision.gturbo \
//!   cargo test -p turbospark-bench --test vision_memory_oracle --release -- \
//!   --ignored --nocapture
//! ```
//!
//! `TURBOSPARK_VISION_ORACLE_PAGES_DIR` overrides where the three fixtures
//! are read from; it defaults to the path the commands above write to.
#![cfg(target_os = "macos")]

use std::path::PathBuf;

use turbospark_bench::memory::{chip_brand_string, AppMemorySampler};
use turbospark_bench::protocol::PROTOCOL_EXPERT_CACHE_SLOTS;
use turbospark_bench::real_model::open_model_runner_with_context;
use turbospark_vision_io::VisionSpecialIds;

mod oracle_common;
mod vision_oracle_rounds;
use vision_oracle_rounds::{params_from, run_rounds, RoundResult};

/// Per-chip rows for `qwen38-27b-vision`, most specific substring first
/// (`memory_oracle.rs` explains the lookup order). Only one chip has ever
/// run this target.
const BASELINES: &[oracle_common::ChipBaseline] = &[oracle_common::ChipBaseline {
    brand_substr: "Apple M4 Max",
    // Matches CEILING_MIB below; the two have to move together, and
    // `the_baselines_agree_with_the_catalogs_measured_rows` is what notices
    // if they stop.
    footprint_ceiling_mib: CEILING_MIB,
    // 0.73 of the slowest reading (12.355 tok/s, the large-page round --
    // slowest because it is the FIRST forward pass this process makes, a
    // cold GPU per AGENTS.md Gotcha 20, not because the page is large: the
    // repeated large-page round reads 15.997, faster than medium or small).
    // The same margin qwen38-27b, qwen3moe and mistral7b's rows take.
    tok_s_floor: 9.0,
    source: "this port, 2026-08-30, Apple M4 Max, AC, 4096 context, one reading of four rounds",
}];

fn env_dir(key: &str) -> Option<PathBuf> {
    let raw = std::env::var(key).ok()?;
    if raw.trim().is_empty() {
        return None;
    }
    Some(PathBuf::from(shellexpand(&raw)))
}

/// `~` only, matching every other real-install test in this crate. A full
/// shell expansion here would be a second, worse shell.
fn shellexpand(raw: &str) -> String {
    match raw.strip_prefix("~/") {
        Some(rest) => format!("{}/{rest}", std::env::var("HOME").unwrap_or_default()),
        None => raw.to_string(),
    }
}

/// One oracle page's markers, sizes and the transcription question live in
/// the shared `vision_oracle_rounds` module beside the sidecar target's copy
/// of this loop, so the two oracles stay one instrument.
///
/// Covers the largest page's ~2,304 merged tokens plus rendering overhead
/// and the generation budget above, with headroom -- see the module header
/// for why this ceiling is mostly a statement about the WINDOW rather than
/// about the vision path (AGENTS.md Gotcha 40).
const VISION_ORACLE_MAX_CONTEXT: u32 = 4096;

/// Measured on the real install, Apple M4 Max, AC, 2026-08-29, four rounds
/// (large/medium/small/large), context 4,096, 16 expert-cache slots (inert
/// on this dense install): peaks read 854.9 / 855.9 / 870.6 / 785.3 MiB --
/// the repeated largest-page round came in BELOW the first, which is the
/// flat-peak claim holding decisively rather than by a thin margin. Ceiling
/// set from the highest reading (870.6) plus ~8% margin, in the same
/// spirit as `mistral_memory_oracle.rs`'s row -- loose enough that
/// allocator jitter cannot flake it, tight enough that a doubling cannot
/// hide. This is a DENSE install (see the module header): with no expert
/// slot cache, this number is almost entirely the 4,096-token KV window
/// plus the tower's fixed 2-slot residency, which is why it sits so far
/// under the other families' 1,700-5,700 MiB rows despite carrying a
/// vision tower those do not.
const CEILING_MIB: u64 = 950;

/// Growth allowed between the first round's peak and the final (repeated
/// largest-page) round's. Vision scratch at this page size is tens to a
/// few hundred MiB, far above `oracle_common`'s 8 MiB text-decode slack, so
/// this is set to catch a leaked PAGE (hundreds of MiB) while tolerating
/// ordinary allocator jitter and the KV cache's own growth as more tokens
/// are decoded across four rounds. The measured round0-vs-round3 delta was
/// -69.6 MiB (round 3 lower), comfortably inside this either way.
const STEADY_STATE_SLACK_MIB: u64 = 64;

/// The combined-install oracle body: open, run the shared four rounds, then
/// assert THIS target's ceiling, tok/s floor and flat-peak constants.
#[test]
#[ignore = "needs a real vision install via TURBOSPARK_QWEN38_VISION_INSTALL_DIR"]
fn peak_footprint_is_flat_across_pages_of_different_sizes() {
    let Some(install) = env_dir("TURBOSPARK_QWEN38_VISION_INSTALL_DIR") else {
        eprintln!(
            "vision_memory_oracle: TURBOSPARK_QWEN38_VISION_INSTALL_DIR is not set; skipping."
        );
        return;
    };
    let pages_dir = env_dir("TURBOSPARK_VISION_ORACLE_PAGES_DIR")
        .unwrap_or_else(|| PathBuf::from(shellexpand("~/models/vision-probe-qwen38/imgs/oracle")));

    let (mut runner, tokenizer) = open_model_runner_with_context(
        &install,
        PROTOCOL_EXPERT_CACHE_SLOTS,
        VISION_ORACLE_MAX_CONTEXT,
    )
    .unwrap_or_else(|e| panic!("the vision install should open: {e}"));
    assert!(
        runner.has_vision_tower(),
        "{} declares no vision tower; point this at the install streamed WITH \
         vision_tower.* (see CLAUDE.local.md)",
        install.display()
    );

    let arch = repack::peek_manifest_arch(&install).expect("peeks");
    let params = params_from(&arch.vision);
    let vision = runner.vision_config();
    let special = VisionSpecialIds {
        vision_start: vision.vision_start_token_id as i32,
        image_pad: vision.image_token_id as i32,
    };

    eprintln!(
        "vision_memory_oracle: context={VISION_ORACLE_MAX_CONTEXT}, \
         expert_cache_slots={PROTOCOL_EXPERT_CACHE_SLOTS} (inert on this dense install), \
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
        "vision_memory_oracle",
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
        "vision_memory_oracle: decode tok/s across {} rounds: {min_tok_s:.3} min, \
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
            "vision_memory_oracle: chip {brand:?} not in the baseline table -> tok/s reported \
             but not asserted"
        );
    }

    // THE FLAT-PEAK CLAIM: the final round repeats the FIRST (largest)
    // page, so any growth here is accumulation across pages rather than a
    // one-time cost the first page alone pays.
    let first_peak = rounds[0].peak_mib;
    let last_peak = rounds[rounds.len() - 1].peak_mib;
    let growth = last_peak - first_peak;
    assert!(
        growth <= STEADY_STATE_SLACK_MIB as f64,
        "peak grew {growth:.1} MiB from the first large-page round ({first_peak:.1} MiB) to \
         the repeated large-page round ({last_peak:.1} MiB), past the {STEADY_STATE_SLACK_MIB} \
         MiB slack: VisionScratch may be accumulating rather than being dropped per page"
    );
}

/// The catalog half of this row, checked offline on every `cargo test`.
///
/// NOT `#[ignore]`d and needs no install: it asserts that `BASELINES` above
/// still agrees with the `measured` block `models.json` carries for
/// `qwen38-27b-vision`. See `oracle_common::assert_agrees_with_catalog`.
#[test]
fn the_baselines_agree_with_the_catalogs_measured_rows() {
    oracle_common::assert_agrees_with_catalog(
        "qwen38-27b-vision",
        BASELINES,
        VISION_ORACLE_MAX_CONTEXT,
        PROTOCOL_EXPERT_CACHE_SLOTS as u32,
    );
}
