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

use runtime::{GenerationConfig, RateControl, RawDecodeProgress};
use selection::ShapingConfig;
use turbospark_bench::memory::{chip_brand_string, AppMemorySampler};
use turbospark_bench::protocol::PROTOCOL_EXPERT_CACHE_SLOTS;
use turbospark_bench::real_model::open_model_runner_with_context;
use turbospark_vision_io::{decode_image_file, preprocess, PreprocessParams, VisionSpecialIds};

mod oracle_common;

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

/// Preprocessing parameters read off the INSTALL's own `ArchConfig`, never
/// restated (`vision_tower_parity.rs`'s rule; the same body
/// `vision_logit_dump.rs` carries).
fn params_from(arch: &model_io::ArchConfig) -> PreprocessParams {
    let v = &arch.vision;
    PreprocessParams {
        patch_size: v.patch_size as usize,
        temporal_patch_size: v.temporal_patch_size as usize,
        merge_size: v.spatial_merge_size as usize,
        in_channels: v.in_channels as usize,
        min_pixels: 65_536,
        max_pixels: 16_777_216,
        image_mean: [0.5; 3],
        image_std: [0.5; 3],
        rescale_factor: 1.0 / 255.0,
    }
}

/// One oracle page. `marker` is a four-word run unique to THIS page's FIRST
/// line, computed offline by replaying `scripts/make_vision_test_page.py`'s
/// own per-page-seeded RNG sequence rather than guessed.
///
/// **Both prior designs were measured wrong, not assumed right.** A first
/// version keyed on each page's opening line's trailing `NNNN.NN` float and
/// failed on a real transcription that was otherwise correct: round 1
/// (`medium`) wrote line 1 as "...decode through" (cut off before its own
/// number) while completing line 3 in full. A second version keyed on the
/// THIRD line instead -- and failed just as validly on `small`, whose real
/// transcription stopped at 40 tokens with all three lines cut mid-word
/// ("...residual expert stre", "...tokenizer embed", "...gradient footpr").
/// Across every round of both real runs, the one thing that reproduced
/// EXACTLY every time -- even in that worst truncation -- was the OPENING
/// of line 1: the model transcribes forward from the start and may stop
/// before the end, so a marker drawn from as early as possible is the only
/// one immune to where it happens to stop. Four words from a 20-word shared
/// vocabulary is long enough that an exact run recurring by chance across
/// three independently-seeded pages is negligible (checked by eye against
/// all three pages' first three lines below).
struct Page {
    file: &'static str,
    label: &'static str,
    marker: &'static str,
}

/// Largest first: the round that establishes the ceiling has to be the
/// biggest scratch allocation, or a smaller-page-first ordering would make
/// "the peak did not grow" trivially true. Sizes vary ~7x (2,304 vs 320
/// merged tokens at this install's patch_size=16/merge_size=2), which is
/// enough spread that a leaked constant-size scratch would be visible
/// against a correctly dropped one.
const PAGES: &[Page] = &[
    Page {
        file: "large.png",
        label: "large (1536x1536)",
        marker: "attention quantize kernel embedding",
    },
    Page {
        file: "medium.png",
        label: "medium (1024x1280)",
        marker: "expert throughput footprint quantize",
    },
    Page {
        file: "small.png",
        label: "small (512x640)",
        marker: "checkpoint router streaming residual",
    },
];

const QUESTION: &str =
    "Transcribe the first three lines of text in this image, exactly as written, \
     including all numbers and punctuation.";

/// Generous enough to reach the third dense line (each carries 9-14 words
/// plus a trailing number) without relying on the model to stop on its own.
const GENERATE_TOKENS: u32 = 300;

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

struct RoundResult {
    label: &'static str,
    peak_mib: f64,
    decode_tok_s: f64,
}

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
    let params = params_from(&arch);
    let vision = runner.vision_config();
    let special = VisionSpecialIds {
        vision_start: vision.vision_start_token_id as i32,
        image_pad: vision.image_token_id as i32,
    };
    let vocab = runner.vocab_size();

    eprintln!(
        "vision_memory_oracle: context={VISION_ORACLE_MAX_CONTEXT}, \
         expert_cache_slots={PROTOCOL_EXPERT_CACHE_SLOTS} (inert on this dense install), \
         ceiling={CEILING_MIB} MiB, steady_state_slack={STEADY_STATE_SLACK_MIB} MiB"
    );

    let mut sampler = AppMemorySampler::new();
    let mut rounds: Vec<RoundResult> = Vec::new();

    // Largest first, then a final repeat of the largest page -- the real
    // steady-state proof that `VisionScratch` is dropped and reallocated
    // per page rather than accumulating.
    let order: Vec<&Page> = PAGES.iter().chain(std::iter::once(&PAGES[0])).collect();

    for (round, page) in order.iter().enumerate() {
        let path = pages_dir.join(page.file);
        let decoded = decode_image_file(&path).unwrap_or_else(|e| {
            panic!(
                "{}: {e}\n  run the setup commands in this file's \
                 module header first",
                path.display()
            )
        });
        let image =
            preprocess(&decoded, &params).unwrap_or_else(|e| panic!("{}: {e}", path.display()));

        let messages = [tokenizer::Message::with_parts(
            tokenizer::Role::User,
            vec![
                tokenizer::ContentPart::Image,
                tokenizer::ContentPart::Text(QUESTION.to_string()),
            ],
        )];
        let rendered = tokenizer
            .apply_chat_template(&messages)
            .expect("the checkpoint's template renders an image part");
        let encoded = tokenizer.encode(&rendered, false);
        let spliced = turbospark_vision_io::splice_and_walk(
            &encoded,
            &[image.grid],
            special,
            params.merge_size,
        )
        .unwrap_or_else(|e| panic!("{}: cannot splice: {e}", page.label));

        // CONSUME the previous round's map before building this one,
        // matching the CLI's `--image-batch` loop
        // (`crates/cli/src/generate/mod.rs`) -- `reset()` deliberately does
        // NOT do this (AGENTS.md Gotcha 29 / crate Gotcha 13).
        runner.clear_prompt_vision();
        let embedding = runner
            .encode_image(&image, &params)
            .unwrap_or_else(|e| panic!("{}: the tower should run: {e}", page.label));
        runner
            .set_prompt_vision(
                std::slice::from_ref(&embedding),
                &spliced.positions,
                spliced.ids.len(),
            )
            .unwrap_or_else(|e| panic!("{}: the injection map should validate: {e}", page.label));

        let shaping = ShapingConfig::new(0.0, 1, None, 1.0, Some(round as u64))
            .expect("a fixed greedy shaping config is always valid");
        let config = GenerationConfig {
            shaping,
            max_new_tokens: GENERATE_TOKENS,
            stop_strings: Vec::new(),
            extra_stop_tokens: Vec::new(),
            rate: RateControl::default(),
        };

        let mut generated_text = String::new();
        let result = runtime::run_raw_completion(
            &mut runner,
            &tokenizer,
            &spliced.ids,
            &config,
            VISION_ORACLE_MAX_CONTEXT,
            vocab,
            |event| match event {
                RawDecodeProgress::Token { delta, .. } => generated_text.push_str(&delta),
                RawDecodeProgress::Tail(tail) => generated_text.push_str(&tail),
                RawDecodeProgress::Prefill { .. } => {}
            },
        )
        .unwrap_or_else(|e| panic!("{}: generation failed: {e}", page.label));

        let peak_bytes = sampler.sample().expect("footprint sampling worked");
        let peak_mib = peak_bytes as f64 / 1_048_576.0;
        let decode_tok_s = if result.decode_seconds > 0.0 {
            result.new_tokens as f64 / result.decode_seconds
        } else {
            0.0
        };
        eprintln!(
            "vision_memory_oracle: round {round} ({}) -> {:?}, {} new tokens, peak {peak_mib:.1} \
             MiB, {decode_tok_s:.3} tok/s ({:.3}s prefill, {:.3}s decode)",
            page.label,
            result.reason,
            result.new_tokens,
            result.prefill_seconds,
            result.decode_seconds
        );
        eprintln!("vision_memory_oracle: round {round} transcription: {generated_text}");

        // THE CONTENT ASSERTION: proves this round's OWN page was read, not
        // a stale embedding from a previous round or no image at all. This
        // is exactly what a memory-shape-only oracle cannot see
        // (`docs/VISION.md`'s M-V5 bug, ROADMAP's "M-V9 is the third").
        assert!(
            generated_text.contains(page.marker),
            "{}: transcription does not contain this page's own marker \
             {:?} -- the model may be answering from a stale or absent \
             embedding rather than this page.\n  full output: {generated_text}",
            page.label,
            page.marker
        );
        // And the NEGATIVE half: no OTHER page's marker leaked in, which is
        // what a genuinely stale (rather than merely absent) embedding
        // would produce.
        for other in PAGES {
            if other.marker == page.marker {
                continue;
            }
            assert!(
                !generated_text.contains(other.marker),
                "{}: transcription contains {:?}, which belongs to {} -- a stale \
                 embedding leaked across rounds",
                page.label,
                other.marker,
                other.label
            );
        }

        rounds.push(RoundResult {
            label: page.label,
            peak_mib,
            decode_tok_s,
        });
    }

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
