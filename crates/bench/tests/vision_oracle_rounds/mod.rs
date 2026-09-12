#![cfg(target_os = "macos")]
//! The multi-page vision round machinery, shared by the two vision memory
//! oracle targets (`vision_memory_oracle.rs` on the combined install,
//! `vision_sidecar_memory_oracle.rs` through a sidecar-attached trunk).
//!
//! ONE MODEL PER PROCESS, for `oracle_common`'s own reason: the footprint
//! assertion is against a WHOLE-SESSION peak, so two opened installs in one
//! process would each be measured against the other's high-water mark. The
//! two targets are therefore separate binaries that include this module,
//! rather than two `#[test]`s in one file.
//!
//! Sharing the ROUND BODY is what keeps the two oracles one instrument: the
//! page order, the marker assertions and the sampling points live here once,
//! so a combined-install row and a sidecar row can be compared as two runs
//! of the same measurement rather than as two measurements that happen to
//! agree. What stays per-target: the install (and how it is opened), the
//! ceiling, the steady-state slack and the tok/s baselines, because those
//! are properties of the install shape being certified.

use std::path::Path;

use runtime::{GenerationConfig, RateControl, RawDecodeProgress, RealForwardRunner};
use selection::ShapingConfig;
use tokenizer::MfTokenizer;
use turbospark_bench::memory::AppMemorySampler;
use turbospark_vision_io::{decode_image_file, preprocess, PreprocessParams, VisionSpecialIds};

/// Preprocessing parameters read off the install's own vision config, never
/// restated (`vision_tower_parity.rs`'s rule). Takes the VISION CONFIG rather
/// than the whole arch so a sidecar-attached trunk can read it off the runner
/// POST-ATTACH -- a text-only trunk's peeked arch has an inactive vision
/// field until the attach mutates it.
pub fn params_from(v: &model_io::VisionConfig) -> PreprocessParams {
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
/// **Both prior designs were measured wrong, not assumed right** (the full
/// account lives in `vision_memory_oracle.rs`'s header): markers drawn from
/// anything but the OPENING of line 1 failed on genuinely correct
/// transcriptions that stopped early.
pub struct Page {
    pub file: &'static str,
    pub label: &'static str,
    pub marker: &'static str,
}

/// Largest first: the round that establishes the ceiling has to be the
/// biggest scratch allocation, or a smaller-page-first ordering would make
/// "the peak did not grow" trivially true. Sizes vary ~7x (2,304 vs 320
/// merged tokens at this install's patch_size=16/merge_size=2), which is
/// enough spread that a leaked constant-size scratch would be visible
/// against a correctly dropped one.
pub const PAGES: &[Page] = &[
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

pub const QUESTION: &str =
    "Transcribe the first three lines of text in this image, exactly as written, \
     including all numbers and punctuation.";

/// Generous enough to reach the third dense line (each carries 9-14 words
/// plus a trailing number) without relying on the model to stop on its own.
pub const GENERATE_TOKENS: u32 = 300;

pub struct RoundResult {
    pub label: &'static str,
    pub peak_mib: f64,
    pub decode_tok_s: f64,
}

/// Runs the four rounds (large/medium/small/large-again) over ONE open
/// runner, asserting each round's content marker and returning the per-round
/// peaks and decode rates. The ceiling, steady-state and tok/s assertions
/// stay with the caller, which owns the constants they compare against.
#[allow(clippy::too_many_arguments)]
pub fn run_rounds(
    runner: &mut RealForwardRunner,
    tokenizer: &MfTokenizer,
    params: &PreprocessParams,
    special: VisionSpecialIds,
    pages_dir: &Path,
    max_context: u32,
    label: &str,
    sampler: &mut AppMemorySampler,
) -> Vec<RoundResult> {
    let vocab = runner.vocab_size();
    let mut rounds: Vec<RoundResult> = Vec::new();

    // Largest first, then a final repeat of the largest page -- the real
    // steady-state proof that `VisionScratch` is dropped and reallocated
    // per page rather than accumulating.
    let order: Vec<&Page> = PAGES.iter().chain(std::iter::once(&PAGES[0])).collect();

    for (round, page) in order.iter().enumerate() {
        let path = pages_dir.join(page.file);
        let decoded = decode_image_file(&path).unwrap_or_else(|e| {
            panic!(
                "{}: {e}\n  run the setup commands in `vision_memory_oracle.rs`'s \
                 module header first",
                path.display()
            )
        });
        let image =
            preprocess(&decoded, params).unwrap_or_else(|e| panic!("{}: {e}", path.display()));

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
            .encode_image(&image, params)
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
            runner,
            tokenizer,
            &spliced.ids,
            &config,
            max_context,
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
            "{label}: round {round} ({}) -> {:?}, {} new tokens, peak {peak_mib:.1} \
             MiB, {decode_tok_s:.3} tok/s ({:.3}s prefill, {:.3}s decode)",
            page.label,
            result.reason,
            result.new_tokens,
            result.prefill_seconds,
            result.decode_seconds
        );
        eprintln!("{label}: round {round} transcription: {generated_text}");

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
    rounds
}
