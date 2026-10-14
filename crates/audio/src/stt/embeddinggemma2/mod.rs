//! EmbeddingGemma 2 word-embedding CTC speech recognition (experiment).
//!
//! Reproduces the portable half of Haku's "Embedding Gemma Transcribe"
//! (`ai.haku.eg2demo`) pipeline: a fine-tuned EmbeddingGemma 2 audio tower
//! truncated at layer 11, a trained 12-layer reader that maps each 40 ms
//! frame to a unit vector in the EmbeddingGemma 2 text-embedding space plus
//! a blank logit, a two-stage vocabulary search over 229457 word rows, and
//! greedy word-level CTC. There is no autoregressive decoder.
//!
//! Module map:
//!
//! - [`frontend`]: the Gemma4 log-mel contract (pinned transformers
//!   reference fixture).
//! - [`windows`]: the 20 s / 18 s window schedule and frame stitching.
//! - [`vocabulary`]: word strings and the FP16 unit-norm embedding table.
//! - [`scorer`]: the 128-d block shortlist (CPU reference of the compiled
//!   scorer) and exact 768-d reranking.
//! - [`ctc`]: the calibrated blank decision, collapse, and timestamps.
//! - [`pipeline`]: orchestration over an [`AcousticBackend`] that runs the
//!   compiled encoder and reader. No backend ships in this crate; the
//!   encoder and reader are Core ML programs and belong in the runtime.
//!
//! Punctuation and casing (`punct_headonly`, `punct_case.json`) are not
//! ported. See the family README for what is verified and what is not.

pub mod ctc;
pub mod frontend;
pub mod pipeline;
pub mod scorer;
pub mod vocabulary;
pub mod windows;

pub use ctc::{Calibration, DecodedWord, TimedWord, READER_P6};
pub use frontend::{LogMel, LogMelFrontend};
pub use pipeline::{AcousticBackend, ReaderOutput, Transcriber, Transcript, ENCODER_DIM};
pub use scorer::{BlockShortlist, BlockShortlister, Candidate, CpuBlockScorer};
pub use vocabulary::{WordList, WordTable, EMBEDDING_DIM};

/// One file of the pinned asset manifest.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AssetFile {
    pub path: &'static str,
    pub size: u64,
    pub sha256: &'static str,
}

/// The pinned reference distribution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Eg2Profile {
    pub name: &'static str,
    pub manifest_url: &'static str,
    pub manifest_version: u32,
    /// Files the portable path reads. The compiled encoder, reader, and
    /// punctuation programs are listed in the README, not here.
    pub portable_files: &'static [AssetFile],
}

pub const EG2DEMO_V5: Eg2Profile = Eg2Profile {
    name: "Embedding Gemma Transcribe eg2demo v5",
    manifest_url: "https://haku-models.tslfiles.org/eg2demo/v5/manifest.json",
    manifest_version: 5,
    portable_files: &[
        AssetFile {
            path: "hann_320.f32",
            size: 1_280,
            sha256: "8ac4fd109da552d6d3ef7a90083306384c9ee2bc32a60dd9f1ecfe931d8caf44",
        },
        AssetFile {
            path: "mel_filters_257x128.f32",
            size: 131_584,
            sha256: "669a99684b146cc243809c72bb0330261a56918ac30bdc0876fc112a8881a6b9",
        },
        AssetFile {
            path: "reader_P6.json",
            size: 294,
            sha256: "7a43e6557e6a202b59761ea0720fad9b807796e24776b2b331b0f03d67a6f700",
        },
        AssetFile {
            path: "reader_words.json",
            size: 2_688_088,
            sha256: "d651e2ec1ebd55de2d31c70f64803894580eb6f5e85239edcd0bbec99c918b53",
        },
        AssetFile {
            path: "word_unit_f16.bin",
            size: 352_445_952,
            sha256: "36edb3cae02101c49ad3f3289ce5c526b1698916eb6b9faf116f4df227a25c9b",
        },
        AssetFile {
            path: "scorer_blk_d128_k16_b4_t500.mlmodelc/weights/weight.bin",
            size: 58_938_560,
            sha256: "823e7cace9dfc1d8c2c6575dc802814f15879c2b124b9da6693e5cf1f6f7034a",
        },
    ],
};

/// Words in the pinned vocabulary.
pub const PINNED_WORDS: usize = 229_457;

#[cfg(test)]
mod asset_tests {
    //! Checks against a local copy of the pinned assets. Point
    //! `TURBOSPARK_EG2_ASSET_DIR` at a directory laid out like the manifest.

    use std::path::PathBuf;

    use super::scorer::{exhaustive_best, rerank, BLOCK_SIZE, SCORER_COLUMNS, SEARCH_DIM};
    use super::*;

    fn asset_dir() -> PathBuf {
        PathBuf::from(
            std::env::var_os("TURBOSPARK_EG2_ASSET_DIR")
                .expect("set TURBOSPARK_EG2_ASSET_DIR to the pinned eg2demo assets"),
        )
    }

    #[test]
    #[ignore = "requires the pinned eg2demo v5 assets in TURBOSPARK_EG2_ASSET_DIR"]
    fn shipped_frontend_tables_match_the_reference_contract() {
        let dir = asset_dir();
        let shipped = LogMelFrontend::load(&dir).unwrap();
        let reference = LogMelFrontend::new().unwrap();
        let window_error = shipped
            .window()
            .iter()
            .zip(reference.window())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let filter_error = shipped
            .filters()
            .weights
            .iter()
            .zip(&reference.filters().weights)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(window_error <= 1e-6, "window max abs error {window_error}");
        assert!(
            filter_error <= 1e-6,
            "filterbank max abs error {filter_error}"
        );
    }

    #[test]
    #[ignore = "requires the pinned eg2demo v5 assets in TURBOSPARK_EG2_ASSET_DIR"]
    fn pinned_vocabulary_scorer_and_calibration_agree() {
        let dir = asset_dir();
        assert_eq!(Calibration::load(&dir).unwrap(), READER_P6);
        let words = WordList::load(&dir).unwrap();
        assert_eq!(words.len(), PINNED_WORDS);
        assert_eq!(words.repeat_id(), Some(PINNED_WORDS as u32 - 1));
        let table = WordTable::load(&dir).unwrap();
        table.check_matches(&words).unwrap();
        let scorer = CpuBlockScorer::load(&dir).unwrap();
        assert_eq!(scorer.column_count(), SCORER_COLUMNS);

        // The projected vocabulary is the word table in identity order times
        // the projection, so block b covers word ids 4b..4b+3.
        let mut worst = 0.0f32;
        for id in (0..PINNED_WORDS as u32)
            .step_by(997)
            .chain([PINNED_WORDS as u32 - 1])
        {
            let row = table.row(id).unwrap();
            let norm: f32 = row.iter().map(|v| v * v).sum::<f32>().sqrt();
            assert!((norm - 1.0).abs() < 2e-3, "word {id} norm {norm}");
            let projected = scorer.project(&row);
            let column = scorer.column(id as usize);
            assert_eq!(column.len(), SEARCH_DIM);
            for (a, b) in projected.iter().zip(column) {
                worst = worst.max((a - b).abs());
            }
        }
        assert!(worst <= 2e-3, "projected column max abs error {worst}");

        // A word's own row must shortlist its block and rerank to itself
        // (or to an identical duplicate string).
        let mut scorer = scorer;
        for id in [0u32, 3, 1_000, 123_457, PINNED_WORDS as u32 - 2] {
            let row = table.row(id).unwrap();
            let shortlist = scorer.shortlist(&row, 1).unwrap();
            assert!(shortlist.row(0).contains(&(id / BLOCK_SIZE as u32)));
            let best = rerank(&row, shortlist.row(0), &table).unwrap();
            assert_eq!(words.word(best.word_id), words.word(id));
            assert_eq!(
                words.word(exhaustive_best(&row, &table).unwrap().word_id),
                words.word(id)
            );
        }
    }
}
