//! End-to-end orchestration over a pluggable acoustic backend.
//!
//! The encoder and reader are compiled Core ML programs in the reference.
//! This crate stays portable, so they sit behind [`AcousticBackend`]; a
//! macOS runtime supplies the Core ML (or a native Metal) implementation.
//! Everything else here, the frontend, window schedule, shortlist rerank,
//! blank decision, collapse, and stitching, is portable Rust.

use crate::{Result, SpeechError};

use super::ctc::{collapse, join, resolve, Calibration, TimedWord};
use super::frontend::{LogMelFrontend, FRAME_LENGTH, HOP_LENGTH, NUM_MELS};
use super::scorer::{rerank, BlockShortlister};
use super::vocabulary::{WordList, WordTable, EMBEDDING_DIM};
use super::windows::{
    plan_windows, valid_encoder_frames, ENCODER_FRAMES, MEL_FRAMES, WINDOW_SAMPLES,
};

/// Width of the layer-11 audio-tower features the reader consumes.
pub const ENCODER_DIM: usize = 1_024;

/// Silence appended after the input (0.5 s). The reader's word spikes land
/// up to about 0.3 s after a word ends, so without trailing audio the last
/// word's spike can fall past the final valid frame and be dropped. A port
/// choice measured on synthetic clips (see the family README); the
/// reference runtime's end handling is unknown.
pub const TAIL_PADDING_SAMPLES: usize = 8_000;

/// Reader outputs for one window.
#[derive(Debug, Clone, PartialEq)]
pub struct ReaderOutput {
    /// Row-major `[ENCODER_FRAMES][EMBEDDING_DIM]`, L2-normalized rows.
    pub vectors: Vec<f32>,
    /// `[ENCODER_FRAMES]` blank logits.
    pub blank: Vec<f32>,
}

/// The fixed-shape encoder and reader (`eg2_audio_l11` and `reader_P6`).
pub trait AcousticBackend {
    /// `features` is `[MEL_FRAMES][NUM_MELS]` log-mel, `mask` is
    /// `[MEL_FRAMES]` 0/1. Returns `[ENCODER_FRAMES][ENCODER_DIM]`.
    fn encode(&mut self, features: &[f32], mask: &[f32]) -> Result<Vec<f32>>;

    /// `hidden` is the encoder output, `mask` the `[ENCODER_FRAMES]` 0/1
    /// encoder mask.
    fn read(&mut self, hidden: &[f32], mask: &[f32]) -> Result<ReaderOutput>;
}

/// A transcript before punctuation and casing.
#[derive(Debug, Clone, PartialEq)]
pub struct Transcript {
    pub text: String,
    pub words: Vec<TimedWord>,
    /// Stitched per-frame labels (`None` = blank), one per 40 ms.
    pub frame_path: Vec<Option<u32>>,
    pub windows: usize,
}

/// The EmbeddingGemma 2 word-CTC transcriber.
pub struct Transcriber<B, S> {
    pub frontend: LogMelFrontend,
    pub backend: B,
    pub shortlister: S,
    pub words: WordList,
    pub table: WordTable,
    pub calibration: Calibration,
    /// Zero samples appended to non-empty input before windowing,
    /// [`TAIL_PADDING_SAMPLES`] by default.
    pub tail_padding: usize,
}

impl<B: AcousticBackend, S: BlockShortlister> Transcriber<B, S> {
    pub fn new(
        frontend: LogMelFrontend,
        backend: B,
        shortlister: S,
        words: WordList,
        table: WordTable,
        calibration: Calibration,
    ) -> Result<Self> {
        table.check_matches(&words)?;
        if table.dim() != EMBEDDING_DIM {
            return Err(SpeechError::BadConfig {
                field: "word table".into(),
                why: format!("rows are {} wide, expected {EMBEDDING_DIM}", table.dim()),
            });
        }
        Ok(Self {
            frontend,
            backend,
            shortlister,
            words,
            table,
            calibration,
            tail_padding: TAIL_PADDING_SAMPLES,
        })
    }

    /// Transcribes 16 kHz mono samples. The stitched path covers the input
    /// plus [`Self::tail_padding`], so a final word's span can end after
    /// the audio does.
    pub fn transcribe(&mut self, samples: &[f32]) -> Result<Transcript> {
        let padded;
        let samples = if samples.is_empty() || self.tail_padding == 0 {
            samples
        } else {
            padded = [samples, &vec![0.0; self.tail_padding]].concat();
            &padded[..]
        };
        let plans = plan_windows(samples.len(), |len| {
            // Matches the frontend mask: mel frame i is real while
            // i * HOP + FRAME < len + FRAME / 2.
            let mel = len.saturating_sub(HOP_LENGTH).div_ceil(HOP_LENGTH);
            valid_encoder_frames(mel)
        });
        let mut frame_path: Vec<Option<u32>> = Vec::new();
        for plan in &plans {
            let window = &samples[plan.sample_start..plan.sample_start + plan.sample_len];
            let features = self
                .frontend
                .features(window, WINDOW_SAMPLES)?
                .fit_to(MEL_FRAMES);
            let valid = valid_encoder_frames(features.valid_frames());
            if plan.keep.end > valid {
                return Err(SpeechError::Input {
                    why: format!(
                        "window {} keeps frames to {} but only {valid} carry audio",
                        plan.index, plan.keep.end
                    ),
                });
            }
            let hidden = self
                .backend
                .encode(&features.values, &features.mask_f32())?;
            check_len("encoder output", hidden.len(), ENCODER_FRAMES * ENCODER_DIM)?;
            let encoder_mask: Vec<f32> = (0..ENCODER_FRAMES)
                .map(|frame| if frame < valid { 1.0 } else { 0.0 })
                .collect();
            let reader = self.backend.read(&hidden, &encoder_mask)?;
            check_len(
                "reader vec",
                reader.vectors.len(),
                ENCODER_FRAMES * EMBEDDING_DIM,
            )?;
            check_len("reader blank", reader.blank.len(), ENCODER_FRAMES)?;
            let shortlist = self
                .shortlister
                .shortlist(&reader.vectors, ENCODER_FRAMES)?;
            if shortlist.frames != ENCODER_FRAMES {
                return Err(SpeechError::Input {
                    why: format!("shortlist covers {} frames", shortlist.frames),
                });
            }
            debug_assert_eq!(frame_path.len(), plan.global_frames().start);
            for frame in plan.keep.clone() {
                let vector = &reader.vectors[frame * EMBEDDING_DIM..(frame + 1) * EMBEDDING_DIM];
                let candidate = rerank(vector, shortlist.row(frame), &self.table);
                frame_path.push(self.calibration.decide(candidate, reader.blank[frame]));
            }
        }
        let words = resolve(&collapse(&frame_path), &self.words);
        Ok(Transcript {
            text: join(&words),
            words,
            frame_path,
            windows: plans.len(),
        })
    }
}

fn check_len(what: &str, actual: usize, expected: usize) -> Result<()> {
    if actual != expected {
        return Err(SpeechError::Input {
            why: format!("{what} holds {actual} values, expected {expected}"),
        });
    }
    Ok(())
}

// Keep the feature geometry the backend contract documents in sync.
const _: () =
    assert!(NUM_MELS == 128 && MEL_FRAMES == 4 * ENCODER_FRAMES && FRAME_LENGTH == 2 * HOP_LENGTH);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stt::embeddinggemma2::ctc::READER_P6;
    use crate::stt::embeddinggemma2::scorer::{BlockShortlist, BLOCK_SIZE};

    /// A backend that ignores the audio and plays a scripted global label
    /// sequence, so the test pins window stitching and decoding: frame `g`
    /// points at word `script(g)` or is blank.
    struct Scripted {
        script: fn(usize) -> Option<u32>,
        window: usize,
        encoded_masks: Vec<usize>,
    }

    fn unit(word: u32) -> Vec<f32> {
        let mut row = vec![0.0f32; EMBEDDING_DIM];
        row[word as usize] = 1.0;
        row
    }

    impl AcousticBackend for Scripted {
        fn encode(&mut self, features: &[f32], mask: &[f32]) -> Result<Vec<f32>> {
            assert_eq!(features.len(), MEL_FRAMES * NUM_MELS);
            assert_eq!(mask.len(), MEL_FRAMES);
            self.encoded_masks
                .push(mask.iter().filter(|&&value| value == 1.0).count());
            Ok(vec![0.0; ENCODER_FRAMES * ENCODER_DIM])
        }

        fn read(&mut self, _hidden: &[f32], _mask: &[f32]) -> Result<ReaderOutput> {
            let offset = self.window * 450;
            self.window += 1;
            let mut vectors = Vec::with_capacity(ENCODER_FRAMES * EMBEDDING_DIM);
            let mut blank = Vec::with_capacity(ENCODER_FRAMES);
            for local in 0..ENCODER_FRAMES {
                match (self.script)(offset + local) {
                    Some(word) => {
                        vectors.extend(unit(word));
                        blank.push(0.0);
                    }
                    None => {
                        vectors.extend(unit(0));
                        blank.push(100.0);
                    }
                }
            }
            Ok(ReaderOutput { vectors, blank })
        }
    }

    /// Shortlists the block holding the frame's hot coordinate first.
    struct Oracle;

    impl BlockShortlister for Oracle {
        fn shortlist(&mut self, vectors: &[f32], frames: usize) -> Result<BlockShortlist> {
            let blocks = vectors
                .chunks_exact(EMBEDDING_DIM)
                .flat_map(|row| {
                    let hot = row.iter().position(|&v| v == 1.0).unwrap();
                    [(hot / BLOCK_SIZE) as u32, 1]
                })
                .collect();
            BlockShortlist::new(frames, 2, blocks)
        }
    }

    fn transcriber(script: fn(usize) -> Option<u32>) -> Transcriber<Scripted, Oracle> {
        let words =
            WordList::from_json_str(r#"["zero","alpha","bravo","charlie","delta","echo","<rep>"]"#)
                .unwrap();
        let mut rows = Vec::new();
        for id in 0..7 {
            rows.extend(unit(id));
        }
        let table = WordTable::from_f32_rows(&rows, EMBEDDING_DIM).unwrap();
        Transcriber::new(
            LogMelFrontend::new().unwrap(),
            Scripted {
                script,
                window: 0,
                encoded_masks: Vec::new(),
            },
            Oracle,
            words,
            table,
            READER_P6,
        )
        .unwrap()
    }

    #[test]
    fn words_spanning_a_window_boundary_are_emitted_once() {
        // "alpha" runs across the window 0 / window 1 seam at global frame
        // 475; "bravo" sits in window 0's discarded tail and window 1's kept
        // head (frames 480..490 are kept from window 1 only).
        fn script(frame: usize) -> Option<u32> {
            match frame {
                10..=20 => Some(1),
                470..=478 => Some(1),
                480..=490 => Some(2),
                700..=705 => Some(3),
                706..=710 => Some(6),
                _ => None,
            }
        }
        let mut model = transcriber(script);
        model.tail_padding = 0;
        let audio = vec![0.0f32; 30 * 16_000];
        let transcript = model.transcribe(&audio).unwrap();
        assert_eq!(transcript.windows, 2);
        assert_eq!(transcript.text, "alpha alpha bravo charlie charlie");
        let alpha = &transcript.words[1];
        assert!((alpha.start_seconds - 470.0 * 0.04).abs() < 1e-9);
        assert!((alpha.end_seconds - 479.0 * 0.04).abs() < 1e-9);
        // 30 s of audio stitches to 750 frames.
        assert_eq!(transcript.frame_path.len(), 750);
        // The second window holds 12 s of real audio: 1199 valid mel frames.
        assert_eq!(model.backend.encoded_masks, vec![1999, 1199]);
    }

    #[test]
    fn tail_padding_keeps_a_spike_just_past_the_audio() {
        // 1 s of audio is 25 encoder frames; the word spikes at frame 30
        // (1.2 s), inside the default 0.5 s tail (38 frames in all).
        fn script(frame: usize) -> Option<u32> {
            (frame == 30).then_some(4)
        }
        let audio = vec![0.0f32; 16_000];
        let mut model = transcriber(script);
        let transcript = model.transcribe(&audio).unwrap();
        assert_eq!(transcript.text, "delta");
        assert_eq!(transcript.frame_path.len(), 38);
        model.tail_padding = 0;
        model.backend.window = 0;
        let unpadded = model.transcribe(&audio).unwrap();
        assert_eq!(unpadded.text, "");
        assert_eq!(unpadded.frame_path.len(), 25);
    }

    #[test]
    fn silence_only_input_yields_no_words() {
        let mut model = transcriber(|_| None);
        let transcript = model.transcribe(&vec![0.0f32; 8_000]).unwrap();
        assert!(transcript.words.is_empty());
        assert_eq!(transcript.text, "");
        assert!(model.transcribe(&[]).unwrap().frame_path.is_empty());
    }
}
