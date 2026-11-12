//! Frame decisions, CTC collapse, and word timestamps.
//!
//! Each 40 ms frame either emits its best word or blank. The word logit is
//! `scale * cosine + bias` with the calibration from `reader_P6.json`; the
//! frame emits the word when that logit is strictly greater than the
//! reader's blank logit (the reference `fcmp` / `csinv ... hi` pair, so a
//! NaN comparison is blank). The path then collapses adjacent identical word
//! ids and drops blanks, so `w, blank, w` yields two words.

use std::path::Path;

use serde_json::Value;
use turbospark_core::TokenId;

use crate::nn::bad_config;
use crate::Result;

use super::scorer::Candidate;
use super::vocabulary::WordList;

/// Seconds per reader frame (2000 mel frames of 10 ms, subsampled 4x).
pub const FRAME_SECONDS: f64 = 0.04;
pub const CALIBRATION_FILE: &str = "reader_P6.json";

/// The word-logit calibration of a trained reader.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Calibration {
    pub scale: f32,
    pub bias: f32,
}

/// The pinned `reader_P6.json` values (sha256 `7a43e655...`).
pub const READER_P6: Calibration = Calibration {
    scale: 39.856_663,
    bias: -0.462_620_76,
};

impl Calibration {
    /// Reads `scale` and `bias` from the reader's JSON sidecar.
    pub fn from_json_str(text: &str) -> Result<Self> {
        let root: Value =
            serde_json::from_str(text).map_err(|error| bad_config(CALIBRATION_FILE, error))?;
        let number = |field: &str| -> Result<f32> {
            root.get(field)
                .and_then(Value::as_f64)
                .filter(|value| value.is_finite())
                .map(|value| value as f32)
                .ok_or_else(|| bad_config(field, "must be a finite number"))
        };
        let calibration = Self {
            scale: number("scale")?,
            bias: number("bias")?,
        };
        if calibration.scale <= 0.0 {
            return Err(bad_config("scale", "must be positive"));
        }
        Ok(calibration)
    }

    pub fn load(asset_dir: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(asset_dir.join(CALIBRATION_FILE))
            .map_err(|error| bad_config(CALIBRATION_FILE, error))?;
        Self::from_json_str(&text)
    }

    pub fn word_logit(&self, cosine: f32) -> f32 {
        self.scale * cosine + self.bias
    }

    /// The frame's label: the candidate word, or `None` for blank.
    pub fn decide(&self, candidate: Option<Candidate>, blank_logit: f32) -> Option<u32> {
        let candidate = candidate?;
        (self.word_logit(candidate.cosine) > blank_logit).then_some(candidate.word_id)
    }
}

/// One collapsed word with its frame span.
#[derive(Debug, Clone, PartialEq)]
pub struct DecodedWord {
    pub word_id: TokenId,
    /// First and last global frame (inclusive) the label held.
    pub first_frame: usize,
    pub last_frame: usize,
}

impl DecodedWord {
    pub fn start_seconds(&self) -> f64 {
        self.first_frame as f64 * FRAME_SECONDS
    }

    pub fn end_seconds(&self) -> f64 {
        (self.last_frame + 1) as f64 * FRAME_SECONDS
    }
}

/// Collapses a frame path: adjacent equal ids merge into one word, blanks
/// separate words and are dropped. Ids are compared by id, not by string,
/// so the list's duplicated strings stay distinct labels.
pub fn collapse(path: &[Option<u32>]) -> Vec<DecodedWord> {
    let mut words: Vec<DecodedWord> = Vec::new();
    let mut previous: Option<u32> = None;
    for (frame, &label) in path.iter().enumerate() {
        match label {
            Some(id) if previous == Some(id) => {
                if let Some(last) = words.last_mut() {
                    last.last_frame = frame;
                }
            }
            Some(id) => words.push(DecodedWord {
                // Word ids are validated against the 32-bit list at load.
                word_id: id as TokenId,
                first_frame: frame,
                last_frame: frame,
            }),
            None => {}
        }
        previous = label;
    }
    words
}

/// A collapsed word resolved to text.
#[derive(Debug, Clone, PartialEq)]
pub struct TimedWord {
    pub text: String,
    pub word_id: TokenId,
    pub start_seconds: f64,
    pub end_seconds: f64,
}

/// Resolves collapsed words to strings. The `<rep>` marker repeats the
/// preceding word's text and is dropped when nothing precedes it. Ids the
/// list does not name are skipped.
pub fn resolve(words: &[DecodedWord], list: &WordList) -> Vec<TimedWord> {
    let mut out: Vec<TimedWord> = Vec::with_capacity(words.len());
    for word in words {
        let Ok(id) = u32::try_from(word.word_id) else {
            continue;
        };
        let text = if Some(id) == list.repeat_id() {
            match out.last() {
                Some(previous) => previous.text.clone(),
                None => continue,
            }
        } else {
            match list.word(id) {
                Some(text) => text.to_owned(),
                None => continue,
            }
        };
        out.push(TimedWord {
            text,
            word_id: word.word_id,
            start_seconds: word.start_seconds(),
            end_seconds: word.end_seconds(),
        });
    }
    out
}

/// Space-joined lowercase transcript, before punctuation and casing.
pub fn join(words: &[TimedWord]) -> String {
    words
        .iter()
        .map(|word| word.text.as_str())
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pinned_calibration_parses_from_the_sidecar_shape() {
        let text = r#"{"scale": 39.85666275024414, "bias": -0.4626207649707794,
            "params_M": 86.45, "arm": {"d": 768}, "epoch": null}"#;
        assert_eq!(Calibration::from_json_str(text).unwrap(), READER_P6);
        assert!(Calibration::from_json_str(r#"{"scale": 1.0}"#).is_err());
        assert!(Calibration::from_json_str(r#"{"scale": 0.0, "bias": 0.0}"#).is_err());
    }

    #[test]
    fn frame_emits_only_when_the_word_logit_beats_blank() {
        let calibration = READER_P6;
        let candidate = Some(Candidate {
            word_id: 7,
            cosine: 0.5,
        });
        let logit = calibration.word_logit(0.5);
        assert!((logit - (39.856_663 * 0.5 - 0.462_620_76)).abs() < 1e-5);
        assert_eq!(calibration.decide(candidate, logit - 0.01), Some(7));
        // Equal is blank: the reference keeps the word only on "higher".
        assert_eq!(calibration.decide(candidate, logit), None);
        assert_eq!(calibration.decide(candidate, f32::NAN), None);
        assert_eq!(calibration.decide(None, -100.0), None);
    }

    #[test]
    fn collapse_merges_runs_and_keeps_blank_separated_repeats() {
        let path = [
            None,
            Some(3),
            Some(3),
            None,
            Some(3),
            Some(5),
            Some(5),
            None,
        ];
        let words = collapse(&path);
        let spans: Vec<_> = words
            .iter()
            .map(|word| (word.word_id, word.first_frame, word.last_frame))
            .collect();
        assert_eq!(spans, vec![(3, 1, 2), (3, 4, 4), (5, 5, 6)]);
        assert!((words[0].start_seconds() - 0.04).abs() < 1e-12);
        assert!((words[0].end_seconds() - 0.12).abs() < 1e-12);
        assert!(collapse(&[None, None]).is_empty());
    }

    #[test]
    fn resolve_expands_the_repeat_marker_and_keeps_duplicate_strings() {
        let list = WordList::from_json_str(r#"["the","cat","the","<rep>"]"#).unwrap();
        // "<rep>" first (dropped), then the, <rep>, cat, and the duplicate
        // "the" id adjacent to id 0 (two words, collapse is by id).
        let path = [Some(3), None, Some(0), Some(3), Some(1), Some(0), Some(2)];
        let words = resolve(&collapse(&path), &list);
        assert_eq!(join(&words), "the the cat the the");
        assert_eq!(words[1].word_id, 3);
    }
}
