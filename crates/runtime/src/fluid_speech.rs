//! Fluid Speech Pro on the portable CPU path: the `audio::stt::parakeet`
//! port of the FluidVoice MLX export, run in the same 15 s windows with 2 s
//! overlap as the Core ML Parakeet runner (`coreml::parakeet`), so long
//! recordings never build one full-length attention matrix.
//!
//! Only this path has the clean transcript style: Clean folds the export's
//! adapter into the weights at load, so a style change reopens the model.
//! One style is resident at a time; the old one is dropped before the new
//! one loads, so a switch never holds both (about 2.5 GB each in f32).

use std::path::{Path, PathBuf};

use audio::stt::nemo::{sentences_to_result, tokens_to_sentences, AlignedResult, AlignedToken};
use audio::stt::parakeet::{ParakeetTdt, TranscriptStyle};

use crate::coreml::parakeet::{merge_windows, plan_windows};

pub const FAMILY: &str = "fluid_speech_pro";
const SAMPLE_RATE: u32 = 16_000;
/// Same window as the Core ML runner, whose merge was gated on real clips.
const WINDOW_SAMPLES: usize = 15 * SAMPLE_RATE as usize;

/// True when `dir` holds a Fluid Speech Pro MLX export (`fluid_speech` in
/// its `config.json`). Core ML exports are detected before this is asked.
pub fn is_fluid_dir(dir: &Path) -> bool {
    std::fs::read(dir.join("config.json"))
        .ok()
        .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok())
        .is_some_and(|c| c.get("fluid_speech").is_some())
}

/// Parses the request's `transcript_style`. `None` is verbatim, the model's
/// default and the only style the Core ML export has.
pub fn parse_style(style: Option<&str>) -> Result<TranscriptStyle, String> {
    match style {
        None | Some("verbatim") => Ok(TranscriptStyle::Verbatim),
        Some("clean") => Ok(TranscriptStyle::Clean),
        Some(other) => Err(format!(
            "transcript_style {other:?} is not one of verbatim, clean"
        )),
    }
}

pub struct FluidSpeechRunner {
    dir: PathBuf,
    model: Option<ParakeetTdt>,
}

impl FluidSpeechRunner {
    pub fn open(dir: &Path) -> Result<Self, String> {
        let model = ParakeetTdt::open(dir).map_err(|e| e.to_string())?;
        Ok(Self {
            dir: dir.to_path_buf(),
            model: Some(model),
        })
    }

    /// The model in `style`, reopened when the resident one differs. A
    /// failed reopen leaves no model, and the next request opens again.
    fn model(&mut self, style: TranscriptStyle) -> Result<&ParakeetTdt, String> {
        if self.model.as_ref().is_some_and(|m| m.style() != style) {
            self.model = None;
        }
        if self.model.is_none() {
            self.model =
                Some(ParakeetTdt::open_with_style(&self.dir, style).map_err(|e| e.to_string())?);
        }
        Ok(self.model.as_ref().expect("opened above"))
    }

    /// Transcribes mono 16 kHz PCM. `progress(window, total)` runs before
    /// each window and stops the run by returning `false`, as `cancelled`
    /// does.
    pub fn decode(
        &mut self,
        samples: &[f32],
        style: TranscriptStyle,
        cancelled: &dyn Fn() -> bool,
        mut progress: impl FnMut(usize, usize) -> bool,
    ) -> Result<AlignedResult, String> {
        if samples.is_empty() || samples.iter().any(|s| !s.is_finite()) {
            return Err("Fluid Speech Pro input must be nonempty finite mono PCM".into());
        }
        let model = self.model(style)?;
        let plan = plan_windows(samples.len(), WINDOW_SAMPLES);
        let overlap_seconds = plan.overlap as f32 / SAMPLE_RATE as f32;
        let mut merged: Vec<AlignedToken> = Vec::new();
        for (i, &start) in plan.starts.iter().enumerate() {
            if cancelled() || !progress(i, plan.starts.len()) {
                return Err("transcription cancelled".into());
            }
            let end = (start + WINDOW_SAMPLES).min(samples.len());
            let offset = start as f32 / SAMPLE_RATE as f32;
            let tokens: Vec<AlignedToken> = model
                .decode(&samples[start..end], SAMPLE_RATE)
                .map_err(|e| e.to_string())?
                .sentences
                .into_iter()
                .flat_map(|s| s.tokens)
                .map(|t| AlignedToken::new(t.id, t.text, t.start + offset, t.duration))
                .collect();
            merged = if i == 0 {
                tokens
            } else {
                merge_windows(merged, tokens, overlap_seconds)
            };
        }
        Ok(sentences_to_result(tokens_to_sentences(merged)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn styles_parse_and_unknown_is_refused() {
        assert_eq!(parse_style(None).unwrap(), TranscriptStyle::Verbatim);
        assert_eq!(
            parse_style(Some("verbatim")).unwrap(),
            TranscriptStyle::Verbatim
        );
        assert_eq!(parse_style(Some("clean")).unwrap(), TranscriptStyle::Clean);
        assert!(parse_style(Some("Clean")).is_err());
    }

    #[test]
    fn only_the_fluid_family_takes_a_transcript_style() {
        use crate::native_audio::{validate_request, AudioRequest};
        let pcm = vec![0.0f32; 16_000];
        let with = |style: &str| AudioRequest {
            transcript_style: Some(style.into()),
            ..AudioRequest::default()
        };
        assert!(validate_request(FAMILY, &with("clean"), &pcm).is_ok());
        assert!(validate_request(FAMILY, &AudioRequest::default(), &pcm).is_ok());
        assert!(validate_request(FAMILY, &with("tidy"), &pcm).is_err());
        // The Core ML export has no clean style; a style there would be ignored.
        let err = validate_request("parakeet_tdt", &with("clean"), &pcm).unwrap_err();
        assert!(err.contains("transcript_style"), "{err}");
    }

    #[test]
    fn fluid_dirs_are_recognized_by_their_config_block() {
        let dir = std::env::temp_dir().join(format!("ts-fluid-dir-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("config.json"), r#"{"encoder":{}}"#).unwrap();
        assert!(!is_fluid_dir(&dir));
        std::fs::write(
            dir.join("config.json"),
            r#"{"fluid_speech":{"insert":"pre"}}"#,
        )
        .unwrap();
        assert!(is_fluid_dir(&dir));
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
