//! The OmniVoice intake facade: the caller-owned pieces the fixture-era
//! port left outside. The HF `tokenizer.json` text tokenizer, the
//! nonverbal-tag atomic tokenization (`_tokenize_with_nonverbal_tags`),
//! the reference-audio encode through the shared Higgs codec (the
//! released `audio_tokenizer/` Transformers format, semantic path
//! included), and a `generate` wrapper over the reference generation
//! defaults.

use std::path::Path;

use tokenizers::Tokenizer;

use super::{build_style_text, build_wrapped_text, duration, Model};
use crate::codec::higgs_audio::HiggsAudioTokenizer;
use crate::{Result, SpeechError};

/// The thirteen nonverbal tags the reference keeps atomic
/// (`_NONVERBAL_PATTERN`); each appears as the bracketed literal.
const NONVERBAL_TAGS: [&str; 13] = [
    "laughter",
    "sigh",
    "confirmation-en",
    "question-en",
    "question-ah",
    "question-oh",
    "question-ei",
    "question-yi",
    "surprise-ah",
    "surprise-oh",
    "surprise-wa",
    "surprise-yo",
    "dissatisfaction-hnn",
];

/// The reference generate defaults (the `generate` signature at the
/// pinned commit); `steps` is the diffusion step count.
#[derive(Debug, Clone)]
pub struct GenerateParams {
    pub language: String,
    pub instruct: String,
    pub steps: usize,
    pub guidance_scale: f32,
    pub class_temperature: f32,
    pub position_temperature: f32,
    pub layer_penalty_factor: f32,
    pub t_shift: f32,
    /// Speaking-rate factor 0.5-2.0. The reference has no speed knob;
    /// the engine maps it onto the estimated target length
    /// (a documented deviation, MegaTTS3 `dur_alpha` precedent).
    pub speed: f32,
    pub seed: u64,
}

impl Default for GenerateParams {
    fn default() -> Self {
        GenerateParams {
            language: "None".into(),
            instruct: "None".into(),
            steps: 32,
            guidance_scale: 2.0,
            class_temperature: 0.0,
            position_temperature: 5.0,
            layer_penalty_factor: 5.0,
            t_shift: 0.1,
            speed: 1.0,
            seed: 0,
        }
    }
}

/// The loaded end-to-end pipeline: bidirectional backbone + audio heads,
/// the text tokenizer, and the Higgs audio tokenizer (decode and, for
/// checkpoints that carry the semantic path, reference encode).
pub struct OmniVoicePipeline {
    model: Model,
    text_tokenizer: Tokenizer,
    audio_tokenizer: HiggsAudioTokenizer,
}

impl OmniVoicePipeline {
    /// Opens an installed model directory: the checkpoint at the root
    /// (`config.json`, `model.safetensors`, `tokenizer.json`) plus the
    /// `audio_tokenizer/` subfolder.
    pub fn open(dir: &Path) -> Result<Self> {
        let model = Model::open(dir)?;
        let text_tokenizer = Tokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| {
            SpeechError::BadConfig {
                field: "tokenizer.json".into(),
                why: e.to_string(),
            }
        })?;
        let audio_tokenizer = HiggsAudioTokenizer::open(&dir.join("audio_tokenizer"))?;
        Ok(OmniVoicePipeline {
            model,
            text_tokenizer,
            audio_tokenizer,
        })
    }

    pub fn sample_rate(&self) -> u32 {
        self.model.sample_rate()
    }

    /// Encodes a voice-clone reference: mono PCM at the checkpoint's
    /// sample rate -> code rows `[T, codebooks]`. Requires the installed
    /// `audio_tokenizer/` to carry the semantic encode path.
    pub fn encode_reference(&self, samples: &[f32]) -> Result<Vec<Vec<i32>>> {
        self.audio_tokenizer.encode_tokens(samples)
    }

    /// The style and wrapped-text token ids. The style string encodes
    /// with the tokenizer's own special tokens (the reference passes the
    /// wrapped style through the tokenizer unmodified); the wrapped text
    /// encodes segment-wise so the bracketed nonverbal tags stay atomic.
    pub fn encode_prompt(
        &self,
        text: &str,
        ref_text: Option<&str>,
        language: &str,
        instruct: &str,
        denoise: bool,
    ) -> Result<(Vec<i32>, Vec<i32>)> {
        let style_ids =
            self.encode_segment(&build_style_text(language, instruct, denoise), true)?;
        let text_ids = self.tokenize_with_nonverbal_tags(&build_wrapped_text(text, ref_text))?;
        Ok((style_ids, text_ids))
    }

    /// The reference duration fallback: the rule estimator calibrated on
    /// `"Nice to meet you."` at 25 seconds, floored at 10 frames.
    pub fn estimate_target_len(&self, text: &str, duration_s: Option<f64>, speed: f32) -> usize {
        let target_len = match duration_s {
            Some(duration) => self.model.target_len_for(duration as f32),
            None => {
                let raw =
                    duration::estimate_duration(text, "Nice to meet you.", 25.0, Some(50.0), 3.0);
                (raw * 1.15).floor().max(10.0) as usize
            }
        };
        if speed != 1.0 && speed.is_finite() && speed > 0.0 {
            ((target_len as f64 / f64::from(speed)).ceil() as usize).max(10)
        } else {
            target_len
        }
    }

    /// The single-utterance generate: assemble the prompt, run the
    /// iterative unmasking with the crate's seeded threefry stream, and
    /// decode through the Higgs codec. Returns the token grid `[T, C]`
    /// and the waveform.
    #[allow(clippy::too_many_arguments)]
    /// `on_step` receives `(completed, total)` diffusion steps and stops
    /// the run on false (the engine's progress and cancellation hooks).
    pub fn generate(
        &mut self,
        text: &str,
        ref_text: Option<&str>,
        ref_tokens: Option<&[Vec<i32>]>,
        duration_s: Option<f64>,
        params: &GenerateParams,
        on_step: Option<&mut dyn FnMut(usize, usize) -> bool>,
    ) -> Result<(Vec<i32>, Vec<f32>)> {
        let (style_ids, text_ids) = self.encode_prompt(
            text,
            ref_text,
            &params.language,
            &params.instruct,
            ref_tokens.is_some(),
        )?;
        let target_len = self.estimate_target_len(text, duration_s, params.speed);
        self.model.generate(
            &style_ids,
            &text_ids,
            target_len,
            ref_tokens,
            params.steps,
            params.guidance_scale,
            params.class_temperature,
            params.position_temperature,
            params.layer_penalty_factor,
            params.t_shift,
            None,
            params.seed,
            &self.audio_tokenizer,
            on_step,
        )
    }

    fn encode_segment(&self, text: &str, add_special_tokens: bool) -> Result<Vec<i32>> {
        let encoding = self
            .text_tokenizer
            .encode(text, add_special_tokens)
            .map_err(|e| SpeechError::Input {
                why: format!("tokenizing {text:?}: {e}"),
            })?;
        encoding
            .get_ids()
            .iter()
            .map(|&id| {
                i32::try_from(id).map_err(|_| SpeechError::Input {
                    why: format!("token id {id} does not fit i32"),
                })
            })
            .collect()
    }

    /// `_tokenize_with_nonverbal_tags`: the thirteen bracketed tags stay
    /// atomic; the segments between them tokenize normally. An input
    /// that produces no parts falls back to one whole-text encode.
    fn tokenize_with_nonverbal_tags(&self, text: &str) -> Result<Vec<i32>> {
        let mut parts: Vec<i32> = Vec::new();
        let mut last_end = 0usize;
        loop {
            let mut best: Option<(usize, &'static str)> = None;
            for tag in NONVERBAL_TAGS {
                let needle = format!("[{tag}]");
                if let Some(offset) = text[last_end..].find(&needle) {
                    let start = last_end + offset;
                    if best.is_none_or(|(position, _)| start < position) {
                        best = Some((start, tag));
                    }
                }
            }
            let Some((start, tag)) = best else {
                break;
            };
            if start > last_end {
                let segment = &text[last_end..start];
                parts.extend(self.encode_segment(segment, false)?);
            }
            parts.extend(self.encode_segment(&format!("[{tag}]"), false)?);
            last_end = start + tag.len() + 2;
        }
        if last_end < text.len() {
            let segment = &text[last_end..];
            parts.extend(self.encode_segment(segment, false)?);
        }
        if parts.is_empty() {
            parts.extend(self.encode_segment(text, false)?);
        }
        Ok(parts)
    }
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};

    use super::{GenerateParams, OmniVoicePipeline};

    fn fixture(segments: &[&str]) -> PathBuf {
        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.push("testdata/tts_omnivoice");
        for segment in segments {
            path.push(segment);
        }
        path
    }

    fn small_tokenizer() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata/tts_omnivoice/tiny_tokenizer.json")
    }

    /// Assembles an installed-directory layout from the tiny fixture:
    /// checkpoint files at the root, the tiny codec under
    /// `audio_tokenizer/`, and a small real BPE `tokenizer.json`.
    fn open_pipeline(tag: &str) -> OmniVoicePipeline {
        let dir =
            std::env::temp_dir().join(format!("omnivoice-pipeline-{tag}-{}", std::process::id()));
        std::fs::create_dir_all(dir.join("audio_tokenizer")).unwrap();
        std::fs::copy(
            fixture(&["generate", "config.json"]),
            dir.join("config.json"),
        )
        .unwrap();
        std::fs::copy(
            fixture(&["generate", "model.safetensors"]),
            dir.join("model.safetensors"),
        )
        .unwrap();
        std::fs::copy(small_tokenizer(), dir.join("tokenizer.json")).unwrap();
        std::fs::copy(
            fixture(&["codec", "config.json"]),
            dir.join("audio_tokenizer/config.json"),
        )
        .unwrap();
        std::fs::copy(
            fixture(&["codec", "model.safetensors"]),
            dir.join("audio_tokenizer/model.safetensors"),
        )
        .unwrap();
        OmniVoicePipeline::open(&dir).expect("pipeline opens")
    }

    #[test]
    fn nonverbal_tags_stay_atomic() {
        let pipeline = open_pipeline("tags");
        // The sentence stream must be exactly the segment encodes: the
        // head, the atomic tag, and the tail - no cross-segment merges.
        let (_, text) = pipeline
            .encode_prompt("hello [laughter] world", None, "en", "None", false)
            .expect("prompt encodes");
        // The wrapped text splits around the tag; the brackets ride
        // with the head and tail segments.
        let head = pipeline
            .encode_segment("<|text_start|>hello ", false)
            .expect("head");
        let tag = pipeline.encode_segment("[laughter]", false).expect("tag");
        let tail = pipeline
            .encode_segment(" world<|text_end|>", false)
            .expect("tail");
        let head_len = head.len();
        let tag_len = tag.len();
        let mut expected = head;
        expected.extend_from_slice(&tag);
        expected.extend_from_slice(&tail);
        assert_eq!(text, expected);
        // The atomic tag ids must equal the standalone tag encode.
        assert_eq!(&text[head_len..head_len + tag_len], &tag[..]);
    }

    #[test]
    fn target_len_follows_the_reference_rules() {
        let pipeline = open_pipeline("duration");
        // "hi" weighs 2.0; the calibrated fallback puts it on the
        // short-duration power curve (50 * (est/50)^(1/3)), then the
        // 1.15 headroom and the frame floor. The recorded reference
        // value for "hi" is 23 frames.
        assert_eq!(pipeline.estimate_target_len("hi", None, 1.0), 23);
        // An explicit duration maps through the codec rate (2 s at 25
        // frames per second).
        assert_eq!(pipeline.estimate_target_len("unused", Some(2.0), 1.0), 50);
        // Speed divides the estimate (documented deviation).
        assert_eq!(pipeline.estimate_target_len("unused", Some(2.0), 2.0), 25);
        // Non-finite or zero speed leaves the estimate alone.
        assert_eq!(pipeline.estimate_target_len("unused", Some(2.0), 0.0), 50);
    }

    #[test]
    fn generate_runs_end_to_end_on_the_tiny_fixture() {
        let mut pipeline = open_pipeline("generate");
        let params = GenerateParams {
            steps: 2,
            ..GenerateParams::default()
        };
        let (tokens, audio) = pipeline
            .generate("Hello world.", None, None, Some(0.5), &params, None)
            .expect("generate");
        assert_eq!(tokens.len() % pipeline.model.config.num_audio_codebook, 0);
        assert!(!tokens.is_empty());
        assert_eq!(
            audio.len(),
            (tokens.len() / pipeline.model.config.num_audio_codebook) * 960
        );
        assert!(audio.iter().all(|value| value.is_finite()));

        // A deterministic seed reproduces the same tokens.
        let (again, _) = pipeline
            .generate("Hello world.", None, None, Some(0.5), &params, None)
            .expect("regenerate");
        assert_eq!(tokens, again);
    }

    #[test]
    fn encode_reference_refuses_a_codec_without_the_semantic_path() {
        let pipeline = open_pipeline("reference");
        let error = pipeline.encode_reference(&[0.1f32; 960]).err().unwrap();
        assert!(error.to_string().contains("semantic encode path"));
    }
}
