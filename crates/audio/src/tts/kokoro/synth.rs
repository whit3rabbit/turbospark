//! Text to audio for Kokoro: the English frontend, the voice pack and the
//! model behind one call, so a host does not re-derive the glue the demo
//! example does by hand.
//!
//! Install layout (the catalog's `kokoro-82m-bf16`): `config.json`,
//! `kokoro-v1_0.safetensors` (or `model.safetensors`) and
//! `voices/af_heart.safetensors` (or `af_heart.safetensors` beside the weights).

use std::path::Path;

use turbospark_model_io::safetensors::SafetensorsFile;

use super::{EnglishFrontend, Kokoro, PhonemeVocabulary, SynthesisRequest, Voice, VoicePack};
use crate::{Result, SpeechError};

/// The catalog's voice file names its tensor `af_heart`. A voice exported for
/// the demo example names it `voice`. Both are the same `[510, 1, 256]` pack.
fn open_voice(path: &Path) -> Result<VoicePack> {
    let file = SafetensorsFile::open(path)?;
    let name = ["af_heart", "voice"]
        .into_iter()
        .find(|n| file.descriptor(n).is_some())
        .ok_or_else(|| SpeechError::Tensor {
            name: "af_heart".into(),
            why: "voice tensor missing (expected `af_heart` or `voice`)".into(),
        })?;
    let shape = file
        .descriptor(name)
        .map(|d| d.shape.clone())
        .unwrap_or_default();
    VoicePack::from_values(Voice::AfHeart, &shape, file.load_as_f32(name)?)
}

/// Kokoro's output rate.
pub const SAMPLE_RATE: u32 = 24_000;

/// Where the one supported voice pack lives inside a catalog install. Ad hoc
/// installs often drop `af_heart.safetensors` beside the weights instead, so
/// that spelling is tried second.
pub const VOICE_FILES: [&str; 2] = ["voices/af_heart.safetensors", "af_heart.safetensors"];

pub struct KokoroSynthesizer {
    model: Kokoro,
    vocab: PhonemeVocabulary,
    voice: VoicePack,
    frontend: EnglishFrontend,
}

impl KokoroSynthesizer {
    pub fn open(dir: &Path) -> Result<Self> {
        let config_path = dir.join("config.json");
        let config: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&config_path).map_err(|e| {
                SpeechError::BadConfig {
                    field: config_path.display().to_string(),
                    why: e.to_string(),
                }
            })?)
            .map_err(|e| SpeechError::BadConfig {
                field: "config.json".to_string(),
                why: e.to_string(),
            })?;
        let model = Kokoro::open(dir)?;
        let vocab = model.phoneme_vocabulary(&config)?;
        let voice_path = VOICE_FILES
            .iter()
            .map(|name| dir.join(name))
            .find(|path| path.is_file())
            .ok_or_else(|| SpeechError::BadConfig {
                field: "voice".to_string(),
                why: format!("none of {VOICE_FILES:?} exists in {}", dir.display()),
            })?;
        let voice = open_voice(&voice_path)?;
        Ok(Self {
            model,
            vocab,
            voice,
            frontend: EnglishFrontend::new(),
        })
    }

    /// Synthesizes `request` one frontend segment at a time. `emit` receives
    /// each segment's 24 kHz mono samples and returns `false` to stop early
    /// (the consumer went away); a stop is not an error.
    pub fn synthesize(
        &self,
        request: &SynthesisRequest,
        mut emit: impl FnMut(Vec<f32>) -> bool,
    ) -> Result<()> {
        for segment in self.frontend.prepare(request, &self.vocab)? {
            let style = self.voice.style_for(&segment);
            let audio = self.model.generate(segment.ids(), style, request.speed)?;
            if !emit(audio) {
                break;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn open_reports_a_missing_directory_instead_of_panicking() {
        let dir = std::env::temp_dir().join("turbospark-kokoro-synth-missing");
        assert!(KokoroSynthesizer::open(&dir).is_err());
    }
}
