//! Which audio pipeline an install is, from its `config.json`.
//!
//! The runtime decides the same thing in `native_audio::Engine::open` (by
//! `model_type`, with Kokoro recognized by layout because its config has no
//! `model_type`). This copy only names the pipeline for coverage and
//! workflows; the worker opens the model through the runtime's own runners,
//! so a disagreement here can mislabel a run but never run the wrong code.

use serde::Serialize;
use serde_json::Value;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AudioKind {
    Whisper,
    Moonshine,
    Qwen3Asr,
    Music3,
    Kokoro,
    MegaTts3,
    OmniVoice,
    FishDrama3,
}

/// What a pipeline consumes and produces.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AudioTaskKind {
    /// PCM in, transcript out.
    Transcribe,
    /// Caption and lyrics in, waveform out.
    Music,
    /// Text in, waveform out.
    Speak,
}

impl AudioKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Whisper => "whisper",
            Self::Moonshine => "moonshine",
            Self::Qwen3Asr => "qwen3_asr",
            Self::Music3 => "minimax_music3",
            Self::Kokoro => "kokoro",
            Self::MegaTts3 => "megatts3",
            Self::OmniVoice => "omnivoice",
            Self::FishDrama3 => "fish_drama3",
        }
    }

    pub fn task(self) -> AudioTaskKind {
        match self {
            Self::Whisper | Self::Moonshine | Self::Qwen3Asr => AudioTaskKind::Transcribe,
            Self::Music3 => AudioTaskKind::Music,
            Self::Kokoro | Self::MegaTts3 | Self::OmniVoice | Self::FishDrama3 => {
                AudioTaskKind::Speak
            }
        }
    }

    pub const ALL: [AudioKind; 8] = [
        Self::Whisper,
        Self::Moonshine,
        Self::Qwen3Asr,
        Self::Music3,
        Self::Kokoro,
        Self::MegaTts3,
        Self::OmniVoice,
        Self::FishDrama3,
    ];
}

/// Pipelines `tslab-worker` can open and run as an audio arm. The rest are
/// covered by kernel benches only, and coverage says so.
pub const WORKER_KINDS: &[AudioKind] = &[];

pub fn runs_in_worker(k: AudioKind) -> bool {
    WORKER_KINDS.contains(&k)
}

/// The pipeline's name: `model_type`, or `kokoro` for a Kokoro layout.
pub fn model_type(cfg: &Value) -> Option<String> {
    if let Some(t) = cfg["model_type"].as_str() {
        return Some(t.to_string());
    }
    // Kokoro-82M's config has no model_type; its vocoder block names it.
    (cfg.get("istftnet").is_some() || cfg.get("plbert").is_some()).then(|| "kokoro".to_string())
}

pub fn kind(cfg: &Value) -> Option<AudioKind> {
    let t = model_type(cfg)?;
    AudioKind::ALL.into_iter().find(|k| k.as_str() == t)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn names_round_trip_and_kokoro_is_found_by_layout() {
        for k in AudioKind::ALL {
            assert_eq!(kind(&json!({"model_type": k.as_str()})), Some(k));
        }
        assert_eq!(kind(&json!({"istftnet": {}})), Some(AudioKind::Kokoro));
        assert_eq!(kind(&json!({"model_type": "llama"})), None);
        assert_eq!(model_type(&json!({})), None);
        assert_eq!(AudioKind::Music3.task(), AudioTaskKind::Music);
    }
}
