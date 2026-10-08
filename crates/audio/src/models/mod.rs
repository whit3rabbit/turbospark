//! Audio model implementations organized by task family:
//! - [`music`][crate::music]: Music Generation (MiniMax Music 0.5)
//! - [`stt`][crate::stt]: Speech-to-Text (Whisper, Moonshine, Parakeet, MMS, Granite Speech, Qwen3 ASR, Nemotron ASR)
//! - [`tts`][crate::tts]: Text-to-Speech (Kokoro)
//! - [`vad`][crate::vad]: Voice Activity Detection & Diarization (Silero VAD, Sortformer, Nemotron Diarization)
//! - [`sts`][crate::sts]: Speech-to-Speech / Audio Enhancement (DeepFilterNet)
//! - [`codec`][crate::codec]: Audio Codecs, Vocoders & Tokenizers
//! - [`lid`][crate::lid]: Language Identification
//!
//! Aligned with the upstream structure of `mlx_audio/<task>/models/<family>`.

pub use crate::codec;
pub use crate::lid;
pub use crate::music;
pub use crate::sts;
pub use crate::stt;
pub use crate::tts;
pub use crate::vad;

// Re-exports preserving flat module paths for backwards compatibility
pub use crate::sts::deepfilternet;
pub use crate::sts::dialogue_sidon;
pub use crate::sts::mossformer2_se;
pub use crate::stt::moonshine;
pub use crate::stt::whisper;
pub use crate::tts::kokoro;
pub use crate::vad::silero_vad;
pub use crate::vad::sortformer;
