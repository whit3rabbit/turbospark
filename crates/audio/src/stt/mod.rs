//! Speech-to-Text (STT) models.
//!
//! Aligned with `mlx_audio/stt/models/`; individual model ports retain their
//! own model-family identity and pinned checkpoint profiles.

pub mod granite_speech5_ctc;
pub mod mms;
pub mod moonshine;
pub mod nemotron_asr;
pub mod parakeet;
pub mod qwen3_asr;
pub mod qwen3_forced_aligner;
pub mod whisper;

pub use granite_speech5_ctc::{GraniteSpeech5Ctc, GraniteSpeech5Profile, GRANITE_SPEECH5_TURBOCTC};
pub use mms::{Mms, MmsConfig, MmsProfile, MMS_1B_FL102_ENGLISH};
pub use moonshine::{Moonshine, MoonshineConfig};
pub use nemotron_asr::{NemotronAsr, NemotronAsrConfig, NemotronAsrProfile, NEMOTRON_3_5_ASR};
pub use parakeet::{
    ParakeetConfig, ParakeetProfile, ParakeetTdt, PARAKEET_REDUX, PARAKEET_TDT_V2, PARAKEET_TDT_V3,
};
pub use qwen3_asr::{Qwen3Asr, Qwen3AsrProfile, QWEN3_ASR_06B_8BIT};
pub use qwen3_forced_aligner::{
    ForcedAlignItem, ForcedAlignResult, Qwen3ForcedAligner, Qwen3ForcedAlignerConfig,
    Qwen3ForcedAlignerProfile, QWEN3_FORCED_ALIGNER_06B_8BIT,
};
pub use whisper::{WhisperConfig, WhisperModel, WhisperSpecialTokens, WhisperWeights};
