//! Speech-to-Text (STT) models.
//!
//! Aligned with `mlx_audio/stt/models/`; individual model ports retain their
//! own model-family identity and pinned checkpoint profiles.

pub mod canary;
pub mod cohere_asr;
pub mod fireredasr2;
pub mod fun_asr_nano;
pub mod glmasr;
pub mod granite_speech;
pub mod granite_speech5_ctc;
pub mod granite_speech_nar;
pub mod higgs_audio_3;
pub mod lasr_ctc;
pub mod mega_asr;
pub mod mms;
pub mod moonshine;
pub mod moss_music;
pub mod moss_transcribe_diarize;
pub mod nemo;
pub mod nemotron_asr;
pub mod parakeet;
pub mod phonon;
pub mod qwen2_audio;
pub mod qwen3_asr;
pub mod qwen3_forced_aligner;
pub mod sensevoice;
pub mod vibevoice_asr;
pub mod voxtral;
pub mod voxtral_realtime;
pub mod wav2vec;
pub mod whisper;

pub use canary::{Canary, CanaryProfile, CANARY_1B_V2_Q8};
pub use fireredasr2::{
    FireRedAsr2, FireRedAsr2Config, FireRedAsr2Options, FireRedAsr2Profile, FIREREDASR2_AED,
};
pub use glmasr::{GlmAsr, GlmAsrProfile, GLM_ASR_NANO_2512};
pub use granite_speech::{GraniteSpeech, GraniteSpeechProfile, GRANITE_4_0_1B_SPEECH};
pub use granite_speech5_ctc::{GraniteSpeech5Ctc, GraniteSpeech5Profile, GRANITE_SPEECH5_TURBOCTC};
pub use granite_speech_nar::{
    GraniteSpeechNar, GraniteSpeechNarProfile, GRANITE_SPEECH_4_1_2B_NAR,
};
pub use higgs_audio_3::{HiggsAudioV3Profile, HiggsAudioV3Stt, HIGGS_AUDIO_V3_STT};
pub use lasr_ctc::{LasrCtc, MEDASR_MLX_FP32};
pub use mega_asr::{MegaAsr, MegaAsrProfile, MEGA_ASR_8BIT};
pub use mms::{Mms, MmsConfig, MmsProfile, MMS_1B_FL102_ENGLISH};
pub use moonshine::{Moonshine, MoonshineConfig};
pub use moss_music::{MossMusic, MossMusicProfile, MOSS_MUSIC_8B_THINKING_4BIT};
pub use moss_transcribe_diarize::{
    MossTranscribeDiarize, MossTranscribeDiarizeProfile, MOSS_TRANSCRIBE_DIARIZE_4BIT,
};
pub use nemo::{AlignedResult, AlignedSentence, AlignedToken};
pub use nemotron_asr::{NemotronAsr, NemotronAsrConfig, NemotronAsrProfile, NEMOTRON_3_5_ASR};
pub use parakeet::{
    ParakeetConfig, ParakeetProfile, ParakeetTdt, PARAKEET_REDUX, PARAKEET_TDT_V2, PARAKEET_TDT_V3,
};
pub use phonon::{Phonon, PhononProfile, PHONON_1};
pub use qwen2_audio::{Qwen2Audio, Qwen2AudioProfile, QWEN2_AUDIO_7B_INSTRUCT_4BIT};
pub use qwen3_asr::{Qwen3Asr, Qwen3AsrProfile, QWEN3_ASR_06B_8BIT};
pub use qwen3_forced_aligner::{
    ForcedAlignItem, ForcedAlignResult, Qwen3ForcedAligner, Qwen3ForcedAlignerConfig,
    Qwen3ForcedAlignerProfile, QWEN3_FORCED_ALIGNER_06B_8BIT,
};
pub use sensevoice::{SenseVoiceOutput, SenseVoiceProfile, SenseVoiceSmall, SENSEVOICE_SMALL};
pub use vibevoice_asr::{VibeVoiceAsr, VibeVoiceAsrProfile, VIBEVOICE_ASR_STREAMING_1_5B};
pub use voxtral::{Voxtral, VoxtralProfile, VOXTRAL_MINI_3B_BF16};
pub use voxtral_realtime::{
    VoxtralRealtime, VoxtralRealtimeProfile, VOXTRAL_MINI_4B_REALTIME_4BIT,
};
pub use wav2vec::{Wav2Vec, Wav2VecConfig, Wav2VecProfile, WAV2VEC2_BASE_960H};
pub use whisper::{WhisperConfig, WhisperModel, WhisperSpecialTokens, WhisperWeights};
