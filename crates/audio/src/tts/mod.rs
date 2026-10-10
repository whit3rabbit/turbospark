//! Text-to-Speech (TTS) models.
//!
//! Aligned with `mlx_audio/tts/models/`.

pub mod bailingmm;
pub mod breeze_tts;
pub mod higgs_audio;
pub mod higgs_audio_v3;
pub mod indextts;
pub mod kokoro;
pub mod llama;
pub mod moss_tts_nano;
pub mod outetts;
pub mod qwen3;
pub mod rumik_oss;
pub mod sampling;
pub mod sesame;
pub mod soprano;

pub use bailingmm::Model as BailingmmTts;
pub use breeze_tts::Model as BreezeTts;
pub use higgs_audio::Model as HiggsAudioTts;
pub use higgs_audio_v3::Model as HiggsAudioV3Tts;
pub use indextts::Model as IndexTts;
pub use kokoro::Kokoro;
pub use llama::Model as LlamaTts;
pub use moss_tts_nano::Model as MossTtsNano;
pub use outetts::Model as OuteTts;
pub use qwen3::Model as Qwen3Tts;
pub use rumik_oss::Model as RumikOss;
pub use sesame::Model as SesameTts;
pub use soprano::Model as SopranoTts;
