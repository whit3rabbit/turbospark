//! Audio codecs, learned tokenizers, and neural vocoders.
//!
//! Aligned with `mlx_audio/codec/models/` (e.g. SNAC, EnCodec, Mimi, DAC,
//! Vocos, BigVGAN). Shared weight-norm layers live in [`wnconv`], the shared
//! conv containers and weight-layout helpers in [`conv`]; each
//! family owns its block stack and loader under its subfolder. The
//! [central inventory](../../MODELS.md#codec) tracks upstream source
//! families, checkpoint profiles, and verification status.

pub mod bigvgan;
pub mod conv;
pub(crate) mod dac;
pub mod dacvae;
pub mod descript;
pub mod ecapa_tdnn;
pub mod encodec;
pub mod fish_s1_dac;
pub mod higgs_audio;
pub mod mimi;
pub mod mimo_audio_tokenizer;
pub mod moss_audio_tokenizer;
pub mod nemotron_voicechat;
pub mod s3;
pub mod snac;
pub mod stepaudio2;
pub mod vocos;
pub mod vq;
pub mod wnconv;
