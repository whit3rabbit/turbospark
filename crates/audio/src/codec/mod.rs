//! Audio codecs, learned tokenizers, and neural vocoders.
//!
//! Aligned with `mlx_audio/codec/models/` (e.g. SNAC, EnCodec, Mimi, DAC,
//! Vocos, BigVGAN). Shared weight-norm layers live in [`wnconv`]; each
//! family owns its block stack and loader under its subfolder. The
//! [central inventory](../../MODELS.md#codec) tracks upstream source
//! families, checkpoint profiles, and verification status.

pub mod bigvgan;
pub mod descript;
pub mod ecapa_tdnn;
pub mod encodec;
pub mod snac;
pub mod vocos;
pub mod vq;
pub mod wnconv;
