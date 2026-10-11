//! MegaTTS3 voice-cloning TTS (ByteDance), staged port.
//!
//! Reference: [bytedance/MegaTTS3](https://github.com/bytedance/MegaTTS3)
//! (Apache-2.0); there is no mlx-audio port, so parity is against the
//! unmodified PyTorch classes. Weights come from `tools/convert_megatts3.py`.
//!
//! Landed so far:
//! - [`facade`]: [`MegaTts3`], the end-to-end pipeline.
//! - [`postprocess`], [`rng`]: loudness matching, crossfade, and the randomness contract.
//! - [`aligner`]: the Whisper-style prompt aligner and its token parser.
//! - [`dit`]: the 24-layer latent diffusion transformer with three-row CFG.
//! - [`duration`]: the autoregressive duration LM and chunk post-processing.
//! - [`g2p`]: the Qwen2 grapheme-to-phoneme model.
//! - [`text`]: English language gate, normalizer, and chunker.
//! - [`sampler`]: the sway/AMO Euler sampler with injectable noise.
//! - [`rel_transformer`]: the relative-position text encoder (also the
//!   duration LM's text encoder).
//!
//! The pipeline joins these with [`crate::codec::wavvae`]; see [`MegaTts3`].

pub mod aligner;
pub mod dit;
pub mod duration;
mod execution;
pub mod facade;
pub mod g2p;
pub mod postprocess;
pub mod rel_transformer;
pub mod rng;
pub mod sampler;
pub mod text;

#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_aligner;
#[cfg(test)]
mod tests_duration;
#[cfg(test)]
mod tests_facade;
#[cfg(test)]
mod tests_g2p;
#[cfg(test)]
mod tests_post;
#[cfg(test)]
mod tests_text;

pub use aligner::{AlignedPrompt, Aligner, AlignerConfig};
pub use dit::{Dit, DitCondition, DitConfig};
pub use duration::{DurationConfig, DurationLm, DurationPrompt};
pub use facade::{
    activation_reserve_bytes, plan_chunks, resample_reference, MegaTts3, SynthParams,
    SynthProgress, VoicePrompt, MAX_PROMPT_SECONDS, MODEL_TYPE, SAMPLE_RATE,
};
pub use g2p::G2p;
pub use postprocess::Crossfader;
pub use rng::{Randomness, ReplayRng, SeededRng};
pub use sampler::SamplerParams;
