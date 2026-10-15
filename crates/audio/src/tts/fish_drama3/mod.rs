//! Fish Audio Drama 3: a Qwen3.5-MoE text backbone, a temporal codebook
//! tower, and the Fish S1 DAC codec (`crate::codec::fish_s1_dac`).
//!
//! Reference: mlx-audio `mlx_audio/tts/models/fish_drama3/` at
//! [`REFERENCE_COMMIT`] (Blaizzy/mlx-audio PR 1018). That is newer than the
//! crate's normative mlx-audio commit, which has no Drama 3; the codec keeps
//! the crate's commit. The backbone runs outside this crate (a headless
//! `QwenGdnMoe` trunk in `crates/runtime`), so this module holds what is
//! portable: the prompt's text rules and chunking. Facts and gates:
//! `docs/FISH_DRAMA3_PHASE0.md`.

pub mod chunking;
pub mod config;
pub mod generate;
pub mod prompt;
pub mod sampling;
pub mod tower;

/// Pinned mlx-audio commit this port was transcribed from.
pub const REFERENCE_COMMIT: &str = "dad2087f790dfe96b07d5aac1405db4b578b411f";
