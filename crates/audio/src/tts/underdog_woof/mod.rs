//! Underdog Woof: a small English text-to-speech model with two fixed voices
//! (Juno, Mira). First slice only: the checkpoint reader and the dequantized
//! speech-stage weights, gated by the frozen tensor hashes. There is no
//! forward pass, decoder, normalizer or RNG here yet; see
//! [README.md](README.md) and `docs/UNDERDOG_WOOF.md` for what is open.
//!
//! Reference: upstream `ConwayResearch/Underdog-Woof` at revision
//! `be702960539777fc3f698e54335276a8391b00f9` (`underdog_woof/runtime.py`,
//! NumPy). There is no mlx-audio implementation to compare against, so the
//! NumPy runtime is the normative reference for this family.
//!
//! License: Underdog Non-Commercial License 1.0. Non-commercial use only,
//! credit "Voice by Underdog Woof", no training or distillation on the model
//! or its audio. See the README before wiring this into anything shipped.

mod npz;
mod weights;

#[cfg(test)]
mod tests;

pub use npz::{f16_to_f32, Dtype, Npy, Npz};
pub use weights::{
    deq, lin_w, tensor_record, tensor_records, tensor_values_digest, Dense, Linear, NamedTensor,
    SpeechConfig, SpeechWeights, TensorRecord,
};

/// Upstream revision the weights and the frozen hashes were taken from.
pub const SOURCE_REVISION: &str = "be702960539777fc3f698e54335276a8391b00f9";
/// SHA-256 of the upstream `woof_int8.npz` (40,595,426 bytes).
pub const SOURCE_WEIGHTS_SHA256: &str =
    "d895fbbe84b8f4f461e3e94ab48b1164353b1988ce8d365f5aa8263d0de1203e";
/// Output sample rate.
pub const SAMPLE_RATE: u32 = 24_000;
