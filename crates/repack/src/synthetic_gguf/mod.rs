//! Builds byte-exact GGUF v3 files in memory, for exercising
//! [`crate::gguf_header`] and the GGUF repack walk without a multi-GB
//! download. Sibling of [`crate::synthetic_model`] and
//! [`crate::synthetic_qwen`], and used the same way.
//!
//! This is a WRITER FOR TESTS, not a production encoder: it lays every
//! tensor out at the next aligned offset in the order pushed, and it does
//! not quantize anything. Callers hand it already-packed bytes, which is
//! also exactly how the repack walk treats a real file, so a round trip
//! through here proves the same byte-identity property the walk must
//! preserve.
//!
//! The weights it emits are deterministic but meaningless. The existing
//! rule for the other synthetic builders applies unchanged: a test may
//! assert structure, offsets, and byte identity, never generated text.

mod builder;
mod gemma4;
mod gemma4_shape;
mod gptoss;

pub use builder::{GgufBuilder, GgufFileAndRanges};
pub use gemma4::{build_synthetic_gemma4_gguf, QuantMix, SyntheticGgufShape};
pub use gptoss::{build_synthetic_gpt_oss_gguf, SyntheticGptOssShape};
