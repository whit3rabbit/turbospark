//! GGUF block-quantization reference (ROADMAP Phase G Stage 2).
//!
//! Sibling of [`crate::quant`], and deliberately a separate module because the
//! layout is not a variant of the affine one. MLX affine stores three planar
//! regions per tensor (nibbles, BF16 scales, BF16 biases) at group 64. A GGUF
//! block is self-contained and interleaved: its scale sits immediately before
//! the weights it scales, so a row is one contiguous byte run with no
//! companion arrays. Keeping the two apart is what stops a caller passing a
//! Q8_0 row to an affine helper, where the argument types would otherwise
//! agree.
//!
//! Q8_0 is also SYMMETRIC where affine is not: `w = q * d` with signed `q`,
//! against affine's `w = q * scale + bias` with unsigned `q`. There is no bias
//! to carry, and a value of zero is exactly representable, which affine's
//! min/max derivation does not guarantee.
//!
//! Q8_0 leads Q4_K here for the reason recorded in ROADMAP Phase G: a Q8_0
//! block is 34 bytes with one scale, so this reference proves the plumbing
//! rather than fighting Q4_K's 6-bit packed sub-scales.
//!
//! Q4_K is the second type, and it is NOT a wider Q8_0. It is a two-level
//! scheme: a 256-element superblock carries two f16 super-scales, then eight
//! 32-element sub-blocks each carry a 6-bit scale and a 6-bit min quantized
//! against those. It is also ASYMMETRIC (`w = d*sc*q - dmin*m`, unsigned
//! 4-bit `q`), which puts it closer to this port's affine layout than to
//! Q8_0 on that one axis while differing from both on every other.
//!
//! Q6_K is the third, and it is a third shape again rather than a wider Q4_K.
//! A 256-element superblock carries one f16 super-scale and SIXTEEN signed
//! int8 sub-block scales (plain bytes, no 6-bit packing), and an element's
//! six bits are split across two runs: four low bits in `ql`, two high bits
//! in `qh`. It is symmetric like Q8_0, with a fixed bias of 32 subtracted off
//! the stored value rather than a per-sub-block min. Qwen 3.6's Q4_K_M ships
//! exactly one Q6_K tensor (`output.weight`), which is why it is here.

pub mod pearson;
pub mod q4_k;
pub mod q5_k;
pub mod q6_k;
pub mod q8_0;

pub use pearson::pearson;
pub use q4_k::{
    dequant_q4_k_gemv, dequantize_q4_k, quantize_q4_k, Q4_K_BLOCK_BYTES, Q4_K_BLOCK_ELEMS,
    Q4_K_SUB_ELEMS,
};
pub use q5_k::{
    dequant_q5_k_gemv, dequantize_q5_k, Q5_K_BLOCK_BYTES, Q5_K_BLOCK_ELEMS, Q5_K_SUB_ELEMS,
};
pub use q6_k::{
    dequant_q6_k_gemv, dequantize_q6_k, quantize_q6_k, Q6_K_BLOCK_BYTES, Q6_K_BLOCK_ELEMS,
    Q6_K_SUB_ELEMS,
};
pub use q8_0::{
    dequant_q8_0_gemv, dequantize_q8_0, quantize_q8_0, Q8_0_BLOCK_BYTES, Q8_0_BLOCK_ELEMS,
};
