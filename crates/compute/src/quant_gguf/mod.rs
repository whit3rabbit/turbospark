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
//!
//! Q3_K is the fourth, and it is the one whose every "obvious" reading of a
//! sibling is wrong: sixteen 16-element groups per superblock rather than
//! eight 32-element ones, 2-bit levels whose SIGN comes from a separate
//! high-bit run instead of an unsigned value with a min, sixteen 6-bit
//! scales packed into twelve bytes through a four-word shuffle, and the f16
//! super-scale as the LAST field (only Q2_K among the K-quants agrees).

/// Pearson correlation helper for validating dequantized weight similarities.
pub mod pearson;
/// Q2_0 block quantization (64-element blocks with signed 2-bit levels).
pub mod q2_0;
/// Q2_K block quantization (256-element superblocks, 16 sub-blocks, asymmetric 2-bit weights).
pub mod q2_k;
/// Q3_K block quantization (256-element superblocks, 16 groups of 16, signed
/// 2-bit levels through a high-bit run, packed 6-bit scales, super-scale last).
pub mod q3_k;
/// Q4_0 block quantization (32-element blocks with signed 4-bit levels).
pub mod q4_0;
/// Q4_K block quantization (256-element superblocks, 8 sub-blocks, asymmetric 4-bit weights).
pub mod q4_k;
/// Q5_0 block quantization (32-element blocks with signed 5-bit levels).
pub mod q5_0;
/// Q5_K block quantization (256-element superblocks, 8 sub-blocks, 5-bit weights with min scale).
pub mod q5_k;
/// Q6_K block quantization (256-element superblocks, 16 sub-blocks, 6-bit weights split into low/high nibbles).
pub mod q6_k;
/// Q8_0 block quantization (32-element symmetric blocks with f16 delta scale).
pub mod q8_0;

pub use pearson::pearson;
pub use q2_0::{dequant_q2_0_gemv, dequantize_q2_0, Q2_0_BLOCK_BYTES, Q2_0_BLOCK_ELEMS};
pub use q2_k::{
    dequant_q2_k_gemv, dequantize_q2_k, quantize_q2_k, Q2_K_BLOCK_BYTES, Q2_K_BLOCK_ELEMS,
    Q2_K_SUB_ELEMS,
};
pub use q3_k::{
    dequant_q3_k_gemv, dequantize_q3_k, q3_k_decode_scales, quantize_q3_k, Q3_K_BLOCK_BYTES,
    Q3_K_BLOCK_ELEMS, Q3_K_SUB_ELEMS,
};
pub use q4_0::{dequantize_q4_0, Q4_0_BLOCK_BYTES, Q4_0_BLOCK_ELEMS};
pub use q4_k::{
    dequant_q4_k_gemv, dequantize_q4_k, quantize_q4_k, Q4_K_BLOCK_BYTES, Q4_K_BLOCK_ELEMS,
    Q4_K_SUB_ELEMS,
};
pub use q5_0::{dequantize_q5_0, Q5_0_BLOCK_BYTES, Q5_0_BLOCK_ELEMS};
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
