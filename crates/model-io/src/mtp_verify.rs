//! What the speculative (MTP) batched verify dispatches, by resident name
//! and dtype tag.
//!
//! Shared by the runtime's open-time gate (`MtpState::speculation_blocker`)
//! and repack's GGUF header prediction (`repack::gguf_speculation`), so a
//! probe and the session it leads to cannot disagree about which installs
//! speculate. Here because both crates already depend on this one.

/// Every trunk projection the batched verify sends through the runtime's
/// `encode_gemm_any`, per layer. A linear layer has no `self_attn.*` and a
/// full one no `linear_attn.*`, so absent names are skipped by both readers.
pub const MTP_VERIFY_PROJECTIONS: [&str; 12] = [
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "linear_attn.in_proj_qkv.weight",
    "linear_attn.in_proj_z.weight",
    "linear_attn.in_proj_a.weight",
    "linear_attn.in_proj_b.weight",
    "linear_attn.out_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
];

/// The resident dtype tags the runtime's `encode_gemm_any` has a batched arm
/// for. Mirrors that function's literal arms: 1 raw BF16, 4 INT4 affine, 6
/// Q8_0, 7 Q4_K, 8 Q6_K, 13 Q5_K, and 15/16 the 1-bit/2-bit affine pair.
/// Each is held bit-exact against its GEMV by the gpu crate's parity tests.
pub const MTP_BATCHED_GEMM_DTYPES: [u8; 8] = [1, 4, 6, 7, 8, 13, 15, 16];

/// The GGUF block types in [`MTP_BATCHED_GEMM_DTYPES`]. A GGUF trunk mixes
/// block types by tensor, so for these the gate scans every projection.
pub const MTP_GGUF_BATCHED_DTYPES: [u8; 4] = [6, 7, 8, 13];
