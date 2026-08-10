//! Builds a small, fully in-memory-generated "tiny Gemma 4" `.gturbo`
//! install: real INT4-affine-quantized weights (deterministic, not
//! trained), written through the real resident-tensor writer and readable
//! back through every `turbospark_model_io` loader. This is what
//! `crates/runtime`'s `RealForwardRunner` (macOS/GPU only) drives for its
//! end-to-end real-forward-pass test, since no trained `.gturbo` checkpoint
//! is available in this environment.
//!
//! Every non-shape architecture field is set to Gemma 4's own real
//! canonical baseline value (`turbospark_model_io::gemma4_26b_a4b`): this is
//! honestly a tiny Gemma-4-architecture model, not an invented one.

pub mod arch;
pub mod dense;
pub mod moe;

pub use arch::{
    down_proj_name, embed_lm_head_name, expert_down_proj_name, expert_gate_proj_name,
    expert_up_proj_name, gate_proj_name, k_proj_name, o_proj_name, q_proj_name, router_name,
    tiny_gemma4_arch, up_proj_name,
};
pub use dense::{build_synthetic_gemma4_install, build_synthetic_gemma4_swa_install};
pub use moe::{build_synthetic_gemma4_moe_install, build_synthetic_gemma4_moe_streamed_install};
