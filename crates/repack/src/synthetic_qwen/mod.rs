//! Synthetic Qwen model fixture generators (MoE and Dense).

mod dense;
mod moe;

pub use dense::{
    build_synthetic_qwen_gdn_dense_install, build_synthetic_qwen_gdn_dense_install_at_bits,
    build_synthetic_qwen_gdn_dense_install_with_mtp,
    build_synthetic_qwen_gdn_dense_install_with_mtp_streamed, tiny_qwen_gdn_dense_arch,
};
pub use moe::{build_synthetic_qwen_gdn_moe_install, tiny_qwen_gdn_moe_arch};
