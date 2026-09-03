//! Synthetic Qwen model fixture generators (MoE and Dense).

mod dense;
mod dense_arch;
mod dense_tensors;
mod moe;
mod qwen4;
mod vision;

pub use dense::{
    build_synthetic_qwen_gdn_dense_install, build_synthetic_qwen_gdn_dense_install_at_bits,
    build_synthetic_qwen_gdn_dense_install_with_both_drafters,
    build_synthetic_qwen_gdn_dense_install_with_dflash,
    build_synthetic_qwen_gdn_dense_install_with_dflash_streamed,
    build_synthetic_qwen_gdn_dense_install_with_mtp,
    build_synthetic_qwen_gdn_dense_install_with_mtp_streamed,
    build_synthetic_qwen_gdn_dense_install_with_vision,
    build_synthetic_qwen_gdn_dense_install_with_vision_streamed, tiny_qwen_gdn_dense_arch,
};
pub use moe::{
    build_synthetic_qwen_gdn_moe_install, build_synthetic_qwen_gdn_moe_install_with_mtp,
    tiny_qwen_gdn_moe_arch,
};
pub use qwen4::{
    build_synthetic_qwen4_exp_install, build_synthetic_qwen4_exp_install_streamed,
    tiny_qwen4_exp_arch,
};
pub use vision::tiny_vision_config;
