//! Synthetic Qwen model fixture generators (MoE and Dense).

mod dense;
mod dense_arch;
mod dense_tensors;
mod moe;
mod qwen4;
mod qwen4_decode;
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
pub use qwen4_decode::{
    build_synthetic_qwen4_exp_decode_install, build_synthetic_qwen4_exp_decode_install_raw_router,
    tiny_qwen4_exp_decode_arch, HC_COUNT, HEAD_DIM, HIDDEN as QWEN4_DECODE_HIDDEN,
    NGRAM_EOS_TOKEN_ID as QWEN4_DECODE_NGRAM_EOS_TOKEN_ID, NUM_EXPERTS,
    NUM_HEADS as QWEN4_DECODE_NUM_HEADS, NUM_KV_HEADS as QWEN4_DECODE_NUM_KV_HEADS,
    NUM_LAYERS as QWEN4_DECODE_NUM_LAYERS, PLE_LAYER as QWEN4_DECODE_PLE_LAYER, TOP_K,
};
pub use vision::tiny_vision_config;
