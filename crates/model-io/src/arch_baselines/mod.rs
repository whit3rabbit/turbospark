//! The canonical architecture baselines, checked against an installed
//! model's manifest at load time. Ported from the `ArchConfig` static
//! members in `Infrastructure/ModelIO/ModelTypes.swift`.

mod deepseek;
mod gemma;
mod gpt_oss;
mod llama;
mod muse_glimmer;
mod qwen;

pub use deepseek::deepseek_v4_flash_284b_a13b;
pub use gemma::gemma4_26b_a4b;
pub use gpt_oss::gpt_oss_20b;
pub use llama::mixtral_8x7b;
pub use muse_glimmer::{muse_glimmer_30b, muse_glimmer_layer_mask};
pub use qwen::{qwen3_30b_a3b, qwen4_exp_125b_a6b, qwen_gdn_dense_27b, qwen_gdn_moe_35b_a3b};

use crate::arch_config::{ArchConfig, ModelFamily};

/// Registry keyed by `manifest.arch.family` for auto-detection at load.
pub fn known_architecture(family: ModelFamily) -> ArchConfig {
    match family {
        ModelFamily::Gemma4 => gemma4_26b_a4b(),
        ModelFamily::QwenGdnMoe => qwen_gdn_moe_35b_a3b(),
        ModelFamily::DeepseekV4Flash => deepseek_v4_flash_284b_a13b(),
        ModelFamily::Llama => mixtral_8x7b(),
        ModelFamily::Qwen3Moe => qwen3_30b_a3b(),
        ModelFamily::GptOss => gpt_oss_20b(),
        ModelFamily::QwenGdnDense => qwen_gdn_dense_27b(),
        ModelFamily::MuseGlimmer => muse_glimmer_30b(),
        ModelFamily::Qwen4Exp => qwen4_exp_125b_a6b(),
    }
}

pub fn all_known_architectures() -> Vec<ArchConfig> {
    vec![
        gemma4_26b_a4b(),
        qwen_gdn_moe_35b_a3b(),
        deepseek_v4_flash_284b_a13b(),
        mixtral_8x7b(),
        qwen3_30b_a3b(),
        gpt_oss_20b(),
        qwen_gdn_dense_27b(),
        muse_glimmer_30b(),
        qwen4_exp_125b_a6b(),
    ]
}
