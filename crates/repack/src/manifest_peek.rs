//! Reconstruct a full `ArchConfig` from an installed `.gturbo` directory's
//! `manifest.json` `arch` object, without knowing the shape up front: shape
//! fields are read as written, family-extension fields fall back to the
//! resolved FAMILY's baseline values (the same fallback rule
//! `arch_validation` applies to omitted manifest fields, once the family is
//! known). Shared by the CLI's real generation path and the bench
//! harness's real-install mode.

use std::path::Path;

use model_io::{ArchConfig, LinearAttentionConfig, ModelFamily};

/// Read `model_dir/manifest.json` and rebuild the `ArchConfig` it
/// describes. Accepts `gemma4` and `qwen36`; rejects `deepseekV4Flash`,
/// whose compressed-attention kernels are unported.
pub fn peek_manifest_arch(model_dir: &Path) -> Result<ArchConfig, String> {
    let manifest_path = model_dir.join("manifest.json");
    let bytes = std::fs::read(&manifest_path)
        .map_err(|e| format!("no manifest.json at {}: {e}", manifest_path.display()))?;
    let value: serde_json::Value =
        serde_json::from_slice(&bytes).map_err(|e| format!("manifest.json: {e}"))?;
    let m: model_io::ManifestArch = serde_json::from_value(value["arch"].clone())
        .map_err(|e| format!("manifest.json arch: {e}"))?;

    let family = match m.family.as_deref() {
        None => ModelFamily::Gemma4,
        Some(raw) => {
            ModelFamily::parse(raw).ok_or_else(|| format!("unknown arch.family {raw:?}"))?
        }
    };
    if family == ModelFamily::DeepseekV4Flash {
        return Err(
            "manifest family deepseekV4Flash is not supported by real generation yet".to_string(),
        );
    }

    // Start from that family's baseline -- the fallback `arch_validation`
    // itself uses for omitted family-extension fields -- then overwrite
    // every field the manifest actually carries.
    let mut arch = model_io::known_architecture(family);
    arch.hidden_size = m.hidden_size;
    arch.intermediate_size = m.ffn_intermediate;
    arch.moe_intermediate_size = m.moe_intermediate_size;
    arch.num_heads = m.num_heads;
    arch.num_kv_heads = m.num_kv_heads;
    arch.num_full_kv_heads = m.num_full_kv_heads;
    arch.head_dim = m.head_dim;
    arch.full_head_dim = m.full_head_dim;
    arch.vocab_size = m.vocab_size;
    arch.num_layers = m.num_layers;
    arch.sliding_window = m.sliding_window;
    arch.final_logit_softcap = m.final_logit_softcap;
    arch.rope_theta = m.rope_theta;
    arch.full_rope_theta = m.full_rope_theta;
    arch.partial_rotary_factor = m.partial_rotary_factor;
    arch.num_experts = m.num_experts;
    arch.top_k_experts = m.top_k_experts;
    arch.tie_word_embeddings = m.tie_word_embeddings;
    arch.attention_k_eq_v = m.attention_k_eq_v;
    arch.hidden_activation = m.hidden_activation.clone();
    arch.full_attention_layer_mask = m
        .full_attention_layer_mask
        .iter()
        .map(|&v| v as u8)
        .collect();

    // Family extensions. Every one of these has to be read back, not
    // assumed: a synthetic install of a family has the family's flags but
    // NOT its shapes, and `validate_arch` compares all of them.
    let base = arch.clone();
    arch.attn_output_gate = m.attn_output_gate.unwrap_or(base.attn_output_gate);
    arch.attention_scale = m.attention_scale.unwrap_or(base.attention_scale);
    arch.embedding_scaled_by_sqrt_hidden = m
        .embedding_scaled_by_sqrt_hidden
        .unwrap_or(base.embedding_scaled_by_sqrt_hidden);
    arch.router_scaled = m.router_scaled.unwrap_or(base.router_scaled);
    arch.ffn_sandwich_norms = m.ffn_sandwich_norms.unwrap_or(base.ffn_sandwich_norms);
    arch.shared_expert_gated = m.shared_expert_gated.unwrap_or(base.shared_expert_gated);
    arch.rope_neox_subdim = m.rope_neox_subdim.unwrap_or(base.rope_neox_subdim);
    arch.linear_attention = LinearAttentionConfig {
        num_k_heads: m.linear_num_k_heads.unwrap_or(0),
        num_v_heads: m.linear_num_v_heads.unwrap_or(0),
        key_head_dim: m.linear_key_head_dim.unwrap_or(0),
        value_head_dim: m.linear_value_head_dim.unwrap_or(0),
        conv_kernel_size: m.linear_conv_kernel_size.unwrap_or(0),
    };
    Ok(arch)
}
