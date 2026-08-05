//! Reconstruct a full `ArchConfig` from an installed `.gturbo` directory's
//! `manifest.json` `arch` object, without knowing the shape up front: shape
//! fields are read as written, family-extension fields fall back to the
//! Gemma 4 baseline values (the same fallback rule `arch_validation`
//! applies to omitted manifest fields). Shared by the CLI's real
//! generation path and the bench harness's real-install mode.

use std::path::Path;

use crate::synthetic_model::tiny_gemma4_arch;

/// Read `model_dir/manifest.json` and rebuild the `ArchConfig` it
/// describes. Rejects non-`gemma4` families: they are the only ones real
/// generation supports today.
pub fn peek_manifest_arch(model_dir: &Path) -> Result<model_io::ArchConfig, String> {
    let manifest_path = model_dir.join("manifest.json");
    let bytes = std::fs::read(&manifest_path)
        .map_err(|e| format!("no manifest.json at {}: {e}", manifest_path.display()))?;
    let value: serde_json::Value =
        serde_json::from_slice(&bytes).map_err(|e| format!("manifest.json: {e}"))?;
    let m: model_io::ManifestArch = serde_json::from_value(value["arch"].clone())
        .map_err(|e| format!("manifest.json arch: {e}"))?;

    // Start from the Gemma 4 baseline (the manifest's own fallback rule
    // for omitted family-extension fields) and overwrite every shape
    // field with what the manifest actually says.
    let mut arch = tiny_gemma4_arch(m.vocab_size, m.num_layers);
    arch.hidden_size = m.hidden_size;
    arch.intermediate_size = m.ffn_intermediate;
    arch.moe_intermediate_size = m.moe_intermediate_size;
    arch.num_heads = m.num_heads;
    arch.num_kv_heads = m.num_kv_heads;
    arch.num_full_kv_heads = m.num_full_kv_heads;
    arch.head_dim = m.head_dim;
    arch.full_head_dim = m.full_head_dim;
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
    if let Some(scale) = m.attention_scale {
        arch.attention_scale = scale;
    }
    if m.family.as_deref().is_some_and(|f| f != "gemma4") {
        return Err(format!(
            "manifest family {:?} is not supported by real generation yet",
            m.family
        ));
    }
    Ok(arch)
}
