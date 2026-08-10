//! Helpers for building manifest JSON and resident index structures.

use std::path::Path;

use model_io::ArchConfig;

use super::types::{io_err, WriterError};

/// A `model_weights.bin` with a valid 24-byte `ResidentIndexHeader`
/// (`index_size == HEADER_BYTES`, zero entries) followed by the raw
/// resident tensor region.
pub(crate) fn build_empty_resident_index(resident_tensor_bytes: &[u8]) -> Vec<u8> {
    const HEADER_BYTES: u64 = 24;
    let mut bytes = Vec::with_capacity(HEADER_BYTES as usize + resident_tensor_bytes.len());
    bytes.extend_from_slice(&HEADER_BYTES.to_le_bytes()); // index_size
    bytes.extend_from_slice(&(resident_tensor_bytes.len() as u64).to_le_bytes()); // resident_size
    bytes.extend_from_slice(&0u64.to_le_bytes()); // entry_count
    bytes.extend_from_slice(resident_tensor_bytes);
    bytes
}

pub(crate) fn build_manifest_json(
    arch: &ArchConfig,
    model_id: &str,
    expert_stride: u64,
    num_layers: usize,
    experts_per_layer: usize,
    dir: &Path,
) -> Result<serde_json::Value, WriterError> {
    let mut files = serde_json::Map::new();
    for relative in ["model_weights.bin", "packed_experts/layout.json"]
        .into_iter()
        .map(String::from)
        .chain((0..num_layers).map(|l| format!("packed_experts/layer_{l:02}.bin")))
    {
        let path = dir.join(&relative);
        let bytes = std::fs::read(&path).map_err(|e| io_err(&path, e))?;
        files.insert(
            relative,
            serde_json::json!({
                "size": bytes.len() as u64,
                "sha256": model_io::hash_data(&bytes),
            }),
        );
    }

    Ok(serde_json::json!({
        "magic": "GTURBO",
        "versionMajor": 1,
        "versionMinor": 0,
        "flags": {},
        "modelID": model_id,
        "sourceSnapshotHash": null,
        "arch": {
            "hiddenSize": arch.hidden_size,
            "ffnIntermediate": arch.intermediate_size,
            "moeIntermediateSize": arch.moe_intermediate_size,
            "numHeads": arch.num_heads,
            "numKVHeads": arch.num_kv_heads,
            "numFullKVHeads": arch.num_full_kv_heads,
            "headDim": arch.head_dim,
            "fullHeadDim": arch.full_head_dim,
            "vocabSize": arch.vocab_size,
            "slidingWindow": arch.sliding_window,
            "finalLogitSoftcap": arch.final_logit_softcap,
            "ropeTheta": arch.rope_theta,
            "fullRopeTheta": arch.full_rope_theta,
            "partialRotaryFactor": arch.partial_rotary_factor,
            "numLayers": arch.num_layers,
            "numExperts": arch.num_experts,
            "topKExperts": arch.top_k_experts,
            "tieWordEmbeddings": arch.tie_word_embeddings,
            "attentionKEqV": arch.attention_k_eq_v,
            "hiddenActivation": arch.hidden_activation,
            "fullAttentionLayerMask": arch.full_attention_layer_mask,
            // Family-extension fields, written UNCONDITIONALLY. `arch_validation`
            // falls back to the Gemma 4 baseline for anything omitted, so a
            // non-Gemma install that leaves them out can never validate (it was
            // compared against Gemma's values). Gemma installs are unaffected:
            // these are exactly the fallbacks for that family.
            "family": arch.family.as_str(),
            "attnOutputGate": arch.attn_output_gate,
            "attentionScale": arch.attention_scale,
            "embeddingScaledBySqrtHidden": arch.embedding_scaled_by_sqrt_hidden,
            "routerScaled": arch.router_scaled,
            "ffnSandwichNorms": arch.ffn_sandwich_norms,
            "sharedExpertGated": arch.shared_expert_gated,
            "ropeNeoxSubdim": arch.rope_neox_subdim,
            "linearNumKHeads": arch.linear_attention.num_k_heads,
            "linearNumVHeads": arch.linear_attention.num_v_heads,
            "linearKeyHeadDim": arch.linear_attention.key_head_dim,
            "linearValueHeadDim": arch.linear_attention.value_head_dim,
            "linearConvKernelSize": arch.linear_attention.conv_kernel_size,
        },
        "quant": null,
        "files": files,
        "expertsPerLayer": experts_per_layer,
        "numLayers": num_layers,
        "expertStride": expert_stride,
    }))
}
