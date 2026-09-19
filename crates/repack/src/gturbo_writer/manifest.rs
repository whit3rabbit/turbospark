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
    // The vision tower's two files, listed only when the arch declares one.
    // Keyed on `arch.vision.is_active()` and not on the directory existing,
    // because a `files` entry names a file this reader will then `read` --
    // probing the filesystem instead would turn a walk that forgot to write
    // the blobs into an install that validates and is missing its tower.
    let vision: &[&str] = if arch.vision.is_active() {
        &["packed_vision/layout.json", "packed_vision/blobs.bin"]
    } else {
        &[]
    };
    for relative in ["model_weights.bin", "packed_experts/layout.json"]
        .into_iter()
        .map(String::from)
        .chain((0..num_layers).map(|l| format!("packed_experts/layer_{l:02}.bin")))
        .chain(vision.iter().map(|s| String::from(*s)))
    {
        let path = dir.join(&relative);
        // Streamed rather than read whole: `model_weights.bin` alone can be
        // tens of gigabytes on a real MoE install, and this function already
        // runs after that file is fully written to disk, so there is no
        // reason to hold a second full copy of it in memory just to hash it.
        let size = std::fs::metadata(&path)
            .map_err(|e| io_err(&path, e))?
            .len();
        let sha256 = model_io::hash_file(&path, 1 << 20).map_err(|e| WriterError::Io {
            path: path.display().to_string(),
            detail: e.to_string(),
        })?;
        files.insert(
            relative,
            serde_json::json!({
                "size": size,
                "sha256": sha256,
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
            "routerScoringFunc": arch.router_scoring_func,
            "ffnSandwichNorms": arch.ffn_sandwich_norms,
            "sharedExpertGated": arch.shared_expert_gated,
            "ropeNeoxSubdim": arch.rope_neox_subdim,
            "linearNumKHeads": arch.linear_attention.num_k_heads,
            "linearNumVHeads": arch.linear_attention.num_v_heads,
            "linearKeyHeadDim": arch.linear_attention.key_head_dim,
            "linearValueHeadDim": arch.linear_attention.value_head_dim,
            "linearConvKernelSize": arch.linear_attention.conv_kernel_size,
            "linearOutputGateSigmoid": arch.linear_attention.output_gate_sigmoid,
            // ROADMAP M5. `swigluLimit` was VALIDATED AND NEVER WRITTEN, and
            // that was invisible for exactly as long as every writable
            // family's value was the `unwrap_or(0.0)` fallback. `gpt-oss` is
            // the first with a non-zero one (7.0), so an omitted field would
            // have compared 0.0 against 7.0 and refused a correct install.
            // Same species as the gotcha the block comment above describes,
            // one field further along.
            "swigluLimit": arch.swiglu_limit,
            "ropeScalingFactor": arch.rope_scaling.factor,
            "ropeScalingOriginalContext": arch.rope_scaling.original_context,
            "ropeScalingBetaFast": arch.rope_scaling.beta_fast,
            "ropeScalingBetaSlow": arch.rope_scaling.beta_slow,
            "ropeScalingMscale": arch.rope_scaling.mscale,
            // `deepseek2`'s MLA block and its dense lead, unconditional for
            // the same reason every field above is: validation falls back on
            // an omitted field, so an MLA install that said nothing would
            // validate its latent rank against zero and a dense-lead
            // install would validate its FFN width against zero. Zero is
            // `MlaConfig::NONE`'s value and every non-MLA family writes it.
            "mlaKvLoraRank": arch.mla.kv_lora_rank,
            "mlaQLoraRank": arch.mla.q_lora_rank,
            "mlaNopeHeadDim": arch.mla.nope_head_dim,
            "mlaRopeHeadDim": arch.mla.rope_head_dim,
            "mlaVHeadDim": arch.mla.v_head_dim,
            "denseLeadIntermediateSize": arch.dense_lead_intermediate_size,
            "numDenseLeadingLayers": arch.num_dense_leading_layers,
            // ROADMAP M-V3, the vision tower, and unconditional for the block
            // comment's reason rather than a new one: an omitted field is
            // resolved against a baseline, so an install that HAS a tower and
            // says nothing would validate its 27 blocks against zero. Every
            // pre-M-V3 family writes `VisionConfig::NONE`'s zeros here, which
            // is what those installs already validate against.
            "visionDepth": arch.vision.depth,
            "visionHiddenSize": arch.vision.hidden_size,
            "visionIntermediateSize": arch.vision.intermediate_size,
            "visionNumHeads": arch.vision.num_heads,
            "visionPatchSize": arch.vision.patch_size,
            "visionTemporalPatchSize": arch.vision.temporal_patch_size,
            "visionInChannels": arch.vision.in_channels,
            "visionSpatialMergeSize": arch.vision.spatial_merge_size,
            "visionNumPositionEmbeddings": arch.vision.num_position_embeddings,
            "visionOutHiddenSize": arch.vision.out_hidden_size,
            "visionMropeSection": arch.vision.mrope_section,
            "visionStartTokenId": arch.vision.vision_start_token_id,
            "visionEndTokenId": arch.vision.vision_end_token_id,
            "visionImageTokenId": arch.vision.image_token_id,
            "visionVideoTokenId": arch.vision.video_token_id,
            // Written only when non-empty, so every install whose tower has
            // no deepstack serializes the same manifest it always did and an
            // empty list stays indistinguishable from the absence that
            // `vision_deepstack_visual_indexes.unwrap_or_default` resolves
            // to the same way.
            "visionDeepstackVisualIndexes": if arch.vision.deepstack_visual_indexes.is_empty() {
                serde_json::Value::Null
            } else {
                serde_json::json!(arch.vision.deepstack_visual_indexes)
            },
            // **THE HOLE THE COMMENT HERE USED TO DESCRIBE IS NOW CLOSED, and
            // it was closed by a family arriving rather than by anyone finding
            // it.** Every `ca*` and `hc*` field was VALIDATED and written by
            // nothing, which was unreachable only because the one family using
            // them (DeepSeek-V4-Flash) is refused at open and no walk can
            // produce such an install. This comment said so and said it would
            // become real "the day a DSV4 install can be written".
            //
            // `qwen4_exp` is that day by a different door: it declares
            // `hyper_connections` AND `compressed_attention`, and it HAS a
            // repack path. Left unwritten, `arch_validation` would resolve its
            // `hcMult` to 0 and compare against 4, so every install this walk
            // produced would fail to open -- loud, which is the good
            // direction, and still an install nobody could use.
            //
            // Unconditional for the block comment's reason, not a new one: an
            // omitted field is resolved against a BASELINE, so an install that
            // has four residual streams and says nothing validates them
            // against Gemma's zero. Every earlier family writes the `NONE`
            // zeros it already validates against, so none of them moves.
            "caIndexNHeads": arch.compressed_attention.index_n_heads,
            "caIndexKvHeads": arch.compressed_attention.index_kv_heads,
            "caIndexHeadDim": arch.compressed_attention.index_head_dim,
            "caIndexTopK": arch.compressed_attention.index_top_k,
            "caIndexBudget": arch.compressed_attention.index_budget,
            // SERDE RENAMES: `caCSACompressRate`, not the camelCase this file
            // otherwise derives. A mismatch here deserializes to None and then
            // validates against 0, which is silent.
            "caCSACompressRate": arch.compressed_attention.csa_compress_rate,
            "caQLoraRank": arch.compressed_attention.q_lora_rank,
            "caOLoraRank": arch.compressed_attention.o_lora_rank,
            "caOGroups": arch.compressed_attention.o_groups,
            "caRopeHeadDim": arch.compressed_attention.rope_head_dim,
            "caHCACompressRate": arch.compressed_attention.hca_compress_rate,
            "caCompressRopeTheta": arch.compressed_attention.compress_rope_theta,
            "caRopeScalingFactor": arch.compressed_attention.rope_scaling_factor,
            "caRopeScalingOriginalMax": arch.compressed_attention.rope_scaling_original_max,
            "caRopeScalingBetaFast": arch.compressed_attention.rope_scaling_beta_fast,
            "caRopeScalingBetaSlow": arch.compressed_attention.rope_scaling_beta_slow,
            "hcMult": arch.hyper_connections.mult,
            "hcLowrank": arch.hyper_connections.lowrank,
            "hcSinkhornIters": arch.hyper_connections.sinkhorn_iters,
            "hcEps": arch.hyper_connections.eps,
            // The n-gram PLE table (`qwen4_exp`), unconditional for the same
            // reason. `PleConfig::NONE`'s zeros and an EMPTY id list are what
            // every other family writes.
            "pleNgramSize": arch.ple.ngram_size,
            "pleHeadsPerNgram": arch.ple.heads_per_ngram,
            "pleNgramVocabSizeBase": arch.ple.ngram_vocab_size_base,
            "pleMakeDivisibleBy": arch.ple.make_divisible_by,
            "pleSplitNgramParts": arch.ple.split_ngram_parts,
            "pleEmbedDim": arch.ple.ple_embed_dim,
            "pleConvKernelSize": arch.ple.conv_kernel_size,
            "pleLayerIds": arch.ple.layer_ids,
            "pleSeed": arch.ple.seed,
            "pleEosTokenId": arch.ple.eos_token_id,
        },
        "quant": null,
        "files": files,
        "expertsPerLayer": experts_per_layer,
        "numLayers": num_layers,
        "expertStride": expert_stride,
    }))
}
