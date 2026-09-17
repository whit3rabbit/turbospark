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
        output_gate_sigmoid: m.linear_output_gate_sigmoid.unwrap_or(false),
    };

    // THE THREE `qwen4_exp` BLOCKS, and `unwrap_or` on the ZERO rather than on
    // the baseline for the vision block's reason below: `arch_validation`
    // compares each against `PleConfig::NONE`'s and `NONE`'s zeros, because an
    // absent field means the install declares no such component and another
    // family's answer about ITS component is not evidence.
    //
    // Wired in the SAME change as the writer and the validator, which is the
    // whole lesson of the vision block's own story: a manifest field has three
    // consumers and only two of them fail loudly when one is missed. Reading
    // these back matters more than the tower's did, because
    // `hyper_connections.mult` decides how WIDE the residual stream is -- a
    // peeker resolving it to zero hands a caller an `ArchConfig` that says one
    // stream for a model that has four.
    arch.compressed_attention = model_io::CompressedAttentionConfig {
        index_n_heads: m.ca_index_n_heads.unwrap_or(0),
        index_kv_heads: m.ca_index_kv_heads.unwrap_or(0),
        index_head_dim: m.ca_index_head_dim.unwrap_or(0),
        index_top_k: m.ca_index_top_k.unwrap_or(0),
        index_budget: m.ca_index_budget.unwrap_or(0),
        csa_compress_rate: m.ca_csa_compress_rate.unwrap_or(0),
        q_lora_rank: m.ca_q_lora_rank.unwrap_or(0),
        o_lora_rank: m.ca_o_lora_rank.unwrap_or(0),
        o_groups: m.ca_o_groups.unwrap_or(0),
        rope_head_dim: m.ca_rope_head_dim.unwrap_or(0),
        hca_compress_rate: m.ca_hca_compress_rate.unwrap_or(0),
        compress_rope_theta: m.ca_compress_rope_theta.unwrap_or(0.0),
        rope_scaling_factor: m.ca_rope_scaling_factor.unwrap_or(0.0),
        rope_scaling_original_max: m.ca_rope_scaling_original_max.unwrap_or(0),
        rope_scaling_beta_fast: m.ca_rope_scaling_beta_fast.unwrap_or(0.0),
        rope_scaling_beta_slow: m.ca_rope_scaling_beta_slow.unwrap_or(0.0),
    };
    arch.hyper_connections = model_io::HyperConnectionConfig {
        mult: m.hc_mult.unwrap_or(0),
        lowrank: m.hc_lowrank.unwrap_or(0),
        sinkhorn_iters: m.hc_sinkhorn_iters.unwrap_or(0),
        eps: m.hc_eps.unwrap_or(0.0),
    };
    arch.ple = model_io::PleConfig {
        ngram_size: m.ple_ngram_size.unwrap_or(0),
        heads_per_ngram: m.ple_heads_per_ngram.unwrap_or(0),
        ngram_vocab_size_base: m.ple_ngram_vocab_size_base.unwrap_or(0),
        make_divisible_by: m.ple_make_divisible_by.unwrap_or(0),
        split_ngram_parts: m.ple_split_ngram_parts.unwrap_or(0),
        ple_embed_dim: m.ple_embed_dim.unwrap_or(0),
        conv_kernel_size: m.ple_conv_kernel_size.unwrap_or(0),
        layer_ids: m.ple_layer_ids.clone().unwrap_or_default(),
        seed: m.ple_seed.unwrap_or(0),
        eos_token_id: m.ple_eos_token_id.unwrap_or(0),
    };
    // Multi-head latent attention (`deepseek2`) and the dense lead, resolved
    // on the same zero-fallback rule as the three blocks above.
    arch.mla = model_io::MlaConfig {
        kv_lora_rank: m.mla_kv_lora_rank.unwrap_or(0),
        q_lora_rank: m.mla_q_lora_rank.unwrap_or(0),
        nope_head_dim: m.mla_nope_head_dim.unwrap_or(0),
        rope_head_dim: m.mla_rope_head_dim.unwrap_or(0),
        v_head_dim: m.mla_v_head_dim.unwrap_or(0),
    };
    arch.dense_lead_intermediate_size = m.dense_lead_intermediate_size.unwrap_or(0);
    arch.num_dense_leading_layers = m.num_dense_leading_layers.unwrap_or(0);
    // YaRN, read back rather than left on the baseline. The writer has
    // carried these five fields since ROADMAP M5, and `arch_validation`
    // compares each against the baseline's value, so an install that omits
    // them can only be a pre-M5 one -- which can only belong to a family
    // whose baseline is `NONE`. `deepseek2` is the first family whose
    // scaling differs from the only other yarn baseline's, which is what
    // makes reading it back load-bearing rather than tidy: the rope
    // frequency table and both rope kernels take their mscale from here.
    arch.rope_scaling = model_io::RopeScalingConfig {
        factor: m.rope_scaling_factor.unwrap_or(0.0),
        original_context: m.rope_scaling_original_context.unwrap_or(0),
        beta_fast: m.rope_scaling_beta_fast.unwrap_or(0.0),
        beta_slow: m.rope_scaling_beta_slow.unwrap_or(0.0),
        mscale: m.rope_scaling_mscale.unwrap_or(0.0),
    };

    // THE VISION TOWER, AND `unwrap_or(0)` RATHER THAN THE BASELINE, which is
    // the one place this function's own fallback rule does not apply.
    //
    // Every other family-extension field above falls back to the resolved
    // family's baseline, because that is what `arch_validation` does for an
    // omitted one. For the tower it does NOT: `arch_validation` compares
    // `a.vision_depth.unwrap_or(0)`, because an absent field means the
    // install declares no tower and another family's answer about ITS tower
    // is not evidence. Reading it the baseline way costs nothing today (every
    // shipped baseline carries `NONE`) and would be wrong the moment one does
    // not.
    //
    // **This block was MISSING for a release and the failure was silent.**
    // M-V3 added the fields to the manifest and to `arch_validation` and not
    // here, so every caller of this function -- the CLI's real generation
    // path and the bench harness -- resolved a vision install to
    // `VisionConfig::NONE`. The install opened, decoded text correctly, and
    // refused an image as though it carried no tower. Found by M-V4's parity
    // gate on the first real run; nothing else asks this function about a
    // component the peeker was never taught.
    //
    // `ManifestArch::vision_config()` is the SAME `unwrap_or(0)` resolution
    // `arch_validation` applies inline, codified once so this reader and the
    // loader cannot drift apart a second time.
    arch.vision = m.vision_config();
    Ok(arch)
}
