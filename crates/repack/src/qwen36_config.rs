//! Qwen 3.6 `config.json` -> [`ArchConfig`].
//!
//! The ONE family-specific piece of the repack path. Everything downstream
//! of it -- `classify_for_family`, `manifest_quant`, `write_qwen36_install`,
//! `write_qwen36_install_streamed` -- already takes a `ModelFamily` and
//! needs nothing else from this module.
//!
//! Shaped like [`crate::parse_gemma4_config`], but almost every key name
//! differs, so it is a separate function rather than a parameterized one.
//! Measured against `mlx-community/Qwen3.6-35B-A3B-4bit`:
//!
//! | `ArchConfig` field | `config.json -> text_config` key |
//! |---|---|
//! | `intermediate_size` | `shared_expert_intermediate_size` (NOT `intermediate_size`, which the text config does not carry) |
//! | `top_k_experts` | `num_experts_per_tok` |
//! | `hidden_activation` | `hidden_act` (Gemma spells it `hidden_activation`) |
//! | `full_attention_layer_mask` | `layer_types`, `linear_attention` -> **2**, `full_attention` -> **1** (Gemma's mapping is 1/0) |
//! | `rope_theta`, `partial_rotary_factor` | `rope_parameters`, a FLAT object here; Gemma nests one sub-object per attention kind |
//! | `num_full_kv_heads`, `full_head_dim` | no separate global-attention keys: the full-attention layers reuse `num_key_value_heads` / `head_dim` |
//! | `linear_attention` | the five `linear_*` keys |
//!
//! `attention_scale` has no key at all. It is `head_dim ** -0.5`, taken from
//! the reference implementation (mlx-lm `Qwen3NextAttention.__init__` sets
//! `self.scale = self.head_dim**-0.5`), not assumed from the formula --
//! `docs/NEW_MODEL.md` Phase 0 explains why that distinction matters. At
//! `head_dim = 256` it is exactly 1/16, a binary fraction, so it survives
//! the serde_json round trip AGENTS.md Gotcha 24 warns about.
//!
//! Fields with no config key (`router_scaled`, `ffn_sandwich_norms`,
//! `rope_neox_subdim`, ...) are family constants, written out the same way
//! `parse_gemma4_config` writes Gemma's.

use model_io::{
    ArchConfig, CompressedAttentionConfig, HyperConnectionConfig, LinearAttentionConfig,
    ModelFamily,
};

use crate::gemma4_checkpoint::Gemma4Error;

/// Layer-mask code for a gated-DeltaNet linear-attention layer.
const MASK_LINEAR: u8 = 2;
/// Layer-mask code for a full-attention layer.
const MASK_FULL: u8 = 1;

/// Parses a Qwen 3.6 `config.json` into an [`ArchConfig`].
///
/// The checkpoint is multimodal (`Qwen3_5MoeForConditionalGeneration`), so
/// the text tower lives under a `text_config` wrapper and `vision_config`
/// is ignored -- `classify_for_family` drops the `vision_tower.` tensors
/// separately.
///
/// Cross-check the result against [`model_io::qwen36_35b_a3b`]: if the two
/// disagree field for field, one of them is wrong.
pub fn parse_qwen36_config(json: &str) -> Result<ArchConfig, Gemma4Error> {
    let root: serde_json::Value =
        serde_json::from_str(json).map_err(|e| Gemma4Error::Config(e.to_string()))?;
    // Text-only conversions drop the wrapper; accept both shapes.
    let tc = root.get("text_config").unwrap_or(&root);

    let i = |k: &str| -> Result<i64, Gemma4Error> {
        tc.get(k)
            .and_then(|v| v.as_i64())
            .ok_or_else(|| Gemma4Error::Config(format!("missing {k}")))
    };
    let b = |k: &str| tc.get(k).and_then(|v| v.as_bool()).unwrap_or(false);

    // `layer_types` is what makes a Qwen install a hybrid rather than a
    // plain transformer, and an empty or short mask would still validate
    // structurally while running the wrong block on every layer, so it is
    // required and length-checked rather than defaulted.
    let num_layers = i("num_hidden_layers")?;
    let types = tc
        .get("layer_types")
        .and_then(|v| v.as_array())
        .ok_or_else(|| Gemma4Error::Config("missing layer_types".to_string()))?;
    let mask = types
        .iter()
        .map(|t| match t.as_str() {
            Some("linear_attention") => Ok(MASK_LINEAR),
            Some("full_attention") => Ok(MASK_FULL),
            other => Err(Gemma4Error::Config(format!(
                "unknown layer_types entry {other:?}"
            ))),
        })
        .collect::<Result<Vec<u8>, _>>()?;
    if mask.len() as i64 != num_layers {
        return Err(Gemma4Error::Config(format!(
            "layer_types has {} entries but num_hidden_layers is {num_layers}",
            mask.len()
        )));
    }

    // Flat here, unlike Gemma's per-attention-kind sub-objects. The
    // partial rotary factor is duplicated at the text_config top level;
    // prefer the rope object and fall back to it.
    let rope = tc.get("rope_parameters");
    let rope_f = |key: &str| rope.and_then(|r| r.get(key)).and_then(|v| v.as_f64());
    let rope_theta = rope_f("rope_theta")
        .ok_or_else(|| Gemma4Error::Config("missing rope_parameters.rope_theta".to_string()))?;
    let prf = rope_f("partial_rotary_factor")
        .or_else(|| tc.get("partial_rotary_factor").and_then(|v| v.as_f64()))
        .ok_or_else(|| Gemma4Error::Config("missing partial_rotary_factor".to_string()))?;

    let head_dim = i("head_dim")?;
    // The NeoX sub-dimension RoPE rotates `rotary_dim / 2` pairs, so an odd
    // or non-integral rotary_dim would silently drop a channel.
    let rotary_dim = prf * head_dim as f64;
    if rotary_dim <= 0.0 || rotary_dim.fract() != 0.0 || (rotary_dim as i64) % 2 != 0 {
        return Err(Gemma4Error::Config(format!(
            "partial_rotary_factor {prf} x head_dim {head_dim} = {rotary_dim}, \
             which is not a positive even integer"
        )));
    }

    let kv_heads = i("num_key_value_heads")?;

    Ok(ArchConfig {
        hidden_size: i("hidden_size")?,
        intermediate_size: i("shared_expert_intermediate_size")?,
        moe_intermediate_size: i("moe_intermediate_size")?,
        num_heads: i("num_attention_heads")?,
        num_kv_heads: kv_heads,
        num_full_kv_heads: kv_heads,
        head_dim,
        full_head_dim: head_dim,
        vocab_size: i("vocab_size")?,
        sliding_window: 0,
        final_logit_softcap: 0.0,
        rope_theta,
        full_rope_theta: rope_theta,
        partial_rotary_factor: prf,
        num_layers,
        num_experts: i("num_experts")?,
        top_k_experts: i("num_experts_per_tok")?,
        tie_word_embeddings: b("tie_word_embeddings"),
        attention_k_eq_v: false,
        full_attention_layer_mask: mask,
        hidden_activation: tc
            .get("hidden_act")
            .and_then(|v| v.as_str())
            .unwrap_or("silu")
            .to_string(),
        family: ModelFamily::Qwen36,
        attn_output_gate: b("attn_output_gate"),
        attention_scale: (head_dim as f64).powf(-0.5),
        embedding_scaled_by_sqrt_hidden: false,
        router_scaled: false,
        ffn_sandwich_norms: false,
        shared_expert_gated: true,
        rope_neox_subdim: true,
        linear_attention: LinearAttentionConfig {
            num_k_heads: i("linear_num_key_heads")?,
            num_v_heads: i("linear_num_value_heads")?,
            key_head_dim: i("linear_key_head_dim")?,
            value_head_dim: i("linear_value_head_dim")?,
            conv_kernel_size: i("linear_conv_kernel_dim")?,
        },
        compressed_attention: CompressedAttentionConfig::NONE,
        hyper_connections: HyperConnectionConfig::NONE,
        num_hash_routed_layers: 0,
        router_scoring_func: "softmax".to_string(),
        routed_scaling_factor: 1.0,
        swiglu_limit: 0.0,
    })
}
