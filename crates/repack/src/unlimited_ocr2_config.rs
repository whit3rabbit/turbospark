//! Unlimited-OCR-2 `config.json` parsing and header canonicalization.
//!
//! The architecture keys live twice, at the root and under `language_config`,
//! and they agree on the shipped checkpoint. The text block is read, because
//! that is the wrapper's meaning everywhere else.
//!
//! v1 shares `model_type: "unlimited-ocr"` and has no DASH, so the first check
//! here refuses it rather than decoding it with the wrong cache. Facts:
//! `docs/UNLIMITED_OCR2_PHASE0.md`.

use model_io::{
    ArchConfig, CompressedAttentionConfig, DashConfig, HyperConnectionConfig,
    LinearAttentionConfig, MlaConfig, ModelFamily, PleConfig, RopeScalingConfig, VisionConfig,
};

use crate::gemma4_checkpoint::Gemma4Error;
use crate::safetensors_header::SafetensorsHeader;

/// The namespace tower tensors take in the renamed header. Outside
/// `language_model.`, so the resident walk drops them and
/// `unlimited_ocr2_tower` writes them, and spelled with `.blocks.` throughout
/// because `layer_index` reads `.layers.` as a trunk layer.
pub const UOCR2_TOWER_PREFIX: &str = "uocr2_tower.";

fn config_err(detail: impl Into<String>) -> Gemma4Error {
    Gemma4Error::Config(detail.into())
}

/// Parse a checkpoint's `config.json` into the canonical architecture shape.
pub fn parse_unlimited_ocr2_config(json: &str) -> Result<ArchConfig, Gemma4Error> {
    let root: serde_json::Value =
        serde_json::from_str(json).map_err(|e| config_err(e.to_string()))?;
    crate::arch_registry::refuse_foreign_config(&root, ModelFamily::UnlimitedOcr2)
        .map_err(Gemma4Error::Config)?;
    let sparse = root.get("sparse_attention_config").ok_or_else(|| {
        config_err(
            "no sparse_attention_config: this is Unlimited-OCR v1, whose R-SWA ring cache has \
             no flow here. v1 runs through the Core ML profile (unlimited-ocr-coreml).",
        )
    })?;
    let tc = root
        .get("language_config")
        .ok_or_else(|| config_err("missing language_config"))?;

    let i = |key: &str| -> Result<i64, Gemma4Error> {
        tc.get(key)
            .and_then(serde_json::Value::as_i64)
            .ok_or_else(|| config_err(format!("missing language_config.{key}")))
    };
    let hidden_size = i("hidden_size")?;
    let num_heads = i("num_attention_heads")?;
    let num_kv_heads = i("num_key_value_heads")?;
    let num_layers = i("num_hidden_layers")?;
    let dense_width = i("intermediate_size")?;
    let moe_width = i("moe_intermediate_size")?;
    let shared = i("n_shared_experts")?;
    let num_experts = i("n_routed_experts")?;
    let top_k = i("num_experts_per_tok")?;
    let vocab_size = i("vocab_size")?;
    let dense_lead = i("first_k_dense_replace")?;
    for (name, v) in [
        ("hidden_size", hidden_size),
        ("num_attention_heads", num_heads),
        ("num_key_value_heads", num_kv_heads),
        ("num_hidden_layers", num_layers),
        ("intermediate_size", dense_width),
        ("moe_intermediate_size", moe_width),
        ("n_shared_experts", shared),
        ("n_routed_experts", num_experts),
        ("num_experts_per_tok", top_k),
        ("vocab_size", vocab_size),
    ] {
        if v <= 0 {
            return Err(config_err(format!("{name} must be positive, got {v}")));
        }
    }
    if hidden_size % num_heads != 0 || num_heads != num_kv_heads {
        return Err(config_err(format!(
            "hidden_size {hidden_size} over {num_heads} heads and {num_kv_heads} KV heads: the \
             flow needs equal head counts that divide the width"
        )));
    }
    if top_k > num_experts || dense_lead < 0 || dense_lead >= num_layers {
        return Err(config_err(format!(
            "top-k {top_k} of {num_experts} experts with {dense_lead} dense leading layers of \
             {num_layers} is not a shape the flow runs"
        )));
    }
    let head_dim = hidden_size / num_heads;

    // Behaviours the flow does not implement. Each would stay fluent.
    if tc.get("use_mla").and_then(serde_json::Value::as_bool) != Some(false) {
        return Err(config_err(
            "language_config.use_mla must be false; MLA is the deepseek2 flow",
        ));
    }
    let root_str = |key: &str| root.get(key).and_then(serde_json::Value::as_str);
    if root_str("hidden_act") != Some("silu") {
        return Err(config_err("hidden_act must be silu"));
    }
    if root_str("scoring_func") != Some("softmax")
        || root_str("topk_method") != Some("greedy")
        || root
            .get("norm_topk_prob")
            .and_then(serde_json::Value::as_bool)
            != Some(false)
    {
        return Err(config_err(
            "the router must be greedy softmax top-k with norm_topk_prob false",
        ));
    }
    let routed_scale = root
        .get("routed_scaling_factor")
        .and_then(serde_json::Value::as_f64)
        .ok_or_else(|| config_err("missing routed_scaling_factor"))?;
    if routed_scale != 1.0 {
        return Err(config_err(format!(
            "routed_scaling_factor {routed_scale} is not wired; the flow assumes 1.0"
        )));
    }
    if root
        .get("moe_layer_freq")
        .and_then(serde_json::Value::as_i64)
        != Some(1)
    {
        return Err(config_err("moe_layer_freq must be 1"));
    }
    if root
        .get("attention_bias")
        .and_then(serde_json::Value::as_bool)
        != Some(false)
    {
        return Err(config_err("attention_bias must be false"));
    }
    if root
        .get("tie_word_embeddings")
        .and_then(serde_json::Value::as_bool)
        != Some(false)
    {
        return Err(config_err("the lm_head must be untied"));
    }
    let rms_eps = root
        .get("rms_norm_eps")
        .and_then(serde_json::Value::as_f64)
        .ok_or_else(|| config_err("missing rms_norm_eps"))?;
    if (rms_eps - 1e-6).abs() > 1e-12 {
        return Err(config_err(format!(
            "rms_norm_eps must be 1e-6, got {rms_eps}"
        )));
    }

    let rs = tc
        .get("rope_scaling")
        .ok_or_else(|| config_err("missing language_config.rope_scaling"))?;
    let rs_f = |key: &str| -> Result<f64, Gemma4Error> {
        rs.get(key)
            .and_then(serde_json::Value::as_f64)
            .ok_or_else(|| config_err(format!("missing rope_scaling.{key}")))
    };
    if rs_f("attention_factor")? != 1.0 {
        return Err(config_err("rope_scaling.attention_factor must be 1.0"));
    }
    let rope_theta = tc
        .get("rope_theta")
        .and_then(serde_json::Value::as_f64)
        .ok_or_else(|| config_err("missing language_config.rope_theta"))?;

    DashConfig::parse(sparse, num_layers as usize, head_dim as usize)
        .map_err(Gemma4Error::Config)?;

    Ok(ArchConfig {
        hidden_size,
        // The fused shared expert, which is what the tensor's own rows say.
        intermediate_size: moe_width * shared,
        moe_intermediate_size: moe_width,
        num_heads,
        num_kv_heads,
        num_full_kv_heads: num_kv_heads,
        head_dim,
        full_head_dim: head_dim,
        vocab_size,
        sliding_window: 0,
        final_logit_softcap: 0.0,
        rope_theta,
        full_rope_theta: rope_theta,
        partial_rotary_factor: 1.0,
        num_layers,
        num_experts,
        top_k_experts: top_k,
        tie_word_embeddings: false,
        attention_k_eq_v: false,
        full_attention_layer_mask: vec![1u8; num_layers as usize],
        hidden_activation: "silu".to_string(),
        family: ModelFamily::UnlimitedOcr2,
        attn_output_gate: false,
        attention_scale: (head_dim as f64).powf(-0.5),
        embedding_scaled_by_sqrt_hidden: false,
        router_scaled: false,
        ffn_sandwich_norms: false,
        shared_expert_gated: false,
        rope_neox_subdim: false,
        linear_attention: LinearAttentionConfig::NONE,
        mla: MlaConfig::NONE,
        compressed_attention: CompressedAttentionConfig::NONE,
        hyper_connections: HyperConnectionConfig::NONE,
        num_hash_routed_layers: 0,
        router_scoring_func: "softmax".to_string(),
        routed_scaling_factor: routed_scale,
        swiglu_limit: 0.0,
        rope_scaling: RopeScalingConfig {
            factor: rs_f("factor")?,
            original_context: rs_f("original_max_position_embeddings")? as i64,
            beta_fast: rs_f("beta_fast")?,
            beta_slow: rs_f("beta_slow")?,
            mscale: 1.0,
        },
        vision: VisionConfig::NONE,
        short_conv_kernel_size: 0,
        ple: PleConfig::NONE,
        dense_lead_intermediate_size: dense_width,
        num_dense_leading_layers: dense_lead,
    })
}

/// The source name's install spelling, or `None` for a name this walk has
/// never seen, which then falls through to the walk's own refusal.
fn canonical_name(name: &str) -> Option<String> {
    if let Some(tail) = name.strip_prefix("language_model.") {
        return Some(format!(
            "language_model.{}",
            tail.replace(".mlp.shared_experts.", ".mlp.shared_expert.")
        ));
    }
    if let Some(tail) = name.strip_prefix("sam_model.") {
        return Some(format!("{UOCR2_TOWER_PREFIX}sam.{tail}"));
    }
    if let Some(tail) = name.strip_prefix("vision_model.") {
        let tail = tail.replace("transformer.layers.", "transformer.blocks.");
        return Some(format!("{UOCR2_TOWER_PREFIX}clip.{tail}"));
    }
    if let Some(tail) = name.strip_prefix("projector.layers.") {
        return Some(format!("{UOCR2_TOWER_PREFIX}projector.{tail}"));
    }
    match name {
        "image_newline" | "view_separator" => Some(format!("{UOCR2_TOWER_PREFIX}{name}")),
        _ => None,
    }
}

/// Rename a shard header onto the install namespace. Renames every tower
/// tensor, so a name with no canonical spelling is refused here, naming it,
/// rather than as `Unknown` partway through a stream.
pub fn canonicalize_unlimited_ocr2_header(
    header: &mut SafetensorsHeader,
) -> Result<(), Gemma4Error> {
    let mut renamed = std::collections::BTreeMap::new();
    for (name, info) in std::mem::take(&mut header.tensors) {
        let canonical = canonical_name(&name).ok_or_else(|| {
            config_err(format!(
                "tensor {name} is not part of Unlimited-OCR-2's layout"
            ))
        })?;
        if renamed.insert(canonical, info).is_some() {
            return Err(config_err(format!("tensor {name} collides after renaming")));
        }
    }
    header.tensors = renamed;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const REAL_CONFIG: &str = include_str!("../tests/fixtures/unlimited_ocr2_4bit_config.json");

    #[test]
    fn the_shipped_config_is_the_baseline() {
        let parsed = parse_unlimited_ocr2_config(REAL_CONFIG).unwrap();
        assert_eq!(parsed, model_io::unlimited_ocr2_12l());
    }

    #[test]
    fn v1_is_refused_by_name() {
        let mut root: serde_json::Value = serde_json::from_str(REAL_CONFIG).unwrap();
        root.as_object_mut()
            .unwrap()
            .remove("sparse_attention_config");
        let err = parse_unlimited_ocr2_config(&root.to_string()).unwrap_err();
        assert!(err.to_string().contains("v1"), "{err}");
    }

    #[test]
    fn refuses_behaviours_the_flow_does_not_implement() {
        for (key, value) in [
            ("hidden_act", serde_json::json!("gelu")),
            ("scoring_func", serde_json::json!("sigmoid")),
            ("topk_method", serde_json::json!("noaux_tc")),
            ("norm_topk_prob", serde_json::json!(true)),
            ("routed_scaling_factor", serde_json::json!(2.5)),
            ("moe_layer_freq", serde_json::json!(2)),
            ("attention_bias", serde_json::json!(true)),
            ("tie_word_embeddings", serde_json::json!(true)),
            ("rms_norm_eps", serde_json::json!(1e-5)),
        ] {
            let mut root: serde_json::Value = serde_json::from_str(REAL_CONFIG).unwrap();
            root[key] = value;
            assert!(
                parse_unlimited_ocr2_config(&root.to_string()).is_err(),
                "{key} was accepted"
            );
        }
        let mut root: serde_json::Value = serde_json::from_str(REAL_CONFIG).unwrap();
        root["language_config"]["use_mla"] = serde_json::json!(true);
        assert!(parse_unlimited_ocr2_config(&root.to_string()).is_err());
    }

    #[test]
    fn every_real_tensor_family_has_a_home() {
        for (source, install) in [
            (
                "language_model.model.layers.3.mlp.shared_experts.up_proj.weight",
                "language_model.model.layers.3.mlp.shared_expert.up_proj.weight",
            ),
            (
                "language_model.model.layers.3.mlp.switch_mlp.up_proj.weight",
                "language_model.model.layers.3.mlp.switch_mlp.up_proj.weight",
            ),
            (
                "language_model.model.layers.3.self_attn.dash_router.head_cls",
                "language_model.model.layers.3.self_attn.dash_router.head_cls",
            ),
            ("sam_model.net_2.weight", "uocr2_tower.sam.net_2.weight"),
            (
                "vision_model.transformer.layers.7.mlp.fc1.bias",
                "uocr2_tower.clip.transformer.blocks.7.mlp.fc1.bias",
            ),
            ("projector.layers.scales", "uocr2_tower.projector.scales"),
            ("image_newline", "uocr2_tower.image_newline"),
            ("view_separator", "uocr2_tower.view_separator"),
        ] {
            assert_eq!(canonical_name(source).as_deref(), Some(install));
        }
        assert_eq!(canonical_name("audio_tower.x.weight"), None);
    }

    #[test]
    fn a_renamed_tower_tensor_cannot_read_as_a_trunk_layer() {
        let name = canonical_name("vision_model.transformer.layers.7.mlp.fc1.bias").unwrap();
        // `layer_index` finds this substring in anything spelled like a layer.
        assert!(!name.contains(".layers."), "{name}");
    }
}
