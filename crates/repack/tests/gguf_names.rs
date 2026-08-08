//! GGUF-to-canonical tensor name mapping for both supported families.
//!
//! The rows under test were read off real files (see the module doc on
//! `gguf_names.rs`); these assertions pin them so a later edit cannot
//! quietly rename a tensor the runtime looks up by string.

use model_io::ModelFamily;
use mrefrust_repack::{
    family_for_architecture, gguf_architecture, map_gguf_name, GgufMapping, GgufNameError,
};

fn resident(name: &str, family: ModelFamily) -> String {
    match map_gguf_name(name, family) {
        Ok(GgufMapping::Resident(n)) => n,
        other => panic!("{name}: expected a resident mapping, got {other:?}"),
    }
}

#[test]
fn maps_gemma4_attention_and_norms() {
    let f = ModelFamily::Gemma4;
    let p = "language_model.model.layers.7.";
    for (gguf, canonical) in [
        ("blk.7.attn_q.weight", "self_attn.q_proj.weight"),
        ("blk.7.attn_k.weight", "self_attn.k_proj.weight"),
        ("blk.7.attn_v.weight", "self_attn.v_proj.weight"),
        ("blk.7.attn_output.weight", "self_attn.o_proj.weight"),
        ("blk.7.attn_q_norm.weight", "self_attn.q_norm.weight"),
        ("blk.7.attn_k_norm.weight", "self_attn.k_norm.weight"),
        ("blk.7.attn_norm.weight", "input_layernorm.weight"),
        (
            "blk.7.post_attention_norm.weight",
            "post_attention_layernorm.weight",
        ),
        ("blk.7.ffn_norm.weight", "pre_feedforward_layernorm.weight"),
        (
            "blk.7.pre_ffw_norm_2.weight",
            "pre_feedforward_layernorm_2.weight",
        ),
        (
            "blk.7.post_ffw_norm.weight",
            "post_feedforward_layernorm.weight",
        ),
        (
            "blk.7.post_ffw_norm_1.weight",
            "post_feedforward_layernorm_1.weight",
        ),
        (
            "blk.7.post_ffw_norm_2.weight",
            "post_feedforward_layernorm_2.weight",
        ),
        ("blk.7.layer_output_scale.weight", "layer_scalar"),
    ] {
        assert_eq!(resident(gguf, f), format!("{p}{canonical}"), "{gguf}");
    }
}

/// The two `.scale` tensors are matched by shape, and one of them lives
/// under a misleading prefix: `ffn_down_exps.scale` is the ROUTER's
/// per-expert scale, not anything to do with the down projection.
#[test]
fn maps_gemma4_router_including_the_misleadingly_named_scale() {
    let f = ModelFamily::Gemma4;
    assert_eq!(
        resident("blk.3.ffn_gate_inp.weight", f),
        "language_model.model.layers.3.router.proj.weight"
    );
    assert_eq!(
        resident("blk.3.ffn_gate_inp.scale", f),
        "language_model.model.layers.3.router.scale"
    );
    assert_eq!(
        resident("blk.3.ffn_down_exps.scale", f),
        "language_model.model.layers.3.router.per_expert_scale"
    );
}

#[test]
fn gemma4_fuses_gate_and_up_but_qwen_does_not() {
    assert_eq!(
        map_gguf_name("blk.2.ffn_gate_up_exps.weight", ModelFamily::Gemma4).unwrap(),
        GgufMapping::RoutedFusedGateUp { layer: 2 }
    );
    assert_eq!(
        map_gguf_name("blk.2.ffn_down_exps.weight", ModelFamily::Gemma4).unwrap(),
        GgufMapping::Routed {
            layer: 2,
            role: "down"
        }
    );

    // Qwen keeps all three separate, so no split is needed there.
    for (gguf, role) in [
        ("blk.2.ffn_gate_exps.weight", "gate"),
        ("blk.2.ffn_up_exps.weight", "up"),
        ("blk.2.ffn_down_exps.weight", "down"),
    ] {
        assert_eq!(
            map_gguf_name(gguf, ModelFamily::Qwen36).unwrap(),
            GgufMapping::Routed { layer: 2, role },
            "{gguf}"
        );
    }
    // And Qwen has no fused tensor at all.
    assert!(map_gguf_name("blk.2.ffn_gate_up_exps.weight", ModelFamily::Qwen36).is_err());
}

#[test]
fn maps_qwen36_gated_deltanet_and_shared_expert() {
    let f = ModelFamily::Qwen36;
    let p = "language_model.model.layers.11.";
    for (gguf, canonical) in [
        ("blk.11.attn_qkv.weight", "linear_attn.in_proj_qkv.weight"),
        ("blk.11.attn_gate.weight", "linear_attn.in_proj_z.weight"),
        ("blk.11.ssm_alpha.weight", "linear_attn.in_proj_a.weight"),
        ("blk.11.ssm_beta.weight", "linear_attn.in_proj_b.weight"),
        ("blk.11.ssm_out.weight", "linear_attn.out_proj.weight"),
        ("blk.11.ssm_conv1d.weight", "linear_attn.conv1d.weight"),
        ("blk.11.ssm_norm.weight", "linear_attn.norm.weight"),
        ("blk.11.ffn_gate_inp.weight", "mlp.gate.weight"),
        (
            "blk.11.ffn_gate_inp_shexp.weight",
            "mlp.shared_expert_gate.weight",
        ),
        (
            "blk.11.ffn_gate_shexp.weight",
            "mlp.shared_expert.gate_proj.weight",
        ),
        (
            "blk.11.ffn_up_shexp.weight",
            "mlp.shared_expert.up_proj.weight",
        ),
        (
            "blk.11.ffn_down_shexp.weight",
            "mlp.shared_expert.down_proj.weight",
        ),
    ] {
        assert_eq!(resident(gguf, f), format!("{p}{canonical}"), "{gguf}");
    }
}

/// AGENTS.md Gotcha 26: these two carry no `.weight` suffix on either side
/// of the mapping. Appending one by analogy produces a `MissingTensor` at
/// open, which is the good failure; a name filter keyed on `.weight` that
/// drops them is the bad one.
#[test]
fn qwen36_a_log_and_dt_bias_keep_their_suffixless_names() {
    let f = ModelFamily::Qwen36;
    assert_eq!(
        resident("blk.4.ssm_a", f),
        "language_model.model.layers.4.linear_attn.A_log"
    );
    assert_eq!(
        resident("blk.4.ssm_dt.bias", f),
        "language_model.model.layers.4.linear_attn.dt_bias"
    );
}

#[test]
fn maps_top_level_tensors_for_both_families() {
    for f in [ModelFamily::Gemma4, ModelFamily::Qwen36] {
        assert_eq!(
            resident("token_embd.weight", f),
            "language_model.model.embed_tokens.weight"
        );
        assert_eq!(
            resident("output_norm.weight", f),
            "language_model.model.norm.weight"
        );
        // Present only when embeddings are untied (Qwen). Mapping it for
        // both families is right: absence is handled by not requiring it,
        // not by refusing to name it.
        assert_eq!(
            resident("output.weight", f),
            "language_model.lm_head.weight"
        );
    }
}

#[test]
fn ignores_derived_rope_frequencies_visibly() {
    assert!(matches!(
        map_gguf_name("rope_freqs.weight", ModelFamily::Gemma4).unwrap(),
        GgufMapping::Ignored { .. }
    ));
}

/// An unmapped name must be an error rather than a skip. Gotcha 26 records
/// what a silent skip costs: every expert quietly becomes resident and the
/// install runs correctly at many times its intended footprint.
#[test]
fn refuses_to_silently_skip_an_unknown_tensor() {
    match map_gguf_name("blk.0.something_new.weight", ModelFamily::Gemma4) {
        Err(GgufNameError::Unmapped { name, family }) => {
            assert_eq!(name, "blk.0.something_new.weight");
            assert_eq!(family, "gemma4");
        }
        other => panic!("expected Unmapped, got {other:?}"),
    }
    assert!(map_gguf_name("mystery.weight", ModelFamily::Qwen36).is_err());
    // A Qwen tensor offered to the Gemma mapping is also unmapped, rather
    // than falling through to a same-named Gemma row.
    assert!(map_gguf_name("blk.0.ssm_a", ModelFamily::Gemma4).is_err());
}

#[test]
fn rejects_a_malformed_layer_index() {
    assert!(matches!(
        map_gguf_name("blk.x.attn_q.weight", ModelFamily::Gemma4),
        Err(GgufNameError::BadLayerIndex { .. })
    ));
}

/// llama.cpp named Qwen 3.6's converter after the 3.5 series, so the GGUF
/// architecture string is NOT the family's own `as_str()`. Deriving it
/// would fail to recognize every Qwen GGUF published.
#[test]
fn architecture_strings_are_the_converters_names_not_the_familys() {
    assert_eq!(gguf_architecture(ModelFamily::Gemma4), Some("gemma4"));
    assert_eq!(gguf_architecture(ModelFamily::Qwen36), Some("qwen35moe"));
    assert_ne!(
        gguf_architecture(ModelFamily::Qwen36),
        Some(ModelFamily::Qwen36.as_str())
    );

    assert_eq!(family_for_architecture("gemma4"), Some(ModelFamily::Gemma4));
    assert_eq!(
        family_for_architecture("qwen35moe"),
        Some(ModelFamily::Qwen36)
    );
    assert_eq!(family_for_architecture("qwen36"), None);
    assert_eq!(family_for_architecture("llama"), None);
}

/// DeepSeek V4 has no repack path and must not acquire one by accident.
#[test]
fn deepseek_v4_is_refused() {
    assert_eq!(gguf_architecture(ModelFamily::DeepseekV4Flash), None);
    assert!(map_gguf_name("blk.0.attn_q.weight", ModelFamily::DeepseekV4Flash).is_err());
}
