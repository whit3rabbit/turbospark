//! GGUF-to-canonical tensor name mapping for both supported families.
//!
//! The rows under test were read off real files (see the module doc on
//! `gguf_names.rs`); these assertions pin them so a later edit cannot
//! quietly rename a tensor the runtime looks up by string.

use model_io::ModelFamily;
use turbospark_repack::{
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
            map_gguf_name(gguf, ModelFamily::QwenGdnMoe).unwrap(),
            GgufMapping::Routed { layer: 2, role },
            "{gguf}"
        );
    }
    // And Qwen has no fused tensor at all.
    assert!(map_gguf_name("blk.2.ffn_gate_up_exps.weight", ModelFamily::QwenGdnMoe).is_err());
}

#[test]
fn maps_qwen36_gated_deltanet_and_shared_expert() {
    let f = ModelFamily::QwenGdnMoe;
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
    let f = ModelFamily::QwenGdnMoe;
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
    for f in [ModelFamily::Gemma4, ModelFamily::QwenGdnMoe] {
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

/// `rope_freqs.weight` is REFUSED, for every family, and it used to be
/// ignored (ROADMAP M4).
///
/// The old row read "RoPE frequencies are derived from `rope_theta` at
/// runtime", which is true of every checkpoint that OMITS this tensor and
/// false of every checkpoint that ships it: Llama 3.1's is a learned `[64]`
/// F32 scaling vector and the two rope kernels here take a scalar theta.
/// Dropping it gives an install that loads, decodes, and is wrong only past
/// the original training length -- no symptom at any length a smoke test
/// reaches.
///
/// Asserted for every family because the check sits ahead of the per-family
/// tables, so no family can grow a row for it by accident.
#[test]
fn refuses_a_learned_rope_frequency_scaling_for_every_family() {
    for family in [
        ModelFamily::Gemma4,
        ModelFamily::QwenGdnMoe,
        ModelFamily::Llama,
        ModelFamily::Qwen3Moe,
    ] {
        match map_gguf_name("rope_freqs.weight", family) {
            Err(GgufNameError::UnsupportedRopeScaling { name }) => {
                assert_eq!(name, "rope_freqs.weight");
            }
            other => panic!("{}: expected a refusal, got {other:?}", family.as_str()),
        }
    }
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
    assert!(map_gguf_name("mystery.weight", ModelFamily::QwenGdnMoe).is_err());
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
    assert_eq!(
        gguf_architecture(ModelFamily::QwenGdnMoe),
        Some("qwen35moe")
    );
    assert_ne!(
        gguf_architecture(ModelFamily::QwenGdnMoe),
        Some(ModelFamily::QwenGdnMoe.as_str())
    );

    assert_eq!(family_for_architecture("gemma4"), Some(ModelFamily::Gemma4));
    assert_eq!(
        family_for_architecture("qwen35moe"),
        Some(ModelFamily::QwenGdnMoe)
    );
    assert_eq!(family_for_architecture("qwen36"), None);
    // `llama` is the one architecture whose GGUF string EQUALS its family
    // name, and it gained a flow in ROADMAP Phase M2. `phi3` stands in as
    // the recognized-but-unported case this line used to make.
    assert_eq!(gguf_architecture(ModelFamily::Llama), Some("llama"));
    assert_eq!(family_for_architecture("llama"), Some(ModelFamily::Llama));
    assert_eq!(family_for_architecture("phi3"), None);
    // The second architecture whose GGUF string equals its family name.
    assert_eq!(gguf_architecture(ModelFamily::Qwen3Moe), Some("qwen3moe"));
    assert_eq!(
        family_for_architecture("qwen3moe"),
        Some(ModelFamily::Qwen3Moe)
    );
    // And it is NOT the Qwen 3.6 family, whose string it superficially
    // resembles. Two different models, two different flows.
    assert_ne!(
        family_for_architecture("qwen3moe"),
        family_for_architecture("qwen35moe")
    );
}

/// Qwen3-MoE's per-layer table: the `llama` set plus the two per-head norms.
///
/// The `llama` rows are asserted TOO, not taken on trust, because the two
/// tables are separate functions that happen to agree; if one drifts, the
/// decode flow they share would read the wrong tensor for one family only.
#[test]
fn maps_qwen3moe_as_llama_plus_the_two_per_head_norms() {
    let f = ModelFamily::Qwen3Moe;
    let p = "language_model.model.layers.3.";
    for (gguf, canonical) in [
        ("blk.3.attn_q.weight", "self_attn.q_proj.weight"),
        ("blk.3.attn_k.weight", "self_attn.k_proj.weight"),
        ("blk.3.attn_v.weight", "self_attn.v_proj.weight"),
        ("blk.3.attn_output.weight", "self_attn.o_proj.weight"),
        ("blk.3.attn_norm.weight", "input_layernorm.weight"),
        ("blk.3.ffn_norm.weight", "post_attention_layernorm.weight"),
        ("blk.3.ffn_gate_inp.weight", "mlp.gate.weight"),
        // The two rows `llama` does not have.
        ("blk.3.attn_q_norm.weight", "self_attn.q_norm.weight"),
        ("blk.3.attn_k_norm.weight", "self_attn.k_norm.weight"),
    ] {
        assert_eq!(resident(gguf, f), format!("{p}{canonical}"), "{gguf}");
    }
    for norm in ["attn_q_norm", "attn_k_norm"] {
        assert!(
            map_gguf_name(&format!("blk.3.{norm}.weight"), ModelFamily::Llama).is_err(),
            "{norm} must stay unmapped for the llama architecture, which has no per-head norms"
        );
    }
}

/// Its experts are unfused and there is no shared expert to map, which is
/// the MoE half of "the llama table plus two rows".
#[test]
fn qwen3moe_routes_three_unfused_experts_and_has_no_shared_expert() {
    let f = ModelFamily::Qwen3Moe;
    for (gguf, role) in [
        ("blk.5.ffn_gate_exps.weight", "gate"),
        ("blk.5.ffn_up_exps.weight", "up"),
        ("blk.5.ffn_down_exps.weight", "down"),
    ] {
        assert_eq!(
            map_gguf_name(gguf, f),
            Ok(GgufMapping::Routed { layer: 5, role }),
            "{gguf}"
        );
    }
    // Qwen 3.6's shared-expert names must not resolve here: this model has
    // no shared expert at all, and silently mapping one would put a tensor
    // in the install that no dispatch ever reads.
    for shared in [
        "blk.5.ffn_gate_shexp.weight",
        "blk.5.ffn_gate_inp_shexp.weight",
    ] {
        assert!(map_gguf_name(shared, f).is_err(), "{shared}");
    }
}

/// DeepSeek V4 has no repack path and must not acquire one by accident.
#[test]
fn deepseek_v4_is_refused() {
    assert_eq!(gguf_architecture(ModelFamily::DeepseekV4Flash), None);
    assert!(map_gguf_name("blk.0.attn_q.weight", ModelFamily::DeepseekV4Flash).is_err());
}

/// ROADMAP M5. Every row read off the real `gpt-oss-20b-MXFP4.gguf` header
/// (`gguf_checkpoint_network.rs::scopes_phase_m5_gpt_oss_layer`), pinned here
/// so the walk cannot lose one silently -- an unmapped name is refused, but a
/// name mapped to the WRONG canonical target is not.
#[test]
fn maps_gpt_oss_attention_biases_and_sinks() {
    let f = ModelFamily::GptOss;
    let p = "language_model.model.layers.3.";
    for (gguf, canonical) in [
        ("attn_q.weight", "self_attn.q_proj.weight"),
        ("attn_k.weight", "self_attn.k_proj.weight"),
        ("attn_v.weight", "self_attn.v_proj.weight"),
        ("attn_output.weight", "self_attn.o_proj.weight"),
        // The four rows no other family has: every projection is biased.
        ("attn_q.bias", "self_attn.q_proj.bias"),
        ("attn_k.bias", "self_attn.k_proj.bias"),
        ("attn_v.bias", "self_attn.v_proj.bias"),
        ("attn_output.bias", "self_attn.o_proj.bias"),
        // One learned logit per q head.
        ("attn_sinks.weight", "self_attn.sinks.weight"),
        ("attn_norm.weight", "input_layernorm.weight"),
        // `post_attention_norm`, where `llama` and `qwen3moe` say `ffn_norm`.
        (
            "post_attention_norm.weight",
            "post_attention_layernorm.weight",
        ),
        ("ffn_gate_inp.weight", "mlp.gate.weight"),
        ("ffn_gate_inp.bias", "mlp.gate.bias"),
    ] {
        assert_eq!(
            resident(&format!("blk.3.{gguf}"), f),
            format!("{p}{canonical}"),
            "{gguf}"
        );
    }
}

/// The per-expert biases go into the BLOB, under the three `*_biases` roles
/// the INT4-affine layout already defines and every GGUF install so far has
/// left empty.
///
/// This is the row most worth pinning, because the alternative spelling is
/// silent rather than fatal: mapping them `Resident` would produce an install
/// that writes every bias into `model_weights.bin`, opens, and decodes
/// without them -- `MoeExpertOffsets`' bias fields would simply stay 0, which
/// is what a blob with no biases looks like.
#[test]
fn gpt_oss_routes_its_per_expert_biases_into_the_blob() {
    for (gguf, role) in [
        ("ffn_gate_exps.weight", "gate"),
        ("ffn_up_exps.weight", "up"),
        ("ffn_down_exps.weight", "down"),
        ("ffn_gate_exps.bias", "gate_biases"),
        ("ffn_up_exps.bias", "up_biases"),
        ("ffn_down_exps.bias", "down_biases"),
    ] {
        assert_eq!(
            map_gguf_name(&format!("blk.7.{gguf}"), ModelFamily::GptOss).unwrap(),
            GgufMapping::Routed { layer: 7, role },
            "{gguf}"
        );
    }
}

/// gpt-oss does NOT fuse gate and up, so Gemma's fused tensor must not map
/// for it -- the same guard `gemma4_fuses_gate_and_up_but_qwen_does_not`
/// applies to Qwen.
#[test]
fn gpt_oss_does_not_take_gemmas_fused_expert_tensor() {
    assert!(map_gguf_name("blk.0.ffn_gate_up_exps.weight", ModelFamily::GptOss).is_err());
}
