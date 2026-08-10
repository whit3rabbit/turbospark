//! Tests for the architecture baselines and their derived layer/shape
//! helpers.

use turbospark_model_io::{
    deepseek_v4_flash_284b_a13b, gemma4_26b_a4b, known_architecture, qwen36_35b_a3b, ModelFamily,
};

#[test]
fn gemma4_layer_mask_marks_every_sixth_layer_full() {
    let arch = gemma4_26b_a4b();
    assert_eq!(arch.full_attention_layer_mask.len(), 30);
    for (i, &kind) in arch.full_attention_layer_mask.iter().enumerate() {
        let expected_full = i >= 5 && (i - 5) % 6 == 0;
        assert_eq!(kind == 1, expected_full, "layer {i}");
    }
    assert!(!arch.has_linear_attention_layers());
    assert!(!arch.has_compressed_attention_layers());
}

#[test]
fn qwen_layer_mask_is_mostly_linear_with_full_every_fourth() {
    let arch = qwen36_35b_a3b();
    assert_eq!(arch.full_attention_layer_mask.len(), 40);
    assert!(arch.has_linear_attention_layers());
    assert!(arch.layer_is_linear(0));
    assert!(arch.layer_is_full(3));
    assert!(!arch.layer_is_full(2));
}

#[test]
fn deepseek_layer_mask_has_csa_and_hca_from_layer_two() {
    let arch = deepseek_v4_flash_284b_a13b();
    assert_eq!(arch.full_attention_layer_mask.len(), 43);
    assert!(!arch.layer_is_compressed(0));
    assert!(!arch.layer_is_compressed(1));
    assert!(arch.layer_is_csa(2));
    assert!(arch.layer_is_hca(3));
    assert!(arch.layer_is_hash_routed(0));
    assert!(!arch.layer_is_hash_routed(3));
}

#[test]
fn known_architecture_matches_the_named_baseline() {
    assert_eq!(known_architecture(ModelFamily::Gemma4), gemma4_26b_a4b());
    assert_eq!(known_architecture(ModelFamily::Qwen36), qwen36_35b_a3b());
    assert_eq!(
        known_architecture(ModelFamily::DeepseekV4Flash),
        deepseek_v4_flash_284b_a13b()
    );
}

#[test]
fn model_family_round_trips_through_its_string_form() {
    for family in [
        ModelFamily::Gemma4,
        ModelFamily::Qwen36,
        ModelFamily::DeepseekV4Flash,
    ] {
        assert_eq!(ModelFamily::parse(family.as_str()), Some(family));
    }
    assert_eq!(ModelFamily::parse("unknown"), None);
}

#[test]
fn decode_int4_gemv_shapes_nonempty_for_every_baseline() {
    for arch in [
        gemma4_26b_a4b(),
        qwen36_35b_a3b(),
        deepseek_v4_flash_284b_a13b(),
    ] {
        let shapes = arch.decode_int4_gemv_shapes();
        assert!(!shapes.is_empty());
        assert!(shapes.iter().all(|(m, n)| *m > 0 && *n > 0));
    }
}

#[test]
fn decode_int8_gemv_shapes_include_router_and_optional_shared_expert_gate() {
    let gemma = gemma4_26b_a4b();
    assert_eq!(gemma.decode_int8_gemv_shapes().len(), 1);
    let qwen = qwen36_35b_a3b();
    assert_eq!(qwen.decode_int8_gemv_shapes().len(), 2);
}

/// AGENTS.md Gotcha 24: `validate_arch` compares float fields with `!=` on
/// `f64` and serde_json's default parser is only accurate to ~1 ULP
/// (exactness is behind its `float_roundtrip` feature), so a value that does
/// not survive a serialize/parse round trip makes its install unloadable.
///
/// Gemma's 1.0 and Qwen's 0.0625 are binary fractions and were never at risk.
/// Mixtral's is `128^-0.5 = 2^-3.5`, which is NOT, so this stopped being a
/// theoretical concern in ROADMAP Phase M2 -- and it is cheaper to assert
/// here than to discover 55 minutes into a streamed install.
#[test]
fn every_baseline_float_survives_the_manifest_round_trip() {
    for arch in turbospark_model_io::all_known_architectures() {
        for (field, value) in [
            ("attentionScale", arch.attention_scale),
            ("ropeTheta", arch.rope_theta),
            ("fullRopeTheta", arch.full_rope_theta),
            ("partialRotaryFactor", arch.partial_rotary_factor),
            ("finalLogitSoftcap", arch.final_logit_softcap),
            ("routedScalingFactor", arch.routed_scaling_factor),
        ] {
            let text = serde_json::to_string(&value).expect("serialize");
            let back: f64 = serde_json::from_str(&text).expect("parse");
            assert_eq!(
                back,
                value,
                "{} {field} = {value} does not round-trip through serde_json (got {back} from {text})",
                arch.family.as_str()
            );
        }
    }
}
