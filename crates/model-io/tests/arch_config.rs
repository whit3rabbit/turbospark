//! Tests for the architecture baselines and their derived layer/shape
//! helpers.

use turbospark_model_io::{
    deepseek_v4_flash_284b_a13b, gemma4_26b_a4b, known_architecture, qwen2_5_7b, qwen3_vl_4b,
    qwen_gdn_dense_27b, qwen_gdn_moe_35b_a3b, ModelFamily,
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
    let arch = qwen_gdn_moe_35b_a3b();
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
    assert_eq!(
        known_architecture(ModelFamily::QwenGdnMoe),
        qwen_gdn_moe_35b_a3b()
    );
    assert_eq!(
        known_architecture(ModelFamily::DeepseekV4Flash),
        deepseek_v4_flash_284b_a13b()
    );
    assert_eq!(
        known_architecture(ModelFamily::QwenGdnDense),
        qwen_gdn_dense_27b()
    );
    assert_eq!(known_architecture(ModelFamily::Qwen2Dense), qwen2_5_7b());
    assert_eq!(known_architecture(ModelFamily::Qwen3Vl), qwen3_vl_4b());
}

#[test]
fn qwen2_baseline_carries_the_dense_qwen2_contract() {
    let arch = qwen2_5_7b();
    assert_eq!(arch.family, ModelFamily::Qwen2Dense);
    assert_eq!(arch.hidden_size, 3584);
    assert_eq!(arch.intermediate_size, 18944);
    assert_eq!(arch.num_heads, 28);
    assert_eq!(arch.num_kv_heads, 4);
    assert_eq!(arch.num_layers, 28);
    assert_eq!(arch.vocab_size, 152_064);
    assert_eq!(arch.full_attention_layer_mask, vec![1; 28]);
    assert!(!arch.rope_neox_subdim);
    assert!(!arch.attn_output_gate);
    assert_eq!(arch.attention_scale, 0.088_388_347_648_318_45);
}

/// The `qwen3_5` layer graph: 64 layers, full attention on every 4th, and so
/// 48 gated-DeltaNet layers to 16 full-attention ones.
///
/// The counts are the check rather than the pattern, because they are what
/// each checkpoint's own tensor inventory independently reports: 48 of each
/// `linear_attn.in_proj_*` against 16 of each `self_attn.{q,k,v,o}_proj`.
/// BOTH published checkpoints report exactly that, which is one of the
/// observations that says they share this baseline.
#[test]
fn qwen_gdn_dense_layer_mask_is_48_linear_to_16_full() {
    let arch = qwen_gdn_dense_27b();
    assert_eq!(arch.full_attention_layer_mask.len(), 64);
    assert_eq!(
        arch.full_attention_layer_mask
            .iter()
            .filter(|&&k| k == 1)
            .count(),
        16
    );
    assert_eq!(
        arch.full_attention_layer_mask
            .iter()
            .filter(|&&k| k == 2)
            .count(),
        48
    );
    // Every 4th, and layer 0 is linear.
    assert_eq!(arch.full_attention_layer_mask[0], 2);
    assert_eq!(arch.full_attention_layer_mask[3], 1);
    assert_eq!(arch.full_attention_layer_mask[63], 1);
}

/// **The claim the `QwenGdnDense` family rests on, asserted rather than commented:
/// every BEHAVIOURAL field is Qwen 3.6's and the shape fields differ.**
///
/// That is what makes it a variant sharing `families/moe/`'s flow rather
/// than a sixth flow, and it is the same relationship dense Mistral has to
/// Mixtral. If a future edit moves one of these apart, the shared flow stops
/// being correct and this is where it should be noticed -- not in a
/// coherence smoke, which cannot see a behavioural field at all.
#[test]
fn qwen_gdn_dense_shares_the_moe_flows_behaviour_and_differs_in_shape() {
    let dense = qwen_gdn_dense_27b();
    let moe = qwen_gdn_moe_35b_a3b();

    // Behavioural: identical.
    assert_eq!(dense.attn_output_gate, moe.attn_output_gate);
    assert_eq!(dense.attention_scale, moe.attention_scale);
    assert_eq!(dense.rope_neox_subdim, moe.rope_neox_subdim);
    assert_eq!(dense.ffn_sandwich_norms, moe.ffn_sandwich_norms);
    assert_eq!(dense.router_scaled, moe.router_scaled);
    assert_eq!(
        dense.embedding_scaled_by_sqrt_hidden,
        moe.embedding_scaled_by_sqrt_hidden
    );
    assert_eq!(dense.attention_k_eq_v, moe.attention_k_eq_v);
    assert_eq!(dense.tie_word_embeddings, moe.tie_word_embeddings);
    assert_eq!(dense.hidden_activation, moe.hidden_activation);
    assert_eq!(dense.final_logit_softcap, moe.final_logit_softcap);
    assert_eq!(dense.sliding_window, moe.sliding_window);
    assert_eq!(dense.rope_theta, moe.rope_theta);
    assert_eq!(dense.partial_rotary_factor, moe.partial_rotary_factor);
    assert_eq!(dense.head_dim, moe.head_dim);
    assert_eq!(dense.vocab_size, moe.vocab_size);
    assert_eq!(
        dense.linear_attention.key_head_dim,
        moe.linear_attention.key_head_dim
    );
    assert_eq!(
        dense.linear_attention.conv_kernel_size,
        moe.linear_attention.conv_kernel_size
    );

    // Shape: different, and the DENSE difference is the one that needs a
    // branch rather than just a baseline.
    assert_ne!(dense.hidden_size, moe.hidden_size);
    assert_ne!(dense.num_layers, moe.num_layers);
    assert_ne!(dense.num_heads, moe.num_heads);
    assert_ne!(
        dense.linear_attention.num_v_heads,
        moe.linear_attention.num_v_heads
    );
    assert_eq!(dense.num_experts, 0, "both checkpoints are dense");
    assert_eq!(dense.top_k_experts, 0);
    assert!(moe.num_experts > 0, "the comparison is not vacuous");
    // ...and `shared_expert_gated` differs BECAUSE it is dense, which is the
    // one behavioural field that legitimately parts company.
    assert!(moe.shared_expert_gated && !dense.shared_expert_gated);
}

#[test]
fn model_family_round_trips_through_its_string_form() {
    for family in [
        ModelFamily::Gemma4,
        ModelFamily::QwenGdnMoe,
        ModelFamily::DeepseekV4Flash,
    ] {
        assert_eq!(ModelFamily::parse(family.as_str()), Some(family));
    }
    assert_eq!(ModelFamily::parse("unknown"), None);
}

/// **The wire strings are an ON-DISK FORMAT, pinned here as literals so a
/// rename cannot change them by accident.**
///
/// Every `.gturbo` install records `as_str()` in its `manifest.json` and
/// `parse()` reads it back at load, so changing one of these does not
/// rename a concept -- it makes every existing install of that family
/// unloadable, with an "unknown family" error pointing nowhere near the
/// commit that caused it.
///
/// TWO OF THESE DELIBERATELY NO LONGER MATCH THEIR VARIANT NAME. The two
/// gated-DeltaNet families were called `Qwen35` and `Qwen36` until
/// 2026-08-15, named after checkpoint versions that had already drifted
/// (upstream keeps `model_type: qwen3_5` stable across the 3.5, 3.6 and 3.8
/// releases, and `Qwen36` matched `qwen3_5_moe` anyway). The Rust names were
/// fixed and the strings were not, because only one of the two is free.
/// If this test fails, do not update the literals -- restore the strings.
#[test]
fn the_on_disk_family_strings_are_frozen() {
    for (family, expected) in [
        (ModelFamily::Gemma4, "gemma4"),
        (ModelFamily::QwenGdnMoe, "qwen36"),
        (ModelFamily::DeepseekV4Flash, "deepseekV4Flash"),
        (ModelFamily::Llama, "llama"),
        (ModelFamily::Qwen3Moe, "qwen3moe"),
        (ModelFamily::GptOss, "gptOss"),
        (ModelFamily::QwenGdnDense, "qwen35"),
        // The FIFTEENTH family, and new enough that the string is free to
        // match the HF `model_type` -- pinned here from birth so it stays
        // free: the first install written with it makes it a format
        // constant.
        (ModelFamily::Qwen3Vl, "qwen3_vl"),
    ] {
        assert_eq!(
            family.as_str(),
            expected,
            "{family:?}'s on-disk string changed; every install carrying the old \
             one becomes unloadable"
        );
        assert_eq!(ModelFamily::parse(expected), Some(family));
    }
}

#[test]
fn decode_int4_gemv_shapes_nonempty_for_every_baseline() {
    for arch in [
        gemma4_26b_a4b(),
        qwen_gdn_moe_35b_a3b(),
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
    let moe = qwen_gdn_moe_35b_a3b();
    assert_eq!(moe.decode_int8_gemv_shapes().len(), 2);
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
