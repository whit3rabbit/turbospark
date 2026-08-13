//! `parse_qwen35_config` against the PRODUCTION field values
//! (`prism-ml/Bonsai-27B-mlx-1bit`, ROADMAP's 1-bit entry).
//!
//! Its own target rather than a case in `qwen36_config.rs`, and the fixture
//! is the real `text_config` verbatim for that file's reason: the whole test
//! reduces to one assertion, that the parse equals
//! `model_io::bonsai_27b()` field for field. A shape-only fixture cannot
//! catch a key mapped to the wrong field, because every dimension would be
//! some other made-up number either way.
//!
//! **THE ASSERTION IS SHARPER HERE THAN FOR ANY OTHER FAMILY, and that is
//! the point of the file.** This config and Qwen 3.6's differ in exactly
//! four fields, all of them about the FFN, so a parser that silently fell
//! through to the MoE branch would produce something that still validates
//! structurally. What catches it is the baseline comparison plus the
//! `num_experts` cases below.

use turbospark_repack::{parse_qwen35_config, parse_qwen36_config};

/// 64 layers, gated-DeltaNet everywhere except every 4th
/// (`full_attention_interval = 4`), which reproduces the checkpoint's own
/// `layer_types` list.
fn layer_types() -> Vec<&'static str> {
    (0..64)
        .map(|i| {
            if (i + 1) % 4 == 0 {
                "full_attention"
            } else {
                "linear_attention"
            }
        })
        .collect()
}

/// The production `text_config`, trimmed to the keys the parser reads plus
/// the mrope fields it must ignore and the `mtp_*` ones nothing implements.
fn text_config() -> serde_json::Value {
    serde_json::json!({
        "attn_output_gate": true,
        "full_attention_interval": 4,
        "head_dim": 256,
        "hidden_act": "silu",
        "hidden_size": 5120,
        // The DENSE FFN width. Qwen 3.6's text_config does not carry this
        // key at all; its FFN width lives in
        // `shared_expert_intermediate_size`.
        "intermediate_size": 17408,
        "layer_types": layer_types(),
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 128,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 48,
        "linear_value_head_dim": 128,
        "max_position_embeddings": 262_144,
        "model_type": "qwen3_5_text",
        // Declared and unimplemented; no `mtp` tensor ships in the file.
        "mtp_num_hidden_layers": 1,
        "mtp_use_dedicated_embeddings": false,
        "num_attention_heads": 24,
        "num_hidden_layers": 64,
        "num_key_value_heads": 4,
        "output_gate_type": "swish",
        "partial_rotary_factor": 0.25,
        "rms_norm_eps": 1.0e-6,
        "rope_parameters": {
            "mrope_interleaved": true,
            "mrope_section": [11, 11, 10],
            "partial_rotary_factor": 0.25,
            "rope_theta": 10_000_000,
            "rope_type": "default"
        },
        "tie_word_embeddings": false,
        "vocab_size": 248_320
    })
}

fn config_json() -> String {
    serde_json::json!({
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "language_model_only": false,
        "model_type": "qwen3_5",
        "text_config": text_config(),
        "vision_config": {"depth": 27},
        // The real object, verbatim: two keys and no per-tensor overrides.
        "quantization": {"group_size": 128, "bits": 1}
    })
    .to_string()
}

#[test]
fn parses_the_production_config_into_the_pinned_baseline() {
    let arch = parse_qwen35_config(&config_json()).expect("config parses");
    assert_eq!(
        arch,
        model_io::bonsai_27b(),
        "parsed config does not match the pinned Bonsai-27B baseline"
    );
}

/// The dense FFN width comes from `intermediate_size`, and the three MoE
/// fields are zero rather than inherited.
///
/// Stated separately from the baseline comparison because it is the one
/// place the two Qwen parsers part company, and because a `0` that came from
/// a missing key would look identical to a `0` that was decided.
#[test]
fn the_dense_ffn_width_is_read_and_the_moe_fields_are_zero() {
    let arch = parse_qwen35_config(&config_json()).expect("config parses");
    assert_eq!(arch.intermediate_size, 17408);
    assert_eq!(arch.moe_intermediate_size, 0);
    assert_eq!(arch.num_experts, 0);
    assert_eq!(arch.top_k_experts, 0);
    assert!(!arch.shared_expert_gated);
}

/// A config that declares BOTH a dense FFN and experts is refused rather
/// than half-read.
///
/// Half-reading it yields an `ArchConfig` whose FFN width and expert count
/// disagree, which validates structurally and then dispatches the wrong
/// branch -- the failure mode this whole family split exists to avoid, since
/// `qwen3_5` and `qwen3_5_moe` are one suffix apart.
#[test]
fn a_dense_config_that_also_declares_experts_is_refused() {
    let mut root: serde_json::Value = serde_json::from_str(&config_json()).unwrap();
    root["text_config"]["num_experts"] = serde_json::json!(256);
    let err = parse_qwen35_config(&root.to_string()).expect_err("must be refused");
    let text = format!("{err:?}");
    assert!(text.contains("num_experts"), "{text}");
    assert!(text.contains("qwen3_5_moe"), "{text}");

    // `num_experts: 0` is not a contradiction and still parses: it is what
    // a dense config would say if it said anything.
    root["text_config"]["num_experts"] = serde_json::json!(0);
    assert!(parse_qwen35_config(&root.to_string()).is_ok());
}

/// The two parsers refuse each other's configs, which is what
/// `refuse_foreign_config` is for.
///
/// Without it every parser hardcodes `family:` on the way out, so a Bonsai
/// config parsed by the MoE parser would yield an `ArchConfig` labelled
/// `qwen36` -- with a 5120 hidden size it never had and a shared-expert
/// width read from a key that is not there.
#[test]
fn the_two_qwen_parsers_refuse_each_others_configs() {
    assert!(parse_qwen36_config(&config_json()).is_err());

    let moe = serde_json::json!({
        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        "model_type": "qwen3_5_moe",
        "text_config": text_config(),
    })
    .to_string();
    assert!(parse_qwen35_config(&moe).is_err());
}

/// The layer graph: 64 entries, 48 linear to 16 full, layer 0 linear.
#[test]
fn the_layer_mask_is_48_linear_to_16_full() {
    let arch = parse_qwen35_config(&config_json()).expect("config parses");
    assert_eq!(arch.full_attention_layer_mask.len(), 64);
    assert_eq!(
        arch.full_attention_layer_mask
            .iter()
            .filter(|&&k| k == 2)
            .count(),
        48
    );
    assert!(arch.layer_is_linear(0));
    assert!(arch.layer_is_full(3));
}

/// `attention_scale` is `head_dim ** -0.5` and, at 256, exactly 1/16 -- a
/// binary fraction, so it survives the serde_json round trip AGENTS.md
/// Gotcha 24 warns about.
#[test]
fn attention_scale_is_the_reference_head_dim_power() {
    let arch = parse_qwen35_config(&config_json()).expect("config parses");
    assert_eq!(arch.attention_scale, 0.0625);
    let back: f64 = serde_json::from_str(&serde_json::to_string(&arch.attention_scale).unwrap())
        .expect("round trips");
    assert_eq!(back, arch.attention_scale);
}
