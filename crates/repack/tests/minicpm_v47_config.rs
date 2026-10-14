//! `mlx-community/MiniCPM-V-4.7-35B-A3B-mxfp4`'s published `config.json`
//! (rev 810419dd09762d63aad6da5b6a418c4af8946e60, copied verbatim into
//! `fixtures/`): a MiniCPM-V wrapper over a Qwen 3.6 35B-A3B trunk,
//! quantized with MLX's mxfp4 mode.
//!
//! The headline is one equality against the QwenGdnMoe baseline, with the
//! one field this checkpoint really changes (its vocabulary) set from the
//! file, so a key wired to the wrong field shows (`docs/NEW_MODEL.md`).

use turbospark_repack::{
    config_json_family, has_foreign_vision_tower, parse_gemma4_quantization,
    parse_qwen_gdn_moe_config, QuantMode,
};

const CONFIG: &str = include_str!("fixtures/minicpm_v47_mxfp4_config.json");

#[test]
fn the_trunk_is_qwen36_with_its_own_vocabulary() {
    let root: serde_json::Value = serde_json::from_str(CONFIG).unwrap();
    assert_eq!(root["model_type"], "minicpmv4_7");
    // The unknown root falls through to `text_config.model_type`.
    assert_eq!(
        config_json_family(&root),
        Some(model_io::ModelFamily::QwenGdnMoe)
    );
    let parsed = parse_qwen_gdn_moe_config(CONFIG).expect("config parses");
    let mut expected = model_io::qwen_gdn_moe_35b_a3b();
    expected.vocab_size = 248_144;
    assert_eq!(parsed, expected);
    // Repack never derives a tower from config; this wrapper's is MiniCPM's.
    assert!(!parsed.vision.is_active());
}

#[test]
fn the_wrapper_is_a_foreign_vision_tower_and_qwen36_is_not() {
    let root: serde_json::Value = serde_json::from_str(CONFIG).unwrap();
    assert!(has_foreign_vision_tower(&root));
    let qwen = serde_json::json!({"model_type": "qwen3_5_moe", "text_config": root["text_config"]});
    assert!(!has_foreign_vision_tower(&qwen));
}

#[test]
fn the_quantization_block_is_mxfp4_with_affine_int8_gates() {
    let quant = parse_gemma4_quantization(CONFIG).expect("quantization parses");
    assert_eq!(quant.mode, QuantMode::Mxfp4);
    assert_eq!((quant.default_bits, quant.group_size), (4, 32));
    // 40 layers x (router + shared-expert gate), each affine 8-bit at 64.
    assert_eq!(quant.bits_overrides.len(), 80);
    for l in 0..40 {
        for name in ["gate", "shared_expert_gate"] {
            let base = format!("language_model.model.layers.{l}.mlp.{name}");
            assert_eq!(quant.bits_for(&base), 8, "{base}");
            assert_eq!(quant.group_for(&base), 64, "{base}");
            assert!(!quant.is_mxfp4(&base), "{base}");
        }
    }
    let expert = "language_model.model.layers.0.mlp.switch_mlp.gate_proj";
    assert!(quant.is_mxfp4(expert));
    assert_eq!(quant.group_for(expert), 32);
}

fn quant_block(q: serde_json::Value) -> String {
    serde_json::json!({ "quantization": q }).to_string()
}

#[test]
fn mxfp4_outside_its_one_shape_is_refused() {
    for (bits, group) in [(4, 64), (8, 32)] {
        let json =
            quant_block(serde_json::json!({"mode": "mxfp4", "bits": bits, "group_size": group}));
        let err = parse_gemma4_quantization(&json).expect_err("refused");
        assert!(err.to_string().contains("mxfp4"), "{err}");
    }
}

#[test]
fn an_unknown_mode_is_refused() {
    let json = quant_block(serde_json::json!({"mode": "nvfp4", "bits": 4, "group_size": 16}));
    let err = parse_gemma4_quantization(&json).expect_err("refused");
    assert!(err.to_string().contains("nvfp4"), "{err}");
}

#[test]
fn affine_checkpoints_still_refuse_a_second_group_size() {
    // The widening is mxfp4-only: the same override under an affine default
    // is a group size the affine kernels cannot pair with the default.
    // A supported affine default (2-bit at 128) carrying the exact gate
    // override MiniCPM's mxfp4 checkpoint declares. Asserting the message
    // matters: without the guard this still fails, later, as an unsupported
    // (8, 128) shape.
    let json = quant_block(serde_json::json!({
        "bits": 2, "group_size": 128,
        "language_model.model.layers.0.mlp.gate": {"bits": 8, "group_size": 64}
    }));
    let err = parse_gemma4_quantization(&json).expect_err("refused");
    assert!(err.to_string().contains("differs from the global"), "{err}");
    let json = quant_block(serde_json::json!({
        "bits": 4, "group_size": 64,
        "language_model.model.layers.0.mlp.gate": {"bits": 8, "group_size": 32}
    }));
    let err = parse_gemma4_quantization(&json).expect_err("refused");
    assert!(err.to_string().contains("differs from the global"), "{err}");
}

#[test]
fn an_mxfp4_override_must_be_an_affine_shape_with_kernels() {
    let json = quant_block(serde_json::json!({
        "mode": "mxfp4", "bits": 4, "group_size": 32,
        "language_model.model.layers.0.mlp.gate": {"bits": 8, "group_size": 128}
    }));
    assert!(parse_gemma4_quantization(&json).is_err());
    let json = quant_block(serde_json::json!({
        "mode": "mxfp4", "bits": 4, "group_size": 32,
        "language_model.model.layers.0.mlp.gate": {"bits": 8, "group_size": 64, "mode": "mxfp8"}
    }));
    assert!(parse_gemma4_quantization(&json).is_err());
}
