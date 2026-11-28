//! `mlx-community/MiniCPM-V-4.6-4bit`'s published `config.json` (rev
//! 86cd463d33a946e4481b77e3c10fc63121b60a19, copied verbatim into
//! `fixtures/`): a MiniCPM-V wrapper over a DENSE qwen3.5 trunk
//! (`docs/MINICPMV_VISION_PHASE0.md`, "4.6").

use model_io::{ModelFamily, VisionTowerKind};
use turbospark_repack::{
    classify_for_family, config_json_family, parse_gemma4_quantization,
    parse_qwen_gdn_dense_config, parse_vision_config, quantize_on_install, Gemma4Bucket,
};

const CONFIG: &str = include_str!("fixtures/minicpm_v46_4bit_config.json");

fn mutated(edit: impl FnOnce(&mut serde_json::Value)) -> String {
    let mut v: serde_json::Value = serde_json::from_str(CONFIG).unwrap();
    edit(&mut v);
    v.to_string()
}

#[test]
fn the_trunk_is_a_tied_dense_qwen35_quantized_at_group_64() {
    let root: serde_json::Value = serde_json::from_str(CONFIG).unwrap();
    assert_eq!(root["model_type"], "minicpmv4_6");
    // The unknown root falls through to `text_config.model_type`.
    assert_eq!(config_json_family(&root), Some(ModelFamily::QwenGdnDense));
    let arch = parse_qwen_gdn_dense_config(CONFIG).expect("trunk parses");
    assert_eq!(
        (arch.num_layers, arch.hidden_size, arch.vocab_size),
        (24, 1024, 248_094)
    );
    assert!(arch.tie_word_embeddings);
    assert!(!arch.vision.is_active());
    let quant = parse_gemma4_quantization(CONFIG).expect("quant");
    assert_eq!((quant.default_bits, quant.group_size), (4, 64));
}

#[test]
fn the_tower_parses_to_the_46_kind_with_the_vocabulary_markers() {
    let v = parse_vision_config(CONFIG).expect("tower parses");
    assert_eq!(v.tower, VisionTowerKind::MiniCpmV46);
    assert!(v.tower.is_minicpm_v());
    assert_eq!(
        VisionTowerKind::parse(v.tower.as_str()),
        Some(VisionTowerKind::MiniCpmV46)
    );
    assert_eq!(
        (v.depth, v.hidden_size, v.intermediate_size, v.num_heads),
        (27, 1152, 4304, 16)
    );
    assert_eq!((v.patch_size, v.spatial_merge_size), (14, 4));
    assert_eq!(v.num_position_embeddings, 4900);
    // The merger projects into this trunk, not the 4.7 MoE's 2048.
    assert_eq!(v.out_hidden_size, 1024);
    assert_eq!(v.mrope_section, [11, 11, 10]);
    // `<image>` and `</image>` are not in the file; the other two are.
    assert_eq!(
        (
            v.vision_start_token_id,
            v.vision_end_token_id,
            v.image_token_id,
            v.video_token_id
        ),
        (248078, 248079, 248056, 248057)
    );
}

#[test]
fn the_references_defaults_may_be_spelled_out() {
    for text in [
        mutated(|v| v["downsample_mode"] = "16x".into()),
        mutated(|v| v["drop_vision_last_layer"] = false.into()),
        mutated(|v| v["image_start_id"] = 248078.into()),
        mutated(|v| v["merge_kernel_size"] = serde_json::json!([2, 2])),
    ] {
        assert!(parse_vision_config(&text).is_ok(), "{text}");
    }
}

#[test]
fn a_46_checkpoint_off_the_kind_constants_is_refused() {
    for (what, text) in [
        ("4x", mutated(|v| v["downsample_mode"] = "4x".into())),
        (
            "non-string mode",
            mutated(|v| v["downsample_mode"] = 16.into()),
        ),
        ("insert layer", mutated(|v| v["insert_layer_id"] = 8.into())),
        ("canvas", mutated(|v| v["mrope_mode"] = "canvas".into())),
        (
            "dropped layer",
            mutated(|v| v["drop_vision_last_layer"] = true.into()),
        ),
        ("other marker", mutated(|v| v["image_start_id"] = 7.into())),
        (
            "other vocabulary",
            mutated(|v| v["image_token_id"] = 151655.into()),
        ),
        ("other newline", mutated(|v| v["newline_id"] = 10.into())),
        (
            "exact gelu",
            mutated(|v| v["vision_config"]["hidden_act"] = "gelu".into()),
        ),
        (
            "moe trunk",
            mutated(|v| v["text_config"]["model_type"] = "qwen3_5_moe_text".into()),
        ),
        ("merger times", mutated(|v| v["merger_times"] = 2.into())),
    ] {
        assert!(parse_vision_config(&text).is_err(), "{what} was accepted");
    }
}

#[test]
fn upstream_bf16_is_refused_rather_than_quantized_on_install() {
    let root: serde_json::Value = serde_json::from_str(CONFIG).unwrap();
    let why =
        quantize_on_install::quantize_refusal(ModelFamily::QwenGdnDense, &root).expect("refused");
    assert!(why.contains("minicpmv4_6"), "{why}");
    // The plain dense trunk is still admitted.
    let plain = serde_json::json!({"model_type": "qwen3_5"});
    assert!(quantize_on_install::quantize_refusal(ModelFamily::QwenGdnDense, &plain).is_none());
}

/// The three tower prefixes of the 4-bit repo's header land in the vision
/// bucket under the dense trunk, so a vision install reads them and a text
/// install drops them. Under an unrelated family `merger.` is never ingested.
#[test]
fn the_adapters_are_tower_tensors_for_the_dense_trunk() {
    for name in [
        "vision_tower.encoder.layers.26.mlp.fc2.weight",
        "vit_merger.linear_1.weight",
        "merger.mlp.0.linear_2.bias",
    ] {
        assert!(
            matches!(
                classify_for_family(name, 24, ModelFamily::QwenGdnDense),
                Gemma4Bucket::VisionTower
            ),
            "{name}"
        );
    }
    for name in ["vit_merger.linear_1.weight", "merger.mlp.0.linear_2.bias"] {
        assert!(
            matches!(
                classify_for_family(name, 24, ModelFamily::Gemma4),
                Gemma4Bucket::ExcludedMultimodal
            ),
            "{name}"
        );
    }
}
