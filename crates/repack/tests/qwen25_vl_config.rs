//! `qwen2_5_vl` config intake, offline (`docs/QWEN25_VL_VISION_PHASE0.md`).
//!
//! The fixture is `mlx-community/Qwen2.5-VL-3B-Instruct-4bit`'s own
//! `config.json` at the revision the Phase 0 doc read, so a parser that agrees
//! with this text agrees with a published file rather than with a guess.

use model_io::{ModelFamily, VisionTowerKind};
use turbospark_repack::{
    config_json_family, parse_qwen2_config, parse_vision_config, VISION_QWEN25_BLOCK_ROLES,
    VISION_QWEN25_RESIDENT_TENSORS,
};

const CONFIG_3B: &str = r#"{
  "architectures": ["Qwen2_5_VLForConditionalGeneration"],
  "hidden_act": "silu",
  "hidden_size": 2048,
  "image_token_id": 151655,
  "intermediate_size": 11008,
  "max_position_embeddings": 128000,
  "model_type": "qwen2_5_vl",
  "num_attention_heads": 16,
  "num_hidden_layers": 36,
  "num_key_value_heads": 2,
  "quantization": {"group_size": 64, "bits": 4},
  "rms_norm_eps": 1e-06,
  "rope_scaling": {"type": "mrope", "mrope_section": [16, 24, 24]},
  "rope_theta": 1000000.0,
  "sliding_window": 32768,
  "tie_word_embeddings": true,
  "use_sliding_window": false,
  "video_token_id": 151656,
  "vision_config": {
    "depth": 32,
    "hidden_act": "silu",
    "hidden_size": 1280,
    "intermediate_size": 3420,
    "num_heads": 16,
    "in_chans": 3,
    "out_hidden_size": 2048,
    "patch_size": 14,
    "spatial_merge_size": 2,
    "spatial_patch_size": 14,
    "window_size": 112,
    "fullatt_block_indexes": [7, 15, 23, 31],
    "tokens_per_second": 2,
    "temporal_patch_size": 2
  },
  "vision_end_token_id": 151653,
  "vision_start_token_id": 151652,
  "vision_token_id": 151654,
  "vocab_size": 151936
}"#;

fn mutated(edit: impl FnOnce(&mut serde_json::Value)) -> String {
    let mut v: serde_json::Value = serde_json::from_str(CONFIG_3B).unwrap();
    edit(&mut v);
    v.to_string()
}

#[test]
fn the_published_config_parses_to_the_qwen25_tower() {
    let v = parse_vision_config(CONFIG_3B).expect("parses");
    assert_eq!(v.tower, VisionTowerKind::Qwen25Vl);
    assert!(v.is_active());
    assert_eq!(
        (v.depth, v.hidden_size, v.intermediate_size, v.num_heads),
        (32, 1280, 3420, 16)
    );
    assert_eq!(
        (v.patch_size, v.temporal_patch_size, v.in_channels),
        (14, 2, 3)
    );
    assert_eq!((v.spatial_merge_size, v.out_hidden_size), (2, 2048));
    assert_eq!(v.head_dim(), 80);
    // No learned position table; positions come from rope alone.
    assert_eq!(v.num_position_embeddings, 0);
    assert_eq!(v.mrope_section, [16, 24, 24]);
    assert!(v.deepstack_visual_indexes.is_empty());
    assert_eq!(
        (
            v.vision_start_token_id,
            v.vision_end_token_id,
            v.image_token_id,
            v.video_token_id
        ),
        (151652, 151653, 151655, 151656)
    );
    // The mrope split covers the trunk's rotary pairs: 16 + 24 + 24 = 128 / 2.
    assert_eq!(v.mrope_section.iter().sum::<i64>(), 2048 / 16 / 2);
}

#[test]
fn the_wrapper_resolves_to_the_qwen2_family_and_its_trunk_parses_text_only() {
    let root: serde_json::Value = serde_json::from_str(CONFIG_3B).unwrap();
    assert_eq!(config_json_family(&root), Some(ModelFamily::Qwen2Dense));
    // The trunk parser reads the flat text keys and leaves the tower to the
    // vision parser: an install is vision-active only when the BYTES say so.
    let arch = parse_qwen2_config(CONFIG_3B).expect("trunk parses");
    assert_eq!(arch.family, ModelFamily::Qwen2Dense);
    assert_eq!((arch.hidden_size, arch.num_layers), (2048, 36));
    assert!(!arch.vision.is_active());
}

#[test]
fn a_window_size_other_than_112_is_refused() {
    let text = mutated(|v| v["vision_config"]["window_size"] = 224.into());
    let err = parse_vision_config(&text).unwrap_err().to_string();
    assert!(err.contains("window_size 224"), "{err}");
}

#[test]
fn a_different_full_attention_pattern_is_refused() {
    let text = mutated(|v| v["vision_config"]["fullatt_block_indexes"] = serde_json::json!([3, 7]));
    let err = parse_vision_config(&text).unwrap_err().to_string();
    assert!(err.contains("fullatt_block_indexes"), "{err}");
}

#[test]
fn an_unsupported_depth_is_refused() {
    let text = mutated(|v| v["vision_config"]["depth"] = 24.into());
    let err = parse_vision_config(&text).unwrap_err().to_string();
    assert!(err.contains("depth 24"), "{err}");
}

#[test]
fn a_tower_without_the_mrope_split_is_refused() {
    let text = mutated(|v| {
        v["rope_scaling"]
            .as_object_mut()
            .unwrap()
            .remove("mrope_section");
    });
    let err = parse_vision_config(&text).unwrap_err().to_string();
    assert!(err.contains("mrope_section"), "{err}");
}

#[test]
fn a_trunk_width_the_merger_cannot_write_into_is_refused() {
    let text = mutated(|v| v["hidden_size"] = 3584.into());
    let err = parse_vision_config(&text).unwrap_err().to_string();
    assert!(err.contains("trunk hidden_size"), "{err}");
}

#[test]
fn the_role_tables_cover_the_published_tower() {
    // 12 per block x 32 blocks + 6 non-block = 390, the count the published
    // 3B index carries under vision_tower.
    assert_eq!(
        VISION_QWEN25_BLOCK_ROLES.len() * 32 + VISION_QWEN25_RESIDENT_TENSORS.len(),
        390
    );
    let mut suffixes: Vec<&str> = VISION_QWEN25_BLOCK_ROLES.iter().map(|(_, s)| *s).collect();
    suffixes.sort_unstable();
    suffixes.dedup();
    assert_eq!(suffixes.len(), 12, "a role suffix is listed twice");
    // RMSNorm weights have no bias tensor; the gated MLP has three projections.
    assert!(!suffixes.iter().any(|s| s.ends_with("norm1.bias")));
    assert!(suffixes.contains(&"mlp.gate_proj.weight"));
}
