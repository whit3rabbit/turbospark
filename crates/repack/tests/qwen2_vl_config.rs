//! `qwen2_vl` config intake, offline (`docs/QWEN2VL_VISION_PHASE0.md`).
//!
//! The fixture is `mlx-community/Qwen2-VL-2B-Instruct-4bit`'s own
//! `config.json` at the revision the Phase 0 doc read (@
//! `01af461cdb9574acc09084a0ef94e216e142b085`), so a parser that agrees with
//! this text agrees with a published file rather than with a guess. NOTE the
//! key vocabulary: the tower width is `embed_dim`, the MLP width is the
//! float relation `mlp_ratio * embed_dim`, there is no `out_hidden_size`
//! (`vision_config.hidden_size` names the TRUNK width on this family), and
//! there are no window keys at all.

use model_io::{ModelFamily, VisionTowerKind};
use turbospark_repack::{
    config_json_family, parse_qwen2_config, parse_vision_config, VISION_QWEN2_VL_BLOCK_ROLES,
    VISION_QWEN2_VL_RESIDENT_TENSORS,
};

const CONFIG_2B: &str = r#"{
  "architectures": ["Qwen2VLForConditionalGeneration"],
  "hidden_act": "silu",
  "hidden_size": 1536,
  "image_token_id": 151655,
  "intermediate_size": 8960,
  "max_position_embeddings": 32768,
  "model_type": "qwen2_vl",
  "num_attention_heads": 12,
  "num_hidden_layers": 28,
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
    "embed_dim": 1280,
    "mlp_ratio": 4,
    "num_heads": 16,
    "in_chans": 3,
    "hidden_size": 1536,
    "patch_size": 14,
    "spatial_merge_size": 2,
    "spatial_patch_size": 14,
    "temporal_patch_size": 2
  },
  "vision_end_token_id": 151653,
  "vision_start_token_id": 151652,
  "vision_token_id": 151654,
  "vocab_size": 151936
}"#;

fn mutated(edit: impl FnOnce(&mut serde_json::Value)) -> String {
    let mut v: serde_json::Value = serde_json::from_str(CONFIG_2B).unwrap();
    edit(&mut v);
    v.to_string()
}

#[test]
fn the_published_config_parses_to_the_qwen2_tower() {
    let v = parse_vision_config(CONFIG_2B).expect("parses");
    assert_eq!(v.tower, VisionTowerKind::Qwen2Vl);
    assert!(v.is_active());
    // embed_dim is the tower's own width; the MLP width is mlp_ratio x it.
    assert_eq!(
        (v.depth, v.hidden_size, v.intermediate_size, v.num_heads),
        (32, 1280, 5120, 16)
    );
    assert_eq!(v.head_dim(), 80);
    assert_eq!(
        (v.patch_size, v.temporal_patch_size, v.in_channels),
        (14, 2, 3)
    );
    // vision_config.hidden_size names the TRUNK width on this family, and
    // the merger writes into it.
    assert_eq!((v.spatial_merge_size, v.out_hidden_size), (2, 1536));
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
    assert_eq!(v.mrope_section.iter().sum::<i64>(), 1536 / 12 / 2);
}

#[test]
fn the_wrapper_resolves_to_the_qwen2_family_and_its_trunk_parses_text_only() {
    let root: serde_json::Value = serde_json::from_str(CONFIG_2B).unwrap();
    assert_eq!(config_json_family(&root), Some(ModelFamily::Qwen2Dense));
    // The trunk parser reads the flat text keys and leaves the tower to the
    // vision parser: an install is vision-active only when the BYTES say so.
    let arch = parse_qwen2_config(CONFIG_2B).expect("trunk parses");
    assert_eq!(arch.family, ModelFamily::Qwen2Dense);
    assert_eq!((arch.hidden_size, arch.num_layers), (1536, 28));
    assert!(!arch.vision.is_active());
}

#[test]
fn a_foreign_model_type_is_refused_by_name() {
    let text = mutated(|v| v["model_type"] = "qwen2_5_vl".into());
    let err = parse_vision_config(&text).unwrap_err().to_string();
    // A 2.5 root string on a qwen2_vl-shaped tower must reach the 2.5
    // READER, not parse as a qwen2_vl tower: that reader wants the 2.5 key
    // vocabulary (`intermediate_size`, `window_size`), and refuses this file
    // for the first key it is missing.
    assert!(err.contains("missing vision_config"), "{err}");
    let text = mutated(|v| v["model_type"] = "qwen3_vl".into());
    let err = parse_vision_config(&text).unwrap_err().to_string();
    // The qwen3 reader refuses the tower-shaped file by name: its key
    // vocabulary wants `intermediate_size` (the qwen2_vl file carries
    // `mlp_ratio` instead).
    assert!(
        err.contains("missing vision_config.intermediate_size"),
        "{err}"
    );
}

#[test]
fn a_deepstack_key_is_refused_by_name() {
    let text = mutated(|v| {
        v["vision_config"]["deepstack_visual_indexes"] = serde_json::json!([5, 11, 17]);
    });
    let err = parse_vision_config(&text).unwrap_err().to_string();
    assert!(err.contains("deepstack_visual_indexes"), "{err}");
    assert!(err.contains("qwen3_vl"), "{err}");
}

#[test]
fn window_keys_are_refused_as_25_tower_keys() {
    for (key, value) in [
        ("window_size", serde_json::json!(112)),
        ("fullatt_block_indexes", serde_json::json!([7, 15, 23, 31])),
    ] {
        let text = mutated(|v| v["vision_config"][key] = value);
        let err = parse_vision_config(&text).unwrap_err().to_string();
        assert!(err.contains(key), "{err}");
    }
}

#[test]
fn the_mrope_section_is_read_from_both_spellings() {
    // The published conversion's spelling: flat root `rope_scaling`.
    let v = parse_vision_config(CONFIG_2B).expect("parses");
    assert_eq!(v.mrope_section, [16, 24, 24]);
    // The HF-native wrapper spelling: `text_config.rope_scaling`.
    let text = mutated(|v| {
        let scaling = v["rope_scaling"].clone();
        v.as_object_mut().unwrap().remove("rope_scaling");
        v["text_config"]["rope_scaling"] = scaling;
    });
    let v = parse_vision_config(&text).expect("wrapper spelling parses");
    assert_eq!(v.mrope_section, [16, 24, 24]);
    // The `rope_parameters` spelling some conversions use.
    let text = mutated(|v| {
        let section = v["rope_scaling"]["mrope_section"].clone();
        let params = serde_json::json!({"mrope_section": section});
        v.as_object_mut().unwrap().remove("rope_scaling");
        v["text_config"]["rope_parameters"] = params;
    });
    let v = parse_vision_config(&text).expect("rope_parameters spelling parses");
    assert_eq!(v.mrope_section, [16, 24, 24]);
}

#[test]
fn a_trunk_without_the_mrope_split_is_refused() {
    let text = mutated(|v| {
        v.as_object_mut().unwrap().remove("rope_scaling");
    });
    let err = parse_vision_config(&text).unwrap_err().to_string();
    assert!(err.contains("mrope_section"), "{err}");
}

#[test]
fn a_foreign_mrope_type_is_refused() {
    let text = mutated(|v| v["rope_scaling"]["type"] = "yarn".into());
    let err = parse_vision_config(&text).unwrap_err().to_string();
    assert!(err.contains("yarn"), "{err}");
}

#[test]
fn a_fractional_or_nonpositive_mlp_width_is_refused() {
    // 4.5 x 1280 is 5760, a whole width the parser must ACCEPT (the guard is
    // the product, not the ratio); a ratio whose product is fractional is
    // refused.
    let text = mutated(|v| v["vision_config"]["mlp_ratio"] = serde_json::json!(4.5));
    let v = parse_vision_config(&text).expect("a whole product is a width");
    assert_eq!(v.intermediate_size, 5760);
    let text = mutated(|v| v["vision_config"]["mlp_ratio"] = serde_json::json!(4.0001));
    let err = parse_vision_config(&text).unwrap_err().to_string();
    assert!(err.contains("mlp_ratio"), "{err}");
    let text = mutated(|v| v["vision_config"]["mlp_ratio"] = 0.into());
    assert!(parse_vision_config(&text).is_err());
}

#[test]
fn a_tower_width_disagreeing_with_the_trunk_is_refused() {
    let text = mutated(|v| v["vision_config"]["hidden_size"] = 3584.into());
    let err = parse_vision_config(&text).unwrap_err().to_string();
    assert!(err.contains("trunk hidden_size"), "{err}");
}

#[test]
fn an_unsupported_depth_or_geometry_is_refused() {
    let text = mutated(|v| v["vision_config"]["depth"] = 27.into());
    let err = parse_vision_config(&text).unwrap_err().to_string();
    assert!(err.contains("depth 27"), "{err}");
    let text = mutated(|v| v["vision_config"]["patch_size"] = 16.into());
    let err = parse_vision_config(&text).unwrap_err().to_string();
    assert!(err.contains("14 x 2 x 2 x 3"), "{err}");
    let text = mutated(|v| v["vision_config"]["num_heads"] = 18.into());
    let err = parse_vision_config(&text).unwrap_err().to_string();
    assert!(err.contains("divisible by num_heads 18"), "{err}");
}

#[test]
fn a_declared_eps_or_act_that_disagrees_with_the_kind_is_refused() {
    // The published file declares neither; when a checkpoint does, they must
    // match what the runtime encodes.
    let text = mutated(|v| v["vision_config"]["layer_norm_eps"] = serde_json::json!(1e-5));
    let err = parse_vision_config(&text).unwrap_err().to_string();
    assert!(err.contains("layer_norm_eps"), "{err}");
    let text = mutated(|v| v["vision_config"]["layer_norm_eps"] = serde_json::json!(1e-6));
    assert!(parse_vision_config(&text).is_ok());
    // transformers' own spelling, written by every config it re-saves.
    let text = mutated(|v| v["vision_config"]["hidden_act"] = "quick_gelu".into());
    assert!(parse_vision_config(&text).is_ok());
    // "gelu" is the erf form to transformers, a different function.
    for act in ["gelu", "relu"] {
        let text = mutated(|v| v["vision_config"]["hidden_act"] = act.into());
        let err = parse_vision_config(&text).unwrap_err().to_string();
        assert!(err.contains("hidden_act"), "{err}");
    }
}

#[test]
fn the_role_tables_cover_the_published_tower() {
    // 12 per block x 32 blocks + 7 non-block = 391.
    assert_eq!(
        VISION_QWEN2_VL_BLOCK_ROLES.len() * 32 + VISION_QWEN2_VL_RESIDENT_TENSORS.len(),
        391
    );
    let mut suffixes: Vec<&str> = VISION_QWEN2_VL_BLOCK_ROLES
        .iter()
        .map(|(_, s)| *s)
        .collect();
    suffixes.sort_unstable();
    suffixes.dedup();
    assert_eq!(suffixes.len(), 12, "a role suffix is listed twice");
    // LayerNorm blocks carry norm BIASES (unlike the 2.5 RMSNorm tower), and
    // the MLP is the plain fc1/fc2 pair (no gate/up split).
    assert!(suffixes.contains(&"norm1.bias"));
    assert!(suffixes.contains(&"mlp.fc1.weight"));
    assert!(suffixes.contains(&"mlp.fc2.weight"));
    assert!(!suffixes.iter().any(|s| s.contains("gate_proj")));
    // The merger's ln_q carries a bias here.
    assert!(VISION_QWEN2_VL_RESIDENT_TENSORS.contains(&"merger.ln_q.bias"));
    assert!(VISION_QWEN2_VL_RESIDENT_TENSORS.contains(&"patch_embed.proj.weight"));
    assert!(!VISION_QWEN2_VL_RESIDENT_TENSORS.contains(&"patch_embed.proj.bias"));
}
