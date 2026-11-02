//! `muse_glimmer` vision intake, offline (`docs/MUSEGLIMMER_VISION_PHASE0.md`).
//!
//! The fixture mirrors `mlx-community/Muse-Glimmer-30B-4bit`'s `config.json` at
//! revision `3e7677d7a40d348a3daba263a2b1c0aa41910710`, with the 50-entry
//! `layer_types` list generated from the published rule so the file stays
//! readable.

use model_io::VisionTowerKind;
use serde_json::json;
use turbospark_repack::{parse_vision_config, MUSE_BLOCK_SOURCES, MUSE_RESIDENT_TENSORS};

fn config() -> serde_json::Value {
    let layer_types: Vec<&str> = (0..50)
        .map(|i| {
            if (i + 1) % 4 == 0 || i == 49 {
                "full_attention"
            } else {
                "window_attention"
            }
        })
        .collect();
    json!({
        "model_type": "muse_glimmer",
        "image_token_id": 200092,
        "video_token_id": 200091,
        "out_hidden_size": 6144,
        "projector_hidden_size": 4096,
        "text_config": {"hidden_size": 6656},
        "vision_config": {
            "hidden_act": "gelu",
            "hidden_size": 1536,
            "intermediate_size": 8960,
            "layer_norm_eps": 1e-5,
            "layer_types": layer_types,
            "max_position_embeddings": 1024,
            "merge_size": 2,
            "num_attention_heads": 16,
            "num_hidden_layers": 50,
            "patch_size": 14,
            "patch_temporal": 2,
            "pos_emb_height": 32,
            "pos_emb_width": 32,
            "rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"}
        }
    })
}

fn parse(v: &serde_json::Value) -> Result<model_io::VisionConfig, String> {
    parse_vision_config(&v.to_string()).map_err(|e| e.to_string())
}

#[test]
fn the_published_config_parses_to_the_muse_tower() {
    let v = parse(&config()).expect("parses");
    assert_eq!(v.tower, VisionTowerKind::MuseGlimmer);
    assert_eq!(
        (v.depth, v.hidden_size, v.intermediate_size, v.num_heads),
        (50, 1536, 8960, 16)
    );
    assert_eq!(v.head_dim(), 96);
    assert_eq!(
        (v.patch_size, v.temporal_patch_size, v.in_channels),
        (14, 2, 3)
    );
    assert_eq!(v.spatial_merge_size, 2);
    assert_eq!(v.num_position_embeddings, 1024);
    // The projection writes into the TRUNK's residual stream, not the root's
    // 6144 (the adapter's input width).
    assert_eq!(v.out_hidden_size, 6656);
    assert_eq!(v.mrope_section, [0, 0, 0]);
    // The markers the config does not name come from the kind.
    assert_eq!(
        (v.vision_start_token_id, v.vision_end_token_id),
        (200080, 200081)
    );
    assert_eq!((v.image_token_id, v.video_token_id), (200092, 200091));
}

#[test]
fn a_window_of_another_size_is_refused() {
    let mut c = config();
    c["vision_config"]["pos_emb_height"] = 16.into();
    c["vision_config"]["pos_emb_width"] = 16.into();
    c["vision_config"]["max_position_embeddings"] = 256.into();
    assert!(parse(&c).unwrap_err().contains("32x32"));
}

#[test]
fn a_layer_pattern_that_moves_one_block_is_refused() {
    let mut c = config();
    c["vision_config"]["layer_types"][4] = "full_attention".into();
    assert!(parse(&c).unwrap_err().contains("layer_types"));
}

#[test]
fn another_norm_epsilon_rope_theta_or_activation_is_refused() {
    let mut c = config();
    c["vision_config"]["layer_norm_eps"] = 1e-6.into();
    assert!(parse(&c).unwrap_err().contains("layer_norm_eps"));
    let mut c = config();
    c["vision_config"]["rope_parameters"]["rope_theta"] = 1e6.into();
    assert!(parse(&c).unwrap_err().contains("rope_theta"));
    let mut c = config();
    c["vision_config"]["hidden_act"] = "gelu_pytorch_tanh".into();
    assert!(parse(&c).unwrap_err().contains("hidden_act"));
}

#[test]
fn an_unsupported_depth_or_a_mismatched_adapter_width_is_refused() {
    let mut c = config();
    c["vision_config"]["num_hidden_layers"] = 24.into();
    assert!(parse(&c).unwrap_err().contains("depth 24"));
    let mut c = config();
    c["out_hidden_size"] = 4608.into();
    assert!(parse(&c).unwrap_err().contains("out_hidden_size"));
}

#[test]
fn the_source_tables_cover_the_published_inventory() {
    // 16 source tensors per block x 50 + the 9 residents = the 809 bases the
    // index lists (806 under vision_tower, 2 adapter, 1 projection).
    let per_block: usize = MUSE_BLOCK_SOURCES.iter().map(|s| s.len()).sum();
    assert_eq!(per_block * 50 + MUSE_RESIDENT_TENSORS.len(), 809);
    // Every resident is under one of the three source prefixes.
    for (source, _) in MUSE_RESIDENT_TENSORS {
        assert!(
            ["vision_tower.", "vision_adapter.", "vision_projection."]
                .iter()
                .any(|p| source.starts_with(p)),
            "{source}"
        );
    }
}
