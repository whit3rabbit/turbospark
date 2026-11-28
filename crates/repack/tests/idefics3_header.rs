//! Idefics3 / SmolVLM tensor namespace normalization and bucketing, over the
//! names `mlx-community/SmolVLM-256M-Instruct-4bit` publishes.

use std::collections::BTreeMap;

use model_io::ModelFamily;
use turbospark_repack::{
    canonicalize_idefics3_header, classify_for_family, Gemma4Bucket, SafetensorsHeader, TensorInfo,
};

fn header(names: &[&str]) -> SafetensorsHeader {
    SafetensorsHeader {
        tensors: names
            .iter()
            .map(|name| {
                (
                    (*name).to_string(),
                    TensorInfo {
                        dtype: "BF16".into(),
                        shape: vec![1],
                        data_offsets: (0, 2),
                    },
                )
            })
            .collect::<BTreeMap<_, _>>(),
        metadata: None,
        header_len: 0,
    }
}

#[test]
fn the_trunk_moves_under_language_model_model_and_the_tower_stays() {
    let mut h = header(&[
        "language_model.embed_tokens.weight",
        "language_model.layers.0.self_attn.q_proj.weight",
        "language_model.norm.weight",
        "language_model.lm_head.weight",
        "vision_model.post_layernorm.weight",
        "connector.modality_projection.proj.weight",
    ]);
    canonicalize_idefics3_header(&mut h).expect("renames");
    let names: Vec<&str> = h.tensors.keys().map(String::as_str).collect();
    assert_eq!(
        names,
        [
            "connector.modality_projection.proj.weight",
            "language_model.lm_head.weight",
            "language_model.model.embed_tokens.weight",
            "language_model.model.layers.0.self_attn.q_proj.weight",
            "language_model.model.norm.weight",
            "vision_model.post_layernorm.weight",
        ]
    );
}

#[test]
fn a_rename_that_collides_is_refused() {
    let mut h = header(&[
        "language_model.norm.weight",
        "language_model.model.norm.weight",
    ]);
    assert!(canonicalize_idefics3_header(&mut h).is_err());
}

#[test]
fn the_tower_and_connector_bucket_as_vision_for_llama_only() {
    for name in [
        "vision_model.encoder.layers.0.mlp.fc1.weight",
        "connector.modality_projection.proj.weight",
    ] {
        assert_eq!(
            classify_for_family(name, 30, ModelFamily::Llama),
            Gemma4Bucket::VisionTower,
            "{name}"
        );
    }
    assert_ne!(
        classify_for_family(
            "connector.modality_projection.proj.weight",
            30,
            ModelFamily::Gemma4
        ),
        Gemma4Bucket::VisionTower
    );
}
