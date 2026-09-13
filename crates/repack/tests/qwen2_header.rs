//! Qwen2 MLX tensor namespace normalization.

use std::collections::BTreeMap;

use turbospark_repack::{canonicalize_qwen2_header, SafetensorsHeader, TensorInfo};

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
fn canonicalizes_the_text_only_mlx_namespace() {
    let mut h = header(&[
        "model.embed_tokens.weight",
        "model.layers.0.self_attn.q_proj.bias",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
        "model.norm.weight",
        "lm_head.weight",
    ]);
    canonicalize_qwen2_header(&mut h).expect("Qwen2 source names canonicalize");
    assert!(h
        .tensors
        .contains_key("language_model.model.embed_tokens.weight"));
    assert!(h
        .tensors
        .contains_key("language_model.model.layers.0.self_attn.q_proj.bias"));
    assert!(h
        .tensors
        .contains_key("language_model.model.layers.0.self_attn.k_proj.weight"));
    assert!(h.tensors.contains_key("language_model.model.norm.weight"));
    assert!(h.tensors.contains_key("language_model.lm_head.weight"));
    assert!(h
        .tensors
        .keys()
        .all(|name| !name.starts_with("model.") && !name.starts_with("lm_head.")));
}

#[test]
fn canonical_namespace_is_idempotent() {
    let mut h = header(&[
        "language_model.model.embed_tokens.weight",
        "language_model.model.norm.weight",
        "language_model.lm_head.weight",
    ]);
    let before = h.clone();
    canonicalize_qwen2_header(&mut h).expect("canonical names are accepted");
    assert_eq!(h, before);
}

#[test]
fn mixed_source_and_canonical_names_are_refused() {
    let mut h = header(&[
        "model.embed_tokens.weight",
        "language_model.model.norm.weight",
    ]);
    let error = canonicalize_qwen2_header(&mut h)
        .expect_err("mixed Qwen2 namespaces must be refused")
        .to_string();
    assert!(error.contains("mixes"), "{error}");
    assert!(error.contains("model.*"), "{error}");
    assert!(error.contains("language_model.*"), "{error}");
}
