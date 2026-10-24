//! `canonicalize_flint_header` and the Flint classifier against the real
//! tensor names and shapes of `RecursiveSI/flint-int4` (read off the shard
//! headers, revision 3b93f9053b530488405ae9fca51d4900d6ce21f0).

use std::collections::BTreeMap;

use model_io::ModelFamily;
use turbospark_repack::{
    canonicalize_flint_header, classify_for_family, Gemma4Bucket, SafetensorsHeader, TensorInfo,
};

fn info(dtype: &str, shape: &[u64], bytes: u64) -> TensorInfo {
    TensorInfo {
        dtype: dtype.to_string(),
        shape: shape.to_vec(),
        data_offsets: (0, bytes),
    }
}

fn header(entries: &[(&str, TensorInfo)]) -> SafetensorsHeader {
    SafetensorsHeader {
        tensors: entries
            .iter()
            .map(|(n, i)| (n.to_string(), i.clone()))
            .collect::<BTreeMap<_, _>>(),
        metadata: None,
        header_len: 0,
    }
}

#[test]
fn relabels_u8_packing_as_u32_without_touching_bytes() {
    // q_proj INT4 [4096, 2048 in] -> U8 [4096, 1024]; embed INT8 -> U8 [V, 2048].
    let mut h = header(&[
        (
            "model.layers.0.self_attn.q_proj.weight",
            info("U8", &[4096, 1024], 4096 * 1024),
        ),
        (
            "model.layers.0.self_attn.q_proj.scales",
            info("BF16", &[4096, 32], 4096 * 64),
        ),
        (
            "model.layers.0.self_attn.q_proj.biases",
            info("BF16", &[4096, 32], 4096 * 64),
        ),
        (
            "model.embed_tokens.weight",
            info("U8", &[150144, 2048], 150144 * 2048),
        ),
        (
            "model.embed_tokens.scales",
            info("BF16", &[150144, 32], 150144 * 64),
        ),
        (
            "model.embed_tokens.biases",
            info("BF16", &[150144, 32], 150144 * 64),
        ),
        ("lm_head.weight", info("U8", &[150144, 1024], 150144 * 1024)),
        ("lm_head.scales", info("BF16", &[150144, 32], 150144 * 64)),
        ("lm_head.biases", info("BF16", &[150144, 32], 150144 * 64)),
        // Stacked experts are rank 3.
        (
            "model.layers.2.mlp.switch_mlp.down_proj.weight",
            info("U8", &[256, 2048, 256], 256 * 2048 * 256),
        ),
        (
            "model.layers.2.mlp.switch_mlp.down_proj.scales",
            info("BF16", &[256, 2048, 8], 1),
        ),
        (
            "model.layers.2.mlp.switch_mlp.down_proj.biases",
            info("BF16", &[256, 2048, 8], 1),
        ),
        ("model.norm.weight", info("BF16", &[2048], 4096)),
    ]);
    canonicalize_flint_header(&mut h).unwrap();

    let q = &h.tensors["language_model.model.layers.0.self_attn.q_proj.weight"];
    assert_eq!(
        (q.dtype.as_str(), q.shape.clone()),
        ("U32", vec![4096, 256])
    );
    // The byte range is untouched: 4096 * 256 words * 4 bytes == 4096 * 1024.
    assert_eq!(q.data_offsets, (0, 4096 * 1024));
    assert_eq!(
        h.tensors["language_model.model.embed_tokens.weight"].shape,
        vec![150144, 512]
    );
    assert_eq!(
        h.tensors["language_model.lm_head.weight"].shape,
        vec![150144, 256]
    );
    assert_eq!(
        h.tensors["language_model.model.layers.2.mlp.switch_mlp.down_proj.weight"].shape,
        vec![256, 2048, 64]
    );
    // Companions and norms keep their dtype and shape.
    assert_eq!(h.tensors["language_model.model.norm.weight"].dtype, "BF16");
    assert_eq!(
        h.tensors["language_model.model.layers.0.self_attn.q_proj.scales"].shape,
        vec![4096, 32]
    );
    // Every U32 weight's words times four equals its byte length.
    for (name, t) in &h.tensors {
        if t.dtype == "U32" {
            let words: u64 = t.shape.iter().product();
            assert_eq!(words * 4, t.data_offsets.1 - t.data_offsets.0, "{name}");
        }
    }
}

#[test]
fn refuses_a_u8_tensor_that_is_not_an_affine_weight() {
    let mut h = header(&[("model.layers.0.mystery.weight", info("U8", &[8, 8], 64))]);
    assert!(canonicalize_flint_header(&mut h).is_err());
}

#[test]
fn routed_experts_are_not_resident_and_sinks_are_resident() {
    let f = ModelFamily::Flint;
    let n = 49;
    let routed = classify_for_family(
        "language_model.model.layers.2.mlp.switch_mlp.gate_proj.weight",
        n,
        f,
    );
    assert!(matches!(
        routed,
        Gemma4Bucket::RoutedExpert {
            role: "gate",
            layer: 2
        }
    ));
    for name in [
        "language_model.model.layers.2.mlp.switch_mlp.up_proj.scales",
        "language_model.model.layers.2.mlp.switch_mlp.down_proj.biases",
    ] {
        assert!(
            matches!(
                classify_for_family(name, n, f),
                Gemma4Bucket::RoutedExpert { .. }
            ),
            "{name}"
        );
    }
    // Dense and shared tensors, the router, the norms and the sink stay resident.
    for name in [
        "language_model.model.layers.0.mlp.gate_proj.weight",
        "language_model.model.layers.2.mlp.shared_expert.gate_proj.weight",
        "language_model.model.layers.2.mlp.gate.weight",
        "language_model.model.layers.2.self_attn.q_norm.weight",
        "language_model.model.layers.2.self_attn.softmax_offset",
        "language_model.lm_head.weight",
    ] {
        assert!(
            matches!(classify_for_family(name, n, f), Gemma4Bucket::LmResident),
            "{name}"
        );
    }
}
