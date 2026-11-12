//! The tiny `lfm3` install, end to end through the real GGUF walk: the
//! per-layer `head_count_kv` mask, the `shortconv.*` table, and the
//! `rope_freqs.weight` value check.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use model_io::ModelFamily;
use turbospark_repack::{
    build_synthetic_lfm3_install, build_synthetic_lfm3_install_with_rope_freqs, peek_manifest_arch,
    tiny_lfm3_rope_freqs,
};

static COUNTER: AtomicU64 = AtomicU64::new(0);

fn temp_dir() -> PathBuf {
    let n = COUNTER.fetch_add(1, Ordering::SeqCst);
    let dir = std::env::temp_dir().join(format!(
        "turbospark-lfm3-synthetic-{}-{n}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).expect("create temp dir");
    dir
}

/// Six layers: `[conv, conv, full, conv, sliding, conv]`, every kind present.
const LAYERS: i64 = 6;

#[test]
fn the_manifest_loads_validates_and_peeks_back() {
    let dir = temp_dir();
    let arch = build_synthetic_lfm3_install(&dir, 64, LAYERS, "lfm3-toy").expect("writes");
    assert_eq!(arch.family, ModelFamily::Lfm3);
    assert_eq!(arch.full_attention_layer_mask, vec![6, 6, 1, 6, 0, 6]);
    assert_eq!(arch.short_conv_kernel_size, 7);
    model_io::load_manifest(&dir, &arch, 4 * 1024 * 1024).expect("manifest validates");
    // The install's own manifest round-trips the whole arch, the new
    // `shortConvKernelSize` field included (AGENTS.md Gotcha 24).
    assert_eq!(peek_manifest_arch(&dir).expect("peek"), arch);
}

#[test]
fn conv_layers_carry_conv_tensors_and_attention_layers_carry_attention() {
    let dir = temp_dir();
    build_synthetic_lfm3_install(&dir, 64, LAYERS, "lfm3-toy").expect("writes");
    let index = model_io::load_resident_index(&dir.join("model_weights.bin")).expect("index");
    let has = |l: usize, tail: &str| {
        index
            .entries
            .contains_key(&format!("language_model.model.layers.{l}.{tail}"))
    };
    for (l, conv) in [(0, true), (1, true), (2, false), (3, true), (4, false)] {
        assert_eq!(has(l, "conv.in_proj.weight"), conv, "layer {l}");
        assert_eq!(has(l, "self_attn.q_proj.weight"), !conv, "layer {l}");
        assert_eq!(has(l, "self_attn.gate_proj.weight"), !conv, "layer {l}");
        for tail in [
            "operator_norm.weight",
            "post_operator_norm.weight",
            "ffn_norm.weight",
            "post_ffn_norm.weight",
            "feed_forward.w1.weight",
            "feed_forward.w2.weight",
            "feed_forward.w3.weight",
        ] {
            assert!(has(l, tail), "layer {l} {tail}");
        }
    }
    // Logical `[hidden, taps]`: the layout the short-conv kernel reads.
    let conv = &index.entries["language_model.model.layers.0.conv.conv.weight"];
    assert_eq!((conv.shape.0, conv.shape.1), (128, 7));
    assert!(index
        .entries
        .contains_key("language_model.model.embedding_norm.weight"));
    // Dropped after verification, never carried.
    assert!(!index.entries.keys().any(|k| k.contains("rope_freqs")));
}

/// One real scaling factor in the vector refuses the whole install: the
/// same tensor name means learned scaling in Llama 3.1.
#[test]
fn a_scaled_rope_freqs_vector_refuses_the_install() {
    let mut freqs = tiny_lfm3_rope_freqs();
    freqs[3] = 1.5;
    let err = build_synthetic_lfm3_install_with_rope_freqs(
        &temp_dir(),
        64,
        LAYERS,
        "lfm3-toy",
        Some(&freqs),
    )
    .unwrap_err()
    .to_string();
    assert!(err.contains("factor 3 is 1.5"), "{err}");
}

/// The partial factor this port applies is the one the vector encodes, so
/// a conversion without it is refused rather than assumed.
#[test]
fn a_missing_rope_freqs_vector_refuses_the_install() {
    let err =
        build_synthetic_lfm3_install_with_rope_freqs(&temp_dir(), 64, LAYERS, "lfm3-toy", None)
            .unwrap_err()
            .to_string();
    assert!(err.contains("rope_freqs.weight"), "{err}");
}

/// Every other family keeps the by-name refusal.
#[test]
fn rope_freqs_stays_refused_for_other_families() {
    let err = turbospark_repack::map_gguf_name("rope_freqs.weight", ModelFamily::Llama)
        .unwrap_err()
        .to_string();
    assert!(err.contains("learned per-dimension RoPE"), "{err}");
}
