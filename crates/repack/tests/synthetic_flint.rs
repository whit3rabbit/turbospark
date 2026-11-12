//! The tiny Flint install, end to end through the real walk, in the
//! checkpoint's own source form (U8 packing, BF16 router, INT8 embedding).

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use model_io::ModelFamily;

/// The resident index's dtype tags (`model_io::resident_index`).
const DTYPE_INT4: u8 = 4;
const DTYPE_INT8: u8 = 5;
use turbospark_repack::build_synthetic_flint_install;

static COUNTER: AtomicU64 = AtomicU64::new(0);

fn temp_dir() -> PathBuf {
    let n = COUNTER.fetch_add(1, Ordering::SeqCst);
    let dir = std::env::temp_dir().join(format!(
        "turbospark-flint-synthetic-{}-{n}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).expect("create temp dir");
    dir
}

fn build() -> (PathBuf, model_io::ArchConfig) {
    let dir = temp_dir();
    let arch = build_synthetic_flint_install(&dir, 64, "flint-toy").expect("the install writes");
    (dir, arch)
}

#[test]
fn the_manifest_loads_and_validates() {
    let (dir, arch) = build();
    let manifest = model_io::load_manifest(&dir, &arch, 4 * 1024 * 1024)
        .expect("the manifest loads and validates");
    assert_eq!(manifest.arch.family.as_deref(), Some("flint"));
    assert_eq!(manifest.arch.num_layers, 4);
    assert_eq!(arch.family, ModelFamily::Flint);
}

/// The dense lead writes NO expert blobs: layers 0 and 1 are dense, so the
/// two packed files are layers 2 and 3 at positions 0 and 1. A walk that
/// still looped from layer 0 would either error or write dense layers as
/// experts.
#[test]
fn only_the_moe_layers_write_expert_files() {
    let (dir, _) = build();
    let mut files: Vec<String> = std::fs::read_dir(dir.join("packed_experts"))
        .expect("packed_experts exists")
        .filter_map(Result::ok)
        .map(|e| e.file_name().to_string_lossy().to_string())
        .filter(|n| n.starts_with("layer_"))
        .collect();
    files.sort();
    assert_eq!(files, vec!["layer_00.bin", "layer_01.bin"]);
}

#[test]
fn resident_index_has_the_right_tensors_in_the_right_forms() {
    let (dir, _) = build();
    let index = model_io::load_resident_index(&dir.join("model_weights.bin")).expect("index");
    let names: Vec<&String> = index.entries.keys().collect();

    // No routed expert may be resident (AGENTS.md Gotcha 26).
    assert!(
        !names.iter().any(|n| n.contains("switch_mlp")),
        "a routed expert became resident: {names:?}"
    );
    // The sink and the per-head norms are resident and keep their names.
    for l in 0..4 {
        let p = format!("language_model.model.layers.{l}.self_attn");
        for tail in ["softmax_offset", "q_norm.weight", "k_norm.weight"] {
            assert!(
                index.entries.contains_key(&format!("{p}.{tail}")),
                "{p}.{tail}"
            );
        }
    }
    // Dense lead layers carry a dense FFN; MoE layers carry a shared expert.
    assert!(index
        .entries
        .contains_key("language_model.model.layers.1.mlp.gate_proj.weight"));
    assert!(index
        .entries
        .contains_key("language_model.model.layers.2.mlp.shared_expert.gate_proj.weight"));
    assert!(!index
        .entries
        .contains_key("language_model.model.layers.0.mlp.gate.weight"));

    // The BF16 router was force-quantized to INT8 [experts, hidden].
    let router = &index.entries["language_model.model.layers.2.mlp.gate.weight"];
    assert_eq!(router.dtype, DTYPE_INT8, "router must be INT8 affine");
    assert_eq!((router.shape.0, router.shape.1), (8, 64));
    // The embedding is INT8, the head INT4: different packed widths.
    let embed = &index.entries["language_model.model.embed_tokens.weight"];
    let head = &index.entries["language_model.lm_head.weight"];
    assert_eq!(embed.dtype, DTYPE_INT8);
    assert_eq!(head.dtype, DTYPE_INT4);
}
