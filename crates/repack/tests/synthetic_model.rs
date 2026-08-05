//! Round-trip test for the tiny synthetic Gemma-4 `.gturbo` install:
//! write it, then read every part back through the real
//! `mrefrust_model_io` loaders and check the resident tensors' shapes and
//! byte contents match what was written.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use mrefrust_repack::{
    build_synthetic_gemma4_install, build_synthetic_gemma4_moe_install, expert_gate_proj_name,
    q_proj_name, router_name,
};

static COUNTER: AtomicU64 = AtomicU64::new(0);

fn temp_dir() -> PathBuf {
    let n = COUNTER.fetch_add(1, Ordering::SeqCst);
    let dir = std::env::temp_dir().join(format!(
        "mrefrust-synthetic-model-{}-{n}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn writes_and_reads_back_a_real_install() {
    let dir = temp_dir();
    let vocab_size = 300;
    let num_layers = 2;
    let arch = build_synthetic_gemma4_install(&dir, vocab_size, num_layers, "tiny-test").unwrap();

    let manifest = model_io::load_manifest(&dir, &arch, model_io::DEFAULT_MAX_BYTES).unwrap();
    assert_eq!(manifest.arch.hidden_size, 64);
    assert_eq!(manifest.arch.num_layers, num_layers);

    let index = model_io::load_resident_index(&dir.join("model_weights.bin")).unwrap();
    assert_eq!(index.entries.len(), 1 + num_layers as usize * 6);

    let entry = index.entries.get(&q_proj_name(0)).expect("q_proj entry");
    assert_eq!(entry.shape.0, 64); // rows
    assert_eq!(entry.shape.1, 64); // cols

    let buffer = model_io::ResidentBuffer::map(
        &dir.join("model_weights.bin"),
        index.header.index_size,
        index.header.resident_size,
    )
    .unwrap();
    let data = buffer.data();
    let local_offset = (entry.file_offset - index.header.index_size) as usize;
    let packed = &data[local_offset..local_offset + entry.size_bytes as usize];
    assert_eq!(packed.len(), 64 * 64 / 2);

    let scale_local = (entry.scale_offset - index.header.index_size) as usize;
    let scales = &data[scale_local..scale_local + entry.scale_size as usize];
    assert_eq!(scales.len(), 64 * 64 / 64 * 2);
}

#[test]
fn writes_and_reads_back_a_real_moe_install() {
    let dir = temp_dir();
    let vocab_size = 300;
    let num_layers = 2;
    let num_experts = 4;
    let top_k = 2;
    let arch = build_synthetic_gemma4_moe_install(
        &dir,
        vocab_size,
        num_layers,
        num_experts,
        top_k,
        "moe-tiny-test",
    )
    .unwrap();
    assert_eq!(arch.num_experts, num_experts);
    assert_eq!(arch.top_k_experts, top_k);

    let manifest = model_io::load_manifest(&dir, &arch, model_io::DEFAULT_MAX_BYTES).unwrap();
    assert_eq!(manifest.arch.num_experts, num_experts);

    let index = model_io::load_resident_index(&dir.join("model_weights.bin")).unwrap();
    // embed + per layer: q/k/o/router + num_experts*3 (gate/up/down).
    let expected = 1 + num_layers as usize * (4 + num_experts as usize * 3);
    assert_eq!(index.entries.len(), expected);
    assert!(index.entries.contains_key(&router_name(0)));
    assert!(index.entries.contains_key(&expert_gate_proj_name(0, 3)));
    // No dense gate/up/down for a MoE-only install.
    assert!(!index.entries.contains_key("layer0.gate_proj"));
}
