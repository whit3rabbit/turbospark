//! Network-gated proof that the orchestrator works against a REAL
//! downloaded HF checkpoint: `HuggingFaceTB/SmolLM2-135M` (~270MB, real
//! bf16 Llama-family weights on the Hugging Face Hub). Not run by default
//! `cargo test --workspace` (real network access and a few hundred MB of
//! download); run explicitly:
//!
//! ```sh
//! cargo test -p mrefrust-repack --test hf_checkpoint_network -- --ignored --nocapture
//! ```

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use mrefrust_repack::{
    build_resident_weights_bin, fetch_safetensors_header, orchestrate_llama_checkpoint,
    tiny_gemma4_arch, write_gturbo_install_with_resident_index, HttpRangeSource,
    LlamaCheckpointDims,
};

static COUNTER: AtomicU64 = AtomicU64::new(0);

fn temp_dir() -> PathBuf {
    let n = COUNTER.fetch_add(1, Ordering::SeqCst);
    let dir = std::env::temp_dir().join(format!("mrefrust-hf-real-{}-{n}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
#[ignore = "downloads a real ~270MB HF checkpoint over the network"]
fn orchestrates_a_real_downloaded_smollm2_checkpoint() {
    let dims = LlamaCheckpointDims {
        hidden_size: 576,
        intermediate_size: 1536,
        num_heads: 9,
        num_kv_heads: 3,
        head_dim: 64,
        num_layers: 30,
        vocab_size: 49152,
        tie_word_embeddings: true,
    };

    let url = "https://huggingface.co/HuggingFaceTB/SmolLM2-135M/resolve/main/model.safetensors";
    let source = HttpRangeSource::new(url);
    let header = fetch_safetensors_header(&source).expect("real header fetch over the network");
    eprintln!("real safetensors header: {} tensors", header.tensors.len());

    let specs = orchestrate_llama_checkpoint(&header, &source, dims)
        .expect("real checkpoint orchestration (download + decode + quantize)");
    // embed_tokens + 9 tensors/layer * 30 layers + final_norm; no lm_head
    // since tie_word_embeddings is true.
    assert_eq!(specs.len(), 1 + dims.num_layers * 9 + 1);
    eprintln!("orchestrated {} real resident tensors", specs.len());

    let mut arch = tiny_gemma4_arch(dims.vocab_size as i64, dims.num_layers as i64);
    arch.hidden_size = dims.hidden_size as i64;
    arch.intermediate_size = dims.intermediate_size as i64;
    arch.num_heads = dims.num_heads as i64;
    arch.num_kv_heads = dims.num_kv_heads as i64;
    arch.num_full_kv_heads = dims.num_kv_heads as i64;
    arch.head_dim = dims.head_dim as i64;
    arch.full_head_dim = dims.head_dim as i64;
    arch.attention_k_eq_v = false;
    arch.hidden_activation = "silu".to_string();
    arch.tie_word_embeddings = dims.tie_word_embeddings;
    arch.rope_theta = 100_000.0;
    arch.full_rope_theta = 100_000.0;
    arch.full_attention_layer_mask = vec![1u8; dims.num_layers];

    let resident_bytes = build_resident_weights_bin(&specs);
    let dir = temp_dir();
    write_gturbo_install_with_resident_index(&dir, &arch, "smollm2-135m-real", &resident_bytes)
        .expect("real install writes");

    let manifest = model_io::load_manifest(&dir, &arch, model_io::DEFAULT_MAX_BYTES)
        .expect("real install's manifest reads back and validates");
    assert_eq!(manifest.arch.vocab_size, dims.vocab_size as i64);
    assert_eq!(manifest.arch.num_layers, dims.num_layers as i64);

    let index = model_io::load_resident_index(&dir.join("model_weights.bin"))
        .expect("real install's resident index reads back");
    assert_eq!(index.entries.len(), specs.len());
    assert!(index.entries.contains_key("embed_tokens"));
    assert!(index.entries.contains_key("layer0.q_proj"));
    assert!(index.entries.contains_key("final_norm"));

    eprintln!(
        "SUCCESS: real HuggingFaceTB/SmolLM2-135M checkpoint orchestrated into a real, \
         readable .gturbo install at {}",
        dir.display()
    );
}
