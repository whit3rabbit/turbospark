//! Hermetic test for the Llama-checkpoint orchestrator: builds a small,
//! real-format (but in-memory, not downloaded) `safetensors` blob by hand
//! and checks every expected resident tensor is extracted, quantized, and
//! named correctly. The real-network proof (downloading an actual small
//! HF checkpoint) is a separate, explicitly network-gated test — see
//! `hf_checkpoint_network.rs`.

use std::collections::BTreeMap;

use turbospark_repack::{
    fetch_safetensors_header, orchestrate_llama_checkpoint, LlamaCheckpointDims, MemoryRangeSource,
    OrchestrateError,
};

fn f32_tensor_bytes(rows: usize, cols: usize, seed: f32) -> Vec<u8> {
    let mut out = Vec::with_capacity(rows * cols * 4);
    for i in 0..rows * cols {
        let v = ((i as f32 * 0.01 + seed) % 2.0) - 1.0;
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

/// Builds a minimal real-format safetensors blob: 8-byte header length,
/// JSON header, then raw tensor bytes back to back in header order.
fn build_safetensors(tensors: &[(&str, usize, usize)]) -> Vec<u8> {
    let mut data = Vec::new();
    let mut header_entries = BTreeMap::new();
    for (i, (name, rows, cols)) in tensors.iter().enumerate() {
        let bytes = f32_tensor_bytes(*rows, *cols, i as f32);
        let start = data.len() as u64;
        data.extend_from_slice(&bytes);
        let end = data.len() as u64;
        header_entries.insert(
            name.to_string(),
            serde_json::json!({
                "dtype": "F32",
                "shape": [*rows, *cols],
                "data_offsets": [start, end],
            }),
        );
    }
    let header_json = serde_json::to_vec(&header_entries).unwrap();
    let mut out = Vec::with_capacity(8 + header_json.len() + data.len());
    out.extend_from_slice(&(header_json.len() as u64).to_le_bytes());
    out.extend_from_slice(&header_json);
    out.extend_from_slice(&data);
    out
}

#[test]
fn orchestrates_a_one_layer_checkpoint_end_to_end() {
    let dims = LlamaCheckpointDims {
        hidden_size: 64,
        intermediate_size: 64,
        num_heads: 1,
        num_kv_heads: 1,
        head_dim: 64,
        num_layers: 1,
        vocab_size: 64,
        tie_word_embeddings: true,
    };
    let h = dims.hidden_size;

    let tensors: Vec<(&str, usize, usize)> = vec![
        ("model.embed_tokens.weight", dims.vocab_size, h),
        ("model.layers.0.self_attn.q_proj.weight", h, h),
        ("model.layers.0.self_attn.k_proj.weight", h, h),
        ("model.layers.0.self_attn.v_proj.weight", h, h),
        ("model.layers.0.self_attn.o_proj.weight", h, h),
        ("model.layers.0.mlp.gate_proj.weight", h, h),
        ("model.layers.0.mlp.up_proj.weight", h, h),
        ("model.layers.0.mlp.down_proj.weight", h, h),
        ("model.layers.0.input_layernorm.weight", 1, h),
        ("model.layers.0.post_attention_layernorm.weight", 1, h),
        ("model.norm.weight", 1, h),
    ];
    let blob = build_safetensors(&tensors);
    let source = MemoryRangeSource::new(&blob);
    let header = fetch_safetensors_header(&source).expect("header parses");

    let specs =
        orchestrate_llama_checkpoint(&header, &source, dims).expect("orchestration succeeds");

    assert_eq!(specs.len(), tensors.len());
    let names: Vec<&str> = specs.iter().map(|s| s.name.as_str()).collect();
    assert!(names.contains(&"embed_tokens"));
    assert!(names.contains(&"layer0.q_proj"));
    assert!(names.contains(&"layer0.k_proj"));
    assert!(names.contains(&"layer0.v_proj"));
    assert!(names.contains(&"layer0.o_proj"));
    assert!(names.contains(&"layer0.gate_proj"));
    assert!(names.contains(&"layer0.up_proj"));
    assert!(names.contains(&"layer0.down_proj"));
    assert!(names.contains(&"layer0.input_norm"));
    assert!(names.contains(&"layer0.post_attn_norm"));
    assert!(names.contains(&"final_norm"));
    // No lm_head: tie_word_embeddings is true.
    assert!(!names.contains(&"lm_head"));

    let embed = specs.iter().find(|s| s.name == "embed_tokens").unwrap();
    assert_eq!(embed.rows, dims.vocab_size as u32);
    assert_eq!(embed.cols, h as u32);
    assert_eq!(embed.packed.len(), dims.vocab_size * h / 2);
}

#[test]
fn reports_a_missing_tensor_by_name() {
    let dims = LlamaCheckpointDims {
        hidden_size: 64,
        intermediate_size: 64,
        num_heads: 1,
        num_kv_heads: 1,
        head_dim: 64,
        num_layers: 1,
        vocab_size: 64,
        tie_word_embeddings: true,
    };
    // Missing everything past the embedding table.
    let tensors: Vec<(&str, usize, usize)> = vec![(
        "model.embed_tokens.weight",
        dims.vocab_size,
        dims.hidden_size,
    )];
    let blob = build_safetensors(&tensors);
    let source = MemoryRangeSource::new(&blob);
    let header = fetch_safetensors_header(&source).unwrap();

    let err = orchestrate_llama_checkpoint(&header, &source, dims).unwrap_err();
    assert!(err.to_string().contains("q_proj"));
}

/// One-tensor blob whose embedding table declares `data_offsets` verbatim, so
/// the parser's ordering check passes and only `absolute_range` can refuse it.
fn blob_with_embed_offsets(start: u64, end: u64) -> Vec<u8> {
    let mut entries = BTreeMap::new();
    entries.insert(
        "model.embed_tokens.weight".to_string(),
        serde_json::json!({
            "dtype": "F32",
            "shape": [1, 64],
            "data_offsets": [start, end],
        }),
    );
    let header_json = serde_json::to_vec(&entries).unwrap();
    let mut out = (header_json.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(&header_json);
    out
}

fn small_dims() -> LlamaCheckpointDims {
    LlamaCheckpointDims {
        hidden_size: 64,
        intermediate_size: 64,
        num_heads: 1,
        num_kv_heads: 1,
        head_dim: 64,
        num_layers: 1,
        vocab_size: 1,
        tie_word_embeddings: true,
    }
}

/// `absolute_range` returns `None` for a present tensor when the data region
/// start plus an offset overflows `u64`. A hostile header must get a named
/// error for that tensor, not a panic in the orchestrator.
#[test]
fn offset_overflowing_the_data_region_is_a_named_error_not_a_panic() {
    let blob = blob_with_embed_offsets(u64::MAX - 1, u64::MAX);
    let source = MemoryRangeSource::new(&blob);
    let header = fetch_safetensors_header(&source).expect("offsets are ordered, so header parses");
    assert!(header.tensors.contains_key("model.embed_tokens.weight"));
    assert_eq!(header.absolute_range("model.embed_tokens.weight"), None);

    let err = orchestrate_llama_checkpoint(&header, &source, small_dims()).unwrap_err();
    assert!(
        matches!(
            &err,
            OrchestrateError::TensorRangeOverflow(name) if name == "model.embed_tokens.weight"
        ),
        "unexpected error: {err}"
    );
}

/// A huge offset that still fits in `u64` resolves to a real range, and the
/// range source refuses it. Pins the boundary of the overflow refusal above.
#[test]
fn huge_addressable_offset_is_refused_by_the_range_source() {
    let blob = blob_with_embed_offsets(1 << 63, 1 << 63);
    let source = MemoryRangeSource::new(&blob);
    let header = fetch_safetensors_header(&source).unwrap();
    assert!(header.absolute_range("model.embed_tokens.weight").is_some());

    let err = orchestrate_llama_checkpoint(&header, &source, small_dims()).unwrap_err();
    assert!(
        matches!(err, OrchestrateError::Download(_)),
        "unexpected error: {err}"
    );
}
