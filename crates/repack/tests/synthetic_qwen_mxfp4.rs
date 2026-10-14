//! The MLX mxfp4 Qwen 3.6 install (the MiniCPM-V 4.7 trunk's layout): it
//! writes, validates, and carries each tensor in the form its kernel reads.

use std::sync::atomic::{AtomicU64, Ordering};

use turbospark_repack::{
    build_synthetic_qwen_gdn_moe_mxfp4_install, gguf_mxfp4_row_bytes, peek_manifest_arch,
};

const VOCAB: i64 = 128;
const LAYERS: i64 = 4;
const EXPERTS: i64 = 8;

/// Resident dtype tags (`crates/runtime/src/real_forward_layout.rs`).
const DTYPE_RAW_BF16: u8 = 1;
const DTYPE_INT8: u8 = 5;

static COUNTER: AtomicU64 = AtomicU64::new(0);

fn temp_dir(name: &str) -> std::path::PathBuf {
    let n = COUNTER.fetch_add(1, Ordering::SeqCst);
    let dir = std::env::temp_dir().join(format!(
        "turbospark-qwen-mxfp4-{name}-{}-{n}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn build(name: &str) -> (std::path::PathBuf, model_io::ArchConfig) {
    let dir = temp_dir(name);
    let arch =
        build_synthetic_qwen_gdn_moe_mxfp4_install(&dir, VOCAB, LAYERS, EXPERTS, "mxfp4-toy")
            .expect("synthetic mxfp4 install");
    (dir, arch)
}

#[test]
fn install_validates_and_states_what_it_wrote() {
    let (dir, built) = build("validate");
    let peeked = peek_manifest_arch(&dir).expect("peek");
    assert_eq!(peeked, built);
    let manifest = model_io::load_manifest(&dir, &peeked, model_io::DEFAULT_MAX_BYTES)
        .expect("manifest validates");
    let quant = manifest.quant.expect("quant object");
    for (slot, s) in [
        ("embedding", &quant.embedding),
        ("attention", &quant.attention),
        ("sharedExpert", &quant.shared_expert),
    ] {
        assert_eq!((s.scheme.as_str(), s.weight_bits), ("raw", 16), "{slot}");
    }
    assert_eq!(
        (
            quant.router.scheme.as_str(),
            quant.router.weight_bits,
            quant.router.group_size
        ),
        ("affine", 8, 64)
    );
    assert_eq!(quant.routed_expert.scheme, "gguf");
    assert_eq!(quant.routed_expert.ggml_type.as_deref(), Some("mxfp4"));
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn experts_are_gguf_mxfp4_blocks_with_no_companions() {
    let (dir, arch) = build("experts");
    let layout =
        model_io::load_packed_experts_layout(&dir, model_io::DEFAULT_MAX_BYTES).expect("layout");
    let (hidden, inter) = (arch.hidden_size as u64, arch.moe_intermediate_size as u64);
    let gate_bytes = inter * gguf_mxfp4_row_bytes(hidden as usize) as u64;
    let down_bytes = hidden * gguf_mxfp4_row_bytes(inter as usize) as u64;
    for layer in &layout.layers {
        for expert in &layer.experts {
            let roles: Vec<&str> = expert.sub_tensors.keys().map(String::as_str).collect();
            // Exactly three: no `gate_scales`, which is the key that would
            // make the runtime classify the layer as affine.
            assert_eq!(roles, ["down", "gate", "up"], "layer {}", layer.layer);
            for (role, size) in [
                ("gate", gate_bytes),
                ("up", gate_bytes),
                ("down", down_bytes),
            ] {
                let sub = &expert.sub_tensors[role];
                assert_eq!(sub.dtype, "mxfp4");
                assert_eq!(sub.size, size, "{role}");
            }
            assert_eq!(
                expert.sub_tensors["down"].offset % 4,
                0,
                "down is uint-aligned"
            );
        }
    }
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn resident_mxfp4_is_bf16_and_overrides_stay_int8() {
    let (dir, arch) = build("resident");
    let index = model_io::load_resident_index(&dir.join("model_weights.bin")).expect("index");
    let full = (0..arch.num_layers as usize)
        .find(|&l| arch.layer_is_full(l))
        .unwrap();
    let linear = (0..arch.num_layers as usize)
        .find(|&l| !arch.layer_is_full(l))
        .unwrap();
    let p = |l: usize| format!("language_model.model.layers.{l}");
    for (name, dtype) in [
        (
            "language_model.model.embed_tokens.weight".to_string(),
            DTYPE_RAW_BF16,
        ),
        ("language_model.lm_head.weight".to_string(), DTYPE_RAW_BF16),
        (
            format!("{}.self_attn.q_proj.weight", p(full)),
            DTYPE_RAW_BF16,
        ),
        (
            format!("{}.mlp.shared_expert.down_proj.weight", p(full)),
            DTYPE_RAW_BF16,
        ),
        (
            format!("{}.linear_attn.in_proj_qkv.weight", p(linear)),
            DTYPE_RAW_BF16,
        ),
        (format!("{}.mlp.gate.weight", p(linear)), DTYPE_INT8),
        (
            format!("{}.mlp.shared_expert_gate.weight", p(linear)),
            DTYPE_INT8,
        ),
    ] {
        let entry = index
            .entries
            .get(&name)
            .unwrap_or_else(|| panic!("{name} missing"));
        assert_eq!(entry.dtype, dtype, "{name}");
    }
    // The dequantized matrix keeps its logical shape and is 2 bytes a value.
    let q = &index.entries[&format!("{}.self_attn.q_proj.weight", p(full))];
    let rows = 2 * arch.num_heads as u64 * arch.head_dim as u64;
    assert_eq!(
        (q.shape.0 as u64, q.shape.1 as u64),
        (rows, arch.hidden_size as u64)
    );
    assert_eq!(q.size_bytes, rows * arch.hidden_size as u64 * 2);
    // The wrapper's tower and mergers did not become resident tensors.
    assert!(
        index
            .entries
            .keys()
            .all(|k| k.starts_with("language_model.")),
        "a non-text tensor was ingested"
    );
    std::fs::remove_dir_all(&dir).unwrap();
}
