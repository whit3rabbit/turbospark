//! The synthetic Qwen 3.6 install: it writes, it validates against the
//! arch peeked back out of its own manifest, and the family-specific
//! classification actually fires.

use turbospark_repack::{
    build_synthetic_qwen36_real_install, classify_for_family, peek_manifest_arch, Gemma4Bucket,
};

use model_io::ModelFamily;

const VOCAB: i64 = 128;
const LAYERS: i64 = 4;
const EXPERTS: i64 = 8;

fn build(dir: &std::path::Path) -> model_io::ArchConfig {
    build_synthetic_qwen36_real_install(dir, VOCAB, LAYERS, EXPERTS, "qwen-toy")
        .expect("synthetic qwen install")
}

fn temp_dir(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("turbospark-qwen-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// The manifest has to carry the family-extension fields, or validation
/// compares them against the Gemma defaults and every one of them
/// mismatches. This is the load `RealForwardRunner::open` performs.
#[test]
fn install_validates_against_its_own_peeked_arch() {
    let dir = temp_dir("validate");
    let built = build(&dir);
    let peeked = peek_manifest_arch(&dir).expect("peek");
    assert_eq!(peeked, built);
    model_io::load_manifest(&dir, &peeked, model_io::DEFAULT_MAX_BYTES)
        .expect("manifest validates");
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn peek_round_trips_family_mask_and_linear_config() {
    let dir = temp_dir("peek");
    let built = build(&dir);
    let peeked = peek_manifest_arch(&dir).expect("peek");
    assert_eq!(peeked.family, ModelFamily::Qwen36);
    assert_eq!(peeked.full_attention_layer_mask, vec![2u8, 1, 2, 1]);
    assert_eq!(peeked.linear_attention, built.linear_attention);
    assert_eq!(peeked.linear_attention.qkv_dim(), 256);
    assert_eq!(peeked.linear_attention.value_dim(), 128);
    assert!(peeked.attn_output_gate);
    assert!(peeked.shared_expert_gated);
    assert!(peeked.rope_neox_subdim);
    assert!(!peeked.ffn_sandwich_norms);
    assert!(!peeked.router_scaled);
    assert!(!peeked.embedding_scaled_by_sqrt_hidden);
    std::fs::remove_dir_all(&dir).unwrap();
}

/// The misclassification this guards against is SILENT: an unrecognized
/// routed marker makes every expert a resident tensor, which still loads
/// and still generates -- just with the whole expert table pinned.
#[test]
fn switch_mlp_is_routed_only_under_qwen() {
    let qwen_expert = "language_model.model.layers.2.mlp.switch_mlp.gate_proj.weight";
    assert_eq!(
        classify_for_family(qwen_expert, 4, ModelFamily::Qwen36),
        Gemma4Bucket::RoutedExpert {
            role: "gate",
            layer: 2
        }
    );
    assert_eq!(
        classify_for_family(qwen_expert, 4, ModelFamily::Gemma4),
        Gemma4Bucket::LmResident
    );

    let gemma_expert = "language_model.model.layers.2.experts.switch_glu.up_proj.weight";
    assert_eq!(
        classify_for_family(gemma_expert, 4, ModelFamily::Qwen36),
        Gemma4Bucket::LmResident
    );

    // The shared expert lives under `.mlp.` too and must NOT be routed.
    for name in [
        "language_model.model.layers.2.mlp.shared_expert.gate_proj.weight",
        "language_model.model.layers.2.mlp.shared_expert_gate.weight",
        "language_model.model.layers.2.mlp.gate.weight",
        "language_model.model.layers.0.linear_attn.A_log",
    ] {
        assert_eq!(
            classify_for_family(name, 4, ModelFamily::Qwen36),
            Gemma4Bucket::LmResident,
            "{name}"
        );
    }
}

#[test]
fn resident_index_has_linear_tensors_and_no_experts() {
    let dir = temp_dir("resident");
    build(&dir);
    let index = model_io::load_resident_index(&dir.join("model_weights.bin")).expect("index");

    for name in [
        "language_model.model.embed_tokens.weight",
        "language_model.lm_head.weight",
        "language_model.model.norm.weight",
        // Layer 0 is linear: the GDN tensor set, A_log/dt_bias suffix-less.
        "language_model.model.layers.0.linear_attn.in_proj_qkv.weight",
        "language_model.model.layers.0.linear_attn.in_proj_z.weight",
        "language_model.model.layers.0.linear_attn.in_proj_a.weight",
        "language_model.model.layers.0.linear_attn.in_proj_b.weight",
        "language_model.model.layers.0.linear_attn.out_proj.weight",
        "language_model.model.layers.0.linear_attn.conv1d.weight",
        "language_model.model.layers.0.linear_attn.A_log",
        "language_model.model.layers.0.linear_attn.dt_bias",
        "language_model.model.layers.0.linear_attn.norm.weight",
        // Layer 1 is full attention.
        "language_model.model.layers.1.self_attn.q_proj.weight",
        "language_model.model.layers.1.self_attn.k_norm.weight",
        // MoE on every layer.
        "language_model.model.layers.0.mlp.gate.weight",
        "language_model.model.layers.0.mlp.shared_expert_gate.weight",
        "language_model.model.layers.0.mlp.shared_expert.down_proj.weight",
    ] {
        assert!(index.entries.contains_key(name), "missing {name}");
    }
    assert!(
        !index.entries.keys().any(|k| k.contains(".mlp.switch_mlp.")),
        "routed experts leaked into the resident index"
    );

    // conv1d is BF16 [C, K, 1] -- rank 3, dtype tag 1 (BF16), and sized
    // for the whole 256-channel x 4-tap kernel.
    let conv = &index.entries["language_model.model.layers.0.linear_attn.conv1d.weight"];
    assert_eq!(conv.dtype, turbospark_repack::DTYPE_BF16);
    assert_eq!(conv.size_bytes, 256 * 4 * 2);
    assert_eq!((conv.shape.0, conv.shape.1, conv.shape.2), (256, 4, 1));

    // in_proj_qkv is INT4 [qkv_dim, hidden] = [256, 64] -> 8192 nibbles.
    let qkv = &index.entries["language_model.model.layers.0.linear_attn.in_proj_qkv.weight"];
    assert_eq!(qkv.size_bytes, 256 * 64 / 2);

    let layout = model_io::load_packed_experts_layout(
        &dir,
        model_io::PACKED_EXPERTS_LAYOUT_DEFAULT_MAX_BYTES,
    )
    .expect("layout");
    assert_eq!(layout.num_layers, LAYERS as usize);
    assert_eq!(layout.experts_per_layer, EXPERTS as usize);
    std::fs::remove_dir_all(&dir).unwrap();
}
