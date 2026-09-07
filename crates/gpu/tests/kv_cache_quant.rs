//! Unit coverage for `KvCacheManager::new_with_kv_quant` at
//! `KvQuant::TurboQuant` (AGENTS.md/CLAUDE.md T6). `tests/kv_cache.rs`
//! only ever calls `new` (the `KvQuant::Off` wrapper); this file is the
//! citation `crates/model-io/src/kv_quant.rs` makes for the TurboQuant
//! arm's own coverage.
#![cfg(target_os = "macos")]

use model_io::KvQuant;
use turbospark_gpu::{KvCacheManager, LayerKind, MetalContext};

/// `full_head_dim` must be `rht_supported` (a power of two in 32..=512)
/// for `KvQuantTables::new` to build at all; 64 is the smallest real value
/// any family here uses.
fn toy_arch(mask: Vec<u8>) -> model_io::ArchConfig {
    model_io::ArchConfig {
        hidden_size: 64,
        intermediate_size: 128,
        moe_intermediate_size: 64,
        num_heads: 2,
        num_kv_heads: 2,
        num_full_kv_heads: 2,
        head_dim: 64,
        full_head_dim: 64,
        vocab_size: 100,
        sliding_window: 4,
        final_logit_softcap: 0.0,
        rope_theta: 10000.0,
        full_rope_theta: 10000.0,
        partial_rotary_factor: 1.0,
        num_layers: mask.len() as i64,
        num_experts: 1,
        top_k_experts: 1,
        tie_word_embeddings: true,
        attention_k_eq_v: true,
        full_attention_layer_mask: mask,
        hidden_activation: "gelu_pytorch_tanh".to_string(),
        family: model_io::ModelFamily::Gemma4,
        attn_output_gate: false,
        attention_scale: 1.0,
        embedding_scaled_by_sqrt_hidden: true,
        router_scaled: true,
        ffn_sandwich_norms: true,
        shared_expert_gated: false,
        rope_neox_subdim: false,
        linear_attention: model_io::LinearAttentionConfig::NONE,
        compressed_attention: model_io::CompressedAttentionConfig::NONE,
        hyper_connections: model_io::HyperConnectionConfig::NONE,
        num_hash_routed_layers: 0,
        router_scoring_func: "softmax".to_string(),
        routed_scaling_factor: 1.0,
        swiglu_limit: 0.0,
        rope_scaling: model_io::RopeScalingConfig::NONE,
        vision: model_io::VisionConfig::NONE,
        ple: model_io::PleConfig::NONE,
    }
}

/// swa, full (quantized: not the model's last layer), full (NOT quantized:
/// `layer_is_quantized` excludes the final layer). This one arch exercises
/// both the on and off gate at `mask == 1` in a single manager.
fn open(quant: KvQuant) -> (MetalContext, KvCacheManager) {
    let context = MetalContext::new().unwrap();
    let arch = toy_arch(vec![0, 1, 1]);
    let manager =
        KvCacheManager::new_with_kv_quant(context.device(), &arch, 32, false, None, 8, None, quant)
            .expect("open");
    (context, manager)
}

#[test]
fn per_layer_quantization_gating_matches_layer_is_quantized() {
    let (_context, manager) = open(KvQuant::TurboQuant {
        k_bits: 3,
        v_bits: 4,
    });

    // Layer 0: SWA, never quantized regardless of kv_quant.
    assert_eq!(manager.layer_kind(0), LayerKind::Swa);
    assert_eq!(manager.layer_quant(0), None);

    // Layer 1: full, not the model's last layer -- quantized.
    assert_eq!(manager.layer_kind(1), LayerKind::Full);
    assert_eq!(manager.layer_quant(1), Some((3, 4)));

    // Layer 2: full, but IS the model's last layer -- `layer_is_quantized`
    // excludes it, so this layer stays FP16 despite mask == 1.
    assert_eq!(manager.layer_kind(2), LayerKind::Full);
    assert_eq!(manager.layer_quant(2), None);
}

#[test]
fn k_and_v_strides_differ_at_different_bit_widths() {
    let (_context, manager) = open(KvQuant::TurboQuant {
        k_bits: 3,
        v_bits: 4,
    });

    // Layer 1 is quantized (see the gating test); K3/V4 pack to different
    // word counts per row, so the two strides must differ -- the whole
    // reason this manager tracks them as two arrays (see its module doc).
    assert_ne!(manager.k_stride(1), manager.v_stride(1));

    // An unquantized layer's two sides stay the historical FP16-symmetric
    // case: equal strides (`num_kv_heads * head_dim * 2`).
    assert_eq!(manager.k_stride(2), manager.v_stride(2));

    // `quant_tables()` is `Some` whenever ANY layer quantizes, shared
    // across every quantized layer in the model.
    assert!(manager.quant_tables().is_some());
}

#[test]
fn kv_quant_off_reports_no_quantized_layers_and_matched_strides() {
    let (_context, manager) = open(KvQuant::Off);

    for layer in 0..3 {
        assert_eq!(manager.layer_quant(layer), None, "layer {layer}");
    }
    assert_eq!(manager.k_stride(1), manager.v_stride(1));
    assert!(manager.quant_tables().is_none());
}

#[test]
#[should_panic(expected = "TurboQuant-quantized")]
fn write_k_refuses_a_quantized_layer() {
    let (_context, manager) = open(KvQuant::TurboQuant {
        k_bits: 4,
        v_bits: 4,
    });
    manager.write_k(1, 0, &[0u8; 4]);
}

#[test]
#[should_panic(expected = "TurboQuant-quantized")]
fn write_v_refuses_a_quantized_layer() {
    let (_context, manager) = open(KvQuant::TurboQuant {
        k_bits: 4,
        v_bits: 4,
    });
    manager.write_v(1, 0, &[0u8; 4]);
}

/// `write_k`/`write_v` must stay usable on a layer `layer_is_quantized`
/// excludes (the model's last full layer), and on an unquantized install
/// entirely -- the refusal is per-LAYER, not per-manager.
#[test]
fn write_k_and_write_v_succeed_on_an_unquantized_layer() {
    let (_context, manager) = open(KvQuant::TurboQuant {
        k_bits: 4,
        v_bits: 4,
    });
    let k_bytes = vec![0u8; manager.k_stride(2)];
    let v_bytes = vec![0u8; manager.v_stride(2)];
    manager.write_k(2, 0, &k_bytes);
    manager.write_v(2, 0, &v_bytes);
}
