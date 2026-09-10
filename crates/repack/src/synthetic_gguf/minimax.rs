//! Small real-named MiniMax GGUF: full-width norms, FP32 routing and mixed K-quants.
use super::{GgufBuilder, GgufFileAndRanges};

pub fn build_synthetic_minimax_gguf() -> GgufFileAndRanges {
    let mut b = GgufBuilder::new().metadata_str("general.architecture", "minimax-m2");
    for (key, n) in [
        ("block_count", 2),
        ("embedding_length", 256),
        ("feed_forward_length", 256),
        ("expert_feed_forward_length", 256),
        ("attention.head_count", 4),
        ("attention.head_count_kv", 2),
        ("attention.key_length", 128),
        ("attention.value_length", 128),
        ("rope.dimension_count", 64),
        ("expert_count", 16),
        ("expert_used_count", 8),
        ("expert_gating_func", 2),
    ] {
        b = b.metadata_u32(&format!("minimax-m2.{key}"), n);
    }
    b = b
        .metadata_f32("minimax-m2.rope.freq_base", 5_000_000.0)
        .metadata_f32("minimax-m2.attention.layer_norm_rms_epsilon", 1e-6)
        .q4_k_tensor("token_embd.weight", &[256, 64], 3)
        .q6_k_tensor("output.weight", &[256, 64], 4);
    let norm = |b: GgufBuilder, name: &str, n: u64| {
        let bytes = (0..n)
            .flat_map(|i| (0.75f32 + (i % 23) as f32 / 32.0).to_le_bytes())
            .collect();
        b.tensor(name, 0, &[n], bytes)
    };
    b = norm(b, "output_norm.weight", 256);
    for l in 0..2 {
        let p = format!("blk.{l}.");
        for (tail, n) in [
            ("attn_norm.weight", 256),
            ("ffn_norm.weight", 256),
            ("attn_q_norm.weight", 512),
            ("attn_k_norm.weight", 256),
        ] {
            b = norm(b, &format!("{p}{tail}"), n);
        }
        for (tail, cols, rows, seed) in [
            ("attn_q.weight", 256, 512, 11),
            ("attn_k.weight", 256, 256, 12),
            ("attn_v.weight", 256, 256, 13),
            ("attn_output.weight", 512, 256, 14),
        ] {
            b = b.q4_k_tensor(&format!("{p}{tail}"), &[cols, rows], seed + l);
        }
        let router = (0..256 * 16)
            .flat_map(|i| (((i * 31 + l as usize * 7) % 113) as f32 / 512.0 - 0.11).to_le_bytes())
            .collect();
        let bias = (0..16)
            .flat_map(|i| (i as f32 * 0.003_013).to_le_bytes())
            .collect();
        b = b
            .tensor(&format!("{p}ffn_gate_inp.weight"), 0, &[256, 16], router)
            .tensor(&format!("{p}exp_probs_b.bias"), 0, &[16], bias)
            .q4_k_tensor(&format!("{p}ffn_gate_exps.weight"), &[256, 256, 16], 20 + l)
            .q4_k_tensor(&format!("{p}ffn_up_exps.weight"), &[256, 256, 16], 30 + l);
        b = if l == 0 {
            b.q6_k_tensor(&format!("{p}ffn_down_exps.weight"), &[256, 256, 16], 40 + l)
        } else {
            b.q4_k_tensor(&format!("{p}ffn_down_exps.weight"), &[256, 256, 16], 40 + l)
        };
    }
    b.build()
}
