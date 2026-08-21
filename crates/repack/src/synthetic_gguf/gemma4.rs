use super::builder::{GgufBuilder, GgufFileAndRanges};
use crate::gguf_header::GgufValue;

pub use super::gemma4_shape::{QuantMix, SyntheticGgufShape};

/// A complete, tiny Gemma 4 GGUF: every tensor name and metadata key the
/// real `gemma-4-26B-A4B-it-Q8_0.gguf` carries, at toy dimensions.
///
/// The name and metadata inventory is copied from the real file (see
/// `gguf_names.rs`), including the parts that are easy to get wrong: gate
/// and up FUSED into `ffn_gate_up_exps`, the router's per-expert scale
/// hiding under `ffn_down_exps.scale`, no `output.weight` because Gemma
/// ties its embeddings, and NO `attn_v` on global layers.
pub fn build_synthetic_gemma4_gguf(shape: SyntheticGgufShape) -> GgufFileAndRanges {
    let s = shape;
    let mut b = GgufBuilder::new()
        .metadata_str("general.architecture", "gemma4")
        .metadata_u32("gemma4.block_count", s.num_layers as u32)
        .metadata_u32("gemma4.embedding_length", s.hidden as u32)
        .metadata_u32("gemma4.attention.head_count", s.num_heads as u32)
        .metadata_u32("gemma4.expert_count", s.num_experts as u32)
        .metadata_u32("gemma4.expert_used_count", s.top_k as u32)
        .metadata_u32(
            "gemma4.expert_feed_forward_length",
            s.moe_intermediate as u32,
        )
        .metadata_u32("gemma4.feed_forward_length", s.intermediate as u32)
        .metadata_u32("gemma4.attention.key_length", s.full_head_dim as u32)
        .metadata_u32("gemma4.attention.key_length_swa", s.head_dim as u32)
        .metadata_u32("gemma4.attention.sliding_window", s.sliding_window as u32)
        .metadata_f32("gemma4.final_logit_softcapping", 30.0)
        .metadata_f32("gemma4.rope.freq_base", 1_000_000.0)
        .metadata_f32("gemma4.rope.freq_base_swa", 10_000.0)
        .metadata(
            "gemma4.attention.sliding_window_pattern",
            GgufValue::Array(
                (0..s.num_layers)
                    .map(|l| GgufValue::Bool(s.slides(l)))
                    .collect(),
            ),
        )
        .metadata(
            "gemma4.attention.head_count_kv",
            GgufValue::Array(
                (0..s.num_layers)
                    .map(|l| {
                        GgufValue::I32(if s.slides(l) {
                            s.num_kv_heads as i32
                        } else {
                            s.num_full_kv_heads as i32
                        })
                    })
                    .collect(),
            ),
        )
        .f32_upcast_bf16_tensor("output_norm.weight", &[s.hidden], 2);
    b = embed_tensor(b, &s, "token_embd.weight", &[s.hidden, s.vocab], 1);

    for l in 0..s.num_layers {
        let sliding = s.slides(l);
        let hd = if sliding { s.head_dim } else { s.full_head_dim };
        let kv = if sliding {
            s.num_kv_heads
        } else {
            s.num_full_kv_heads
        };
        let q_dim = s.num_heads * hd;
        let seed = (l as u8).wrapping_mul(37).wrapping_add(3);

        b = attn_tensor(
            b,
            &s,
            &format!("blk.{l}.attn_q.weight"),
            &[s.hidden, q_dim],
            seed,
        );
        b = attn_tensor(
            b,
            &s,
            &format!("blk.{l}.attn_k.weight"),
            &[s.hidden, kv * hd],
            seed.wrapping_add(1),
        );
        b = attn_tensor(
            b,
            &s,
            &format!("blk.{l}.attn_output.weight"),
            &[q_dim, s.hidden],
            seed.wrapping_add(2),
        );
        b = b
            .f32_upcast_bf16_tensor(&format!("blk.{l}.attn_q_norm.weight"), &[hd], seed)
            .f32_upcast_bf16_tensor(
                &format!("blk.{l}.attn_k_norm.weight"),
                &[hd],
                seed.wrapping_add(1),
            );

        // Global layers carry no V projection at all - a property of the
        // model, reproduced here so the walk is exercised against it.
        if sliding {
            b = attn_tensor(
                b,
                &s,
                &format!("blk.{l}.attn_v.weight"),
                &[s.hidden, kv * hd],
                seed.wrapping_add(3),
            );
        }

        for (n, norm) in [
            "attn_norm",
            "post_attention_norm",
            "ffn_norm",
            "pre_ffw_norm_2",
            "post_ffw_norm",
            "post_ffw_norm_1",
            "post_ffw_norm_2",
        ]
        .iter()
        .enumerate()
        {
            b = b.f32_upcast_bf16_tensor(
                &format!("blk.{l}.{norm}.weight"),
                &[s.hidden],
                seed.wrapping_add(10 + n as u8),
            );
        }

        b = b
            .q8_0_tensor(
                &format!("blk.{l}.ffn_gate.weight"),
                &[s.hidden, s.intermediate],
                seed.wrapping_add(4),
            )
            .q8_0_tensor(
                &format!("blk.{l}.ffn_up.weight"),
                &[s.hidden, s.intermediate],
                seed.wrapping_add(5),
            )
            .q8_0_tensor(
                &format!("blk.{l}.ffn_down.weight"),
                &[s.intermediate, s.hidden],
                seed.wrapping_add(6),
            )
            // Router: F32 in GGUF, where an MLX install carries INT8. Real
            // values rather than upcast BF16, because the repack quantizes
            // this one instead of narrowing it.
            .f32_tensor(
                &format!("blk.{l}.ffn_gate_inp.weight"),
                &[s.hidden, s.num_experts],
                seed.wrapping_add(20),
            )
            .f32_upcast_bf16_tensor(
                &format!("blk.{l}.ffn_gate_inp.scale"),
                &[s.hidden],
                seed.wrapping_add(21),
            )
            .f32_upcast_bf16_tensor(
                &format!("blk.{l}.ffn_down_exps.scale"),
                &[s.num_experts],
                seed.wrapping_add(22),
            )
            .f32_upcast_bf16_tensor(
                &format!("blk.{l}.layer_output_scale.weight"),
                &[1],
                seed.wrapping_add(23),
            );

        // Gate and up fused along the output dim.
        b = expert_tensor(
            b,
            &s,
            &format!("blk.{l}.ffn_gate_up_exps.weight"),
            &[s.hidden, 2 * s.moe_intermediate, s.num_experts],
            seed.wrapping_add(7),
            l,
            true,
        );
        b = expert_tensor(
            b,
            &s,
            &format!("blk.{l}.ffn_down_exps.weight"),
            &[s.moe_intermediate, s.hidden, s.num_experts],
            seed.wrapping_add(8),
            l,
            false,
        );
    }

    b.build()
}

/// The roles a mixed fixture moves off Q8_0, each mirroring where the real
/// file it models puts that block type. Written as functions rather than
/// methods on the builder because the choice belongs to the fixture's shape,
/// not to GGUF.
fn embed_tensor(
    b: GgufBuilder,
    s: &SyntheticGgufShape,
    name: &str,
    dims: &[u64],
    seed: u8,
) -> GgufBuilder {
    match s.mix {
        QuantMix::Q8_0 => b.q8_0_tensor(name, dims, seed),
        QuantMix::KQuant => b.q4_k_tensor(name, dims, seed),
        // The Phase S candidate keeps `token_embd` at Q6_K and ties the head
        // to it, which is what made `embed_lookup_q6_k` worth writing.
        QuantMix::Iq => b.q6_k_tensor(name, dims, seed),
        // gpt-oss keeps `token_embd` at Q8_0 and unties its head.
        QuantMix::Mxfp4 => b.q8_0_tensor(name, dims, seed),
    }
}

/// A routed expert tensor. Takes the LAYER and which half it is, because
/// [`QuantMix::Iq`] is the first mixture where those matter: its phases carry
/// different types from each other, and its last layer differs from the rest.
fn expert_tensor(
    b: GgufBuilder,
    s: &SyntheticGgufShape,
    name: &str,
    dims: &[u64],
    seed: u8,
    layer: usize,
    gate_up: bool,
) -> GgufBuilder {
    match (s.mix, gate_up, s.odd_expert_layer(layer)) {
        (QuantMix::Q8_0, _, _) => b.q8_0_tensor(name, dims, seed),
        (QuantMix::KQuant, _, _) => b.q4_k_tensor(name, dims, seed),
        (QuantMix::Iq, true, false) => b.iq_tensor(name, 18, dims, seed),
        (QuantMix::Iq, true, true) => b.iq_tensor(name, 23, dims, seed),
        (QuantMix::Iq, false, false) => b.iq_tensor(name, 20, dims, seed),
        (QuantMix::Iq, false, true) => b.q8_0_tensor(name, dims, seed),
        // Both phases, every layer: gpt-oss is uniform in a way the two
        // mixtures above deliberately are not.
        (QuantMix::Mxfp4, _, _) => b.mxfp4_tensor(name, dims, seed),
    }
}

/// Q6_K, where a real `Q4_K_M` would carry it on `output.weight`. A Gemma
/// fixture ties its embeddings and so has no such tensor, and the attention
/// projections are the next place a resident GEMV reads every token. The IQ
/// mixture leaves attention at Q8_0, as its candidate does.
fn attn_tensor(
    b: GgufBuilder,
    s: &SyntheticGgufShape,
    name: &str,
    dims: &[u64],
    seed: u8,
) -> GgufBuilder {
    match s.mix {
        QuantMix::KQuant => b.q6_k_tensor(name, dims, seed),
        QuantMix::Q8_0 | QuantMix::Iq | QuantMix::Mxfp4 => b.q8_0_tensor(name, dims, seed),
    }
}
