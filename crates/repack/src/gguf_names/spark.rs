//! Spark-X2.5 GGUF tensor name mappings.
//!
//! Every row read off the real published
//! `XHToken/Spark-X2.5-4B-GGUF` Q4_K_M header (docs/SPARK_PHASE0.md): 290
//! tensors, 8 per layer, no biases, no `output.weight` (tied head). The
//! conventions are the upstream llama.cpp merge (PR 27868), not a fork.
//!
//! TWO THINGS DIFFER FROM THE `llama` TABLE NEXT DOOR.
//!
//! The QKV projection is FUSED: one `attn_qkv.weight` at `[2560, 6144]`,
//! q then k then v along the output dim (4096 | 1024 | 1024). It maps to a
//! canonical fused name and the runtime flow reads it as ONE tensor and
//! splits after the GEMV, the same shape as Qwen's packed `[q; gate]`
//! handling. The split lands on clean byte boundaries in every executable
//! block type (2560 elements is a whole number of Q4_K/Q6_K/Q8_0 blocks),
//! but this port does not need that: it never splits on disk.
//!
//! The attention output gate is its own tensor, `attn_gate.weight` at
//! `[2560, 16]` -- one SCALAR per head, a shape no other family's gate has.
//! It maps to the HF spelling `self_attn.g_proj.weight`.

use super::{layer_prefix, GgufMapping};

pub fn map_spark_layer(suffix: &str, layer: usize) -> Option<GgufMapping> {
    let p = layer_prefix(layer);
    let resident = |tail: &str| Some(GgufMapping::Resident(format!("{p}{tail}")));
    match suffix {
        // Fused QKV, q | k | v along the output dim. NOT split here: the
        // resident stays whole and `families/spark`'s flow owns the split.
        "attn_qkv.weight" => resident("self_attn.q_k_v_proj.weight"),
        "attn_gate.weight" => resident("self_attn.g_proj.weight"),
        "attn_output.weight" => resident("self_attn.o_proj.weight"),
        "attn_norm.weight" => resident("input_layernorm.weight"),
        // GGUF's `ffn_norm` is the PRE-FFN norm, which HF spells
        // `post_attention_layernorm` (same reversal as the `llama` table).
        "ffn_norm.weight" => resident("post_attention_layernorm.weight"),
        "ffn_gate.weight" => resident("mlp.gate_proj.weight"),
        "ffn_up.weight" => resident("mlp.up_proj.weight"),
        "ffn_down.weight" => resident("mlp.down_proj.weight"),
        _ => None,
    }
}
