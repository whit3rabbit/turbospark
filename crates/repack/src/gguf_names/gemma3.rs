//! Gemma 3 GGUF tensor name mappings.
//!
//! Read off the real `ggml-org/gemma-3-1b-it-GGUF` Q4_K_M header (340
//! tensors: 13 per layer over 26 blocks, plus `token_embd` and `output_norm`;
//! no `output.weight`, the head is tied). The canonical spellings are the
//! Gemma 4 install's, so the one flow reads both. Everything Gemma 4 adds is
//! deliberately absent: no router, no fused experts, no `layer_output_scale`.
//! A file carrying one is another architecture and is refused as unmapped.
//!
//! The VALUE convention is in `gguf_checkpoint::conventions`: every
//! `*norm.weight` carries llama.cpp's baked `+1`, which the walk removes
//! because this family's flow applies `1 + w` itself.

use super::{layer_prefix, GgufMapping};

pub fn map_gemma3_layer(suffix: &str, layer: usize) -> Option<GgufMapping> {
    let p = layer_prefix(layer);
    let resident = |tail: &str| Some(GgufMapping::Resident(format!("{p}{tail}")));
    match suffix {
        "attn_q.weight" => resident("self_attn.q_proj.weight"),
        "attn_k.weight" => resident("self_attn.k_proj.weight"),
        "attn_v.weight" => resident("self_attn.v_proj.weight"),
        "attn_output.weight" => resident("self_attn.o_proj.weight"),
        "attn_q_norm.weight" => resident("self_attn.q_norm.weight"),
        "attn_k_norm.weight" => resident("self_attn.k_norm.weight"),
        "attn_norm.weight" => resident("input_layernorm.weight"),
        "post_attention_norm.weight" => resident("post_attention_layernorm.weight"),
        // GGUF's `ffn_norm` is the PRE-FFN norm.
        "ffn_norm.weight" => resident("pre_feedforward_layernorm.weight"),
        "post_ffw_norm.weight" => resident("post_feedforward_layernorm.weight"),
        "ffn_gate.weight" => resident("mlp.gate_proj.weight"),
        "ffn_up.weight" => resident("mlp.up_proj.weight"),
        "ffn_down.weight" => resident("mlp.down_proj.weight"),
        _ => None,
    }
}
