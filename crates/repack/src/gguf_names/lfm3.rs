//! LFM3 GGUF tensor name mappings.
//!
//! Every row read off the real published `LiquidAI/LFM3-1.7B-GGUF` Q8_0 and
//! Q4_K_M headers (docs/LFM3_PHASE0.md), and every right-hand name is the
//! MLX checkpoint's own spelling (`LiquidAI/LFM3-1.7B-MLX-4bit`), so the two
//! intakes produce one resident index and one flow reads both.
//!
//! A layer carries EITHER the three `shortconv.*` tensors (13 conv layers)
//! OR the attention set (q/k/v/o, the headwise `attn_gate`, per-head q/k
//! norms), never both; the norms and the FFN are on every layer. No biases,
//! no `output.weight` (tied head).
//!
//! `shortconv.conv.weight` is ggml `[7, 2048]`, i.e. logical `[2048, 7]`:
//! channel-major with the taps contiguous, the same bytes the MLX
//! `[2048, 7, 1]` and HF `[2048, 1, 7]` layouts hold. It ships F32 and takes
//! the BF16 default like the norms.

use super::{layer_prefix, GgufMapping};

pub fn map_lfm3_layer(suffix: &str, layer: usize) -> Option<GgufMapping> {
    let p = layer_prefix(layer);
    let resident = |tail: &str| Some(GgufMapping::Resident(format!("{p}{tail}")));
    match suffix {
        "shortconv.in_proj.weight" => resident("conv.in_proj.weight"),
        "shortconv.conv.weight" => resident("conv.conv.weight"),
        "shortconv.out_proj.weight" => resident("conv.out_proj.weight"),
        "attn_q.weight" => resident("self_attn.q_proj.weight"),
        "attn_k.weight" => resident("self_attn.k_proj.weight"),
        "attn_v.weight" => resident("self_attn.v_proj.weight"),
        "attn_output.weight" => resident("self_attn.o_proj.weight"),
        // `[hidden, num_heads]`: one sigmoid scalar per head.
        "attn_gate.weight" => resident("self_attn.gate_proj.weight"),
        "attn_q_norm.weight" => resident("self_attn.q_norm.weight"),
        "attn_k_norm.weight" => resident("self_attn.k_norm.weight"),
        // GGUF `attn_norm` is the PRE-mixer norm for conv and attention
        // layers alike; HF calls it `operator_norm`.
        "attn_norm.weight" => resident("operator_norm.weight"),
        // The two sandwich norms. llama.cpp names them after the Gemma
        // tensors that share the role.
        "post_attention_norm.weight" => resident("post_operator_norm.weight"),
        "post_ffw_norm.weight" => resident("post_ffn_norm.weight"),
        "ffn_norm.weight" => resident("ffn_norm.weight"),
        // `w1` is the gate, `w3` the up, `w2` the down (`Lfm3MLP`).
        "ffn_gate.weight" => resident("feed_forward.w1.weight"),
        "ffn_up.weight" => resident("feed_forward.w3.weight"),
        "ffn_down.weight" => resident("feed_forward.w2.weight"),
        _ => None,
    }
}

/// The final norm. HF and MLX name it `embedding_norm` although it runs
/// AFTER the last layer (`Lfm3Model.forward`), and the install keeps the
/// MLX spelling rather than the shared `model.norm` row.
pub fn map_lfm3_top_level(name: &str) -> Option<GgufMapping> {
    match name {
        "output_norm.weight" => Some(GgufMapping::Resident(
            "language_model.model.embedding_norm.weight".to_string(),
        )),
        // The zero-tail partial-rotary encoding, NOT learned scaling: the
        // walk reads its bytes and refuses anything but the exact pattern.
        "rope_freqs.weight" => Some(GgufMapping::ZeroTailRopeFreqs),
        _ => None,
    }
}
