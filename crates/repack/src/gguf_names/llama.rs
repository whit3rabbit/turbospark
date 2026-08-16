//! LLaMA / Mixtral / Mistral GGUF tensor name mappings.

use super::{layer_prefix, GgufMapping};

/// The `llama` architecture's per-layer suffixes, verified against
/// `Mixtral-8x7B-Instruct-v0.1.Q4_K_M.gguf` and
/// `Meta-Llama-3.1-8B-Instruct-Q6_K.gguf` (ROADMAP Phase M2).
///
/// ONE TABLE FOR BOTH HALVES of the architecture, which is the whole reason
/// the family is worth having: a dense Llama carries `ffn_gate`/`ffn_up`/
/// `ffn_down` and no router, a Mixtral carries `ffn_gate_inp` plus the three
/// `_exps` tensors and no dense FFN, and nothing else differs. Neither half
/// carries q/k norms, and neither carries a shared expert.
pub fn map_llama_layer(suffix: &str, layer: usize) -> Option<GgufMapping> {
    let p = layer_prefix(layer);
    let resident = |tail: &str| Some(GgufMapping::Resident(format!("{p}{tail}")));
    match suffix {
        "attn_q.weight" => resident("self_attn.q_proj.weight"),
        "attn_k.weight" => resident("self_attn.k_proj.weight"),
        "attn_v.weight" => resident("self_attn.v_proj.weight"),
        "attn_output.weight" => resident("self_attn.o_proj.weight"),
        "attn_norm.weight" => resident("input_layernorm.weight"),
        // GGUF's `ffn_norm` is the PRE-FFN norm, which HF and this install
        // spell `post_attention_layernorm`. The two names describe the same
        // position from opposite sides.
        "ffn_norm.weight" => resident("post_attention_layernorm.weight"),
        // The dense half (Llama 2/3.x, Mistral).
        "ffn_gate.weight" => resident("mlp.gate_proj.weight"),
        "ffn_up.weight" => resident("mlp.up_proj.weight"),
        "ffn_down.weight" => resident("mlp.down_proj.weight"),
        // The MoE half (Mixtral). Same router slot name Qwen uses.
        "ffn_gate_inp.weight" => resident("mlp.gate.weight"),
        "ffn_gate_exps.weight" => Some(GgufMapping::Routed {
            layer,
            role: "gate",
        }),
        "ffn_up_exps.weight" => Some(GgufMapping::Routed { layer, role: "up" }),
        "ffn_down_exps.weight" => Some(GgufMapping::Routed {
            layer,
            role: "down",
        }),
        _ => None,
    }
}

/// True for a pre-merge per-expert tensor suffix: `ffn_down.3.weight` and
/// friends, which a 2023-era Mixtral conversion carries 256 of per role.
///
/// Matched on SHAPE rather than by listing indices, since the expert count
/// varies by model, and kept separate from the table above so the refusal can
/// say what is wrong instead of "no mapping".
pub fn is_pre_merge_expert(suffix: &str) -> bool {
    let mut parts = suffix.split('.');
    let head = parts.next().unwrap_or_default();
    if !matches!(head, "ffn_gate" | "ffn_up" | "ffn_down") {
        return false;
    }
    let Some(index) = parts.next() else {
        return false;
    };
    !index.is_empty()
        && index.bytes().all(|b| b.is_ascii_digit())
        && parts.next() == Some("weight")
        && parts.next().is_none()
}
