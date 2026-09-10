//! MiniMax-M2's 13 per-layer tensor roles, witnessed across all three shards.
use super::{layer_prefix, GgufMapping};

pub fn map_layer(suffix: &str, layer: usize) -> Option<GgufMapping> {
    let tail = match suffix {
        "attn_q_norm.weight" => "self_attn.q_norm.weight",
        "attn_k_norm.weight" => "self_attn.k_norm.weight",
        "exp_probs_b.bias" => "mlp.e_score_correction_bias",
        // MiniMax has no dense or shared FFN; do not accept those names.
        "ffn_gate.weight" | "ffn_up.weight" | "ffn_down.weight" => return None,
        _ => return super::llama::map_llama_layer(suffix, layer),
    };
    Some(GgufMapping::Resident(format!(
        "{}{tail}",
        layer_prefix(layer)
    )))
}
