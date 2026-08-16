//! Gemma 4 GGUF tensor name mappings.

use super::{layer_prefix, GgufMapping};

/// Gemma 4's per-layer suffixes, verified against
/// `gemma-4-26B-A4B-it-Q8_0.gguf` and `~/models/gemma4.gturbo`.
pub fn map_gemma4_layer(suffix: &str, layer: usize) -> Option<GgufMapping> {
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
        // GGUF calls the pre-FFN norm `ffn_norm` and only numbers the
        // SECOND one; the install spells both out.
        "ffn_norm.weight" => resident("pre_feedforward_layernorm.weight"),
        "pre_ffw_norm_2.weight" => resident("pre_feedforward_layernorm_2.weight"),
        "post_ffw_norm.weight" => resident("post_feedforward_layernorm.weight"),
        "post_ffw_norm_1.weight" => resident("post_feedforward_layernorm_1.weight"),
        "post_ffw_norm_2.weight" => resident("post_feedforward_layernorm_2.weight"),
        // The dense/shared-expert FFN.
        "ffn_gate.weight" => resident("mlp.gate_proj.weight"),
        "ffn_up.weight" => resident("mlp.up_proj.weight"),
        "ffn_down.weight" => resident("mlp.down_proj.weight"),
        // Router. The two `.scale` tensors are matched by shape against the
        // install: `ffn_gate_inp.scale` is [hidden] like `router.scale`, and
        // `ffn_down_exps.scale` is [experts] like `router.per_expert_scale`
        // -- note that the latter is a ROUTER tensor despite its name
        // living under the down-projection's prefix.
        "ffn_gate_inp.weight" => resident("router.proj.weight"),
        "ffn_gate_inp.scale" => resident("router.scale"),
        "ffn_down_exps.scale" => resident("router.per_expert_scale"),
        "layer_output_scale.weight" => resident("layer_scalar"),
        "ffn_gate_up_exps.weight" => Some(GgufMapping::RoutedFusedGateUp { layer }),
        "ffn_down_exps.weight" => Some(GgufMapping::Routed {
            layer,
            role: "down",
        }),
        _ => None,
    }
}
