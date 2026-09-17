//! `deepseek2` GGUF tensor name mappings, verified against
//! `mradermacher/DeepSeek-V2-Lite-Chat-GGUF`'s Q8_0 header
//! (`docs/DEEPSEEK2_PHASE0.md`). 27 layers: one dense LEAD FFN, then 26
//! MoE layers with a fused shared expert and MLA attention.

use super::{layer_prefix, GgufMapping};

/// The MLA projections have no neighbours here: this is the only family
/// whose attention compresses its KV, so `attn_kv_a_mqa` / `attn_kv_b` /
/// `attn_kv_a_norm` are new canonical names, kept whole like spark's fused
/// `q_k_v_proj` -- the resident stays what the file stores and the flow
/// owns the split.
///
/// The MoE rows are `map_qwen3moe_layer`'s four (router, three routed) plus
/// the three shared-expert rows `map_qwen_gdn_moe_layer` carries: DeepSeek
/// is the first GGUF family with BOTH. The dense lead layer's plain
/// `ffn_gate`/`ffn_up`/`ffn_down` take the llama table's dense triple, so
/// one table covers both layer kinds and the classifier sorts by presence
/// of the routed tensors, as llama.cpp's own loader does.
pub fn map_deepseek2_layer(suffix: &str, layer: usize) -> Option<GgufMapping> {
    let p = layer_prefix(layer);
    let resident = |tail: &str| Some(GgufMapping::Resident(format!("{p}{tail}")));
    match suffix {
        // Attention (MLA). `attn_q` is [hidden, heads*(nope+rope)] and
        // stays whole; the per-head nope/pe split is the flow's job.
        "attn_q.weight" => resident("self_attn.q_proj.weight"),
        // The down-projection [hidden, latent+rope] and its latent norm.
        "attn_kv_a_mqa.weight" => resident("self_attn.kv_a_proj.weight"),
        "attn_kv_a_norm.weight" => resident("self_attn.kv_a_norm.weight"),
        // The up-projection [latent, heads*(nope+v)]: absorbed as W_k/W_v
        // per head by the flow, never expanded at repack.
        "attn_kv_b.weight" => resident("self_attn.kv_b_proj.weight"),
        "attn_output.weight" => resident("self_attn.o_proj.weight"),
        "attn_norm.weight" => resident("input_layernorm.weight"),
        "ffn_norm.weight" => resident("post_attention_layernorm.weight"),
        // Router (F32 in the file; transcodes to INT8 like qwen3moe's).
        "ffn_gate_inp.weight" => resident("mlp.gate.weight"),
        // Routed experts, separate gate/up, exactly the qwen3moe shape.
        "ffn_gate_exps.weight" => Some(GgufMapping::Routed {
            layer,
            role: "gate",
        }),
        "ffn_up_exps.weight" => Some(GgufMapping::Routed { layer, role: "up" }),
        "ffn_down_exps.weight" => Some(GgufMapping::Routed {
            layer,
            role: "down",
        }),
        // The fused shared expert (n_shared experts concatenated along the
        // output dim), a RESIDENT plain SwiGLU.
        "ffn_gate_shexp.weight" => resident("mlp.shared_expert.gate_proj.weight"),
        "ffn_up_shexp.weight" => resident("mlp.shared_expert.up_proj.weight"),
        "ffn_down_shexp.weight" => resident("mlp.shared_expert.down_proj.weight"),
        // The dense lead layer's FFN (layer 0 only, but a name table is
        // not per-layer-indexed and the real file only ever pairs these
        // with a layer that has no routed tensors).
        "ffn_gate.weight" => resident("mlp.gate_proj.weight"),
        "ffn_up.weight" => resident("mlp.up_proj.weight"),
        "ffn_down.weight" => resident("mlp.down_proj.weight"),
        _ => None,
    }
}
