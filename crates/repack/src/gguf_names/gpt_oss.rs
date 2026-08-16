//! GPT-OSS GGUF tensor name mappings.

use super::{layer_prefix, GgufMapping};

/// `gpt-oss` (ROADMAP M5). Every row read off the real published
/// `gpt-oss-20b-MXFP4.gguf` header, `blk.0` and `blk.1` both, since the
/// window alternates and the two halves of the pattern did not have to carry
/// the same tensors (they do).
///
/// THREE THINGS DIFFER FROM THE `llama` TABLE NEXT DOOR and each would
/// otherwise surface as an unmapped name mid-walk.
///
/// The post-attention norm is `post_attention_norm`, not `ffn_norm`. Both
/// map to the same canonical name; only the GGUF spelling differs.
///
/// EVERY PROJECTION HAS A BIAS, which no family here has ever had, and so
/// does the router. Those are resident and take the ordinary `.bias`
/// canonical spelling.
///
/// THE PER-EXPERT BIASES GO INTO THE BLOB, under the `*_biases` roles the
/// INT4-affine layout already defines and every GGUF install so far has left
/// empty. That is not a pun on the name: `MoeExpertOffsets` has three unused
/// bias fields, `moe_offsets_from_layout` already resolves them through
/// `companion()` (absent means 0), and the affine-vs-GGUF discriminator keys
/// on `gate_scales` rather than on these, so a blob with biases and no scales
/// is still correctly read as GGUF. Packing them per expert also means the
/// bias travels with the weights it belongs to: the streamer reads one
/// contiguous blob per miss and the kernel never needs to know which EXPERT a
/// slot holds. It costs 0.27% of the blob (34.6 KiB against 12.6 MiB).
pub fn map_gpt_oss_layer(suffix: &str, layer: usize) -> Option<GgufMapping> {
    let p = layer_prefix(layer);
    let resident = |tail: &str| Some(GgufMapping::Resident(format!("{p}{tail}")));
    match suffix {
        "attn_q.weight" => resident("self_attn.q_proj.weight"),
        "attn_k.weight" => resident("self_attn.k_proj.weight"),
        "attn_v.weight" => resident("self_attn.v_proj.weight"),
        "attn_output.weight" => resident("self_attn.o_proj.weight"),
        "attn_q.bias" => resident("self_attn.q_proj.bias"),
        "attn_k.bias" => resident("self_attn.k_proj.bias"),
        "attn_v.bias" => resident("self_attn.v_proj.bias"),
        "attn_output.bias" => resident("self_attn.o_proj.bias"),
        // One learned logit per q head, added to the softmax denominator.
        "attn_sinks.weight" => resident("self_attn.sinks.weight"),
        "attn_norm.weight" => resident("input_layernorm.weight"),
        // The one spelling difference from `llama`'s `ffn_norm`.
        "post_attention_norm.weight" => resident("post_attention_layernorm.weight"),
        "ffn_gate_inp.weight" => resident("mlp.gate.weight"),
        "ffn_gate_inp.bias" => resident("mlp.gate.bias"),
        "ffn_gate_exps.weight" => Some(GgufMapping::Routed {
            layer,
            role: "gate",
        }),
        "ffn_up_exps.weight" => Some(GgufMapping::Routed { layer, role: "up" }),
        "ffn_down_exps.weight" => Some(GgufMapping::Routed {
            layer,
            role: "down",
        }),
        "ffn_gate_exps.bias" => Some(GgufMapping::Routed {
            layer,
            role: "gate_biases",
        }),
        "ffn_up_exps.bias" => Some(GgufMapping::Routed {
            layer,
            role: "up_biases",
        }),
        "ffn_down_exps.bias" => Some(GgufMapping::Routed {
            layer,
            role: "down_biases",
        }),
        _ => None,
    }
}
