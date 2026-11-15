//! Muse Glimmer GGUF tensor name mappings.
//!
//! Every row read off the real `ggml-org/Muse-Glimmer-30B-GGUF` Q4_K_M header
//! (731 tensors: 14 per layer over 52 layers, plus `token_embd`,
//! `output_norm` and an UNTIED `output`) and cross-read against llama.cpp's
//! `src/models/muse-glimmer.cpp` and `conversion/muse_glimmer.py`. Three
//! things differ from a plain name map, and the first two are VALUE
//! conventions the walk applies, not names:
//!
//! - **The four per-layer norms carry `+1`.** The converter adds 1 to every
//!   tensor named `*layernorm.weight` (input, post-attention, pre-FFN,
//!   post-FFN); the final norm is not shifted. This family's flow applies
//!   `1 + w` itself (`encode_rms_norm_bf16w_centered`), so the install must
//!   hold the CENTERED value and the walk subtracts 1 again
//!   (`conventions::apply_source_convention`). Measured: GGUF minus the MLX
//!   conversion's tensor is exactly 1.0 on the four, 0.0 on the final norm.
//! - **q and k rows are permuted for ggml's interleaved RoPE**, the same
//!   permutation as `llama`'s, undone the same way
//!   (`conventions::needs_rotary_unpermute`).
//! - **The q/k norm tensors are synthesized.** The reference's norms are
//!   scale-free, followed by `qk_scale_factor` on Q alone, which the flow
//!   keeps as a constant. The converter materializes that as a `q_norm`
//!   filled with the factor and a `k_norm` of ones. They are not carried, but
//!   only after their VALUES match (`gguf_checkpoint::muse_qk_norms`), so a
//!   re-conversion with a different factor refuses instead of installing a
//!   model whose Q scale is silently wrong.

use super::{layer_prefix, GgufMapping};

pub fn map_muse_layer(suffix: &str, layer: usize) -> Option<GgufMapping> {
    let p = layer_prefix(layer);
    let resident = |tail: &str| Some(GgufMapping::Resident(format!("{p}{tail}")));
    match suffix {
        "attn_q.weight" => resident("self_attn.q_proj.weight"),
        "attn_k.weight" => resident("self_attn.k_proj.weight"),
        "attn_v.weight" => resident("self_attn.v_proj.weight"),
        // The attention OUTPUT gate: its own projection from the normed
        // input, `[hidden, heads * head_dim]`, not a packed q_proj.
        "attn_gate.weight" => resident("self_attn.gate_proj.weight"),
        "attn_output.weight" => resident("self_attn.o_proj.weight"),
        "attn_norm.weight" => resident("input_layernorm.weight"),
        "post_attention_norm.weight" => resident("post_attention_layernorm.weight"),
        // GGUF's `ffn_norm` is the PRE-FFN norm.
        "ffn_norm.weight" => resident("pre_feedforward_layernorm.weight"),
        "post_ffw_norm.weight" => resident("post_feedforward_layernorm.weight"),
        "attn_q_norm.weight" => Some(GgufMapping::MuseQkNorm { is_q: true }),
        "attn_k_norm.weight" => Some(GgufMapping::MuseQkNorm { is_q: false }),
        "ffn_gate.weight" => resident("mlp.gate_proj.weight"),
        "ffn_up.weight" => resident("mlp.up_proj.weight"),
        "ffn_down.weight" => resident("mlp.down_proj.weight"),
        _ => None,
    }
}
