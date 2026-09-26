//! Qwen GGUF tensor name mappings (Qwen 3.6 GDN MoE and Qwen3-MoE).

use super::{layer_prefix, GgufMapping};

/// Qwen3.8-Flash-Next's explicit GGUF-to-runtime name map. It deliberately
/// lists the whole inventory because its hyper-connections, indexer and PLE
/// tensors do not follow the older Qwen GDN names.
pub fn map_qwen4_exp_layer(suffix: &str, layer: usize) -> Option<GgufMapping> {
    let p = layer_prefix(layer);
    let resident = |tail: &str| Some(GgufMapping::Resident(format!("{p}{tail}")));
    match suffix {
        "attn_q.weight" => resident("self_attn.q_proj.weight"),
        "attn_k.weight" => resident("self_attn.k_proj.weight"),
        "attn_v.weight" => resident("self_attn.v_proj.weight"),
        "attn_output.weight" => resident("self_attn.o_proj.weight"),
        "attn_q_norm.weight" => resident("self_attn.q_norm.weight"),
        "attn_k_norm.weight" => resident("self_attn.k_norm.weight"),
        "attn_qkv.weight" => resident("linear_attn.in_proj_qkv.weight"),
        "attn_gate.weight" => resident("linear_attn.in_proj_z.weight"),
        "ssm_alpha.weight" => resident("linear_attn.in_proj_a.weight"),
        "ssm_beta.weight" => resident("linear_attn.in_proj_b.weight"),
        "ssm_out.weight" => resident("linear_attn.out_proj.weight"),
        "ssm_conv1d.weight" => resident("linear_attn.conv1d.weight"),
        "ssm_norm.weight" => resident("linear_attn.norm.weight"),
        "ssm_a" => resident("linear_attn.A_log"),
        "ssm_dt.bias" => resident("linear_attn.dt_bias"),
        "ffn_gate_inp.weight" => resident("mlp.gate.weight"),
        "ffn_gate_inp_shexp.weight" => resident("mlp.shared_expert_gate.weight"),
        "ffn_gate_shexp.weight" => resident("mlp.shared_expert.gate_proj.weight"),
        "ffn_up_shexp.weight" => resident("mlp.shared_expert.up_proj.weight"),
        "ffn_down_shexp.weight" => resident("mlp.shared_expert.down_proj.weight"),
        "ffn_gate_exps.weight" => Some(GgufMapping::Routed {
            layer,
            role: "gate",
        }),
        "ffn_up_exps.weight" => Some(GgufMapping::Routed { layer, role: "up" }),
        "ffn_down_exps.weight" => Some(GgufMapping::Routed {
            layer,
            role: "down",
        }),
        "hc_attn_norm.weight" => resident("attn_hyper_connection.hc_norm.weight"),
        "hc_attn_down.weight" => resident("attn_hyper_connection.input_mix_weight_down.weight"),
        "hc_attn_up.weight" => resident("attn_hyper_connection.input_mix_weight_up.weight"),
        "hc_attn_inject.weight" => resident("attn_hyper_connection.block_inject_weight.weight"),
        "hc_ffn_norm.weight" => resident("mlp_hyper_connection.hc_norm.weight"),
        "hc_ffn_down.weight" => resident("mlp_hyper_connection.input_mix_weight_down.weight"),
        "hc_ffn_up.weight" => resident("mlp_hyper_connection.input_mix_weight_up.weight"),
        "hc_ffn_inject.weight" => resident("mlp_hyper_connection.block_inject_weight.weight"),
        "indexer.q_proj.weight" => resident("self_attn.indexer.q_proj.weight"),
        "indexer.k_proj.weight" => resident("self_attn.indexer.k_proj.weight"),
        "indexer.q_norm.weight" => resident("self_attn.indexer.q_layernorm.weight"),
        "indexer.k_norm.weight" => resident("self_attn.indexer.k_layernorm.weight"),
        "ple_conv1d.weight" => resident("ple.conv1d.weight"),
        "ple_key.weight" => resident("ple.key_proj.weight"),
        "ple_norm_conv.weight" => resident("ple.norm_conv.weight"),
        "ple_norm_key.weight" => resident("ple.norm_key.weight"),
        "ple_norm_query.weight" => resident("ple.norm_query.weight"),
        "ple_value.weight" => resident("ple.value_proj.weight"),
        _ => None,
    }
}

/// The final hyper-connection mixer lives at model scope in the runtime.
pub fn map_qwen4_exp_top_level(name: &str) -> Option<GgufMapping> {
    let prefix = "language_model.model.hyper_connection_mixer.";
    let suffix = match name {
        "output_hc_norm.weight" => "hc_norm.weight",
        "output_hc_down.weight" => "input_mix_weight_down.weight",
        "output_hc_up.weight" => "input_mix_weight_up.weight",
        _ => return None,
    };
    Some(GgufMapping::Resident(format!("{prefix}{suffix}")))
}

/// Qwen 3.6's per-layer suffixes, verified against
/// `Qwen3.6-35B-A3B-Q4_K_M.gguf` and `~/models/qwen36.gturbo`. Note the
/// hybrid layer split: the 10 full-attention layers carry `attn_q/k/v`, the
/// 30 linear-attention layers carry `attn_qkv` plus the `ssm_*` family, and
/// no layer carries both.
pub fn map_qwen_gdn_moe_layer(suffix: &str, layer: usize) -> Option<GgufMapping> {
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
        // Gated DeltaNet. GGUF names these after the SSM family it borrows
        // its tensor slots from; the install names them after what they do.
        "attn_qkv.weight" => resident("linear_attn.in_proj_qkv.weight"),
        "attn_gate.weight" => resident("linear_attn.in_proj_z.weight"),
        "ssm_alpha.weight" => resident("linear_attn.in_proj_a.weight"),
        "ssm_beta.weight" => resident("linear_attn.in_proj_b.weight"),
        "ssm_out.weight" => resident("linear_attn.out_proj.weight"),
        "ssm_conv1d.weight" => resident("linear_attn.conv1d.weight"),
        "ssm_norm.weight" => resident("linear_attn.norm.weight"),
        // Gotcha 26: no `.weight` suffix on either of these, on either side.
        "ssm_a" => resident("linear_attn.A_log"),
        "ssm_dt.bias" => resident("linear_attn.dt_bias"),
        "ffn_gate_inp.weight" => resident("mlp.gate.weight"),
        "ffn_gate_inp_shexp.weight" => resident("mlp.shared_expert_gate.weight"),
        "ffn_gate_shexp.weight" => resident("mlp.shared_expert.gate_proj.weight"),
        "ffn_up_shexp.weight" => resident("mlp.shared_expert.up_proj.weight"),
        "ffn_down_shexp.weight" => resident("mlp.shared_expert.down_proj.weight"),
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

/// The DENSE `qwen35` half's per-layer suffixes, verified against
/// `ornith-ai/Ornith-1.5-9B-GGUF/Ornith-1.5-9B-Q4_K_M.gguf` -- the first
/// published GGUF of this architecture, and the reason the family stopped
/// being MLX-safetensors-only.
///
/// **The MoE table above minus its four MoE rows, plus a dense triple.** All
/// twenty of the real file's distinct suffixes are accounted for: the
/// seventeen shared with the MoE half (the hybrid split is identical -- 24
/// linear layers carrying `attn_qkv` plus the `ssm_*` family against 8 full
/// ones carrying `attn_q/k/v`, at the same `full_attention_interval` of 4),
/// and `ffn_{gate,up,down}` where the MoE half has `ffn_*_exps`,
/// `ffn_*_shexp` and `ffn_gate_inp`.
///
/// Deliberately NOT delegating to [`map_qwen_gdn_moe_layer`] with the MoE rows
/// filtered, for the reason [`map_qwen3moe_layer`] gives about the `llama`
/// table: the shared rows are equal today by observation of two real files,
/// not by construction, and a delegation would make a future divergence in
/// either file silently adopt the other half's answer.
pub fn map_qwen_gdn_dense_layer(suffix: &str, layer: usize) -> Option<GgufMapping> {
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
        // Gated DeltaNet, identical to the MoE half's.
        "attn_qkv.weight" => resident("linear_attn.in_proj_qkv.weight"),
        "attn_gate.weight" => resident("linear_attn.in_proj_z.weight"),
        "ssm_alpha.weight" => resident("linear_attn.in_proj_a.weight"),
        "ssm_beta.weight" => resident("linear_attn.in_proj_b.weight"),
        "ssm_out.weight" => resident("linear_attn.out_proj.weight"),
        "ssm_conv1d.weight" => resident("linear_attn.conv1d.weight"),
        "ssm_norm.weight" => resident("linear_attn.norm.weight"),
        // Gotcha 26: no `.weight` suffix on either of these, on either side.
        "ssm_a" => resident("linear_attn.A_log"),
        "ssm_dt.bias" => resident("linear_attn.dt_bias"),
        // The four rows the MoE table does not have, replacing its four.
        // `families/qwen/dense.rs` reads exactly these three names, and its
        // width is `intermediate_size` (12288 here) rather than
        // `moe_intermediate_size`, which a dense install sets to 0.
        "ffn_gate.weight" => resident("mlp.gate_proj.weight"),
        "ffn_up.weight" => resident("mlp.up_proj.weight"),
        "ffn_down.weight" => resident("mlp.down_proj.weight"),
        _ => None,
    }
}

/// Qwen3-MoE's per-layer suffixes, verified against
/// `Qwen3-30B-A3B-Q4_K_M.gguf` (ROADMAP Phase M, the fine-grained follow-on).
///
/// **The `llama` table plus exactly two rows.** Qwen3 norms q and k per head
/// before RoPE where a Mixtral norms neither, and those two `[head_dim]`
/// tensors are the only per-layer name the two architectures do not share.
/// Everything else -- the GQA projections, `ffn_norm` sitting where HF says
/// `post_attention_layernorm`, `ffn_gate_inp` as the router, the three
/// unfused `_exps` tensors -- is identical, which is why one decode flow
/// serves both.
///
/// Deliberately NOT delegating to `map_llama_layer`: the two tables are
/// equal today by observation of two real files, not by construction, and a
/// delegation would make a future divergence in either file silently adopt
/// the other family's answer.
pub fn map_qwen3moe_layer(suffix: &str, layer: usize) -> Option<GgufMapping> {
    let p = layer_prefix(layer);
    let resident = |tail: &str| Some(GgufMapping::Resident(format!("{p}{tail}")));
    match suffix {
        "attn_q.weight" => resident("self_attn.q_proj.weight"),
        "attn_k.weight" => resident("self_attn.k_proj.weight"),
        "attn_v.weight" => resident("self_attn.v_proj.weight"),
        "attn_output.weight" => resident("self_attn.o_proj.weight"),
        // The two rows the `llama` table does not have.
        "attn_q_norm.weight" => resident("self_attn.q_norm.weight"),
        "attn_k_norm.weight" => resident("self_attn.k_norm.weight"),
        "attn_norm.weight" => resident("input_layernorm.weight"),
        "ffn_norm.weight" => resident("post_attention_layernorm.weight"),
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

/// Dense `qwen3` (`ModelFamily::Qwen3Dense`): [`map_qwen3moe_layer`]'s table
/// minus the four routed-expert rows, plus plain `ffn_gate`/`ffn_up`/
/// `ffn_down`. Confirmed structurally against llama.cpp's own
/// `gguf-py/gguf/constants.py`: `MODEL_ARCH.QWEN3`'s tensor list is
/// `MODEL_ARCH.QWEN3MOE`'s list with `FFN_GATE_INP`/`FFN_GATE_EXPS`/
/// `FFN_UP_EXPS`/`FFN_DOWN_EXPS` removed and plain `FFN_GATE`/`FFN_DOWN`/
/// `FFN_UP` added -- nothing else differs, including the two QK-norm rows
/// (`docs/QWEN3_PHASE0.md`).
pub fn map_qwen3_dense_layer(suffix: &str, layer: usize) -> Option<GgufMapping> {
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
        "ffn_norm.weight" => resident("post_attention_layernorm.weight"),
        "ffn_gate.weight" => resident("mlp.gate_proj.weight"),
        "ffn_up.weight" => resident("mlp.up_proj.weight"),
        "ffn_down.weight" => resident("mlp.down_proj.weight"),
        _ => None,
    }
}
