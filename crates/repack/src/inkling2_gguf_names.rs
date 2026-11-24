//! GGUF tensor names of Inkling 2 Mini, mapped back to the HF checkpoint
//! names the Inkling intake and flow use (`model.llm.*`), with the transform
//! llama.cpp's converter (`conversion/inkling2.py`) applied to each.
//!
//! Two conventions exist:
//! - llama.cpp master (`ggml-org/Inkling-2-Mini-GGUF`, arch `inkling2_loop`,
//!   24 blocks): loop gates `blk.N.loop_{q,k,v,ffn}_ctrl_{down,gain,forget}`
//!   and per-core-layer message queries `blk.{core_start + i}.loop_msg_query`
//!   (`[q[2i], kv[2i], kv[2i+1], q[2i+1]]`).
//! - the older converter (`unsloth/Inkling-2-Mini-GGUF`, arch `inkling2`):
//!   loop gates `blk.N.loop_{attn,k,v,mlp}_{down,gain,forget}` and the two
//!   global query tensors `loop_res_query` / `loop_kv_res_query`, already in
//!   the HF `[2 * core, hidden]` layout.
//!
//! Only names are decided here. Running a GGUF also needs its block types:
//! experts can be ggml NVFP4 (the same block layout as the install's) with a
//! per-expert F32 `.scale`, or a re-quantized type (Q8_0, Q5_K, IQ*) the
//! Inkling MoE kernels do not read yet.

/// What the converter did to a tensor, to be undone by an intake.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgufTransform {
    /// Same values and layout (dtype may differ: F32 norms were BF16).
    None,
    /// `rel_logits_proj.proj` was transposed (`[extent, d_rel]` stored).
    Transposed,
    /// A conv weight squeezed from `[ch, 1, taps]` to `[ch, taps]`; on the
    /// full-attention `{k,v}_sconv` it is `static_w` flipped and transposed,
    /// which is exactly the install's rewritten `{k,v}_sconv.weight`.
    SqueezedConv,
    /// Routed `w13`: gate rows first, then up rows (the HF checkpoint
    /// interleaves them, even rows gate, odd rows up).
    ExpertGateUpSplit,
    /// Routed `w2`, rows as in HF.
    ExpertDown,
    /// The per-expert F32 scale of an NVFP4 routed projection (`[experts]`).
    /// The HF checkpoint has no such tensor (it stores `amax` and a
    /// denominator), so its name here, `{w13,w2}_weight_outer`, is synthetic:
    /// the install's per-expert outer-scale record.
    ExpertScale,
    /// One core layer's four message queries (master convention); the HF
    /// tensors are the two cross-pass query sets, recombined by the intake.
    MessageQueries { core_index: usize },
}

/// The HF name (or names) and transform of one GGUF tensor, or None for a
/// tensor the text install does not read (vision, the MTP draft) or a name
/// this table does not know. `core_start` is the loop's first core layer.
pub fn gguf_to_hf(name: &str, core_start: usize) -> Option<(String, GgufTransform)> {
    use GgufTransform as T;
    let global = |hf: &str, t| Some((hf.to_string(), t));
    match name {
        "token_embd.weight" => return global("model.llm.embed.weight", T::None),
        "token_embd_norm.weight" => return global("model.llm.embed_norm.weight", T::None),
        "output.weight" => return global("model.llm.unembed.weight", T::None),
        "output_norm.weight" => return global("model.llm.norm.weight", T::None),
        "output_res_query.weight" => return global("model.llm.attn_res.output_query", T::None),
        "loop_res_query.weight" => {
            return global("model.llm.middle_cycle_cross_pass_queries", T::None)
        }
        "loop_kv_res_query.weight" => {
            return global("model.llm.middle_cycle_cross_pass_kv_queries", T::None)
        }
        _ => {}
    }
    let rest = name.strip_prefix("blk.")?;
    let (l, tail) = rest.split_once('.')?;
    let l: usize = l.parse().ok()?;
    let layer = |s: &str, t| Some((format!("model.llm.layers.{l}.{s}"), t));
    match tail {
        "attn_q.weight" => layer("attn.wq_du.weight", T::None),
        "attn_k.weight" => layer("attn.wk_dv.weight", T::None),
        "attn_v.weight" => layer("attn.wv_dv.weight", T::None),
        "attn_output.weight" => layer("attn.wo_ud.weight", T::None),
        "attn_gate.weight" => layer("attn.gate_attn.weight", T::None),
        "attn_rel.weight" => layer("attn.wr_du.weight", T::None),
        "attn_sink_proj.weight" => layer("attn.attn_sink_proj.weight", T::None),
        "attn_rel_bias.weight" => layer("attn.rel_logits_proj.proj", T::Transposed),
        "attn_q_clip.weight" => layer("attn.q_l2_clip.scale", T::None),
        "attn_rel_clip.weight" => layer("attn.rel_l2_clip.scale", T::None),
        "attn_k_conv.weight" => layer("attn.k_sconv.weight", T::SqueezedConv),
        "attn_v_conv.weight" => layer("attn.v_sconv.weight", T::SqueezedConv),
        "attn_k_conv_proj.weight" => layer("attn.k_sconv.weight_proj", T::None),
        "attn_v_conv_proj.weight" => layer("attn.v_sconv.weight_proj", T::None),
        "attn_post_conv.weight" => layer("attn_sconv.weight", T::SqueezedConv),
        "ffn_post_conv.weight" => layer("mlp_sconv.weight", T::SqueezedConv),
        "attn_norm.weight" => layer("attn_norm.weight", T::None),
        "attn_q_inp_norm.weight" => layer("q_input_norm.weight", T::None),
        "attn_k_inp_norm.weight" => layer("k_input_norm.weight", T::None),
        "attn_v_inp_norm.weight" => layer("v_input_norm.weight", T::None),
        "ffn_norm.weight" => layer("mlp_norm.weight", T::None),
        "ffn_gate_inp.weight" => layer("mlp.gate.weight", T::None),
        "exp_probs_b.bias" => layer("mlp.gate.bias", T::None),
        "ffn_gate_up_exps.weight" => layer("mlp.experts.w13_weight", T::ExpertGateUpSplit),
        "ffn_down_exps.weight" => layer("mlp.experts.w2_weight", T::ExpertDown),
        "ffn_gate_up_exps.scale" => layer("mlp.experts.w13_weight_outer", T::ExpertScale),
        "ffn_down_exps.scale" => layer("mlp.experts.w2_weight_outer", T::ExpertScale),
        "ffn_res_query.weight" => Some((format!("model.llm.attn_res.mlp_queries.{l}"), T::None)),
        // Block L's attention input is HF query L - 1 (block 0 has none).
        "attn_res_query.weight" => Some((
            format!("model.llm.attn_res.attn_queries.{}", l.checked_sub(1)?),
            T::None,
        )),
        "loop_msg_query.weight" => Some((
            "model.llm.middle_cycle_cross_pass_queries+kv".into(),
            T::MessageQueries {
                core_index: l.checked_sub(core_start)?,
            },
        )),
        other => {
            // Loop gates, both conventions.
            let (ctrl, part) = other
                .strip_prefix("loop_")?
                .strip_suffix(".weight")?
                .rsplit_once('_')?;
            let ctrl = match ctrl {
                "q_ctrl" | "attn" => "attn",
                "k_ctrl" | "k" => "k",
                "v_ctrl" | "v" => "v",
                "ffn_ctrl" | "mlp" => "mlp",
                _ => return None,
            };
            let part = match part {
                "down" | "gain" | "forget" => part,
                _ => return None,
            };
            layer(&format!("controllers.{ctrl}.{part}_weight"), T::None)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{gguf_to_hf, GgufTransform as T};

    #[test]
    fn both_conventions_map_to_the_hf_names() {
        let core = 6;
        for (gguf, hf, t) in [
            ("token_embd.weight", "model.llm.embed.weight", T::None),
            ("output.weight", "model.llm.unembed.weight", T::None),
            (
                "output_res_query.weight",
                "model.llm.attn_res.output_query",
                T::None,
            ),
            (
                "blk.3.attn_q.weight",
                "model.llm.layers.3.attn.wq_du.weight",
                T::None,
            ),
            (
                "blk.3.attn_rel.weight",
                "model.llm.layers.3.attn.wr_du.weight",
                T::None,
            ),
            (
                "blk.3.attn_rel_bias.weight",
                "model.llm.layers.3.attn.rel_logits_proj.proj",
                T::Transposed,
            ),
            (
                "blk.5.attn_k_conv.weight",
                "model.llm.layers.5.attn.k_sconv.weight",
                T::SqueezedConv,
            ),
            (
                "blk.3.ffn_post_conv.weight",
                "model.llm.layers.3.mlp_sconv.weight",
                T::SqueezedConv,
            ),
            (
                "blk.5.attn_q_inp_norm.weight",
                "model.llm.layers.5.q_input_norm.weight",
                T::None,
            ),
            (
                "blk.3.exp_probs_b.bias",
                "model.llm.layers.3.mlp.gate.bias",
                T::None,
            ),
            (
                "blk.3.ffn_gate_up_exps.weight",
                "model.llm.layers.3.mlp.experts.w13_weight",
                T::ExpertGateUpSplit,
            ),
            (
                "blk.5.attn_res_query.weight",
                "model.llm.attn_res.attn_queries.4",
                T::None,
            ),
            (
                "blk.5.ffn_res_query.weight",
                "model.llm.attn_res.mlp_queries.5",
                T::None,
            ),
            // master and older loop gates
            (
                "blk.7.loop_q_ctrl_down.weight",
                "model.llm.layers.7.controllers.attn.down_weight",
                T::None,
            ),
            (
                "blk.7.loop_attn_down.weight",
                "model.llm.layers.7.controllers.attn.down_weight",
                T::None,
            ),
            (
                "blk.7.loop_ffn_ctrl_gain.weight",
                "model.llm.layers.7.controllers.mlp.gain_weight",
                T::None,
            ),
            (
                "blk.7.loop_mlp_gain.weight",
                "model.llm.layers.7.controllers.mlp.gain_weight",
                T::None,
            ),
            (
                "loop_kv_res_query.weight",
                "model.llm.middle_cycle_cross_pass_kv_queries",
                T::None,
            ),
        ] {
            assert_eq!(gguf_to_hf(gguf, core), Some((hf.to_string(), t)), "{gguf}");
        }
        assert_eq!(
            gguf_to_hf("blk.8.loop_msg_query.weight", core).map(|(_, t)| t),
            Some(T::MessageQueries { core_index: 2 })
        );
        for unknown in [
            "blk.0.attn_res_query.weight",
            "v.blk.0.attn_q.weight",
            "blk.3.attn_bogus.weight",
            "blk.7.loop_x_ctrl_down.weight",
        ] {
            assert_eq!(gguf_to_hf(unknown, core), None, "{unknown}");
        }
    }
}
