//! Manifest quantization generation across families.

use model_io::ModelFamily;

use super::config::Gemma4Quant;

/// The `manifest.json -> quant` object for a Gemma 4 install, derived
/// from the checkpoint's own per-tensor bits (slot bits are read from the
/// layer-0 base names; every slot is affine/BF16/group-64 in this format).
/// Production-shape manifests are rejected by `turbospark_model_io` without
/// this object.
pub fn gemma4_manifest_quant(quant: &Gemma4Quant) -> serde_json::Value {
    manifest_quant(quant, ModelFamily::Gemma4)
}

/// [`gemma4_manifest_quant`] for any family. The probe names are the only
/// difference: Qwen 3.6's layer 0 is a LINEAR layer, so its "attention"
/// slot has to be probed at `linear_attn.in_proj_qkv` -- there is no
/// `self_attn.q_proj` under layer 0 at all.
pub fn manifest_quant(quant: &Gemma4Quant, family: ModelFamily) -> serde_json::Value {
    manifest_quant_for(quant, family, true)
}

/// [`manifest_quant`] told whether the model HAS routed experts.
///
/// **`has_experts = false` makes the three MoE slots mirror the attention
/// one, which is `crates/repack` Gotcha 8's `or_attention` rule arriving on
/// the safetensors side.** The GGUF walk learned it in ROADMAP M4, at one
/// five-minute re-stream per slot; this side kept the old shape only because
/// no dense safetensors checkpoint existed until `qwen3_5`, and because its
/// dense path wrote no quant block at all so nothing ever read these.
///
/// A probe here cannot tell "found, at the default width" from "not found":
/// `bits_for` reads the OVERRIDES map, not the tensor list. So denseness is
/// passed in rather than inferred, and a slot describing a component the
/// model does not have says the same thing the attention slot does --
/// executable exactly when the install is.
pub fn manifest_quant_for(
    quant: &Gemma4Quant,
    family: ModelFamily,
    has_experts: bool,
) -> serde_json::Value {
    // The companion dtype and the group size are read off the CHECKPOINT, not
    // written as constants. They used to be `bf16`/64 literals, which was a
    // true statement about every install that existed and became a false one
    // the moment a 1-bit checkpoint could be walked: its companions are FP16
    // at group 128, and a manifest claiming otherwise describes bytes the
    // install does not contain. `model_io::validate_quant` reads exactly
    // these three fields together and refuses any other combination.
    //
    // Keying the dtype on the DEFAULT bits is sound only because
    // `is_supported_affine_shape` has already refused every mixture that
    // would break it: 1 and 2 bits exist at group 128 and nowhere else, and 4
    // and 8 exist at group 64 and nowhere else, so a per-tensor override can
    // change the width within a group size but can never straddle the two
    // shapes. It CAN move between 1 and 2 bits, which share both the group
    // size and the companion dtype, so this stays one answer per install.
    let companions = if quant.default_bits <= 2 {
        "fp16"
    } else {
        "bf16"
    };
    let group_size = quant.group_size;
    let slot = |bits: u32| {
        serde_json::json!({
            "weightBits": bits,
            "scheme": "affine",
            "scaleType": companions,
            "biasType": companions,
            "groupSize": group_size,
        })
    };
    let l0 = "language_model.model.layers.0";
    let (attention, router, shared, routed) = match family {
        ModelFamily::QwenGdnMoe => (
            format!("{l0}.linear_attn.in_proj_qkv"),
            format!("{l0}.mlp.gate"),
            format!("{l0}.mlp.shared_expert.gate_proj"),
            format!("{l0}.mlp.switch_mlp.gate_proj"),
        ),
        // `qwen3_5` shares Qwen 3.6's layer 0 (a LINEAR layer, so the
        // attention probe is `in_proj_qkv` and not `self_attn.q_proj`) and
        // is DENSE. Its router and routed-expert probes therefore find
        // nothing and fall back to the default bits, which is the whole
        // model's width -- `crates/repack` Gotcha 8's rule, and the reason
        // `model_io::validate_quant` accepts the 1-bit shape on all five
        // slots. The shared-expert probe deliberately names the DENSE FFN,
        // which does exist, rather than a `shared_expert` path that never
        // will: the slot then reports a width that was actually measured.
        ModelFamily::QwenGdnDense => (
            format!("{l0}.linear_attn.in_proj_qkv"),
            format!("{l0}.mlp.gate"),
            format!("{l0}.mlp.gate_proj"),
            format!("{l0}.mlp.switch_mlp.gate_proj"),
        ),
        // The `llama` architecture shares Gemma's routed marker but names
        // its router the way Qwen does (`mlp.gate`, from GGUF's
        // `ffn_gate_inp`) and has NO shared expert -- that slot's probe finds
        // nothing and takes the default, which `validate_quant` accepts at
        // 4 or 8 bits either way.
        // `qwen3moe` names every one of these the way the `llama`
        // architecture does, shared expert included (it has none either).
        ModelFamily::Llama | ModelFamily::Qwen3Moe => (
            format!("{l0}.self_attn.q_proj"),
            format!("{l0}.mlp.gate"),
            format!("{l0}.mlp.gate_proj"),
            format!("{l0}.experts.switch_glu.gate_proj"),
        ),
        // gpt-oss has no safetensors path at all (it is GGUF-only here), so
        // it never reaches this probe; grouped with the families whose names
        // it shares rather than given an arm that cannot run.
        ModelFamily::Gemma4 | ModelFamily::DeepseekV4Flash | ModelFamily::GptOss => (
            format!("{l0}.self_attn.q_proj"),
            format!("{l0}.router.proj"),
            format!("{l0}.mlp.gate_proj"),
            format!("{l0}.experts.switch_glu.gate_proj"),
        ),
    };
    // The three MoE slots mirror ATTENTION when there are no experts: see
    // this function's doc, and `crates/repack` Gotcha 8 for what refusing
    // them instead cost the GGUF side.
    let attention_bits = quant.bits_for(&attention);
    let moe_slot = |probe: &str| {
        if has_experts {
            slot(quant.bits_for(probe))
        } else {
            slot(attention_bits)
        }
    };
    serde_json::json!({
        "embedding": slot(quant.bits_for("language_model.model.embed_tokens")),
        "attention": slot(attention_bits),
        "router": moe_slot(&router),
        "sharedExpert": moe_slot(&shared),
        "routedExpert": moe_slot(&routed),
    })
}
