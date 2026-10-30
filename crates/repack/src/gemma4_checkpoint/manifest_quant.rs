//! Manifest quantization generation across families.

use model_io::ModelFamily;

use super::config::{Gemma4Quant, QuantMode};

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
        // `qwen4_exp` probes at the same four paths, `l0` included. Its layer
        // 0 is LINEAR for the same reason Qwen 3.6's is (`(i + 1) % 4 == 0`
        // puts the first full-attention layer at index 3), it has a gated
        // shared expert, and its routed container is spelled the same.
        //
        // VERIFIED AGAINST THE PUBLISHED TENSOR INDEX, not assumed: all 3,215
        // names carry the `language_model.model.` prefix `l0` already builds.
        // The reference implementation reads them WITHOUT it, because its own
        // `sanitize()` strips the prefix on load -- so the reference is
        // evidence about that reader and not about the artifact, and taking
        // its spelling would have missed every tensor in the file.
        ModelFamily::QwenGdnMoe | ModelFamily::Qwen4Exp => (
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
        // Dense `qwen3` shares the same tensor names too (plain
        // `self_attn.q_proj`, dense `mlp.gate_proj`, no router, no shared
        // expert) but has no safetensors path this pass (GGUF only, see
        // `docs/QWEN3_PHASE0.md`), so like gpt-oss below it is grouped with
        // the family whose names it shares rather than given an arm that
        // cannot run.
        ModelFamily::Llama
        | ModelFamily::Qwen3Moe
        | ModelFamily::Qwen3Dense
        | ModelFamily::Qwen2Dense
        // `deepseek2` has no safetensors path (GGUF-only), so this arm is
        // unreachable; its canonical names match this shape (self_attn.q_proj,
        // mlp.gate router) if one is ever written. Its shared-expert probe
        // would find nothing here and take the default, which is the same
        // behavior the llama arm gives its shared-expert-less families.
        | ModelFamily::Deepseek2
        | ModelFamily::MiniMaxM2 => (
            format!("{l0}.self_attn.q_proj"),
            format!("{l0}.mlp.gate"),
            format!("{l0}.mlp.gate_proj"),
            format!("{l0}.experts.switch_glu.gate_proj"),
        ),
        // `muse_glimmer`'s layer 0 IS a plain attention layer, unlike both
        // Qwen halves, so the attention probe is the ordinary
        // `self_attn.q_proj`. It is DENSE, so the router and routed probes
        // find nothing and fall back to the default bits (the whole model's
        // width) -- `crates/repack` Gotcha 8. The shared-expert probe names
        // the DENSE FFN, which exists, so that slot reports a width that was
        // actually measured rather than a default.
        //
        // `spark2_5` is dense the same way and takes the same probe shape.
        // It has no safetensors intake this pass (GGUF only), so the arm is
        // unreachable today; it is written in the shape the future
        // safetensors pass would want rather than parked on another
        // family's arm whose probes name tensors Spark does not have.
        //
        // `qwen3_vl` IS reached today (its pinned witness is an MLX
        // conversion) and its layer 0 is a plain attention layer with dense
        // `mlp.{gate,up,down}_proj`, so it takes this exact shape: the
        // attention probe measures `self_attn.q_proj`, the shared-expert
        // probe measures the dense FFN, and the router/routed probes take
        // the default.
        ModelFamily::MuseGlimmer | ModelFamily::Spark25 | ModelFamily::Qwen3Vl => (
            format!("{l0}.self_attn.q_proj"),
            format!("{l0}.mlp.gate"),
            format!("{l0}.mlp.gate_proj"),
            format!("{l0}.experts.switch_glu.gate_proj"),
        ),
        // `flint`'s layers 0 and 1 are DENSE (`mlp_only_layers`), so the
        // router, shared-expert and routed probes must name layer 2, the
        // first MoE layer; every other arm probes layer 0. The attention
        // probe stays on layer 0, which is a plain `self_attn.q_proj`.
        // Names carry the canonical `language_model.` prefix because
        // `canonicalize_flint_header` has already renamed the header.
        ModelFamily::Flint => {
            let l2 = "language_model.model.layers.2";
            (
                format!("{l0}.self_attn.q_proj"),
                format!("{l2}.mlp.gate"),
                format!("{l2}.mlp.shared_expert.gate_proj"),
                format!("{l2}.mlp.switch_mlp.gate_proj"),
            )
        }
        // `lfm3`'s layer 0 is a SHORT-CONV layer with no `self_attn` at all,
        // so the attention probe names layer 2, the first attention layer
        // (`config.layer_types`). The shared-expert probe measures the dense
        // FFN's gate, `feed_forward.w1`, which exists on every layer; the
        // router and routed probes find nothing and take the default, the
        // muse/spark rule for a dense family (`crates/repack` Gotcha 8).
        ModelFamily::Lfm3 => (
            "language_model.model.layers.2.self_attn.q_proj".to_string(),
            format!("{l0}.mlp.gate"),
            format!("{l0}.feed_forward.w1"),
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
    if quant.mode == QuantMode::Mxfp4 {
        return mxfp4_manifest_quant(quant, has_experts, [&attention, &router, &shared, &routed]);
    }
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
    // `qwen4_exp`'s router is force-quantized to INT8 at write time
    // regardless of what the checkpoint's own config declares
    // (`narrow.rs`'s `quantize_gating_matrix_int8`, called from
    // `orchestrate.rs` because this family's real checkpoint ships the
    // router raw rather than U32-prepacked). Its manifest slot states that
    // unconditionally instead
    // of trusting `quant.bits_for`, which for this checkpoint reports the
    // model's default width and not the router's -- mirroring the GGUF
    // walk's router slot, which is also a stated fact about the bytes rather
    // than a probe (`crates/repack` CLAUDE.md Gotcha 6).
    let router_slot = if !has_experts {
        slot(attention_bits)
    } else if matches!(family, ModelFamily::Qwen4Exp | ModelFamily::Flint) {
        // `flint` ships its router as an unquantized BF16 gate that the
        // writer force-quantizes to INT8 (`int8_force_targets`), so the
        // probe would report the model's 4-bit default for a tensor that
        // carries no companions and `validate_quant` would refuse the
        // install. State the width of the bytes written, as `qwen4_exp` does.
        slot(8)
    } else {
        slot(quant.bits_for(&router))
    };
    serde_json::json!({
        "embedding": slot(quant.bits_for("language_model.model.embed_tokens")),
        "attention": slot(attention_bits),
        "router": router_slot,
        "sharedExpert": moe_slot(&shared),
        "routedExpert": moe_slot(&routed),
    })
}

/// The quant object for an MLX mxfp4 checkpoint, stated as what the writer
/// PUTS ON DISK rather than what the source declared, since the two differ:
///
/// - a resident tensor that is mxfp4 at source is dequantized to BF16
///   (`orchestrate.rs`'s `read_mxfp4_as_bf16`), and one that was never
///   quantized stays BF16, so either way its slot is raw BF16. The probe
///   cannot tell those two apart and does not need to;
/// - routed experts are re-blocked to GGUF MXFP4 (`expert_blobs.rs`);
/// - a per-tensor override is affine at its own bits and group size, with
///   BF16 companions (`pass_through_packed` refuses anything else).
fn mxfp4_manifest_quant(
    quant: &Gemma4Quant,
    has_experts: bool,
    [attention, router, shared, routed]: [&str; 4],
) -> serde_json::Value {
    let raw_bf16 = serde_json::json!({
        "weightBits": 16,
        "scheme": "raw",
        "scaleType": "none",
        "biasType": "none",
        "groupSize": 0,
    });
    let resident = |probe: &str| {
        if quant.is_mxfp4(probe) {
            raw_bf16.clone()
        } else {
            serde_json::json!({
                "weightBits": quant.bits_for(probe),
                "scheme": "affine",
                "scaleType": "bf16",
                "biasType": "bf16",
                "groupSize": quant.group_for(probe),
            })
        }
    };
    let attention_slot = resident(attention);
    let (router, shared, routed) = if has_experts {
        let routed = if quant.is_mxfp4(routed) {
            serde_json::json!({
                "weightBits": 0,
                "scheme": "gguf",
                "ggmlType": "mxfp4",
                "scaleType": "inline",
                "biasType": "inline",
                "groupSize": 0,
            })
        } else {
            resident(routed)
        };
        (resident(router), resident(shared), routed)
    } else {
        let mirror = attention_slot.clone();
        (mirror.clone(), mirror.clone(), mirror)
    };
    serde_json::json!({
        "embedding": resident("language_model.model.embed_tokens"),
        "attention": attention_slot,
        "router": router,
        "sharedExpert": shared,
        "routedExpert": routed,
    })
}
