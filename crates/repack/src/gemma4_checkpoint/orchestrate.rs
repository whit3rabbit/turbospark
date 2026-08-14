//! Orchestration, layer planning, and manifest quantization generation.

use std::collections::BTreeMap;

use model_io::{ArchConfig, ModelFamily};

use super::config::{Gemma4Error, Gemma4Quant};
use super::shards::{
    classify_for_family, lm_order_key, narrow_raw_to_bf16, pass_through_packed, shape4,
    Gemma4Bucket, Gemma4Shards, GTURBO_PAGE_BYTES,
};
use crate::gturbo_writer::{ExpertBlob, LayerBlobs, SubTensor};
use crate::ranged_download::RangeSource;
use crate::resident_writer::{RawTensorSpec, ResidentEntrySpec};
use crate::safetensors_header::{SafetensorsHeader, TensorInfo};

/// Everything the writer needs for a full Gemma 4 `.gturbo` install:
/// ordered resident entries (pass-through quantized + raw), per-layer
/// routed-expert blobs, and the one model-wide page-rounded expert stride
/// (0 when the checkpoint has no routed experts).
pub struct Gemma4RepackOutput {
    pub resident: Vec<ResidentEntrySpec>,
    pub layers: Vec<LayerBlobs>,
    pub expert_stride: u64,
    pub excluded_multimodal: Vec<String>,
    /// One `(tensor, values that lost bits)` row per unquantized tensor this
    /// walk had to narrow to BF16. See `narrow_raw_to_bf16`.
    pub lossy_narrowing: Vec<(String, usize)>,
}

/// Walks a Gemma 4 checkpoint's tensors: classifies every name, orders and
/// reads the resident LM set (pass-through for `U32` `.weight` tensors with
/// their BF16 companions, raw bytes for everything else), and slices each
/// layer's routed-expert `gate/up/down` bundles into per-expert blobs.
pub fn orchestrate_gemma4_checkpoint(
    header: &SafetensorsHeader,
    source: &dyn RangeSource,
    arch: &ArchConfig,
    quant: &Gemma4Quant,
) -> Result<Gemma4RepackOutput, Gemma4Error> {
    orchestrate_gemma4_checkpoint_sharded(&Gemma4Shards::single(header, source), arch, quant)
}

/// Multi-shard variant of [`orchestrate_gemma4_checkpoint`] -- what a real
/// (three-shard) checkpoint goes through.
pub fn orchestrate_gemma4_checkpoint_sharded(
    shards: &Gemma4Shards<'_>,
    arch: &ArchConfig,
    quant: &Gemma4Quant,
) -> Result<Gemma4RepackOutput, Gemma4Error> {
    let plan = classify_all(shards, arch)?;
    let resident = read_resident_entries(shards, &plan.resident_bases, quant)?;
    let expert_stride = expert_stride_from_headers(shards, arch, quant, &plan.routed)?;
    let mut layers = Vec::new();
    if !plan.routed.is_empty() {
        for layer in 0..arch.num_layers as usize {
            let (blobs, _used) = plan_one_expert_layer(shards, arch, quant, &plan.routed, layer)?;
            layers.push(blobs);
        }
    }
    Ok(Gemma4RepackOutput {
        resident: resident.entries,
        layers,
        expert_stride,
        excluded_multimodal: plan.excluded,
        lossy_narrowing: resident.lossy_narrowing,
    })
}

pub struct ClassifiedNames<'a> {
    pub resident_bases: Vec<&'a str>,
    pub routed: BTreeMap<usize, BTreeMap<&'static str, &'a str>>,
    pub excluded: Vec<String>,
}

pub fn classify_all<'a>(
    shards: &Gemma4Shards<'a>,
    arch: &ArchConfig,
) -> Result<ClassifiedNames<'a>, Gemma4Error> {
    let num_layers = arch.num_layers as usize;
    let mut resident_bases: Vec<&str> = Vec::new();
    let mut excluded: Vec<String> = Vec::new();
    let mut routed: BTreeMap<usize, BTreeMap<&'static str, &str>> = BTreeMap::new();

    for name in shards.names() {
        if name.ends_with(".scales") || name.ends_with(".biases") {
            continue;
        }
        match classify_for_family(name, num_layers, arch.family) {
            Gemma4Bucket::LmResident => resident_bases.push(name),
            Gemma4Bucket::RoutedExpert { role, layer } => {
                if routed
                    .entry(layer)
                    .or_default()
                    .insert(role, name.as_str())
                    .is_some()
                {
                    return Err(Gemma4Error::ShapeMismatch {
                        tensor: name.to_string(),
                        detail: format!("two routed-expert tensors for layer {layer} role {role}"),
                    });
                }
            }
            Gemma4Bucket::ExcludedMultimodal => excluded.push(name.clone()),
            Gemma4Bucket::Unknown => return Err(Gemma4Error::UnknownTensor(name.clone())),
        }
    }
    resident_bases.sort_by(|a, b| lm_order_key(a).cmp(&lm_order_key(b)));
    excluded.sort();
    Ok(ClassifiedNames {
        resident_bases,
        routed,
        excluded,
    })
}

/// The resident set plus what narrowing it to BF16 cost.
pub struct ResidentRead {
    pub entries: Vec<ResidentEntrySpec>,
    /// One `(tensor, values that lost bits)` row per lossily-narrowed tensor,
    /// mirroring `GgufRepackOutput::lossy_narrowing`. Empty for every
    /// BF16-source checkpoint, which is every one but Bonsai-27B.
    pub lossy_narrowing: Vec<(String, usize)>,
}

pub fn read_resident_entries(
    shards: &Gemma4Shards<'_>,
    resident_bases: &[&str],
    quant: &Gemma4Quant,
) -> Result<ResidentRead, Gemma4Error> {
    let mut entries = Vec::with_capacity(resident_bases.len());
    let mut lossy_narrowing = Vec::new();
    for &name in resident_bases {
        let t = shards.info(name)?;
        if t.dtype == "U32" && name.ends_with(".weight") {
            entries.push(pass_through_packed(shards, name, quant)?);
        } else {
            // NARROWED, not tagged. The deleted `raw_dtype_tag` recorded F16
            // or F32 and nothing downstream reads either tag: every consumer
            // of an unquantized tensor decodes it as BF16 off its byte size,
            // so an F16 norm written verbatim is misread rather than
            // refused. See `narrow_raw_to_bf16` for the measurement behind
            // accepting the loss.
            let narrowed = narrow_raw_to_bf16(name, &t.dtype, shards.read(name)?)?;
            if narrowed.lossy > 0 {
                lossy_narrowing.push((name.to_string(), narrowed.lossy));
            }
            entries.push(ResidentEntrySpec::Raw(RawTensorSpec {
                name: name.to_string(),
                dtype: narrowed.dtype,
                bytes: narrowed.bytes,
                shape: shape4(&t.shape),
            }));
        }
    }
    Ok(ResidentRead {
        entries,
        lossy_narrowing,
    })
}

/// The one model-wide expert stride, computed from shard HEADERS alone
/// (per-expert weight+scales+biases byte totals, max across layers,
/// rounded to 16 KiB) -- so a streaming writer knows the stride before any
/// expert byte downloads.
pub fn expert_stride_from_headers(
    shards: &Gemma4Shards<'_>,
    arch: &ArchConfig,
    quant: &Gemma4Quant,
    routed: &BTreeMap<usize, BTreeMap<&'static str, &str>>,
) -> Result<u64, Gemma4Error> {
    if routed.is_empty() {
        return Ok(0);
    }
    let expert_count = arch.num_experts as u64;
    let mut max_blob = 0u64;
    for layer in 0..arch.num_layers as usize {
        let bundle = routed.get(&layer).ok_or_else(|| {
            Gemma4Error::MissingTensor(format!("layer {layer} routed-expert bundle"))
        })?;
        let mut blob = 0u64;
        for role in ["gate", "up", "down"] {
            let name = *bundle
                .get(role)
                .ok_or_else(|| Gemma4Error::MissingTensor(format!("layer {layer} {role}_proj")))?;
            let base = name.strip_suffix(".weight").unwrap_or(name);
            let bits = quant.bits_for(base);
            if bits != 4 {
                return Err(Gemma4Error::UnsupportedDtype {
                    tensor: name.to_string(),
                    dtype: format!("{bits}-bit routed expert (kernels are int4-only)"),
                });
            }
            for suffix in ["", ".scales", ".biases"] {
                let full = if suffix.is_empty() {
                    name.to_string()
                } else {
                    format!("{base}{suffix}")
                };
                let t = shards.info(&full)?;
                let bytes = t.data_offsets.1 - t.data_offsets.0;
                if bytes % expert_count != 0 {
                    return Err(Gemma4Error::ShapeMismatch {
                        tensor: full,
                        detail: format!("bytes not divisible by {expert_count} experts"),
                    });
                }
                blob += bytes / expert_count;
            }
        }
        max_blob = max_blob.max(blob);
    }
    Ok(max_blob.div_ceil(GTURBO_PAGE_BYTES) * GTURBO_PAGE_BYTES)
}

/// Builds one routed layer's per-expert blobs. Blob layout matches the
/// synthetic MoE installs (and `RealForwardRunner`'s `MoeExpertOffsets`):
/// `gate, gate_scales, gate_biases, up, ..., down, ...` back to back. The
/// down projection's blob offset must land 4-byte aligned (the phase-2
/// kernel reads its weights with `uint` loads); u32 weights and even-sized
/// BF16 companions keep that true for any real shape, and we verify it.
/// Returns the blobs plus the per-expert bytes used (callers pad to the
/// model-wide stride).
pub fn plan_one_expert_layer(
    shards: &Gemma4Shards<'_>,
    arch: &ArchConfig,
    quant: &Gemma4Quant,
    routed: &BTreeMap<usize, BTreeMap<&'static str, &str>>,
    layer: usize,
) -> Result<(LayerBlobs, u64), Gemma4Error> {
    let expert_count = arch.num_experts as usize;
    {
        let bundle = routed.get(&layer).ok_or_else(|| {
            Gemma4Error::MissingTensor(format!("layer {layer} routed-expert bundle"))
        })?;
        let mut experts: Vec<ExpertBlob> = (0..expert_count)
            .map(|e| ExpertBlob {
                expert: e,
                sub_tensors: Vec::with_capacity(9),
            })
            .collect();
        let mut blob_used = 0u64;

        for role in ["gate", "up", "down"] {
            let name = *bundle
                .get(role)
                .ok_or_else(|| Gemma4Error::MissingTensor(format!("layer {layer} {role}_proj")))?;
            let w = shards.info(name)?;
            if w.dtype != "U32" || w.shape.len() != 3 || w.shape[0] as usize != expert_count {
                return Err(Gemma4Error::ShapeMismatch {
                    tensor: name.to_string(),
                    detail: format!(
                        "expected U32 rank-3 with leading {expert_count}, got {} {:?}",
                        w.dtype, w.shape
                    ),
                });
            }
            let base = name.strip_suffix(".weight").unwrap_or(name);
            let bits = quant.bits_for(base);
            if bits != 4 {
                return Err(Gemma4Error::UnsupportedDtype {
                    tensor: name.to_string(),
                    dtype: format!("{bits}-bit routed expert (kernels are int4-only)"),
                });
            }
            let s_name = format!("{base}.scales");
            let b_name = format!("{base}.biases");
            let s = shards.info(&s_name)?;
            let b = shards.info(&b_name)?;
            if s.dtype != "BF16" || b.dtype != "BF16" {
                return Err(Gemma4Error::UnsupportedDtype {
                    tensor: name.to_string(),
                    dtype: format!("{}/{} companions", s.dtype, b.dtype),
                });
            }

            let w_bytes = shards.read(name)?;
            let s_bytes = shards.read(&s_name)?;
            let b_bytes = shards.read(&b_name)?;
            let per = |total: usize, what: &str| -> Result<usize, Gemma4Error> {
                if total % expert_count != 0 {
                    return Err(Gemma4Error::ShapeMismatch {
                        tensor: name.to_string(),
                        detail: format!("{what} bytes not divisible by {expert_count} experts"),
                    });
                }
                Ok(total / expert_count)
            };
            let w_per = per(w_bytes.len(), "weight")?;
            let s_per = per(s_bytes.len(), "scales")?;
            let b_per = per(b_bytes.len(), "biases")?;

            let rows = w.shape[1];
            let cols = w.shape[2] * 8;
            let comp_shape = |t: &TensorInfo| t.shape[1..].to_vec();
            if role == "down" && blob_used % 4 != 0 {
                return Err(Gemma4Error::ShapeMismatch {
                    tensor: name.to_string(),
                    detail: format!("down offset {blob_used} not 4-byte aligned"),
                });
            }
            for (e, blob) in experts.iter_mut().enumerate() {
                for (suffix, bytes, dtype, shape) in [
                    (
                        "",
                        w_bytes[e * w_per..(e + 1) * w_per].to_vec(),
                        "u32",
                        vec![rows, cols],
                    ),
                    (
                        "_scales",
                        s_bytes[e * s_per..(e + 1) * s_per].to_vec(),
                        "bf16",
                        comp_shape(s),
                    ),
                    (
                        "_biases",
                        b_bytes[e * b_per..(e + 1) * b_per].to_vec(),
                        "bf16",
                        comp_shape(b),
                    ),
                ] {
                    blob.sub_tensors.push(SubTensor {
                        role: format!("{role}{suffix}"),
                        bytes,
                        dtype: dtype.to_string(),
                        shape,
                    });
                }
            }
            blob_used += (w_per + s_per + b_per) as u64;
        }
        Ok((LayerBlobs { layer, experts }, blob_used))
    }
}

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
    // would break it: 1 bit exists at group 128 and nowhere else, and 4 and 8
    // exist at group 64 and nowhere else, so a per-tensor override can change
    // the width within a group size but can never straddle the two shapes.
    let companions = if quant.default_bits == 1 {
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
        ModelFamily::Qwen36 => (
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
        ModelFamily::Qwen35 => (
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
