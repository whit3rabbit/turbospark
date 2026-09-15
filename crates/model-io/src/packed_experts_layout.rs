//! `packed_experts/layout.json` decode: per-layer, per-expert byte offsets
//! inside the packed-expert blob files. Ported from
//! `Infrastructure/ModelIO/PackedExpertsLayout.swift`.

use std::collections::BTreeMap;
use std::path::Path;

use serde_json::Value;

use crate::error::ModelError;

/// Entry describing a sub-tensor (weight, scale, or bias) inside an expert blob.
#[derive(Debug, Clone, PartialEq)]
pub struct SubTensorEntry {
    /// Offset relative to the expert blob's start.
    pub offset: u64,
    /// Bytes; scale slices encode the group count.
    pub size: u64,
    /// What the writer called this run's element type: `"int4"`, `"bf16"`,
    /// `"q8_0"`, `"iq3_xxs"` and so on, lowercased ggml names for GGUF
    /// installs.
    ///
    /// The writer has always emitted it and the loader has always required it
    /// to be present; until ROADMAP Phase S it then threw it away, because
    /// every install was uniform and the manifest's one `ggmlType` said
    /// everything. A mixed install needs it PER SUB-TENSOR: the Phase S
    /// candidate's expert is IQ3_XXS gate/up over an IQ4_NL down, so the
    /// manifest cannot say which kernel a given dispatch wants and this can.
    pub dtype: String,
}

/// Entry describing the offsets and sub-tensors of a single routed expert.
#[derive(Debug, Clone, PartialEq)]
pub struct ExpertEntry {
    /// Logical routed-expert id used by the model/router.
    pub expert: usize,
    /// Absolute byte offset of this expert blob's start inside its layer file.
    pub offset: u64,
    /// Total bytes consumed by this expert blob (== `expert_stride`).
    pub size: u64,
    /// Sub-tensors keyed by role (gate/up/down/shared) and component.
    pub sub_tensors: BTreeMap<String, SubTensorEntry>,
}

/// Layout information for all experts within a single layer.
#[derive(Debug, Clone, PartialEq)]
pub struct LayerLayout {
    /// Layer index.
    pub layer: usize,
    /// Basename, e.g. "layer_00.bin".
    pub file: String,
    /// Bytes per expert blob IN THIS LAYER, page-aligned.
    ///
    /// Per layer rather than per model because a mixed sub-4-bit checkpoint
    /// is not uniform across layers (ROADMAP Phase S). The candidate there
    /// puts IQ3_XXS + IQ4_NL experts on 29 layers and IQ4_XS + Q8_0 on the
    /// thirtieth, whose blob is 1.6x the others: padding every layer to the
    /// model-wide maximum would cost 16.2 GB against 10.3, turning the
    /// phase's whole -24.2% into a +35% regression, and inflating per-miss
    /// read bytes by the same factor on 29 of 30 layers.
    ///
    /// Falls back to the top-level `expertStride` when the layer does not
    /// declare one, which every install written before Phase S does.
    pub expert_stride: u64,
    /// List of expert layout entries in this layer.
    pub experts: Vec<ExpertEntry>,
}

/// Complete layout layout descriptor for packed experts across all model layers.
#[derive(Debug, Clone, PartialEq)]
pub struct PackedExpertsLayout {
    /// The model-wide MAXIMUM of [`LayerLayout::expert_stride`], which is what
    /// `manifest.json`'s `expertStride` declares and what a consumer wanting
    /// one number for "how big can an expert blob be" should read. Do NOT use
    /// it to size or address a specific layer: see the field above.
    pub expert_stride: u64,
    /// Total layer count.
    pub num_layers: usize,
    /// Expert count per layer.
    pub experts_per_layer: usize,
    /// Per-layer expert layout descriptors.
    pub layers: Vec<LayerLayout>,
}

impl PackedExpertsLayout {
    /// Resolve `(layer, expert)` -> `ExpertEntry`. O(1).
    pub fn expert(&self, layer: usize, expert: usize) -> &ExpertEntry {
        &self.layers[layer].experts[expert]
    }
}

/// 64 MiB: Qwen 3.6's 40 layers x 256 experts x 9 sub-tensors produce a
/// ~22 MB layout.json; Gemma's is ~5 MB.
pub const DEFAULT_MAX_BYTES: u64 = 64 * 1024 * 1024;

/// The routed experts' subdirectory, and the only one until ROADMAP M-V3.
pub const PACKED_EXPERTS_DIR: &str = "packed_experts";

/// The vision tower's subdirectory (ROADMAP M-V3). It carries the SAME schema
/// this module decodes: one `LayerLayout`, `experts` = the tower's blocks,
/// free-form roles with a dtype each. What makes the reuse honest rather than
/// a pun is that `StreamLayout` interprets none of it -- a "layer" is a file
/// and an "expert" is a fixed-stride blob inside it, which is exactly what a
/// double-buffered block loop wants.
pub const PACKED_VISION_DIR: &str = "packed_vision";

/// Loads packed experts layout from `dir/packed_experts/layout.json`.
pub fn load(dir: &Path, max_bytes: u64) -> Result<PackedExpertsLayout, ModelError> {
    load_from(dir, PACKED_EXPERTS_DIR, max_bytes)
}

/// [`load`] against an arbitrary subdirectory of the install.
///
/// The subdirectory is a PARAMETER rather than two copies of this decoder,
/// because a second copy is a second place for the `expert_stride` fallback,
/// the per-layer stride ceiling and the missing-entry check to drift -- and
/// Gotcha 2 is already about one of those being got wrong. The vision tower
/// passes [`PACKED_VISION_DIR`].
pub fn load_from(
    dir: &Path,
    subdir: &str,
    max_bytes: u64,
) -> Result<PackedExpertsLayout, ModelError> {
    let path = dir.join(subdir).join("layout.json");
    if !path.exists() {
        return Err(ModelError::MissingFile {
            name: format!("{subdir}/layout.json"),
        });
    }
    let size = std::fs::metadata(&path)
        .map_err(|e| ModelError::IoFailed {
            call: "stat".to_string(),
            detail: e.to_string(),
        })?
        .len();
    if size > max_bytes {
        return Err(ModelError::IndexCorrupt {
            detail: format!("layout.json size {size} exceeds metadata cap {max_bytes}"),
        });
    }
    let data = std::fs::read(&path).map_err(|e| ModelError::IoFailed {
        call: "read".to_string(),
        detail: e.to_string(),
    })?;
    let root: Value = serde_json::from_slice(&data).map_err(|e| ModelError::IndexCorrupt {
        detail: format!("layout.json: {e}"),
    })?;

    let corrupt = |detail: &str| ModelError::IndexCorrupt {
        detail: format!("layout.json: {detail}"),
    };

    let expert_stride = root
        .get("expertStride")
        .and_then(Value::as_u64)
        .ok_or_else(|| corrupt("missing top-level keys"))?;
    let num_layers = root
        .get("numLayers")
        .and_then(Value::as_u64)
        .ok_or_else(|| corrupt("missing top-level keys"))? as usize;
    let experts_per_layer = root
        .get("expertsPerLayer")
        .and_then(Value::as_u64)
        .ok_or_else(|| corrupt("missing top-level keys"))? as usize;
    let layers_arr = root
        .get("layers")
        .and_then(Value::as_array)
        .ok_or_else(|| corrupt("missing top-level keys"))?;

    let mut layers = Vec::with_capacity(layers_arr.len());
    for layer_obj in layers_arr {
        let layer_idx = layer_obj
            .get("layer")
            .and_then(Value::as_u64)
            .ok_or_else(|| corrupt("malformed layer entry"))? as usize;
        // `PackedExpertsLayout::expert()` and `committed_bytes` index
        // `layers[layer]` BY POSITION, so a layout whose `layer` fields are
        // out of order or gapped resolves the wrong blob with no error at
        // the dispatch site. Refusing here, at load, is the one place that
        // can name which layer disagreed.
        if layer_idx != layers.len() {
            return Err(corrupt(&format!(
                "layers must appear in order 0..numLayers with no gaps; expected layer \
                 {} next, found {layer_idx}",
                layers.len()
            )));
        }
        let file = layer_obj
            .get("file")
            .and_then(Value::as_str)
            .ok_or_else(|| corrupt("malformed layer entry"))?
            .to_string();
        let experts_arr = layer_obj
            .get("experts")
            .and_then(Value::as_array)
            .ok_or_else(|| corrupt("malformed layer entry"))?;
        // Absent on every install written before ROADMAP Phase S, where the
        // stride really was model-wide, so the top-level value is the right
        // fallback rather than an error.
        let layer_stride = layer_obj
            .get("expertStride")
            .and_then(Value::as_u64)
            .unwrap_or(expert_stride);
        if layer_stride > expert_stride {
            return Err(corrupt(&format!(
                "layer {layer_idx} declares stride {layer_stride}, above the top-level \
                 {expert_stride} that sizes every consumer's slot"
            )));
        }

        let mut experts: Vec<Option<ExpertEntry>> = vec![None; experts_per_layer];
        for expert_obj in experts_arr {
            let offset = expert_obj
                .get("offset")
                .and_then(Value::as_u64)
                .ok_or_else(|| corrupt("malformed expert entry"))?;
            let size = expert_obj
                .get("size")
                .and_then(Value::as_u64)
                .ok_or_else(|| corrupt("malformed expert entry"))?;
            if size > layer_stride {
                return Err(corrupt(&format!(
                    "layer {layer_idx} expert blob size {size} exceeds its stride {layer_stride}"
                )));
            }
            let tensors_obj = expert_obj
                .get("tensors")
                .and_then(Value::as_object)
                .ok_or_else(|| corrupt("malformed expert entry"))?;

            let mut sub_tensors = BTreeMap::new();
            for (role, t) in tensors_obj {
                let toff = t.get("offset").and_then(Value::as_u64);
                let tsize = t.get("size").and_then(Value::as_u64);
                let dtype = t.get("dtype").and_then(Value::as_str);
                let shape = t.get("shape").and_then(Value::as_array);
                let (Some(toff), Some(tsize)) = (toff, tsize) else {
                    return Err(corrupt(&format!("malformed tensor {role}")));
                };
                let (Some(dtype), Some(shape)) = (dtype, shape) else {
                    return Err(corrupt(&format!("malformed tensor {role}")));
                };
                let end = toff
                    .checked_add(tsize)
                    .ok_or_else(|| corrupt(&format!("tensor {role} byte range overflows u64")))?;
                if end > size || end > layer_stride {
                    return Err(corrupt(&format!(
                        "tensor {role} byte range {toff}..{end} exceeds expert blob size {size}"
                    )));
                }
                if matches!(role.as_str(), "gate_biases" | "up_biases" | "down_biases") {
                    // A routed bias is written by the writer whose bytes the
                    // layout describes, and the families disagree by DESIGN:
                    // the quantized families carry BF16/FP16 companion
                    // planes shaped like the scales (`[rows, groups]`; the
                    // real gemma4 install's `down_biases` is bf16
                    // `[2816, 11]`), while gpt-oss's per-expert biases are
                    // F32 and rank-2 (`[width, experts]`, accepted by
                    // `gguf_checkpoint/plan.rs::routed_body_dims` for
                    // exactly these three roles). What every family shares
                    // is the arithmetic: the byte size must equal the
                    // shape's element count times the dtype's width, and
                    // the offset must be aligned to that width. An F32-
                    // flat-rows-only contract here refused the repo's own
                    // writer output on BOTH conventions and every MoE
                    // install with them. Absence stays legal: experts with
                    // no bias write no `_biases` role at all.
                    let width = if dtype.eq_ignore_ascii_case("f32") {
                        4u64
                    } else if dtype.eq_ignore_ascii_case("bf16")
                        || dtype.eq_ignore_ascii_case("f16")
                    {
                        2
                    } else {
                        return Err(corrupt(&format!(
                            "tensor {role} bias dtype must be F32, BF16 or FP16, not {dtype}"
                        )));
                    };
                    let mut elements = 1u64;
                    let mut malformed = false;
                    for d in shape {
                        match d.as_u64() {
                            Some(v) => elements = elements.saturating_mul(v),
                            None => malformed = true,
                        }
                    }
                    let expected_size = elements.checked_mul(width).ok_or_else(|| {
                        corrupt(&format!("tensor {role} bias byte count overflows u64"))
                    })?;
                    if malformed || elements == 0 {
                        return Err(corrupt(&format!(
                            "tensor {role} bias shape must name at least one element"
                        )));
                    }
                    if tsize != expected_size {
                        return Err(corrupt(&format!(
                            "tensor {role} bias must be {expected_size} bytes for its {dtype} \
                             shape, found {tsize} bytes"
                        )));
                    }
                    if toff % width != 0 {
                        return Err(corrupt(&format!(
                            "tensor {role} bias offset {toff} is not {width}-byte aligned"
                        )));
                    }
                }
                if let Some(bits) = t.get("bits") {
                    if !bits.is_i64() && !bits.is_u64() {
                        return Err(corrupt(&format!("malformed tensor bits {role}")));
                    }
                }
                sub_tensors.insert(
                    role.clone(),
                    SubTensorEntry {
                        offset: toff,
                        size: tsize,
                        dtype: dtype.to_ascii_lowercase(),
                    },
                );
            }

            let expert_id = expert_obj
                .get("expert")
                .and_then(Value::as_u64)
                .map(|v| v as usize)
                .unwrap_or_else(|| experts.iter().filter(|e| e.is_some()).count());
            if expert_id >= experts_per_layer {
                return Err(corrupt("expert id out of range"));
            }
            if let Some(rank) = expert_obj.get("physicalRank").and_then(Value::as_u64) {
                if rank as usize >= experts_per_layer {
                    return Err(corrupt("physicalRank out of range"));
                }
            }
            // A duplicate id would otherwise overwrite silently, resolving
            // one of the two blobs to nothing and leaving the OTHER
            // `expert_id` slot permanently `None` -- caught below as
            // "missing expert entries" with no hint that the real cause was
            // a duplicate rather than an omission.
            if experts[expert_id].is_some() {
                return Err(corrupt(&format!(
                    "layer {layer_idx} declares expert {expert_id} more than once"
                )));
            }
            experts[expert_id] = Some(ExpertEntry {
                expert: expert_id,
                offset,
                size,
                sub_tensors,
            });
        }
        if experts.iter().any(Option::is_none) {
            return Err(corrupt("missing expert entries"));
        }
        let experts: Vec<ExpertEntry> = experts.into_iter().map(Option::unwrap).collect();
        if let Some(first) = experts.first() {
            for expert in &experts[1..] {
                if expert.size != first.size || expert.sub_tensors != first.sub_tensors {
                    return Err(corrupt(&format!(
                        "layer {layer_idx} expert {} layout differs from expert {}",
                        expert.expert, first.expert
                    )));
                }
            }
        }
        layers.push(LayerLayout {
            layer: layer_idx,
            file,
            expert_stride: layer_stride,
            experts,
        });
    }
    if layers.len() != num_layers {
        return Err(corrupt(&format!(
            "numLayers declares {num_layers} but the layers array has {}",
            layers.len()
        )));
    }

    Ok(PackedExpertsLayout {
        expert_stride,
        num_layers,
        experts_per_layer,
        layers,
    })
}
