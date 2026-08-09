//! `packed_experts/layout.json` decode: per-layer, per-expert byte offsets
//! inside the packed-expert blob files. Ported from
//! `Infrastructure/ModelIO/PackedExpertsLayout.swift`.

use std::collections::BTreeMap;
use std::path::Path;

use serde_json::Value;

use crate::error::ModelError;

#[derive(Debug, Clone, PartialEq)]
pub struct SubTensorEntry {
    /// Offset relative to the expert blob's start.
    pub offset: u64,
    /// Bytes; scale slices encode the group count.
    pub size: u64,
}

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

#[derive(Debug, Clone, PartialEq)]
pub struct LayerLayout {
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
    pub experts: Vec<ExpertEntry>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PackedExpertsLayout {
    /// The model-wide MAXIMUM of [`LayerLayout::expert_stride`], which is what
    /// `manifest.json`'s `expertStride` declares and what a consumer wanting
    /// one number for "how big can an expert blob be" should read. Do NOT use
    /// it to size or address a specific layer: see the field above.
    pub expert_stride: u64,
    pub num_layers: usize,
    pub experts_per_layer: usize,
    pub layers: Vec<LayerLayout>,
}

impl PackedExpertsLayout {
    /// Resolve `(layer, expert)` -> `ExpertEntry`. O(1).
    pub fn expert(&self, layer: usize, expert: usize) -> &ExpertEntry {
        &self.layers[layer].experts[expert]
    }
}

// 64 MiB: Qwen 3.6's 40 layers x 256 experts x 9 sub-tensors produce a
// ~22 MB layout.json; Gemma's is ~5 MB.
pub const DEFAULT_MAX_BYTES: u64 = 64 * 1024 * 1024;

pub fn load(dir: &Path, max_bytes: u64) -> Result<PackedExpertsLayout, ModelError> {
    let path = dir.join("packed_experts").join("layout.json");
    if !path.exists() {
        return Err(ModelError::MissingFile {
            name: "packed_experts/layout.json".to_string(),
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
            let tensors_obj = expert_obj
                .get("tensors")
                .and_then(Value::as_object)
                .ok_or_else(|| corrupt("malformed expert entry"))?;

            let mut sub_tensors = BTreeMap::new();
            for (role, t) in tensors_obj {
                let toff = t.get("offset").and_then(Value::as_u64);
                let tsize = t.get("size").and_then(Value::as_u64);
                let has_dtype = t.get("dtype").is_some_and(Value::is_string);
                let has_shape = t.get("shape").is_some_and(Value::is_array);
                let (Some(toff), Some(tsize)) = (toff, tsize) else {
                    return Err(corrupt(&format!("malformed tensor {role}")));
                };
                if !has_dtype || !has_shape {
                    return Err(corrupt(&format!("malformed tensor {role}")));
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
        layers.push(LayerLayout {
            layer: layer_idx,
            file,
            expert_stride: layer_stride,
            experts: experts.into_iter().map(Option::unwrap).collect(),
        });
    }

    Ok(PackedExpertsLayout {
        expert_stride,
        num_layers,
        experts_per_layer,
        layers,
    })
}
