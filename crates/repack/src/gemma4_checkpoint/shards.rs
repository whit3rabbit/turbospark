//! Multi-shard reading, tensor classification, and resident ordering.

use model_io::ModelFamily;

use super::config::{Gemma4Error, Gemma4Quant};
use crate::ranged_download::RangeSource;
use crate::resident_writer::{
    ResidentEntrySpec, ResidentTensorSpec, DTYPE_BF16, DTYPE_FP16, DTYPE_FP32,
};
use crate::safetensors_header::{SafetensorsHeader, TensorInfo};

/// On-disk page alignment unit for `.gturbo` files (the Swift repacker's
/// `Layout.pageBytes`): fixed at 16 KiB regardless of host page size.
pub const GTURBO_PAGE_BYTES: u64 = 16_384;

/// Classification bucket for a Gemma 4 source checkpoint tensor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Gemma4Bucket {
    /// Text model resident tensor loaded permanently into RAM/VRAM.
    LmResident,
    /// Routed expert tensor with role ('gate', 'up', 'down') and layer index.
    RoutedExpert {
        /// Role of the expert tensor ('gate', 'up', or 'down').
        role: &'static str,
        /// Layer index containing the expert.
        layer: usize,
    },
    /// Multimodal vision or audio tensor excluded from the text language model.
    ExcludedMultimodal,
    /// Tensor not matching known language model or multimodal patterns.
    Unknown,
}

/// Extracts the layer index from a layer-scoped tensor name (e.g. `...layers.12...`).
pub fn layer_index(name: &str) -> Option<usize> {
    let tail = &name[name.find(".layers.")? + ".layers.".len()..];
    tail[..tail.find('.')?].parse().ok()
}

/// The container path a family's routed (per-expert) weights live under.
fn routed_marker(family: ModelFamily) -> &'static str {
    match family {
        ModelFamily::Qwen36 => ".mlp.switch_mlp.",
        // DeepSeek V4 has no repack path yet; Gemma's marker is the default.
        ModelFamily::Gemma4 | ModelFamily::DeepseekV4Flash => ".experts.switch_glu.",
    }
}

/// Classifies source tensor name under specified model family contract.
pub fn classify_for_family(name: &str, num_layers: usize, family: ModelFamily) -> Gemma4Bucket {
    if name.starts_with("language_model.") {
        if name.contains(routed_marker(family)) {
            let role = if name.contains(".gate_proj.") {
                Some("gate")
            } else if name.contains(".up_proj.") {
                Some("up")
            } else if name.contains(".down_proj.") {
                Some("down")
            } else {
                None
            };
            if let (Some(role), Some(layer)) = (role, layer_index(name)) {
                if layer < num_layers {
                    return Gemma4Bucket::RoutedExpert { role, layer };
                }
            }
        }
        return Gemma4Bucket::LmResident;
    }
    if name.starts_with("vision_tower.")
        || name.starts_with("embed_vision.")
        || name.starts_with("audio_tower.")
    {
        return Gemma4Bucket::ExcludedMultimodal;
    }
    Gemma4Bucket::Unknown
}

/// Classify a source tensor name under the Gemma 4 family contract: the
/// text tower lives under `language_model.`, routed experts under
/// `.experts.switch_glu.`, and the vision/audio towers are excluded.
pub fn classify_gemma4(name: &str, num_layers: usize) -> Gemma4Bucket {
    classify_for_family(name, num_layers, ModelFamily::Gemma4)
}

/// Within-layer slot order, mirroring `RepackPlanner.swift`'s Gemma table.
pub fn slot_rank(n: &str) -> usize {
    const CONTAINS: [&str; 12] = [
        ".self_attn.q_proj.weight",
        ".self_attn.k_proj.weight",
        ".self_attn.v_proj.weight",
        ".self_attn.o_proj.weight",
        ".self_attn.q_norm.weight",
        ".self_attn.k_norm.weight",
        ".router.proj.weight",
        ".router.scale",
        ".router.per_expert_scale",
        ".mlp.gate_proj.weight",
        ".mlp.up_proj.weight",
        ".mlp.down_proj.weight",
    ];
    for (rank, pat) in CONTAINS.iter().enumerate() {
        if n.contains(pat) {
            return rank;
        }
    }
    const SUFFIXES: [&str; 8] = [
        ".input_layernorm.weight",
        ".post_attention_layernorm.weight",
        ".pre_feedforward_layernorm.weight",
        ".pre_feedforward_layernorm_2.weight",
        ".post_feedforward_layernorm.weight",
        ".post_feedforward_layernorm_1.weight",
        ".post_feedforward_layernorm_2.weight",
        ".layer_scalar",
    ];
    for (j, pat) in SUFFIXES.iter().enumerate() {
        if n.ends_with(pat) {
            return CONTAINS.len() + j;
        }
    }
    100
}

/// Stable resident order: embedding first, then per-layer groups in layer
/// order (slot-ranked within a layer), then top-level extras, the final
/// norm, and `lm_head` last.
pub fn lm_order_key(n: &str) -> (usize, usize, usize, &str) {
    match n {
        "language_model.model.embed_tokens.weight" => (0, 0, 0, n),
        "language_model.model.norm.weight" => (3, 0, 0, n),
        "language_model.lm_head.weight" => (4, 0, 0, n),
        _ => match layer_index(n) {
            Some(li) => (1, li, slot_rank(n), n),
            None => (2, 0, 0, n),
        },
    }
}

/// A multi-shard checkpoint view: real HF checkpoints split their tensors
/// across several `model-NNNNN-of-NNNNN.safetensors` files (the
/// `model.safetensors.index.json` weight map), and companion tensors may
/// live in a different shard than their weight -- so lookups go through one
/// merged name registry, exactly like the Swift planner's `registry`.
pub struct Gemma4Shards<'a> {
    shards: Vec<(&'a SafetensorsHeader, &'a dyn RangeSource)>,
    by_name: std::collections::HashMap<&'a str, usize>,
}

impl<'a> Gemma4Shards<'a> {
    /// Creates multi-shard tensor registry mapping names to shard index.
    pub fn new(shards: Vec<(&'a SafetensorsHeader, &'a dyn RangeSource)>) -> Self {
        let mut by_name = std::collections::HashMap::new();
        for (i, (header, _)) in shards.iter().enumerate() {
            for name in header.tensors.keys() {
                by_name.insert(name.as_str(), i);
            }
        }
        Self { shards, by_name }
    }

    /// Creates single-shard tensor registry wrapper.
    pub fn single(header: &'a SafetensorsHeader, source: &'a dyn RangeSource) -> Self {
        Self::new(vec![(header, source)])
    }

    /// Resolves the shard header and byte source for a given tensor name.
    pub fn shard_of(
        &self,
        name: &str,
    ) -> Result<&(&'a SafetensorsHeader, &'a dyn RangeSource), Gemma4Error> {
        let i = *self
            .by_name
            .get(name)
            .ok_or_else(|| Gemma4Error::MissingTensor(name.to_string()))?;
        Ok(&self.shards[i])
    }

    /// Looks up metadata information for a given tensor name across shards.
    pub fn info(&self, name: &str) -> Result<&'a TensorInfo, Gemma4Error> {
        let (header, _) = self.shard_of(name)?;
        header
            .tensors
            .get(name)
            .ok_or_else(|| Gemma4Error::MissingTensor(name.to_string()))
    }

    /// Returns true if a tensor with the given name exists in any shard.
    pub fn contains(&self, name: &str) -> bool {
        self.by_name.contains_key(name)
    }

    /// Reads raw tensor byte payload from its hosting shard range source.
    pub fn read(&self, name: &str) -> Result<Vec<u8>, Gemma4Error> {
        let (header, source) = self.shard_of(name)?;
        let (start, end) = header
            .absolute_range(name)
            .ok_or_else(|| Gemma4Error::MissingTensor(name.to_string()))?;
        Ok(source.read_range(start, end)?)
    }

    /// Returns an iterator over all tensor names present across all shards.
    pub fn names(&self) -> impl Iterator<Item = &'a String> + '_ {
        self.shards.iter().flat_map(|(h, _)| h.tensors.keys())
    }
}

/// Converts string data type to raw byte dtype tag.
pub fn raw_dtype_tag(tensor: &str, dtype: &str) -> Result<u8, Gemma4Error> {
    match dtype {
        "BF16" => Ok(DTYPE_BF16),
        "F16" => Ok(DTYPE_FP16),
        "F32" => Ok(DTYPE_FP32),
        other => Err(Gemma4Error::UnsupportedDtype {
            tensor: tensor.to_string(),
            dtype: other.to_string(),
        }),
    }
}

/// Normalizes tensor shape slice to 4-tuple of u32 dimensions.
pub fn shape4(shape: &[u64]) -> (u32, u32, u32, u32) {
    let get = |i: usize| shape.get(i).copied().unwrap_or(0) as u32;
    (get(0), get(1), get(2), get(3))
}

/// Packs quantized tensor weights and companion scale/bias arrays into resident entry spec.
pub fn pass_through_packed(
    shards: &Gemma4Shards<'_>,
    name: &str,
    quant: &Gemma4Quant,
) -> Result<ResidentEntrySpec, Gemma4Error> {
    let w = shards.info(name)?;
    if w.shape.len() != 2 {
        return Err(Gemma4Error::ShapeMismatch {
            tensor: name.to_string(),
            detail: format!("expected rank-2 packed weight, got {:?}", w.shape),
        });
    }
    let base = name.strip_suffix(".weight").unwrap_or(name);
    let bits = quant.bits_for(base);
    let factor = match bits {
        4 | 8 => 32 / bits as u64,
        other => {
            return Err(Gemma4Error::UnsupportedDtype {
                tensor: name.to_string(),
                dtype: format!("{other}-bit quantization"),
            })
        }
    };
    let scales_name = format!("{base}.scales");
    let biases_name = format!("{base}.biases");
    for companion in [&scales_name, &biases_name] {
        if !shards.contains(companion) {
            return Err(Gemma4Error::MissingCompanion(name.to_string()));
        }
        let c = shards.info(companion)?;
        if c.dtype != "BF16" {
            return Err(Gemma4Error::UnsupportedDtype {
                tensor: companion.to_string(),
                dtype: c.dtype.clone(),
            });
        }
    }
    let rows = w.shape[0];
    let cols = w.shape[1] * factor;
    let packed = shards.read(name)?;
    let scales = le_u16(&shards.read(&scales_name)?);
    let biases = le_u16(&shards.read(&biases_name)?);
    let expected_groups = (rows * cols / 64) as usize;
    if cols % 64 != 0 || scales.len() != expected_groups || biases.len() != expected_groups {
        return Err(Gemma4Error::ShapeMismatch {
            tensor: name.to_string(),
            detail: format!(
                "shape {rows}x{cols} with {} scales / {} biases does not match \
                 the 64-element groups this port's kernels assume",
                scales.len(),
                biases.len()
            ),
        });
    }
    let spec = ResidentTensorSpec {
        name: name.to_string(),
        packed,
        scales,
        biases,
        rows: rows as u32,
        cols: cols as u32,
    };
    Ok(match bits {
        4 => ResidentEntrySpec::Int4(spec),
        _ => ResidentEntrySpec::Int8(spec),
    })
}

/// Converts little-endian byte slice into u16 vector.
pub fn le_u16(bytes: &[u8]) -> Vec<u16> {
    bytes
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect()
}
