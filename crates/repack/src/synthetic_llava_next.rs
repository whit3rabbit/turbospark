//! A tiny LLaVA 1.6 (`llava_next`) tower checkpoint under its published names,
//! for the intake and runtime tests (`docs/LLAVA_NEXT_VISION_PHASE0.md`).
//!
//! Only the sizes are shrunk. The 336 pixel image, the patch of 14, the
//! 577-row position table and the five grid pinpoints are the published ones,
//! because the preprocessing and the packing are built on them.

use std::path::Path;

use crate::gemma4_checkpoint::{
    parse_llava_next_vision_config, parse_llava_vision_config, read_vision_entries,
    write_vision_sidecar, Gemma4Shards, CLIP_MODEL_PREFIX, CLIP_NEXT_NEWLINE,
};
use crate::ranged_download::MemoryRangeSource;
use crate::safetensors_header::{parse_header, DEFAULT_MAX_HEADER_BYTES};
use crate::synthetic_llama::SyntheticTowerTensor;
use crate::synthetic_tensors::{assemble_safetensors, deterministic_row, u16_le, Tensor};

/// The tower's own width.
pub const SYNTHETIC_NEXT_HIDDEN: usize = 16;
/// The tower's attention heads.
pub const SYNTHETIC_NEXT_HEADS: usize = 2;
/// The `<image>` id the config declares.
pub const SYNTHETIC_NEXT_IMAGE_ID: i64 = 200;

/// The files a `pull-vision` of the checkpoint reads.
pub struct SyntheticLlavaNext {
    pub config_json: String,
    pub preprocessor_json: String,
    /// One `model.safetensors`.
    pub safetensors: Vec<u8>,
    /// The tensors as written, for an independent CPU forward pass.
    pub tensors: Vec<SyntheticTowerTensor>,
}

/// Build the checkpoint with a projector `out_hidden` wide. Three layers are
/// written and `vision_feature_layer -2` runs two of them.
pub fn synthetic_llava_next_checkpoint(out_hidden: usize) -> SyntheticLlavaNext {
    const H: usize = SYNTHETIC_NEXT_HIDDEN;
    const MLP: usize = 24;
    let mut tensors: Vec<SyntheticTowerTensor> = Vec::new();
    let mut raw: Vec<Tensor> = Vec::new();
    let mut add = |name: String, shape: Vec<usize>, center: f32, amplitude: f32| {
        let n: usize = shape.iter().product();
        let bits: Vec<u16> = deterministic_row(11_000 + raw.len() as u64, n)
            .iter()
            .map(|&j| compute::f32_to_bf16(center + j * amplitude))
            .collect();
        tensors.push(SyntheticTowerTensor {
            name: name.clone(),
            shape: shape.clone(),
            values: bits.iter().map(|&b| compute::bf16_to_f32(b)).collect(),
        });
        raw.push(Tensor {
            name,
            dtype: "BF16",
            shape: shape.iter().map(|&d| d as u64).collect(),
            bytes: u16_le(&bits),
        });
    };
    let base = "vision_tower.vision_model.";
    add(
        format!("{base}embeddings.patch_embedding.weight"),
        vec![H, 3, 14, 14],
        0.0,
        0.05,
    );
    add(
        format!("{base}embeddings.class_embedding"),
        vec![H],
        0.0,
        0.3,
    );
    add(
        format!("{base}embeddings.position_embedding.weight"),
        vec![577, H],
        0.0,
        0.3,
    );
    add(format!("{base}pre_layrnorm.weight"), vec![H], 1.0, 0.2);
    add(format!("{base}pre_layrnorm.bias"), vec![H], 0.0, 0.1);
    for layer in 0..3 {
        let p = format!("{base}encoder.layers.{layer}.");
        for norm in ["layer_norm1", "layer_norm2"] {
            add(format!("{p}{norm}.weight"), vec![H], 1.0, 0.2);
            add(format!("{p}{norm}.bias"), vec![H], 0.0, 0.1);
        }
        for proj in ["q_proj", "k_proj", "v_proj", "out_proj"] {
            add(format!("{p}self_attn.{proj}.weight"), vec![H, H], 0.0, 0.4);
            add(format!("{p}self_attn.{proj}.bias"), vec![H], 0.0, 0.1);
        }
        add(format!("{p}mlp.fc1.weight"), vec![MLP, H], 0.0, 0.6);
        add(format!("{p}mlp.fc1.bias"), vec![MLP], 0.0, 0.5);
        add(format!("{p}mlp.fc2.weight"), vec![H, MLP], 0.0, 0.3);
        add(format!("{p}mlp.fc2.bias"), vec![H], 0.0, 0.1);
    }
    add(format!("{base}post_layernorm.weight"), vec![H], 1.0, 0.1);
    add(format!("{base}post_layernorm.bias"), vec![H], 0.0, 0.1);
    let proj = "multi_modal_projector.";
    add(
        format!("{proj}linear_1.weight"),
        vec![out_hidden, H],
        0.0,
        0.4,
    );
    add(format!("{proj}linear_1.bias"), vec![out_hidden], 0.0, 0.2);
    add(
        format!("{proj}linear_2.weight"),
        vec![out_hidden, out_hidden],
        0.0,
        0.2,
    );
    add(format!("{proj}linear_2.bias"), vec![out_hidden], 0.0, 0.1);
    add("image_newline".to_string(), vec![out_hidden], 0.0, 0.5);

    let config_json = serde_json::json!({
        "model_type": "llava_next",
        "image_token_index": SYNTHETIC_NEXT_IMAGE_ID,
        "projector_hidden_act": "gelu",
        "use_image_newline_parameter": true,
        "vision_feature_layer": -2,
        "vision_feature_select_strategy": "default",
        "text_config": {"model_type": "llama", "hidden_size": out_hidden},
        "vision_config": {
            "model_type": "clip_vision_model", "hidden_size": H, "image_size": 336,
            "intermediate_size": MLP, "num_attention_heads": SYNTHETIC_NEXT_HEADS,
            "num_hidden_layers": 3, "patch_size": 14
        }
    })
    .to_string();
    let preprocessor_json = serde_json::json!({
        "aspect_ratio_setting": "anyres",
        "crop_size": {"height": 336, "width": 336},
        "do_center_crop": true, "do_normalize": true, "do_pad": true,
        "do_rescale": true, "do_resize": true,
        "image_grid_pinpoints": [[336, 672], [672, 336], [672, 672], [1008, 336], [336, 1008]],
        "image_mean": [0.48145466, 0.4578275, 0.40821073],
        "image_processor_type": "LlavaNextImageProcessor",
        "image_std": [0.26862954, 0.26130258, 0.27577711],
        "resample": 3, "rescale_factor": 0.00392156862745098,
        "size": {"shortest_edge": 336}
    })
    .to_string();
    SyntheticLlavaNext {
        config_json,
        preprocessor_json,
        safetensors: assemble_safetensors(&raw),
        tensors,
    }
}

/// Install the synthetic tower as a `llama` sidecar in `dir`, through the
/// parser, reader and writer the catalog uses, with its processor config
/// beside it. `image_token_id` replaces the config's `<image>` id, for a test
/// whose tokenizer numbers it differently.
pub fn build_synthetic_llava_next_sidecar(
    dir: &Path,
    out_hidden: usize,
    image_token_id: i64,
) -> Result<Vec<SyntheticTowerTensor>, Box<dyn std::error::Error>> {
    build_sidecar(dir, out_hidden, image_token_id, true)
}

/// The same weights installed as a LLaVA-1.5 sidecar (no newline, a center
/// crop processor config), so a test can hold the 1.5 path to the same
/// independent computation.
pub fn build_synthetic_llava_15_sidecar(
    dir: &Path,
    out_hidden: usize,
    image_token_id: i64,
) -> Result<Vec<SyntheticTowerTensor>, Box<dyn std::error::Error>> {
    build_sidecar(dir, out_hidden, image_token_id, false)
}

fn build_sidecar(
    dir: &Path,
    out_hidden: usize,
    image_token_id: i64,
    next: bool,
) -> Result<Vec<SyntheticTowerTensor>, Box<dyn std::error::Error>> {
    let ck = synthetic_llava_next_checkpoint(out_hidden);
    let mut config: serde_json::Value = serde_json::from_str(&ck.config_json)?;
    config["image_token_index"] = serde_json::json!(image_token_id);
    let (vision, processor) = if next {
        (
            parse_llava_next_vision_config(&config.to_string())?,
            ck.preprocessor_json.clone(),
        )
    } else {
        config["model_type"] = serde_json::json!("llava");
        let mut processor: serde_json::Value = serde_json::from_str(&ck.preprocessor_json)?;
        processor["image_processor_type"] = serde_json::json!("CLIPImageProcessor");
        (
            parse_llava_vision_config(&config.to_string())?,
            processor.to_string(),
        )
    };
    let header = parse_header(&ck.safetensors, DEFAULT_MAX_HEADER_BYTES)?;
    let source = MemoryRangeSource::new(&ck.safetensors);
    let shards = Gemma4Shards::single(&header, &source);
    // A 1.5 fetch filters by prefix, so the newline never reaches its reader.
    let bases: Vec<&str> = header
        .tensors
        .keys()
        .map(String::as_str)
        .filter(|name| next || *name != CLIP_NEXT_NEWLINE)
        .collect();
    let read = read_vision_entries(&shards, &bases, &vision)?;
    let record = model_io::SidecarRecord {
        kind: model_io::SIDECAR_KIND.to_string(),
        pairs_with: model_io::PairsWith {
            family: "llama".to_string(),
            hidden_size: vision.out_hidden_size,
        },
        source: model_io::SidecarSource {
            repo: "synthetic/llava-next".to_string(),
            revision: "main".to_string(),
            prefix: CLIP_MODEL_PREFIX.to_string(),
            file: "model.safetensors".to_string(),
        },
        tower_blocks: vision.depth,
        block_stride: read.block_stride,
    };
    write_vision_sidecar(
        dir,
        model_io::ModelFamily::Llama,
        &vision,
        "synthetic/llava-next",
        &read,
        record,
    )?;
    std::fs::write(dir.join("preprocessor_config.json"), processor)?;
    Ok(ck.tensors)
}
