//! Manifest metadata derived from a staged image install.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use super::builder_files::collect_files;
use crate::install::{
    required_metadata_for, ImageComponent, ImageManifest, ImageSupported, IMAGE_BATCH,
    IMAGE_CAPABILITY, IMAGE_FORWARDS, IMAGE_GUIDANCE, IMAGE_HEIGHT, IMAGE_MANIFEST_MAGIC,
    IMAGE_MANIFEST_SCHEMA, IMAGE_PROMPT_MAX_TOKENS, IMAGE_STEPS, IMAGE_WIDTH,
};
use crate::packed::{PackedIndex, PackedTensorReport, PACKED_INDEX_NAME};

const COMPONENTS_DIR: &str = "components";

pub(super) fn make_manifest(
    spec: &super::ImageInstallSpec,
    staging: &Path,
    reports: &BTreeMap<String, PackedTensorReport>,
) -> Result<ImageManifest, String> {
    let indices = load_packed_indices(staging)?;
    let text = reports
        .get("text_encoder")
        .ok_or_else(|| "text encoder pack report is missing".to_string())?;
    let transformer = reports
        .get("transformer")
        .ok_or_else(|| "transformer pack report is missing".to_string())?;
    let vae = reports
        .get("vae_decoder")
        .ok_or_else(|| "VAE pack report is missing".to_string())?;

    let mut components = BTreeMap::new();
    components.insert(
        "tokenizer".to_string(),
        component("text_encoder_stage", tokenizer_metadata(staging)?),
    );
    components.insert(
        "text_encoder".to_string(),
        component(
            "text_encoder_stage",
            text_encoder_metadata(&spec.model_revision, text, &indices["text_encoder"]),
        ),
    );
    components.insert(
        "transformer".to_string(),
        component(
            "transformer_stage",
            transformer_metadata(&spec.model_revision, transformer, &indices["transformer"]),
        ),
    );
    components.insert(
        "scheduler".to_string(),
        component("pipeline_control", scheduler_metadata(staging)?),
    );
    components.insert(
        "vae_decoder".to_string(),
        component(
            "vae_stage",
            vae_metadata(&spec.model_revision, vae, &indices["vae_decoder"]),
        ),
    );

    Ok(ImageManifest {
        magic: IMAGE_MANIFEST_MAGIC.to_string(),
        version: IMAGE_MANIFEST_SCHEMA,
        capability: IMAGE_CAPABILITY.to_string(),
        source: serde_json::json!({
            "model_id": &spec.model_id,
            "model_revision": &spec.model_revision,
            "license": "Apache-2.0",
        }),
        supported: ImageSupported {
            width: IMAGE_WIDTH,
            height: IMAGE_HEIGHT,
            batch: IMAGE_BATCH,
            scheduler_steps: IMAGE_STEPS,
            transformer_forwards: IMAGE_FORWARDS,
            guidance_scale: IMAGE_GUIDANCE,
            prompt_max_tokens: IMAGE_PROMPT_MAX_TOKENS,
            latent_shape: vec![1, 16, 128, 128],
        },
        components,
        files: collect_files(staging, reports)?,
        verification: serde_json::json!({
            "hash": "sha256",
            "packed_index": "tensor_inventory_sha256",
            "runtime_gate": "ig1-captured-inputs-and-real-vae",
        }),
        resource_envelope: serde_json::json!({
            "reference_contract": "docs/verification/z-image-ig0-resource-contract.json",
            "ownership": "one-heavyweight-stage-at-a-time",
        }),
    })
}

fn component(owner: &str, mut metadata: BTreeMap<String, serde_json::Value>) -> ImageComponent {
    let name = metadata
        .get("component_name")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("")
        .to_string();
    metadata.remove("component_name");
    let required_metadata = required_metadata_for(&name)
        .iter()
        .map(|value| serde_json::Value::String((*value).to_string()))
        .collect();
    metadata.insert(
        "required_metadata".to_string(),
        serde_json::Value::Array(required_metadata),
    );
    ImageComponent {
        required: true,
        owner: owner.to_string(),
        metadata,
    }
}

fn tokenizer_metadata(staging: &Path) -> Result<BTreeMap<String, serde_json::Value>, String> {
    let mut assets = BTreeMap::new();
    for file in
        super::builder_files::collect_paths(&staging.join(COMPONENTS_DIR).join("tokenizer"))?
    {
        let rel = file
            .strip_prefix(staging)
            .map_err(|e| format!("tokenizer path is outside staging: {e}"))?;
        let bytes =
            fs::read(&file).map_err(|e| format!("failed to read {}: {e}", file.display()))?;
        assets.insert(
            rel.to_string_lossy().replace('\\', "/"),
            serde_json::Value::String(model_io::hash_data(&bytes)),
        );
    }
    if assets.is_empty() {
        return Err("image tokenizer source contains no files".to_string());
    }
    Ok(BTreeMap::from([
        ("component_name".to_string(), serde_json::json!("tokenizer")),
        (
            "asset_paths".to_string(),
            serde_json::json!(assets.keys().collect::<Vec<_>>()),
        ),
        ("asset_sha256".to_string(), serde_json::json!(assets)),
        (
            "chat_template".to_string(),
            serde_json::json!("tokenizer_config.json:chat_template"),
        ),
        ("enable_thinking".to_string(), serde_json::json!(true)),
        (
            "truncation_policy".to_string(),
            serde_json::json!("right_truncate"),
        ),
        ("padding_policy".to_string(), serde_json::json!("right_pad")),
        ("max_tokens".to_string(), serde_json::json!(512)),
    ]))
}

fn scheduler_metadata(staging: &Path) -> Result<BTreeMap<String, serde_json::Value>, String> {
    let path = staging.join(COMPONENTS_DIR).join("scheduler/config.json");
    let bytes = fs::read(&path).map_err(|e| format!("failed to read scheduler config: {e}"))?;
    Ok(BTreeMap::from([
        ("component_name".to_string(), serde_json::json!("scheduler")),
        (
            "config_sha256".to_string(),
            serde_json::json!(model_io::hash_data(&bytes)),
        ),
        ("num_train_timesteps".to_string(), serde_json::json!(1000)),
        ("shift".to_string(), serde_json::json!(3.0)),
        ("sigma_endpoints".to_string(), serde_json::json!([1.0, 0.0])),
        ("evaluation_count".to_string(), serde_json::json!(9)),
        ("guidance_policy".to_string(), serde_json::json!("zero")),
        (
            "noise_provenance".to_string(),
            serde_json::json!("request-seed"),
        ),
    ]))
}

fn text_encoder_metadata(
    revision: &str,
    report: &PackedTensorReport,
    index: &PackedIndex,
) -> BTreeMap<String, serde_json::Value> {
    BTreeMap::from([
        (
            "component_name".to_string(),
            serde_json::json!("text_encoder"),
        ),
        (
            "canonical_revision".to_string(),
            serde_json::json!(revision),
        ),
        (
            "tensor_inventory".to_string(),
            serde_json::json!(&report.tensor_inventory_sha256),
        ),
        (
            "hidden_state_output".to_string(),
            serde_json::json!("hidden_states[-2]"),
        ),
        ("causal_mask".to_string(), serde_json::json!("causal")),
        ("qk_norm_epsilon".to_string(), serde_json::json!(1e-6)),
        (
            "quantized_tensor_names".to_string(),
            serde_json::json!(quantized_names(index)),
        ),
        (
            "protected_tensor_names".to_string(),
            serde_json::json!(protected_names(index)),
        ),
    ])
}

fn transformer_metadata(
    revision: &str,
    report: &PackedTensorReport,
    index: &PackedIndex,
) -> BTreeMap<String, serde_json::Value> {
    let quantization_scheme = super::builder_files::quantization_scheme(index);
    let quantization_label = super::builder_files::quantization_label(index);
    let mlx_affine = quantization_scheme == crate::runtime::IMAGE_MLX_QUANTIZATION;
    let nibble_order = if mlx_affine {
        "lsb_first_bit_fields"
    } else {
        "low_nibble_even_high_nibble_odd"
    };
    let scale_and_bias_convention = if mlx_affine {
        "value = q * scale + bias"
    } else {
        "value = nibble * bf16_scale + bf16_bias"
    };
    BTreeMap::from([
        (
            "component_name".to_string(),
            serde_json::json!("transformer"),
        ),
        (
            "canonical_revision".to_string(),
            serde_json::json!(revision),
        ),
        (
            "tensor_inventory".to_string(),
            serde_json::json!(&report.tensor_inventory_sha256),
        ),
        (
            "patch_order".to_string(),
            serde_json::json!("channel-height-width"),
        ),
        (
            "learned_pad_token".to_string(),
            serde_json::json!("x_pad_token-and-cap_pad_token"),
        ),
        (
            "sequence_padding".to_string(),
            serde_json::json!("multiple-of-32"),
        ),
        (
            "timestep_modulation".to_string(),
            serde_json::json!("adaLN"),
        ),
        ("three_axis_rope".to_string(), serde_json::json!(true)),
        (
            "quantization_scheme".to_string(),
            serde_json::json!(quantization_scheme),
        ),
        (
            "quantization_label".to_string(),
            serde_json::json!(quantization_label),
        ),
        ("group_size".to_string(), serde_json::json!(64)),
        (
            "affine_bit_widths".to_string(),
            serde_json::json!(crate::packed::MLX_AFFINE_BITS),
        ),
        (
            "observed_affine_bit_widths".to_string(),
            serde_json::json!(super::builder_files::observed_mlx_bit_widths(index)),
        ),
        ("nibble_order".to_string(), serde_json::json!(nibble_order)),
        (
            "scale_and_bias_convention".to_string(),
            serde_json::json!(scale_and_bias_convention),
        ),
        (
            "quantized_tensor_names".to_string(),
            serde_json::json!(quantized_names(index)),
        ),
        (
            "protected_tensor_names".to_string(),
            serde_json::json!(protected_names(index)),
        ),
    ])
}

fn vae_metadata(
    revision: &str,
    report: &PackedTensorReport,
    index: &PackedIndex,
) -> BTreeMap<String, serde_json::Value> {
    BTreeMap::from([
        (
            "component_name".to_string(),
            serde_json::json!("vae_decoder"),
        ),
        (
            "canonical_revision".to_string(),
            serde_json::json!(revision),
        ),
        (
            "decoder_tensor_subset".to_string(),
            serde_json::json!("decoder-only"),
        ),
        (
            "latent_layout".to_string(),
            serde_json::json!("[16,height,width]"),
        ),
        (
            "normalization".to_string(),
            serde_json::json!("group-norm-32"),
        ),
        (
            "scale_and_shift".to_string(),
            serde_json::json!({"scale": 0.3611, "shift": 0.1159}),
        ),
        ("output_range".to_string(), serde_json::json!("[-1,1]")),
        (
            "pixel_conversion".to_string(),
            serde_json::json!("clamp-and-round-rgb8"),
        ),
        (
            "quantized_tensor_names".to_string(),
            serde_json::json!(quantized_names(index)),
        ),
        (
            "protected_tensor_names".to_string(),
            serde_json::json!(protected_names(index)),
        ),
        (
            "tensor_inventory".to_string(),
            serde_json::json!(&report.tensor_inventory_sha256),
        ),
    ])
}

fn load_packed_indices(staging: &Path) -> Result<BTreeMap<String, PackedIndex>, String> {
    let mut indices = BTreeMap::new();
    for name in ["text_encoder", "transformer", "vae_decoder"] {
        let path = staging
            .join(COMPONENTS_DIR)
            .join(name)
            .join(PACKED_INDEX_NAME);
        let bytes =
            fs::read(&path).map_err(|e| format!("failed to read {}: {e}", path.display()))?;
        let index = serde_json::from_slice(&bytes)
            .map_err(|e| format!("failed to parse packed index {}: {e}", path.display()))?;
        indices.insert(name.to_string(), index);
    }
    Ok(indices)
}

fn quantized_names(index: &PackedIndex) -> Vec<String> {
    index
        .tensors
        .iter()
        .filter(|(_, tensor)| matches!(tensor.storage_dtype.as_str(), "INT4_AFFINE" | "MLX_AFFINE"))
        .map(|(name, _)| name.clone())
        .collect()
}

fn protected_names(index: &PackedIndex) -> Vec<String> {
    index
        .tensors
        .iter()
        .filter(|(_, tensor)| {
            !matches!(tensor.storage_dtype.as_str(), "INT4_AFFINE" | "MLX_AFFINE")
        })
        .map(|(name, _)| name.clone())
        .collect()
}
