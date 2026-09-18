//! Validation for the image-generation install contract.
//!
//! Image installs intentionally do not reuse the text-model manifest. Their
//! components have different owners and are admitted one heavyweight stage at
//! a time, so accepting a text manifest here would make an invalid install
//! look runnable.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Component, Path};

use serde::{Deserialize, Serialize};

pub const IMAGE_MANIFEST_NAME: &str = "manifest.json";
pub const IMAGE_RECEIPT_NAME: &str = "verified-install.json";
pub const IMAGE_MANIFEST_MAGIC: &str = "GTURBO";
pub const IMAGE_MANIFEST_SCHEMA: u32 = 1;
pub const IMAGE_CAPABILITY: &str = "image-generation";
pub const IMAGE_WIDTH: u32 = 1024;
pub const IMAGE_HEIGHT: u32 = 1024;
pub const IMAGE_BATCH: u32 = 1;
pub const IMAGE_STEPS: u32 = 9;
pub const IMAGE_FORWARDS: u32 = 9;
pub const IMAGE_GUIDANCE: f32 = 0.0;
pub const IMAGE_PROMPT_MAX_TOKENS: u32 = 512;

pub const COMPONENT_ORDER: [&str; 5] = [
    "tokenizer",
    "text_encoder",
    "transformer",
    "scheduler",
    "vae_decoder",
];

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageManifest {
    pub magic: String,
    pub version: u32,
    pub capability: String,
    pub source: serde_json::Value,
    pub supported: ImageSupported,
    pub components: BTreeMap<String, ImageComponent>,
    pub files: Vec<ImageManifestFile>,
    pub verification: serde_json::Value,
    pub resource_envelope: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageSupported {
    pub width: u32,
    pub height: u32,
    pub batch: u32,
    pub scheduler_steps: u32,
    pub transformer_forwards: u32,
    pub guidance_scale: f32,
    pub prompt_max_tokens: u32,
    pub latent_shape: Vec<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageComponent {
    pub required: bool,
    pub owner: String,
    #[serde(flatten)]
    pub metadata: BTreeMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageManifestFile {
    pub path: String,
    pub owner: String,
    pub size_bytes: u64,
    pub sha256: String,
    pub storage_dtype: String,
    pub quantization: serde_json::Value,
    pub tensor_inventory_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImageManifestValidation {
    pub file_count: usize,
    pub component_order: Vec<String>,
    pub dimensions: (u32, u32),
    pub batch: u32,
    pub scheduler_steps: u32,
    pub transformer_forwards: u32,
}

impl ImageManifest {
    pub fn load(root: &Path) -> Result<Self, String> {
        let path = root.join(IMAGE_MANIFEST_NAME);
        let bytes = fs::read(&path)
            .map_err(|e| format!("failed to read image manifest {}: {e}", path.display()))?;
        serde_json::from_slice(&bytes)
            .map_err(|e| format!("failed to parse image manifest {}: {e}", path.display()))
    }

    /// Validate the frozen IG2 admission envelope without touching payloads.
    pub fn validate(&self) -> Result<ImageManifestValidation, String> {
        if self.magic != IMAGE_MANIFEST_MAGIC {
            return Err(format!(
                "image manifest magic {:?} is not {:?}",
                self.magic, IMAGE_MANIFEST_MAGIC
            ));
        }
        if self.version != IMAGE_MANIFEST_SCHEMA {
            return Err(format!(
                "unsupported image manifest version {}, expected {}",
                self.version, IMAGE_MANIFEST_SCHEMA
            ));
        }
        if self.capability != IMAGE_CAPABILITY {
            return Err(format!(
                "manifest capability {:?} is not {:?}",
                self.capability, IMAGE_CAPABILITY
            ));
        }

        let supported = &self.supported;
        if (supported.width, supported.height) != (IMAGE_WIDTH, IMAGE_HEIGHT) {
            return Err(format!(
                "image dimensions {}x{} are outside the IG2 envelope of {}x{}",
                supported.width, supported.height, IMAGE_WIDTH, IMAGE_HEIGHT
            ));
        }
        if supported.batch != IMAGE_BATCH
            || supported.scheduler_steps != IMAGE_STEPS
            || supported.transformer_forwards != IMAGE_FORWARDS
            || supported.guidance_scale.to_bits() != IMAGE_GUIDANCE.to_bits()
            || supported.prompt_max_tokens != IMAGE_PROMPT_MAX_TOKENS
        {
            return Err(format!(
                "unsupported image envelope: batch={}, steps={}, forwards={}, guidance={}, prompt_max_tokens={}",
                supported.batch,
                supported.scheduler_steps,
                supported.transformer_forwards,
                supported.guidance_scale,
                supported.prompt_max_tokens
            ));
        }
        if supported.latent_shape != [1, 16, 128, 128] {
            return Err(format!(
                "unsupported latent shape {:?}, expected [1, 16, 128, 128]",
                supported.latent_shape
            ));
        }

        for component_name in COMPONENT_ORDER {
            let component = self.components.get(component_name).ok_or_else(|| {
                format!("image manifest is missing required component {component_name}")
            })?;
            if !component.required {
                return Err(format!(
                    "required component {component_name} is not marked required"
                ));
            }
            if component.owner.is_empty() {
                return Err(format!("component {component_name} has an empty owner"));
            }
            let required_metadata = component
                .metadata
                .get("required_metadata")
                .and_then(serde_json::Value::as_array)
                .ok_or_else(|| {
                    format!("component {component_name} is missing required_metadata")
                })?;
            if required_metadata.is_empty()
                || required_metadata
                    .iter()
                    .any(|value| value.as_str().is_none())
            {
                return Err(format!(
                    "component {component_name} has an invalid required_metadata list"
                ));
            }
            for required in required_metadata_for(component_name) {
                if !required_metadata
                    .iter()
                    .any(|value| value.as_str() == Some(required))
                {
                    return Err(format!(
                        "component {component_name} is missing required metadata {required}"
                    ));
                }
            }
        }
        if self.components.len() != COMPONENT_ORDER.len() {
            return Err(format!(
                "image manifest has {} components, expected exactly {}",
                self.components.len(),
                COMPONENT_ORDER.len()
            ));
        }
        let expected_quantization = crate::runtime::image_quantization_label(self)?;
        let declared_quantization = self.components["transformer"]
            .metadata
            .get("quantization_label")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| {
                "image transformer metadata is missing quantization_label".to_string()
            })?;
        if declared_quantization != expected_quantization {
            return Err(format!(
                "image transformer quantization label {:?} does not match {:?}",
                declared_quantization, expected_quantization
            ));
        }

        let mut paths = BTreeMap::new();
        for file in &self.files {
            validate_relative_path(&file.path)?;
            if paths.insert(file.path.clone(), ()).is_some() {
                return Err(format!(
                    "image manifest lists file {} more than once",
                    file.path
                ));
            }
            if file.size_bytes == 0 {
                return Err(format!("image manifest file {} has zero size", file.path));
            }
            validate_sha256(&file.sha256, &file.path)?;
            validate_sha256(&file.tensor_inventory_sha256, &file.path)?;
            if !self
                .components
                .values()
                .any(|component| component.owner == file.owner)
            {
                return Err(format!(
                    "image manifest file {} has unknown owner {}",
                    file.path, file.owner
                ));
            }
            if file.storage_dtype.is_empty() {
                return Err(format!(
                    "image manifest file {} has no storage_dtype",
                    file.path
                ));
            }
        }
        if self.files.is_empty() {
            return Err("image manifest declares no install files".to_string());
        }

        Ok(ImageManifestValidation {
            file_count: self.files.len(),
            component_order: COMPONENT_ORDER.iter().map(|s| (*s).to_string()).collect(),
            dimensions: (supported.width, supported.height),
            batch: supported.batch,
            scheduler_steps: supported.scheduler_steps,
            transformer_forwards: supported.transformer_forwards,
        })
    }

    /// Validate the manifest, then verify every declared payload size and hash.
    pub fn verify_files(&self, root: &Path) -> Result<ImageManifestValidation, String> {
        self.verify_files_inner(root, None)
    }

    /// Variant used while an install is staged under a temporary name. The
    /// receipt already binds the final path because publication is an atomic
    /// directory rename.
    pub(crate) fn verify_files_for_publication(
        &self,
        root: &Path,
        final_directory_path: &str,
    ) -> Result<ImageManifestValidation, String> {
        self.verify_files_inner(root, Some(final_directory_path))
    }

    fn verify_files_inner(
        &self,
        root: &Path,
        final_directory_path: Option<&str>,
    ) -> Result<ImageManifestValidation, String> {
        let validation = self.validate()?;
        for file in &self.files {
            let path = root.join(&file.path);
            let metadata = fs::metadata(&path).map_err(|e| {
                format!("failed to stat image install file {}: {e}", path.display())
            })?;
            if !metadata.is_file() {
                return Err(format!(
                    "image install path {} is not a file",
                    path.display()
                ));
            }
            if metadata.len() != file.size_bytes {
                return Err(format!(
                    "image install file {} has size {}, expected {}",
                    file.path,
                    metadata.len(),
                    file.size_bytes
                ));
            }
            let actual = model_io::hash_file(&path, 1 << 20)
                .map_err(|e| format!("failed to hash image install file {}: {e}", file.path))?;
            if actual != file.sha256.to_ascii_lowercase() {
                return Err(format!(
                    "image install file {} has SHA-256 {}, expected {}",
                    file.path, actual, file.sha256
                ));
            }
        }
        verify_receipt(self, root, final_directory_path)?;
        Ok(validation)
    }
}

fn verify_receipt(
    manifest: &ImageManifest,
    root: &Path,
    final_directory_path: Option<&str>,
) -> Result<(), String> {
    let receipt = model_io::load_install_receipt(root, model_io::INSTALL_RECEIPT_DEFAULT_MAX_BYTES)
        .map_err(|error| format!("image install receipt is invalid: {error}"))?;
    let manifest_path = root.join(IMAGE_MANIFEST_NAME);
    let manifest_sha256 = model_io::hash_file(&manifest_path, 1 << 20)
        .map_err(|error| format!("failed to hash image manifest: {error}"))?;
    if receipt.schema_version != 1 {
        return Err(format!(
            "image install receipt has unsupported schemaVersion {}",
            receipt.schema_version
        ));
    }
    if receipt.manifest_sha256.to_ascii_lowercase() != manifest_sha256 {
        return Err("image install receipt manifest SHA mismatch".to_string());
    }
    let expected_directory_path = match final_directory_path {
        Some(path) => path.to_string(),
        None => root
            .canonicalize()
            .unwrap_or_else(|_| root.to_path_buf())
            .display()
            .to_string(),
    };
    if receipt.model_directory_path != expected_directory_path {
        return Err(format!(
            "image install receipt directory mismatch: {}, expected {}",
            receipt.model_directory_path, expected_directory_path
        ));
    }

    let mut expected = BTreeMap::new();
    let manifest_size = fs::metadata(&manifest_path)
        .map_err(|error| format!("failed to stat image manifest: {error}"))?
        .len();
    expected.insert(
        IMAGE_MANIFEST_NAME.to_string(),
        (manifest_size, manifest_sha256),
    );
    for file in &manifest.files {
        expected.insert(
            file.path.clone(),
            (file.size_bytes, file.sha256.to_ascii_lowercase()),
        );
    }
    if receipt.files.len() != expected.len() {
        return Err(format!(
            "image install receipt file count {} does not match {}",
            receipt.files.len(),
            expected.len()
        ));
    }
    for (path, (size, sha256)) in expected {
        let entry = receipt
            .files
            .get(&path)
            .ok_or_else(|| format!("image install receipt is missing {path}"))?;
        if entry.size != size || entry.sha256.to_ascii_lowercase() != sha256 {
            return Err(format!(
                "image install receipt entry for {path} does not match"
            ));
        }
    }
    Ok(())
}

fn validate_relative_path(path: &str) -> Result<(), String> {
    let candidate = Path::new(path);
    if path.is_empty() || candidate.is_absolute() {
        return Err(format!(
            "image manifest path {path:?} is not a relative file path"
        ));
    }
    if candidate.components().any(|component| {
        matches!(
            component,
            Component::ParentDir | Component::RootDir | Component::Prefix(_)
        )
    }) {
        return Err(format!(
            "image manifest path {path:?} escapes the install root"
        ));
    }
    Ok(())
}

fn validate_sha256(value: &str, path: &str) -> Result<(), String> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(format!("image manifest file {path} has an invalid SHA-256"));
    }
    Ok(())
}

pub(crate) fn required_metadata_for(component_name: &str) -> &'static [&'static str] {
    match component_name {
        "tokenizer" => &[
            "asset_paths",
            "asset_sha256",
            "chat_template",
            "enable_thinking",
            "truncation_policy",
            "padding_policy",
            "max_tokens",
        ],
        "text_encoder" => &[
            "canonical_revision",
            "tensor_inventory",
            "hidden_state_output",
            "causal_mask",
            "qk_norm_epsilon",
            "quantized_tensor_names",
            "protected_tensor_names",
        ],
        "transformer" => &[
            "canonical_revision",
            "tensor_inventory",
            "patch_order",
            "learned_pad_token",
            "sequence_padding",
            "timestep_modulation",
            "three_axis_rope",
            "quantization_scheme",
            "quantization_label",
            "group_size",
            "nibble_order",
            "scale_and_bias_convention",
            "quantized_tensor_names",
            "protected_tensor_names",
        ],
        "scheduler" => &[
            "config_sha256",
            "num_train_timesteps",
            "shift",
            "sigma_endpoints",
            "evaluation_count",
            "guidance_policy",
            "noise_provenance",
        ],
        "vae_decoder" => &[
            "canonical_revision",
            "decoder_tensor_subset",
            "latent_layout",
            "normalization",
            "scale_and_shift",
            "output_range",
            "pixel_conversion",
            "quantized_tensor_names",
            "protected_tensor_names",
        ],
        _ => &[],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_the_frozen_ig2_envelope_and_stage_owners() {
        let mut components: BTreeMap<String, ImageComponent> = COMPONENT_ORDER
            .iter()
            .map(|name| {
                let metadata = BTreeMap::from([(
                    "required_metadata".to_string(),
                    serde_json::Value::Array(
                        required_metadata_for(name)
                            .iter()
                            .map(|value| serde_json::Value::String((*value).to_string()))
                            .collect(),
                    ),
                )]);
                let owner = match *name {
                    "tokenizer" | "text_encoder" => "text_encoder_stage",
                    "transformer" => "transformer_stage",
                    "scheduler" => "pipeline_control",
                    "vae_decoder" => "vae_stage",
                    _ => unreachable!(),
                };
                (
                    (*name).to_string(),
                    ImageComponent {
                        required: true,
                        owner: owner.to_string(),
                        metadata,
                    },
                )
            })
            .collect();
        components
            .get_mut("transformer")
            .expect("transformer component")
            .metadata
            .extend([
                (
                    "quantization_scheme".to_string(),
                    serde_json::json!(crate::runtime::IMAGE_QUANTIZATION),
                ),
                (
                    "observed_affine_bit_widths".to_string(),
                    serde_json::json!([]),
                ),
                (
                    "quantization_label".to_string(),
                    serde_json::json!(crate::runtime::IMAGE_QUANTIZATION),
                ),
            ]);
        let manifest = ImageManifest {
            magic: IMAGE_MANIFEST_MAGIC.to_string(),
            version: IMAGE_MANIFEST_SCHEMA,
            capability: IMAGE_CAPABILITY.to_string(),
            source: serde_json::json!({"model_revision": "test"}),
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
            files: vec![ImageManifestFile {
                path: "components/scheduler/config.json".to_string(),
                owner: "pipeline_control".to_string(),
                size_bytes: 1,
                sha256: "00".repeat(32),
                storage_dtype: "JSON".to_string(),
                quantization: serde_json::json!({"scheme": "none"}),
                tensor_inventory_sha256: "11".repeat(32),
            }],
            verification: serde_json::json!({}),
            resource_envelope: serde_json::json!({}),
        };

        let validation = manifest.validate().expect("valid image manifest");
        assert_eq!(validation.file_count, 1);
        assert_eq!(validation.dimensions, (IMAGE_WIDTH, IMAGE_HEIGHT));

        let mut forged = manifest.clone();
        forged
            .components
            .get_mut("transformer")
            .expect("transformer component")
            .metadata
            .insert(
                "quantization_label".to_string(),
                serde_json::json!("unquantized"),
            );
        let error = forged
            .validate()
            .expect_err("mismatched quantization label must be rejected");
        assert!(error.contains("quantization label"), "got {error}");
    }

    #[test]
    fn rejects_manifest_paths_that_escape_the_install_root() {
        let error = validate_relative_path("../outside.bin").expect_err("path must be rejected");
        assert!(error.contains("escapes"));
    }
}
