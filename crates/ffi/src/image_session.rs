//! Native image-generation session shared by the C ABI and Swift wrapper.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};

use crate::heavy::HeavyWorkGuard;

#[derive(Debug, Clone, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct ImageGenerateOptions {
    pub prompt: String,
    pub seed: u64,
    pub width: u32,
    pub height: u32,
    pub steps: u32,
}

impl Default for ImageGenerateOptions {
    fn default() -> Self {
        Self {
            prompt: String::new(),
            seed: 0,
            width: image::IMAGE_WIDTH,
            height: image::IMAGE_HEIGHT,
            steps: image::IMAGE_STEPS,
        }
    }
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ImageGenerationInfo {
    pub status: String,
    pub error: Option<String>,
    pub metadata: Option<image::ImageMetadata>,
}

#[derive(Debug)]
pub struct ImageGenerationOutput {
    pub png: Vec<u8>,
    pub info: ImageGenerationInfo,
}

#[derive(Debug)]
pub struct ImageSession {
    #[cfg(target_os = "macos")]
    backend: Mutex<image::MetalImageBackend>,
    cancellation: Mutex<Arc<image::CancellationToken>>,
    model_id: String,
    model_revision: String,
    component_revisions: BTreeMap<String, String>,
    quantization: String,
    #[cfg(target_os = "macos")]
    _model_path: PathBuf,
}

impl ImageSession {
    #[cfg(target_os = "macos")]
    pub fn open(model_arg: &str) -> Result<Self, String> {
        let model_path = catalog::resolve_image_arg(model_arg);
        let manifest = image::ImageManifest::load(&model_path)?;
        manifest.validate()?;
        manifest.verify_files(&model_path)?;
        let model_id = manifest
            .source
            .get("model_id")
            .and_then(serde_json::Value::as_str)
            .unwrap_or(model_arg)
            .to_string();
        let model_revision = manifest
            .source
            .get("model_revision")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("installed")
            .to_string();
        let component_revisions = manifest
            .components
            .iter()
            .filter_map(|(name, component)| {
                component
                    .metadata
                    .get("canonical_revision")
                    .and_then(serde_json::Value::as_str)
                    .map(|revision| (name.clone(), revision.to_string()))
            })
            .collect();
        let quantization = image::image_quantization_label(&manifest)?;
        let backend = image::MetalImageBackend::open(&model_path)?;
        Ok(Self {
            backend: Mutex::new(backend),
            cancellation: Mutex::new(Arc::new(image::CancellationToken::new())),
            model_id,
            model_revision,
            component_revisions,
            quantization,
            _model_path: model_path,
        })
    }

    #[cfg(not(target_os = "macos"))]
    pub fn open(_model_arg: &str) -> Result<Self, String> {
        Err("native image generation requires macOS Metal".to_string())
    }

    pub fn cancel(&self) {
        if let Ok(token) = self.cancellation.lock() {
            token.cancel();
        }
    }

    #[cfg(target_os = "macos")]
    pub fn generate<F: FnMut(image::ImageProgress)>(
        &self,
        options: ImageGenerateOptions,
        mut on_progress: F,
    ) -> Result<ImageGenerationOutput, String> {
        let cancellation = Arc::new(image::CancellationToken::new());
        {
            let mut current = self
                .cancellation
                .lock()
                .map_err(|_| "image session cancellation state is poisoned".to_string())?;
            *current = Arc::clone(&cancellation);
        }
        let _gate = HeavyWorkGuard::acquire();
        let mut backend = self
            .backend
            .lock()
            .map_err(|_| "image session backend is poisoned".to_string())?;
        let request = image::ImageRequest {
            model_id: self.model_id.clone(),
            model_revision: self.model_revision.clone(),
            component_revisions: self.component_revisions.clone(),
            prompt: options.prompt,
            width: options.width,
            height: options.height,
            batch: image::IMAGE_BATCH,
            scheduler_steps: options.steps,
            guidance_scale: image::IMAGE_GUIDANCE,
            seed: options.seed,
            quantization: self.quantization.clone(),
            noise_provenance: "zimage_metal_xorshift_box_muller_v1".to_string(),
        };
        match image::generate(&mut *backend, &request, &cancellation, |progress| {
            on_progress(progress)
        }) {
            Ok(result) => Ok(ImageGenerationOutput {
                png: result.png,
                info: ImageGenerationInfo {
                    status: "completed".to_string(),
                    error: None,
                    metadata: Some(result.metadata),
                },
            }),
            Err(error) if error == image::IMAGE_CANCELLED => Ok(ImageGenerationOutput {
                png: Vec::new(),
                info: ImageGenerationInfo {
                    status: "cancelled".to_string(),
                    error: Some(error),
                    metadata: None,
                },
            }),
            Err(error) => Err(error),
        }
    }

    #[cfg(not(target_os = "macos"))]
    pub fn generate<F: FnMut(image::ImageProgress)>(
        &self,
        _options: ImageGenerateOptions,
        _on_progress: F,
    ) -> Result<ImageGenerationOutput, String> {
        Err("native image generation requires macOS Metal".to_string())
    }
}
