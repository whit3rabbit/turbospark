//! Stage ownership and output contract for IG2 image generation.
//!
//! The backend owns device-specific work. This module owns the lifecycle that
//! every backend must obey: text conditioning, transformer denoising, VAE
//! decode, and only then publication of a complete PNG.

use std::collections::BTreeMap;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

use serde::{Deserialize, Serialize};

use crate::install::{
    IMAGE_BATCH, IMAGE_FORWARDS, IMAGE_GUIDANCE, IMAGE_HEIGHT, IMAGE_PROMPT_MAX_TOKENS,
    IMAGE_STEPS, IMAGE_WIDTH,
};
use crate::scheduler::FlowMatchEulerScheduler;
use crate::vae::decoded_to_rgb8;

pub const IMAGE_CANCELLED: &str = "image generation cancelled";
pub const IMAGE_ENGINE_REVISION: &str = "ig2-runtime-v1";
pub const IMAGE_QUANTIZATION: &str = "four-bit-linear-weights-group-64";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImageStage {
    TextEncoder,
    Transformer,
    VaeDecoder,
    PngEncode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImageProgress {
    pub stage: ImageStage,
    pub completed: u32,
    pub total: u32,
}

#[derive(Debug, Clone, Default)]
pub struct CancellationToken {
    cancelled: Arc<AtomicBool>,
}

impl CancellationToken {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }
}

#[derive(Debug, Clone)]
pub struct ImageRequest {
    pub model_id: String,
    pub model_revision: String,
    pub component_revisions: BTreeMap<String, String>,
    pub prompt: String,
    pub width: u32,
    pub height: u32,
    pub batch: u32,
    pub scheduler_steps: u32,
    pub guidance_scale: f32,
    pub seed: u64,
    pub quantization: String,
    pub noise_provenance: String,
}

impl ImageRequest {
    pub fn validate(&self) -> Result<(), String> {
        if self.model_id.is_empty() || self.model_revision.is_empty() {
            return Err("image request requires model_id and model_revision".to_string());
        }
        if self.prompt.trim().is_empty() {
            return Err("image request prompt cannot be empty".to_string());
        }
        if (self.width, self.height) != (IMAGE_WIDTH, IMAGE_HEIGHT) {
            return Err(format!(
                "image dimensions {}x{} are outside the IG2 envelope of {}x{}",
                self.width, self.height, IMAGE_WIDTH, IMAGE_HEIGHT
            ));
        }
        if self.batch != IMAGE_BATCH
            || self.scheduler_steps != IMAGE_STEPS
            || self.guidance_scale.to_bits() != IMAGE_GUIDANCE.to_bits()
        {
            return Err(format!(
                "unsupported image request envelope: batch={}, steps={}, guidance={}",
                self.batch, self.scheduler_steps, self.guidance_scale
            ));
        }
        if self.quantization != IMAGE_QUANTIZATION {
            return Err(format!(
                "unsupported image quantization {:?}, expected {:?}",
                self.quantization, IMAGE_QUANTIZATION
            ));
        }
        if self.noise_provenance.is_empty() {
            return Err("image request requires noise provenance".to_string());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ImageMetadata {
    pub prompt: String,
    pub seed: u64,
    pub width: u32,
    pub height: u32,
    pub batch: u32,
    pub scheduler_steps: u32,
    pub transformer_forwards: u32,
    pub guidance_scale: f32,
    pub model_id: String,
    pub model_revision: String,
    pub component_revisions: BTreeMap<String, String>,
    pub quantization: String,
    pub scheduler: SchedulerMetadata,
    pub noise_provenance: String,
    pub engine_revision: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SchedulerMetadata {
    pub num_train_timesteps: f32,
    pub shift: f32,
    pub timesteps: Vec<f32>,
    pub sigmas: Vec<f32>,
    pub evaluation_count: u32,
    pub guidance_policy: String,
}

#[derive(Debug, Clone)]
pub struct ImageResult {
    pub png: Vec<u8>,
    pub metadata: ImageMetadata,
}

pub trait ImageBackend {
    fn encode_conditioning(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        cancellation: &CancellationToken,
        progress: &mut dyn FnMut(u32, u32),
    ) -> Result<Vec<f32>, String>;

    fn denoise(
        &mut self,
        conditioning: &[f32],
        request: &ImageRequest,
        scheduler: &FlowMatchEulerScheduler,
        cancellation: &CancellationToken,
        progress: &mut dyn FnMut(u32, u32),
    ) -> Result<Vec<f32>, String>;

    fn decode(
        &mut self,
        latents: &[f32],
        width: u32,
        height: u32,
        cancellation: &CancellationToken,
        progress: &mut dyn FnMut(u32, u32),
    ) -> Result<Vec<f32>, String>;
}

/// Run one image through the staged lifecycle and return a complete PNG.
pub fn generate<F: FnMut(ImageProgress)>(
    backend: &mut dyn ImageBackend,
    request: &ImageRequest,
    cancellation: &CancellationToken,
    mut on_progress: F,
) -> Result<ImageResult, String> {
    request.validate()?;
    check_cancelled(cancellation)?;

    let mut scheduler = FlowMatchEulerScheduler::default();
    scheduler.set_timesteps(request.scheduler_steps as usize);
    if scheduler.timesteps.len() != IMAGE_FORWARDS as usize {
        return Err("scheduler evaluation count does not match the IG2 contract".to_string());
    }

    let mut stage_progress = |stage: ImageStage, completed: u32, total: u32| {
        on_progress(ImageProgress {
            stage,
            completed,
            total,
        });
    };

    stage_progress(ImageStage::TextEncoder, 0, 1);
    let conditioning = backend.encode_conditioning(
        &request.prompt,
        IMAGE_PROMPT_MAX_TOKENS as usize,
        cancellation,
        &mut |completed, total| stage_progress(ImageStage::TextEncoder, completed, total),
    )?;
    check_cancelled(cancellation)?;
    stage_progress(ImageStage::TextEncoder, 1, 1);

    stage_progress(ImageStage::Transformer, 0, request.scheduler_steps);
    let latents = backend.denoise(
        &conditioning,
        request,
        &scheduler,
        cancellation,
        &mut |completed, total| stage_progress(ImageStage::Transformer, completed, total),
    )?;
    check_cancelled(cancellation)?;
    stage_progress(
        ImageStage::Transformer,
        request.scheduler_steps,
        request.scheduler_steps,
    );

    stage_progress(ImageStage::VaeDecoder, 0, 1);
    let decoded = backend.decode(
        &latents,
        request.width,
        request.height,
        cancellation,
        &mut |completed, total| stage_progress(ImageStage::VaeDecoder, completed, total),
    )?;
    check_cancelled(cancellation)?;
    let rgb = decoded_to_rgb8(&decoded, request.height as usize, request.width as usize)?;
    stage_progress(ImageStage::VaeDecoder, 1, 1);

    let metadata = ImageMetadata {
        prompt: request.prompt.clone(),
        seed: request.seed,
        width: request.width,
        height: request.height,
        batch: request.batch,
        scheduler_steps: request.scheduler_steps,
        transformer_forwards: IMAGE_FORWARDS,
        guidance_scale: request.guidance_scale,
        model_id: request.model_id.clone(),
        model_revision: request.model_revision.clone(),
        component_revisions: request.component_revisions.clone(),
        quantization: request.quantization.clone(),
        scheduler: SchedulerMetadata {
            num_train_timesteps: scheduler.num_train_timesteps,
            shift: scheduler.shift,
            timesteps: scheduler.timesteps.clone(),
            sigmas: scheduler.sigmas.clone(),
            evaluation_count: request.scheduler_steps,
            guidance_policy: "zero".to_string(),
        },
        noise_provenance: request.noise_provenance.clone(),
        engine_revision: IMAGE_ENGINE_REVISION.to_string(),
    };
    let metadata_json = serde_json::to_string(&metadata)
        .map_err(|e| format!("failed to serialize image metadata: {e}"))?;
    check_cancelled(cancellation)?;
    stage_progress(ImageStage::PngEncode, 0, 1);
    let png = encode_png_with_metadata(&rgb, request.width, request.height, &metadata_json)?;
    check_cancelled(cancellation)?;
    stage_progress(ImageStage::PngEncode, 1, 1);

    Ok(ImageResult { png, metadata })
}

fn encode_png_with_metadata(
    rgb: &[u8],
    width: u32,
    height: u32,
    metadata_json: &str,
) -> Result<Vec<u8>, String> {
    let mut output = Vec::new();
    let mut encoder = png::Encoder::new(&mut output, width, height);
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);
    encoder
        .add_itxt_chunk("turbospark".to_string(), metadata_json.to_string())
        .map_err(|e| format!("failed to add PNG metadata: {e}"))?;
    let mut writer = encoder
        .write_header()
        .map_err(|e| format!("failed to write PNG header: {e}"))?;
    writer
        .write_image_data(rgb)
        .map_err(|e| format!("failed to encode PNG: {e}"))?;
    drop(writer);
    Ok(output)
}

fn check_cancelled(cancellation: &CancellationToken) -> Result<(), String> {
    if cancellation.is_cancelled() {
        Err(IMAGE_CANCELLED.to_string())
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request() -> ImageRequest {
        ImageRequest {
            model_id: "z-image-turbo".to_string(),
            model_revision: "model-rev".to_string(),
            component_revisions: BTreeMap::from([("transformer".to_string(), "rev".to_string())]),
            prompt: "a lighthouse in winter".to_string(),
            width: IMAGE_WIDTH,
            height: IMAGE_HEIGHT,
            batch: IMAGE_BATCH,
            scheduler_steps: IMAGE_STEPS,
            guidance_scale: IMAGE_GUIDANCE,
            seed: 42,
            quantization: IMAGE_QUANTIZATION.to_string(),
            noise_provenance: "test-xorshift".to_string(),
        }
    }

    struct TestBackend {
        cancel_at: Option<ImageStage>,
    }

    impl ImageBackend for TestBackend {
        fn encode_conditioning(
            &mut self,
            _prompt: &str,
            _max_tokens: usize,
            cancellation: &CancellationToken,
            progress: &mut dyn FnMut(u32, u32),
        ) -> Result<Vec<f32>, String> {
            progress(1, 1);
            if self.cancel_at == Some(ImageStage::TextEncoder) {
                cancellation.cancel();
            }
            Ok(vec![0.0; 2560])
        }

        fn denoise(
            &mut self,
            _conditioning: &[f32],
            _request: &ImageRequest,
            _scheduler: &FlowMatchEulerScheduler,
            cancellation: &CancellationToken,
            progress: &mut dyn FnMut(u32, u32),
        ) -> Result<Vec<f32>, String> {
            progress(9, 9);
            if self.cancel_at == Some(ImageStage::Transformer) {
                cancellation.cancel();
            }
            Ok(vec![0.0; 16 * 128 * 128])
        }

        fn decode(
            &mut self,
            _latents: &[f32],
            _width: u32,
            _height: u32,
            cancellation: &CancellationToken,
            progress: &mut dyn FnMut(u32, u32),
        ) -> Result<Vec<f32>, String> {
            progress(1, 1);
            if self.cancel_at == Some(ImageStage::VaeDecoder) {
                cancellation.cancel();
            }
            Ok(vec![0.0; 3 * 1024 * 1024])
        }
    }

    #[test]
    fn generation_embeds_contract_metadata_and_stage_progress() {
        let mut backend = TestBackend { cancel_at: None };
        let cancellation = CancellationToken::new();
        let mut progress = Vec::new();
        let result = generate(&mut backend, &request(), &cancellation, |event| {
            progress.push(event)
        })
        .expect("test generation");

        assert_eq!(result.metadata.seed, 42);
        assert_eq!(result.metadata.scheduler.evaluation_count, IMAGE_STEPS);
        assert!(result.png.starts_with(b"\x89PNG\r\n\x1a\n"));
        assert!(result
            .png
            .windows(b"turbospark".len())
            .any(|window| window == b"turbospark"));
        assert!(progress
            .iter()
            .any(|event| event.stage == ImageStage::TextEncoder));
        assert!(progress
            .iter()
            .any(|event| event.stage == ImageStage::Transformer));
        assert!(progress
            .iter()
            .any(|event| event.stage == ImageStage::VaeDecoder));
        assert!(progress
            .iter()
            .any(|event| event.stage == ImageStage::PngEncode));
    }

    #[test]
    fn cancellation_stops_after_each_owned_stage() {
        for stage in [
            ImageStage::TextEncoder,
            ImageStage::Transformer,
            ImageStage::VaeDecoder,
            ImageStage::PngEncode,
        ] {
            let mut backend = TestBackend {
                cancel_at: Some(stage),
            };
            let cancellation = CancellationToken::new();
            let cancel_for_png = cancellation.clone();
            let result = generate(&mut backend, &request(), &cancellation, |event| {
                if stage == ImageStage::PngEncode && event.stage == ImageStage::PngEncode {
                    cancel_for_png.cancel();
                }
            });
            assert!(
                matches!(result, Err(error) if error == IMAGE_CANCELLED),
                "cancellation must stop the {:?} stage",
                stage
            );
        }
    }

    #[test]
    fn cancellation_stops_before_transformer_stage() {
        let mut backend = TestBackend {
            cancel_at: Some(ImageStage::TextEncoder),
        };
        let cancellation = CancellationToken::new();
        let result = generate(&mut backend, &request(), &cancellation, |_| {});
        assert!(matches!(result, Err(error) if error == IMAGE_CANCELLED));
    }
}
