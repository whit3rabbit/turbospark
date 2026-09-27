//! CPU reference backend for the IG2 packed component path.
//!
//! This backend proves install layout, staging, cancellation, and metadata
//! without claiming the final Metal performance or memory envelope. Each
//! heavyweight component is opened only for the stage that consumes it.

use std::path::{Path, PathBuf};

use tokenizer::MfTokenizer;

use crate::conditioning::{frame_prompt, load_tokenizer, tokenize_prompt};
use crate::install::{ImageManifest, IMAGE_HEIGHT, IMAGE_WIDTH};
use crate::pipeline::ZImageTransformer;
use crate::runtime::{CancellationToken, ImageBackend, ImageRequest, IMAGE_CANCELLED};
use crate::scheduler::FlowMatchEulerScheduler;
use crate::text_encoder::{encode_tokens_with_cancel, ShardedSafetensors};
use crate::vae::VaeDecoder;

pub const IMAGE_COMPONENTS_DIR: &str = "components";

pub struct CpuReferenceBackend {
    root: PathBuf,
    tokenizer: Option<MfTokenizer>,
}

impl CpuReferenceBackend {
    /// Open and verify a complete packed image install.
    pub fn open(root: &Path) -> Result<Self, String> {
        let manifest = ImageManifest::load(root)?;
        manifest.verify_files(root)?;
        for component in [
            "tokenizer",
            "text_encoder",
            "transformer",
            "scheduler",
            "vae_decoder",
        ] {
            let path = component_path(root, component);
            if !path.is_dir() {
                return Err(format!(
                    "image component directory is missing: {}",
                    path.display()
                ));
            }
        }
        Ok(Self {
            root: root.to_path_buf(),
            tokenizer: None,
        })
    }

    fn tokenizer(&mut self) -> Result<&MfTokenizer, String> {
        if self.tokenizer.is_none() {
            let loaded = load_tokenizer(&component_path(&self.root, "tokenizer"))
                .map_err(|e| format!("failed to load image tokenizer: {e}"))?;
            self.tokenizer = Some(loaded);
        }
        self.tokenizer
            .as_ref()
            .ok_or_else(|| "image tokenizer was not loaded".to_string())
    }
}

impl ImageBackend for CpuReferenceBackend {
    fn encode_conditioning(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        cancellation: &CancellationToken,
        progress: &mut dyn FnMut(u32, u32),
    ) -> Result<Vec<f32>, String> {
        if cancellation.is_cancelled() {
            return Err(IMAGE_CANCELLED.to_string());
        }
        let tokenizer = self.tokenizer()?;
        let framed = frame_prompt(prompt, tokenizer)
            .map_err(|e| format!("failed to frame image prompt: {e}"))?;
        let (token_ids, attention_mask) = tokenize_prompt(&framed, tokenizer, max_tokens);
        let retained = attention_mask.iter().filter(|&&value| value != 0).count();
        if retained == 0 {
            return Err("image prompt produced no tokens".to_string());
        }

        let shards = ShardedSafetensors::open_packed(&component_path(&self.root, "text_encoder"))?;
        encode_tokens_with_cancel(&token_ids[..retained], &shards, |completed, total| {
            if cancellation.is_cancelled() {
                false
            } else {
                progress(completed as u32, total as u32);
                true
            }
        })
        .map_err(|error| {
            if cancellation.is_cancelled() {
                IMAGE_CANCELLED.to_string()
            } else {
                error
            }
        })
    }

    fn denoise(
        &mut self,
        conditioning: &[f32],
        request: &ImageRequest,
        scheduler: &FlowMatchEulerScheduler,
        cancellation: &CancellationToken,
        progress: &mut dyn FnMut(u32, u32),
    ) -> Result<Vec<f32>, String> {
        if (request.width, request.height) != (IMAGE_WIDTH, IMAGE_HEIGHT) {
            return Err(format!(
                "CPU image reference backend only supports {IMAGE_WIDTH}x{IMAGE_HEIGHT}"
            ));
        }
        let transformer =
            ZImageTransformer::open_packed(&component_path(&self.root, "transformer"))?;
        let mut latent = seeded_noise(request.seed, 16 * 128 * 128);
        for step in 0..request.scheduler_steps as usize {
            if cancellation.is_cancelled() {
                return Err(IMAGE_CANCELLED.to_string());
            }
            let velocity = transformer
                .forward_with_progress_cancel(
                    &latent,
                    128,
                    128,
                    scheduler.normalized_time(step),
                    conditioning,
                    |_| !cancellation.is_cancelled(),
                )
                .map_err(|error| {
                    if cancellation.is_cancelled() {
                        IMAGE_CANCELLED.to_string()
                    } else {
                        error
                    }
                })?;
            latent = scheduler.step(&velocity, step, &latent);
            progress((step + 1) as u32, request.scheduler_steps);
        }
        Ok(latent)
    }

    fn decode(
        &mut self,
        latents: &[f32],
        width: u32,
        height: u32,
        cancellation: &CancellationToken,
        progress: &mut dyn FnMut(u32, u32),
    ) -> Result<Vec<f32>, String> {
        if cancellation.is_cancelled() {
            return Err(IMAGE_CANCELLED.to_string());
        }
        if (width, height) != (1024, 1024) {
            return Err("CPU image reference backend only supports 1024x1024".to_string());
        }
        let weights = ShardedSafetensors::open_packed(&component_path(&self.root, "vae_decoder"))?;
        let decoder = VaeDecoder::from_tensor_loader(&weights)?;
        let decoded = decoder.decode(latents, 128, 128)?;
        if cancellation.is_cancelled() {
            return Err(IMAGE_CANCELLED.to_string());
        }
        progress(1, 1);
        Ok(decoded)
    }
}

fn component_path(root: &Path, component: &str) -> PathBuf {
    root.join(IMAGE_COMPONENTS_DIR).join(component)
}

fn seeded_noise(seed: u64, count: usize) -> Vec<f32> {
    let mut state = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut output = Vec::with_capacity(count);
    while output.len() < count {
        let u1 = uniform(&mut state).max(f64::MIN_POSITIVE);
        let u2 = uniform(&mut state);
        let radius = (-2.0 * u1.ln()).sqrt();
        let angle = std::f64::consts::TAU * u2;
        output.push((radius * angle.cos()) as f32);
        if output.len() < count {
            output.push((radius * angle.sin()) as f32);
        }
    }
    output
}

fn uniform(state: &mut u64) -> f64 {
    *state ^= *state << 7;
    *state ^= *state >> 9;
    *state ^= *state << 8;
    (*state as f64) / (u64::MAX as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reference_backend_refuses_alternate_dimensions_before_loading_weights() {
        let mut backend = CpuReferenceBackend {
            root: PathBuf::from("/missing-image-install"),
            tokenizer: None,
        };
        let request = ImageRequest {
            model_id: "test".to_string(),
            model_revision: "test".to_string(),
            component_revisions: Default::default(),
            prompt: "a lighthouse".to_string(),
            width: 768,
            height: 1024,
            batch: 1,
            scheduler_steps: 9,
            guidance_scale: 0.0,
            seed: 42,
            quantization: "test".to_string(),
            noise_provenance: "test".to_string(),
        };
        let error = backend
            .denoise(
                &[],
                &request,
                &FlowMatchEulerScheduler::default(),
                &CancellationToken::new(),
                &mut |_, _| {},
            )
            .expect_err("CPU reference geometry is fixed at 1024x1024");

        assert!(error.contains("only supports 1024x1024"), "{error}");
    }
}
