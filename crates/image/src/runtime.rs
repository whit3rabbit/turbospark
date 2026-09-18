//! Stage ownership and output contract for IG2 image generation.
//!
//! The backend owns device-specific work. This module owns the lifecycle that
//! every backend must obey: text conditioning, transformer denoising, VAE
//! decode, and only then publication of a complete PNG.

use std::collections::BTreeMap;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Condvar, Mutex,
};

use serde::{Deserialize, Serialize};

use crate::install::{
    ImageManifest, IMAGE_BATCH, IMAGE_FORWARDS, IMAGE_GUIDANCE, IMAGE_HEIGHT,
    IMAGE_PROMPT_MAX_TOKENS, IMAGE_STEPS, IMAGE_WIDTH,
};
use crate::packed::MLX_AFFINE_BITS;
use crate::scheduler::FlowMatchEulerScheduler;
use crate::vae::decoded_to_rgb8;

pub const IMAGE_CANCELLED: &str = "image generation cancelled";
pub const IMAGE_ENGINE_REVISION: &str = "ig2-runtime-v1";
pub const IMAGE_QUANTIZATION: &str = "four-bit-linear-weights-group-64";
pub const IMAGE_MLX_QUANTIZATION: &str = "mlx-affine-linear-weights-group-64";
pub const IMAGE_UNQUANTIZED: &str = "unquantized";

/// Return the quantization label carried by an installed image manifest.
///
/// The old INT4 label remains the default for the original packed profile, but
/// MLX installs need the observed width in their request and PNG metadata so
/// a 2-, 3-, 4-, 5-, 6-, or 8-bit run is not reported as the legacy INT4
/// profile.
pub fn image_quantization_label(manifest: &ImageManifest) -> Result<String, String> {
    let transformer = manifest
        .components
        .get("transformer")
        .ok_or_else(|| "image manifest is missing transformer metadata".to_string())?;
    let scheme = transformer
        .metadata
        .get("quantization_scheme")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| "image transformer metadata is missing quantization_scheme".to_string())?;
    match scheme {
        IMAGE_QUANTIZATION | IMAGE_UNQUANTIZED => Ok(scheme.to_string()),
        IMAGE_MLX_QUANTIZATION => {
            let widths = transformer
                .metadata
                .get("observed_affine_bit_widths")
                .and_then(serde_json::Value::as_array)
                .ok_or_else(|| {
                    "MLX transformer metadata is missing observed_affine_bit_widths".to_string()
                })?;
            let mut widths = widths
                .iter()
                .map(|value| {
                    let width = value.as_u64().ok_or_else(|| {
                        "MLX observed affine bit width is not an integer".to_string()
                    })?;
                    u8::try_from(width)
                        .map_err(|_| "MLX observed affine bit width overflows u8".to_string())
                })
                .collect::<Result<Vec<_>, _>>()?;
            widths.sort_unstable();
            widths.dedup();
            if widths.is_empty() || widths.iter().any(|width| !MLX_AFFINE_BITS.contains(width)) {
                return Err(format!(
                    "MLX observed affine bit widths {:?} are unsupported",
                    widths
                ));
            }
            let suffix = widths
                .iter()
                .map(u8::to_string)
                .collect::<Vec<_>>()
                .join("-");
            Ok(format!("{IMAGE_MLX_QUANTIZATION}-bits-{suffix}"))
        }
        other => Err(format!("unsupported image quantization scheme {other:?}")),
    }
}

fn is_supported_quantization_label(label: &str) -> bool {
    if matches!(label, IMAGE_QUANTIZATION | IMAGE_UNQUANTIZED) {
        return true;
    }
    let Some(suffix) = label.strip_prefix("mlx-affine-linear-weights-group-64-bits-") else {
        return false;
    };
    let mut widths = suffix
        .split('-')
        .map(|part| part.parse::<u8>())
        .collect::<Result<Vec<_>, _>>()
        .unwrap_or_default();
    if widths.is_empty() {
        return false;
    }
    widths.sort_unstable();
    if widths.windows(2).any(|pair| pair[0] == pair[1]) {
        return false;
    }
    widths.iter().all(|width| MLX_AFFINE_BITS.contains(width))
}

/// The categories used by the image admission and resource ledgers.
///
/// These are deliberately separate from `phys_footprint`: the latter is a
/// process observation and includes allocator and driver state that this
/// crate cannot attribute to a tensor or a kernel workspace.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ImageMemoryPlan {
    /// Total packed payload bytes when all component weights are resident.
    pub component_weights: u64,
    /// Capacity of the largest component's two synchronous stream slots.
    pub streamed_component_weights: u64,
    pub conditioning: u64,
    pub latents: u64,
    pub activations: u64,
    pub scratch: u64,
    pub staging: u64,
    pub in_flight_gpu: u64,
    pub allocator_retention: u64,
    pub largest_block: u64,
    pub block_workspace: u64,
}

impl ImageMemoryPlan {
    pub fn managed_bytes(self) -> u64 {
        self.component_weights
            .saturating_add(self.conditioning)
            .saturating_add(self.latents)
            .saturating_add(self.activations)
            .saturating_add(self.scratch)
            .saturating_add(self.staging)
            .saturating_add(self.in_flight_gpu)
            .saturating_add(self.allocator_retention)
    }

    /// The lower bound for the streamed execution represented by this plan.
    /// It is intentionally not a whole-machine RAM claim.
    pub fn streamed_lower_bound(self) -> u64 {
        self.streamed_component_weights
            .saturating_add(self.conditioning)
            .saturating_add(self.latents)
            .saturating_add(self.activations)
            .saturating_add(self.scratch)
            .saturating_add(self.staging)
            .saturating_add(self.in_flight_gpu)
            .saturating_add(self.allocator_retention)
    }

    pub fn resident_lower_bound(self) -> u64 {
        self.managed_bytes()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageMemoryBudget {
    pub max_bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageBudgetError {
    InvalidBudget,
    LargestBlockDoesNotFit { required: u64, budget: u64 },
    ManagedPlanDoesNotFit { required: u64, budget: u64 },
}

impl std::fmt::Display for ImageBudgetError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidBudget => write!(f, "image memory budget must be nonzero"),
            Self::LargestBlockDoesNotFit { required, budget } => write!(
                f,
                "image memory budget refused before execution: largest block and workspace require {required} bytes, budget is {budget}"
            ),
            Self::ManagedPlanDoesNotFit { required, budget } => write!(
                f,
                "image memory budget refused before execution: managed plan requires {required} bytes, budget is {budget}"
            ),
        }
    }
}

pub fn admit_image_budget(
    plan: ImageMemoryPlan,
    budget: ImageMemoryBudget,
    streamed: bool,
) -> Result<(), ImageBudgetError> {
    if budget.max_bytes == 0 {
        return Err(ImageBudgetError::InvalidBudget);
    }
    let block_required = plan.largest_block.saturating_add(plan.block_workspace);
    if block_required > budget.max_bytes {
        return Err(ImageBudgetError::LargestBlockDoesNotFit {
            required: block_required,
            budget: budget.max_bytes,
        });
    }
    let required = if streamed {
        plan.streamed_lower_bound()
    } else {
        plan.resident_lower_bound()
    };
    if required > budget.max_bytes {
        return Err(ImageBudgetError::ManagedPlanDoesNotFit {
            required,
            budget: budget.max_bytes,
        });
    }
    Ok(())
}

/// Tracks consumers of a mapped component or staging slot.
///
/// A lease is not released until its owner has finished consuming the work.
/// Backend code can therefore wait for idle during cancellation and teardown
/// without guessing whether a dropped output still aliases a component.
#[derive(Debug, Clone, Default)]
pub struct ImageWorkTracker {
    state: Arc<(Mutex<usize>, Condvar)>,
}

impl ImageWorkTracker {
    pub fn begin(&self) -> ImageWorkLease {
        let (lock, _) = &*self.state;
        *lock.lock().expect("image work tracker mutex poisoned") += 1;
        ImageWorkLease {
            tracker: self.clone(),
        }
    }

    pub fn wait_for_idle(&self) {
        let (lock, ready) = &*self.state;
        let mut outstanding = lock.lock().expect("image work tracker mutex poisoned");
        while *outstanding != 0 {
            outstanding = ready
                .wait(outstanding)
                .expect("image work tracker mutex poisoned");
        }
    }

    fn finish(&self) {
        let (lock, ready) = &*self.state;
        let mut outstanding = lock.lock().expect("image work tracker mutex poisoned");
        *outstanding = outstanding
            .checked_sub(1)
            .expect("image work tracker lease underflow");
        if *outstanding == 0 {
            ready.notify_all();
        }
    }
}

#[derive(Debug)]
pub struct ImageWorkLease {
    tracker: ImageWorkTracker,
}

impl Drop for ImageWorkLease {
    fn drop(&mut self) {
        self.tracker.finish();
    }
}

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

    /// Request cancellation and wait until all registered consumers release
    /// their leases. This is the teardown seam used by GPU and I/O owners.
    pub fn cancel_and_wait(&self, work: &ImageWorkTracker) {
        self.cancel();
        work.wait_for_idle();
    }
}

/// Accounting-only state machine for dense sequential two-slot streaming.
/// A slot cannot be acquired twice until its consumer calls `release`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SequentialImageSlots {
    in_use: [bool; 2],
    live_bytes: u64,
    peak_live_bytes: u64,
}

impl SequentialImageSlots {
    pub fn acquire(&mut self, bytes: u64) -> Result<usize, &'static str> {
        let slot = self
            .in_use
            .iter()
            .position(|in_use| !in_use)
            .ok_or("both image stream slots are still in use")?;
        self.in_use[slot] = true;
        self.live_bytes = self.live_bytes.saturating_add(bytes);
        self.peak_live_bytes = self.peak_live_bytes.max(self.live_bytes);
        Ok(slot)
    }

    pub fn release(&mut self, slot: usize, bytes: u64) -> Result<(), &'static str> {
        let in_use = self
            .in_use
            .get_mut(slot)
            .ok_or("image stream slot index is out of range")?;
        if !*in_use {
            return Err("image stream slot was released before acquisition");
        }
        *in_use = false;
        self.live_bytes = self
            .live_bytes
            .checked_sub(bytes)
            .ok_or("image stream live-byte accounting underflow")?;
        Ok(())
    }

    pub fn live_bytes(self) -> u64 {
        self.live_bytes
    }

    pub fn peak_live_bytes(self) -> u64 {
        self.peak_live_bytes
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
        if !is_supported_quantization_label(&self.quantization) {
            return Err(format!(
                "unsupported image quantization {:?}",
                self.quantization
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
    #[serde(rename = "modelID")]
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

    /// Wait for consumers that outlive an individual backend stage.
    ///
    /// GPU-backed implementations must fence their submitted work before
    /// returning from the stage itself. This hook closes the cancellation
    /// seam for I/O or staging consumers registered across that stage.
    fn wait_for_idle(&mut self) {}
}

/// Run one image through the staged lifecycle and return a complete PNG.
pub fn generate<F: FnMut(ImageProgress)>(
    backend: &mut dyn ImageBackend,
    request: &ImageRequest,
    cancellation: &CancellationToken,
    on_progress: F,
) -> Result<ImageResult, String> {
    let result = generate_inner(backend, request, cancellation, on_progress);
    if cancellation.is_cancelled() {
        backend.wait_for_idle();
    }
    result
}

fn generate_inner<F: FnMut(ImageProgress)>(
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

/// Admit an image plan before entering the generation lifecycle.
///
/// Callers that have a manifest-backed plan should use this seam instead of
/// checking a budget after opening a component or scheduling GPU work.
pub fn generate_with_memory_budget<F: FnMut(ImageProgress)>(
    backend: &mut dyn ImageBackend,
    request: &ImageRequest,
    cancellation: &CancellationToken,
    plan: ImageMemoryPlan,
    budget: ImageMemoryBudget,
    streamed: bool,
    on_progress: F,
) -> Result<ImageResult, String> {
    admit_image_budget(plan, budget, streamed).map_err(|error| error.to_string())?;
    generate(backend, request, cancellation, on_progress)
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
    use std::sync::atomic::AtomicUsize;

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

    #[test]
    fn image_requests_accept_every_mlx_width_label() {
        for bits in MLX_AFFINE_BITS {
            let mut request = request();
            request.quantization = format!("mlx-affine-linear-weights-group-64-bits-{bits}");
            request.validate().expect("MLX image request label");
        }
        let mut mixed = request();
        mixed.quantization = "mlx-affine-linear-weights-group-64-bits-2-3-4-5-6-8".to_string();
        mixed.validate().expect("mixed MLX image request label");
    }

    #[test]
    fn image_requests_reject_non_mlx_or_unsupported_width_labels() {
        for label in [
            "mlx-affine-linear-weights-group-64-bits-1",
            "mlx-affine-linear-weights-group-64-bits-7",
            "mlx-affine-linear-weights-group-64-bits-2-2",
            "not-a-quantization",
        ] {
            let mut request = request();
            request.quantization = label.to_string();
            assert!(
                request.validate().is_err(),
                "unsupported image quantization label {label:?} was accepted"
            );
        }
    }

    struct TestBackend {
        cancel_at: Option<ImageStage>,
        idle_waits: Option<Arc<AtomicUsize>>,
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

        fn wait_for_idle(&mut self) {
            if let Some(idle_waits) = &self.idle_waits {
                idle_waits.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    #[test]
    fn generation_embeds_contract_metadata_and_stage_progress() {
        let mut backend = TestBackend {
            cancel_at: None,
            idle_waits: None,
        };
        let cancellation = CancellationToken::new();
        let mut progress = Vec::new();
        let result = generate(&mut backend, &request(), &cancellation, |event| {
            progress.push(event)
        })
        .expect("test generation");

        assert_eq!(result.metadata.seed, 42);
        assert_eq!(result.metadata.scheduler.evaluation_count, IMAGE_STEPS);
        let metadata = serde_json::to_value(&result.metadata).expect("metadata JSON");
        assert_eq!(metadata["modelID"], "z-image-turbo");
        assert!(metadata.get("modelId").is_none());
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
    fn generation_preserves_the_selected_mlx_width_in_metadata() {
        let mut backend = TestBackend {
            cancel_at: None,
            idle_waits: None,
        };
        let mut request = request();
        request.quantization = "mlx-affine-linear-weights-group-64-bits-6".to_string();
        let result = generate(&mut backend, &request, &CancellationToken::new(), |_| {})
            .expect("MLX metadata generation");

        assert_eq!(
            result.metadata.quantization,
            "mlx-affine-linear-weights-group-64-bits-6"
        );
        let metadata = serde_json::to_value(&result.metadata).expect("metadata JSON");
        assert_eq!(
            metadata["quantization"],
            "mlx-affine-linear-weights-group-64-bits-6"
        );
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
                idle_waits: None,
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
            idle_waits: None,
        };
        let cancellation = CancellationToken::new();
        let result = generate(&mut backend, &request(), &cancellation, |_| {});
        assert!(matches!(result, Err(error) if error == IMAGE_CANCELLED));
    }

    #[test]
    fn budget_refusal_happens_before_execution() {
        let plan = ImageMemoryPlan {
            component_weights: 80,
            streamed_component_weights: 30,
            conditioning: 10,
            latents: 10,
            activations: 20,
            scratch: 5,
            staging: 5,
            in_flight_gpu: 5,
            allocator_retention: 5,
            largest_block: 60,
            block_workspace: 15,
        };
        assert_eq!(plan.managed_bytes(), 140);
        assert_eq!(plan.streamed_lower_bound(), 90);
        assert!(admit_image_budget(plan, ImageMemoryBudget { max_bytes: 74 }, true).is_err());
        assert!(admit_image_budget(plan, ImageMemoryBudget { max_bytes: 139 }, false).is_err());
        assert!(admit_image_budget(plan, ImageMemoryBudget { max_bytes: 89 }, true).is_err());
        assert!(admit_image_budget(plan, ImageMemoryBudget { max_bytes: 90 }, true).is_ok());

        let mut backend = TestBackend {
            cancel_at: None,
            idle_waits: None,
        };
        let error = generate_with_memory_budget(
            &mut backend,
            &request(),
            &CancellationToken::new(),
            plan,
            ImageMemoryBudget { max_bytes: 74 },
            true,
            |_| {},
        )
        .expect_err("generation must be refused before entering the backend");
        assert!(error.contains("refused before execution"));
    }

    #[test]
    fn two_slots_reject_early_reuse_and_bound_live_storage() {
        let mut slots = SequentialImageSlots::default();
        let first = slots.acquire(10).expect("first slot");
        let second = slots.acquire(20).expect("second slot");
        assert_eq!(
            slots.acquire(1),
            Err("both image stream slots are still in use")
        );
        assert_eq!(slots.peak_live_bytes(), 30);
        slots.release(first, 10).expect("release first slot");
        let reused = slots.acquire(7).expect("reuse only after release");
        assert_eq!(reused, first);
        slots.release(reused, 7).expect("release reused slot");
        slots.release(second, 20).expect("release second slot");
        assert_eq!(slots.live_bytes(), 0);
    }

    #[test]
    fn cancellation_waits_for_registered_consumers() {
        let work = ImageWorkTracker::default();
        let lease = work.begin();
        let worker = std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(10));
            drop(lease);
        });
        let cancellation = CancellationToken::new();
        cancellation.cancel_and_wait(&work);
        worker.join().expect("consumer thread");
        assert!(cancellation.is_cancelled());
    }

    #[test]
    fn generation_waits_for_backend_consumers_after_cancellation() {
        let idle_waits = Arc::new(AtomicUsize::new(0));
        let mut backend = TestBackend {
            cancel_at: Some(ImageStage::Transformer),
            idle_waits: Some(Arc::clone(&idle_waits)),
        };
        let result = generate(&mut backend, &request(), &CancellationToken::new(), |_| {});
        assert!(matches!(result, Err(error) if error == IMAGE_CANCELLED));
        assert_eq!(idle_waits.load(Ordering::Relaxed), 1);
    }
}
