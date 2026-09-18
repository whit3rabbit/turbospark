//! Native Z-Image-Turbo pipeline components: scheduler, conditioning,
//! and text encoder forward pass.
//!
//! Provides portable, pure-Rust references and execution routines for
//! diffusion conditioning and scheduling.

#![forbid(unsafe_code)]

pub mod builder;
pub mod conditioning;
pub mod fixtures;
pub mod install;
pub mod packed;
pub mod patchify;
pub mod pipeline;
pub mod reference;
pub mod rope;
pub mod runtime;
pub mod scheduler;
pub mod text_encoder;
pub mod transformer;
pub mod vae;

#[cfg(target_os = "macos")]
pub mod metal;
#[cfg(target_os = "macos")]
mod metal_ops;

pub use builder::{build_image_install, ImageInstallReport, ImageInstallSpec};
pub use conditioning::{frame_prompt, tokenize_prompt, MAX_SEQUENCE_LENGTH, PAD_TOKEN_ID};
pub use fixtures::{
    read_npy_bool, read_npy_complex64, read_npy_f32, read_npy_file_bool, read_npy_file_complex64,
    read_npy_file_f32, read_npy_file_i64, read_npy_i64, read_npz, read_npz_file, NpyArray,
};
pub use install::{
    ImageComponent, ImageManifest, ImageManifestFile, ImageManifestValidation, ImageSupported,
    IMAGE_BATCH, IMAGE_FORWARDS, IMAGE_GUIDANCE, IMAGE_HEIGHT, IMAGE_PROMPT_MAX_TOKENS,
    IMAGE_RECEIPT_NAME, IMAGE_STEPS, IMAGE_WIDTH,
};
#[cfg(target_os = "macos")]
pub use metal::{
    ImageFirstStepBoundary, ImageFirstStepTrace, ImageIntraBlockTrace, ImageResidency,
    ImageStreamMetrics, MetalImageBackend,
};
pub use packed::{
    pack_component, PackedQuantization, PackedTensor, PackedTensorReport, PackedTensorStore,
    PACKED_DATA_NAME, PACKED_INDEX_NAME,
};
pub use patchify::{
    build_unified_sequence, create_coordinate_grid, pad_with_ids, patchify_image, unpatchify,
    PATCH_DIM, SEQ_MULTI_OF,
};
pub use pipeline::{timestep_embedding, ZImageTransformer, Z_IMAGE_DIM, Z_IMAGE_HEADS};
pub use reference::CpuReferenceBackend;
pub use rope::{RopeEmbedder, DEFAULT_AXES_DIMS, DEFAULT_AXES_LENS, DEFAULT_ROPE_THETA};
pub use runtime::{
    admit_image_budget, generate, generate_with_memory_budget, CancellationToken, ImageBackend,
    ImageBudgetError, ImageMemoryBudget, ImageMemoryPlan, ImageMetadata, ImageProgress,
    ImageRequest, ImageResult, ImageStage, ImageWorkLease, ImageWorkTracker, SchedulerMetadata,
    SequentialImageSlots, IMAGE_CANCELLED, IMAGE_ENGINE_REVISION, IMAGE_QUANTIZATION,
};
pub use scheduler::FlowMatchEulerScheduler;
pub use text_encoder::{encode_tokens, ShardedSafetensors};
pub use transformer::{AdaLnModulation, FinalLayer, ModulationParams, ZImageTransformerBlock};
pub use vae::{
    conv2d, decoded_to_rgb8, encode_rgb8_png, group_norm, upsample_nearest_2x, VaeDecoder,
};
