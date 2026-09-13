//! Native Z-Image-Turbo pipeline components: scheduler, conditioning,
//! and text encoder forward pass.
//!
//! Provides portable, pure-Rust references and execution routines for
//! diffusion conditioning and scheduling.

#![forbid(unsafe_code)]

pub mod conditioning;
pub mod fixtures;
pub mod patchify;
pub mod pipeline;
pub mod rope;
pub mod scheduler;
pub mod text_encoder;
pub mod transformer;
pub mod vae;

pub use conditioning::{frame_prompt, tokenize_prompt, MAX_SEQUENCE_LENGTH, PAD_TOKEN_ID};
pub use fixtures::{
    read_npy_bool, read_npy_complex64, read_npy_f32, read_npy_file_bool, read_npy_file_complex64,
    read_npy_file_f32, read_npy_file_i64, read_npy_i64, read_npz, read_npz_file, NpyArray,
};
pub use patchify::{
    build_unified_sequence, create_coordinate_grid, pad_with_ids, patchify_image, unpatchify,
    PATCH_DIM, SEQ_MULTI_OF,
};
pub use pipeline::{timestep_embedding, ZImageTransformer, Z_IMAGE_DIM, Z_IMAGE_HEADS};
pub use rope::{RopeEmbedder, DEFAULT_AXES_DIMS, DEFAULT_AXES_LENS, DEFAULT_ROPE_THETA};
pub use scheduler::FlowMatchEulerScheduler;
pub use text_encoder::{encode_tokens, ShardedSafetensors};
pub use transformer::{AdaLnModulation, FinalLayer, ModulationParams, ZImageTransformerBlock};
pub use vae::{
    conv2d, decoded_to_rgb8, encode_rgb8_png, group_norm, upsample_nearest_2x, VaeDecoder,
};
