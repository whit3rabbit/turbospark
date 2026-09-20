//! Compile-time architecture baselines and the family-dependent config
//! types they're built from. Ported from
//! `Infrastructure/ModelIO/ModelTypes.swift`.
//!
//! `manifest.json -> arch` must match the resolved [`ArchConfig`]
//! field-by-field at load time (see `manifest::validate_arch`); mismatches
//! produce [`crate::ModelError::ArchMismatch`].

mod config;
mod family;
mod sub_configs;

pub use config::ArchConfig;
pub use family::ModelFamily;
pub use sub_configs::{
    CompressedAttentionConfig, HyperConnectionConfig, LinearAttentionConfig, MlaConfig, PleConfig,
    RopeScalingConfig, VisionConfig, MAX_VISION_DEEPSTACK_MERGERS,
};
