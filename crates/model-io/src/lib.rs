//! Model install layout: manifest/arch validation, packed-expert layout,
//! resident tensor index, SHA-256 verification, and the trusted install
//! receipt. Ported from `Infrastructure/ModelIO` (minus the Metal-specific
//! buffer wrapping, which belongs to the `gpu` crate in a later phase).
//!
//! Unlike the earlier phases, this crate is allowed a narrow amount of
//! `unsafe` (confined to the `mmap` call in `resident_buffer`) and platform
//! assumptions (4 KiB pages), per the workspace's cross-cutting rule that
//! model I/O and streaming are where that trade-off is made.

mod arch_baselines;
mod arch_config;
mod arch_validation;
mod error;
mod install_receipt;
mod manifest;
mod packed_experts_layout;
mod resident_buffer;
mod resident_index;
mod sha256;

pub use arch_baselines::{
    all_known_architectures, deepseek_v4_flash_284b_a13b, gemma4_26b_a4b, gpt_oss_20b,
    known_architecture, mixtral_8x7b, qwen36_35b_a3b, qwen3_30b_a3b,
};
pub use arch_config::{
    ArchConfig, CompressedAttentionConfig, HyperConnectionConfig, LinearAttentionConfig,
    ModelFamily, RopeScalingConfig,
};
pub use error::ModelError;
pub use install_receipt::{
    load as load_install_receipt, validate as validate_install_receipt,
    validate_manifest_binding as validate_install_receipt_manifest_binding,
    FileEntry as InstallReceiptFileEntry, ModelIntegrityPolicy, VerifiedInstallReceipt,
    DEFAULT_MAX_BYTES as INSTALL_RECEIPT_DEFAULT_MAX_BYTES,
};
pub use manifest::{
    known_flags, load as load_manifest, peek_family, validate as validate_manifest, Manifest,
    ManifestArch, ManifestFileEntry, ManifestQuant, ManifestQuantSlot, DEFAULT_MAX_BYTES,
    EXECUTABLE_GGUF_TYPES, REQUIRED_FILES,
};
pub use packed_experts_layout::{
    load as load_packed_experts_layout, ExpertEntry, LayerLayout, PackedExpertsLayout,
    SubTensorEntry, DEFAULT_MAX_BYTES as PACKED_EXPERTS_LAYOUT_DEFAULT_MAX_BYTES,
};
pub use resident_buffer::ResidentBuffer;
pub use resident_index::{
    load as load_resident_index, ResidentIndex, ResidentIndexEntry, ResidentIndexHeader,
    ENTRY_BYTES, HEADER_BYTES,
};
pub use sha256::{hash_data, hash_file, verify_file};

// Token id width consumed from the core primitives, keeping the dependency
// edge live and documenting the interchange type this crate uses throughout.
pub use foundation::TokenId;
