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
mod cgroup;
mod context_policy;
pub mod encoder_config;
mod error;
mod expert_cache_policy;
mod install_receipt;
mod kv_quant;
mod load_guard;
mod manifest;
mod ngram_hash;
mod ngram_table;
mod packed_experts_layout;
mod resident_buffer;
mod resident_index;
pub mod safetensors;
mod sha256;
mod steering_set;
mod vision_sidecar;

pub use arch_baselines::{
    all_known_architectures, deepseek_v2_lite_16b, deepseek_v4_flash_284b_a13b, gemma4_26b_a4b,
    gpt_oss_20b, known_architecture, minimax_m2, mixtral_8x7b, muse_glimmer_30b,
    muse_glimmer_layer_mask, qwen2_5_7b, qwen3_30b_a3b, qwen3_4b, qwen3_vl_4b, qwen4_exp_125b_a6b,
    qwen_gdn_dense_27b, qwen_gdn_moe_35b_a3b, spark_layer_mask, spark_x25_4b,
};
pub use arch_config::{
    ArchConfig, CompressedAttentionConfig, HyperConnectionConfig, LinearAttentionConfig, MlaConfig,
    ModelFamily, PleConfig, RopeScalingConfig, VisionConfig,
};
pub use cgroup::{
    parse_limit as parse_cgroup_limit, probe_self as probe_cgroup_memory, CgroupMemoryLimit,
};
pub use context_policy::{
    committed_breakdown, committed_breakdown_with_residency, committed_bytes, gdn_state_bytes,
    kv_bytes_for_context, kv_bytes_for_context_with, largest_context_within,
    largest_context_within_with, resolve_max_context, resolve_max_context_with, session_pool_bytes,
    session_pool_bytes_with, CommittedBytes, ContextCap, ContextFloorUnmet, ContextOverCap,
    ContextPlan, ContextRefused, ContextTooLarge, MaxContext, CONTEXT_BUDGET_FRACTION,
    CONTEXT_GRANULARITY, CONTEXT_RESERVE_BYTES, MAX_SUPPORTED_CONTEXT,
};
pub use encoder_config::{EncoderConfig, EncoderQuantization};
pub use error::ModelError;
pub use expert_cache_policy::{
    auto_residency_prefers_mapped, ExpertCacheSlots, ExpertResidency, ResolvedExpertResidency,
    HEADROOM_FRACTION, HEADROOM_RESERVE_BYTES,
};
pub use install_receipt::{
    load as load_install_receipt, validate as validate_install_receipt,
    validate_manifest_binding as validate_install_receipt_manifest_binding,
    FileEntry as InstallReceiptFileEntry, ModelIntegrityPolicy, VerifiedInstallReceipt,
    DEFAULT_MAX_BYTES as INSTALL_RECEIPT_DEFAULT_MAX_BYTES,
};
pub use kv_quant::{
    kv_layer_strides, layer_is_quantized, rht_supported, tq_packed_words, tq_row_bytes, KvQuant,
};
pub use load_guard::{GuardBudget, LoadGuard, LoadPolicy};
pub use manifest::{
    known_flags, load as load_manifest, peek_family, validate as validate_manifest, Manifest,
    ManifestArch, ManifestFileEntry, ManifestHadamard, ManifestHadamardSigns, ManifestQuant,
    ManifestQuantSlot, DEFAULT_MAX_BYTES, EXECUTABLE_GGUF_TYPES, REQUIRED_FILES,
};
pub use ngram_hash::{
    build_layer_multipliers, derive_head_vocab_and_offsets, find_nth_prime_after, ple_ngram_rows,
    NgramContext,
};
pub use ngram_table::{
    load_ngram_table_layout, NgramTableLayout, NGRAM_HEADER_MAX_BYTES, NGRAM_TABLE_BLOB,
    NGRAM_TABLE_DIR, NGRAM_TABLE_HEADER,
};
pub use packed_experts_layout::{
    load as load_packed_experts_layout, load_from as load_packed_layout_from, ExpertEntry,
    LayerLayout, PackedExpertsLayout, SubTensorEntry,
    DEFAULT_MAX_BYTES as PACKED_EXPERTS_LAYOUT_DEFAULT_MAX_BYTES, PACKED_EXPERTS_DIR,
    PACKED_VISION_DIR,
};
pub use resident_buffer::ResidentBuffer;
pub use resident_index::{
    load as load_resident_index, ResidentIndex, ResidentIndexEntry, ResidentIndexHeader,
    ENTRY_BYTES, HEADER_BYTES,
};
pub use safetensors::SafetensorsFile;
pub use sha256::{hash_data, hash_file, verify_file};
pub use steering_set::{LayerDirection, SteeringSet};
pub use vision_sidecar::{
    is_sidecar_dir, load as load_vision_sidecar, sidecar_arch, PairsWith, SidecarRecord,
    SidecarSource, SIDECAR_KIND, SIDECAR_RECORD_FILE,
};

// Token id width consumed from the core primitives, keeping the dependency
// edge live and documenting the interchange type this crate uses throughout.
pub use foundation::TokenId;
