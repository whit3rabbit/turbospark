//! `.gturbo` repack support: safetensors header parsing, ranged-download
//! planning, per-row int4/int8 quantization repack, and full-SHA256 install
//! verification. Ported from `Infrastructure/ModelIO/VerifiedInstallReceipt.swift`
//! (the verifier) and the intent of the (unported, closed-source) HF
//! streaming installer described in the ROADMAP.
#![forbid(unsafe_code)]
// `gturbo_writer::manifest`'s one `serde_json::json!` literal writes every
// arch field, and each field is one level of macro recursion. ROADMAP M5's
// five additions (`swigluLimit` plus the four YaRN scalars) took it past the
// default 128. Raising the limit is serde_json's own documented answer; the
// alternative is splitting the literal, which would hide the "written
// UNCONDITIONALLY" rule that block comment exists to enforce.
#![recursion_limit = "256"]

mod arch_registry;
mod gemma4_checkpoint;
mod gguf_checkpoint;
mod gguf_config;
mod gguf_header;
mod gguf_names;
mod gturbo_writer;
mod hf_checkpoint;
mod install_verifier;
mod manifest_peek;
mod qwen36_config;
mod ranged_download;
mod repack;
mod resident_writer;
mod safetensors_header;
mod synthetic_gguf;
mod synthetic_llama;
mod synthetic_model;
mod synthetic_qwen;
mod synthetic_qwen35;
mod synthetic_real;

pub use arch_registry::{
    config_json_family, describe_gguf_architecture, gguf_arch_support, hf_family_for_model_type,
    planned_gguf_architectures, refuse_foreign_config, ArchSupport, PlannedArch,
};
pub use gemma4_checkpoint::{
    classify_for_family, classify_gemma4, gemma4_manifest_quant, is_supported_affine_shape,
    manifest_quant, narrow_raw_to_bf16, orchestrate_gemma4_checkpoint,
    orchestrate_gemma4_checkpoint_sharded, parse_gemma4_config, parse_gemma4_quantization,
    pass_through_packed, write_gemma4_install, write_gemma4_install_streamed,
    write_qwen_gdn_dense_install, write_qwen_gdn_dense_install_streamed,
    write_qwen_gdn_moe_install, write_qwen_gdn_moe_install_streamed, Gemma4Bucket, Gemma4Error,
    Gemma4Quant, Gemma4RepackOutput, Gemma4Shards, NarrowedRaw, AFFINE_1BIT_GROUP_SIZE,
    AFFINE_2BIT_GROUP_SIZE, AFFINE_GROUP_SIZE, GTURBO_PAGE_BYTES,
};
pub use gguf_checkpoint::{
    dtype_tag_for_ggml_type, gguf_manifest_quant, orchestrate_gguf_checkpoint,
    write_gguf_install_streamed, GgufRepackError, GgufRepackOutput, FUSED_GATE_FIRST,
};
pub use gguf_config::{arch_from_gguf, GgufConfigError};
pub use gguf_header::{
    ggml_type_block, ggml_type_name, parse_header as parse_gguf_header, GgufHeader,
    GgufHeaderError, GgufTensorInfo, GgufValue, DEFAULT_ALIGNMENT,
    DEFAULT_MAX_HEADER_BYTES as GGUF_DEFAULT_MAX_HEADER_BYTES, SUPPORTED_VERSION as GGUF_VERSION,
};
pub use gguf_names::{
    family_for_architecture, gguf_architecture, map_gguf_name, GgufMapping, GgufNameError,
};
pub use gturbo_writer::{
    write_gturbo_install, write_gturbo_install_with_resident_index,
    write_gturbo_install_with_resident_index_and_experts, ExpertBlob, LayerBlobs,
    StreamingGturboWriter, SubTensor, WriterError,
};
pub use hf_checkpoint::{orchestrate_llama_checkpoint, LlamaCheckpointDims, OrchestrateError};
pub use install_verifier::verify_install_full_sha256;
pub use manifest_peek::peek_manifest_arch;
pub use qwen36_config::{parse_qwen_gdn_dense_config, parse_qwen_gdn_moe_config};
pub use ranged_download::{
    fetch_gguf_header, fetch_safetensors_header, DownloadError, HttpRangeSource, MemoryRangeSource,
    RangeSource, GGUF_INITIAL_FETCH_BYTES,
};
pub use repack::{
    int4_packed_bytes, int8_packed_bytes, quantize_matrix_int4, quantize_matrix_int8, RepackError,
};
pub use resident_writer::{
    build_resident_weights_bin, build_resident_weights_bin_mixed, RawTensorSpec, ResidentEntrySpec,
    ResidentTensorSpec, DTYPE_BF16, DTYPE_FP16, DTYPE_FP32, DTYPE_GGUF_Q4_0, DTYPE_GGUF_Q4_K,
    DTYPE_GGUF_Q6_K, DTYPE_GGUF_Q8_0, DTYPE_INT8_AFFINE, GGUF_BLOCK_DTYPES,
};
pub use safetensors_header::{
    parse_header, required_prefix_len, SafetensorsHeader, SafetensorsHeaderError, TensorInfo,
    DEFAULT_MAX_HEADER_BYTES,
};
pub use synthetic_gguf::{
    build_synthetic_gemma4_gguf, build_synthetic_gpt_oss_gguf, GgufBuilder, GgufFileAndRanges,
    QuantMix, SyntheticGgufShape, SyntheticGptOssShape,
};
pub use synthetic_llama::{
    build_synthetic_dense_llama_install, build_synthetic_gqa_moe_install,
    build_synthetic_llama_real_install, tiny_dense_llama_arch, tiny_gqa_moe_arch, tiny_llama_arch,
};
pub use synthetic_model::{
    build_synthetic_gemma4_install, build_synthetic_gemma4_moe_install,
    build_synthetic_gemma4_moe_streamed_install, build_synthetic_gemma4_swa_install,
    down_proj_name, embed_lm_head_name, expert_down_proj_name, expert_gate_proj_name,
    expert_up_proj_name, gate_proj_name, k_proj_name, o_proj_name, q_proj_name, router_name,
    tiny_gemma4_arch, up_proj_name,
};
pub use synthetic_qwen::{build_synthetic_qwen_gdn_moe_install, tiny_qwen_gdn_moe_arch};
pub use synthetic_qwen35::{
    build_synthetic_qwen_gdn_dense_install, build_synthetic_qwen_gdn_dense_install_at_bits,
    tiny_qwen_gdn_dense_arch,
};
pub use synthetic_real::build_synthetic_gemma4_real_install;

// Token id width consumed from the core primitives, keeping the dependency
// edge live and documenting the interchange type this crate uses throughout.
pub use foundation::TokenId;
