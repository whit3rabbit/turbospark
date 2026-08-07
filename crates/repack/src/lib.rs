//! `.gturbo` repack support: safetensors header parsing, ranged-download
//! planning, per-row int4/int8 quantization repack, and full-SHA256 install
//! verification. Ported from `Infrastructure/ModelIO/VerifiedInstallReceipt.swift`
//! (the verifier) and the intent of the (unported, closed-source) HF
//! streaming installer described in the ROADMAP.
#![forbid(unsafe_code)]

mod gemma4_checkpoint;
mod gturbo_writer;
mod hf_checkpoint;
mod install_verifier;
mod manifest_peek;
mod qwen36_config;
mod ranged_download;
mod repack;
mod resident_writer;
mod safetensors_header;
mod synthetic_model;
mod synthetic_qwen;
mod synthetic_real;

pub use gemma4_checkpoint::{
    classify_for_family, classify_gemma4, gemma4_manifest_quant, manifest_quant,
    orchestrate_gemma4_checkpoint, orchestrate_gemma4_checkpoint_sharded, parse_gemma4_config,
    parse_gemma4_quantization, write_gemma4_install, write_gemma4_install_streamed,
    write_qwen36_install, write_qwen36_install_streamed, Gemma4Bucket, Gemma4Error, Gemma4Quant,
    Gemma4RepackOutput, Gemma4Shards, GTURBO_PAGE_BYTES,
};
pub use gturbo_writer::{
    write_gturbo_install, write_gturbo_install_with_resident_index,
    write_gturbo_install_with_resident_index_and_experts, ExpertBlob, LayerBlobs,
    StreamingGturboWriter, SubTensor, WriterError,
};
pub use hf_checkpoint::{orchestrate_llama_checkpoint, LlamaCheckpointDims, OrchestrateError};
pub use install_verifier::verify_install_full_sha256;
pub use manifest_peek::peek_manifest_arch;
pub use qwen36_config::parse_qwen36_config;
pub use ranged_download::{
    fetch_safetensors_header, DownloadError, HttpRangeSource, MemoryRangeSource, RangeSource,
};
pub use repack::{
    int4_packed_bytes, int8_packed_bytes, quantize_matrix_int4, quantize_matrix_int8, RepackError,
};
pub use resident_writer::{
    build_resident_weights_bin, build_resident_weights_bin_mixed, RawTensorSpec, ResidentEntrySpec,
    ResidentTensorSpec, DTYPE_BF16, DTYPE_FP16, DTYPE_FP32, DTYPE_INT8_AFFINE,
};
pub use safetensors_header::{
    parse_header, required_prefix_len, SafetensorsHeader, SafetensorsHeaderError, TensorInfo,
    DEFAULT_MAX_HEADER_BYTES,
};
pub use synthetic_model::{
    build_synthetic_gemma4_install, build_synthetic_gemma4_moe_install,
    build_synthetic_gemma4_moe_streamed_install, build_synthetic_gemma4_swa_install,
    down_proj_name, embed_lm_head_name, expert_down_proj_name, expert_gate_proj_name,
    expert_up_proj_name, gate_proj_name, k_proj_name, o_proj_name, q_proj_name, router_name,
    tiny_gemma4_arch, up_proj_name,
};
pub use synthetic_qwen::{build_synthetic_qwen36_real_install, tiny_qwen36_arch};
pub use synthetic_real::build_synthetic_gemma4_real_install;

// Token id width consumed from the core primitives, keeping the dependency
// edge live and documenting the interchange type this crate uses throughout.
pub use foundation::TokenId;
