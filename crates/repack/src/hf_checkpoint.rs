//! Orchestrates a real, downloaded Llama-family HF checkpoint's
//! `safetensors` file into a `.gturbo` install: walks the checkpoint's
//! standard tensor names (`model.embed_tokens.weight`, per-layer
//! `self_attn.{q,k,v,o}_proj.weight`, `mlp.{gate,up,down}_proj.weight`,
//! the two per-layer norms, the final norm, and `lm_head.weight` when not
//! tied), reads each tensor's real bytes via a [`RangeSource`], decodes
//! them to `f32`, quantizes with [`crate::quantize_matrix_int4`], and
//! returns one [`ResidentTensorSpec`] per tensor ready for
//! [`crate::build_resident_weights_bin`]. This is the piece Phase 8's own
//! module docs called out as missing: "walking a real downloaded HF
//! checkpoint's safetensors files, mapping tensor names to `.gturbo`
//! roles, and calling the quantizer + writer end to end".
//!
//! Scope: one concrete, standard tensor-naming convention (the Llama
//! family, which most small dense HF checkpoints share), not a universal
//! any-architecture mapper. Supports `BF16` and `F32` source dtypes (the
//! two real small checkpoints this was exercised against use); `F16` is
//! not decoded. Whether the resulting install can also be *run* through
//! `crates/runtime`'s `RealForwardRunner` is a separate, narrower question
//! (that runner is dense/Gemma-4-shaped-only, and quantizing norm weights
//! to INT4 — done here, since this module's job is producing a valid
//! install, not numerical fidelity — is not how a production repacker
//! would treat them); see `DEVIATIONS.md`.

use crate::ranged_download::{DownloadError, RangeSource};
use crate::repack::{quantize_matrix_int4, RepackError};
use crate::resident_writer::ResidentTensorSpec;
use crate::safetensors_header::SafetensorsHeader;

#[derive(Debug)]
pub enum OrchestrateError {
    MissingTensor(String),
    UnsupportedDtype { tensor: String, dtype: String },
    Download(DownloadError),
    Quantize(RepackError),
}

impl std::fmt::Display for OrchestrateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OrchestrateError::MissingTensor(name) => {
                write!(f, "checkpoint is missing tensor: {name}")
            }
            OrchestrateError::UnsupportedDtype { tensor, dtype } => {
                write!(f, "tensor {tensor} has unsupported dtype {dtype}")
            }
            OrchestrateError::Download(e) => write!(f, "{e}"),
            OrchestrateError::Quantize(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for OrchestrateError {}

impl From<DownloadError> for OrchestrateError {
    fn from(e: DownloadError) -> Self {
        OrchestrateError::Download(e)
    }
}

/// The real dimensions this orchestrator needs, read from the checkpoint's
/// own `config.json` by the caller (this module has no HTTP/JSON
/// dependency of its own beyond what `RangeSource` already provides).
#[derive(Debug, Clone, Copy)]
pub struct LlamaCheckpointDims {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub num_layers: usize,
    pub vocab_size: usize,
    pub tie_word_embeddings: bool,
}

fn decode_tensor_f32(
    header: &SafetensorsHeader,
    source: &dyn RangeSource,
    name: &str,
) -> Result<Vec<f32>, OrchestrateError> {
    let info = header
        .tensors
        .get(name)
        .ok_or_else(|| OrchestrateError::MissingTensor(name.to_string()))?;
    let (start, end) = header
        .absolute_range(name)
        .expect("tensor present in header.tensors, so absolute_range must resolve");
    let bytes = source.read_range(start, end)?;

    match info.dtype.as_str() {
        "F32" => Ok(bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()),
        "BF16" => Ok(bytes
            .chunks_exact(2)
            .map(|c| compute::bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect()),
        other => Err(OrchestrateError::UnsupportedDtype {
            tensor: name.to_string(),
            dtype: other.to_string(),
        }),
    }
}

fn quantize_named(
    header: &SafetensorsHeader,
    source: &dyn RangeSource,
    tensor_name: &str,
    spec_name: &str,
    rows: usize,
    cols: usize,
) -> Result<ResidentTensorSpec, OrchestrateError> {
    let data = decode_tensor_f32(header, source, tensor_name)?;
    let quantized = quantize_matrix_int4(&data, rows, cols).map_err(OrchestrateError::Quantize)?;
    Ok(crate::repack::resident_spec_from_int4_rows(
        spec_name, &quantized, cols,
    ))
}

/// Walks a real Llama-family checkpoint's tensors (per `dims`) and returns
/// one quantized [`ResidentTensorSpec`] per resident weight: `embed_tokens`,
/// per layer `layerN.{q,k,v,o,gate,up,down}_proj` plus
/// `layerN.{input,post_attn}_norm`, `final_norm`, and (only when
/// `!dims.tie_word_embeddings`) `lm_head`.
pub fn orchestrate_llama_checkpoint(
    header: &SafetensorsHeader,
    source: &dyn RangeSource,
    dims: LlamaCheckpointDims,
) -> Result<Vec<ResidentTensorSpec>, OrchestrateError> {
    let h = dims.hidden_size;
    let qdim = dims.num_heads * dims.head_dim;
    let kvdim = dims.num_kv_heads * dims.head_dim;
    let f = dims.intermediate_size;

    let mut specs = Vec::with_capacity(3 + dims.num_layers * 9);
    specs.push(quantize_named(
        header,
        source,
        "model.embed_tokens.weight",
        "embed_tokens",
        dims.vocab_size,
        h,
    )?);

    for l in 0..dims.num_layers {
        let p = format!("model.layers.{l}");
        specs.push(quantize_named(
            header,
            source,
            &format!("{p}.self_attn.q_proj.weight"),
            &format!("layer{l}.q_proj"),
            qdim,
            h,
        )?);
        specs.push(quantize_named(
            header,
            source,
            &format!("{p}.self_attn.k_proj.weight"),
            &format!("layer{l}.k_proj"),
            kvdim,
            h,
        )?);
        specs.push(quantize_named(
            header,
            source,
            &format!("{p}.self_attn.v_proj.weight"),
            &format!("layer{l}.v_proj"),
            kvdim,
            h,
        )?);
        specs.push(quantize_named(
            header,
            source,
            &format!("{p}.self_attn.o_proj.weight"),
            &format!("layer{l}.o_proj"),
            h,
            qdim,
        )?);
        specs.push(quantize_named(
            header,
            source,
            &format!("{p}.mlp.gate_proj.weight"),
            &format!("layer{l}.gate_proj"),
            f,
            h,
        )?);
        specs.push(quantize_named(
            header,
            source,
            &format!("{p}.mlp.up_proj.weight"),
            &format!("layer{l}.up_proj"),
            f,
            h,
        )?);
        specs.push(quantize_named(
            header,
            source,
            &format!("{p}.mlp.down_proj.weight"),
            &format!("layer{l}.down_proj"),
            h,
            f,
        )?);
        // Norm weights: a single row of length `h`, quantized as a 1-row
        // matrix. Real repackers would keep these unquantized (FP16); this
        // orchestrator quantizes everything uniformly to keep the writer's
        // one INT4-only resident-tensor format sufficient — see module docs.
        specs.push(quantize_named(
            header,
            source,
            &format!("{p}.input_layernorm.weight"),
            &format!("layer{l}.input_norm"),
            1,
            h,
        )?);
        specs.push(quantize_named(
            header,
            source,
            &format!("{p}.post_attention_layernorm.weight"),
            &format!("layer{l}.post_attn_norm"),
            1,
            h,
        )?);
    }

    specs.push(quantize_named(
        header,
        source,
        "model.norm.weight",
        "final_norm",
        1,
        h,
    )?);

    if !dims.tie_word_embeddings {
        specs.push(quantize_named(
            header,
            source,
            "lm_head.weight",
            "lm_head",
            dims.vocab_size,
            h,
        )?);
    }

    Ok(specs)
}
