//! Resident-set transcoding, source conventions, and raw tensor specs.

use foundation::LogitValue as F16;
use model_io::{ArchConfig, ModelFamily};
use std::collections::BTreeMap;

use super::types::{
    dtype_tag_for_ggml_type, logical_shape, read_tensor, GgufRepackError, GGML_TYPE_F32,
};
use crate::gguf_header::GgufHeader;
use crate::gguf_names::{map_gguf_name, GgufMapping};
use crate::ranged_download::RangeSource;
use crate::resident_writer::{RawTensorSpec, ResidentEntrySpec};

const GGML_TYPE_F16: u32 = 1;
const GGML_TYPE_Q4_0: u32 = 2;
const GGML_TYPE_Q5_0: u32 = 6;
const GGML_TYPE_Q3_K: u32 = 11;
const GGML_TYPE_Q4_K: u32 = 12;
const GGML_TYPE_Q5_K: u32 = 13;
const GGML_TYPE_Q6_K: u32 = 14;
const GGML_TYPE_IQ3_S: u32 = 21;
const GGML_TYPE_IQ4_XS: u32 = 23;
const GGML_TYPE_Q2_0: u32 = 42;
const QWEN4EXP_SIMPLE_TRANSCODE_TYPES: [u32; 4] = [
    GGML_TYPE_F16,
    GGML_TYPE_Q2_0,
    GGML_TYPE_Q4_0,
    GGML_TYPE_Q5_0,
];
const QWEN4EXP_SSM_OUT_TRANSCODE_TYPES: [u32; 6] = [
    GGML_TYPE_Q3_K,
    GGML_TYPE_Q4_K,
    GGML_TYPE_Q5_K,
    GGML_TYPE_Q6_K,
    GGML_TYPE_IQ3_S,
    GGML_TYPE_IQ4_XS,
];

/// Whether the Qwen4Exp walk converts this resident source tensor to BF16.
/// Kept beside the writer so the catalog probe can use the same source-type
/// policy instead of guessing from the global executable-kernel list.
pub fn qwen4exp_tensor_is_transcoded(canonical: &str, ggml_type: u32) -> bool {
    QWEN4EXP_SIMPLE_TRANSCODE_TYPES.contains(&ggml_type)
        || (canonical.ends_with("linear_attn.out_proj.weight")
            && QWEN4EXP_SSM_OUT_TRANSCODE_TYPES.contains(&ggml_type))
}

fn int8_transcode_targets(family: ModelFamily) -> &'static [&'static str] {
    match family {
        ModelFamily::Gemma4 => &["router.proj.weight"],
        ModelFamily::QwenGdnMoe | ModelFamily::Qwen4Exp => {
            &["mlp.gate.weight", "mlp.shared_expert_gate.weight"]
        }
        // Mixtral's `ffn_gate_inp` maps to the same canonical name Qwen's
        // router takes, and it is F16 in the 2023 conversion and F32 in the
        // 2025 one -- both narrow to the INT8 affine the runtime's router
        // GEMV reads. A dense Llama has no router and so no target here.
        // Same canonical router name, same F32 source tensor. `qwen3moe`
        // additionally ships its q/k norms as F32, but those take the BF16
        // default: the runtime's `norm_view` reads BF16.
        // gpt-oss names its router `ffn_gate_inp` too, so it maps to the
        // same canonical name and takes the same INT8 transcode. Its BIASES
        // are F32 and are NOT here: they take the BF16 default, which is
        // what a bias-add kernel reads.
        ModelFamily::Llama | ModelFamily::Qwen3Moe | ModelFamily::GptOss => &["mlp.gate.weight"],
        // `deepseek2`'s router is `ffn_gate_inp` F32 exactly as `qwen3moe`'s
        // is (read off the V2-Lite header), and its INT8-transcoded canonical
        // name is the same `mlp.gate.weight`.
        ModelFamily::Deepseek2 => &["mlp.gate.weight"],
        // `qwen3_5` DOES have a real GGUF file now
        // (`ornith-ai/Ornith-1.5-9B-GGUF`, walked by
        // `tests/ornith_gguf_network.rs`), so the empty list is not about
        // an unreached family. It is empty because the dense half of this
        // architecture has no router and no shared-expert gate at all,
        // unlike Qwen 3.6's MoE list.
        // `muse_glimmer` is MLX-safetensors-only for the same reason, and is
        // additionally DENSE, so it has no router to transcode even if a
        // GGUF of it were ever published.
        // `qwen4_exp` is the one family here that DOES have published GGUFs
        // (unsloth's and bartowski's) and is still empty, for a different
        // reason than its neighbours: this port ingests the MLX safetensors
        // and REFUSES the GGUF at `gguf_config`, so this arm is unreachable.
        // Listing its router here would be a claim about a file nobody in
        // this repo has parsed. It gains a list when the GGUF walk gains an
        // arm, not before.
        //
        // `spark2_5` is DENSE and has no router and no shared-expert gate:
        // its only small per-layer tensor is the headwise `attn_gate`, which
        // the flow reads as FP16 and which takes the BF16 default like every
        // other small tensor.
        //
        // Dense `qwen3` is DENSE the same way (`num_experts == 0`) and its
        // real GGUFs ARE walked by this function -- unlike `qwen3_5` above,
        // this is not an unreached arm. There is simply no router or
        // shared-expert gate tensor in the file to transcode.
        //
        // `qwen3_vl` is dense like its neighbours and has no GGUF intake
        // this pass (`gguf_names` maps nothing for it), so the arm is
        // unreachable today and stays empty for the same reason a dense
        // family's is: no router, no shared-expert gate.
        ModelFamily::DeepseekV4Flash
        | ModelFamily::QwenGdnDense
        | ModelFamily::MuseGlimmer
        | ModelFamily::Spark25
        | ModelFamily::Qwen3Dense
        | ModelFamily::Qwen2Dense
        | ModelFamily::MiniMaxM2
        | ModelFamily::Qwen3Vl => &[],
    }
}

fn row_and_col(dims: &[u64]) -> (usize, usize) {
    let (r, c, _, _) = logical_shape(dims);
    if dims.len() == 1 {
        (r as usize, 1)
    } else {
        (r as usize, c as usize)
    }
}
use super::conventions::{apply_source_convention, apply_source_convention_bytes};

/// Result of transcoding an F32 GGUF tensor into BF16 or INT8 resident format.
pub struct Transcoded {
    /// Transcoded resident entry specification.
    pub spec: ResidentEntrySpec,
    /// Number of values where precision loss occurred during transcoding.
    pub lossy: usize,
}

fn block_tensor_values(
    name: &str,
    dims: &[u64],
    bytes: &[u8],
    block_elems: usize,
    block_bytes: usize,
    dtype_name: &str,
    dequantize: fn(&[u8], usize) -> Vec<f32>,
) -> Result<Vec<f32>, GgufRepackError> {
    let shape_error = |detail: String| GgufRepackError::ShapeMismatch {
        tensor: name.to_string(),
        detail,
    };
    if dims.is_empty() || dims[0] == 0 || dims[0] as usize % block_elems != 0 {
        return Err(shape_error(format!(
            "{dtype_name} row width must be a positive multiple of {block_elems}, got {dims:?}"
        )));
    }
    let elements = dims
        .iter()
        .try_fold(1u64, |count, &dim| count.checked_mul(dim))
        .and_then(|count| usize::try_from(count).ok())
        .ok_or_else(|| shape_error(format!("{dtype_name} element count overflows")))?;
    let expected_bytes = elements
        .checked_div(block_elems)
        .and_then(|blocks| blocks.checked_mul(block_bytes))
        .ok_or_else(|| shape_error(format!("{dtype_name} byte count overflows")))?;
    if bytes.len() != expected_bytes {
        return Err(shape_error(format!(
            "{dtype_name} tensor has {} bytes, expected {expected_bytes}",
            bytes.len()
        )));
    }
    Ok(dequantize(bytes, elements))
}

fn dequantized_bf16_spec(
    name: &str,
    canonical: String,
    dims: &[u64],
    arch: &ArchConfig,
    mut values: Vec<f32>,
) -> Result<Transcoded, GgufRepackError> {
    let elements = dims
        .iter()
        .try_fold(1u64, |count, &dim| count.checked_mul(dim))
        .and_then(|count| usize::try_from(count).ok())
        .ok_or_else(|| GgufRepackError::ShapeMismatch {
            tensor: name.to_string(),
            detail: "dequantized tensor element count overflows".into(),
        })?;
    if values.len() != elements || values.iter().any(|value| !value.is_finite()) {
        return Err(GgufRepackError::ShapeMismatch {
            tensor: name.to_string(),
            detail: format!(
                "dequantized to {} finite values, expected {elements}",
                values.len()
            ),
        });
    }
    apply_source_convention(name, &canonical, arch, row_and_col(dims).1, &mut values)?;
    let (spec, lossy) = raw_bf16_spec(canonical, &values, dims);
    Ok(Transcoded { spec, lossy })
}

fn transcode_q2_0(
    name: &str,
    canonical: String,
    bytes: &[u8],
    dims: &[u64],
    arch: &ArchConfig,
) -> Result<Transcoded, GgufRepackError> {
    let values = block_tensor_values(
        name,
        dims,
        bytes,
        compute::quant_gguf::Q2_0_BLOCK_ELEMS,
        compute::quant_gguf::Q2_0_BLOCK_BYTES,
        "Q2_0",
        compute::quant_gguf::dequantize_q2_0,
    )?;
    dequantized_bf16_spec(name, canonical, dims, arch, values)
}

fn transcode_q4_0(
    name: &str,
    canonical: String,
    bytes: &[u8],
    dims: &[u64],
    arch: &ArchConfig,
) -> Result<Transcoded, GgufRepackError> {
    let values = block_tensor_values(
        name,
        dims,
        bytes,
        compute::quant_gguf::Q4_0_BLOCK_ELEMS,
        compute::quant_gguf::Q4_0_BLOCK_BYTES,
        "Q4_0",
        compute::quant_gguf::dequantize_q4_0,
    )?;
    dequantized_bf16_spec(name, canonical, dims, arch, values)
}

fn transcode_q5_0(
    name: &str,
    canonical: String,
    bytes: &[u8],
    dims: &[u64],
    arch: &ArchConfig,
) -> Result<Transcoded, GgufRepackError> {
    let values = block_tensor_values(
        name,
        dims,
        bytes,
        compute::quant_gguf::Q5_0_BLOCK_ELEMS,
        compute::quant_gguf::Q5_0_BLOCK_BYTES,
        "Q5_0",
        compute::quant_gguf::dequantize_q5_0,
    )?;
    dequantized_bf16_spec(name, canonical, dims, arch, values)
}

fn transcode_f16(
    name: &str,
    canonical: String,
    bytes: &[u8],
    dims: &[u64],
    arch: &ArchConfig,
) -> Result<Transcoded, GgufRepackError> {
    let elements = dims
        .iter()
        .try_fold(1u64, |count, &dim| count.checked_mul(dim))
        .and_then(|count| usize::try_from(count).ok())
        .ok_or_else(|| GgufRepackError::ShapeMismatch {
            tensor: name.to_string(),
            detail: "F16 element count overflows".into(),
        })?;
    let expected_bytes = elements
        .checked_mul(2)
        .ok_or_else(|| GgufRepackError::ShapeMismatch {
            tensor: name.to_string(),
            detail: "F16 byte count overflows".into(),
        })?;
    if bytes.len() != expected_bytes {
        return Err(GgufRepackError::ShapeMismatch {
            tensor: name.to_string(),
            detail: format!(
                "F16 tensor has {} bytes, expected {expected_bytes}",
                bytes.len()
            ),
        });
    }
    let values = bytes
        .chunks_exact(2)
        .map(|pair| f32::from(F16::from_bits(u16::from_le_bytes([pair[0], pair[1]]))))
        .collect();
    dequantized_bf16_spec(name, canonical, dims, arch, values)
}

fn f32_values(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Transcodes an F32 GGUF tensor into a BF16 raw spec or INT8 affine spec.
pub fn transcode_f32(
    name: &str,
    canonical: String,
    bytes: &[u8],
    dims: &[u64],
    arch: &ArchConfig,
) -> Result<Transcoded, GgufRepackError> {
    let mut values = f32_values(bytes);
    if values.len() * 4 != bytes.len() {
        return Err(GgufRepackError::ShapeMismatch {
            tensor: name.to_string(),
            detail: format!("{} bytes is not a whole number of F32 values", bytes.len()),
        });
    }
    let family = arch.family;
    if family == ModelFamily::MiniMaxM2
        && (canonical.ends_with("mlp.gate.weight")
            || canonical.ends_with("mlp.e_score_correction_bias"))
    {
        return Ok(Transcoded {
            spec: ResidentEntrySpec::Raw(RawTensorSpec {
                name: canonical,
                dtype: crate::resident_writer::DTYPE_FP32,
                bytes: bytes.to_vec(),
                shape: logical_shape(dims),
            }),
            lossy: 0,
        });
    }
    apply_source_convention(name, &canonical, arch, row_and_col(dims).1, &mut values)?;

    let wants_int8 = int8_transcode_targets(family)
        .iter()
        .any(|suffix| canonical.ends_with(suffix));
    if !wants_int8 {
        let (spec, lossy) = raw_bf16_spec(canonical, &values, dims);
        return Ok(Transcoded { spec, lossy });
    }

    let (rows, cols) = match dims.len() {
        1 => (1usize, dims[0] as usize),
        2 => {
            let (r, c, _, _) = logical_shape(dims);
            (r as usize, c as usize)
        }
        _ => {
            return Err(GgufRepackError::ShapeMismatch {
                tensor: name.to_string(),
                detail: format!("INT8 transcode needs a rank-1 or rank-2 tensor, got {dims:?}"),
            })
        }
    };
    if rows * cols != values.len() {
        return Err(GgufRepackError::ShapeMismatch {
            tensor: name.to_string(),
            detail: format!("{} values do not fill {rows}x{cols}", values.len()),
        });
    }
    if cols % 64 != 0 {
        return Err(GgufRepackError::ShapeMismatch {
            tensor: name.to_string(),
            detail: format!("row length {cols} is not a multiple of the 64-element group"),
        });
    }
    let quantized: Vec<_> = (0..rows)
        .map(|r| compute::quantize_int8_affine(&values[r * cols..(r + 1) * cols]))
        .collect();
    Ok(Transcoded {
        spec: ResidentEntrySpec::Int8(crate::repack::resident_spec_from_int8_rows(
            canonical, &quantized, cols,
        )),
        lossy: 0,
    })
}

fn raw_bf16_spec(canonical: String, values: &[f32], dims: &[u64]) -> (ResidentEntrySpec, usize) {
    let mut lossy = 0usize;
    let mut out = Vec::with_capacity(values.len() * 2);
    for &value in values {
        let bits = compute::f32_to_bf16(value);
        if compute::bf16_to_f32(bits) != value {
            lossy += 1;
        }
        out.extend_from_slice(&bits.to_le_bytes());
    }
    (
        ResidentEntrySpec::Raw(RawTensorSpec {
            name: canonical,
            dtype: crate::resident_writer::DTYPE_BF16,
            bytes: out,
            shape: logical_shape(dims),
        }),
        lossy,
    )
}

/// Tuple holding a list of resident tensor specs and a list of lossy transcode counts per tensor.
pub type ResidentSet = (Vec<ResidentEntrySpec>, Vec<(String, usize)>);

/// Reads and transcodes the requested resident tensors from a GGUF checkpoint.
pub fn resident_entries(
    header: &GgufHeader,
    source: &dyn RangeSource,
    arch: &ArchConfig,
    names: &[&str],
) -> Result<ResidentSet, GgufRepackError> {
    let family = arch.family;
    let mut out = Vec::with_capacity(names.len());
    let mut lossy = Vec::new();
    let mut indexer_projection_parts: BTreeMap<usize, (Option<&str>, Option<&str>)> =
        BTreeMap::new();
    for name in names {
        let info = &header.tensors[*name];
        let GgufMapping::Resident(canonical) = map_gguf_name(name, family)? else {
            continue;
        };
        let qwen4exp_dequantized_type = family == ModelFamily::Qwen4Exp
            && qwen4exp_tensor_is_transcoded(&canonical, info.ggml_type);
        let dtype = if qwen4exp_dequantized_type {
            crate::resident_writer::DTYPE_BF16
        } else {
            dtype_tag_for_ggml_type(info.ggml_type).ok_or_else(|| {
                GgufRepackError::UnsupportedType {
                    tensor: (*name).to_string(),
                    ggml_type: info.ggml_type,
                }
            })?
        };
        if family == ModelFamily::Qwen4Exp {
            if let Some((layer, is_query)) = qwen4_indexer_projection_part(name) {
                if arch.layer_is_linear(layer) {
                    return Err(GgufRepackError::ShapeMismatch {
                        tensor: (*name).to_string(),
                        detail: "Qwen4Exp indexer projections occur only on full-attention layers"
                            .to_string(),
                    });
                }
                let parts = indexer_projection_parts.entry(layer).or_default();
                let slot = if is_query { &mut parts.0 } else { &mut parts.1 };
                if slot.replace(*name).is_some() {
                    return Err(GgufRepackError::ShapeMismatch {
                        tensor: (*name).to_string(),
                        detail: "duplicate Qwen4Exp indexer projection part".to_string(),
                    });
                }
                continue;
            }
        }
        let bytes = read_tensor(header, source, name)?;
        if family == ModelFamily::Qwen4Exp {
            let t = match info.ggml_type {
                GGML_TYPE_F16 => Some(transcode_f16(
                    name,
                    canonical.clone(),
                    &bytes,
                    &info.dims,
                    arch,
                )?),
                GGML_TYPE_Q2_0 => Some(transcode_q2_0(
                    name,
                    canonical.clone(),
                    &bytes,
                    &info.dims,
                    arch,
                )?),
                GGML_TYPE_Q4_0 => Some(transcode_q4_0(
                    name,
                    canonical.clone(),
                    &bytes,
                    &info.dims,
                    arch,
                )?),
                GGML_TYPE_Q5_0 => Some(transcode_q5_0(
                    name,
                    canonical.clone(),
                    &bytes,
                    &info.dims,
                    arch,
                )?),
                _ => None,
            };
            if let Some(t) = t {
                if t.lossy > 0 {
                    lossy.push(((*name).to_string(), t.lossy));
                }
                out.push(t.spec);
                continue;
            }
        }
        if info.ggml_type == GGML_TYPE_F32 {
            let t = transcode_f32(name, canonical, &bytes, &info.dims, arch)?;
            if t.lossy > 0 {
                lossy.push(((*name).to_string(), t.lossy));
            }
            out.push(t.spec);
            continue;
        }
        if family == ModelFamily::Qwen4Exp
            && canonical.ends_with("linear_attn.out_proj.weight")
            && QWEN4EXP_SSM_OUT_TRANSCODE_TYPES.contains(&info.ggml_type)
        {
            // Qwen3.8's 48 V heads are 128 columns wide, while this artifact
            // quantizes `ssm_out` in 256-element blocks. A block spans two
            // heads, so no byte permutation can restore the runtime's
            // even/odd V-head order. Dequantize one matrix, permute its
            // columns, and carry the values as BF16; this keeps the source
            // quantization values while avoiding a false byte-level shuffle.
            let elements = info
                .dims
                .iter()
                .try_fold(1u64, |count, &dim| count.checked_mul(dim))
                .and_then(|count| usize::try_from(count).ok())
                .ok_or_else(|| GgufRepackError::ShapeMismatch {
                    tensor: (*name).to_string(),
                    detail: "V-head output projection element count overflows".to_string(),
                })?;
            let values = match info.ggml_type {
                GGML_TYPE_Q3_K => compute::quant_gguf::dequantize_q3_k(&bytes, elements),
                GGML_TYPE_Q4_K => compute::quant_gguf::dequantize_q4_k(&bytes, elements),
                GGML_TYPE_Q5_K => compute::quant_gguf::dequantize_q5_k(&bytes, elements),
                GGML_TYPE_Q6_K => compute::quant_gguf::dequantize_q6_k(&bytes, elements),
                GGML_TYPE_IQ3_S => compute::quant_gguf_iq::dequantize_iq3_s(&bytes, elements),
                GGML_TYPE_IQ4_XS => compute::quant_gguf_iq::dequantize_iq4_xs(&bytes, elements),
                _ => unreachable!(),
            };
            if values.len() != elements || values.iter().any(|value| !value.is_finite()) {
                return Err(GgufRepackError::ShapeMismatch {
                    tensor: (*name).to_string(),
                    detail: format!(
                        "type {} dequantized to {} finite values, expected {elements}",
                        info.ggml_type,
                        values.len()
                    ),
                });
            }
            let mut values = values;
            apply_source_convention(
                name,
                &canonical,
                arch,
                row_and_col(&info.dims).1,
                &mut values,
            )?;
            let (spec, precision_loss) = raw_bf16_spec(canonical, &values, &info.dims);
            if precision_loss > 0 {
                lossy.push(((*name).to_string(), precision_loss));
            }
            out.push(spec);
            continue;
        }
        let mut bytes = bytes;
        apply_source_convention_bytes(
            name,
            &canonical,
            arch,
            row_and_col(&info.dims),
            info.ggml_type,
            &mut bytes,
        )?;
        out.push(ResidentEntrySpec::Raw(RawTensorSpec {
            name: canonical,
            dtype,
            bytes,
            shape: logical_shape(&info.dims),
        }));
    }
    for (layer, (query_name, key_name)) in indexer_projection_parts {
        let query_name = query_name.ok_or_else(|| GgufRepackError::MissingTensor {
            name: format!("blk.{layer}.indexer.q_proj.weight"),
        })?;
        let key_name = key_name.ok_or_else(|| GgufRepackError::MissingTensor {
            name: format!("blk.{layer}.indexer.k_proj.weight"),
        })?;
        let query_info = &header.tensors[query_name];
        let key_info = &header.tensors[key_name];
        let expected_q = arch
            .compressed_attention
            .index_n_heads
            .checked_mul(arch.compressed_attention.index_head_dim)
            .ok_or_else(|| GgufRepackError::ShapeMismatch {
                tensor: query_name.to_string(),
                detail: "QSA query projection width overflows".to_string(),
            })?;
        let hidden =
            u64::try_from(arch.hidden_size).map_err(|_| GgufRepackError::ShapeMismatch {
                tensor: query_name.to_string(),
                detail: "hidden size is negative".to_string(),
            })?;
        let query_rows = u64::try_from(expected_q).map_err(|_| GgufRepackError::ShapeMismatch {
            tensor: query_name.to_string(),
            detail: "QSA query projection width is negative".to_string(),
        })?;
        let key_rows = u64::try_from(arch.compressed_attention.index_head_dim).map_err(|_| {
            GgufRepackError::ShapeMismatch {
                tensor: key_name.to_string(),
                detail: "QSA key projection width is negative".to_string(),
            }
        })?;
        for (name, info, output_rows) in [
            (query_name, query_info, query_rows),
            (key_name, key_info, key_rows),
        ] {
            if info.dims != [hidden, output_rows] || info.ggml_type != 30 {
                return Err(GgufRepackError::ShapeMismatch {
                    tensor: name.to_string(),
                    detail: format!(
                        "Qwen4Exp QSA parts must be BF16 type 30 with GGUF shape \
                         [{hidden}, {output_rows}], got type {} shape {:?}",
                        info.ggml_type, info.dims
                    ),
                });
            }
        }
        if query_info.ggml_type != key_info.ggml_type {
            return Err(GgufRepackError::ShapeMismatch {
                tensor: key_name.to_string(),
                detail: "Qwen4Exp QSA query and key projections use different dtypes".to_string(),
            });
        }
        let mut bytes = read_tensor(header, source, query_name)?;
        bytes.extend_from_slice(&read_tensor(header, source, key_name)?);
        let combined_rows =
            query_rows
                .checked_add(key_rows)
                .ok_or_else(|| GgufRepackError::ShapeMismatch {
                    tensor: query_name.to_string(),
                    detail: "combined QSA projection row count overflows".to_string(),
                })?;
        out.push(ResidentEntrySpec::Raw(RawTensorSpec {
            name: format!(
                "language_model.model.layers.{layer}.self_attn.indexer.index_qk_proj.weight"
            ),
            dtype: crate::resident_writer::DTYPE_BF16,
            bytes,
            shape: logical_shape(&[hidden, combined_rows]),
        }));
    }
    Ok((out, lossy))
}

fn qwen4_indexer_projection_part(name: &str) -> Option<(usize, bool)> {
    let rest = name.strip_prefix("blk.")?;
    let (raw_layer, suffix) = rest.split_once('.')?;
    let layer = raw_layer.parse().ok()?;
    match suffix {
        "indexer.q_proj.weight" => Some((layer, true)),
        "indexer.k_proj.weight" => Some((layer, false)),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use crate::gguf_header::{GgufHeader, GgufTensorInfo};
    use crate::ranged_download::DownloadError;

    struct MemorySource(Vec<u8>);

    impl RangeSource for MemorySource {
        fn read_range(&self, start: u64, end: u64) -> Result<Vec<u8>, DownloadError> {
            let start = usize::try_from(start).map_err(|_| DownloadError::InvalidRange {
                start,
                end_exclusive: end,
            })?;
            let end = usize::try_from(end).map_err(|_| DownloadError::InvalidRange {
                start: start as u64,
                end_exclusive: end,
            })?;
            self.0
                .get(start..end)
                .map(ToOwned::to_owned)
                .ok_or(DownloadError::ShortRead {
                    expected: (end - start) as u64,
                    actual: self.0.len().saturating_sub(start) as u64,
                })
        }
    }

    #[test]
    fn qwen4_indexer_parts_merge_and_quantized_v_heads_deinterleave() {
        let mut tensors = BTreeMap::new();
        tensors.insert(
            "blk.0.indexer.q_proj.weight".into(),
            GgufTensorInfo {
                ggml_type: 30,
                dims: vec![4, 4],
                offset: 0,
            },
        );
        tensors.insert(
            "blk.0.indexer.k_proj.weight".into(),
            GgufTensorInfo {
                ggml_type: 30,
                dims: vec![4, 2],
                offset: 32,
            },
        );
        tensors.insert(
            "blk.0.ssm_out.weight".into(),
            GgufTensorInfo {
                ggml_type: 23,
                dims: vec![6144, 1],
                offset: 48,
            },
        );
        let v_head_types = [
            (1, GGML_TYPE_Q5_K, compute::quant_gguf::Q5_K_BLOCK_BYTES),
            (2, GGML_TYPE_Q3_K, compute::quant_gguf::Q3_K_BLOCK_BYTES),
            (3, GGML_TYPE_Q4_K, compute::quant_gguf::Q4_K_BLOCK_BYTES),
            (4, GGML_TYPE_Q6_K, compute::quant_gguf::Q6_K_BLOCK_BYTES),
            (
                5,
                GGML_TYPE_IQ3_S,
                compute::quant_gguf_iq::IQ3_S_BLOCK_BYTES,
            ),
        ];
        let mut offset = (48 + 24 * compute::quant_gguf_iq::IQ4_XS_BLOCK_BYTES) as u64;
        for (layer, ggml_type, block_bytes) in v_head_types {
            tensors.insert(
                format!("blk.{layer}.ssm_out.weight"),
                GgufTensorInfo {
                    ggml_type,
                    dims: vec![6144, 1],
                    offset,
                },
            );
            offset += (24 * block_bytes) as u64;
        }
        let header = GgufHeader {
            version: 3,
            metadata: BTreeMap::new(),
            tensors,
            alignment: 32,
            data_region_start: 0,
        };
        let mut source_bytes = (0..48).collect::<Vec<u8>>();
        let mut quantized = vec![0u8; 24 * compute::quant_gguf_iq::IQ4_XS_BLOCK_BYTES];
        for (block_index, block) in quantized
            .chunks_exact_mut(compute::quant_gguf_iq::IQ4_XS_BLOCK_BYTES)
            .enumerate()
        {
            block[0..2].copy_from_slice(&0x3c00u16.to_le_bytes()); // d = 1.0
            block[4..8].fill(0x88); // nonzero sub-block scales
            for (byte_index, byte) in block[8..].iter_mut().enumerate() {
                *byte = (byte_index as u8).wrapping_add(block_index as u8 * 7);
            }
        }
        source_bytes.extend_from_slice(&quantized);
        for (_, _, block_bytes) in v_head_types {
            source_bytes.extend_from_slice(&vec![0u8; 24 * block_bytes]);
        }
        let source = MemorySource(source_bytes);
        let mut arch = model_io::known_architecture(ModelFamily::Qwen4Exp);
        arch.hidden_size = 4;
        arch.full_attention_layer_mask = vec![1; 6];
        arch.compressed_attention.index_n_heads = 2;
        arch.compressed_attention.index_head_dim = 2;
        arch.linear_attention.num_k_heads = 16;
        arch.linear_attention.key_head_dim = 128;
        arch.linear_attention.num_v_heads = 48;
        arch.linear_attention.value_head_dim = 128;

        let (entries, lossy) = resident_entries(
            &header,
            &source,
            &arch,
            &[
                "blk.0.indexer.q_proj.weight",
                "blk.0.indexer.k_proj.weight",
                "blk.0.ssm_out.weight",
                "blk.1.ssm_out.weight",
                "blk.2.ssm_out.weight",
                "blk.3.ssm_out.weight",
                "blk.4.ssm_out.weight",
                "blk.5.ssm_out.weight",
            ],
        )
        .expect("transcode QSA and V-head tensors");
        assert_eq!(lossy.len(), 1, "BF16 narrowing is recorded for this tensor");
        assert_eq!(lossy[0].0, "blk.0.ssm_out.weight");
        assert_eq!(entries.len(), 7);

        let indexer = entries
            .iter()
            .find_map(|entry| match entry {
                ResidentEntrySpec::Raw(spec)
                    if spec.name.ends_with("indexer.index_qk_proj.weight") =>
                {
                    Some(spec)
                }
                _ => None,
            })
            .expect("fused indexer projection");
        assert_eq!(indexer.shape, (6, 4, 0, 0));
        assert_eq!(indexer.dtype, crate::resident_writer::DTYPE_BF16);
        assert_eq!(indexer.bytes, (0..48).collect::<Vec<u8>>());

        let output = entries
            .iter()
            .find_map(|entry| match entry {
                ResidentEntrySpec::Raw(spec)
                    if spec.name.ends_with("linear_attn.out_proj.weight") =>
                {
                    Some(spec)
                }
                _ => None,
            })
            .expect("deinterleaved output projection");
        assert_eq!(output.shape, (1, 6144, 0, 0));
        assert_eq!(output.dtype, crate::resident_writer::DTYPE_BF16);
        assert_eq!(output.bytes.len(), 6144 * 2);
        let mut expected = compute::quant_gguf_iq::dequantize_iq4_xs(&quantized, 6144);
        let original = expected.clone();
        let num_k_heads = arch.linear_attention.num_k_heads as usize;
        let values_per_k_head = arch.linear_attention.num_v_heads as usize / num_k_heads;
        for key_head in 0..num_k_heads {
            for value_head in 0..values_per_k_head {
                let grouped_head = key_head * values_per_k_head + value_head;
                let tiled_head = value_head * num_k_heads + key_head;
                expected[grouped_head * 128..(grouped_head + 1) * 128]
                    .copy_from_slice(&original[tiled_head * 128..(tiled_head + 1) * 128]);
            }
        }
        for (bytes, value) in output.bytes.chunks_exact(2).zip(expected) {
            assert_eq!(
                u16::from_le_bytes([bytes[0], bytes[1]]),
                compute::f32_to_bf16(value),
                "dequantized and deinterleaved output element"
            );
        }

        for layer in 1..=5 {
            let output = entries
                .iter()
                .find_map(|entry| match entry {
                    ResidentEntrySpec::Raw(spec)
                        if spec
                            .name
                            .ends_with(&format!("layers.{layer}.linear_attn.out_proj.weight")) =>
                    {
                        Some(spec)
                    }
                    _ => None,
                })
                .expect("K-quant and IQ V-head output projections are dequantized");
            assert_eq!(output.shape, (1, 6144, 0, 0));
            assert_eq!(output.dtype, crate::resident_writer::DTYPE_BF16);
            assert!(output
                .bytes
                .chunks_exact(2)
                .all(|bytes| matches!(u16::from_le_bytes([bytes[0], bytes[1]]), 0 | 0x8000)));
        }
    }

    #[test]
    fn qwen4exp_resident_types_convert_or_remain_block_quantized_as_required() {
        let mut q2_0 = vec![0u8; compute::quant_gguf::Q2_0_BLOCK_BYTES];
        q2_0[..2].copy_from_slice(&0x3800u16.to_le_bytes());
        let mut q4_0 = vec![0u8; compute::quant_gguf::Q4_0_BLOCK_BYTES];
        q4_0[..2].copy_from_slice(&0x3800u16.to_le_bytes());
        let mut q5_0 = vec![0u8; compute::quant_gguf::Q5_0_BLOCK_BYTES];
        q5_0[..2].copy_from_slice(&0x3800u16.to_le_bytes());

        let mut source_bytes = q2_0.clone();
        source_bytes.extend_from_slice(&q5_0);
        source_bytes.extend_from_slice(&q4_0);
        source_bytes.extend_from_slice(&0x3c00u16.to_le_bytes()); // F16 1.0
        source_bytes.extend_from_slice(&0x3800u16.to_le_bytes()); // F16 0.5
        let q5_k = vec![0u8; 5_120 * compute::quant_gguf::Q5_K_BLOCK_BYTES];
        source_bytes.extend_from_slice(&q5_k);
        let source = MemorySource(source_bytes);
        let mut tensors = BTreeMap::new();
        tensors.insert(
            "blk.1.ffn_down_shexp.weight".into(),
            GgufTensorInfo {
                ggml_type: GGML_TYPE_Q2_0,
                dims: vec![64, 1],
                offset: 0,
            },
        );
        tensors.insert(
            "blk.0.ffn_down_shexp.weight".into(),
            GgufTensorInfo {
                ggml_type: GGML_TYPE_Q5_0,
                dims: vec![32, 1],
                offset: q2_0.len() as u64,
            },
        );
        tensors.insert(
            "blk.2.ffn_down_shexp.weight".into(),
            GgufTensorInfo {
                ggml_type: GGML_TYPE_Q4_0,
                dims: vec![32, 1],
                offset: (q2_0.len() + q5_0.len()) as u64,
            },
        );
        tensors.insert(
            "blk.1.ple_conv1d.weight".into(),
            GgufTensorInfo {
                ggml_type: GGML_TYPE_F16,
                dims: vec![2, 1],
                offset: (q2_0.len() + q5_0.len() + q4_0.len()) as u64,
            },
        );
        tensors.insert(
            "blk.3.attn_v.weight".into(),
            GgufTensorInfo {
                ggml_type: GGML_TYPE_Q5_K,
                dims: vec![2_560, 512],
                offset: (q2_0.len() + q5_0.len() + q4_0.len() + 4) as u64,
            },
        );
        let header = GgufHeader {
            version: 3,
            metadata: BTreeMap::new(),
            tensors,
            alignment: 32,
            data_region_start: 0,
        };
        let arch = model_io::known_architecture(ModelFamily::Qwen4Exp);
        let (entries, lossy) = resident_entries(
            &header,
            &source,
            &arch,
            &[
                "blk.0.ffn_down_shexp.weight",
                "blk.1.ffn_down_shexp.weight",
                "blk.2.ffn_down_shexp.weight",
                "blk.1.ple_conv1d.weight",
                "blk.3.attn_v.weight",
            ],
        )
        .expect("convert the Qwen4Exp resident-only block types");
        assert_eq!(entries.len(), 5);
        assert!(lossy.is_empty(), "fixture values are exact in BF16");
        for entry in entries {
            let ResidentEntrySpec::Raw(spec) = entry else {
                panic!("resident tensor should become raw BF16")
            };
            if spec.name.ends_with("layers.3.self_attn.v_proj.weight") {
                assert_eq!(spec.dtype, crate::resident_writer::DTYPE_GGUF_Q5_K);
                assert_eq!(spec.bytes, q5_k);
                assert_eq!(spec.shape, (512, 2_560, 0, 0));
                continue;
            }
            assert_eq!(spec.dtype, crate::resident_writer::DTYPE_BF16);
            let elements: usize = if spec.name.ends_with("ple.conv1d.weight") {
                2
            } else if spec
                .name
                .contains("layers.1.mlp.shared_expert.down_proj.weight")
            {
                64
            } else {
                32
            };
            assert_eq!((spec.shape.0 * spec.shape.1) as usize, elements);
            assert_eq!(spec.bytes.len(), elements * 2);
        }
    }
}
