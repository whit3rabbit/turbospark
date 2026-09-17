//! Resident set transcoding (F32 to BF16/INT8), V-head interleaving conventions, and raw tensor specs.

use model_io::{ArchConfig, ModelFamily};

use super::types::{
    dtype_tag_for_ggml_type, logical_shape, read_tensor, GgufRepackError, GGML_TYPE_F32,
};
use crate::gguf_header::GgufHeader;
use crate::gguf_names::{map_gguf_name, GgufMapping};
use crate::ranged_download::RangeSource;
use crate::resident_writer::{RawTensorSpec, ResidentEntrySpec};

fn int8_transcode_targets(family: ModelFamily) -> &'static [&'static str] {
    match family {
        ModelFamily::Gemma4 => &["router.proj.weight"],
        ModelFamily::QwenGdnMoe => &["mlp.gate.weight", "mlp.shared_expert_gate.weight"],
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
        ModelFamily::DeepseekV4Flash
        | ModelFamily::QwenGdnDense
        | ModelFamily::MuseGlimmer
        | ModelFamily::Qwen4Exp
        | ModelFamily::Spark25
        | ModelFamily::Qwen3Dense
        | ModelFamily::Qwen2Dense
        | ModelFamily::MiniMaxM2 => &[],
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
        let mut lossy = 0usize;
        let mut out = Vec::with_capacity(values.len() * 2);
        for &v in &values {
            let bits = compute::f32_to_bf16(v);
            if compute::bf16_to_f32(bits) != v {
                lossy += 1;
            }
            out.extend_from_slice(&bits.to_le_bytes());
        }
        return Ok(Transcoded {
            spec: ResidentEntrySpec::Raw(RawTensorSpec {
                name: canonical,
                dtype: crate::resident_writer::DTYPE_BF16,
                bytes: out,
                shape: logical_shape(dims),
            }),
            lossy,
        });
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
    for name in names {
        let info = &header.tensors[*name];
        let dtype = dtype_tag_for_ggml_type(info.ggml_type).ok_or_else(|| {
            GgufRepackError::UnsupportedType {
                tensor: (*name).to_string(),
                ggml_type: info.ggml_type,
            }
        })?;
        let GgufMapping::Resident(canonical) = map_gguf_name(name, family)? else {
            continue;
        };
        let bytes = read_tensor(header, source, name)?;
        if info.ggml_type == GGML_TYPE_F32 {
            let t = transcode_f32(name, canonical, &bytes, &info.dims, arch)?;
            if t.lossy > 0 {
                lossy.push(((*name).to_string(), t.lossy));
            }
            out.push(t.spec);
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
    Ok((out, lossy))
}
