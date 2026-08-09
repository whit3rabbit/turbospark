//! Resident set transcoding (F32 to BF16/INT8), V-head interleaving conventions, and raw tensor specs.

use model_io::{ArchConfig, ModelFamily};

use super::types::{
    dtype_tag_for_ggml_type, logical_shape, read_tensor, GgufRepackError, GGML_TYPE_F32,
};
use crate::gguf_header::GgufHeader;
use crate::gguf_names::{map_gguf_name, GgufMapping};
use crate::ranged_download::RangeSource;
use crate::resident_writer::{RawTensorSpec, ResidentEntrySpec, ResidentTensorSpec};

fn int8_transcode_targets(family: ModelFamily) -> &'static [&'static str] {
    match family {
        ModelFamily::Gemma4 => &["router.proj.weight"],
        ModelFamily::Qwen36 => &["mlp.gate.weight", "mlp.shared_expert_gate.weight"],
        ModelFamily::DeepseekV4Flash => &[],
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

struct VHeadAxis {
    columns: bool,
    base: usize,
    span: usize,
}

fn v_head_axis(canonical: &str, arch: &ArchConfig) -> Option<VHeadAxis> {
    if arch.family != ModelFamily::Qwen36 {
        return None;
    }
    let la = &arch.linear_attention;
    let rows = |base: usize, span: usize| {
        Some(VHeadAxis {
            columns: false,
            base,
            span,
        })
    };
    let v_at = 2 * la.num_k_heads as usize * la.key_head_dim as usize;
    let width = la.value_head_dim as usize;
    match canonical.rsplit_once("linear_attn.")?.1 {
        "A_log" | "dt_bias" | "in_proj_a.weight" | "in_proj_b.weight" => rows(0, 1),
        "conv1d.weight" | "in_proj_qkv.weight" => rows(v_at, width),
        "in_proj_z.weight" => rows(0, width),
        "out_proj.weight" => Some(VHeadAxis {
            columns: true,
            base: 0,
            span: width,
        }),
        _ => None,
    }
}

fn permute_v_heads<T: Copy>(data: &mut [T], base: usize, span: usize, heads: usize) {
    let source = data.to_owned();
    for h in 0..heads {
        let to = if h < heads / 2 {
            2 * h
        } else {
            2 * (h - heads / 2) + 1
        };
        data[base + to * span..base + (to + 1) * span]
            .copy_from_slice(&source[base + h * span..base + (h + 1) * span]);
    }
}

fn even_v_heads(name: &str, arch: &ArchConfig) -> Result<usize, GgufRepackError> {
    let heads = arch.linear_attention.num_v_heads as usize;
    if heads < 2 || heads % 2 != 0 {
        return Err(GgufRepackError::ShapeMismatch {
            tensor: name.to_string(),
            detail: format!("the V-head de-interleave needs an even num_v_heads, got {heads}"),
        });
    }
    Ok(heads)
}

fn apply_source_convention(
    name: &str,
    canonical: &str,
    arch: &ArchConfig,
    cols: usize,
    values: &mut [f32],
) -> Result<(), GgufRepackError> {
    let Some(axis) = v_head_axis(canonical, arch) else {
        return Ok(());
    };
    let heads = even_v_heads(name, arch)?;
    let shape_err = |detail: String| GgufRepackError::ShapeMismatch {
        tensor: name.to_string(),
        detail,
    };
    if axis.columns {
        return Err(shape_err(
            "a column-axis V-head tensor is not expected to arrive as F32".to_string(),
        ));
    }
    let (base, span) = (axis.base * cols, axis.span * cols);
    if base + heads * span != values.len() {
        return Err(shape_err(format!(
            "{} values do not fill {base} + {heads} x {span}",
            values.len()
        )));
    }

    if canonical.ends_with("linear_attn.A_log") {
        for v in values.iter_mut() {
            if !v.is_finite() || *v >= 0.0 {
                return Err(shape_err(format!(
                    "ssm_a holds -exp(A_log) and must be finite and negative, found {v}"
                )));
            }
            *v = (-*v).ln();
        }
    }
    permute_v_heads(values, base, span, heads);
    Ok(())
}

fn apply_source_convention_bytes(
    name: &str,
    canonical: &str,
    arch: &ArchConfig,
    (rows, cols): (usize, usize),
    bytes: &mut [u8],
) -> Result<(), GgufRepackError> {
    let Some(axis) = v_head_axis(canonical, arch) else {
        return Ok(());
    };
    let heads = even_v_heads(name, arch)?;
    let shape_err = |detail: String| GgufRepackError::ShapeMismatch {
        tensor: name.to_string(),
        detail,
    };
    let along = if axis.columns { cols } else { rows };
    if axis.base + heads * axis.span != along {
        return Err(shape_err(format!(
            "a {rows}x{cols} tensor's V axis does not fill {} + {heads} x {}",
            axis.base, axis.span
        )));
    }
    if rows == 0 || bytes.len() % rows != 0 {
        return Err(shape_err(format!(
            "{} bytes is not a whole number of {rows} rows",
            bytes.len()
        )));
    }
    let row_bytes = bytes.len() / rows;

    if !axis.columns {
        permute_v_heads(bytes, axis.base * row_bytes, axis.span * row_bytes, heads);
        return Ok(());
    }
    if row_bytes * axis.span % cols != 0 {
        return Err(shape_err(format!(
            "a {row_bytes}-byte row of {cols} columns has no whole-byte \
             {}-column V head: the block is wider than one head",
            axis.span
        )));
    }
    let head_bytes = row_bytes * axis.span / cols;
    let base_bytes = row_bytes * axis.base / cols;
    for r in 0..rows {
        permute_v_heads(
            &mut bytes[r * row_bytes..(r + 1) * row_bytes],
            base_bytes,
            head_bytes,
            heads,
        );
    }
    Ok(())
}

pub struct Transcoded {
    pub spec: ResidentEntrySpec,
    pub lossy: usize,
}

fn f32_values(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

pub fn transcode_f32(
    name: &str,
    canonical: String,
    bytes: &[u8],
    dims: &[u64],
    arch: &ArchConfig,
) -> Result<Transcoded, GgufRepackError> {
    let family = arch.family;
    let mut values = f32_values(bytes);
    if values.len() * 4 != bytes.len() {
        return Err(GgufRepackError::ShapeMismatch {
            tensor: name.to_string(),
            detail: format!("{} bytes is not a whole number of F32 values", bytes.len()),
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
    let mut packed = Vec::with_capacity(rows * cols);
    let mut scales = Vec::with_capacity(rows * cols / 64);
    let mut biases = Vec::with_capacity(rows * cols / 64);
    for r in 0..rows {
        let row = compute::quantize_int8_affine(&values[r * cols..(r + 1) * cols]);
        packed.extend_from_slice(&row.packed);
        scales.extend_from_slice(&row.scales);
        biases.extend_from_slice(&row.biases);
    }
    Ok(Transcoded {
        spec: ResidentEntrySpec::Int8(ResidentTensorSpec {
            name: canonical,
            packed,
            scales,
            biases,
            rows: rows as u32,
            cols: cols as u32,
        }),
        lossy: 0,
    })
}

pub type ResidentSet = (Vec<ResidentEntrySpec>, Vec<(String, usize)>);

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
        apply_source_convention_bytes(name, &canonical, arch, row_and_col(&info.dims), &mut bytes)?;
        out.push(ResidentEntrySpec::Raw(RawTensorSpec {
            name: canonical,
            dtype,
            bytes,
            shape: logical_shape(&info.dims),
        }));
    }
    Ok((out, lossy))
}
