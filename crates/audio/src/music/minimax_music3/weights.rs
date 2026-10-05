//! Weight access shared by the converted-tree and official-tree loaders.
//!
//! Two backends feed one interface: safetensors files (converted MLX
//! trees, which may carry MLX groupwise-quantized linears) and an
//! in-memory map (the official modular tree after key sanitizing and
//! layout remap, which is never quantized). Both are strict: every
//! tensor in the source must be consumed by exactly one model field,
//! otherwise `finish` fails, mirroring `load_model(strict=True)`.

use super::backend::{ComputeBackend, Weight, WeightData, WeightEncoding};
use super::precision::{DType, Music3Precision};
use std::collections::HashMap;
use std::path::Path;
use std::rc::Rc;

use turbospark_model_io::safetensors::SafetensorsFile;

use crate::quant::{self, QuantScheme};
use crate::Result;
use crate::SpeechError;

#[derive(Clone)]
pub(crate) struct Tensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
    pub dtype: DType,
}

impl Tensor {
    pub(crate) fn promote(&self, input: DType) -> DType {
        if self.is_empty() {
            input
        } else {
            input.promote(self.dtype)
        }
    }
}
impl std::ops::Deref for Tensor {
    type Target = [f32];
    fn deref(&self) -> &[f32] {
        &self.data
    }
}
impl Default for Tensor {
    fn default() -> Self {
        Self {
            data: Vec::new(),
            shape: Vec::new(),
            dtype: DType::F32,
        }
    }
}

/// Weight encoding selected by `config.json::quantization`.
#[derive(Debug, Clone, Copy)]
pub(crate) enum LinearQuantization {
    Dense,
    Affine(QuantScheme),
    MxFp4,
    MxFp8,
    NvFp4,
}

impl LinearQuantization {
    fn float_grouped(self) -> Option<(u32, usize, FloatQuantKind)> {
        match self {
            Self::MxFp4 => Some((4, 32, FloatQuantKind::MxFp4)),
            Self::MxFp8 => Some((8, 32, FloatQuantKind::MxFp8)),
            Self::NvFp4 => Some((4, 16, FloatQuantKind::NvFp4)),
            Self::Dense | Self::Affine(_) => None,
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum FloatQuantKind {
    MxFp4,
    MxFp8,
    NvFp4,
}

pub(crate) enum WeightStore {
    File {
        files: Vec<SafetensorsFile>,
        scheme: LinearQuantization,
        precision: Music3Precision,
        backend: Option<Rc<dyn ComputeBackend>>,
        consumed: std::collections::HashSet<String>,
    },
    Map {
        precision: Music3Precision,
        tensors: HashMap<String, Tensor>,
    },
}

fn missing(name: &str) -> SpeechError {
    SpeechError::Tensor {
        name: name.to_string(),
        why: "missing".to_string(),
    }
}

impl WeightStore {
    pub(crate) fn from_files(
        files: Vec<SafetensorsFile>,
        scheme: LinearQuantization,
    ) -> WeightStore {
        WeightStore::File {
            files,
            scheme,
            precision: Music3Precision::Float32,
            backend: None,
            consumed: std::collections::HashSet::new(),
        }
    }

    pub(crate) fn from_files_with_backend(
        files: Vec<SafetensorsFile>,
        scheme: LinearQuantization,
        backend: Rc<dyn ComputeBackend>,
    ) -> Self {
        Self::File {
            files,
            scheme,
            precision: Music3Precision::Float32,
            backend: Some(backend),
            consumed: Default::default(),
        }
    }

    pub(crate) fn with_precision(mut self, precision: Music3Precision) -> Self {
        match &mut self {
            Self::File { precision: p, .. } | Self::Map { precision: p, .. } => *p = precision,
        }
        self
    }
    pub(crate) fn precision(&self) -> Music3Precision {
        match self {
            Self::File { precision, .. } | Self::Map { precision, .. } => *precision,
        }
    }

    pub(crate) fn backend(&self) -> Option<Rc<dyn ComputeBackend>> {
        match self {
            Self::File { backend, .. } => backend.clone(),
            Self::Map { .. } => None,
        }
    }

    pub(crate) fn dense_weight(
        &self,
        data: Vec<f32>,
        shape: &[usize],
        dtype: DType,
    ) -> Result<Weight> {
        match self.backend() {
            Some(backend) => {
                let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
                let elements = data.len();
                Ok(Weight::Device {
                    elements,
                    dtype,
                    packed: false,
                    dynamic: false,
                    weight: backend.load_weight(WeightData {
                        name: "convolution",
                        shape,
                        bytes: &bytes,
                        encoding: WeightEncoding::F32,
                        dtype,
                        scales_dtype: None,
                        offsets_dtype: None,
                        scales: &[],
                        offsets: &[],
                        block_scales: &[],
                    })?,
                })
            }
            None => Ok(Weight::cpu_typed(data, dtype, false, false)),
        }
    }

    pub(crate) fn embedding(&mut self, name: &str) -> Result<Weight> {
        if let Self::File {
            files,
            consumed,
            backend: Some(backend),
            precision,
            ..
        } = self
        {
            let file = files
                .iter()
                .find(|f| f.contains_tensor(name))
                .ok_or_else(|| missing(name))?;
            let desc = file.descriptor(name).ok_or_else(|| missing(name))?;
            if desc.shape.len() != 2 || desc.shape.contains(&0) {
                return Err(missing(name));
            }
            let encoding = dense_encoding(&desc.dtype, name)?;
            let dtype = precision.dtype(DType::from_checkpoint(&desc.dtype)?);
            let elements = desc.shape.iter().product();
            let weight = backend.load_weight(WeightData {
                name,
                shape: &desc.shape,
                bytes: file.raw_bytes(name)?,
                encoding,
                dtype,
                scales_dtype: None,
                offsets_dtype: None,
                scales: &[],
                offsets: &[],
                block_scales: &[],
            })?;
            consumed.insert(name.into());
            return Ok(Weight::Device {
                elements,
                dtype,
                packed: false,
                dynamic: false,
                weight,
            });
        }
        let tensor = self.tensor(name)?;
        Ok(Weight::cpu_typed(tensor.data, tensor.dtype, false, false))
    }

    pub(crate) fn from_map(tensors: HashMap<String, Tensor>) -> WeightStore {
        WeightStore::Map {
            tensors,
            precision: Music3Precision::Float32,
        }
    }

    pub(crate) fn has(&self, name: &str) -> bool {
        match self {
            WeightStore::File { files, .. } => files.iter().any(|f| f.contains_tensor(name)),
            WeightStore::Map { tensors, .. } => tensors.contains_key(name),
        }
    }

    /// Take a plain (non-linear) tensor by exact name.
    pub(crate) fn tensor(&mut self, name: &str) -> Result<Tensor> {
        match self {
            WeightStore::File {
                files,
                consumed,
                precision,
                ..
            } => {
                let file = files.iter().find(|f| f.contains_tensor(name));
                let file = file.ok_or_else(|| missing(name))?;
                let shape = file
                    .descriptor(name)
                    .map(|d| d.shape.clone())
                    .ok_or_else(|| missing(name))?;
                let dtype = precision.dtype(DType::from_checkpoint(
                    &file.descriptor(name).ok_or_else(|| missing(name))?.dtype,
                )?);
                let data = file.load_as_f32(name)?;
                consumed.insert(name.to_string());
                Ok(Tensor { data, shape, dtype })
            }
            WeightStore::Map { tensors, precision } => {
                let mut tensor = tensors.remove(name).ok_or_else(|| missing(name))?;
                tensor.dtype = precision.dtype(tensor.dtype);
                Ok(tensor)
            }
        }
    }

    /// Load one linear as `(weight [out, in], bias)`; quantized
    /// linears dequantize through the shared affine kernel.
    pub(crate) fn linear(&mut self, base: &str) -> Result<(Weight, Option<Tensor>)> {
        match self {
            WeightStore::File {
                files,
                scheme,
                backend,
                consumed,
                precision,
            } => {
                let weight_name = format!("{base}.weight");
                let file = files
                    .iter()
                    .find(|f| f.contains_tensor(&weight_name))
                    .ok_or_else(|| missing(&weight_name))?;
                let loaded = if let Some(backend) = backend {
                    device_linear(file, base, *scheme, backend.as_ref(), *precision)?
                } else if file.contains_tensor(&format!("{base}.scales")) {
                    if let Some((bits, group_size, kind)) = scheme.float_grouped() {
                        let (w, b) = load_float_quantized(file, base, bits, group_size, kind)?;
                        (Weight::cpu_typed(w, DType::F32, true, true), b)
                    } else if let LinearQuantization::Affine(affine) = scheme {
                        let (w, b) = quant::load_quantized(file, base, *affine)?;
                        let scales_dtype = precision.dtype(DType::from_checkpoint(
                            &file
                                .descriptor(&format!("{base}.scales"))
                                .ok_or_else(|| missing(base))?
                                .dtype,
                        )?);
                        let offsets_dtype = precision.dtype(DType::from_checkpoint(
                            &file
                                .descriptor(&format!("{base}.biases"))
                                .ok_or_else(|| missing(base))?
                                .dtype,
                        )?);
                        (
                            Weight::cpu_typed(w, scales_dtype.promote(offsets_dtype), true, false),
                            b,
                        )
                    } else {
                        return Err(SpeechError::Unsupported {
                            why: format!(
                                "{weight_name} is packed, but config.json does not declare a supported quantization mode"
                            ),
                        });
                    }
                } else {
                    let weight = file.load_as_f32(&weight_name)?;
                    let bias = if file.contains_tensor(&format!("{base}.bias")) {
                        Some(file.load_as_f32(&format!("{base}.bias"))?)
                    } else {
                        None
                    };
                    (
                        Weight::cpu_typed(
                            weight,
                            precision.dtype(DType::from_checkpoint(
                                &file
                                    .descriptor(&weight_name)
                                    .ok_or_else(|| missing(&weight_name))?
                                    .dtype,
                            )?),
                            false,
                            false,
                        ),
                        bias,
                    )
                };
                consumed.insert(weight_name);
                if file.contains_tensor(&format!("{base}.bias")) {
                    consumed.insert(format!("{base}.bias"));
                }
                if file.contains_tensor(&format!("{base}.scales")) {
                    consumed.insert(format!("{base}.scales"));
                    if file.contains_tensor(&format!("{base}.biases")) {
                        consumed.insert(format!("{base}.biases"));
                    }
                }
                let (weight, bias) = loaded;
                let bias = bias
                    .map(|data| -> Result<Tensor> {
                        let name = format!("{base}.bias");
                        Ok(Tensor {
                            shape: vec![data.len()],
                            data,
                            dtype: precision.dtype(DType::from_checkpoint(
                                &file.descriptor(&name).ok_or_else(|| missing(&name))?.dtype,
                            )?),
                        })
                    })
                    .transpose()?;
                Ok((weight, bias))
            }
            WeightStore::Map { tensors, precision } => {
                if tensors.contains_key(&format!("{base}.scales")) {
                    return Err(SpeechError::Unsupported {
                        why: "quantized weights are not expected in an official tree".to_string(),
                    });
                }
                let weight = tensors
                    .remove(&format!("{base}.weight"))
                    .ok_or_else(|| missing(&format!("{base}.weight")))?;
                let bias = tensors.remove(&format!("{base}.bias")).map(|mut tensor| {
                    tensor.dtype = precision.dtype(tensor.dtype);
                    tensor
                });
                Ok((
                    Weight::cpu_typed(weight.data, precision.dtype(weight.dtype), false, false),
                    bias,
                ))
            }
        }
    }

    /// Fail when any source tensor was not consumed.
    pub(crate) fn finish(self) -> Result<()> {
        match self {
            WeightStore::File {
                files, consumed, ..
            } => {
                let extras: Vec<String> = files
                    .iter()
                    .flat_map(|file| file.tensor_names().map(str::to_string))
                    .filter(|name| !consumed.contains(name))
                    .collect();
                if extras.is_empty() {
                    Ok(())
                } else {
                    Err(SpeechError::Tensor {
                        name: extras[0].clone(),
                        why: format!("unconsumed tensors: {}", extras.join(", ")),
                    })
                }
            }
            WeightStore::Map { tensors, .. } => {
                if tensors.is_empty() {
                    Ok(())
                } else {
                    let mut names: Vec<&String> = tensors.keys().collect();
                    names.sort();
                    Err(SpeechError::Tensor {
                        name: names[0].clone(),
                        why: format!(
                            "unconsumed tensors: {}",
                            names
                                .iter()
                                .map(|n| n.as_str())
                                .collect::<Vec<_>>()
                                .join(", ")
                        ),
                    })
                }
            }
        }
    }
}

fn dense_encoding(dtype: &str, name: &str) -> Result<WeightEncoding> {
    match dtype {
        "F32" => Ok(WeightEncoding::F32),
        "F16" => Ok(WeightEncoding::F16),
        "BF16" => Ok(WeightEncoding::Bf16),
        _ => Err(SpeechError::Tensor {
            name: name.into(),
            why: format!("unsupported dense dtype {dtype}"),
        }),
    }
}

fn device_linear(
    file: &SafetensorsFile,
    base: &str,
    scheme: LinearQuantization,
    backend: &dyn ComputeBackend,
    precision: Music3Precision,
) -> Result<(Weight, Option<Vec<f32>>)> {
    let name = format!("{base}.weight");
    let desc = file.descriptor(&name).ok_or_else(|| missing(&name))?;
    if desc.shape.len() != 2 || desc.shape.contains(&0) {
        return Err(SpeechError::Tensor {
            name,
            why: "expected a non-empty matrix".into(),
        });
    }
    let mut dtype = DType::F32;
    let mut scales_dtype = None;
    let mut offsets_dtype = None;
    let mut dynamic = false;
    let mut packed = false;
    let mut shape = desc.shape.clone();
    let mut scales = Vec::new();
    let mut offsets = Vec::new();
    let mut block_scales = &[][..];
    let encoding = if file.contains_tensor(&format!("{base}.scales")) {
        packed = true;
        let scale_name = format!("{base}.scales");
        let scale_desc = file
            .descriptor(&scale_name)
            .ok_or_else(|| missing(&scale_name))?;
        if desc.dtype != "U32"
            || scale_desc.shape.len() != 2
            || scale_desc.shape[0] != shape[0]
            || scale_desc.shape[1] == 0
        {
            return Err(SpeechError::Tensor {
                name,
                why: "packed matrix/scale geometry mismatch".into(),
            });
        }
        let (bits, group, encoding) = match scheme {
            LinearQuantization::Affine(q) => {
                if !matches!(q.bits, 2 | 3 | 4 | 5 | 6 | 8) || q.group_size == 0 {
                    return Err(SpeechError::Unsupported {
                        why: "invalid affine encoding".into(),
                    });
                }
                dtype = precision.dtype(DType::from_checkpoint(&scale_desc.dtype)?);
                scales_dtype = Some(dtype);
                scales = file.load_as_f32(&scale_name)?;
                let offset_name = format!("{base}.biases");
                let offset_desc = file
                    .descriptor(&offset_name)
                    .ok_or_else(|| missing(&offset_name))?;
                if offset_desc.shape != scale_desc.shape {
                    return Err(missing(&offset_name));
                }
                dtype = dtype.promote(precision.dtype(DType::from_checkpoint(&offset_desc.dtype)?));
                offsets_dtype = Some(precision.dtype(DType::from_checkpoint(&offset_desc.dtype)?));
                offsets = file.load_as_f32(&offset_name)?;
                (
                    q.bits,
                    q.group_size,
                    WeightEncoding::Affine {
                        bits: q.bits,
                        group_size: q.group_size,
                    },
                )
            }
            LinearQuantization::MxFp4 | LinearQuantization::MxFp8 | LinearQuantization::NvFp4 => {
                if scale_desc.dtype != "U8" || file.contains_tensor(&format!("{base}.biases")) {
                    return Err(SpeechError::Tensor {
                        name: scale_name,
                        why: "float packed scales require U8 without affine biases".into(),
                    });
                }
                dynamic = true;
                block_scales = file.raw_bytes(&scale_name)?;
                match scheme {
                    LinearQuantization::MxFp4 => (4, 32, WeightEncoding::MxFp4),
                    LinearQuantization::MxFp8 => (8, 32, WeightEncoding::MxFp8),
                    _ => (4, 16, WeightEncoding::NvFp4),
                }
            }
            _ => {
                return Err(SpeechError::Unsupported {
                    why: "packed matrix without quantization declaration".into(),
                })
            }
        };
        shape[1] = scale_desc.shape[1]
            .checked_mul(group)
            .ok_or_else(|| missing(&name))?;
        let row_bits = shape[1]
            .checked_mul(bits as usize)
            .ok_or_else(|| SpeechError::Tensor {
                name: name.clone(),
                why: "packed row width overflows".into(),
            })?;
        if desc.shape[1] != row_bits.div_ceil(32) {
            return Err(SpeechError::Tensor {
                name,
                why: "packed input width does not match scales".into(),
            });
        }
        encoding
    } else {
        dtype = precision.dtype(DType::from_checkpoint(&desc.dtype)?);
        dense_encoding(&desc.dtype, &name)?
    };
    let bias_name = format!("{base}.bias");
    let bias = if file.contains_tensor(&bias_name) {
        let b = file.load_as_f32(&bias_name)?;
        if b.len() != shape[0] {
            return Err(missing(&bias_name));
        }
        Some(b)
    } else {
        None
    };
    let elements = shape[0]
        .checked_mul(shape[1])
        .filter(|&n| n <= isize::MAX as usize / 4)
        .ok_or_else(|| SpeechError::Tensor {
            name: name.clone(),
            why: "matrix shape overflows".into(),
        })?;
    let weight = backend.load_weight(WeightData {
        name: &name,
        shape: &shape,
        bytes: file.raw_bytes(&name)?,
        encoding,
        dtype,
        scales_dtype,
        offsets_dtype,
        scales: &scales,
        offsets: &offsets,
        block_scales,
    })?;
    Ok((
        Weight::Device {
            elements,
            dtype,
            packed,
            dynamic,
            weight,
        },
        bias,
    ))
}

fn load_float_quantized(
    file: &SafetensorsFile,
    base: &str,
    bits: u32,
    group_size: usize,
    kind: FloatQuantKind,
) -> Result<(Vec<f32>, Option<Vec<f32>>)> {
    let weight_name = format!("{base}.weight");
    let scales_name = format!("{base}.scales");
    if file.contains_tensor(&format!("{base}.biases")) {
        return Err(SpeechError::Tensor {
            name: format!("{base}.biases"),
            why: "MXFP/NVFP4 linears do not carry affine biases".to_string(),
        });
    }
    let weight = file
        .descriptor(&weight_name)
        .ok_or_else(|| missing(&weight_name))?;
    let scales = file
        .descriptor(&scales_name)
        .ok_or_else(|| missing(&scales_name))?;
    if weight.dtype != "U32" || weight.shape.len() != 2 || weight.shape[0] == 0 {
        return Err(SpeechError::Tensor {
            name: weight_name,
            why: format!("expected a non-empty 2-D U32 packed weight, got {weight:?}"),
        });
    }
    if scales.dtype != "U8" || scales.shape.len() != 2 || scales.shape[0] != weight.shape[0] {
        return Err(SpeechError::Tensor {
            name: scales_name,
            why: format!(
                "expected a 2-D U8 scale matrix with {} rows, got {scales:?}",
                weight.shape[0]
            ),
        });
    }
    if group_size == 0 || bits == 0 || 32 % bits != 0 {
        return Err(SpeechError::Unsupported {
            why: format!("invalid float-quantized mode: {bits} bits, group {group_size}"),
        });
    }
    let out_dim = weight.shape[0];
    let groups = scales.shape[1];
    let in_dim = groups
        .checked_mul(group_size)
        .ok_or_else(|| SpeechError::Tensor {
            name: scales_name.clone(),
            why: "input width overflows".to_string(),
        })?;
    let expected_words = in_dim
        .checked_mul(bits as usize)
        .filter(|width| width % 32 == 0)
        .map(|width| width / 32)
        .ok_or_else(|| SpeechError::Tensor {
            name: weight_name.clone(),
            why: "packed input width does not divide into U32 words".to_string(),
        })?;
    if groups == 0 || weight.shape[1] != expected_words {
        return Err(SpeechError::Tensor {
            name: weight_name.clone(),
            why: format!(
                "packed row width {} != {expected_words} U32 words for {in_dim} values",
                weight.shape[1]
            ),
        });
    }
    let expected_scale_bytes = out_dim
        .checked_mul(groups)
        .ok_or_else(|| SpeechError::Tensor {
            name: scales_name.clone(),
            why: "scale tensor size overflows".to_string(),
        })?;
    if scales.shape[1] != in_dim / group_size
        || file.raw_bytes(&scales_name)?.len() != expected_scale_bytes
    {
        return Err(SpeechError::Tensor {
            name: scales_name,
            why: "scale tensor byte count or group count does not match packed weights".to_string(),
        });
    }
    let packed = file.raw_bytes(&weight_name)?;
    let expected_weight_bytes = out_dim
        .checked_mul(expected_words)
        .and_then(|count| count.checked_mul(4))
        .ok_or_else(|| SpeechError::Tensor {
            name: weight_name.clone(),
            why: "packed weight tensor size overflows".to_string(),
        })?;
    if packed.len() != expected_weight_bytes {
        return Err(SpeechError::Tensor {
            name: weight_name.clone(),
            why: "packed weight byte count does not match its shape".to_string(),
        });
    }
    let scale_bytes = file.raw_bytes(&scales_name)?;
    let output_len = out_dim
        .checked_mul(in_dim)
        .filter(|&count| count <= isize::MAX as usize / 4)
        .ok_or_else(|| SpeechError::Tensor {
            name: weight_name.clone(),
            why: "dequantized shape overflows".to_string(),
        })?;
    let mask = (1u32 << bits) - 1;
    let mut output = Vec::with_capacity(output_len);
    for row in 0..out_dim {
        for column in 0..in_dim {
            let bit_offset = column * bits as usize;
            let word_offset = (row * expected_words + bit_offset / 32) * 4;
            let word = u32::from_le_bytes(
                packed[word_offset..word_offset + 4]
                    .try_into()
                    .expect("validated U32 word"),
            );
            let code = (word >> (bit_offset % 32)) & mask;
            let scale_byte = scale_bytes[row * groups + column / group_size];
            let scale = match kind {
                FloatQuantKind::MxFp4 | FloatQuantKind::MxFp8 => decode_e8m0(scale_byte),
                FloatQuantKind::NvFp4 => decode_e4m3(scale_byte),
            };
            let decoded = match kind {
                FloatQuantKind::MxFp4 | FloatQuantKind::NvFp4 => decode_e2m1(code),
                FloatQuantKind::MxFp8 => decode_e4m3(code as u8),
            };
            let value = decoded * scale;
            if !value.is_finite() {
                return Err(SpeechError::Tensor {
                    name: weight_name.clone(),
                    why: format!("non-finite value at row {row}, column {column}"),
                });
            }
            output.push(value);
        }
    }
    let bias_name = format!("{base}.bias");
    let bias = if file.contains_tensor(&bias_name) {
        let values = file.load_as_f32(&bias_name)?;
        if values.len() != out_dim {
            return Err(SpeechError::Tensor {
                name: bias_name,
                why: format!("bias length {} != output width {out_dim}", values.len()),
            });
        }
        Some(values)
    } else {
        None
    };
    Ok((output, bias))
}

/// OCP E2M1 has representable magnitudes 0, 0.5, 1, 1.5, 2, 3, 4, and 6.
fn decode_e2m1(code: u32) -> f32 {
    let sign = if code & 0x8 == 0 { 1.0 } else { -1.0 };
    let exponent = (code >> 1) & 0x3;
    let mantissa = code & 0x1;
    let magnitude = if exponent == 0 {
        mantissa as f32 * 0.5
    } else {
        (1.0 + mantissa as f32 * 0.5) * 2.0f32.powi(exponent as i32 - 1)
    };
    sign * magnitude
}

/// MLX's signed finite E4M3 encoding, including subnormals and its NaN code.
fn decode_e4m3(code: u8) -> f32 {
    let sign = if code & 0x80 == 0 { 1.0 } else { -1.0 };
    let exponent = (code >> 3) & 0x0f;
    let mantissa = code & 0x07;
    if exponent == 0x0f && mantissa == 0x07 {
        return f32::NAN;
    }
    let magnitude = if exponent == 0 {
        mantissa as f32 * 2.0f32.powi(-9)
    } else {
        (1.0 + mantissa as f32 / 8.0) * 2.0f32.powi(exponent as i32 - 7)
    };
    sign * magnitude
}

fn decode_e8m0(code: u8) -> f32 {
    if code == u8::MAX {
        f32::NAN
    } else {
        2.0f32.powi(code as i32 - 127)
    }
}

/// Open the safetensors shards of a converted tree: the index file's
/// `weight_map` order when present, otherwise the single model file.
pub(crate) fn open_converted_shards(dir: &Path) -> Result<Vec<SafetensorsFile>> {
    let mut names: Vec<String> = Vec::new();
    let index = dir.join("model.safetensors.index.json");
    if index.is_file() {
        let value = crate::quant::read_json(&index)?;
        let empty = serde_json::Map::new();
        let map = value
            .get("weight_map")
            .and_then(|w| w.as_object())
            .unwrap_or(&empty);
        let mut shards: Vec<String> = Vec::new();
        for (tensor, shard) in map {
            let shard = shard.as_str().ok_or_else(|| SpeechError::BadConfig {
                field: format!("weight_map[{tensor}]"),
                why: "shard name is not a string".to_string(),
            })?;
            if !shards.iter().any(|s| s == shard) {
                shards.push(shard.to_string());
            }
        }
        shards.sort();
        names = shards;
    } else {
        names.push("model.safetensors".to_string());
    }
    if names.is_empty() {
        return Err(SpeechError::Tensor {
            name: "model.safetensors".to_string(),
            why: "no shards listed for the converted tree".to_string(),
        });
    }
    let mut files = Vec::with_capacity(names.len());
    for name in names {
        let path = dir.join(&name);
        if !path.is_file() {
            return Err(SpeechError::Tensor {
                name,
                why: "shard file missing from the converted tree".to_string(),
            });
        }
        files.push(SafetensorsFile::open(&path)?);
    }
    Ok(files)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;
    use std::io::Write;

    fn write_quantized_fixture(path: &Path, bits: usize, group_size: usize, code: u32, scale: u8) {
        let words_per_row = bits * group_size / 32;
        let mut words = vec![0u32; words_per_row];
        words[0] = code;
        let mut weight_bytes = Vec::with_capacity(words_per_row * 4);
        for word in words {
            weight_bytes.extend_from_slice(&word.to_le_bytes());
        }

        let mut header = BTreeMap::<String, serde_json::Value>::new();
        header.insert(
            "layer.scales".to_string(),
            serde_json::json!({
                "dtype": "U8", "shape": [1, 1],
                "data_offsets": [weight_bytes.len(), weight_bytes.len() + 1]
            }),
        );
        header.insert(
            "layer.weight".to_string(),
            serde_json::json!({
                "dtype": "U32", "shape": [1, words_per_row],
                "data_offsets": [0, weight_bytes.len()]
            }),
        );
        let mut header_bytes = serde_json::to_vec(&header).unwrap();
        while header_bytes.len() % 8 != 0 {
            header_bytes.push(b' ');
        }
        let mut file = std::fs::File::create(path).unwrap();
        file.write_all(&(header_bytes.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(&header_bytes).unwrap();
        file.write_all(&weight_bytes).unwrap();
        file.write_all(&[scale]).unwrap();
    }

    #[test]
    fn checkpoint_loading_preserves_parameter_and_bias_precision() {
        let tensor = |data: Vec<f32>, shape: Vec<usize>, dtype| Tensor { data, shape, dtype };
        let parameters = HashMap::from([
            ("norm".into(), tensor(vec![1.0], vec![1], DType::Bf16)),
            (
                "layer.weight".into(),
                tensor(vec![1.0, 2.0], vec![1, 2], DType::F16),
            ),
            ("layer.bias".into(), tensor(vec![0.1], vec![1], DType::F32)),
        ]);
        let mut checkpoint =
            WeightStore::from_map(parameters.clone()).with_precision(Music3Precision::Checkpoint);
        assert_eq!(checkpoint.tensor("norm").unwrap().dtype, DType::Bf16);
        let (weight, bias) = checkpoint.linear("layer").unwrap();
        assert_eq!(weight.dtype(), DType::F16);
        assert_eq!(bias.unwrap().dtype, DType::F32);
        checkpoint.finish().unwrap();
        let mut float32 = WeightStore::from_map(parameters);
        assert_eq!(float32.tensor("norm").unwrap().dtype, DType::F32);
        let (weight, bias) = float32.linear("layer").unwrap();
        assert_eq!(weight.dtype(), DType::F32);
        assert_eq!(bias.unwrap().dtype, DType::F32);
        float32.finish().unwrap();
    }

    #[test]
    fn packed_mxfp4_mxfp8_and_nvfp4_linears_decode() {
        let root =
            std::env::temp_dir().join(format!("turbospark-music3-quant-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        let cases = [
            ("mxfp4", 4, 32, 1, 127, LinearQuantization::MxFp4, 0.5),
            ("mxfp8", 8, 32, 0x38, 127, LinearQuantization::MxFp8, 1.0),
            ("nvfp4", 4, 16, 2, 0x38, LinearQuantization::NvFp4, 1.0),
        ];
        for (name, bits, group_size, code, scale, scheme, first) in cases {
            let path = root.join(format!("{name}.safetensors"));
            write_quantized_fixture(&path, bits, group_size, code, scale);
            let file = SafetensorsFile::open(&path).unwrap();
            let mut store = WeightStore::from_files(vec![file], scheme);
            let (weight, bias) = store.linear("layer").unwrap();
            assert_eq!(weight.len(), group_size);
            let values = match &weight {
                Weight::Cpu { data: v, .. } => v.as_slice(),
                Weight::Device { .. } => panic!("cpu store returned a device weight"),
            };
            assert!((values[0] - first).abs() < 1e-6, "{name}: {}", values[0]);
            assert!(values[1..].iter().all(|value| *value == 0.0));
            assert!(bias.is_none());
            store.finish().unwrap();
        }
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn packed_float_quantization_rejects_unexpected_affine_biases() {
        let root = std::env::temp_dir().join(format!(
            "turbospark-music3-quant-bias-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let path = root.join("bad.safetensors");
        write_quantized_fixture(&path, 4, 32, 1, 127);
        let mut bytes = std::fs::read(&path).unwrap();
        let header_len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
        let mut header: serde_json::Value =
            serde_json::from_slice(&bytes[8..8 + header_len]).unwrap();
        header["layer.biases"] = serde_json::json!({
            "dtype": "U8", "shape": [1, 1], "data_offsets": [17, 18]
        });
        let mut header_bytes = serde_json::to_vec(&header).unwrap();
        while header_bytes.len() % 8 != 0 {
            header_bytes.push(b' ');
        }
        let payload = bytes.split_off(8 + header_len);
        let mut file = std::fs::File::create(&path).unwrap();
        file.write_all(&(header_bytes.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(&header_bytes).unwrap();
        file.write_all(&payload).unwrap();
        let safetensors = SafetensorsFile::open(&path).unwrap();
        let mut store = WeightStore::from_files(vec![safetensors], LinearQuantization::MxFp4);
        assert!(store.linear("layer").is_err());
        std::fs::remove_dir_all(root).unwrap();
    }
}

#[cfg(test)]
mod device_validation {
    use super::super::backend::{AttentionShape, DeviceWeight};
    use super::*;
    struct RejectUpload;
    impl ComputeBackend for RejectUpload {
        fn load_weight(&self, _data: WeightData<'_>) -> Result<Rc<dyn DeviceWeight>> {
            panic!("invalid packed geometry reached the device upload");
        }
        fn attention(
            &self,
            _q: &[f32],
            _k: &[f32],
            _v: &[f32],
            _s: AttentionShape,
            _dtype: DType,
        ) -> Result<Vec<f32>> {
            unreachable!()
        }
    }
    #[test]
    fn oversized_affine_group_is_refused_before_upload() {
        let dir =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata/minimax_music3/converted_q8");
        let files = open_converted_shards(&dir).unwrap();
        let mut store = WeightStore::from_files_with_backend(
            files,
            LinearQuantization::Affine(QuantScheme {
                bits: 8,
                group_size: 1usize << (usize::BITS - 3),
            }),
            Rc::new(RejectUpload),
        );
        let error = match store.linear("language_model.model.layers.0.self_attn.q_proj") {
            Err(e) => e,
            Ok(_) => panic!("overflowing group accepted"),
        };
        assert!(error.to_string().contains("overflows"), "{error}");
    }
}
