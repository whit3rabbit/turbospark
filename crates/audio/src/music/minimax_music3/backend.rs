//! Device seam for Music 3. The portable model owns sampling and sequencing;
//! a device owns resident weights and the expensive tensor operations.

use super::precision::{self, DType};
use crate::{ops, Result, SpeechError};
use std::rc::Rc;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WeightEncoding {
    F32,
    F16,
    Bf16,
    Affine { bits: u32, group_size: usize },
    MxFp4,
    MxFp8,
    NvFp4,
}

/// Borrowed checkpoint data. Affine companions are decoded to f32, while
/// floating-point block scales retain their one-byte encoding.
pub struct WeightData<'a> {
    pub name: &'a str,
    pub shape: &'a [usize],
    pub bytes: &'a [u8],
    pub encoding: WeightEncoding,
    pub dtype: DType,
    pub scales_dtype: Option<DType>,
    pub offsets_dtype: Option<DType>,
    pub scales: &'a [f32],
    pub offsets: &'a [f32],
    pub block_scales: &'a [u8],
}

#[derive(Clone, Copy, Debug)]
pub struct ConvShape {
    pub input_channels: usize,
    pub output_channels: usize,
    pub kernel: usize,
    pub stride: usize,
    pub padding: usize,
    pub dilation: usize,
    pub transpose: bool,
}

/// Q is batch/position/head/dimension. K/V use that layout unless
/// `kv_time_major` is set, which addresses the AR cache as time/batch/head.
#[derive(Clone, Copy, Debug)]
pub struct AttentionShape {
    pub batch: usize,
    pub queries: usize,
    pub keys: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub dim: usize,
    pub kv_time_major: bool,
    pub causal: bool,
    pub offset: usize,
}

pub trait DeviceWeight {
    fn linear(
        &self,
        input: &[f32],
        bias: Option<&[f32]>,
        rows: usize,
        input_dim: usize,
        output_dim: usize,
        dtype: DType,
    ) -> Result<Vec<f32>>;
    fn embedding(&self, ids: &[i32], width: usize) -> Result<Vec<f32>>;
    fn convolution(
        &self,
        input: &[f32],
        bias: Option<&[f32]>,
        shape: ConvShape,
        dtype: DType,
    ) -> Result<Vec<f32>>;
}

#[derive(Clone, Copy, Debug)]
pub struct RopeShape {
    pub batch: usize,
    pub seq: usize,
    pub heads: usize,
    pub dim: usize,
    pub offset: usize,
    pub theta: f32,
}

pub trait ComputeBackend {
    fn load_weight(&self, data: WeightData<'_>) -> Result<Rc<dyn DeviceWeight>>;
    fn attention(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        shape: AttentionShape,
        dtype: DType,
    ) -> Result<Vec<f32>>;
    fn rms_norm(
        &self,
        x: &[f32],
        w: &[f32],
        rows: usize,
        cols: usize,
        eps: f32,
        dtype: DType,
    ) -> Result<Vec<f32>> {
        Ok(precision::rms_norm(x, w, rows, cols, eps, dtype))
    }
    #[allow(clippy::too_many_arguments)]
    fn layer_norm(
        &self,
        x: &[f32],
        w: &[f32],
        bias: Option<&[f32]>,
        rows: usize,
        cols: usize,
        eps: f32,
        dtype: DType,
    ) -> Result<Vec<f32>> {
        Ok(precision::layer_norm(x, w, bias, rows, cols, eps, dtype))
    }
    fn rope(&self, x: &[f32], shape: RopeShape, dtype: DType) -> Result<Vec<f32>> {
        Ok(rope(x, shape, dtype))
    }
    fn rotary_tables(
        &self,
        len: usize,
        rotary_dim: usize,
        theta: f32,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        Ok(rotary_tables(len, rotary_dim, theta))
    }
    /// Uniforms are the f32 MLX erfinv inputs in (-1, 1), after its
    /// uniform-range remap. The Gaussian output is rounded to logical dtype.
    fn normal_from_uniform(&self, uniforms: &[f32], dtype: DType) -> Result<Vec<f32>> {
        Ok(super::rng::normal_from_uniform(uniforms, dtype))
    }
    /// Device backends reproduce native transcendental rounding in Snake.
    /// The portable default uses exact multiplication for the squared sine.
    fn snake(
        &self,
        x: &[f32],
        alpha: &[f32],
        channels: usize,
        frames: usize,
        dtype: DType,
    ) -> Result<Vec<f32>> {
        let mut out = x.to_vec();
        super::vocoder::snake(&mut out, alpha, channels, frames, dtype);
        Ok(out)
    }
    fn trace(&self, _stage: &str, _data: &[f32], _dtype: DType, _shape: &[usize]) {}
}

pub(crate) fn rotary_tables(len: usize, rotary_dim: usize, theta: f32) -> (Vec<f32>, Vec<f32>) {
    let half = rotary_dim / 2;
    // The reference computes positive powers and then reciprocals. Negative
    // powers can land on a different f32 value before partial rotary rounds.
    let frequencies: Vec<f32> = (0..half)
        .map(|i| 1.0f32 / theta.powf((2 * i) as f32 / rotary_dim as f32))
        .collect();
    let mut cos = Vec::with_capacity(len * half);
    let mut sin = Vec::with_capacity(len * half);
    for t in 0..len {
        for frequency in &frequencies {
            let angle = t as f32 * frequency;
            cos.push(angle.cos());
            sin.push(angle.sin());
        }
    }
    (cos, sin)
}

pub(crate) fn rope(x: &[f32], shape: RopeShape, dtype: DType) -> Vec<f32> {
    let mut out = x.to_vec();
    let (cos, sin) = ops::rope_tables_range(shape.offset, shape.seq, shape.dim, shape.theta);
    let half = shape.dim / 2;
    for b in 0..shape.batch {
        for t in 0..shape.seq {
            for h in 0..shape.heads {
                for d in 0..half {
                    let base = ((b * shape.seq + t) * shape.heads + h) * shape.dim;
                    let a = x[base + d];
                    let b = x[base + half + d];
                    let c = cos[t * half + d];
                    let s = sin[t * half + d];
                    out[base + d] = dtype.round(a * c - b * s);
                    out[base + half + d] = dtype.round(b * c + a * s);
                }
            }
        }
    }
    out
}

pub(crate) fn rms_norm(
    backend: &Option<Rc<dyn ComputeBackend>>,
    x: &[f32],
    w: &[f32],
    rows: usize,
    cols: usize,
    eps: f32,
    dtype: DType,
) -> Result<Vec<f32>> {
    match backend {
        Some(b) if dtype != DType::F32 => b.rms_norm(x, w, rows, cols, eps, dtype),
        _ => Ok(precision::rms_norm(x, w, rows, cols, eps, dtype)),
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn layer_norm(
    backend: &Option<Rc<dyn ComputeBackend>>,
    x: &[f32],
    w: &[f32],
    bias: Option<&[f32]>,
    rows: usize,
    cols: usize,
    eps: f32,
    dtype: DType,
) -> Result<Vec<f32>> {
    match backend {
        Some(b) if dtype != DType::F32 => b.layer_norm(x, w, bias, rows, cols, eps, dtype),
        _ => Ok(precision::layer_norm(x, w, bias, rows, cols, eps, dtype)),
    }
}

pub(crate) fn trace(
    backend: &Option<Rc<dyn ComputeBackend>>,
    stage: &str,
    data: &[f32],
    dtype: DType,
    shape: &[usize],
) {
    if let Some(b) = backend {
        b.trace(stage, data, dtype, shape);
    }
}

pub(crate) fn attention(
    backend: &Option<Rc<dyn ComputeBackend>>,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    s: AttentionShape,
    dtype: DType,
) -> Result<Vec<f32>> {
    if let Some(b) = backend {
        return b.attention(q, k, v, s, dtype);
    }
    let mut out = vec![0.0; q.len()];
    let fallback = dtype != DType::F32 && !matches!(s.dim, 64 | 80 | 96 | 128 | 256);
    let scale = if fallback {
        dtype.round(1.0 / (s.dim as f32).sqrt())
    } else {
        1.0 / (s.dim as f32).sqrt()
    };
    for b in 0..s.batch {
        for t in 0..s.queries {
            for h in 0..s.heads {
                let kvh = h / (s.heads / s.kv_heads);
                let qb = ((b * s.queries + t) * s.heads + h) * s.dim;
                let mut scores = vec![0.0; s.keys];
                for (key, score) in scores.iter_mut().enumerate() {
                    if s.causal && key > s.offset + t {
                        *score = f32::NEG_INFINITY;
                        continue;
                    }
                    let kb = if s.kv_time_major {
                        ((key * s.batch + b) * s.kv_heads + kvh) * s.dim
                    } else {
                        ((b * s.keys + key) * s.kv_heads + kvh) * s.dim
                    };
                    let mut dot = 0.0;
                    for d in 0..s.dim {
                        let query = if fallback {
                            dtype.round(q[qb + d] * scale)
                        } else {
                            q[qb + d]
                        };
                        dot += query * k[kb + d];
                    }
                    *score = if fallback {
                        dtype.round(dot)
                    } else {
                        dot * scale
                    };
                }
                ops::softmax_row(&mut scores);
                if fallback {
                    dtype.round_slice(&mut scores);
                }
                for d in 0..s.dim {
                    let mut value = 0.0;
                    for (key, probability) in scores.iter().enumerate() {
                        let vb = if s.kv_time_major {
                            ((key * s.batch + b) * s.kv_heads + kvh) * s.dim
                        } else {
                            ((b * s.keys + key) * s.kv_heads + kvh) * s.dim
                        };
                        value += probability * v[vb + d];
                    }
                    out[qb + d] = dtype.round(value);
                }
            }
        }
    }
    Ok(out)
}

#[derive(Clone)]
pub(crate) enum Weight {
    Cpu {
        data: Rc<Vec<f32>>,
        dtype: DType,
        packed: bool,
        dynamic: bool,
    },
    Device {
        elements: usize,
        dtype: DType,
        packed: bool,
        dynamic: bool,
        weight: Rc<dyn DeviceWeight>,
    },
}

impl Weight {
    pub(crate) fn cpu_typed(data: Vec<f32>, dtype: DType, packed: bool, dynamic: bool) -> Self {
        Self::Cpu {
            data: Rc::new(data),
            dtype,
            packed,
            dynamic,
        }
    }
    pub(crate) fn dtype(&self) -> DType {
        match self {
            Self::Cpu { dtype, .. } | Self::Device { dtype, .. } => *dtype,
        }
    }
    pub(crate) fn output_dtype(&self, input: DType) -> DType {
        match self {
            Self::Cpu { dynamic: true, .. } | Self::Device { dynamic: true, .. } => input,
            _ => input.promote(self.dtype()),
        }
    }
    fn packed(&self) -> bool {
        match self {
            Self::Cpu { packed, .. } | Self::Device { packed, .. } => *packed,
        }
    }
    pub(crate) fn len(&self) -> usize {
        match self {
            Self::Cpu { data: v, .. } => v.len(),
            Self::Device { elements, .. } => *elements,
        }
    }
    pub(crate) fn embedding(&self, ids: &[i32], width: usize) -> Result<Vec<f32>> {
        if width == 0
            || ids
                .iter()
                .any(|&id| id < 0 || id as usize >= self.len() / width)
        {
            return Err(SpeechError::Input {
                why: "embedding id outside vocabulary".into(),
            });
        }
        match self {
            Self::Cpu { data: v, .. } => Ok(ops::embedding(v, width, ids)),
            Self::Device { weight, .. } => weight.embedding(ids, width),
        }
    }
    pub(crate) fn convolution(
        &self,
        input: &[f32],
        bias: Option<&[f32]>,
        s: ConvShape,
        dtype: DType,
    ) -> Result<Vec<f32>> {
        let cpu_bias = if dtype == DType::F32 { bias } else { None };
        let mut output = match self {
            Self::Cpu { data: v, .. } if s.transpose => {
                Ok::<_, SpeechError>(ops::conv_transpose1d(
                    input,
                    v,
                    cpu_bias,
                    s.input_channels,
                    s.output_channels,
                    s.kernel,
                    s.stride,
                    s.padding,
                    0,
                    1,
                ))
            }
            Self::Cpu { data: v, .. } => Ok::<_, SpeechError>(ops::conv1d(
                input,
                v,
                cpu_bias,
                s.input_channels,
                s.output_channels,
                s.kernel,
                s.stride,
                s.padding,
                s.dilation,
                1,
            )),
            Self::Device { weight, .. } => return weight.convolution(input, bias, s, dtype),
        }?;
        dtype.round_slice(&mut output);
        if dtype != DType::F32 {
            if let Some(bias) = bias {
                let frames = output.len() / s.output_channels;
                for (c, row) in output.chunks_exact_mut(frames).enumerate() {
                    for v in row {
                        *v = dtype.round(*v + bias[c]);
                    }
                }
            }
        }
        Ok(output)
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn linear(
    input: &[f32],
    weight: &Weight,
    bias: Option<&super::weights::Tensor>,
    rows: usize,
    input_dim: usize,
    output_dim: usize,
    input_dtype: DType,
) -> Result<Vec<f32>> {
    if input.len() != rows * input_dim
        || weight.len() != input_dim * output_dim
        || bias.is_some_and(|b| !b.is_empty() && b.len() != output_dim)
    {
        return Err(SpeechError::Input {
            why: "linear geometry does not match weights".into(),
        });
    }
    let bias = bias.filter(|b| !b.is_empty());
    let projection_dtype = weight.output_dtype(input_dtype);
    let dtype = bias.map_or(projection_dtype, |b| projection_dtype.promote(b.dtype));
    let separate_bias = (weight.packed()
        && (projection_dtype != DType::F32 || dtype != projection_dtype))
        || (!weight.packed() && rows == 1 && dtype != DType::F32);
    let bias_values = bias.map(|b| b.data.as_slice());
    match weight {
        Weight::Cpu { data, .. } => {
            let mut output = ops::linear(
                input,
                data,
                if separate_bias { None } else { bias_values },
                rows,
                input_dim,
                output_dim,
            );
            (if separate_bias {
                projection_dtype
            } else {
                dtype
            })
            .round_slice(&mut output);
            if separate_bias {
                if let Some(bias) = bias {
                    for row in output.chunks_exact_mut(output_dim) {
                        for (v, b) in row.iter_mut().zip(bias.iter()) {
                            *v = dtype.round(*v + b);
                        }
                    }
                }
            }
            Ok(output)
        }
        Weight::Device { weight, .. } => {
            if separate_bias && dtype != projection_dtype {
                let mut output =
                    weight.linear(input, None, rows, input_dim, output_dim, projection_dtype)?;
                if let Some(bias) = bias {
                    for row in output.chunks_exact_mut(output_dim) {
                        for (v, b) in row.iter_mut().zip(bias.iter()) {
                            *v = dtype.round(*v + b);
                        }
                    }
                }
                Ok(output)
            } else {
                weight.linear(input, bias_values, rows, input_dim, output_dim, dtype)
            }
        }
    }
}
