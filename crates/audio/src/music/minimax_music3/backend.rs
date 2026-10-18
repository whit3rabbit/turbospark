//! Device seam for Music 3. The portable model owns sampling and sequencing;
//! a device owns resident weights and the expensive tensor operations.

use super::precision::{self, DType};
use super::weights::Tensor;
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
    /// Output rows of the full matrix when `shape` is a row window of it.
    /// The device plans width-dependent kernel choices (the dense split-K)
    /// from the full matrix, so the window reproduces its columns bit for bit.
    pub planned_rows: Option<usize>,
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
    /// [`Self::convolution`] for one element of `batch` independent inputs that
    /// MLX runs as a single call. Each output is unchanged; only a backend's
    /// choice of K split can depend on `batch` (MLX's split-K rule sees
    /// `batch * outlen` rows). Backends without that rule ignore it.
    fn convolution_batched(
        &self,
        input: &[f32],
        bias: Option<&[f32]>,
        shape: ConvShape,
        batch: usize,
        dtype: DType,
    ) -> Result<Vec<f32>> {
        let _ = batch;
        self.convolution(input, bias, shape, dtype)
    }
    /// Groups are explicit because Kokoro includes depthwise transpose pools.
    fn convolution_grouped(
        &self,
        input: &[f32],
        bias: Option<&[f32]>,
        shape: ConvShape,
        groups: usize,
        dtype: DType,
    ) -> Result<Vec<f32>> {
        if groups != 1 {
            return Err(SpeechError::Unsupported {
                why: "backend does not implement grouped audio convolution".into(),
            });
        }
        self.convolution(input, bias, shape, dtype)
    }
    /// Recurrent weights are [4 * hidden, hidden], with i/f/g/o gate order.
    fn lstm_recurrence(
        &self,
        _projection: &[f32],
        _hidden: usize,
        _backward: bool,
    ) -> Result<Vec<f32>> {
        Err(SpeechError::Unsupported {
            why: "backend does not implement audio LSTM recurrence".into(),
        })
    }
    /// The concrete weight, for a device [`Sequence`] that binds its buffers.
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        None
    }
}

/// A persistent device f32 buffer that outlives one [`Sequence`] (the AR KV
/// cache). The host writes and reads it only between submissions.
pub trait DeviceBuffer {
    fn write(&self, first: usize, data: &[f32]) -> Result<()>;
    fn read(&self, first: usize, len: usize) -> Result<Vec<f32>>;
    fn as_any(&self) -> &dyn std::any::Any;
}

/// A tensor inside a [`Sequence`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Slot(pub usize);

/// Device operations recorded together and read back once, so a device can
/// run a whole depth step in one submission. Every operation must produce
/// exactly what the matching per-operation call produces.
pub trait Sequence {
    fn upload(&mut self, data: &[f32]) -> Result<Slot>;
    /// A persistent buffer as a slot, without a copy.
    fn bind(&mut self, buffer: &dyn DeviceBuffer) -> Result<Slot>;
    /// `len` elements of `slot` starting at element `first`, without a copy.
    fn view(&mut self, slot: Slot, first: usize, len: usize) -> Result<Slot>;
    /// Copy `src` into the existing `dst` bit for bit.
    fn copy(&mut self, src: Slot, dst: Slot) -> Result<()>;
    /// [`ComputeBackend::rope`] on a slot.
    fn rope(&mut self, x: Slot, shape: RopeShape, dtype: DType) -> Result<Slot>;
    /// A projection; `bias` and `dtype` (the output dtype) as passed to
    /// [`DeviceWeight::linear`].
    #[allow(clippy::too_many_arguments)]
    fn linear(
        &mut self,
        x: Slot,
        weight: &dyn DeviceWeight,
        bias: Option<&[f32]>,
        rows: usize,
        input_dim: usize,
        output_dim: usize,
        dtype: DType,
    ) -> Result<Slot>;
    /// [`ComputeBackend::layer_norm`] on a slot.
    #[allow(clippy::too_many_arguments)]
    fn layer_norm(
        &mut self,
        x: Slot,
        w: &[f32],
        bias: Option<&[f32]>,
        rows: usize,
        cols: usize,
        eps: f32,
        dtype: DType,
    ) -> Result<Slot>;
    /// The DiT's typed partial NeoX rotary (`dit::partial_rope_typed`) over
    /// `[len, heads * head_dim]` with `[len, rotary / 2]` tables.
    #[allow(clippy::too_many_arguments)]
    fn partial_rope(
        &mut self,
        x: Slot,
        cos: Slot,
        sin: Slot,
        len: usize,
        heads: usize,
        head_dim: usize,
        rotary: usize,
        dtype: DType,
    ) -> Result<Slot>;
    /// `dtype.round(state * dtype.silu(gate))` over rows of `[state | gate]`,
    /// each `half` wide.
    fn gated_silu(&mut self, x: Slot, rows: usize, half: usize, dtype: DType) -> Result<Slot>;
    fn rms_norm(
        &mut self,
        x: Slot,
        w: &[f32],
        rows: usize,
        cols: usize,
        eps: f32,
        dtype: DType,
    ) -> Result<Slot>;
    fn attention(
        &mut self,
        q: Slot,
        k: Slot,
        v: Slot,
        shape: AttentionShape,
        dtype: DType,
    ) -> Result<Slot>;
    /// A single-group convolution with the bias added in the kernel; `shape`,
    /// `batch` and `dtype` as passed to [`DeviceWeight::convolution_batched`].
    /// Backends that cannot record it report [`SpeechError::Unsupported`] and
    /// the caller keeps the per-operation path.
    #[allow(clippy::too_many_arguments)]
    fn convolution(
        &mut self,
        _x: Slot,
        _weight: &dyn DeviceWeight,
        _bias: Option<&[f32]>,
        _shape: ConvShape,
        _batch: usize,
        _dtype: DType,
    ) -> Result<Slot> {
        Err(SpeechError::Unsupported {
            why: "sequence does not record convolutions".into(),
        })
    }
    /// [`ComputeBackend::snake`] on a slot.
    fn snake(
        &mut self,
        _x: Slot,
        _alpha: &[f32],
        _channels: usize,
        _frames: usize,
        _dtype: DType,
    ) -> Result<Slot> {
        Err(SpeechError::Unsupported {
            why: "sequence does not record Snake".into(),
        })
    }
    /// `dtype.round(a + b)`.
    fn add_round(&mut self, a: Slot, b: Slot, dtype: DType) -> Result<Slot>;
    /// `fused.round(gate.silu(g) * u)`.
    fn swiglu(&mut self, g: Slot, u: Slot, gate: DType, fused: DType) -> Result<Slot>;
    /// Run everything and read `read` back. `Ok(None)` means an operation met
    /// a case the device cannot round exactly like the host; the caller
    /// recomputes on the per-operation path.
    fn finish(self: Box<Self>, read: &[Slot]) -> Result<Option<Vec<Vec<f32>>>>;
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

/// Arithmetic layout before materializing channel-major normalization input.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NormalizationLayout {
    Rows,
    Columns,
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
    fn normalize(
        &self,
        _x: &[f32],
        _rows: usize,
        _cols: usize,
        _eps: f32,
        _layout: NormalizationLayout,
    ) -> Result<Vec<f32>> {
        Err(SpeechError::Unsupported {
            why: "backend does not implement audio normalization layout".into(),
        })
    }
    fn sine(&self, _x: &[f32]) -> Result<Vec<f32>> {
        Err(SpeechError::Unsupported {
            why: "backend does not implement precise audio sine".into(),
        })
    }
    fn tanh(&self, _x: &[f32]) -> Result<Vec<f32>> {
        Err(SpeechError::Unsupported {
            why: "backend does not implement precise audio tanh".into(),
        })
    }
    /// Kokoro uses the compiled erf activation rather than an erf approximation.
    fn gelu_erf(&self, _x: &[f32]) -> Result<Vec<f32>> {
        Err(SpeechError::Unsupported {
            why: "backend does not implement audio erf-GELU".into(),
        })
    }
    /// Normalize checkpoint [channel,input,kernel] weights in sanitized reduction order.
    fn weight_norm(
        &self,
        _x: &[f32],
        _g: &[f32],
        _rows: usize,
        _input: usize,
        _kernel: usize,
    ) -> Result<Vec<f32>> {
        Err(SpeechError::Unsupported {
            why: "backend does not implement audio weight normalization".into(),
        })
    }
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
    /// Inclusive scan along rows of a row-major [rows, columns] F32 tensor.
    fn cumulative_sum(&self, _x: &[f32], _rows: usize, _columns: usize) -> Result<Vec<f32>> {
        Err(SpeechError::Unsupported {
            why: "backend does not implement strided audio scan".into(),
        })
    }
    /// Centered, reflect-padded periodic-window STFT, channel-major magnitude then phase.
    fn stft_magnitude_phase(&self, _x: &[f32], _window: &[f32], _hop: usize) -> Result<Vec<f32>> {
        Err(SpeechError::Unsupported {
            why: "backend does not implement audio STFT".into(),
        })
    }
    fn trace(&self, _stage: &str, _data: &[f32], _dtype: DType, _shape: &[usize]) {}
    /// A [`Sequence`] for one `op` of geometry `shape`, or `None` to keep the
    /// per-operation path (the default, and any time traces are observed).
    fn sequence(&self, _op: &'static str, _shape: &[usize]) -> Option<Box<dyn Sequence + '_>> {
        None
    }
    /// A persistent buffer of `len` f32 values for sequences to bind, or
    /// `None` when the backend has no sequences.
    fn buffer(&self, _len: usize) -> Option<Result<Rc<dyn DeviceBuffer>>> {
        None
    }
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
    /// [`Self::convolution`] when MLX would run `batch` inputs as one call.
    pub(crate) fn convolution_batched(
        &self,
        input: &[f32],
        bias: Option<&[f32]>,
        s: ConvShape,
        batch: usize,
        dtype: DType,
    ) -> Result<Vec<f32>> {
        match self {
            Self::Device { weight, .. } => weight.convolution_batched(input, bias, s, batch, dtype),
            _ => self.convolution(input, bias, s, dtype),
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

/// [`linear`] without bias, recorded into `sequence`: the same output dtype
/// and the same device call as the per-operation path.
pub(crate) fn sequence_linear(
    sequence: &mut dyn Sequence,
    x: Slot,
    weight: &Weight,
    rows: usize,
    input_dim: usize,
    output_dim: usize,
    input_dtype: DType,
) -> Result<Slot> {
    sequence_linear_bias(
        sequence,
        x,
        weight,
        None,
        rows,
        input_dim,
        output_dim,
        input_dtype,
    )
}

/// Whether [`linear`] passes `bias` to the device kernel rather than adding
/// it on the host: the bias dtype matches the projection's output dtype (or
/// there is no bias). Only then can a sequence reproduce it.
pub(crate) fn bias_in_kernel(weight: &Weight, bias: Option<&Tensor>, input_dtype: DType) -> bool {
    let projection = weight.output_dtype(input_dtype);
    bias.filter(|b| !b.is_empty())
        .is_none_or(|b| projection.promote(b.dtype) == projection)
}

/// [`linear`] with an in-kernel bias (see [`bias_in_kernel`]), recorded into
/// `sequence`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn sequence_linear_bias(
    sequence: &mut dyn Sequence,
    x: Slot,
    weight: &Weight,
    bias: Option<&Tensor>,
    rows: usize,
    input_dim: usize,
    output_dim: usize,
    input_dtype: DType,
) -> Result<Slot> {
    let bias = bias.filter(|b| !b.is_empty());
    match weight {
        Weight::Device { weight: device, .. }
            if weight.len() == input_dim * output_dim
                && bias.is_none_or(|b| b.len() == output_dim)
                && bias_in_kernel(weight, bias, input_dtype) =>
        {
            let dtype = weight.output_dtype(input_dtype);
            sequence.linear(
                x,
                device.as_ref(),
                bias.map(|b| b.data.as_slice()),
                rows,
                input_dim,
                output_dim,
                dtype,
            )
        }
        _ => Err(SpeechError::Input {
            why: "sequence linear needs a matching device weight and kernel bias".into(),
        }),
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
