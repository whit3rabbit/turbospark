//! MLX Conv1d / ConvTranspose1d adapters over the shared kernels.
//!
//! MLX stores both conv weight layouts as `[out, kernel, in]` and runs
//! on channels-last `[N, W, C]` inputs; the crate kernels use PyTorch
//! layouts (`[out, in, K]`, `[in, out, K]`) on channel-major `[C, W]`
//! buffers. The loaders here permute once at load time. Mathematically
//! the two conventions compute the same convolution, which the fixture
//! parity tests pin.

use super::backend::{ConvShape, Weight};
use crate::Result;
use crate::SpeechError;

use super::precision::DType;
use super::weights::{Tensor, WeightStore};

pub(crate) struct MlxConv1d {
    weight: Weight, // PyTorch [out, in, K]
    bias: Option<Tensor>,
    out_ch: usize,
    in_ch: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
}

pub(crate) struct MlxConvTranspose1d {
    weight: Weight, // PyTorch [in, out, K]
    bias: Option<Tensor>,
    out_ch: usize,
    in_ch: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
}

pub(crate) struct ConvSpec {
    pub kernel: usize,
    pub stride: usize,
    pub padding: usize,
    pub dilation: usize,
}

fn permute_out_k_in_to_out_in_k(w: &[f32], shape: &[usize]) -> Vec<f32> {
    let [out, k, inch] = [shape[0], shape[1], shape[2]];
    let mut pt = Vec::with_capacity(w.len());
    for o in 0..out {
        for i in 0..inch {
            for kk in 0..k {
                pt.push(w[(o * k + kk) * inch + i]);
            }
        }
    }
    pt
}

fn permute_out_k_in_to_in_out_k(w: &[f32], shape: &[usize]) -> Vec<f32> {
    let [out, k, inch] = [shape[0], shape[1], shape[2]];
    let mut pt = Vec::with_capacity(w.len());
    for i in 0..inch {
        for o in 0..out {
            for kk in 0..k {
                pt.push(w[(o * k + kk) * inch + i]);
            }
        }
    }
    pt
}

impl MlxConv1d {
    pub(crate) fn load(store: &mut WeightStore, name: &str, spec: ConvSpec) -> Result<MlxConv1d> {
        let weight = store.tensor(&format!("{name}.weight"))?;
        if weight.shape.len() != 3 {
            return Err(SpeechError::Tensor {
                name: format!("{name}.weight"),
                why: format!("expected 3-D conv weight, got {:?}", weight.shape),
            });
        }
        let out_ch = weight.shape[0];
        let kernel = weight.shape[1];
        let in_ch = weight.shape[2];
        if kernel != spec.kernel || spec.dilation < 1 {
            return Err(SpeechError::Tensor {
                name: format!("{name}.weight"),
                why: format!("kernel {kernel} does not match expected {}", spec.kernel),
            });
        }
        let bias = load_optional_bias(store, name, out_ch)?;
        Ok(MlxConv1d {
            weight: store.dense_weight(
                permute_out_k_in_to_out_in_k(&weight.data, &weight.shape),
                &[out_ch, in_ch, kernel],
                weight.dtype,
            )?,
            bias,
            out_ch,
            in_ch,
            kernel,
            stride: spec.stride,
            padding: spec.padding,
            dilation: spec.dilation,
        })
    }

    /// Channel-major `[in, seq]` in, `[out, seq']` out.
    pub(crate) fn dtype(&self, input: DType) -> DType {
        self.bias
            .as_ref()
            .map_or(self.weight.output_dtype(input), |b| {
                self.weight.output_dtype(input).promote(b.dtype)
            })
    }
    pub(crate) fn forward_typed(
        &self,
        x: &[f32],
        _seq: usize,
        input_dtype: DType,
    ) -> Result<Vec<f32>> {
        let projection_dtype = self.weight.output_dtype(input_dtype);
        let dtype = self.dtype(input_dtype);
        let separate_bias = dtype != projection_dtype;
        let mut output = self.weight.convolution(
            x,
            if separate_bias {
                None
            } else {
                self.bias.as_deref()
            },
            ConvShape {
                input_channels: self.in_ch,
                output_channels: self.out_ch,
                kernel: self.kernel,
                stride: self.stride,
                padding: self.padding,
                dilation: self.dilation,
                transpose: false,
            },
            projection_dtype,
        )?;
        if separate_bias {
            if let Some(bias) = &self.bias {
                let frames = output.len() / self.out_ch;
                for (c, row) in output.chunks_exact_mut(frames).enumerate() {
                    for value in row {
                        *value = dtype.round(*value + bias[c]);
                    }
                }
            }
        }
        Ok(output)
    }
}

impl MlxConvTranspose1d {
    pub(crate) fn load(
        store: &mut WeightStore,
        name: &str,
        spec: ConvSpec,
    ) -> Result<MlxConvTranspose1d> {
        let weight = store.tensor(&format!("{name}.weight"))?;
        if weight.shape.len() != 3 {
            return Err(SpeechError::Tensor {
                name: format!("{name}.weight"),
                why: format!("expected 3-D conv weight, got {:?}", weight.shape),
            });
        }
        let out_ch = weight.shape[0];
        let kernel = weight.shape[1];
        let in_ch = weight.shape[2];
        if kernel != spec.kernel {
            return Err(SpeechError::Tensor {
                name: format!("{name}.weight"),
                why: format!("kernel {kernel} does not match expected {}", spec.kernel),
            });
        }
        let bias = load_optional_bias(store, name, out_ch)?;
        Ok(MlxConvTranspose1d {
            weight: store.dense_weight(
                permute_out_k_in_to_in_out_k(&weight.data, &weight.shape),
                &[in_ch, out_ch, kernel],
                weight.dtype,
            )?,
            bias,
            out_ch,
            in_ch,
            kernel,
            stride: spec.stride,
            padding: spec.padding,
        })
    }

    /// Channel-major `[in, seq]` in, `[out, seq']` out.
    pub(crate) fn dtype(&self, input: DType) -> DType {
        self.bias
            .as_ref()
            .map_or(self.weight.output_dtype(input), |b| {
                self.weight.output_dtype(input).promote(b.dtype)
            })
    }
    pub(crate) fn forward_typed(
        &self,
        x: &[f32],
        _seq: usize,
        input_dtype: DType,
    ) -> Result<Vec<f32>> {
        let projection_dtype = self.weight.output_dtype(input_dtype);
        let dtype = self.dtype(input_dtype);
        let separate_bias = dtype != projection_dtype;
        let mut output = self.weight.convolution(
            x,
            if separate_bias {
                None
            } else {
                self.bias.as_deref()
            },
            ConvShape {
                input_channels: self.in_ch,
                output_channels: self.out_ch,
                kernel: self.kernel,
                stride: self.stride,
                padding: self.padding,
                dilation: 1,
                transpose: true,
            },
            projection_dtype,
        )?;
        if separate_bias {
            if let Some(bias) = &self.bias {
                let frames = output.len() / self.out_ch;
                for (c, row) in output.chunks_exact_mut(frames).enumerate() {
                    for value in row {
                        *value = dtype.round(*value + bias[c]);
                    }
                }
            }
        }
        Ok(output)
    }
}

/// Bias-free convolutions (the DiT's kernel-1 residuals) omit the
/// tensor entirely.
fn load_optional_bias(
    store: &mut WeightStore,
    name: &str,
    out_ch: usize,
) -> Result<Option<Tensor>> {
    let bias_name = format!("{name}.bias");
    if !store.has(&bias_name) {
        return Ok(None);
    }
    let bias = store.tensor(&bias_name)?;
    if bias.data.len() != out_ch {
        return Err(SpeechError::Tensor {
            name: bias_name,
            why: format!("expected {out_ch} bias values"),
        });
    }
    Ok(Some(bias))
}
