//! MLX Conv1d / ConvTranspose1d adapters over the shared kernels.
//!
//! MLX stores both conv weight layouts as `[out, kernel, in]` and runs
//! on channels-last `[N, W, C]` inputs; the crate kernels use PyTorch
//! layouts (`[out, in, K]`, `[in, out, K]`) on channel-major `[C, W]`
//! buffers. The loaders here permute once at load time. Mathematically
//! the two conventions compute the same convolution, which the fixture
//! parity tests pin.

use crate::ops;
use crate::Result;
use crate::SpeechError;

use super::weights::WeightStore;

pub(crate) struct MlxConv1d {
    weight: Vec<f32>, // PyTorch [out, in, K]
    bias: Option<Vec<f32>>,
    out_ch: usize,
    in_ch: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
}

pub(crate) struct MlxConvTranspose1d {
    weight: Vec<f32>, // PyTorch [in, out, K]
    bias: Option<Vec<f32>>,
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
            weight: permute_out_k_in_to_out_in_k(&weight.data, &weight.shape),
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
    pub(crate) fn forward(&self, x: &[f32], _seq: usize) -> Vec<f32> {
        ops::conv1d(
            x,
            &self.weight,
            self.bias.as_deref(),
            self.in_ch,
            self.out_ch,
            self.kernel,
            self.stride,
            self.padding,
            self.dilation,
            1,
        )
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
            weight: permute_out_k_in_to_in_out_k(&weight.data, &weight.shape),
            bias,
            out_ch,
            in_ch,
            kernel,
            stride: spec.stride,
            padding: spec.padding,
        })
    }

    /// Channel-major `[in, seq]` in, `[out, seq']` out.
    pub(crate) fn forward(&self, x: &[f32], _seq: usize) -> Vec<f32> {
        ops::conv_transpose1d(
            x,
            &self.weight,
            self.bias.as_deref(),
            self.in_ch,
            self.out_ch,
            self.kernel,
            self.stride,
            self.padding,
            1,
        )
    }
}

/// Bias-free convolutions (the DiT's kernel-1 residuals) omit the
/// tensor entirely.
fn load_optional_bias(
    store: &mut WeightStore,
    name: &str,
    out_ch: usize,
) -> Result<Option<Vec<f32>>> {
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
    Ok(Some(bias.data))
}
