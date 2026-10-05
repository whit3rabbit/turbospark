//! Weight-normalized convolutions and the Snake activation shared by the
//! codec families (SNAC, Descript DAC, and their derivatives).
//!
//! Reference: `mlx_audio/codec/models/snac/layers.py` and
//! `mlx_audio/codec/models/descript/nn/layers.py` at the pinned
//! mlx-audio commit; both define the same WNConv1d math.
//!
//! Layout contract (crate-wide): activations are channel-major
//! `[ch, seq]`; weights are stored in the PyTorch layout
//! (`[out, in/groups, K]` for conv, `[in, out/groups, K]` for transpose
//! conv) and converted from the MLX checkpoint layout
//! (`[out, K, in]` / `[in, K, out]`) at load time. Weight normalization
//! is folded once at load: `weight = weight_g * weight_v /
//! normalize(weight_v)`, matching the reference elementwise order.

use turbospark_model_io::safetensors::SafetensorsFile;

use crate::{ops, Result, SpeechError};

/// Loads a tensor and requires its safetensors shape to be `want`.
pub(crate) fn load_f32_shaped(
    file: &SafetensorsFile,
    name: &str,
    want: &[usize],
) -> Result<Vec<f32>> {
    let desc = file.descriptor(name).ok_or_else(|| SpeechError::Tensor {
        name: name.to_string(),
        why: "missing from checkpoint".to_string(),
    })?;
    if desc.shape != want {
        return Err(SpeechError::Tensor {
            name: name.to_string(),
            why: format!("expected shape {want:?}, got {:?}", desc.shape),
        });
    }
    file.load_as_f32(name).map_err(SpeechError::from)
}

/// Snake activation as the codec families define it:
/// `x + sin(alpha * x)^2 / (alpha + 1e-9)`. The `1e-9` guard inside the
/// reciprocal is part of the reference (`snac/layers.py::snake`);
/// `ops::snake` uses the unguarded vocoder variant, so codecs call this.
pub fn snake1d(x: &mut [f32], alpha: &[f32], channels: usize, frames: usize) {
    assert_eq!(alpha.len(), channels);
    for ch in 0..channels {
        let recip = 1.0 / (alpha[ch] + 1e-9);
        let a = alpha[ch];
        for f in 0..frames {
            let v = &mut x[ch * frames + f];
            *v += recip * (a * *v).sin().powi(2);
        }
    }
}

/// Weight-normalized Conv1d. `forward` maps `[in_ch, seq]` to
/// `[out_ch, out_seq]`.
#[derive(Debug, Clone)]
pub struct WnConv1d {
    pub in_ch: usize,
    pub out_ch: usize,
    pub kernel: usize,
    pub stride: usize,
    pub padding: usize,
    pub dilation: usize,
    pub groups: usize,
    /// PyTorch layout `[out_ch, in_ch / groups, kernel]`, weight norm
    /// already folded.
    weight: Vec<f32>,
    bias: Option<Vec<f32>>,
}

impl WnConv1d {
    /// Loads `{prefix}.weight_g`, `{prefix}.weight_v`, and the optional
    /// `{prefix}.bias` from an MLX-converted checkpoint (stored layout
    /// `[out, K, in/groups]`, `weight_g` `[out, 1, 1]`).
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        file: &SafetensorsFile,
        prefix: &str,
        in_ch: usize,
        out_ch: usize,
        kernel: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<Self> {
        let in_g = in_ch / groups;
        let v = load_f32_shaped(file, &format!("{prefix}.weight_v"), &[out_ch, kernel, in_g])?;
        let g = load_f32_shaped(file, &format!("{prefix}.weight_g"), &[out_ch, 1, 1])?;
        let bias = if file.contains_tensor(&format!("{prefix}.bias")) {
            Some(load_f32_shaped(file, &format!("{prefix}.bias"), &[out_ch])?)
        } else {
            None
        };
        // normalize_weight sums over all axes except the output channel
        // in the stored (out, K, in) order: K outer, in inner.
        let mut weight = vec![0.0f32; out_ch * in_g * kernel];
        for (oc, &gv) in g.iter().enumerate() {
            let mut acc = 0.0f32;
            let base = oc * kernel * in_g;
            for kk in 0..kernel {
                for ii in 0..in_g {
                    acc += v[base + kk * in_g + ii] * v[base + kk * in_g + ii];
                }
            }
            let norm = acc.sqrt();
            for kk in 0..kernel {
                for ii in 0..in_g {
                    // PyTorch layout [out, in, K] from stored [out, K, in].
                    weight[oc * in_g * kernel + ii * kernel + kk] =
                        gv * v[base + kk * in_g + ii] / norm;
                }
            }
        }
        Ok(WnConv1d {
            in_ch,
            out_ch,
            kernel,
            stride,
            padding,
            dilation,
            groups,
            weight,
            bias,
        })
    }

    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
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
            self.groups,
        )
    }
}

/// Weight-normalized ConvTranspose1d. `forward` maps `[in_ch, seq]` to
/// `[out_ch, out_seq]`.
#[derive(Debug, Clone)]
pub struct WnConvTranspose1d {
    pub in_ch: usize,
    pub out_ch: usize,
    pub kernel: usize,
    pub stride: usize,
    pub padding: usize,
    pub output_padding: usize,
    pub groups: usize,
    /// PyTorch layout `[in_ch, out_ch / groups, kernel]`, weight norm
    /// already folded.
    weight: Vec<f32>,
    bias: Option<Vec<f32>>,
}

impl WnConvTranspose1d {
    /// Loads `{prefix}.weight_g`, `{prefix}.weight_v`, and
    /// `{prefix}.bias` from an MLX-converted checkpoint (stored layout
    /// `[in, K, out/groups]`, `weight_g` `[in, 1, 1]`).
    ///
    /// The reference calls `mx.conv_transpose1d(x, weight, stride,
    /// padding, dilation, groups)` positionally, but MLX's parameter
    /// order is `(stride, padding, dilation, output_padding, groups)`,
    /// so the groups value lands in the output_padding slot. Codec
    /// transpose convs always use groups=1, so every family effectively
    /// decodes with output_padding=1; the loader sets that and asserts
    /// the groups=1 precondition.
    pub fn load(
        file: &SafetensorsFile,
        prefix: &str,
        in_ch: usize,
        out_ch: usize,
        kernel: usize,
        stride: usize,
        padding: usize,
    ) -> Result<Self> {
        let out_g = out_ch;
        let v = load_f32_shaped(file, &format!("{prefix}.weight_v"), &[in_ch, kernel, out_g])?;
        let g = load_f32_shaped(file, &format!("{prefix}.weight_g"), &[in_ch, 1, 1])?;
        let bias = if file.contains_tensor(&format!("{prefix}.bias")) {
            Some(load_f32_shaped(file, &format!("{prefix}.bias"), &[out_ch])?)
        } else {
            None
        };
        // Stored (in, K, out): normalize over axes (K, out), K outer.
        let mut weight = vec![0.0f32; in_ch * out_g * kernel];
        for (ic, &gv) in g.iter().enumerate() {
            let mut acc = 0.0f32;
            let base = ic * kernel * out_g;
            for kk in 0..kernel {
                for oo in 0..out_g {
                    acc += v[base + kk * out_g + oo] * v[base + kk * out_g + oo];
                }
            }
            let norm = acc.sqrt();
            for kk in 0..kernel {
                for oo in 0..out_g {
                    // PyTorch layout [in, out, K] from stored [in, K, out].
                    weight[ic * out_g * kernel + oo * kernel + kk] =
                        gv * v[base + kk * out_g + oo] / norm;
                }
            }
        }
        Ok(WnConvTranspose1d {
            in_ch,
            out_ch,
            kernel,
            stride,
            padding,
            output_padding: 1,
            groups: 1,
            weight,
            bias,
        })
    }

    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        ops::conv_transpose1d(
            x,
            &self.weight,
            self.bias.as_deref(),
            self.in_ch,
            self.out_ch,
            self.kernel,
            self.stride,
            self.padding,
            self.output_padding,
            self.groups,
        )
    }

    /// Descript DAC variant (`descript/nn/layers.py`): the checkpoint
    /// stores the transposed-conv weight out-first `[out, K, in/groups]`
    /// with `weight_g` `[1, 1, in/groups]` (per-in-channel norm, axes
    /// except 2), and the reference passes MLX's parameters by keyword,
    /// so output_padding stays 0 and groups=1 here.
    pub fn load_out_first(
        file: &SafetensorsFile,
        prefix: &str,
        in_ch: usize,
        out_ch: usize,
        kernel: usize,
        stride: usize,
        padding: usize,
    ) -> Result<Self> {
        let in_g = in_ch;
        let v = load_f32_shaped(file, &format!("{prefix}.weight_v"), &[out_ch, kernel, in_g])?;
        let g = load_f32_shaped(file, &format!("{prefix}.weight_g"), &[1, 1, in_g])?;
        let bias = if file.contains_tensor(&format!("{prefix}.bias")) {
            Some(load_f32_shaped(file, &format!("{prefix}.bias"), &[out_ch])?)
        } else {
            None
        };
        // normalize_weight(v, except_dim=2) sums over axes (out, K) in
        // row-major order for each in-channel.
        let mut weight = vec![0.0f32; in_g * out_ch * kernel];
        for (ic, &gv) in g.iter().enumerate() {
            let mut acc = 0.0f32;
            for oo in 0..out_ch {
                for kk in 0..kernel {
                    acc += v[oo * kernel * in_g + kk * in_g + ic]
                        * v[oo * kernel * in_g + kk * in_g + ic];
                }
            }
            let norm = acc.sqrt();
            for oo in 0..out_ch {
                for kk in 0..kernel {
                    // PyTorch layout [in, out, K] from stored [out, K, in].
                    weight[ic * out_ch * kernel + oo * kernel + kk] =
                        gv * v[oo * kernel * in_g + kk * in_g + ic] / norm;
                }
            }
        }
        Ok(WnConvTranspose1d {
            in_ch,
            out_ch,
            kernel,
            stride,
            padding,
            output_padding: 0,
            groups: 1,
            weight,
            bias,
        })
    }
}
