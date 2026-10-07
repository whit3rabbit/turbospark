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

use crate::codec::conv::{
    load_bias, permute_okc_to_cok, permute_okc_to_ock, wn_fold_in_place, BiasMode, Conv1d,
    ConvTranspose1d, WnFold,
};
use crate::{Result, SpeechError};

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

/// Weight-normalized Conv1d: the shared [`Conv1d`] with the fold already
/// applied. `forward` maps `[in_ch, seq]` to `[out_ch, out_seq]`.
pub type WnConv1d = Conv1d;

impl Conv1d {
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
        let mut v = load_f32_shaped(file, &format!("{prefix}.weight_v"), &[out_ch, kernel, in_g])?;
        let g = load_f32_shaped(file, &format!("{prefix}.weight_g"), &[out_ch, 1, 1])?;
        let bias = load_bias(file, &format!("{prefix}.bias"), out_ch, BiasMode::Optional)?;
        // normalize_weight sums over all axes except the output channel
        // in the stored (out, K, in) order: K outer, in inner, which is
        // the contiguous row order, so the fold runs before the permute.
        wn_fold_in_place(&mut v, &g, kernel * in_g, WnFold::Div);
        // PyTorch layout [out, in, K] from stored [out, K, in].
        let weight = permute_okc_to_ock(&v, out_ch, kernel, in_g);
        Ok(Conv1d {
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
}

/// Weight-normalized ConvTranspose1d: the shared [`ConvTranspose1d`]
/// with the fold already applied. `forward` maps `[in_ch, seq]` to
/// `[out_ch, out_seq]`.
pub type WnConvTranspose1d = ConvTranspose1d;

impl ConvTranspose1d {
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
        let mut v = load_f32_shaped(file, &format!("{prefix}.weight_v"), &[in_ch, kernel, out_g])?;
        let g = load_f32_shaped(file, &format!("{prefix}.weight_g"), &[in_ch, 1, 1])?;
        let bias = load_bias(file, &format!("{prefix}.bias"), out_ch, BiasMode::Optional)?;
        // Stored (in, K, out): normalize over axes (K, out), K outer.
        wn_fold_in_place(&mut v, &g, kernel * out_g, WnFold::Div);
        // PyTorch layout [in, out, K] from stored [in, K, out].
        let weight = permute_okc_to_ock(&v, in_ch, kernel, out_g);
        Ok(ConvTranspose1d {
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
        let bias = load_bias(file, &format!("{prefix}.bias"), out_ch, BiasMode::Optional)?;
        // normalize_weight(v, except_dim=2) sums over axes (out, K) in
        // row-major order for each in-channel; after the permute to
        // [in, out, K] that is exactly the contiguous row order.
        let mut weight = permute_okc_to_cok(&v, out_ch, kernel, in_g);
        wn_fold_in_place(&mut weight, &g, out_ch * kernel, WnFold::Div);
        Ok(ConvTranspose1d {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snake1d_computes_guarded_activation() {
        let mut x = vec![0.0f32, std::f32::consts::FRAC_PI_2];
        let alpha = vec![1.0f32];
        snake1d(&mut x, &alpha, 1, 2);
        assert!((x[0] - 0.0).abs() < 1e-6);
        let expected_1 = std::f32::consts::FRAC_PI_2 + 1.0 / (1.0 + 1e-9);
        assert!((x[1] - expected_1).abs() < 1e-5);
    }

    #[test]
    fn wn_conv1d_forward_shapes_and_values() {
        // 1 in_ch, 1 out_ch, kernel 3, stride 1, padding 1 (same conv).
        let conv = WnConv1d {
            in_ch: 1,
            out_ch: 1,
            kernel: 3,
            stride: 1,
            padding: 1,
            dilation: 1,
            groups: 1,
            weight: vec![0.0, 1.0, 0.0],
            bias: Some(vec![0.5]),
        };
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let output = conv.forward(&input);
        assert_eq!(output.len(), 4);
        for (out_val, in_val) in output.iter().zip(&input) {
            assert!((out_val - (in_val + 0.5)).abs() < 1e-6);
        }
    }
}
