//! EnCodec: SEANet-based streaming neural audio codec (Meta's EnCodec
//! as ported by mlx-audio).
//!
//! Reference: `mlx_audio/codec/models/encodec/encodec.py` at mlx-audio
//! [`e1b19b9054bf163f5d812221a54fcc346f1890e9`](https://github.com/Blaizzy/mlx-audio/tree/e1b19b9054bf163f5d812221a54fcc346f1890e9/mlx_audio/codec/models/encodec).
//! Causal convolutions with asymmetric extra padding and reflect
//! padding, SEANet resnet blocks with a compress bottleneck and 1x1
//! shortcut, a two-layer LSTM with residual add, and a Euclidean
//! (unnormalized) residual vector quantizer whose level count comes
//! from the target bandwidth.
//!
//! Deviations from the pinned reference (see the family README):
//! mono (`audio_channels == 1`) only, `norm_type == "weight_norm"` only
//! (the deployed checkpoint values), and the single-frame decode path;
//! chunked encode/decode with overlap-add is refused. The reference
//! consumes channels-last arrays; this port keeps its channel-major
//! layout and converts at the boundaries.

use std::path::Path;

use turbospark_model_io::safetensors::SafetensorsFile;

use crate::codec::conv::{
    load_bias, load_mlx_conv_weight, load_mlx_convt_weight, BiasMode, Conv1d, ConvTranspose1d,
};
use crate::codec::wnconv::load_f32_shaped;
use crate::ops;
use crate::{Result, SpeechError};

/// Pinned mlx-audio commit this port was transcribed from.
pub const REFERENCE_COMMIT: &str = "e1b19b9054bf163f5d812221a54fcc346f1890e9";

/// Checkpoint geometry, one-to-one with the reference dataclass.
#[derive(Debug, Clone)]
pub struct EncodecConfig {
    pub audio_channels: usize,
    pub num_filters: usize,
    pub kernel_size: usize,
    pub num_residual_layers: usize,
    pub dilation_growth_rate: usize,
    pub codebook_size: usize,
    pub codebook_dim: usize,
    pub hidden_size: usize,
    pub num_lstm_layers: usize,
    pub residual_kernel_size: usize,
    pub use_causal_conv: bool,
    pub normalize: bool,
    pub pad_mode: String,
    pub norm_type: String,
    pub last_kernel_size: usize,
    pub trim_right_ratio: f32,
    pub compress: usize,
    pub upsampling_ratios: Vec<usize>,
    pub target_bandwidths: Vec<f64>,
    pub sampling_rate: u32,
}

impl EncodecConfig {
    /// Parses the checkpoint `config.json` with the dataclass defaults;
    /// keys outside the dataclass are ignored (the reference filters
    /// the same way).
    pub fn from_json(value: &serde_json::Value) -> Result<Self> {
        let u =
            |field: &str| -> Result<Option<u64>> { Ok(value.get(field).and_then(|v| v.as_u64())) };
        let f =
            |field: &str| -> Result<Option<f64>> { Ok(value.get(field).and_then(|v| v.as_f64())) };
        let s = |field: &str| -> Result<Option<String>> {
            Ok(value
                .get(field)
                .and_then(|v| v.as_str())
                .map(|v| v.to_string()))
        };
        let usize_vec = |field: &str| -> Result<Option<Vec<usize>>> {
            match value.get(field) {
                None | Some(serde_json::Value::Null) => Ok(None),
                Some(v) => {
                    let arr = v.as_array().ok_or_else(|| SpeechError::BadConfig {
                        field: field.to_string(),
                        why: "expected a list".to_string(),
                    })?;
                    Ok(Some(
                        arr.iter()
                            .map(|x| x.as_u64().map(|n| n as usize))
                            .collect::<Option<Vec<_>>>()
                            .ok_or_else(|| SpeechError::BadConfig {
                                field: field.to_string(),
                                why: "expected a list of integers".to_string(),
                            })?,
                    ))
                }
            }
        };
        let f64_vec = |field: &str| -> Result<Option<Vec<f64>>> {
            match value.get(field) {
                None | Some(serde_json::Value::Null) => Ok(None),
                Some(v) => {
                    let arr = v.as_array().ok_or_else(|| SpeechError::BadConfig {
                        field: field.to_string(),
                        why: "expected a list".to_string(),
                    })?;
                    Ok(Some(
                        arr.iter()
                            .map(|x| x.as_f64())
                            .collect::<Option<Vec<_>>>()
                            .ok_or_else(|| SpeechError::BadConfig {
                                field: field.to_string(),
                                why: "expected a list of numbers".to_string(),
                            })?,
                    ))
                }
            }
        };
        Ok(EncodecConfig {
            audio_channels: u("audio_channels")?.unwrap_or(1) as usize,
            num_filters: u("num_filters")?.unwrap_or(32) as usize,
            kernel_size: u("kernel_size")?.unwrap_or(7) as usize,
            num_residual_layers: u("num_residual_layers")?.unwrap_or(1) as usize,
            dilation_growth_rate: u("dilation_growth_rate")?.unwrap_or(2) as usize,
            codebook_size: u("codebook_size")?.unwrap_or(1024) as usize,
            codebook_dim: u("codebook_dim")?.unwrap_or(128) as usize,
            hidden_size: u("hidden_size")?.unwrap_or(128) as usize,
            num_lstm_layers: u("num_lstm_layers")?.unwrap_or(2) as usize,
            residual_kernel_size: u("residual_kernel_size")?.unwrap_or(3) as usize,
            use_causal_conv: value
                .get("use_causal_conv")
                .and_then(|v| v.as_bool())
                .unwrap_or(true),
            normalize: value
                .get("normalize")
                .and_then(|v| v.as_bool())
                .unwrap_or(false),
            pad_mode: s("pad_mode")?.unwrap_or_else(|| "reflect".to_string()),
            norm_type: s("norm_type")?.unwrap_or_else(|| "weight_norm".to_string()),
            last_kernel_size: u("last_kernel_size")?.unwrap_or(7) as usize,
            trim_right_ratio: f("trim_right_ratio")?.unwrap_or(1.0) as f32,
            compress: u("compress")?.unwrap_or(2) as usize,
            upsampling_ratios: usize_vec("upsampling_ratios")?.unwrap_or_else(|| vec![8, 8, 4, 2]),
            target_bandwidths: f64_vec("target_bandwidths")?
                .unwrap_or_else(|| vec![1.5, 3.0, 6.0, 12.0, 24.0]),
            sampling_rate: u("sampling_rate")?.unwrap_or(24000) as u32,
        })
    }

    /// Total encoder downsampling; one latent frame per `hop_length`
    /// samples.
    pub fn hop_length(&self) -> usize {
        self.upsampling_ratios.iter().product()
    }
}

/// `(out, K, in)` of the conv weight at `{prefix}.conv.weight`, stored
/// MLX layout `[out, K, in]` (weight norm already folded at conversion).
fn plain_conv_dims(file: &SafetensorsFile, prefix: &str) -> Result<(usize, usize, usize)> {
    let v_name = format!("{prefix}.conv.weight");
    let desc = file
        .descriptor(&v_name)
        .ok_or_else(|| SpeechError::Tensor {
            name: v_name.clone(),
            why: "missing conv weight".to_string(),
        })?;
    if desc.shape.len() != 3 {
        return Err(SpeechError::Tensor {
            name: v_name.clone(),
            why: format!("expected 3-D weight, got {:?}", desc.shape),
        });
    }
    Ok((desc.shape[0], desc.shape[1], desc.shape[2]))
}

fn load_plain_conv(file: &SafetensorsFile, prefix: &str) -> Result<Conv1d> {
    let (out_ch, kernel, in_ch) = plain_conv_dims(file, prefix)?;
    let weight = load_mlx_conv_weight(
        file,
        &format!("{prefix}.conv.weight"),
        out_ch,
        kernel,
        in_ch,
    )?;
    let bias = load_bias(
        file,
        &format!("{prefix}.conv.bias"),
        out_ch,
        BiasMode::Optional,
    )?;
    Ok(Conv1d {
        in_ch,
        out_ch,
        kernel,
        stride: 1,
        padding: 0,
        dilation: 1,
        groups: 1,
        weight,
        bias,
    })
}

/// Reflect or zero padding of `x [ch, seq]`, matching the reference's
/// slicing (`prefix = x[1:pad_left+1][::-1]`, the edge sample is not
/// repeated).
fn pad1d(
    x: &[f32],
    ch: usize,
    pad_left: usize,
    pad_right: usize,
    reflect: bool,
) -> Result<Vec<f32>> {
    let seq = x.len() / ch;
    if !reflect {
        let mut out = vec![0.0f32; ch * (pad_left + seq + pad_right)];
        for c in 0..ch {
            out[c * (pad_left + seq + pad_right) + pad_left
                ..c * (pad_left + seq + pad_right) + pad_left + seq]
                .copy_from_slice(&x[c * seq..(c + 1) * seq]);
        }
        return Ok(out);
    }
    if seq == 0 {
        return Err(SpeechError::Input {
            why: "reflect padding of an empty sequence".to_string(),
        });
    }
    if pad_left >= seq || pad_right >= seq {
        return Err(SpeechError::Input {
            why: format!("reflect padding {pad_left}/{pad_right} exceeds sequence length {seq}"),
        });
    }
    let out_len = pad_left + seq + pad_right;
    let mut out = vec![0.0f32; ch * out_len];
    for c in 0..ch {
        let src = &x[c * seq..(c + 1) * seq];
        let dst = &mut out[c * out_len..(c + 1) * out_len];
        // prefix = x[1 .. pad_left + 1] reversed.
        for p in 0..pad_left {
            dst[p] = src[1 + pad_left - 1 - p];
        }
        dst[pad_left..pad_left + seq].copy_from_slice(src);
        // suffix = x[len - (pad_right + 1) .. len - 1] reversed.
        for p in 0..pad_right {
            dst[pad_left + seq + p] = src[seq - 2 - p];
        }
    }
    Ok(out)
}

/// Conv1d with causal or asymmetric extra padding (the reference
/// `EncodecConv1d`); `norm_type == "weight_norm"` means no norm module.
struct EncodecConv1d {
    conv: Conv1d,
    causal: bool,
    reflect: bool,
    /// Effective kernel with dilations.
    kernel_eff: usize,
    padding_total: usize,
    stride: usize,
}

impl EncodecConv1d {
    #[allow(clippy::too_many_arguments)]
    fn load(
        file: &SafetensorsFile,
        prefix: &str,
        kernel: usize,
        stride: usize,
        dilation: usize,
        causal: bool,
        reflect: bool,
    ) -> Result<Self> {
        let mut conv = load_plain_conv(file, prefix)?;
        conv.stride = stride;
        conv.dilation = dilation;
        Ok(EncodecConv1d {
            conv,
            causal,
            reflect,
            kernel_eff: (kernel - 1) * dilation + 1,
            padding_total: kernel - stride,
            stride,
        })
    }

    fn forward(&self, x: &[f32]) -> Result<Vec<f32>> {
        let ch = self.conv.in_ch;
        let len = x.len() / ch;
        let n_frames = ((len as f64 - self.kernel_eff as f64 + self.padding_total as f64)
            / self.stride as f64
            + 1.0)
            .ceil() as i64
            - 1;
        let ideal_length =
            n_frames.max(0) as usize * self.stride + self.kernel_eff - self.padding_total;
        let extra_padding = ideal_length.saturating_sub(len);
        let (left, right) = if self.causal {
            (self.padding_total, extra_padding)
        } else {
            let padding_right = self.padding_total / 2;
            (
                self.padding_total - padding_right,
                padding_right + extra_padding,
            )
        };
        let padded = pad1d(x, ch, left, right, self.reflect)?;
        Ok(self.conv.forward(&padded))
    }
}

/// ConvTranspose1d with the reference's post-conv trim.
struct EncodecConvT1d {
    conv: ConvTranspose1d,
    causal: bool,
    trim_right_ratio: f32,
    padding_total: usize,
}

impl EncodecConvT1d {
    fn load(
        file: &SafetensorsFile,
        prefix: &str,
        kernel: usize,
        stride: usize,
        causal: bool,
        trim_right_ratio: f32,
    ) -> Result<Self> {
        let (out_ch, k, in_ch) = plain_conv_dims(file, prefix)?;
        // MLX ConvTranspose1d stores [out, K, in] like conv1d, but the
        // op expects the PyTorch [in, out, K] orientation.
        let weight =
            load_mlx_convt_weight(file, &format!("{prefix}.conv.weight"), out_ch, k, in_ch)?;
        let bias = load_bias(
            file,
            &format!("{prefix}.conv.bias"),
            out_ch,
            BiasMode::Optional,
        )?;
        Ok(EncodecConvT1d {
            conv: ConvTranspose1d {
                in_ch,
                out_ch,
                kernel: k,
                stride,
                padding: 0,
                output_padding: 0,
                groups: 1,
                weight,
                bias,
            },
            causal,
            trim_right_ratio,
            padding_total: kernel - stride,
        })
    }

    fn forward(&self, x: &[f32]) -> Vec<f32> {
        let y = self.conv.forward(x);
        let ch = self.conv.out_ch;
        let out_len = y.len() / ch;
        let padding_right = if self.causal {
            (self.padding_total as f32 * self.trim_right_ratio).ceil() as usize
        } else {
            self.padding_total / 2
        };
        let padding_left = self.padding_total - padding_right;
        let end = out_len.saturating_sub(padding_right);
        let kept = end.saturating_sub(padding_left);
        let mut out = vec![0.0f32; ch * kept];
        for c in 0..ch {
            out[c * kept..(c + 1) * kept]
                .copy_from_slice(&y[c * out_len + padding_left..c * out_len + padding_left + kept]);
        }
        out
    }
}

/// One LSTM layer. The reference runs a Metal kernel per time step with
/// the standard i, f, g, o gate order and a numerically stable sigmoid;
/// the recurrence here is the portable equivalent.
struct Lstm {
    /// `[4H, in]`.
    wx: Vec<f32>,
    /// `[4H, H]`.
    wh: Vec<f32>,
    bias: Option<Vec<f32>>,
    hidden: usize,
    in_dim: usize,
}

impl Lstm {
    fn load(file: &SafetensorsFile, prefix: &str) -> Result<Self> {
        let wx_desc =
            file.descriptor(&format!("{prefix}.Wx"))
                .ok_or_else(|| SpeechError::Tensor {
                    name: format!("{prefix}.Wx"),
                    why: "missing LSTM Wx".to_string(),
                })?;
        if wx_desc.shape.len() != 2 {
            return Err(SpeechError::Tensor {
                name: format!("{prefix}.Wx"),
                why: format!("expected 2-D Wx, got {:?}", wx_desc.shape),
            });
        }
        let (four_h, in_dim) = (wx_desc.shape[0], wx_desc.shape[1]);
        let wx = load_f32_shaped(file, &format!("{prefix}.Wx"), &[four_h, in_dim])?;
        let wh = load_f32_shaped(file, &format!("{prefix}.Wh"), &[four_h, in_dim])?;
        let bias = if file.contains_tensor(&format!("{prefix}.bias")) {
            Some(load_f32_shaped(file, &format!("{prefix}.bias"), &[four_h])?)
        } else {
            None
        };
        Ok(Lstm {
            wx,
            wh,
            bias,
            hidden: in_dim,
            in_dim,
        })
    }

    /// `x [in_dim, T]` -> `[hidden, T]`.
    fn forward(&self, x: &[f32]) -> Vec<f32> {
        let t = x.len() / self.in_dim;
        let h = self.hidden;
        // Transpose once so each step's input projection reads one
        // contiguous row; the sums still walk `i` ascending, bias first.
        let mut x_rows = vec![0.0f32; t * self.in_dim];
        for i in 0..self.in_dim {
            for step in 0..t {
                x_rows[step * self.in_dim + i] = x[i * t + step];
            }
        }
        let mut out = vec![0.0f32; h * t];
        let mut cell = vec![0.0f32; h];
        // Hidden state of the previous step, kept contiguous (the strided
        // `out` column holds the same values).
        let mut h_prev = vec![0.0f32; h];
        // pre[j] = bias[j] + sum_i x[i, step] * wx[j, i] for this step.
        let mut pre = vec![0.0f32; 4 * h];
        // h_pre is the (4H) hidden-side gate input; the reference skips
        // the Wh multiply on step 0 (zeros), which is identical.
        let mut h_pre = vec![0.0f32; 4 * h];
        for step in 0..t {
            let x_row = &x_rows[step * self.in_dim..(step + 1) * self.in_dim];
            for j in 0..4 * h {
                let mut acc = self.bias.as_deref().map_or(0.0, |b| b[j]);
                let w_row = &self.wx[j * self.in_dim..(j + 1) * self.in_dim];
                for i in 0..self.in_dim {
                    acc += x_row[i] * w_row[i];
                }
                pre[j] = acc;
            }
            if step > 0 {
                // h_pre = h_prev @ wh.T: wh [4H, H], h_prev [H].
                for j in 0..4 * h {
                    let mut acc = 0.0f32;
                    let w_row = &self.wh[j * h..(j + 1) * h];
                    for i in 0..h {
                        acc += h_prev[i] * w_row[i];
                    }
                    h_pre[j] = acc;
                }
            }
            // Gates in the kernel's stable form. `h_pre` is complete, so
            // the new hidden value can overwrite `h_prev` in place.
            for i in 0..h {
                let pi = h_pre[i] + pre[i];
                let pf = h_pre[h + i] + pre[h + i];
                let pg = h_pre[2 * h + i] + pre[2 * h + i];
                let po = h_pre[3 * h + i] + pre[3 * h + i];
                let gi = stable_sigmoid(pi);
                let gf = stable_sigmoid(pf);
                let gg = pg.tanh();
                let go = stable_sigmoid(po);
                cell[i] = gf * cell[i] + gi * gg;
                let new_hidden = go * cell[i].tanh();
                h_prev[i] = new_hidden;
                out[i * t + step] = new_hidden;
            }
        }
        out
    }
}

/// The Metal kernel's sigmoid: `y = 1 / (1 + exp(-|x|))`, `x < 0` maps
/// to `1 - y`.
fn stable_sigmoid(x: f32) -> f32 {
    let y = 1.0 / (1.0 + (-x.abs()).exp());
    if x < 0.0 {
        1.0 - y
    } else {
        y
    }
}

/// Stacked LSTMs with a residual add (`EncodecLSTM`).
struct EncodecLstm {
    lstms: Vec<Lstm>,
}

impl EncodecLstm {
    fn load(file: &SafetensorsFile, prefix: &str, count: usize) -> Result<Self> {
        let mut lstms = Vec::with_capacity(count);
        for i in 0..count {
            lstms.push(Lstm::load(file, &format!("{prefix}.lstm.{i}"))?);
        }
        Ok(EncodecLstm { lstms })
    }

    fn forward(&self, x: &[f32]) -> Vec<f32> {
        let mut h = x.to_vec();
        for lstm in &self.lstms {
            h = lstm.forward(&h);
        }
        h.iter().zip(x).map(|(a, b)| a + b).collect()
    }
}

/// SEANet resnet block: two convs through a compress bottleneck with a
/// 1x1 shortcut (the reference defaults `use_conv_shortcut` on).
struct ResnetBlock {
    first: EncodecConv1d,
    second: EncodecConv1d,
    shortcut: EncodecConv1d,
}

impl ResnetBlock {
    #[allow(clippy::too_many_arguments)]
    fn load(
        file: &SafetensorsFile,
        prefix: &str,
        dim: usize,
        dilations: [usize; 2],
        kernel_sizes: [usize; 2],
        causal: bool,
        reflect: bool,
    ) -> Result<Self> {
        let first = EncodecConv1d::load(
            file,
            &format!("{prefix}.block.1"),
            kernel_sizes[0],
            1,
            dilations[0],
            causal,
            reflect,
        )?;
        if first.conv.in_ch != dim {
            return Err(SpeechError::Tensor {
                name: format!("{prefix}.block.1.conv.weight"),
                why: format!(
                    "resnet input {} contradicts block dim {dim}",
                    first.conv.in_ch
                ),
            });
        }
        let second = EncodecConv1d::load(
            file,
            &format!("{prefix}.block.3"),
            kernel_sizes[1],
            1,
            dilations[1],
            causal,
            reflect,
        )?;
        let shortcut = EncodecConv1d::load(
            file,
            &format!("{prefix}.shortcut"),
            1,
            1,
            1,
            causal,
            reflect,
        )?;
        Ok(ResnetBlock {
            first,
            second,
            shortcut,
        })
    }

    fn forward(&self, x: &[f32]) -> Result<Vec<f32>> {
        let mut h = x.to_vec();
        ops::elu(&mut h);
        let h = self.first.forward(&h)?;
        let mut h = h;
        ops::elu(&mut h);
        let h = self.second.forward(&h)?;
        let sc = self.shortcut.forward(x)?;
        Ok(sc.iter().zip(h).map(|(a, b)| a + b).collect())
    }
}

enum EncLayer {
    Conv(EncodecConv1d),
    Resnet(ResnetBlock),
    Lstm(EncodecLstm),
    Elu,
}

enum DecLayer {
    Conv(EncodecConv1d),
    ConvT(EncodecConvT1d),
    Resnet(ResnetBlock),
    Lstm(EncodecLstm),
    Elu,
}

struct Encoder {
    layers: Vec<EncLayer>,
}

impl Encoder {
    fn forward(&self, x: &[f32]) -> Result<Vec<f32>> {
        let mut h = x.to_vec();
        for layer in &self.layers {
            match layer {
                EncLayer::Conv(l) => h = l.forward(&h)?,
                EncLayer::Resnet(l) => h = l.forward(&h)?,
                EncLayer::Lstm(l) => h = l.forward(&h),
                EncLayer::Elu => ops::elu(&mut h),
            }
        }
        Ok(h)
    }
}

struct Decoder {
    layers: Vec<DecLayer>,
}

impl Decoder {
    fn forward(&self, x: &[f32]) -> Result<Vec<f32>> {
        let mut h = x.to_vec();
        for layer in &self.layers {
            match layer {
                DecLayer::Conv(l) => h = l.forward(&h)?,
                DecLayer::ConvT(l) => h = l.forward(&h),
                DecLayer::Resnet(l) => h = l.forward(&h)?,
                DecLayer::Lstm(l) => h = l.forward(&h),
                DecLayer::Elu => ops::elu(&mut h),
            }
        }
        Ok(h)
    }
}

/// Euclidean (unnormalized) nearest-neighbor codebook.
struct EuclideanCodebook {
    /// `[size, dim]`.
    embed: Vec<f32>,
    /// Per-row squared norms.
    embed_sq: Vec<f32>,
    size: usize,
    dim: usize,
}

impl EuclideanCodebook {
    fn load(file: &SafetensorsFile, prefix: &str) -> Result<Self> {
        let name = format!("{prefix}.codebook.embed");
        let desc = file.descriptor(&name).ok_or_else(|| SpeechError::Tensor {
            name: name.clone(),
            why: "missing codebook embed".to_string(),
        })?;
        if desc.shape.len() != 2 {
            return Err(SpeechError::Tensor {
                name: name.clone(),
                why: format!("expected 2-D embed, got {:?}", desc.shape),
            });
        }
        let (size, dim) = (desc.shape[0], desc.shape[1]);
        let embed = load_f32_shaped(file, &name, &[size, dim])?;
        let mut embed_sq = vec![0.0f32; size];
        for r in 0..size {
            embed_sq[r] = embed[r * dim..(r + 1) * dim].iter().map(|v| v * v).sum();
        }
        Ok(EuclideanCodebook {
            embed,
            embed_sq,
            size,
            dim,
        })
    }

    /// Nearest row for one `col [dim]` frame: argmin of the squared
    /// distance, first-wins ties (matching `argmax` of the negative).
    fn nearest(&self, col: &[f32]) -> usize {
        let e2: f32 = col.iter().map(|v| v * v).sum();
        let mut best = f32::INFINITY;
        let mut best_idx = 0usize;
        for r in 0..self.size {
            let row = &self.embed[r * self.dim..(r + 1) * self.dim];
            let mut dot = 0.0f32;
            for (a, &b) in col.iter().zip(row) {
                dot += a * b;
            }
            // dist = e2 - 2 dot + sq; argmax of -dist == argmin with <.
            let dist = e2 - 2.0 * dot + self.embed_sq[r];
            if dist < best {
                best = dist;
                best_idx = r;
            }
        }
        best_idx
    }

    fn row(&self, idx: usize) -> &[f32] {
        &self.embed[idx * self.dim..(idx + 1) * self.dim]
    }
}

/// Residual vector quantizer over the codebook stack; level count is
/// bandwidth-driven (`get_num_quantizers_for_bandwidth`).
struct ResidualVq {
    layers: Vec<EuclideanCodebook>,
    codebook_size: usize,
    frame_rate: usize,
    num_quantizers: usize,
}

impl ResidualVq {
    fn load(config: &EncodecConfig, file: &SafetensorsFile) -> Result<Self> {
        // frame_rate = ceil(sampling_rate / hop_length).
        let frame_rate = config.sampling_rate.div_ceil(config.hop_length() as u32) as usize;
        let num_quantizers = (1000.0 * config.target_bandwidths.last().copied().unwrap_or(24.0)
            / (frame_rate as f64 * 10.0)) as usize;
        let mut layers = Vec::with_capacity(num_quantizers);
        for i in 0..num_quantizers {
            layers.push(EuclideanCodebook::load(
                file,
                &format!("quantizer.layers.{i}"),
            )?);
        }
        Ok(ResidualVq {
            layers,
            codebook_size: config.codebook_size,
            frame_rate,
            num_quantizers,
        })
    }

    fn num_for_bandwidth(&self, bandwidth: f64) -> usize {
        let bw_per_q = (self.codebook_size as f64).log2() * self.frame_rate as f64;
        if bandwidth > 0.0 {
            ((bandwidth * 1000.0 / bw_per_q).floor() as usize).max(1)
        } else {
            self.num_quantizers
        }
    }

    /// `embeddings [dim, T]` -> per-level codes.
    fn encode(&self, embeddings: &[f32], dim: usize, bandwidth: f64) -> Vec<Vec<i32>> {
        let n_q = self.num_for_bandwidth(bandwidth);
        let frames = embeddings.len() / dim;
        let mut residual = embeddings.to_vec();
        let mut codes = Vec::with_capacity(n_q.min(self.layers.len()));
        for layer in self.layers.iter().take(n_q) {
            let mut indices = vec![0i32; frames];
            let mut quantized = vec![0.0f32; dim * frames];
            for f in 0..frames {
                let col: Vec<f32> = (0..dim).map(|d| residual[d * frames + f]).collect();
                let idx = layer.nearest(&col);
                indices[f] = idx as i32;
                for (d, &v) in layer.row(idx).iter().enumerate() {
                    quantized[d * frames + f] = v;
                }
            }
            for (r, q) in residual.iter_mut().zip(&quantized) {
                *r -= q;
            }
            codes.push(indices);
        }
        codes
    }

    /// Codes `[n_q][T]` -> quantized latents `[dim, T]`.
    fn decode(&self, codes: &[Vec<i32>], dim: usize) -> Result<Vec<f32>> {
        if codes.len() > self.layers.len() {
            return Err(SpeechError::Input {
                why: format!(
                    "{} code levels exceed the {} available",
                    codes.len(),
                    self.layers.len()
                ),
            });
        }
        let frames = codes.first().map(|c| c.len()).unwrap_or(0);
        let mut out = vec![0.0f32; dim * frames];
        for (layer, level) in self.layers.iter().zip(codes) {
            if level.len() != frames {
                return Err(SpeechError::Input {
                    why: "ragged code levels".to_string(),
                });
            }
            for (f, &code) in level.iter().enumerate() {
                let idx = code as usize;
                for (d, &v) in layer.row(idx).iter().enumerate() {
                    out[d * frames + f] += v;
                }
            }
        }
        Ok(out)
    }
}

/// Loaded EnCodec, batch-of-one, mono.
pub struct Encodec {
    pub config: EncodecConfig,
    encoder: Encoder,
    decoder: Decoder,
    quantizer: ResidualVq,
}

impl Encodec {
    /// Opens a checkpoint directory holding `config.json` plus
    /// `model.safetensors` (the `from_pretrained` layout).
    pub fn open(dir: &Path) -> Result<Encodec> {
        let config_text = std::fs::read_to_string(dir.join("config.json")).map_err(|e| {
            SpeechError::BadConfig {
                field: "config.json".to_string(),
                why: e.to_string(),
            }
        })?;
        let value: serde_json::Value =
            serde_json::from_str(&config_text).map_err(|e| SpeechError::BadConfig {
                field: "config.json".to_string(),
                why: e.to_string(),
            })?;
        let config = EncodecConfig::from_json(&value)?;
        let file = SafetensorsFile::open(&dir.join("model.safetensors"))?;
        Encodec::load(config, &file)
    }

    /// Loads the model from a parsed config and a safetensors file.
    pub fn load(config: EncodecConfig, file: &SafetensorsFile) -> Result<Encodec> {
        if config.audio_channels != 1 {
            return Err(SpeechError::Unsupported {
                why: format!(
                    "audio_channels = {}; this port is mono-only",
                    config.audio_channels
                ),
            });
        }
        if config.norm_type != "weight_norm" {
            return Err(SpeechError::Unsupported {
                why: format!(
                    "norm_type = {:?}; only the weight_norm (norm-free conv) \
                     checkpoints are verified",
                    config.norm_type
                ),
            });
        }
        let reflect = match config.pad_mode.as_str() {
            "reflect" => true,
            "constant" | "zero" => false,
            other => {
                return Err(SpeechError::Unsupported {
                    why: format!("pad_mode {other:?} not supported"),
                })
            }
        };
        let causal = config.use_causal_conv;
        let encoder = Self::load_encoder(&config, file, causal, reflect)?;
        let decoder = Self::load_decoder(&config, file, causal, reflect)?;
        let quantizer = ResidualVq::load(&config, file)?;
        Ok(Encodec {
            config,
            encoder,
            decoder,
            quantizer,
        })
    }

    fn load_encoder(
        config: &EncodecConfig,
        file: &SafetensorsFile,
        causal: bool,
        reflect: bool,
    ) -> Result<Encoder> {
        let mut layers = Vec::new();
        let mut index = 0usize;
        layers.push(EncLayer::Conv(EncodecConv1d::load(
            file,
            &format!("encoder.layers.{index}"),
            config.kernel_size,
            1,
            1,
            causal,
            reflect,
        )?));
        index += 1;
        let mut scaling = 1usize;
        for ratio in config.upsampling_ratios.iter().rev() {
            let current = scaling * config.num_filters;
            for j in 0..config.num_residual_layers {
                layers.push(EncLayer::Resnet(ResnetBlock::load(
                    file,
                    &format!("encoder.layers.{index}"),
                    current,
                    [config.dilation_growth_rate.pow(j as u32), 1],
                    [config.residual_kernel_size, 1],
                    causal,
                    reflect,
                )?));
                index += 1;
            }
            layers.push(EncLayer::Elu);
            index += 1;
            layers.push(EncLayer::Conv(EncodecConv1d::load(
                file,
                &format!("encoder.layers.{index}"),
                ratio * 2,
                *ratio,
                1,
                causal,
                reflect,
            )?));
            index += 1;
            scaling *= 2;
        }
        layers.push(EncLayer::Lstm(EncodecLstm::load(
            file,
            &format!("encoder.layers.{index}"),
            config.num_lstm_layers,
        )?));
        index += 1;
        layers.push(EncLayer::Elu);
        index += 1;
        layers.push(EncLayer::Conv(EncodecConv1d::load(
            file,
            &format!("encoder.layers.{index}"),
            config.last_kernel_size,
            1,
            1,
            causal,
            reflect,
        )?));
        Ok(Encoder { layers })
    }

    fn load_decoder(
        config: &EncodecConfig,
        file: &SafetensorsFile,
        causal: bool,
        reflect: bool,
    ) -> Result<Decoder> {
        let mut layers = Vec::new();
        let mut index = 0usize;
        let mut scaling = 1usize << config.upsampling_ratios.len();
        layers.push(DecLayer::Conv(EncodecConv1d::load(
            file,
            &format!("decoder.layers.{index}"),
            config.kernel_size,
            1,
            1,
            causal,
            reflect,
        )?));
        index += 1;
        layers.push(DecLayer::Lstm(EncodecLstm::load(
            file,
            &format!("decoder.layers.{index}"),
            config.num_lstm_layers,
        )?));
        index += 1;
        for ratio in &config.upsampling_ratios {
            let current = scaling * config.num_filters;
            layers.push(DecLayer::Elu);
            index += 1;
            layers.push(DecLayer::ConvT(EncodecConvT1d::load(
                file,
                &format!("decoder.layers.{index}"),
                ratio * 2,
                *ratio,
                causal,
                config.trim_right_ratio,
            )?));
            index += 1;
            for j in 0..config.num_residual_layers {
                layers.push(DecLayer::Resnet(ResnetBlock::load(
                    file,
                    &format!("decoder.layers.{index}"),
                    current / 2,
                    [config.dilation_growth_rate.pow(j as u32), 1],
                    [config.residual_kernel_size, 1],
                    causal,
                    reflect,
                )?));
                index += 1;
            }
            scaling /= 2;
        }
        layers.push(DecLayer::Elu);
        index += 1;
        layers.push(DecLayer::Conv(EncodecConv1d::load(
            file,
            &format!("decoder.layers.{index}"),
            config.last_kernel_size,
            1,
            1,
            causal,
            reflect,
        )?));
        Ok(Decoder { layers })
    }

    /// Frame rate of the latent codes.
    pub fn frame_rate(&self) -> usize {
        self.quantizer.frame_rate
    }

    /// Encodes mono samples into per-level codes. `bandwidth` (in
    /// kbps) must be one of `config.target_bandwidths`; `None` picks
    /// the first.
    pub fn encode(&self, samples: &[f32], bandwidth: Option<f64>) -> Result<Vec<Vec<i32>>> {
        if samples.is_empty() {
            return Err(SpeechError::Input {
                why: "empty waveform".to_string(),
            });
        }
        let bandwidth = bandwidth.unwrap_or(self.config.target_bandwidths[0]);
        if !self
            .config
            .target_bandwidths
            .iter()
            .any(|&b| b == bandwidth)
        {
            return Err(SpeechError::Input {
                why: format!(
                    "bandwidth {bandwidth} not in {:?}",
                    self.config.target_bandwidths
                ),
            });
        }
        let mut x = samples.to_vec();
        let mut scale = None;
        if self.config.normalize {
            let rms = (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32).sqrt() + 1e-8;
            for v in &mut x {
                *v /= rms;
            }
            scale = Some(rms);
        }
        let embeddings = self.encoder.forward(&x)?;
        let dim = self.config.hidden_size;
        let codes = self.quantizer.encode(&embeddings, dim, bandwidth);
        let _ = scale;
        Ok(codes)
    }

    /// Decodes code levels to mono samples. The output is exactly
    /// `frames * hop_length` long; callers trim to the original length.
    /// With `config.normalize`, pass the encode-time scale to undo it.
    pub fn decode_codes(&self, codes: &[Vec<i32>], scale: Option<f32>) -> Result<Vec<f32>> {
        let dim = self.config.hidden_size;
        let quantized = self.quantizer.decode(codes, dim)?;
        let mut audio = self.decoder.forward(&quantized)?;
        if let Some(s) = scale {
            for v in &mut audio {
                *v *= s;
            }
        }
        Ok(audio)
    }
}

#[cfg(test)]
mod tests;
