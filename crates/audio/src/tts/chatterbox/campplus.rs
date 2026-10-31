//! The chatterbox CAMPPlus speaker embedding
//! (`s3gen/xvector.py::CAMPPlus`): the kaldi fbank front end, the FCM
//! 2D-conv head with batchnorms, the strided TDNN, three dense TDNN
//! blocks with CAM squeeze-and-excitation, transit layers, statistics
//! pooling, and the affine-free batchnorm dense head. BatchNorm layers
//! run in inference mode against the stored running stats.

use std::path::Path;

use turbospark_model_io::safetensors::SafetensorsFile;

use crate::fft::RealFftPlan;
use crate::mel::{self, MelScale};
use crate::nn::load_tensor;
use crate::{Result, SpeechError};

/// Kaldi fbank geometry (the `kaldi_fbank` defaults at 16 kHz).
const FBANK_SAMPLE_RATE: u32 = 16_000;
const FBANK_FRAME_MS: f32 = 25.0;
const FBANK_SHIFT_MS: f32 = 10.0;
const FBANK_NUM_MELS: usize = 80;

/// `kaldi_fbank` (`s3gen/xvector.py`): snip-edges framing of a 16 kHz
/// waveform, per-frame DC removal, 0.97 pre-emphasis, Povey window
/// (hann^0.85, size-1 denominator), power spectrum on the next power of
/// two, HTK mel (fmin 20, fmax sr/2), log floored at f32 epsilon.
/// Returns rows `[frames, num_mels]`.
pub fn kaldi_fbank(samples: &[f32]) -> Result<Vec<Vec<f32>>> {
    let sample_rate = FBANK_SAMPLE_RATE as usize;
    let win_length = (sample_rate as f32 * FBANK_FRAME_MS / 1000.0) as usize;
    let hop_length = (sample_rate as f32 * FBANK_SHIFT_MS / 1000.0) as usize;
    let n_fft = win_length.next_power_of_two();
    let mut num_frames = (samples.len().saturating_sub(win_length)) / hop_length + 1;
    if samples.len() < win_length {
        num_frames = 1;
    }

    // Povey window: hann^0.85 with the size-1 denominator.
    let window: Vec<f32> = (0..win_length)
        .map(|n| {
            let hann =
                0.5 - 0.5 * (2.0 * std::f32::consts::PI * n as f32 / (win_length - 1) as f32).cos();
            hann.powf(0.85)
        })
        .collect();

    let mut frames = Vec::with_capacity(num_frames * n_fft);
    for f in 0..num_frames {
        let mut frame = vec![0.0f32; win_length];
        if samples.len() >= win_length {
            let start = f * hop_length;
            frame.copy_from_slice(&samples[start..start + win_length]);
        } else {
            let len = samples.len().min(win_length);
            frame[..len].copy_from_slice(&samples[..len]);
        }
        // DC removal, then pre-emphasis keeping the first sample.
        let mean = frame.iter().sum::<f32>() / win_length as f32;
        for v in &mut frame {
            *v -= mean;
        }
        for i in (1..win_length).rev() {
            frame[i] -= 0.97 * frame[i - 1];
        }
        for (v, w) in frame.iter_mut().zip(&window) {
            *v *= w;
        }
        frame.resize(n_fft, 0.0);
        frames.extend_from_slice(&frame);
    }

    let plan = RealFftPlan::new(n_fft)?;
    let filterbank = mel::mel_filterbank(
        FBANK_NUM_MELS,
        n_fft,
        FBANK_SAMPLE_RATE,
        20.0,
        Some(FBANK_SAMPLE_RATE as f32 / 2.0),
        MelScale::Htk,
    )?;
    let mut out = Vec::with_capacity(num_frames);
    for frame in frames.chunks(n_fft) {
        let spectrum = plan.forward(frame)?;
        let power: Vec<f32> = spectrum.iter().map(|c| c.re * c.re + c.im * c.im).collect();
        let mel = filterbank.project(&power)?;
        out.push(
            mel.into_iter()
                .map(|v| v.max(1.192_092_9e-7).ln())
                .collect(),
        );
    }
    Ok(out)
}

/// Geometry (the `CAMPPlus.__init__` defaults; the fixture shrinks the
/// dense-block widths).
#[derive(Debug, Clone)]
pub struct CampPlusConfig {
    pub feat_dim: usize,
    pub embedding_size: usize,
    /// The FCM head width.
    pub fcm_channels: usize,
    pub init_channels: usize,
    pub growth_rate: usize,
    pub bn_size: usize,
    /// Dense block layer counts (reference `(12, 24, 16)`).
    pub num_layers: [usize; 3],
}

impl Default for CampPlusConfig {
    fn default() -> Self {
        CampPlusConfig {
            feat_dim: 80,
            embedding_size: 192,
            fcm_channels: 32,
            init_channels: 128,
            growth_rate: 32,
            bn_size: 4,
            num_layers: [12, 24, 16],
        }
    }
}

impl CampPlusConfig {
    /// Parses a JSON object over the reference defaults.
    pub fn from_json_with_defaults(
        value: &serde_json::Value,
        defaults: CampPlusConfig,
    ) -> Result<Self> {
        let int = |name: &str, fallback: usize| -> Result<usize> {
            match value.get(name) {
                None | Some(serde_json::Value::Null) => Ok(fallback),
                Some(v) => v
                    .as_u64()
                    .map(|n| n as usize)
                    .ok_or_else(|| SpeechError::BadConfig {
                        field: name.to_string(),
                        why: "expected an integer".to_string(),
                    }),
            }
        };
        let layers = match value.get("num_layers") {
            None | Some(serde_json::Value::Null) => defaults.num_layers,
            Some(v) => {
                let arr = v.as_array().ok_or_else(|| SpeechError::BadConfig {
                    field: "num_layers".to_string(),
                    why: "expected an array".to_string(),
                })?;
                if arr.len() != 3 {
                    return Err(SpeechError::BadConfig {
                        field: "num_layers".to_string(),
                        why: "expected three entries".to_string(),
                    });
                }
                let mut out = [0usize; 3];
                for (slot, item) in out.iter_mut().zip(arr) {
                    *slot = item.as_u64().ok_or_else(|| SpeechError::BadConfig {
                        field: "num_layers".to_string(),
                        why: "expected integers".to_string(),
                    })? as usize;
                }
                out
            }
        };
        Ok(CampPlusConfig {
            feat_dim: int("feat_dim", defaults.feat_dim)?,
            embedding_size: int("embedding_size", defaults.embedding_size)?,
            fcm_channels: int("fcm_channels", defaults.fcm_channels)?,
            init_channels: int("init_channels", defaults.init_channels)?,
            growth_rate: int("growth_rate", defaults.growth_rate)?,
            bn_size: int("bn_size", defaults.bn_size)?,
            num_layers: layers,
        })
    }
}

/// Inference-mode batch norm over `[planes, n]` data (channel-first).
struct BatchNorm {
    weight: Option<Vec<f32>>,
    bias: Option<Vec<f32>>,
    mean: Vec<f32>,
    var: Vec<f32>,
    channels: usize,
}

impl BatchNorm {
    fn load(file: &SafetensorsFile, name: &str, channels: usize) -> Result<Self> {
        let mean = load_tensor(file, &format!("{name}.running_mean"), &[channels])?;
        let var = load_tensor(file, &format!("{name}.running_var"), &[channels])?;
        let (weight, bias) = if file.contains_tensor(&format!("{name}.weight")) {
            (
                Some(load_tensor(file, &format!("{name}.weight"), &[channels])?),
                Some(load_tensor(file, &format!("{name}.bias"), &[channels])?),
            )
        } else {
            (None, None)
        };
        Ok(BatchNorm {
            weight,
            bias,
            mean,
            var,
            channels,
        })
    }

    fn apply(&self, x: &mut [f32]) {
        let n = x.len() / self.channels;
        for c in 0..self.channels {
            let inv = 1.0 / (self.var[c] + 1e-5).sqrt();
            for value in &mut x[c * n..(c + 1) * n] {
                let norm = (*value - self.mean[c]) * inv;
                *value = match (&self.weight, &self.bias) {
                    (Some(w), Some(b)) => norm * w[c] + b[c],
                    _ => norm,
                };
            }
        }
    }
}

fn relu(x: &mut [f32]) {
    for value in x.iter_mut() {
        *value = value.max(0.0);
    }
}

/// 2D conv with an H-only stride (`stride=(s, 1)` in the reference),
/// channels-first `[in, h, w]`, PyTorch weight `[out, in, kh, kw]`
/// row-major, symmetric padding.
#[allow(clippy::too_many_arguments)]
fn conv2d_h_stride(
    x: &[f32],
    w: &[f32],
    in_ch: usize,
    out_ch: usize,
    h: usize,
    width: usize,
    kh: usize,
    kw: usize,
    stride_h: usize,
    pad: usize,
) -> Vec<f32> {
    let out_h = (h + 2 * pad - kh) / stride_h + 1;
    let out_w = width + 2 * pad - kw + 1;
    let mut out = vec![0.0f32; out_ch * out_h * out_w];
    for oc in 0..out_ch {
        for oh in 0..out_h {
            for ow in 0..out_w {
                let mut acc = 0.0f32;
                for ic in 0..in_ch {
                    for kh_i in 0..kh {
                        for kw_i in 0..kw {
                            let ih = oh * stride_h + kh_i;
                            let iw = ow + kw_i;
                            if ih >= h + pad || iw >= width + pad || (ih < pad) || (iw < pad) {
                                continue;
                            }
                            let ih = ih - pad;
                            let iw = iw - pad;
                            if ih >= h || iw >= width {
                                continue;
                            }
                            let x_value = x[(ic * h + ih) * width + iw];
                            let w_value = w[((oc * in_ch + ic) * kh + kh_i) * kw + kw_i];
                            acc += x_value * w_value;
                        }
                    }
                }
                out[(oc * out_h + oh) * out_w + ow] = acc;
            }
        }
    }
    out
}

/// 1x1 conv over `[channels, frames]` with optional bias.
fn conv1x1(
    x: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    in_ch: usize,
    out_ch: usize,
    frames: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; out_ch * frames];
    for (oc, w_row) in weight.chunks_exact(in_ch).enumerate() {
        for t in 0..frames {
            let mut acc = bias.map_or(0.0, |b| b[oc]);
            for (ic, w) in w_row.iter().enumerate() {
                acc += w * x[ic * frames + t];
            }
            out[oc * frames + t] = acc;
        }
    }
    out
}

/// A strided TDNN conv (`TDNNLayer`): k5 s2, batchnorm + relu. The
/// kernel slides over the time axis of `[channels, frames]`.
fn tdnn_conv(
    x: &[f32],
    w: &[f32],
    in_ch: usize,
    out_ch: usize,
    frames: usize,
    stride: usize,
) -> Vec<f32> {
    let kernel = 5;
    let pad = (kernel - 1) / 2;
    let out_frames = (frames + 2 * pad - kernel) / stride + 1;
    let mut out = vec![0.0f32; out_ch * out_frames];
    for oc in 0..out_ch {
        for ot in 0..out_frames {
            let mut acc = 0.0f32;
            for ic in 0..in_ch {
                for k in 0..kernel {
                    let t = ot * stride + k;
                    if t < pad || t >= frames + pad {
                        continue;
                    }
                    acc += x[ic * frames + (t - pad)] * w[(oc * in_ch + ic) * kernel + k];
                }
            }
            out[oc * out_frames + ot] = acc;
        }
    }
    out
}

/// MLX Conv1d stores `[out, k, in]`; the local convs run
/// `[out, in, k]` (kernel-major last).
fn load_conv1d_weight(
    file: &SafetensorsFile,
    name: &str,
    out_ch: usize,
    in_ch: usize,
    k: usize,
) -> Result<Vec<f32>> {
    let raw = load_tensor(file, name, &[out_ch, k, in_ch])?;
    let mut out = vec![0.0f32; raw.len()];
    for o in 0..out_ch {
        for i in 0..in_ch {
            for kk in 0..k {
                out[(o * in_ch + i) * k + kk] = raw[(o * k + kk) * in_ch + i];
            }
        }
    }
    Ok(out)
}

/// MLX Conv2d stores NHWC `[out, kh, kw, in]`; the local conv runs
/// channels-first `[out, in, kh, kw]`.
fn load_conv2d_weight(
    file: &SafetensorsFile,
    name: &str,
    out_ch: usize,
    in_ch: usize,
    kh: usize,
    kw: usize,
) -> Result<Vec<f32>> {
    let raw = load_tensor(file, name, &[out_ch, kh, kw, in_ch])?;
    let mut out = vec![0.0f32; raw.len()];
    for o in 0..out_ch {
        for h in 0..kh {
            for w in 0..kw {
                for i in 0..in_ch {
                    out[((o * in_ch + i) * kh + h) * kw + w] =
                        raw[((o * kh + h) * kw + w) * in_ch + i];
                }
            }
        }
    }
    Ok(out)
}

struct ResBlock {
    conv1: Vec<f32>,
    bn1: BatchNorm,
    conv2: Vec<f32>,
    bn2: BatchNorm,
    shortcut_conv: Option<Vec<f32>>,
    shortcut_bn: Option<BatchNorm>,
    channels: usize,
}

impl ResBlock {
    fn load(
        file: &SafetensorsFile,
        prefix: &str,
        in_planes: usize,
        planes: usize,
        stride: usize,
    ) -> Result<Self> {
        let conv1 = load_conv2d_weight(
            file,
            &format!("{prefix}.conv1.weight"),
            planes,
            in_planes,
            3,
            3,
        )?;
        let bn1 = BatchNorm::load(file, &format!("{prefix}.bn1"), planes)?;
        let conv2 = load_conv2d_weight(
            file,
            &format!("{prefix}.conv2.weight"),
            planes,
            planes,
            3,
            3,
        )?;
        let bn2 = BatchNorm::load(file, &format!("{prefix}.bn2"), planes)?;
        let has_shortcut = stride != 1 || in_planes != planes;
        let (shortcut_conv, shortcut_bn) = if has_shortcut {
            (
                Some(load_conv2d_weight(
                    file,
                    &format!("{prefix}.shortcut.0.weight"),
                    planes,
                    in_planes,
                    1,
                    1,
                )?),
                Some(BatchNorm::load(
                    file,
                    &format!("{prefix}.shortcut.1"),
                    planes,
                )?),
            )
        } else {
            (None, None)
        };
        Ok(ResBlock {
            conv1,
            bn1,
            conv2,
            bn2,
            shortcut_conv,
            shortcut_bn,
            channels: planes,
        })
    }

    /// `[in, h, w]` -> `[planes, h/stride, w]`. The k3 convs pad 1; the
    /// 1x1 shortcut conv is built without padding in the reference.
    fn forward(
        &self,
        x: &[f32],
        in_planes: usize,
        h: usize,
        width: usize,
        stride: usize,
    ) -> Vec<f32> {
        let out1 = conv2d_h_stride(
            x,
            &self.conv1,
            in_planes,
            self.channels,
            h,
            width,
            3,
            3,
            stride,
            1,
        );
        let mut out1 = out1;
        self.bn1.apply(&mut out1);
        relu(&mut out1);
        let out_h = (h + 2 - 3) / stride + 1;
        let out2 = conv2d_h_stride(
            &out1,
            &self.conv2,
            self.channels,
            self.channels,
            out_h,
            width,
            3,
            3,
            1,
            1,
        );
        let mut out2 = out2;
        self.bn2.apply(&mut out2);
        let shortcut = match (&self.shortcut_conv, &self.shortcut_bn) {
            (Some(conv), Some(bn)) => {
                let s =
                    conv2d_h_stride(x, conv, in_planes, self.channels, h, width, 1, 1, stride, 0);
                let mut s = s;
                bn.apply(&mut s);
                s
            }
            _ => x.to_vec(),
        };
        for (o, s) in out2.iter_mut().zip(&shortcut) {
            *o += s;
        }
        relu(&mut out2);
        out2
    }
}

struct CamLayer {
    nonlinear1: BatchNorm,
    linear1: Vec<f32>,
    nonlinear2: BatchNorm,
    linear_local: Vec<f32>,
    cam_linear1: Vec<f32>,
    cam_linear1_bias: Vec<f32>,
    cam_linear2: Vec<f32>,
    cam_linear2_bias: Vec<f32>,
    in_channels: usize,
    bn_channels: usize,
    out_channels: usize,
    kernel: usize,
    dilation: usize,
}

/// Loaded CAMPPlus speaker encoder (`speaker_encoder.` subtree).
pub struct CampPlus {
    config: CampPlusConfig,
    conv1: Vec<f32>,
    bn1: BatchNorm,
    layer1: Vec<ResBlock>,
    layer2: Vec<ResBlock>,
    conv2: Vec<f32>,
    bn2: BatchNorm,
    tdnn: Vec<f32>,
    tdnn_bn: BatchNorm,
    blocks: Vec<Vec<CamLayer>>,
    transits: Vec<(BatchNorm, Vec<f32>)>,
    out_bn: BatchNorm,
    dense: Vec<f32>,
    dense_bn: BatchNorm,
    tdnn_in: usize,
    block_out: usize,
}

impl CampPlus {
    pub fn open(path: &Path, config: CampPlusConfig) -> Result<Self> {
        let file = SafetensorsFile::open(path).map_err(SpeechError::from)?;
        Self::load(&file, "", config)
    }

    pub fn load(file: &SafetensorsFile, prefix: &str, config: CampPlusConfig) -> Result<Self> {
        let m = config.fcm_channels;
        let conv1 = load_conv2d_weight(file, &format!("{prefix}head.conv1.weight"), m, 1, 3, 3)?;
        let bn1 = BatchNorm::load(file, &format!("{prefix}head.bn1"), m)?;
        let mut layer1 = Vec::new();
        let mut in_planes = m;
        for (index, stride) in [2usize, 1].iter().enumerate() {
            layer1.push(ResBlock::load(
                file,
                &format!("{prefix}head.layer1.{index}"),
                in_planes,
                m,
                *stride,
            )?);
            in_planes = m;
        }
        let mut layer2 = Vec::new();
        for (index, stride) in [2usize, 1].iter().enumerate() {
            layer2.push(ResBlock::load(
                file,
                &format!("{prefix}head.layer2.{index}"),
                in_planes,
                m,
                *stride,
            )?);
            in_planes = m;
        }
        let conv2 = load_conv2d_weight(file, &format!("{prefix}head.conv2.weight"), m, m, 3, 3)?;
        let bn2 = BatchNorm::load(file, &format!("{prefix}head.bn2"), m)?;
        // After the FCM the channel count is m * (feat_dim / 8).
        let tdnn_in = m * (config.feat_dim / 8);
        let tdnn = load_conv1d_weight(
            file,
            &format!("{prefix}tdnn.linear.weight"),
            config.init_channels,
            tdnn_in,
            5,
        )?;
        let tdnn_bn = BatchNorm::load(
            file,
            &format!("{prefix}tdnn.nonlinear.0"),
            config.init_channels,
        )?;
        let bn_channels = config.bn_size * config.growth_rate;
        let mut blocks: Vec<Vec<CamLayer>> = Vec::new();
        let mut transits: Vec<(BatchNorm, Vec<f32>)> = Vec::new();
        let mut channels = config.init_channels;
        for (block_index, &num_layers) in config.num_layers.iter().enumerate() {
            let mut layers = Vec::new();
            for layer_index in 0..num_layers {
                let p = format!("{prefix}blocks.{block_index}.layers.{layer_index}");
                let in_channels = channels + layer_index * config.growth_rate;
                layers.push(CamLayer {
                    nonlinear1: BatchNorm::load(file, &format!("{p}.nonlinear1.0"), in_channels)?,
                    linear1: load_conv1d_weight(
                        file,
                        &format!("{p}.linear1.weight"),
                        bn_channels,
                        in_channels,
                        1,
                    )?,
                    nonlinear2: BatchNorm::load(file, &format!("{p}.nonlinear2.0"), bn_channels)?,
                    linear_local: load_conv1d_weight(
                        file,
                        &format!("{p}.cam_layer.linear_local.weight"),
                        config.growth_rate,
                        bn_channels,
                        3,
                    )?,
                    cam_linear1: load_conv1d_weight(
                        file,
                        &format!("{p}.cam_layer.linear1.weight"),
                        bn_channels / 2,
                        bn_channels,
                        1,
                    )?,
                    cam_linear1_bias: load_tensor(
                        file,
                        &format!("{p}.cam_layer.linear1.bias"),
                        &[bn_channels / 2],
                    )?,
                    cam_linear2: load_conv1d_weight(
                        file,
                        &format!("{p}.cam_layer.linear2.weight"),
                        config.growth_rate,
                        bn_channels / 2,
                        1,
                    )?,
                    cam_linear2_bias: load_tensor(
                        file,
                        &format!("{p}.cam_layer.linear2.bias"),
                        &[config.growth_rate],
                    )?,
                    in_channels,
                    bn_channels,
                    out_channels: config.growth_rate,
                    kernel: 3,
                    dilation: [1usize, 2, 2][block_index],
                });
            }
            blocks.push(layers);
            channels += num_layers * config.growth_rate;
            // Transit: bn + relu + 1x1 conv halving.
            let transit_bn = BatchNorm::load(
                file,
                &format!("{prefix}transits.{block_index}.nonlinear.0"),
                channels,
            )?;
            let transit_linear = load_conv1d_weight(
                file,
                &format!("{prefix}transits.{block_index}.linear.weight"),
                channels / 2,
                channels,
                1,
            )?;
            transits.push((transit_bn, transit_linear));
            channels /= 2;
        }
        let out_bn = BatchNorm::load(file, &format!("{prefix}out_nonlinear.0"), channels)?;
        let dense = load_conv1d_weight(
            file,
            &format!("{prefix}dense.linear.weight"),
            config.embedding_size,
            channels * 2,
            1,
        )?;
        let dense_bn = BatchNorm::load(
            file,
            &format!("{prefix}dense.nonlinear.0"),
            config.embedding_size,
        )?;
        Ok(CampPlus {
            config,
            conv1,
            bn1,
            layer1,
            layer2,
            conv2,
            bn2,
            tdnn,
            tdnn_bn,
            blocks,
            transits,
            out_bn,
            dense,
            dense_bn,
            tdnn_in,
            block_out: channels,
        })
    }

    /// `inference`: the mean-normalized kaldi fbank `[frames, feat_dim]`
    /// -> the embedding `[embedding_size]`.
    pub fn inference(&self, fbank: &[f32], frames: usize) -> Vec<f32> {
        let feat_dim = self.config.feat_dim;
        let m = self.config.fcm_channels;
        // FCM: the reference treats the mel bins as H, time as W, with
        // H-only strides. Channels-first [C, H, W].
        let h0 = feat_dim;
        let w0 = frames;
        let mut x = vec![0.0f32; h0 * w0];
        for h in 0..h0 {
            for w in 0..w0 {
                x[h * w0 + w] = fbank[w * feat_dim + h];
            }
        }
        let c1 = conv2d_h_stride(&x, &self.conv1, 1, m, h0, w0, 3, 3, 1, 1);
        let mut c1 = c1;
        self.bn1.apply(&mut c1);
        relu(&mut c1);
        let mut t = c1;
        let mut h = h0;
        let mut in_planes = m;
        for (index, block) in self.layer1.iter().enumerate() {
            let stride = if index == 0 { 2 } else { 1 };
            t = block.forward(&t, in_planes, h, w0, stride);
            h = (h + 2 - 3) / stride + 1;
            in_planes = m;
        }
        for (index, block) in self.layer2.iter().enumerate() {
            let stride = if index == 0 { 2 } else { 1 };
            t = block.forward(&t, in_planes, h, w0, stride);
            h = (h + 2 - 3) / stride + 1;
            in_planes = m;
        }
        let c2 = conv2d_h_stride(&t, &self.conv2, in_planes, m, h, w0, 3, 3, 2, 1);
        let mut c2 = c2;
        self.bn2.apply(&mut c2);
        relu(&mut c2);
        let h_out = (h + 2 - 3) / 2 + 1;
        // The conv output flattened is already the reference (B, C', H')
        // planes over W in [C * H'] order, so no gather is needed.
        debug_assert_eq!(c2.len(), m * h_out * w0);
        let mut t = c2;
        let mut channels = self.tdnn_in;
        // TDNN k5 s2 + bn + relu over [channels, frames].
        {
            let tdnn_out = tdnn_conv(&t, &self.tdnn, channels, self.config.init_channels, w0, 2);
            let mut tdnn_out = tdnn_out;
            self.tdnn_bn.apply(&mut tdnn_out);
            relu(&mut tdnn_out);
            t = tdnn_out;
            channels = self.config.init_channels;
        }
        let frames_now = t.len() / channels;
        for (block_index, layers) in self.blocks.iter().enumerate() {
            // Dense: x = concat(x, layer(x)) per layer.
            for layer in layers {
                let out = cam_forward(layer, &t, frames_now);
                t.extend_from_slice(&out);
            }
            // Transit: bn + relu + 1x1 conv halving.
            let (bn, linear) = &self.transits[block_index];
            let mut bn_out = t.clone();
            bn.apply(&mut bn_out);
            relu(&mut bn_out);
            let in_ch = bn.channels;
            let out_ch = linear.len() / in_ch;
            t = conv1x1(&bn_out, linear, None, in_ch, out_ch, frames_now);
            channels = out_ch;
            let _ = channels;
        }
        let mut out = t;
        self.out_bn.apply(&mut out);
        relu(&mut out);
        // Statistics pooling over time: mean and std (sqrt(var + 1e-5)).
        let frames_out = out.len() / self.block_out;
        let mut stats = vec![0.0f32; self.block_out * 2];
        for c in 0..self.block_out {
            let mut mean = 0.0f32;
            for f in 0..frames_out {
                mean += out[c * frames_out + f];
            }
            mean /= frames_out as f32;
            let mut var = 0.0f32;
            for f in 0..frames_out {
                let d = out[c * frames_out + f] - mean;
                var += d * d;
            }
            stats[c] = mean;
            stats[self.block_out + c] = (var / frames_out as f32 + 1e-5).sqrt();
        }
        // Dense: [channels*2] -> embedding over the single segment
        // (weight [embedding, channels*2, 1]).
        let mut embedding = vec![0.0f32; self.config.embedding_size];
        for (e, w_row) in self.dense.chunks_exact(self.block_out * 2).enumerate() {
            let mut acc = 0.0f32;
            for (i, w) in w_row.iter().enumerate() {
                acc += w * stats[i];
            }
            embedding[e] = acc;
        }
        // The dense head config is "batchnorm_": batchnorm only,
        // no trailing relu.
        self.dense_bn.apply(&mut embedding);
        embedding
    }
}

fn cam_forward(layer: &CamLayer, x: &[f32], frames: usize) -> Vec<f32> {
    // nonlinear1 (bn + relu) on [in_ch, frames].
    let mut h = x.to_vec();
    layer.nonlinear1.apply(&mut h);
    relu(&mut h);
    // linear1 1x1.
    let mut h = conv1x1(
        &h,
        &layer.linear1,
        None,
        layer.in_channels,
        layer.bn_channels,
        frames,
    );
    layer.nonlinear2.apply(&mut h);
    relu(&mut h);
    // CAM: local conv k3 with dilation, gated by the context branch.
    let pad = layer.dilation;
    let out_frames = h.len() / layer.bn_channels;
    let mut y = vec![0.0f32; layer.out_channels * out_frames];
    for oc in 0..layer.out_channels {
        for t in 0..out_frames {
            let mut acc = 0.0f32;
            for ic in 0..layer.bn_channels {
                for k in 0..layer.kernel {
                    let src = t as i64 + k as i64 * layer.dilation as i64 - pad as i64;
                    if src < 0 || src as usize >= out_frames {
                        continue;
                    }
                    acc += h[ic * out_frames + src as usize]
                        * layer.linear_local[(oc * layer.bn_channels + ic) * layer.kernel + k];
                }
            }
            y[oc * out_frames + t] = acc;
        }
    }
    // Context: mean over time + segment pooling, then 1x1 relu sigmoid.
    let mut context = vec![0.0f32; layer.bn_channels];
    for c in 0..layer.bn_channels {
        context[c] =
            h[c * out_frames..(c + 1) * out_frames].iter().sum::<f32>() / out_frames as f32;
    }
    // Segment pooling: the reference zero-pads the tail to a whole
    // number of segments, so every segment mean divides by seg_len.
    let seg_len = 100;
    let n_segs = out_frames.div_ceil(seg_len);
    for c in 0..layer.bn_channels {
        let mut seg_pool = vec![0.0f32; out_frames];
        for s in 0..n_segs {
            let start = s * seg_len;
            let end = (start + seg_len).min(out_frames);
            let mut acc = 0.0f32;
            for t in start..end {
                acc += h[c * out_frames + t];
            }
            let mean = acc / seg_len as f32;
            seg_pool[start..end].fill(mean);
        }
        context[c] += seg_pool.iter().sum::<f32>() / out_frames as f32;
    }
    // context [bn_channels] -> linear1 (bn/2) -> relu -> linear2 (out)
    // -> sigmoid, broadcast over time.
    let mid = layer.bn_channels / 2;
    let mut ctx = vec![0.0f32; mid];
    for (oc, w_row) in layer
        .cam_linear1
        .chunks_exact(layer.bn_channels)
        .enumerate()
    {
        let mut acc = layer.cam_linear1_bias[oc];
        for (ic, w) in w_row.iter().enumerate() {
            acc += w * context[ic];
        }
        ctx[oc] = acc.max(0.0);
    }
    let mut m = vec![0.0f32; layer.out_channels];
    for (oc, w_row) in layer.cam_linear2.chunks_exact(mid).enumerate() {
        let mut acc = layer.cam_linear2_bias[oc];
        for (ic, w) in w_row.iter().enumerate() {
            acc += w * ctx[ic];
        }
        m[oc] = 1.0 / (1.0 + (-acc).exp());
    }
    for (oc, m_value) in m.iter().enumerate() {
        for t in 0..out_frames {
            y[oc * out_frames + t] *= m_value;
        }
    }
    y
}
