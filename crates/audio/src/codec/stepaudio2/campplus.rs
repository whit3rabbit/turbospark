//! The StepAudio2 CAMPPlus speaker encoder.
//!
//! Reference: `mlx_audio/codec/models/stepaudio2/speaker.py`
//! (`StepAudio2CAMPPlus`, the fused single-conv variant matching the
//! folded ONNX export) and `s3gen/xvector.py` (`kaldi_fbank`,
//! `StatsPool`, `DenseLayer`), all at the pinned commit recorded in the
//! parent module.
//!
//! Structure: a Kaldi-style fbank front end (Povey window, DC removal,
//! pre-emphasis, HTK mel, log floored at float epsilon, per-feature
//! mean removal) feeds a 2D-conv ResNet head, a strided TDNN conv, and
//! three dense CAM blocks with SE-style context gating and transit
//! convs. A statistics pooling (mean + population std) and a BN-ReLU
//! dense head produce the 192-dim embedding, which the flow
//! L2-normalizes before its speaker affine.
//!
//! The fused variant has no batchnorm inside the conv head (the ONNX
//! export folded it into the conv weights); the TDNN path keeps
//! batchnorms in the CAM layers, transits, and dense head.

use turbospark_model_io::safetensors::SafetensorsFile;

use super::SaConv1d;
use crate::codec::wnconv::load_f32_shaped;
use crate::fft::RealFftPlan;
use crate::mel::{mel_filterbank, MelScale};
use crate::{Result, SpeechError};

/// Kaldi fbank geometry (`xvector.py::kaldi_fbank` at 16 kHz).
const FBANK_SAMPLE_RATE: u32 = 16_000;
const FBANK_FRAME_MS: f32 = 25.0;
const FBANK_SHIFT_MS: f32 = 10.0;
const FBANK_NUM_MELS: usize = 80;

/// CAMPPlus geometry (`StepAudio2CAMPPlus.__init__` defaults).
const FEAT_DIM: usize = 80;
const EMBEDDING_SIZE: usize = 192;
const GROWTH_RATE: usize = 32;
const BN_SIZE: usize = 4;
const INIT_CHANNELS: usize = 128;
const M_CHANNELS: usize = 32;
/// `(num_layers, kernel_size, dilation)` per dense block.
const BLOCK_SPECS: [(usize, usize, usize); 3] = [(12, 3, 1), (24, 3, 2), (16, 3, 2)];
const SEG_POOL_LEN: usize = 100;
const BN_EPS: f32 = 1e-5;

/// Kaldi fbank: `T x num_mels` log-mel rows
/// (`xvector.py::kaldi_fbank`). Frames are `snip_edges=True` (only
/// fully contained frames), each DC-removed and pre-emphasized, with
/// the Povey window (symmetric Hann to the 0.85) and the next power of
/// two FFT size.
pub fn kaldi_fbank(samples: &[f32]) -> Result<Vec<Vec<f32>>> {
    let win_length = (FBANK_SAMPLE_RATE as f32 * FBANK_FRAME_MS / 1000.0) as usize;
    let hop_length = (FBANK_SAMPLE_RATE as f32 * FBANK_SHIFT_MS / 1000.0) as usize;
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
    let filterbank = mel_filterbank(
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

/// Inference batch norm (`mlx.nn.BatchNorm`, running statistics).
#[derive(Debug, Clone)]
struct BatchNorm {
    ch: usize,
    scale: Vec<f32>,
    shift: Vec<f32>,
}

impl BatchNorm {
    fn load(file: &SafetensorsFile, prefix: &str, ch: usize, affine: bool) -> Result<Self> {
        let mean = load_f32_shaped(file, &format!("{prefix}.running_mean"), &[ch])?;
        let var = load_f32_shaped(file, &format!("{prefix}.running_var"), &[ch])?;
        let (weight, bias) = if affine {
            (
                Some(load_f32_shaped(file, &format!("{prefix}.weight"), &[ch])?),
                Some(load_f32_shaped(file, &format!("{prefix}.bias"), &[ch])?),
            )
        } else {
            (None, None)
        };
        let mut scale = vec![0.0f32; ch];
        let mut shift = vec![0.0f32; ch];
        for c in 0..ch {
            let inv = 1.0 / (var[c] + BN_EPS).sqrt();
            match (&weight, &bias) {
                (Some(w), Some(b)) => {
                    scale[c] = w[c] * inv;
                    shift[c] = b[c] - mean[c] * w[c] * inv;
                }
                _ => {
                    scale[c] = inv;
                    shift[c] = -mean[c] * inv;
                }
            }
        }
        Ok(BatchNorm { ch, scale, shift })
    }

    /// In-place over channel-major `[ch, T]`.
    fn forward(&self, x: &mut [f32], frames: usize) {
        for c in 0..self.ch {
            for v in &mut x[c * frames..(c + 1) * frames] {
                *v = *v * self.scale[c] + self.shift[c];
            }
        }
    }
}

/// Conv2d over NHWC `[H, W, C]` planes with the MLX stored weight
/// `[out, kh, kw, in]`, separate H/W strides and symmetric padding.
#[derive(Debug, Clone)]
struct Conv2dNhwc {
    in_ch: usize,
    out_ch: usize,
    kh: usize,
    kw: usize,
    stride_h: usize,
    stride_w: usize,
    pad: usize,
    weight: Vec<f32>,
    bias: Option<Vec<f32>>,
}

impl Conv2dNhwc {
    fn load(
        file: &SafetensorsFile,
        prefix: &str,
        in_ch: usize,
        out_ch: usize,
        kh: usize,
        kw: usize,
        stride_h: usize,
        stride_w: usize,
        pad: usize,
    ) -> Result<Self> {
        let weight = load_f32_shaped(file, &format!("{prefix}.weight"), &[out_ch, kh, kw, in_ch])?;
        let bias_name = format!("{prefix}.bias");
        let bias = if file.contains_tensor(&bias_name) {
            Some(load_f32_shaped(file, &bias_name, &[out_ch])?)
        } else {
            None
        };
        Ok(Conv2dNhwc {
            in_ch,
            out_ch,
            kh,
            kw,
            stride_h,
            stride_w,
            pad,
            weight,
            bias,
        })
    }

    /// `x` is `[h, w, in_ch]`; returns `[oh, ow, out_ch]`.
    fn forward(&self, x: &[f32], h: usize, w: usize) -> Vec<f32> {
        let oh = (h + 2 * self.pad - self.kh) / self.stride_h + 1;
        let ow = (w + 2 * self.pad - self.kw) / self.stride_w + 1;
        let mut out = vec![0.0f32; oh * ow * self.out_ch];
        for oc in 0..self.out_ch {
            for ohh in 0..oh {
                for oww in 0..ow {
                    let mut acc = self.bias.as_ref().map_or(0.0, |b| b[oc]);
                    for kh_i in 0..self.kh {
                        let ih = ohh * self.stride_h + kh_i;
                        if ih < self.pad || ih - self.pad >= h {
                            continue;
                        }
                        for kw_i in 0..self.kw {
                            let iw = oww * self.stride_w + kw_i;
                            if iw < self.pad || iw - self.pad >= w {
                                continue;
                            }
                            for ic in 0..self.in_ch {
                                let xv = x[(ih - self.pad) * w * self.in_ch
                                    + (iw - self.pad) * self.in_ch
                                    + ic];
                                let wv = self.weight
                                    [((oc * self.kh + kh_i) * self.kw + kw_i) * self.in_ch + ic];
                                acc += xv * wv;
                            }
                        }
                    }
                    out[(ohh * ow + oww) * self.out_ch + oc] = acc;
                }
            }
        }
        out
    }
}

/// `FusedBasicResBlock`: strided conv + ReLU, conv, shortcut conv when
/// the stride or width changes, ReLU. All convs are 3x3 except the 1x1
/// shortcut.
#[derive(Debug, Clone)]
struct FusedResBlock {
    conv1: Conv2dNhwc,
    conv2: Conv2dNhwc,
    shortcut: Option<Conv2dNhwc>,
}

impl FusedResBlock {
    fn load(
        file: &SafetensorsFile,
        prefix: &str,
        in_planes: usize,
        planes: usize,
        stride: usize,
    ) -> Result<Self> {
        let shortcut = if stride != 1 || in_planes != planes {
            Some(Conv2dNhwc::load(
                file,
                &format!("{prefix}.shortcut.0"),
                in_planes,
                planes,
                1,
                1,
                stride,
                1,
                0,
            )?)
        } else {
            None
        };
        Ok(FusedResBlock {
            conv1: Conv2dNhwc::load(
                file,
                &format!("{prefix}.conv1"),
                in_planes,
                planes,
                3,
                3,
                stride,
                1,
                1,
            )?,
            conv2: Conv2dNhwc::load(
                file,
                &format!("{prefix}.conv2"),
                planes,
                planes,
                3,
                3,
                1,
                1,
                1,
            )?,
            shortcut,
        })
    }

    fn forward(&self, x: &mut Vec<f32>, h: usize, w: usize) -> (usize, usize) {
        let mut out = self.conv1.forward(x, h, w);
        for v in &mut out {
            *v = v.max(0.0);
        }
        let oh = (h + 2 - 3) / self.conv1.stride_h + 1;
        let ow = (w + 2 - 3) / self.conv1.stride_w + 1;
        let out2 = self.conv2.forward(&out, oh, ow);
        // The conv2 and shortcut planes share the (oh, ow) geometry.
        let shortcut = match &self.shortcut {
            Some(sc) => sc.forward(x, h, w),
            None => out,
        };
        let mut merged = out2;
        for (v, s) in merged.iter_mut().zip(&shortcut) {
            *v += s;
        }
        for v in &mut merged {
            *v = v.max(0.0);
        }
        *x = merged;
        (oh, ow)
    }
}

/// `FusedFCM` head: conv, two stride-2 res layers, a final stride-2
/// conv, then the (C, H) channel merge to `[C * H, T]`.
#[derive(Debug, Clone)]
struct FusedFcm {
    conv1: Conv2dNhwc,
    layer1: Vec<FusedResBlock>,
    layer2: Vec<FusedResBlock>,
    conv2: Conv2dNhwc,
    out_channels: usize,
}

impl FusedFcm {
    fn load(file: &SafetensorsFile, prefix: &str) -> Result<Self> {
        let m = M_CHANNELS;
        let layer = |name: &str, num: usize| -> Result<Vec<FusedResBlock>> {
            let mut in_planes = m;
            let mut blocks = Vec::with_capacity(num);
            for i in 0..num {
                let stride = if i == 0 { 2 } else { 1 };
                blocks.push(FusedResBlock::load(
                    file,
                    &format!("{prefix}.{name}.{i}"),
                    in_planes,
                    m,
                    stride,
                )?);
                in_planes = m;
            }
            Ok(blocks)
        };
        Ok(FusedFcm {
            conv1: Conv2dNhwc::load(file, &format!("{prefix}.conv1"), 1, m, 3, 3, 1, 1, 1)?,
            layer1: layer("layer1", 2)?,
            layer2: layer("layer2", 2)?,
            conv2: Conv2dNhwc::load(file, &format!("{prefix}.conv2"), m, m, 3, 3, 2, 1, 1)?,
            out_channels: m * (FEAT_DIM / 8),
        })
    }

    /// `x` is the `[80, T]` fbank (used directly as the `[H, W, C]`
    /// planes with C = 1). Returns channel-major `[C * H', W]`.
    fn forward(&self, x: &[f32], t: usize) -> Vec<f32> {
        let h = FEAT_DIM;
        let planes = x.to_vec();
        let mut out = self.conv1.forward(&planes, h, t);
        for v in &mut out {
            *v = v.max(0.0);
        }
        let mut hw = (
            (h + 2 - 3) / self.conv1.stride_h + 1,
            (t + 2 - 3) / self.conv1.stride_w + 1,
        );
        for block in &self.layer1 {
            hw = block.forward(&mut out, hw.0, hw.1);
        }
        for block in &self.layer2 {
            hw = block.forward(&mut out, hw.0, hw.1);
        }
        let mut out = self.conv2.forward(&out, hw.0, hw.1);
        for v in &mut out {
            *v = v.max(0.0);
        }
        let oh = (hw.0 + 2 - 3) / self.conv2.stride_h + 1;
        let ow = (hw.1 + 2 - 3) / self.conv2.stride_w + 1;
        // NHWC [oh, ow, C] -> (C * oh, ow) channel-major.
        let mut cm = vec![0.0f32; self.out_channels * ow];
        for ohh in 0..oh {
            for oww in 0..ow {
                for c in 0..self.conv2.out_ch {
                    cm[(c * oh + ohh) * ow + oww] = out[(ohh * ow + oww) * self.conv2.out_ch + c];
                }
            }
        }
        cm
    }
}

/// SE context gating layer (`xvector.py::CAMLayer`).
#[derive(Debug, Clone)]
struct CamLayer {
    linear_local: SaConv1d,
    linear1: SaConv1d,
    linear2: SaConv1d,
}

impl CamLayer {
    fn load(
        file: &SafetensorsFile,
        prefix: &str,
        bn_channels: usize,
        out_channels: usize,
        kernel: usize,
        dilation: usize,
    ) -> Result<Self> {
        let padding = (kernel - 1) / 2 * dilation;
        Ok(CamLayer {
            linear_local: SaConv1d::load(
                file,
                &format!("{prefix}.linear_local"),
                bn_channels,
                out_channels,
                kernel,
                1,
                padding,
                dilation,
                1,
            )?,
            linear1: SaConv1d::load(
                file,
                &format!("{prefix}.linear1"),
                bn_channels,
                bn_channels / 2,
                1,
                1,
                0,
                1,
                1,
            )?,
            linear2: SaConv1d::load(
                file,
                &format!("{prefix}.linear2"),
                bn_channels / 2,
                out_channels,
                1,
                1,
                0,
                1,
                1,
            )?,
        })
    }

    /// Channel-major `[bn_channels, T]` in, `[out_channels, T]` out.
    fn forward(&self, x: &[f32], in_ch: usize, frames: usize) -> Vec<f32> {
        let mut y = self.linear_local.forward(x);
        // context = mean over time + segment pool, both of the input.
        let mut context = vec![0.0f32; in_ch];
        for c in 0..in_ch {
            context[c] = x[c * frames..(c + 1) * frames].iter().sum::<f32>() / frames as f32;
        }
        let pooled = seg_pool(x, in_ch, frames);
        for (c, v) in context.iter_mut().enumerate() {
            *v += pooled[c];
        }
        let mut ctx = self.linear1.forward(&context);
        for v in &mut ctx {
            *v = v.max(0.0);
        }
        let mut m = self.linear2.forward(&ctx);
        for v in &mut m {
            *v = 1.0 / (1.0 + (-*v).exp());
        }
        for (i, v) in y.iter_mut().enumerate() {
            *v *= m[i / frames];
        }
        y
    }
}

/// `seg_pooling`: average over 100-frame segments (ceil mode), then
/// broadcast back and truncate. Channel-major `[ch, T]`.
fn seg_pool(x: &[f32], ch: usize, frames: usize) -> Vec<f32> {
    let n_segs = (frames + SEG_POOL_LEN - 1) / SEG_POOL_LEN;
    let mut out = vec![0.0f32; ch * frames];
    for c in 0..ch {
        let row = &x[c * frames..(c + 1) * frames];
        for s in 0..n_segs {
            let start = s * SEG_POOL_LEN;
            let end = (start + SEG_POOL_LEN).min(frames);
            let mean = row[start..end].iter().sum::<f32>() / (end - start) as f32;
            for v in &mut out[c * frames + start..c * frames + end] {
                *v = mean;
            }
        }
    }
    out
}

/// `FusedCAMDenseTDNNLayer`: BN + ReLU, 1x1 conv, ReLU, CAM layer.
#[derive(Debug, Clone)]
struct CamDenseLayer {
    bn: BatchNorm,
    linear1: SaConv1d,
    cam: CamLayer,
    in_channels: usize,
    out_channels: usize,
}

impl CamDenseLayer {
    fn load(
        file: &SafetensorsFile,
        prefix: &str,
        in_channels: usize,
        out_channels: usize,
        bn_channels: usize,
        kernel: usize,
        dilation: usize,
    ) -> Result<Self> {
        Ok(CamDenseLayer {
            bn: BatchNorm::load(file, &format!("{prefix}.nonlinear1.0"), in_channels, true)?,
            linear1: SaConv1d::load(
                file,
                &format!("{prefix}.linear1"),
                in_channels,
                bn_channels,
                1,
                1,
                0,
                1,
                1,
            )?,
            cam: CamLayer::load(
                file,
                &format!("{prefix}.cam_layer"),
                bn_channels,
                out_channels,
                kernel,
                dilation,
            )?,
            in_channels,
            out_channels,
        })
    }

    /// Channel-major `[in, T]`; output `[out, T]`.
    fn forward(&self, x: &[f32], frames: usize) -> Vec<f32> {
        let mut h = x.to_vec();
        self.bn.forward(&mut h, frames);
        for v in &mut h {
            *v = v.max(0.0);
        }
        let mut h = self.linear1.forward(&h);
        for v in &mut h {
            *v = v.max(0.0);
        }
        self.cam.forward(&h, self.linear1.out_ch, frames)
    }
}

/// Dense CAM block: layers concatenate their outputs
/// (`x = concat([x, layer(x)])`).
#[derive(Debug, Clone)]
struct CamDenseBlock {
    layers: Vec<CamDenseLayer>,
}

impl CamDenseBlock {
    fn forward(&self, x: &mut Vec<f32>, in_channels: usize, frames: usize) -> usize {
        let mut channels = in_channels;
        for layer in &self.layers {
            let out = layer.forward(x, frames);
            x.extend_from_slice(&out);
            channels += layer.out_channels;
        }
        channels
    }
}

/// `FusedTransitLayer`: BN + ReLU + 1x1 conv.
#[derive(Debug, Clone)]
struct TransitLayer {
    bn: BatchNorm,
    linear: SaConv1d,
}

impl TransitLayer {
    fn load(
        file: &SafetensorsFile,
        prefix: &str,
        in_channels: usize,
        out_channels: usize,
        bias: bool,
    ) -> Result<Self> {
        // The last transit is the only one with a conv bias; the
        // loader picks the bias up from the checkpoint when present.
        let _ = bias;
        Ok(TransitLayer {
            bn: BatchNorm::load(file, &format!("{prefix}.nonlinear.0"), in_channels, true)?,
            linear: SaConv1d::load(
                file,
                &format!("{prefix}.linear"),
                in_channels,
                out_channels,
                1,
                1,
                0,
                1,
                1,
            )?,
        })
    }
}

/// The loaded CAMPPlus speaker encoder, batch-of-one.
pub struct StepAudio2CampPlus {
    head: FusedFcm,
    tdnn: SaConv1d,
    blocks: Vec<CamDenseBlock>,
    transits: Vec<TransitLayer>,
    dense_linear: SaConv1d,
    dense_bn: BatchNorm,
    embedding_size: usize,
}

impl StepAudio2CampPlus {
    /// Number of embedding dims.
    pub fn embedding_size(&self) -> usize {
        self.embedding_size
    }

    pub fn load(file: &SafetensorsFile) -> Result<Self> {
        let prefix = "campplus";
        let head = FusedFcm::load(file, &format!("{prefix}.head"))?;
        let channels = head.out_channels;
        let tdnn = SaConv1d::load(
            file,
            &format!("{prefix}.tdnn.linear"),
            channels,
            INIT_CHANNELS,
            5,
            2,
            2,
            1,
            1,
        )?;

        let mut blocks = Vec::with_capacity(BLOCK_SPECS.len());
        let mut transits = Vec::with_capacity(BLOCK_SPECS.len());
        let mut channels = INIT_CHANNELS;
        for (i, &(num_layers, kernel, dilation)) in BLOCK_SPECS.iter().enumerate() {
            let bn_channels = BN_SIZE * GROWTH_RATE;
            let mut in_channels = channels;
            let mut layers = Vec::with_capacity(num_layers);
            for l in 0..num_layers {
                layers.push(CamDenseLayer::load(
                    file,
                    &format!("{prefix}.blocks.{i}.layers.{l}"),
                    in_channels,
                    GROWTH_RATE,
                    bn_channels,
                    kernel,
                    dilation,
                )?);
                in_channels += GROWTH_RATE;
            }
            blocks.push(CamDenseBlock { layers });
            let out_channels = channels + num_layers * GROWTH_RATE;
            transits.push(TransitLayer::load(
                file,
                &format!("{prefix}.transits.{i}"),
                out_channels,
                out_channels / 2,
                i == BLOCK_SPECS.len() - 1,
            )?);
            channels = out_channels / 2;
        }

        let dense_linear = SaConv1d::load(
            file,
            &format!("{prefix}.dense.linear"),
            channels * 2,
            EMBEDDING_SIZE,
            1,
            1,
            0,
            1,
            1,
        )?;
        // `config_str="batchnorm_"`: batchnorm without affine.
        let dense_bn = BatchNorm::load(
            file,
            &format!("{prefix}.dense.nonlinear.0"),
            EMBEDDING_SIZE,
            false,
        )?;

        Ok(StepAudio2CampPlus {
            head,
            tdnn,
            blocks,
            transits,
            dense_linear,
            dense_bn,
            embedding_size: EMBEDDING_SIZE,
        })
    }

    /// Forward over a flat row-major fbank `[T * 80]`; returns the
    /// 192-dim embedding.
    pub fn forward(&self, fbank: &[f32]) -> Result<Vec<f32>> {
        if fbank.is_empty() || fbank.len() % FEAT_DIM != 0 {
            return Err(SpeechError::Input {
                why: format!("fbank rows must have {FEAT_DIM} bins"),
            });
        }
        let t = fbank.len() / FEAT_DIM;
        let cm = super::to_cm(fbank, t, FEAT_DIM);
        let mut x = self.head.forward(&cm, t);

        x = self.tdnn.forward(&x);
        let mut frames = x.len() / self.tdnn.out_ch;
        for v in &mut x {
            *v = v.max(0.0);
        }

        let mut channels = self.tdnn.out_ch;
        for (block, transit) in self.blocks.iter().zip(&self.transits) {
            block.forward(&mut x, channels, frames);
            // Transit: BN -> ReLU -> 1x1 conv.
            let mut b = std::mem::take(&mut x);
            transit.bn.forward(&mut b, frames);
            for v in &mut b {
                *v = v.max(0.0);
            }
            x = transit.linear.forward(&b);
            frames = x.len() / transit.linear.out_ch;
            channels = transit.linear.out_ch;
        }

        for v in &mut x {
            *v = v.max(0.0);
        }

        // Statistics pooling over time: [mean; std] per channel.
        let mut stats = vec![0.0f32; channels * 2];
        for c in 0..channels {
            let row = &x[c * frames..(c + 1) * frames];
            let mean = row.iter().sum::<f32>() / frames as f32;
            let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / frames as f32;
            stats[c] = mean;
            stats[channels + c] = (var + 1e-5).sqrt();
        }

        // Dense head: 1x1 conv (no bias) -> BN (no affine) -> ReLU.
        let mut out = self.dense_linear.forward(&stats);
        self.dense_bn.forward(&mut out, 1);
        for v in &mut out {
            *v = v.max(0.0);
        }
        Ok(out)
    }

    /// `inference`: kaldi fbank of one 16 kHz waveform, per-feature
    /// mean removal, forward.
    pub fn inference(&self, audio: &[f32]) -> Result<Vec<f32>> {
        let mut fbank = kaldi_fbank(audio)?;
        if fbank.is_empty() {
            return Err(SpeechError::Input {
                why: "audio too short for a single fbank frame".to_string(),
            });
        }
        let n = fbank.len();
        let mut mean = vec![0.0f32; FBANK_NUM_MELS];
        for frame in &fbank {
            for (m, v) in mean.iter_mut().zip(frame) {
                *m += v;
            }
        }
        for m in &mut mean {
            *m /= n as f32;
        }
        for frame in &mut fbank {
            for (v, m) in frame.iter_mut().zip(&mean) {
                *v -= m;
            }
        }
        let flat: Vec<f32> = fbank.into_iter().flatten().collect();
        let embedding = self.forward(&flat)?;
        debug_assert_eq!(embedding.len(), self.embedding_size());
        Ok(embedding)
    }
}
