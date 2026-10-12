//! The WAV-VAE codec for AudioDiT (`vae.py`): a channels-last conv
//! autoencoder (24 kHz, downsampling ratio = product of the strides,
//! encoder latent twice the model latent after the mean/scale split)
//! with SnakeBeta activations, residual units, pixel
//! shuffle/unshuffle shortcuts, and the reparameterization draw in the
//! encoder.

use crate::{Result, SpeechError};
use turbospark_model_io::safetensors::SafetensorsFile;

fn tensor(file: &SafetensorsFile, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
    let descriptor = file.descriptor(name).ok_or_else(|| SpeechError::Tensor {
        name: name.to_owned(),
        why: "tensor is missing".into(),
    })?;
    if descriptor.shape.iter().product::<usize>() != shape.iter().product::<usize>() {
        return Err(SpeechError::Tensor {
            name: name.to_owned(),
            why: format!("expected {shape:?} elements, got {:?}", descriptor.shape),
        });
    }
    crate::nn::load_tensor(file, name, &descriptor.shape)
}

/// Channels-last conv1d with the MLX weight layout `[out, kernel,
/// in]` (groups = 1).
#[allow(
    clippy::too_many_arguments,
    reason = "conv kernels carry the tensor geometry explicitly"
)]
fn conv1d(
    x: &[f32],
    seq: usize,
    in_ch: usize,
    w: &[f32],
    bias: Option<&[f32]>,
    out_ch: usize,
    kernel: usize,
    dilation: usize,
    stride: usize,
    padding: usize,
) -> Vec<f32> {
    let k_eff = (kernel - 1) * dilation + 1;
    let out_seq = (seq + 2 * padding - k_eff) / stride + 1;
    let mut out = vec![0.0f32; out_seq * out_ch];
    for t in 0..out_seq {
        for o in 0..out_ch {
            let mut acc = bias.map_or(0.0, |b| b[o]);
            for k in 0..kernel {
                let src = t as isize * stride as isize + (k * dilation) as isize - padding as isize;
                if src < 0 || src as usize >= seq {
                    continue;
                }
                let base = src as usize * in_ch;
                let w_base = (o * kernel + k) * in_ch;
                for c in 0..in_ch {
                    acc += x[base + c] * w[w_base + c];
                }
            }
            out[t * out_ch + o] = acc;
        }
    }
    out
}

/// Channels-last conv transpose with the MLX weight layout `[out,
/// kernel, in]`; output length `(seq - 1) * stride - 2 * padding +
/// kernel` (output_padding 0, the reference call).
#[allow(
    clippy::too_many_arguments,
    reason = "conv kernels carry the tensor geometry explicitly"
)]
fn conv_transpose1d(
    x: &[f32],
    seq: usize,
    in_ch: usize,
    w: &[f32],
    bias: Option<&[f32]>,
    out_ch: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
) -> Vec<f32> {
    let out_seq = (seq - 1) * stride + kernel - 2 * padding;
    let mut out = vec![0.0f32; out_seq * out_ch];
    for t in 0..seq {
        for k in 0..kernel {
            let dst = t * stride + k;
            if dst < padding || dst >= padding + out_seq {
                continue;
            }
            let o = dst - padding;
            for i in 0..in_ch {
                let xv = x[t * in_ch + i];
                if xv == 0.0 {
                    continue;
                }
                for oc in 0..out_ch {
                    out[o * out_ch + oc] += xv * w[(oc * kernel + k) * in_ch + i];
                }
            }
        }
    }
    if let Some(b) = bias {
        for frame in out.chunks_exact_mut(out_ch) {
            for (o, v) in frame.iter_mut().enumerate() {
                *v += b[o];
            }
        }
    }
    out
}

/// `pixel_unshuffle_1d`: `[seq, in_ch] -> [seq / f, in_ch * f]` with
/// channel `c * f + j` reading source position `w * f + j`.
fn pixel_unshuffle(x: &[f32], seq: usize, in_ch: usize, factor: usize) -> (Vec<f32>, usize, usize) {
    let out_seq = seq / factor;
    let out_ch = in_ch * factor;
    let mut out = vec![0.0f32; out_seq * out_ch];
    for w in 0..out_seq {
        for c in 0..in_ch {
            for j in 0..factor {
                out[w * out_ch + c * factor + j] = x[(w * factor + j) * in_ch + c];
            }
        }
    }
    (out, out_seq, out_ch)
}

/// `pixel_shuffle_1d`: `[seq, in_ch] -> [seq * f, in_ch / f]`.
fn pixel_shuffle(x: &[f32], seq: usize, in_ch: usize, factor: usize) -> (Vec<f32>, usize, usize) {
    let out_ch = in_ch / factor;
    let mut out = vec![0.0f32; seq * factor * out_ch];
    for w in 0..seq {
        for c2 in 0..out_ch {
            for j in 0..factor {
                out[(w * factor + j) * out_ch + c2] = x[w * in_ch + c2 * factor + j];
            }
        }
    }
    (out, seq * factor, out_ch)
}

/// `DownsampleShortcut`: unshuffle then group-average down to
/// `out_channels`.
fn downsample_shortcut(
    x: &[f32],
    seq: usize,
    in_ch: usize,
    out_ch: usize,
    factor: usize,
) -> Vec<f32> {
    let (unshuffled, n, wide) = pixel_unshuffle(x, seq, in_ch, factor);
    let group = in_ch * factor / out_ch;
    let mut out = vec![0.0f32; n * out_ch];
    for t in 0..n {
        for o in 0..out_ch {
            let mut acc = 0.0f32;
            for g in 0..group {
                acc += unshuffled[t * wide + o * group + g];
            }
            out[t * out_ch + o] = acc / group as f32;
        }
    }
    out
}

/// `UpsampleShortcut`: repeat channels then pixel shuffle.
fn upsample_shortcut(
    x: &[f32],
    seq: usize,
    in_ch: usize,
    out_ch: usize,
    factor: usize,
) -> Vec<f32> {
    let repeats = out_ch * factor / in_ch;
    let mut repeated = vec![0.0f32; seq * in_ch * repeats];
    for t in 0..seq {
        for (c, &v) in x[t * in_ch..(t + 1) * in_ch].iter().enumerate() {
            for r in 0..repeats {
                repeated[t * in_ch * repeats + c * repeats + r] = v;
            }
        }
    }
    let (shuffled, _, _) = pixel_shuffle(&repeated, seq, in_ch * repeats, factor);
    shuffled
}

/// `SnakeBeta`: `x + sin(x * exp(alpha))^2 / (exp(beta) + 1e-9)`.
struct SnakeBeta {
    alpha: Vec<f32>,
    beta: Vec<f32>,
    ch: usize,
}

impl SnakeBeta {
    fn load(file: &SafetensorsFile, prefix: &str, ch: usize) -> Result<Self> {
        Ok(Self {
            alpha: tensor(file, &format!("{prefix}.alpha"), &[ch])?,
            beta: tensor(file, &format!("{prefix}.beta"), &[ch])?,
            ch,
        })
    }

    fn apply(&self, x: &mut [f32]) {
        for frame in x.chunks_exact_mut(self.ch) {
            for (c, v) in frame.iter_mut().enumerate() {
                let alpha = self.alpha[c].exp();
                let beta = self.beta[c].exp();
                let sine = (*v * alpha).sin();
                *v += (1.0 / (beta + 1e-9)) * sine * sine;
            }
        }
    }
}

/// The residual unit (Snake, dilated conv, Snake, pointwise conv,
/// residual add). Every unit in this architecture keeps its width.
struct ResidualUnit {
    snake1: SnakeBeta,
    conv1: Vec<f32>,
    bias1: Option<Vec<f32>>,
    dilation: usize,
    conv1_padding: usize,
    snake2: SnakeBeta,
    conv2: Vec<f32>,
    bias2: Option<Vec<f32>>,
    ch: usize,
}

impl ResidualUnit {
    fn load(
        file: &SafetensorsFile,
        prefix: &str,
        ch: usize,
        dilation: usize,
        kernel: usize,
    ) -> Result<Self> {
        Ok(Self {
            snake1: SnakeBeta::load(file, &format!("{prefix}.layers.0"), ch)?,
            conv1: tensor(
                file,
                &format!("{prefix}.layers.1.weight"),
                &[ch * kernel * ch],
            )?,
            bias1: Some(tensor(file, &format!("{prefix}.layers.1.bias"), &[ch])?),
            dilation,
            conv1_padding: (dilation * (kernel - 1)) / 2,
            snake2: SnakeBeta::load(file, &format!("{prefix}.layers.2"), ch)?,
            conv2: tensor(file, &format!("{prefix}.layers.3.weight"), &[ch * ch])?,
            bias2: Some(tensor(file, &format!("{prefix}.layers.3.bias"), &[ch])?),
            ch,
        })
    }

    fn forward(&self, x: Vec<f32>, seq: usize) -> Vec<f32> {
        let ch = self.ch;
        let mut h = x.clone();
        self.snake1.apply(&mut h);
        h = conv1d(
            &h,
            seq,
            ch,
            &self.conv1,
            self.bias1.as_deref(),
            ch,
            7,
            self.dilation,
            1,
            self.conv1_padding,
        );
        self.snake2.apply(&mut h);
        h = conv1d(
            &h,
            seq,
            ch,
            &self.conv2,
            self.bias2.as_deref(),
            ch,
            1,
            1,
            1,
            0,
        );
        let mut out = x;
        for (dst, src) in out.iter_mut().zip(&h) {
            *dst += src;
        }
        out
    }
}

/// One encoder stage: three residual units, Snake, and the strided
/// down conv, with the optional averaging shortcut.
struct EncoderStage {
    units: [ResidualUnit; 3],
    snake: SnakeBeta,
    down_w: Vec<f32>,
    down_b: Option<Vec<f32>>,
    out_ch: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
    shortcut: bool,
}

impl EncoderStage {
    fn forward(&self, x: Vec<f32>, seq: usize, in_ch: usize) -> (Vec<f32>, usize) {
        let debug = std::env::var("LC_DEBUG").is_ok();
        let mut cur = x.clone();
        for (u, unit) in self.units.iter().enumerate() {
            cur = unit.forward(cur, seq);
            if debug {
                eprintln!(
                    "[stage] unit {u} first20 {}",
                    cur.iter().take(20).sum::<f32>()
                );
            }
        }
        self.snake.apply(&mut cur);
        if debug {
            eprintln!("[stage] snake first20 {}", cur.iter().take(20).sum::<f32>());
        }
        let mut h = conv1d(
            &cur,
            seq,
            in_ch,
            &self.down_w,
            self.down_b.as_deref(),
            self.out_ch,
            self.kernel,
            1,
            self.stride,
            self.padding,
        );
        if debug {
            eprintln!("[stage] conv first20 {}", h.iter().take(20).sum::<f32>());
        }
        if self.shortcut {
            let sc = downsample_shortcut(&x, seq, in_ch, self.out_ch, self.stride);
            if debug {
                eprintln!(
                    "[stage] shortcut first20 {}",
                    sc.iter().take(20).sum::<f32>()
                );
            }
            for (dst, src) in h.iter_mut().zip(&sc) {
                *dst += src;
            }
        }
        let out_seq = h.len() / self.out_ch;
        (h, out_seq)
    }
}

/// One decoder stage: Snake, transposed-conv upsample, three residual
/// units, with the optional duplicating shortcut off the stage input.
struct DecoderStage {
    snake: SnakeBeta,
    up_w: Vec<f32>,
    up_b: Option<Vec<f32>>,
    units: [ResidualUnit; 3],
    out_ch: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
    shortcut: bool,
}

impl DecoderStage {
    fn forward(&self, x: Vec<f32>, seq: usize, in_ch: usize) -> (Vec<f32>, usize) {
        let debug = std::env::var("LC_DEBUG").is_ok();
        let mut cur = x.clone();
        self.snake.apply(&mut cur);
        if debug {
            eprintln!(
                "[dec] snake full {:.4} first6 {:?}",
                cur.iter().sum::<f32>(),
                &cur[..6]
            );
        }
        let mut cur = x.clone();
        self.snake.apply(&mut cur);
        let upsampled = conv_transpose1d(
            &cur,
            seq,
            in_ch,
            &self.up_w,
            self.up_b.as_deref(),
            self.out_ch,
            self.kernel,
            self.stride,
            self.padding,
        );
        let up_seq = upsampled.len() / self.out_ch;
        let mut h = upsampled;
        for unit in &self.units {
            h = unit.forward(h, up_seq);
        }
        // The reference adds the duplicating shortcut to the POST-unit
        // value: `h + res(x)`.
        if self.shortcut {
            let sc = upsample_shortcut(&x, seq, in_ch, self.out_ch, self.stride);
            for (dst, src) in h.iter_mut().zip(&sc) {
                *dst += src;
            }
        }
        (h, up_seq)
    }
}

/// The loaded WAV-VAE (encoder and decoder; the generate path decodes
/// and voice prompts encode).
pub(crate) struct AudioDiTVae {
    encoder: Vec<EncoderLayerKind>,
    encoder_shortcut: Option<(usize, usize, usize)>,
    decoder: Vec<DecoderLayerKind>,
    decoder_shortcut: Option<(usize, usize, usize)>,
    pub(crate) config: super::VaeConfig,
}

#[allow(
    clippy::large_enum_variant,
    reason = "the weights live in exactly one variant per layer"
)]
enum EncoderLayerKind {
    Conv {
        w: Vec<f32>,
        b: Option<Vec<f32>>,
        out_ch: usize,
        kernel: usize,
        stride: usize,
        padding: usize,
    },
    Stage(EncoderStage),
}

#[allow(
    clippy::large_enum_variant,
    reason = "the weights live in exactly one variant per layer"
)]
enum DecoderLayerKind {
    Conv {
        w: Vec<f32>,
        b: Option<Vec<f32>>,
        out_ch: usize,
        kernel: usize,
        stride: usize,
        padding: usize,
    },
    Stage(DecoderStage),
    Snake(SnakeBeta),
}

impl AudioDiTVae {
    pub(crate) fn load(
        file: &SafetensorsFile,
        prefix: &str,
        config: &super::VaeConfig,
    ) -> Result<Self> {
        config.validate()?;
        let ch = config.channels;
        let mut c_mults = vec![1usize];
        c_mults.extend(config.c_mults.iter().copied());

        let mut encoder = Vec::new();
        encoder.push(EncoderLayerKind::Conv {
            w: tensor(
                file,
                &format!("{prefix}.encoder.layers.0.weight"),
                &[c_mults[0] * ch * 7 * config.in_channels],
            )?,
            b: Some(tensor(
                file,
                &format!("{prefix}.encoder.layers.0.bias"),
                &[c_mults[0] * ch],
            )?),
            out_ch: c_mults[0] * ch,
            kernel: 7,
            stride: 1,
            padding: 3,
        });
        let mut index = 1usize;
        for i in 0..config.c_mults.len() {
            let in_c = c_mults[i] * ch;
            let out_c = c_mults[i + 1] * ch;
            let stride = config.strides[i];
            let base = format!("{prefix}.encoder.layers.{index}");
            let unit = |j: usize, d: usize| -> Result<ResidualUnit> {
                ResidualUnit::load(file, &format!("{base}.layers.{j}"), in_c, d, 7)
            };
            encoder.push(EncoderLayerKind::Stage(EncoderStage {
                units: [unit(0, 1)?, unit(1, 3)?, unit(2, 9)?],
                snake: SnakeBeta::load(file, &format!("{base}.layers.3"), in_c)?,
                down_w: tensor(
                    file,
                    &format!("{base}.layers.4.weight"),
                    &[out_c * 2 * stride * in_c],
                )?,
                down_b: Some(tensor(file, &format!("{base}.layers.4.bias"), &[out_c])?),
                out_ch: out_c,
                kernel: 2 * stride,
                stride,
                padding: stride.div_ceil(2),
                shortcut: config.downsample_shortcut == "averaging",
            }));
            index += 1;
        }
        let last_mult = *c_mults.last().expect("c_mults");
        encoder.push(EncoderLayerKind::Conv {
            w: tensor(
                file,
                &format!("{prefix}.encoder.layers.{index}.weight"),
                &[config.encoder_latent_dim * 3 * last_mult * ch],
            )?,
            b: Some(tensor(
                file,
                &format!("{prefix}.encoder.layers.{index}.bias"),
                &[config.encoder_latent_dim],
            )?),
            out_ch: config.encoder_latent_dim,
            kernel: 3,
            stride: 1,
            padding: 1,
        });
        let encoder_shortcut = if config.out_shortcut == "averaging" {
            Some((last_mult * ch, config.encoder_latent_dim, 1))
        } else {
            None
        };

        let mut decoder = Vec::new();
        decoder.push(DecoderLayerKind::Conv {
            w: tensor(
                file,
                &format!("{prefix}.decoder.layers.0.weight"),
                &[last_mult * ch * 7 * config.latent_dim],
            )?,
            b: Some(tensor(
                file,
                &format!("{prefix}.decoder.layers.0.bias"),
                &[last_mult * ch],
            )?),
            out_ch: last_mult * ch,
            kernel: 7,
            stride: 1,
            padding: 3,
        });
        let mut index = 1usize;
        for i in (1..c_mults.len()).rev() {
            let in_c = c_mults[i] * ch;
            let out_c = c_mults[i - 1] * ch;
            let stride = config.strides[i - 1];
            let base = format!("{prefix}.decoder.layers.{index}");
            let unit = |j: usize, d: usize| -> Result<ResidualUnit> {
                ResidualUnit::load(file, &format!("{base}.layers.{j}"), out_c, d, 7)
            };
            decoder.push(DecoderLayerKind::Stage(DecoderStage {
                snake: SnakeBeta::load(file, &format!("{base}.layers.0"), in_c)?,
                up_w: tensor(
                    file,
                    &format!("{base}.layers.1.weight"),
                    &[out_c * 2 * stride * in_c],
                )?,
                up_b: Some(tensor(file, &format!("{base}.layers.1.bias"), &[out_c])?),
                units: [unit(2, 1)?, unit(3, 3)?, unit(4, 9)?],
                out_ch: out_c,
                kernel: 2 * stride,
                stride,
                padding: stride.div_ceil(2),
                shortcut: config.upsample_shortcut == "duplicating",
            }));
            index += 1;
        }
        decoder.push(DecoderLayerKind::Snake(SnakeBeta::load(
            file,
            &format!("{prefix}.decoder.layers.{index}"),
            c_mults[0] * ch,
        )?));
        index += 1;
        decoder.push(DecoderLayerKind::Conv {
            w: tensor(
                file,
                &format!("{prefix}.decoder.layers.{index}.weight"),
                &[config.in_channels * 7 * c_mults[0] * ch],
            )?,
            b: None,
            out_ch: config.in_channels,
            kernel: 7,
            stride: 1,
            padding: 3,
        });
        let decoder_shortcut = if config.in_shortcut == "duplicating" {
            Some((config.latent_dim, last_mult * ch, 1))
        } else {
            None
        };

        Ok(Self {
            encoder,
            encoder_shortcut,
            decoder,
            decoder_shortcut,
            config: config.clone(),
        })
    }

    /// Encodes `wav [seq, 1]` (already padded to the hop multiple by
    /// the caller) to latents `[frames, latent_dim]`, drawing the
    /// reparameterization noise from `noise`.
    pub(crate) fn encode(&self, wav: &[f32], noise: &mut dyn super::Noise) -> Result<Vec<f32>> {
        let in_ch = self.config.in_channels;
        let mut h = wav.to_vec();
        let mut seq = wav.len() / in_ch;
        let mut channels = in_ch;
        let head = self.encoder.len() - 1;
        for (i, layer) in self.encoder.iter().enumerate() {
            let apply_shortcut = self.encoder_shortcut.is_some() && i == head;
            let input = h;
            let mut output = match layer {
                EncoderLayerKind::Conv {
                    w,
                    b,
                    out_ch,
                    kernel,
                    stride,
                    padding,
                } => conv1d(
                    &input,
                    seq,
                    channels,
                    w,
                    b.as_deref(),
                    *out_ch,
                    *kernel,
                    1,
                    *stride,
                    *padding,
                ),
                EncoderLayerKind::Stage(stage) => {
                    let (out, out_seq) = stage.forward(input.clone(), seq, channels);
                    seq = out_seq;
                    out
                }
            };
            channels = output.len() / seq.max(1);
            if apply_shortcut {
                if let Some((sc_in, sc_out, factor)) = self.encoder_shortcut {
                    let sc = downsample_shortcut(&input, seq, sc_in, sc_out, factor);
                    for (dst, src) in output.iter_mut().zip(&sc) {
                        *dst += src;
                    }
                    channels = sc_out;
                }
            }
            h = output;
        }
        seq = h.len() / self.config.encoder_latent_dim;
        // mean / softplus(scale) split with the reparameterization draw.
        let latent_dim = self.config.encoder_latent_dim / 2;
        let mut out = vec![0.0f32; seq * latent_dim];
        let draws = noise.normal(seq * latent_dim);
        for t in 0..seq {
            for d in 0..latent_dim {
                let mean = h[t * self.config.encoder_latent_dim + d];
                let scale_param = h[t * self.config.encoder_latent_dim + latent_dim + d];
                let stdev = softplus(scale_param) + 1e-4;
                out[t * latent_dim + d] = draws[t * latent_dim + d] * stdev + mean;
            }
        }
        for v in &mut out {
            *v /= self.config.scale;
        }
        Ok(out)
    }

    /// Decodes latents `[frames, latent_dim]` to samples `[out_seq]`.
    pub(crate) fn decode(&self, latents: &[f32], frames: usize) -> Vec<f32> {
        let scale = self.config.scale;
        let latent_dim = self.config.latent_dim;
        let mut h: Vec<f32> = latents
            .iter()
            .take(frames * latent_dim)
            .map(|v| v * scale)
            .collect();
        let mut seq = frames;
        let mut channels = latent_dim;
        let mut iter = self.decoder.iter().peekable();
        // The stem conv sums with the duplicating shortcut.
        if let Some((sc_in, sc_out, factor)) = self.decoder_shortcut {
            let shortcut = upsample_shortcut(&h, seq, sc_in, sc_out, factor);
            let stem = match iter.peek() {
                Some(DecoderLayerKind::Conv {
                    w,
                    b,
                    out_ch,
                    kernel,
                    stride,
                    padding,
                }) => conv1d(
                    &h,
                    seq,
                    channels,
                    w,
                    b.as_deref(),
                    *out_ch,
                    *kernel,
                    1,
                    *stride,
                    *padding,
                ),
                _ => unreachable!("decoder starts with the stem conv"),
            };
            let mut combined = shortcut;
            if std::env::var("LC_DEBUG").is_ok() {
                eprintln!(
                    "[dec] stem+h full {:.4} first6 {:?}",
                    combined.iter().sum::<f32>(),
                    &combined[..6]
                );
            }
            for (dst, src) in combined.iter_mut().zip(&stem) {
                *dst += src;
            }
            h = combined;
            channels = stem.len() / seq.max(1);
            iter.next();
        }
        for layer in iter {
            match layer {
                DecoderLayerKind::Conv {
                    w,
                    b,
                    out_ch,
                    kernel,
                    stride,
                    padding,
                } => {
                    h = conv1d(
                        &h,
                        seq,
                        channels,
                        w,
                        b.as_deref(),
                        *out_ch,
                        *kernel,
                        1,
                        *stride,
                        *padding,
                    );
                    channels = *out_ch;
                }
                DecoderLayerKind::Stage(stage) => {
                    let (out, out_seq) = stage.forward(h, seq, channels);
                    h = out;
                    seq = out_seq;
                    channels = stage.out_ch;
                }
                DecoderLayerKind::Snake(snake) => {
                    snake.apply(&mut h);
                }
            }
        }
        h
    }
}

/// `softplus` without overflow: `log1p(exp(-|x|)) + max(x, 0)`.
fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else {
        x.exp().ln_1p()
    }
}
