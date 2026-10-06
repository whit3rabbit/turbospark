//! The StepAudio2 HiFT vocoder: predicted mel -> 24 kHz waveform.
//!
//! Reference: `mlx_audio/codec/models/stepaudio2/hift.py`
//! (`StepAudio2HiFTGenerator`, which configures the chatterbox s3gen
//! `HiFTGenerator` at the 24 kHz geometry) plus
//! `s3gen/hifigan.py` (SineGen, ResBlock, STFT/iSTFT helpers) and
//! `s3gen/f0_predictor.py` (`ConvRNNF0Predictor`), all at the pinned
//! commit recorded in the parent module.
//!
//! Structure: an ELU conv stack predicts f0 per mel frame; the f0 is
//! nearest-upsampled 480x (the product of the three upsample rates and
//! the iSTFT hop), rendered through the neural source filter
//! (`SineGen` with interpolated phase), and fused with the mel
//! upsampling path as STFT-domain residual blocks. The output head
//! emits log-magnitude and phase halves that an iSTFT turns into
//! samples, clipped to +/-0.99.
//!
//! The sine generator's random initial phase and noise bed are
//! inference-time randomness in the reference; here they are explicit
//! [`SineDraws`] inputs (the fixture records and replays one
//! realization). `None` uses zeros: deterministic initial phase and a
//! silent noise bed.

use turbospark_model_io::safetensors::SafetensorsFile;

use super::{leaky_relu, to_cm, to_rows, SaConv1d, SaConvTr1d, SaLinear};
use crate::codec::wnconv::load_f32_shaped;
use crate::dsp;
use crate::fft::{ComplexF32, ComplexFftPlan};
use crate::{Result, SpeechError};

/// Fixed reference geometry (`StepAudio2HiFTGenerator.__init__`).
const UPSAMPLE_RATES: [usize; 3] = [8, 5, 3];
const UPSAMPLE_KERNELS: [usize; 3] = [16, 11, 7];
const RESBLOCK_KERNELS: [usize; 3] = [3, 7, 11];
const RESBLOCK_DILATIONS: [usize; 3] = [1, 3, 5];
const SOURCE_RESBLOCK_KERNELS: [usize; 3] = [7, 7, 11];
const ISTFT_N_FFT: usize = 16;
const ISTFT_HOP: usize = 4;
const BASE_CHANNELS: usize = 512;
const MEL_CHANNELS: usize = 80;
const LRELU_SLOPE: f32 = 0.1;
const AUDIO_LIMIT: f32 = 0.99;
const NB_HARMONICS: usize = 8;
const SINE_AMP: f32 = 0.1;
const NOISE_STD: f32 = 0.003;
const VOICED_THRESHOLD: f32 = 10.0;
const F0_CHANNELS: usize = 512;
const SOURCE_IN: usize = ISTFT_N_FFT + 2;

/// Total mel-frame to waveform-sample factor (8 * 5 * 3 * hop 4).
const UPSCALE: usize = 480;

/// Sine-generator randomness the reference draws at inference.
#[derive(Debug, Clone, Default)]
pub struct SineDraws {
    /// Initial phase offsets per harmonic `[harmonics + 1]`; the first
    /// entry is always zero (the reference zeroes it after the draw).
    pub rand_ini: Vec<f32>,
    /// Standard-normal noise bed `[harmonics + 1, T_wav]`, scaled by
    /// the reference's noise amplitude inside `infer`.
    pub sine_noise: Vec<f32>,
}

/// Snake activation exactly as `s3gen/hifigan.py::Snake` defines it:
/// the reciprocal and the sine argument both use the magnitude-clamped
/// alpha (`|alpha| >= 1e-4`, exact zero becomes `1e-4`).
fn snake_hift(x: &mut [f32], alpha: &[f32], channels: usize, frames: usize) {
    assert_eq!(alpha.len(), channels);
    for ch in 0..channels {
        let a = alpha[ch];
        let abs = a.abs();
        let sign = if a < 0.0 {
            -1.0
        } else if a > 0.0 {
            1.0
        } else {
            0.0
        };
        let clamped = if abs < 1e-9 {
            1e-4
        } else {
            sign * abs.max(1e-4)
        };
        for v in &mut x[ch * frames..(ch + 1) * frames] {
            *v += (*v * clamped).sin().powi(2) / clamped;
        }
    }
}

/// Symmetric conv padding the reference computes as
/// `get_padding(kernel, dilation) = (kernel * dilation - dilation) / 2`.
fn get_padding(kernel: usize, dilation: usize) -> usize {
    (kernel * dilation - dilation) / 2
}

/// ResBlock (`s3gen/hifigan.py::ResBlock`): per dilation, Snake -> conv
/// -> Snake -> conv, residual. Channel-major `[ch, T]`.
#[derive(Debug, Clone)]
struct ResBlock {
    channels: usize,
    convs1: Vec<SaConv1d>,
    convs2: Vec<SaConv1d>,
    alphas1: Vec<Vec<f32>>,
    alphas2: Vec<Vec<f32>>,
}

impl ResBlock {
    fn load(
        file: &SafetensorsFile,
        prefix: &str,
        channels: usize,
        kernel: usize,
        dilations: &[usize],
    ) -> Result<Self> {
        let mut convs1 = Vec::with_capacity(dilations.len());
        let mut convs2 = Vec::with_capacity(dilations.len());
        let mut alphas1 = Vec::with_capacity(dilations.len());
        let mut alphas2 = Vec::with_capacity(dilations.len());
        for (d, &dilation) in dilations.iter().enumerate() {
            convs1.push(SaConv1d::load(
                file,
                &format!("{prefix}.convs1.{d}"),
                channels,
                channels,
                kernel,
                1,
                get_padding(kernel, dilation),
                dilation,
                1,
            )?);
            convs2.push(SaConv1d::load(
                file,
                &format!("{prefix}.convs2.{d}"),
                channels,
                channels,
                kernel,
                1,
                get_padding(kernel, 1),
                1,
                1,
            )?);
            alphas1.push(load_f32_shaped(
                file,
                &format!("{prefix}.activations1.{d}.alpha"),
                &[channels],
            )?);
            alphas2.push(load_f32_shaped(
                file,
                &format!("{prefix}.activations2.{d}.alpha"),
                &[channels],
            )?);
        }
        Ok(ResBlock {
            channels,
            convs1,
            convs2,
            alphas1,
            alphas2,
        })
    }

    fn forward(&self, x: &mut Vec<f32>, frames: usize) {
        for d in 0..self.convs1.len() {
            let mut xt = x.clone();
            snake_hift(&mut xt, &self.alphas1[d], self.channels, frames);
            xt = self.convs1[d].forward(&xt);
            snake_hift(&mut xt, &self.alphas2[d], self.channels, frames);
            xt = self.convs2[d].forward(&xt);
            for (v, r) in x.iter_mut().zip(&xt) {
                *v += r;
            }
        }
    }
}

/// Linear interpolation of the last axis to `new_size`, mirroring
/// `s3gen/hifigan.py::_linear_interpolate_1d_to_size` (linspace
/// positions, floor index, clamped upper neighbor).
fn interp_to_size(x: &[f32], new_size: usize) -> Vec<f32> {
    let t = x.len();
    if new_size == t {
        return x.to_vec();
    }
    if new_size == 0 {
        return Vec::new();
    }
    let mut out = vec![0.0f32; new_size];
    if new_size == 1 {
        // linspace(0, T-1, 1) is [0]; position 0 interpolates x[0].
        out[0] = x[0];
        return out;
    }
    let step = (t - 1) as f32 / (new_size - 1) as f32;
    for (i, o) in out.iter_mut().enumerate() {
        // numpy/MLX linspace pins the endpoint.
        let pos = if i == new_size - 1 {
            (t - 1) as f32
        } else {
            i as f32 * step
        };
        let low = pos.floor();
        let low_i = (low as usize).min(t - 1);
        let high_i = (low_i + 1).min(t - 1);
        let w = pos - low;
        *o = x[low_i] + w * (x[high_i] - x[low_i]);
    }
    out
}

/// Loaded HiFT generator, batch-of-one.
pub struct StepAudio2HiFT {
    f0_condnet: Vec<SaConv1d>,
    f0_classifier: SaLinear,
    source_linear: SaLinear,
    conv_pre: SaConv1d,
    ups: Vec<SaConvTr1d>,
    source_downs: Vec<SaConv1d>,
    source_resblocks: Vec<ResBlock>,
    resblocks: Vec<ResBlock>,
    conv_post: SaConv1d,
    /// The synthesis window is a checkpoint tensor; the derived
    /// periodic Hann is the fallback the reference documents.
    stft_window: Vec<f32>,
    stft_plan: ComplexFftPlan,
}

impl StepAudio2HiFT {
    pub fn load(file: &SafetensorsFile) -> Result<Self> {
        let prefix = "hift";
        let f0_condnet = (0..5)
            .map(|i| {
                let (in_ch, out_ch) = if i == 0 {
                    (MEL_CHANNELS, F0_CHANNELS)
                } else {
                    (F0_CHANNELS, F0_CHANNELS)
                };
                SaConv1d::load(
                    file,
                    &format!("{prefix}.f0_predictor.condnet.{i}"),
                    in_ch,
                    out_ch,
                    3,
                    1,
                    1,
                    1,
                    1,
                )
            })
            .collect::<Result<_>>()?;
        let f0_classifier = SaLinear::load(
            file,
            &format!("{prefix}.f0_predictor.classifier"),
            F0_CHANNELS,
            1,
        )?;
        let source_linear = SaLinear::load(
            file,
            &format!("{prefix}.m_source.l_linear"),
            NB_HARMONICS + 1,
            1,
        )?;

        let conv_pre = SaConv1d::load(
            file,
            &format!("{prefix}.conv_pre"),
            MEL_CHANNELS,
            BASE_CHANNELS,
            7,
            1,
            3,
            1,
            1,
        )?;

        // Upsampling transpose convs: channels halve per stage.
        let mut ups = Vec::with_capacity(UPSAMPLE_RATES.len());
        for (i, (&u, &k)) in UPSAMPLE_RATES.iter().zip(&UPSAMPLE_KERNELS).enumerate() {
            let (in_ch, out_ch) = (BASE_CHANNELS >> i, BASE_CHANNELS >> (i + 1));
            ups.push(SaConvTr1d::load(
                file,
                &format!("{prefix}.ups.{i}"),
                in_ch,
                out_ch,
                k,
                u,
                (k - u) / 2,
            )?);
        }

        // Source downsampling convs: cumulative rates [1, 3, 15]
        // reversed, kernel `2u`, stride `u`, padding `u / 2` (the
        // rate-1 stage uses a 1x1 conv).
        let mut source_downs = Vec::with_capacity(UPSAMPLE_RATES.len());
        for (i, &u) in [15usize, 3, 1].iter().enumerate() {
            let (kernel, stride, padding) = if u == 1 { (1, 1, 0) } else { (u * 2, u, u / 2) };
            source_downs.push(SaConv1d::load(
                file,
                &format!("{prefix}.source_downs.{i}"),
                SOURCE_IN,
                BASE_CHANNELS >> (i + 1),
                kernel,
                stride,
                padding,
                1,
                1,
            )?);
        }
        let source_resblocks = (0..UPSAMPLE_RATES.len())
            .map(|i| {
                ResBlock::load(
                    file,
                    &format!("{prefix}.source_resblocks.{i}"),
                    BASE_CHANNELS >> (i + 1),
                    SOURCE_RESBLOCK_KERNELS[i],
                    &RESBLOCK_DILATIONS,
                )
            })
            .collect::<Result<_>>()?;

        // Main resblocks: after upsample stage i the channel count is
        // base >> (i + 1); three blocks with kernels [3, 7, 11].
        let mut resblocks = Vec::with_capacity(3 * RESBLOCK_KERNELS.len());
        for i in 0..UPSAMPLE_RATES.len() {
            let ch = BASE_CHANNELS >> (i + 1);
            for kernel in &RESBLOCK_KERNELS {
                resblocks.push(ResBlock::load(
                    file,
                    &format!("{prefix}.resblocks.{}", resblocks.len()),
                    ch,
                    *kernel,
                    &RESBLOCK_DILATIONS,
                )?);
            }
        }

        let conv_post = SaConv1d::load(
            file,
            &format!("{prefix}.conv_post"),
            BASE_CHANNELS >> UPSAMPLE_RATES.len(),
            SOURCE_IN,
            7,
            1,
            3,
            1,
            1,
        )?;

        let stft_window = if file.contains_tensor(&format!("{prefix}.stft_window")) {
            load_f32_shaped(file, &format!("{prefix}.stft_window"), &[ISTFT_N_FFT])?
        } else {
            dsp::hann_window(ISTFT_N_FFT)
        };
        let stft_plan = ComplexFftPlan::new(ISTFT_N_FFT)?;

        Ok(StepAudio2HiFT {
            f0_condnet,
            f0_classifier,
            source_linear,
            conv_pre,
            ups,
            source_downs,
            source_resblocks,
            resblocks,
            conv_post,
            stft_window,
            stft_plan,
        })
    }

    /// F0 in Hz per mel frame (the reference returns the absolute
    /// classifier output).
    pub fn predict_f0(&self, speech_feat: &[f32]) -> Result<Vec<f32>> {
        let frames = speech_feat.len() / MEL_CHANNELS;
        if speech_feat.len() % MEL_CHANNELS != 0 {
            return Err(SpeechError::Input {
                why: "speech_feat must be row-major mel frames".to_string(),
            });
        }
        let mut h = to_cm(speech_feat, frames, MEL_CHANNELS);
        for conv in &self.f0_condnet {
            h = conv.forward(&h);
            super::elu(&mut h);
        }
        let rows = to_rows(&h, F0_CHANNELS, frames);
        let out = self.f0_classifier.forward(&rows, frames);
        Ok(out.into_iter().map(f32::abs).collect())
    }

    /// Renders mel frames (row-major `[T, 80]`) to 24 kHz samples.
    pub fn infer(&self, speech_feat: &[f32], draws: Option<&SineDraws>) -> Result<Vec<f32>> {
        let mel_frames = speech_feat.len() / MEL_CHANNELS;
        let f0 = self.predict_f0(speech_feat)?;

        // Nearest-upsample f0 to the waveform rate.
        let wav_frames = mel_frames * UPSCALE;
        let mut source = vec![0.0f32; wav_frames];
        for (i, s) in source.iter_mut().enumerate() {
            *s = f0[i / UPSCALE];
        }

        // Neural source filter: sine harmonics with interpolated phase.
        let sine = self.sine_gen(&source, draws)?;
        // Merge harmonics: tanh(Linear(9 -> 1)) per sample.
        let mut merged = vec![0.0f32; wav_frames];
        for (t, m) in merged.iter_mut().enumerate() {
            let mut acc = 0.0f32;
            for h in 0..NB_HARMONICS + 1 {
                acc += self.source_linear.weight[h] * sine[h * wav_frames + t];
            }
            acc += self.source_linear.bias.as_deref().unwrap_or(&[0.0])[0];
            *m = acc.tanh();
        }

        // STFT of the source: real and imaginary halves stacked to 18.
        let (s_real, s_imag) = self.stft(&merged)?;
        let stft_frames = s_real.len() / (ISTFT_N_FFT / 2 + 1);
        let mut s_stft = vec![0.0f32; SOURCE_IN * stft_frames];
        s_stft[..(ISTFT_N_FFT / 2 + 1) * stft_frames].copy_from_slice(&s_real);
        s_stft[(ISTFT_N_FFT / 2 + 1) * stft_frames..].copy_from_slice(&s_imag);

        // Pre-conv on the mel.
        let mut x = to_cm(speech_feat, mel_frames, MEL_CHANNELS);
        x = self.conv_pre.forward(&x);

        for i in 0..self.ups.len() {
            leaky_relu(&mut x, LRELU_SLOPE);
            x = self.ups[i].forward(&x);
            let frames_i = x.len() / self.ups[i].out_ch;
            if i == self.ups.len() - 1 {
                // Reflection pad of one sample on the left (prepend the
                // second sample).
                let ch = self.ups[i].out_ch;
                let mut padded = vec![0.0f32; ch * (frames_i + 1)];
                for c in 0..ch {
                    padded[c * (frames_i + 1)] = x[c * frames_i + 1];
                    padded[c * (frames_i + 1) + 1..(c + 1) * (frames_i + 1)]
                        .copy_from_slice(&x[c * frames_i..(c + 1) * frames_i]);
                }
                x = padded;
            }
            let frames_i = x.len() / self.ups[i].out_ch;

            // Source fusion branch.
            let mut si = self.source_downs[i].forward(&s_stft);
            let si_frames = si.len() / self.source_downs[i].out_ch;
            self.source_resblocks[i].forward(&mut si, si_frames);
            for (v, s) in x.iter_mut().zip(&si) {
                *v += s;
            }

            // Average the three parallel resblocks.
            let ch = self.ups[i].out_ch;
            let mut acc = vec![0.0f32; ch * frames_i];
            for j in 0..RESBLOCK_KERNELS.len() {
                let mut block_out = x.clone();
                self.resblocks[i * RESBLOCK_KERNELS.len() + j].forward(&mut block_out, frames_i);
                for (a, b) in acc.iter_mut().zip(&block_out) {
                    *a += b;
                }
            }
            for v in &mut acc {
                *v /= RESBLOCK_KERNELS.len() as f32;
            }
            x = acc;
        }

        leaky_relu(&mut x, LRELU_SLOPE);
        let out_frames = x.len() / (BASE_CHANNELS >> self.ups.len());
        let x = self.conv_post.forward(&x);
        let bins = ISTFT_N_FFT / 2 + 1;
        let mut magnitude = vec![0.0f32; bins * out_frames];
        let mut phase = vec![0.0f32; bins * out_frames];
        for (i, v) in x.iter().enumerate() {
            let bin = i / out_frames;
            let frame = i % out_frames;
            if bin < bins {
                magnitude[bin * out_frames + frame] = v.exp();
            } else {
                phase[(bin - bins) * out_frames + frame] = v.sin();
            }
        }
        let wav = self.istft(&magnitude, &phase, out_frames)?;
        Ok(wav
            .into_iter()
            .map(|v| v.clamp(-AUDIO_LIMIT, AUDIO_LIMIT))
            .collect())
    }

    /// `SineGen` with the interpolation phase path. Returns
    /// `[harmonics + 1, T_wav]`.
    fn sine_gen(&self, f0: &[f32], draws: Option<&SineDraws>) -> Result<Vec<f32>> {
        let t_wav = f0.len();
        let harmonics = NB_HARMONICS + 1;
        let t_mel = (t_wav / UPSCALE).max(1);
        let default_ini = vec![0.0f32; harmonics];
        let rand_ini = draws.map(|d| d.rand_ini.as_slice()).unwrap_or(&default_ini);
        if rand_ini.len() != harmonics {
            return Err(SpeechError::Input {
                why: format!("rand_ini must have {harmonics} entries"),
            });
        }
        if let Some(d) = draws {
            if d.sine_noise.len() != harmonics * t_wav {
                return Err(SpeechError::Input {
                    why: format!(
                        "sine_noise must have {} entries, got {}",
                        harmonics * t_wav,
                        d.sine_noise.len()
                    ),
                });
            }
        }

        // Voiced flag and per-harmonic normalized radian rates.
        let uv: Vec<f32> = f0
            .iter()
            .map(|&f| (f > VOICED_THRESHOLD) as i32 as f32)
            .collect();
        // rad[h][t] = (f0[t] * (h+1) / sr) % 1, plus the drawn initial
        // phase at t = 0.
        let mut rad = vec![0.0f32; harmonics * t_wav];
        for (h, row) in rad.chunks_mut(t_wav).enumerate() {
            for (t, r) in row.iter_mut().enumerate() {
                *r = (f0[t] * (h + 1) as f32 / 24_000.0) % 1.0;
            }
            row[0] += rand_ini[h];
        }

        // Downsample the rates to the mel rate, accumulate phase, scale
        // back up, and take the sine.
        let mut sine = vec![0.0f32; harmonics * t_wav];
        for (h, row) in rad.chunks(t_wav).enumerate() {
            let down = interp_to_size(row, t_mel);
            let mut phase = 0.0f32;
            let mut mel_phase = vec![0.0f32; t_mel];
            for (tm, v) in down.iter().enumerate() {
                phase += v;
                mel_phase[tm] = phase * 2.0 * std::f32::consts::PI;
            }
            let up = interp_to_size(
                &mel_phase
                    .iter()
                    .map(|p| p * UPSCALE as f32)
                    .collect::<Vec<_>>(),
                t_wav,
            );
            for (t, v) in up.iter().enumerate() {
                sine[h * t_wav + t] = v.sin() * SINE_AMP;
            }
        }

        // Voiced gating and the noise bed.
        let zeros = vec![0.0f32; harmonics * t_wav];
        let noise = draws.map(|d| d.sine_noise.as_slice()).unwrap_or(&zeros);
        for t in 0..t_wav {
            let amp = uv[t] * NOISE_STD + (1.0 - uv[t]) * SINE_AMP / 3.0;
            for h in 0..harmonics {
                let i = h * t_wav + t;
                sine[i] = sine[i] * uv[t] + amp * noise[i];
            }
        }
        Ok(sine)
    }

    /// The reference `stft`: reflect-pad by n_fft/2, uncentered frames,
    /// windowed complex FFT. Returns channel-major `[bins, F]`.
    fn stft(&self, x: &[f32]) -> Result<(Vec<f32>, Vec<f32>)> {
        let pad = ISTFT_N_FFT / 2;
        if x.len() < pad + 2 {
            return Err(SpeechError::Input {
                why: "source signal too short for the reflect pad".to_string(),
            });
        }
        let mut padded = Vec::with_capacity(x.len() + ISTFT_N_FFT);
        padded.extend(x[1..pad + 1].iter().rev());
        padded.extend_from_slice(x);
        padded.extend(x[x.len() - pad - 1..x.len() - 1].iter().rev());
        let frames = (padded.len() - ISTFT_N_FFT) / ISTFT_HOP + 1;
        let bins = ISTFT_N_FFT / 2 + 1;
        let mut real = vec![0.0f32; bins * frames];
        let mut imag = vec![0.0f32; bins * frames];
        let mut frame_in = vec![ComplexF32::new(0.0, 0.0); ISTFT_N_FFT];
        for f in 0..frames {
            for (t, slot) in frame_in.iter_mut().enumerate() {
                slot.re = padded[f * ISTFT_HOP + t] * self.stft_window[t];
            }
            let spectrum = self.stft_plan.forward(&frame_in)?;
            for (b, c) in spectrum.iter().take(bins).enumerate() {
                real[b * frames + f] = c.re;
                imag[b * frames + f] = c.im;
            }
        }
        Ok((real, imag))
    }

    /// The reference `istft`: clipped magnitude, Hermitian rebuild,
    /// inverse FFT, windowed overlap-add normalized by the squared
    /// window sum, trimmed by n_fft/2 per side.
    fn istft(&self, magnitude: &[f32], phase: &[f32], frames: usize) -> Result<Vec<f32>> {
        let bins = ISTFT_N_FFT / 2 + 1;
        let out_len = (frames - 1) * ISTFT_HOP + ISTFT_N_FFT;
        let mut ola = vec![0.0f32; out_len];
        let mut denom = vec![0.0f32; out_len];
        let mut spectrum = vec![ComplexF32::new(0.0, 0.0); ISTFT_N_FFT];
        for f in 0..frames {
            for b in 0..bins {
                let mag = magnitude[b * frames + f].min(1e2);
                let ph = phase[b * frames + f];
                spectrum[b] = ComplexF32::new(mag * ph.cos(), mag * ph.sin());
            }
            spectrum[0].im = 0.0;
            spectrum[bins - 1].im = 0.0;
            for k in bins..ISTFT_N_FFT {
                spectrum[k] = spectrum[ISTFT_N_FFT - k].conj();
            }
            let frame = self.stft_plan.inverse(&spectrum)?;
            let start = f * ISTFT_HOP;
            for (t, v) in frame.iter().enumerate() {
                ola[start + t] += v.re * self.stft_window[t];
            }
            for (t, &w) in self.stft_window.iter().enumerate() {
                denom[start + t] += w * w;
            }
        }
        let trim = ISTFT_N_FFT / 2;
        let mut out = Vec::with_capacity(out_len - 2 * trim);
        for t in trim..out_len - trim {
            let d = denom[t];
            out.push(ola[t] / d.max(1e-8));
        }
        Ok(out)
    }
}
