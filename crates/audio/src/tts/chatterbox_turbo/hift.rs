//! The chatterbox_turbo HiFT vocoder: predicted mel -> 24 kHz waveform.
//!
//! Reference: `chatterbox_turbo/models/s3gen/hifigan.py`
//! (`HiFTGenerator`, `Snake`, `ResBlock`, `SineGen`, `SourceModule`,
//! `F0Predictor`, the `_stft`/`_istft` helpers) at the pinned commit
//! recorded in the parent module, configured by
//! `s3gen/s3gen.py::S3Token2Wav` at the 24 kHz geometry.
//!
//! Structure: an ELU conv stack predicts f0 per mel frame; the f0 is
//! nearest-upsampled 480x (8 * 5 * 3 * iSTFT hop 4), rendered through
//! the neural source filter, and fused with the mel upsampling path as
//! STFT-domain residual blocks. The output head emits log-magnitude
//! (`exp`) and phase (`sin`) halves that an iSTFT turns into samples,
//! clipped to +/-0.99.
//!
//! Differences against the sibling chatterbox vocoder at the same
//! commit, all ported faithfully:
//! - the source STFT is UNCENTERED (frames of the raw waveform, no
//!   reflect pad; a zero tail pad only when the input is shorter than
//!   one window), so the fusion lengths run one-two frames short and
//!   the reference's `min_len` clamp truncates the mel path at every
//!   fusion stage;
//! - the last upsample stage pads ONE ZERO on the left (the reference
//!   `mx.pad` call, despite its "reflection" comment), not a reflected
//!   sample;
//! - the sine generator's phase is `(2*pi*cumsum(f0*(h+1)/sr)) mod
//!   2*pi` and the initial-phase draw covers harmonics 1..8 only (eight
//!   uniforms, harmonic 0 a structural zero) - the sibling draws nine
//!   and masks the first;
//! - the final leaky ReLU takes the `nn.leaky_relu` default slope
//!   (0.01), not the 0.1 the upsample stages use;
//! - the f0 predictor keys route through the `Conv1dPT` wrapper
//!   (`condnet.N.conv.conv`) though its math matches the sibling's
//!   `ConvRNNF0Predictor`.
//!
//! The sine randomness (initial phases, the sine noise bed, and the
//! discarded source-module noise bed) is an explicit [`SineDraws`]
//! input; `None` means deterministic zeros, which leaves the sine bed
//! and the noise bed silent.

use turbospark_model_io::safetensors::SafetensorsFile;

use crate::codec::wnconv::load_f32_shaped;
use crate::dsp;
use crate::fft::{ComplexF32, ComplexFftPlan};
use crate::{Result, SpeechError};

use crate::tts::chatterbox::s3gen::{
    elu, leaky_relu, to_cm, to_rows, SaConv1d, SaConvTr1d, SaLinear,
};

/// Fixed reference geometry (`S3Token2Wav.__init__`).
const UPSAMPLE_RATES: [usize; 3] = [8, 5, 3];
const UPSAMPLE_KERNELS: [usize; 3] = [16, 11, 7];
const RESBLOCK_KERNELS: [usize; 3] = [3, 7, 11];
const RESBLOCK_DILATIONS: [usize; 3] = [1, 3, 5];
const SOURCE_RESBLOCK_KERNELS: [usize; 3] = [7, 7, 11];
const ISTFT_N_FFT: usize = 16;
const ISTFT_HOP: usize = 4;
const NB_HARMONICS: usize = 8;
const SINE_AMP: f32 = 0.1;
const NOISE_STD: f32 = 0.003;
const VOICED_THRESHOLD: f32 = 10.0;
/// The `nn.leaky_relu` default slope the reference `decode` uses for
/// its final activation (the upsample stages pass 0.1 explicitly).
const FINAL_LRELU_SLOPE: f32 = 0.01;
const UP_LRELU_SLOPE: f32 = 0.1;
const AUDIO_LIMIT: f32 = 0.99;
const SOURCE_IN: usize = ISTFT_N_FFT + 2;
/// Total mel-frame to waveform-sample factor (8 * 5 * 3 * hop 4).
const UPSCALE: usize = 480;

/// Configurable widths (the reference constructor arguments the
/// `S3Token2Wav` hardwires; the fixture shrinks them, the defaults are
/// the reference values).
#[derive(Debug, Clone)]
pub struct HiftConfig {
    pub base_channels: usize,
    pub in_channels: usize,
    pub f0_channels: usize,
}

impl Default for HiftConfig {
    fn default() -> Self {
        HiftConfig {
            base_channels: 512,
            in_channels: 80,
            f0_channels: 512,
        }
    }
}

impl HiftConfig {
    /// Parses a JSON object over the reference defaults.
    pub fn from_json_with_defaults(
        value: &serde_json::Value,
        defaults: HiftConfig,
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
        Ok(HiftConfig {
            base_channels: int("base_channels", defaults.base_channels)?,
            in_channels: int("in_channels", defaults.in_channels)?,
            f0_channels: int("f0_channels", defaults.f0_channels)?,
        })
    }
}

/// Sine-generator randomness the reference draws at inference. The
/// phase covers harmonics 1..8 only (harmonic 0 is a structural zero);
/// the sine noise bed is `[harmonics + 1, T_wav]`; the source-module
/// noise bed is discarded upstream and recorded only to keep the draw
/// order honest.
#[derive(Debug, Clone, Default)]
pub struct SineDraws {
    pub phase: Vec<f32>,
    pub sine_noise: Vec<f32>,
    pub source_noise: Vec<f32>,
}

/// Snake activation exactly as `hifigan.py::Snake` defines it: the
/// reciprocal and the sine argument both use the magnitude-clamped
/// alpha (`|alpha| >= 1e-4`, exact zero becomes `1e-4`).
fn snake(x: &mut [f32], alpha: &[f32], channels: usize, frames: usize) {
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

/// Symmetric conv padding `(kernel * dilation - dilation) / 2`.
fn get_padding(kernel: usize, dilation: usize) -> usize {
    (kernel * dilation - dilation) / 2
}

/// ResBlock (`hifigan.py::ResBlock`): per dilation, Snake -> conv ->
/// Snake -> conv, residual. Channel-major `[ch, T]`.
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
                &format!("{prefix}.convs1.{d}.conv"),
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
                &format!("{prefix}.convs2.{d}.conv"),
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

    fn forward(&self, x: &mut [f32], frames: usize) {
        for d in 0..self.convs1.len() {
            let mut xt = x.to_vec();
            snake(&mut xt, &self.alphas1[d], self.channels, frames);
            xt = self.convs1[d].forward(&xt);
            snake(&mut xt, &self.alphas2[d], self.channels, frames);
            xt = self.convs2[d].forward(&xt);
            for (v, r) in x.iter_mut().zip(&xt) {
                *v += r;
            }
        }
    }
}

/// Loaded HiFT generator, batch-of-one (`S3Token2Wav.mel2wav`).
pub struct Hift {
    config: HiftConfig,
    f0_condnet: Vec<SaConv1d>,
    f0_classifier: SaLinear,
    source_linear: SaLinear,
    conv_pre: SaConv1d,
    ups: Vec<SaConvTr1d>,
    source_downs: Vec<SaConv1d>,
    source_resblocks: Vec<ResBlock>,
    resblocks: Vec<ResBlock>,
    conv_post: SaConv1d,
    /// The synthesis window the reference generates at init
    /// (`hanning(istft n_fft, periodic=True)`).
    stft_window: Vec<f32>,
    stft_plan: ComplexFftPlan,
}

impl Hift {
    pub fn load(config: &HiftConfig, file: &SafetensorsFile, prefix: &str) -> Result<Self> {
        let base = config.base_channels;
        let mel = config.in_channels;
        let f0_ch = config.f0_channels;
        // Every turbo conv routes through the `Conv1dPT` wrapper, one
        // `.conv` deeper than the sibling checkpoint's names (the f0
        // predictor list holds the wrapper directly).
        let f0_condnet = (0..5)
            .map(|i| {
                let (in_ch, out_ch) = if i == 0 { (mel, f0_ch) } else { (f0_ch, f0_ch) };
                SaConv1d::load(
                    file,
                    &format!("{prefix}.f0_predictor.condnet.{i}.conv"),
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
        let f0_classifier =
            SaLinear::load(file, &format!("{prefix}.f0_predictor.classifier"), f0_ch, 1)?;
        let source_linear = SaLinear::load(
            file,
            &format!("{prefix}.m_source.l_linear"),
            NB_HARMONICS + 1,
            1,
        )?;
        let conv_pre = SaConv1d::load(
            file,
            &format!("{prefix}.conv_pre.conv"),
            mel,
            base,
            7,
            1,
            3,
            1,
            1,
        )?;

        // Upsampling transpose convs: channels halve per stage.
        let mut ups = Vec::with_capacity(UPSAMPLE_RATES.len());
        for (i, (&u, &k)) in UPSAMPLE_RATES.iter().zip(&UPSAMPLE_KERNELS).enumerate() {
            let (in_ch, out_ch) = (base >> i, base >> (i + 1));
            ups.push(SaConvTr1d::load(
                file,
                &format!("{prefix}.ups.{i}.conv"),
                in_ch,
                out_ch,
                k,
                u,
                (k - u) / 2,
            )?);
        }

        // Source downsampling convs: cumulative rates of [1, 3, 15]
        // reversed, kernel `2u`, stride `u`, padding `u / 2` (the
        // rate-1 stage uses a 1x1 conv).
        let mut source_downs = Vec::with_capacity(UPSAMPLE_RATES.len());
        for (i, &u) in [15usize, 3, 1].iter().enumerate() {
            let (kernel, stride, padding) = if u == 1 { (1, 1, 0) } else { (u * 2, u, u / 2) };
            source_downs.push(SaConv1d::load(
                file,
                &format!("{prefix}.source_downs.{i}.conv"),
                SOURCE_IN,
                base >> (i + 1),
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
                    base >> (i + 1),
                    SOURCE_RESBLOCK_KERNELS[i],
                    &RESBLOCK_DILATIONS,
                )
            })
            .collect::<Result<_>>()?;

        // Main resblocks: after upsample stage i the channel count is
        // base >> (i + 1); three blocks with kernels [3, 7, 11].
        let mut resblocks = Vec::with_capacity(UPSAMPLE_RATES.len() * RESBLOCK_KERNELS.len());
        for i in 0..UPSAMPLE_RATES.len() {
            let ch = base >> (i + 1);
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
            &format!("{prefix}.conv_post.conv"),
            base >> UPSAMPLE_RATES.len(),
            SOURCE_IN,
            7,
            1,
            3,
            1,
            1,
        )?;

        // The window is a periodic Hann (fftbins=True convention).
        let stft_window = dsp::hann_window(ISTFT_N_FFT);
        let stft_plan = ComplexFftPlan::new(ISTFT_N_FFT)?;

        Ok(Hift {
            config: config.clone(),
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

    /// F0 in Hz per mel frame (`F0Predictor`: ELU conv stack, linear,
    /// absolute value).
    pub fn predict_f0(&self, speech_feat: &[f32]) -> Result<Vec<f32>> {
        let mel = self.config.in_channels;
        let frames = speech_feat.len() / mel;
        if speech_feat.len() % mel != 0 {
            return Err(SpeechError::Input {
                why: "speech_feat must be row-major mel frames".to_string(),
            });
        }
        let mut h = to_cm(speech_feat, frames, mel);
        for conv in &self.f0_condnet {
            h = conv.forward(&h);
            elu(&mut h);
        }
        let rows = to_rows(&h, self.config.f0_channels, frames);
        let out = self.f0_classifier.forward(&rows, frames);
        Ok(out.into_iter().map(f32::abs).collect())
    }

    /// Renders mel frames (row-major `[T, 80]`) to 24 kHz samples
    /// (`HiFTGenerator.inference` with an empty `cache_source`).
    pub fn infer(&self, speech_feat: &[f32], draws: Option<&SineDraws>) -> Result<Vec<f32>> {
        let mel = self.config.in_channels;
        let mel_frames = speech_feat.len() / mel;
        let f0 = self.predict_f0(speech_feat)?;

        // Nearest-upsample f0 to the waveform rate.
        let wav_frames = mel_frames * UPSCALE;
        let mut source = vec![0.0f32; wav_frames];
        for (i, s) in source.iter_mut().enumerate() {
            *s = f0[i / UPSCALE];
        }

        // Neural source filter: harmonics at the sample rate, constant
        // initial phase per harmonic (harmonic 0 structural zero),
        // voiced-gated noise bed.
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

        // STFT of the source (uncentered): real and imaginary halves
        // stacked to 18.
        let (s_real, s_imag) = self.stft(&merged)?;
        let stft_frames = s_real.len() / (ISTFT_N_FFT / 2 + 1);
        let mut s_stft = vec![0.0f32; SOURCE_IN * stft_frames];
        s_stft[..(ISTFT_N_FFT / 2 + 1) * stft_frames].copy_from_slice(&s_real);
        s_stft[(ISTFT_N_FFT / 2 + 1) * stft_frames..].copy_from_slice(&s_imag);

        // Pre-conv on the mel.
        let mut x = to_cm(speech_feat, mel_frames, mel);
        x = self.conv_pre.forward(&x);

        let base = self.config.base_channels;
        for i in 0..self.ups.len() {
            leaky_relu(&mut x, UP_LRELU_SLOPE);
            x = self.ups[i].forward(&x);
            let ch = self.ups[i].out_ch;
            let mut frames_i = x.len() / ch;
            if i == self.ups.len() - 1 {
                // The reference pads ONE ZERO on the left of the time
                // axis (`mx.pad(x, [(0, 0), (0, 0), (1, 0)])`).
                let mut padded = vec![0.0f32; ch * (frames_i + 1)];
                for c in 0..ch {
                    padded[c * (frames_i + 1) + 1..(c + 1) * (frames_i + 1)]
                        .copy_from_slice(&x[c * frames_i..(c + 1) * frames_i]);
                }
                x = padded;
                frames_i += 1;
            }

            // Source fusion branch, clamped to the common length (the
            // reference truncates the mel path to `min_len`).
            let mut si = self.source_downs[i].forward(&s_stft);
            let si_frames = si.len() / self.source_downs[i].out_ch;
            self.source_resblocks[i].forward(&mut si, si_frames);
            let min_len = frames_i.min(si_frames);
            let mut fused = vec![0.0f32; ch * min_len];
            for c in 0..ch {
                let src = &x[c * frames_i..c * frames_i + min_len];
                let other = &si[c * si_frames..c * si_frames + min_len];
                for (f, (a, b)) in src.iter().zip(other).enumerate() {
                    fused[c * min_len + f] = a + b;
                }
            }
            x = fused;
            let frames_i = min_len;

            // Average the three parallel resblocks.
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

        // The final leaky ReLU takes the nn default slope.
        leaky_relu(&mut x, FINAL_LRELU_SLOPE);
        let out_frames = x.len() / (base >> self.ups.len());
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

    /// `SineGen` at the turbo phase path: `theta[t] =
    /// (2*pi*cumsum(f0[t]*(h+1)/sr))[t] mod 2*pi` plus the constant
    /// initial phase (harmonic 0 zero). Returns
    /// `[harmonics + 1, T_wav]`.
    fn sine_gen(&self, f0: &[f32], draws: Option<&SineDraws>) -> Result<Vec<f32>> {
        let t_wav = f0.len();
        let harmonics = NB_HARMONICS + 1;
        let default_phase = vec![0.0f32; NB_HARMONICS];
        let phase = draws.map(|d| d.phase.as_slice()).unwrap_or(&default_phase);
        if phase.len() != NB_HARMONICS {
            return Err(SpeechError::Input {
                why: format!("sine phase must have {NB_HARMONICS} entries"),
            });
        }
        if let Some(d) = draws {
            if d.sine_noise.len() != harmonics * t_wav {
                return Err(SpeechError::Input {
                    why: format!(
                        "sine noise must have {} entries, got {}",
                        harmonics * t_wav,
                        d.sine_noise.len()
                    ),
                });
            }
            if !d.source_noise.is_empty() && d.source_noise.len() != t_wav {
                return Err(SpeechError::Input {
                    why: format!(
                        "source noise must have {t_wav} entries, got {}",
                        d.source_noise.len()
                    ),
                });
            }
        }

        let mut sine = vec![0.0f32; harmonics * t_wav];
        let uv: Vec<f32> = f0
            .iter()
            .map(|&f| (f > VOICED_THRESHOLD) as i32 as f32)
            .collect();
        let two_pi = 2.0 * std::f32::consts::PI;
        for (h, row) in sine.chunks_mut(t_wav).enumerate() {
            let mut cum = 0.0f32;
            let initial = if h == 0 { 0.0 } else { phase[h - 1] };
            for (t, slot) in row.iter_mut().enumerate() {
                cum += f0[t] * (h + 1) as f32 / 24_000.0;
                let theta = (two_pi * cum) % two_pi;
                *slot = SINE_AMP * (theta + initial).sin();
            }
        }

        // Voiced gating and the noise bed.
        let zeros = vec![0.0f32; harmonics * t_wav];
        let noise = draws.map(|d| d.sine_noise.as_slice()).unwrap_or(&zeros);
        for (t, &u) in uv.iter().enumerate() {
            let amp = u * NOISE_STD + (1.0 - u) * SINE_AMP / 3.0;
            for h in 0..harmonics {
                let i = h * t_wav + t;
                sine[i] = sine[i] * u + amp * noise[i];
            }
        }
        Ok(sine)
    }

    /// The turbo `_stft`: UNCENTERED frames of the raw waveform (a zero
    /// tail pad only when the input is shorter than one window),
    /// windowed, complex FFT. Returns channel-major `[bins, F]`.
    fn stft(&self, x: &[f32]) -> Result<(Vec<f32>, Vec<f32>)> {
        let mut samples = x.to_vec();
        if samples.len() < ISTFT_N_FFT {
            samples.resize(ISTFT_N_FFT, 0.0);
        }
        let frames = (samples.len() - ISTFT_N_FFT) / ISTFT_HOP + 1;
        let bins = ISTFT_N_FFT / 2 + 1;
        let mut real = vec![0.0f32; bins * frames];
        let mut imag = vec![0.0f32; bins * frames];
        let mut frame_in = vec![ComplexF32::new(0.0, 0.0); ISTFT_N_FFT];
        for f in 0..frames {
            for (t, slot) in frame_in.iter_mut().enumerate() {
                slot.re = samples[f * ISTFT_HOP + t] * self.stft_window[t];
            }
            let spectrum = self.stft_plan.forward(&frame_in)?;
            for (b, c) in spectrum.iter().take(bins).enumerate() {
                real[b * frames + f] = c.re;
                imag[b * frames + f] = c.im;
            }
        }
        Ok((real, imag))
    }

    /// The reference `_istft`: clipped magnitude, inverse FFT, windowed
    /// overlap-add normalized by the squared window sum, trimmed by
    /// n_fft/2 at the front (center=True convention).
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
        // `audio[:, pad : pad + (T - 1) * hop]` with pad = n_fft // 2.
        let trim = ISTFT_N_FFT / 2;
        let keep = (frames - 1) * ISTFT_HOP;
        let mut out = Vec::with_capacity(keep);
        for t in trim..trim + keep {
            let d = denom[t];
            out.push(ola[t] / d.max(1e-8));
        }
        Ok(out)
    }
}
