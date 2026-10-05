//! Short-time Fourier transform and its inverse.
//!
//! Framing follows the torch.stft conventions the model frontends assume:
//! optional centering with reflect padding, a caller-supplied window
//! zero-padded to the FFT size, and `1 + len / hop` frames when centered.
//! The inverse reconstructs by overlap-add with window-sum-of-squares
//! normalization, so Hann windows at hop <= fft/2 reconstruct to round-off.
//!
//! Frames are `Vec<ComplexF32>` rows of `fft_size / 2 + 1` bins, emitted in
//! time order. A plan is built once per call and reused across frames; when
//! transforming many signals of the same shape, build the plan once
//! yourself and frame through it (the mel module shows the pattern).

use crate::error::AudioError;
use crate::fft::{ComplexF32, RealFftPlan};

#[derive(Debug, Clone)]
pub struct StftOptions {
    /// FFT size per frame; even (or 1), at least as large as the window.
    pub fft_size: usize,
    /// Hop between frame starts in samples; must be positive.
    pub hop: usize,
    /// Analysis window, at most `fft_size` long (zero-padded on the right).
    pub window: Vec<f32>,
    /// Reflect-pad `fft_size / 2` samples on both sides before framing, the
    /// torch `center=True` behavior.
    pub center: bool,
}

/// Signal-edge handling for centered STFT frames.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StftPaddingMode {
    /// Reflect-pad without repeating the edge sample, matching torch.stft.
    Reflect,
    /// Pad both sides with zero, matching MLX audio preprocessing.
    Constant,
}

/// Placement of a short analysis window inside the FFT frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StftWindowPlacement {
    /// Start at FFT index 0 and zero-pad on the right.
    Left,
    /// Split zero-padding around the window, with the extra sample on the right.
    Center,
}

/// Forward STFT: one complex spectrum row per frame.
pub fn stft(samples: &[f32], options: &StftOptions) -> Result<Vec<Vec<ComplexF32>>, AudioError> {
    stft_with_modes(
        samples,
        options,
        StftPaddingMode::Reflect,
        StftWindowPlacement::Left,
    )
}

/// Forward STFT with explicit signal padding and short-window placement.
///
/// The default [stft] contract remains reflect-centered signal padding and a
/// left-aligned analysis window. MLX audio frontends use zero signal padding
/// and center the window within each FFT frame.
pub fn stft_with_modes(
    samples: &[f32],
    options: &StftOptions,
    padding_mode: StftPaddingMode,
    window_placement: StftWindowPlacement,
) -> Result<Vec<Vec<ComplexF32>>, AudioError> {
    let plan = RealFftPlan::new(options.fft_size)?;
    let buffer = framed_samples(samples, options, padding_mode)?;
    let window_start = match window_placement {
        StftWindowPlacement::Left => 0,
        StftWindowPlacement::Center => (options.fft_size - options.window.len()) / 2,
    };
    let mut out =
        Vec::with_capacity(1 + (buffer.len().saturating_sub(options.fft_size)) / options.hop);
    let mut fft_input = vec![0.0f32; options.fft_size];
    for start in (0..buffer.len()).step_by(options.hop) {
        if start + options.fft_size > buffer.len() {
            break;
        }
        for (t, slot) in fft_input.iter_mut().enumerate() {
            let window_index = t.checked_sub(window_start);
            *slot = window_index
                .and_then(|i| options.window.get(i))
                .map_or(0.0, |&w| buffer[start + t] * w);
        }
        out.push(plan.forward(&fft_input)?);
    }
    Ok(out)
}

/// Inverse STFT: overlap-add the spectra back to a real signal.
///
/// With `center` set, the `fft_size / 2` padding is trimmed from both ends,
/// so the output length equals the input length that produced the frames.
/// Positions whose window energy sums to zero (possible only for windows
/// that vanish where every frame boundary lands) stay zero rather than
/// dividing by nothing.
pub fn istft(
    spectra: &[Vec<ComplexF32>],
    options: &StftOptions,
    original_len: usize,
) -> Result<Vec<f32>, AudioError> {
    validate(options)?;
    let plan = RealFftPlan::new(options.fft_size)?;
    let pad = if options.center {
        options.fft_size / 2
    } else {
        0
    };
    if spectra.is_empty() {
        return Ok(Vec::new());
    }
    let padded_len = (spectra.len() - 1)
        .checked_mul(options.hop)
        .and_then(|n| n.checked_add(options.fft_size))
        .filter(|&n| n <= 1 << 24)
        .ok_or(AudioError::BufferTooLarge {
            what: "inverse STFT output",
            samples: usize::MAX,
        })?;
    let mut acc = vec![0.0f64; padded_len];
    // The inverse FFT returns the windowed frame w(t) * s(...), so the
    // overlap-add numerator needs one more window factor: w^2 * s summed
    // over frames, divided by the window energy sum. Squaring twice here
    // scales the output by sum(w^3) / sum(w^2) -- for a periodic Hann at
    // quarter overlap, exactly 5/6.
    let mut window_sq = vec![0.0f64; options.fft_size];
    for (t, &w) in options.window.iter().enumerate() {
        window_sq[t] = f64::from(w) * f64::from(w);
    }
    for (frame, spectrum) in spectra.iter().enumerate() {
        let frames = plan.inverse(spectrum)?;
        let start = frame * options.hop;
        for (t, &x) in frames.iter().enumerate() {
            acc[start + t] +=
                f64::from(x) * f64::from(options.window.get(t).copied().unwrap_or(0.0));
        }
    }
    // Window energy per output position, accumulated per frame so hop >
    // window length (energy holes at frame boundaries) reconstructs as
    // silence instead of dividing by zero.
    let mut energy = vec![0.0f64; padded_len];
    for frame in 0..spectra.len() {
        let start = frame * options.hop;
        for (t, &w) in window_sq.iter().enumerate() {
            energy[start + t] += w;
        }
    }
    let mut out = Vec::with_capacity(original_len.min(padded_len.saturating_sub(pad)));
    for t in 0..original_len {
        let padded_t = t + pad;
        if padded_t >= padded_len {
            break;
        }
        let value = if energy[padded_t] > 1e-12 {
            acc[padded_t] / energy[padded_t]
        } else {
            0.0
        };
        out.push(value as f32);
    }
    Ok(out)
}

/// Reflect-pads when centered, then returns the buffer frames run over.
fn framed_samples(
    samples: &[f32],
    options: &StftOptions,
    padding_mode: StftPaddingMode,
) -> Result<Vec<f32>, AudioError> {
    validate(options)?;
    if !options.center {
        if samples.len() < options.fft_size {
            return Ok(Vec::new());
        }
        return Ok(samples.to_vec());
    }
    let pad = options.fft_size / 2;
    if padding_mode == StftPaddingMode::Reflect && samples.len() < pad + 1 {
        return Err(AudioError::InvalidParameter {
            name: "samples".to_string(),
            value: samples.len().to_string(),
            why: format!(
                "centered STFT needs at least {} samples to reflect-pad both sides",
                pad + 1
            ),
        });
    }
    let padded_len = samples
        .len()
        .checked_add(options.fft_size)
        .filter(|&n| n <= 1 << 24)
        .ok_or(AudioError::BufferTooLarge {
            what: "centered STFT input",
            samples: usize::MAX,
        })?;
    let mut padded = vec![0.0f32; padded_len];
    padded[pad..pad + samples.len()].copy_from_slice(samples);
    if padding_mode == StftPaddingMode::Reflect {
        // torch reflect: the edge sample is not repeated.
        for i in 0..pad {
            padded[i] = samples[pad - i];
        }
        for j in 0..pad {
            padded[pad + samples.len() + j] = samples[samples.len() - 2 - j];
        }
    }
    Ok(padded)
}

fn validate(options: &StftOptions) -> Result<(), AudioError> {
    if options.hop == 0 {
        return Err(AudioError::InvalidParameter {
            name: "hop".to_string(),
            value: "0".to_string(),
            why: "hop must be positive".to_string(),
        });
    }
    if options.window.len() > options.fft_size {
        return Err(AudioError::InvalidParameter {
            name: "window length".to_string(),
            value: options.window.len().to_string(),
            why: format!("window must be at most the FFT size {}", options.fft_size),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsp::hann_window;

    fn sine(len: usize, cycles: f64) -> Vec<f32> {
        (0..len)
            .map(|i| (2.0 * std::f64::consts::PI * cycles * i as f64 / len as f64).sin() as f32)
            .collect()
    }

    fn options(fft: usize, hop: usize, center: bool) -> StftOptions {
        StftOptions {
            fft_size: fft,
            hop,
            window: hann_window(fft),
            center,
        }
    }

    #[test]
    fn frame_counts_match_torch_conventions() {
        let samples = vec![0.0f32; 1000];
        // Centered: 1 + 1000/256 = 4 frames at fft 512.
        assert_eq!(stft(&samples, &options(512, 256, true)).unwrap().len(), 4);
        // Uncentered: 1 + (1000-512)/256 = 2.
        assert_eq!(stft(&samples, &options(512, 256, false)).unwrap().len(), 2);
        // Uncentered with a shorter-than-frame input is zero frames.
        assert!(stft(&[0.0; 10], &options(512, 256, false))
            .unwrap()
            .is_empty());
    }

    #[test]
    fn centered_short_input_is_refused() {
        let err = stft(&[0.0; 4], &options(512, 256, true)).unwrap_err();
        assert!(err.to_string().contains("reflect-pad"), "{err}");
    }

    #[test]
    fn rejects_zero_hop_and_long_window() {
        let opts = StftOptions {
            fft_size: 512,
            hop: 0,
            window: hann_window(512),
            center: false,
        };
        let err = stft(&[0.0; 600], &opts).unwrap_err();
        assert!(err.to_string().contains("hop must be positive"), "{err}");
        let opts = StftOptions {
            fft_size: 512,
            hop: 256,
            window: hann_window(600),
            center: false,
        };
        let err = stft(&[0.0; 600], &opts).unwrap_err();
        assert!(err.to_string().contains("at most the FFT size"), "{err}");
    }

    #[test]
    fn perfect_reconstruction_hann_quarter_overlap() {
        for center in [true, false] {
            let original = sine(4096, 37.0);
            let opts = options(512, 128, center);
            let spectra = stft(&original, &opts).unwrap();
            let back = istft(&spectra, &opts, original.len()).unwrap();
            assert_eq!(back.len(), original.len());
            // The interior reconstructs to round-off. Without centering the
            // first and last fft-hop samples hang on a single frame whose
            // window tap is near zero: exact in theory, ill-conditioned in
            // f32, so the tails get a loose bound instead.
            let (interior_start, interior_end) = match center {
                true => (0, original.len()),
                false => (512, original.len() - 512),
            };
            let interior_err = original
                .iter()
                .zip(&back)
                .enumerate()
                .filter(|(i, _)| *i >= interior_start && *i < interior_end)
                .map(|(_, (a, b))| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                interior_err < 2e-4,
                "center={center} interior error {interior_err}"
            );
            if !center {
                let tail_err = original
                    .iter()
                    .zip(&back)
                    .enumerate()
                    .filter(|(i, _)| *i < 512 || *i >= original.len() - 512)
                    .map(|(_, (a, b))| (a - b).abs())
                    .fold(0.0f32, f32::max);
                assert!(tail_err < 1e-2, "uncentered tail error {tail_err}");
            }
        }
    }

    #[test]
    fn tone_lands_in_the_right_bin() {
        let fft = 512usize;
        let bin = 16;
        let original: Vec<f32> = (0..4096)
            .map(|i| (2.0 * std::f64::consts::PI * bin as f64 * i as f64 / fft as f64).sin() as f32)
            .collect();
        let opts = options(fft, 128, true);
        let spectra = stft(&original, &opts).unwrap();
        let peak = spectra[4]
            .iter()
            .enumerate()
            .max_by(|a, b| {
                let am = a.1.re * a.1.re + a.1.im * a.1.im;
                let bm = b.1.re * b.1.re + b.1.im * b.1.im;
                am.partial_cmp(&bm).unwrap()
            })
            .map(|(i, _)| i)
            .unwrap();
        assert_eq!(peak, bin);
    }
}

#[cfg(test)]
mod short_window_regression {
    use super::*;

    #[test]
    fn inverse_accepts_the_zero_padded_analysis_window() {
        let options = StftOptions {
            fft_size: 8,
            hop: 4,
            window: vec![1.0; 4],
            center: false,
        };
        let input: Vec<f32> = (1..=12).map(|n| n as f32).collect();
        let spectra = stft(&input, &options).unwrap();
        let output = istft(&spectra, &options, input.len()).unwrap();
        for (actual, expected) in output[..8].iter().zip(&input[..8]) {
            assert!((actual - expected).abs() < 1e-5);
        }
        assert_eq!(&output[8..], &[0.0; 4]);
    }
}

#[cfg(test)]
mod allocation_regression {
    use super::*;
    #[test]
    fn inverse_refuses_frame_span_overflow_and_bounds_requested_capacity() {
        let options = StftOptions {
            fft_size: 4,
            hop: usize::MAX,
            window: vec![1.0; 4],
            center: false,
        };
        let spectra = vec![vec![ComplexF32::default(); 3]; 2];
        assert!(matches!(
            istft(&spectra, &options, 4),
            Err(AudioError::BufferTooLarge { .. })
        ));
        let options = StftOptions { hop: 4, ..options };
        assert_eq!(istft(&spectra, &options, usize::MAX).unwrap().len(), 8);
        assert!(istft(&[], &options, usize::MAX).unwrap().is_empty());
    }
}
