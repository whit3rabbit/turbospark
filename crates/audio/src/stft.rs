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
use crate::fft::{ComplexF32, RealFftPlan, RealFftScratch};
use std::borrow::Cow;

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
    let plan = RealFftPlan::cached(options.fft_size)?;
    let framer = Framer::new(samples, options, padding_mode, window_placement)?;
    let count = framer.frame_count();
    let mut out = Vec::with_capacity(count);
    let mut fft_input = vec![0.0f32; options.fft_size];
    let mut scratch = RealFftScratch::default();
    for frame in 0..count {
        framer.fill(frame, &mut fft_input);
        let mut spectrum = Vec::with_capacity(options.fft_size / 2 + 1);
        plan.forward_into(&fft_input, &mut spectrum, &mut scratch)?;
        out.push(spectrum);
    }
    Ok(out)
}

/// Streams the forward STFT one frame at a time: `visit` receives each
/// `fft_size / 2 + 1` bin spectrum in time order, borrowed from a buffer that
/// is reused for the next frame. Returns the frame count.
///
/// Numerically identical to [`stft_with_modes`], but a frontend that reduces
/// every frame (power, mel projection) never has to hold, or allocate, the
/// full `frames x bins` spectrogram. Errors match `stft_with_modes`; an error
/// from `visit` stops the transform and is returned.
pub fn stft_each(
    samples: &[f32],
    options: &StftOptions,
    padding_mode: StftPaddingMode,
    window_placement: StftWindowPlacement,
    mut visit: impl FnMut(&[ComplexF32]) -> Result<(), AudioError>,
) -> Result<usize, AudioError> {
    let plan = RealFftPlan::cached(options.fft_size)?;
    let framer = Framer::new(samples, options, padding_mode, window_placement)?;
    let count = framer.frame_count();
    let mut fft_input = vec![0.0f32; options.fft_size];
    let mut spectrum = Vec::new();
    let mut scratch = RealFftScratch::default();
    for frame in 0..count {
        framer.fill(frame, &mut fft_input);
        plan.forward_into(&fft_input, &mut spectrum, &mut scratch)?;
        visit(&spectrum)?;
    }
    Ok(count)
}

/// Forward STFT into one caller-owned buffer: `out` is cleared and filled
/// with `frames * (fft_size / 2 + 1)` bins, frame-major. Returns the frame
/// count. Same values as [`stft_with_modes`] without a `Vec` per frame.
pub fn stft_into(
    samples: &[f32],
    options: &StftOptions,
    padding_mode: StftPaddingMode,
    window_placement: StftWindowPlacement,
    out: &mut Vec<ComplexF32>,
) -> Result<usize, AudioError> {
    out.clear();
    stft_each(samples, options, padding_mode, window_placement, |bins| {
        out.extend_from_slice(bins);
        Ok(())
    })
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
    let plan = RealFftPlan::cached(options.fft_size)?;
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
    //
    // The window is widened to f64 once, zero-extended to the FFT size
    // (the frame past the window still multiplies by 0.0, as before, so a
    // non-finite sample still poisons its slot exactly as it used to).
    let mut window_f64 = vec![0.0f64; options.fft_size];
    for (t, &w) in options.window.iter().enumerate() {
        window_f64[t] = f64::from(w);
    }
    let window_sq: Vec<f64> = window_f64.iter().map(|&w| w * w).collect();
    let mut frame_samples = Vec::new();
    let mut scratch = RealFftScratch::default();
    for (frame, spectrum) in spectra.iter().enumerate() {
        plan.inverse_into(spectrum, &mut frame_samples, &mut scratch)?;
        let start = frame * options.hop;
        for ((slot, &x), &w) in acc[start..start + options.fft_size]
            .iter_mut()
            .zip(&frame_samples)
            .zip(&window_f64)
        {
            *slot += f64::from(x) * w;
        }
    }
    let energy = window_energy(&window_sq, spectra.len(), options.hop, padded_len);
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

/// Window energy per output position: the sum of `window_sq[p - f * hop]`
/// over every frame `f` covering `p`, accumulated in ascending frame order in
/// f64 so hop > window length (energy holes at frame boundaries)
/// reconstructs as silence instead of dividing by zero.
///
/// The textbook loop adds the whole window once per frame, O(frames * fft).
/// Away from the two edges, every position with the same `p % hop` is covered
/// by the same window taps in the same order, so its sum is bit-for-bit one
/// table entry; only the edge positions (where frames are missing) are summed
/// directly. Terms are added in the same order from 0.0, so every position
/// holds the identical f64 as the per-frame loop would produce.
fn window_energy(window_sq: &[f64], frames: usize, hop: usize, padded_len: usize) -> Vec<f64> {
    let fft = window_sq.len();
    let mut energy = vec![0.0f64; padded_len];
    if hop >= fft {
        // At most one frame covers any position: nothing to share.
        for frame in 0..frames {
            let start = frame * hop;
            for (t, &w) in window_sq.iter().enumerate() {
                energy[start + t] += w;
            }
        }
        return energy;
    }
    // Positions in [fft, frames * hop) see a full, steady set of frames.
    let (mid_lo, mid_hi) = if frames * hop > fft {
        (fft, (frames * hop).min(padded_len))
    } else {
        (0, 0)
    };
    let mut table = Vec::new();
    if mid_lo < mid_hi {
        // Frame f covers p = q * hop + r at tap r + (q - f) * hop, so
        // ascending frames walk the taps downward from the highest one.
        table = (0..hop)
            .map(|r| {
                let mut acc = 0.0f64;
                for k in (0..=(fft - 1 - r) / hop).rev() {
                    acc += window_sq[r + k * hop];
                }
                acc
            })
            .collect();
    }
    let mut r = mid_lo % hop;
    for (p, slot) in energy.iter_mut().enumerate() {
        if p >= mid_lo && p < mid_hi {
            *slot = table[r];
            r += 1;
            if r == hop {
                r = 0;
            }
            continue;
        }
        let first = if p >= fft { (p - fft) / hop + 1 } else { 0 };
        let last = (p / hop).min(frames - 1);
        let mut acc = 0.0f64;
        if first <= last {
            for frame in first..=last {
                acc += window_sq[p - frame * hop];
            }
        }
        *slot = acc;
    }
    energy
}

/// Reflect-pads when centered, then returns the buffer frames run over.
/// An uncentered signal is borrowed as is; only padding needs a copy.
fn framed_samples<'a>(
    samples: &'a [f32],
    options: &StftOptions,
    padding_mode: StftPaddingMode,
) -> Result<Cow<'a, [f32]>, AudioError> {
    validate(options)?;
    if !options.center {
        if samples.len() < options.fft_size {
            return Ok(Cow::Borrowed(&[]));
        }
        return Ok(Cow::Borrowed(samples));
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
    Ok(Cow::Owned(padded))
}

/// The padded signal plus the geometry needed to window frame `i` into an
/// FFT input buffer.
struct Framer<'a> {
    buffer: Cow<'a, [f32]>,
    fft_size: usize,
    hop: usize,
    window: &'a [f32],
    /// FFT index of the first window tap.
    window_start: usize,
}

impl<'a> Framer<'a> {
    fn new(
        samples: &'a [f32],
        options: &'a StftOptions,
        padding_mode: StftPaddingMode,
        window_placement: StftWindowPlacement,
    ) -> Result<Self, AudioError> {
        let buffer = framed_samples(samples, options, padding_mode)?;
        // Computed after `framed_samples` validated window <= fft_size.
        let window_start = match window_placement {
            StftWindowPlacement::Left => 0,
            StftWindowPlacement::Center => (options.fft_size - options.window.len()) / 2,
        };
        Ok(Self {
            buffer,
            fft_size: options.fft_size,
            hop: options.hop,
            window: &options.window,
            window_start,
        })
    }

    /// Frames whose full `fft_size` span fits in the buffer.
    fn frame_count(&self) -> usize {
        if self.buffer.len() < self.fft_size {
            0
        } else {
            (self.buffer.len() - self.fft_size) / self.hop + 1
        }
    }

    /// Writes the windowed samples of `frame` into `input`. Only the window
    /// span is written: every other slot is the zero the caller initialized
    /// (a zero-padded window tail or head), and stays zero across frames.
    fn fill(&self, frame: usize, input: &mut [f32]) {
        let first = frame * self.hop + self.window_start;
        let src = &self.buffer[first..first + self.window.len()];
        let dst = &mut input[self.window_start..self.window_start + self.window.len()];
        for ((slot, &x), &w) in dst.iter_mut().zip(src).zip(self.window) {
            *slot = x * w;
        }
    }
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

/// Bitwise parity of the streaming/cached STFT paths and the periodic window
/// energy against the previous implementations, retained verbatim in `old`.
#[cfg(test)]
mod parity_tests {
    use super::*;
    use crate::dsp::{hann_window, TestRng};

    mod old {
        use super::super::*;

        pub(super) fn stft_with_modes(
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
            let mut out = Vec::with_capacity(
                1 + (buffer.len().saturating_sub(options.fft_size)) / options.hop,
            );
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

        pub(super) fn istft(
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
            let energy = energy(&window_sq, spectra.len(), options.hop, padded_len);
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

        /// The per-frame window-sum loop `istft` used to run inline.
        pub(super) fn energy(
            window_sq: &[f64],
            frames: usize,
            hop: usize,
            padded_len: usize,
        ) -> Vec<f64> {
            let mut energy = vec![0.0f64; padded_len];
            for frame in 0..frames {
                let start = frame * hop;
                for (t, &w) in window_sq.iter().enumerate() {
                    energy[start + t] += w;
                }
            }
            energy
        }
    }

    fn assert_spectra_bits(what: &str, got: &[Vec<ComplexF32>], want: &[Vec<ComplexF32>]) {
        assert_eq!(got.len(), want.len(), "{what}: frame count");
        for (f, (g, w)) in got.iter().zip(want).enumerate() {
            assert_eq!(g.len(), w.len(), "{what}: frame {f} length");
            for (i, (a, b)) in g.iter().zip(w).enumerate() {
                assert_eq!(a.re.to_bits(), b.re.to_bits(), "{what}: [{f}][{i}].re");
                assert_eq!(a.im.to_bits(), b.im.to_bits(), "{what}: [{f}][{i}].im");
            }
        }
    }

    fn assert_f32_bits(what: &str, got: &[f32], want: &[f32]) {
        assert_eq!(got.len(), want.len(), "{what}: length");
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            assert_eq!(g.to_bits(), w.to_bits(), "{what}[{i}]: {g} vs {w}");
        }
    }

    /// (fft, hop, window length, center, signal length)
    const CASES: [(usize, usize, usize, bool, usize); 10] = [
        (400, 160, 400, true, 3_000),
        (512, 160, 400, true, 4_100),
        (256, 80, 200, true, 1_000),
        (16, 4, 16, true, 70),
        (16, 4, 10, true, 70),
        (64, 16, 64, false, 500),
        (64, 16, 33, false, 500),
        (64, 100, 64, false, 800),
        (32, 5, 32, true, 129),
        (400, 160, 400, false, 100),
    ];

    fn random_window(rng: &mut TestRng, len: usize) -> Vec<f32> {
        // Non-negative taps with exact zeros, unlike a plain Hann.
        rng.vec(len).into_iter().map(f32::abs).collect()
    }

    #[test]
    fn streaming_stft_matches_old_stft_bitwise() {
        let mut rng = TestRng(0x1234_5678_9ABC_DEF1);
        for (case, &(fft, hop, wlen, center, len)) in CASES.iter().enumerate() {
            for window in [hann_window(wlen), random_window(&mut rng, wlen)] {
                let options = StftOptions {
                    fft_size: fft,
                    hop,
                    window,
                    center,
                };
                let samples = rng.vec(len);
                for padding in [StftPaddingMode::Reflect, StftPaddingMode::Constant] {
                    for placement in [StftWindowPlacement::Left, StftWindowPlacement::Center] {
                        let want =
                            old::stft_with_modes(&samples, &options, padding, placement).unwrap();
                        let tag = format!("case {case} {padding:?} {placement:?}");
                        let got = stft_with_modes(&samples, &options, padding, placement).unwrap();
                        assert_spectra_bits(&tag, &got, &want);

                        let mut visited = Vec::new();
                        let n = stft_each(&samples, &options, padding, placement, |bins| {
                            visited.push(bins.to_vec());
                            Ok(())
                        })
                        .unwrap();
                        assert_eq!(n, want.len(), "{tag}: stft_each count");
                        assert_spectra_bits(&tag, &visited, &want);

                        let mut flat = vec![ComplexF32::new(9.0, 9.0); 3];
                        let n =
                            stft_into(&samples, &options, padding, placement, &mut flat).unwrap();
                        assert_eq!(n, want.len());
                        let bins = fft / 2 + 1;
                        let rows: Vec<Vec<ComplexF32>> =
                            flat.chunks(bins).map(<[ComplexF32]>::to_vec).collect();
                        assert_eq!(flat.len(), n * bins, "{tag}: stft_into length");
                        assert_spectra_bits(&tag, &rows, &want);
                    }
                }
            }
        }
    }

    #[test]
    fn stft_errors_match_old_stft() {
        let ok = StftOptions {
            fft_size: 16,
            hop: 4,
            window: hann_window(16),
            center: true,
        };
        let bad = [
            // Odd FFT size, zero hop, long window, too-short reflect input.
            (
                StftOptions {
                    fft_size: 7,
                    ..ok.clone()
                },
                40usize,
            ),
            (
                StftOptions {
                    hop: 0,
                    ..ok.clone()
                },
                40,
            ),
            (
                StftOptions {
                    window: hann_window(20),
                    ..ok.clone()
                },
                40,
            ),
            (ok.clone(), 4),
            (
                StftOptions {
                    fft_size: 0,
                    ..ok.clone()
                },
                40,
            ),
        ];
        for (options, len) in bad {
            let samples = vec![0.5f32; len];
            let want = old::stft_with_modes(
                &samples,
                &options,
                StftPaddingMode::Reflect,
                StftWindowPlacement::Left,
            )
            .unwrap_err();
            let got = stft_each(
                &samples,
                &options,
                StftPaddingMode::Reflect,
                StftWindowPlacement::Left,
                |_| Ok(()),
            )
            .unwrap_err();
            assert_eq!(got.to_string(), want.to_string());
            let got = stft(&samples, &options).unwrap_err();
            assert_eq!(got.to_string(), want.to_string());
        }
    }

    #[test]
    fn istft_matches_old_istft_bitwise() {
        let mut rng = TestRng(0x0F0F_1234_ABCD_9876);
        for (case, &(fft, hop, wlen, center, len)) in CASES.iter().enumerate() {
            for window in [hann_window(wlen), random_window(&mut rng, wlen)] {
                let options = StftOptions {
                    fft_size: fft,
                    hop,
                    window,
                    center,
                };
                let samples = rng.vec(len);
                let spectra = old::stft_with_modes(
                    &samples,
                    &options,
                    StftPaddingMode::Reflect,
                    StftWindowPlacement::Left,
                )
                .unwrap();
                let want = old::istft(&spectra, &options, len).unwrap();
                let got = istft(&spectra, &options, len).unwrap();
                assert_f32_bits(&format!("istft case {case}"), &got, &want);
                // A non-finite sample must poison the same slots as before
                // (zero-extended window taps still multiply by 0.0).
                let mut poisoned = spectra.clone();
                if let Some(first) = poisoned.first_mut() {
                    first[1].re = f32::INFINITY;
                }
                let want = old::istft(&poisoned, &options, len).unwrap();
                let got = istft(&poisoned, &options, len).unwrap();
                assert_f32_bits(&format!("istft inf case {case}"), &got, &want);
            }
        }
    }

    #[test]
    fn window_energy_matches_per_frame_loop_bitwise() {
        let mut rng = TestRng(0xC0FF_EE00_DEAD_BEEF);
        // (fft, hop, frames): hop below, equal to, and above the FFT size,
        // hops that do not divide it, and frame counts around the point
        // where the steady region first appears.
        let shapes = [
            (512usize, 128usize, 1usize),
            (512, 128, 2),
            (512, 128, 5),
            (512, 128, 40),
            (400, 160, 3),
            (400, 160, 30),
            (64, 100, 5),
            (64, 64, 4),
            (64, 1, 70),
            (64, 1, 3),
            (16, 3, 9),
            (16, 5, 2),
            (16, 15, 8),
            (16, 7, 1),
            (32, 4, 8),
            (32, 4, 9),
            (32, 31, 6),
            (2, 1, 5),
            (1, 1, 4),
        ];
        for (fft, hop, frames) in shapes {
            let padded_len = (frames - 1) * hop + fft;
            for wlen in [fft, fft.div_ceil(2)] {
                for window in [hann_window(wlen), random_window(&mut rng, wlen)] {
                    let mut window_sq = vec![0.0f64; fft];
                    for (t, &w) in window.iter().enumerate() {
                        window_sq[t] = f64::from(w) * f64::from(w);
                    }
                    let want = old::energy(&window_sq, frames, hop, padded_len);
                    let got = window_energy(&window_sq, frames, hop, padded_len);
                    assert_eq!(got.len(), want.len());
                    for (p, (g, w)) in got.iter().zip(&want).enumerate() {
                        assert_eq!(
                            g.to_bits(),
                            w.to_bits(),
                            "fft {fft} hop {hop} frames {frames} wlen {wlen} p {p}"
                        );
                    }
                }
            }
        }
    }
}
