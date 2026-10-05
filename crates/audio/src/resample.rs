//! Mono resampling: linear interpolation and the torchaudio-compatible
//! sinc-Hann polyphase filter.
//!
//! Port of upstream audio.cpp's `resampling.cpp`, minus the soxr backend.
//! Upstream tries a dynamically loaded libsoxr first and falls back to
//! linear interpolation when the library is absent. Dynamic loading breaks
//! this crate's portable pure-Rust contract, so the soxr arm is not ported;
//! the two portable strategies cover the frontend need:
//!
//! - [`resample_mono_linear`], the upstream fallback, for cheap preview and
//!   length-matching work.
//! - [`resample_mono_sinc_hann`], a port of the torchaudio `resample`
//!   kernel: a polyphase FIR with a Hann-windowed sinc, with the same rate
//!   reduction, cutoff, kernel width, and numeric modes, so results line up
//!   with torchaudio's default path for the same options.
//!
//! Two documented deviations from upstream: the per-parameter kernel cache
//! (a mutex-guarded global) is replaced by recomputation per call, and the
//! soxr output-length policies do not apply. Neither changes the sinc-Hann
//! or linear results themselves.

use crate::error::AudioError;

/// Upper bound on one resampler output allocation (16 GiB of f32). `vec!`
/// aborts the process on allocation failure instead of returning an error,
/// so absurd output sizes are refused before the buffer is reserved.
const MAX_RESAMPLE_OUTPUT_SAMPLES: usize = 1 << 32;
// Coprime rates can expand the polyphase table independently of input size.
// Refuse more than 64 MiB before constructing that table.
const MAX_RESAMPLE_KERNEL_SAMPLES: usize = 1 << 24;

/// Kernel construction precision for [`resample_mono_sinc_hann`], mirroring
/// torchaudio's numeric modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelComputation {
    /// Whole kernel math in f32, matching torchaudio's default GPU-style
    /// path.
    Float32,
    /// f64 kernel math, rounded through f32 on store.
    Float64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Accumulation {
    Float32,
    Float64,
}

/// Options for [`resample_mono_sinc_hann`]. Defaults replicate
/// torchaudio's `resample` defaults and upstream's
/// `torchaudio_sinc_hann_float32_options()`: rolloff 0.99, filter width 6,
/// f32 kernel math, f32 accumulation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SincHannOptions {
    /// Lowpass cutoff as a fraction of Nyquist; must be in (0, 1].
    pub rolloff: f64,
    /// Half-width of the sinc in zero crossings; larger is sharper and
    /// slower.
    pub lowpass_filter_width: usize,
    pub computation: KernelComputation,
    pub accumulation: Accumulation,
}

impl Default for SincHannOptions {
    fn default() -> Self {
        Self {
            rolloff: 0.99,
            lowpass_filter_width: 6,
            computation: KernelComputation::Float32,
            accumulation: Accumulation::Float32,
        }
    }
}

/// Resamples mono audio by linear interpolation between neighboring input
/// samples, the upstream fallback kernel.
///
/// Output index `i` maps to source position `i * source / target`; edges
/// clamp to the last sample. Output count is `round(n * target / source)`.
pub fn resample_mono_linear(
    input: &[f32],
    source_rate: u32,
    target_rate: u32,
) -> Result<Vec<f32>, AudioError> {
    check_rates(source_rate, target_rate)?;
    if input.is_empty() || source_rate == target_rate {
        return Ok(input.to_vec());
    }
    let scale = f64::from(target_rate) / f64::from(source_rate);
    let out_count = (input.len() as f64 * scale).round() as usize;
    if out_count > MAX_RESAMPLE_OUTPUT_SAMPLES {
        return Err(AudioError::BufferTooLarge {
            what: "resampled output",
            samples: out_count,
        });
    }
    let mut out = Vec::with_capacity(out_count);
    let last = input.len() - 1;
    for i in 0..out_count {
        let pos = i as f64 / scale;
        let left = (pos as usize).min(last);
        let right = (left + 1).min(last);
        let frac = (pos - left as f64) as f32;
        let a = input[left];
        let b = input[right];
        out.push(a + (b - a) * frac);
    }
    Ok(out)
}

/// Resamples mono audio with a polyphase FIR whose kernel is a Hann-windowed
/// sinc, replicating torchaudio's `resample` for the same options.
///
/// Rates are reduced by their GCD before the kernel is built (torchaudio's
/// reduction), so 44100 -> 16000 shares the kernel torchaudio builds for the
/// reduced pair. Input outside `[0, len)` is treated as zeros at the edges.
/// Output count is `ceil(n * target / source)`.
pub fn resample_mono_sinc_hann(
    input: &[f32],
    source_rate: u32,
    target_rate: u32,
    options: &SincHannOptions,
) -> Result<Vec<f32>, AudioError> {
    check_rates(source_rate, target_rate)?;
    if !(0.0 < options.rolloff && options.rolloff <= 1.0) {
        return Err(AudioError::InvalidParameter {
            name: "rolloff".to_string(),
            value: options.rolloff.to_string(),
            why: "must be in (0, 1]".to_string(),
        });
    }
    if options.lowpass_filter_width == 0 {
        return Err(AudioError::InvalidParameter {
            name: "lowpass_filter_width".to_string(),
            value: "0".to_string(),
            why: "must be positive".to_string(),
        });
    }
    if input.is_empty() || source_rate == target_rate {
        return Ok(input.to_vec());
    }

    let g = gcd(source_rate, target_rate);
    let orig_freq = (source_rate / g) as usize;
    let new_freq = (target_rate / g) as usize;

    let base_freq = (orig_freq.min(new_freq) as f64) * options.rolloff;
    let zero_taps = (options.lowpass_filter_width as f64) * orig_freq as f64 / base_freq;
    let width = zero_taps.ceil() as usize;
    let kernel_size = width
        .checked_mul(2)
        .and_then(|n| n.checked_add(orig_freq))
        .ok_or(AudioError::BufferTooLarge {
            what: "resampling kernel",
            samples: usize::MAX,
        })?;
    let kernel_samples = new_freq
        .checked_mul(kernel_size)
        .ok_or(AudioError::BufferTooLarge {
            what: "resampling kernel",
            samples: usize::MAX,
        })?;
    if kernel_samples > MAX_RESAMPLE_KERNEL_SAMPLES {
        return Err(AudioError::BufferTooLarge {
            what: "resampling kernel",
            samples: kernel_samples,
        });
    }

    let kernel = build_kernel(orig_freq, new_freq, base_freq, width, options);

    // Output index (b, p) in block b, phase p samples the input around
    // (b * new_freq + p) * orig_freq / new_freq; kernel tap j reads input
    // block_start + j - width. Taps whose unclamped |t| reaches the filter
    // width are exactly zero (the clamped window closes there), so each
    // phase only iterates its active range.
    let len = input.len();
    let blocks = len.div_ceil(orig_freq);
    let out_count = len
        .checked_mul(new_freq)
        .ok_or(AudioError::BufferTooLarge {
            what: "resampled output",
            samples: usize::MAX,
        })?
        .div_ceil(orig_freq);
    let allocated = blocks
        .checked_mul(new_freq)
        .ok_or(AudioError::BufferTooLarge {
            what: "resampled output",
            samples: usize::MAX,
        })?;
    if allocated > MAX_RESAMPLE_OUTPUT_SAMPLES {
        return Err(AudioError::BufferTooLarge {
            what: "resampled output",
            samples: allocated,
        });
    }
    let mut out = vec![0.0f32; allocated];

    for (block, out_row) in out.chunks_mut(new_freq).enumerate() {
        let block_start = block * orig_freq;
        for (phase, out_sample) in out_row.iter_mut().enumerate() {
            let center_tap = width as f64 + phase as f64 * orig_freq as f64 / new_freq as f64;
            let half = zero_taps + 1.0;
            let start = ((center_tap - half).floor().max(0.0) as usize).min(kernel_size);
            let end = ((center_tap + half).ceil() as usize).min(kernel_size);
            let in_first = block_start as isize + start as isize - width as isize;
            let in_last = block_start + end - width; // exclusive when in range

            let acc = if in_first >= 0 && in_last <= len {
                let window = &input[in_first as usize..in_last];
                sum_taps(
                    &kernel[phase * kernel_size + start..phase * kernel_size + end],
                    window,
                )
            } else {
                sum_taps_zero_padded(
                    &kernel[phase * kernel_size + start..phase * kernel_size + end],
                    input,
                    in_first,
                )
            };
            *out_sample = match options.accumulation {
                Accumulation::Float32 => acc.0,
                Accumulation::Float64 => acc.1 as f32,
            };
        }
    }
    out.truncate(out_count);
    Ok(out)
}

/// Sums `kernel[j] * window[j]` over aligned slices. Returns both
/// accumulations so the caller picks per its precision mode; the unused one
/// is a short sum the optimizer folds away.
fn sum_taps(kernel: &[f32], window: &[f32]) -> (f32, f64) {
    let mut acc32 = 0.0f32;
    let mut acc64 = 0.0f64;
    for (&k, &s) in kernel.iter().zip(window) {
        acc32 += k * s;
        acc64 += f64::from(k) * f64::from(s);
    }
    (acc32, acc64)
}

/// Same sum with the window conceptually extended by zeros on both sides.
/// `first` may be negative; `kernel.len()` taps are consumed from `first`.
fn sum_taps_zero_padded(kernel: &[f32], input: &[f32], first: isize) -> (f32, f64) {
    let mut acc32 = 0.0f32;
    let mut acc64 = 0.0f64;
    let len = input.len() as isize;
    for (j, &k) in kernel.iter().enumerate() {
        let idx = first + j as isize;
        if idx < 0 || idx >= len {
            continue;
        }
        let s = input[idx as usize];
        acc32 += k * s;
        acc64 += f64::from(k) * f64::from(s);
    }
    (acc32, acc64)
}

fn build_kernel(
    orig_freq: usize,
    new_freq: usize,
    base_freq: f64,
    width: usize,
    options: &SincHannOptions,
) -> Vec<f32> {
    let kernel_size = 2 * width + orig_freq;
    let lowpass = options.lowpass_filter_width;
    let scale = base_freq / orig_freq as f64;
    let mut kernel = vec![0.0f32; new_freq * kernel_size];
    for phase in 0..new_freq {
        for (j, slot) in kernel[phase * kernel_size..(phase + 1) * kernel_size]
            .iter_mut()
            .enumerate()
        {
            // Kernel taps carry idx = j - width, spanning
            // [-width, width + orig_freq); the sinc peak for a phase sits
            // at idx = phase * orig_freq / new_freq, the fractional input
            // position that phase interpolates.
            *slot = match options.computation {
                KernelComputation::Float32 => {
                    let t = ((j as f32 - width as f32) / orig_freq as f32
                        - phase as f32 / new_freq as f32)
                        * base_freq as f32;
                    let t = t.clamp(-(lowpass as f32), lowpass as f32);
                    let w = 0.5 + 0.5 * (std::f32::consts::PI * t / lowpass as f32).cos();
                    scale as f32 * sinc_f32(t * std::f32::consts::PI) * w
                }
                KernelComputation::Float64 => {
                    let t = ((j as f64 - width as f64) / orig_freq as f64
                        - phase as f64 / new_freq as f64)
                        * base_freq;
                    let t = t.clamp(-(lowpass as f64), lowpass as f64);
                    let w = 0.5 + 0.5 * (std::f64::consts::PI * t / lowpass as f64).cos();
                    (scale * sinc_f64(t * std::f64::consts::PI) * w) as f32
                }
            };
        }
    }
    kernel
}

fn sinc_f32(x: f32) -> f32 {
    if x == 0.0 {
        1.0
    } else {
        x.sin() / x
    }
}

fn sinc_f64(x: f64) -> f64 {
    if x == 0.0 {
        1.0
    } else {
        x.sin() / x
    }
}

fn gcd(a: u32, b: u32) -> u32 {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

fn check_rates(source_rate: u32, target_rate: u32) -> Result<(), AudioError> {
    for (name, rate) in [("source_rate", source_rate), ("target_rate", target_rate)] {
        if rate == 0 {
            return Err(AudioError::InvalidParameter {
                name: name.to_string(),
                value: "0".to_string(),
                why: "sample rates must be positive".to_string(),
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ramp(n: usize) -> Vec<f32> {
        (0..n).map(|i| i as f32).collect()
    }

    #[test]
    fn linear_identity_at_same_rate() {
        let input = ramp(17);
        let out = resample_mono_linear(&input, 8_000, 8_000).unwrap();
        assert_eq!(out, input);
    }

    #[test]
    fn linear_rejects_zero_rate() {
        let err = resample_mono_linear(&[0.0], 0, 16_000).unwrap_err();
        assert!(err.to_string().contains("source_rate"), "{err}");
    }

    #[test]
    fn linear_length_formula() {
        // 100 samples at 48000 -> 16000 is 33.33 -> 33 output samples.
        let out = resample_mono_linear(&ramp(100), 48_000, 16_000).unwrap();
        assert_eq!(out.len(), 33);
        // 100 samples at 16000 -> 48000 is 300 exactly.
        let out = resample_mono_linear(&ramp(100), 16_000, 48_000).unwrap();
        assert_eq!(out.len(), 300);
    }

    #[test]
    fn linear_preserves_constants() {
        let input = vec![0.25f32; 50];
        let out = resample_mono_linear(&input, 44_100, 22_050).unwrap();
        assert!(out.iter().all(|&s| (s - 0.25).abs() < 1e-6));
    }

    #[test]
    fn linear_downsample_2_to_1_lands_on_input_grid() {
        // Integer 2:1 decimation with a ramp: every output position is an
        // even input index, so the values match exactly.
        let input = ramp(20);
        let out = resample_mono_linear(&input, 2, 1).unwrap();
        assert_eq!(
            out,
            vec![0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0]
        );
    }

    #[test]
    fn sinc_hann_identity_and_empty() {
        let input = ramp(9);
        assert_eq!(
            resample_mono_sinc_hann(&input, 16_000, 16_000, &SincHannOptions::default()).unwrap(),
            input
        );
        assert!(
            resample_mono_sinc_hann(&[], 16_000, 8_000, &SincHannOptions::default())
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn sinc_hann_rejects_bad_options() {
        let bad = SincHannOptions {
            rolloff: 1.5,
            ..SincHannOptions::default()
        };
        let err = resample_mono_sinc_hann(&[0.0], 8_000, 4_000, &bad).unwrap_err();
        assert!(err.to_string().contains("rolloff"), "{err}");
        let bad = SincHannOptions {
            lowpass_filter_width: 0,
            ..SincHannOptions::default()
        };
        let err = resample_mono_sinc_hann(&[0.0], 8_000, 4_000, &bad).unwrap_err();
        assert!(err.to_string().contains("lowpass_filter_width"), "{err}");
    }

    #[test]
    fn sinc_hann_length_formula() {
        let out = resample_mono_sinc_hann(&ramp(100), 48_000, 16_000, &SincHannOptions::default())
            .unwrap();
        assert_eq!(out.len(), 34); // ceil(100 * 16000 / 48000)
        let out = resample_mono_sinc_hann(&ramp(100), 16_000, 48_000, &SincHannOptions::default())
            .unwrap();
        assert_eq!(out.len(), 300);
    }

    #[test]
    fn sinc_hann_preserves_dc_level() {
        // A constant signal must resample to the same constant: each
        // phase's kernel sums to 1. The edges legitimately dip because the
        // kernel treats input outside the buffer as zeros, so the check
        // measures the steady state.
        let input = vec![0.5f32; 1000];
        let out =
            resample_mono_sinc_hann(&input, 48_000, 16_000, &SincHannOptions::default()).unwrap();
        let interior = &out[100..out.len() - 100];
        let max_err = interior
            .iter()
            .map(|&s| (s - 0.5).abs())
            .fold(0.0f32, f32::max);
        assert!(max_err < 1e-3, "max dc error {max_err}");
    }

    #[test]
    fn sinc_hann_preserves_sine_frequency() {
        // A 1 kHz sine at 48 kHz resampled to 16 kHz stays a 1 kHz sine;
        // count zero crossings, the cheap frequency proxy.
        let rate = 48_000usize;
        let freq = 1_000.0f32;
        let input: Vec<f32> = (0..rate)
            .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / rate as f32).sin())
            .collect();
        let out =
            resample_mono_sinc_hann(&input, 48_000, 16_000, &SincHannOptions::default()).unwrap();
        let crossings = out.windows(2).filter(|w| w[0] * w[1] < 0.0).count();
        // One second of a 1 kHz sine is ~2000 crossings at any rate.
        assert!(
            (1900..=2100).contains(&crossings),
            "zero crossings {crossings}"
        );
    }

    #[test]
    fn sinc_hann_accumulation_modes_agree() {
        let input: Vec<f32> = (0..333).map(|i| ((i as f32) * 0.37).sin()).collect();
        let a =
            resample_mono_sinc_hann(&input, 44_100, 16_000, &SincHannOptions::default()).unwrap();
        let b = resample_mono_sinc_hann(
            &input,
            44_100,
            16_000,
            &SincHannOptions {
                accumulation: Accumulation::Float64,
                computation: KernelComputation::Float64,
                ..SincHannOptions::default()
            },
        )
        .unwrap();
        assert_eq!(a.len(), b.len());
        let max_diff = a
            .iter()
            .zip(&b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        assert!(max_diff < 1e-5, "mode disagreement {max_diff}");
    }
}

#[cfg(test)]
mod kernel_budget_regression {
    use super::*;

    #[test]
    fn refuses_coprime_kernel_expansion_before_allocating() {
        for (source, target) in [(44_100, 44_101), (1, u32::MAX)] {
            assert!(matches!(
                resample_mono_sinc_hann(&[1.0], source, target, &SincHannOptions::default()),
                Err(AudioError::BufferTooLarge {
                    what: "resampling kernel",
                    ..
                })
            ));
        }
        let options = SincHannOptions {
            lowpass_filter_width: usize::MAX,
            ..SincHannOptions::default()
        };
        assert!(matches!(
            resample_mono_sinc_hann(&[1.0], 48000, 16000, &options),
            Err(AudioError::BufferTooLarge { .. })
        ));
    }
}
