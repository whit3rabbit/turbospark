//! Streaming windowed-sinc sample-rate conversion for one channel.
//!
//! Written here rather than pulled in (`rubato`) because the need is one
//! shape: anything to 16 kHz for speech, and the occasional 44.1/48 kHz
//! export. A Blackman-windowed sinc with 32 taps a side, cut off at the lower
//! Nyquist, keeps aliasing below the floor a speech model can hear, and it
//! is about 64 multiply-adds per output sample.

use std::f64::consts::PI;

/// Taps on each side of the interpolation point.
const HALF_TAPS: usize = 32;

/// Converts one channel from `input_rate` to `output_rate`, chunk by chunk.
///
/// Feed input with [`Resampler::push`] in any chunk sizes, then call
/// [`Resampler::finish`] once. The concatenated output has exactly
/// `floor(total_input * output_rate / input_rate)` samples, so a 1.0 s clip
/// at any rate is 1.0 s at the new rate.
#[derive(Debug, Clone)]
pub struct Resampler {
    /// Input samples advanced per output sample (`input_rate / output_rate`).
    step: f64,
    /// Low-pass cutoff relative to the input Nyquist: 1 when upsampling,
    /// `output_rate / input_rate` when downsampling.
    cutoff: f64,
    /// Input samples not yet discarded. `buffer[0]` is absolute input index
    /// `buffer_start`.
    buffer: Vec<f32>,
    buffer_start: u64,
    /// Absolute index of the next output sample.
    next_output: u64,
    total_input: u64,
    passthrough: bool,
}

impl Resampler {
    /// Panics on a zero rate; callers validate options first
    /// (`ConvertOptions::validate`).
    pub fn new(input_rate: u32, output_rate: u32) -> Self {
        assert!(
            input_rate > 0 && output_rate > 0,
            "sample rates must be non-zero"
        );
        let step = f64::from(input_rate) / f64::from(output_rate);
        Self {
            step,
            cutoff: (1.0 / step).min(1.0),
            buffer: Vec::new(),
            buffer_start: 0,
            next_output: 0,
            total_input: 0,
            passthrough: input_rate == output_rate,
        }
    }

    /// Consumes `input` and appends every output sample it makes computable.
    pub fn push(&mut self, input: &[f32], output: &mut Vec<f32>) {
        self.total_input += input.len() as u64;
        if self.passthrough {
            output.extend_from_slice(input);
            return;
        }
        self.buffer.extend_from_slice(input);
        self.drain(output, false);
    }

    /// Flushes the tail. The resampler is spent afterwards.
    pub fn finish(&mut self, output: &mut Vec<f32>) {
        if self.passthrough {
            return;
        }
        self.drain(output, true);
    }

    fn drain(&mut self, output: &mut Vec<f32>, at_end: bool) {
        let expected_total = (self.total_input as f64 / self.step).floor() as u64;
        let buffered_end = self.buffer_start + self.buffer.len() as u64;
        while self.next_output < expected_total {
            let position = self.next_output as f64 * self.step;
            let center = position.floor() as u64;
            // Need HALF_TAPS samples to the right unless the input is over,
            // in which case the missing right side is silence.
            if !at_end && center + HALF_TAPS as u64 >= buffered_end {
                break;
            }
            output.push(self.interpolate(position));
            self.next_output += 1;
        }
        // Keep HALF_TAPS of history behind the next interpolation point.
        let next_center = (self.next_output as f64 * self.step).floor() as u64;
        let keep_from = next_center.saturating_sub(HALF_TAPS as u64);
        if keep_from > self.buffer_start {
            let drop = ((keep_from - self.buffer_start) as usize).min(self.buffer.len());
            self.buffer.drain(..drop);
            self.buffer_start += drop as u64;
        }
    }

    fn interpolate(&self, position: f64) -> f32 {
        let center = position.floor() as i64;
        let mut sum = 0.0f64;
        let mut weight_sum = 0.0f64;
        let first = center - HALF_TAPS as i64 + 1;
        let last = center + HALF_TAPS as i64;
        for index in first..=last {
            let distance = position - index as f64;
            let weight = self.kernel(distance);
            weight_sum += weight;
            if index < self.buffer_start as i64 {
                continue;
            }
            let offset = (index - self.buffer_start as i64) as usize;
            if let Some(sample) = self.buffer.get(offset) {
                sum += f64::from(*sample) * weight;
            }
        }
        // Normalizing by the kernel's own sum keeps DC gain at exactly 1, so
        // a constant signal stays constant across the conversion.
        if weight_sum.abs() > f64::EPSILON {
            (sum / weight_sum) as f32
        } else {
            0.0
        }
    }

    /// Blackman-windowed sinc at `distance` input samples from the point.
    fn kernel(&self, distance: f64) -> f64 {
        let span = HALF_TAPS as f64;
        if distance.abs() >= span {
            return 0.0;
        }
        let x = distance * self.cutoff;
        let sinc = if x.abs() < 1e-9 {
            1.0
        } else {
            (PI * x).sin() / (PI * x)
        };
        let n = (distance + span) / (2.0 * span);
        let window = 0.42 - 0.5 * (2.0 * PI * n).cos() + 0.08 * (4.0 * PI * n).cos();
        sinc * window
    }
}

/// Resamples a whole buffer. Convenience for tests and short clips.
pub fn resample(input: &[f32], input_rate: u32, output_rate: u32) -> Vec<f32> {
    let mut resampler = Resampler::new(input_rate, output_rate);
    let mut output = Vec::with_capacity(
        (input.len() as f64 * f64::from(output_rate) / f64::from(input_rate)) as usize + 1,
    );
    resampler.push(input, &mut output);
    resampler.finish(&mut output);
    output
}

/// Averages channels into one. A single channel is returned as-is.
pub fn downmix(channels: &[Vec<f32>]) -> Vec<f32> {
    match channels.len() {
        0 => Vec::new(),
        1 => channels[0].clone(),
        count => {
            let frames = channels[0].len();
            let scale = 1.0 / count as f32;
            (0..frames)
                .map(|i| channels.iter().map(|c| c[i]).sum::<f32>() * scale)
                .collect()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine(rate: u32, freq: f64, seconds: f64) -> Vec<f32> {
        let n = (f64::from(rate) * seconds) as usize;
        (0..n)
            .map(|i| (2.0 * PI * freq * i as f64 / f64::from(rate)).sin() as f32)
            .collect()
    }

    #[test]
    fn output_length_tracks_the_rate_ratio_exactly() {
        for (from, to) in [
            (48_000, 16_000),
            (44_100, 16_000),
            (16_000, 48_000),
            (8_000, 16_000),
        ] {
            let input = sine(from, 440.0, 1.0);
            let output = resample(&input, from, to);
            assert_eq!(output.len(), to as usize, "{from} -> {to}");
        }
    }

    #[test]
    fn chunked_push_matches_one_shot() {
        let input = sine(48_000, 300.0, 0.5);
        let whole = resample(&input, 48_000, 16_000);
        let mut resampler = Resampler::new(48_000, 16_000);
        let mut chunked = Vec::new();
        for piece in input.chunks(1_023) {
            resampler.push(piece, &mut chunked);
        }
        resampler.finish(&mut chunked);
        assert_eq!(whole.len(), chunked.len());
        for (a, b) in whole.iter().zip(&chunked) {
            assert!((a - b).abs() < 1e-6, "{a} vs {b}");
        }
    }

    #[test]
    fn constant_signal_keeps_its_level() {
        let input = vec![0.5f32; 48_000];
        let output = resample(&input, 48_000, 16_000);
        // Away from the edges, where the window sees only real samples.
        for sample in &output[100..output.len() - 100] {
            assert!((sample - 0.5).abs() < 1e-4, "{sample}");
        }
    }

    #[test]
    fn a_tone_below_the_new_nyquist_survives_downsampling() {
        let input = sine(48_000, 1_000.0, 1.0);
        let output = resample(&input, 48_000, 16_000);
        let reference = sine(16_000, 1_000.0, 1.0);
        let interior = 200..output.len() - 200;
        let error: f32 = output[interior.clone()]
            .iter()
            .zip(&reference[interior])
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        assert!(error < 0.01, "max error {error}");
    }

    #[test]
    fn a_tone_above_the_new_nyquist_is_attenuated() {
        // 12 kHz cannot exist at 16 kHz; it must not alias back in loudly.
        let input = sine(48_000, 12_000.0, 1.0);
        let output = resample(&input, 48_000, 16_000);
        let interior = &output[200..output.len() - 200];
        let rms = (interior.iter().map(|s| s * s).sum::<f32>() / interior.len() as f32).sqrt();
        assert!(rms < 0.05, "aliased rms {rms}");
    }

    #[test]
    fn same_rate_is_bit_exact_passthrough() {
        let input = sine(16_000, 440.0, 0.1);
        assert_eq!(resample(&input, 16_000, 16_000), input);
    }

    #[test]
    fn downmix_averages_channels() {
        let mixed = downmix(&[vec![1.0, 0.0], vec![0.0, 1.0]]);
        assert_eq!(mixed, vec![0.5, 0.5]);
    }
}
