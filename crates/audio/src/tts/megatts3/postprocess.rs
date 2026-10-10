//! Waveform post-processing of `MegaTTS3DiTInfer.forward`: BS.1770 loudness
//! matching to the prompt (pyloudnorm 0.1.1) and the Hann crossfade that joins
//! text chunks (`combine_audio_segments`).

use crate::{Result, SpeechError};

/// pyloudnorm `Meter` defaults: 400 ms blocks, 75% overlap.
const BLOCK_SECONDS: f64 = 0.400;
const OVERLAP_STEP: f64 = 0.25;
const ABSOLUTE_GATE: f64 = -70.0;
/// `combine_audio_segments` default.
pub const CROSSFADE_SECONDS: f64 = 0.16;

/// One RBJ biquad, normalized by `a0`, run in transposed direct form II like
/// `scipy.signal.lfilter`.
struct Biquad {
    b: [f64; 3],
    a: [f64; 2],
}

impl Biquad {
    fn high_shelf(gain_db: f64, q: f64, fc: f64, rate: f64) -> Self {
        let a = 10f64.powf(gain_db / 40.0);
        let w0 = 2.0 * std::f64::consts::PI * (fc / rate);
        let alpha = w0.sin() / (2.0 * q);
        let cos = w0.cos();
        let sq = 2.0 * a.sqrt() * alpha;
        let b0 = a * ((a + 1.0) + (a - 1.0) * cos + sq);
        let b1 = -2.0 * a * ((a - 1.0) + (a + 1.0) * cos);
        let b2 = a * ((a + 1.0) + (a - 1.0) * cos - sq);
        let a0 = (a + 1.0) - (a - 1.0) * cos + sq;
        let a1 = 2.0 * ((a - 1.0) - (a + 1.0) * cos);
        let a2 = (a + 1.0) - (a - 1.0) * cos - sq;
        Self::normalized([b0, b1, b2], [a0, a1, a2])
    }

    fn high_pass(q: f64, fc: f64, rate: f64) -> Self {
        let w0 = 2.0 * std::f64::consts::PI * (fc / rate);
        let alpha = w0.sin() / (2.0 * q);
        let cos = w0.cos();
        Self::normalized(
            [(1.0 + cos) / 2.0, -(1.0 + cos), (1.0 + cos) / 2.0],
            [1.0 + alpha, -2.0 * cos, 1.0 - alpha],
        )
    }

    fn normalized(b: [f64; 3], a: [f64; 3]) -> Self {
        Self {
            b: [b[0] / a[0], b[1] / a[0], b[2] / a[0]],
            a: [a[1] / a[0], a[2] / a[0]],
        }
    }

    fn apply(&self, x: &mut [f64]) {
        let (mut z1, mut z2) = (0.0f64, 0.0f64);
        for v in x.iter_mut() {
            let xn = *v;
            let y = self.b[0] * xn + z1;
            z1 = self.b[1] * xn + z2 - self.a[0] * y;
            z2 = self.b[2] * xn - self.a[1] * y;
            *v = y;
        }
    }
}

/// Integrated gated loudness (LUFS) of mono audio, ITU-R BS.1770-4 as
/// implemented by pyloudnorm 0.1.1 with K-weighting. Returns `-inf` when every
/// block is below the absolute gate (silence), like the reference. Audio
/// shorter than one 400 ms block is an error (pyloudnorm raises).
pub fn integrated_loudness(samples: &[f32], rate: u32) -> Result<f64> {
    let rate_f = f64::from(rate);
    if samples.iter().any(|v| !v.is_finite()) {
        return Err(SpeechError::Input {
            why: "audio contains NaN or infinity".to_string(),
        });
    }
    if (samples.len() as f64) < BLOCK_SECONDS * rate_f {
        return Err(SpeechError::Input {
            why: format!(
                "audio of {} samples is shorter than the 400 ms loudness block",
                samples.len()
            ),
        });
    }
    let mut x: Vec<f64> = samples.iter().map(|&v| f64::from(v)).collect();
    Biquad::high_shelf(4.0, 1.0 / 2f64.sqrt(), 1500.0, rate_f).apply(&mut x);
    Biquad::high_pass(0.5, 38.0, rate_f).apply(&mut x);

    let t = x.len() as f64 / rate_f;
    // int(np.round((T - T_g) / (T_g * step)) + 1), np.round ties to even.
    let blocks =
        (((t - BLOCK_SECONDS) / (BLOCK_SECONDS * OVERLAP_STEP)).round_ties_even() + 1.0) as usize;
    let mut z = Vec::with_capacity(blocks);
    for j in 0..blocks {
        let lo = (BLOCK_SECONDS * (j as f64 * OVERLAP_STEP) * rate_f) as usize;
        let hi = (BLOCK_SECONDS * (j as f64 * OVERLAP_STEP + 1.0) * rate_f) as usize;
        let sum: f64 = x[lo.min(x.len())..hi.min(x.len())]
            .iter()
            .map(|v| v * v)
            .sum();
        z.push(sum / (BLOCK_SECONDS * rate_f));
    }
    let block_loudness: Vec<f64> = z.iter().map(|&zj| -0.691 + 10.0 * zj.log10()).collect();
    let gated: Vec<f64> = z
        .iter()
        .zip(&block_loudness)
        .filter(|(_, &l)| l >= ABSOLUTE_GATE)
        .map(|(&zj, _)| zj)
        .collect();
    if gated.is_empty() {
        return Ok(f64::NEG_INFINITY);
    }
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let relative_gate = -0.691 + 10.0 * mean(&gated).log10() - 10.0;
    let kept: Vec<f64> = z
        .iter()
        .zip(&block_loudness)
        .filter(|(_, &l)| l > relative_gate && l > ABSOLUTE_GATE)
        .map(|(&zj, _)| zj)
        .collect();
    if kept.is_empty() {
        return Ok(f64::NEG_INFINITY);
    }
    Ok(-0.691 + 10.0 * mean(&kept).log10())
}

/// `pyln.normalize.loudness` followed by the reference's peak guard: gain the
/// signal from `input_lufs` to `target_lufs`, then if any sample reaches 1.0,
/// rescale to a 0.95 peak. A non-finite loudness (silence) leaves the audio
/// unchanged instead of producing NaN.
pub fn match_loudness(samples: &[f32], input_lufs: f64, target_lufs: f64) -> Vec<f32> {
    if !input_lufs.is_finite() || !target_lufs.is_finite() {
        return samples.to_vec();
    }
    let gain = 10f64.powf((target_lufs - input_lufs) / 20.0);
    let mut out: Vec<f64> = samples.iter().map(|&v| gain * f64::from(v)).collect();
    let peak = out.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    if peak >= 1.0 {
        for v in out.iter_mut() {
            *v = *v / peak * 0.95;
        }
    }
    out.into_iter().map(|v| v as f32).collect()
}

/// `combine_audio_segments`: joins chunks with a Hann crossfade of
/// [`CROSSFADE_SECONDS`]. The reference's symmetric `np.hanning(2W)` halves are
/// reproduced as written (they do not sum exactly to one). A segment shorter
/// than the fade is an error (the reference's broadcast would fail).
pub fn combine_segments(segments: &[Vec<f32>], rate: u32) -> Result<Vec<f32>> {
    let w = (f64::from(rate) * CROSSFADE_SECONDS) as usize;
    let n = 2 * w;
    let hann: Vec<f64> = (0..n)
        .map(|i| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / (n as f64 - 1.0)).cos())
        .collect();
    let mut iter = segments.iter();
    let Some(first) = iter.next() else {
        return Ok(Vec::new());
    };
    let mut combined = first.clone();
    for segment in iter {
        if combined.len() < w || segment.len() < w {
            return Err(SpeechError::Input {
                why: format!(
                    "segments of {} and {} samples cannot be crossfaded over {w} samples",
                    combined.len(),
                    segment.len()
                ),
            });
        }
        let tail_start = combined.len() - w;
        let mut joined = combined[..tail_start].to_vec();
        for i in 0..w {
            let v =
                f64::from(combined[tail_start + i]) * hann[w + i] + f64::from(segment[i]) * hann[i];
            joined.push(v as f32);
        }
        joined.extend_from_slice(&segment[w..]);
        combined = joined;
    }
    Ok(combined)
}
