//! The chatterbox_turbo CAMPPlus fbank front end.
//!
//! Reference: `chatterbox_turbo/models/s3gen/xvector.py`
//! (`extract_fbank_features`, `_get_mel_banks_kaldi`, `_povey_window`)
//! at the pinned commit recorded in the parent module.
//!
//! This is the piece that genuinely differs from the sibling
//! chatterbox family's front end: the sibling computes an f32 STFT over
//! a librosa-style HTK filterbank, while turbo reimplements the Kaldi
//! pipeline in float64 - a Kaldi `GetMelBanks` triangle construction
//! (strict bin-edge inequalities), snip-edges framing, per-frame DC
//! removal before the 0.97 pre-emphasis, and a 512-point rfft. The
//! two front ends disagree by up to 0.015 on the log-mel values, so
//! the port reimplements the turbo pipeline (including the f64
//! arithmetic) instead of reusing the sibling helper.
//!
//! The CAMPPlus network that consumes these features is the shared
//! sibling port (`crate::tts::chatterbox::campplus`), which is
//! bit-for-bit the same module tree in both references.

/// Kaldi fbank geometry (the `extract_fbank_features` defaults at
/// 16 kHz).
const SAMPLE_RATE: usize = 16_000;
const FRAME_LENGTH: usize = 400;
const FRAME_SHIFT: usize = 160;
const NUM_MELS: usize = 80;
const LOW_FREQ: f64 = 20.0;
const HIGH_FREQ: f64 = 0.0;
const PREEMPH: f64 = 0.97;
const EPSILON: f64 = 1.192_092_9e-7;

/// `_povey_window`: `hann(length - 1 denominator) ** 0.85` in f64.
fn povey_window(length: usize) -> Vec<f64> {
    (0..length)
        .map(|n| {
            let hann =
                0.5 - 0.5 * (2.0 * std::f64::consts::PI * n as f64 / (length - 1) as f64).cos();
            hann.powf(0.85)
        })
        .collect()
}

/// `_get_mel_banks_kaldi`: the Kaldi triangle construction over the
/// 1127*log(1 + f/700) mel axis with strict bin-edge inequalities.
/// Returns `[num_bins, num_fft_bins]` (padded with a zero column to
/// match the rfft output width, as the reference does).
fn kaldi_mel_banks(num_bins: usize, padded_window_size: usize, sample_freq: f64) -> Vec<Vec<f64>> {
    let num_fft_bins = padded_window_size / 2;
    let nyquist = 0.5 * sample_freq;
    let mut high_freq = HIGH_FREQ;
    if high_freq <= 0.0 {
        high_freq += nyquist;
    }
    let fft_bin_width = sample_freq / padded_window_size as f64;
    let mel_low = 1127.0 * (1.0 + LOW_FREQ / 700.0).ln();
    let mel_high = 1127.0 * (1.0 + high_freq / 700.0).ln();
    let mel_delta = (mel_high - mel_low) / (num_bins + 1) as f64;
    let mut bins = vec![vec![0.0f64; num_fft_bins]; num_bins];
    for (i, row) in bins.iter_mut().enumerate() {
        let left_mel = mel_low + i as f64 * mel_delta;
        let center_mel = mel_low + (i + 1) as f64 * mel_delta;
        let right_mel = mel_low + (i + 2) as f64 * mel_delta;
        for (j, slot) in row.iter_mut().enumerate() {
            let freq = fft_bin_width * j as f64;
            let mel = 1127.0 * (1.0 + freq / 700.0).ln();
            if mel > left_mel && mel <= center_mel {
                *slot = (mel - left_mel) / (center_mel - left_mel);
            } else if mel > center_mel && mel < right_mel {
                *slot = (right_mel - mel) / (right_mel - center_mel);
            }
        }
    }
    // The reference pads a zero column on the right so the bank width
    // matches padded_length / 2 + 1.
    for row in &mut bins {
        row.push(0.0);
    }
    bins
}

/// In-place iterative radix-2 complex FFT over f64 pairs
/// (`(re, im)` interleaved), length a power of two.
fn fft_inplace(data: &mut [(f64, f64)]) {
    let n = data.len();
    debug_assert!(n.is_power_of_two());
    // Bit reversal.
    let mut j = 0usize;
    for i in 0..n - 1 {
        if i < j {
            data.swap(i, j);
        }
        let mut m = n >> 1;
        while m >= 1 && j & m != 0 {
            j ^= m;
            m >>= 1;
        }
        j |= m;
    }
    // Butterflies.
    let mut len = 2usize;
    while len <= n {
        let ang = -2.0 * std::f64::consts::PI / len as f64;
        let (sin, cos) = ang.sin_cos();
        let mut base = 0usize;
        while base < n {
            let (mut wr, mut wi) = (1.0f64, 0.0f64);
            for k in 0..len / 2 {
                let a = base + k;
                let b = a + len / 2;
                let (ar, ai) = data[a];
                let (br, bi) = data[b];
                let tr = br * wr - bi * wi;
                let ti = br * wi + bi * wr;
                data[a] = (ar + tr, ai + ti);
                data[b] = (ar - tr, ai - ti);
                let (nr, ni) = (wr * cos - wi * sin, wr * sin + wi * cos);
                wr = nr;
                wi = ni;
            }
            base += len;
        }
        len <<= 1;
    }
}

/// `extract_fbank_features` (snip-edges, defaults): returns rows
/// `[frames, num_mels]` of log power mel energies floored at the f32
/// epsilon, exactly as the reference computes them in float64.
pub fn kaldi_fbank(samples: &[f32]) -> Vec<Vec<f32>> {
    let padded_length = FRAME_LENGTH.next_power_of_two();
    let window = povey_window(FRAME_LENGTH);
    let banks = kaldi_mel_banks(NUM_MELS, padded_length, SAMPLE_RATE as f64);
    let num_frames = if samples.len() < FRAME_LENGTH {
        0
    } else {
        1 + (samples.len() - FRAME_LENGTH) / FRAME_SHIFT
    };
    let mut out = Vec::with_capacity(num_frames.max(1));
    for f in 0..num_frames {
        // Frame, DC removal, per-frame pre-emphasis, Povey window.
        let mut frame = vec![0.0f64; FRAME_LENGTH];
        for (i, slot) in frame.iter_mut().enumerate() {
            *slot = samples[f * FRAME_SHIFT + i] as f64;
        }
        let mean = frame.iter().sum::<f64>() / FRAME_LENGTH as f64;
        for v in &mut frame {
            *v -= mean;
        }
        for i in (1..FRAME_LENGTH).rev() {
            frame[i] -= PREEMPH * frame[i - 1];
        }
        for (v, w) in frame.iter_mut().zip(&window) {
            *v *= w;
        }
        // Zero pad to the FFT length, rfft, power spectrum.
        let mut spectrum = vec![(0.0f64, 0.0f64); padded_length];
        for (i, v) in frame.iter().enumerate() {
            spectrum[i] = (*v, 0.0);
        }
        fft_inplace(&mut spectrum);
        let bins = padded_length / 2 + 1;
        let mut power = vec![0.0f64; bins];
        for (i, c) in spectrum.iter().take(bins).enumerate() {
            power[i] = c.0 * c.0 + c.1 * c.1;
        }
        // Mel filterbank, log with the f32 epsilon floor.
        let mut row = Vec::with_capacity(NUM_MELS);
        for bank in &banks {
            let energy: f64 = bank.iter().zip(&power).map(|(w, p)| w * p).sum();
            row.push(energy.max(EPSILON).ln() as f32);
        }
        out.push(row);
    }
    if out.is_empty() {
        out.push(vec![0.0f32; NUM_MELS]);
    }
    out
}

/// The `CAMPPlus.inference` conditioning: the fbank followed by the
/// per-frequency-bin mean removal over frames (the reference's
/// `features - features.mean(axis=1, keepdims=True)`).
pub fn mean_normalized_fbank(samples: &[f32]) -> Vec<Vec<f32>> {
    let mut fbank = kaldi_fbank(samples);
    let frames = fbank.len();
    if frames > 0 {
        let bins = fbank[0].len();
        for bin in 0..bins {
            let mean = fbank.iter().map(|row| row[bin]).sum::<f32>() / frames as f32;
            for row in &mut fbank {
                row[bin] -= mean;
            }
        }
    }
    fbank
}
