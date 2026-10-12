//! Confucius4 DSP frontends: the SeamlessM4T-style 160-d fbank for the
//! w2v-bert encoder (`features.py`), the kaldi fbank + CAMPPlus
//! frontend (`xvector.py`), and the reference-audio 80-band slaney mel
//! (`confucius4.py::_ref_mel`), all portable f32.

use crate::fft::{real_fft_forward, ComplexF32};
use crate::{Result, SpeechError};

const MEL_FLOOR: f32 = 1.192092955078125e-07;

/// Power spectrum rows of a real FFT: `|rfft(frame)|^2`.
fn power_spectrum(frame: &[f32], nfft: usize) -> Result<Vec<f32>> {
    let plan = crate::fft::RealFftPlan::cached(nfft).map_err(SpeechError::from)?;
    let _ = &plan;
    let mut padded = frame.to_vec();
    padded.resize(nfft, 0.0);
    let spectrum: Vec<ComplexF32> = real_fft_forward(&padded).map_err(SpeechError::from)?;
    Ok(spectrum.iter().map(|c| c.re * c.re + c.im * c.im).collect())
}

/// `features.py::fbank_160`: 16 kHz audio -> `(1, frames / 2, 160)`.
/// `mel_filters` and `window` come from the checkpoint's
/// `fbank_filters.npz` (mel `[257, 80]` column-major as saved, window
/// `[400]`), precomputed at convert time.
pub fn fbank_160(audio: &[f32], mel_filters: &[f32], window: &[f32]) -> Result<Vec<f32>> {
    const FRAME: usize = 400;
    const HOP: usize = 160;
    const NFFT: usize = 512;
    const MELS: usize = 80;
    let wav: Vec<f32> = audio.iter().map(|v| v * (1u32 << 15) as f32).collect();
    if wav.len() < FRAME {
        return Err(SpeechError::Input {
            why: format!(
                "reference audio too short for the w2v fbank: {} samples",
                wav.len()
            ),
        });
    }
    let nfr = 1 + (wav.len() - FRAME) / HOP;
    let mut out = vec![0.0f32; nfr * MELS];
    for frame in 0..nfr {
        let base = frame * HOP;
        let mut buf = vec![0.0f32; NFFT];
        // remove_dc_offset then preemphasis inside the frame.
        let mut frame_data = vec![0.0f32; FRAME];
        frame_data.copy_from_slice(&wav[base..base + FRAME]);
        let mean: f32 = frame_data.iter().sum::<f32>() / FRAME as f32;
        for value in &mut frame_data {
            *value -= mean;
        }
        let mut emph = vec![0.0f32; FRAME];
        emph[0] = frame_data[0] * 0.03;
        for i in 1..FRAME {
            emph[i] = frame_data[i] - 0.97 * frame_data[i - 1];
        }
        for i in 0..FRAME {
            buf[i] = emph[i] * window[i];
        }
        let spec = power_spectrum(&buf, NFFT)?;
        // spec @ mel_filters with mel [257, 80] row-major.
        for m in 0..MELS {
            let mut acc = 0.0f32;
            for f in 0..spec.len() {
                acc += spec[f] * mel_filters[f * MELS + m];
            }
            out[frame * MELS + m] = acc.max(MEL_FLOOR).ln();
        }
    }
    // Per-mel-bin CMVN (unbiased variance, ddof=1).
    let mut mean = vec![0.0f32; MELS];
    for frame in 0..nfr {
        for m in 0..MELS {
            mean[m] += out[frame * MELS + m];
        }
    }
    for value in &mut mean {
        *value /= nfr as f32;
    }
    let mut var = vec![0.0f32; MELS];
    for frame in 0..nfr {
        for m in 0..MELS {
            let d = out[frame * MELS + m] - mean[m];
            var[m] += d * d;
        }
    }
    for m in 0..MELS {
        let std = (var[m] / (nfr - 1) as f32 + 1e-7f32).sqrt();
        for frame in 0..nfr {
            out[frame * MELS + m] = (out[frame * MELS + m] - mean[m]) / std;
        }
    }
    // Stride-2 frame stacking: drop the odd tail frame.
    let n = nfr - (nfr % 2);
    let mut stacked = Vec::with_capacity((n / 2) * 160);
    for pair in 0..n / 2 {
        stacked.extend_from_slice(&out[pair * 2 * MELS..(pair * 2 + 2) * MELS]);
    }
    Ok(stacked)
}

/// The CAMPPlus kaldi fbank (`xvector.py::kaldi_fbank`): povey window,
/// snip-edges framing, preemphasis, HTK mel (the shared
/// `crate::mel` filterbank, `f_min` 20, `f_max` sr / 2, no
/// normalization), log with the f32 epsilon floor. Returns
/// `[frames, n_mels]`.
pub fn kaldi_fbank(audio: &[f32], sample_rate: usize, num_mel_bins: usize) -> Result<Vec<f32>> {
    let win_length = sample_rate * 25 / 1000;
    let hop_length = sample_rate * 10 / 1000;
    let n_fft = win_length.next_power_of_two();
    let signal_len = audio.len();
    let num_frames = (signal_len - win_length) / hop_length + 1;
    // Povey window: hann^0.85.
    let window: Vec<f32> = (0..win_length)
        .map(|n| {
            let hann =
                0.5 - 0.5 * (2.0 * std::f32::consts::PI * n as f32 / (win_length - 1) as f32).cos();
            hann.powf(0.85)
        })
        .collect();
    let filters = crate::mel::mel_filterbank(
        num_mel_bins,
        n_fft,
        sample_rate as u32,
        20.0,
        Some(sample_rate as f32 / 2.0),
        crate::mel::MelScale::Htk,
    )
    .map_err(SpeechError::from)?;
    let n_freqs = n_fft / 2 + 1;
    let mut out = vec![0.0f32; num_frames.max(1) * num_mel_bins];
    for frame in 0..num_frames.max(1) {
        let start = frame * hop_length;
        let mut frame_data = vec![0.0f32; win_length];
        if start + win_length <= signal_len {
            frame_data.copy_from_slice(&audio[start..start + win_length]);
        }
        let mean: f32 = frame_data.iter().sum::<f32>() / win_length as f32;
        for value in &mut frame_data {
            *value -= mean;
        }
        let mut emph = vec![0.0f32; win_length];
        emph[0] = frame_data[0];
        for i in 1..win_length {
            emph[i] = frame_data[i] - 0.97 * frame_data[i - 1];
        }
        let mut buf = vec![0.0f32; n_fft];
        for i in 0..win_length {
            buf[i] = emph[i] * window[i];
        }
        let Ok(spec) = power_spectrum(&buf, n_fft) else {
            continue;
        };
        for m in 0..num_mel_bins {
            let mut acc = 0.0f32;
            for f in 0..n_freqs {
                acc += spec[f] * filters.weights[m * n_freqs + f];
            }
            out[frame * num_mel_bins + m] = acc.max(MEL_FLOOR).ln();
        }
    }
    Ok(out)
}

/// The slaney mel filterbank (`confucius4.py::_slaney_mel`, librosa
/// `htk=false, norm="slaney"`): the shared `crate::mel` bank covers it
/// exactly; returns `[n_mels, n_freqs]`.
pub fn slaney_mel(sample_rate: usize, n_fft: usize, n_mels: usize) -> Result<Vec<f32>> {
    let bank = crate::mel::mel_filterbank(
        n_mels,
        n_fft,
        sample_rate as u32,
        0.0,
        None,
        crate::mel::MelScale::Slaney,
    )
    .map_err(SpeechError::from)?;
    Ok(bank.weights)
}

/// `confucius4.py::_ref_mel`: 16 kHz reference audio resampled to
/// 22050 (the mlx-audio polyphase kaiser-best resampler), framed with
/// the periodic hann of the reference (np.hanning(1025) minus its last
/// sample), reflect-padded, and projected through the slaney mel. The
/// resampler is the shared `stt::vibevoice_asr::resample_poly` port of
/// `mlx_audio.resample`.
pub fn ref_mel(audio_16k: &[f32]) -> Result<Vec<f32>> {
    const SR: usize = 22050;
    const NFFT: usize = 1024;
    const HOP: usize = 256;
    const WIN: usize = 1024;
    const MELS: usize = 80;
    let resampled = crate::stt::vibevoice_asr::resample_poly::resample_mono_polyphase(
        audio_16k, 16000, SR as u32,
    )
    .map_err(SpeechError::from)?;
    // np.hanning(WIN + 1)[:-1]: symmetric hann of length 1025 without
    // the final zero sample, i.e. periodic-ish over WIN samples.
    let hann: Vec<f32> = (0..WIN)
        .map(|n| 0.5f32 * (1.0 - (2.0 * std::f32::consts::PI * n as f32 / WIN as f32).cos()))
        .collect();
    let filters = slaney_mel(SR, NFFT, MELS)?;
    let pad = (NFFT - HOP) / 2;
    // Reflect pad along time: numpy 'reflect' copies a[p-1..] mirrored
    // on the left and a[n-2], a[n-3], ... left-to-right on the right.
    let mut y = vec![0.0f32; resampled.len() + 2 * pad];
    for i in 0..pad {
        y[i] = resampled[pad - i];
        let tail = pad + resampled.len() + i;
        let src = resampled.len() - 2 - i;
        y[tail] = resampled[src];
    }
    y[pad..pad + resampled.len()].copy_from_slice(&resampled);
    let nfr = 1 + (y.len() - NFFT) / HOP;
    let n_freqs = NFFT / 2 + 1;
    let mut out = vec![0.0f32; nfr * MELS];
    for frame in 0..nfr {
        let base = frame * HOP;
        let mut buf = vec![0.0f32; NFFT];
        for i in 0..NFFT {
            buf[i] = y[base + i] * hann[i];
        }
        let spec = power_spectrum(&buf, NFFT)?;
        // sqrt(|spec|^2 + 1e-9): the reference takes sqrt of the power
        // plus a tiny epsilon.
        let mag: Vec<f32> = spec.iter().map(|p| (p + 1e-9f32).sqrt()).collect();
        for m in 0..MELS {
            let mut acc = 0.0f32;
            for f in 0..n_freqs {
                acc += filters[m * n_freqs + f] * mag[f];
            }
            out[frame * MELS + m] = acc.max(1e-5f32).ln();
        }
    }
    // Returned as [1, frames, mels] -> caller transposes; keep
    // [frames, mels] row-major.
    Ok(out)
}
