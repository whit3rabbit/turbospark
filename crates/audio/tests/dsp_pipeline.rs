//! End-to-end pipeline checks over the public surface: waveform in, mono
//! resample, mel out -- the shape every model frontend will call. The mel
//! reference is recomputed here through a direct DFT instead of the crate's
//! FFT, so a packing or split bug cannot cancel itself across modules.

use turbospark_audio::conversion::{to_mono_resampled, MonoResampleStrategy};
use turbospark_audio::dsp::hann_window;
use turbospark_audio::mel::{log_mel_spectrogram, mel_filterbank, MelScale, MelSpectrogramOptions};
use turbospark_audio::resample::{resample_mono_linear, resample_mono_sinc_hann, SincHannOptions};
use turbospark_audio::stft::StftOptions;
use turbospark_audio::wav::{read_wav_f32_bytes, write_wav_f32};
use turbospark_audio::Waveform;

fn stereo_tone(sample_rate: u32, seconds: f64, freq: f64) -> Waveform {
    let frames = (sample_rate as f64 * seconds) as usize;
    let mut samples = Vec::with_capacity(frames * 2);
    for i in 0..frames {
        let s = (2.0 * std::f64::consts::PI * freq * i as f64 / sample_rate as f64).sin() as f32;
        samples.push(s);
        samples.push(s * 0.5);
    }
    Waveform::new(sample_rate, 2, samples).unwrap()
}

#[test]
fn wav_to_mono_resampled_to_mel_end_to_end() {
    // 44.1 kHz stereo tone down to the 16 kHz mono the speech frontends
    // want, then a log-mel spectrogram.
    let wave = stereo_tone(44_100, 1.0, 440.0);
    let mono = to_mono_resampled(
        &wave,
        16_000,
        &MonoResampleStrategy::SincHann(SincHannOptions::default()),
    )
    .unwrap();
    // ceil(44100 * 16000 / 44100) = 16000 output samples.
    assert_eq!(mono.len(), 16_000);

    let options = MelSpectrogramOptions::default();
    let log_mel = log_mel_spectrogram(&mono, &options, 1e-10).unwrap();
    // Centered: 1 + 16000/160 frames.
    assert_eq!(log_mel.len(), 101);
    for frame in &log_mel {
        assert_eq!(frame.len(), 80);
        assert!(frame.iter().all(|x| x.is_finite()));
    }
    // A 440 Hz tone lands in the low-frequency mel bands; band 79 covers
    // ~8 kHz and must stay near the silence floor.
    let hot = log_mel[50]
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0;
    assert!(hot < 20, "440 Hz energy in band {hot}");
    assert!(log_mel[50][79] < log_mel[50][hot] - 1.0);
}

#[test]
fn sinc_hann_resample_beats_linear_on_a_tone() {
    // 44100 -> 16000 with a 997 Hz tone: no peak lands on the output grid,
    // so linear interpolation between input samples 2.76 apart loses a
    // measurable few percent of amplitude while the polyphase kernel
    // carries the tone at near unity gain.
    let rate_in = 44_100u32;
    let rate_out = 16_000u32;
    let input: Vec<f32> = (0..rate_in)
        .map(|i| (2.0 * std::f64::consts::PI * 997.0 * i as f64 / rate_in as f64).sin() as f32)
        .collect();
    let fine =
        resample_mono_sinc_hann(&input, rate_in, rate_out, &SincHannOptions::default()).unwrap();
    let coarse = resample_mono_linear(&input, rate_in, rate_out).unwrap();
    // RMS deviation from the ideal phase-continuous output sine, away from
    // the zero-padded startup transient. The polyphase kernel tracks the
    // tone; linear interpolation across ~2.8-sample gaps loses more.
    let rms_err = |s: &[f32]| {
        let interior = &s[2000..s.len() - 2000];
        let se: f64 = interior
            .iter()
            .enumerate()
            .map(|(j, x)| {
                let ideal =
                    (2.0 * std::f64::consts::PI * 997.0 * (j + 2000) as f64 / 16_000.0).sin();
                let d = f64::from(*x) - ideal;
                d * d
            })
            .sum();
        (se / interior.len() as f64).sqrt()
    };
    let err_fine = rms_err(&fine);
    let err_coarse = rms_err(&coarse);
    assert!(
        err_fine * 3.0 < err_coarse,
        "sinc-hann rms error {err_fine} should beat linear {err_coarse}"
    );
    assert!(err_fine < 1e-3, "sinc-hann rms error {err_fine}");
}

#[test]
fn mel_spectrogram_matches_direct_dft_reference() {
    // Small case, fully recomputed by hand: 64 samples at 8 kHz, fft 32,
    // hop 16, 4 HTK bands over [0, 4000). The reference path below shares
    // nothing with the crate's FFT.
    let sample_rate = 8_000u32;
    let samples: Vec<f32> = (0..64)
        .map(|i| (2.0 * std::f64::consts::PI * 1_000.0 * i as f64 / 8_000.0).sin() as f32)
        .collect();
    let fft = 32usize;
    let hop = 16usize;
    let window = hann_window(fft);
    let options = MelSpectrogramOptions {
        stft: StftOptions {
            fft_size: fft,
            hop,
            window: window.clone(),
            center: true,
        },
        num_mels: 4,
        sample_rate,
        fmin: 0.0,
        fmax: Some(4_000.0),
        scale: MelScale::Htk,
        power: 2.0,
    };
    let got = turbospark_audio::mel::mel_spectrogram(&samples, &options).unwrap();
    assert_eq!(got.len(), 1 + 64 / hop);

    let filterbank =
        mel_filterbank(4, fft, sample_rate, 0.0, Some(4_000.0), MelScale::Htk).unwrap();
    // The same reflect padding the stft module applies: pad = fft/2, edge
    // sample not repeated.
    let pad = fft / 2;
    let mut padded = Vec::with_capacity(samples.len() + 2 * pad);
    for i in 0..pad {
        padded.push(samples[pad - i]);
    }
    padded.extend_from_slice(&samples);
    for j in 0..pad {
        padded.push(samples[samples.len() - 2 - j]);
    }
    // Frames at padded starts 0, 16, 32, 48, 64 (length 96, fft 32).
    let mut want = Vec::new();
    for frame in 0..=64 / hop {
        let start = frame * hop;
        let mut power = vec![0.0f64; fft / 2 + 1];
        for (k, slot) in power.iter_mut().enumerate() {
            let (mut re, mut im) = (0.0f64, 0.0f64);
            for t in 0..fft {
                let x = f64::from(padded[start + t]) * f64::from(window[t]);
                let a = -2.0 * std::f64::consts::PI * k as f64 * t as f64 / fft as f64;
                re += x * a.cos();
                im += x * a.sin();
            }
            *slot = re * re + im * im;
        }
        let spectrum: Vec<f32> = power.iter().map(|p| *p as f32).collect();
        want.push(filterbank.project(&spectrum).unwrap());
    }
    assert_eq!(got.len(), want.len());
    for (frame, (g, w)) in got.iter().zip(&want).enumerate() {
        for (b, (gv, wv)) in g.iter().zip(w).enumerate() {
            let diff = (gv - wv).abs() / wv.abs().max(1e-9);
            assert!(diff < 1e-3, "frame {frame} band {b}: {gv} vs {wv}");
        }
    }
}

#[test]
fn wav_bytes_feed_the_pipeline() {
    // Writer output read back through the reader must produce the same
    // spectrogram as the original waveform.
    let wave = stereo_tone(16_000, 0.5, 440.0);
    let bytes = write_wav_f32(&wave);
    let back = read_wav_f32_bytes(&bytes).unwrap();
    assert_eq!(back, wave);
    let options = MelSpectrogramOptions::default();
    let a = log_mel_spectrogram(&back.samples, &options, 1e-10).unwrap();
    let b = log_mel_spectrogram(&wave.samples, &options, 1e-10).unwrap();
    assert_eq!(a, b);
}
