//! Incremental log-mel frontend for live Multitalker input.
//!
//! [`MultitalkerAsr::features`](super::MultitalkerAsr::features) computes the
//! whole recording at once, as NeMo's `CacheAwareStreamingAudioBuffer` does,
//! and `nemo_mel` refuses waveforms past 2^24 samples (about 17.5 minutes).
//! [`LiveFrontend`] produces the same frames from PCM as it arrives, with
//! bounded memory.
//!
//! Frame `f` is centered on sample `f * hop`, so it needs raw samples
//! `f * hop - fft/2 - 1 ..= f * hop + fft/2 - 1` (the extra one is
//! pre-emphasis's look-back). It is complete once `f * hop + fft/2` samples
//! have arrived, which is the whole latency cost of going live. Frames are
//! computed by the same `nemo_mel` call on a short hop-aligned window of raw
//! samples, so each is bit-identical to its whole-file value. Only the
//! right edge differs while streaming: the last `fft/2 / hop` frames wait
//! for [`finish`](LiveFrontend::finish), which supplies the zero padding and
//! the "past `samples / hop`" mask exactly as the whole-file call does.

use turbospark_audio::nemo_mel::{nemo_log_mel_spectrogram_with_padding, NemoMelOptions};
use turbospark_audio::stft::StftPaddingMode;

use crate::{Result, SpeechError};

/// Raw samples kept behind the next frame's center: enough for the FFT
/// frame's left half plus pre-emphasis's look-back, rounded up to whole hops
/// so segment frames stay aligned with the recording's.
fn lookback_hops(fft: usize, hop: usize) -> usize {
    (fft / 2 + 1).div_ceil(hop)
}

/// Streams PCM in and frame-major `[frames, mels]` log-mel out.
pub struct LiveFrontend {
    options: NemoMelOptions,
    hop: usize,
    half: usize,
    mels: usize,
    /// Raw samples from absolute index `base`.
    tail: Vec<f32>,
    base: usize,
    total: usize,
    next_frame: usize,
    finished: bool,
}

impl LiveFrontend {
    /// `options` are the checkpoint's whole-file options (normalization
    /// `None`, constant padding); the valid-frame mask is applied here at
    /// [`finish`](Self::finish), not per segment.
    pub fn new(mut options: NemoMelOptions) -> Result<Self> {
        if options.pad_to != 0 || options.stft.hop == 0 || !options.stft.center {
            return Err(SpeechError::Unsupported {
                why: "live frontend needs centered frames, a positive hop and no pad_to".into(),
            });
        }
        if !matches!(
            options.normalize,
            turbospark_audio::nemo_mel::NemoMelNormalization::None
        ) {
            return Err(SpeechError::Unsupported {
                why: "live frontend cannot normalize per recording".into(),
            });
        }
        options.normalize_valid_frames = false;
        Ok(Self {
            hop: options.stft.hop,
            half: options.stft.fft_size / 2,
            mels: options.num_mels,
            options,
            tail: Vec::new(),
            base: 0,
            total: 0,
            next_frame: 0,
            finished: false,
        })
    }

    /// Frames returned so far.
    pub fn frames_emitted(&self) -> usize {
        self.next_frame
    }

    /// Samples received so far.
    pub fn samples_received(&self) -> usize {
        self.total
    }

    /// Adds PCM and returns the frames it completed, frame-major.
    pub fn push(&mut self, samples: &[f32]) -> Result<Vec<f32>> {
        if self.finished {
            return Err(SpeechError::Input {
                why: "live frontend was finished; open a new one".into(),
            });
        }
        if samples.iter().any(|s| !s.is_finite()) {
            return Err(SpeechError::Input {
                why: "Multitalker input must be finite mono PCM".into(),
            });
        }
        self.tail.extend_from_slice(samples);
        self.total += samples.len();
        // Frame f is complete once f * hop + half samples exist.
        let ready = match self.total.checked_sub(self.half) {
            Some(span) => span / self.hop + 1,
            None => 0,
        };
        let out = self.emit(self.next_frame, ready)?;
        self.trim();
        Ok(out)
    }

    /// Ends the recording and returns the remaining frames: the right-edge
    /// frames with zero padding, and the frame centered at `samples / hop`
    /// or later zeroed. Errors on a recording too short for the whole-file
    /// call.
    pub fn finish(&mut self) -> Result<Vec<f32>> {
        if self.finished {
            return Err(SpeechError::Input {
                why: "live frontend was already finished".into(),
            });
        }
        let valid = self.total / self.hop;
        if self.total == 0 || valid < 2 {
            return Err(SpeechError::Input {
                why: "Multitalker input needs at least two hops of PCM".into(),
            });
        }
        let end = valid + 1;
        let mut out = self.emit(self.next_frame, end)?;
        // NeMo masks frames at or past `samples / hop`.
        let first_masked = valid.saturating_sub(self.next_frame - out.len() / self.mels);
        out[first_masked * self.mels..].fill(0.0);
        self.finished = true;
        self.tail = Vec::new();
        Ok(out)
    }

    /// Frames `from..to` (absolute), whose windows lie inside the received
    /// samples or, for `finish`, the zero padding past them.
    fn emit(&mut self, from: usize, to: usize) -> Result<Vec<f32>> {
        if to <= from {
            return Ok(Vec::new());
        }
        let back = lookback_hops(self.options.stft.fft_size, self.hop);
        let seg_first_frame = from.saturating_sub(back);
        let seg_lo = seg_first_frame * self.hop;
        let rel = seg_lo - self.base;
        let mel = nemo_log_mel_spectrogram_with_padding(
            &self.tail[rel..],
            &self.options,
            StftPaddingMode::Constant,
        )?;
        let mut out = Vec::with_capacity((to - from) * self.mels);
        for frame in from..to {
            let row = mel.get(frame - seg_first_frame).ok_or_else(|| {
                SpeechError::Audio(format!("live frontend segment is missing frame {frame}"))
            })?;
            out.extend_from_slice(row);
        }
        self.next_frame = to;
        Ok(out)
    }

    /// Drops raw samples no later frame reads.
    fn trim(&mut self) {
        let back = lookback_hops(self.options.stft.fft_size, self.hop);
        let keep_from = self.next_frame.saturating_sub(back) * self.hop;
        if keep_from > self.base {
            self.tail.drain(..keep_from - self.base);
            self.base = keep_from;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::symmetric_hann;
    use turbospark_audio::nemo_mel::NemoMelNormalization;
    use turbospark_audio::stft::StftOptions;

    /// Multitalker's frontend (128 mels, FFT 512, 400-sample Hann, hop 160).
    fn options() -> NemoMelOptions {
        NemoMelOptions {
            stft: StftOptions {
                fft_size: 512,
                hop: 160,
                window: symmetric_hann(400),
                center: true,
            },
            sample_rate: 16_000,
            num_mels: 128,
            normalize: NemoMelNormalization::None,
            preemphasis: 0.97,
            log_zero_guard_value: 2.0f32.powi(-24),
            pad_to: 0,
            pad_value: 0.0,
            normalize_valid_frames: true,
        }
    }

    fn signal(samples: usize, seed: u32) -> Vec<f32> {
        let mut state = seed;
        (0..samples)
            .map(|i| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let noise = (state >> 8) as f32 / (1u32 << 24) as f32 - 0.5;
                0.3 * (i as f32 * 0.07).sin() + 0.1 * noise
            })
            .collect()
    }

    fn whole_file(samples: &[f32]) -> Vec<f32> {
        nemo_log_mel_spectrogram_with_padding(samples, &options(), StftPaddingMode::Constant)
            .unwrap()
            .into_iter()
            .flatten()
            .collect()
    }

    fn live(samples: &[f32], pushes: &[usize]) -> Vec<f32> {
        let mut frontend = LiveFrontend::new(options()).unwrap();
        let mut out = Vec::new();
        let mut at = 0;
        let mut sizes = pushes.iter().cycle();
        while at < samples.len() {
            let n = (*sizes.next().unwrap()).min(samples.len() - at);
            out.extend(frontend.push(&samples[at..at + n]).unwrap());
            at += n;
        }
        out.extend(frontend.finish().unwrap());
        out
    }

    /// Every frame, including the right edge and the masked final frame, is
    /// bit-identical to the whole-file call for pushes smaller than a hop,
    /// odd, hop-aligned and larger than the lookback, and for lengths on and
    /// off a hop boundary.
    #[test]
    fn live_frames_equal_whole_file_bitwise() {
        for (len, seed) in [
            (320usize, 1u32),
            (321, 2),
            (5_000, 3),
            (16_000, 4),
            (48_017, 5),
        ] {
            let samples = signal(len, seed);
            let want = whole_file(&samples);
            for pushes in [
                vec![1usize],
                vec![7, 13],
                vec![160],
                vec![161, 159],
                vec![512],
                vec![1_000, 3],
                vec![len],
            ] {
                let got = live(&samples, &pushes);
                assert_eq!(got.len(), want.len(), "len {len} pushes {pushes:?}");
                for (i, (a, b)) in got.iter().zip(&want).enumerate() {
                    assert_eq!(
                        a.to_bits(),
                        b.to_bits(),
                        "len {len} pushes {pushes:?} frame {} mel {}",
                        i / 128,
                        i % 128
                    );
                }
            }
        }
    }

    /// While streaming, only frames whose whole FFT frame has arrived are
    /// returned, so the output never holds a value `finish` would change.
    #[test]
    fn push_withholds_frames_that_need_the_right_edge() {
        let samples = signal(10_000, 9);
        let mut frontend = LiveFrontend::new(options()).unwrap();
        assert!(frontend.push(&samples[..255]).unwrap().is_empty());
        // 256 samples complete frame 0 only.
        assert_eq!(frontend.push(&samples[255..256]).unwrap().len(), 128);
        let more = frontend.push(&samples[256..]).unwrap();
        let emitted = frontend.frames_emitted();
        assert_eq!(emitted, (10_000 - 256) / 160 + 1);
        assert_eq!(more.len(), (emitted - 1) * 128);
        let rest = frontend.finish().unwrap();
        assert_eq!(emitted + rest.len() / 128, 10_000 / 160 + 1);
    }

    /// Memory stays bounded: the raw tail never holds more than the unread
    /// samples plus the lookback, however long the recording.
    #[test]
    fn tail_is_bounded_on_long_input() {
        let mut frontend = LiveFrontend::new(options()).unwrap();
        let chunk = signal(16_000, 11);
        for _ in 0..60 {
            frontend.push(&chunk).unwrap();
            assert!(
                frontend.tail.len() <= 16_000 + 3 * 160 + 512,
                "{}",
                frontend.tail.len()
            );
        }
        assert_eq!(frontend.samples_received(), 960_000);
    }

    #[test]
    fn rejects_bad_input_and_reuse() {
        let mut frontend = LiveFrontend::new(options()).unwrap();
        assert!(frontend.push(&[0.0, f32::NAN]).is_err());
        assert!(frontend.finish().is_err(), "an empty recording");
        let mut frontend = LiveFrontend::new(options()).unwrap();
        frontend.push(&signal(1_000, 1)).unwrap();
        frontend.finish().unwrap();
        assert!(frontend.push(&[0.0]).is_err());
        assert!(frontend.finish().is_err());
        let mut normalized = options();
        normalized.normalize = NemoMelNormalization::PerFeature;
        assert!(LiveFrontend::new(normalized).is_err());
    }
}
