//! Streaming isolation: the voice leaves as the audio arrives
//! (`streaming.Stream`, the plain stream).
//!
//! Samples are pushed in pieces of any size. A frame is computed as soon as
//! its analysis window is complete, up to `block_frames` frames together,
//! with the state carried: time attention over a cache of rotated past keys,
//! the time Mamba's recurrence, and the overlap-add of the synthesis. Output
//! samples are released once every frame touching them is synthesized, which
//! for frame `t` waits for frame `t + lookahead`. Only `flush` ends a stream.
//!
//! Divergence from the reference: the reference `VoiceIsolator.stream()`
//! defaults to windowed lanes (600 frames, 200 crossfade) to bound the drift
//! of an unwindowed cache. The released realtime checkpoint was trained with
//! a 1200-frame attention window, which the cache keeps exactly (a frame sees
//! itself and the 1199 before it), so this port streams without lanes; with
//! a cache of at least the window its output does not depend on how the
//! audio is pushed.

use super::layers::BlockState;
use super::{correction, VoiceIsolator};
use crate::fft::{ComplexF32, RealFftPlan, RealFftScratch};
use crate::{Result, SpeechError};

/// Default frames computed together (the reference `block_frames`).
pub const DEFAULT_BLOCK_FRAMES: usize = 4;

/// One stream's state. It borrows nothing: every call takes the model, so a
/// host can own both side by side.
pub struct VoiceIsolatorStream {
    block_frames: usize,
    lookahead: usize,
    n_fft: usize,
    hop: usize,
    states: Vec<BlockState>,
    audio: Vec<f32>,
    audio_base: usize,
    frames: usize,
    emitted: usize,
    raw: std::collections::VecDeque<Vec<f32>>,
    spectra: std::collections::VecDeque<Vec<ComplexF32>>,
    pending_base: usize,
    ola: Vec<f32>,
    envelope: Vec<f32>,
    ola_base: usize,
    released: usize,
    closed: bool,
    plan: std::sync::Arc<RealFftPlan>,
    scratch: RealFftScratch,
}

impl VoiceIsolatorStream {
    /// A stream of `model` at its streaming position. `cache_frames` bounds
    /// the time attention's past (default: the checkpoint's attention
    /// window, else its training chunk).
    pub fn new(
        model: &VoiceIsolator,
        block_frames: usize,
        cache_frames: Option<usize>,
    ) -> Result<Self> {
        if model.offline() {
            return Err(SpeechError::Input {
                why: "streaming needs a causal checkpoint at a streaming position".into(),
            });
        }
        if block_frames == 0 {
            return Err(SpeechError::Input {
                why: "process at least one frame per block".into(),
            });
        }
        let c = &model.config;
        let cache = match cache_frames {
            Some(n) if n > 0 => n,
            _ if c.attention_window_frames > 0 => c.attention_window_frames,
            _ => ((c.chunk_seconds * f64::from(c.sample_rate) / c.hop_length as f64).round()
                as usize)
                .max(1),
        };
        let capacity = cache + (cache / 2).max(block_frames);
        let bands = model.bands();
        let states = model
            .blocks
            .iter()
            .map(|b| b.stream_state(bands, c.width, cache, capacity, c.attention_window_frames))
            .collect();
        Ok(Self {
            block_frames,
            lookahead: model.lookahead_frames(),
            n_fft: c.n_fft,
            hop: c.hop_length,
            states,
            audio: Vec::new(),
            audio_base: 0,
            frames: 0,
            emitted: 0,
            raw: Default::default(),
            spectra: Default::default(),
            pending_base: 0,
            ola: vec![0.0; c.n_fft],
            envelope: vec![0.0; c.n_fft],
            ola_base: 0,
            released: 0,
            closed: false,
            plan: RealFftPlan::cached(c.n_fft)
                .map_err(|e| SpeechError::Input { why: e.to_string() })?,
            scratch: RealFftScratch::default(),
        })
    }

    /// Input samples that must follow an output sample before its release.
    pub fn latency_samples(&self) -> usize {
        self.n_fft + self.hop * self.lookahead
    }

    /// Feeds new mono samples at the model rate; returns the output samples
    /// that became final.
    pub fn push(&mut self, model: &VoiceIsolator, samples: &[f32]) -> Result<Vec<f32>> {
        if self.closed {
            return Err(SpeechError::Input {
                why: "the stream was flushed".into(),
            });
        }
        if samples.iter().any(|v| !v.is_finite()) {
            return Err(SpeechError::Input {
                why: "samples must be finite".into(),
            });
        }
        self.audio.extend_from_slice(samples);
        let total = self.audio_base + self.audio.len();
        let half = self.n_fft / 2;
        let computable = if total >= half {
            (total - half) / self.hop + 1
        } else {
            0
        };
        self.advance(model, computable)?;
        let upto = (self.emitted * self.hop).saturating_sub(half);
        let out = self.release(upto);
        self.trim();
        Ok(out)
    }

    /// Ends the stream and returns every remaining output sample.
    pub fn flush(&mut self, model: &VoiceIsolator) -> Result<Vec<f32>> {
        if self.closed {
            return Ok(Vec::new());
        }
        self.closed = true;
        let total = self.audio_base + self.audio.len();
        self.advance(model, total / self.hop + 1)?;
        while self.emitted < self.frames {
            self.synthesize(model)?;
        }
        Ok(self.release(total))
    }

    /// Spectrum of frame `f`: its window of the audio, zeros where there is
    /// none (`_analyze`).
    fn analyze(&mut self, model: &VoiceIsolator, f: usize) -> Result<Vec<ComplexF32>> {
        let start = (f * self.hop) as isize - (self.n_fft / 2) as isize;
        let total = (self.audio_base + self.audio.len()) as isize;
        let mut frame = vec![0.0f32; self.n_fft];
        for (i, slot) in frame.iter_mut().enumerate() {
            let at = start + i as isize;
            if at >= self.audio_base as isize && at < total {
                *slot = self.audio[at as usize - self.audio_base] * model.window[i];
            }
        }
        let mut spectrum = Vec::with_capacity(self.n_fft / 2 + 1);
        self.plan
            .forward_into(&frame, &mut spectrum, &mut self.scratch)
            .map_err(|e| SpeechError::Input { why: e.to_string() })?;
        Ok(spectrum)
    }

    /// Computes every frame below `limit` in blocks, then synthesizes the
    /// frames whose look-ahead frame exists (`_advance`).
    fn advance(&mut self, model: &VoiceIsolator, limit: usize) -> Result<()> {
        while self.frames < limit {
            let count = self.block_frames.min(limit - self.frames);
            let spectra: Vec<Vec<ComplexF32>> = (self.frames..self.frames + count)
                .map(|f| self.analyze(model, f))
                .collect::<Result<_>>()?;
            let mut x = model.encode(&spectra);
            let bands = model.bands();
            for (block, state) in model.blocks.iter().zip(self.states.iter_mut()) {
                block.forward_stream(&model.exec, &mut x, bands, count, state);
            }
            let raw = model.decoder.forward(&model.exec, &x, count, &model.edges);
            let stride = raw.len() / count;
            for (i, spectrum) in spectra.into_iter().enumerate() {
                self.raw
                    .push_back(raw[i * stride..(i + 1) * stride].to_vec());
                self.spectra.push_back(spectrum);
            }
            self.frames += count;
        }
        while self.emitted + self.lookahead < self.frames {
            self.synthesize(model)?;
        }
        self.trim();
        Ok(())
    }

    /// Synthesizes the next frame from the decoder output `lookahead`
    /// frames later; the last computed frame serves every frame beyond it.
    fn synthesize(&mut self, model: &VoiceIsolator) -> Result<()> {
        let index = self.emitted;
        let last = self.frames - 1;
        let source = (index + self.lookahead).min(last);
        let bins = self.n_fft / 2 + 1;
        let bound = model.config.correction_bound;
        let raw = &self.raw[source - self.pending_base];
        let spectrum = &self.spectra[index - self.pending_base];
        let mut delta: Vec<ComplexF32> = (0..bins)
            .map(|f| correction(spectrum[f], &raw[f * 4..(f + 1) * 4], bound))
            .collect();
        // irfft ignores the imaginary part of DC and Nyquist.
        delta[0].im = 0.0;
        delta[bins - 1].im = 0.0;
        let mut frame = Vec::with_capacity(self.n_fft);
        self.plan
            .inverse_into(&delta, &mut frame, &mut self.scratch)
            .map_err(|e| SpeechError::Input { why: e.to_string() })?;
        let start = index * self.hop - self.ola_base;
        let needed = start + self.n_fft;
        if needed > self.ola.len() {
            self.ola.resize(needed, 0.0);
            self.envelope.resize(needed, 0.0);
        }
        for i in 0..self.n_fft {
            let w = model.window[i];
            self.ola[start + i] += frame[i] * w;
            self.envelope[start + i] += w * w;
        }
        self.emitted += 1;
        Ok(())
    }

    /// Drops what no frame still needs (`_trim`).
    fn trim(&mut self) {
        let drop = self.emitted - self.pending_base;
        self.raw.drain(..drop.min(self.raw.len()));
        self.spectra.drain(..drop.min(self.spectra.len()));
        self.pending_base = self.emitted;
        let keep_from = self.released.saturating_sub(self.ola_base);
        if keep_from > 0 {
            self.ola.drain(..keep_from.min(self.ola.len()));
            self.envelope.drain(..keep_from.min(self.envelope.len()));
            self.ola_base = self.released;
        }
        let needed_from = self
            .released
            .min((self.frames * self.hop).saturating_sub(self.n_fft / 2));
        if needed_from > self.audio_base {
            let drop = needed_from - self.audio_base;
            self.audio.drain(..drop);
            self.audio_base = needed_from;
        }
    }

    /// Output samples `[released, upto)`: input plus synthesized change.
    fn release(&mut self, upto: usize) -> Vec<f32> {
        let total = self.audio_base + self.audio.len();
        let (start, stop) = (self.released, upto.min(total));
        if stop <= start {
            return Vec::new();
        }
        let offset = self.n_fft / 2;
        let out = (start..stop)
            .map(|s| {
                let at = s + offset - self.ola_base;
                let (o, e) = (
                    self.ola.get(at).copied().unwrap_or(0.0),
                    self.envelope.get(at).copied().unwrap_or(0.0),
                );
                self.audio[s - self.audio_base] + o / e.max(1e-11)
            })
            .collect();
        self.released = stop;
        out
    }
}
