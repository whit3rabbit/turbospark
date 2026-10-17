//! The Frozen E coupled scheduler: NeMo Speech 218d0b4
//! `SpeakerTaggedASR.perform_parallel_streaming_stt_spk` (parallel speaker
//! strategy, speaker-kernel conditioning, cache gating), driven the way the
//! multitalker driver's `CacheAwareStreamingAudioBuffer` chunks a file with
//! `pad_and_drop_preencoded = true`.
//!
//! Per step, with `R = chunk_rows` (14 for `[70, 13]`):
//! 1. The diarizer takes the ASR chunk minus its pre-encode cache and its
//!    80 ms predictions are appended to the cumulative `P`; speakers at or
//!    past `max_speakers` are zeroed.
//! 2. `chunk_preds = P[-R:]`; the gate window is `P[-R * gate_chunks:]`.
//! 3. Active speakers: per-speaker max over the window, top `max_speakers`,
//!    kept when `> 0.5`, in ascending id order.
//! 4. Per active `s`: target `chunk_preds[:, s]`, background
//!    `any(chunk_preds[:, other active] > 0.5)`, both binarized at 0.5.
//! 5. A new id allocates fresh streams for every lower missing id.
//! 6. One shared pre-encode, then each active speaker's own encoder caches
//!    and decoder state; inactive speakers keep their state untouched.
//! 7. Sentences update from every decoded speaker's full hypothesis.

use crate::models::vad::nemotron_diarization::CoupledDiarizer;
use crate::{Result, SpeechError};

use super::seglst::{HypothesisView, SeglstEntry, SentenceBuilder};
use super::{Features, MultitalkerAsr, SpeakerStream, Token};

/// Scheduler settings; [`CoupledConfig::frozen_e`] is the qualified one.
#[derive(Debug, Clone, PartialEq)]
pub struct CoupledConfig {
    /// `max_num_of_spks`.
    pub max_speakers: usize,
    /// `cache_gating`.
    pub cache_gating: bool,
    /// `cache_gating_buffer_size` (gate window in chunks).
    pub gate_chunks: usize,
    /// Activity threshold for gating and background targets.
    pub threshold: f32,
    /// `binary_diar_preds`.
    pub binary_targets: bool,
    /// `sent_break_sec`.
    pub sentence_break: f64,
    /// Encoder frame duration in seconds.
    pub frame_seconds: f64,
}

impl CoupledConfig {
    /// The Frozen E driver settings.
    pub fn frozen_e() -> Self {
        Self {
            max_speakers: 4,
            cache_gating: true,
            gate_chunks: 2,
            threshold: 0.5,
            binary_targets: true,
            sentence_break: 1.0,
            frame_seconds: 0.08,
        }
    }
}

/// A source of per-chunk 80 ms speaker probabilities.
pub trait DiarizationSource {
    /// Output speakers per row.
    fn num_speakers(&self) -> usize;
    /// `mel` is channel-major `[n_mels, frames]` with `valid` real frames;
    /// returns `[ceil(frames / 8), num_speakers]`.
    fn step(&mut self, mel: &[f32], frames: usize, valid: usize) -> Result<Vec<f32>>;
}

impl DiarizationSource for CoupledDiarizer<'_> {
    fn num_speakers(&self) -> usize {
        CoupledDiarizer::num_speakers(self)
    }

    fn step(&mut self, mel: &[f32], frames: usize, valid: usize) -> Result<Vec<f32>> {
        Ok(CoupledDiarizer::step(self, mel, frames, valid)?.to_vec())
    }
}

/// Recorded per-step predictions (for example NeMo's, from a trace), so ASR
/// divergence can be isolated from diarizer divergence.
pub struct ReplayedDiarization {
    steps: Vec<Vec<f32>>,
    num_speakers: usize,
    next: usize,
}

impl ReplayedDiarization {
    pub fn new(steps: Vec<Vec<f32>>, num_speakers: usize) -> Self {
        Self {
            steps,
            num_speakers,
            next: 0,
        }
    }
}

impl DiarizationSource for ReplayedDiarization {
    fn num_speakers(&self) -> usize {
        self.num_speakers
    }

    fn step(&mut self, _mel: &[f32], frames: usize, _valid: usize) -> Result<Vec<f32>> {
        let preds = self
            .steps
            .get(self.next)
            .cloned()
            .ok_or_else(|| SpeechError::Input {
                why: format!("replayed diarization has no step {}", self.next),
            })?;
        self.next += 1;
        // Recorded tails may carry earlier rows; keep this chunk's rows.
        let rows = frames.div_ceil(8);
        let have = preds.len() / self.num_speakers;
        if have < rows {
            return Err(SpeechError::Input {
                why: format!("replayed step has {have} rows, chunk needs {rows}"),
            });
        }
        Ok(preds[(have - rows) * self.num_speakers..].to_vec())
    }
}

/// Zeroes speakers at or past `max_speakers` in `[rows, num_speakers]`
/// predictions (the driver's `diar_preds[:, :, max_num_of_spks:] = 0`).
pub fn zero_extra_speakers(preds: &mut [f32], num_speakers: usize, max_speakers: usize) {
    for row in preds.chunks_exact_mut(num_speakers) {
        for p in &mut row[max_speakers.min(num_speakers)..] {
            *p = 0.0;
        }
    }
}

/// Active speakers from a `[frames, num_speakers]` window: per-speaker max
/// over time, the `max_active` largest (ties to the lower id), kept when
/// above `threshold`, ascending.
pub fn gate(window: &[f32], num_speakers: usize, max_active: usize, threshold: f32) -> Vec<usize> {
    let mut maxima = vec![f32::NEG_INFINITY; num_speakers];
    for row in window.chunks_exact(num_speakers) {
        for (m, &p) in maxima.iter_mut().zip(row) {
            *m = m.max(p);
        }
    }
    let mut order: Vec<usize> = (0..num_speakers).collect();
    order.sort_by(|&a, &b| maxima[b].total_cmp(&maxima[a]).then(a.cmp(&b)));
    let mut active: Vec<usize> = order
        .into_iter()
        .take(max_active)
        .filter(|&s| maxima[s] > threshold)
        .collect();
    active.sort_unstable();
    active
}

/// Target and background activity for `speaker` from `[rows, num_speakers]`
/// chunk predictions: the speaker's own probability, and whether any other
/// active speaker is above `threshold`. Binarized when `binary`.
pub fn targets(
    chunk: &[f32],
    num_speakers: usize,
    active: &[usize],
    speaker: usize,
    threshold: f32,
    binary: bool,
) -> (Vec<f32>, Vec<f32>) {
    let mut target = Vec::with_capacity(chunk.len() / num_speakers);
    let mut background = Vec::with_capacity(chunk.len() / num_speakers);
    for row in chunk.chunks_exact(num_speakers) {
        let t = row[speaker];
        let bg = active.iter().any(|&o| o != speaker && row[o] > threshold);
        target.push(if binary {
            f32::from(u8::from(t > threshold))
        } else {
            t
        });
        background.push(f32::from(u8::from(bg)));
    }
    (target, background)
}

/// What one step did, for traces and divergence reports.
#[derive(Debug, Clone, PartialEq)]
pub struct StepReport {
    pub step: usize,
    /// Chunk start time in seconds (NeMo's accumulated offset).
    pub offset: f64,
    pub asr_frames: usize,
    pub diar_frames: usize,
    /// `[R, num_speakers]` after zeroing speakers past `max_speakers`.
    pub chunk_preds: Vec<f32>,
    pub gate_max: Vec<f32>,
    pub active: Vec<usize>,
    pub spk_targets: Vec<Vec<f32>>,
    pub bg_targets: Vec<Vec<f32>>,
    pub pre_encoded_rows: usize,
    pub pre_encoded_valid: usize,
    /// Per active speaker: tokens emitted this step.
    pub new_tokens: Vec<Vec<Token>>,
    /// Per active speaker: every token id so far (NeMo `y_sequence`).
    pub tokens: Vec<Vec<usize>>,
    /// Per active speaker: every token's global frame (NeMo `timestamp`).
    pub token_frames: Vec<Vec<usize>>,
    /// Per active speaker after the step.
    pub decoded_frames: Vec<usize>,
    pub cache_len: Vec<usize>,
    pub texts: Vec<String>,
    pub last: bool,
}

/// One recording's coupled diarization and multitalker transcription.
pub struct CoupledSession<'m, D: DiarizationSource> {
    asr: &'m MultitalkerAsr,
    diar: D,
    cfg: CoupledConfig,
    preds: Vec<f32>,
    speakers: Vec<SpeakerStream>,
    decoded: Vec<bool>,
    sentences: SentenceBuilder,
    offset: f64,
    step: usize,
}

impl<'m, D: DiarizationSource> CoupledSession<'m, D> {
    pub fn new(asr: &'m MultitalkerAsr, diar: D, cfg: CoupledConfig) -> Self {
        Self {
            sentences: SentenceBuilder::new(cfg.frame_seconds, cfg.sentence_break),
            // NeMo's per-session ASR state starts with speaker 0's caches.
            speakers: vec![asr.new_stream()],
            decoded: vec![false],
            preds: Vec::new(),
            offset: 0.0,
            step: 0,
            asr,
            diar,
            cfg,
        }
    }

    /// Streams the whole recording through `step`, chunked as NeMo's
    /// `CacheAwareStreamingAudioBuffer` does with `pad_and_drop_preencoded`:
    /// every chunk is `[pre-encode cache ; shift frames]` (zeros before the
    /// start), the shift is `chunk_rows * subsampling_factor`, and a tail
    /// shorter than one subsampling group ends the stream.
    pub fn run(&mut self, features: &Features) -> Result<Vec<StepReport>> {
        let shift = self.asr.shift_frames();
        let cache = self.asr.pre_encode_cache_frames();
        let factor = self.asr.config().subsampling_factor;
        let mut reports = Vec::new();
        let mut idx = 0usize;
        while idx < features.frames {
            let new = shift.min(features.frames - idx);
            if idx > 0 && new < factor {
                break;
            }
            let last = idx + shift >= features.frames;
            reports.push(self.step(features, idx, new, cache, last)?);
            idx += shift;
        }
        Ok(reports)
    }

    fn step(
        &mut self,
        features: &Features,
        idx: usize,
        new: usize,
        cache: usize,
        last: bool,
    ) -> Result<StepReport> {
        let asr = self.asr;
        let n_spk = self.diar.num_speakers();
        let rows_per_chunk = asr.chunk_rows();

        // 1. Diarization on the chunk without the ASR pre-encode cache.
        let diar_mel = features.channel_major(idx, new);
        let mut chunk = self.diar.step(&diar_mel, new, new)?;
        zero_extra_speakers(&mut chunk, n_spk, self.cfg.max_speakers);
        self.preds.extend_from_slice(&chunk);
        let total_rows = self.preds.len() / n_spk;

        // 2-3. Chunk predictions and gating.
        let take = rows_per_chunk.min(total_rows);
        let chunk_preds = self.preds[(total_rows - take) * n_spk..].to_vec();
        let window_rows = (rows_per_chunk * self.cfg.gate_chunks).min(total_rows);
        let window = &self.preds[(total_rows - window_rows) * n_spk..];
        let mut gate_max = vec![f32::NEG_INFINITY; n_spk];
        for row in window.chunks_exact(n_spk) {
            for (m, &p) in gate_max.iter_mut().zip(row) {
                *m = m.max(p);
            }
        }
        let active = if self.cfg.cache_gating {
            gate(window, n_spk, self.cfg.max_speakers, self.cfg.threshold)
        } else {
            (0..self.cfg.max_speakers).collect()
        };

        let offset = self.offset;
        self.offset += rows_per_chunk as f64 * self.cfg.frame_seconds;

        // 6. One pre-encode shared by every active speaker.
        let frames = cache + new;
        let asr_chunk = features.frame_major(idx as isize - cache as isize, frames);
        let pre = asr.pre_encode(&asr_chunk, frames, frames, asr.drop_extra_pre_encoded())?;

        let mut report = StepReport {
            step: self.step,
            offset,
            asr_frames: frames,
            diar_frames: new,
            chunk_preds: chunk_preds.clone(),
            gate_max,
            active: active.clone(),
            spk_targets: Vec::new(),
            bg_targets: Vec::new(),
            pre_encoded_rows: pre.rows,
            pre_encoded_valid: pre.valid,
            new_tokens: Vec::new(),
            tokens: Vec::new(),
            token_frames: Vec::new(),
            decoded_frames: Vec::new(),
            cache_len: Vec::new(),
            texts: Vec::new(),
            last,
        };
        self.step += 1;
        if active.is_empty() {
            return Ok(report);
        }

        // 5. Lazy allocation of every id up to the largest active one.
        let top = *active.last().expect("nonempty");
        while self.speakers.len() <= top {
            self.speakers.push(asr.new_stream());
            self.decoded.push(false);
        }

        // 4 and 6. Conditioned per-speaker steps.
        for &s in &active {
            let (target, background) = targets(
                &chunk_preds,
                n_spk,
                &active,
                s,
                self.cfg.threshold,
                self.cfg.binary_targets,
            );
            let out = asr.step(&pre, &target, &background, &mut self.speakers[s], last)?;
            self.decoded[s] = true;
            let ids: Vec<usize> = self.speakers[s].tokens().iter().map(|t| t.id).collect();
            report.texts.push(asr.detokenize(&ids));
            report
                .token_frames
                .push(self.speakers[s].tokens().iter().map(|t| t.frame).collect());
            report.tokens.push(ids);
            report.new_tokens.push(out.tokens);
            report
                .decoded_frames
                .push(self.speakers[s].decoded_frames());
            report.cache_len.push(self.speakers[s].cache_len());
            report.spk_targets.push(target);
            report.bg_targets.push(background);
        }

        // 7. Sentence update from every decoded speaker.
        let texts: Vec<Option<(String, Vec<usize>, usize)>> = self
            .speakers
            .iter()
            .zip(&self.decoded)
            .map(|(stream, &decoded)| {
                decoded.then(|| {
                    let ids: Vec<usize> = stream.tokens().iter().map(|t| t.id).collect();
                    let frames = stream.tokens().iter().map(|t| t.frame).collect();
                    (asr.detokenize(&ids), frames, stream.decoded_frames())
                })
            })
            .collect();
        let views: Vec<Option<HypothesisView<'_>>> = texts
            .iter()
            .map(|t| {
                t.as_ref().map(|(text, frames, decoded)| HypothesisView {
                    text,
                    frames,
                    decoded_frames: *decoded,
                })
            })
            .collect();
        self.sentences.update(offset, &views);
        Ok(report)
    }

    /// The session's SegLST so far.
    pub fn seglst(&self) -> Vec<SeglstEntry> {
        self.sentences.entries()
    }

    /// Cumulative 80 ms predictions after speaker zeroing, `[frames, num_speakers]`.
    pub fn diarization(&self) -> &[f32] {
        &self.preds
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Window maxima pick the active set; ties go to the lower id; the
    /// threshold is strict; the result is sorted.
    #[test]
    fn gate_topk_threshold_sorted_known_answer() {
        // 2 rows x 4 speakers.
        let window = [0.2, 0.9, 0.51, 0.5, 0.7, 0.1, 0.4, 0.3];
        assert_eq!(gate(&window, 4, 4, 0.5), vec![0, 1, 2]);
        assert_eq!(gate(&window, 4, 2, 0.5), vec![0, 1]);
        // Equal maxima: the lower id wins the last slot.
        let tie = [0.8, 0.8, 0.8, 0.1];
        assert_eq!(gate(&tie, 4, 2, 0.5), vec![0, 1]);
        assert!(gate(&[0.5, 0.5], 2, 2, 0.5).is_empty());
    }

    /// Speakers 4..8 of the 8-speaker diarizer are zeroed for Frozen E's
    /// `max_num_of_spks = 4`, so the gate can never select them.
    #[test]
    fn speakers_ge4_zeroed() {
        let mut preds: Vec<f32> = (0..16).map(|i| 0.9 - 0.01 * i as f32).collect();
        zero_extra_speakers(&mut preds, 8, 4);
        assert!(preds[..4].iter().all(|&p| p > 0.5));
        assert!(preds[4..8].iter().all(|&p| p == 0.0));
        assert!(preds[12..].iter().all(|&p| p == 0.0));
        assert_eq!(gate(&preds, 8, 4, 0.5), vec![0, 1, 2, 3]);
        let mut all = vec![0.9f32; 8];
        zero_extra_speakers(&mut all, 8, 8);
        assert_eq!(all, vec![0.9; 8]);
    }

    /// Background counts only the other ACTIVE speakers, and targets are
    /// binarized at 0.5.
    #[test]
    fn bg_uses_other_active_only() {
        // 3 rows x 4 speakers; speaker 3 is loud but not active.
        let chunk = [
            0.9, 0.2, 0.0, 0.9, //
            0.4, 0.6, 0.0, 0.9, //
            0.6, 0.7, 0.0, 0.1,
        ];
        let (t0, bg0) = targets(&chunk, 4, &[0, 1], 0, 0.5, true);
        assert_eq!(t0, [1.0, 0.0, 1.0]);
        assert_eq!(bg0, [0.0, 1.0, 1.0]);
        let (t1, bg1) = targets(&chunk, 4, &[0, 1], 1, 0.5, false);
        assert_eq!(t1, [0.2, 0.6, 0.7]);
        assert_eq!(bg1, [1.0, 0.0, 1.0]);
    }

    /// Replayed predictions keep the newest rows the chunk needs (NeMo's
    /// final chunk has fewer rows than the recorded 14-row tail).
    #[test]
    fn replayed_diarization_keeps_this_chunks_rows() {
        let tail: Vec<f32> = (0..14 * 2).map(|i| i as f32).collect();
        let mut replay = ReplayedDiarization::new(vec![tail.clone(), tail], 2);
        assert_eq!(replay.step(&[], 112, 112).unwrap().len(), 28);
        assert_eq!(
            replay.step(&[], 17, 17).unwrap(),
            [22.0, 23.0, 24.0, 25.0, 26.0, 27.0]
        );
        assert!(replay.step(&[], 112, 112).is_err());
    }
}
