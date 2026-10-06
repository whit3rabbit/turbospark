//! VAD chunking for Higgs Audio v3 transcription.
//!
//! Reference: `mlx_audio/stt/models/higgs_audio_3/vad.py` at mlx-audio 0.5.7,
//! commit `e1b19b9054bf163f5d812221a54fcc346f1890e9`. The reference calls a
//! Silero VAD backend for speech ranges, merges them into waveform-covering
//! spans (or keeps them raw under `split_vads`), splits every span into
//! chunks of at most `chunk_samples`, and falls back to a plain uniform
//! split whenever the backend yields nothing or raises.

use crate::vad::silero_vad::SileroVad;
use crate::{Result, SpeechError};

/// A speech-range source: `(start, end)` sample pairs clipped to the clip.
pub trait SpeechRanges {
    fn speech_ranges(&self, wav: &[f32]) -> Result<Vec<(usize, usize)>>;
}

/// The Silero VAD backend with the reference's default timing
/// (threshold 0.5, 250 ms minimum speech, 100 ms minimum silence, 30 ms
/// speech padding), backed by the shared `vad::silero_vad` model.
pub struct SileroRanges {
    model: SileroVad,
}

impl SileroRanges {
    /// Loads the local Silero VAD checkpoint directory
    /// (`model.safetensors`). Loading is local-only; tests never download.
    pub fn load(dir: &std::path::Path) -> Result<Self> {
        Ok(Self {
            model: SileroVad::load(dir)?,
        })
    }
}

impl SpeechRanges for SileroRanges {
    fn speech_ranges(&self, wav: &[f32]) -> Result<Vec<(usize, usize)>> {
        let ranges = self
            .model
            .detect(wav, 16_000)
            .map_err(|error| SpeechError::Input {
                why: format!("Silero VAD detection failed: {error}"),
            })?;
        Ok(ranges.into_iter().filter(|(s, e)| e > s).collect())
    }
}

/// `_split_long`: contiguous `[pos, min(end, pos + max_samples)]` pairs.
pub fn split_long(start: usize, end: usize, max_samples: usize) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut pos = start;
    while pos < end {
        let next = end.min(pos + max_samples);
        out.push((pos, next));
        pos = next;
    }
    out
}

/// The reference `vad_chunk_ranges`: no usable speech ranges means a plain
/// uniform split of the whole clip; raw ranges pass through under
/// `split_vads`; otherwise ranges merge into covering spans where each span
/// starts no earlier than the previous range's end and the final span always
/// reaches the end of the clip. Every span is then split into
/// `max_samples` chunks; an empty result falls back to the uniform split.
pub fn chunk_ranges(
    total: usize,
    chunk_samples: usize,
    cuts: &[(usize, usize)],
    split_vads: bool,
) -> Vec<(usize, usize)> {
    if cuts.is_empty() {
        return split_long(0, total, chunk_samples);
    }

    let spans: Vec<(usize, usize)> = if split_vads {
        cuts.to_vec()
    } else {
        let mut spans = Vec::new();
        let mut previous_end = 0usize;
        for (index, &(start, end)) in cuts.iter().enumerate() {
            let s = previous_end.min(start);
            let e = if index == cuts.len() - 1 { total } else { end };
            if e > s {
                spans.push((s, e));
            }
            previous_end = end;
        }
        spans
    };

    let mut out = Vec::new();
    for (s, e) in spans {
        out.extend(split_long(s, e, chunk_samples));
    }
    if out.is_empty() {
        split_long(0, total, chunk_samples)
    } else {
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_long_tiles_contiguous_ranges() {
        assert_eq!(split_long(0, 10, 4), vec![(0, 4), (4, 8), (8, 10)]);
        assert_eq!(split_long(5, 9, 10), vec![(5, 9)]);
        assert_eq!(split_long(3, 3, 4), Vec::<(usize, usize)>::new());
    }

    #[test]
    fn no_cuts_falls_back_to_a_uniform_split() {
        assert_eq!(
            chunk_ranges(10, 4, &[], false),
            vec![(0, 4), (4, 8), (8, 10)]
        );
    }

    #[test]
    fn merged_ranges_cover_the_whole_clip_and_the_last_span_reaches_the_end() {
        // A 20 s clip with speech at 2-6 s and 7-12 s, chunks of 8 samples:
        // spans become (0, 6) and (6, 20); each span tiles from its own
        // start, so the second span splits at 14 rather than at 8.
        let cuts = vec![(2usize, 6usize), (7usize, 12usize)];
        assert_eq!(
            chunk_ranges(20, 8, &cuts, false),
            vec![(0, 6), (6, 14), (14, 20)]
        );
    }

    #[test]
    fn split_vads_keeps_the_raw_ranges_and_tiles_them() {
        let cuts = vec![(2usize, 6usize), (7usize, 12usize)];
        assert_eq!(chunk_ranges(20, 8, &cuts, true), vec![(2, 6), (7, 12)]);
        // Long raw ranges still tile at the chunk size.
        let cuts = vec![(0usize, 30usize)];
        assert_eq!(
            chunk_ranges(30, 8, &cuts, true),
            vec![(0, 8), (8, 16), (16, 24), (24, 30)]
        );
    }

    #[test]
    fn an_empty_split_result_falls_back_to_the_uniform_split() {
        // Cuts whose spans all degenerate (e > s never holds after merging
        // is impossible for nonempty cuts, so exercise the guard through a
        // total of zero).
        let cuts = vec![(1usize, 2usize)];
        assert_eq!(
            chunk_ranges(0, 4, &cuts, false),
            Vec::<(usize, usize)>::new()
        );
    }

    #[test]
    fn merged_spans_tile_from_their_own_starts() {
        // Two speech ranges below the chunk size still produce two merged
        // spans: (0, 10000) and (10000, total); neither reaches the chunk
        // size, so no tiling happens inside them.
        let total = 44_715usize;
        let chunk = 64_000usize;
        let cuts = vec![(0usize, 10_000usize), (20_000usize, total)];
        assert_eq!(
            chunk_ranges(total, chunk, &cuts, false),
            vec![(0, 10_000), (10_000, total)]
        );
    }

    #[test]
    fn a_single_range_covers_whole_clip_and_stays_one_chunk() {
        // The smoke-clip property the pinned verification relies on: the
        // Silero backend reported one range (32, 44715), which merges into
        // (0, total) and stays one chunk below the chunk size.
        let total = 44_715usize;
        let chunk = 64_000usize;
        assert_eq!(chunk_ranges(total, chunk, &[], false), vec![(0, total)]);
        assert_eq!(
            chunk_ranges(total, chunk, &[(32usize, total)], false),
            vec![(0, total)]
        );
        assert_eq!(
            chunk_ranges(total, chunk, &[(1_000usize, 40_000usize)], false),
            vec![(0, total)]
        );
    }
}
