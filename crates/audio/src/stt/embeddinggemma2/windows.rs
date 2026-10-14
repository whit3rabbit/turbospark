//! Fixed-shape window schedule and frame stitching.
//!
//! The compiled encoder takes exactly 20 s (2000 mel frames) and emits 500
//! reader frames of 40 ms. Windows start every 18 s, so consecutive windows
//! overlap by 2 s. An interior window keeps only its local frames
//! `[25, 475)`, the central 18 s, which tile the timeline exactly: window
//! `w` covers global frames `[450 w + 25, 450 w + 475)`. Constants come from
//! the reference runtime (`WIN`, `STEP`, `KEEP0`, `KEEP1`).
//!
//! The first window keeps from local frame 0 and the last window keeps
//! through its final valid frame, so audio near either end is not dropped.
//! That edge handling is this port's choice: the reference constants alone
//! do not show how its runtime treats the first and last second.

use std::ops::Range;

use super::frontend::HOP_LENGTH;

pub const WINDOW_SAMPLES: usize = 320_000;
pub const STEP_SAMPLES: usize = 288_000;
pub const MEL_FRAMES: usize = 2_000;
pub const ENCODER_FRAMES: usize = 500;
/// Mel frames per encoder frame (the audio tower's 4x subsampling).
pub const SUBSAMPLING: usize = MEL_FRAMES / ENCODER_FRAMES;
pub const KEEP_START: usize = 25;
pub const KEEP_END: usize = 475;
/// Encoder frames per window step (`STEP_SAMPLES / (HOP_LENGTH * 4)`).
pub const STEP_FRAMES: usize = STEP_SAMPLES / (HOP_LENGTH * SUBSAMPLING);

/// One encoder window over the input.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WindowPlan {
    pub index: usize,
    /// First input sample of the window.
    pub sample_start: usize,
    /// Real samples in the window (`<= WINDOW_SAMPLES`); the rest is zero
    /// padding the frontend masks.
    pub sample_len: usize,
    /// Local encoder frames whose labels enter the global path.
    pub keep: Range<usize>,
    /// Global frame index of local frame 0.
    pub frame_offset: usize,
}

impl WindowPlan {
    /// Global frames this window contributes.
    pub fn global_frames(&self) -> Range<usize> {
        self.frame_offset + self.keep.start..self.frame_offset + self.keep.end
    }
}

/// Encoder frames carrying real audio for a window whose mel mask has
/// `valid_mel` set frames. The encoder mask takes every 4th mel mask entry
/// (the Gemma audio subsampling convention), hence the ceiling.
pub fn valid_encoder_frames(valid_mel: usize) -> usize {
    valid_mel.div_ceil(SUBSAMPLING).min(ENCODER_FRAMES)
}

/// Plans the windows covering `total_samples`. `valid_frames(len)` returns
/// the encoder frames with real audio for a window of `len` real samples;
/// the pipeline passes the frontend's mask count through
/// [`valid_encoder_frames`].
pub fn plan_windows(
    total_samples: usize,
    valid_frames: impl Fn(usize) -> usize,
) -> Vec<WindowPlan> {
    let mut plans = Vec::new();
    if total_samples == 0 {
        return plans;
    }
    let mut start = 0usize;
    loop {
        let len = (total_samples - start).min(WINDOW_SAMPLES);
        let last = start + WINDOW_SAMPLES >= total_samples;
        let index = plans.len();
        let keep_start = if index == 0 { 0 } else { KEEP_START };
        let keep_end = if last {
            valid_frames(len).min(ENCODER_FRAMES)
        } else {
            KEEP_END
        };
        plans.push(WindowPlan {
            index,
            sample_start: start,
            sample_len: len,
            keep: keep_start..keep_end.max(keep_start),
            frame_offset: index * STEP_FRAMES,
        });
        if last {
            return plans;
        }
        start += STEP_SAMPLES;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn by_samples(len: usize) -> usize {
        // Valid mel frames for `len` real samples: i * 160 + 320 < len + 160.
        let mel = if len <= HOP_LENGTH {
            0
        } else {
            (len - HOP_LENGTH).div_ceil(HOP_LENGTH)
        };
        valid_encoder_frames(mel)
    }

    #[test]
    fn constants_tile_the_timeline() {
        assert_eq!(STEP_FRAMES, 450);
        assert_eq!(KEEP_END - KEEP_START, STEP_FRAMES);
        assert_eq!(SUBSAMPLING, 4);
    }

    #[test]
    fn short_audio_is_one_window_kept_to_its_valid_frames() {
        let plans = plan_windows(16_000, by_samples);
        assert_eq!(plans.len(), 1);
        assert_eq!(plans[0].sample_len, 16_000);
        // 99 valid mel frames -> 25 encoder frames.
        assert_eq!(plans[0].keep, 0..25);
        assert!(plan_windows(0, by_samples).is_empty());
    }

    #[test]
    fn exactly_twenty_seconds_is_one_full_window() {
        let plans = plan_windows(WINDOW_SAMPLES, by_samples);
        assert_eq!(plans.len(), 1);
        assert_eq!(plans[0].keep, 0..ENCODER_FRAMES);
    }

    #[test]
    fn long_audio_windows_stitch_without_gaps_or_overlap() {
        let total = 60 * 16_000 + 1234;
        let plans = plan_windows(total, by_samples);
        assert_eq!(plans.len(), 4);
        let mut next = 0;
        for plan in &plans {
            assert_eq!(plan.sample_start, plan.index * STEP_SAMPLES);
            let frames = plan.global_frames();
            assert_eq!(frames.start, next, "window {} leaves a gap", plan.index);
            next = frames.end;
        }
        assert_eq!(plans[0].keep, 0..KEEP_END);
        assert_eq!(plans[1].keep, KEEP_START..KEEP_END);
        let last = plans.last().unwrap();
        assert_eq!(last.sample_start + last.sample_len, total);
        // The stitched path ends at the last frame with real audio.
        assert_eq!(next, last.frame_offset + by_samples(last.sample_len));
        assert!((next as f64 * 0.04 - total as f64 / 16_000.0).abs() < 0.05);
    }

    #[test]
    fn one_sample_past_a_window_opens_a_second_window() {
        let plans = plan_windows(WINDOW_SAMPLES + 1, by_samples);
        assert_eq!(plans.len(), 2);
        assert_eq!(plans[0].keep, 0..KEEP_END);
        assert_eq!(plans[1].sample_len, WINDOW_SAMPLES + 1 - STEP_SAMPLES);
        assert!(plans[1].keep.start == KEEP_START && plans[1].keep.end > KEEP_START);
    }
}
