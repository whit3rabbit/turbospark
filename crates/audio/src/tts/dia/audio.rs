//! Dia audio-token plumbing (`audio.py`): the forward delay pattern
//! applied to prompt-audio codes, the revert applied to generated codes
//! before the DAC decode, and the codebook-to-audio conversion with its
//! invalid-code clamp.

use crate::codec::descript::Dac;
use crate::{Result, SpeechError};

/// `apply_audio_delay` over a `[frames][channels]` code grid: position
/// `t` of channel `c` reads `t - delay[c]`, with BOS where the source
/// index is negative. (The pad branch of the reference can never fire
/// because the output length equals the input length; kept for shape
/// fidelity.)
pub(crate) fn apply_audio_delay(
    codes: &[Vec<i32>],
    delay: &[usize],
    pad_value: i32,
    bos_value: i32,
) -> Vec<Vec<i32>> {
    let frames = codes.len();
    let channels = delay.len();
    let mut out = vec![vec![pad_value; channels]; frames];
    for (t, row) in out.iter_mut().enumerate() {
        for (c, &d) in delay.iter().enumerate() {
            let source = t as isize - d as isize;
            row[c] = if source < 0 {
                bos_value
            } else {
                codes[source as usize][c]
            };
        }
    }
    out
}

/// `revert_audio_delay` over a `[frames][channels]` grid: position `t`
/// of channel `c` reads `min(t + delay[c], frames - 1)`. The reference
/// caps the index before comparing against the pad threshold, so the
/// pad branch never fires; the cap itself repeats the last frame for
/// the delayed channels and is part of the pinned behavior.
pub(crate) fn revert_audio_delay(
    codes: &[Vec<i32>],
    delay: &[usize],
    pad_value: i32,
) -> Vec<Vec<i32>> {
    let frames = codes.len();
    let channels = delay.len();
    let mut out = vec![vec![pad_value; channels]; frames];
    for (t, row) in out.iter_mut().enumerate() {
        for (c, &d) in delay.iter().enumerate() {
            let source = (t + d).min(frames.saturating_sub(1));
            row[c] = codes[source][c];
        }
    }
    out
}

/// `codebook_to_audio`: strips the first frame (the reference drops the
/// frame after the BOS position), truncates to `max_tokens`, reverts
/// the delay, drops the final 30 frames (the `extra_steps_after_eos`
/// tail), clamps codes outside `[0, 1023]` to 0, and decodes through
/// the DAC. Returns the waveform and the number of clamped codes.
pub(crate) fn codebook_to_audio(
    codes: &[Vec<i32>],
    dac: &Dac,
    delay: &[usize],
    max_tokens: usize,
) -> Result<(Vec<f32>, usize)> {
    let channels = delay.len();
    let mut frames: Vec<Vec<i32>> = Vec::new();
    for row in codes {
        let mut row = row.clone();
        if !row.is_empty() {
            row.remove(0);
        }
        frames.push(row);
    }
    for row in &mut frames {
        row.truncate(max_tokens);
    }
    let seq_length = frames.first().map(|r| r.len()).unwrap_or(0);
    // Transpose to [frames][channels] for the revert.
    let mut grid = vec![vec![0i32; channels]; seq_length];
    for (c, row) in frames.iter().enumerate().take(channels) {
        for (t, &v) in row.iter().enumerate() {
            grid[t][c] = v;
        }
    }
    let reverted = revert_audio_delay(&grid, delay, 0);
    // `[:, :-30, :]`: the 30 extra post-EOS frames are dropped; a
    // sequence of 30 frames or fewer decodes to nothing (an empty
    // slice, not an error, exactly like MLX).
    let kept = seq_length.saturating_sub(30);
    let mut clamped = 0usize;
    let mut levels: Vec<Vec<i32>> = vec![Vec::with_capacity(kept); channels];
    for row in reverted.iter().take(kept) {
        for (c, level) in levels.iter_mut().enumerate() {
            let mut code = row[c];
            if !(0..=1023).contains(&code) {
                code = 0;
                clamped += 1;
            }
            level.push(code);
        }
    }
    if kept == 0 {
        // The reference crashes here (a conv over a zero-length grid
        // inside the quantizer from_codes path); the port refuses with
        // a descriptive error instead.
        return Err(SpeechError::Input {
            why: "empty decode: the sliced code grid drops the 30-frame post-EOS tail and nothing remains".into(),
        });
    }
    let audio = dac.decode_codes(&levels)?;
    Ok((audio, clamped))
}

/// `audio_to_codebook`: encodes mono prompt audio through the DAC
/// (which right-pads to the hop multiple exactly like `preprocess`) and
/// applies the delay pattern. Returns `[frames][channels]`.
pub(crate) fn audio_to_codebook(
    dac: &Dac,
    samples: &[f32],
    delay: &[usize],
    bos: i32,
    pad: i32,
) -> Result<Vec<Vec<i32>>> {
    let levels = dac.encode(samples)?;
    let frames = levels.first().map(|l| l.len()).unwrap_or(0);
    let channels = delay.len();
    let mut grid = vec![vec![0i32; channels]; frames];
    for (c, level) in levels.iter().enumerate().take(channels) {
        for (t, &code) in level.iter().enumerate() {
            grid[t][c] = code;
        }
    }
    Ok(apply_audio_delay(&grid, delay, pad, bos))
}
