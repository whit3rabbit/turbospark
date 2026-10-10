//! The family sampling of `moss_tts_nano/sampling.py`.
//!
//! Differences from the shared [../sampling.rs](../sampling.rs) that
//! keep this a family-local port: the repetition penalty CLAMPS
//! out-of-range context ids into the vocabulary instead of refusing
//! them, it runs over the full per-channel history (no 20-token
//! window), the top-k filter runs BEFORE top-p on the
//! temperature-scaled logits (the lm `make_sampler` order is the
//! reverse), and both filters map their mask back onto the raw logits
//! (`_mask_logits_from_logprobs`) so the categorical draw sees
//! unnormalized scores. Draws use the crate-local MLX-compatible
//! threefry RNG; `do_sample=False` takes the argmax exactly like the
//! reference and never draws.

use crate::music::minimax_music3::rng;
use crate::tts::sampling as shared;
use crate::{Result, SpeechError};

/// `apply_repetition_penalty`: the sign-aware penalty over the full
/// `previous_token_ids` history, with the reference's clamping of
/// out-of-range ids to 0 (`mx.where(previous < 0, 0, ...)`) instead of a
/// refusal. Each distinct id in the history is penalized once.
pub fn apply_repetition_penalty(
    logits: &mut [f32],
    previous_token_ids: &[i32],
    repetition_penalty: f32,
) {
    if previous_token_ids.is_empty() || repetition_penalty == 1.0 {
        return;
    }
    let vocab_size = logits.len();
    let mut seen: Vec<usize> = Vec::new();
    for &token in previous_token_ids {
        let clamped = if token < 0 || token as usize >= vocab_size {
            0usize
        } else {
            token as usize
        };
        if !seen.contains(&clamped) {
            seen.push(clamped);
        }
    }
    for index in seen {
        logits[index] = if logits[index] < 0.0 {
            logits[index] * repetition_penalty
        } else {
            logits[index] / repetition_penalty
        };
    }
}

/// `apply_top_k`: log-softmax, the shared top-k mask, and the mask
/// mapped back onto the raw logits. `None`/0/`>= vocab` disables.
pub fn apply_top_k(logits: &[f32], top_k: Option<usize>) -> Result<Vec<f32>> {
    let disable = match top_k {
        None => true,
        Some(top_k) => top_k == 0 || top_k >= logits.len(),
    };
    if disable {
        return Ok(logits.to_vec());
    }
    let logprobs = shared::log_softmax(logits);
    let mut filtered = logprobs.clone();
    shared::apply_top_k(&mut filtered, top_k.unwrap())?;
    Ok(mask_from_logprobs(logits, &filtered))
}

/// `apply_top_p`: log-softmax, the shared top-p mask, and the mask
/// mapped back onto the raw logits. `None`/`<= 0`/`>= 1` disables.
pub fn apply_top_p(logits: &[f32], top_p: Option<f32>) -> Result<Vec<f32>> {
    let disable = match top_p {
        None => true,
        Some(top_p) => top_p <= 0.0 || top_p >= 1.0,
    };
    if disable {
        return Ok(logits.to_vec());
    }
    let logprobs = shared::log_softmax(logits);
    let mut filtered = logprobs;
    shared::apply_top_p(&mut filtered, top_p.unwrap());
    Ok(mask_from_logprobs(logits, &filtered))
}

/// `_mask_logits_from_logprobs`: keep the raw score where the filtered
/// log-probability survived, negative infinity where it was masked.
fn mask_from_logprobs(logits: &[f32], filtered_logprobs: &[f32]) -> Vec<f32> {
    logits
        .iter()
        .zip(filtered_logprobs)
        .map(|(&score, &logprob)| {
            if logprob == f32::NEG_INFINITY {
                f32::NEG_INFINITY
            } else {
                score
            }
        })
        .collect()
}

/// `sample_next_token` for one row: repetition penalty, then the greedy
/// argmax or the scaled top-k -> top-p -> categorical chain.
#[allow(clippy::too_many_arguments)]
pub fn sample_next_token(
    logits: &[f32],
    do_sample: bool,
    temperature: f32,
    top_k: Option<usize>,
    top_p: Option<f32>,
    previous_token_ids: Option<&[i32]>,
    repetition_penalty: f32,
    key: rng::Key,
) -> Result<usize> {
    let mut scores = logits.to_vec();
    if let Some(previous) = previous_token_ids {
        apply_repetition_penalty(&mut scores, previous, repetition_penalty);
    }
    if !do_sample {
        return Ok(shared::argmax(&scores));
    }
    if temperature <= 0.0 {
        return Err(SpeechError::Input {
            why: "temperature must be positive when do_sample=True".into(),
        });
    }
    for score in &mut scores {
        *score /= temperature;
    }
    let scores = apply_top_k(&scores, top_k)?;
    let scores = apply_top_p(&scores, top_p)?;
    Ok(rng::categorical(&scores, key))
}

/// `sample_assistant_text_token`: the continue/stop choice runs the same
/// sampler over the two gathered candidate logits
/// (`[audio_assistant_slot, audio_end]`) with the top-k capped at the
/// candidate count; the returned index maps back to the candidate ids.
#[allow(clippy::too_many_arguments)]
pub fn sample_assistant_text_token(
    text_logits: &[f32],
    audio_assistant_slot_token_id: i32,
    audio_end_token_id: i32,
    do_sample: bool,
    temperature: f32,
    top_k: Option<usize>,
    top_p: Option<f32>,
    key: rng::Key,
) -> Result<usize> {
    let candidates = [audio_assistant_slot_token_id, audio_end_token_id];
    let mut candidate_scores = Vec::with_capacity(2);
    for &id in &candidates {
        let index = usize::try_from(id).map_err(|_| SpeechError::Input {
            why: format!("negative candidate token {id}"),
        })?;
        let score = text_logits
            .get(index)
            .copied()
            .ok_or_else(|| SpeechError::Input {
                why: format!("candidate token {id} outside vocab {}", text_logits.len()),
            })?;
        candidate_scores.push(score);
    }
    let capped_top_k = top_k.map(|top_k| top_k.min(2));
    sample_next_token(
        &candidate_scores,
        do_sample,
        temperature,
        capped_top_k,
        top_p,
        None,
        1.0,
        key,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repetition_penalty_clamps_and_deduplicates() {
        // Vocabulary of 4; ids 9 and -3 clamp to 0, id 0 appears once in
        // the penalty set despite three occurrences.
        let mut logits = vec![2.0f32, -4.0, 6.0, 1.0];
        apply_repetition_penalty(&mut logits, &[0, 0, 9, -3], 2.0);
        assert_eq!(logits, vec![1.0, -4.0, 6.0, 1.0]);
        // A penalty of 1.0 and an empty history are no-ops.
        let mut logits = vec![2.0f32, -4.0];
        apply_repetition_penalty(&mut logits, &[0], 1.0);
        assert_eq!(logits, vec![2.0, -4.0]);
        apply_repetition_penalty(&mut logits, &[], 2.0);
        assert_eq!(logits, vec![2.0, -4.0]);
    }

    #[test]
    fn top_k_and_top_p_mask_map_back_to_raw_logits() {
        // Logprobs of [0.5, 0.3, 0.2] probabilities.
        let logits = [10.0f32, 9.407605, 8.988122];
        let top_k = apply_top_k(&logits, Some(2)).unwrap();
        assert_eq!(top_k[0], 10.0);
        assert_eq!(top_k[1], 9.407605);
        assert!(top_k[2] == f32::NEG_INFINITY);
        // top_p over the full range keeps everything.
        let kept = apply_top_p(&logits, Some(1.0)).unwrap();
        assert_eq!(kept, logits.to_vec());
        // top_p 0.5 keeps only the nucleus.
        let top_p = apply_top_p(&logits, Some(0.5)).unwrap();
        assert_eq!(top_p[0], 10.0);
        assert!(top_p[1] == f32::NEG_INFINITY);
        assert!(top_p[2] == f32::NEG_INFINITY);
    }

    #[test]
    fn sample_next_token_takes_argmax_when_not_sampling() {
        let logits = [0.2f32, 3.0, -1.0];
        let token = sample_next_token(
            &logits,
            false,
            0.8,
            Some(2),
            Some(0.9),
            None,
            1.2,
            rng::Key::new(7),
        )
        .unwrap();
        assert_eq!(token, 1);
    }

    #[test]
    fn sample_next_token_refuses_a_non_positive_temperature() {
        let logits = [0.2f32, 3.0];
        let error = sample_next_token(&logits, true, 0.0, None, None, None, 1.0, rng::Key::new(1))
            .unwrap_err();
        assert!(error.to_string().contains("temperature"), "{error}");
    }

    #[test]
    fn assistant_text_choice_follows_the_stronger_candidate() {
        // text_logits with the audio_end candidate (7) dominating.
        let mut text_logits = vec![0.0f32; 10];
        text_logits[7] = 4.0;
        text_logits[9] = 1.0;
        let picked = sample_assistant_text_token(
            &text_logits,
            9,
            7,
            false,
            1.0,
            Some(50),
            Some(1.0),
            rng::Key::new(3),
        )
        .unwrap();
        assert_eq!(picked, 1, "candidate index 1 is the audio_end token");
        text_logits[9] = 9.0;
        let picked = sample_assistant_text_token(
            &text_logits,
            9,
            7,
            false,
            1.0,
            Some(50),
            Some(1.0),
            rng::Key::new(3),
        )
        .unwrap();
        assert_eq!(picked, 0, "candidate index 0 is the assistant slot");
    }
}
