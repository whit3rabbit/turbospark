//! The family sampling of `moss_tts/sampling.py`, shared verbatim by the
//! delay-pattern and local-transformer variants.
//!
//! Same shape as the moss_tts_nano family sampler (the reference file is
//! the sibling's `sampling.py` lifted into this model): the repetition
//! penalty clamps out-of-range context ids into the vocabulary instead of
//! refusing them and runs over the full history, the top-k filter runs
//! BEFORE top-p on scores the caller already temperature-scaled, and both
//! filters map their log-probability mask back onto the raw logits so the
//! categorical draw sees unnormalized scores. Temperature division lives
//! at the call sites here (the reference divides `logits[:, -1, :]` by
//! the text or audio temperature before `sample_token`), unlike the nano
//! sampler which scales inside. Draws use the crate-local MLX-compatible
//! threefry RNG; greedy (`do_sample=False`) takes the argmax and never
//! draws.

use crate::music::minimax_music3::rng;
use crate::tts::sampling as shared;
use crate::Result;

/// `apply_repetition_penalty_delay_pattern` for one logits row: the
/// sign-aware penalty over the full `prev_tokens` history, with the
/// reference's clamping of negative and out-of-range ids to 0 instead of
/// a refusal. Each distinct id in the history is penalized once.
pub fn apply_repetition_penalty(logits: &mut [f32], prev_tokens: &[i32], penalty: f32) {
    if prev_tokens.is_empty() || penalty == 1.0 {
        return;
    }
    let vocab_size = logits.len();
    let mut seen: Vec<usize> = Vec::new();
    for &token in prev_tokens {
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
            logits[index] * penalty
        } else {
            logits[index] / penalty
        };
    }
}

/// `apply_top_k`: log-softmax, the shared top-k mask, and the mask
/// mapped back onto the raw logits. `None`/`<= 0`/`>= vocab` disables.
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

/// `sample_token` for one row: the repetition penalty first, then the
/// greedy argmax or the top-k -> top-p -> categorical chain over the
/// scores as given (the caller divides by temperature beforehand).
pub fn sample_token(
    logits: &[f32],
    prev_tokens: Option<&[i32]>,
    repetition_penalty: f32,
    top_p: Option<f32>,
    top_k: Option<usize>,
    do_sample: bool,
    key: rng::Key,
) -> Result<usize> {
    let mut scores = logits.to_vec();
    if let Some(previous) = prev_tokens {
        apply_repetition_penalty(&mut scores, previous, repetition_penalty);
    }
    if !do_sample {
        return Ok(shared::argmax(&scores));
    }
    let scores = apply_top_k(&scores, top_k)?;
    let scores = apply_top_p(&scores, top_p)?;
    Ok(rng::categorical(&scores, key))
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
        let top_p = apply_top_p(&logits, Some(0.5)).unwrap();
        assert_eq!(top_p[0], 10.0);
        assert!(top_p[1] == f32::NEG_INFINITY);
        assert!(top_p[2] == f32::NEG_INFINITY);
        // Disabled filters pass the scores through unchanged.
        let kept = apply_top_k(&logits, Some(3)).unwrap();
        assert_eq!(kept, logits.to_vec());
        let kept = apply_top_p(&logits, Some(1.0)).unwrap();
        assert_eq!(kept, logits.to_vec());
    }

    #[test]
    fn sample_token_takes_argmax_when_not_sampling() {
        let logits = [0.2f32, 3.0, -1.0];
        let token = sample_token(
            &logits,
            Some(&[2, 2]),
            1.1,
            Some(0.9),
            Some(25),
            false,
            rng::Key::new(7),
        )
        .unwrap();
        assert_eq!(token, 1);
    }

    #[test]
    fn sample_token_applies_the_penalty_before_the_filters() {
        // The history token 0 would win the raw argmax; the penalty drops
        // it below token 1, which the greedy path then takes.
        let logits = [5.0f32, 4.0, -1.0];
        let token = sample_token(
            &logits,
            Some(&[0]),
            2.0,
            Some(1.0),
            Some(25),
            false,
            rng::Key::new(3),
        )
        .unwrap();
        assert_eq!(token, 1);
    }
}
