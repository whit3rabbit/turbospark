//! Deterministic sampler and logits processors shared by the LM-backed
//! TTS families.
//!
//! Port of the `mlx_audio/lm/sample_utils.py` subset the TTS families
//! exercise at the pinned mlx-audio commit
//! e1b19b9054bf163f5d812221a54fcc346f1890e9: the sign-aware repetition
//! penalty (defaults 1.3 over a 20-token context), the logit-bias
//! processor, the log-softmax normalization the reference generate step
//! applies after the processors, and the top-p / min-p / top-k filters
//! (`make_sampler` order: top-p first, then min-p, then XTC, then top-k,
//! then the categorical draw). XTC sampling is not ported: no reference
//! family enables it (`xtc_probability` defaults to 0).
//!
//! The categorical draw itself diverges from MLX by design: the reference
//! samples through `mx.random.state`, which is not reproducible from
//! Rust. Draws here use the crate-local MLX-compatible threefry RNG
//! (`crate::music::minimax_music3::rng`), so a fixed seed fixes the Rust
//! sequence while the filtered distribution is pinned by fixtures against
//! Python. Temperature 0 never draws: it takes the argmax exactly like
//! the reference `make_sampler(0, ...)`.

use crate::music::minimax_music3::rng;
use crate::{Result, SpeechError};

/// Adds each `(token, bias)` to the logit vector, the reference
/// `logit_bias_processor`.
pub fn apply_logit_bias(logits: &mut [f32], bias: &[(i32, f32)]) -> Result<()> {
    for &(token, value) in bias {
        let index = index_of(token, logits.len())?;
        logits[index] += value;
    }
    Ok(())
}

/// The sign-aware repetition penalty (`make_repetition_penalty`): every
/// token in the last `context_size` context tokens gets its logit
/// divided by `penalty` when positive and multiplied when negative. The
/// reference gathers and scatters in one step, so a token repeated in
/// the window is penalized once, not once per occurrence.
pub fn apply_repetition_penalty(
    logits: &mut [f32],
    context: &[i32],
    penalty: f32,
    context_size: usize,
) -> Result<()> {
    if penalty < 0.0 {
        return Err(SpeechError::Input {
            why: format!("repetition penalty must be non-negative, got {penalty}"),
        });
    }
    if context.is_empty() || penalty == 0.0 {
        return Ok(());
    }
    let start = context.len().saturating_sub(context_size);
    let mut seen: Vec<usize> = Vec::new();
    for &token in &context[start..] {
        let index = index_of(token, logits.len())?;
        if !seen.contains(&index) {
            seen.push(index);
        }
    }
    for index in seen {
        logits[index] = if logits[index] < 0.0 {
            logits[index] * penalty
        } else {
            logits[index] / penalty
        };
    }
    Ok(())
}

/// `logprobs = logits - logsumexp(logits)`, the normalization the
/// reference generate step applies before sampling.
pub fn log_softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for &x in logits {
        sum += (x - max).exp();
    }
    let lse = max + sum.ln();
    logits.iter().map(|&x| x - lse).collect()
}

/// Top-p (nucleus) filter (`apply_top_p`): keeps the tokens whose
/// ascending-order cumulative probability passes `1 - top_p` and masks
/// the rest to negative infinity. Call on normalized log-probabilities.
pub fn apply_top_p(logprobs: &mut [f32], top_p: f32) {
    let len = logprobs.len();
    let probs: Vec<f32> = logprobs.iter().map(|&x| x.exp()).collect();
    let mut order: Vec<usize> = (0..len).collect();
    // Ascending logprob order; the fixture vectors have no ties, which
    // is also where the reference argsort tie order is unspecified.
    order.sort_by(|&a, &b| {
        logprobs[a]
            .partial_cmp(&logprobs[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut cumulative = vec![0.0f32; len];
    let mut sum = 0.0f32;
    for &i in &order {
        sum += probs[i];
        cumulative[i] = sum;
    }
    let threshold = 1.0f32 - top_p;
    for (logprob, &mass) in logprobs.iter_mut().zip(&cumulative) {
        // Keep only where the ascending cumulative mass clears
        // `1 - top_p`; NaN masses mask too, matching `mx.where`.
        if mass > threshold {
            continue;
        }
        *logprob = f32::NEG_INFINITY;
    }
}

/// Top-k filter (`apply_top_k`): keeps the `top_k` largest log-probs and
/// masks the rest to negative infinity.
pub fn apply_top_k(logprobs: &mut [f32], top_k: usize) -> Result<()> {
    let len = logprobs.len();
    if top_k == 0 || top_k >= len {
        return Err(SpeechError::Input {
            why: format!("top_k has to be in the (0, {len}) interval, but is {top_k}"),
        });
    }
    let mut order: Vec<usize> = (0..len).collect();
    order.sort_by(|&a, &b| {
        logprobs[b]
            .partial_cmp(&logprobs[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    for &i in &order[top_k..] {
        logprobs[i] = f32::NEG_INFINITY;
    }
    Ok(())
}

/// Min-p filter (`apply_min_p`): masks every token whose log-probability
/// falls below `max + ln(min_p)`, keeping at least `min_tokens_to_keep`
/// tokens. Call on normalized log-probabilities; the reference guards the
/// call on `min_p != 0` (`make_sampler`), so 0 is refused here rather than
/// hitting `ln(0)`.
pub fn apply_min_p(logprobs: &mut [f32], min_p: f32, min_tokens_to_keep: usize) -> Result<()> {
    if !(0.0..=1.0).contains(&min_p) {
        return Err(SpeechError::Input {
            why: format!("min_p has to be a float in the [0, 1] interval, but is {min_p}"),
        });
    }
    if min_tokens_to_keep < 1 {
        return Err(SpeechError::Input {
            why: format!(
                "min_tokens_to_keep has to be a positive integer, but is {min_tokens_to_keep}"
            ),
        });
    }
    let len = logprobs.len();
    let top = logprobs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let scaled_min_p = top + min_p.ln();
    // The reference `argpartition` tie order is unspecified; the fixture
    // vectors have no ties. Descending stable order keeps the lowest
    // indices among equals, matching a stable sort.
    let mut order: Vec<usize> = (0..len).collect();
    order.sort_by(|&a, &b| {
        logprobs[b]
            .partial_cmp(&logprobs[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    // The top min_tokens_to_keep indices survive the cutoff unconditionally.
    let guaranteed = if min_tokens_to_keep > 1 {
        min_tokens_to_keep.min(len)
    } else {
        0
    };
    let mut keep = vec![false; len];
    for &index in &order[..guaranteed] {
        keep[index] = true;
    }
    for (index, logprob) in logprobs.iter_mut().enumerate() {
        if *logprob < scaled_min_p && !keep[index] {
            *logprob = f32::NEG_INFINITY;
        }
    }
    Ok(())
}

/// Index of the largest value; ties take the first occurrence.
pub fn argmax(values: &[f32]) -> usize {
    let mut best = 0usize;
    let mut best_value = f32::NEG_INFINITY;
    for (index, &value) in values.iter().enumerate() {
        if value > best_value {
            best = index;
            best_value = value;
        }
    }
    best
}

/// The reference `categorical_sampling(logprobs, temp)`:
/// `mx.random.categorical(logprobs * (1 / temp))` for one row. The draw
/// comes from the crate-local key sequence; see the module docs.
pub fn sample_categorical(logprobs: &[f32], temperature: f32, key: rng::Key) -> usize {
    let scale = 1.0 / temperature;
    let scaled: Vec<f32> = logprobs.iter().map(|&x| x * scale).collect();
    rng::categorical(&scaled, key)
}

fn index_of(token: i32, len: usize) -> Result<usize> {
    let index = usize::try_from(token).map_err(|_| SpeechError::Input {
        why: format!("negative token id {token} in a logits processor context"),
    })?;
    if index >= len {
        return Err(SpeechError::Input {
            why: format!("token id {token} outside logits of width {len}"),
        });
    }
    Ok(index)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn log_softmax_is_normalized_and_shift_invariant() {
        let logits = [1.5f32, -2.0, 0.25, 3.75];
        let logprobs = log_softmax(&logits);
        let total: f32 = logprobs.iter().map(|x| x.exp()).sum();
        assert!((total - 1.0).abs() < 1e-6, "total {total}");
        let shifted = log_softmax(&[
            logits[0] + 7.0,
            logits[1] + 7.0,
            logits[2] + 7.0,
            logits[3] + 7.0,
        ]);
        for (a, b) in logprobs.iter().zip(&shifted) {
            assert!((a - b).abs() < 1e-6, "{a} vs {b}");
        }
    }

    #[test]
    fn repetition_penalty_is_sign_aware_and_deduplicates() {
        let mut logits = vec![2.0f32, -4.0, 6.0, 1.0];
        apply_repetition_penalty(&mut logits, &[0, 0, 1], 2.0, 20).unwrap();
        assert_eq!(logits, vec![1.0, -8.0, 6.0, 1.0]);
    }

    #[test]
    fn repetition_penalty_truncates_the_context_window() {
        let mut logits = vec![2.0f32, 4.0, 8.0];
        apply_repetition_penalty(&mut logits, &[0, 1, 2], 2.0, 1).unwrap();
        // Only the last context token (2) is in the window.
        assert_eq!(logits, vec![2.0, 4.0, 4.0]);
    }

    #[test]
    fn top_p_keeps_the_nucleus_and_masks_the_tail() {
        // Logprobs of [0.5, 0.3, 0.2] probabilities.
        let logprobs = log_softmax(&[10.0f32, 9.407605, 8.988122]);
        let mut filtered = logprobs.clone();
        apply_top_p(&mut filtered, 0.5);
        // Ascending: 0.2 (cum 0.2, masked), 0.3 (cum 0.5, masked, not >),
        // 0.5 (cum 1.0, kept).
        assert!(filtered[0].is_finite());
        assert!(filtered[1].is_infinite() && filtered[1] < 0.0);
        assert!(filtered[2].is_infinite() && filtered[2] < 0.0);
    }

    #[test]
    fn top_k_keeps_exactly_k_entries() {
        let mut logprobs = vec![-1.0f32, -5.0, -2.0, -9.0];
        apply_top_k(&mut logprobs, 2).unwrap();
        assert!(logprobs[0].is_finite());
        assert!(logprobs[2].is_finite());
        assert!(logprobs[1] < 0.0 && logprobs[1].is_infinite());
        assert!(logprobs[3] < 0.0 && logprobs[3].is_infinite());
    }

    #[test]
    fn top_k_bounds_are_validated() {
        let mut logprobs = vec![-1.0f32, -2.0];
        assert!(apply_top_k(&mut logprobs, 0).is_err());
        assert!(apply_top_k(&mut logprobs, 2).is_err());
        assert!(apply_top_k(&mut logprobs, 3).is_err());
    }

    #[test]
    fn min_p_keeps_the_scaled_cutoff_and_the_guaranteed_tail() {
        // Logprobs of [0.5, 0.3, 0.12, 0.08] probabilities.
        let logprobs = log_softmax(&[10.0f32, 9.407605, 8.714476, 8.468172]);
        let top = logprobs[0];
        let mut filtered = logprobs.clone();
        apply_min_p(&mut filtered, 0.3, 1).unwrap();
        // Cutoff: 0.5 * 0.3 = 0.15; 0.12 and 0.08 fall below it.
        assert!(filtered[0].is_finite());
        assert!(filtered[1].is_finite());
        assert!(filtered[2].is_infinite() && filtered[2] < 0.0);
        assert!(filtered[3].is_infinite() && filtered[3] < 0.0);
        let _ = top;
        // min_tokens_to_keep = 3 keeps the third token despite the cutoff.
        let mut filtered = logprobs.clone();
        apply_min_p(&mut filtered, 0.3, 3).unwrap();
        assert!(filtered[2].is_finite());
        assert!(filtered[3].is_infinite() && filtered[3] < 0.0);
        // A min_p of 1 keeps only tokens equal to the max.
        let mut filtered = logprobs.clone();
        apply_min_p(&mut filtered, 1.0, 1).unwrap();
        assert!(filtered[0].is_finite());
        assert!(filtered[1].is_infinite());
    }

    #[test]
    fn min_p_bounds_are_validated() {
        let mut logprobs = vec![-1.0f32, -2.0];
        assert!(apply_min_p(&mut logprobs, -0.1, 1).is_err());
        assert!(apply_min_p(&mut logprobs, 1.1, 1).is_err());
        assert!(apply_min_p(&mut logprobs, 0.5, 0).is_err());
    }

    #[test]
    fn logit_bias_and_repetition_penalty_validate_indices() {
        let mut logits = vec![0.0f32; 4];
        assert!(apply_logit_bias(&mut logits, &[(-1, 1.0)]).is_err());
        assert!(apply_logit_bias(&mut logits, &[(9, 1.0)]).is_err());
        assert!(apply_repetition_penalty(&mut logits, &[9], 2.0, 20).is_err());
        assert!(apply_repetition_penalty(&mut logits, &[-1], 2.0, 20).is_err());
        assert!(apply_repetition_penalty(&mut logits, &[], 2.0, 20).is_ok());
        assert!(apply_repetition_penalty(&mut logits, &[0], -1.0, 20).is_err());
    }

    #[test]
    fn categorical_follows_overwhelming_mass_and_greedy_takes_argmax() {
        let mut logprobs = vec![0.0f32; 16];
        logprobs[11] = 100.0;
        assert_eq!(sample_categorical(&logprobs, 0.7, rng::Key::new(5)), 11);
        assert_eq!(argmax(&logprobs), 11);
    }
}
