//! The selection pipeline: penalty application, truncation composition,
//! temperature reweighting, and the final draw.

use foundation::{LogitsView, TokenId};

use crate::derive::{derive_step_value, to_unit_interval};
use crate::penalty::apply_repetition_penalty;
use crate::shaping::{SelectionError, ShapingConfig};
use crate::truncation::{rank_indices, softmax, truncate_by_probability_mass, truncate_by_rank};

/// Select exactly one candidate identifier.
///
/// At the deterministic temperature value, this always agrees with the
/// single highest-scoring candidate under the raw score vector (a fixed,
/// consistent tie-break on equal scores), ignoring both truncation modes
/// and the seed.
///
/// At a positive temperature: the repetition penalty attenuates history
/// candidates first, then probability-mass truncation is evaluated against
/// the full distribution, then rank-based truncation caps that surviving
/// set, then temperature reweighting is applied immediately before a
/// seeded-or-clock-derived random draw. The result is always drawn from the
/// surviving set and is always a valid identifier in the declared candidate
/// domain, including on the defensive fallback branch that would otherwise
/// leave the result undefined.
///
/// Rejects a score vector containing a non-finite entry, and rejects an
/// empty candidate domain, each with a distinguishable descriptive error
/// rather than an undefined result.
pub fn select(
    scores: LogitsView<'_>,
    config: &ShapingConfig,
    history: &[TokenId],
    position: u64,
) -> Result<TokenId, SelectionError> {
    let raw = scores.as_slice();
    if raw.is_empty() {
        return Err(SelectionError {
            reason: "candidate domain must not be empty".to_string(),
        });
    }

    let mut working: Vec<f32> = raw.iter().map(|v| v.to_f32()).collect();
    if working.iter().any(|v| !v.is_finite()) {
        return Err(SelectionError {
            reason: "score vector must not contain a non-finite entry".to_string(),
        });
    }

    if config.is_deterministic() {
        return Ok(argmax(&working));
    }

    apply_repetition_penalty(&mut working, history, config.repetition_penalty());

    let probs = softmax(&working);
    let ranked = rank_indices(&probs);
    let mass_kept = truncate_by_probability_mass(&ranked, &probs, config.top_p());
    let surviving = truncate_by_rank(&mass_kept, config.top_k());

    if surviving.is_empty() {
        // Defensive fallback: never leave the selected identifier
        // undefined. Unreachable in the normal post-normalization case.
        return Ok(argmax(&working));
    }

    let reweighted = reweight(&surviving, &probs, config.temperature());
    let chosen = draw(&surviving, &reweighted, config.seed(), position);
    Ok(chosen as TokenId)
}

/// The single highest-scoring candidate, with ties broken by the first
/// (lowest-index) occurrence for a fixed, consistent result.
fn argmax(scores: &[f32]) -> TokenId {
    let mut best = 0usize;
    for (i, &s) in scores.iter().enumerate().skip(1) {
        if s > scores[best] {
            best = i;
        }
    }
    best as TokenId
}

/// Reweight the surviving candidates' full-distribution probabilities by
/// the inverse-temperature power, then renormalize over just the surviving
/// set.
fn reweight(surviving: &[usize], probs: &[f64], temperature: f64) -> Vec<f64> {
    let inv_t = 1.0 / temperature;
    let weighted: Vec<f64> = surviving.iter().map(|&i| probs[i].powf(inv_t)).collect();
    let sum: f64 = weighted.iter().sum();
    if sum <= 0.0 {
        let n = weighted.len() as f64;
        return weighted.iter().map(|_| 1.0 / n).collect();
    }
    weighted.into_iter().map(|w| w / sum).collect()
}

/// Draw one surviving candidate using a seeded-or-clock-derived uniform
/// value over the reweighted distribution.
fn draw(surviving: &[usize], weights: &[f64], seed: Option<u64>, position: u64) -> usize {
    let step_value = derive_step_value(seed, position);
    let target = to_unit_interval(step_value);
    let mut cumulative = 0.0;
    for (i, &w) in weights.iter().enumerate() {
        cumulative += w;
        if target < cumulative {
            return surviving[i];
        }
    }
    // Numerical rounding at the tail: fall back to the last surviving
    // candidate rather than leaving the draw undefined.
    *surviving
        .last()
        .expect("surviving set is checked non-empty by the caller")
}
