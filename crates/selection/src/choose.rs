//! The selection pipeline: penalty application, truncation composition,
//! temperature reweighting, and the final draw.

use std::cell::RefCell;

use foundation::{LogitsView, TokenId};

use crate::derive::{derive_step_value, to_unit_interval};
use crate::penalty::{apply_presence_and_frequency_penalty, apply_repetition_penalty};
use crate::shaping::{SelectionError, ShapingConfig};
use crate::truncation::{rank_indices_u32_into, rank_top_k_u32_into};

thread_local! {
    // Reused across calls on purpose: at the real Gemma 4 vocabulary
    // (262144) these buffers are multi-megabyte, and this function runs
    // once per decoded token OUTSIDE `produce`, where no profiler in this
    // repo can see it (AGENTS.md Gotcha 23). Fresh allocations per token
    // were a measurable share of the sampler's ~2.6 ms/token.
    static SCRATCH: RefCell<Scratch> = RefCell::new(Scratch::default());
}

#[derive(Default)]
struct Scratch {
    working: Vec<f32>,
    exps: Vec<f64>,
    ranked: Vec<u32>,
}

/// Select exactly one candidate identifier.
///
/// At the deterministic temperature value, this always agrees with the
/// single highest-scoring candidate under the raw score vector (a fixed,
/// consistent tie-break on equal scores), ignoring both truncation modes
/// and the seed.
///
/// At a positive temperature: the repetition penalty attenuates history
/// candidates first, then the presence and frequency penalties attenuate
/// the GENERATED suffix of history (`&history[history.len() - position..]`
/// -- `position` doubles as the generated-token count, which is what every
/// caller in `crates/runtime` passes), then probability-mass truncation is
/// evaluated against the full distribution, then rank-based truncation caps
/// that surviving set, then min-p truncation (if enabled) drops whatever
/// remains below its threshold of the ranked prefix's own top score, then
/// temperature reweighting is applied immediately before a
/// seeded-or-clock-derived random draw. The result is always drawn from the
/// surviving set and is always a valid identifier in the declared candidate
/// domain, including on the defensive fallback branch that would otherwise
/// leave the result undefined.
///
/// Rejects a score vector containing a non-finite entry, and rejects an
/// empty candidate domain, each with a distinguishable descriptive error
/// rather than an undefined result.
///
/// Internally the full probability vector is never materialized: candidates
/// are ranked by their unnormalized `exp(s - max)` values (division by the
/// positive normalizer is monotone, so the order is the same), and the
/// normalized probability `exps[i] / sum` is computed on the fly only for
/// the ranked prefix the probability-mass walk touches and for the
/// surviving set the reweight touches -- bit-identical divisions to the
/// retired full vector's entries.
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

    SCRATCH.with(|scratch| {
        let Scratch {
            working,
            exps,
            ranked,
        } = &mut *scratch.borrow_mut();

        let (surviving, sum) =
            shape_candidates(raw, working, exps, ranked, config, history, position)?;

        // The deterministic arm short-circuits INSIDE `shape_candidates`
        // (argmax ignores penalties and truncation), and its surviving set
        // is that single candidate. Returning HERE rather than falling
        // through to the reweight is load-bearing: at this point `exps` is
        // still empty -- the softmax never ran -- and a point mass needs
        // neither it nor a draw.
        if config.is_deterministic() {
            return Ok(surviving[0] as TokenId);
        }

        let reweighted = reweight(&surviving, exps, sum, config.temperature());
        let chosen = draw(&surviving, &reweighted, config.seed(), position);
        Ok(chosen as TokenId)
    })
}

/// The shared pipeline half: finiteness, penalties, stable softmax, ranking
/// and truncation, ending at the SURVIVING candidate list and the softmax's
/// normalizer. [`select`] draws from the reweighted survivors;
/// [`shaped_distribution`] packages them. One body rather than two is the
/// whole point: rejection sampling's exactness claim is "over the same
/// distribution `select` samples from", and a second implementation that
/// agrees with the first until it does not is how that claim would rot.
///
/// The buffers are the caller's SCRATCH (see the thread-local's doc); the
/// surviving list is returned OWNED because both callers reweight inside
/// the scratch borrow.
fn shape_candidates(
    raw: &[foundation::LogitValue],
    working: &mut Vec<f32>,
    exps: &mut Vec<f64>,
    ranked: &mut Vec<u32>,
    config: &ShapingConfig,
    history: &[TokenId],
    position: u64,
) -> Result<(Vec<u32>, f64), SelectionError> {
    working.clear();
    working.reserve(raw.len());
    let mut all_finite = true;
    for v in raw {
        let f = v.to_f32();
        all_finite &= f.is_finite();
        working.push(f);
    }
    if !all_finite {
        return Err(SelectionError {
            reason: "score vector must not contain a non-finite entry".to_string(),
        });
    }

    if config.is_deterministic() {
        return Ok((vec![argmax(working) as u32], 1.0));
    }

    apply_repetition_penalty(working, history, config.repetition_penalty());

    // `position` IS the generated-token count: every caller in
    // `crates/runtime` passes `sink.generated as u64` here (the
    // speculative path commits each accepted proposal through the same
    // sink before its next `select` call, which is what keeps this
    // exact across both decode loops -- see `speculative.rs`'s own
    // comment on that invariant). Clamped defensively rather than
    // trusted, so a future caller that violates it degrades to "treat
    // all of history as generated" instead of underflowing.
    let generated = (position as usize).min(history.len());
    apply_presence_and_frequency_penalty(
        working,
        &history[history.len() - generated..],
        config.presence_penalty(),
        config.frequency_penalty(),
    );

    // Numerically stable softmax, kept as unnormalized exps plus their
    // sum. The summation order matches the retired `softmax`'s sequential
    // `iter().sum()`, so `exps[i] / sum` reproduces its probabilities bit
    // for bit.
    let max = working.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
    exps.clear();
    exps.reserve(working.len());
    for &w in working.iter() {
        let e = (w as f64 - max).exp();
        exps.push(e);
    }
    let mut sum = 0.0f64;
    for &e in exps.iter() {
        sum += e;
    }

    // Rank-based truncation caps the surviving set at `top_k`, and
    // probability-mass truncation only ever keeps a prefix of the
    // ranked order, so ranking beyond `top_k` cannot change the
    // result. Ranking the whole domain when it is disabled is the
    // fallback, not the path.
    match config.top_k() {
        0 => rank_indices_u32_into(exps, ranked),
        k => rank_top_k_u32_into(exps, k as usize, ranked),
    }

    // Probability-mass truncation over the ranked prefix, against the
    // full-distribution probabilities.
    let mut surviving: Vec<u32> = Vec::new();
    match config.top_p() {
        None => surviving.extend_from_slice(ranked),
        Some(threshold) => {
            let mut cumulative = 0.0;
            for &i in ranked.iter() {
                surviving.push(i);
                cumulative += exps[i as usize] / sum;
                if cumulative >= threshold {
                    break;
                }
            }
            if surviving.is_empty() && !ranked.is_empty() {
                surviving.push(ranked[0]);
            }
        }
    }
    // Rank-based truncation caps whatever survived the mass step.
    if config.top_k() != 0 {
        surviving.truncate(config.top_k() as usize);
    }

    // Min-p composes AFTER the top-k cut, against the ranked prefix's
    // own top score -- `ranked[0]`, since `ranked` is sorted descending
    // and always starts from the full domain, so it is the true global
    // max whether or not top-k pre-limited the ranking. Pinned by
    // `tests/min_p.rs`'s composition-order test.
    if let Some(min_p) = config.min_p() {
        if let Some(&top) = ranked.first() {
            let threshold = min_p * exps[top as usize];
            surviving.retain(|&i| exps[i as usize] >= threshold);
            if surviving.is_empty() {
                surviving.push(top);
            }
        }
    }

    if surviving.is_empty() {
        // Defensive fallback: never leave the selected identifier
        // undefined. Unreachable in the normal post-normalization case.
        return Ok((vec![argmax(working) as u32], 1.0));
    }

    Ok((surviving, sum))
}

/// The distribution [`select`] samples from, materialized: the surviving
/// support and its temperature-reweighted probabilities -- the target `p`
/// of Leviathan/Chen rejection sampling, and the drafter-side `q` when
/// built from DRAFT logits under the same shaping.
///
/// At the deterministic temperature this is the point mass `select`'s argmax
/// arm samples, penalties and truncation ignored exactly as that arm
/// ignores them -- a caller asking for "the same distribution" at T=0 gets
/// the same answer `select` gives.
pub fn shaped_distribution(
    scores: LogitsView<'_>,
    config: &ShapingConfig,
    history: &[TokenId],
    position: u64,
) -> Result<crate::distribution::ShapedDistribution, SelectionError> {
    let raw = scores.as_slice();
    if raw.is_empty() {
        return Err(SelectionError {
            reason: "candidate domain must not be empty".to_string(),
        });
    }

    SCRATCH.with(|scratch| {
        let Scratch {
            working,
            exps,
            ranked,
        } = &mut *scratch.borrow_mut();

        let (surviving, sum) =
            shape_candidates(raw, working, exps, ranked, config, history, position)?;

        if config.is_deterministic() {
            return Ok(crate::distribution::ShapedDistribution::point_mass(
                surviving[0],
            ));
        }

        let probs = reweight(&surviving, exps, sum, config.temperature());
        Ok(crate::distribution::ShapedDistribution::new(
            surviving, probs,
        ))
    })
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
fn reweight(surviving: &[u32], exps: &[f64], sum: f64, temperature: f64) -> Vec<f64> {
    let inv_t = 1.0 / temperature;
    let weighted: Vec<f64> = surviving
        .iter()
        .map(|&i| (exps[i as usize] / sum).powf(inv_t))
        .collect();
    let total: f64 = weighted.iter().sum();
    if total <= 0.0 {
        let n = weighted.len() as f64;
        return weighted.iter().map(|_| 1.0 / n).collect();
    }
    weighted.into_iter().map(|w| w / total).collect()
}

/// Draw one surviving candidate using a seeded-or-clock-derived uniform
/// value over the reweighted distribution.
fn draw(surviving: &[u32], weights: &[f64], seed: Option<u64>, position: u64) -> u32 {
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
