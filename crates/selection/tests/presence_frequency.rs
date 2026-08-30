//! Presence- and frequency-penalty effect scenarios, observed through the
//! composed selection pipeline -- the same shape as `tests/penalty.rs`, for
//! the two OpenAI-style knobs `ShapingConfig::with_presence_penalty` and
//! `ShapingConfig::with_frequency_penalty` add.

use std::collections::HashMap;

use foundation::{LogitValue, LogitsView, TokenId};
use turbospark_selection::{select, ShapingConfig};

fn logits(values: &[f32]) -> Vec<LogitValue> {
    values.iter().map(|&v| LogitValue::from_f32(v)).collect()
}

/// `position` doubles as the generated-token count `select` reads (see
/// `choose.rs`'s doc comment), so trial `t` here also means "t tokens
/// generated so far" -- consistent with how every real caller in
/// `crates/runtime` uses it, and what makes the prompt-vs-generated split
/// below observable at all.
fn frequency(config: &ShapingConfig, history: &[TokenId], trials: u64) -> HashMap<TokenId, u32> {
    let scores = logits(&[2.0, 2.0, 2.0, 2.0]);
    let view = LogitsView::new(&scores);
    let mut counts = HashMap::new();
    for position in 0..trials {
        let chosen = select(view, config, history, position).unwrap();
        *counts.entry(chosen).or_insert(0) += 1;
    }
    counts
}

#[test]
fn a_presence_penalized_candidate_is_selected_materially_less_often() {
    let history = vec![0i32];
    let trials = 400;

    let unpenalized = ShapingConfig::new(1.0, 0, None, 1.0, Some(1)).unwrap();
    let penalized = ShapingConfig::new(1.0, 0, None, 1.0, Some(1))
        .unwrap()
        .with_presence_penalty(1.5)
        .unwrap();

    // `position` runs 0..trials in `frequency`, so by the time most trials
    // run, `position` (the generated count) exceeds `history.len()` and the
    // whole of `history` counts as generated -- exercising the penalty on
    // every trial rather than none of them.
    let baseline = frequency(&unpenalized, &history, trials);
    let attenuated = frequency(&penalized, &history, trials);

    let baseline_share = *baseline.get(&0).unwrap_or(&0);
    let attenuated_share = *attenuated.get(&0).unwrap_or(&0);
    assert!(
        attenuated_share < baseline_share,
        "expected presence penalty to reduce candidate 0's share: baseline {baseline_share}, attenuated {attenuated_share}"
    );
}

#[test]
fn a_frequency_penalized_candidate_is_selected_materially_less_often() {
    let history = vec![0i32, 0, 0];
    let trials = 400;

    let unpenalized = ShapingConfig::new(1.0, 0, None, 1.0, Some(1)).unwrap();
    let penalized = ShapingConfig::new(1.0, 0, None, 1.0, Some(1))
        .unwrap()
        .with_frequency_penalty(0.5)
        .unwrap();

    let baseline = frequency(&unpenalized, &history, trials);
    let attenuated = frequency(&penalized, &history, trials);

    let baseline_share = *baseline.get(&0).unwrap_or(&0);
    let attenuated_share = *attenuated.get(&0).unwrap_or(&0);
    assert!(
        attenuated_share < baseline_share,
        "expected frequency penalty to reduce candidate 0's share: baseline {baseline_share}, attenuated {attenuated_share}"
    );
}

/// **THE HEADLINE CASE**: a token sitting in the PROMPT half of `history`
/// (before `position`, i.e. not yet "generated") must be untouched by
/// either penalty -- this is what distinguishes this port's llama.cpp-
/// compatible convention from OpenAI's own (which penalizes the whole
/// context, prompt included; see `DEVIATIONS.md`).
#[test]
fn a_prompt_only_candidate_is_not_penalized_at_all() {
    let scores = logits(&[2.0, 2.0, 2.0, 2.0]);
    let view = LogitsView::new(&scores);
    // Candidate 0 sits in `history`, but `position: 0` means ZERO tokens
    // have been generated -- the whole of `history` is prompt, so the
    // generated suffix `select` penalizes is empty.
    let history = vec![0i32, 0, 0];

    let mut baseline = HashMap::new();
    let mut with_penalty = HashMap::new();
    for trial_seed in 0..400u64 {
        let unpenalized_at_seed = ShapingConfig::new(1.0, 0, None, 1.0, Some(trial_seed)).unwrap();
        let penalized_at_seed = ShapingConfig::new(1.0, 0, None, 1.0, Some(trial_seed))
            .unwrap()
            .with_presence_penalty(2.0)
            .unwrap()
            .with_frequency_penalty(2.0)
            .unwrap();
        *baseline
            .entry(select(view, &unpenalized_at_seed, &history, 0).unwrap())
            .or_insert(0u32) += 1;
        *with_penalty
            .entry(select(view, &penalized_at_seed, &history, 0).unwrap())
            .or_insert(0u32) += 1;
    }

    assert_eq!(
        baseline, with_penalty,
        "a candidate that only appears in the PROMPT half of history must not be penalized"
    );
}

#[test]
fn zero_presence_and_frequency_penalty_is_bit_identical_to_the_default() {
    let history = vec![0i32, 1, 0];
    let scores = logits(&[1.0, 2.0, 0.5, 3.0]);
    let view = LogitsView::new(&scores);

    let default_config = ShapingConfig::new(0.8, 4, Some(0.95), 1.2, Some(7)).unwrap();
    let explicit_zero = ShapingConfig::new(0.8, 4, Some(0.95), 1.2, Some(7))
        .unwrap()
        .with_presence_penalty(0.0)
        .unwrap()
        .with_frequency_penalty(0.0)
        .unwrap();

    for position in 0..50u64 {
        assert_eq!(
            select(view, &default_config, &history, position).unwrap(),
            select(view, &explicit_zero, &history, position).unwrap(),
            "position {position}: explicit zero penalties must select identically to the default"
        );
    }
}
