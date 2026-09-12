//! Rejection-sampling prerequisites (ROADMAP P1 item 4): the distribution
//! [`shaped_distribution`] materializes is the one [`select`] samples from,
//! verified as a draw-for-draw identity rather than as an agreement of
//! summary statistics.
//!
//! The identity is the exactness claim's foundation: Leviathan/Chen
//! acceptance compares p(x)/q(x) over the SAME shaped distributions a
//! sequential decode samples from, so a `shaped_distribution` that ranked,
//! truncated or reweighted one candidate differently from `select` would
//! make every downstream "exact" statement approximate in exactly the
//! places nobody looks (the tails).

use foundation::{LogitValue, LogitsView, TokenId};
use turbospark_selection::{select, shaped_distribution, ShapingConfig};

/// A deterministic score vector with structure: one clear mode, a shoulder,
/// and a long tail, so truncation and reweight both have real work to do.
fn structured_scores() -> Vec<LogitValue> {
    (0..64u32)
        .map(|i| {
            let base = match i {
                7 => 6.0,
                19 => 4.5,
                20 => 4.2,
                33 => 3.0,
                _ => -1.0 - (i as f32) * 0.01,
            };
            LogitValue::from_f32(base)
        })
        .collect()
}

#[allow(clippy::excessive_precision)]
fn sampled_config(temperature: f64, top_k: u32, top_p: Option<f64>, seed: u64) -> ShapingConfig {
    ShapingConfig::new(temperature, top_k, top_p, 1.0, Some(seed)).expect("valid shaping")
}

/// THE IDENTITY: for a fixed seed and position, `select` returns exactly
/// the token `shaped_distribution(...).draw(seed, position)` returns --
/// same surviving set, same reweighted probabilities, same cumulative
/// walk. Swept across positions and several shaping combinations so a
/// divergence in any one truncation mode reddens.
#[test]
fn shaped_distribution_draws_what_select_draws() {
    let scores = structured_scores();
    let history: Vec<TokenId> = vec![3, 7, 11];
    let shapings = [
        sampled_config(0.2, 64, Some(0.95), 7),
        sampled_config(0.7, 0, None, 9),
        sampled_config(1.3, 8, Some(0.9), 11),
        sampled_config(0.5, 64, Some(0.95), 20260911),
    ];
    for config in shapings {
        for position in [0u64, 1, 2, 5, 17, 100] {
            let selected = select(LogitsView::new(&scores), &config, &history, position)
                .expect("select succeeds on a finite vector");
            let distribution =
                shaped_distribution(LogitsView::new(&scores), &config, &history, position)
                    .expect("shaped_distribution succeeds on the same vector");
            let drawn = distribution.draw(config.seed(), position);
            assert_eq!(
                selected,
                drawn as TokenId,
                "config T={} k={} p={:?} position {position}: select and the materialized \
                 distribution disagree",
                config.temperature(),
                config.top_k(),
                config.top_p()
            );
        }
    }
}

/// The materialized distribution is normalized over its support, and the
/// support is exactly the candidates `select` can ever return under that
/// shaping: sweeping many positions, every `select` result is inside the
/// support (the converse is not asserted -- some support entries can be
/// too light for any of the swept uniforms to land on).
#[test]
fn the_distribution_is_normalized_and_covers_selects_results() {
    let scores = structured_scores();
    let history: Vec<TokenId> = vec![3, 7, 11];
    let config = sampled_config(0.8, 16, Some(0.95), 4242);
    let distribution =
        shaped_distribution(LogitsView::new(&scores), &config, &history, 0).expect("builds");
    let total: f64 = distribution
        .support()
        .iter()
        .map(|&t| distribution.probability(t))
        .sum();
    assert!(
        (total - 1.0).abs() < 1e-9,
        "support probabilities sum to {total}, not 1"
    );
    for position in 0..256u64 {
        let selected =
            select(LogitsView::new(&scores), &config, &history, position).expect("select succeeds");
        assert!(
            distribution.support().contains(&(selected as u32)),
            "position {position}: select returned {} outside the support",
            selected
        );
    }
}

/// At the deterministic temperature the distribution is the point mass
/// `select`'s argmax arm returns -- penalties and truncation ignored, the
/// same exception the arm itself makes.
#[test]
fn deterministic_shaping_is_the_argmax_point_mass() {
    let scores = structured_scores();
    let history: Vec<TokenId> = vec![7, 7, 7]; // penalties must NOT move it
    let config = sampled_config(0.0, 1, None, 1);
    for position in [0u64, 3, 9] {
        let selected =
            select(LogitsView::new(&scores), &config, &history, position).expect("argmax arm");
        let distribution =
            shaped_distribution(LogitsView::new(&scores), &config, &history, position)
                .expect("point mass");
        assert_eq!(distribution.support(), &[selected as u32]);
        assert_eq!(distribution.probability(selected as u32), 1.0);
    }
}
