//! Nucleus sampling with top-k off ranks a short prefix before it sorts the
//! whole vocabulary. The surviving set must equal the full-sort reference on a
//! narrow nucleus, on one wider than the prefix, and on a flat tie.

use foundation::{LogitValue, LogitsView};
use turbospark_selection::truncation::{rank_indices, softmax, truncate_by_probability_mass};
use turbospark_selection::{shaped_distribution, ShapingConfig};

fn check(scores: &[f32], top_p: f64) {
    let raw: Vec<LogitValue> = scores.iter().map(|&s| LogitValue::from_f32(s)).collect();
    let rounded: Vec<f32> = raw.iter().map(|v| v.to_f32()).collect();
    let probs = softmax(&rounded);
    let want = truncate_by_probability_mass(&rank_indices(&probs), &probs, Some(top_p));
    let config = ShapingConfig::new(1.0, 0, Some(top_p), 1.0, Some(1)).unwrap();
    let got = shaped_distribution(LogitsView::new(&raw), &config, &[], 0).unwrap();
    let got: Vec<usize> = got.support().iter().map(|&i| i as usize).collect();
    assert_eq!(got, want, "top_p {top_p}");
}

#[test]
fn prefix_ranking_matches_the_full_sort() {
    let n = 5000usize;
    let peaked: Vec<f32> = (0..n)
        .map(|i| {
            if i % 97 == 3 {
                12.0 - (i % 13) as f32 * 0.25
            } else {
                0.0
            }
        })
        .collect();
    let spread: Vec<f32> = (0..n)
        .map(|i| ((i * 7919) % 4001) as f32 / 1000.0)
        .collect();
    let flat = vec![1.0f32; n];
    for scores in [&peaked, &spread, &flat] {
        for top_p in [0.01, 0.5, 0.9, 0.99, 1.0] {
            check(scores, top_p);
        }
    }
}
