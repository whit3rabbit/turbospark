//! Drift guard: the live pipeline in `choose.rs:shape_candidates` inlines
//! softmax, top-p, top-k and min-p, while `truncation.rs` keeps the reference
//! helpers (`softmax`, `rank_indices`, `truncate_by_*`) that `select` never
//! calls. This test composes the reference helpers in the documented order and
//! requires the live support (observed through `shaped_distribution`) to match
//! exactly, in rank order, so a change to one side alone fails here.
//!
//! Penalties are left at their identity values: they run before the softmax
//! and are covered by `penalty.rs` and `presence_frequency.rs`.

use foundation::{LogitValue, LogitsView};
use turbospark_selection::shaped_distribution;
use turbospark_selection::shaping::ShapingConfig;
use turbospark_selection::truncation::{
    rank_indices, softmax, truncate_by_min_p, truncate_by_probability_mass, truncate_by_rank,
};

/// Reference composition: softmax, full rank, top-p, top-k cap, min-p.
fn reference_support(
    scores: &[f32],
    top_k: u32,
    top_p: Option<f64>,
    min_p: Option<f64>,
) -> Vec<u32> {
    // `select` receives half-precision logits, so the reference must rank the
    // same rounded values or near-ties would differ for reasons unrelated to
    // drift.
    let scores: Vec<f32> = scores
        .iter()
        .map(|&v| LogitValue::from_f32(v).to_f32())
        .collect();
    let probs = softmax(&scores);
    let ranked = rank_indices(&probs);
    let mass = truncate_by_probability_mass(&ranked, &probs, top_p);
    let capped = truncate_by_rank(&mass, top_k);
    truncate_by_min_p(&capped, &ranked, &probs, min_p)
        .into_iter()
        .map(|i| i as u32)
        .collect()
}

fn live_support(scores: &[f32], top_k: u32, top_p: Option<f64>, min_p: Option<f64>) -> Vec<u32> {
    let logits: Vec<LogitValue> = scores.iter().map(|&v| LogitValue::from_f32(v)).collect();
    let mut config = ShapingConfig::new(1.0, top_k, top_p, 1.0, Some(1)).unwrap();
    if let Some(m) = min_p {
        config = config.with_min_p(m).unwrap();
    }
    shaped_distribution(LogitsView::new(&logits), &config, &[], 0)
        .unwrap()
        .support()
        .to_vec()
}

/// Deterministic xorshift, so the synthetic distributions are reproducible.
fn synthetic(seed: u64, n: usize, scale: f32, quantum: Option<f32>) -> Vec<f32> {
    let mut s = seed.max(1);
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let v = ((s >> 11) as f64 / (1u64 << 53) as f64) as f32 * scale - scale / 2.0;
            // A coarse grid forces exact ties, which exercise the index tie-break.
            quantum.map_or(v, |q| (v / q).round() * q)
        })
        .collect()
}

#[test]
fn live_path_matches_reference_helpers() {
    let mut cases: Vec<Vec<f32>> = vec![
        vec![10.0, 9.5, 0.0, -1.0, -2.0],
        vec![1.0, 1.0, 1.0, 1.0],
        vec![5.0],
        vec![3.0, 3.0, 2.0, 2.0, 1.0, 1.0],
    ];
    for (seed, n, scale, q) in [
        (1u64, 50usize, 8.0f32, None),
        (2, 300, 20.0, None),
        (3, 300, 6.0, Some(0.5)),
        (4, 1000, 30.0, None),
        (5, 64, 1.0, Some(0.25)),
    ] {
        cases.push(synthetic(seed, n, scale, q));
    }

    let top_ks = [0u32, 1, 3, 40, 256];
    let top_ps = [None, Some(0.3), Some(0.9), Some(1.0)];
    let min_ps = [None, Some(0.05), Some(0.5)];

    for (ci, scores) in cases.iter().enumerate() {
        for &k in &top_ks {
            for &p in &top_ps {
                for &m in &min_ps {
                    assert_eq!(
                        live_support(scores, k, p, m),
                        reference_support(scores, k, p, m),
                        "case {ci}: top_k={k} top_p={p:?} min_p={m:?}"
                    );
                }
            }
        }
    }
}
