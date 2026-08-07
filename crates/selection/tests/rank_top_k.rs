//! `rank_top_k` must be indistinguishable from the prefix of a full sort,
//! and the `u32` scratch-buffer variants the hot path uses must be
//! indistinguishable from both.
//!
//! The partial ranking exists only because the full sort cost ~17 ms per
//! token at the real vocabulary size. A partial selection that disagreed
//! with the full order on ties, or on equal probabilities, would change
//! sampled output while every throughput number improved -- exactly the
//! class of bug the repo's greedy-only smoke tests cannot see.

use mrefrust_selection::truncation::{
    rank_indices, rank_indices_u32_into, rank_top_k, rank_top_k_u32_into,
};

/// Deterministic pseudorandom probabilities with deliberate exact ties,
/// which is where a tie-break difference would show up.
fn probs(len: usize) -> Vec<f64> {
    let mut state = 0x2545_f491_4f6c_dd1du64;
    (0..len)
        .map(|i| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            // Every 16th entry collapses onto one of four values, so the
            // ascending-index tie-break is exercised, not just assumed.
            if i % 16 == 0 {
                (i % 4) as f64
            } else {
                (state >> 11) as f64 / (1u64 << 53) as f64
            }
        })
        .collect()
}

#[test]
fn agrees_with_the_full_sort_prefix() {
    for len in [1usize, 2, 7, 64, 65, 1000, 4096] {
        let p = probs(len);
        let full = rank_indices(&p);
        for k in [0usize, 1, 2, 63, 64, 256, len, len + 1] {
            let want: Vec<usize> = full.iter().copied().take(k.min(len)).collect();
            assert_eq!(rank_top_k(&p, k), want, "len {len}, k {k}");
        }
    }
}

#[test]
fn u32_scratch_variants_agree_with_the_reference() {
    let mut out = Vec::new();
    for len in [1usize, 2, 7, 64, 65, 1000, 4096] {
        let p = probs(len);
        let full: Vec<u32> = rank_indices(&p).into_iter().map(|i| i as u32).collect();
        // Scratch is deliberately reused dirty across every call below.
        rank_indices_u32_into(&p, &mut out);
        assert_eq!(out, full, "rank_indices_u32_into, len {len}");
        for k in [0usize, 1, 2, 63, 64, 256, len, len + 1] {
            rank_top_k_u32_into(&p, k, &mut out);
            let want: Vec<u32> = full.iter().copied().take(k.min(len)).collect();
            assert_eq!(out, want, "rank_top_k_u32_into, len {len}, k {k}");
        }
    }
}

/// The hot path ranks unnormalized `exp(s - max)` values, so an all-tied
/// key vector (every key equal, e.g. a flat distribution) must come back
/// in ascending index order from every variant.
#[test]
fn all_equal_keys_rank_by_ascending_index() {
    let p = vec![0.25f64; 128];
    let want: Vec<usize> = (0..128).collect();
    assert_eq!(rank_indices(&p), want);
    assert_eq!(rank_top_k(&p, 64), want[..64].to_vec());
    let mut out = Vec::new();
    rank_top_k_u32_into(&p, 64, &mut out);
    let want32: Vec<u32> = (0..64).collect();
    assert_eq!(out, want32);
}

#[test]
fn keeps_no_more_than_the_domain() {
    let p = probs(5);
    assert_eq!(rank_top_k(&p, 100).len(), 5);
    assert!(rank_top_k(&p, 0).is_empty());
    assert!(rank_top_k(&[], 8).is_empty());
    let mut out = vec![7u32];
    rank_top_k_u32_into(&p, 0, &mut out);
    assert!(out.is_empty());
    rank_top_k_u32_into(&[], 8, &mut out);
    assert!(out.is_empty());
}

/// Timing, not correctness: prints the per-call cost at the real Gemma 4
/// vocabulary so the reason this function exists stays checkable.
/// `cargo test -p mrefrust-selection --release --test rank_top_k -- --ignored --nocapture`
#[test]
#[ignore = "timing aid, not an assertion; needs --release to mean anything"]
fn cost_at_the_real_vocabulary() {
    let p = probs(262_144);
    for (label, f) in [
        (
            "full sort   ",
            &(|p: &[f64]| rank_indices(p)[0]) as &dyn Fn(&[f64]) -> usize,
        ),
        (
            "top-k 64    ",
            &(|p: &[f64]| rank_top_k(p, 64)[0]) as &dyn Fn(&[f64]) -> usize,
        ),
    ] {
        let started = std::time::Instant::now();
        let mut sink = 0usize;
        for _ in 0..10 {
            sink += f(&p);
        }
        println!(
            "{label}: {:>7.2} ms/call (sink {sink})",
            started.elapsed().as_secs_f64() * 1e3 / 10.0
        );
    }
    // The scratch variant, reusing one buffer the way `select` does.
    let mut out = Vec::new();
    let started = std::time::Instant::now();
    let mut sink = 0u32;
    for _ in 0..10 {
        rank_top_k_u32_into(&p, 64, &mut out);
        sink += out[0];
    }
    println!(
        "top-k 64 u32: {:>7.2} ms/call (sink {sink})",
        started.elapsed().as_secs_f64() * 1e3 / 10.0
    );
}
