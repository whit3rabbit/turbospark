//! `rank_top_k` must be indistinguishable from the prefix of a full sort,
//! and the `u32` scratch-buffer variants the hot path uses must be
//! indistinguishable from both.
//!
//! The partial ranking exists only because the full sort cost ~17 ms per
//! token at the real vocabulary size. A partial selection that disagreed
//! with the full order on ties, or on equal probabilities, would change
//! sampled output while every throughput number improved -- exactly the
//! class of bug the repo's greedy-only smoke tests cannot see.

use turbospark_selection::truncation::{
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

/// The sequential-cut path added 2026-08-15 compares raw `f64` keys with
/// `>=`, and every fixture above is non-negative because the hot path feeds
/// it `exp(s - max)`. `keys` is public and documented as "any monotone image
/// of the probabilities", so a negative, a zero and an infinity all have to
/// land in the same order the full sort puts them.
#[test]
fn mixed_sign_and_infinite_keys_agree_with_the_reference() {
    let keys = vec![
        0.0,
        -1.0,
        f64::INFINITY,
        -0.0,
        5.0,
        f64::NEG_INFINITY,
        -1.0,
        1e-300,
        -5.0,
        5.0,
        0.0,
        f64::INFINITY,
    ];
    let full: Vec<u32> = rank_indices(&keys).into_iter().map(|i| i as u32).collect();
    let mut out = Vec::new();
    for k in 0..=keys.len() + 1 {
        rank_top_k_u32_into(&keys, k, &mut out);
        let want: Vec<u32> = full.iter().copied().take(k.min(keys.len())).collect();
        assert_eq!(out, want, "k {k}");
    }
}

/// A NaN key used to make the comparator INTRANSITIVE, and Rust's sorts
/// detect that and abort the process: `partial_cmp(..).unwrap_or(Equal)`
/// reports NaN equal to every real key, so on `[1.0, NaN, 5.0]` the old rule
/// gave `0 < 1`, `1 < 2` and `0 > 2` at once. Small inputs merely returned a
/// scrambled answer; larger ones panicked with "user-provided comparison
/// function does not correctly implement a total order", which is why this
/// sat unnoticed in a public API. `select` never reached it -- it rejects a
/// non-finite score vector first.
///
/// NaN now ranks LAST, which is the choice `f64::total_cmp` would have got
/// wrong for this use: under IEEE totalOrder a positive NaN outranks
/// infinity, so a NaN score would be selected rather than discarded.
#[test]
fn nan_keys_rank_last_and_no_longer_panic() {
    // The exact intransitivity, now well-defined: 5.0, then 1.0, then NaN.
    assert_eq!(rank_indices(&[1.0, f64::NAN, 5.0]), vec![2, 0, 1]);

    // An input large enough that the old comparator aborted rather than
    // merely scrambling. 1366 NaNs of 4096, spread through the domain.
    let many: Vec<f64> = (0..4096)
        .map(|i| {
            if i % 3 == 0 {
                f64::NAN
            } else {
                ((i * 7) % 101) as f64
            }
        })
        .collect();
    let full = rank_indices(&many);
    assert_eq!(full.len(), many.len());

    // Every NaN sits after every real key, and the real prefix is exactly
    // the non-NaN count -- so "last" means last, not merely "somewhere".
    let first_nan = full
        .iter()
        .position(|&i| many[i].is_nan())
        .expect("the fixture carries NaNs");
    assert_eq!(first_nan, many.iter().filter(|v| !v.is_nan()).count());
    assert!(full[first_nan..].iter().all(|&i| many[i].is_nan()));

    // The partial paths agree with the full reference, which is only
    // possible because both comparators are now the same total order: a
    // stable and an unstable sort may disagree under a broken one.
    let want: Vec<u32> = full.iter().map(|&i| i as u32).collect();
    let mut out = Vec::new();
    rank_indices_u32_into(&many, &mut out);
    assert_eq!(out, want, "rank_indices_u32_into");
    for k in [1usize, 64, first_nan, first_nan + 1, many.len()] {
        rank_top_k_u32_into(&many, k, &mut out);
        assert_eq!(out, want[..k.min(many.len())].to_vec(), "k {k}");
    }
}

/// The cut admits every tie at the boundary, so when the boundary value is
/// heavily repeated the admitted set is far larger than `k` and the
/// truncation after the sort is what makes the answer right. A cut path
/// that admitted only `k` entries, or compared with `>` instead of `>=`,
/// would return too few here rather than the wrong ones -- which is the
/// failure a spot check on distinct values cannot see.
#[test]
fn a_heavily_tied_cut_admits_the_ties_and_truncates_after_ordering() {
    // 4 clear winners, then 200 exact ties straddling any k in 5..205.
    let mut keys = vec![9.0, 8.0, 7.0, 6.0];
    keys.extend(std::iter::repeat_n(1.0, 200));
    keys.push(0.5);
    let full: Vec<u32> = rank_indices(&keys).into_iter().map(|i| i as u32).collect();
    let mut out = Vec::new();
    for k in [4usize, 5, 6, 100, 203, 204, 205] {
        rank_top_k_u32_into(&keys, k, &mut out);
        assert_eq!(
            out.len(),
            k.min(keys.len()),
            "k {k} returned the wrong count"
        );
        assert_eq!(out, full[..k.min(keys.len())].to_vec(), "k {k}");
    }
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
/// `cargo test -p turbospark-selection --release --test rank_top_k -- --ignored --nocapture`
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
