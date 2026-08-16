//! What is left in the host sampler after the 2026-08-07 partial-ranking fix,
//! split by part, at the real Gemma 4 vocabulary.
//!
//! `select` runs once per decoded token OUTSIDE `produce`, so no profiling
//! surface in this repo can see it (AGENTS.md Gotcha 23) and the only way to
//! know what it costs is to time it here. The 2026-08-07 session left an
//! estimate -- "~2.6 ms/token, softmax over 262144 f64 plus the index Vec" --
//! and never split it, so the obvious next move (narrowing the exp pass to
//! f32) has never had a number attached to what it would buy.
//!
//! It answered the opposite of what was assumed: the exp pass is 23% and the
//! partial RANKING was 70%, so the lever was the ranking's access pattern and
//! not the exp's precision. `truncation.rs`'s sequential cut followed.
//!
//! Everything here is TIMING except
//! `the_rankings_sensitivity_to_exp_precision_is_measured_not_assumed`, which
//! is an assertion and is the one that settles the f32 question: a cheaper exp
//! is only worth discussing if what it moves is bits nobody reads.
//!
//! `cargo test -p turbospark-selection --release --test host_sampler_cost -- --ignored --nocapture`

use foundation::{LogitValue, LogitsView, TokenId};
use turbospark_selection::truncation::rank_top_k_u32_into;
use turbospark_selection::{select, ShapingConfig};

const VOCAB: usize = 262_144;
const TOP_K: usize = 64;

/// Deterministic logits shaped like a real post-softcap head: bounded well
/// inside Gemma 4's +/-30 softcap, most of the domain flat and low, a sparse
/// set of plausible continuations carrying the mass. A uniform random vector
/// would make both the exp pass and the ranking's tie structure
/// unrepresentative of what the sampler actually sees.
fn logits(len: usize) -> Vec<LogitValue> {
    let mut state = 0x2545_f491_4f6c_dd1du64;
    (0..len)
        .map(|i| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let u = (state >> 11) as f32 / (1u64 << 53) as f32;
            let v = if i % 1301 == 0 {
                4.0 + u * 8.0
            } else {
                -12.0 + u * 6.0
            };
            LogitValue::from_f32(v)
        })
        .collect()
}

fn widened(raw: &[LogitValue]) -> Vec<f32> {
    raw.iter().map(|v| v.to_f32()).collect()
}

fn time<T>(label: &str, iters: usize, mut f: impl FnMut() -> T) -> f64 {
    // One discarded warmup, per the repo's standing measurement discipline.
    std::hint::black_box(f());
    let started = std::time::Instant::now();
    for _ in 0..iters {
        std::hint::black_box(f());
    }
    let ms = started.elapsed().as_secs_f64() * 1e3 / iters as f64;
    println!("{label:<32}{ms:>8.3} ms/call");
    ms
}

/// The split. Read the parts against the whole: if they do not add up, the
/// missing time is somewhere nobody is looking, which is precisely the
/// mistake that hid the full sort for the life of the port.
#[test]
#[ignore = "timing aid, not an assertion; needs --release to mean anything"]
fn where_the_host_sampler_time_goes() {
    let raw = logits(VOCAB);
    // The CLI's sampled defaults, which is what the frozen sampled digests
    // and the real-model sampled smoke both run at.
    let config = ShapingConfig::new(0.2, TOP_K as u32, Some(0.95), 1.0, Some(1))
        .expect("the CLI's own sampled defaults must be a valid configuration");
    let history: Vec<TokenId> = (0..64).collect();

    println!("\n-- host sampler, V={VOCAB}, T=0.2 top-k {TOP_K} top-p 0.95 --");

    let whole = time("select (whole)", 20, || {
        select(LogitsView::new(&raw), &config, &history, 7).expect("valid inputs")
    });

    // Part 1: the f16 -> f32 widen plus the finiteness check.
    let widen = time("  widen f16->f32 + finite", 20, || {
        let mut working = Vec::with_capacity(raw.len());
        let mut all_finite = true;
        for v in &raw {
            let f = v.to_f32();
            all_finite &= f.is_finite();
            working.push(f);
        }
        (working, all_finite)
    });

    let working = widened(&raw);
    let max = working.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
    let max32 = max as f32;

    // Part 2: the exp pass, as shipped.
    let exp64 = time("  exp pass f64 (as shipped)", 20, || {
        let mut exps: Vec<f64> = Vec::with_capacity(working.len());
        for &w in working.iter() {
            exps.push((w as f64 - max).exp());
        }
        let mut sum = 0.0f64;
        for &e in exps.iter() {
            sum += e;
        }
        (exps, sum)
    });

    // Part 3: the same pass at f32, which is the candidate change.
    let exp32 = time("  exp pass f32 (candidate)", 20, || {
        let mut exps: Vec<f32> = Vec::with_capacity(working.len());
        for &w in working.iter() {
            exps.push((w - max32).exp());
        }
        let mut sum = 0.0f32;
        for &e in exps.iter() {
            sum += e;
        }
        (exps, sum)
    });

    // Part 4: the partial ranking, already optimized. Here for the
    // subtraction, not because anything is expected of it.
    let exps: Vec<f64> = working.iter().map(|&w| (w as f64 - max).exp()).collect();
    let mut ranked = Vec::new();
    let rank = time("  rank_top_k_u32_into", 20, || {
        rank_top_k_u32_into(&exps, TOP_K, &mut ranked);
        ranked[0]
    });

    let parts = widen + exp64 + rank;
    println!(
        "\n  parts {parts:.3} of whole {whole:.3} ms, residual {:.3} ms ({:.1}%)",
        whole - parts,
        (whole - parts) / whole * 100.0
    );
    println!(
        "  f32 exp would save {:.3} ms/call, {:.1}% of the sampler",
        exp64 - exp32,
        (exp64 - exp32) / whole * 100.0
    );
    println!(
        "  ...which against a ~25 ms token is {:.2}%\n",
        (exp64 - exp32) / 25.0 * 100.0
    );
}

/// `rank_top_k_u32_into` exactly as it stood before the 2026-08-15
/// sequential-cut path, so the two can be timed against each other in ONE
/// process with the arms interleaved. Absolute timings on this machine drift
/// ~25% with desktop load (AGENTS.md Gotcha 43); a paired ratio does not,
/// which is why this arm exists rather than a comparison against a number
/// written down in an earlier session.
fn pre_cut_path(keys: &[f64], k: usize, out: &mut Vec<u32>) {
    let k = k.min(keys.len());
    if k == 0 {
        out.clear();
        return;
    }
    let cmp = |&a: &u32, &b: &u32| {
        keys[b as usize]
            .partial_cmp(&keys[a as usize])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    };
    out.clear();
    out.extend(0..keys.len() as u32);
    if k < out.len() {
        out.select_nth_unstable_by(k - 1, cmp);
        out.truncate(k);
    }
    out.sort_unstable_by(cmp);
}

/// The paired ratio, arms alternating WITHIN each round rather than as two
/// consecutive batches, which is the repo's standing A/B discipline because
/// consecutive batches carry drift. Also asserts the two agree, so a ratio
/// can never be reported for arms computing different answers.
#[test]
#[ignore = "timing aid, not an assertion; needs --release to mean anything"]
fn the_cut_path_against_the_partition_it_replaced() {
    let raw = logits(VOCAB);
    let working = widened(&raw);
    let max = working.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
    let exps: Vec<f64> = working.iter().map(|&w| (w as f64 - max).exp()).collect();

    let mut old_out = Vec::new();
    let mut new_out = Vec::new();
    pre_cut_path(&exps, TOP_K, &mut old_out);
    rank_top_k_u32_into(&exps, TOP_K, &mut new_out);
    assert_eq!(
        old_out, new_out,
        "the two ranking paths disagree, so any ratio below is meaningless"
    );

    println!("\n-- top-{TOP_K} of {VOCAB}, interleaved pairs --");
    for round in 1..=3 {
        let t0 = std::time::Instant::now();
        for _ in 0..10 {
            pre_cut_path(&exps, TOP_K, &mut old_out);
            std::hint::black_box(&old_out);
        }
        let old = t0.elapsed().as_secs_f64() * 1e3 / 10.0;

        let t1 = std::time::Instant::now();
        for _ in 0..10 {
            rank_top_k_u32_into(&exps, TOP_K, &mut new_out);
            std::hint::black_box(&new_out);
        }
        let new = t1.elapsed().as_secs_f64() * 1e3 / 10.0;

        println!(
            "  round {round}: partition {old:>6.3} ms, cut {new:>6.3} ms  ({:.2}x)",
            old / new
        );
    }
    println!();
}

/// Rounds `x` to `bits` significant mantissa bits, so the ranking's
/// sensitivity to exp PRECISION can be measured as a number rather than
/// argued about at two fixed widths.
fn round_to_bits(x: f64, bits: i32) -> f64 {
    if x == 0.0 || !x.is_finite() {
        return x;
    }
    let e = x.abs().log2().floor();
    let scale = 2f64.powi(bits - 1 - e as i32);
    (x * scale).round() / scale
}

/// THE DECIDING FACT, and it is the opposite of what the roadmap assumed.
///
/// The plan costed an f32 exp pass as "re-freeze the sampled digests",
/// taking for granted that a narrower exp reorders the surviving set --
/// the exps ARE `rank_top_k_u32_into`'s sort key, so that is the obvious
/// reading. It does not, and the reason is upstream of the sampler
/// entirely: `LogitValue` is f16. Two logits that differ at all differ by
/// at least one f16 ULP (~0.008 at magnitude 8), `exp` is monotone, and
/// that gap survives at any float width down to far fewer bits than f32
/// carries. Equal f16 logits produce exactly equal exps at BOTH widths and
/// fall to the same ascending-index tie-break.
///
/// So this measures the BREAKING POINT rather than asserting either
/// answer, which is also what keeps it from going vacuous: a fixture whose
/// top-k were so widely separated that nothing could reorder it would
/// report a breaking point of 1 bit and be visibly useless.
#[test]
fn the_rankings_sensitivity_to_exp_precision_is_measured_not_assumed() {
    let raw = logits(VOCAB);
    let working = widened(&raw);
    let max = working.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
    let max32 = max as f32;

    let exps64: Vec<f64> = working.iter().map(|&w| (w as f64 - max).exp()).collect();
    // Widened back to f64 so the two rankings differ by the EXP's precision
    // alone and not by the comparator's element width.
    let exps32: Vec<f64> = working.iter().map(|&w| (w - max32).exp() as f64).collect();

    let mut want = Vec::new();
    let mut got = Vec::new();
    rank_top_k_u32_into(&exps64, TOP_K, &mut want);
    rank_top_k_u32_into(&exps32, TOP_K, &mut got);
    assert_eq!(
        want, got,
        "an f32 exp pass reordered the top-{TOP_K}. The f16 input alphabet is \
         supposed to make that impossible, so either the fixture is no longer \
         f16-derived or `rank_top_k_u32_into`'s tie-break has moved."
    );

    // Where DOES it break? Walk the exps down in precision until the order
    // moves. f32 carries 24 mantissa bits and f64 53, so the answer says
    // how much headroom the invariance above actually has.
    let mut breaking = 0;
    for bits in (2..=52).rev() {
        let rounded: Vec<f64> = exps64.iter().map(|&e| round_to_bits(e, bits)).collect();
        rank_top_k_u32_into(&rounded, TOP_K, &mut got);
        if got != want {
            breaking = bits;
            break;
        }
    }
    println!(
        "top-{TOP_K} order is unchanged by an f32 exp pass; it first moves at \
         {breaking} mantissa bits (f32 has 24, f64 53)"
    );
    assert!(
        breaking > 1,
        "the ranking survived rounding to 2 mantissa bits, so this fixture \
         cannot detect a reordering at any precision and proves nothing"
    );
    assert!(
        breaking < 24,
        "the ranking moves at {breaking} bits, at or above f32's 24 -- the \
         invariance asserted above is then luck rather than structure"
    );
}
