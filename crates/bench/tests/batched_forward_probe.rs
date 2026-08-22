#![cfg(target_os = "macos")]
//! Where does the batched verify stop agreeing with a sequential decode?
//!
//! A greedy speculative stream parts from a greedy sequential one at ~154
//! tokens on the real `qwen3_5` install (`docs/DFLASH2.md`). The cause was
//! attributed for a while to `dequant_int4_gemm_simd` accumulating
//! differently from `dequant_int4_gemv_simd`; that was read off the two
//! shaders rather than measured, and it is refuted -- the pair agrees
//! bit-for-bit on data proven able to see a reassociation, against a
//! positive control that does differ
//! (`crates/gpu/tests/dequant_int4_gemm_parity.rs`). The `gdn.metal`
//! multi-row kernels are exact too.
//!
//! **SO THE DIFFERENCE IS SOMEWHERE IN `produce_batched` AS A WHOLE, AND
//! THIS SPLITS THAT IN TWO.** The block-size evidence cannot: every
//! speculative arm calls `produce_batched` and the sequential arm calls
//! `produce`, so identical text across blocks 2/4/7/8 is consistent with any
//! difference between the two functions.
//!
//!   - **M=1.** A one-token "batch" does no batching at all: one row, one
//!     query, one KV write, the same causal span. A difference here is a
//!     COMPOSITION difference between the two functions -- a kernel reached
//!     by one and not the other, a norm reading a different row, a residual
//!     laid out differently.
//!   - **M=2.** If M=1 agrees and M=2 does not, the cause is the batching
//!     itself, and the first place to look is the attention span each row
//!     sees: `produce_batched` writes all M KV rows before any query reads,
//!     and row m is supposed to look at `[0, start + m]` and no further.
//!
//! It reports rather than asserts -- an instrument, not a gate. Every arm
//! compares BIT-IDENTICAL logits rather than text: a last-bit difference is
//! invisible in the output until it lands on a near-tie, which is the whole
//! reason the divergence took 154 tokens to show up.
//!
//! **MEASURED 2026-08-21 ON `qwen38-27b-mtp`, and it is M=1:**
//!
//!   M=1, position 22    219001/248320 logits differ, worst 2.1e-2, argmax agrees
//!   M=1, position 0          0/248320 differ, worst 0
//!   M=2, position 22    219001 and 222737/248320 differ
//!
//! So the batch WIDTH is not the variable and never was -- M=1 already
//! differs by as much as M=2, on a call that does no batching. The magnitude
//! rules out a reassociation: 88% of a vocabulary moving by up to 2e-2 is a
//! systematic numerical difference, not a last-bit one. That the argmax
//! still agrees is exactly why the streams track for 154 tokens first.
//!
//! Position 0 is what narrows it. Bit-identical there exonerates everything
//! observable at position 0 -- the embedding, every norm, the linear/GDN
//! layers, the FFN, the head, every GEMM. It is blind to two things only:
//! the q/k path, since a softmax over one key returns V whatever the score,
//! and carried history, since there is none. History is separately excluded
//! (`gdn_parity`'s carried-state case).
//!
//! **THE ONSET IS A STEP AND NOT A DRIFT, which narrows it further and rules
//! out the obvious remaining suspect.** One row at each position, history
//! built by `produce` in BOTH arms so only the row under test differs:
//!
//!   keys  1     0/248320 differ
//!   keys  2     0/248320 differ
//!   keys  3  224516/248320 differ, worst 2.7e-2
//!   keys  4  188467/248320 differ
//!   keys  5  149951/248320 differ
//!   keys  6  155934/248320 differ
//!
//! Two keys is BIT-IDENTICAL, and a two-key softmax is not degenerate -- it
//! exposes the score in full -- so q, k and V are all exact at that point,
//! and so is everything upstream of them. Then three keys differs at full
//! magnitude. Nothing accumulates gradually; it switches on.
//!
//! And it is NOT the split-KV combine, which is the natural guess for a
//! threshold in the key count: `MIN_POSITIONS_PER_CHUNK` is 16, so
//! `chunks_for` returns 1 for every span in this table and the cross-chunk
//! max and denominator pass does not run at any of them. Both paths also
//! dispatch the same `encode_attention_decode` with argument-for-argument
//! identical spans, the same per-head q/k norm at the same epsilon, and the
//! same `rope_neox_subdim` at the same theta and rotary dim.
//!
//! It is also NOT stale scratch: the same batched call twice from one
//! checkpoint gives 0/248320 differing, so this is arithmetic rather than a
//! buffer carrying state across calls (Gotcha 27's shape, checked because a
//! comparison against `produce` alone cannot tell the two apart).
//!
//! **AND IT IS NOT A DEFECT. Measured in NATS it is this port's own shape
//! floor, which is where the whole hunt should have started.** The arms above
//! read 6.2e-8 to 1.5e-5 nats with the argmax agreeing on every row, against
//! a dense batched-vs-cached shape floor of 7.4e-6 measured on MLX for this
//! same architecture -- and against 1.57e-5, this repo's own cross-engine
//! result for the family, published as "no detectable kernel gap"
//! (`crates/bench/CLAUDE.md` Gotcha 8). Every engine's batched and cached
//! passes disagree by about this much; that disagreement is what `kld.py`
//! measures as the shape floor by running the REFERENCE twice.
//!
//! **THE COUNT AND THE MAX DELTA ARE THE WRONG UNITS, and reading them as the
//! magnitude is what made this look like a bug worth bisecting.** "88% of the
//! vocabulary differs, worst 2e-2" sounds enormous and describes a
//! distributional difference of 1e-5 nats. The tell was available the whole
//! time and was not used: the argmax never moved. A greedy stream parting at
//! ~154 tokens is then exactly what it should do -- it tracks until the first
//! near-tie and then falls the other way, which is ordinary FP behaviour a
//! batched verify has in every engine.
//!
//! So there is nothing here to fix and no per-layer bisect to run. What the
//! eliminations below are still worth is bounding the floor's CAUSE: it is
//! not the INT4 GEMM (bit-exact at fixture and at real shapes, against a
//! positive control that differs), not the GDN multi-row kernels (bit-exact
//! from zero, carried and filling states), not the split-KV combine
//! (`chunks_for` is 1 until span 32), and not stale scratch. It appears at
//! three keys and not two, so it is the attention reduction, which is the
//! only thing left that a key count reaches.
//!
//! One thing a future reader should NOT spend time on: a synthetic install
//! cannot see any of this -- `real_forward_qwen35_batched_onset.rs` sweeps a
//! dense INT4 fixture and reads 0 differing at every span.
//!
//! ```sh
//! TURBOSPARK_PROBE_INSTALL_DIR=~/models/qwen38-27b-mtp.gturbo \
//!   cargo test -p turbospark-bench --test batched_forward_probe --release -- --ignored --nocapture
//! ```
//!
//! Needs an install `produce_batched` accepts: DENSE and INT4. It refuses a
//! MoE or sub-4-bit one by name, which this reports rather than swallowing.

use foundation::LogitValue;
use runtime::LogitProducer;
use tokenizer::{Message, MfTokenizer, Role};
use turbospark_bench::real_model::open_model_runner_speculative;

fn ids_for(tokenizer: &MfTokenizer, text: &str) -> Vec<i32> {
    let rendered = tokenizer
        .apply_chat_template(&[Message::new(Role::User, text)])
        .expect("chat template renders");
    tokenizer.encode(&rendered, false)
}

fn bits(logits: &[LogitValue]) -> Vec<u16> {
    logits.iter().map(|v| v.to_bits()).collect()
}

/// KL(p || q) in NATS over the two rows' softmaxes.
///
/// **THIS IS THE UNIT THAT MAKES THE GAP READABLE, and the count of differing
/// logits is not.** Every engine's batched and cached passes disagree -- that
/// is what `scripts/kld.py` calls the SHAPE FLOOR, measured by running the
/// REFERENCE twice, once token-by-token through a cache and once as one
/// batched pass, holding weights and kernels fixed. In mlx-lm on Gemma it
/// costs 0.0352 nats and 4% of the argmaxes. So "the batched pass differs
/// from the cached one" is not on its own a defect, and a raw count of moved
/// logits cannot say whether this port's gap is ordinary or anomalous.
///
/// The floors to read this against are per SHAPE, not per port
/// (`crates/bench/CLAUDE.md` Gotcha 8). A dense model's two passes are nearly
/// the same computation, so its floor nearly vanishes:
///
///   dense `qwen35` 9B, llama.cpp batched vs cached   0.0000024   99.65%
///   dense `qwen3_5` 27B, mlx batched vs cached       0.0000074  100.00%
///   MoE `qwen3moe`, llama.cpp                        0.00135     99.1%
///   MoE ornith-35B, mlx                              0.02780     92.84%
///
/// The install this probe runs on is DENSE `qwen3_5`, so the ~1e-6 rows are
/// its comparison. Computed in f64 off an f16 input, which is exact.
fn kl_nats(p: &[LogitValue], q: &[LogitValue]) -> f64 {
    let softmax = |v: &[LogitValue]| {
        let m = v
            .iter()
            .map(|x| x.to_f32() as f64)
            .fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = v.iter().map(|x| ((x.to_f32() as f64) - m).exp()).collect();
        let sum: f64 = exps.iter().sum();
        exps.into_iter().map(|e| e / sum).collect::<Vec<f64>>()
    };
    let (p, q) = (softmax(p), softmax(q));
    p.iter()
        .zip(q.iter())
        .filter(|(pi, _)| **pi > 0.0)
        .map(|(pi, qi)| pi * (pi / qi.max(f64::MIN_POSITIVE)).ln())
        .sum()
}

/// How far apart two logit rows are, as a count and as the worst absolute
/// gap. The count alone would not distinguish a last-bit difference from a
/// broken pass, and only the first of those is what this is hunting.
fn compare(label: &str, a: &[u16], b: &[u16], af: &[LogitValue], bf: &[LogitValue]) -> usize {
    let differing = a.iter().zip(b.iter()).filter(|(x, y)| x != y).count();
    let worst = af
        .iter()
        .zip(bf.iter())
        .map(|(x, y)| (x.to_f32() - y.to_f32()).abs())
        .fold(0.0f32, f32::max);
    let argmax = |v: &[LogitValue]| {
        v.iter()
            .enumerate()
            .max_by(|x, y| x.1.to_f32().total_cmp(&y.1.to_f32()))
            .map(|(i, _)| i)
            .unwrap_or(0)
    };
    println!(
        "  {label}: {differing}/{} logits differ, worst |delta| {worst:e}, KL {:.3e} nats, \
         argmax {} vs {}",
        a.len(),
        kl_nats(af, bf),
        argmax(af),
        argmax(bf)
    );
    differing
}

#[test]
#[ignore = "needs a real dense INT4 install via TURBOSPARK_PROBE_INSTALL_DIR"]
fn a_batched_forward_at_one_row_is_compared_against_a_sequential_step() {
    let dir = std::path::PathBuf::from(
        std::env::var_os("TURBOSPARK_PROBE_INSTALL_DIR").expect("TURBOSPARK_PROBE_INSTALL_DIR"),
    );
    // THE M-ROW SCRATCH IS ALLOCATED ONLY WHEN A DRAFTER IS ASKED FOR at
    // open, so this has to go through the speculative door even though it
    // never drafts a token: every other opener in this crate pins drafting
    // OFF, and `produce_batched` refuses by name without the scratch. The
    // block is 2 because that is what ships (`DFLASH_SERVING_BLOCK`); the
    // probe drives `produce_batched` itself and never runs a round.
    let (mut runner, tokenizer) = open_model_runner_speculative(
        &dir,
        16,
        runtime::DraftPolicies::mtp(runtime::MtpDraftPolicy::Fixed(2)),
    )
    .expect("install opens");
    let vocab = tokenizer.vocab_size;

    let prompt = ids_for(
        &tokenizer,
        "Explain how coastal wetlands reduce flood damage.",
    );
    // Two tokens that are ordinary continuations rather than anything
    // special, since what is being compared is arithmetic and not routing.
    let follow: Vec<i32> = ids_for(&tokenizer, "Coastal wetlands absorb storm surge.")
        .into_iter()
        .take(2)
        .collect();
    assert_eq!(follow.len(), 2, "need two continuation tokens");
    let base = prompt.len();
    println!(
        "batched_forward_probe: {base} prompt tokens, vocab {vocab}, install {}",
        dir.display()
    );

    // Prime once and checkpoint, so every arm below starts from the SAME
    // state without paying the prompt again.
    runner.reset();
    let mut scratch = vec![LogitValue::from_f32(0.0); vocab];
    for (i, &token) in prompt.iter().enumerate() {
        runner.produce(token, i, &mut scratch).expect("produce");
    }
    let point = runner.checkpoint();

    // --- M=1: is a one-row batch the same computation as a decode step? ---
    runner
        .produce(follow[0], base, &mut scratch)
        .expect("produce");
    let seq_1 = scratch.clone();
    runner.rollback(&point);

    let mut batched = vec![LogitValue::from_f32(0.0); vocab];
    match runner.produce_batched(&follow[..1], base, &mut batched) {
        Ok(()) => {}
        Err(e) => {
            println!("produce_batched refused this install: {e}");
            println!("point TURBOSPARK_PROBE_INSTALL_DIR at a DENSE INT4 install");
            return;
        }
    }
    runner.rollback(&point);

    println!("M=1 (no batching happens at all):");
    let m1 = compare("row 0", &bits(&seq_1), &bits(&batched), &seq_1, &batched);

    // --- M=1 AT POSITION 0: the same comparison, minus two things ---
    //
    // At position 0 a causal softmax is over ONE key and is exactly 1.0, so
    // attention returns V unchanged whatever the q/k path does (the same
    // blindness AGENTS.md Gotcha 51 records for a digest taken there -- used
    // deliberately here rather than tripped over), and there is no
    // accumulated history for the recurrent half to carry.
    //
    // **SO AN AGREEMENT HERE EXONERATES EVERYTHING ELSE AND SUSPECTS EXACTLY
    // TWO THINGS**, which is the whole reason to take the reading: the
    // embedding, every norm, the linear/GDN layers, the FFN, the head and
    // every GEMM are all observable at position 0 and all exact if this is
    // zero. What is NOT observable is the q/k path and carried history.
    // History is separately excluded --
    // `crates/gpu/tests/gdn_parity.rs::one_prefill_row_matches_one_decode_step_from_a_state_that_carries_history`
    // reads 0 of 128 outputs and 0 of 4096 state elements differing -- which
    // leaves the q/k path and the multi-key reduction that consumes it.
    runner.reset();
    let mut zero_seq = vec![LogitValue::from_f32(0.0); vocab];
    runner
        .produce(prompt[0], 0, &mut zero_seq)
        .expect("produce at 0");
    runner.reset();
    let mut zero_batched = vec![LogitValue::from_f32(0.0); vocab];
    runner
        .produce_batched(&prompt[..1], 0, &mut zero_batched)
        .expect("batched at 0");
    println!("M=1 at position 0 (softmax over one key: attention is the identity on V):");
    let m1_pos0 = compare(
        "row 0",
        &bits(&zero_seq),
        &bits(&zero_batched),
        &zero_seq,
        &zero_batched,
    );
    // --- THE SHAPE OF THE ONSET: positions 0..5, one row each ---
    //
    // Position 0 agreeing and position 22 differing leaves two shapes, and
    // they point at different components. If the difference appears in FULL
    // the moment there are two keys, it is the multi-key reduction -- the
    // split-KV combine has a cross-chunk max and denominator that a
    // single-key softmax skips entirely. If it GROWS with history instead,
    // something is accumulating and the per-step difference is small.
    println!("onset (M=1, one row at each position, fresh each time):");
    for p in 0..6usize {
        runner.reset();
        let mut a = vec![LogitValue::from_f32(0.0); vocab];
        for (i, &token) in prompt[..=p].iter().enumerate() {
            runner.produce(token, i, &mut a).expect("produce");
        }
        runner.reset();
        let mut b = vec![LogitValue::from_f32(0.0); vocab];
        for (i, &token) in prompt[..p].iter().enumerate() {
            runner.produce(token, i, &mut b).expect("produce");
        }
        runner
            .produce_batched(&prompt[p..=p], p, &mut b)
            .expect("batched");
        compare(
            &format!("position {p} ({} keys)", p + 1),
            &bits(&a),
            &bits(&b),
            &a,
            &b,
        );
    }

    runner.reset();
    for (i, &token) in prompt.iter().enumerate() {
        runner.produce(token, i, &mut scratch).expect("produce");
    }
    let point = runner.checkpoint();

    // --- IS `produce_batched` EVEN DETERMINISTIC? ---
    //
    // Every arm above compares it against `produce`, which cannot tell an
    // arithmetic difference from a STATE one: a scratch buffer left
    // uninitialised, or carrying the previous call's values, produces a
    // stable-looking wrong answer that also happens to differ from the
    // sequential path. Running it twice from the same checkpoint separates
    // those -- and a difference here would be Gotcha 27's shape (output as a
    // function of hidden state rather than of the input), which is a bug in
    // its own right and a much better lead than any of the arithmetic.
    let mut twice_a = vec![LogitValue::from_f32(0.0); vocab];
    let mut twice_b = vec![LogitValue::from_f32(0.0); vocab];
    runner
        .produce_batched(&follow[..1], base, &mut twice_a)
        .expect("batched once");
    runner.rollback(&point);
    runner
        .produce_batched(&follow[..1], base, &mut twice_b)
        .expect("batched twice");
    runner.rollback(&point);
    println!("determinism (the same batched call twice from one checkpoint):");
    let unstable = compare(
        "run 1 vs run 2",
        &bits(&twice_a),
        &bits(&twice_b),
        &twice_a,
        &twice_b,
    );

    // --- M=2: the first case where rows can influence each other ---
    runner
        .produce(follow[0], base, &mut scratch)
        .expect("produce");
    let seq_2_row0 = scratch.clone();
    runner
        .produce(follow[1], base + 1, &mut scratch)
        .expect("produce");
    let seq_2_row1 = scratch.clone();
    runner.rollback(&point);

    // One buffer of `M * vocab`, token-major, so row m starts at `m * vocab`.
    let mut both = vec![LogitValue::from_f32(0.0); 2 * vocab];
    runner
        .produce_batched(&follow, base, &mut both)
        .expect("batched M=2");
    let b0 = both[..vocab].to_vec();
    let b1 = both[vocab..].to_vec();
    runner.rollback(&point);

    println!("M=2:");
    let m2_row0 = compare("row 0", &bits(&seq_2_row0), &bits(&b0), &seq_2_row0, &b0);
    let m2_row1 = compare("row 1", &bits(&seq_2_row1), &bits(&b1), &seq_2_row1, &b1);

    // THE VERDICT, stated rather than left to the reader: exactly one of
    // these is the case, and which one decides where to look next.
    // THE VERDICT IS READ IN NATS, NOT IN THE COUNT. See `kl_nats`: every
    // engine's batched and cached passes disagree, and the question is
    // whether this port's gap is bigger than the floor its SHAPE implies.
    const DENSE_SHAPE_FLOOR_NATS: f64 = 7.4e-6;
    let worst_kl = [&seq_1, &seq_2_row0, &seq_2_row1]
        .iter()
        .zip([&batched, &b0, &b1])
        .map(|(s, b)| kl_nats(s, b))
        .fold(0.0f64, f64::max);
    println!(
        "\nworst KL over the arms above: {worst_kl:.3e} nats, against a dense \
         batched-vs-cached shape floor of {DENSE_SHAPE_FLOOR_NATS:.1e} measured on \
         mlx for this same architecture"
    );

    println!("\nVERDICT:");
    if worst_kl < 10.0 * DENSE_SHAPE_FLOOR_NATS {
        println!(
            "  WITHIN THE SHAPE FLOOR. The two paths differ by about as much as the\n  \
             REFERENCE engines' own batched and cached passes differ from each other on\n  \
             this architecture (mlx 7.4e-6 nats; this repo's accepted cross-engine result\n  \
             for the family is 1.57e-5 at 100% top-1, published as no detectable gap).\n  \
             The argmax agrees on every row here, which is why a greedy stream tracks for\n  \
             ~154 tokens and then parts at the first near-tie -- ordinary FP behaviour a\n  \
             batched verify has in every engine, not a defect in this one.\n  \
             DO NOT read the differing-LOGIT COUNT as the magnitude: 88% of a vocabulary\n  \
             moving at 1e-5 nats is a tiny distributional difference, and that count is\n  \
             what made this look like a bug worth bisecting."
        );
    } else if unstable > 0 {
        println!(
            "  `produce_batched` IS NOT DETERMINISTIC -- the same call from the same\n  \
             checkpoint gave different logits twice. Stop reading the arithmetic: this\n  \
             is a scratch buffer carrying state across calls, and every comparison\n  \
             above is measuring that rather than a kernel (AGENTS.md Gotcha 27)."
        );
    } else if m1 > 0 {
        println!(
            "  `produce_batched` differs from `produce` AT ONE ROW, where no batching\n  \
             happens. The cause is a COMPOSITION difference between the two functions --\n  \
             a kernel one reaches and the other does not, a norm reading a different row,\n  \
             a residual laid out differently. Bisect the layer stack, not the batch."
        );
        if m1_pos0 > 0 {
            println!(
                "  AND it survives at position 0, where the softmax is over one key and the\n  \
                 recurrent state is empty. So it is NOT the q/k path and NOT carried\n  \
                 history: look at the norms, the FFN and the head."
            );
        } else {
            println!(
                "  BUT the two agree at position 0 -- BIT-IDENTICALLY. That exonerates the\n  \
                 embedding, every norm, the linear/GDN layers, the FFN, the head and every\n  \
                 GEMM, all of which are observable there. Position 0 is blind to exactly\n  \
                 two things: the q/k path (a softmax over one key returns V whatever the\n  \
                 score) and carried history (there is none). History is separately\n  \
                 excluded -- gdn_parity's carried-state case reads 0/128 outputs and\n  \
                 0/4096 state differing -- so what is left is the q/k path and the\n  \
                 multi-key reduction that consumes it. Bisect THERE."
            );
        }
    } else if m2_row0 > 0 || m2_row1 > 0 {
        println!(
            "  The two agree at M=1 and differ at M=2, so the BATCHING is the cause and\n  \
             the kernels are not. First place to look: the attention span each row sees.\n  \
             `produce_batched` writes all M KV rows before any query reads, and row m must\n  \
             look at [0, start+m] and no further."
        );
    } else {
        println!(
            "  Bit-identical at M=1 AND M=2 on this prompt. The divergence is real\n  \
             (measured at 154 tokens on the protocol's prose case), so it is either\n  \
             deeper into a block than two tokens, or it needs the drafter's own writes\n  \
             rather than the verify alone. Widen M and re-run before concluding anything."
        );
    }
}
