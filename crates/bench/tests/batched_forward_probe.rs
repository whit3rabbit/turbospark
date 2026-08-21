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
//! **So the next step is a per-layer bisect and not more elimination by
//! reading.** What is wanted is the first layer at which the two residual
//! streams part at span 3, which needs a readback this probe does not have.
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
        "  {label}: {differing}/{} logits differ, worst |delta| {worst:e}, argmax {} vs {}",
        a.len(),
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
    println!("\nVERDICT:");
    if m1 > 0 {
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
