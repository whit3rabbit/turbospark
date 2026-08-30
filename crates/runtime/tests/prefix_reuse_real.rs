//! Prefix KV reuse against a real install: does continuing from the previous
//! turn's KV produce the SAME tokens as re-prefilling from zero?
//!
//! That is the whole correctness claim, and nothing cheaper can make it. The
//! scripted tests in `raw_completion.rs` prove the LOOP skips the right
//! positions and skips the reset; they cannot prove the resulting attention
//! is the same, because a scripted producer has no KV to be wrong about.
//! Sliding-window rings and GDN recurrent state are exactly the things a
//! synthetic fixture does not have.
//!
//! ```sh
//! TURBOSPARK_GEMMA4_INSTALL_DIR=~/models/gemma4.gturbo \
//!   cargo test -p turbospark-runtime --test prefix_reuse_real --release -- --ignored --nocapture
//! ```

#![cfg(target_os = "macos")]

use std::path::{Path, PathBuf};

use selection::ShapingConfig;
use tokenizer::MfTokenizer;
use turbospark_runtime::{
    run_raw_completion, run_raw_completion_chunked, GenerationConfig, RateControl,
    RealForwardRunner, StopReason,
};

fn install_dir() -> Option<PathBuf> {
    std::env::var_os("TURBOSPARK_GEMMA4_INSTALL_DIR").map(PathBuf::from)
}

fn greedy(max_new: u32) -> GenerationConfig {
    GenerationConfig {
        // Greedy: two runs of the same math must agree exactly, and any
        // sampling would let them differ for a reason that is not the
        // feature under test.
        shaping: ShapingConfig::new(0.0, 0, None, 1.0, None).unwrap(),
        max_new_tokens: max_new,
        stop_strings: Vec::new(),
        extra_stop_tokens: Vec::new(),
        rate: RateControl::default(),
    }
}

fn open(dir: &Path) -> (RealForwardRunner, MfTokenizer) {
    let arch = turbospark_repack::peek_manifest_arch(dir).expect("install manifest reads");
    let runner = RealForwardRunner::open(dir, arch).expect("install opens");
    let tokenizer = MfTokenizer::load_from_dir(dir).expect("install tokenizer loads");
    (runner, tokenizer)
}

/// Generate `turns` sequentially, each prompt extending the last, returning
/// the token ids each turn produced.
fn converse(
    runner: &mut RealForwardRunner,
    tokenizer: &MfTokenizer,
    turns: &[Vec<i32>],
    max_new: u32,
) -> Vec<Vec<i32>> {
    let vocab = runner.vocab_size();
    let config = greedy(max_new);
    let mut out = Vec::new();
    for prompt in turns {
        let mut generated = Vec::new();
        let result = run_raw_completion(runner, tokenizer, prompt, &config, 4096, vocab, |event| {
            if let turbospark_runtime::RawDecodeProgress::Token { id, .. } = event {
                generated.push(id);
            }
        })
        .expect("generation succeeds");
        assert!(
            matches!(result.reason, StopReason::MaxTokens | StopReason::EndOfTurn),
            "unexpected stop: {:?}",
            result.reason
        );
        out.push(generated);
    }
    out
}

/// Turn 2's prompt = turn 1's prompt + turn 1's output + a continuation, so
/// the recorded state IS a strict prefix of it and reuse can fire.
fn two_turns(tokenizer: &MfTokenizer, first_out: &[i32]) -> Vec<Vec<i32>> {
    let first = tokenizer.encode("The ocean is", false);
    let mut second = first.clone();
    second.extend_from_slice(first_out);
    second.extend(tokenizer.encode(" and also", false));
    vec![first, second]
}

#[test]
#[ignore = "needs a real Gemma 4 .gturbo install (TURBOSPARK_GEMMA4_INSTALL_DIR)"]
fn reusing_the_previous_turns_kv_generates_the_same_tokens_as_re_prefilling() {
    let Some(dir) = install_dir() else {
        eprintln!("SKIP: TURBOSPARK_GEMMA4_INSTALL_DIR unset");
        return;
    };
    let max_new = 24;

    // Establish turn 1's output first, so turn 2's prompt is built from the
    // real continuation rather than from a guess about it.
    let (mut probe, tokenizer) = open(&dir);
    let first_out = converse(
        &mut probe,
        &tokenizer,
        &[tokenizer.encode("The ocean is", false)],
        max_new,
    )
    .pop()
    .unwrap();
    drop(probe);
    let turns = two_turns(&tokenizer, &first_out);

    // Arm A: reuse OFF. Every turn resets and re-prefills, which is the
    // engine's behaviour before this feature and therefore the reference.
    let (mut a, _) = open(&dir);
    a.set_prefix_reuse(false);
    let baseline = converse(&mut a, &tokenizer, &turns, max_new);
    drop(a);

    // Arm B: reuse ON. Turn 2 continues from turn 1's KV.
    let (mut b, _) = open(&dir);
    b.set_prefix_reuse(true);
    let reused = converse(&mut b, &tokenizer, &turns, max_new);

    assert_eq!(
        baseline[0], reused[0],
        "turn 1 cannot differ: nothing precedes it, so no prefix exists to reuse"
    );
    assert_eq!(
        baseline[1], reused[1],
        "turn 2 diverged. Reuse must be a pure skip of work, not a change of \
         math: same positions, same KV rows, same tokens"
    );
    println!(
        "turn 1: {} tokens, turn 2: {} tokens, identical across both arms",
        reused[0].len(),
        reused[1].len()
    );
}

#[test]
#[ignore = "needs a real Gemma 4 .gturbo install (TURBOSPARK_GEMMA4_INSTALL_DIR)"]
fn a_diverging_prompt_falls_back_and_still_matches_the_reference() {
    let Some(dir) = install_dir() else {
        eprintln!("SKIP: TURBOSPARK_GEMMA4_INSTALL_DIR unset");
        return;
    };
    let max_new = 24;
    let (_, tokenizer) = open(&dir);

    // Turn 2 shares only a leading fragment and then diverges, so the record
    // cannot match and the loop must fall back to a full re-prefill. The
    // dangerous outcome is not an error: it is answering turn 2 from turn
    // 1's state, which reads as a fluent reply to a question nobody asked.
    let turns = vec![
        tokenizer.encode("The ocean is", false),
        tokenizer.encode("The mountain range was", false),
    ];

    let (mut a, _) = open(&dir);
    a.set_prefix_reuse(false);
    let baseline = converse(&mut a, &tokenizer, &turns, max_new);
    drop(a);

    let (mut b, _) = open(&dir);
    b.set_prefix_reuse(true);
    let reused = converse(&mut b, &tokenizer, &turns, max_new);

    assert_eq!(
        baseline, reused,
        "a diverging turn must re-prefill and match"
    );
}

#[test]
#[ignore = "needs a real Gemma 4 .gturbo install (TURBOSPARK_GEMMA4_INSTALL_DIR)"]
fn reuse_actually_fires_on_the_second_turn() {
    // The two tests above pass trivially if reuse never engages: falling back
    // to a full prefill agrees with the reference by construction. This one
    // asserts the optimisation HAPPENED, so they cannot both be green for the
    // wrong reason -- the shape `docs/TESTING.md` warns about, where a
    // feature's tests prove only that disabling it is safe.
    let Some(dir) = install_dir() else {
        eprintln!("SKIP: TURBOSPARK_GEMMA4_INSTALL_DIR unset");
        return;
    };
    let (mut runner, tokenizer) = open(&dir);
    runner.set_prefix_reuse(true);
    let max_new = 8;

    let first = tokenizer.encode("The ocean is", false);
    let out = converse(
        &mut runner,
        &tokenizer,
        std::slice::from_ref(&first),
        max_new,
    )
    .pop()
    .unwrap();

    let mut second = first.clone();
    second.extend_from_slice(&out);
    second.extend(tokenizer.encode(" and also", false));

    // Read off the RESULT rather than by calling the seam: `try_reuse_prefix`
    // COMMITS (it moves the KV cursor back), so using it as a query would
    // perform the very thing it is being asked about.
    let vocab = runner.vocab_size();
    let result = run_raw_completion(
        &mut runner,
        &tokenizer,
        &second,
        &greedy(max_new),
        4096,
        vocab,
        |_| {},
    )
    .expect("second turn generates");
    let reusable = result.reused_prefix_tokens;
    assert!(
        reusable >= first.len(),
        "expected at least the first turn's {} prompt tokens to be reused, got {reusable}",
        first.len()
    );
    println!("reused {reusable} of {} prompt tokens", second.len());
}

#[test]
#[ignore = "needs a real Gemma 4 .gturbo install (TURBOSPARK_GEMMA4_INSTALL_DIR)"]
fn reuse_cuts_prefill_time_on_a_long_shared_prefix() {
    // The VALUE claim, reported rather than tightly asserted: a wall-clock
    // threshold would be a thermal flake (AGENTS.md Gotcha 28). The assertion
    // is only that reuse is not SLOWER, which is the regression worth
    // catching; the printed ratio is the number to read.
    let Some(dir) = install_dir() else {
        eprintln!("SKIP: TURBOSPARK_GEMMA4_INSTALL_DIR unset");
        return;
    };
    let (_, tokenizer) = open(&dir);

    // A transcript-shaped prompt: a long shared body, then a short new turn,
    // which is exactly the shape a chat client resends every message.
    let body = tokenizer.encode(
        "Coastal wetlands reduce flood damage through several mechanisms. \
         Wave attenuation happens as vegetation adds friction. Storm surge \
         buffering spreads water over a wide shallow zone. Floodwater storage \
         holds volume in soil and roots. Sediment trapping builds elevation \
         over time, which sustains the marsh against sea level rise.",
        false,
    );
    let config = greedy(4);

    let time_turn_two = |reuse: bool| {
        let (mut runner, tok) = open(&dir);
        runner.set_prefix_reuse(reuse);
        let vocab = runner.vocab_size();

        // Turn 1 builds the state, and its OUTPUT is part of what that state
        // covers: the decode loop feeds each sampled token back. So turn 2's
        // prompt must carry those ids too, which is what a chat client does
        // when it resends the transcript with the assistant's reply in it.
        // Omitting them makes the record diverge at the first generated
        // position and reuse silently never fires -- measured at a 1.05x
        // "speedup" that reads as the feature being worthless.
        let mut first_out = Vec::new();
        run_raw_completion(&mut runner, &tok, &body, &config, 4096, vocab, |e| {
            if let turbospark_runtime::RawDecodeProgress::Token { id, .. } = e {
                first_out.push(id);
            }
        })
        .unwrap();
        let mut second = body.clone();
        second.extend_from_slice(&first_out);
        second.extend(tok.encode(" Summarize that in one sentence.", false));

        let start = std::time::Instant::now();
        let result =
            run_raw_completion(&mut runner, &tok, &second, &config, 4096, vocab, |_| {}).unwrap();
        (
            start.elapsed().as_secs_f64(),
            result.prefill_seconds,
            second.len(),
        )
    };

    // Interleaved rather than batched, per CLAUDE.local.md's A/B rule.
    let (off_total, off_prefill, prompt_len) = time_turn_two(false);
    let (on_total, on_prefill, _) = time_turn_two(true);

    println!(
        "prompt {prompt_len} tokens, shared prefix {} tokens",
        body.len()
    );
    println!("  reuse OFF: prefill {off_prefill:.3}s, turn {off_total:.3}s");
    println!("  reuse ON:  prefill {on_prefill:.3}s, turn {on_total:.3}s");
    println!(
        "  prefill speedup: {:.2}x",
        off_prefill / on_prefill.max(1e-9)
    );

    assert!(
        on_prefill <= off_prefill * 1.5,
        "reuse made prefill SLOWER ({on_prefill:.3}s against {off_prefill:.3}s), \
         which means it is doing work instead of skipping it"
    );
}

/// The same claim through the CHUNKED loop, which is the one that matters:
/// every family supporting chunked prefill routes there, so the CLI and the
/// server reach it rather than `run_raw_completion`.
///
/// This test exists because three mutations of the chunked wiring -- starting
/// the spans at 0, dropping the absolute-position offset, and slicing the
/// chunk from the wrong base -- ALL survived the tests above. They drive the
/// sequential loop, so they could not see the chunked one at all (AGENTS.md:
/// a surviving mutation that applied is a missing test, not a weak one).
fn converse_chunked(
    runner: &mut RealForwardRunner,
    tokenizer: &MfTokenizer,
    turns: &[Vec<i32>],
    max_new: u32,
) -> Vec<(Vec<i32>, usize)> {
    let vocab = runner.vocab_size();
    let config = greedy(max_new);
    let mut out = Vec::new();
    for prompt in turns {
        let mut generated = Vec::new();
        let result = run_raw_completion_chunked(
            runner,
            tokenizer,
            prompt,
            &config,
            4096,
            vocab,
            // Small enough that a multi-token prompt spans several chunks,
            // so an off-by-one in the span walk cannot hide inside one.
            8,
            |event| {
                if let turbospark_runtime::RawDecodeProgress::Token { id, .. } = event {
                    generated.push(id);
                }
            },
        )
        .expect("chunked generation succeeds");
        out.push((generated, result.reused_prefix_tokens));
    }
    out
}

#[test]
#[ignore = "needs a real Gemma 4 .gturbo install (TURBOSPARK_GEMMA4_INSTALL_DIR)"]
fn chunked_prefill_reuse_generates_the_same_tokens_and_actually_fires() {
    let Some(dir) = install_dir() else {
        eprintln!("SKIP: TURBOSPARK_GEMMA4_INSTALL_DIR unset");
        return;
    };
    let max_new = 24;

    let (mut probe, tokenizer) = open(&dir);
    let first_out = converse(
        &mut probe,
        &tokenizer,
        &[tokenizer.encode("The ocean is", false)],
        max_new,
    )
    .pop()
    .unwrap();
    drop(probe);
    let turns = two_turns(&tokenizer, &first_out);

    let (mut a, _) = open(&dir);
    a.set_prefix_reuse(false);
    let baseline = converse_chunked(&mut a, &tokenizer, &turns, max_new);
    drop(a);

    let (mut b, _) = open(&dir);
    b.set_prefix_reuse(true);
    let reused = converse_chunked(&mut b, &tokenizer, &turns, max_new);

    assert_eq!(
        baseline[0].0, reused[0].0,
        "turn 1 cannot differ: nothing precedes it"
    );
    assert_eq!(
        baseline[1].0, reused[1].0,
        "turn 2 diverged under chunked prefill. The span walk starts at the \
         reused offset and its `completed_count` is walk-relative while the KV \
         cursor is absolute; getting either wrong reads rows nobody wrote"
    );
    assert_eq!(baseline[0].1, 0, "the OFF arm must never reuse");
    assert_eq!(baseline[1].1, 0, "the OFF arm must never reuse");
    // And the ON arm must actually have done it, or the equality above is the
    // trivial one: two full prefills agreeing with each other.
    assert!(
        reused[1].1 > 0,
        "chunked reuse never fired, so this test proves only that disabling \
         it is safe"
    );
    println!(
        "chunked: turn 2 reused {} of {} prompt tokens, tokens identical",
        reused[1].1,
        turns[1].len()
    );
}
