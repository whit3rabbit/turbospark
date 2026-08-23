#![cfg(target_os = "macos")]
//! Where does a steering edit stop steering and start breaking the model?
//! (`docs/OBLITERATION.md`, "Next, in order" items 1 and 2.)
//!
//! `steering_probe.rs` answers "is the edit applied, and is it inert at
//! zero" at ONE alpha. It cannot answer "which alpha", and its collapse
//! detector is a binary proxy: first token is end-of-turn, or the whole
//! continuation is one or two distinct tokens. That is enough to stop a
//! degenerate run being reported as a success and no help at all in
//! choosing an operating point.
//!
//! This sweeps alpha and reports, per arm, how surprising the steered
//! output is TO THE UNSTEERED MODEL.
//!
//! # The measurement, and the thing it is not
//!
//! Generate greedily under a steered engine, then teacher-force those exact
//! ids through the UNSTEERED engine and take the mean NLL. It needs no
//! second checkpoint, for the reason the whole page exists: the edit is
//! rank-1 and reversible, so one install is both models.
//!
//! **THIS IS NOT A COHERENCE SCORE.** A steered model is supposed to say
//! things the unsteered model would not, so the number rises for two
//! unrelated reasons -- the edit WORKING and the edit doing DAMAGE -- and no
//! single value separates them. Two things make it readable anyway:
//!
//!   - **The shape across the sweep.** Steering costs a gentle rise;
//!     collapse costs orders of magnitude. The knee is the answer, and it is
//!     visible only because there are several points.
//!   - **An anchor.** The unsteered model's perplexity on the frozen
//!     reference answer is what ORDINARY FLUENT PROSE costs it (4.9432 on
//!     `qwen38-27b`, the number `qwen38_quality_gate` freezes). An arm near
//!     that is as predictable as human writing; an arm hundreds of times
//!     above it is not writing.
//!
//! # Why the alpha 0 arm is in the sweep rather than assumed
//!
//! It is the null control again, and here it does a second job for free: it
//! must reproduce the unsteered continuation EXACTLY, which makes it the
//! zero point of the NLL column as well. If arm 0 scores anything other than
//! the unsteered model's own greedy output, every other row is measuring
//! something other than the edit.
//!
//! ```sh
//! TURBOSPARK_PROBE_INSTALL_DIR=~/models/qwen38-27b.gturbo \
//! TURBOSPARK_STEERING_VECTOR=/tmp/steer/ocean.gguf \
//!   cargo test -p turbospark-bench --test steering_sweep --release -- --ignored --nocapture
//! ```
//!
//! Six opens of one install, ~2 min. `TURBOSPARK_STEERING_ALPHAS` overrides
//! the sweep as a comma-separated list.

use foundation::LogitValue;
use runtime::{LogitProducer, RealForwardRunner};
use tokenizer::{Message, MfTokenizer, Role};
use turbospark_bench::protocol::PROTOCOL_EXPERT_CACHE_SLOTS;
use turbospark_bench::real_model::{open_model_runner, open_model_runner_steered};

mod quality_common;
use quality_common::{teacher_forced_nll, user_turn_ids, REFERENCE_ANSWER};

/// Long enough for a collapse to be unambiguous and for a knee to have
/// somewhere to appear, short enough that six arms stay inside a couple of
/// minutes.
const GENERATE: usize = 40;

/// The default sweep. Even spacing rather than a cluster, because the point
/// is to SEE the knee rather than to confirm where it was expected: the
/// derived ceiling on this page's corpus is 0.36, and bracketing it with
/// points at 0.2 and 0.4 would beg the question.
const DEFAULT_ALPHAS: &[f32] = &[0.0, 0.2, 0.4, 0.6, 0.8, 1.0];

fn ids_for(tokenizer: &MfTokenizer, text: &str) -> Vec<i32> {
    let rendered = tokenizer
        .apply_chat_template(&[Message::new(Role::User, text)])
        .expect("chat template renders");
    tokenizer.encode(&rendered, false)
}

fn argmax(v: &[LogitValue]) -> i32 {
    v.iter()
        .enumerate()
        .max_by(|x, y| x.1.to_f32().total_cmp(&y.1.to_f32()))
        .map(|(i, _)| i as i32)
        .unwrap_or(0)
}

/// Greedy continuation from a fresh state over `prompt`, returning the
/// generated ids only.
fn generate(runner: &mut RealForwardRunner, prompt: &[i32], n: usize) -> Vec<i32> {
    let mut scratch = vec![LogitValue::from_f32(0.0); runner.vocab_size()];
    runner.reset();
    for (i, &token) in prompt.iter().enumerate() {
        runner.produce(token, i, &mut scratch).expect("prefill");
    }
    let bad = scratch.iter().filter(|x| !x.to_f32().is_finite()).count();
    assert_eq!(
        bad, 0,
        "the steered prefill produced {bad} non-finite logits, which read as a PERFECT \
         score on any argmax or rank instrument (AGENTS.md Gotcha 59). Nothing downstream \
         of this would mean anything."
    );
    let mut out = Vec::with_capacity(n);
    let mut token = argmax(&scratch);
    for step in 0..n {
        out.push(token);
        runner
            .produce(token, prompt.len() + step, &mut scratch)
            .expect("decode");
        token = argmax(&scratch);
    }
    out
}

struct Arm {
    alpha: f32,
    tokens: Vec<i32>,
    /// Mean NLL under the UNSTEERED model, filled in phase B.
    nll: f64,
}

#[test]
#[ignore = "needs a real install via TURBOSPARK_PROBE_INSTALL_DIR and a vector via TURBOSPARK_STEERING_VECTOR"]
fn the_alpha_sweep_shows_where_steering_becomes_damage() {
    let dir = std::path::PathBuf::from(
        std::env::var_os("TURBOSPARK_PROBE_INSTALL_DIR").expect("TURBOSPARK_PROBE_INSTALL_DIR"),
    );
    let vector = std::path::PathBuf::from(
        std::env::var_os("TURBOSPARK_STEERING_VECTOR").expect("TURBOSPARK_STEERING_VECTOR"),
    );
    let alphas: Vec<f32> = match std::env::var("TURBOSPARK_STEERING_ALPHAS") {
        Ok(s) => s
            .split(',')
            .map(|p| p.trim().parse().expect("alphas parse as floats"))
            .collect(),
        Err(_) => DEFAULT_ALPHAS.to_vec(),
    };
    assert!(
        alphas.first() == Some(&0.0),
        "the sweep must START at alpha 0: that arm is the null control AND the zero point \
         of the NLL column, so without it every other row is a number with no origin"
    );

    let set = repack::control_vector::load_control_vector(&vector)
        .unwrap_or_else(|e| panic!("{}: {e:?}", vector.display()));
    let mode = set.declared_mode.unwrap_or_default();
    println!(
        "steering_sweep: install {}\n  vector {} ({} covered layers), mode {}, {} tokens per arm",
        dir.display(),
        vector.display(),
        set.covered_layers(),
        mode.as_str(),
        GENERATE,
    );

    let prompt_text = "Describe what you notice about the water.";

    // ---- PHASE A: generate every steered arm, ONE RUNNER ALIVE AT A TIME ----
    //
    // Deliberately not "open the unsteered engine and keep it while the
    // steered ones come and go", which would be the obvious shape. Alpha is
    // fixed at OPEN (`SteeringState` stores it), so a sweep is one open per
    // point either way, and holding two engines buys nothing while putting
    // two KV caches and two Metal contexts in flight at once.
    let mut arms: Vec<Arm> = Vec::with_capacity(alphas.len());
    let mut prompt: Vec<i32> = Vec::new();
    for &alpha in &alphas {
        let policy = runtime::SteeringPolicy {
            set: Some(set.clone()),
            mode,
            alpha,
            target: 0.0,
            gate_threshold: 0.0,
        };
        let (mut runner, tokenizer) =
            match open_model_runner_steered(&dir, PROTOCOL_EXPERT_CACHE_SLOTS, policy) {
                Ok(pair) => pair,
                Err(e) => {
                    println!("\nsteering refused this install: {e}");
                    println!(
                        "point TURBOSPARK_PROBE_INSTALL_DIR at a family the steering \
                         dispatch serves (docs/OBLITERATION.md)"
                    );
                    return;
                }
            };
        if prompt.is_empty() {
            prompt = ids_for(&tokenizer, prompt_text);
        }
        let tokens = generate(&mut runner, &prompt, GENERATE);
        println!("  alpha {alpha:>4}: generated {} tokens", tokens.len());
        arms.push(Arm {
            alpha,
            tokens,
            nll: f64::NAN,
        });
    }

    // ---- PHASE B: score every arm under the UNSTEERED engine ----
    let (mut runner, tokenizer) = open_model_runner(&dir, PROTOCOL_EXPERT_CACHE_SLOTS)
        .unwrap_or_else(|e| panic!("install opens unsteered: {e}"));
    assert!(
        runner.steering_line().is_none(),
        "the scoring engine must be UNSTEERED; if it is not, every row is one edit scored \
         against another and the column has no meaning"
    );

    // THE ANCHOR, and it is what makes the NLL column readable at all.
    // `reference_perplexity`'s corpus is a fixed human-written answer, so
    // this is what ORDINARY FLUENT PROSE costs this model. On `qwen38-27b`
    // it should land on `qwen38_quality_gate`'s frozen 4.9432, which is a
    // free cross-check that this target and that gate walk the same ids.
    let anchor_prompt = user_turn_ids(&tokenizer);
    let anchor_ids = tokenizer.encode(REFERENCE_ANSWER, false);
    let anchor_nll = teacher_forced_nll(&mut runner, &anchor_prompt, &anchor_ids);
    let anchor_ppl = anchor_nll.exp();
    assert!(
        anchor_ppl.is_finite(),
        "the anchor perplexity is not finite, so no row below can be read against it"
    );

    let unsteered = generate(&mut runner, &prompt, GENERATE);
    for arm in arms.iter_mut() {
        arm.nll = teacher_forced_nll(&mut runner, &prompt, &arm.tokens);
    }

    // ---- THE NULL CONTROL, restated here because it anchors the column ----
    let zero = &arms[0];
    assert_eq!(
        zero.tokens, unsteered,
        "alpha 0 produced a different {GENERATE}-token continuation than steering off. \
         That arm is this sweep's zero point, so until it matches, every NLL below is \
         measuring the kernel rather than the edit."
    );

    println!(
        "\nanchor: the unsteered model on the frozen reference answer reads perplexity \
         {anchor_ppl:.4}.\n  That is what ORDINARY FLUENT PROSE costs it. Read the column \
         against this, not against 1.0."
    );
    println!(
        "\n  {:>5}  {:>10}  {:>9}  {:>8}  first divergence",
        "alpha", "ppl", "x anchor", "distinct"
    );
    for arm in &arms {
        let ppl = arm.nll.exp();
        let distinct: std::collections::BTreeSet<i32> = arm.tokens.iter().copied().collect();
        let diverge = arm
            .tokens
            .iter()
            .zip(unsteered.iter())
            .position(|(a, b)| a != b)
            .map(|i| i.to_string())
            .unwrap_or_else(|| "none".to_string());
        println!(
            "  {:>5}  {:>10.4}  {:>8.1}x  {:>8}  {}",
            arm.alpha,
            ppl,
            ppl / anchor_ppl,
            distinct.len(),
            diverge
        );
    }

    println!("\nwhat each arm said:");
    for arm in &arms {
        println!(
            "  alpha {:>4}: {:?}",
            arm.alpha,
            tokenizer.decode(&arm.tokens, true)
        );
    }

    // ---- THE USABLE BAND, which is the deliverable ----
    //
    // **THE CRITERION IS THE ANCHOR CROSSING, NOT THE STEEPEST STEP.** The
    // steepest step was the first version of this and it gave the WRONG
    // answer on the corpus it was written against: it picked 0.6 -> 0.8 at
    // 60.7x and called everything at or below 0.6 usable, while alpha 0.6
    // was already emitting template markup at 13 distinct tokens. The
    // largest MULTIPLICATIVE jump lands well INSIDE the wreckage, because
    // once output is degenerate the number keeps climbing.
    //
    // The anchor crossing is not a fabricated threshold (Gotcha 38's rule).
    // The anchor is MEASURED -- what human-written prose costs this model --
    // and the argument for comparing against it is structural: a model's own
    // GREEDY output is the argmax path, so it should be far MORE predictable
    // to that model than human writing is. The fluent arms here sit at 0.3x
    // the anchor. An arm whose own greedy output is as surprising as human
    // prose has stopped producing its own distribution's typical text.
    let usable = arms
        .iter()
        .take_while(|a| a.nll.exp() < anchor_ppl)
        .last()
        .map(|a| a.alpha);
    match usable {
        Some(alpha) => println!(
            "\nUSABLE BAND: up to alpha {alpha} on THIS direction and THIS prompt.\n  \
             The last arm whose own greedy output stays more predictable to the unsteered\n  \
             model than ordinary prose is ({anchor_ppl:.4}). \
             `scripts/extract_direction.py` predicts a\n  ceiling from the captures alone \
             and the two should agree."
        ),
        None => println!(
            "\nNO USABLE BAND: even the lowest alpha in the sweep scores above the \
             {anchor_ppl:.4} anchor.\n  Either the sweep starts too high, or this direction \
             damages the model at any strength."
        ),
    }

    // Reported SECOND and explicitly not the criterion, so nobody reinstates
    // it as one.
    let mut worst_step = 0.0f64;
    let mut steepest = 1usize;
    for i in 1..arms.len() {
        let ratio = arms[i].nll.exp() / arms[i - 1].nll.exp().max(f64::MIN_POSITIVE);
        if ratio > worst_step {
            worst_step = ratio;
            steepest = i;
        }
    }
    println!(
        "  (steepest step is alpha {} -> {} at {worst_step:.1}x, which is INSIDE the damage \
         and is\n  why it is not the criterion. Past the crossing the column also stops \
         being monotone --\n  ordering broken outputs by perplexity means nothing.)",
        arms[steepest - 1].alpha,
        arms[steepest].alpha
    );

    // Finiteness on every row, at the point the measurement is taken rather
    // than where it is used (Gotcha 59).
    for arm in &arms {
        assert!(
            arm.nll.is_finite(),
            "alpha {} scored a non-finite NLL, so its row and the knee derived from it are \
             both meaningless",
            arm.alpha
        );
    }
}
