#![cfg(target_os = "macos")]
//! Does live directional steering do what it claims, and does OFF really
//! cost nothing? (`docs/OBLITERATION.md`, ROADMAP item 9 phase 3.)
//!
//! The edit is rank-1 and reversible, so **no second model and no second
//! process is needed** to compare a steered engine against its original --
//! which is the whole reason this work is a runtime edit rather than a
//! weight rewrite. Three opens of ONE install answer everything below.
//!
//! # The four arms, and which one is load-bearing
//!
//! 1. **The null control.** `alpha = 0` with the kernel DISPATCHED at every
//!    covered layer must be BIT-IDENTICAL to steering off. This is the arm
//!    worth having: `x' = x - 0 * c * d_hat` is the identity in exact
//!    arithmetic, so any difference at all is the kernel writing something
//!    it should not -- a wrong reduction, a wrong buffer offset, a wrong row
//!    stride -- and one comparison covers all three. It is an ASSERTION.
//! 2. **Steered vs unsteered, in NATS against the shape floor.** Here a
//!    LARGE divergence is the success signal and the floor is what says it
//!    is the edit rather than noise. That direction is unusual enough to be
//!    worth stating: every other divergence measurement in this crate wants
//!    a small number.
//! 3. **The coefficient trace.** `c_l` per steered layer, read back from the
//!    kernel's own scratch. This is the "reference" without a reference
//!    model -- it says how much of the direction was present at each layer
//!    before the edit, which is both the diagnostic and the input to
//!    in-kernel gating (`--steering-gate`).
//! 4. **Determinism.** The same steered generation twice from one checkpoint
//!    must produce ONE output, which is `gguf_nondeterminism_probe.rs`'s
//!    assertion applied to the new dispatch. A steered engine whose output
//!    depended on hidden state would be AGENTS.md Gotcha 27 arriving in a
//!    new kernel.
//!
//! # What it cannot see, stated so nobody reads it as more than it is
//!
//! Nothing here says the direction is a good direction, or that it names the
//! concept whoever extracted it meant. A control vector's SEMANTICS are a
//! judgement about generated text; this measures that the arithmetic is
//! applied, is inert at zero, is reproducible, and moves the distribution
//! by an amount the numerics cannot account for.
//!
//! ```sh
//! TURBOSPARK_PROBE_INSTALL_DIR=~/models/qwen38-27b.gturbo \
//! TURBOSPARK_STEERING_VECTOR=/tmp/steer/ocean.gguf \
//!   cargo test -p turbospark-bench --test steering_probe --release -- --ignored --nocapture
//! ```
//!
//! `TURBOSPARK_STEERING_SCALE` overrides the strength. **The default is
//! `0.3`, not `1.0`**, and the reason is arm 2's one blind spot: `ablate` at
//! full strength over every layer COLLAPSES the turn on this family, and a
//! collapsed turn scores a huge divergence -- so the natural default would
//! measure the documented failure mode and report it as a success. See the
//! collapse note at the end of the run. The mode comes from the file's own
//! declaration when it has one, exactly as the CLI resolves it: a vector
//! built for one edit should apply that edit.
//!
//! Needs a family the steering dispatch serves; every other one is refused
//! at open BY NAME, which this reports rather than swallowing.

use foundation::LogitValue;
use runtime::{LogitProducer, RealForwardRunner};
use tokenizer::{Message, MfTokenizer, Role};
use turbospark_bench::protocol::PROTOCOL_EXPERT_CACHE_SLOTS;
use turbospark_bench::real_model::{open_model_runner, open_model_runner_steered};

/// Long enough that a steered continuation can visibly part from an
/// unsteered one, short enough that three opens plus five walks stay inside
/// a couple of minutes.
const GREEDY_TOKENS: usize = 24;

/// The prompt every arm walks unless `TURBOSPARK_STEERING_PROMPT` says
/// otherwise.
///
/// Kept verbatim so `docs/OBLITERATION.md`'s frozen arms reproduce; it is
/// ocean-adjacent because the first direction measured on this surface was,
/// and that coupling is exactly what the override exists to break.
const DEFAULT_STEERING_PROMPT: &str = "Describe what you notice about the water.";

/// The scale a divergence here is read against.
///
/// A dense model's batched and cached passes are nearly the same
/// computation, so this family's shape floor nearly vanishes -- 0.0000074
/// nats measured on mlx for this exact architecture, which is also what
/// `batched_forward_probe.rs` reads its own arms against. It is the
/// magnitude of "differences that are not the edit", so a steered arm
/// ABOVE it is the edit and one at or below it is noise wearing the edit's
/// name.
///
/// **This is why a zero direction is the failure mode to fear rather than a
/// wrong one.** `SteeringState::build` computes `inv_norm` from the values
/// it will actually read and returns 0.0 when they sum to zero, so a
/// direction file full of zeros -- which is what a capture taken on a family
/// whose flow contains no copy extracts to -- makes the whole pipeline run,
/// report a covered layer count, print a summary line, and steer NOTHING.
/// No error anywhere. Asserting against this floor is what catches it.
const DENSE_SHAPE_FLOOR_NATS: f64 = 7.4e-6;

fn ids_for(tokenizer: &MfTokenizer, text: &str) -> Vec<i32> {
    let rendered = tokenizer
        .apply_chat_template(&[Message::new(Role::User, text)])
        .expect("chat template renders");
    tokenizer.encode(&rendered, false)
}

fn bits(logits: &[LogitValue]) -> Vec<u16> {
    logits.iter().map(|v| v.to_bits()).collect()
}

fn argmax(v: &[LogitValue]) -> i32 {
    v.iter()
        .enumerate()
        .max_by(|x, y| x.1.to_f32().total_cmp(&y.1.to_f32()))
        .map(|(i, _)| i as i32)
        .unwrap_or(0)
}

/// KL(p || q) in nats over the two rows' softmaxes, in f64 off an f16 input,
/// which is exact. Same function `batched_forward_probe.rs` uses, and for
/// the same reason: a COUNT of differing logits cannot distinguish a
/// last-bit difference from a rewritten distribution, and only one of those
/// is what this is looking for.
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

/// **EVERY row is checked finite before anything reads it.**
///
/// A NaN scores the BEST POSSIBLE value on an argmax and on any rank
/// instrument, because every comparison against NaN is false (AGENTS.md
/// Gotcha 59), and it also makes `kl_nats` return NaN, which a `>` against a
/// floor silently fails rather than flagging. Steering is exactly the kind
/// of edit that can produce one: `Add` and `Clamp` both ADD to a residual
/// stream stored in FP16, whose largest finite value is 65,504.
fn require_finite(label: &str, v: &[LogitValue]) {
    let bad = v.iter().filter(|x| !x.to_f32().is_finite()).count();
    assert_eq!(
        bad,
        0,
        "{label}: {bad} of {} logits are not finite. A non-finite row reads as a PERFECT \
         score on every rank instrument downstream and makes every KL below NaN, so nothing \
         after this point would mean anything. `Add` and `Clamp` add to an FP16 residual \
         stream that saturates at 65,504; lower --steering-scale or use Ablate, which \
         cannot overflow.",
        v.len()
    );
}

/// Walks a prompt from a fresh state and returns the last token's logits.
fn walk(runner: &mut RealForwardRunner, ids: &[i32]) -> Vec<LogitValue> {
    let mut scratch = vec![LogitValue::from_f32(0.0); runner.vocab_size()];
    runner.reset();
    for (i, &token) in ids.iter().enumerate() {
        runner.produce(token, i, &mut scratch).expect("produce");
    }
    scratch
}

/// Greedy continuation from wherever the runner already is.
///
/// A LOCAL argmax rather than `selection::select`, deliberately: this arm is
/// asking whether the ENGINE is deterministic under the new dispatch, and
/// routing it through the sampler would put a second component inside the
/// thing being isolated. The sampler is exercised by the quality gates.
fn greedy(runner: &mut RealForwardRunner, from: usize, first: i32, n: usize) -> Vec<i32> {
    let mut scratch = vec![LogitValue::from_f32(0.0); runner.vocab_size()];
    let mut out = Vec::with_capacity(n);
    let mut token = first;
    for step in 0..n {
        runner
            .produce(token, from + step, &mut scratch)
            .expect("produce");
        token = argmax(&scratch);
        out.push(token);
    }
    out
}

#[test]
#[ignore = "needs a real install via TURBOSPARK_PROBE_INSTALL_DIR and a vector via TURBOSPARK_STEERING_VECTOR"]
fn a_steering_edit_is_inert_at_zero_and_moves_the_distribution_at_one() {
    let dir = std::path::PathBuf::from(
        std::env::var_os("TURBOSPARK_PROBE_INSTALL_DIR").expect("TURBOSPARK_PROBE_INSTALL_DIR"),
    );
    let vector = std::path::PathBuf::from(
        std::env::var_os("TURBOSPARK_STEERING_VECTOR").expect("TURBOSPARK_STEERING_VECTOR"),
    );
    // **THE DEFAULT IS 0.3 AND NOT 1.0, AND THAT IS THE WHOLE OPERATING
    // POINT QUESTION.** `docs/OBLITERATION.md` records `ablate` at
    // `alpha = 1` across all 64 layers as COLLAPSING the turn -- the model
    // emits its end-of-turn token immediately and generates nothing --
    // because this direction's norm at the late layers is a sixth of the
    // residual stream's and the stream IS the output head's input.
    //
    // That matters here rather than being trivia, because a collapsed turn
    // still passes arm 2 handsomely: it reads tens of nats from the
    // unsteered distribution, which is exactly what "the edit works" looks
    // like in that number. This probe's first run defaulted to 1.0, measured
    // the collapse, and reported it as a successful steer. The doc's own
    // coherent row is what the default reproduces now.
    let alpha: f32 = std::env::var("TURBOSPARK_STEERING_SCALE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.3);
    assert!(
        alpha.is_finite() && alpha != 0.0,
        "TURBOSPARK_STEERING_SCALE must be finite and non-zero; the ZERO case is arm 1 and \
         is run unconditionally, so setting it here would make arms 2-4 re-measure the null \
         control and report it as the edit"
    );

    let set = repack::control_vector::load_control_vector(&vector)
        .unwrap_or_else(|e| panic!("{}: {e:?}", vector.display()));

    // `TURBOSPARK_STEERING_MODE` points the four arms at a different edit
    // without rewriting the vector, which is what makes two modes comparable
    // on ONE artifact. REFUSED on an unknown spelling rather than falling
    // back to the file's: a probe that silently measured a different edit
    // than the one asked for would report the wrong model as the right one
    // (`SteeringMode::parse` returns `None` for exactly that reason).
    let mode = match std::env::var("TURBOSPARK_STEERING_MODE") {
        Ok(raw) => foundation::SteeringMode::parse(raw.trim()).unwrap_or_else(|| {
            panic!(
                "TURBOSPARK_STEERING_MODE={raw:?} is not a steering mode. \
                 Accepted: ablate, add, clamp, renorm."
            )
        }),
        Err(_) => set.declared_mode.unwrap_or_default(),
    };
    println!(
        "steering_probe: install {}\n  vector {} ({} covered layers), mode {}, alpha {alpha}",
        dir.display(),
        vector.display(),
        set.covered_layers(),
        mode.as_str(),
    );

    let policy = |alpha: f32| runtime::SteeringPolicy {
        set: Some(set.clone()),
        mode,
        alpha,
        target: 0.0,
        gate_threshold: 0.0,
    };

    // ONE prompt for every arm. The comparison is arithmetic, so what the
    // prompt says does not matter -- but it has to be the SAME text in all
    // three engines, or a difference in the tokens would read as the edit.
    //
    // The DEFAULT is ocean-adjacent because the first direction measured here
    // was, and it is kept verbatim so `docs/OBLITERATION.md`'s frozen arms
    // reproduce. `TURBOSPARK_STEERING_PROMPT` is what makes a SECOND direction
    // measurable at all: a vector extracted from some other concept pair has
    // no reason to move this text, and an arm that reads a small divergence
    // because the prompt was wrong for the direction is indistinguishable from
    // one that reads it because the edit does not work.
    let prompt_text = std::env::var("TURBOSPARK_STEERING_PROMPT")
        .unwrap_or_else(|_| DEFAULT_STEERING_PROMPT.to_string());
    let prompt_text = prompt_text.as_str();
    println!("  prompt {prompt_text:?}");

    // ---- ARM 0: the unsteered reference ----
    //
    // Opened FIRST and dropped before the steered ones, so a footprint or a
    // Metal-allocation difference cannot be blamed on ordering.
    let (off_logits, off_tokens, prompt, tokenizer) = {
        let (mut runner, tokenizer) = open_model_runner(&dir, PROTOCOL_EXPERT_CACHE_SLOTS)
            .unwrap_or_else(|e| panic!("install opens unsteered: {e}"));
        assert!(
            runner.steering_line().is_none(),
            "an engine opened through `open_model_runner` must report NO steering; if it \
             does, the reference arm is itself steered and every comparison below is \
             measuring one edit against another"
        );
        let prompt = ids_for(&tokenizer, prompt_text);
        let logits = walk(&mut runner, &prompt);
        require_finite("unsteered", &logits);
        let first = argmax(&logits);
        let tokens = greedy(&mut runner, prompt.len(), first, GREEDY_TOKENS);
        // The tokenizer outlives its runner deliberately: the two steered
        // arms need it to DETOKENIZE, and it is a property of the install
        // rather than of the engine, so re-loading it per arm would only
        // add a way for the arms to disagree about what the ids mean.
        (logits, tokens, prompt, tokenizer)
    };
    println!(
        "  {} prompt tokens, vocab {}",
        prompt.len(),
        off_logits.len()
    );

    // ---- ARM 1: THE NULL CONTROL, and the one that is an assertion ----
    //
    // The kernel runs at every covered layer and computes a real coefficient;
    // only the scale it applies is zero. So this is not "steering off by
    // another route" -- it is the full dispatch with an edit of zero, which
    // is what makes a bit-identical result evidence about the KERNEL rather
    // than about the flag.
    {
        let (mut runner, _tokenizer) =
            match open_model_runner_steered(&dir, PROTOCOL_EXPERT_CACHE_SLOTS, policy(0.0)) {
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
        println!(
            "\narm 1, null control -- {}",
            runner.steering_line().expect("steering is on at alpha 0")
        );
        let null_logits = walk(&mut runner, &prompt);
        require_finite("alpha 0", &null_logits);
        let differing = bits(&off_logits)
            .iter()
            .zip(bits(&null_logits).iter())
            .filter(|(a, b)| a != b)
            .count();
        println!(
            "  {differing}/{} logits differ from steering off, KL {:.3e} nats",
            off_logits.len(),
            kl_nats(&off_logits, &null_logits)
        );
        assert_eq!(
            differing, 0,
            "alpha 0 must be BIT-IDENTICAL to steering off and {differing} logits differ. \
             `x - 0 * c * d_hat` is the identity, so the kernel is writing something it \
             should not: a wrong reduction, a wrong direction offset, or a wrong row \
             stride. One of those three, and this comparison does not say which -- but it \
             does say the edit's arithmetic cannot be trusted at any other alpha either."
        );

        // The same check on GENERATED TEXT and not just one row. A last-bit
        // difference is invisible in a single logit comparison's argmax and
        // shows up only once it lands on a near-tie, which is exactly how
        // the batched verify's divergence took 154 tokens to appear.
        let null_tokens = greedy(
            &mut runner,
            prompt.len(),
            argmax(&null_logits),
            GREEDY_TOKENS,
        );
        assert_eq!(
            null_tokens, off_tokens,
            "alpha 0 produced a different {GREEDY_TOKENS}-token greedy continuation than \
             steering off. The single-row comparison above passed, so this is a difference \
             too small to move one argmax that compounds over a generation."
        );
        println!("  {GREEDY_TOKENS}-token greedy continuation identical to steering off");
    }

    // ---- ARMS 2, 3, 4: the real edit ----
    let (mut runner, _tokenizer) =
        open_model_runner_steered(&dir, PROTOCOL_EXPERT_CACHE_SLOTS, policy(alpha))
            .unwrap_or_else(|e| panic!("install opens steered: {e}"));
    println!(
        "\narms 2-4 -- {}",
        runner.steering_line().expect("steering is on")
    );

    let steered_logits = walk(&mut runner, &prompt);
    require_finite(&format!("alpha {alpha}"), &steered_logits);

    // --- ARM 3: the coefficient trace, read before anything else runs ---
    //
    // Taken HERE because the buffer is rewritten every pass, so it describes
    // the last `produce` and nothing else. A trace read after the greedy
    // continuation below would describe that continuation's final token,
    // which is a different question.
    match runner.steering_coefficients() {
        Some(coeffs) => {
            let present: Vec<(usize, f32)> = coeffs
                .iter()
                .enumerate()
                .filter_map(|(l, c)| c.map(|v| (l, v)))
                .collect();
            let finite = present.iter().filter(|(_, c)| c.is_finite()).count();
            assert_eq!(
                finite,
                present.len(),
                "a steered layer reported a non-finite coefficient, which is the \
                 measurement itself going bad rather than the model"
            );
            let mag = |f: fn(f64, f64) -> f64, init: f64| {
                present.iter().map(|(_, c)| c.abs() as f64).fold(init, f)
            };
            println!(
                "arm 3, coefficient trace at the last prompt token: {} steered layers, \
                 |c| min {:.4} max {:.4}",
                present.len(),
                mag(f64::min, f64::INFINITY),
                mag(f64::max, 0.0)
            );
            // Printed sparsely: sixty-four rows is a wall, and what a reader
            // needs is the SHAPE -- which layers carry the direction.
            for (l, c) in present.iter().step_by(present.len().div_ceil(8).max(1)) {
                println!("    layer {l:>3}  c = {c:+.4}");
            }
            assert!(
                present.iter().any(|(_, c)| c.abs() > 0.0),
                "every steered layer reported a coefficient of exactly zero. Either the \
                 direction is all zeros -- which `inv_norm` turns into an inert edit with \
                 no error anywhere -- or the readback is not wired to the kernel that \
                 writes it. Both look like a working steer from the outside."
            );
        }
        None => panic!("a steered runner must report coefficients"),
    }

    // --- ARM 2: the divergence, and here a LARGE number is the good one ---
    let steered_kl = kl_nats(&off_logits, &steered_logits);
    let differing = bits(&off_logits)
        .iter()
        .zip(bits(&steered_logits).iter())
        .filter(|(a, b)| a != b)
        .count();
    println!(
        "\narm 2, steered vs unsteered: {differing}/{} logits differ, KL {steered_kl:.4e} nats \
         ({:.0}x the {DENSE_SHAPE_FLOOR_NATS:.1e} dense shape floor), argmax {} vs {}",
        off_logits.len(),
        steered_kl / DENSE_SHAPE_FLOOR_NATS,
        argmax(&off_logits),
        argmax(&steered_logits),
    );
    assert!(
        steered_kl.is_finite() && steered_kl > DENSE_SHAPE_FLOOR_NATS,
        "the steered distribution is {steered_kl:.4e} nats from the unsteered one, at or \
         below the {DENSE_SHAPE_FLOOR_NATS:.1e} floor that ordinary numerical differences \
         already cost on this architecture. So this run did not measurably steer, and the \
         likeliest cause is a direction that is all zeros or is not covering the layers \
         this model runs: `inv_norm` makes a zero direction inert, and every surface \
         around it -- the covered-layer count, the summary line, the coefficient \
         readback -- still reports a healthy steer."
    );

    // --- ARM 4: determinism, which no comparison against the OFF arm can see ---
    //
    // Every arm above compares the steered engine against a DIFFERENT engine,
    // and that cannot tell an arithmetic difference from a STATE one: a
    // scratch buffer carrying the previous call's values produces a stable
    // wrong answer that also differs from the unsteered path. Running the
    // same generation twice from one checkpoint separates them, and a
    // difference here would be AGENTS.md Gotcha 27 in a new kernel -- output
    // as a function of hidden state rather than of the input.
    //
    // MUTATION-CHECKED, AND THE RESULT IS WORTH RECORDING RATHER THAN
    // TUNING AWAY: deleting the rollback between the two runs reddens this
    // test through the ENGINE's own position guard ("non-sequential position
    // 20; KV cache is at 44") and never reaches the equality below. So an
    // invariant is doing the work, and the only state leak this arm can
    // actually observe is one that preserves the KV cursor while changing
    // what the kernels read -- which is precisely the shape of a scratch
    // buffer carrying values across calls, and precisely what a steering
    // dispatch could introduce. It is a regression detector for a future
    // non-determinism, not a check that passes today by luck.
    let point = runner.checkpoint();
    let first = argmax(&steered_logits);
    let run_a = greedy(&mut runner, prompt.len(), first, GREEDY_TOKENS);
    runner.rollback(&point);
    let run_b = greedy(&mut runner, prompt.len(), first, GREEDY_TOKENS);
    runner.rollback(&point);
    assert_eq!(
        run_a, run_b,
        "the same steered generation from one checkpoint produced two different token \
         streams. Stop reading the divergence above: this is state leaking across calls, \
         and every comparison in this file is measuring that rather than the edit."
    );
    println!(
        "arm 4, determinism: two {GREEDY_TOKENS}-token runs from one checkpoint agree, \
         and the steered continuation differs from the unsteered one at token {}",
        run_a
            .iter()
            .zip(off_tokens.iter())
            .position(|(a, b)| a != b)
            .map(|i| i.to_string())
            .unwrap_or_else(|| format!("NONE of {GREEDY_TOKENS}")),
    );

    // BOTH CONTINUATIONS ARE PRINTED, and that is not decoration.
    //
    // A KL of tens of nats is what a STEERED model and a DESTROYED one both
    // look like, and no number in this file tells them apart -- ablating a
    // direction out of all sixty-four layers at full strength is a large
    // edit, and "the distribution moved enormously" is exactly as consistent
    // with fluent-but-redirected text as with token soup. This repo's
    // characteristic failure is an instrument reading a plausible value on
    // degenerate input (Gotchas 30, 57, 59), so the cheap defence is to show
    // the reader the thing the number is about. It is REPORTED and not
    // asserted: fluency is a judgement, and a probe that tried to score it
    // would be inventing a threshold.
    println!("\nwhat the two engines actually said (greedy, {GREEDY_TOKENS} tokens):");
    println!("  unsteered: {:?}", tokenizer.decode(&off_tokens, true));
    println!("  steered:   {:?}", tokenizer.decode(&run_a, true));

    // --- COLLAPSE, which has an OBJECTIVE proxy even though fluency does not ---
    //
    // Judging whether steered text reads well is not something a probe can
    // do. Judging whether the turn ENDED is: the documented collapse mode is
    // an immediate end-of-turn token, and a run that emits its stop token at
    // position 0 or repeats a single token has produced no turn to judge.
    // Reported rather than asserted, because a caller deliberately measuring
    // the collapse point is doing something legitimate -- what must not
    // happen is that it passes silently while arm 2 calls it a success.
    let distinct: std::collections::BTreeSet<i32> = run_a.iter().copied().collect();
    let collapsed = run_a.first() == Some(&tokenizer.end_of_turn_id) || distinct.len() <= 2;
    if collapsed {
        println!(
            "\n  *** COLLAPSED: the steered turn produced {} distinct token(s){}. The KL \
             above is real and means nothing about steering -- a model that stops \
             generating diverges enormously from one that does not. `ablate` at alpha 1 \
             over ALL layers is the documented collapse point (docs/OBLITERATION.md); \
             bound the edit with a smaller TURBOSPARK_STEERING_SCALE or a layer band \
             before reading arm 2 as a result.",
            distinct.len(),
            if run_a.first() == Some(&tokenizer.end_of_turn_id) {
                ", starting with end-of-turn"
            } else {
                ""
            }
        );
    } else {
        println!(
            "  steered turn is not collapsed: {} distinct tokens, no immediate end-of-turn",
            distinct.len()
        );
    }

    println!(
        "\nVERDICT: the edit is inert at alpha 0 (bit-identical, {GREEDY_TOKENS} tokens \
         deep), deterministic, and moves the distribution {:.0}x the shape floor at alpha \
         {alpha}{}. What this does NOT say is whether the direction names the concept it \
         was extracted for -- that is a judgement about text, not a number.",
        steered_kl / DENSE_SHAPE_FLOOR_NATS,
        if collapsed {
            " -- BUT THE TURN COLLAPSED, so that multiple describes a model that stopped \
             generating rather than one that was steered"
        } else {
            ""
        }
    );
}
