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
//! # The two other axes, and why they are env vars rather than files
//!
//! `TURBOSPARK_STEERING_MODE` overrides the edit. Without it the mode comes
//! from the vector file's own `declared_mode`, so comparing two modes would
//! mean writing two vectors -- and then the comparison has two variables in
//! it, the mode and the artifact. One file swept twice is single-variable.
//! An unknown value is REFUSED and never falls back to the file's mode, for
//! the reason `SteeringMode::parse` gives: a caller who asked for one edit
//! and silently got another measures the wrong model and reports it as the
//! right one.
//!
//! `TURBOSPARK_STEERING_BANDS` sweeps the LAYER BAND beside alpha --
//! comma-separated, `all` or `START:END` inclusive and 0-based. It is the
//! other lever on the collapse: `docs/OBLITERATION.md`'s stream-share column
//! says the damage concentrates in the late layers, so a band excluding them
//! should push the usable alpha up. The default is a single `all` arm, so a
//! plain run is unchanged.
//!
//! **A band covering ZERO layers is refused rather than run.** It steers
//! nothing, so every alpha would read as perfectly usable and the table would
//! report a flawless result for an engine doing nothing -- an instrument
//! returning a plausible value on degenerate input, which is the failure this
//! whole page keeps meeting (AGENTS.md Gotchas 30, 57, 59).
//!
//! Six opens of one install, ~2 min for the default single band;
//! `bands x alphas` opens otherwise, at roughly 20 s each.
//! `TURBOSPARK_STEERING_ALPHAS` overrides the alpha sweep as a
//! comma-separated list.

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

/// The prompt every arm generates from unless `TURBOSPARK_STEERING_PROMPT`
/// says otherwise.
///
/// Kept verbatim so this page's frozen sweep tables reproduce. It is
/// ocean-adjacent because the first direction measured on this surface was --
/// and a sweep is where that coupling does real damage, because the usable
/// band is read off where the output degrades, and a direction the prompt
/// never invites has less to degrade.
const DEFAULT_STEERING_PROMPT: &str = "Describe what you notice about the water.";

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
    band: usize,
    alpha: f32,
    tokens: Vec<i32>,
    /// Mean NLL under the UNSTEERED model, filled in phase B.
    nll: f64,
}

/// One layer band of the sweep: the whole covered range, or a restriction of
/// it applied through `SteeringSet::restrict_to_range` -- the same call the
/// CLI's `--steering-layers` and the server's make, so a band measured here
/// is a band a caller can actually ask for.
struct Band {
    label: String,
    /// `None` is every layer the vector covers.
    range: Option<(usize, usize)>,
    covered: usize,
}

/// Parses `all` or `START:END` (inclusive, 0-based), the spelling
/// `--steering-layers` takes.
fn parse_bands(raw: &str) -> Vec<(String, Option<(usize, usize)>)> {
    raw.split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| {
            if s == "all" {
                return (s.to_string(), None);
            }
            let (a, b) = s
                .split_once(':')
                .unwrap_or_else(|| panic!("band {s:?} is not `all` or `START:END`"));
            let start: usize = a.trim().parse().expect("band start parses");
            let end: usize = b.trim().parse().expect("band end parses");
            assert!(start <= end, "band {s:?} runs backwards");
            (s.to_string(), Some((start, end)))
        })
        .collect()
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

    // REFUSED on an unknown spelling rather than falling back to the file's
    // declaration: silently measuring a different edit than the one asked for
    // is the failure `SteeringMode::parse` returns `None` to prevent, and a
    // sweep is exactly where it would go unnoticed.
    let mode = match std::env::var("TURBOSPARK_STEERING_MODE") {
        Ok(raw) => foundation::SteeringMode::parse(raw.trim()).unwrap_or_else(|| {
            panic!(
                "TURBOSPARK_STEERING_MODE={raw:?} is not a steering mode. \
                 Accepted: ablate, add, clamp, renorm."
            )
        }),
        Err(_) => set.declared_mode.unwrap_or_default(),
    };

    let bands: Vec<Band> = parse_bands(
        &std::env::var("TURBOSPARK_STEERING_BANDS").unwrap_or_else(|_| "all".to_string()),
    )
    .into_iter()
    .map(|(label, range)| {
        let mut restricted = set.clone();
        if let Some((start, end)) = range {
            restricted.restrict_to_range(start, end);
        }
        let covered = restricted.covered_layers();
        // A band covering nothing steers nothing, so every alpha under it
        // would score like the unsteered model and the table would report a
        // perfect result for an engine doing no work.
        assert!(
            covered > 0,
            "band {label:?} covers 0 of the vector's {} layers, so it would steer nothing \
             and every alpha under it would read as usable",
            set.covered_layers()
        );
        Band {
            label,
            range,
            covered,
        }
    })
    .collect();

    println!(
        "steering_sweep: install {}\n  vector {} ({} covered layers), mode {}, {} tokens per arm",
        dir.display(),
        vector.display(),
        set.covered_layers(),
        mode.as_str(),
        GENERATE,
    );
    println!(
        "  {} layer band(s): {}",
        bands.len(),
        bands
            .iter()
            .map(|b| format!("{} ({} layers)", b.label, b.covered))
            .collect::<Vec<_>>()
            .join(", ")
    );

    let prompt_text = std::env::var("TURBOSPARK_STEERING_PROMPT")
        .unwrap_or_else(|_| DEFAULT_STEERING_PROMPT.to_string());
    let prompt_text = prompt_text.as_str();
    println!("  prompt {prompt_text:?}");

    // ---- PHASE A: generate every steered arm, ONE RUNNER ALIVE AT A TIME ----
    //
    // Deliberately not "open the unsteered engine and keep it while the
    // steered ones come and go", which would be the obvious shape. Alpha is
    // fixed at OPEN (`SteeringState` stores it), so a sweep is one open per
    // point either way, and holding two engines buys nothing while putting
    // two KV caches and two Metal contexts in flight at once.
    let mut arms: Vec<Arm> = Vec::with_capacity(alphas.len() * bands.len());
    let mut prompt: Vec<i32> = Vec::new();
    for (band_idx, band) in bands.iter().enumerate() {
        for &alpha in &alphas {
            let mut banded = set.clone();
            if let Some((start, end)) = band.range {
                banded.restrict_to_range(start, end);
            }
            let policy = runtime::SteeringPolicy {
                set: Some(banded),
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
            println!(
                "  band {:>7} alpha {alpha:>4}: generated {} tokens",
                band.label,
                tokens.len()
            );
            arms.push(Arm {
                band: band_idx,
                alpha,
                tokens,
                nll: f64::NAN,
            });
        }
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
    //
    // Once per BAND, and that is stronger than once overall: alpha 0 must be
    // inert whatever subset of layers the dispatch runs over, so a band that
    // failed here would be a restriction bug rather than an edit.
    for (band_idx, band) in bands.iter().enumerate() {
        let zero = arms
            .iter()
            .find(|a| a.band == band_idx && a.alpha == 0.0)
            .expect("every band has an alpha 0 arm");
        assert_eq!(
            zero.tokens, unsteered,
            "band {} at alpha 0 produced a different {GENERATE}-token continuation than \
             steering off. That arm is this sweep's zero point, so until it matches, every \
             NLL below is measuring the kernel rather than the edit.",
            band.label
        );
    }

    println!(
        "\nanchor: the unsteered model on the frozen reference answer reads perplexity \
         {anchor_ppl:.4}.\n  That is what ORDINARY FLUENT PROSE costs it. Read the column \
         against this, not against 1.0."
    );

    // ---- THE USABLE BAND, per layer band, and it is the deliverable ----
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
    // to that model than human writing is. The fluent arms sit at 0.3x the
    // anchor. An arm whose own greedy output is as surprising as human prose
    // has stopped producing its own distribution's typical text.
    //
    // Every verdict below is "on THIS direction, THIS prompt, THIS mode and
    // THIS band". `scripts/extract_direction.py` predicts a ceiling from the
    // captures alone, and for `ablate` over all layers the two should agree.
    let mut verdicts: Vec<(String, Option<f32>)> = Vec::with_capacity(bands.len());
    for (band_idx, band) in bands.iter().enumerate() {
        let rows: Vec<&Arm> = arms.iter().filter(|a| a.band == band_idx).collect();
        println!(
            "\n=== mode {}, layers {} ({} covered) ===",
            mode.as_str(),
            band.label,
            band.covered
        );
        println!(
            "  {:>5}  {:>10}  {:>9}  {:>8}  first divergence",
            "alpha", "ppl", "x anchor", "distinct"
        );
        for arm in &rows {
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

        println!("  what each arm said:");
        for arm in &rows {
            println!(
                "    alpha {:>4}: {:?}",
                arm.alpha,
                tokenizer.decode(&arm.tokens, true)
            );
        }

        let usable = rows
            .iter()
            .take_while(|a| a.nll.exp() < anchor_ppl)
            .last()
            .map(|a| a.alpha);

        // ---- THE SECOND SIGNAL, AND WHAT TO DO WHEN IT DISAGREES ----
        //
        // The anchor crossing is the criterion and it is not sufficient on
        // its own. MEASURED on the real `qwen38-27b` at a fine grid: `ablate`
        // at alpha 0.55 reads 2.1309, i.e. 0.4x the anchor and comfortably
        // "usable", while its actual output is one sentence repeated with
        // template markup between the copies -- 18 distinct tokens against
        // the unsteered arm's 34. The crossing UNDER-CALLS the damage there,
        // which is the third time a verdict rule on this page has been wrong
        // in a way only the text revealed (see the steepest step, and the
        // probe's alpha-1.0 default).
        //
        // `distinct` is the independent signal, and the honest thing is to
        // REPORT THE DISAGREEMENT rather than invent a second threshold to
        // resolve it (Gotcha 38's rule -- there is no measured basis for a
        // "distinct must stay above N" line). So this prints the largest
        // consecutive drop and says when it lands at or before the crossing.
        // Nothing here is asserted: an operator choosing a strength wants
        // both numbers, not one of them silently preferred.
        let mut biggest_drop = 0i64;
        let mut drop_at = 0usize;
        let counts: Vec<usize> = rows
            .iter()
            .map(|a| {
                a.tokens
                    .iter()
                    .copied()
                    .collect::<std::collections::BTreeSet<i32>>()
                    .len()
            })
            .collect();
        for i in 1..counts.len() {
            let d = counts[i - 1] as i64 - counts[i] as i64;
            if d > biggest_drop {
                biggest_drop = d;
                drop_at = i;
            }
        }
        match usable {
            Some(alpha) => println!("  USABLE BAND: up to alpha {alpha}"),
            None => println!("  NO USABLE BAND: even the lowest alpha scores above the anchor"),
        }
        if biggest_drop > 0 {
            println!(
                "  distinct tokens fall hardest {} -> {} (alpha {} -> {}), against {} at alpha 0",
                counts[drop_at - 1],
                counts[drop_at],
                rows[drop_at - 1].alpha,
                rows[drop_at].alpha,
                counts[0]
            );
            if let Some(crossing) = usable {
                if rows[drop_at].alpha <= crossing {
                    println!(
                        "  ** THE TWO SIGNALS DISAGREE. The vocabulary collapses at or BELOW the\n  \
                         anchor crossing, so the crossing is the OPTIMISTIC reading here and the\n  \
                         top of this band is likely already degenerate. Read the text."
                    );
                }
            }
        }

        // Reported SECOND and explicitly not the criterion, so nobody
        // reinstates it as one.
        let mut worst_step = 0.0f64;
        let mut steepest = 1usize;
        for i in 1..rows.len() {
            let ratio = rows[i].nll.exp() / rows[i - 1].nll.exp().max(f64::MIN_POSITIVE);
            if ratio > worst_step {
                worst_step = ratio;
                steepest = i;
            }
        }
        if rows.len() > 1 {
            println!(
                "  (steepest step alpha {} -> {} at {worst_step:.1}x, INSIDE the damage and \
                 therefore not the criterion)",
                rows[steepest - 1].alpha,
                rows[steepest].alpha
            );
        }
        verdicts.push((band.label.clone(), usable));
    }

    // ---- THE CROSS-BAND COMPARISON, which is what the band axis is FOR ----
    //
    // `docs/OBLITERATION.md`'s stream-share column predicts the damage is
    // concentrated in the late layers, so a band excluding them should carry a
    // HIGHER usable alpha than `all`. Printed rather than asserted: the
    // prediction is what is under test, and a test that asserted it could only
    // ever confirm it.
    if verdicts.len() > 1 {
        println!("\nusable alpha by layer band (mode {}):", mode.as_str());
        for (label, usable) in &verdicts {
            match usable {
                Some(a) => println!("  {label:>10}: up to {a}"),
                None => println!("  {label:>10}: none"),
            }
        }
        println!(
            "  A band that excludes the high-share layers should read HIGHER than `all`.\n  \
             That is the stream-share column's prediction, and this table is what tests it."
        );
    }

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
