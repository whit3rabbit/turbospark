#![cfg(target_os = "macos")]
// This module is compiled into three separate test binaries and each uses
// a different part of it (`quality_sensitivity` wants only
// `measure_perplexity`), so per-binary dead code here means nothing.
#![allow(dead_code)]
//! The quality gate's body, shared by the per-family targets
//! (`quality_gate.rs`, `qwen36_quality_gate.rs`). ROADMAP Phase Q's first
//! two deliverables: a perplexity number and frozen golden digests per
//! real install.
//!
//! WHAT THIS IS FOR. Every other gate in this repo checks that generation
//! RAN (token counts, stop reason, footprint, tok/s) or that a human found
//! the text coherent. Neither notices quality drifting a few percent,
//! which is what a quantization change does when it is subtly wrong rather
//! than broken. Phase S (sub-4-bit experts) cannot be judged without a
//! number here, which is why Phase Q gates it.
//!
//! ONLY ASSISTANT-POSITION TOKENS ARE SCORED, and that is not a detail.
//! Both supported checkpoints are instruction-tuned, and instruction
//! tuning masks the loss on the prompt: the model is never trained to
//! predict the user's turn or the turn markup that closes it. Measured
//! here first, on the real Gemma 4 install, teacher-forcing the PROMPT
//! text gave a mean NLL of 15.3 nats against a uniform-distribution bound
//! of 12.5 -- literally worse than guessing -- while the assistant-side
//! markup at the end of the same sequence scored 0.000. It was not a bug
//! in the measurement (a replay of the model's own greedy output agreed
//! 39 times in 40, the one miss a genuine near-tie at 0.96 nats); it is
//! what an SFT checkpoint does outside the region it was trained on. So
//! the corpus is a fixed reference ANSWER, teacher-forced into the
//! assistant slot after the frozen protocol's own first prompt, and only
//! its tokens are scored.
//!
//! WHAT THE NUMBER IS NOT. One install, one fixed passage, this port's
//! own tokenizer and chat template. It is a REGRESSION SENTINEL against
//! this port's past, not a figure comparable to a published wikitext
//! perplexity. Do not quote it against anyone else's number.
//!
//! ONE MODEL PER PROCESS, the same rule and the same reason as
//! `oracle_common`: two `#[test]`s in one binary run on parallel threads,
//! so a second family opened here would double the resident footprint of
//! a test whose whole point is to run a real install. Separate
//! integration targets are separate binaries, and cargo runs those one
//! after another.
//!
//! DIGESTS DEPEND ON THE EXPERT CACHE STATE, so the run order below is
//! part of the frozen protocol. Each layer's routed slots are ordered
//! cache misses first, then hits, which permutes the phase-2 reduce order,
//! and FP addition is not associative. A cold-cache generation therefore
//! does NOT match a warm one: measured here, the first greedy run of a
//! process and the third produced different digests from identical
//! settings. Every digest below is taken after a discarded warmup of the
//! same generation, which is the same discipline the throughput protocol
//! already uses, and everything is pinned to
//! `PROTOCOL_EXPERT_CACHE_SLOTS`.
//!
//! THE CONSTRAINED-WORKING-SET ARM (Phase Q's fourth deliverable) reopens
//! the same install at `PRESSURE_EXPERT_CACHE_SLOTS` and repeats the greedy
//! digest. The expert cache is the port's memory knob -- roughly 3.2 MB of
//! pinned host memory per slot per layer, the axis that took footprint from
//! ~2.1 to ~3.7 GiB going 16 -> 32 slots -- so halving it IS the memory cap
//! that upstream's acceptance proof constrains, and it needs no root, no
//! balloon process, and no OS memory-pressure simulation.
//!
//! Upstream states that proof as "byte-identical output at unchanged
//! throughput under a constrained working set", and BOTH FAMILIES NOW MEET
//! IT LITERALLY: this arm asserts the 8-slot digest EQUALS the 16-slot one,
//! not that it matches a second frozen golden.
//!
//! It did not always. Gemma used to order a layer's routed slots misses
//! first then hits, so the resident hits' phase-1 GEMV could ride its own
//! command buffer; that order fed phase 2's reduce, and FP addition is not
//! associative, so Gemma's bytes moved with the hit/miss split and the two
//! slot counts needed two rows. Worse, the hit/miss split is a function of
//! CACHE STATE rather than of the prompt, so two warm greedy runs of one
//! prompt in one process could differ (2026-08-08: 4 distinct outputs in 6
//! runs on a Q8_0 GGUF install at 16 slots, 2 in 6 on the MLX install at
//! 32 -- this gate had only ever run at 16 and 8). The slots are now
//! dispatched in the router's own ranking, which depends on the route
//! alone, and the whole class went away with it.

use std::path::Path;

use foundation::LogitValue;
use runtime::{
    run_raw_completion, GenerationConfig, LogitProducer, RawDecodeProgress, RealForwardRunner,
};
use selection::ShapingConfig;
use tokenizer::{Message, MfTokenizer, Role};
use turbospark_bench::memory::chip_brand_string;
use turbospark_bench::protocol::{
    PROTOCOL_CASES, PROTOCOL_EXPERT_CACHE_SLOTS, PROTOCOL_MAX_CONTEXT, PROTOCOL_TEMPERATURE,
    PROTOCOL_TOP_K, PROTOCOL_TOP_P,
};
use turbospark_bench::real_model::open_model_runner;

/// The reference answer, teacher-forced into the assistant slot. Original
/// prose written for this repo, so it is not in any training set verbatim
/// and carries no licence question. It answers the frozen protocol's
/// `short-explanation` prompt, which is what goes in the user slot, so the
/// pair reads as a real turn rather than as text dropped into a template.
/// `pub` for `logit_dump.rs`, which teacher-forces the same sequence and
/// writes out the per-position logits instead of collapsing them to a
/// perplexity. Same corpus on purpose: a cross-engine KLD and the
/// perplexity number are then two readings of one forward pass.
pub const REFERENCE_ANSWER: &str = include_str!("../../prompts/quality-v1/assistant-reference.txt");

/// A recorded quality baseline for one chip and one install.
///
/// Chip-keyed for the same reason the memory oracle's rows are: the GPU
/// reduce order is a property of the device, so neither the digests nor
/// the last digits of the perplexity are portable. A chip with no row
/// still runs and still asserts determinism; it just has nothing to
/// compare against.
pub struct ChipQuality {
    pub brand_substr: &'static str,
    pub perplexity: f64,
    pub greedy_digest: &'static str,
    pub sampled_digest: &'static str,
    /// Where the row came from: date, chip, power source. Printed every
    /// run and quoted in every failure. No row here is or can be a Swift
    /// parity claim -- the Swift original publishes no quality numbers at
    /// all, which is the gap Phase Q exists to close.
    pub source: &'static str,
}

/// Perplexity moves in its last digits with the GPU reduce order and with
/// anything that changes FP16 rounding in the head, neither of which is a
/// quality regression. 2% is far under the smallest degradation worth
/// acting on (routed experts at 3-bit are expected to move this by whole
/// percent) and far over the run-to-run noise of one binary.
pub const PERPLEXITY_REL_TOLERANCE: f64 = 0.02;

/// Digest generation length. Long enough that a distribution bug diverges
/// the continuation, short enough that five generations plus the
/// perplexity pass stay inside a few minutes.
const DIGEST_MAX_NEW: u32 = 256;

/// The constrained arm's expert-cache size: half the protocol's, and the
/// smallest value `--expert-cache-slots` allows. Below `num_experts`, so it
/// forces a real hit/miss mix rather than caching the whole table
/// (AGENTS.md Gotcha 12).
const PRESSURE_EXPERT_CACHE_SLOTS: usize = 8;

/// How much decode throughput the constrained arm is allowed to lose.
///
/// Deliberately loose. The property being asserted is that halving the
/// working set does not COLLAPSE decode (every miss re-reads an expert, so
/// some loss is the mechanism working, not a regression); a tight band here
/// would flake on machine state, which is the failure mode the throughput
/// rules in CLAUDE.local.md exist to avoid. Prefill attribution at 32 -> 8
/// slots moved wall clock by 33%, so 2x is several times the observed
/// effect.
const PRESSURE_THROUGHPUT_FLOOR_RATIO: f64 = 0.5;

/// The gate's first arm on its own: open the install at `dir` and return
/// the teacher-forced perplexity of the frozen corpus.
///
/// Split out for `quality_sensitivity.rs`, which needs the same number from
/// a deliberately damaged copy of an install. The runner is dropped before
/// returning so a caller can measure two installs in one process without
/// holding both resident.
///
pub fn measure_perplexity(dir: &Path) -> f64 {
    let (mut runner, tokenizer) =
        open_model_runner(dir, PROTOCOL_EXPERT_CACHE_SLOTS).expect("install should open");
    let prompt_ids = user_turn_ids(&tokenizer);
    reference_perplexity(&mut runner, &tokenizer, &prompt_ids)
}

/// Run Phase Q's in-repo half against `dir` and assert what `rows` records
/// for this chip.
pub fn run_quality_gate(dir: &Path, rows: &[ChipQuality]) {
    let brand = chip_brand_string();
    let row = brand
        .as_deref()
        .and_then(|b| rows.iter().find(|row| b.contains(row.brand_substr)));
    match row {
        Some(row) => eprintln!(
            "quality_gate: chip {:?} -> asserting against a recorded row (source: {})",
            brand, row.source
        ),
        None => eprintln!(
            "quality_gate: chip {brand:?} has no recorded row -> everything is \
             measured and printed, only determinism is asserted"
        ),
    }

    let (mut runner, tokenizer) =
        open_model_runner(dir, PROTOCOL_EXPERT_CACHE_SLOTS).expect("real install should open");
    let prompt_ids = user_turn_ids(&tokenizer);

    // 1. Perplexity of the reference answer in the assistant slot.
    let perplexity = reference_perplexity(&mut runner, &tokenizer, &prompt_ids);
    eprintln!("quality_gate: reference-answer perplexity {perplexity:.4}");

    // 2. The two digests. Exactly 0.0 is the argmax fast path; 0.0001 is
    //    NOT (AGENTS.md Gotcha 16, selection Gotcha 6), and a golden
    //    digest wants the path with no RNG in it at all.
    let greedy = ShapingConfig::new(0.0, 1, None, 1.0, None).expect("greedy shaping is valid");
    let sampled = ShapingConfig::new(
        PROTOCOL_TEMPERATURE,
        PROTOCOL_TOP_K,
        Some(PROTOCOL_TOP_P),
        1.0,
        Some(PROTOCOL_CASES[0].seed),
    )
    .expect("protocol shaping is valid");

    // Warmup, discarded: a cold expert cache generates different bytes
    // (see the module doc).
    generation_digest(&mut runner, &tokenizer, &prompt_ids, &greedy);
    let (greedy_digest, baseline_tok_s) =
        generation_digest(&mut runner, &tokenizer, &prompt_ids, &greedy);
    let (greedy_again, _) = generation_digest(&mut runner, &tokenizer, &prompt_ids, &greedy);
    generation_digest(&mut runner, &tokenizer, &prompt_ids, &sampled);
    let (sampled_digest, _) = generation_digest(&mut runner, &tokenizer, &prompt_ids, &sampled);
    eprintln!("quality_gate: greedy  digest {greedy_digest}");
    eprintln!("quality_gate: sampled digest {sampled_digest}");

    // 3. Determinism. The one assertion that holds on any chip, and what
    //    makes freezing a digest meaningful in the first place: a digest
    //    that does not reproduce inside one session cannot regress
    //    detectably across sessions either.
    assert_eq!(
        greedy_digest, greedy_again,
        "two warm greedy runs of the same prompt at the same settings \
         produced different output in one session: generation is not \
         deterministic, so no golden digest can hold"
    );

    // 4. The constrained working set. The 16-slot runner is dropped first
    //    so its slot capacity and its resident mapping are released before
    //    the second one is opened: two live runners would double the
    //    footprint of a test whose whole point is a real install.
    drop(runner);
    let (mut constrained, _) = open_model_runner(dir, PRESSURE_EXPERT_CACHE_SLOTS)
        .expect("real install should reopen with a smaller expert cache");
    generation_digest(&mut constrained, &tokenizer, &prompt_ids, &greedy);
    let (constrained_digest, constrained_tok_s) =
        generation_digest(&mut constrained, &tokenizer, &prompt_ids, &greedy);
    let (constrained_again, _) =
        generation_digest(&mut constrained, &tokenizer, &prompt_ids, &greedy);
    let throughput_ratio = constrained_tok_s / baseline_tok_s;
    eprintln!(
        "quality_gate: constrained ({PRESSURE_EXPERT_CACHE_SLOTS} slots) digest \
         {constrained_digest}, {constrained_tok_s:.3} tok/s against \
         {baseline_tok_s:.3} at {PROTOCOL_EXPERT_CACHE_SLOTS} ({:.2}x)",
        throughput_ratio
    );
    assert_eq!(
        constrained_digest, constrained_again,
        "two warm greedy runs at {PRESSURE_EXPERT_CACHE_SLOTS} expert-cache \
         slots produced different output in one session: a constrained \
         working set is not deterministic, which is a stronger failure than \
         any digest drift"
    );
    // The slot order is the router's ranking, which the cache cannot reach,
    // so halving the working set must not move a single byte. This is the
    // assertion that fails if misses-first ordering ever comes back.
    assert_eq!(
        constrained_digest, greedy_digest,
        "greedy output at {PRESSURE_EXPERT_CACHE_SLOTS} expert-cache slots \
         differs from the same generation at {PROTOCOL_EXPERT_CACHE_SLOTS}: \
         the routed-slot dispatch order has become a function of cache state \
         again, so output depends on how many experts happened to be resident"
    );
    assert!(
        throughput_ratio >= PRESSURE_THROUGHPUT_FLOOR_RATIO,
        "halving the expert cache to {PRESSURE_EXPERT_CACHE_SLOTS} slots cut \
         decode to {constrained_tok_s:.3} tok/s from {baseline_tok_s:.3} \
         ({:.2}x, floor {PRESSURE_THROUGHPUT_FLOOR_RATIO:.2}x): throughput \
         collapses under a constrained working set rather than degrading",
        throughput_ratio
    );

    // 5. The recorded row, when this chip has one.
    let Some(row) = row else {
        eprintln!(
            "quality_gate: nothing asserted beyond determinism. To freeze these, \
             add a ChipQuality row for {brand:?}."
        );
        return;
    };
    let drift = (perplexity - row.perplexity).abs() / row.perplexity;
    assert!(
        drift <= PERPLEXITY_REL_TOLERANCE,
        "reference-answer perplexity {perplexity:.4} drifted {:.2}% from the \
         recorded {:.4} (tolerance {:.0}%, source: {}): model quality moved",
        drift * 100.0,
        row.perplexity,
        PERPLEXITY_REL_TOLERANCE * 100.0,
        row.source
    );
    assert_eq!(
        greedy_digest, row.greedy_digest,
        "greedy output changed against the recorded golden (source: {}). \
         Deterministic settings, so this is a real change in what the model \
         emits, not sampling noise.",
        row.source
    );
    assert_eq!(
        sampled_digest, row.sampled_digest,
        "sampled output changed against the recorded golden (source: {}). \
         The sampled arm sees distribution bugs greedy cannot (AGENTS.md \
         Gotcha 16); the greedy line above tells the two apart.",
        row.source
    );
}

/// The user turn, chat-formatted exactly as `run_protocol_case` and the
/// CLI format it (the dialect template renders the turn markup and its own
/// `<bos>`, hence `add_bos = false`). Its last token is the position the
/// model would start answering from.
pub fn user_turn_ids(tokenizer: &MfTokenizer) -> Vec<i32> {
    let messages = [Message::new(Role::User, PROTOCOL_CASES[0].content)];
    let rendered = tokenizer
        .apply_chat_template(&messages)
        .expect("chat template renders");
    tokenizer.encode(&rendered, false)
}

/// Teacher-forced perplexity of [`REFERENCE_ANSWER`] in the assistant slot
/// after `prompt_ids`.
///
/// `produce` writes the logits for the NEXT position, so feeding token `i`
/// scores token `i + 1`. Scoring starts at the last prompt token, whose
/// logits predict the first answer token, and every scored target is
/// therefore assistant-side. `produce_prefill` must NOT be used anywhere
/// here: it is allowed to skip the output head, which is the only thing
/// this function reads.
fn reference_perplexity(
    runner: &mut RealForwardRunner,
    tokenizer: &MfTokenizer,
    prompt_ids: &[i32],
) -> f64 {
    let answer_ids = tokenizer.encode(REFERENCE_ANSWER, false);
    assert!(!answer_ids.is_empty(), "the reference answer must tokenize");
    let mut ids = prompt_ids.to_vec();
    ids.extend(&answer_ids);
    assert!(
        ids.len() <= PROTOCOL_MAX_CONTEXT as usize,
        "prompt plus reference answer is {} tokens, over the \
         {PROTOCOL_MAX_CONTEXT}-token KV the runner was opened with",
        ids.len()
    );

    runner.reset();
    // The MODEL's head width, not the tokenizer dialect's constant: two
    // checkpoints can share a dialect and pad their heads differently
    // (`RealForwardRunner::vocab_size`).
    let mut logits = vec![LogitValue::from_f32(0.0); runner.vocab_size()];
    let mut nll_sum = 0.0f64;
    let first_scored = prompt_ids.len() - 1;
    for (position, &token) in ids.iter().take(ids.len() - 1).enumerate() {
        runner
            .produce(token, position, &mut logits)
            .expect("forward pass over the corpus");
        if position >= first_scored {
            nll_sum += negative_log_prob(&logits, ids[position + 1]);
        }
    }
    (nll_sum / answer_ids.len() as f64).exp()
}

/// `-log softmax(logits)[target]`, as `logsumexp(z) - z[target]`.
///
/// Accumulated in f64 over the full vocabulary (262,144 on both families),
/// max-subtracted so the exponentials cannot overflow. The logits arrive
/// softcapped where the family caps (`softcap * tanh(z / softcap)`), which
/// is what `*ForCausalLM.forward` returns upstream too, so this is the
/// distribution the sampler actually sees.
fn negative_log_prob(logits: &[LogitValue], target: i32) -> f64 {
    let target = target as usize;
    assert!(
        target < logits.len(),
        "target token id is outside the vocab"
    );
    let max = logits
        .iter()
        .fold(f32::NEG_INFINITY, |acc, &v| acc.max(v.to_f32()));
    assert!(
        max.is_finite(),
        "the forward pass produced no finite logit; perplexity is meaningless"
    );
    let max = max as f64;
    let sum_exp: f64 = logits
        .iter()
        .map(|&v| (v.to_f32() as f64 - max).exp())
        .sum();
    (max + sum_exp.ln()) - logits[target].to_f32() as f64
}

/// Generate from `prompt_ids` and return the lowercase hex SHA-256 of the
/// visible text, which is every `Token` delta plus the final `Tail`
/// (`raw_completion.rs`'s decode loop emits each byte exactly once across
/// the two), plus decode tok/s.
///
/// The throughput is the loop's own `decode_seconds`, not wall clock, so it
/// excludes prefill and excludes the digest hashing -- the same number the
/// bench harness reports.
fn generation_digest(
    runner: &mut RealForwardRunner,
    tokenizer: &MfTokenizer,
    prompt_ids: &[i32],
    shaping: &ShapingConfig,
) -> (String, f64) {
    let config = GenerationConfig {
        shaping: *shaping,
        max_new_tokens: DIGEST_MAX_NEW,
        stop_strings: Vec::new(),
        extra_stop_tokens: Vec::new(),
        rate: Default::default(),
    };

    // Read before the mutable borrow: the logits width is the MODEL's,
    // not the tokenizer dialect's constant.
    let vocab_size = runner.vocab_size();
    let mut text = String::new();
    let result = run_raw_completion(
        runner,
        tokenizer,
        prompt_ids,
        &config,
        PROTOCOL_MAX_CONTEXT,
        vocab_size,
        |event| match event {
            RawDecodeProgress::Token { delta, .. } => text.push_str(&delta),
            RawDecodeProgress::Tail(tail) => text.push_str(&tail),
            RawDecodeProgress::Prefill { .. } => {}
        },
    )
    .expect("generation runs");

    assert!(
        result.decode_seconds > 0.0 && result.new_tokens > 0,
        "generation produced no timed decode, so its throughput is undefined"
    );
    (
        model_io::hash_data(text.as_bytes()),
        result.new_tokens as f64 / result.decode_seconds,
    )
}
