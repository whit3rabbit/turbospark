//! The speculative decode loop's control flow, against a scripted target and
//! a scripted drafter.
//!
//! **THE BAR IS BYTE-IDENTITY WITH THE SEQUENTIAL LOOP, and the reference
//! arm is `run_raw_completion` rather than a second speculative run.**
//! Comparing two speculative arms against each other passes whenever both
//! are wrong the same way, which is exactly what happened to the first draft
//! of `mtp_accept_length_probe`'s losslessness gate (`docs/MTP.md`).
//!
//! What a scripted producer can prove here is the half a real install cannot
//! cheaply prove: that the loop is lossless at EVERY drafter quality
//! including the degenerate ones, that it never leaves the engine's cursor
//! disagreeing with the tokens it reported, and that the stop ladder,
//! the budget and cancellation behave as they do sequentially even when the
//! stop lands in the middle of an accepted block. A real drafter accepts
//! some middling fraction and cannot be steered to those corners.

use std::cell::Cell;
use std::path::PathBuf;

use foundation::LogitValue;
use selection::ShapingConfig;
use tokenizer::MfTokenizer;
use turbospark_runtime::{
    run_raw_completion, run_raw_completion_speculative, run_raw_completion_speculative_cancellable,
    GenerationConfig, LogitProducer, RawDecodeProgress, RawDecodeResult, RuntimeError,
    SpeculativeProducer, StopReason,
};

fn load_tokenizer() -> MfTokenizer {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/ChatMLTokenizer");
    MfTokenizer::load_from_dir(&dir).expect("fixture tokenizer should load")
}

fn greedy_config(max_new_tokens: u32) -> GenerationConfig {
    GenerationConfig {
        shaping: ShapingConfig::new(0.0, 0, None, 1.0, None).unwrap(),
        max_new_tokens,
        stop_strings: Vec::new(),
        extra_stop_tokens: Vec::new(),
        rate: Default::default(),
    }
}

fn sampled_config(max_new_tokens: u32) -> GenerationConfig {
    GenerationConfig {
        shaping: ShapingConfig::new(0.7, 64, Some(0.95), 1.0, Some(7)).unwrap(),
        max_new_tokens,
        stop_strings: Vec::new(),
        extra_stop_tokens: Vec::new(),
        rate: Default::default(),
    }
}

/// How good the scripted drafter is, which is the axis the loop has to be
/// lossless across.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Drafter {
    /// Proposes exactly what the target will say. Every proposal is
    /// accepted, so no round ever rolls back -- the case that exercises the
    /// extra drafter step taken for its cache row alone.
    Perfect,
    /// Never proposes what the target will say. Every round rejects at
    /// position 0, so every round rolls back and the committed stream must
    /// still come out identical.
    Useless,
    /// Right on even positions and wrong on odd ones, so acceptance lands
    /// strictly inside the block and the rollback replays a non-empty
    /// prefix.
    Alternating,
}

/// A target model that is a pure function of POSITION, plus a drafter of
/// selectable quality.
///
/// Making the target ignore the token it is fed is what makes the reference
/// arm meaningful: the sequential and speculative runs then have to agree on
/// the stream for reasons of control flow alone, and any disagreement is the
/// loop's doing rather than a scripted script drifting between arms.
///
/// It also POLICES ITS OWN CURSOR. Every `produce` and `verify` asserts it
/// was called at the position the engine has actually absorbed, which is the
/// property no output comparison can see: a loop that rolls back too little
/// leaves the cursor ahead of the tokens it reported, so a caller continuing
/// from `kv_backed_token_ids` would resume against a cache holding rows for
/// tokens that were never committed.
struct ScriptedTarget {
    vocab: usize,
    truth: Vec<i32>,
    drafter: Drafter,
    /// Positions absorbed by the target.
    cursor: usize,
    /// Positions absorbed by the drafter.
    drafter_cursor: usize,
    /// Counts, so a test can assert the fast path was actually taken.
    verify_calls: Cell<usize>,
    verify_rows: Cell<usize>,
    /// The furthest position ever written. A speculative round feeds
    /// `block + 1` positions BEFORE knowing how many it will keep, so this
    /// is the only way to see a round that wrote KV rows past the window the
    /// caller sized for.
    high_water: Cell<usize>,
}

impl ScriptedTarget {
    fn new(vocab: usize, truth: Vec<i32>, drafter: Drafter) -> Self {
        Self {
            vocab,
            truth,
            drafter,
            cursor: 0,
            drafter_cursor: 0,
            verify_calls: Cell::new(0),
            verify_rows: Cell::new(0),
            high_water: Cell::new(0),
        }
    }

    /// What the target says after the token at `position`.
    fn truth_at(&self, position: usize) -> i32 {
        self.truth[position % self.truth.len()]
    }

    fn write_one_hot(&self, id: i32, out: &mut [LogitValue]) {
        out.fill(LogitValue::from_f32(0.0));
        out[id as usize] = LogitValue::from_f32(1.0);
    }
}

impl LogitProducer for ScriptedTarget {
    fn reset(&mut self) {
        self.cursor = 0;
        self.drafter_cursor = 0;
    }

    fn produce(
        &mut self,
        _token: i32,
        position: usize,
        logits: &mut [LogitValue],
    ) -> Result<(), String> {
        assert_eq!(
            position, self.cursor,
            "target fed out of order: the engine has absorbed {} positions",
            self.cursor
        );
        let id = self.truth_at(position);
        self.write_one_hot(id, logits);
        self.cursor += 1;
        self.high_water.set(self.high_water.get().max(self.cursor));
        Ok(())
    }
}

impl SpeculativeProducer for ScriptedTarget {
    /// The scripted engine's whole restorable state.
    type Checkpoint = (usize, usize);

    fn prime_drafter(&mut self, _next: i32, position: usize) -> Result<(), String> {
        assert_eq!(position, self.drafter_cursor, "drafter primed out of order");
        self.drafter_cursor += 1;
        Ok(())
    }

    fn draft_step(
        &mut self,
        _token: i32,
        position: usize,
        logits: &mut [LogitValue],
    ) -> Result<(), String> {
        assert_eq!(
            position, self.drafter_cursor,
            "drafter stepped off its cursor"
        );
        // A proposal produced at `position` is checked against what the
        // target says after the token at `position + 1`.
        let id = match self.drafter {
            Drafter::Perfect => self.truth_at(position + 1),
            Drafter::Useless => {
                let wrong = (self.truth_at(position + 1) + 1) % self.vocab as i32;
                // Must stay a real, non-stop token or the test would be
                // measuring the stop ladder instead of acceptance.
                if wrong == 0 {
                    1
                } else {
                    wrong
                }
            }
            Drafter::Alternating => {
                if position % 2 == 0 {
                    self.truth_at(position + 1)
                } else {
                    let wrong = (self.truth_at(position + 1) + 1) % self.vocab as i32;
                    if wrong == 0 {
                        1
                    } else {
                        wrong
                    }
                }
            }
        };
        self.write_one_hot(id, logits);
        self.drafter_cursor += 1;
        Ok(())
    }

    fn rewind_drafter(&mut self, position: usize) -> Result<(), String> {
        assert!(
            position <= self.drafter_cursor,
            "drafter rewind target {position} is ahead of its cursor {}",
            self.drafter_cursor
        );
        self.drafter_cursor = position;
        Ok(())
    }

    fn checkpoint(&mut self) -> (usize, usize) {
        (self.cursor, self.drafter_cursor)
    }

    fn rollback(&mut self, point: &(usize, usize)) {
        assert!(
            point.0 <= self.cursor,
            "rollback target is ahead of the cursor"
        );
        self.cursor = point.0;
    }

    fn verify(
        &mut self,
        feed: &[i32],
        base: usize,
        logits: &mut [LogitValue],
    ) -> Result<(), String> {
        assert_eq!(
            base, self.cursor,
            "batched verify fed out of order: the engine has absorbed {} positions",
            self.cursor
        );
        assert_eq!(
            logits.len(),
            feed.len() * self.vocab,
            "verify was given the wrong number of rows"
        );
        self.verify_calls.set(self.verify_calls.get() + 1);
        self.verify_rows.set(self.verify_rows.get() + feed.len());
        for (i, _) in feed.iter().enumerate() {
            let id = self.truth_at(base + i);
            self.write_one_hot(id, &mut logits[i * self.vocab..(i + 1) * self.vocab]);
        }
        self.cursor += feed.len();
        self.high_water.set(self.high_water.get().max(self.cursor));
        Ok(())
    }
}

/// A continuation with no stop token in it, so a run ends on the budget.
fn plain_truth(tokenizer: &MfTokenizer) -> Vec<i32> {
    ["h", "e", "l", "o", "w", "r", "d"]
        .iter()
        .map(|t| {
            tokenizer
                .token_to_id(t)
                .unwrap_or_else(|| panic!("{t} is in the base vocab"))
        })
        .collect()
}

fn collect_tokens(events: &[RawDecodeProgress]) -> Vec<i32> {
    events
        .iter()
        .filter_map(|e| match e {
            RawDecodeProgress::Token { id, .. } => Some(*id),
            _ => None,
        })
        .collect()
}

fn run_sequential(
    tokenizer: &MfTokenizer,
    truth: &[i32],
    prompt: &[i32],
    config: &GenerationConfig,
    vocab: usize,
) -> (RawDecodeResult, Vec<i32>) {
    let mut producer = ScriptedTarget::new(vocab, truth.to_vec(), Drafter::Perfect);
    let mut events = Vec::new();
    let result = run_raw_completion(&mut producer, tokenizer, prompt, config, 4096, vocab, |e| {
        events.push(e)
    })
    .expect("sequential run");
    (result, collect_tokens(&events))
}

fn run_speculative(
    tokenizer: &MfTokenizer,
    truth: &[i32],
    prompt: &[i32],
    config: &GenerationConfig,
    vocab: usize,
    drafter: Drafter,
    block: usize,
) -> (RawDecodeResult, Vec<i32>, usize) {
    let mut producer = ScriptedTarget::new(vocab, truth.to_vec(), drafter);
    let mut events = Vec::new();
    let result = run_raw_completion_speculative(
        &mut producer,
        tokenizer,
        prompt,
        config,
        4096,
        vocab,
        block,
        |e| events.push(e),
    )
    .expect("speculative run");
    let calls = producer.verify_calls.get();
    // The cursor the engine really holds must be the one reported.
    assert_eq!(
        producer.cursor, result.kv_position,
        "reported kv_position disagrees with the engine's cursor"
    );
    assert_eq!(
        result.kv_backed_token_ids.len(),
        result.kv_position,
        "kv_backed_token_ids and kv_position describe different caches"
    );
    (result, collect_tokens(&events), calls)
}

#[test]
fn every_drafter_quality_reproduces_the_sequential_stream() {
    let tokenizer = load_tokenizer();
    let vocab = tokenizer.vocab_size;
    let truth = plain_truth(&tokenizer);
    let prompt = tokenizer.encode("hi", false);
    let config = greedy_config(12);

    let (seq, seq_tokens) = run_sequential(&tokenizer, &truth, &prompt, &config, vocab);
    assert_eq!(seq.reason, StopReason::MaxTokens);
    assert_eq!(seq_tokens.len(), 12);

    for drafter in [Drafter::Perfect, Drafter::Useless, Drafter::Alternating] {
        for block in [1usize, 2, 4, 7] {
            let (spec, spec_tokens, _) =
                run_speculative(&tokenizer, &truth, &prompt, &config, vocab, drafter, block);
            assert_eq!(
                spec_tokens, seq_tokens,
                "{drafter:?} at block {block} changed the token stream"
            );
            assert_eq!(spec.new_tokens, seq.new_tokens, "{drafter:?} block {block}");
            assert_eq!(spec.reason, seq.reason, "{drafter:?} block {block}");
            assert_eq!(
                spec.kv_backed_token_ids, seq.kv_backed_token_ids,
                "{drafter:?} block {block} left a different cache"
            );
        }
    }
}

#[test]
fn a_perfect_drafter_needs_fewer_verify_passes_than_tokens() {
    let tokenizer = load_tokenizer();
    let vocab = tokenizer.vocab_size;
    let truth = plain_truth(&tokenizer);
    let prompt = tokenizer.encode("hi", false);
    let config = greedy_config(12);

    // The point of the whole feature: at block 4 a perfect drafter commits
    // 5 tokens per round, so 12 tokens cost far fewer target passes than 12.
    // Without this the losslessness tests above would still pass against a
    // loop that quietly verified one token at a time.
    let (_, tokens, calls) = run_speculative(
        &tokenizer,
        &truth,
        &prompt,
        &config,
        vocab,
        Drafter::Perfect,
        4,
    );
    assert_eq!(tokens.len(), 12);
    assert!(
        calls <= 4,
        "a perfect drafter at block 4 should need at most 4 verify passes for 12 tokens, took {calls}"
    );

    // And a useless drafter must not do BETTER than one round per token,
    // which would mean acceptance was not really being checked.
    let (_, _, useless_calls) = run_speculative(
        &tokenizer,
        &truth,
        &prompt,
        &config,
        vocab,
        Drafter::Useless,
        4,
    );
    assert!(
        useless_calls >= 12,
        "a useless drafter accepts nothing, so it cannot beat one verify per token, took {useless_calls}"
    );
}

#[test]
fn a_stop_token_inside_an_accepted_block_stops_where_the_sequential_run_does() {
    let tokenizer = load_tokenizer();
    let vocab = tokenizer.vocab_size;
    let h = tokenizer.token_to_id("h").expect("'h'");
    let e = tokenizer.token_to_id("e").expect("'e'");
    // The stop lands third, i.e. strictly inside a block of 4, which is the
    // case that has to roll the engine back to the committed prefix rather
    // than to the start of the round.
    let truth = vec![h, e, tokenizer.end_of_turn_id, h, e];
    let prompt = tokenizer.encode("hi", false);
    let config = greedy_config(50);

    let (seq, seq_tokens) = run_sequential(&tokenizer, &truth, &prompt, &config, vocab);
    assert_eq!(seq.reason, StopReason::EndOfTurn);

    for block in [2usize, 4, 8] {
        let (spec, spec_tokens, _) = run_speculative(
            &tokenizer,
            &truth,
            &prompt,
            &config,
            vocab,
            Drafter::Perfect,
            block,
        );
        assert_eq!(spec.reason, StopReason::EndOfTurn, "block {block}");
        assert_eq!(spec_tokens, seq_tokens, "block {block}");
        // The stopping token is never fed, so it must not be in the cache.
        assert_eq!(
            spec.kv_backed_token_ids, seq.kv_backed_token_ids,
            "block {block} left the stop token in the cache"
        );
    }
}

#[test]
fn the_budget_is_exact_even_when_a_block_would_overshoot_it() {
    let tokenizer = load_tokenizer();
    let vocab = tokenizer.vocab_size;
    let truth = plain_truth(&tokenizer);
    let prompt = tokenizer.encode("hi", false);

    // 5 is not a multiple of any block size here, so a loop that committed
    // whole blocks would overshoot.
    for max_new in [1u32, 2, 5, 9] {
        let config = greedy_config(max_new);
        for block in [1usize, 2, 4, 8] {
            let (spec, tokens, _) = run_speculative(
                &tokenizer,
                &truth,
                &prompt,
                &config,
                vocab,
                Drafter::Perfect,
                block,
            );
            assert_eq!(
                tokens.len(),
                max_new as usize,
                "block {block} generated the wrong count for a budget of {max_new}"
            );
            assert_eq!(spec.new_tokens, max_new as usize);
            assert_eq!(spec.reason, StopReason::MaxTokens);
        }
    }
}

#[test]
fn a_round_never_feeds_past_the_context_window_it_was_admitted_under() {
    let tokenizer = load_tokenizer();
    let vocab = tokenizer.vocab_size;
    let truth = plain_truth(&tokenizer);
    let prompt = tokenizer.encode("hi", false);
    let max_new = 6u32;
    // The window `check_admission` accepts is exactly the COMMITTED stream,
    // with nothing spare. A speculative round feeds `block + 1` positions
    // before it knows how many it will keep, so an uncapped block writes KV
    // rows the caller never sized for -- and gives them back on the rollback,
    // so no output comparison can see it. On a real engine those rows are
    // past the end of the allocated cache.
    let max_context = (prompt.len() + max_new as usize) as u32;
    let config = greedy_config(max_new);

    for block in [2usize, 4, 8, 16] {
        let mut producer = ScriptedTarget::new(vocab, truth.clone(), Drafter::Perfect);
        let result = run_raw_completion_speculative(
            &mut producer,
            &tokenizer,
            &prompt,
            &config,
            max_context,
            vocab,
            block,
            |_| {},
        )
        .expect("run fits the window it was admitted under");
        assert_eq!(result.new_tokens, max_new as usize, "block {block}");
        assert!(
            producer.high_water.get() <= max_context as usize,
            "block {block} wrote position {} past a {max_context}-token window",
            producer.high_water.get()
        );
    }
}

#[test]
fn a_sampled_configuration_is_refused_rather_than_approximated() {
    let tokenizer = load_tokenizer();
    let vocab = tokenizer.vocab_size;
    let truth = plain_truth(&tokenizer);
    let prompt = tokenizer.encode("hi", false);
    let config = sampled_config(10);

    let mut producer = ScriptedTarget::new(vocab, truth, Drafter::Perfect);
    let err = run_raw_completion_speculative(
        &mut producer,
        &tokenizer,
        &prompt,
        &config,
        4096,
        vocab,
        2,
        |_| {},
    )
    .expect_err("a sampled run must not be served by argmax acceptance");

    match err {
        RuntimeError::SpeculationUnavailable(detail) => {
            assert!(
                detail.contains("rejection sampling"),
                "the refusal must name what is missing, got: {detail}"
            );
        }
        other => panic!("expected SpeculationUnavailable, got {other:?}"),
    }
    // Nothing was generated and nothing was absorbed: the refusal is at
    // admission, before any state moved.
    assert_eq!(producer.cursor, 0);
}

#[test]
fn a_zero_block_is_refused_rather_than_silently_decoding_sequentially() {
    let tokenizer = load_tokenizer();
    let vocab = tokenizer.vocab_size;
    let truth = plain_truth(&tokenizer);
    let prompt = tokenizer.encode("hi", false);
    let config = greedy_config(4);

    let mut producer = ScriptedTarget::new(vocab, truth, Drafter::Perfect);
    let err = run_raw_completion_speculative(
        &mut producer,
        &tokenizer,
        &prompt,
        &config,
        4096,
        vocab,
        0,
        |_| {},
    )
    .expect_err("block 0 proposes nothing");
    assert!(matches!(err, RuntimeError::SpeculationUnavailable(_)));
}

#[test]
fn cancelling_mid_block_reports_cancelled_and_leaves_an_honest_cache() {
    let tokenizer = load_tokenizer();
    let vocab = tokenizer.vocab_size;
    let truth = plain_truth(&tokenizer);
    let prompt = tokenizer.encode("hi", false);
    let config = greedy_config(50);

    // Cancel after the third committed token, which at block 4 lands inside
    // an accepted block rather than on a round boundary.
    let seen = Cell::new(0usize);
    let cancel = || seen.get() >= 3;

    let mut producer = ScriptedTarget::new(vocab, truth, Drafter::Perfect);
    let mut events = Vec::new();
    let result = run_raw_completion_speculative_cancellable(
        &mut producer,
        &tokenizer,
        &prompt,
        &config,
        4096,
        vocab,
        4,
        &cancel,
        |e| {
            if matches!(e, RawDecodeProgress::Token { .. }) {
                seen.set(seen.get() + 1);
            }
            events.push(e);
        },
    )
    .expect("cancelled run is not an error");

    assert_eq!(result.reason, StopReason::Cancelled);
    assert_eq!(collect_tokens(&events).len(), 3);
    assert_eq!(
        producer.cursor, result.kv_position,
        "a cancelled speculative run must not leave the engine ahead of what it reported"
    );
    assert_eq!(result.kv_backed_token_ids.len(), result.kv_position);
}
