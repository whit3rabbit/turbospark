//! Cancellation: `run_raw_completion_cancellable` and its chunked sibling.
//!
//! The same ChatML fixture and scripted-logits arrangement
//! `tests/raw_completion.rs` uses, so a reader comparing the two sees only
//! the cancel predicate. Token ids are resolved from the loaded tokenizer
//! rather than hardcoded, for the reason that file's header gives.
//!
//! Three of these are about a property that is easy to lose and impossible
//! to see from the outside: a cancelled run has to take the SAME exit path
//! the other stop reasons take. If it breaks out early instead, it silently
//! drops whatever the stop matcher was withholding, and the caller gets a
//! truncated reply rather than a cancelled one.

use std::cell::Cell;
use std::path::PathBuf;

use foundation::LogitValue;
use selection::ShapingConfig;
use tokenizer::MfTokenizer;
use turbospark_runtime::{
    run_raw_completion, run_raw_completion_cancellable, run_raw_completion_chunked,
    run_raw_completion_chunked_cancellable, GenerationConfig, RawDecodeProgress,
    ScriptedLogitProducer, StopReason,
};

fn load_tokenizer() -> MfTokenizer {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/ChatMLTokenizer");
    MfTokenizer::load_from_dir(&dir).expect("fixture tokenizer should load")
}

fn one_hot(vocab_size: usize, index: usize) -> Vec<LogitValue> {
    let mut v = vec![LogitValue::from_f32(0.0); vocab_size];
    v[index] = LogitValue::from_f32(1.0);
    v
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

fn id_of(tokenizer: &MfTokenizer, ch: &str) -> usize {
    tokenizer
        .token_to_id(ch)
        .unwrap_or_else(|| panic!("{ch:?} is in the fixture's base vocab")) as usize
}

/// A predicate that first returns true on poll number `n`, counting from 0.
///
/// The poll schedule is worth stating, because every expectation below is
/// derived from it: one poll per prompt token (or per CHUNK on the chunked
/// path), then one per decoded token. So poll `prompt_len` is the first
/// decode poll, and `cancel_on_poll(0)` fires after exactly one prompt token
/// has been committed.
///
/// `Cell` rather than an `AtomicBool` because the loop is single-threaded and
/// the predicate is `&dyn Fn`. The FFI's real one is atomic; nothing here
/// needs to be, and a counter makes "cancel on exactly the Nth poll"
/// expressible where a bool cannot.
fn cancel_on_poll(n: usize) -> impl Fn() -> bool {
    let polls = Cell::new(0usize);
    move || {
        let seen = polls.get();
        polls.set(seen + 1);
        seen >= n
    }
}

/// The steps a scripted producer needs to emit `chars` and then stop.
///
/// The first `prompt_len - 1` steps are consumed by prefill's
/// `produce_prefill` calls, so decode step 1 is seeded by the LAST prefill
/// step. Getting this offset wrong starts the generation mid-sequence.
fn scripted(vocab_size: usize, prompt_len: usize, ids: &[usize]) -> Vec<Vec<LogitValue>> {
    let mut steps = vec![vec![LogitValue::from_f32(0.0); vocab_size]; prompt_len - 1];
    steps.extend(ids.iter().map(|&id| one_hot(vocab_size, id)));
    steps
}

#[test]
fn a_never_true_predicate_reproduces_the_uncancelled_run_exactly() {
    // The delegation guard. `run_raw_completion` passes a predicate that is
    // always false, so this is the assertion that the added branch changed
    // nothing on the path every existing caller takes.
    let tokenizer = load_tokenizer();
    let vocab_size = tokenizer.vocab_size;
    let prompt_ids = tokenizer.encode("hi", false);
    let ids = [
        id_of(&tokenizer, "h"),
        id_of(&tokenizer, "i"),
        tokenizer.end_of_turn_id as usize,
    ];

    let mut plain_events = Vec::new();
    let mut plain = ScriptedLogitProducer::new(scripted(vocab_size, prompt_ids.len(), &ids));
    let plain_result = run_raw_completion(
        &mut plain,
        &tokenizer,
        &prompt_ids,
        &greedy_config(10),
        4096,
        vocab_size,
        |e| plain_events.push(e),
    )
    .expect("uncancelled run should succeed");

    let mut cancellable_events = Vec::new();
    let mut cancellable = ScriptedLogitProducer::new(scripted(vocab_size, prompt_ids.len(), &ids));
    let cancellable_result = run_raw_completion_cancellable(
        &mut cancellable,
        &tokenizer,
        &prompt_ids,
        &greedy_config(10),
        4096,
        vocab_size,
        &|| false,
        |e| cancellable_events.push(e),
    )
    .expect("never-cancelled run should succeed");

    assert_eq!(plain_events, cancellable_events);
    assert_eq!(plain_result.reason, cancellable_result.reason);
    assert_eq!(plain_result.new_tokens, cancellable_result.new_tokens);
    assert_eq!(
        plain_result.kv_backed_token_ids,
        cancellable_result.kv_backed_token_ids
    );
    assert_eq!(plain_result.reason, StopReason::EndOfTurn);
}

#[test]
fn cancelling_during_decode_stops_and_keeps_what_was_generated() {
    let tokenizer = load_tokenizer();
    let vocab_size = tokenizer.vocab_size;
    let prompt_ids = tokenizer.encode("hi", false);
    let h = id_of(&tokenizer, "h");
    // Far more steps than the run will consume: the point is that the loop
    // stops because it was cancelled, not because it ran out of script.
    let ids = vec![h; 20];

    let mut producer = ScriptedLogitProducer::new(scripted(vocab_size, prompt_ids.len(), &ids));
    let mut tokens = 0usize;
    // Poll `prompt_len` is the first decode poll, so this fires on the third
    // decoded token.
    let cancel = cancel_on_poll(prompt_ids.len() + 2);

    let result = run_raw_completion_cancellable(
        &mut producer,
        &tokenizer,
        &prompt_ids,
        &greedy_config(50),
        4096,
        vocab_size,
        &cancel,
        |e| {
            if matches!(e, RawDecodeProgress::Token { .. }) {
                tokens += 1;
            }
        },
    )
    .expect("a cancelled run is not an error");

    assert_eq!(result.reason, StopReason::Cancelled);
    assert_eq!(
        result.new_tokens, 3,
        "two full tokens plus the one that saw the cancel"
    );
    assert_eq!(tokens, 3);
    // The KV cache has to describe itself honestly, or a caller cannot
    // continue the conversation from a cancelled turn.
    assert_eq!(result.prompt_tokens, prompt_ids.len());
    assert_eq!(result.kv_backed_token_ids.len(), result.kv_position);
    assert!(result.kv_position >= prompt_ids.len());
}

#[test]
fn cancelling_during_decode_still_flushes_the_withheld_tail() {
    // THE ONE THAT MATTERS. The stop matcher withholds any text that could
    // be the start of a stop string. A cancelled run that breaks early
    // instead of taking the shared exit path drops it, and the caller sees a
    // reply missing its last few characters with nothing to indicate why.
    let tokenizer = load_tokenizer();
    let vocab_size = tokenizer.vocab_size;
    let prompt_ids = tokenizer.encode("hi", false);
    let h = id_of(&tokenizer, "h");

    let mut config = greedy_config(50);
    // "hz" never completes: the model emits "h", the matcher holds it as a
    // possible prefix, and only a flush can release it.
    config.stop_strings = vec!["hz".to_string()];

    let mut producer = ScriptedLogitProducer::new(scripted(vocab_size, prompt_ids.len(), &[h; 20]));
    let mut visible = String::new();
    let mut tail = String::new();
    // Exactly ONE decoded token, so the matcher is holding that single "h"
    // and has emitted nothing. Let a second through and it releases the
    // first (the suffix of "hh" that could still become "hz" is only the
    // last character), which would weaken the assertion below to nothing.
    let cancel = cancel_on_poll(prompt_ids.len());

    let result = run_raw_completion_cancellable(
        &mut producer,
        &tokenizer,
        &prompt_ids,
        &config,
        4096,
        vocab_size,
        &cancel,
        |e| match e {
            RawDecodeProgress::Token { delta, .. } => visible.push_str(&delta),
            RawDecodeProgress::Tail(t) => tail.push_str(&t),
            RawDecodeProgress::Prefill { .. } => {}
        },
    )
    .expect("a cancelled run is not an error");

    assert_eq!(result.reason, StopReason::Cancelled);
    assert_eq!(result.new_tokens, 1);
    // The one "h" was withheld as a possible "hz" prefix, so the run emitted
    // no visible text at all...
    assert!(
        visible.is_empty(),
        "the matcher should have withheld the token, got {visible:?}"
    );
    // ...and the ONLY way the caller ever sees it is the flush on the way
    // out. This is the assertion a break-early cancel fails.
    assert_eq!(
        tail, "h",
        "the withheld text must be released when the run is cancelled"
    );
}

#[test]
fn a_real_stop_reason_wins_over_a_simultaneous_cancel() {
    // Precedence. A Stop button pressed as the model finishes must not
    // relabel a complete turn as a truncated one: the run stopped because it
    // was done, and `EndOfTurn` is what a caller acts on.
    let tokenizer = load_tokenizer();
    let vocab_size = tokenizer.vocab_size;
    let prompt_ids = tokenizer.encode("hi", false);
    // The FIRST decoded token is the stop token, and the predicate is armed
    // for exactly that step. Note the predicate must stay false through
    // prefill: an always-true one cancels on the first prompt token and
    // never reaches decode at all, which tests nothing.
    let ids = [tokenizer.end_of_turn_id as usize];

    let mut producer = ScriptedLogitProducer::new(scripted(vocab_size, prompt_ids.len(), &ids));
    let cancel = cancel_on_poll(prompt_ids.len());
    let result = run_raw_completion_cancellable(
        &mut producer,
        &tokenizer,
        &prompt_ids,
        &greedy_config(50),
        4096,
        vocab_size,
        &cancel,
        |_| {},
    )
    .expect("a cancelled run is not an error");

    // The stop-token branch breaks out ABOVE the cancel poll, so the
    // predicate is never even consulted on this step.
    assert_eq!(result.reason, StopReason::EndOfTurn);
}

#[test]
fn max_tokens_wins_over_a_simultaneous_cancel() {
    // The other half of the precedence rule, on the arm that shares the
    // cancel's `if`. `hit_max` is tested first, so a run that filled its
    // budget on the same token the cancel arrived reports `MaxTokens`.
    let tokenizer = load_tokenizer();
    let vocab_size = tokenizer.vocab_size;
    let prompt_ids = tokenizer.encode("hi", false);
    let h = id_of(&tokenizer, "h");

    let mut producer = ScriptedLogitProducer::new(scripted(vocab_size, prompt_ids.len(), &[h; 10]));
    // Armed for the first decode step, which is also the step that fills a
    // budget of one. Both conditions are true in the same `if`.
    let cancel = cancel_on_poll(prompt_ids.len());
    let result = run_raw_completion_cancellable(
        &mut producer,
        &tokenizer,
        &prompt_ids,
        &greedy_config(1),
        4096,
        vocab_size,
        &cancel,
        |_| {},
    )
    .expect("a cancelled run is not an error");

    assert_eq!(result.reason, StopReason::MaxTokens);
    assert_eq!(result.new_tokens, 1);
}

#[test]
fn cancelling_during_prefill_generates_nothing_and_reports_an_honest_cursor() {
    let tokenizer = load_tokenizer();
    let vocab_size = tokenizer.vocab_size;
    // Long enough that prefill has several tokens to be cancelled between.
    let prompt_ids = tokenizer.encode("hello there", false);
    assert!(
        prompt_ids.len() > 2,
        "this case needs a prompt prefill can be interrupted inside"
    );
    let h = id_of(&tokenizer, "h");

    let mut producer = ScriptedLogitProducer::new(scripted(vocab_size, prompt_ids.len(), &[h; 5]));
    let mut prefilled = 0usize;
    let mut tokens = 0usize;
    // Fires on the very first poll, i.e. once one prompt token is committed.
    let cancel = cancel_on_poll(0);

    let result = run_raw_completion_cancellable(
        &mut producer,
        &tokenizer,
        &prompt_ids,
        &greedy_config(10),
        4096,
        vocab_size,
        &cancel,
        |e| match e {
            RawDecodeProgress::Prefill { done, .. } => prefilled = done,
            RawDecodeProgress::Token { .. } => tokens += 1,
            RawDecodeProgress::Tail(_) => {}
        },
    )
    .expect("a cancelled prefill is not an error");

    assert_eq!(result.reason, StopReason::Cancelled);
    assert_eq!(result.new_tokens, 0);
    assert_eq!(tokens, 0);
    // Cancelled after the first prompt token was committed, and before the
    // whole prompt was walked.
    assert_eq!(prefilled, 1);
    assert_eq!(result.kv_position, 1);
    assert_eq!(result.kv_backed_token_ids, vec![prompt_ids[0]]);
    // Decoding never started, so a caller dividing tokens by seconds must
    // not see a rate.
    assert_eq!(result.decode_seconds, 0.0);
}

#[test]
fn the_chunked_path_cancels_at_a_chunk_boundary() {
    let tokenizer = load_tokenizer();
    let vocab_size = tokenizer.vocab_size;
    let prompt_ids = tokenizer.encode("hello there", false);
    let h = id_of(&tokenizer, "h");
    // One scripted step per chunk, then the decode steps.
    let mut steps = vec![vec![LogitValue::from_f32(0.0); vocab_size]; prompt_ids.len()];
    steps.extend((0..5).map(|_| one_hot(vocab_size, h)));

    let mut producer = ScriptedLogitProducer::new(steps);
    let mut prefilled = 0usize;
    // The chunked path polls once per CHUNK, so the first poll lands after
    // two prompt tokens rather than one.
    let cancel = cancel_on_poll(0);

    let result = run_raw_completion_chunked_cancellable(
        &mut producer,
        &tokenizer,
        &prompt_ids,
        &greedy_config(10),
        4096,
        vocab_size,
        2,
        &cancel,
        |e| {
            if let RawDecodeProgress::Prefill { done, .. } = e {
                prefilled = done;
            }
        },
    )
    .expect("a cancelled chunked prefill is not an error");

    assert_eq!(result.reason, StopReason::Cancelled);
    assert_eq!(result.new_tokens, 0);
    // A whole chunk, never a partial one: `PrefillChunkCommitState` makes a
    // mid-chunk bail unreachable, so the cursor lands on a chunk boundary.
    assert_eq!(prefilled, 2);
    assert_eq!(result.kv_position, 2);
    assert_eq!(result.kv_backed_token_ids, prompt_ids[..2].to_vec());
}

#[test]
fn the_chunked_delegation_reproduces_the_uncancelled_run() {
    let tokenizer = load_tokenizer();
    let vocab_size = tokenizer.vocab_size;
    let prompt_ids = tokenizer.encode("hello there", false);
    let ids = [id_of(&tokenizer, "h"), tokenizer.end_of_turn_id as usize];

    let steps = || {
        let mut s = vec![vec![LogitValue::from_f32(0.0); vocab_size]; prompt_ids.len()];
        s.extend(ids.iter().map(|&id| one_hot(vocab_size, id)));
        s
    };

    let mut plain_events = Vec::new();
    let mut plain = ScriptedLogitProducer::new(steps());
    let plain_result = run_raw_completion_chunked(
        &mut plain,
        &tokenizer,
        &prompt_ids,
        &greedy_config(10),
        4096,
        vocab_size,
        2,
        |e| plain_events.push(e),
    )
    .expect("uncancelled chunked run should succeed");

    let mut cancellable_events = Vec::new();
    let mut cancellable = ScriptedLogitProducer::new(steps());
    let cancellable_result = run_raw_completion_chunked_cancellable(
        &mut cancellable,
        &tokenizer,
        &prompt_ids,
        &greedy_config(10),
        4096,
        vocab_size,
        2,
        &|| false,
        |e| cancellable_events.push(e),
    )
    .expect("never-cancelled chunked run should succeed");

    assert_eq!(plain_events, cancellable_events);
    assert_eq!(plain_result.reason, cancellable_result.reason);
    assert_eq!(
        plain_result.kv_backed_token_ids,
        cancellable_result.kv_backed_token_ids
    );
}
