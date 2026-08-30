//! End-to-end raw-completion loop tests: real ChatML tokenizer fixture,
//! scripted logits standing in for a model forward pass, exercising
//! prefill, greedy sampling, detokenization, and stop-token handling.
//!
//! Token ids are resolved from the loaded tokenizer rather than hardcoded:
//! the fixture's `tokenizer.json` embeds high placeholder ids (e.g. 248044)
//! for its `added_tokens`, but the loader renumbers them sequentially after
//! the 258-entry base vocab, so the *actual* ids only exist at load time.

use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

use foundation::LogitValue;
use selection::ShapingConfig;
use tokenizer::MfTokenizer;
use turbospark_runtime::{
    run_raw_completion, GenerationConfig, RateControl, RawDecodeProgress, ScriptedLogitProducer,
    StopReason, ThermalLevel,
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

#[test]
fn stops_at_end_of_turn_and_emits_visible_content() {
    let tokenizer = load_tokenizer();
    let vocab_size = tokenizer.vocab_size;
    let h_id = tokenizer
        .token_to_id("h")
        .expect("'h' is in the base vocab") as usize;
    let im_end_id = tokenizer.end_of_turn_id as usize;

    let prompt_ids = tokenizer.encode("hi", false);
    assert!(!prompt_ids.is_empty());

    let mut steps = vec![vec![LogitValue::from_f32(0.0); vocab_size]; prompt_ids.len() - 1];
    steps.push(one_hot(vocab_size, h_id)); // last prefill step seeds decode step 1
    steps.push(one_hot(vocab_size, im_end_id)); // decode step 2: stop

    let mut producer = ScriptedLogitProducer::new(steps);
    let config = greedy_config(10);
    let mut events = Vec::new();

    let result = run_raw_completion(
        &mut producer,
        &tokenizer,
        &prompt_ids,
        &config,
        4096,
        vocab_size,
        |e| {
            events.push(e);
        },
    )
    .unwrap();

    assert_eq!(result.reason, StopReason::EndOfTurn);
    assert_eq!(result.new_tokens, 2);
    let h_id = h_id as i32;
    let im_end_id = im_end_id as i32;
    assert!(events
        .iter()
        .any(|e| matches!(e, RawDecodeProgress::Token { id, .. } if *id == h_id)));
    assert!(!events
        .iter()
        .any(|e| matches!(e, RawDecodeProgress::Token { id, .. } if *id == im_end_id)));
}

#[test]
fn stops_at_max_new_tokens_when_no_stop_token_arrives() {
    let tokenizer = load_tokenizer();
    let vocab_size = tokenizer.vocab_size;
    let h_id = tokenizer
        .token_to_id("h")
        .expect("'h' is in the base vocab") as usize;
    let prompt_ids = tokenizer.encode("hi", false);

    let mut steps = vec![vec![LogitValue::from_f32(0.0); vocab_size]; prompt_ids.len() - 1];
    // Every step (prefill seed + every decode step) keeps producing 'h'.
    for _ in 0..5 {
        steps.push(one_hot(vocab_size, h_id));
    }

    let mut producer = ScriptedLogitProducer::new(steps);
    let config = greedy_config(3);

    let result = run_raw_completion(
        &mut producer,
        &tokenizer,
        &prompt_ids,
        &config,
        4096,
        vocab_size,
        |_| {},
    )
    .unwrap();
    assert_eq!(result.reason, StopReason::MaxTokens);
    assert_eq!(result.new_tokens, 3);
}

#[test]
fn rejects_empty_prompt() {
    let tokenizer = load_tokenizer();
    let vocab_size = tokenizer.vocab_size;
    let mut producer = ScriptedLogitProducer::new(Vec::new());
    let config = greedy_config(5);
    let err = run_raw_completion(
        &mut producer,
        &tokenizer,
        &[],
        &config,
        4096,
        vocab_size,
        |_| {},
    )
    .unwrap_err();
    assert_eq!(err, turbospark_runtime::RuntimeError::EmptyPrompt);
}

#[test]
fn rejects_context_overflow() {
    let tokenizer = load_tokenizer();
    let vocab_size = tokenizer.vocab_size;
    let prompt_ids = tokenizer.encode("hi", false);
    let mut producer = ScriptedLogitProducer::new(Vec::new());
    let config = greedy_config(1000);
    let err = run_raw_completion(
        &mut producer,
        &tokenizer,
        &prompt_ids,
        &config,
        4,
        vocab_size,
        |_| {},
    )
    .unwrap_err();
    assert!(matches!(
        err,
        turbospark_runtime::RuntimeError::ContextOverflow { .. }
    ));
}

#[test]
fn stop_string_truncates_visible_output() {
    let tokenizer = load_tokenizer();
    let vocab_size = tokenizer.vocab_size;
    let h_id = tokenizer
        .token_to_id("h")
        .expect("'h' is in the base vocab") as usize;
    let prompt_ids = tokenizer.encode("hi", false);

    let mut steps = vec![vec![LogitValue::from_f32(0.0); vocab_size]; prompt_ids.len() - 1];
    for _ in 0..5 {
        steps.push(one_hot(vocab_size, h_id));
    }
    let mut producer = ScriptedLogitProducer::new(steps);
    let config = GenerationConfig {
        shaping: ShapingConfig::new(0.0, 0, None, 1.0, None).unwrap(),
        max_new_tokens: 5,
        stop_strings: vec!["hh".to_string()],
        extra_stop_tokens: Vec::new(),
        rate: Default::default(),
    };

    let result = run_raw_completion(
        &mut producer,
        &tokenizer,
        &prompt_ids,
        &config,
        4096,
        vocab_size,
        |_| {},
    )
    .unwrap();
    assert_eq!(result.reason, StopReason::StopString);
}

// --- ROADMAP Phase P2: decode-rate control -------------------------------

/// Counts how often the decode loop polls thermal pressure. A `fn` pointer
/// cannot capture, so the count lives in a static; only the test below
/// installs this probe.
static THERMAL_POLLS: AtomicUsize = AtomicUsize::new(0);

static MEMORY_POLLS: AtomicUsize = AtomicUsize::new(0);

/// Always `Critical`, so the memory ladder is unambiguously the thing
/// imposing the cap. A probe returning `Normal` would leave the run
/// indistinguishable from one with no memory watcher at all.
fn counting_critical_memory_probe() -> turbospark_runtime::MemoryPressure {
    MEMORY_POLLS.fetch_add(1, Ordering::Relaxed);
    turbospark_runtime::MemoryPressure::Critical
}

fn counting_nominal_probe() -> ThermalLevel {
    THERMAL_POLLS.fetch_add(1, Ordering::Relaxed);
    ThermalLevel::Nominal
}

/// Scripts `count` decode steps that all pick the same token, so a run's
/// length is set by `max_new_tokens` rather than by a stop.
fn repeating_steps(
    tokenizer: &MfTokenizer,
    prompt_len: usize,
    token: usize,
    count: usize,
) -> Vec<Vec<LogitValue>> {
    let vocab_size = tokenizer.vocab_size;
    let mut steps = vec![vec![LogitValue::from_f32(0.0); vocab_size]; prompt_len - 1];
    steps.extend((0..count).map(|_| one_hot(vocab_size, token)));
    steps
}

fn run_ids(tokenizer: &MfTokenizer, config: &GenerationConfig, max_new: usize) -> Vec<i32> {
    let prompt_ids = tokenizer.encode("hi", false);
    let h_id = tokenizer
        .token_to_id("h")
        .expect("'h' is in the base vocab") as usize;
    let mut producer =
        ScriptedLogitProducer::new(repeating_steps(tokenizer, prompt_ids.len(), h_id, max_new));
    let mut ids = Vec::new();
    run_raw_completion(
        &mut producer,
        tokenizer,
        &prompt_ids,
        config,
        4096,
        tokenizer.vocab_size,
        |event| {
            if let RawDecodeProgress::Token { id, .. } = event {
                ids.push(id);
            }
        },
    )
    .unwrap();
    ids
}

#[test]
fn a_rate_cap_holds_decode_to_at_least_its_schedule() {
    let tokenizer = load_tokenizer();
    let prompt_ids = tokenizer.encode("hi", false);
    let h_id = tokenizer
        .token_to_id("h")
        .expect("'h' is in the base vocab") as usize;

    // Six tokens at 50/s. The sixth hits `max_new_tokens` and breaks out
    // before pacing, so five slots of 20 ms are scheduled and the fifth
    // one closes 100 ms after decode starts. Asserted as a LOWER bound
    // only: a busy machine may take longer, but it can never finish
    // sooner without the cap having failed to apply.
    let mut producer =
        ScriptedLogitProducer::new(repeating_steps(&tokenizer, prompt_ids.len(), h_id, 6));
    let config = GenerationConfig {
        max_new_tokens: 6,
        rate: RateControl {
            max_tokens_per_sec: Some(50.0),
            thermal_probe: None,
            memory_probe: None,
        },
        ..greedy_config(6)
    };

    let result = run_raw_completion(
        &mut producer,
        &tokenizer,
        &prompt_ids,
        &config,
        4096,
        tokenizer.vocab_size,
        |_| {},
    )
    .unwrap();

    assert_eq!(result.reason, StopReason::MaxTokens);
    assert_eq!(result.new_tokens, 6);
    assert!(
        result.decode_seconds >= 0.1,
        "five 20 ms slots must have been waited out, got {}",
        result.decode_seconds
    );
}

#[test]
fn pacing_polls_thermal_pressure_without_changing_the_tokens() {
    let tokenizer = load_tokenizer();
    const MAX_NEW: usize = 40;

    let uncapped = run_ids(&tokenizer, &greedy_config(MAX_NEW as u32), MAX_NEW);

    THERMAL_POLLS.store(0, Ordering::Relaxed);
    let paced = run_ids(
        &tokenizer,
        &GenerationConfig {
            rate: RateControl {
                // Fast enough to keep the test short; the ladder itself is
                // unit-tested in `power.rs` and the re-anchoring in
                // `pacing.rs`.
                max_tokens_per_sec: Some(400.0),
                thermal_probe: Some(counting_nominal_probe),
                memory_probe: None,
            },
            ..greedy_config(MAX_NEW as u32)
        },
        MAX_NEW,
    );

    // Once before the loop, then every 16th token: 40 tokens gives polls
    // at 16 and 32.
    assert_eq!(THERMAL_POLLS.load(Ordering::Relaxed), 3);
    assert_eq!(uncapped.len(), MAX_NEW);
    assert_eq!(
        paced, uncapped,
        "pacing runs after selection and must not move a single token"
    );
}

/// The memory watcher rides the SAME poll block as the thermal one, on the
/// same cadence, and is subject to the same rule: pacing runs downstream of
/// selection, so it must not move a token.
///
/// **The memory probe alone, with no thermal probe**, which is the case a
/// shared `if let Some(probe) = thermal_probe` guard silently drops: the
/// block would never run, the cap would never be applied, and the watcher
/// would be dead code that every other assertion still passes over.
#[test]
fn the_memory_watcher_polls_on_the_same_cadence_without_changing_the_tokens() {
    let tokenizer = load_tokenizer();
    const MAX_NEW: usize = 40;

    let uncapped = run_ids(&tokenizer, &greedy_config(MAX_NEW as u32), MAX_NEW);

    MEMORY_POLLS.store(0, Ordering::Relaxed);
    let watched = run_ids(
        &tokenizer,
        &GenerationConfig {
            rate: RateControl {
                max_tokens_per_sec: Some(400.0),
                thermal_probe: None,
                memory_probe: Some(counting_critical_memory_probe),
            },
            ..greedy_config(MAX_NEW as u32)
        },
        MAX_NEW,
    );

    // Once before the loop, then every 16th token: the same 3 the thermal
    // case counts, which is what says the two share one block rather than
    // each having acquired its own schedule.
    assert_eq!(MEMORY_POLLS.load(Ordering::Relaxed), 3);
    assert_eq!(
        watched, uncapped,
        "the memory watcher runs after selection and must not move a token"
    );
}

/// A run with no probes reports `Normal`, which is the ABSENCE of a reading.
/// Asserting this is what stops a host reading the default decode path as a
/// positive report that memory was fine.
#[test]
fn an_unwatched_run_reports_normal_pressure() {
    let tokenizer = load_tokenizer();
    let prompt_ids = tokenizer.encode("hi", false);
    let h_id = tokenizer
        .token_to_id("h")
        .expect("'h' is in the base vocab") as usize;
    let mut producer =
        ScriptedLogitProducer::new(repeating_steps(&tokenizer, prompt_ids.len(), h_id, 4));
    let result = run_raw_completion(
        &mut producer,
        &tokenizer,
        &prompt_ids,
        &greedy_config(4),
        4096,
        tokenizer.vocab_size,
        |_| {},
    )
    .unwrap();
    assert_eq!(
        result.peak_memory_pressure,
        turbospark_runtime::MemoryPressure::Normal
    );
}

/// A producer that records every `(token, position)` it is fed and answers
/// `reusable_prefix` from a length the test sets, so the LOOP's half of
/// prefix reuse is observable without a GPU.
///
/// Deliberately not `ScriptedLogitProducer` with a flag bolted on: what
/// needs asserting is which positions were re-fed, and only a producer that
/// records its inputs can say.
struct RecordingProducer {
    fed: Vec<(i32, usize)>,
    reusable: usize,
    resets: usize,
    vocab: usize,
    next: usize,
}

impl RecordingProducer {
    fn new(vocab: usize, reusable: usize, next: usize) -> Self {
        Self {
            fed: Vec::new(),
            reusable,
            resets: 0,
            vocab,
            next,
        }
    }
}

impl turbospark_runtime::LogitProducer for RecordingProducer {
    fn reset(&mut self) {
        self.resets += 1;
    }

    fn try_reuse_prefix(&mut self, _prompt_ids: &[i32]) -> usize {
        self.reusable
    }

    fn produce(
        &mut self,
        token: i32,
        position: usize,
        logits: &mut [LogitValue],
    ) -> Result<(), String> {
        self.fed.push((token, position));
        logits.copy_from_slice(&one_hot(self.vocab, self.next));
        Ok(())
    }
}

fn run_with(
    producer: &mut RecordingProducer,
    prompt: &[i32],
) -> turbospark_runtime::RawDecodeResult {
    let tokenizer = load_tokenizer();
    let config = GenerationConfig {
        shaping: ShapingConfig::new(0.0, 0, None, 1.0, None).unwrap(),
        max_new_tokens: 1,
        stop_strings: Vec::new(),
        extra_stop_tokens: Vec::new(),
        rate: RateControl::default(),
    };
    let vocab = producer.vocab;
    run_raw_completion(producer, &tokenizer, prompt, &config, 4096, vocab, |_| {})
        .expect("run should succeed")
}

#[test]
fn a_reusable_prefix_skips_those_positions_and_the_reset() {
    let tokenizer = load_tokenizer();
    let vocab = tokenizer.vocab_size;
    let mut producer = RecordingProducer::new(vocab, 3, 7);
    let result = run_with(&mut producer, &[10, 11, 12, 13, 14]);

    // Only the tail was fed, at its ORIGINAL positions -- a reused prefix
    // that re-based the cursor would attend over the wrong rows.
    assert_eq!(producer.fed, vec![(13, 3), (14, 4)]);
    // And the state was NOT cleared, which is the whole point.
    assert_eq!(producer.resets, 0);
    // The result still describes the full prompt, not just the tail: a
    // caller computing tokens-per-second or a context budget from this must
    // see the conversation, not this turn's slice of it.
    assert_eq!(result.prompt_tokens, 5);
    // 5, not 6: the one sampled token is never fed back, so the KV holds
    // positions 0..4 only. Worth pinning, because it is exactly the kind of
    // per-caller arithmetic that `kv_prefix` refuses to derive a reuse
    // length from -- the producer records what it was FED instead.
    assert_eq!(result.kv_position, 5);
    assert_eq!(result.kv_backed_token_ids[..5], [10, 11, 12, 13, 14]);
}

#[test]
fn no_reusable_prefix_resets_and_feeds_every_position() {
    let tokenizer = load_tokenizer();
    let vocab = tokenizer.vocab_size;
    let mut producer = RecordingProducer::new(vocab, 0, 7);
    run_with(&mut producer, &[10, 11, 12, 13, 14]);

    assert_eq!(
        producer.fed,
        vec![(10, 0), (11, 1), (12, 2), (13, 3), (14, 4)]
    );
    assert_eq!(producer.resets, 1);
}

#[test]
fn a_prefix_covering_the_whole_prompt_is_clamped_so_one_token_is_always_fed() {
    // `decode` samples from `logits` before generating, so a prefill that
    // fed nothing would hand it the PREVIOUS turn's logits and emit a token
    // this prompt never justified. The loop clamps to `len - 1` rather than
    // trusting the producer, because the two guards protect different
    // things: the producer's is about what its cache holds, the loop's is
    // about its own next statement.
    let tokenizer = load_tokenizer();
    let vocab = tokenizer.vocab_size;
    let mut producer = RecordingProducer::new(vocab, 99, 7);
    run_with(&mut producer, &[10, 11, 12]);

    assert_eq!(producer.fed, vec![(12, 2)]);
    assert_eq!(producer.resets, 0);
}
