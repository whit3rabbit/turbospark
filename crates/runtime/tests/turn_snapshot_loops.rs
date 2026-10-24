//! The generation loops' half of the GDN turn snapshot
//! (`crate::turn_snapshot`): a producer that names a cut must see its
//! prefill stop there for exactly one call, with exactly the cut's tokens
//! fed, and a producer that names none must see the passes it always saw.
//! The runner's half (what it keeps and when it restores) needs a real GDN
//! install and is pinned by `turn_snapshot_real.rs`.

use std::path::PathBuf;

use foundation::{prefill_chunk_spans, LogitValue, TokenId};
use selection::ShapingConfig;
use tokenizer::MfTokenizer;
use turbospark_runtime::{
    run_raw_completion, run_raw_completion_chunked, ChunkedPrefillRunner, GenerationConfig,
    LogitProducer, RateControl,
};

fn load_tokenizer() -> MfTokenizer {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/ChatMLTokenizer");
    MfTokenizer::load_from_dir(&dir).expect("fixture tokenizer should load")
}

#[derive(Debug, PartialEq)]
enum Event {
    /// A pass over `count` tokens from `start`.
    Pass { start: usize, count: usize },
    /// `take_turn_snapshot(position)` with `fed` tokens in the state.
    Snapshot { position: usize, fed: usize },
}

struct Recorder {
    next: usize,
    cut: Option<usize>,
    fed: usize,
    events: Vec<Event>,
}

impl Recorder {
    fn new(next: usize, cut: Option<usize>) -> Self {
        Self {
            next,
            cut,
            fed: 0,
            events: Vec::new(),
        }
    }

    fn write(&self, logits: &mut [LogitValue]) {
        logits.fill(LogitValue::from_f32(0.0));
        logits[self.next] = LogitValue::from_f32(1.0);
    }
}

impl LogitProducer for Recorder {
    fn reset(&mut self) {
        self.fed = 0;
    }

    fn turn_snapshot_position(&self, _prompt_ids: &[TokenId], _reused: usize) -> Option<usize> {
        self.cut
    }

    fn take_turn_snapshot(&mut self, position: usize) {
        self.events.push(Event::Snapshot {
            position,
            fed: self.fed,
        });
    }

    fn produce(
        &mut self,
        _token: i32,
        position: usize,
        logits: &mut [LogitValue],
    ) -> Result<(), String> {
        assert_eq!(position, self.fed, "fed out of order");
        self.events.push(Event::Pass {
            start: position,
            count: 1,
        });
        self.fed += 1;
        self.write(logits);
        Ok(())
    }
}

impl ChunkedPrefillRunner for Recorder {
    fn prefill_chunk(
        &mut self,
        tokens: &[i32],
        start_position: usize,
        logits: &mut [LogitValue],
    ) -> Result<(), String> {
        assert_eq!(start_position, self.fed, "chunk fed out of order");
        self.events.push(Event::Pass {
            start: start_position,
            count: tokens.len(),
        });
        self.fed += tokens.len();
        self.write(logits);
        Ok(())
    }
}

fn config() -> GenerationConfig {
    GenerationConfig {
        shaping: ShapingConfig::new(0.0, 0, None, 1.0, None).unwrap(),
        max_new_tokens: 1,
        stop_strings: Vec::new(),
        extra_stop_tokens: Vec::new(),
        rate: RateControl::default(),
    }
}

fn prompt(len: usize) -> Vec<i32> {
    (0..len as i32).map(|i| 10 + i).collect()
}

/// The prefill's events only; decode adds one pass after.
fn prefill_events(events: &[Event], prompt_len: usize) -> Vec<&Event> {
    events
        .iter()
        .filter(|e| match e {
            Event::Pass { start, .. } => *start < prompt_len,
            Event::Snapshot { .. } => true,
        })
        .collect()
}

#[test]
fn the_chunked_loop_splits_only_the_chunk_holding_the_cut() {
    let tokenizer = load_tokenizer();
    let vocab = tokenizer.vocab_size;
    let ids = prompt(20);
    let mut producer = Recorder::new(7, Some(15));
    run_raw_completion_chunked(
        &mut producer,
        &tokenizer,
        &ids,
        &config(),
        4096,
        vocab,
        4,
        |_| {},
    )
    .expect("run succeeds");
    let pass = |start, count| Event::Pass { start, count };
    assert_eq!(
        prefill_events(&producer.events, ids.len()),
        vec![
            &pass(0, 4),
            &pass(4, 4),
            &pass(8, 4),
            &pass(12, 3),
            &Event::Snapshot {
                position: 15,
                fed: 15
            },
            &pass(15, 4),
            &pass(19, 1),
        ]
    );
}

#[test]
fn the_chunked_loop_without_a_cut_keeps_the_plain_spans() {
    let tokenizer = load_tokenizer();
    let vocab = tokenizer.vocab_size;
    let ids = prompt(20);
    let mut producer = Recorder::new(7, None);
    run_raw_completion_chunked(
        &mut producer,
        &tokenizer,
        &ids,
        &config(),
        4096,
        vocab,
        4,
        |_| {},
    )
    .expect("run succeeds");
    let want: Vec<Event> = prefill_chunk_spans(ids.len(), 0, 4)
        .iter()
        .map(|s| Event::Pass {
            start: s.start_position,
            count: s.token_count,
        })
        .collect();
    assert_eq!(
        prefill_events(&producer.events, ids.len()),
        want.iter().collect::<Vec<_>>()
    );
}

#[test]
fn a_cut_on_a_chunk_boundary_adds_no_pass() {
    let tokenizer = load_tokenizer();
    let vocab = tokenizer.vocab_size;
    let ids = prompt(20);
    let mut producer = Recorder::new(7, Some(12));
    run_raw_completion_chunked(
        &mut producer,
        &tokenizer,
        &ids,
        &config(),
        4096,
        vocab,
        4,
        |_| {},
    )
    .expect("run succeeds");
    let passes = prefill_events(&producer.events, ids.len())
        .into_iter()
        .filter(|e| matches!(e, Event::Pass { .. }))
        .count();
    assert_eq!(passes, 5);
    assert!(producer.events.contains(&Event::Snapshot {
        position: 12,
        fed: 12
    }));
}

#[test]
fn the_sequential_loop_snapshots_with_exactly_the_cut_fed() {
    let tokenizer = load_tokenizer();
    let vocab = tokenizer.vocab_size;
    let ids = prompt(12);
    let mut producer = Recorder::new(7, Some(9));
    run_raw_completion(
        &mut producer,
        &tokenizer,
        &ids,
        &config(),
        4096,
        vocab,
        |_| {},
    )
    .expect("run succeeds");
    let snapshots: Vec<_> = producer
        .events
        .iter()
        .filter(|e| matches!(e, Event::Snapshot { .. }))
        .collect();
    assert_eq!(
        snapshots,
        vec![&Event::Snapshot {
            position: 9,
            fed: 9
        }]
    );
}
