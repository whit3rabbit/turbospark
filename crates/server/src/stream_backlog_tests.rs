//! A live stream whose client stops reading (but stays connected) must stop
//! its generation once the queued backlog is full, instead of queuing every
//! chunk until `max_tokens` is spent.
//!
//! Neither disconnect detector covers this client: the receiver is alive, so
//! sends succeed, and the body is never dropped, so `CancelOnDrop` never
//! fires. The handlers are called directly and the body is held unpolled
//! after its first frame, so no socket buffer adds slack: the only bound is
//! `STREAM_BACKLOG` (shrunk under `cfg(test)` so the plateau comes quickly in
//! a debug build).

use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use axum::extract::State;
use axum::response::Response;
use axum::{Extension, Json};
use futures::StreamExt;
use runtime::{LogitProducer, RawDecodeResult, RuntimeError};
use tokenizer::MfTokenizer;

use crate::handler::RequestLeaseSlot;
use crate::model::ChatModel;
use crate::ServerState;

const BUDGET: u64 = 1_000_000;

struct CountingProducer {
    h_id: i32,
    calls: Arc<AtomicUsize>,
}

impl LogitProducer for CountingProducer {
    fn reset(&mut self) {}

    fn produce(
        &mut self,
        _token: i32,
        _position: usize,
        logits: &mut [foundation::LogitValue],
    ) -> Result<(), String> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        logits.fill(foundation::LogitValue::from_f32(0.0));
        logits[self.h_id as usize] = foundation::LogitValue::from_f32(1.0);
        Ok(())
    }
}

/// Replays one token forever, so only cancellation or the budget ends a run.
struct CountingModel {
    tokenizer: MfTokenizer,
    h_id: i32,
    calls: Arc<AtomicUsize>,
}

impl ChatModel for CountingModel {
    fn tokenizer(&self) -> &MfTokenizer {
        &self.tokenizer
    }
    fn vocab_size(&self) -> usize {
        self.tokenizer.vocab_size
    }
    fn max_context(&self) -> u32 {
        // Clears `prompt + max_new <= max_context` for `BUDGET`; nothing
        // here allocates in proportion to it.
        2_000_000
    }
    fn model_id(&self) -> &str {
        "counting"
    }
    fn with_producer(
        &self,
        f: &mut dyn FnMut(&mut dyn LogitProducer) -> Result<RawDecodeResult, RuntimeError>,
    ) -> Result<RawDecodeResult, RuntimeError> {
        let mut producer = CountingProducer {
            h_id: self.h_id,
            calls: self.calls.clone(),
        };
        f(&mut producer)
    }
}

fn setup() -> (ServerState, Arc<AtomicUsize>) {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/ChatMLTokenizer");
    let tok = MfTokenizer::load_from_dir(&dir).expect("fixture tokenizer should load");
    let h_id = tok.token_to_id("h").unwrap();
    let calls = Arc::new(AtomicUsize::new(0));
    let model: Arc<dyn ChatModel> = Arc::new(CountingModel {
        tokenizer: tok,
        h_id,
        calls: calls.clone(),
    });
    (model.into(), calls)
}

fn lease() -> Extension<RequestLeaseSlot> {
    Extension(RequestLeaseSlot::default())
}

/// Reads the first frame, then holds the body unread and waits for the
/// producer call count to stop moving. Panics if it is still moving at the
/// deadline, which is what an unbounded queue does for a `BUDGET` this size.
async fn plateau(name: &str, response: Response, calls: &AtomicUsize) -> usize {
    assert_eq!(response.status(), 200, "{name}");
    let mut body = response.into_body().into_data_stream();
    let first = body.next().await.expect("first frame").expect("frame");
    assert!(!first.is_empty(), "{name}");

    let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
    let mut last = calls.load(Ordering::SeqCst);
    loop {
        tokio::time::sleep(Duration::from_millis(300)).await;
        let now = calls.load(Ordering::SeqCst);
        if now == last && now > 0 {
            drop(body);
            assert!((now as u64) < BUDGET, "{name}: {now}");
            return now;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "{name}: generation still advancing ({now} calls) 10 s after the \
             client stopped reading; chunks are queuing without bound"
        );
        last = now;
    }
}

#[tokio::test]
async fn stalled_chat_completions_reader_stops_generation() {
    let (state, calls) = setup();
    let request = serde_json::from_value(serde_json::json!({
        "model": "m", "max_tokens": BUDGET, "temperature": 0.0, "stream": true,
        "messages": [{"role": "user", "content": "hi"}]
    }))
    .unwrap();
    let response =
        crate::handler::chat_completions(State(state), None, lease(), Json(request)).await;
    plateau("chat", response, &calls).await;
}

#[tokio::test]
async fn stalled_completions_reader_stops_generation() {
    let (state, calls) = setup();
    let request = serde_json::from_value(serde_json::json!({
        "model": "m", "max_tokens": BUDGET, "temperature": 0.0, "stream": true,
        "prompt": "hi"
    }))
    .unwrap();
    let response =
        crate::completions::completions(State(state), None, lease(), Json(request)).await;
    plateau("completions", response, &calls).await;
}

#[tokio::test]
async fn stalled_messages_reader_stops_generation() {
    let (state, calls) = setup();
    let body = serde_json::json!({
        "model": "m", "max_tokens": BUDGET, "temperature": 0.0, "stream": true,
        "messages": [{"role": "user", "content": "hi"}]
    });
    let response = crate::messages::messages(State(state), None, lease(), Json(body)).await;
    plateau("messages", response, &calls).await;
}

#[tokio::test]
async fn stalled_responses_reader_stops_generation() {
    let (state, calls) = setup();
    let request = serde_json::from_value(serde_json::json!({
        "model": "m", "max_output_tokens": BUDGET, "temperature": 0.0,
        "stream": true, "input": "hi"
    }))
    .unwrap();
    let response = crate::responses::responses(State(state), None, lease(), Json(request)).await;
    plateau("responses", response, &calls).await;
}

#[tokio::test]
async fn stalled_ollama_reader_stops_generation() {
    let (state, calls) = setup();
    let request = serde_json::from_value(serde_json::json!({
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}],
        "options": {"num_predict": BUDGET, "temperature": 0.0},
    }))
    .unwrap();
    let response = crate::ollama::chat(State(state), None, lease(), Json(request)).await;
    plateau("ollama", response, &calls).await;
}
