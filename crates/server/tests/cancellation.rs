//! Client-disconnect cancellation: dropping a streaming response mid-generation
//! must shorten the generation still running on the server, rather than let
//! it run to whatever `max_tokens` asked for with nobody left to read it.
//!
//! `CountingModel` replays a fixed token forever (no scripted step budget to
//! run out of, unlike `ScriptedChatModel`) while counting every call into
//! `produce`, so the test can see directly how far generation actually got
//! rather than inferring it from wall-clock timing.

use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use runtime::{LogitProducer, RawDecodeResult, RuntimeError};
use tokenizer::MfTokenizer;
use turbospark_server::{build_router, ChatModel};

fn load_tokenizer() -> MfTokenizer {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/ChatMLTokenizer");
    MfTokenizer::load_from_dir(&dir).expect("fixture tokenizer should load")
}

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
        // Large enough that `max_tokens: 1_000_000` below clears
        // `check_admission`'s `prompt + max_new <= max_context` gate; a
        // real install's window would refuse that request outright, but
        // nothing here allocates memory proportional to it.
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

async fn serve(model: Arc<dyn ChatModel>) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, build_router(model)).await.unwrap();
    });
    format!("http://{addr}")
}

/// **THE HEADLINE CASE.** A request that asks for far more tokens than any
/// test should wait for (`max_tokens: 1_000_000`) is dropped after its first
/// chunk arrives -- proving the stream is actually under way -- and the
/// server must notice and stop within a short, fixed window rather than
/// grind through the full budget for a client that is no longer reading.
#[tokio::test]
async fn dropping_a_streaming_response_stops_generation_early() {
    let tok = load_tokenizer();
    let h_id = tok.token_to_id("h").unwrap();
    let calls = Arc::new(AtomicUsize::new(0));
    let model: Arc<dyn ChatModel> = Arc::new(CountingModel {
        tokenizer: tok,
        h_id,
        calls: calls.clone(),
    });
    let base = serve(model).await;

    let mut response = reqwest::Client::new()
        .post(format!("{base}/v1/chat/completions"))
        .json(&serde_json::json!({
            "model": "m", "max_tokens": 1_000_000, "temperature": 0.0, "stream": true,
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);

    // Prove the stream is actually flowing before cutting it off: an
    // early drop that raced the connection itself would prove nothing.
    let first = response
        .chunk()
        .await
        .unwrap()
        .expect("the role-delta chunk should have arrived");
    assert!(!first.is_empty());
    drop(response);

    // A fixed window rather than a poll-until-quiet loop: at any plausible
    // per-token cost in this decode loop (sampling, detokenizing, stop
    // matching -- AGENTS.md Gotcha 23 measured tens of milliseconds per
    // token even on a real model before that path was optimized), reaching
    // anywhere near 1,000,000 calls in a few hundred milliseconds is not
    // achievable, so a low count here can only mean cancellation fired.
    tokio::time::sleep(Duration::from_millis(300)).await;

    let stopped_at = calls.load(Ordering::SeqCst);
    assert!(
        stopped_at < 5_000,
        "generation reached {stopped_at} calls within 300ms of the client \
         disconnecting; cancellation did not stop it short of the 1,000,000 \
         token budget"
    );
}
