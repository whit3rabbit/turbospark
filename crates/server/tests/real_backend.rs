//! End-to-end test of the real (`RealForwardRunner`-backed) server backend.
//!
//! Needs a real `.gturbo` install and a Metal device, so it is `#[ignore]`d
//! and gated on the same env var the memory oracle uses:
//!
//!   TURBOSPARK_GEMMA4_INSTALL_DIR=~/models/gemma4.gturbo \
//!     cargo test -p turbospark-server --test real_backend --release -- --ignored --nocapture
//!
//! One model open serves both requests, in sequence: that is the point, it
//! proves a second request through the same mutex-held runner still works
//! (the raw-completion loop resets the KV cache on entry). Generated text is
//! never asserted, only that generation ran (docs/TESTING.md).
#![cfg(target_os = "macos")]

use std::path::PathBuf;
use std::sync::Arc;

use turbospark_server::{build_router, RealChatModel};

fn install_dir() -> Option<PathBuf> {
    std::env::var_os("TURBOSPARK_GEMMA4_INSTALL_DIR").map(PathBuf::from)
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs a real Gemma 4 .gturbo install (TURBOSPARK_GEMMA4_INSTALL_DIR)"]
async fn real_backend_serves_streaming_and_non_streaming_requests() {
    let Some(dir) = install_dir() else {
        eprintln!(
            "real_backend: TURBOSPARK_GEMMA4_INSTALL_DIR is not set; skipping. \
             Point it at a repacked Gemma 4 .gturbo install to run this test."
        );
        return;
    };

    // Slots AND the context window both PINNED, not `auto`. This is a gate,
    // and a gate that let the machine pick either would be asserting against
    // a different configuration on every host (AGENTS.md Gotcha 35).
    //
    // SPECULATION IS PINNED OFF for the same reason, and it is the newest
    // instance of that rule here: `Speculation::Auto` reads the INSTALL, so a
    // gate left on it would decode speculatively or sequentially depending on
    // which drafter the install this env var happens to point at carries.
    // Off is also what every host has always run, so no assertion below moves.
    let model = RealChatModel::open(
        &dir,
        Some(1024),
        Some(16),
        Default::default(),
        runtime::Speculation::Off,
        runtime::SpeculativeDrafter::Auto,
        // PINNED for the same reason speculation is, and this one would be
        // inert anyway: no request below carries tools, so every guardrail
        // short-circuits on the empty offer set. Pinning says so rather than
        // leaving a gate's path to a default that can move.
        turbospark_server::GuardrailConfig::OFF,
        // PINNED OFF, and this is the one flag here that would change the
        // TOKENS rather than the path taken to them: a steered model is a
        // different model, so a test asserting what this install says must
        // not be able to acquire one by default.
        runtime::SteeringPolicy::off(),
    )
    .expect("real install should open");
    let model: Arc<dyn turbospark_server::ChatModel> = Arc::new(model);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, build_router(model)).await.unwrap();
    });
    let base = format!("http://{addr}");
    let client = reqwest::Client::new();

    // top_p without an explicit top_k is the standard OpenAI shape, and the
    // shaping config rejects it unless the handler defaults top_k.
    let response = client
        .post(format!("{base}/v1/chat/completions"))
        .json(&serde_json::json!({
            "model": "gemma4",
            "messages": [{"role": "user", "content": "Name one benefit of wetlands."}],
            "max_tokens": 24,
            "temperature": 0.2,
            "top_p": 0.95
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    let body: serde_json::Value = response.json().await.unwrap();
    let content = body["choices"][0]["message"]["content"].as_str().unwrap();
    assert!(!content.is_empty(), "expected generated text");
    let reason = body["choices"][0]["finish_reason"].as_str().unwrap();
    assert!(
        reason == "stop" || reason == "length",
        "unexpected finish_reason {reason}"
    );
    assert!(body["usage"]["completion_tokens"].as_u64().unwrap() > 0);

    // Second request, same runner, streamed.
    let response = client
        .post(format!("{base}/v1/chat/completions"))
        .json(&serde_json::json!({
            "model": "gemma4",
            "messages": [{"role": "user", "content": "Name one benefit of mangroves."}],
            "max_tokens": 24,
            "temperature": 0.2,
            "stream": true
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    let body = response.text().await.unwrap();
    assert!(body.contains("chat.completion.chunk"));
    assert!(body.contains("[DONE]"));
    assert!(
        !body.contains("\"error\""),
        "streamed run reported an error: {body}"
    );
    let deltas = body
        .lines()
        .filter(|l| l.contains("\"content\":\"") && !l.contains("\"content\":\"\""))
        .count();
    assert!(deltas > 0, "expected at least one non-empty content delta");
}
