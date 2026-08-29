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
        runtime::LoadPolicy::default(),
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

/// **THE SERVING HALF OF M-V8, AND THE ARM THE ROADMAP DEMANDED.**
///
/// `tests/images.rs` covers every refusal with a scripted backend and no
/// model. It cannot cover the thing that matters: whether a picture actually
/// reaches the model. That is exactly the gap M-V5's injection bug lived in
/// for two milestones -- every test drove `produce` directly, the shapes and
/// lengths all agreed, and the model answered fluently about a page it had
/// never seen.
///
/// So this asserts on the OUTPUT, against a page whose content is known: a
/// generated table of words with five-digit line numbers down the left. A
/// server that dropped the image answers from the question alone and matches
/// none of it.
///
///   TURBOSPARK_QWEN38_VISION_INSTALL_DIR=~/models/qwen38-27b-vision.gturbo \
///   TURBOSPARK_VISION_PAGE=~/models/vision-probe-qwen38/imgs/page.png \
///     cargo test -p turbospark-server --test real_backend --release -- \
///     --ignored --nocapture
#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs a real vision install (TURBOSPARK_QWEN38_VISION_INSTALL_DIR) and a page \
            (TURBOSPARK_VISION_PAGE)"]
async fn real_backend_reads_an_image_sent_over_both_endpoints() {
    let (Some(dir), Some(page)) = (
        std::env::var_os("TURBOSPARK_QWEN38_VISION_INSTALL_DIR").map(PathBuf::from),
        std::env::var_os("TURBOSPARK_VISION_PAGE").map(PathBuf::from),
    ) else {
        eprintln!(
            "real_backend: TURBOSPARK_QWEN38_VISION_INSTALL_DIR and TURBOSPARK_VISION_PAGE are \
             not both set; skipping."
        );
        return;
    };

    let png = std::fs::read(&page).expect("the test page should be readable");
    let data_url = format!("data:image/png;base64,{}", base64_encode(&png));

    // Pinned exactly as the test above pins them, and for the same reason.
    let model = RealChatModel::open(
        &dir,
        Some(4096),
        Some(16),
        Default::default(),
        runtime::Speculation::Off,
        runtime::SpeculativeDrafter::Auto,
        turbospark_server::GuardrailConfig::OFF,
        runtime::SteeringPolicy::off(),
        runtime::LoadPolicy::default(),
    )
    .expect("the vision install should open");
    let model: Arc<dyn turbospark_server::ChatModel> = Arc::new(model);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, build_router(model)).await.unwrap();
    });
    let base = format!("http://{addr}");
    let client = reqwest::Client::new();

    // The page is a table of generated words with five-digit line numbers.
    // Asserting on a DIGIT RUN rather than on specific words: the words are
    // random per page, the line numbers are structural, and a model answering
    // from the question alone produces neither.
    let reads_the_page = |text: &str| -> bool {
        text.contains("00012") || text.contains("00030") || text.contains(" | ")
    };

    // ---- OpenAI, non-streaming ------------------------------------------
    let response = client
        .post(format!("{base}/v1/chat/completions"))
        .json(&serde_json::json!({
            "model": "qwen38-vision",
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": "Transcribe the text in this image."}
            ]}],
            "max_tokens": 48,
            "temperature": 0
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    // A served image must NOT be reported as dropped, which is the pair to
    // `images.rs`'s text-only cases.
    assert!(
        response.headers().get("x-anyllm-degradation").is_none(),
        "a served image was reported as degraded"
    );
    let body: serde_json::Value = response.json().await.unwrap();
    let text = body["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or_default()
        .to_string();
    eprintln!("openai non-streaming: {text}");
    assert!(
        reads_the_page(&text),
        "the model did not read the page; it answered {text:?}"
    );

    // ---- Anthropic, the other wire format on the same backend -----------
    let response = client
        .post(format!("{base}/v1/messages"))
        .json(&serde_json::json!({
            "model": "claude-sonnet-4-6",
            "max_tokens": 48,
            "temperature": 0,
            "messages": [{"role": "user", "content": [
                {"type": "image", "source": {
                    "type": "base64", "media_type": "image/png",
                    "data": base64_encode(&png)}},
                {"type": "text", "text": "Transcribe the text in this image."}
            ]}]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    let body: serde_json::Value = response.json().await.unwrap();
    let text = body["content"][0]["text"]
        .as_str()
        .unwrap_or_default()
        .to_string();
    eprintln!("anthropic: {text}");
    assert!(
        reads_the_page(&text),
        "the model did not read the page through /v1/messages; it answered {text:?}"
    );
}

/// Standard base64. Test-local, because this crate's own decoder is the thing
/// under test and encoding with it would make the round trip agree with
/// itself whatever either half does.
fn base64_encode(bytes: &[u8]) -> String {
    const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let b = [
            chunk[0],
            *chunk.get(1).unwrap_or(&0),
            *chunk.get(2).unwrap_or(&0),
        ];
        let n = ((b[0] as u32) << 16) | ((b[1] as u32) << 8) | b[2] as u32;
        out.push(ALPHABET[(n >> 18) as usize & 63] as char);
        out.push(ALPHABET[(n >> 12) as usize & 63] as char);
        out.push(if chunk.len() > 1 {
            ALPHABET[(n >> 6) as usize & 63] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            ALPHABET[n as usize & 63] as char
        } else {
            '='
        });
    }
    out
}
