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
use std::sync::{Arc, Mutex};

use turbospark_server::observe::{ServerEvent, ServerObserver};
use turbospark_server::{build_router, build_router_with_options, RealChatModel, RouterOptions};

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
        tokenizer::ReasoningEffort::Off,
        None,
        // PINNED OFF: this test asserts nothing about prefix reuse and stays
        // on the engine's baseline path, same reasoning as steering above.
        false,
        // PINNED at 1 (no pool): this test asserts nothing about session
        // multiplexing.
        1,
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

#[derive(Default)]
struct Recorder(Mutex<Vec<ServerEvent>>);

impl ServerObserver for Recorder {
    fn record(&self, event: ServerEvent) {
        self.0.lock().unwrap().push(event);
    }
}

impl Recorder {
    /// Every `Generated` event's `reusedPrefixTokens`, in the order they were
    /// recorded. A request the guardrails re-asked would produce two events
    /// for one HTTP call (`crates/server/CLAUDE.md` Gotcha 29), so this reads
    /// per-event rather than assuming one event per request.
    fn generated_reused_counts(&self) -> Vec<u64> {
        self.0
            .lock()
            .unwrap()
            .iter()
            .map(|e| serde_json::to_value(e).unwrap())
            .filter(|v| v["kind"] == "generated")
            .map(|v| v["reusedPrefixTokens"].as_u64().unwrap())
            .collect()
    }

    /// The same, paired with `sessionSlotEvicted`, for the session-pool test
    /// below.
    fn generated_reuse_and_eviction(&self) -> Vec<(u64, bool)> {
        self.0
            .lock()
            .unwrap()
            .iter()
            .map(|e| serde_json::to_value(e).unwrap())
            .filter(|v| v["kind"] == "generated")
            .map(|v| {
                (
                    v["reusedPrefixTokens"].as_u64().unwrap(),
                    v["sessionSlotEvicted"].as_bool().unwrap(),
                )
            })
            .collect()
    }
}

/// Proves prefix KV reuse actually fires over the server's own HTTP surface,
/// not just inside the runtime crate. `crates/runtime/tests/prefix_reuse_real.rs`
/// proves the mechanism itself; this proves the wiring in `real_model.rs` and
/// `main.rs` reaches it, on the code path the server actually takes
/// (`run_raw_completion_chunked_cancellable`, since Gemma 4 supports chunked
/// prefill and `RealChatModel::run_completion`'s non-speculative branch
/// always prefers it -- `crates/server/CLAUDE.md` Gotcha 19).
///
/// Two sequential requests, same server, same process, same runner, forming
/// a continuing transcript exactly the way a chat client resends one:
/// request 2 carries request 1's own reply back as an `assistant` turn. A
/// test that only checked the response bodies would pass trivially whether
/// or not reuse ever engaged, which is exactly the shape
/// `reuse_actually_fires_on_the_second_turn` in the runtime crate's own test
/// guards against -- so this asserts the SECOND `Generated` event's
/// `reusedPrefixTokens` is nonzero, not just that both requests succeeded.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs a real Gemma 4 .gturbo install (TURBOSPARK_GEMMA4_INSTALL_DIR)"]
async fn real_backend_reuses_kv_across_two_chat_turns() {
    let Some(dir) = install_dir() else {
        eprintln!(
            "real_backend: TURBOSPARK_GEMMA4_INSTALL_DIR is not set; skipping. \
             Point it at a repacked Gemma 4 .gturbo install to run this test."
        );
        return;
    };

    let model = RealChatModel::open(
        &dir,
        Some(1024),
        Some(16),
        Default::default(),
        runtime::Speculation::Off,
        runtime::SpeculativeDrafter::Auto,
        turbospark_server::GuardrailConfig::OFF,
        runtime::SteeringPolicy::off(),
        runtime::LoadPolicy::default(),
        tokenizer::ReasoningEffort::Off,
        None,
        // THE ONE FLAG THIS TEST IS ABOUT: on, unlike every other test in
        // this file.
        true,
        // PINNED at 1: this test is about single-session prefix reuse, not
        // multi-session pooling. See `real_backend_reuses_kv_across_two_
        // interleaved_conversations` for that.
        1,
    )
    .expect("real install should open");
    let model: Arc<dyn turbospark_server::ChatModel> = Arc::new(model);
    let recorder = Arc::new(Recorder::default());
    let router = build_router_with_options(
        model,
        RouterOptions {
            api_key: None,
            observer: Some(recorder.clone() as Arc<dyn ServerObserver>),
        },
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });
    let base = format!("http://{addr}");
    let client = reqwest::Client::new();

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
    let reply = body["choices"][0]["message"]["content"]
        .as_str()
        .unwrap()
        .to_string();
    assert!(!reply.is_empty(), "expected generated text on turn 1");

    // Turn 2 carries turn 1's own reply back as an assistant message, exactly
    // as a chat client resends the whole transcript every turn -- this is
    // what makes the re-rendered prompt's leading tokens agree with what
    // turn 1 left in the KV.
    let response = client
        .post(format!("{base}/v1/chat/completions"))
        .json(&serde_json::json!({
            "model": "gemma4",
            "messages": [
                {"role": "user", "content": "Name one benefit of wetlands."},
                {"role": "assistant", "content": reply},
                {"role": "user", "content": "Name another one."}
            ],
            "max_tokens": 24,
            "temperature": 0.2,
            "top_p": 0.95
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    let body: serde_json::Value = response.json().await.unwrap();
    let reply2 = body["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or_default();
    assert!(!reply2.is_empty(), "expected generated text on turn 2");

    let counts = recorder.generated_reused_counts();
    assert_eq!(counts.len(), 2, "expected one Generated event per request");
    assert_eq!(counts[0], 0, "turn 1 has nothing to continue from");
    assert!(
        counts[1] > 0,
        "turn 2's prompt shares a prefix with turn 1's KV; reuse should have fired, \
         but reusedPrefixTokens read 0"
    );
}

/// **THE SERVER-LEVEL PROOF OF ROADMAP SECTION 4's OPTION 3, AND OF
/// `crates/server/CLAUDE.md`'s `--session-slots` GOTCHA.** The prior test
/// proves prefix reuse fires for ONE client talking to itself; this proves
/// the thing that test's own comment says the server cannot do by default:
/// serve TWO interleaved conversations off one runner without one turn
/// discarding the other's reusable state. At `--session-slots 1` (every
/// other test in this file), turn B1 landing between A1 and A2 would
/// `reset()` A's session outright, and A2 would read `reusedPrefixTokens ==
/// 0` -- exactly Gotcha 31's stated limitation. At `--session-slots 2`, B1's
/// `reset()` parks A's session instead of clobbering it, and A2 finds it
/// again through the pool.
///
/// Two independent "clients" (A: wetlands/mangroves, B: volcanoes/glaciers,
/// deliberately unrelated topics), interleaved A1, B1, A2, B2 on the SAME
/// server. Asserts both A2 and B2 -- not just one -- found more than the
/// harmless template-boilerplate overlap any two prompts share (measured via
/// B1, see the comment at the assertions), and that neither turn had to
/// evict a still-useful session (`sessionSlotEvicted == false` throughout):
/// two conversations exactly fill a two-slot pool, so nothing here should be
/// under memory pressure.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs a real Gemma 4 .gturbo install (TURBOSPARK_GEMMA4_INSTALL_DIR)"]
async fn real_backend_reuses_kv_across_two_interleaved_conversations() {
    let Some(dir) = install_dir() else {
        eprintln!(
            "real_backend: TURBOSPARK_GEMMA4_INSTALL_DIR is not set; skipping. \
             Point it at a repacked Gemma 4 .gturbo install to run this test."
        );
        return;
    };

    let model = RealChatModel::open(
        &dir,
        Some(1024),
        Some(16),
        Default::default(),
        runtime::Speculation::Off,
        runtime::SpeculativeDrafter::Auto,
        turbospark_server::GuardrailConfig::OFF,
        runtime::SteeringPolicy::off(),
        runtime::LoadPolicy::default(),
        tokenizer::ReasoningEffort::Off,
        None,
        true,
        // THE ONE FLAG THIS TEST IS ABOUT: 2, unlike every other test in this
        // file, so the pool holds one live plus one parked session.
        2,
    )
    .expect("real install should open");
    let model: Arc<dyn turbospark_server::ChatModel> = Arc::new(model);
    let recorder = Arc::new(Recorder::default());
    let router = build_router_with_options(
        model,
        RouterOptions {
            api_key: None,
            observer: Some(recorder.clone() as Arc<dyn ServerObserver>),
        },
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });
    let base = format!("http://{addr}");
    let client = reqwest::Client::new();

    async fn turn(
        client: &reqwest::Client,
        base: &str,
        history: &mut Vec<serde_json::Value>,
        prompt: &str,
    ) {
        history.push(serde_json::json!({"role": "user", "content": prompt}));
        let response = client
            .post(format!("{base}/v1/chat/completions"))
            .json(&serde_json::json!({
                "model": "gemma4",
                "messages": history,
                "max_tokens": 24,
                "temperature": 0.2,
                "top_p": 0.95
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), 200);
        let body: serde_json::Value = response.json().await.unwrap();
        let reply = body["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or_default()
            .to_string();
        assert!(!reply.is_empty(), "expected generated text for {prompt:?}");
        history.push(serde_json::json!({"role": "assistant", "content": reply}));
    }

    let mut conversation_a = Vec::new();
    let mut conversation_b = Vec::new();

    turn(
        &client,
        &base,
        &mut conversation_a,
        "Name one benefit of wetlands.",
    )
    .await;
    turn(
        &client,
        &base,
        &mut conversation_b,
        "Name one hazard of volcanoes.",
    )
    .await;
    turn(&client, &base, &mut conversation_a, "Name another one.").await;
    turn(&client, &base, &mut conversation_b, "Name another one.").await;

    let events = recorder.generated_reuse_and_eviction();
    assert_eq!(events.len(), 4, "expected one Generated event per request");
    let (a1, b1, a2, b2) = (events[0], events[1], events[2], events[3]);

    // A1 is the FIRST request against a brand-new server: both the live
    // session and the one parked slot are provably empty
    // (`KvPrefix::common_prefix` short-circuits to 0 on an empty record no
    // matter what the prompt is), so this is the one turn in the test
    // guaranteed to read exactly 0.
    assert_eq!(a1.0, 0, "A's first turn has nothing to continue from");
    // B1 is NOT guaranteed to read 0, and asserting so was this test's own
    // bug, caught by the real install rather than assumed away: B1's LIVE
    // session at that point is A1's real content, and `common_prefix` is a
    // plain longest-common-prefix walk over token ids with no notion of
    // "conversation" -- two chat turns rendered through the SAME template
    // legitimately share its opening boilerplate (measured here: 6 tokens),
    // and reusing those exact shared bytes is correct and harmless (attention
    // over identical tokens is identical regardless of which conversation
    // produced them). `b1.0` is exactly that harmless template-only overlap,
    // which is why it is the BASELINE the two real assertions below compare
    // against, rather than a value asserted to be 0.
    assert!(
        a2.0 > b1.0,
        "A's second turn should have found A's own FULL session (parked from B1's turn) \
         rather than just the template boilerplate any two prompts share (measured on B1: \
         {} tokens); got reusedPrefixTokens={}, expected more than the boilerplate baseline",
        b1.0,
        a2.0
    );
    assert!(
        b2.0 > b1.0,
        "B's second turn should have found B's own FULL session (parked from A2's turn), \
         which includes B1's real exchange and so must exceed B1's own template-only overlap \
         with A ({} tokens); got reusedPrefixTokens={}",
        b1.0,
        b2.0
    );
    for (label, (_, evicted)) in [("A1", a1), ("B1", b1), ("A2", a2), ("B2", b2)] {
        assert!(
            !evicted,
            "{label} should not have evicted a still-useful session: two conversations \
             exactly fill a two-slot pool"
        );
    }
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
        tokenizer::ReasoningEffort::Off,
        None,
        // PINNED OFF: the vision path taints `kv_prefix` on every call anyway
        // (`set_prompt_vision` taints, `crates/runtime/CLAUDE.md` Gotcha 30),
        // so reuse is inert here regardless. Pinning `false` documents that
        // rather than leaving it implicit.
        false,
        // PINNED at 1 (no pool): this test asserts nothing about session
        // multiplexing.
        1,
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
