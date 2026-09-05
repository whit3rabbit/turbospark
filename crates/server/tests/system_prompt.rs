//! The deployment-wide default system prompt (`--system` / `--system-file`).
//!
//! Every assertion here is an EQUALITY between two prompt-token counts, never
//! a threshold, because the property under test is exact: injecting the
//! default must produce the same prompt the caller would have produced by
//! sending that system message itself, and a request that sends its own must
//! produce a prompt the default did not touch at all.
//!
//! Counting is the observable rather than the rendered text because `plan` is
//! `pub(crate)`. That is not a weaker check: the count is a pure function of
//! the rendered prompt, so a default that landed in the wrong place, twice, or
//! not at all moves it.

use std::path::PathBuf;
use std::sync::Arc;

use tokenizer::MfTokenizer;
use turbospark_server::{build_router, ScriptedChatModel};

/// The prompt this file configures. Multi-word so its token count is well
/// clear of any single-token rounding.
const DEFAULT_SYSTEM: &str = "You are a deployment-wide test assistant. Answer briefly.";

/// A DIFFERENT prompt the caller sends, so a test cannot pass by the two
/// happening to tokenize to the same length.
const CALLER_SYSTEM: &str = "Ignore the deployment and speak only in questions, at length.";

fn load_tokenizer() -> MfTokenizer {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/ChatMLTokenizer");
    MfTokenizer::load_from_dir(&dir).expect("fixture tokenizer should load")
}

fn one_hot(vocab_size: usize, index: usize) -> Vec<foundation::LogitValue> {
    let mut v = vec![foundation::LogitValue::from_f32(0.0); vocab_size];
    v[index] = foundation::LogitValue::from_f32(1.0);
    v
}

/// `default_system: None` is a server started without the flag; `Some` is one
/// started with it. Both otherwise identical, which is what makes every
/// comparison below single-variable.
async fn spawn_server(default_system: Option<&str>) -> String {
    let tok = load_tokenizer();
    let h = tok.token_to_id("h").unwrap() as usize;
    let steps = vec![one_hot(tok.vocab_size, h); 1024];
    let scripted = ScriptedChatModel::new(tok, 4096, steps);
    let scripted = match default_system {
        Some(s) => scripted.with_default_system(s),
        None => scripted,
    };
    let model: Arc<dyn turbospark_server::ChatModel> = Arc::new(scripted);
    let router = build_router(model);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });
    format!("http://{addr}")
}

async fn post(base: &str, path: &str, body: serde_json::Value) -> serde_json::Value {
    reqwest::Client::new()
        .post(format!("{base}{path}"))
        .json(&body)
        .send()
        .await
        .unwrap()
        .json::<serde_json::Value>()
        .await
        .unwrap()
}

/// `POST /v1/messages/count_tokens` runs the same `plan` a real call does, so
/// it prices the prompt without decoding anything.
async fn anthropic_count(base: &str, body: serde_json::Value) -> u64 {
    let v = post(base, "/v1/messages/count_tokens", body).await;
    v["input_tokens"]
        .as_u64()
        .unwrap_or_else(|| panic!("no input_tokens in {v}"))
}

async fn openai_prompt_tokens(base: &str, body: serde_json::Value) -> u64 {
    let v = post(base, "/v1/chat/completions", body).await;
    v["usage"]["prompt_tokens"]
        .as_u64()
        .unwrap_or_else(|| panic!("no usage.prompt_tokens in {v}"))
}

// ---------------------------------------------------------------------------
// The default is applied, and applied as though the caller had sent it

/// The load-bearing equality. A server with the default configured must price
/// a bare user turn EXACTLY as a server without it prices that same turn with
/// the prompt sent explicitly. Anything else means the injection landed in the
/// wrong position, or rendered through a different path.
#[tokio::test]
async fn the_default_renders_exactly_as_a_caller_sent_system_message() {
    let with_default = spawn_server(Some(DEFAULT_SYSTEM)).await;
    let without = spawn_server(None).await;

    let injected = anthropic_count(
        &with_default,
        serde_json::json!({
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}]
        }),
    )
    .await;
    let sent_by_hand = anthropic_count(
        &without,
        serde_json::json!({
            "model": "m",
            "system": DEFAULT_SYSTEM,
            "messages": [{"role": "user", "content": "hi"}]
        }),
    )
    .await;

    assert_eq!(
        injected, sent_by_hand,
        "a configured default must render as the same prompt the caller would have produced"
    );
}

/// And it must actually DO something. Without this the equality above is also
/// satisfied by a default that is silently dropped on both servers.
#[tokio::test]
async fn the_default_lengthens_a_prompt_that_carries_none() {
    let with_default = spawn_server(Some(DEFAULT_SYSTEM)).await;
    let without = spawn_server(None).await;
    let body = serde_json::json!({
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}]
    });

    let injected = anthropic_count(&with_default, body.clone()).await;
    let bare = anthropic_count(&without, body).await;

    assert!(
        injected > bare,
        "configured default should add tokens: with={injected} without={bare}"
    );
}

// ---------------------------------------------------------------------------
// The caller wins

/// Anthropic's system is a TOP-LEVEL field rather than a message, prepended by
/// `anyllm_translate` before `plan` ever sees it. This is the case that proves
/// the suppression check reads the translated request and not the wire body.
#[tokio::test]
async fn an_anthropic_top_level_system_suppresses_the_default() {
    let with_default = spawn_server(Some(DEFAULT_SYSTEM)).await;
    let without = spawn_server(None).await;
    let body = serde_json::json!({
        "model": "m",
        "system": CALLER_SYSTEM,
        "messages": [{"role": "user", "content": "hi"}]
    });

    let served_with = anthropic_count(&with_default, body.clone()).await;
    let served_without = anthropic_count(&without, body).await;

    assert_eq!(
        served_with, served_without,
        "a caller that sent its own system prompt must get a prompt the default never touched"
    );
}

#[tokio::test]
async fn an_openai_system_message_suppresses_the_default() {
    let with_default = spawn_server(Some(DEFAULT_SYSTEM)).await;
    let without = spawn_server(None).await;
    let body = serde_json::json!({
        "model": "m",
        "max_tokens": 4,
        "messages": [
            {"role": "system", "content": CALLER_SYSTEM},
            {"role": "user", "content": "hi"}
        ]
    });

    let served_with = openai_prompt_tokens(&with_default, body.clone()).await;
    let served_without = openai_prompt_tokens(&without, body).await;

    assert_eq!(served_with, served_without);
}

// **NO `developer`-ROLE CASE, AND THAT IS THE FIXTURE'S LIMIT RATHER THAN A
// GAP IN THE RULE.** The suppression check reads
// `ChatRole::System | ChatRole::Developer`, mirroring `role_from`. The ChatML
// fixture's own `chat_template.jinja` refuses that role before any of this is
// reached ("Unexpected message role", chat:144), so a test here would assert
// the template's refusal rather than the default's suppression. Cover it when
// a fixture whose template renders `developer` lands.

// ---------------------------------------------------------------------------
// One insertion, every wire format

/// `plan` is the single funnel, so a default configured once must reach the
/// Ollama route as well without that route knowing anything about it.
#[tokio::test]
async fn the_ollama_route_gets_the_default_too() {
    let with_default = spawn_server(Some(DEFAULT_SYSTEM)).await;
    let without = spawn_server(None).await;
    let body = serde_json::json!({
        "model": "m",
        "stream": false,
        // BOUNDED, or this case decodes to the scripted step limit and takes
        // over a minute. Nothing about the prompt count depends on it.
        "options": {"num_predict": 4},
        "messages": [{"role": "user", "content": "hi"}]
    });

    let injected = post(&with_default, "/api/chat", body.clone()).await["prompt_eval_count"]
        .as_u64()
        .unwrap();
    let bare = post(&without, "/api/chat", body).await["prompt_eval_count"]
        .as_u64()
        .unwrap();

    assert!(injected > bare, "with={injected} without={bare}");
}

/// And a caller's own system message on that route suppresses it, the same way
/// it does on the other two.
#[tokio::test]
async fn an_ollama_system_message_suppresses_the_default() {
    let with_default = spawn_server(Some(DEFAULT_SYSTEM)).await;
    let without = spawn_server(None).await;
    let body = serde_json::json!({
        "model": "m",
        "stream": false,
        "options": {"num_predict": 4},
        "messages": [
            {"role": "system", "content": CALLER_SYSTEM},
            {"role": "user", "content": "hi"}
        ]
    });

    let served_with = post(&with_default, "/api/chat", body.clone()).await["prompt_eval_count"]
        .as_u64()
        .unwrap();
    let served_without = post(&without, "/api/chat", body).await["prompt_eval_count"]
        .as_u64()
        .unwrap();

    assert_eq!(served_with, served_without);
}

/// `/v1/responses` maps its `instructions` field to a leading system message,
/// so that field must suppress the default for the same reason Anthropic's
/// top-level `system` does.
#[tokio::test]
async fn responses_instructions_suppress_the_default() {
    let with_default = spawn_server(Some(DEFAULT_SYSTEM)).await;
    let without = spawn_server(None).await;
    let body = serde_json::json!({
        "model": "m",
        "max_output_tokens": 4,
        "instructions": CALLER_SYSTEM,
        "input": "hi"
    });

    let served_with = post(&with_default, "/v1/responses", body.clone()).await["usage"]
        ["input_tokens"]
        .as_u64()
        .unwrap();
    let served_without = post(&without, "/v1/responses", body).await["usage"]["input_tokens"]
        .as_u64()
        .unwrap();

    assert_eq!(served_with, served_without);
}

/// The counting route must price what a real call prefills, or a client that
/// budgets against it under-reserves by the whole system prompt.
#[tokio::test]
async fn count_tokens_prices_the_injected_default() {
    let base = spawn_server(Some(DEFAULT_SYSTEM)).await;
    let messages = serde_json::json!([{"role": "user", "content": "hi"}]);

    let counted = anthropic_count(
        &base,
        serde_json::json!({"model": "m", "messages": messages.clone()}),
    )
    .await;
    let really_prefilled = openai_prompt_tokens(
        &base,
        serde_json::json!({"model": "m", "max_tokens": 4, "messages": messages}),
    )
    .await;

    assert_eq!(counted, really_prefilled);
}
