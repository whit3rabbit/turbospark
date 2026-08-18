//! A ChatML `<think>` block reaches the client as reasoning, but ONLY when
//! the request asked to think.
//!
//! The Harmony sibling of this file covers the dialect that always separates
//! its channels. This one covers the conditional case, which is where the
//! damage would be: `needs_decoder` keys the ChatML arm on the REQUEST, so
//! getting that condition wrong is invisible until a real model is driven
//! with `reasoning_effort` set and answers with its scratch work glued to
//! the front of the reply.
//!
//! No model: `ScriptedChatModel` forces an exact token sequence, so the
//! `<think>` frame is produced deterministically. The prefill consumes one
//! scripted step per prompt token but the last (see `steps_for_turn`).

use std::path::PathBuf;
use std::sync::Arc;

use tokenizer::{Message, MfTokenizer, ReasoningEffort, Role};
use turbospark_server::{build_router, ScriptedChatModel};

const REASONING: &str = "seventeen minus eight";
const ANSWER: &str = "Nine sheep.";
const USER: &str = "how many sheep";

fn load_tokenizer() -> MfTokenizer {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/ChatMLTokenizer");
    MfTokenizer::load_from_dir(&dir).expect("ChatML fixture tokenizer should load")
}

fn one_hot(vocab_size: usize, index: usize) -> Vec<foundation::LogitValue> {
    let mut v = vec![foundation::LogitValue::from_f32(0.0); vocab_size];
    v[index] = foundation::LogitValue::from_f32(1.0);
    v
}

/// A thinking assistant turn: `<think>` reasoning `</think>` answer, ending
/// at the turn-end token so generation stops the way a real one does.
fn thinking_turn(tok: &MfTokenizer) -> Vec<i32> {
    let mut ids = vec![tok.think_start_id.expect("ChatML resolves <think>")];
    ids.extend(tok.encode(REASONING, false));
    ids.push(tok.think_end_id.expect("ChatML resolves </think>"));
    ids.extend(tok.encode(ANSWER, false));
    ids.push(tok.end_of_turn_id);
    ids
}

/// The scripted steps for `effort`, whose PROMPT LENGTH differs from the
/// default one's -- the template renders an extra pre-closed `<think>` block
/// when thinking is off, so the prefill offset has to be computed against the
/// same level the request will send.
fn steps_for(tok: &MfTokenizer, effort: ReasoningEffort) -> Vec<Vec<foundation::LogitValue>> {
    let prompt = tok
        .apply_chat_template_with_reasoning(&[Message::new(Role::User, USER)], effort)
        .expect("the fixture ships a template");
    let prompt_ids = tok.encode(&prompt, false);

    let mut steps = vec![one_hot(tok.vocab_size, 0); prompt_ids.len() - 1];
    steps.extend(
        thinking_turn(tok)
            .iter()
            .map(|&id| one_hot(tok.vocab_size, id as usize)),
    );
    steps
}

async fn serve(effort: ReasoningEffort) -> String {
    let tok = load_tokenizer();
    let steps = steps_for(&tok, effort);
    let model: Arc<dyn turbospark_server::ChatModel> =
        Arc::new(ScriptedChatModel::new(tok, 4096, steps));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, build_router(model)).await.unwrap();
    });
    format!("http://{addr}")
}

fn body(effort: Option<&str>) -> serde_json::Value {
    let mut body = serde_json::json!({
        "model": "m",
        "max_tokens": 64,
        "temperature": 0.0,
        "messages": [{"role": "user", "content": USER}]
    });
    if let Some(effort) = effort {
        body["reasoning_effort"] = serde_json::json!(effort);
    }
    body
}

async fn post(base: &str, path: &str, body: serde_json::Value) -> serde_json::Value {
    let raw = reqwest::Client::new()
        .post(format!("{base}{path}"))
        .json(&body)
        .send()
        .await
        .expect("request should reach the server")
        .text()
        .await
        .expect("body should read");
    serde_json::from_str(&raw).unwrap_or_else(|e| panic!("{e}: {raw}"))
}

/// THE HEADLINE: with `reasoning_effort` set, the `<think>` body arrives as
/// `reasoning_content` and the answer arrives as content, separately.
#[tokio::test]
async fn a_think_block_becomes_reasoning_content_when_a_level_was_asked_for() {
    let base = serve(ReasoningEffort::Low).await;
    let json = post(&base, "/v1/chat/completions", body(Some("low"))).await;

    let message = &json["choices"][0]["message"];
    let content = message["content"].as_str().unwrap_or_default();
    let reasoning = message["reasoning_content"].as_str().unwrap_or_default();

    assert!(reasoning.contains(REASONING), "reasoning = {reasoning:?}");
    assert!(content.contains(ANSWER), "content = {content:?}");
    // The half that matters most: the scratch work must not be in the reply.
    assert!(
        !content.contains(REASONING),
        "reasoning leaked into the answer: {content:?}"
    );
}

/// The same turn on the Anthropic endpoint, where `reasoning_content` becomes
/// a `thinking` block ahead of the text block.
#[tokio::test]
async fn the_anthropic_endpoint_carries_it_as_a_thinking_block() {
    let base = serve(ReasoningEffort::Low).await;
    let json = post(&base, "/v1/messages", body(Some("low"))).await;

    let kinds: Vec<&str> = json["content"]
        .as_array()
        .expect("content blocks")
        .iter()
        .map(|b| b["type"].as_str().unwrap_or_default())
        .collect();
    assert_eq!(kinds, vec!["thinking", "text"], "{json}");
    assert!(json["content"][0]["thinking"]
        .as_str()
        .unwrap_or_default()
        .contains(REASONING));
    assert!(json["content"][1]["text"]
        .as_str()
        .unwrap_or_default()
        .contains(ANSWER));
}

/// **THE CONDITION, FROM THE OTHER SIDE.** With no level asked for, this
/// dialect keeps the pass-through path it has always had: no decoder is
/// built, so nothing is separated and nothing is swallowed.
///
/// Without this case the pair above would pass just as happily with
/// `needs_decoder` widened to every ChatML request, which is the change that
/// would alter four shipped families' output for callers who asked for
/// nothing.
#[tokio::test]
async fn no_level_means_no_decoder_and_no_reasoning_field() {
    let base = serve(ReasoningEffort::Off).await;
    let json = post(&base, "/v1/chat/completions", body(None)).await;

    let message = &json["choices"][0]["message"];
    assert!(
        message["reasoning_content"].is_null(),
        "an unasked-for request grew a reasoning field: {message}"
    );
    // The frame tokens render to the empty string, so an undecoded turn is
    // the two bodies run together -- which is exactly what a client got
    // before the decoder was wired, and is still what it gets here.
    let content = message["content"].as_str().unwrap_or_default();
    assert!(content.contains(ANSWER), "content = {content:?}");
    assert!(content.contains(REASONING), "content = {content:?}");
}
