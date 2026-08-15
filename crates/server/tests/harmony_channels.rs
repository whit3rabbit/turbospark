//! `gpt-oss`'s reasoning channel, end to end over both endpoints, with NO
//! model (ROADMAP Later list).
//!
//! The scripted backend replays a fixed logit sequence, so a test can force
//! the exact token stream a Harmony turn produces:
//!
//! ```text
//! <|channel|>analysis<|message|>REASONING<|end|>
//! <|start|>assistant<|channel|>final<|message|>ANSWER<|return|>
//! ```
//!
//! What is under test is the whole chain from that stream to the wire:
//! `StructuredAssistantDecoder` splits the channels, `stream_blocking` builds
//! a decoder for this dialect even with no tools in the request, and
//! `reasoning_content` reaches the client as an Anthropic `thinking` block.
//!
//! The fixture carries the real checkpoint's special-token NAMES and not its
//! ids (tokenizer crate Gotcha 3), so every id here is resolved from the
//! loaded tokenizer.

use std::path::PathBuf;
use std::sync::Arc;

use tokenizer::{Message, MfTokenizer, Role};
use turbospark_server::{build_router, ScriptedChatModel};

const REASONING: &str = "user asks about the sky";
const ANSWER: &str = "Rayleigh scattering.";
const USER: &str = "why is the sky blue";

fn load_tokenizer() -> MfTokenizer {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/HarmonyTokenizer");
    MfTokenizer::load_from_dir(&dir).expect("Harmony fixture tokenizer should load")
}

fn one_hot(vocab_size: usize, index: usize) -> Vec<foundation::LogitValue> {
    let mut v = vec![foundation::LogitValue::from_f32(0.0); vocab_size];
    v[index] = foundation::LogitValue::from_f32(1.0);
    v
}

/// The token ids of a complete Harmony assistant turn, ending at `<|return|>`
/// so the generation stops the way a real one does.
fn harmony_turn(tok: &MfTokenizer) -> Vec<i32> {
    let mark = |name: &str| tok.token_to_id(name).unwrap_or_else(|| panic!("{name}"));
    let mut ids = vec![tok.channel_start_id];
    ids.extend(tok.encode("analysis", false));
    ids.push(tok.message_start_id);
    ids.extend(tok.encode(REASONING, false));
    ids.push(tok.message_end_id);
    ids.push(mark("<|start|>"));
    ids.extend(tok.encode("assistant", false));
    ids.push(tok.channel_start_id);
    ids.extend(tok.encode("final", false));
    ids.push(tok.message_start_id);
    ids.extend(tok.encode(ANSWER, false));
    ids.push(tok.end_of_turn_id);
    ids
}

/// `plan` renders through the checkpoint's own template and encodes with
/// `add_bos = false`; the prefill consumes one scripted step per prompt token
/// but the last, so the decode steps have to start at exactly that offset.
fn steps_for_turn(tok: &MfTokenizer) -> Vec<Vec<foundation::LogitValue>> {
    let prompt = tok
        .apply_chat_template(&[Message::new(Role::User, USER)])
        .expect("the fixture ships a Harmony template");
    let prompt_ids = tok.encode(&prompt, false);

    let mut steps = vec![one_hot(tok.vocab_size, 0); prompt_ids.len() - 1];
    steps.extend(
        harmony_turn(tok)
            .iter()
            .map(|&id| one_hot(tok.vocab_size, id as usize)),
    );
    steps
}

async fn spawn_server() -> String {
    let tok = load_tokenizer();
    let steps = steps_for_turn(&tok);
    let model: Arc<dyn turbospark_server::ChatModel> =
        Arc::new(ScriptedChatModel::new(tok, 4096, steps));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, build_router(model)).await.unwrap();
    });
    format!("http://{addr}")
}

fn body(messages_max_tokens: u32) -> serde_json::Value {
    serde_json::json!({
        "model": "claude-sonnet-4-6",
        "max_tokens": messages_max_tokens,
        "temperature": 0.0,
        "messages": [{"role": "user", "content": USER}]
    })
}

/// THE HEADLINE: a request carrying NO tools still goes through the decoder on
/// this dialect, and the two channels come back as two Anthropic blocks in the
/// order the model produced them.
#[tokio::test]
async fn the_analysis_channel_becomes_an_anthropic_thinking_block() {
    let base = spawn_server().await;
    let response = reqwest::Client::new()
        .post(format!("{base}/v1/messages"))
        .json(&body(200))
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), 200);
    let body: serde_json::Value = response.json().await.unwrap();
    let kinds: Vec<&str> = body["content"]
        .as_array()
        .unwrap()
        .iter()
        .map(|b| b["type"].as_str().unwrap())
        .collect();
    assert_eq!(kinds, vec!["thinking", "text"], "{body}");

    let thinking = body["content"][0]["thinking"].as_str().unwrap();
    let text = body["content"][1]["text"].as_str().unwrap();
    assert!(thinking.contains(REASONING), "{thinking:?}");
    assert!(text.contains(ANSWER), "{text:?}");

    // The two must not bleed into each other, and none of the frame markup
    // may reach the client as either one. Before the channels were decoded,
    // ALL of this arrived as one run of text.
    assert!(
        !text.contains(REASONING),
        "reasoning leaked as text: {text:?}"
    );
    assert!(!thinking.contains(ANSWER), "answer leaked as reasoning");
    for markup in ["<|channel|>", "<|message|>", "<|end|>", "assistant"] {
        assert!(!text.contains(markup), "{markup} leaked into {text:?}");
    }
    assert_eq!(body["stop_reason"], "end_turn");
}

/// The same split in the streaming shape. Anthropic clients dispatch on the
/// `event:` line and on the delta type, so the ORDER is the contract: a
/// thinking block that opens, takes its deltas and closes before the text
/// block opens.
#[tokio::test]
async fn the_streamed_turn_emits_a_thinking_block_before_the_text_block() {
    let base = spawn_server().await;
    let mut request = body(200);
    request["stream"] = serde_json::Value::Bool(true);
    let response = reqwest::Client::new()
        .post(format!("{base}/v1/messages"))
        .json(&request)
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), 200);
    let raw = response.text().await.unwrap();
    let payloads: Vec<serde_json::Value> = raw
        .lines()
        .filter_map(|l| l.strip_prefix("data:"))
        .filter_map(|d| serde_json::from_str(d.trim()).ok())
        .collect();

    let opened: Vec<&str> = payloads
        .iter()
        .filter(|e| e["type"] == "content_block_start")
        .map(|e| e["content_block"]["type"].as_str().unwrap())
        .collect();
    assert_eq!(opened, vec!["thinking", "text"], "{raw}");

    let mut delta_kinds: Vec<&str> = payloads
        .iter()
        .filter_map(|e| e["delta"]["type"].as_str())
        .filter(|t| *t == "thinking_delta" || *t == "text_delta")
        .collect();
    delta_kinds.dedup();
    assert_eq!(delta_kinds, vec!["thinking_delta", "text_delta"], "{raw}");

    let thinking: String = payloads
        .iter()
        .filter(|e| e["delta"]["type"] == "thinking_delta")
        .filter_map(|e| e["delta"]["thinking"].as_str())
        .collect();
    assert!(thinking.contains(REASONING), "{thinking:?}");
}

/// The OpenAI endpoint carries the same split as `reasoning_content`, which is
/// the field the Anthropic mapping above reads. Asserting it directly is what
/// says the two endpoints agree rather than one of them inventing a shape.
#[tokio::test]
async fn the_openai_endpoint_carries_reasoning_content() {
    let base = spawn_server().await;
    let response = reqwest::Client::new()
        .post(format!("{base}/v1/chat/completions"))
        .json(&serde_json::json!({
            "model": "gpt-oss",
            "max_tokens": 200,
            "temperature": 0.0,
            "messages": [{"role": "user", "content": USER}]
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), 200);
    let body: serde_json::Value = response.json().await.unwrap();
    let message = &body["choices"][0]["message"];
    assert!(
        message["reasoning_content"]
            .as_str()
            .unwrap_or_default()
            .contains(REASONING),
        "{message}"
    );
    let content = message["content"].as_str().unwrap();
    assert!(content.contains(ANSWER), "{content:?}");
    assert!(!content.contains(REASONING), "{content:?}");
}
