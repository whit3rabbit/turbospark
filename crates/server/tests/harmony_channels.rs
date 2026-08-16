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
    serve(tok, steps).await
}

async fn serve(tok: MfTokenizer, steps: Vec<Vec<foundation::LogitValue>>) -> String {
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

// ---------------------------------------------------------------------------
// Tool calls (ROADMAP's Harmony tool-calling item).
// ---------------------------------------------------------------------------

const TOOL: &str = "get_weather";
const ARGUMENTS: &str = r#"{"city":"Oslo"}"#;

/// The ids of a Harmony tool call, ending at `<|call|>` so the generation stops
/// the way a real one does.
///
/// **`<|call|>` IS THE POINT OF THIS FIXTURE.** It is in the dialect's stop set,
/// so `run_raw_completion` breaks before the progress callback and the decoder
/// never sees it: everything downstream depends on the call being emitted from
/// `finish` instead, and on the stop ladder recognizing this token.
fn tool_call_turn(tok: &MfTokenizer) -> Vec<i32> {
    let mark = |name: &str| tok.token_to_id(name).unwrap_or_else(|| panic!("{name}"));
    let mut ids = vec![tok.channel_start_id];
    ids.extend(tok.encode(&format!("commentary to=functions.{TOOL} "), false));
    ids.push(mark("<|constrain|>"));
    ids.extend(tok.encode("json", false));
    ids.push(tok.message_start_id);
    ids.extend(tok.encode(ARGUMENTS, false));
    ids.push(mark("<|call|>"));
    ids
}

fn tool_definition() -> serde_json::Value {
    serde_json::json!({
        "type": "function",
        "function": {
            "name": TOOL,
            "description": "Get the current weather in a given city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    })
}

/// A request carrying tools takes the OTHER prompt path (server crate Gotcha
/// 7), so the prefill offset has to be reconstructed through the same call
/// `plan` makes rather than through `apply_chat_template`.
async fn spawn_tool_server() -> String {
    let tok = load_tokenizer();
    let tools = vec![tokenizer::FunctionDefinition {
        name: TOOL.to_string(),
        description: "Get the current weather in a given city".to_string(),
        parameters: tokenizer::JsonValue::Null,
    }];
    let prompt_ids = tok
        .encode_generic_tool_chat(&[Message::new(Role::User, USER)], &tools, false)
        .expect("the fixture ships a template");

    let mut steps = vec![one_hot(tok.vocab_size, 0); prompt_ids.len() - 1];
    steps.extend(
        tool_call_turn(&tok)
            .iter()
            .map(|&id| one_hot(tok.vocab_size, id as usize)),
    );
    serve(tok, steps).await
}

fn tool_body() -> serde_json::Value {
    serde_json::json!({
        "model": "claude-sonnet-4-6",
        "max_tokens": 200,
        "temperature": 0.0,
        "tools": [{
            "name": TOOL,
            "description": "Get the current weather in a given city",
            "input_schema": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        }],
        "messages": [{"role": "user", "content": USER}]
    })
}

/// THE HEADLINE: a Harmony call reaches an Anthropic client as a `tool_use`
/// block with `stop_reason: "tool_use"`.
///
/// Both halves used to be wrong and in ways that looked like nothing: the call
/// was never emitted (its terminator is a stop token the decoder cannot see)
/// and the finish reason fell through to `end_turn`, because the ladder read
/// Gemma's tool-RESPONSE marker, which this dialect does not have.
#[tokio::test]
async fn a_harmony_tool_call_becomes_an_anthropic_tool_use_block() {
    let base = spawn_tool_server().await;
    let response = reqwest::Client::new()
        .post(format!("{base}/v1/messages"))
        .json(&tool_body())
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), 200);
    let body: serde_json::Value = response.json().await.unwrap();
    let blocks = body["content"].as_array().unwrap();
    let use_block = blocks
        .iter()
        .find(|b| b["type"] == "tool_use")
        .unwrap_or_else(|| panic!("no tool_use block in {body}"));

    assert_eq!(use_block["name"], TOOL, "the namespace is stripped");
    assert_eq!(use_block["input"]["city"], "Oslo");
    assert_eq!(body["stop_reason"], "tool_use", "{body}");

    // The arguments are the model's markup, not its answer: none of the body
    // may also arrive as text.
    let text: String = blocks
        .iter()
        .filter_map(|b| b["text"].as_str())
        .collect::<Vec<_>>()
        .join("");
    assert!(!text.contains("city"), "the call leaked as text: {text:?}");
}

/// The same call over the OpenAI endpoint, which is the shape the Anthropic
/// mapping above is built from. Asserting it directly is what says the two
/// endpoints agree rather than one of them inventing something.
#[tokio::test]
async fn the_openai_endpoint_carries_the_call_and_the_tool_calls_finish_reason() {
    let base = spawn_tool_server().await;
    let mut request = tool_body();
    request["tools"] = serde_json::json!([tool_definition()]);
    let response = reqwest::Client::new()
        .post(format!("{base}/v1/chat/completions"))
        .json(&request)
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), 200);
    let body: serde_json::Value = response.json().await.unwrap();
    let choice = &body["choices"][0];
    let call = &choice["message"]["tool_calls"][0];

    assert_eq!(call["function"]["name"], TOOL, "{body}");
    assert_eq!(call["function"]["arguments"], ARGUMENTS);
    assert_eq!(choice["finish_reason"], "tool_calls", "{body}");
}

/// A STREAMED call has to arrive BEFORE the finish chunk, which is the one
/// ordering constraint emitting from `finish` could plausibly have broken: the
/// call is produced after `run_raw_completion` has already returned.
#[tokio::test]
async fn a_streamed_call_arrives_before_the_finish_chunk() {
    let base = spawn_tool_server().await;
    let mut request = tool_body();
    request["tools"] = serde_json::json!([tool_definition()]);
    request["stream"] = serde_json::Value::Bool(true);
    let response = reqwest::Client::new()
        .post(format!("{base}/v1/chat/completions"))
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

    let call_at = payloads
        .iter()
        .position(|e| e["choices"][0]["delta"]["tool_calls"].is_array())
        .unwrap_or_else(|| panic!("no tool_call delta in {raw}"));
    let finish_at = payloads
        .iter()
        .position(|e| e["choices"][0]["finish_reason"] == "tool_calls")
        .unwrap_or_else(|| panic!("no tool_calls finish in {raw}"));
    assert!(call_at < finish_at, "call after finish chunk: {raw}");

    let call = &payloads[call_at]["choices"][0]["delta"]["tool_calls"][0];
    assert_eq!(call["function"]["name"], TOOL);
    assert_eq!(call["function"]["arguments"], ARGUMENTS);
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
