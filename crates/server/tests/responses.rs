//! End-to-end tests for the OpenAI `POST /v1/responses` endpoint.
//!
//! Same harness as `chat_completions.rs`: a real axum server on an ephemeral
//! loopback port, real tokenizer, scripted "model". The streaming tests
//! assert on the exact `event:` name SEQUENCE, which is the actual contract
//! a Responses client's state machine depends on -- the same reason
//! `messages.rs`'s streaming test checks Anthropic event order rather than
//! just "the response was 200".

use std::path::PathBuf;
use std::sync::Arc;

use runtime::{LogitProducer, RawDecodeResult, RuntimeError};
use tokenizer::MfTokenizer;
use turbospark_server::{build_router, ChatModel, GuardrailConfig, ScriptedChatModel};

fn load_tokenizer() -> MfTokenizer {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/ChatMLTokenizer");
    MfTokenizer::load_from_dir(&dir).expect("fixture tokenizer should load")
}

fn one_hot(vocab_size: usize, index: usize) -> Vec<foundation::LogitValue> {
    let mut v = vec![foundation::LogitValue::from_f32(0.0); vocab_size];
    v[index] = foundation::LogitValue::from_f32(1.0);
    v
}

async fn spawn_server(steps: Vec<Vec<foundation::LogitValue>>) -> String {
    let tok = load_tokenizer();
    let model: Arc<dyn turbospark_server::ChatModel> =
        Arc::new(ScriptedChatModel::new(tok, 4096, steps));
    let router = build_router(model);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });
    format!("http://{addr}")
}

fn h_steps(tok: &MfTokenizer, count: usize) -> Vec<Vec<foundation::LogitValue>> {
    let h_id = tok.token_to_id("h").unwrap() as usize;
    (0..count).map(|_| one_hot(tok.vocab_size, h_id)).collect()
}

fn event_names(body: &str) -> Vec<String> {
    body.lines()
        .filter_map(|l| l.strip_prefix("event:"))
        .map(|n| n.trim().to_string())
        .collect()
}

/// Pairs each `event:` line with the `data:` line that follows it, parsed as
/// JSON -- the same association a real client's SSE parser makes, rather
/// than assuming exact line spacing between the two.
fn events_with_data(body: &str) -> Vec<(String, serde_json::Value)> {
    let mut out = Vec::new();
    let mut current: Option<String> = None;
    for line in body.lines() {
        if let Some(name) = line.strip_prefix("event:") {
            current = Some(name.trim().to_string());
        } else if let Some(data) = line.strip_prefix("data:") {
            if let Some(name) = current.take() {
                if let Ok(value) = serde_json::from_str::<serde_json::Value>(data.trim()) {
                    out.push((name, value));
                }
            }
        }
    }
    out
}

#[tokio::test]
async fn non_streaming_response_returns_a_message_output_item() {
    let tok = load_tokenizer();
    let base = spawn_server(h_steps(&tok, 50)).await;

    let response = reqwest::Client::new()
        .post(format!("{base}/v1/responses"))
        .json(&serde_json::json!({
            "model": "m", "input": "hi", "max_output_tokens": 3, "temperature": 0.0
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), 200);
    let body: serde_json::Value = response.json().await.unwrap();
    assert_eq!(body["type"], "response");
    assert_eq!(body["status"], "incomplete"); // capped by max_output_tokens
    let output = body["output"].as_array().unwrap();
    assert_eq!(output.len(), 1);
    assert_eq!(output[0]["type"], "message");
    assert_eq!(output[0]["role"], "assistant");
    let text = output[0]["content"][0]["text"].as_str().unwrap();
    assert!(!text.is_empty());
    assert_eq!(body["usage"]["output_tokens"], 3);
}

#[tokio::test]
async fn an_items_array_input_is_accepted() {
    let tok = load_tokenizer();
    let base = spawn_server(h_steps(&tok, 50)).await;

    let response = reqwest::Client::new()
        .post(format!("{base}/v1/responses"))
        .json(&serde_json::json!({
            "model": "m",
            "input": [{"type": "message", "role": "user", "content": "hi"}],
            "max_output_tokens": 2, "temperature": 0.0
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
}

#[tokio::test]
async fn previous_response_id_is_refused() {
    let base = spawn_server(Vec::new()).await;
    let response = reqwest::Client::new()
        .post(format!("{base}/v1/responses"))
        .json(&serde_json::json!({
            "model": "m", "input": "hi", "previous_response_id": "resp_1"
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 400);
    let text = response.text().await.unwrap();
    assert!(text.contains("previous_response_id"), "{text}");
}

/// F5: a plan failure on the NON-STREAMING path used to reach `run_guarded`
/// with no prior validation, and `run_guarded`'s own re-plan maps a failure
/// to `GenError::Join` -- a 500 for a request the caller sent wrong, where
/// `/v1/chat/completions` and `/v1/messages` both 400 on the identical
/// mistake because they plan before ever calling `run_guarded`.
#[tokio::test]
async fn a_bad_reasoning_effort_400s_rather_than_500s_non_streaming() {
    let tok = load_tokenizer();
    let base = spawn_server(h_steps(&tok, 50)).await;

    let response = reqwest::Client::new()
        .post(format!("{base}/v1/responses"))
        .json(&serde_json::json!({
            "model": "m", "input": "hi", "reasoning_effort": "bogus"
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 400);
    let text = response.text().await.unwrap();
    assert!(text.contains("reasoning_effort"), "{text}");
}

/// The same gap on the BUFFERED streaming path (tools present, guardrails
/// active by default): `stream_response` used to route straight to
/// `buffered_stream_response` with no validation at all, so this request
/// would have opened an SSE stream and only failed once the guarded
/// generation's own re-plan ran, deep inside a "response.failed" event -- an
/// unnecessary generation attempt and a 200 status for a request that was
/// never going to succeed. It must 400 before the stream ever opens.
#[tokio::test]
async fn a_bad_reasoning_effort_400s_rather_than_streaming_a_failure_event() {
    let base = spawn_server(Vec::new()).await;

    let response = reqwest::Client::new()
        .post(format!("{base}/v1/responses"))
        .json(&serde_json::json!({
            "model": "m", "input": "hi", "stream": true, "reasoning_effort": "bogus",
            "tools": [{
                "type": "function", "name": "get_weather",
                "parameters": {"type": "object", "properties": {}}
            }]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 400);
    let text = response.text().await.unwrap();
    assert!(text.contains("reasoning_effort"), "{text}");
    assert!(!text.contains("response.failed"), "{text}");
}

#[tokio::test]
async fn an_unknown_input_item_type_is_refused() {
    let base = spawn_server(Vec::new()).await;
    let response = reqwest::Client::new()
        .post(format!("{base}/v1/responses"))
        .json(&serde_json::json!({
            "model": "m", "input": [{"type": "reasoning", "summary": []}]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 400);
}

#[tokio::test]
async fn store_true_is_reported_on_the_degradation_header() {
    let tok = load_tokenizer();
    let base = spawn_server(h_steps(&tok, 50)).await;
    let response = reqwest::Client::new()
        .post(format!("{base}/v1/responses"))
        .json(&serde_json::json!({
            "model": "m", "input": "hi", "max_output_tokens": 2, "temperature": 0.0, "store": true
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    let header = response
        .headers()
        .get("x-anyllm-degradation")
        .expect("store: true must be reported")
        .to_str()
        .unwrap()
        .to_string();
    assert!(header.contains("store"), "{header}");
}

#[tokio::test]
async fn a_plain_request_carries_no_degradation_header() {
    let tok = load_tokenizer();
    let base = spawn_server(h_steps(&tok, 50)).await;
    let response = reqwest::Client::new()
        .post(format!("{base}/v1/responses"))
        .json(&serde_json::json!({
            "model": "m", "input": "hi", "max_output_tokens": 2, "temperature": 0.0
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    assert!(response.headers().get("x-anyllm-degradation").is_none());
}

/// **THE POINT OF THIS TEST**: a Responses client's state machine dispatches
/// on the `event:` line, so the SEQUENCE is the actual contract, not just
/// "some events arrived". `response.created` opens the stream and
/// `response.completed` closes it; a text turn opens and closes exactly one
/// message item and one content part in between.
#[tokio::test]
async fn streaming_response_emits_events_in_the_documented_order() {
    let tok = load_tokenizer();
    let base = spawn_server(h_steps(&tok, 50)).await;

    let response = reqwest::Client::new()
        .post(format!("{base}/v1/responses"))
        .json(&serde_json::json!({
            "model": "m", "input": "hi", "max_output_tokens": 2, "temperature": 0.0,
            "stream": true
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), 200);
    let body = response.text().await.unwrap();
    let names = event_names(&body);

    assert_eq!(names.first().map(String::as_str), Some("response.created"));
    assert_eq!(names.last().map(String::as_str), Some("response.completed"));
    assert_eq!(
        names,
        vec![
            "response.created",
            "response.output_item.added",
            "response.content_part.added",
            "response.output_text.delta",
            "response.output_text.delta",
            "response.output_text.done",
            "response.content_part.done",
            "response.output_item.done",
            "response.completed",
        ],
        "{names:?}"
    );
    // `[DONE]` is a Chat-Completions-ism; a Responses stream ends at
    // response.completed, the same convention the Anthropic endpoint uses.
    assert!(!body.contains("[DONE]"));
}

/// A fixed token sequence, ignoring prefill entirely -- `crate::guardrails`'s
/// integration tests use the identical shape so the script is independent of
/// how long the rendered (tool-carrying) prompt is.
struct FixedTurn {
    emit: Vec<i32>,
    cursor: usize,
}

impl FixedTurn {
    fn from_ids(tok: &MfTokenizer, mut emit: Vec<i32>) -> Self {
        emit.push(tok.end_of_turn_id);
        Self { emit, cursor: 0 }
    }
}

impl LogitProducer for FixedTurn {
    fn reset(&mut self) {
        self.cursor = 0;
    }

    fn produce(
        &mut self,
        _token: i32,
        _position: usize,
        logits: &mut [foundation::LogitValue],
    ) -> Result<(), String> {
        let id = *self
            .emit
            .get(self.cursor)
            .ok_or_else(|| "fixed turn exhausted".to_string())?;
        self.cursor += 1;
        logits.fill(foundation::LogitValue::from_f32(0.0));
        logits[id as usize] = foundation::LogitValue::from_f32(1.0);
        Ok(())
    }

    fn produce_prefill(
        &mut self,
        _token: i32,
        _position: usize,
        _scratch: &mut [foundation::LogitValue],
    ) -> Result<(), String> {
        Ok(())
    }
}

/// A model that emits a fixed token sequence with guardrails OFF, so a
/// tool-carrying request stays on the LIVE per-token streaming path
/// (`stream_response`) rather than being buffered
/// (`!tools.is_empty() && model.guardrails().active()`) -- the buffered path
/// cannot reproduce F4's bug, since it always emits text before calls in one
/// shot from `Generated`'s already-separated `text`/`calls` fields, never
/// interleaved as the decoder actually produced them.
struct LiveToolTurn {
    tokenizer: MfTokenizer,
    emit: Vec<i32>,
}

impl ChatModel for LiveToolTurn {
    fn tokenizer(&self) -> &MfTokenizer {
        &self.tokenizer
    }
    fn vocab_size(&self) -> usize {
        self.tokenizer.vocab_size
    }
    fn max_context(&self) -> u32 {
        4096
    }
    fn model_id(&self) -> &str {
        "test-model"
    }
    fn guardrails(&self) -> GuardrailConfig {
        GuardrailConfig::OFF
    }
    fn with_producer(
        &self,
        f: &mut dyn FnMut(&mut dyn LogitProducer) -> Result<RawDecodeResult, RuntimeError>,
    ) -> Result<RawDecodeResult, RuntimeError> {
        let mut producer = FixedTurn::from_ids(&self.tokenizer, self.emit.clone());
        f(&mut producer)
    }
}

/// F4: text streamed AFTER a tool call must not carry the text streamed
/// BEFORE it. The live decoder emits `Piece::Text("Sure, checking. ")`,
/// `Piece::Tool(get_weather)`, `Piece::Text(" All set.")` in that order for
/// this script, so a driver that never clears its text accumulator between
/// the two text spans would send `output_text.done` for the second span
/// carrying "Sure, checking.  All set." -- and would reuse `msg_0` /
/// `output_index 0` for both spans, which is Responses' own contract for
/// "this is the SAME item" and is false here.
#[tokio::test]
async fn a_tool_call_does_not_duplicate_text_streamed_after_it() {
    let tok = load_tokenizer();
    let mark = |name: &str| tok.token_to_id(name).unwrap_or_else(|| panic!("{name}"));
    let mut emit = tok.encode("Sure, checking. ", false);
    emit.push(mark("<tool_call>"));
    emit.extend(tok.encode(
        "\n<function=get_weather>\n<parameter=city>\nOslo\n</parameter>\n</function>\n",
        false,
    ));
    emit.push(mark("</tool_call>"));
    emit.extend(tok.encode(" All set.", false));

    let model: Arc<dyn ChatModel> = Arc::new(LiveToolTurn {
        tokenizer: tok,
        emit,
    });
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, build_router(model)).await.unwrap();
    });
    let base = format!("http://{addr}");

    let response = reqwest::Client::new()
        .post(format!("{base}/v1/responses"))
        .json(&serde_json::json!({
            "model": "m", "input": "what is the weather in Oslo", "stream": true,
            "temperature": 0.0,
            "tools": [{
                "type": "function", "name": "get_weather",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}
            }]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    let body = response.text().await.unwrap();

    let names = event_names(&body);
    // Two distinct message items (one per text span, either side of the
    // call), each opened, closed, and never reopened.
    assert_eq!(
        names
            .iter()
            .filter(|n| *n == "response.output_item.added")
            .count(),
        3,
        "{names:?}"
    );
    assert_eq!(
        names
            .iter()
            .filter(|n| *n == "response.output_text.done")
            .count(),
        2,
        "{names:?}"
    );

    // The regression itself: the SECOND span's done text is exactly its own
    // delta, never the first span's text concatenated with it. Paired up by
    // walking `event:`/`data:` line pairs, the way a real client's SSE
    // parser does, rather than assuming exact line spacing.
    let events = events_with_data(&body);
    let done_texts: Vec<String> = events
        .iter()
        .filter(|(name, _)| name == "response.output_text.done")
        .filter_map(|(_, data)| data["text"].as_str().map(str::to_string))
        .collect();
    assert_eq!(done_texts.len(), 2, "{events:?}");
    assert_eq!(done_texts[0], "Sure, checking. ", "{done_texts:?}");
    assert_eq!(
        done_texts[1], " All set.",
        "second span must not carry the first span's text: {done_texts:?}"
    );

    // The two message items must carry DISTINCT ids -- reusing "msg_0" for
    // both is what let the second span's `output_item.done` silently
    // overwrite the first's in a client keyed by item id.
    let (_, completed) = events
        .iter()
        .find(|(name, _)| name == "response.completed")
        .expect("a response.completed event");
    let output = completed["response"]["output"].as_array().unwrap();
    let message_ids: Vec<&str> = output
        .iter()
        .filter(|item| item["type"] == "message")
        .map(|item| item["id"].as_str().unwrap())
        .collect();
    assert_eq!(message_ids.len(), 2, "{output:?}");
    assert_ne!(message_ids[0], message_ids[1], "{output:?}");
    assert_eq!(
        output
            .iter()
            .filter(|i| i["type"] == "function_call")
            .count(),
        1
    );
}

#[tokio::test]
async fn streaming_response_carries_the_degradation_header() {
    let tok = load_tokenizer();
    let base = spawn_server(h_steps(&tok, 50)).await;
    let response = reqwest::Client::new()
        .post(format!("{base}/v1/responses"))
        .json(&serde_json::json!({
            "model": "m", "input": "hi", "max_output_tokens": 2, "temperature": 0.0,
            "stream": true, "store": true
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    let header = response
        .headers()
        .get("x-anyllm-degradation")
        .expect("store: true must be reported on the streaming path too")
        .to_str()
        .unwrap()
        .to_string();
    assert!(header.contains("store"), "{header}");
}

#[tokio::test]
async fn malformed_request_body_is_rejected() {
    let base = spawn_server(Vec::new()).await;
    let response = reqwest::Client::new()
        .post(format!("{base}/v1/responses"))
        .header("content-type", "application/json")
        .body("not json")
        .send()
        .await
        .unwrap();
    assert!(response.status().is_client_error());
}
