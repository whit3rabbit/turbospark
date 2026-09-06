//! Tool-call guardrails end to end over both endpoints, with NO model.
//!
//! The scripted backend replays a fixed logit sequence, so a test can force
//! the exact token stream a bad turn produces and watch what reaches the wire.
//! Three things are under test and each fails differently without guardrails:
//!
//! - a call the decoder cannot parse (bare JSON on a ChatML checkpoint, which
//!   expects `<tool_call>` markup) reaches the client as PROSE;
//! - a call whose arguments violate the request's own schema reaches the
//!   client as a valid-looking `tool_calls` entry the caller then fails on;
//! - a request with NO tools must be untouched, and must still STREAM.
//!
//! Every case drives [`TwoTurnModel`], whose producer ignores prefill, so a
//! script is independent of how long the rendered prompt is -- which is what
//! makes the RETRY testable: a retry's prompt is longer than the first.
//!
//! The fixture carries the real checkpoint's special-token NAMES and not its
//! ids (tokenizer crate Gotcha 3), so every id here is resolved from the
//! loaded tokenizer.

use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use runtime::{LogitProducer, RawDecodeResult, RuntimeError};
use tokenizer::MfTokenizer;
use turbospark_server::{build_router, ChatModel, GuardrailConfig};

const USER: &str = "what is the weather in Oslo";

/// The call the model SHOULD have made, in a dialect ChatML's decoder does not
/// parse. `rescue_tool_call` recognises OpenAI's `{"name", "arguments"}` shape.
const BARE_JSON_CALL: &str = r#"{"name": "get_weather", "arguments": {"city": "Oslo"}}"#;

fn load_tokenizer() -> MfTokenizer {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/ChatMLTokenizer");
    MfTokenizer::load_from_dir(&dir).expect("ChatML fixture tokenizer should load")
}

/// A producer that emits a fixed TOKEN SEQUENCE on decode and ignores prefill
/// entirely.
///
/// **Not `ScriptedLogitProducer`, and the difference is what makes the retry
/// testable at all.** That one consumes a step per prefill token, so its
/// script has to be sized to the exact prompt -- and a retry's prompt is
/// LONGER than the first (it carries the failed assistant turn and the nudge),
/// so a script sized for the first generation runs out mid-prefill on the
/// second. Overriding `produce_prefill` to a no-op decouples the script from
/// the prompt: decode always starts at token 0 of `emit`, whatever was
/// rendered in front of it.
struct FixedTurn {
    emit: Vec<i32>,
    cursor: usize,
}

impl FixedTurn {
    fn new(tok: &MfTokenizer, text: &str) -> Self {
        let mut emit = tok.encode(text, false);
        emit.push(tok.end_of_turn_id);
        Self { emit, cursor: 0 }
    }

    /// A turn given as raw token ids, for scripts that must carry special
    /// tokens the markup lives in -- `new` can only ever produce them if
    /// `encode` happens to map the literal, which is not a thing to lean on.
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

    /// The whole point: prefill advances nothing, so the emitted sequence is
    /// independent of how long the rendered prompt is.
    fn produce_prefill(
        &mut self,
        _token: i32,
        _position: usize,
        _scratch: &mut [foundation::LogitValue],
    ) -> Result<(), String> {
        Ok(())
    }
}

const WEATHER_SCHEMA: &str =
    r#"{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}"#;

fn weather_body(stream: bool) -> serde_json::Value {
    serde_json::json!({
        "model": "test-model",
        "max_tokens": 200,
        "temperature": 0.0,
        "messages": [{"role": "user", "content": USER}],
        "tools": [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Current weather for a city",
                "parameters": serde_json::from_str::<serde_json::Value>(WEATHER_SCHEMA).unwrap(),
            }
        }],
        "stream": stream,
    })
}

async fn serve(model: Arc<dyn ChatModel>) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, build_router(model)).await.unwrap();
    });
    format!("http://{addr}")
}

/// A backend that answers with the FIRST text until it has been called once,
/// then with the second, counting calls so a test can prove a retry happened.
struct TwoTurnModel {
    tokenizer: MfTokenizer,
    first: String,
    second: String,
    /// The first turn as raw ids, set by [`Self::with_first_ids`]; when
    /// present it replaces `first` in the produced sequence.
    first_ids: Option<Vec<i32>>,
    calls: AtomicUsize,
    guardrails: GuardrailConfig,
}

impl TwoTurnModel {
    fn new(tokenizer: MfTokenizer, first: &str, second: &str, guardrails: GuardrailConfig) -> Self {
        Self {
            tokenizer,
            first: first.to_string(),
            second: second.to_string(),
            first_ids: None,
            calls: AtomicUsize::new(0),
            guardrails,
        }
    }

    fn with_first_ids(
        tokenizer: MfTokenizer,
        first_ids: Vec<i32>,
        second: &str,
        guardrails: GuardrailConfig,
    ) -> Self {
        Self {
            first_ids: Some(first_ids),
            tokenizer,
            first: String::new(),
            second: second.to_string(),
            calls: AtomicUsize::new(0),
            guardrails,
        }
    }
}

impl ChatModel for TwoTurnModel {
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
        self.guardrails
    }
    fn with_producer(
        &self,
        f: &mut dyn FnMut(&mut dyn LogitProducer) -> Result<RawDecodeResult, RuntimeError>,
    ) -> Result<RawDecodeResult, RuntimeError> {
        if self.calls.fetch_add(1, Ordering::SeqCst) == 0 {
            let mut producer = match &self.first_ids {
                Some(ids) => FixedTurn::from_ids(&self.tokenizer, ids.clone()),
                None => FixedTurn::new(&self.tokenizer, &self.first),
            };
            f(&mut producer)
        } else {
            let mut producer = FixedTurn::new(&self.tokenizer, &self.second);
            f(&mut producer)
        }
    }
}

fn tool_calls_of(body: &serde_json::Value) -> &Vec<serde_json::Value> {
    body["choices"][0]["message"]["tool_calls"]
        .as_array()
        .unwrap_or_else(|| panic!("no tool_calls in {body}"))
}

/// THE HEADLINE: a call the ChatML decoder cannot parse still reaches the
/// client as a `tool_calls` entry, and the raw markup does NOT reach it as
/// content.
#[tokio::test]
async fn a_call_the_decoder_missed_is_rescued_onto_the_wire() {
    let tok = load_tokenizer();
    let model = TwoTurnModel::new(
        tok,
        BARE_JSON_CALL,
        BARE_JSON_CALL,
        GuardrailConfig::default(),
    );
    let base = serve(Arc::new(model)).await;

    let response = reqwest::Client::new()
        .post(format!("{base}/v1/chat/completions"))
        .json(&weather_body(false))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    let body: serde_json::Value = response.json().await.unwrap();

    let calls = tool_calls_of(&body);
    assert_eq!(calls.len(), 1, "{body}");
    assert_eq!(calls[0]["function"]["name"], "get_weather", "{body}");
    assert!(
        calls[0]["function"]["arguments"]
            .as_str()
            .unwrap()
            .contains("Oslo"),
        "{body}"
    );
    // The markup must not ALSO arrive as prose: leaking a raw JSON blob to a
    // client as the assistant's answer is the failure being fixed.
    let content = body["choices"][0]["message"]["content"].as_str();
    assert!(
        content.is_none_or(|c| !c.contains("get_weather")),
        "raw markup leaked as content: {body}"
    );
}

/// The same rescue over the Anthropic endpoint, which reaches it through a
/// different translator.
#[tokio::test]
async fn the_anthropic_endpoint_rescues_it_as_a_tool_use_block() {
    let tok = load_tokenizer();
    let model = TwoTurnModel::new(
        tok,
        BARE_JSON_CALL,
        BARE_JSON_CALL,
        GuardrailConfig::default(),
    );
    let base = serve(Arc::new(model)).await;

    let response = reqwest::Client::new()
        .post(format!("{base}/v1/messages"))
        .json(&serde_json::json!({
            "model": "claude-sonnet-4-6",
            "max_tokens": 200,
            "temperature": 0.0,
            "messages": [{"role": "user", "content": USER}],
            "tools": [{
                "name": "get_weather",
                "description": "Current weather for a city",
                "input_schema": serde_json::from_str::<serde_json::Value>(WEATHER_SCHEMA).unwrap(),
            }],
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    let body: serde_json::Value = response.json().await.unwrap();

    let kinds: Vec<&str> = body["content"]
        .as_array()
        .unwrap_or_else(|| panic!("{body}"))
        .iter()
        .filter_map(|b| b["type"].as_str())
        .collect();
    assert!(kinds.contains(&"tool_use"), "{body}");
}

/// With guardrails off the SAME stream comes back as prose, which is what says
/// the test above measures the guardrail rather than the decoder.
#[tokio::test]
async fn with_guardrails_off_the_same_stream_is_prose() {
    let tok = load_tokenizer();
    let model = TwoTurnModel::new(tok, BARE_JSON_CALL, BARE_JSON_CALL, GuardrailConfig::OFF);
    let base = serve(Arc::new(model)).await;

    let response = reqwest::Client::new()
        .post(format!("{base}/v1/chat/completions"))
        .json(&weather_body(false))
        .send()
        .await
        .unwrap();
    let body: serde_json::Value = response.json().await.unwrap();
    assert!(
        body["choices"][0]["message"]["tool_calls"].is_null(),
        "guardrails off should not rescue: {body}"
    );
    let content = body["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or_default();
    assert!(content.contains("get_weather"), "{body}");
}

/// An invalid call is re-asked ONCE, and the second answer is what reaches the
/// client. The counting backend is what proves the re-generation happened at
/// all rather than the first turn being repaired in place.
#[tokio::test]
async fn an_invalid_call_is_retried_and_the_second_answer_wins() {
    let tok = load_tokenizer();
    // First turn: a call missing the required `city`. Second: a correct one.
    let model = Arc::new(TwoTurnModel::new(
        tok,
        r#"{"name": "get_weather", "arguments": {}}"#,
        BARE_JSON_CALL,
        GuardrailConfig::default(),
    ));
    let base = serve(model.clone()).await;

    let response = reqwest::Client::new()
        .post(format!("{base}/v1/chat/completions"))
        .json(&weather_body(false))
        .send()
        .await
        .unwrap();
    let body: serde_json::Value = response.json().await.unwrap();

    assert!(
        model.calls.load(Ordering::SeqCst) >= 2,
        "the turn should have been re-generated, saw {} call(s): {body}",
        model.calls.load(Ordering::SeqCst)
    );
    let calls = tool_calls_of(&body);
    assert!(
        calls[0]["function"]["arguments"]
            .as_str()
            .unwrap()
            .contains("Oslo"),
        "the SECOND answer should be the one returned: {body}"
    );
}

/// The retry budget is one. A backend that is wrong twice returns the second
/// answer rather than looping.
#[tokio::test]
async fn the_retry_budget_is_spent_rather_than_looped() {
    let tok = load_tokenizer();
    let model = Arc::new(TwoTurnModel::new(
        tok,
        r#"{"name": "get_weather", "arguments": {}}"#,
        r#"{"name": "get_weather", "arguments": {}}"#,
        GuardrailConfig::default(),
    ));
    let base = serve(model.clone()).await;

    let response = reqwest::Client::new()
        .post(format!("{base}/v1/chat/completions"))
        .json(&weather_body(false))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200, "a spent budget is not an error");
    assert_eq!(
        model.calls.load(Ordering::SeqCst),
        2,
        "one generation plus one retry, and no more"
    );
}

/// **THE INVARIANT: a request with no tools is untouched, and still STREAMS.**
///
/// The deltas have to arrive as separate chunks rather than one buffered
/// blob, which is what says the no-tools path never reaches
/// `buffered_stream_response`.
#[tokio::test]
async fn a_request_with_no_tools_still_streams_live() {
    let tok = load_tokenizer();
    let answer = "It is four degrees and raining.";
    let model = TwoTurnModel::new(tok, answer, answer, GuardrailConfig::default());
    let base = serve(Arc::new(model)).await;

    let text = reqwest::Client::new()
        .post(format!("{base}/v1/chat/completions"))
        .json(&serde_json::json!({
            "model": "test-model",
            "max_tokens": 200,
            "temperature": 0.0,
            "messages": [{"role": "user", "content": USER}],
            "stream": true,
        }))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();

    assert!(text.contains("[DONE]"), "{text}");
    // More than one content delta is the tell: the buffered path emits the
    // whole answer as a single `text_delta` chunk.
    let content_chunks = text
        .lines()
        .filter(|l| l.contains("\"content\":\"") && !l.contains("\"content\":\"\""))
        .count();
    assert!(
        content_chunks > 1,
        "a no-tools request must stream token by token, saw {content_chunks} content chunk(s):\n{text}"
    );
}

/// A tool-carrying stream is BUFFERED, and the buffered frames are still well
/// formed: role, content/tool calls, finish, `[DONE]`.
#[tokio::test]
async fn a_tool_request_streams_the_buffered_frames_in_order() {
    let tok = load_tokenizer();
    let model = TwoTurnModel::new(
        tok,
        BARE_JSON_CALL,
        BARE_JSON_CALL,
        GuardrailConfig::default(),
    );
    let base = serve(Arc::new(model)).await;

    let text = reqwest::Client::new()
        .post(format!("{base}/v1/chat/completions"))
        .json(&weather_body(true))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();

    let tool_at = text.find("tool_calls").unwrap_or_else(|| panic!("{text}"));
    let finish_at = text
        .find("\"finish_reason\":\"tool_calls\"")
        .or_else(|| text.find("\"finish_reason\":\"stop\""))
        .unwrap_or_else(|| panic!("no finish chunk in {text}"));
    // Same ordering assertion the Harmony streaming test needs: a call that
    // arrives after the finish chunk is invisible to a client that stops
    // reading there.
    assert!(tool_at < finish_at, "call must precede the finish: {text}");
    assert!(text.contains("[DONE]"), "{text}");
}

/// **THE DEGRADED-SPAN REGRESSION: bare JSON inside ChatML's
/// `<tool_call>` / `</tool_call>` special-token pair** -- the shape a
/// pre-3.5 Qwen emits, which the native 3.5+ XML parser refuses. The turn
/// must still complete, the span's body must survive the failed parse, and
/// the rescue layer must recover the call onto the wire. Before the failed
/// span body was released, this test's markup reached `generated.text`
/// with a hole exactly where the JSON was and no rescue was possible: the
/// call was lost twice over, once to the parser and once to the rescue.
#[tokio::test]
async fn a_malformed_chatml_tool_span_is_rescued_onto_the_wire() {
    let tok = load_tokenizer();
    let mut emit = tok.encode(" Sure, checking the weather. ", false);
    emit.push(tok.tool_call_start_id);
    emit.extend(tok.encode(BARE_JSON_CALL, false));
    emit.push(tok.tool_call_end_id);
    let model = TwoTurnModel::with_first_ids(tok, emit, BARE_JSON_CALL, GuardrailConfig::default());
    let base = serve(Arc::new(model)).await;

    let response = reqwest::Client::new()
        .post(format!("{base}/v1/chat/completions"))
        .json(&weather_body(false))
        .send()
        .await
        .unwrap();
    assert_eq!(
        response.status(),
        200,
        "a failed span parse is not a failed request"
    );
    let body: serde_json::Value = response.json().await.unwrap();

    let calls = tool_calls_of(&body);
    assert_eq!(calls.len(), 1, "{body}");
    assert_eq!(calls[0]["function"]["name"], "get_weather", "{body}");
    assert!(
        calls[0]["function"]["arguments"]
            .as_str()
            .unwrap()
            .contains("Oslo"),
        "{body}"
    );
    // The failed span's markup must not ALSO reach the client as prose.
    let content = body["choices"][0]["message"]["content"].as_str();
    assert!(
        content.is_none_or(|c| !c.contains("get_weather")),
        "span markup leaked as content: {body}"
    );
}

/// A GLM-format call reaches the wire as a `tool_calls` entry through the
/// SAME end-to-end path, which is what says `extra_formats` is wired into
/// [`inspect`] and not merely unit-tested beside it. The ChatML fixture's
/// decoder passes GLM markup through as ordinary text (its special tokens
/// are not GLM's), so the only thing standing between the markup and the
/// wire is the rescue.
#[tokio::test]
async fn a_glm_format_call_is_rescued_onto_the_wire() {
    let tok = load_tokenizer();
    let glm_call = "<tool_call>get_weather\n<arg_key>city</arg_key>\n\
                    <arg_value>Oslo</arg_value>\n</tool_call>";
    let model = TwoTurnModel::new(tok, glm_call, glm_call, GuardrailConfig::default());
    let base = serve(Arc::new(model)).await;

    let response = reqwest::Client::new()
        .post(format!("{base}/v1/chat/completions"))
        .json(&weather_body(false))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    let body: serde_json::Value = response.json().await.unwrap();

    let calls = tool_calls_of(&body);
    assert_eq!(calls.len(), 1, "{body}");
    assert_eq!(calls[0]["function"]["name"], "get_weather", "{body}");
    assert!(
        calls[0]["function"]["arguments"]
            .as_str()
            .unwrap()
            .contains("Oslo"),
        "{body}"
    );
    let content = body["choices"][0]["message"]["content"].as_str();
    assert!(
        content.is_none_or(|c| !c.contains("<arg_key>")),
        "GLM markup leaked as content: {body}"
    );
}
