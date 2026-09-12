//! What a host embedding this server can see about its own traffic.
//!
//! The two properties worth pinning are that a REJECTED request still shows
//! up (a console that only records successes is missing the rows somebody
//! opened it to find) and that the generation counters come off
//! `RawDecodeResult` rather than off a count of deltas -- `swift/CLAUDE.md`
//! Gotcha 7 is the worked example of how far apart those two numbers are.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use tokenizer::MfTokenizer;
use turbospark_server::observe::{ServerEvent, ServerObserver};
use turbospark_server::{build_router_with_options, ChatModel, RouterOptions, ScriptedChatModel};

fn load_tokenizer() -> MfTokenizer {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/ChatMLTokenizer");
    MfTokenizer::load_from_dir(&dir).expect("fixture tokenizer should load")
}

fn one_hot(vocab_size: usize, index: usize) -> Vec<foundation::LogitValue> {
    let mut v = vec![foundation::LogitValue::from_f32(0.0); vocab_size];
    v[index] = foundation::LogitValue::from_f32(1.0);
    v
}

#[derive(Default)]
struct Recorder(Mutex<Vec<ServerEvent>>);

impl ServerObserver for Recorder {
    fn record(&self, event: ServerEvent) {
        self.0.lock().unwrap().push(event);
    }
}

impl Recorder {
    /// The event KINDS in order, which is what most assertions here are
    /// really about.
    fn kinds(&self) -> Vec<String> {
        self.0
            .lock()
            .unwrap()
            .iter()
            .map(|e| {
                serde_json::to_value(e).unwrap()["kind"]
                    .as_str()
                    .unwrap()
                    .to_string()
            })
            .collect()
    }

    fn find(&self, kind: &str) -> Option<serde_json::Value> {
        self.0
            .lock()
            .unwrap()
            .iter()
            .map(|e| serde_json::to_value(e).unwrap())
            .find(|v| v["kind"] == kind)
    }
}

async fn serve(api_key: Option<&str>) -> (String, Arc<Recorder>) {
    let tok = load_tokenizer();
    let vocab = tok.vocab_size;
    let steps = (0..64).map(|_| one_hot(vocab, 5)).collect();
    let model: Arc<dyn ChatModel> = Arc::new(ScriptedChatModel::new(tok, 4096, steps));
    let recorder = Arc::new(Recorder::default());
    let router = build_router_with_options(
        model,
        RouterOptions {
            api_key: api_key.map(str::to_string),
            observer: Some(recorder.clone() as Arc<dyn ServerObserver>),
        },
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });
    (format!("http://{addr}"), recorder)
}

async fn chat(base: &str) -> reqwest::Response {
    reqwest::Client::new()
        .post(format!("{base}/v1/chat/completions"))
        .json(&serde_json::json!({
            "model": "whatever",
            "max_tokens": 3,
            "messages": [{"role": "user", "content": "hi"}],
        }))
        .send()
        .await
        .unwrap()
}

#[tokio::test]
async fn a_served_request_reports_started_routed_generated_finished() {
    let (base, recorder) = serve(None).await;
    assert_eq!(chat(&base).await.status().as_u16(), 200);

    assert_eq!(
        recorder.kinds(),
        vec![
            "requestStarted",
            "requestRouted",
            "generated",
            "requestFinished"
        ]
    );
}

/// **THE FIELD THAT MATTERS FOR A MULTI-MODEL CONSOLE.** The request named
/// `whatever` and the single-model fallback served `scripted`; both are
/// reported, because collapsing them hides the one fact somebody debugging
/// routing is looking for.
#[tokio::test]
async fn routing_reports_what_was_asked_for_and_what_answered() {
    let (base, recorder) = serve(None).await;
    chat(&base).await;

    let routed = recorder.find("requestRouted").expect("a routed event");
    assert_eq!(routed["requested"], "whatever");
    assert_eq!(routed["served"], "scripted");
    assert_eq!(routed["stream"], false);
}

/// Counts come off `RawDecodeResult`. A delta count would be LOWER here
/// (the fixture's chosen token renders to text, but special tokens do not
/// and a real dialect emits several), so asserting the exact budget is what
/// discriminates.
#[tokio::test]
async fn the_generation_counters_are_the_decoders_own() {
    let (base, recorder) = serve(None).await;
    chat(&base).await;

    let generated = recorder.find("generated").expect("a generated event");
    assert_eq!(generated["model"], "scripted");
    assert_eq!(
        generated["newTokens"], 3,
        "max_tokens was 3 and the scripted model never stops early"
    );
    assert!(generated["promptTokens"].as_u64().unwrap() > 0);
    // Present rather than checked for a value: the numbers are wall clock.
    // What is asserted is that BOTH phases are reported separately, which is
    // the whole reason there is no `ttftMs` field (see `observe.rs`).
    assert!(generated["prefillSeconds"].is_number());
    assert!(generated["decodeSeconds"].is_number());
}

/// **A REJECTED REQUEST IS STILL A ROW.** The observing layer sits OUTSIDE
/// the auth layer for exactly this: a 401 never reaches a handler, so
/// anything recorded from inside one would miss it entirely.
///
/// Mutation check: applying the observe layer to `protected` instead of to
/// the merged router reddens this case alone.
#[tokio::test]
async fn a_request_rejected_by_auth_is_still_recorded() {
    let (base, recorder) = serve(Some("sk-correct")).await;
    let response = chat(&base).await;
    assert_eq!(response.status().as_u16(), 401);

    assert_eq!(
        recorder.kinds(),
        vec!["requestStarted", "requestFinished"],
        "no handler ran, so there is nothing between the two"
    );
    let finished = recorder.find("requestFinished").expect("a finished event");
    assert_eq!(finished["status"], 401);
}

/// `/health` is exempt from AUTH and deliberately not from observation: a
/// host watching its own server wants to see probes too, and the layer
/// costs a request with no observer configured exactly nothing.
#[tokio::test]
async fn health_is_observed_even_though_it_is_exempt_from_auth() {
    let (base, recorder) = serve(Some("sk-correct")).await;
    let response = reqwest::get(format!("{base}/health")).await.unwrap();
    assert_eq!(response.status().as_u16(), 200);

    let started = recorder.find("requestStarted").expect("a started event");
    assert_eq!(started["path"], "/health");
    assert_eq!(started["method"], "GET");
}

/// A streamed turn still reports its counters, and the `Generated` event is
/// what closes it -- `RequestFinished` fires when the handler hands axum the
/// stream, which is BEFORE any token exists.
#[tokio::test]
async fn a_streamed_turn_reports_its_counters_too() {
    let (base, recorder) = serve(None).await;
    let response = reqwest::Client::new()
        .post(format!("{base}/v1/chat/completions"))
        .json(&serde_json::json!({
            "model": "whatever",
            "max_tokens": 3,
            "stream": true,
            "messages": [{"role": "user", "content": "hi"}],
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status().as_u16(), 200);
    // Drain it: the generation runs on a blocking task and the event lands
    // when that task finishes, not when the headers arrive.
    let _ = response.bytes().await.unwrap();

    let routed = recorder.find("requestRouted").expect("a routed event");
    assert_eq!(routed["stream"], true);
    let generated = recorder.find("generated").expect("a generated event");
    assert_eq!(generated["newTokens"], 3);
}

/// An error response records its status and the error message in `RequestFinished`.
#[tokio::test]
async fn an_error_response_records_its_error_message() {
    let (base, recorder) = serve(None).await;
    let response = reqwest::Client::new()
        .post(format!("{base}/v1/chat/completions"))
        .header("content-type", "application/json")
        .body("{not valid json}")
        .send()
        .await
        .unwrap();
    assert!(response.status().as_u16() >= 400);

    let finished = recorder.find("requestFinished").expect("a finished event");
    assert_eq!(finished["status"], response.status().as_u16());
    assert!(
        finished["error"].as_str().is_some(),
        "finished event should contain error message: {:?}",
        finished
    );
}
