//! `timings`, `cached_tokens` and the text series of `/v1/metrics`, over real
//! HTTP against `ScriptedChatModel`.

use std::path::PathBuf;
use std::sync::Arc;

use serde_json::{json, Value};
use tokenizer::MfTokenizer;
use turbospark_server::{build_router_with_options, ChatModel, RouterOptions, ScriptedChatModel};

const KEY: &str = "test-key";

async fn spawn(steps: usize, api_key: Option<&str>) -> String {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/ChatMLTokenizer");
    let tok = MfTokenizer::load_from_dir(&dir).expect("fixture tokenizer should load");
    let mut one_hot = vec![foundation::LogitValue::from_f32(0.0); tok.vocab_size];
    one_hot[tok.token_to_id("h").unwrap() as usize] = foundation::LogitValue::from_f32(1.0);
    let model: Arc<dyn ChatModel> =
        Arc::new(ScriptedChatModel::new(tok, 4096, vec![one_hot; steps]));
    let router = build_router_with_options(
        model,
        RouterOptions {
            api_key: api_key.map(str::to_string),
            ..Default::default()
        },
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
    format!("http://{addr}")
}

async fn post(base: &str, path: &str, body: Value) -> reqwest::Response {
    reqwest::Client::new()
        .post(format!("{base}{path}"))
        .json(&body)
        .send()
        .await
        .unwrap()
}

fn chat(stream: bool, include_usage: bool) -> Value {
    json!({
        "model": "secret-model-name", "max_tokens": 3, "temperature": 0.0,
        "messages": [{"role": "user", "content": "hi"}],
        "stream": stream, "stream_options": {"include_usage": include_usage}
    })
}

fn completion(stream: bool) -> Value {
    json!({"model": "m", "prompt": "hi", "max_tokens": 3, "temperature": 0.0, "stream": stream})
}

async fn chunks(response: reqwest::Response) -> Vec<Value> {
    response
        .text()
        .await
        .unwrap()
        .lines()
        .filter_map(|l| l.strip_prefix("data: "))
        .filter(|d| *d != "[DONE]")
        .map(|d| serde_json::from_str(d).unwrap())
        .collect()
}

fn check_timings(timings: &Value, prompt_tokens: &Value) {
    assert_eq!(timings["predicted_n"], 3, "{timings}");
    // The scripted backend reuses nothing, so all of the prompt is prefilled.
    assert_eq!(&timings["prompt_n"], prompt_tokens, "{timings}");
    for key in ["prompt_ms", "predicted_ms"] {
        assert!(timings[key].as_f64().unwrap() >= 0.0, "{key} {timings}");
    }
    for key in ["prompt_per_second", "predicted_per_second"] {
        assert!(timings.get(key).is_some(), "{key} {timings}");
    }
}

#[tokio::test]
async fn non_streaming_responses_carry_timings_and_cached_tokens() {
    let base = spawn(64, None).await;
    for (path, request) in [
        ("/v1/chat/completions", chat(false, false)),
        ("/v1/completions", completion(false)),
    ] {
        let body: Value = post(&base, path, request).await.json().await.unwrap();
        check_timings(&body["timings"], &body["usage"]["prompt_tokens"]);
        assert_eq!(
            body["usage"]["prompt_tokens_details"]["cached_tokens"], 0,
            "{path}: {body}"
        );
    }
}

/// Only the last chunk carries `timings`, whichever chunk that is.
#[tokio::test]
async fn the_final_streaming_chunk_carries_timings() {
    let base = spawn(64, None).await;
    for (path, request) in [
        ("/v1/chat/completions", chat(true, true)),
        ("/v1/chat/completions", chat(true, false)),
        ("/v1/completions", completion(true)),
    ] {
        let all = chunks(post(&base, path, request).await).await;
        let (last, earlier) = all.split_last().unwrap();
        assert!(
            earlier.iter().all(|c| c.get("timings").is_none()),
            "{path}: {all:?}"
        );
        assert_eq!(last["timings"]["predicted_n"], 3, "{path}: {last}");
        if let Some(usage) = last.get("usage") {
            check_timings(&last["timings"], &usage["prompt_tokens"]);
            assert_eq!(usage["prompt_tokens_details"]["cached_tokens"], 0);
        }
    }
}

async fn scrape(base: &str) -> String {
    let response = reqwest::Client::new()
        .get(format!("{base}/v1/metrics"))
        .bearer_auth(KEY)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    response.text().await.unwrap()
}

fn value(text: &str, name: &str) -> f64 {
    let prefix = format!("turbospark_text_{name} ");
    text.lines()
        .find_map(|l| l.strip_prefix(&prefix))
        .unwrap_or_else(|| panic!("no {name} in\n{text}"))
        .parse()
        .unwrap()
}

#[tokio::test]
async fn metrics_count_generations_and_need_the_key() {
    // Enough steps for the short prompt, not for the long one, which fails.
    let base = spawn(64, Some(KEY)).await;
    let client = reqwest::Client::new();
    let open = client
        .get(format!("{base}/v1/metrics"))
        .send()
        .await
        .unwrap();
    assert_eq!(open.status(), 401);

    let send = |body: Value| {
        client
            .post(format!("{base}/v1/chat/completions"))
            .bearer_auth(KEY)
            .json(&body)
            .send()
    };
    let ok: Value = send(chat(false, false))
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let mut long = chat(false, false);
    long["messages"][0]["content"] = json!("hi ".repeat(200));
    assert_eq!(send(long).await.unwrap().status(), 500);

    let text = scrape(&base).await;
    assert_eq!(value(&text, "requests_total"), 2.0);
    assert_eq!(value(&text, "requests_failed_total"), 1.0);
    assert_eq!(
        value(&text, "prompt_tokens_total"),
        ok["usage"]["prompt_tokens"].as_f64().unwrap()
    );
    assert_eq!(value(&text, "predicted_tokens_total"), 3.0);
    assert!(value(&text, "prefill_seconds_total") >= 0.0);
    assert!(value(&text, "decode_seconds_total") >= 0.0);
    assert_eq!(value(&text, "queue_waiting"), 0.0);
    assert_eq!(value(&text, "queue_rejected_total"), 0.0);
    // No labels at all, so nothing a client sent can appear.
    assert!(!text.contains('{'), "{text}");
    assert!(!text.contains("secret-model-name"), "{text}");
    // No audio provider, no audio series.
    assert!(!text.contains("turbospark_audio"), "{text}");
}
