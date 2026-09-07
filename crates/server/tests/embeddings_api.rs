use std::path::PathBuf;
use std::sync::Arc;

use tokenizer::MfTokenizer;
use turbospark_server::{build_router, ScriptedChatModel};

fn load_tokenizer() -> MfTokenizer {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/ChatMLTokenizer");
    MfTokenizer::load_from_dir(&dir).expect("fixture tokenizer should load")
}

async fn spawn_server() -> String {
    let tok = load_tokenizer();
    let model: Arc<dyn turbospark_server::ChatModel> =
        Arc::new(ScriptedChatModel::new(tok, 4096, vec![]));
    let router = build_router(model);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });
    format!("http://{addr}")
}

#[tokio::test]
async fn test_v1_embeddings_single_input() {
    let base = spawn_server().await;

    let response = reqwest::Client::new()
        .post(format!("{base}/v1/embeddings"))
        .json(&serde_json::json!({
            "model": "scripted",
            "input": "The quick brown fox"
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), reqwest::StatusCode::OK);
    let body: serde_json::Value = response.json().await.unwrap();

    assert_eq!(body["object"], "list");
    assert_eq!(body["model"], "scripted");
    let data = body["data"].as_array().expect("data array");
    assert_eq!(data.len(), 1);
    assert_eq!(data[0]["object"], "embedding");
    assert_eq!(data[0]["index"], 0);

    let emb: Vec<f32> = serde_json::from_value(data[0]["embedding"].clone()).unwrap();
    assert_eq!(emb.len(), 4);

    let norm: f32 = emb.iter().map(|v| v * v).sum::<f32>().sqrt();
    assert!(
        (norm - 1.0).abs() < 1e-5,
        "embedding must be unit-normalized, got {norm}"
    );
}

#[tokio::test]
async fn test_v1_embeddings_batch_and_dimensions() {
    let base = spawn_server().await;

    let response = reqwest::Client::new()
        .post(format!("{base}/v1/embeddings"))
        .json(&serde_json::json!({
            "model": "scripted",
            "input": ["First sentence", "Second sentence"],
            "dimensions": 2
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), reqwest::StatusCode::OK);
    let body: serde_json::Value = response.json().await.unwrap();

    let data = body["data"].as_array().expect("data array");
    assert_eq!(data.len(), 2);
    assert_eq!(data[0]["index"], 0);
    assert_eq!(data[1]["index"], 1);

    let emb0: Vec<f32> = serde_json::from_value(data[0]["embedding"].clone()).unwrap();
    let emb1: Vec<f32> = serde_json::from_value(data[1]["embedding"].clone()).unwrap();
    assert_eq!(emb0.len(), 2);
    assert_eq!(emb1.len(), 2);

    let norm0: f32 = emb0.iter().map(|v| v * v).sum::<f32>().sqrt();
    assert!((norm0 - 1.0).abs() < 1e-5);
    let norm1: f32 = emb1.iter().map(|v| v * v).sum::<f32>().sqrt();
    assert!((norm1 - 1.0).abs() < 1e-5);
}

/// F8: `dimensions: 0` and a `dimensions` wider than the model's own width
/// (4, for the scripted fixture's fixed 4-element vector) must both 400
/// rather than the old silent behaviour (0 returned an empty vector with
/// 200; a too-wide value was returned unmodified with no error).
#[tokio::test]
async fn dimensions_zero_or_too_wide_is_refused() {
    let base = spawn_server().await;

    let zero = reqwest::Client::new()
        .post(format!("{base}/v1/embeddings"))
        .json(&serde_json::json!({"model": "scripted", "input": "hi", "dimensions": 0}))
        .send()
        .await
        .unwrap();
    assert_eq!(zero.status(), reqwest::StatusCode::BAD_REQUEST);

    let too_wide = reqwest::Client::new()
        .post(format!("{base}/v1/embeddings"))
        .json(&serde_json::json!({"model": "scripted", "input": "hi", "dimensions": 999}))
        .send()
        .await
        .unwrap();
    assert_eq!(too_wide.status(), reqwest::StatusCode::BAD_REQUEST);
    let text = too_wide.text().await.unwrap();
    assert!(text.contains("dimensions"), "{text}");
}

/// F8: `encoding_format: "base64"` (the stock OpenAI Python SDK's default)
/// must return a base64 string decoding to the SAME bytes `"float"` returns
/// as a JSON array, and an unrecognised format must 400 rather than being
/// silently ignored.
#[tokio::test]
async fn encoding_format_base64_round_trips_and_unknown_is_refused() {
    let base = spawn_server().await;

    let float_resp = reqwest::Client::new()
        .post(format!("{base}/v1/embeddings"))
        .json(&serde_json::json!({"model": "scripted", "input": "hi", "encoding_format": "float"}))
        .send()
        .await
        .unwrap();
    assert_eq!(float_resp.status(), reqwest::StatusCode::OK);
    let float_body: serde_json::Value = float_resp.json().await.unwrap();
    let float_vec: Vec<f32> =
        serde_json::from_value(float_body["data"][0]["embedding"].clone()).unwrap();

    let b64_resp = reqwest::Client::new()
        .post(format!("{base}/v1/embeddings"))
        .json(&serde_json::json!({"model": "scripted", "input": "hi", "encoding_format": "base64"}))
        .send()
        .await
        .unwrap();
    assert_eq!(b64_resp.status(), reqwest::StatusCode::OK);
    let b64_body: serde_json::Value = b64_resp.json().await.unwrap();
    let encoded = b64_body["data"][0]["embedding"]
        .as_str()
        .expect("base64 embedding must be a JSON string, not an array");
    let decoded_vec = decode_base64_f32(encoded);
    assert_eq!(
        decoded_vec, float_vec,
        "base64 must decode to the same floats"
    );

    let bad = reqwest::Client::new()
        .post(format!("{base}/v1/embeddings"))
        .json(&serde_json::json!({"model": "scripted", "input": "hi", "encoding_format": "int8"}))
        .send()
        .await
        .unwrap();
    assert_eq!(bad.status(), reqwest::StatusCode::BAD_REQUEST);
}

/// Minimal standard-base64 decoder, independent of the server's own encoder,
/// so this test cannot pass merely because both sides share one bug.
fn decode_base64_f32(s: &str) -> Vec<f32> {
    const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let value = |c: u8| ALPHABET.iter().position(|&a| a == c).unwrap() as u32;
    let clean: Vec<u8> = s.bytes().filter(|&b| b != b'=').collect();
    let mut bits = Vec::new();
    for &b in &clean {
        let v = value(b);
        for shift in (0..6).rev() {
            bits.push((v >> shift) & 1);
        }
    }
    let mut bytes = Vec::new();
    for chunk in bits.chunks(8) {
        if chunk.len() < 8 {
            break;
        }
        let mut byte = 0u8;
        for &bit in chunk {
            byte = (byte << 1) | bit as u8;
        }
        bytes.push(byte);
    }
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

#[tokio::test]
async fn test_ollama_embeddings_and_embed() {
    let base = spawn_server().await;

    // 1. /api/embeddings (legacy)
    let resp = reqwest::Client::new()
        .post(format!("{base}/api/embeddings"))
        .json(&serde_json::json!({
            "model": "scripted",
            "prompt": "Test prompt"
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let body: serde_json::Value = resp.json().await.unwrap();
    let emb: Vec<f32> = serde_json::from_value(body["embedding"].clone()).unwrap();
    assert_eq!(emb.len(), 4);

    // 2. /api/embed (batch)
    let resp2 = reqwest::Client::new()
        .post(format!("{base}/api/embed"))
        .json(&serde_json::json!({
            "model": "scripted",
            "input": ["A", "B"]
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp2.status(), reqwest::StatusCode::OK);
    let body2: serde_json::Value = resp2.json().await.unwrap();
    let embs: Vec<Vec<f32>> = serde_json::from_value(body2["embeddings"].clone()).unwrap();
    assert_eq!(embs.len(), 2);
    assert_eq!(embs[0].len(), 4);
    assert_eq!(embs[1].len(), 4);
}

#[tokio::test]
async fn test_v1_embeddings_omitted_model_and_default_alias() {
    let base = spawn_server().await;

    // 1. Model omitted entirely -> resolves to default embedding model
    let resp1 = reqwest::Client::new()
        .post(format!("{base}/v1/embeddings"))
        .json(&serde_json::json!({
            "input": "Embedding with no model specified"
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp1.status(), reqwest::StatusCode::OK);
    let body1: serde_json::Value = resp1.json().await.unwrap();
    assert_eq!(body1["object"], "list");
    let data1 = body1["data"].as_array().expect("data array");
    assert_eq!(data1.len(), 1);

    // 2. Generic default model name (text-embedding-3-small) -> routes to default embedding model
    let resp2 = reqwest::Client::new()
        .post(format!("{base}/v1/embeddings"))
        .json(&serde_json::json!({
            "model": "text-embedding-3-small",
            "input": "Embedding with OpenAI default model name"
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp2.status(), reqwest::StatusCode::OK);
    let body2: serde_json::Value = resp2.json().await.unwrap();
    assert_eq!(body2["object"], "list");
    let data2 = body2["data"].as_array().expect("data array");
    assert_eq!(data2.len(), 1);
}

/// F10: an oversized batch must 400 before it ever reaches the encoder,
/// rather than tying up a blocking thread on a request size nobody bounded.
#[tokio::test]
async fn an_oversized_input_batch_is_refused() {
    let base = spawn_server().await;
    let too_many: Vec<&str> = std::iter::repeat_n("x", 2049).collect();

    let response = reqwest::Client::new()
        .post(format!("{base}/v1/embeddings"))
        .json(&serde_json::json!({"model": "scripted", "input": too_many}))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), reqwest::StatusCode::BAD_REQUEST);
    let text = response.text().await.unwrap();
    assert!(text.contains("too many inputs"), "{text}");
}

/// The same cap on total text size, reachable with far fewer than the count
/// limit's number of inputs. Sized under axum's own 2 MiB JSON body limit so
/// this exercises the application-level check rather than the framework's.
#[tokio::test]
async fn an_oversized_total_input_is_refused() {
    let base = spawn_server().await;
    let huge = "x".repeat(1_500_000);

    let response = reqwest::Client::new()
        .post(format!("{base}/v1/embeddings"))
        .json(&serde_json::json!({"model": "scripted", "input": huge}))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), reqwest::StatusCode::BAD_REQUEST);
    let text = response.text().await.unwrap();
    assert!(text.contains("input too large"), "{text}");
}
