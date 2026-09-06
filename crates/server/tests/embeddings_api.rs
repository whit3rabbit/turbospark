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
