use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use base64::Engine;
use turbospark_server::registry::StaticRegistry;
use turbospark_server::{
    build_router_with_options, ImageError, ImageGenerateRequest, ImageProvider, RouterOptions,
    ServerState,
};

#[derive(Default)]
struct FakeImages {
    calls: Mutex<Vec<ImageGenerateRequest>>,
}

impl ImageProvider for FakeImages {
    fn models(&self) -> Vec<String> {
        vec!["z-test".into()]
    }

    fn generate(
        &self,
        request: ImageGenerateRequest,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<u8>, ImageError>> + Send + '_>> {
        Box::pin(async move {
            self.calls.lock().unwrap().push(request);
            Ok(b"\x89PNG\r\n\x1a\nfake-bytes".to_vec())
        })
    }
}

async fn spawn(fake: Option<Arc<FakeImages>>) -> String {
    let registry = Arc::new(StaticRegistry::new(vec![]).unwrap());
    let mut state = ServerState::new(registry);
    if let Some(fake) = fake {
        state = state.with_image_provider(fake);
    }
    let router = build_router_with_options(
        state,
        RouterOptions {
            api_key: Some("secret".into()),
            ..Default::default()
        },
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
    format!("http://{addr}")
}

#[tokio::test]
async fn image_route_auth_discovery_and_serial_results() {
    let fake = Arc::new(FakeImages::default());
    let base = spawn(Some(Arc::clone(&fake))).await;
    let client = reqwest::Client::new();
    let body = serde_json::json!({
        "model":"z-test", "prompt":"a fox", "size":"512x512", "n":2, "seed":10
    });
    let unauthorized = client
        .post(format!("{base}/v1/images/generations"))
        .json(&body)
        .send()
        .await
        .unwrap();
    assert_eq!(unauthorized.status(), 401);

    let health: serde_json::Value = client
        .get(format!("{base}/health"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(health["state"], "ready");

    let models: serde_json::Value = client
        .get(format!("{base}/v1/models"))
        .bearer_auth("secret")
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert!(models["data"]
        .as_array()
        .unwrap()
        .iter()
        .any(|model| model["id"] == "z-test"));

    let response = client
        .post(format!("{base}/v1/images/generations"))
        .bearer_auth("secret")
        .json(&body)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    assert_eq!(response.headers()["x-turbospark-seed"], "10");
    let result: serde_json::Value = response.json().await.unwrap();
    assert!(result["created"].as_u64().is_some());
    assert_eq!(result["data"].as_array().unwrap().len(), 2);
    for item in result["data"].as_array().unwrap() {
        assert_eq!(
            base64::engine::general_purpose::STANDARD
                .decode(item["b64_json"].as_str().unwrap())
                .unwrap(),
            b"\x89PNG\r\n\x1a\nfake-bytes"
        );
    }
    let calls = fake.calls.lock().unwrap();
    assert_eq!(calls.len(), 2);
    assert_eq!((calls[0].seed, calls[1].seed), (10, 11));
    assert_eq!((calls[0].width, calls[0].height), (512, 512));
}

#[tokio::test]
async fn image_route_rejects_unsupported_options_and_missing_provider() {
    let base = spawn(Some(Arc::new(FakeImages::default()))).await;
    let client = reqwest::Client::new();
    for (field, value) in [
        ("n", serde_json::json!(5)),
        ("size", serde_json::json!("128x128")),
        ("response_format", serde_json::json!("url")),
        ("stream", serde_json::json!(true)),
        ("mask", serde_json::json!("not supported")),
    ] {
        let mut body = serde_json::json!({"model":"z-test", "prompt":"a fox"});
        body[field] = value;
        let response = client
            .post(format!("{base}/v1/images/generations"))
            .bearer_auth("secret")
            .json(&body)
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), 400, "{field}");
    }
    let empty = spawn(None).await;
    let response = client
        .post(format!("{empty}/v1/images/generations"))
        .bearer_auth("secret")
        .json(&serde_json::json!({"model":"z-test", "prompt":"a fox"}))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 503);
    let edits = client
        .post(format!("{base}/v1/images/edits"))
        .bearer_auth("secret")
        .send()
        .await
        .unwrap();
    assert_eq!(edits.status(), 501);
    let error: serde_json::Value = edits.json().await.unwrap();
    assert!(error["error"]["message"]
        .as_str()
        .unwrap()
        .contains("unsupported"));
}
