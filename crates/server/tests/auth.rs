//! End-to-end tests for `build_router_with_options`'s `--api-key` layer.
//!
//! Same harness as `chat_completions.rs`: a real axum server on an ephemeral
//! loopback port, real tokenizer, scripted "model".

use std::path::PathBuf;
use std::sync::Arc;

use tokenizer::MfTokenizer;
use turbospark_server::{
    build_router, build_router_with_options, RouterOptions, ScriptedChatModel,
};

fn load_tokenizer() -> MfTokenizer {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/ChatMLTokenizer");
    MfTokenizer::load_from_dir(&dir).expect("fixture tokenizer should load")
}

fn one_hot(vocab_size: usize, index: usize) -> Vec<foundation::LogitValue> {
    let mut v = vec![foundation::LogitValue::from_f32(0.0); vocab_size];
    v[index] = foundation::LogitValue::from_f32(1.0);
    v
}

fn h_steps(tok: &MfTokenizer, count: usize) -> Vec<Vec<foundation::LogitValue>> {
    let h_id = tok.token_to_id("h").unwrap() as usize;
    (0..count).map(|_| one_hot(tok.vocab_size, h_id)).collect()
}

async fn spawn_protected(api_key: &str) -> String {
    let tok = load_tokenizer();
    let steps = h_steps(&tok, 50);
    let model: Arc<dyn turbospark_server::ChatModel> =
        Arc::new(ScriptedChatModel::new(tok, 4096, steps));
    let router = build_router_with_options(
        model,
        RouterOptions {
            api_key: Some(api_key.to_string()),
            ..Default::default()
        },
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });
    format!("http://{addr}")
}

fn chat_body() -> serde_json::Value {
    serde_json::json!({
        "model": "m", "max_tokens": 2, "temperature": 0.0,
        "messages": [{"role": "user", "content": "hi"}]
    })
}

#[tokio::test]
async fn a_request_with_no_credential_is_refused() {
    let base = spawn_protected("sk-correct").await;
    let response = reqwest::Client::new()
        .post(format!("{base}/v1/chat/completions"))
        .json(&chat_body())
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 401);
    let body: serde_json::Value = response.json().await.unwrap();
    assert_eq!(body["error"]["type"], "authentication_error");
}

/// **SAME LENGTH AS THE CORRECT KEY, DELIBERATELY.** `constant_time_eq`
/// refuses a length mismatch on its own fast path before ever touching the
/// byte-comparison loop; a wrong key of a DIFFERENT length would pass this
/// test without that loop doing anything, which is exactly the gap that let
/// a first version of this test survive `constant_time_eq`'s loop being
/// deleted outright.
#[tokio::test]
async fn a_wrong_key_of_the_same_length_is_refused() {
    let base = spawn_protected("sk-correct").await;
    let response = reqwest::Client::new()
        .post(format!("{base}/v1/chat/completions"))
        .header("x-api-key", "sk-wrongxx")
        .json(&chat_body())
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 401);
}

#[tokio::test]
async fn a_wrong_key_of_a_different_length_is_refused() {
    let base = spawn_protected("sk-correct").await;
    let response = reqwest::Client::new()
        .post(format!("{base}/v1/chat/completions"))
        .header("x-api-key", "short")
        .json(&chat_body())
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 401);
}

#[tokio::test]
async fn x_api_key_with_the_right_value_is_accepted() {
    let base = spawn_protected("sk-correct").await;
    let response = reqwest::Client::new()
        .post(format!("{base}/v1/chat/completions"))
        .header("x-api-key", "sk-correct")
        .json(&chat_body())
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
}

#[tokio::test]
async fn authorization_bearer_with_the_right_value_is_accepted() {
    let base = spawn_protected("sk-correct").await;
    let response = reqwest::Client::new()
        .post(format!("{base}/v1/chat/completions"))
        .header("authorization", "Bearer sk-correct")
        .json(&chat_body())
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
}

/// **THE ONE ROUTE THE LAYER MUST NEVER REACH.** A liveness probe answering
/// 401 to an unauthenticated caller is indistinguishable from the process
/// being down, which is the one thing it must never be wrong about.
#[tokio::test]
async fn health_is_exempt_from_the_api_key_requirement() {
    let base = spawn_protected("sk-correct").await;
    let response = reqwest::Client::new()
        .get(format!("{base}/health"))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
}

/// **`build_router` (no options) is untouched.** Every existing caller --
/// every other integration test file in this crate, and the FFI's future
/// in-process server -- must keep the exact unauthenticated behavior it had
/// before `--api-key` existed.
#[tokio::test]
async fn build_router_with_no_options_stays_unauthenticated() {
    let tok = load_tokenizer();
    let steps = h_steps(&tok, 50);
    let model: Arc<dyn turbospark_server::ChatModel> =
        Arc::new(ScriptedChatModel::new(tok, 4096, steps));
    let router = build_router(model);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });
    let base = format!("http://{addr}");

    let response = reqwest::Client::new()
        .post(format!("{base}/v1/chat/completions"))
        .json(&chat_body())
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
}

async fn spawn_keyless() -> String {
    let tok = load_tokenizer();
    let steps = h_steps(&tok, 50);
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

/// A page that DNS-rebinds its own name to 127.0.0.1 is same-origin to this
/// server, and its browser still sends that name as `Host`. The connection
/// itself is genuinely loopback, so `Host` is the only thing telling the two
/// apart. Each entry is a name a rebinding page could hold, or a malformed
/// value that must not parse its way into the loopback set.
const REFUSED_HOSTS: &[&str] = &[
    "rebind.example",
    "rebind.example:8080",
    // Loopback text as a prefix, inside a name the attacker owns.
    "127.0.0.1.rebind.example",
    "localhost.rebind.example:8080",
    "[::1].rebind.example",
    // Not a port, so not `localhost:<port>`.
    "localhost:rebind.example",
    "127.0.0.1:8080:rebind.example",
    "localhost:",
    // Malformed brackets and addresses that are not loopback.
    "[::1",
    "::1",
    "[localhost]",
    "[127.0.0.1]",
    "192.168.1.10:8080",
    "0.0.0.0:8080",
    "",
];

#[tokio::test]
async fn a_keyless_server_refuses_a_host_that_is_not_loopback() {
    let base = spawn_keyless().await;
    let client = reqwest::Client::new();
    for host in REFUSED_HOSTS {
        let chat = client
            .post(format!("{base}/v1/chat/completions"))
            .header(reqwest::header::HOST, *host)
            .json(&chat_body())
            .send()
            .await
            .unwrap();
        assert_eq!(chat.status(), 403, "chat, Host: {host:?}");
        let body: serde_json::Value = chat.json().await.unwrap();
        assert_eq!(body["error"]["type"], "permission_error", "Host: {host:?}");

        // A read route and the Ollama surface, not only OpenAI generation:
        // the rebinding page gets whatever the protected router serves.
        for path in ["/v1/models", "/api/tags"] {
            let read = client
                .get(format!("{base}{path}"))
                .header(reqwest::header::HOST, *host)
                .send()
                .await
                .unwrap();
            assert_eq!(read.status(), 403, "{path}, Host: {host:?}");
        }
    }
}

/// Not valid UTF-8, so it cannot be compared as a name at all. Sent apart
/// from `REFUSED_HOSTS` because it cannot be spelled as a `&str`.
#[tokio::test]
async fn a_keyless_server_refuses_a_host_that_is_not_text() {
    let base = spawn_keyless().await;
    let response = reqwest::Client::new()
        .get(format!("{base}/v1/models"))
        .header(
            reqwest::header::HOST,
            reqwest::header::HeaderValue::from_bytes(b"\xfflocalhost").unwrap(),
        )
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 403);
}

#[tokio::test]
async fn a_keyless_server_answers_loopback_hosts() {
    let base = spawn_keyless().await;
    let client = reqwest::Client::new();
    for host in [
        "127.0.0.1",
        "127.0.0.1:8080",
        "127.1.2.3:8080",
        "localhost",
        "localhost:8080",
        // Host names are case-insensitive, and clients do not all lowercase.
        "LocalHost:8080",
        "[::1]",
        "[::1]:8080",
        // What a client in a Docker Desktop container uses to reach a service
        // on the Mac's loopback. `.internal` has no public DNS, so a web page
        // cannot rebind to it.
        "host.docker.internal:8080",
    ] {
        let response = client
            .post(format!("{base}/v1/chat/completions"))
            .header(reqwest::header::HOST, host)
            .json(&chat_body())
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), 200, "Host: {host:?}");
    }
    // What a client sends when it was never told to override anything.
    let plain = client
        .post(format!("{base}/v1/chat/completions"))
        .json(&chat_body())
        .send()
        .await
        .unwrap();
    assert_eq!(plain.status(), 200);
}

/// `/health` answers the same whatever the `Host`, for the reason it skips the
/// key: a probe answered with a refusal reads as a process that is down. It
/// holds only `status` and `state`, so a rebinding page learns nothing it can
/// use with every other route refused.
#[tokio::test]
async fn health_is_exempt_from_the_host_check() {
    let base = spawn_keyless().await;
    let response = reqwest::Client::new()
        .get(format!("{base}/health"))
        .header(reqwest::header::HOST, "rebind.example")
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
}

/// The key is what stops a rebinding page here, since the page does not have
/// it. Tailnet binds require a key and clients reach them by MagicDNS name or
/// `100.x` address, so a loopback rule on this path would refuse every one.
#[tokio::test]
async fn a_server_with_a_key_answers_any_host_and_still_demands_the_key() {
    let base = spawn_protected("sk-correct").await;
    let client = reqwest::Client::new();
    for host in ["machine.tailnet.ts.net:8080", "100.100.1.1:8080"] {
        let keyed = client
            .post(format!("{base}/v1/chat/completions"))
            .header(reqwest::header::HOST, host)
            .header("x-api-key", "sk-correct")
            .json(&chat_body())
            .send()
            .await
            .unwrap();
        assert_eq!(keyed.status(), 200, "keyed, Host: {host:?}");

        let unkeyed = client
            .post(format!("{base}/v1/chat/completions"))
            .header(reqwest::header::HOST, host)
            .json(&chat_body())
            .send()
            .await
            .unwrap();
        assert_eq!(unkeyed.status(), 401, "unkeyed, Host: {host:?}");
    }
}
