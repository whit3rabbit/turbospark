//! Routing a request to one of several attached models, and the
//! single-model fallback that keeps every pre-registry client working.
//!
//! The fallback is the case worth reading first: `crates/server/src/
//! registry.rs`'s header records why a request naming a model this server
//! has never heard of is SERVED rather than refused when only one is
//! attached, and `one_model_serves_a_name_it_does_not_know` below is the
//! documented Claude Code invocation turned into an assertion.

use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{mpsc, Arc, Condvar, Mutex};
use std::thread::{self, ThreadId};
use std::time::Duration;

use tokenizer::MfTokenizer;
use turbospark_server::registry::{
    LoadOutcome, LoadRefusal, ModelLoader, ModelLoaderRegistry, ModelSuggestion, StaticRegistry,
    SuggestionSource,
};
use turbospark_server::{build_router, ChatModel, ScriptedChatModel};

fn load_tokenizer() -> MfTokenizer {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/ChatMLTokenizer");
    MfTokenizer::load_from_dir(&dir).expect("fixture tokenizer should load")
}

fn one_hot(vocab_size: usize, index: usize) -> Vec<foundation::LogitValue> {
    let mut v = vec![foundation::LogitValue::from_f32(0.0); vocab_size];
    v[index] = foundation::LogitValue::from_f32(1.0);
    v
}

/// `ScriptedChatModel::model_id` is the constant `"scripted"`, so a test
/// with two distinguishable backends needs a wrapper that names them. It
/// also makes the ROUTING observable: each model's id comes back on the
/// response's own `model` field.
struct Named(ScriptedChatModel, String);

impl ChatModel for Named {
    fn tokenizer(&self) -> &MfTokenizer {
        self.0.tokenizer()
    }
    fn vocab_size(&self) -> usize {
        self.0.vocab_size()
    }
    fn max_context(&self) -> u32 {
        self.0.max_context()
    }
    fn model_id(&self) -> &str {
        &self.1
    }
    fn with_producer(
        &self,
        f: &mut dyn FnMut(
            &mut dyn runtime::LogitProducer,
        ) -> Result<runtime::RawDecodeResult, runtime::RuntimeError>,
    ) -> Result<runtime::RawDecodeResult, runtime::RuntimeError> {
        self.0.with_producer(f)
    }
}

fn named(id: &str) -> Arc<dyn ChatModel> {
    let tok = load_tokenizer();
    let vocab = tok.vocab_size;
    // Enough steps to prefill a short prompt and decode a couple of tokens.
    let steps = (0..64).map(|_| one_hot(vocab, 5)).collect();
    Arc::new(Named(
        ScriptedChatModel::new(tok, 4096, steps),
        id.to_string(),
    ))
}

async fn serve_registry(registry: Arc<dyn turbospark_server::registry::ModelRegistry>) -> String {
    let router = build_router(registry);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });
    format!("http://{addr}")
}

async fn serve(models: Vec<Arc<dyn ChatModel>>) -> String {
    let registry: Arc<dyn turbospark_server::registry::ModelRegistry> =
        Arc::new(StaticRegistry::new(models).expect("test model ids are unique"));
    serve_registry(registry).await
}

async fn chat(base: &str, model: &str) -> (u16, serde_json::Value) {
    let response = reqwest::Client::new()
        .post(format!("{base}/v1/chat/completions"))
        .json(&serde_json::json!({
            "model": model,
            "max_tokens": 2,
            "messages": [{"role": "user", "content": "hi"}],
        }))
        .send()
        .await
        .unwrap();
    let status = response.status().as_u16();
    (status, response.json().await.unwrap())
}

#[tokio::test]
async fn two_models_are_both_listed_with_their_own_context_windows() {
    let base = serve(vec![named("alpha.gturbo"), named("beta.gturbo")]).await;
    let body: serde_json::Value = reqwest::get(format!("{base}/v1/models"))
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let ids: Vec<&str> = body["data"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| m["id"].as_str().unwrap())
        .collect();
    assert_eq!(
        ids,
        vec![
            "alpha.gturbo",
            "claude-turbospark-alpha.gturbo",
            "beta.gturbo",
            "claude-turbospark-beta.gturbo",
        ]
    );
    assert_eq!(body["data"][0]["context_window"], 4096);
}

#[tokio::test]
async fn an_exact_model_id_reaches_that_model() {
    let base = serve(vec![named("alpha.gturbo"), named("beta.gturbo")]).await;
    for id in ["alpha.gturbo", "beta.gturbo"] {
        let (status, body) = chat(&base, id).await;
        assert_eq!(status, 200, "{id} should have been served");
        // The response echoes the request's own model name, so routing is
        // What this asserts is that both ids are ACCEPTED, which is the half
        // a 404 would break.
        assert_eq!(body["object"], "chat.completion");
    }
}

#[tokio::test]
async fn an_exact_claude_alias_reaches_its_model_on_a_multi_model_server() {
    let base = serve(vec![named("alpha.gturbo"), named("beta.gturbo")]).await;
    let (status, body) = chat(&base, "claude-turbospark-beta.gturbo").await;
    assert_eq!(status, 200);
    assert_eq!(body["object"], "chat.completion");
}

/// **THE CASE THE FALLBACK EXISTS FOR.** `docs/CLI.md` points Claude Code at
/// this server with `ANTHROPIC_BASE_URL`, and it sends
/// `"model": "claude-sonnet-4-6"` at an install named nothing of the sort.
/// Every OpenAI SDK does the same with its own default. Routing strictly on
/// the name would 404 all of them.
///
/// Mutation check: deleting `resolve_among`'s `[only] =>` arm reddens this
/// case alone and leaves the two-model cases green.
#[tokio::test]
async fn one_model_serves_a_name_it_does_not_know() {
    let base = serve(vec![named("gemma4.gturbo")]).await;
    let (status, body) = chat(&base, "claude-sonnet-4-6").await;
    assert_eq!(status, 200);
    assert_eq!(body["object"], "chat.completion");
}

/// With two attached there is a real ambiguity, so the same request is
/// refused -- and the refusal names what IS there, because a caller who
/// guessed wrong has no other way to find out.
#[tokio::test]
async fn several_models_refuse_an_unknown_name_and_name_the_alternatives() {
    let base = serve(vec![named("alpha.gturbo"), named("beta.gturbo")]).await;
    let (status, body) = chat(&base, "claude-sonnet-4-6").await;
    assert_eq!(status, 404);
    assert_eq!(body["error"]["code"], "model_not_found");
    let message = body["error"]["message"].as_str().unwrap();
    assert!(message.contains("alpha.gturbo"), "{message}");
    assert!(message.contains("beta.gturbo"), "{message}");
}

struct SuggestedModelLoader {
    suggestions: Vec<ModelSuggestion>,
    load_count: AtomicUsize,
    candidate_count: AtomicUsize,
}

impl ModelLoader for SuggestedModelLoader {
    fn load(&self, _id: &str) -> LoadOutcome {
        self.load_count.fetch_add(1, Ordering::SeqCst);
        LoadOutcome::Unavailable {
            candidates: Vec::new(),
        }
    }

    fn candidates(&self, _requested: &str) -> Vec<ModelSuggestion> {
        self.candidate_count.fetch_add(1, Ordering::SeqCst);
        self.suggestions.clone()
    }
}

struct CountingModel {
    inner: Arc<dyn ChatModel>,
    generations: Arc<AtomicUsize>,
}

impl ChatModel for CountingModel {
    fn tokenizer(&self) -> &MfTokenizer {
        self.inner.tokenizer()
    }

    fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }

    fn max_context(&self) -> u32 {
        self.inner.max_context()
    }

    fn model_id(&self) -> &str {
        self.inner.model_id()
    }

    fn with_producer(
        &self,
        f: &mut dyn FnMut(
            &mut dyn runtime::LogitProducer,
        ) -> Result<runtime::RawDecodeResult, runtime::RuntimeError>,
    ) -> Result<runtime::RawDecodeResult, runtime::RuntimeError> {
        self.generations.fetch_add(1, Ordering::SeqCst);
        self.inner.with_producer(f)
    }
}

struct RefusingModelLoader {
    loads: AtomicUsize,
}

impl ModelLoader for RefusingModelLoader {
    fn load(&self, _id: &str) -> LoadOutcome {
        self.loads.fetch_add(1, Ordering::SeqCst);
        LoadOutcome::Refused(LoadRefusal::DoesNotFit {
            committed_bytes: 12,
            ceiling_bytes: 10,
        })
    }

    fn candidates(&self, _requested: &str) -> Vec<ModelSuggestion> {
        Vec::new()
    }
}

#[tokio::test]
async fn unknown_model_error_separates_loaded_ids_from_installed_and_catalog_suggestions() {
    let loader = Arc::new(SuggestedModelLoader {
        suggestions: vec![
            ModelSuggestion {
                id: "installed-close-match".to_string(),
                label: "Installed close match".to_string(),
                sources: vec![SuggestionSource::Installed],
            },
            ModelSuggestion {
                id: "catalog-close-match".to_string(),
                label: "Catalog close match".to_string(),
                sources: vec![SuggestionSource::Catalog],
            },
        ],
        load_count: AtomicUsize::new(0),
        candidate_count: AtomicUsize::new(0),
    });
    let base: Arc<dyn turbospark_server::registry::ModelRegistry> = Arc::new(
        StaticRegistry::new(vec![named("alpha.gturbo"), named("beta.gturbo")])
            .expect("test model ids are unique"),
    );
    let registry: Arc<dyn turbospark_server::registry::ModelRegistry> =
        Arc::new(ModelLoaderRegistry::new(base, loader.clone()));
    let base_url = serve_registry(registry).await;

    let (status, body) = chat(&base_url, "requested-name").await;

    assert_eq!(status, 404);
    assert_eq!(body["error"]["code"], "model_not_found");
    assert_eq!(body["error"]["available"][0], "alpha.gturbo");
    assert_eq!(
        body["error"]["suggestions"][0]["id"],
        "installed-close-match"
    );
    assert_eq!(body["error"]["suggestions"][0]["sources"][0], "installed");
    assert_eq!(body["error"]["suggestions"][1]["id"], "catalog-close-match");
    assert_eq!(body["error"]["suggestions"][1]["sources"][0], "catalog");
    assert_eq!(loader.load_count.load(Ordering::SeqCst), 1);
    assert_eq!(loader.candidate_count.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn a_load_refusal_is_retryable_and_does_not_start_generation() {
    let generations = Arc::new(AtomicUsize::new(0));
    let attached: Arc<dyn ChatModel> = Arc::new(CountingModel {
        inner: named("attached.gturbo"),
        generations: Arc::clone(&generations),
    });
    let base: Arc<dyn turbospark_server::registry::ModelRegistry> =
        Arc::new(StaticRegistry::new(vec![attached]).expect("test model ids are unique"));
    let loader = Arc::new(RefusingModelLoader {
        loads: AtomicUsize::new(0),
    });
    let registry: Arc<dyn turbospark_server::registry::ModelRegistry> =
        Arc::new(ModelLoaderRegistry::new(base, loader.clone()));
    let base_url = serve_registry(registry).await;

    let (status, body) = chat(&base_url, "installed.gturbo").await;

    assert_eq!(status, 503);
    assert_eq!(body["error"]["code"], "model_load_refused");
    assert_eq!(body["error"]["retryable"], true);
    assert_eq!(generations.load(Ordering::SeqCst), 0);
    assert_eq!(loader.loads.load(Ordering::SeqCst), 1);
}

/// A server can run with nothing attached: that is the state a GUI starts
/// one in before loading a model. 503 rather than 404, because the request
/// is fine and the server is not ready -- a distinction a client's retry
/// logic acts on.
#[tokio::test]
async fn an_empty_server_reports_itself_unavailable_rather_than_missing() {
    let base = serve(Vec::new()).await;
    let (status, _) = chat(&base, "anything").await;
    assert_eq!(status, 503);

    let health: serde_json::Value = reqwest::get(format!("{base}/health"))
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(health["state"], "empty");
    assert_eq!(health["status"], "ok");
}

/// `/v1/models/:id` answers about the id it was ASKED about, and must not
/// inherit the single-model fallback: the fallback serves generations,
/// where a lookup's whole question is whether this exact id exists.
///
/// Mutation check: routing `model_detail` through `registry.resolve` instead
/// of through `rows()` reddens the second half of this and nothing else.
#[tokio::test]
async fn a_model_lookup_does_not_take_the_single_model_fallback() {
    let base = serve(vec![named("gemma4.gturbo")]).await;

    let found = reqwest::get(format!("{base}/v1/models/gemma4.gturbo"))
        .await
        .unwrap();
    assert_eq!(found.status().as_u16(), 200);

    let alias = reqwest::get(format!("{base}/v1/models/claude-turbospark-gemma4.gturbo"))
        .await
        .unwrap();
    assert_eq!(alias.status().as_u16(), 200);
    let detail: serde_json::Value = alias.json().await.unwrap();
    assert_eq!(detail["id"], "claude-turbospark-gemma4.gturbo");

    let missing = reqwest::get(format!("{base}/v1/models/not-a-model"))
        .await
        .unwrap();
    assert_eq!(
        missing.status().as_u16(),
        404,
        "a lookup must not answer for a name it does not have"
    );
}

struct ThreadRecordingLoader {
    load_thread: Arc<Mutex<Option<ThreadId>>>,
}

impl ModelLoader for ThreadRecordingLoader {
    fn load(&self, _id: &str) -> LoadOutcome {
        *self.load_thread.lock().unwrap() = Some(thread::current().id());
        LoadOutcome::Refused(LoadRefusal::OpenFailed {
            detail: "test refusal".to_string(),
        })
    }

    fn candidates(&self, _requested: &str) -> Vec<ModelSuggestion> {
        Vec::new()
    }
}

#[tokio::test(flavor = "current_thread")]
async fn model_resolution_runs_off_the_request_runtime_thread() {
    let runtime_thread = thread::current().id();
    let load_thread = Arc::new(Mutex::new(None));
    let base: Arc<dyn turbospark_server::registry::ModelRegistry> =
        Arc::new(StaticRegistry::new(Vec::new()).expect("empty registry is valid"));
    let loader = Arc::new(ThreadRecordingLoader {
        load_thread: Arc::clone(&load_thread),
    });
    let registry: Arc<dyn turbospark_server::registry::ModelRegistry> =
        Arc::new(ModelLoaderRegistry::new(base, loader));
    let base_url = serve_registry(registry).await;

    let (status, body) = chat(&base_url, "installed.gturbo").await;

    assert_eq!(status, 503, "{body}");
    let load_thread = load_thread
        .lock()
        .unwrap()
        .expect("the loader should be consulted for the installed id");
    assert_ne!(
        load_thread, runtime_thread,
        "synchronous model loading blocked the request runtime worker"
    );
}

struct BlockingLoader {
    gate: Arc<(Mutex<bool>, Condvar)>,
    started: Mutex<Option<mpsc::SyncSender<()>>>,
    loads: AtomicUsize,
    active: AtomicUsize,
    max_active: AtomicUsize,
}

struct ActiveLoad<'a>(&'a AtomicUsize);

impl Drop for ActiveLoad<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

impl ModelLoader for BlockingLoader {
    fn load(&self, _id: &str) -> LoadOutcome {
        self.loads.fetch_add(1, Ordering::SeqCst);
        let active = self.active.fetch_add(1, Ordering::SeqCst) + 1;
        self.max_active.fetch_max(active, Ordering::SeqCst);
        let _active = ActiveLoad(&self.active);
        if let Some(started) = self.started.lock().unwrap().take() {
            let _ = started.send(());
        }

        let (released, wake) = &*self.gate;
        let mut released = released.lock().unwrap();
        while !*released {
            released = wake.wait(released).unwrap();
        }

        LoadOutcome::Refused(LoadRefusal::OpenFailed {
            detail: "controlled test refusal".to_string(),
        })
    }

    fn candidates(&self, _requested: &str) -> Vec<ModelSuggestion> {
        Vec::new()
    }
}

struct ReleaseGate(Arc<(Mutex<bool>, Condvar)>);

impl Drop for ReleaseGate {
    fn drop(&mut self) {
        let (released, wake) = &*self.0;
        *released.lock().unwrap() = true;
        wake.notify_all();
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn saturated_resolution_refuses_every_inference_route_without_blocking_other_routes() {
    let gate = Arc::new((Mutex::new(false), Condvar::new()));
    let _release_gate = ReleaseGate(Arc::clone(&gate));
    let (started_tx, started_rx) = mpsc::sync_channel(1);
    let loader = Arc::new(BlockingLoader {
        gate,
        started: Mutex::new(Some(started_tx)),
        loads: AtomicUsize::new(0),
        active: AtomicUsize::new(0),
        max_active: AtomicUsize::new(0),
    });
    let base: Arc<dyn turbospark_server::registry::ModelRegistry> =
        Arc::new(StaticRegistry::new(Vec::new()).expect("empty registry is valid"));
    let registry: Arc<dyn turbospark_server::registry::ModelRegistry> =
        Arc::new(ModelLoaderRegistry::new(base, loader.clone()));
    let state = turbospark_server::ServerState::new(registry)
        .with_max_concurrent_resolutions(NonZeroUsize::new(1).unwrap());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base_url = format!("http://{}", listener.local_addr().unwrap());
    tokio::spawn(async move {
        axum::serve(listener, build_router(state)).await.unwrap();
    });

    let client = reqwest::Client::new();
    let first_url = base_url.clone();
    let first_client = client.clone();
    let first = tokio::spawn(async move {
        first_client
            .post(format!("{first_url}/v1/chat/completions"))
            .json(&serde_json::json!({
                "model": "installed.gturbo",
                "max_tokens": 1,
                "messages": [{"role": "user", "content": "hi"}]
            }))
            .send()
            .await
            .expect("the blocked request should eventually return")
    });
    tokio::task::spawn_blocking(move || started_rx.recv_timeout(Duration::from_secs(5)))
        .await
        .expect("loader start waiter should not panic")
        .expect("installed resolution should reach the fake loader");

    let requests = [
        (
            "/v1/chat/completions",
            serde_json::json!({"model":"installed.gturbo","messages":[{"role":"user","content":"hi"}]}),
        ),
        (
            "/v1/completions",
            serde_json::json!({"model":"installed.gturbo","prompt":"hi"}),
        ),
        (
            "/v1/responses",
            serde_json::json!({"model":"installed.gturbo","input":"hi"}),
        ),
        (
            "/v1/messages",
            serde_json::json!({"model":"installed.gturbo","max_tokens":1,"messages":[{"role":"user","content":"hi"}]}),
        ),
        (
            "/v1/messages/count_tokens",
            serde_json::json!({"model":"installed.gturbo","messages":[{"role":"user","content":"hi"}]}),
        ),
        (
            "/api/chat",
            serde_json::json!({"model":"installed.gturbo","stream":false,"messages":[{"role":"user","content":"hi"}]}),
        ),
        (
            "/api/generate",
            serde_json::json!({"model":"installed.gturbo","stream":false,"prompt":"hi"}),
        ),
        (
            "/v1/embeddings",
            serde_json::json!({"model":"installed.gturbo","input":"hi"}),
        ),
        (
            "/api/embeddings",
            serde_json::json!({"model":"installed.gturbo","prompt":"hi"}),
        ),
        (
            "/api/embed",
            serde_json::json!({"model":"installed.gturbo","input":"hi"}),
        ),
    ];
    for (path, body) in requests {
        let response = tokio::time::timeout(
            Duration::from_secs(2),
            client.post(format!("{base_url}{path}")).json(&body).send(),
        )
        .await
        .unwrap_or_else(|_| panic!("{path} waited for a saturated resolution slot"))
        .unwrap_or_else(|error| panic!("{path} request failed: {error}"));
        assert_eq!(response.status().as_u16(), 503, "{path}");
        let body: serde_json::Value = response.json().await.unwrap();
        assert_eq!(body["error"]["code"], "model_resolution_busy", "{path}");
        assert_eq!(body["error"]["retryable"], true, "{path}");
    }

    let health = tokio::time::timeout(
        Duration::from_secs(2),
        client.get(format!("{base_url}/health")).send(),
    )
    .await
    .expect("health should not wait for model resolution")
    .unwrap();
    assert_eq!(health.status().as_u16(), 200);
    let models = tokio::time::timeout(
        Duration::from_secs(2),
        client.get(format!("{base_url}/v1/models")).send(),
    )
    .await
    .expect("model listing should not wait for model resolution")
    .unwrap();
    assert_eq!(models.status().as_u16(), 200);
    assert_eq!(loader.loads.load(Ordering::SeqCst), 1);
    assert_eq!(loader.active.load(Ordering::SeqCst), 1);
    assert_eq!(loader.max_active.load(Ordering::SeqCst), 1);

    drop(_release_gate);
    let first_response = tokio::time::timeout(Duration::from_secs(5), first)
        .await
        .expect("the first resolution should finish after release")
        .expect("the first request task should not panic");
    assert_eq!(first_response.status().as_u16(), 503);
    let first_body: serde_json::Value = first_response.json().await.unwrap();
    assert_eq!(first_body["error"]["code"], "model_load_refused");
    assert_eq!(loader.loads.load(Ordering::SeqCst), 1);
    assert_eq!(loader.active.load(Ordering::SeqCst), 0);
    assert_eq!(loader.max_active.load(Ordering::SeqCst), 1);
}
