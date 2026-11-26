//! `GET /v1/metrics`: Prometheus text. The text-generation series live here;
//! a server with an audio provider appends the audio series to the same body.
//!
//! No series carries a label, so nothing a client sent (and no model path or
//! key) can reach the exposition.

use std::fmt::Write as _;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, Weak};

use axum::extract::State;
use axum::http::{header, HeaderValue};
use axum::response::{IntoResponse, Response};
use runtime::RawDecodeResult;

use crate::queue::GenerationQueue;
use crate::ServerState;

#[derive(Default)]
struct Totals {
    requests: u64,
    failed: u64,
    prompt_tokens: u64,
    predicted_tokens: u64,
    prefill_seconds: f64,
    decode_seconds: f64,
}

#[derive(Default)]
pub(crate) struct TextMetrics {
    totals: Mutex<Totals>,
    rejected: AtomicU64,
    /// Every runner gate a request has resolved to. A gate nothing resolved
    /// to cannot have waiters, so the waiting gauge misses none.
    queues: Mutex<Vec<Weak<GenerationQueue>>>,
}

impl TextMetrics {
    pub(crate) fn watch(&self, queue: &Arc<GenerationQueue>) {
        let mut queues = self.queues.lock().unwrap_or_else(|p| p.into_inner());
        queues.retain(|q| q.strong_count() > 0);
        if !queues.iter().any(|q| q.as_ptr() == Arc::as_ptr(queue)) {
            queues.push(Arc::downgrade(queue));
        }
    }

    pub(crate) fn started(&self) {
        self.totals
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .requests += 1;
    }

    /// `None` is a generation that returned an error.
    pub(crate) fn finished(&self, decode: Option<&RawDecodeResult>) {
        let mut totals = self.totals.lock().unwrap_or_else(|p| p.into_inner());
        let Some(decode) = decode else {
            totals.failed += 1;
            return;
        };
        totals.prompt_tokens += crate::timings::prefilled_tokens(decode);
        totals.predicted_tokens += decode.new_tokens as u64;
        totals.prefill_seconds += decode.prefill_seconds;
        totals.decode_seconds += decode.decode_seconds;
    }

    pub(crate) fn rejected(&self) {
        self.rejected.fetch_add(1, Ordering::Relaxed);
    }

    fn render(&self) -> String {
        let waiting: usize = self
            .queues
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .iter()
            .filter_map(Weak::upgrade)
            .map(|q| q.queued())
            .sum();
        let t = self.totals.lock().unwrap_or_else(|p| p.into_inner());
        let mut out = String::new();
        let mut series = |name: &str, kind: &str, help: &str, value: String| {
            let name = format!("turbospark_text_{name}");
            let _ = writeln!(
                out,
                "# HELP {name} {help}\n# TYPE {name} {kind}\n{name} {value}"
            );
        };
        series(
            "requests_total",
            "counter",
            "Generations started on a model; a guardrail retry counts again.",
            t.requests.to_string(),
        );
        series(
            "requests_failed_total",
            "counter",
            "Generations that ended in a runtime error.",
            t.failed.to_string(),
        );
        series(
            "prompt_tokens_total",
            "counter",
            "Prompt tokens prefilled, reused prefix excluded.",
            t.prompt_tokens.to_string(),
        );
        series(
            "predicted_tokens_total",
            "counter",
            "Tokens generated.",
            t.predicted_tokens.to_string(),
        );
        series(
            "prefill_seconds_total",
            "counter",
            "Time spent in prefill by completed generations.",
            format!("{:.6}", t.prefill_seconds),
        );
        series(
            "decode_seconds_total",
            "counter",
            "Time spent in decode by completed generations.",
            format!("{:.6}", t.decode_seconds),
        );
        series(
            "queue_waiting",
            "gauge",
            "Requests waiting for the generation queue.",
            waiting.to_string(),
        );
        series(
            "queue_rejected_total",
            "counter",
            "Requests refused with 429 because the generation queue was full.",
            self.rejected.load(Ordering::Relaxed).to_string(),
        );
        out
    }
}

pub(crate) async fn metrics_endpoint(State(state): State<ServerState>) -> Response {
    let mut body = state.text_metrics.render();
    if let Some(audio) = &state.audio {
        body.push_str(&crate::audio::render_metrics(audio));
    }
    let mut response = body.into_response();
    response.headers_mut().insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("text/plain; version=0.0.4; charset=utf-8"),
    );
    response
}
