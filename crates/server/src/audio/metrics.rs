//! `GET /v1/metrics`: Prometheus text for the audio routes only. Chat has no
//! equivalent, so the series carry an `audio` prefix rather than claiming to
//! describe the whole process.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::Duration;

use axum::extract::State;
use axum::http::{header, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};

use crate::ServerState;

#[derive(Default)]
struct Series {
    count: u64,
    seconds: f64,
}

#[derive(Default)]
pub(crate) struct AudioMetrics {
    // (endpoint, status code) -> count and total seconds. Both key parts are
    // bounded: endpoints are static strings, status codes are a small set.
    requests: Mutex<BTreeMap<(&'static str, u16), Series>>,
    in_flight: AtomicU64,
    busy: AtomicU64,
}

/// Decrements the in-flight gauge on drop, so a cancelled request cannot
/// leave it stuck high.
pub(crate) struct InFlight<'a>(&'a AtomicU64);

impl Drop for InFlight<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::Relaxed);
    }
}

impl AudioMetrics {
    pub(crate) fn begin(&self) -> InFlight<'_> {
        self.in_flight.fetch_add(1, Ordering::Relaxed);
        InFlight(&self.in_flight)
    }

    pub(crate) fn record(&self, endpoint: &'static str, status: u16, took: Duration) {
        if status == 429 {
            self.busy.fetch_add(1, Ordering::Relaxed);
        }
        let mut map = self.requests.lock().unwrap_or_else(|p| p.into_inner());
        let s = map.entry((endpoint, status)).or_default();
        s.count += 1;
        s.seconds += took.as_secs_f64();
    }

    fn render(&self, models_loaded: usize, jobs: &[(&'static str, usize)]) -> String {
        let mut out = String::new();
        out.push_str(
            "# HELP turbospark_audio_requests_total Audio requests by endpoint and status.\n",
        );
        out.push_str("# TYPE turbospark_audio_requests_total counter\n");
        let map = self.requests.lock().unwrap_or_else(|p| p.into_inner());
        for ((endpoint, status), s) in map.iter() {
            out.push_str(&format!(
                "turbospark_audio_requests_total{{endpoint=\"{endpoint}\",status=\"{status}\"}} {}\n",
                s.count
            ));
        }
        out.push_str(
            "# HELP turbospark_audio_request_duration_seconds_sum Total handler time by endpoint and status.\n",
        );
        out.push_str("# TYPE turbospark_audio_request_duration_seconds_sum counter\n");
        for ((endpoint, status), s) in map.iter() {
            out.push_str(&format!(
                "turbospark_audio_request_duration_seconds_sum{{endpoint=\"{endpoint}\",status=\"{status}\"}} {:.6}\n",
                s.seconds
            ));
        }
        drop(map);
        out.push_str("# HELP turbospark_audio_in_flight Audio requests currently being handled.\n");
        out.push_str("# TYPE turbospark_audio_in_flight gauge\n");
        out.push_str(&format!(
            "turbospark_audio_in_flight {}\n",
            self.in_flight.load(Ordering::Relaxed)
        ));
        out.push_str("# HELP turbospark_audio_busy_total Requests rejected with 429.\n");
        out.push_str("# TYPE turbospark_audio_busy_total counter\n");
        out.push_str(&format!(
            "turbospark_audio_busy_total {}\n",
            self.busy.load(Ordering::Relaxed)
        ));
        out.push_str("# HELP turbospark_audio_models_loaded Audio models currently attached.\n");
        out.push_str("# TYPE turbospark_audio_models_loaded gauge\n");
        out.push_str(&format!("turbospark_audio_models_loaded {models_loaded}\n"));
        out.push_str("# HELP turbospark_audio_jobs Async audio jobs by status.\n");
        out.push_str("# TYPE turbospark_audio_jobs gauge\n");
        for (status, n) in jobs {
            out.push_str(&format!(
                "turbospark_audio_jobs{{status=\"{status}\"}} {n}\n"
            ));
        }
        out
    }
}

pub(crate) async fn metrics_endpoint(State(state): State<ServerState>) -> Response {
    let Some(audio) = state.audio else {
        return super::unavailable();
    };
    let body = audio
        .metrics
        .render(audio.provider.models().len(), &audio.jobs.counts());
    let mut response = (StatusCode::OK, body).into_response();
    response.headers_mut().insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("text/plain; version=0.0.4; charset=utf-8"),
    );
    response
}
