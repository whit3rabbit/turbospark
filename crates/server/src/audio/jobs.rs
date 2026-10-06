//! Async audio jobs: `async=true` on a transcription or generation returns
//! 202 and a job id, and the result is fetched from `/v1/audio/jobs/{id}`.
//!
//! The store is in memory and bounded three ways (active jobs, finished-job
//! age, finished-result bytes). It dies with the process; clients that need
//! durability must fetch results promptly.
//!
//! Cancelling aborts the job's task. If the work has not reached a model
//! worker yet, dropping the future cancels it. A job the worker is already
//! running cannot be interrupted (the runners expose no cancel hook) and runs
//! to completion with its result discarded.

use std::collections::{HashMap, VecDeque};
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use axum::extract::{Path, State};
use axum::http::{header, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;

use super::{error, unix_now};
use crate::ServerState;

/// Jobs that may be unfinished at once. Past this, submission is a 429.
const MAX_ACTIVE_JOBS: usize = 8;
/// How long a finished job stays fetchable.
const FINISHED_TTL: Duration = Duration::from_secs(60 * 60);
/// Total bytes of finished results kept; the oldest are evicted first.
const MAX_RESULT_BYTES: usize = 256 * 1024 * 1024;

pub(crate) struct JobOutput {
    pub(crate) content_type: &'static str,
    pub(crate) body: Vec<u8>,
}

pub(crate) struct JobFailure {
    pub(crate) status: StatusCode,
    pub(crate) message: String,
}

pub(crate) type JobFuture = Pin<Box<dyn Future<Output = Result<JobOutput, JobFailure>> + Send>>;

pub(crate) enum SubmitError {
    Full,
}

enum State_ {
    Running,
    Succeeded(Arc<JobOutput>),
    Failed(StatusCode, String),
    Cancelled,
}

impl State_ {
    fn name(&self) -> &'static str {
        match self {
            State_::Running => "running",
            State_::Succeeded(_) => "succeeded",
            State_::Failed(..) => "failed",
            State_::Cancelled => "cancelled",
        }
    }
}

struct Job {
    kind: &'static str,
    model: String,
    created: u64,
    state: State_,
    finished_at: Option<Instant>,
    abort: Option<tokio::task::AbortHandle>,
}

#[derive(Default)]
struct Inner {
    jobs: HashMap<String, Job>,
    /// Submission order, for evicting the oldest finished job.
    order: VecDeque<String>,
}

#[derive(Default)]
pub(crate) struct JobStore {
    inner: Mutex<Inner>,
    counter: AtomicU64,
}

fn splitmix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

impl JobStore {
    fn lock(&self) -> std::sync::MutexGuard<'_, Inner> {
        self.inner.lock().unwrap_or_else(|p| p.into_inner())
    }

    fn new_id(&self) -> String {
        let n = self.counter.fetch_add(1, Ordering::Relaxed);
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);
        format!("job_{:016x}", splitmix(nanos ^ splitmix(n)))
    }

    /// Drops expired results and, if over the byte budget, the oldest ones.
    fn gc(inner: &mut Inner) {
        let now = Instant::now();
        let expired: Vec<String> = inner
            .jobs
            .iter()
            .filter(|(_, j)| {
                j.finished_at
                    .is_some_and(|t| now.duration_since(t) > FINISHED_TTL)
            })
            .map(|(id, _)| id.clone())
            .collect();
        for id in expired {
            inner.jobs.remove(&id);
        }
        let bytes = |inner: &Inner| -> usize {
            inner
                .jobs
                .values()
                .map(|j| match &j.state {
                    State_::Succeeded(o) => o.body.len(),
                    _ => 0,
                })
                .sum()
        };
        while bytes(inner) > MAX_RESULT_BYTES {
            let oldest = inner
                .order
                .iter()
                .find(|id| {
                    inner
                        .jobs
                        .get(*id)
                        .is_some_and(|j| matches!(j.state, State_::Succeeded(_)))
                })
                .cloned();
            match oldest {
                Some(id) => {
                    inner.jobs.remove(&id);
                }
                None => break,
            }
        }
        let live = &inner.jobs;
        inner.order.retain(|id| live.contains_key(id));
    }

    pub(crate) fn submit(
        self: &Arc<Self>,
        kind: &'static str,
        model: String,
        work: JobFuture,
    ) -> Result<String, SubmitError> {
        let id = self.new_id();
        {
            let mut inner = self.lock();
            Self::gc(&mut inner);
            let active = inner
                .jobs
                .values()
                .filter(|j| matches!(j.state, State_::Running))
                .count();
            if active >= MAX_ACTIVE_JOBS {
                return Err(SubmitError::Full);
            }
            inner.jobs.insert(
                id.clone(),
                Job {
                    kind,
                    model,
                    created: unix_now(),
                    state: State_::Running,
                    finished_at: None,
                    abort: None,
                },
            );
            inner.order.push_back(id.clone());
        }
        let store = Arc::clone(self);
        let task_id = id.clone();
        let handle = tokio::spawn(async move {
            let result = work.await;
            store.finish(&task_id, result);
        });
        if let Some(job) = self.lock().jobs.get_mut(&id) {
            if matches!(job.state, State_::Running) {
                job.abort = Some(handle.abort_handle());
            }
        }
        Ok(id)
    }

    fn finish(&self, id: &str, result: Result<JobOutput, JobFailure>) {
        let mut inner = self.lock();
        let Some(job) = inner.jobs.get_mut(id) else {
            return;
        };
        // A cancelled job keeps its state even if the worker finished anyway.
        if !matches!(job.state, State_::Running) {
            return;
        }
        job.state = match result {
            Ok(out) => State_::Succeeded(Arc::new(out)),
            Err(f) => State_::Failed(f.status, f.message),
        };
        job.finished_at = Some(Instant::now());
        job.abort = None;
        Self::gc(&mut inner);
    }

    /// Per-status job counts for the metrics endpoint.
    pub(crate) fn counts(&self) -> Vec<(&'static str, usize)> {
        let inner = self.lock();
        ["running", "succeeded", "failed", "cancelled"]
            .into_iter()
            .map(|name| {
                (
                    name,
                    inner
                        .jobs
                        .values()
                        .filter(|j| j.state.name() == name)
                        .count(),
                )
            })
            .collect()
    }

    /// Number of jobs not yet finished for `model`; the unload guard.
    pub(crate) fn running_for_model(&self, model: &str) -> usize {
        self.lock()
            .jobs
            .values()
            .filter(|j| j.model == model && matches!(j.state, State_::Running))
            .count()
    }
}

fn describe(id: &str, job: &Job) -> serde_json::Value {
    let mut v = serde_json::json!({
        "id": id,
        "object": "audio.job",
        "kind": job.kind,
        "model": job.model,
        "status": job.state.name(),
        "created": job.created,
    });
    match &job.state {
        State_::Succeeded(_) => {
            v["result_url"] = format!("/v1/audio/jobs/{id}/result").into();
        }
        State_::Failed(status, message) => {
            v["error"] = serde_json::json!({"status": status.as_u16(), "message": message});
        }
        _ => {}
    }
    v
}

pub(crate) async fn get_job(State(state): State<ServerState>, Path(id): Path<String>) -> Response {
    let Some(audio) = state.audio else {
        return super::unavailable();
    };
    let mut inner = audio.jobs.lock();
    JobStore::gc(&mut inner);
    match inner.jobs.get(&id) {
        Some(job) => Json(describe(&id, job)).into_response(),
        None => error(StatusCode::NOT_FOUND, "no such job", Some("id")),
    }
}

pub(crate) async fn get_job_result(
    State(state): State<ServerState>,
    Path(id): Path<String>,
) -> Response {
    let Some(audio) = state.audio else {
        return super::unavailable();
    };
    let inner = audio.jobs.lock();
    let Some(job) = inner.jobs.get(&id) else {
        return error(StatusCode::NOT_FOUND, "no such job", Some("id"));
    };
    match &job.state {
        State_::Succeeded(out) => {
            let mut response = (StatusCode::OK, out.body.clone()).into_response();
            response.headers_mut().insert(
                header::CONTENT_TYPE,
                HeaderValue::from_static(out.content_type),
            );
            response
        }
        // The job's own failure, with the status the synchronous route would
        // have given, so a client can reuse its error handling.
        State_::Failed(status, message) => error(*status, message.clone(), None),
        State_::Cancelled => error(StatusCode::GONE, "job was cancelled", Some("id")),
        State_::Running => error(
            StatusCode::CONFLICT,
            "job is still running; poll /v1/audio/jobs/{id}",
            Some("id"),
        ),
    }
}

/// Cancels a running job, or removes a finished one.
pub(crate) async fn delete_job(
    State(state): State<ServerState>,
    Path(id): Path<String>,
) -> Response {
    let Some(audio) = state.audio else {
        return super::unavailable();
    };
    let mut inner = audio.jobs.lock();
    let Some(job) = inner.jobs.get_mut(&id) else {
        return error(StatusCode::NOT_FOUND, "no such job", Some("id"));
    };
    if matches!(job.state, State_::Running) {
        if let Some(abort) = job.abort.take() {
            abort.abort();
        }
        job.state = State_::Cancelled;
        job.finished_at = Some(Instant::now());
        return Json(serde_json::json!({"id": id, "status": "cancelled"})).into_response();
    }
    inner.jobs.remove(&id);
    inner.order.retain(|x| x != &id);
    Json(serde_json::json!({"id": id, "deleted": true})).into_response()
}
