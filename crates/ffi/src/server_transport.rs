//! Body-byte counters and optional bounded previews, outside the model path.
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use axum::{
    body::Body,
    extract::{Request, State},
    middleware::Next,
    response::Response,
};
use http_body_util::BodyExt;

#[derive(Clone, Debug, Default, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TrafficSnapshot {
    pub received_bytes: u64,
    pub sent_bytes: u64,
    pub capture_text: bool,
    pub previews: Vec<String>,
}

pub(crate) struct Traffic {
    received: AtomicU64,
    sent: AtomicU64,
    next_id: AtomicU64,
    capture: bool,
    previews: Mutex<VecDeque<String>>,
}

impl Traffic {
    pub fn new(capture: bool) -> Self {
        Self {
            received: AtomicU64::new(0),
            sent: AtomicU64::new(0),
            next_id: AtomicU64::new(0),
            capture,
            previews: Mutex::new(VecDeque::new()),
        }
    }

    pub fn snapshot(&self) -> TrafficSnapshot {
        TrafficSnapshot {
            received_bytes: self.received.load(Ordering::Relaxed),
            sent_bytes: self.sent.load(Ordering::Relaxed),
            capture_text: self.capture,
            previews: self
                .previews
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .iter()
                .cloned()
                .collect(),
        }
    }

    fn wrap(self: &Arc<Self>, body: Body, incoming: bool, id: u64) -> Body {
        let traffic = Arc::clone(self);
        // Only a bounded prefix per body is eligible for preview. Counters
        // continue for the whole stream, including after preview truncation.
        let mut remaining = 4096usize;
        Body::new(body.map_frame(move |frame| {
            if let Some(bytes) = frame.data_ref() {
                let counter = if incoming {
                    &traffic.received
                } else {
                    &traffic.sent
                };
                counter.fetch_add(bytes.len() as u64, Ordering::Relaxed);
                if traffic.capture && remaining > 0 && !bytes.is_empty() {
                    let length = bytes.len().min(remaining).min(256);
                    remaining -= bytes.len().min(remaining);
                    let direction = if incoming { "IN" } else { "OUT" };
                    let text = String::from_utf8_lossy(&bytes[..length]);
                    let suffix = if length < bytes.len() {
                        " [truncated]"
                    } else {
                        ""
                    };
                    let mut previews = traffic.previews.lock().unwrap_or_else(|e| e.into_inner());
                    if previews.len() == 64 {
                        previews.pop_front();
                    }
                    previews.push_back(format!("{direction} #{id}: {text}{suffix}"));
                }
            }
            // Preserve frames (including trailers), errors and backpressure.
            frame
        }))
    }
}

pub(crate) async fn observe(
    State(traffic): State<Arc<Traffic>>,
    request: Request,
    next: Next,
) -> Response {
    let id = traffic.next_id.fetch_add(1, Ordering::Relaxed);
    let (parts, body) = request.into_parts();
    let request = Request::from_parts(parts, traffic.wrap(body, true, id));
    let (parts, body) = next.run(request).await.into_parts();
    Response::from_parts(parts, traffic.wrap(body, false, id))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn counts_consumed_bytes_without_changing_bodies_or_capturing_by_default() {
        let traffic = Arc::new(Traffic::new(false));
        let body = traffic.wrap(Body::from("request"), true, 0);
        assert_eq!(traffic.snapshot().received_bytes, 0);
        assert_eq!(axum::body::to_bytes(body, 100).await.unwrap(), "request");
        let body = traffic.wrap(Body::from("response"), false, 0);
        assert_eq!(axum::body::to_bytes(body, 100).await.unwrap(), "response");
        let snapshot = traffic.snapshot();
        assert_eq!(snapshot.received_bytes, 7);
        assert_eq!(snapshot.sent_bytes, 8);
        assert!(snapshot.previews.is_empty());
    }

    #[tokio::test]
    async fn previews_are_bounded_while_full_body_bytes_are_counted() {
        let traffic = Arc::new(Traffic::new(true));
        for id in 0..70 {
            let body = traffic.wrap(Body::from("x".repeat(500)), false, id);
            assert_eq!(axum::body::to_bytes(body, 1000).await.unwrap().len(), 500);
        }
        let snapshot = traffic.snapshot();
        assert_eq!(snapshot.sent_bytes, 35000);
        assert_eq!(snapshot.previews.len(), 64);
        assert!(snapshot.previews[0].starts_with("OUT #6: "));
        assert!(snapshot.previews[0].ends_with(" [truncated]"));
        assert!(snapshot.previews.iter().all(|s| s.len() < 300));
    }
}
