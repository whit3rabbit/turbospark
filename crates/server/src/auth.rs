//! Bearer/`x-api-key` authentication for a `--api-key` server.
//!
//! Opt-in: [`crate::build_router`] never applies this layer (it delegates to
//! [`crate::build_router_with_options`] with `api_key: None`), so every
//! existing test and caller keeps the exact unauthenticated behavior this
//! crate had before the flag existed. `GET /health` is routed OUTSIDE the
//! layer entirely (see `lib.rs`): an auth check answers "who are you", and a
//! liveness probe answering "no" to an unauthenticated caller would be
//! indistinguishable from the process being down, which is the one thing a
//! liveness probe must never be wrong about.
//!
//! Accepts EITHER spelling: `x-api-key: <key>` (what Anthropic-native
//! clients, including Claude Code, send when pointed at this server via
//! `ANTHROPIC_API_KEY`/`ANTHROPIC_BASE_URL`) or `Authorization: Bearer
//! <key>` (the generic/OpenAI convention). `x-api-key` is checked first
//! because it is the one this server's own documented Claude Code walkthrough
//! (`crates/server/CLAUDE.md`) actually exercises.

use std::sync::Arc;

use axum::extract::State;
use axum::http::{Request, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Json, Response};

/// The configured key, cheap to clone into the middleware's own state.
#[derive(Clone)]
pub(crate) struct ApiKey(pub(crate) Arc<str>);

/// A local constant-time byte comparison rather than the `subtle` crate: one
/// short function, no new dependency, and `subtle`'s API (`ConstantTimeEq`
/// over a `Choice`, meant for comparing against a table of many secrets) is
/// built for a bigger problem than comparing a presented header against the
/// ONE key this process holds for its whole lifetime. Still branchless on
/// the byte comparison itself, which is the property that matters: a
/// short-circuiting `==` would let a timing side channel narrow the key
/// character by character.
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff: u8 = 0;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

fn unauthorized() -> Response {
    (
        StatusCode::UNAUTHORIZED,
        Json(serde_json::json!({
            "error": {
                "message": "invalid or missing API key",
                "type": "authentication_error",
            }
        })),
    )
        .into_response()
}

fn presented_key<B>(request: &Request<B>) -> Option<&str> {
    let headers = request.headers();
    headers
        .get("x-api-key")
        .and_then(|v| v.to_str().ok())
        .or_else(|| {
            headers
                .get(axum::http::header::AUTHORIZATION)
                .and_then(|v| v.to_str().ok())
                .and_then(|v| v.strip_prefix("Bearer "))
        })
}

pub(crate) async fn require_api_key(
    State(expected): State<ApiKey>,
    request: Request<axum::body::Body>,
    next: Next,
) -> Response {
    match presented_key(&request) {
        Some(key) if constant_time_eq(key.as_bytes(), expected.0.as_bytes()) => {
            next.run(request).await
        }
        _ => unauthorized(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn equal_keys_compare_equal() {
        assert!(constant_time_eq(b"sk-secret", b"sk-secret"));
    }

    #[test]
    fn different_length_keys_are_not_equal() {
        assert!(!constant_time_eq(b"short", b"much-longer-key"));
    }

    #[test]
    fn same_length_different_content_is_not_equal() {
        assert!(!constant_time_eq(b"aaaaaaaa", b"aaaaaaab"));
    }

    #[test]
    fn empty_compares_equal_only_to_empty() {
        assert!(constant_time_eq(b"", b""));
        assert!(!constant_time_eq(b"", b"x"));
    }

    #[test]
    fn x_api_key_is_read_first() {
        let request = Request::builder()
            .header("x-api-key", "from-x-api-key")
            .header("authorization", "Bearer from-bearer")
            .body(())
            .unwrap();
        assert_eq!(presented_key(&request), Some("from-x-api-key"));
    }

    #[test]
    fn bearer_is_read_when_x_api_key_is_absent() {
        let request = Request::builder()
            .header("authorization", "Bearer from-bearer")
            .body(())
            .unwrap();
        assert_eq!(presented_key(&request), Some("from-bearer"));
    }

    #[test]
    fn a_bare_authorization_header_with_no_bearer_prefix_is_not_a_key() {
        let request = Request::builder()
            .header("authorization", "from-bearer")
            .body(())
            .unwrap();
        assert_eq!(presented_key(&request), None);
    }

    #[test]
    fn no_relevant_header_at_all_presents_nothing() {
        let request = Request::builder().body(()).unwrap();
        assert_eq!(presented_key(&request), None);
    }
}
