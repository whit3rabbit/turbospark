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

/// `x-api-key` first (`x-api-key` is checked BEFORE `Authorization: Bearer`
/// -- crate Gotcha 26), and empty is treated as ABSENT rather than as a
/// value to compare against the real key: a stray empty `x-api-key` header
/// (a proxy or client library default) used to shadow a valid `Authorization:
/// Bearer` sent alongside it, since `Some("")` short-circuited the `or_else`
/// before that header was ever read.
///
/// The `Bearer` scheme is matched CASE-INSENSITIVELY, per RFC 7235 (the
/// scheme token is case-insensitive; only the credentials that follow are
/// not) -- a client sending `bearer` or `BEARER` used to be refused outright
/// by a literal `"Bearer "` prefix match.
fn presented_key<B>(request: &Request<B>) -> Option<&str> {
    let headers = request.headers();
    headers
        .get("x-api-key")
        .and_then(|v| v.to_str().ok())
        .filter(|v| !v.is_empty())
        .or_else(|| {
            headers
                .get(axum::http::header::AUTHORIZATION)
                .and_then(|v| v.to_str().ok())
                .and_then(|v| {
                    let (scheme, token) = v.split_once(' ')?;
                    scheme.eq_ignore_ascii_case("bearer").then_some(token)
                })
                .filter(|v| !v.is_empty())
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

    /// F13: an empty `x-api-key` (a proxy or client library default) must
    /// not shadow a valid `Authorization: Bearer` sent alongside it.
    #[test]
    fn an_empty_x_api_key_falls_through_to_bearer() {
        let request = Request::builder()
            .header("x-api-key", "")
            .header("authorization", "Bearer real-key")
            .body(())
            .unwrap();
        assert_eq!(presented_key(&request), Some("real-key"));
    }

    /// F13: RFC 7235 makes the scheme token case-insensitive.
    #[test]
    fn the_bearer_scheme_is_case_insensitive() {
        for scheme in ["Bearer", "bearer", "BEARER", "BeArEr"] {
            let request = Request::builder()
                .header("authorization", format!("{scheme} real-key"))
                .body(())
                .unwrap();
            assert_eq!(presented_key(&request), Some("real-key"), "{scheme}");
        }
    }
}
