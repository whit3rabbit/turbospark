//! `Host` validation for a server with no API key.
//!
//! A keyless server only ever binds loopback (`main.rs` `resolve_api_key`
//! and the FFI `Server::start` both refuse a Tailnet bind without a key), so
//! every legitimate client names a loopback host. A web page that DNS-rebinds
//! its own name to 127.0.0.1 is same-origin to this server, needs no CORS
//! grant, and still sends its own name as `Host`. The TCP peer is genuinely
//! loopback in that case, so `Host` is the only signal that tells the two
//! apart. With a key configured the key already stops such a page, and a
//! Tailnet client legitimately names a MagicDNS host, so this layer is only
//! applied when there is no key.

use axum::http::{Request, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Json, Response};

fn forbidden() -> Response {
    (
        StatusCode::FORBIDDEN,
        Json(serde_json::json!({
            "error": {
                "message": "Host header must name a loopback address on a server without an API key",
                "type": "permission_error",
            }
        })),
    )
        .into_response()
}

/// `name[:port]` or `[v6][:port]`, where the port, when present, must be
/// digits. Exact parsing rather than a prefix match: `127.0.0.1.evil.example`
/// and `localhost:evil.example` are attacker-owned names.
pub(crate) fn is_loopback_host(value: &str) -> bool {
    let value = value.to_ascii_lowercase();
    let (name, port) = if let Some(rest) = value.strip_prefix('[') {
        let Some((inner, after)) = rest.split_once(']') else {
            return false;
        };
        match inner.parse::<std::net::Ipv6Addr>() {
            Ok(ip) if ip.is_loopback() => {}
            _ => return false,
        }
        let port = match after {
            "" => None,
            p => match p.strip_prefix(':') {
                Some(p) => Some(p),
                None => return false,
            },
        };
        return port.is_none_or(valid_port);
    } else {
        match value.split_once(':') {
            Some((name, port)) => (name.to_string(), Some(port.to_string())),
            None => (value.clone(), None),
        }
    };
    if !port.as_deref().is_none_or(valid_port) {
        return false;
    }
    // `host.docker.internal` is how a Docker Desktop container reaches a
    // service on the Mac's loopback. `.internal` is reserved and has no
    // public DNS, so a web page cannot rebind to it.
    name == "localhost"
        || name == "host.docker.internal"
        || name
            .parse::<std::net::Ipv4Addr>()
            .is_ok_and(|ip| ip.is_loopback())
}

fn valid_port(port: &str) -> bool {
    !port.is_empty() && port.len() <= 5 && port.bytes().all(|b| b.is_ascii_digit())
}

pub(crate) async fn require_loopback_host(
    request: Request<axum::body::Body>,
    next: Next,
) -> Response {
    // HTTP/2 carries the authority in the URI instead of `Host`. A request
    // with neither cannot come from a browser, which always sends one, so it
    // is not a rebinding page and is let through.
    let allowed = match request.headers().get(axum::http::header::HOST) {
        Some(value) => value.to_str().is_ok_and(is_loopback_host),
        None => request
            .uri()
            .authority()
            .is_none_or(|a| is_loopback_host(a.as_str())),
    };
    if allowed {
        next.run(request).await
    } else {
        forbidden()
    }
}

#[cfg(test)]
mod tests {
    use super::is_loopback_host;

    #[test]
    fn port_must_be_digits() {
        assert!(is_loopback_host("localhost:8080"));
        assert!(!is_loopback_host("localhost:80a"));
        assert!(!is_loopback_host("[::1]:"));
        assert!(!is_loopback_host("[::1]x"));
        assert!(!is_loopback_host("localhost:123456"));
    }
}
