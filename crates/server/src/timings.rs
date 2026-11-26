//! llama.cpp's `timings` object and OpenAI's `cached_tokens`, so a client can
//! separate the server's own speed from the latency it measured. Both come
//! straight off [`RawDecodeResult`], never off a count of content events.

use runtime::RawDecodeResult;
use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub(crate) struct Timings {
    /// Tokens actually prefilled: the reused prefix is not in it.
    pub prompt_n: u64,
    pub prompt_ms: f64,
    /// Null when no time was measured, rather than a made-up rate.
    pub prompt_per_second: Option<f64>,
    pub predicted_n: u64,
    pub predicted_ms: f64,
    pub predicted_per_second: Option<f64>,
}

pub(crate) fn prefilled_tokens(decode: &RawDecodeResult) -> u64 {
    decode
        .prompt_tokens
        .saturating_sub(decode.reused_prefix_tokens) as u64
}

impl From<&RawDecodeResult> for Timings {
    fn from(decode: &RawDecodeResult) -> Self {
        let rate = |n: u64, seconds: f64| (seconds > 0.0).then(|| n as f64 / seconds);
        let prompt_n = prefilled_tokens(decode);
        let predicted_n = decode.new_tokens as u64;
        Self {
            prompt_n,
            prompt_ms: decode.prefill_seconds * 1000.0,
            prompt_per_second: rate(prompt_n, decode.prefill_seconds),
            predicted_n,
            predicted_ms: decode.decode_seconds * 1000.0,
            predicted_per_second: rate(predicted_n, decode.decode_seconds),
        }
    }
}

/// The value of `usage.prompt_tokens_details`.
pub(crate) fn prompt_tokens_details(decode: &RawDecodeResult) -> serde_json::Value {
    serde_json::json!({"cached_tokens": decode.reused_prefix_tokens})
}

/// A wire object with `timings` appended after its own fields. Flattened, so
/// the wrapped type's field order is untouched.
#[derive(Serialize)]
pub(crate) struct WithTimings<T> {
    #[serde(flatten)]
    pub body: T,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timings: Option<Timings>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn decode(prefill_seconds: f64, decode_seconds: f64) -> RawDecodeResult {
        RawDecodeResult {
            prompt_tokens: 100,
            reused_prefix_tokens: 60,
            new_tokens: 20,
            prefill_seconds,
            decode_seconds,
            ..crate::handler::cancelled_before_start()
        }
    }

    #[test]
    fn prompt_n_excludes_the_reused_prefix_and_rates_follow() {
        let d = decode(0.5, 2.0);
        let t = Timings::from(&d);
        assert_eq!((t.prompt_n, t.predicted_n), (40, 20));
        assert_eq!((t.prompt_ms, t.predicted_ms), (500.0, 2000.0));
        assert_eq!(t.prompt_per_second, Some(80.0));
        assert_eq!(t.predicted_per_second, Some(10.0));
        assert_eq!(prompt_tokens_details(&d)["cached_tokens"], 60);
    }

    #[test]
    fn an_unmeasured_duration_has_no_rate() {
        let t = Timings::from(&decode(0.0, 0.0));
        assert_eq!((t.prompt_per_second, t.predicted_per_second), (None, None));
    }

    #[test]
    fn timings_are_appended_after_the_wrapped_fields() {
        let chunk =
            crate::response::completion_chunk("i".into(), 1, "m".into(), Default::default(), None);
        let plain = serde_json::to_string(&chunk).unwrap();
        let wrapped = serde_json::to_string(&WithTimings {
            body: chunk,
            timings: Some(Timings::from(&decode(0.5, 2.0))),
        })
        .unwrap();
        let prefix = format!("{},\"timings\":", &plain[..plain.len() - 1]);
        assert!(wrapped.starts_with(&prefix), "{wrapped}");
    }
}
