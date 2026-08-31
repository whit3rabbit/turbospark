//! The one place this crate reports what it is doing.
//!
//! Before this module there was no logging, no metrics and no request
//! counter anywhere in the server: an embedding host could tell that a
//! request had been served only by watching its own client. Everything a
//! console or a graph shows comes from here.
//!
//! **THE FACTS COME FROM TWO PLACES BECAUSE NEITHER CAN SEE THE OTHER'S.**
//! An axum middleware layer knows the method, the path, the status and the
//! wall duration, and it knows them for EVERY request including the ones no
//! handler ever ran -- a 404 on an unknown route, a 401 from `auth.rs`, a
//! rejected body. It cannot know how many tokens were prefilled or when the
//! first one came out. The generation paths know exactly that and nothing
//! about the HTTP envelope around them. So a request produces a
//! [`ServerEvent::RequestStarted`] and a [`ServerEvent::RequestFinished`]
//! from the layer, and zero or more generation events in between, tied
//! together by an id the layer mints and puts in the request extensions.
//!
//! **TOKEN COUNTS COME OFF `RawDecodeResult`, NEVER OFF A COUNT OF CONTENT
//! EVENTS.** `swift/CLAUDE.md` Gotcha 7 records what happens when a caller
//! counts callbacks instead: special tokens decode to the empty string, the
//! streaming detokenizer withholds partial UTF-8, and reasoning goes to a
//! different channel entirely, so the count is low by an amount that varies
//! with the dialect and the turn. That number is fine for a liveness
//! indicator and is not a throughput measurement. Nothing here may carry it.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// A request id, unique within one server's lifetime. Minted by
/// [`RequestIds::next`] in the middleware and echoed by every event
/// belonging to that request.
pub type RequestId = u64;

/// What the middleware puts in the request extensions so a handler can tie
/// its own events to the HTTP-level ones. A newtype rather than a bare
/// `u64`, because extensions are keyed by TYPE and a bare integer would
/// collide with any other layer that had the same idea.
#[derive(Clone, Copy, Debug)]
pub struct RequestTag(pub RequestId);

/// Monotonic id source, one per running server.
#[derive(Default)]
pub struct RequestIds(AtomicU64);

impl RequestIds {
    pub fn next(&self) -> RequestId {
        // Relaxed is right: the only requirement is uniqueness, and nothing
        // downstream orders anything against the counter itself.
        self.0.fetch_add(1, Ordering::Relaxed)
    }
}

/// What happened. Serialized straight through to a host, so the field names
/// are camelCase and the variants are externally tagged under `"kind"`
/// (`crates/ffi/CLAUDE.md` Gotcha 3: a Swift `Codable` should need no
/// `CodingKeys`).
#[derive(Clone, Debug, serde::Serialize)]
#[serde(tag = "kind", rename_all = "camelCase")]
pub enum ServerEvent {
    /// From the middleware, before any handler runs. Carries only what an
    /// unparsed request can actually tell you -- the `model` field and the
    /// `stream` flag are in a body this layer must not consume, and are
    /// reported by [`ServerEvent::RequestRouted`] instead.
    #[serde(rename_all = "camelCase")]
    RequestStarted {
        id: RequestId,
        at_ms: u64,
        method: String,
        path: String,
    },
    /// From a handler, once it has deserialized the body and picked a
    /// backend.
    ///
    /// **`requested` AND `served` ARE SEPARATE FIELDS BECAUSE THEY DIFFER ON
    /// EVERY SINGLE-MODEL FALLBACK**, which is the common case rather than
    /// an edge one (`registry.rs`'s header). Collapsing them would make a
    /// console show a Claude Code request as having asked for the install it
    /// was routed to, hiding the one fact somebody debugging a multi-model
    /// setup is looking for.
    #[serde(rename_all = "camelCase")]
    RequestRouted {
        id: RequestId,
        requested: Option<String>,
        served: String,
        stream: bool,
    },
    /// One completed generation, carrying the counters no middleware can
    /// observe. Emitted once per `ChatModel::run_completion`, so a request
    /// the guardrails re-asked (`guardrails.rs`) produces TWO -- which is
    /// the useful reading rather than a duplicate, since the retry is real
    /// work the machine did.
    ///
    /// **THERE IS NO TIME-TO-FIRST-TOKEN FIELD, AND ITS ABSENCE IS THE
    /// HONEST ANSWER.** Nothing on this path can measure one: a caller
    /// means "request in, first token out", which includes the wait behind
    /// the runner's mutex (`crates/server/CLAUDE.md` Gotcha 1 -- one runner
    /// per process, so requests queue) and this decorator only starts
    /// counting once it already holds the runner. What IS measured is
    /// `prefill_seconds` and `decode_seconds`, straight off
    /// `RawDecodeResult`, plus `RequestFinished.duration_ms` from the
    /// middleware, which DOES include the queue. Subtracting gives the wait.
    /// A field called `ttft_ms` filled from `prefill_seconds` would read as
    /// the first number and be the second.
    #[serde(rename_all = "camelCase")]
    Generated {
        id: RequestId,
        /// The model that actually served it.
        model: String,
        prompt_tokens: u32,
        new_tokens: u32,
        prefill_seconds: f64,
        decode_seconds: f64,
        stop_reason: String,
    },
    /// From the middleware, after the handler returned. On a STREAMING
    /// response this fires when the handler returns the stream, not when
    /// the stream finishes -- axum has already sent the headers by then and
    /// there is nothing further for a layer to observe. `Generated` is the
    /// event that closes a streamed turn.
    #[serde(rename_all = "camelCase")]
    RequestFinished {
        id: RequestId,
        status: u16,
        duration_ms: u32,
    },
    /// A model joined or left a running server's registry.
    #[serde(rename_all = "camelCase")]
    ModelAttached { at_ms: u64, model: String },
    #[serde(rename_all = "camelCase")]
    ModelDetached { at_ms: u64, model: String },
}

/// Where events go. A host implements this and hands it to
/// [`crate::RouterOptions`].
///
/// `record` is called from request-handling threads, including inside a
/// blocking generation task, so an implementation must not block for long
/// and must not call back into the router. The FFI's is a bounded ring
/// behind a mutex, which is the shape to copy.
pub trait ServerObserver: Send + Sync {
    fn record(&self, event: ServerEvent);
}

/// Convenience for the `Option<Arc<dyn ServerObserver>>` every call site
/// holds: builds the event only when somebody is listening.
///
/// Building a `ServerEvent` allocates several `String`s, and with no
/// observer configured -- which is every caller that predates this module,
/// including the whole integration suite -- that work would be pure waste
/// on the hot path.
pub(crate) fn record<F>(observer: &Option<Arc<dyn ServerObserver>>, build: F)
where
    F: FnOnce() -> ServerEvent,
{
    if let Some(o) = observer {
        o.record(build());
    }
}

/// Milliseconds since the Unix epoch, for a host that has to place an event
/// on a timeline it did not observe.
pub(crate) fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// A [`crate::ChatModel`] that reports every generation it runs, wrapping
/// the one a request resolved to.
///
/// **A DECORATOR ON `run_completion` RATHER THAN A PARAMETER THREADED
/// THROUGH THE GENERATION CORE**, and the reason is that this method IS the
/// choke point already. `crates/server/CLAUDE.md` Gotcha 17 records why it
/// exists at all -- the speculative loop is not object-safe, so the trait
/// method owns which decode loop runs -- and the consequence is that every
/// generation in this crate goes through exactly one call to it: the
/// streaming and non-streaming chat paths through `stream_blocking`,
/// `/v1/completions` through its own local pair, the Anthropic and
/// Responses paths through the shared core, and both Ollama routes.
/// Threading a reporter instead would have meant a new argument on
/// `stream_blocking`, `run_full`, `run_guarded` and four handlers, for
/// facts all of them get from the same return value.
///
/// The cost is one vtable hop per REQUEST (never per token) and one `Arc`
/// clone, and it is paid only when an observer is configured -- with none,
/// `crate::handler::resolve_backend` hands back the bare model and this
/// type is never constructed.
pub(crate) struct ReportingModel {
    inner: Arc<dyn crate::ChatModel>,
    observer: Arc<dyn ServerObserver>,
    id: RequestId,
}

impl ReportingModel {
    pub(crate) fn new(
        inner: Arc<dyn crate::ChatModel>,
        observer: Arc<dyn ServerObserver>,
        id: RequestId,
    ) -> Self {
        Self {
            inner,
            observer,
            id,
        }
    }
}

impl crate::ChatModel for ReportingModel {
    fn tokenizer(&self) -> &tokenizer::MfTokenizer {
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
    fn vision(&self) -> Option<crate::vision::VisionInfo> {
        self.inner.vision()
    }
    fn rate_control(&self) -> runtime::RateControl {
        self.inner.rate_control()
    }
    fn guardrails(&self) -> crate::guardrails::GuardrailConfig {
        self.inner.guardrails()
    }
    fn default_reasoning(&self) -> tokenizer::ReasoningEffort {
        self.inner.default_reasoning()
    }
    fn with_producer(
        &self,
        f: &mut dyn FnMut(
            &mut dyn runtime::LogitProducer,
        ) -> Result<runtime::RawDecodeResult, runtime::RuntimeError>,
    ) -> Result<runtime::RawDecodeResult, runtime::RuntimeError> {
        self.inner.with_producer(f)
    }

    /// **DELEGATES RATHER THAN REIMPLEMENTING, which is what keeps the
    /// decode-loop choice where Gotcha 17 put it.** Calling
    /// `self.inner.run_completion` reaches whatever loop the CONCRETE
    /// backend picks -- speculative, chunked, image-under-one-lock, or the
    /// trait's own sequential default. A decorator that called
    /// `with_producer` here instead would silently downgrade every
    /// speculative or chunked request to the sequential loop the moment an
    /// observer was configured, with nothing failing.
    fn run_completion(
        &self,
        prompt_ids: &[foundation::TokenId],
        config: &runtime::GenerationConfig,
        images: Option<&crate::vision::RequestImages>,
        cancel: runtime::CancelFlag<'_>,
        on_progress: &mut dyn FnMut(runtime::RawDecodeProgress),
    ) -> Result<runtime::RawDecodeResult, runtime::RuntimeError> {
        let result = self
            .inner
            .run_completion(prompt_ids, config, images, cancel, on_progress);
        if let Ok(decode) = &result {
            self.observer.record(ServerEvent::Generated {
                id: self.id,
                model: self.inner.model_id().to_string(),
                prompt_tokens: decode.prompt_tokens as u32,
                new_tokens: decode.new_tokens as u32,
                prefill_seconds: decode.prefill_seconds,
                decode_seconds: decode.decode_seconds,
                stop_reason: format!("{:?}", decode.reason),
            });
        }
        // A failure is NOT recorded here. The handler turns it into a status
        // code and the middleware reports that, with the message -- so
        // emitting one from inside would double-count exactly the requests a
        // console is most likely to be counting.
        result
    }
}
