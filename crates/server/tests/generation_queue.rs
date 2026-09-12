//! The FIFO generation gate (ROADMAP P1 item 5): admission order under
//! concurrency, and what happens to a request whose client disconnects
//! while it is still queued.
//!
//! # The fixture, and why a HOLD rather than a timing trick
//!
//! `HeldModel` blocks inside `run_completion` until the test releases it,
//! so the FIRST request observably holds the gate (its generation has
//! started, `started` says so) while later ones queue. Arrival order is
//! made DETERMINISTIC by waiting on the gate's own `queued()` counter
//! between fires, rather than by sleeping and hoping the runtime scheduled
//! in between: request 1 is not fired until request 0 has started, request
//! 2 not until request 1 is observably waiting.
//!
//! The first chunk of each SSE body is sent only after its generation's
//! permit arrives (the whole framing body runs inside `run_gated`), so the
//! order in which the three bodies COMPLETE is the order the gate granted
//! permits -- which is the whole contract under test. Completion order is
//! strictly serial here (a body ends before its permit drops, so the next
//! generation cannot have started), which is what makes the assertion
//! exact rather than timestamp-fuzzy.
//!
//! Both tests are wrapped in a `timeout` so a gate that deadlocks fails in
//! seconds instead of hanging the suite.

use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

use runtime::{LogitProducer, RawDecodeResult, RuntimeError};
use tokenizer::MfTokenizer;
use turbospark_server::{build_router, ChatModel, GenerationQueue};

fn load_tokenizer() -> MfTokenizer {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/ChatMLTokenizer");
    MfTokenizer::load_from_dir(&dir).expect("fixture tokenizer should load")
}

/// The state the fixture shares with the test: the gate itself, how many
/// generations have STARTED (incremented inside the permit-held region),
/// and the hold the first generation waits on.
#[derive(Clone)]
struct Held {
    queue: Arc<GenerationQueue>,
    started: Arc<AtomicUsize>,
    hold: Arc<(Mutex<bool>, Condvar)>,
}

impl Held {
    fn new() -> Self {
        Held {
            queue: GenerationQueue::shared(),
            started: Arc::new(AtomicUsize::new(0)),
            hold: Arc::new((Mutex::new(false), Condvar::new())),
        }
    }

    fn release(&self) {
        let mut g = self.hold.0.lock().unwrap();
        *g = true;
        self.hold.1.notify_all();
    }
}

struct HeldProducer {
    h_id: i32,
}

impl LogitProducer for HeldProducer {
    fn reset(&mut self) {}

    fn produce(
        &mut self,
        _token: i32,
        _position: usize,
        logits: &mut [foundation::LogitValue],
    ) -> Result<(), String> {
        logits.fill(foundation::LogitValue::from_f32(0.0));
        logits[self.h_id as usize] = foundation::LogitValue::from_f32(1.0);
        Ok(())
    }
}

struct HeldModel {
    tokenizer: MfTokenizer,
    h_id: i32,
    held: Held,
}

impl ChatModel for HeldModel {
    fn tokenizer(&self) -> &MfTokenizer {
        &self.tokenizer
    }
    fn vocab_size(&self) -> usize {
        self.tokenizer.vocab_size
    }
    fn max_context(&self) -> u32 {
        2_000_000
    }
    fn model_id(&self) -> &str {
        "held"
    }
    fn with_producer(
        &self,
        f: &mut dyn FnMut(&mut dyn LogitProducer) -> Result<RawDecodeResult, RuntimeError>,
    ) -> Result<RawDecodeResult, RuntimeError> {
        f(&mut HeldProducer { h_id: self.h_id })
    }
    fn generation_queue(&self) -> Option<Arc<GenerationQueue>> {
        Some(self.held.queue.clone())
    }

    /// Records the start (inside the permit-held region, which is the
    /// point), holds until released, then runs the trait's own default
    /// sequential loop through `with_producer`.
    fn run_completion(
        &self,
        prompt_ids: &[foundation::TokenId],
        config: &runtime::GenerationConfig,
        images: Option<&turbospark_server::vision::RequestImages>,
        cancel: runtime::CancelFlag<'_>,
        on_progress: &mut dyn FnMut(runtime::RawDecodeProgress),
    ) -> Result<RawDecodeResult, RuntimeError> {
        self.held.started.fetch_add(1, Ordering::SeqCst);
        // BOUNDED: a mutation that breaks the gate (or the gate deadlocking)
        // would otherwise park this blocking thread forever and the test
        // BINARY never exits after its own panic -- the failure mode becomes
        // a hung suite rather than a red test. Twenty seconds is far past
        // anything the happy path needs and far short of the suite's
        // patience.
        let deadline = std::time::Instant::now() + Duration::from_secs(20);
        let mut gate = self.held.hold.0.lock().unwrap();
        while !*gate {
            let now = std::time::Instant::now();
            if now >= deadline {
                break;
            }
            let (g, _timed_out) = self.held.hold.1.wait_timeout(gate, deadline - now).unwrap();
            gate = g;
        }
        // The trait default's own body (minus its images refusal, which no
        // request here exercises): `ChatModel::run_completion(self, ..)`
        // would dispatch straight back into THIS override, not the default.
        let _ = images;
        self.with_producer(&mut |producer| {
            runtime::run_raw_completion_cancellable(
                producer,
                self.tokenizer(),
                prompt_ids,
                config,
                self.max_context(),
                self.vocab_size(),
                cancel,
                &mut *on_progress,
            )
        })
    }
}

async fn serve_held(held: Held) -> String {
    let tok = load_tokenizer();
    let h_id = tok.token_to_id("h").unwrap();
    let model: Arc<dyn ChatModel> = Arc::new(HeldModel {
        tokenizer: tok,
        h_id,
        held,
    });
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, build_router(model)).await.unwrap();
    });
    format!("http://{addr}")
}

/// A streaming request, fired as its own task; the join handle resolves
/// when the RESPONSE HEADERS are back (which is immediate even while the
/// generation is queued -- the SSE body is what stalls).
async fn fire_streaming(base: String) -> reqwest::Response {
    reqwest::Client::new()
        .post(format!("{base}/v1/chat/completions"))
        .json(&serde_json::json!({
            "model": "held",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 4,
            "temperature": 0.0,
            "stream": true,
        }))
        .send()
        .await
        .expect("the streaming request connects")
}

async fn wait_until(what: &str, mut cond: impl FnMut() -> bool) {
    let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
    while !cond() {
        if tokio::time::Instant::now() > deadline {
            panic!("condition not observed within 10s: {what}");
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

/// The FIFO contract: three requests fired with deterministic arrival
/// order generate in that order, and the gate's queued counter is what
/// made the arrival order deterministic in the first place.
#[tokio::test]
async fn generations_start_in_arrival_order_under_the_gate() {
    let held = Held::new();
    let base = serve_held(held.clone()).await;

    let outcome = tokio::time::timeout(Duration::from_secs(30), async {
        // Request 0 fires first and must be the one holding the permit.
        let r0 = tokio::spawn(fire_streaming(base.clone()));
        wait_until("request 0's generation starts", || {
            held.started.load(Ordering::SeqCst) == 1
        })
        .await;

        // Requests 1 and 2 fire only once observably queued.
        let r1 = tokio::spawn(fire_streaming(base.clone()));
        wait_until("request 1 is queued behind the gate", || {
            held.queue.queued() == 1
        })
        .await;
        let r2 = tokio::spawn(fire_streaming(base.clone()));
        wait_until("request 2 is queued behind the gate", || {
            held.queue.queued() == 2
        })
        .await;

        held.release();
        wait_until("all three generations ran", || {
            held.started.load(Ordering::SeqCst) == 3
        })
        .await;

        // Completion order == permit grant order: a body completes before
        // its permit drops, so the next generation cannot have started.
        let (done_tx, mut done_rx) = tokio::sync::mpsc::unbounded_channel();
        for (id, handle) in [r0, r1, r2].into_iter().enumerate() {
            let tx = done_tx.clone();
            tokio::spawn(async move {
                let response = handle.await.expect("request task lives");
                response.bytes().await.expect("body reads to completion");
                tx.send(id).unwrap();
            });
        }
        drop(done_tx);
        let mut order = Vec::new();
        while let Some(id) = done_rx.recv().await {
            order.push(id);
        }
        order
    })
    .await;

    let order = outcome.expect("the gate must not deadlock");
    assert_eq!(
        order,
        vec![0, 1, 2],
        "generations completed out of arrival order: the gate is not granting FIFO"
    );
    assert_eq!(
        held.started.load(Ordering::SeqCst),
        3,
        "all three requests must have generated exactly once"
    );
}

/// The queued-disconnect contract: a request whose client is gone before
/// its permit arrives never generates, and does not wedge the gate for the
/// requests behind it.
#[tokio::test]
async fn a_request_disconnected_while_queued_never_generates() {
    let held = Held::new();
    let base = serve_held(held.clone()).await;

    let outcome = tokio::time::timeout(Duration::from_secs(30), async {
        // Request 0 holds the gate. Its handle is deliberately unread:
        // only its START is observed, through `started`.
        let _r0 = tokio::spawn(fire_streaming(base.clone()));
        wait_until("request 0's generation starts", || {
            held.started.load(Ordering::SeqCst) == 1
        })
        .await;

        // Request 1 queues, then its client disconnects (the response is
        // dropped without the body ever being read).
        let r1 = tokio::spawn(fire_streaming(base.clone()));
        wait_until("request 1 is queued behind the gate", || {
            held.queue.queued() == 1
        })
        .await;
        let response = r1.await.expect("request task lives");
        drop(response);
        // Give the disconnect a moment to reach the server's CancelOnDrop;
        // loopback TCP close propagates in well under this.
        tokio::time::sleep(Duration::from_millis(500)).await;

        // Release the hold: request 0 finishes (its tiny body drains into
        // the socket buffers nobody needs to read); request 1's acquire
        // finds the cancel flag set and folds into silence without
        // generating.
        held.release();

        // A request arriving AFTER the disconnect must sail through: the
        // dead request did not wedge the gate.
        let sanity = tokio::spawn(fire_streaming(base.clone()));
        wait_until("the post-disconnect request generates", || {
            held.started.load(Ordering::SeqCst) >= 2
        })
        .await;
        let response = sanity.await.expect("sanity request task lives");
        response.bytes().await.expect("sanity body completes");
        held.started.load(Ordering::SeqCst)
    })
    .await;

    let started = outcome.expect("the gate must not deadlock");
    assert_eq!(
        started, 2,
        "expected exactly two generations (the holder and the post-disconnect \
         sanity request): the disconnected request must not have started"
    );
}
