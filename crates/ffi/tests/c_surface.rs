//! The C entry points, driven through the same function bodies the header
//! exposes (this crate is `rlib` as well as `staticlib` for exactly that).
//!
//! Portable and model-free: everything here runs on any platform in
//! milliseconds. What it covers is the plumbing a real-model test cannot
//! isolate -- ownership, error propagation, the panic guard, and above all
//! **cancellation observed from a second thread**, which is the one property
//! of this crate's design that a manual click of a Stop button would be the
//! only other way to check.
//!
//! The tokenizer fixture is borrowed from `crates/tokenizer` by relative
//! path rather than copied. A copy would be a second thing to keep in step
//! with the loader's added-token renumbering for no benefit.

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int, c_void};
use std::path::PathBuf;
use std::ptr;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use foundation::LogitValue;
use tokenizer::MfTokenizer;
use turbospark_ffi::{
    abi, session_for_testing, session_for_testing_named, ts_generate, ts_last_error,
    ts_model_delete, ts_probe_json, ts_recommend_json, ts_server_attach_session,
    ts_server_detach_model, ts_server_info_json, ts_server_poll_events_json, ts_server_start,
    ts_server_stop, ts_session_cancel, ts_session_count_text_tokens, ts_session_count_tokens,
    ts_session_detokenize_json, ts_session_fit_window_json, ts_session_info_json, ts_session_open,
    ts_session_render_prompt, ts_session_tokenize_json, ts_string_free, ts_system_info_json,
    Server, Session, TS_EVENT_CONTENT, TS_EVENT_PREFILL,
};

fn fixture() -> MfTokenizer {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../tokenizer/tests/fixtures/ChatMLTokenizer");
    MfTokenizer::load_from_dir(&dir).expect("fixture tokenizer should load")
}

fn one_hot(vocab: usize, index: usize) -> Vec<LogitValue> {
    let mut v = vec![LogitValue::from_f32(0.0); vocab];
    v[index] = LogitValue::from_f32(1.0);
    v
}

fn c(s: &str) -> CString {
    CString::new(s).unwrap()
}

/// Reads and frees a `char **` out-parameter.
unsafe fn take(out: *mut c_char) -> String {
    assert!(!out.is_null(), "a successful call must write its out-param");
    let s = CStr::from_ptr(out).to_str().unwrap().to_string();
    ts_string_free(out);
    s
}

fn last_error() -> String {
    unsafe {
        // Ask for the length first, exactly as the header tells a caller to.
        let len = ts_last_error(ptr::null_mut(), 0);
        let mut buf = vec![0u8; len + 1];
        let wrote = ts_last_error(buf.as_mut_ptr() as *mut c_char, buf.len());
        assert_eq!(
            wrote, len,
            "the reported length must not depend on the buffer"
        );
        CStr::from_ptr(buf.as_ptr() as *const c_char)
            .to_str()
            .unwrap()
            .to_string()
    }
}

/// A session that decodes `token` forever, so a test controls exactly when a
/// generation ends.
fn endless_session(tokenizer: MfTokenizer, token: &str, steps: usize) -> Session {
    let vocab = tokenizer.vocab_size;
    let id = tokenizer.token_to_id(token).unwrap() as usize;
    session_for_testing(tokenizer, vec![one_hot(vocab, id); steps], vocab, 4096)
}

// ---------------------------------------------------------------- ownership

#[test]
fn a_null_argument_is_an_error_rather_than_a_crash() {
    let mut out: *mut c_char = ptr::null_mut();
    let code = unsafe { ts_session_open(ptr::null(), ptr::null(), ptr::null_mut()) };
    assert_eq!(code, abi::TS_ERR_INVALID_ARGUMENT);
    assert!(last_error().contains("null"), "got {:?}", last_error());

    // And the same for a required string, reached past the out-param check.
    let mut session: *mut Session = ptr::null_mut();
    let code = unsafe { ts_session_open(ptr::null(), ptr::null(), &mut session) };
    assert_eq!(code, abi::TS_ERR_INVALID_ARGUMENT);
    assert!(session.is_null(), "a failed open must not write a handle");
    let _ = &mut out;
}

#[test]
fn the_error_slot_reports_its_own_length_and_truncates_safely() {
    unsafe { ts_session_open(ptr::null(), ptr::null(), ptr::null_mut()) };
    let full = last_error();
    assert!(!full.is_empty());

    // A buffer far too small still produces a valid C string, and the return
    // value still names the length the caller would need.
    let mut tiny = [0u8; 4];
    let reported = unsafe { ts_last_error(tiny.as_mut_ptr() as *mut c_char, tiny.len()) };
    assert_eq!(reported, full.len());
    let truncated = unsafe { CStr::from_ptr(tiny.as_ptr() as *const c_char) };
    assert_eq!(
        truncated.to_bytes().len(),
        3,
        "one byte reserved for the NUL"
    );
    assert!(full.as_bytes().starts_with(truncated.to_bytes()));
}

#[test]
fn a_successful_call_clears_a_previous_error() {
    unsafe { ts_session_open(ptr::null(), ptr::null(), ptr::null_mut()) };
    assert!(!last_error().is_empty());

    let session = endless_session(fixture(), "h", 8);
    let mut out: *mut c_char = ptr::null_mut();
    let code = unsafe { ts_session_info_json(&session, &mut out) };
    assert_eq!(code, abi::TS_OK);
    let _ = unsafe { take(out) };
    assert!(
        last_error().is_empty(),
        "a stale message must not survive a successful call"
    );
}

#[test]
fn session_info_is_json_the_swift_side_can_decode() {
    let session = endless_session(fixture(), "h", 8);
    let mut out: *mut c_char = ptr::null_mut();
    assert_eq!(
        unsafe { ts_session_info_json(&session, &mut out) },
        abi::TS_OK
    );
    let json: serde_json::Value = serde_json::from_str(&unsafe { take(out) }).unwrap();
    // camelCase, so a Codable needs no CodingKeys.
    assert_eq!(json["maxContext"], 4096);
    assert!(json["vocabSize"].as_u64().unwrap() > 0);
    assert!(json.get("reasoningSupport").is_some());
    // The menu a GUI draws. Non-empty and opening at "off" on every install:
    // an empty array would hide the control, and a set not starting at "off"
    // would offer no way back to not thinking. Levels rendering the same
    // prompt are collapsed before they get here, so this is also the guard
    // that the field carries offerable choices rather than five spellings.
    let levels = json["reasoningLevels"]
        .as_array()
        .expect("reasoningLevels is reported as an array");
    assert_eq!(levels[0], "off");
    assert!(levels
        .iter()
        .all(|l| ["off", "low", "medium", "high", "xhigh"].contains(&l.as_str().unwrap())));
    // The speculation block is PRESENT AND NULL rather than absent, which
    // is the difference between "this session does not speculate" and "this
    // build predates the field". A scripted engine implements no drafter,
    // so null is the right answer here and there is no reason to report.
    let speculation = json.get("speculation").expect("speculation is reported");
    assert!(speculation["block"].is_null());
    assert!(speculation["drafter"].is_null());
    assert!(speculation["reason"].is_null());

    let steering = json.get("steering").expect("steering is reported");
    assert_eq!(steering["active"], false);
    assert!(steering["mode"].is_null());
    assert!(steering["scale"].is_null());
    assert!(steering["summary"].is_null());
}

// ------------------------------------------------------------- generation

/// Collects streamed events into `(kind, text)` pairs.
struct Sink {
    events: Mutex<Vec<(c_int, String)>>,
}

unsafe extern "C" fn collect(
    ud: *mut c_void,
    kind: c_int,
    text: *const c_char,
    len: usize,
    _a: u32,
    _b: u32,
) {
    let sink = &*(ud as *const Sink);
    let text = if text.is_null() || len == 0 {
        String::new()
    } else {
        std::str::from_utf8(std::slice::from_raw_parts(text as *const u8, len))
            .unwrap()
            .to_string()
    };
    sink.events.lock().unwrap().push((kind, text));
}

#[test]
fn a_generation_streams_events_and_reports_a_result() {
    let session = endless_session(fixture(), "h", 32);
    let sink = Sink {
        events: Mutex::new(Vec::new()),
    };
    let messages = c(r#"[{"role":"user","content":"hi"}]"#);
    let options = c(r#"{"maxNewTokens":4,"temperature":0.0,"topK":0,"topP":1.0}"#);
    let mut out: *mut c_char = ptr::null_mut();

    let code = unsafe {
        ts_generate(
            &session,
            messages.as_ptr(),
            options.as_ptr(),
            Some(collect),
            &sink as *const Sink as *mut c_void,
            &mut out,
        )
    };
    assert_eq!(code, abi::TS_OK, "{}", last_error());
    let result: serde_json::Value = serde_json::from_str(&unsafe { take(out) }).unwrap();

    assert_eq!(result["stopReason"], "maxTokens");
    assert_eq!(result["newTokens"], 4);
    let events = sink.events.lock().unwrap();
    assert!(events.iter().any(|(k, _)| *k == TS_EVENT_PREFILL));
    // The accumulated content in the result must equal what was streamed, or
    // a caller that trusted one of the two would be wrong about the other.
    let streamed: String = events
        .iter()
        .filter(|(k, _)| *k == TS_EVENT_CONTENT)
        .map(|(_, t)| t.as_str())
        .collect();
    assert_eq!(result["content"].as_str().unwrap(), streamed);
}

#[test]
fn a_null_callback_still_generates() {
    // A caller that only wants the finished turn passes no callback, and the
    // header promises that works.
    let session = endless_session(fixture(), "h", 32);
    let messages = c(r#"[{"role":"user","content":"hi"}]"#);
    let options = c(r#"{"maxNewTokens":3,"temperature":0.0,"topK":0,"topP":1.0}"#);
    let mut out: *mut c_char = ptr::null_mut();
    let code = unsafe {
        ts_generate(
            &session,
            messages.as_ptr(),
            options.as_ptr(),
            None,
            ptr::null_mut(),
            &mut out,
        )
    };
    assert_eq!(code, abi::TS_OK, "{}", last_error());
    let result: serde_json::Value = serde_json::from_str(&unsafe { take(out) }).unwrap();
    assert_eq!(result["newTokens"], 3);
}

#[test]
fn malformed_json_arguments_are_reported_rather_than_defaulted() {
    let session = endless_session(fixture(), "h", 8);
    let messages = c("not json at all");
    let mut out: *mut c_char = ptr::null_mut();
    let code = unsafe {
        ts_generate(
            &session,
            messages.as_ptr(),
            ptr::null(),
            None,
            ptr::null_mut(),
            &mut out,
        )
    };
    assert_eq!(code, abi::TS_ERR_JSON);
    assert!(
        last_error().contains("messagesJson"),
        "got {:?}",
        last_error()
    );
}

#[test]
fn an_unknown_reasoning_level_names_itself() {
    let session = endless_session(fixture(), "h", 8);
    let messages = c(r#"[{"role":"user","content":"hi"}]"#);
    let options = c(r#"{"reasoning":"enormous"}"#);
    let mut out: *mut c_char = ptr::null_mut();
    let code = unsafe {
        ts_generate(
            &session,
            messages.as_ptr(),
            options.as_ptr(),
            None,
            ptr::null_mut(),
            &mut out,
        )
    };
    assert_eq!(code, abi::TS_ERR_GENERATE);
    assert!(last_error().contains("enormous"), "got {:?}", last_error());
}

// ----------------------------------------------------------- cancellation

/// Cancels from a SECOND THREAD while the generation runs.
///
/// **The property this crate's whole session shape exists for.** The cancel
/// flag lives outside the engine mutex, so a Stop pressed on another thread
/// does not queue behind the turn it is trying to stop. If it were inside,
/// this test would not fail -- it would HANG, which is exactly what a user
/// would experience as a frozen window.
#[test]
fn cancelling_from_another_thread_stops_a_running_generation() {
    // Only a few dozen steps, not a budget's worth: the run is meant to stop
    // at token 8, and one scripted step is a whole vocabulary of logits, so
    // sizing this to `maxNewTokens` allocates ~100 MB and costs ten seconds
    // to prove nothing extra.
    let session = Arc::new(endless_session(fixture(), "h", 64));

    struct Counter {
        seen: AtomicUsize,
        session: Arc<Session>,
    }
    unsafe extern "C" fn tick(
        ud: *mut c_void,
        kind: c_int,
        _t: *const c_char,
        _l: usize,
        _a: u32,
        _b: u32,
    ) {
        if kind != TS_EVENT_CONTENT {
            return;
        }
        let counter = &*(ud as *const Counter);
        // On the 8th token, cancel from a DIFFERENT thread and wait for it.
        // Cancelling inline would prove nothing: the interesting claim is
        // that a thread which does not hold the engine lock can do this.
        if counter.seen.fetch_add(1, Ordering::SeqCst) == 7 {
            let session = Arc::clone(&counter.session);
            std::thread::spawn(move || ts_session_cancel(Arc::as_ptr(&session)))
                .join()
                .unwrap();
        }
    }

    let counter = Counter {
        seen: AtomicUsize::new(0),
        session: Arc::clone(&session),
    };
    let messages = c(r#"[{"role":"user","content":"hi"}]"#);
    // A budget far larger than the cancel point, so finishing early can only
    // be the cancel.
    let options = c(r#"{"maxNewTokens":100000,"temperature":0.0,"topK":0,"topP":1.0}"#);
    let mut out: *mut c_char = ptr::null_mut();

    let code = unsafe {
        ts_generate(
            Arc::as_ptr(&session),
            messages.as_ptr(),
            options.as_ptr(),
            Some(tick),
            &counter as *const Counter as *mut c_void,
            &mut out,
        )
    };
    assert_eq!(code, abi::TS_OK, "a cancelled run is not an error");
    let result: serde_json::Value = serde_json::from_str(&unsafe { take(out) }).unwrap();

    assert_eq!(result["stopReason"], "cancelled");
    let produced = result["newTokens"].as_u64().unwrap();
    assert!(
        (8..=9).contains(&produced),
        "should have stopped at the cancel point, produced {produced}"
    );
    // The partial turn is kept, so a GUI can leave it on screen.
    assert!(!result["content"].as_str().unwrap().is_empty());
}

#[test]
fn a_cancel_between_turns_does_not_cancel_the_next_one() {
    let session = endless_session(fixture(), "h", 64);
    // Pressed while nothing is running.
    unsafe { ts_session_cancel(&session) };

    let messages = c(r#"[{"role":"user","content":"hi"}]"#);
    let options = c(r#"{"maxNewTokens":5,"temperature":0.0,"topK":0,"topP":1.0}"#);
    let mut out: *mut c_char = ptr::null_mut();
    let code = unsafe {
        ts_generate(
            &session,
            messages.as_ptr(),
            options.as_ptr(),
            None,
            ptr::null_mut(),
            &mut out,
        )
    };
    assert_eq!(code, abi::TS_OK, "{}", last_error());
    let result: serde_json::Value = serde_json::from_str(&unsafe { take(out) }).unwrap();
    // The flag is cleared on entry, so the stale press is discarded.
    assert_eq!(result["stopReason"], "maxTokens");
    assert_eq!(result["newTokens"], 5);
}

// ----------------------------------------------------------- panic guard

#[test]
fn a_panic_is_caught_and_becomes_an_error_code() {
    // Reaches `guard` directly, since nothing in the shipped surface panics
    // on purpose. The point is that the mechanism works and that the payload
    // survives into the message, so a bug report has something in it.
    let code = abi::guard(|| panic!("deliberate test panic"));
    assert_eq!(code, abi::TS_ERR_PANIC);
    let message = last_error();
    assert!(message.contains("deliberate test panic"), "got {message:?}");
}

// ------------------------------------------------------- model management

#[test]
fn a_malformed_repository_is_rejected_before_any_network_call() {
    // No network: the shape check runs first, which is what makes this test
    // safe to keep in the standing suite.
    for bad in ["nameonly", "too/many/parts", "/leading", "trailing/"] {
        let repo = c(bad);
        let mut out: *mut c_char = ptr::null_mut();
        let code = unsafe { ts_probe_json(repo.as_ptr(), ptr::null(), ptr::null(), &mut out) };
        assert_eq!(code, abi::TS_ERR_JSON, "{bad} should be refused");
        assert!(
            last_error().contains("owner/name"),
            "{bad}: got {:?}",
            last_error()
        );
    }
}

#[test]
fn session_counts_prompt_tokens_correctly() {
    let session = endless_session(fixture(), "h", 10);
    let messages = c(r#"[{"role":"user","content":"Hello world"}]"#);
    let mut count: u32 = 0;
    let code =
        unsafe { ts_session_count_tokens(&session, messages.as_ptr(), ptr::null(), &mut count) };
    assert_eq!(code, abi::TS_OK, "{}", last_error());
    assert!(count > 0, "token count should be positive");
}

#[test]
fn session_counts_raw_text_tokens_correctly() {
    let session = endless_session(fixture(), "h", 10);
    let text = c("The quick brown fox jumps over the lazy dog.");
    let mut count: u32 = 0;
    let code = unsafe { ts_session_count_text_tokens(&session, text.as_ptr(), false, &mut count) };
    assert_eq!(code, abi::TS_OK, "{}", last_error());
    assert!(count > 0, "text token count should be positive");
}

#[test]
fn session_fits_conversation_window_correctly() {
    let session = endless_session(fixture(), "h", 10);
    // Multiple messages: system + older user/assistant turns + newest user turn
    let messages = c(r#"[
        {"role":"system","content":"You are a helpful assistant."},
        {"role":"user","content":"First question that is quite detailed and takes some token space."},
        {"role":"assistant","content":"First detailed answer that also occupies substantial token space."},
        {"role":"user","content":"Second question?"}
    ]"#);
    let mut out: *mut c_char = ptr::null_mut();
    // Use a bound that fits system + newest user, but not all 4 messages
    let code = unsafe {
        ts_session_fit_window_json(&session, messages.as_ptr(), ptr::null(), 40, &mut out)
    };
    assert_eq!(code, abi::TS_OK, "{}", last_error());
    let json_str = unsafe { take(out) };
    let parsed: serde_json::Value = serde_json::from_str(&json_str).unwrap();
    assert!(parsed.get("retained").is_some());
    assert!(parsed["removedTurnCount"].as_u64().unwrap() > 0);
    let retained = parsed["retained"].as_array().unwrap();
    // System message should be preserved
    assert_eq!(retained.first().unwrap()["role"], "system");
    // Newest user turn should be preserved
    assert_eq!(retained.last().unwrap()["role"], "user");
}

#[test]
fn system_info_json_is_valid_json() {
    let mut out: *mut c_char = ptr::null_mut();
    let code = unsafe { ts_system_info_json(&mut out) };
    assert_eq!(code, abi::TS_OK, "{}", last_error());
    let json_str = unsafe { take(out) };
    let parsed: serde_json::Value = serde_json::from_str(&json_str).unwrap();
    assert!(parsed.get("physicalMemoryBytes").is_some());
    assert!(parsed.get("thermalLevel").is_some());
}

#[test]
fn deleting_nonexistent_model_returns_error() {
    let alias = c("nonexistent_alias_12345");
    let code = unsafe { ts_model_delete(alias.as_ptr()) };
    assert_eq!(code, abi::TS_ERR_INVALID_ARGUMENT);
    assert!(last_error().contains("not installed"));
}

#[test]
fn recommend_json_returns_ranked_catalog_rows() {
    let mut out: *mut c_char = ptr::null_mut();
    // NULL options means every default, which is the `relaxed` tier.
    let code = unsafe { ts_recommend_json(4096, ptr::null(), &mut out) };
    // On platforms without memory probe (or CI), it returns an error or JSON array
    if code == abi::TS_OK {
        let json_str = unsafe { take(out) };
        let parsed: serde_json::Value = serde_json::from_str(&json_str).unwrap();
        assert!(parsed.is_array());
    }
}

/// The guard reaches the ranking, and a misspelling is REFUSED rather than
/// quietly honoured as the default -- the rule `sized` already follows for
/// `"atuo"`, and the one that matters most here: silently ranking under
/// `relaxed` when the caller asked for `strict` is the exact disagreement
/// between a recommendation and the open it recommends that this option
/// exists to prevent.
#[test]
fn recommend_json_accepts_a_load_guard_and_refuses_a_misspelling() {
    for tier in ["off", "relaxed", "balanced", "strict"] {
        let mut out: *mut c_char = ptr::null_mut();
        let opts = c(&format!(r#"{{"loadGuard":"{tier}"}}"#));
        let code = unsafe { ts_recommend_json(4096, opts.as_ptr(), &mut out) };
        assert_ne!(
            code,
            abi::TS_ERR_INVALID_ARGUMENT,
            "{tier} should be a recognized tier"
        );
        if code == abi::TS_OK {
            let _ = unsafe { take(out) };
        }
    }

    // A byte ceiling is the `Custom` tier and is equally legal.
    let mut out: *mut c_char = ptr::null_mut();
    let opts = c(r#"{"loadGuard":3221225472}"#);
    let code = unsafe { ts_recommend_json(4096, opts.as_ptr(), &mut out) };
    assert_ne!(code, abi::TS_ERR_INVALID_ARGUMENT);
    if code == abi::TS_OK {
        let _ = unsafe { take(out) };
    }

    let mut out: *mut c_char = ptr::null_mut();
    let opts = c(r#"{"loadGuard":"strcit"}"#);
    let code = unsafe { ts_recommend_json(4096, opts.as_ptr(), &mut out) };
    assert_eq!(code, abi::TS_ERR_INVALID_ARGUMENT);
    assert!(last_error().contains("loadGuard"), "{}", last_error());
}

#[test]
fn session_renders_formatted_chat_prompt() {
    let session = endless_session(fixture(), "h", 10);
    let messages = c(r#"[
        {"role":"system","content":"You are a helpful assistant."},
        {"role":"user","content":"Hello world"}
    ]"#);
    let mut out: *mut c_char = ptr::null_mut();
    let code =
        unsafe { ts_session_render_prompt(&session, messages.as_ptr(), ptr::null(), &mut out) };
    assert_eq!(code, abi::TS_OK, "{}", last_error());
    let prompt = unsafe { take(out) };
    assert!(prompt.contains("<|im_start|>system"));
    assert!(prompt.contains("You are a helpful assistant."));
    assert!(prompt.contains("<|im_start|>user"));
    assert!(prompt.contains("Hello world"));
    assert!(prompt.contains("<|im_start|>assistant"));
}

#[test]
fn session_tokenizes_and_detokenizes_roundtrip() {
    let session = endless_session(fixture(), "h", 10);
    let text = c("The quick brown fox jumps over the lazy dog.");
    let mut out_tokens: *mut c_char = ptr::null_mut();
    let code = unsafe { ts_session_tokenize_json(&session, text.as_ptr(), false, &mut out_tokens) };
    assert_eq!(code, abi::TS_OK, "{}", last_error());
    let tokens_json = unsafe { take(out_tokens) };
    let tokens: Vec<i32> = serde_json::from_str(&tokens_json).unwrap();
    assert!(!tokens.is_empty());

    let tokens_c = c(&tokens_json);
    let mut out_text: *mut c_char = ptr::null_mut();
    let decode_code =
        unsafe { ts_session_detokenize_json(&session, tokens_c.as_ptr(), false, &mut out_text) };
    assert_eq!(decode_code, abi::TS_OK, "{}", last_error());
    let decoded_text = unsafe { take(out_text) };
    assert_eq!(decoded_text, "The quick brown fox jumps over the lazy dog.");
}

#[test]
fn session_info_includes_special_tokens() {
    let session = endless_session(fixture(), "h", 8);
    let mut out: *mut c_char = ptr::null_mut();
    assert_eq!(
        unsafe { ts_session_info_json(&session, &mut out) },
        abi::TS_OK
    );
    let json: serde_json::Value = serde_json::from_str(&unsafe { take(out) }).unwrap();
    let special = json
        .get("specialTokens")
        .expect("specialTokens must be present in SessionInfo");
    assert!(special.get("stopTokenIds").is_some());
    let stop_ids = special["stopTokenIds"].as_array().unwrap();
    assert!(!stop_ids.is_empty());
}

// ----------------------------------------------------- in-process server

/// Starts a server over `session` and asserts success, returning the handle.
unsafe fn start_server(session: *const Session, options: &str) -> *mut Server {
    let opts = c(options);
    let mut server: *mut Server = ptr::null_mut();
    let code = ts_server_start(session, opts.as_ptr(), &mut server);
    assert_eq!(code, abi::TS_OK, "{}", last_error());
    assert!(!server.is_null(), "a successful start must write a handle");
    server
}

/// Reads back the ACTUALLY bound address (`port: 0` in the options asks the
/// OS to choose the port) and builds a base URL from it.
///
/// **BOTH HALVES COME OUT OF `ServerInfo`, INCLUDING THE HOST.** This helper
/// used to interpolate a `127.0.0.1` literal beside the read-back port, which
/// made every server case here pass unchanged under any bind change -- the
/// same restatement the app and the CLI were making, in the one file that
/// could have caught it.
fn server_base_url(server: *const Server) -> String {
    let json = server_info(server);
    let host = json["host"].as_str().expect("info reports a host");
    let port = json["port"].as_u64().unwrap();
    assert_ne!(port, 0, "port 0 must resolve to the actually bound port");
    format!("http://{host}:{port}")
}

/// `ts_server_info_json` as parsed JSON, asserting the call itself succeeded.
fn server_info(server: *const Server) -> serde_json::Value {
    let mut out: *mut c_char = ptr::null_mut();
    let code = unsafe { ts_server_info_json(server, &mut out) };
    assert_eq!(code, abi::TS_OK, "{}", last_error());
    serde_json::from_str(&unsafe { take(out) }).unwrap()
}

/// **THE `guardrails` OPTION IS ACTUALLY READ, and an unrecognized spelling
/// is refused rather than silently defaulted.**
///
/// This is the assertion that says the field is WIRED, not merely declared:
/// a `ts_server_start` that ignored the key entirely would start happily on
/// every input here, and the unit tests over `guardrails_config` would still
/// pass because they call the parser directly. The refusal is the observable
/// that can only happen if the option reached the validator.
///
/// It matters because the failure it guards against is invisible: a host that
/// asked for guardrails off and got them on has no way to tell from the
/// outside, which is exactly the state `swift/TurboSparkApp` was in before
/// this option existed.
#[test]
fn the_server_guardrails_option_is_read_rather_than_ignored() {
    let session = endless_session(fixture(), "h", 4);

    for good in ["{}", r#"{"guardrails":"on"}"#, r#"{"guardrails":"off"}"#] {
        let server = unsafe { start_server(&session, good) };
        unsafe { ts_server_stop(server) };
    }

    // Case matters, and so does an empty string: `crates/server/src/args.rs`
    // matches these two literals and nothing else, so accepting more here
    // would make the ABI looser than the flag it mirrors.
    for bad in [
        r#"{"guardrails":"disabled"}"#,
        r#"{"guardrails":"OFF"}"#,
        r#"{"guardrails":""}"#,
    ] {
        let opts = c(bad);
        let mut server: *mut Server = ptr::null_mut();
        let code = unsafe { ts_server_start(&session, opts.as_ptr(), &mut server) };
        assert_ne!(code, abi::TS_OK, "{bad} must be refused");
        assert!(server.is_null(), "a refused start must write no handle");
        let err = last_error();
        assert!(err.contains("guardrails"), "{bad}: {err}");
    }
}

/// A real HTTP round trip through the server this session started, proving
/// `ts_server_start` actually serves requests rather than merely building a
/// `Router` nothing is listening on.
#[tokio::test]
async fn a_server_started_from_a_session_serves_that_sessions_model() {
    let session = endless_session(fixture(), "h", 32);
    let server = unsafe { start_server(&session, "{}") };
    let base = server_base_url(server);

    let response = reqwest::Client::new()
        .post(format!("{base}/v1/chat/completions"))
        .json(&serde_json::json!({
            "model": "m", "max_tokens": 4, "temperature": 0.0,
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    let body: serde_json::Value = response.json().await.unwrap();
    let content = body["choices"][0]["message"]["content"]
        .as_str()
        .expect("a chat completion response carries message.content");
    assert!(!content.is_empty());

    unsafe { ts_server_stop(server) };
}

/// **THE PROPERTY THE `SessionCore` SPLIT EXISTS FOR.** Dropping the
/// `Session` handle that started a server must not take the engine down
/// with it: the server holds its own `Arc<SessionCore>` clone
/// (`session.rs`'s module doc), independent of the handle a caller passed to
/// `ts_server_start`. If this regressed to a borrow, this test would not
/// compile; if it regressed to sharing state without a true clone, the
/// request below would fail against a dropped engine instead of succeeding.
#[tokio::test]
async fn closing_the_session_does_not_stop_a_server_still_serving_it() {
    let session = endless_session(fixture(), "h", 32);
    let server = unsafe { start_server(&session, "{}") };
    let base = server_base_url(server);

    drop(session);

    let response = reqwest::Client::new()
        .post(format!("{base}/v1/chat/completions"))
        .json(&serde_json::json!({
            "model": "m", "max_tokens": 2, "temperature": 0.0,
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(
        response.status(),
        200,
        "the server's own Arc<SessionCore> clone must keep the engine alive \
         after the session handle that started it was dropped"
    );

    unsafe { ts_server_stop(server) };
}

/// `ts_server_stop` blocks until the background thread has actually exited
/// (`Server::stop`'s doc), so a connection attempt afterward must fail to
/// connect at all rather than merely time out waiting on a reply.
#[tokio::test]
async fn ts_server_stop_actually_stops_serving() {
    let session = endless_session(fixture(), "h", 4);
    let server = unsafe { start_server(&session, "{}") };
    let base = server_base_url(server);

    unsafe { ts_server_stop(server) };

    let result = reqwest::Client::new()
        .get(format!("{base}/health"))
        .send()
        .await;
    assert!(
        result.is_err(),
        "a request after ts_server_stop should fail to connect, got {result:?}"
    );
}

#[test]
fn server_options_accept_an_api_key_and_report_it_enabled() {
    let session = endless_session(fixture(), "h", 4);
    let server = unsafe { start_server(&session, r#"{"apiKey":"sk-test"}"#) };
    let json = server_info(server);
    assert_eq!(json["authEnabled"], true);
    assert_eq!(json["modelId"], "<scripted>");
    unsafe { ts_server_stop(server) };
}

/// **THE HOST IS AN OBSERVATION, AND THIS IS WHAT MAKES IT ONE.**
/// `server.rs` binds a `127.0.0.1` literal and reads the result back out of
/// `local_addr`; nothing else in this workspace can tell whether the reported
/// host came from the read or from the literal. `server_base_url`'s round
/// trips are NOT that check on their own -- a `0.0.0.0` bind still answers a
/// request addressed to `0.0.0.0` on this platform, so they would stay green
/// while the field reported something no caller should copy into a URL.
#[test]
fn the_reported_host_is_the_address_actually_bound() {
    let session = endless_session(fixture(), "h", 4);
    let server = unsafe { start_server(&session, "{}") };
    let json = server_info(server);
    assert_eq!(
        json["host"], "127.0.0.1",
        "this library binds loopback, and info must REPORT that rather than \
         leave a caller to assume it"
    );
    unsafe { ts_server_stop(server) };
}

// ------------------------------------------- attaching, detaching, polling

/// An `endless_session` under a chosen install path, which is what the
/// server derives a model id from.
fn named_session(path: &str, steps: usize) -> Session {
    let tokenizer = fixture();
    let vocab = tokenizer.vocab_size;
    let id = tokenizer.token_to_id("h").unwrap() as usize;
    session_for_testing_named(
        tokenizer,
        vec![one_hot(vocab, id); steps],
        vocab,
        4096,
        path,
    )
}

unsafe fn attach(server: *const Server, session: &Session) -> String {
    let mut out: *mut c_char = ptr::null_mut();
    let code = ts_server_attach_session(server, session, &mut out);
    assert_eq!(code, abi::TS_OK, "{}", last_error());
    take(out)
}

fn poll_events(server: *const Server, max: u32) -> serde_json::Value {
    let mut out: *mut c_char = ptr::null_mut();
    let code = unsafe { ts_server_poll_events_json(server, max, &mut out) };
    assert_eq!(code, abi::TS_OK, "{}", last_error());
    serde_json::from_str(&unsafe { take(out) }).unwrap()
}

async fn chat(base: &str, model: &str) -> (u16, serde_json::Value) {
    let response = reqwest::Client::new()
        .post(format!("{base}/v1/chat/completions"))
        .json(&serde_json::json!({
            "model": model, "max_tokens": 2, "temperature": 0.0,
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .send()
        .await
        .unwrap();
    (response.status().as_u16(), response.json().await.unwrap())
}

/// **A NULL SESSION IS A RUNNING SERVER WITH NOTHING TO SERVE**, which is
/// the state a GUI starts one in before its user has chosen a model. The
/// socket is bound and `/health` answers; generation is 503 rather than a
/// 404 or a hang.
#[tokio::test]
async fn a_server_started_with_no_session_binds_and_reports_itself_empty() {
    let server = unsafe { start_server(ptr::null(), "{}") };
    let base = server_base_url(server);

    let info = server_info(server);
    assert_eq!(info["models"].as_array().unwrap().len(), 0);
    assert_eq!(info["modelId"], "");

    let health: serde_json::Value = reqwest::get(format!("{base}/health"))
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(health["state"], "empty");

    let (status, _) = chat(&base, "anything").await;
    assert_eq!(status, 503, "no model attached is unavailable, not missing");

    unsafe { ts_server_stop(server) };
}

/// Attaching takes effect on a RUNNING server: no rebind, and the port the
/// caller already handed out keeps working.
#[tokio::test]
async fn a_model_attached_to_a_running_server_is_served_on_the_same_port() {
    let server = unsafe { start_server(ptr::null(), "{}") };
    let base = server_base_url(server);
    let session = named_session("/models/alpha.gturbo", 32);

    let id = unsafe { attach(server, &session) };
    assert_eq!(id, "alpha.gturbo", "the id is the install directory's name");

    let (status, body) = chat(&base, "alpha.gturbo").await;
    assert_eq!(status, 200, "{body}");
    assert_eq!(server_info(server)["models"][0], "alpha.gturbo");

    unsafe { ts_server_stop(server) };
}

/// Two models, addressed by id, on one server and one port.
#[tokio::test]
async fn two_attached_models_are_both_addressable_and_an_unknown_name_is_not() {
    let server = unsafe { start_server(ptr::null(), "{}") };
    let base = server_base_url(server);
    let alpha = named_session("/models/alpha.gturbo", 32);
    let beta = named_session("/models/beta.gturbo", 32);
    unsafe {
        attach(server, &alpha);
        attach(server, &beta);
    }

    assert_eq!(
        server_info(server)["models"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect::<Vec<_>>(),
        vec!["alpha.gturbo", "beta.gturbo"],
        "reported in attachment order"
    );

    for id in ["alpha.gturbo", "beta.gturbo"] {
        assert_eq!(chat(&base, id).await.0, 200, "{id}");
    }
    // With two attached there is a real ambiguity, so the fallback that
    // serves any name on a one-model server does not apply.
    assert_eq!(chat(&base, "neither-of-them").await.0, 404);

    unsafe { ts_server_stop(server) };
}

/// **A DUPLICATE ID IS REFUSED BY NAME.** Both scripted sessions here take
/// the default `<scripted>` path, so both derive the same id -- which is the
/// shape of a caller opening one install twice. Renaming the second would
/// make it addressable under a name the caller never learned, and would then
/// make a detach under the name they DO know remove the wrong one.
#[test]
fn attaching_a_second_model_under_the_same_id_is_refused() {
    let server = unsafe { start_server(ptr::null(), "{}") };
    let first = endless_session(fixture(), "h", 4);
    let second = endless_session(fixture(), "h", 4);
    unsafe { attach(server, &first) };

    let mut out: *mut c_char = ptr::null_mut();
    let code = unsafe { ts_server_attach_session(server, &second, &mut out) };
    assert_eq!(code, abi::TS_ERR_INVALID_ARGUMENT);
    assert!(out.is_null(), "a refused attach must not write an id");
    assert!(
        last_error().contains("already attached"),
        "the error must say why: {:?}",
        last_error()
    );
    assert_eq!(
        server_info(server)["models"].as_array().unwrap().len(),
        1,
        "the refused attach must not have half-added anything"
    );

    unsafe { ts_server_stop(server) };
}

/// Detaching removes the model from routing, and detaching one that is not
/// there is reported rather than silently succeeding.
///
/// **WHICH MODEL ANSWERED IS READ OFF THE EVENT STREAM, because the response
/// cannot say.** An OpenAI response echoes the request's own `model` field,
/// so a chat that returns 200 with `"model": "alpha.gturbo"` is equally
/// consistent with alpha having served it and with beta having served it
/// under the fallback -- which is exactly the pair this test has to tell
/// apart. `requestRouted.served` is the only place the answer exists.
#[tokio::test]
async fn detaching_removes_a_model_from_routing() {
    let server = unsafe { start_server(ptr::null(), "{}") };
    let base = server_base_url(server);
    let alpha = named_session("/models/alpha.gturbo", 32);
    let beta = named_session("/models/beta.gturbo", 32);
    unsafe {
        attach(server, &alpha);
        attach(server, &beta);
    }

    let id = c("alpha.gturbo");
    assert_eq!(
        unsafe { ts_server_detach_model(server, id.as_ptr()) },
        abi::TS_OK
    );
    assert_eq!(
        server_info(server)["models"].as_array().unwrap(),
        &vec![serde_json::json!("beta.gturbo")]
    );
    let _ = poll_events(server, 0);

    // **A REQUEST FOR THE DETACHED ID IS STILL ANSWERED, BY THE SURVIVOR.**
    // Down to one model, the single-model fallback applies again and serves
    // any name -- the same rule that keeps a client sending its own default
    // name working. It is worth knowing rather than assuming a 404: a host
    // that detaches a model does NOT stop that name from being accepted, it
    // stops that ENGINE from answering.
    assert_eq!(chat(&base, "alpha.gturbo").await.0, 200);
    let routed = poll_events(server, 0);
    let served = routed["events"]
        .as_array()
        .unwrap()
        .iter()
        .find(|e| e["kind"] == "requestRouted")
        .expect("a routed event");
    assert_eq!(served["requested"], "alpha.gturbo");
    assert_eq!(
        served["served"], "beta.gturbo",
        "the detached engine must not be the one that answered"
    );

    let again = unsafe { ts_server_detach_model(server, id.as_ptr()) };
    assert_eq!(
        again,
        abi::TS_ERR_INVALID_ARGUMENT,
        "a detach of something absent means the caller's list and the \
         server's have gone out of step, which is worth saying"
    );

    unsafe { ts_server_stop(server) };
}

/// **THE EVENTS ARE WHAT A HOST'S CONSOLE AND GRAPHS ARE BUILT ON.** Asserted
/// as a SEQUENCE rather than a set: a console renders them in order, and the
/// generation counters arriving before the request was routed would be
/// unreadable.
#[tokio::test]
async fn polling_drains_the_events_of_a_served_request_in_order() {
    let server = unsafe { start_server(ptr::null(), "{}") };
    let base = server_base_url(server);
    let session = named_session("/models/alpha.gturbo", 32);
    unsafe { attach(server, &session) };

    // The attach itself is an event, and draining it here leaves the next
    // poll showing only the request.
    let attached = poll_events(server, 0);
    assert_eq!(attached["events"][0]["kind"], "modelAttached");
    assert_eq!(attached["events"][0]["model"], "alpha.gturbo");
    assert_eq!(attached["dropped"], 0);

    assert_eq!(chat(&base, "alpha.gturbo").await.0, 200);

    let drained = poll_events(server, 0);
    let kinds: Vec<&str> = drained["events"]
        .as_array()
        .unwrap()
        .iter()
        .map(|e| e["kind"].as_str().unwrap())
        .collect();
    assert_eq!(
        kinds,
        vec![
            "requestStarted",
            "requestRouted",
            "generated",
            "requestFinished"
        ]
    );

    let events = drained["events"].as_array().unwrap();
    assert_eq!(events[0]["path"], "/v1/chat/completions");
    assert_eq!(events[1]["served"], "alpha.gturbo");
    assert_eq!(events[2]["newTokens"], 2, "max_tokens was 2");
    assert_eq!(events[3]["status"], 200);
    // One request, so every event carries the same id -- which is what ties
    // them together in a console.
    let id = events[0]["id"].as_u64().unwrap();
    assert!(events[..4]
        .iter()
        .all(|e| e["id"] == id || e["id"].is_null()));

    unsafe { ts_server_stop(server) };
}

/// A drain returns each event exactly once. A host polling on a timer
/// appends what it gets, so a peek would duplicate every row.
#[tokio::test]
async fn a_second_poll_returns_nothing_already_drained() {
    let server = unsafe { start_server(ptr::null(), "{}") };
    let session = named_session("/models/alpha.gturbo", 4);
    unsafe { attach(server, &session) };

    assert_eq!(
        poll_events(server, 0)["events"].as_array().unwrap().len(),
        1
    );
    assert_eq!(
        poll_events(server, 0)["events"].as_array().unwrap().len(),
        0
    );

    unsafe { ts_server_stop(server) };
}

/// `max` bounds one CALL, not the buffer: the remainder stays queued rather
/// than being discarded, so a burst arrives late and never silently short.
#[tokio::test]
async fn a_bounded_poll_leaves_the_rest_queued() {
    let server = unsafe { start_server(ptr::null(), "{}") };
    let base = server_base_url(server);
    let session = named_session("/models/alpha.gturbo", 32);
    unsafe { attach(server, &session) };
    assert_eq!(chat(&base, "alpha.gturbo").await.0, 200);

    // 1 attach + 4 request events.
    let first = poll_events(server, 2);
    assert_eq!(first["events"].as_array().unwrap().len(), 2);
    let rest = poll_events(server, 0);
    assert_eq!(rest["events"].as_array().unwrap().len(), 3);

    unsafe { ts_server_stop(server) };
}

/// The pre-registry spelling still works: a non-null session at start is
/// exactly an immediate attach, so every caller written against the
/// one-model shape is unaffected.
#[test]
fn starting_with_a_session_is_the_same_as_starting_empty_and_attaching() {
    let session = endless_session(fixture(), "h", 4);
    let server = unsafe { start_server(&session, "{}") };
    let info = server_info(server);
    assert_eq!(info["modelId"], "<scripted>");
    assert_eq!(info["models"][0], "<scripted>");
    unsafe { ts_server_stop(server) };
}

#[test]
fn generation_with_custom_stop_tokens() {
    let tok = fixture();
    let h_id = tok.token_to_id("h").unwrap();
    let session = endless_session(tok, "h", 32);
    let messages = c(r#"[{"role":"user","content":"hi"}]"#);
    let options = c(&format!(
        r#"{{"maxNewTokens":10,"temperature":0.0,"stopTokens":[{h_id}]}}"#
    ));
    let mut out: *mut c_char = ptr::null_mut();
    let code = unsafe {
        ts_generate(
            &session,
            messages.as_ptr(),
            options.as_ptr(),
            None,
            ptr::null_mut(),
            &mut out,
        )
    };
    assert_eq!(code, abi::TS_OK, "{}", last_error());
    let result: serde_json::Value = serde_json::from_str(&unsafe { take(out) }).unwrap();
    // It stopped on the first token because it is in stop_tokens
    assert!(result["newTokens"].as_u64().unwrap() <= 1);
}

/// A slot count outside `ALLOWED_CACHE_SLOTS` must be REFUSED at the boundary
/// rather than reaching `ExpertCacheSlots::Fixed`.
///
/// This engine is linked into its host -- `swift/TurboSparkApp` has no server
/// and no IPC -- so a value the engine will not honour is not an error a GUI
/// can display: `crates/streaming`'s expert cache panics and the abort takes
/// the whole app with it (root `CLAUDE.md` Gotcha 64 is the concrete case, at
/// `slots == top_k` on the first multi-token prompt). Every other front end
/// validated already; this binding did not, and it is the one whose caller is
/// a picker rather than a typed flag.
///
/// The model path here does not exist ON PURPOSE. Options are mapped before
/// anything is read from disk, which is what makes this reachable with no
/// install on the machine -- and asserting the error names the OPTION rather
/// than the path is what proves the check runs where that comment says.
#[test]
fn an_out_of_set_expert_cache_slot_count_is_refused_before_the_model_is_read() {
    for bad in ["4", "60", "200", "0", "7"] {
        let model = CString::new("/nonexistent/model.gturbo").unwrap();
        let options = CString::new(format!("{{\"expertCacheSlots\":{bad}}}")).unwrap();
        let mut session: *mut Session = ptr::null_mut();
        let code = unsafe { ts_session_open(model.as_ptr(), options.as_ptr(), &mut session) };

        assert_ne!(code, abi::TS_OK, "slots={bad} must not open");
        assert!(session.is_null(), "a failed open must not write a handle");
        let err = last_error();
        assert!(
            err.contains("expertCacheSlots"),
            "the refusal must name the option, not the path: got {err:?} for slots={bad}"
        );
    }

    // And the legal set still gets past the option check -- otherwise the
    // guard above would pass by refusing everything, which is the shape of a
    // gate that cannot fail.
    for good in ["8", "16", "24", "32", "48", "64", "96", "128"] {
        let model = CString::new("/nonexistent/model.gturbo").unwrap();
        let options = CString::new(format!("{{\"expertCacheSlots\":{good}}}")).unwrap();
        let mut session: *mut Session = ptr::null_mut();
        unsafe { ts_session_open(model.as_ptr(), options.as_ptr(), &mut session) };
        let err = last_error();
        assert!(
            !err.contains("expertCacheSlots"),
            "slots={good} is in ALLOWED_CACHE_SLOTS and must reach the model read: got {err:?}"
        );
    }

    // `auto` is not a number and must stay accepted.
    let model = CString::new("/nonexistent/model.gturbo").unwrap();
    let options = CString::new("{\"expertCacheSlots\":\"auto\"}").unwrap();
    let mut session: *mut Session = ptr::null_mut();
    unsafe { ts_session_open(model.as_ptr(), options.as_ptr(), &mut session) };
    assert!(
        !last_error().contains("expertCacheSlots"),
        "automatic sizing must not be caught by the allowed-set check"
    );
}
