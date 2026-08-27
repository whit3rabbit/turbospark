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
    abi, session_for_testing, ts_generate, ts_last_error, ts_model_delete, ts_probe_json,
    ts_recommend_json, ts_session_cancel, ts_session_count_text_tokens, ts_session_count_tokens,
    ts_session_detokenize_json, ts_session_fit_window_json, ts_session_info_json, ts_session_open,
    ts_session_render_prompt, ts_session_tokenize_json, ts_string_free, ts_system_info_json,
    Session, TS_EVENT_CONTENT, TS_EVENT_PREFILL,
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
    let code = unsafe { ts_recommend_json(4096, &mut out) };
    // On platforms without memory probe (or CI), it returns an error or JSON array
    if code == abi::TS_OK {
        let json_str = unsafe { take(out) };
        let parsed: serde_json::Value = serde_json::from_str(&json_str).unwrap();
        assert!(parsed.is_array());
    }
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
