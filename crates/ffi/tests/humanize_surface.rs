//! `ts_humanize` through its real function body, on a scripted session.
//!
//! Model-free, like `c_surface.rs`: what this pins is the plumbing around the
//! humanizer model (the wrapper reaching the engine verbatim, the closed
//! option set, the audit riding beside the turn result), not the model's
//! output quality, which the bench gates own.

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int, c_void};
use std::path::PathBuf;
use std::ptr;
use std::sync::Mutex;

use foundation::humanizer::build_humanize_prompt;
use foundation::LogitValue;
use tokenizer::MfTokenizer;
use turbospark_ffi::{
    abi, session_for_testing, ts_humanize, ts_last_error, ts_string_free, Session, TS_EVENT_CONTENT,
};

fn fixture() -> MfTokenizer {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../tokenizer/tests/fixtures/ChatMLTokenizer");
    MfTokenizer::load_from_dir(&dir).expect("fixture tokenizer should load")
}

/// Steps enough to prefill the wrapped `DRAFT` (the scripted producer spends
/// one per prompt token) and then decode a few tokens.
fn steps_for_draft() -> usize {
    fixture().encode(&build_humanize_prompt(DRAFT), true).len() + 16
}

/// Decodes `token` for `steps` forward calls. The logit is large on purpose: the recipe
/// samples at temperature 1.0, and a unit-height spike over a ~150k-entry
/// vocabulary would sample the spike with near-zero probability.
fn endless_session(token: &str, steps: usize) -> Session {
    let tokenizer = fixture();
    let vocab = tokenizer.vocab_size;
    let id = tokenizer.token_to_id(token).unwrap() as usize;
    let mut row = vec![LogitValue::from_f32(0.0); vocab];
    row[id] = LogitValue::from_f32(50.0);
    session_for_testing(tokenizer, vec![row; steps], vocab, 4096)
}

fn c(s: &str) -> CString {
    CString::new(s).unwrap()
}

fn last_error() -> String {
    unsafe {
        let len = ts_last_error(ptr::null_mut(), 0);
        let mut buf = vec![0u8; len + 1];
        ts_last_error(buf.as_mut_ptr() as *mut c_char, buf.len());
        CStr::from_ptr(buf.as_ptr() as *const c_char)
            .to_str()
            .unwrap()
            .to_string()
    }
}

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

fn sink() -> Sink {
    Sink {
        events: Mutex::new(Vec::new()),
    }
}

const DRAFT: &str = "Item 37 shipped in 2024, and the team was pleased.";

#[test]
fn a_rewrite_streams_and_carries_the_audit_beside_the_turn_result() {
    let session = endless_session("h", steps_for_draft());
    let sink = sink();
    let (draft, options) = (c(DRAFT), c(r#"{"maxNewTokens":4,"seed":7}"#));
    let mut out: *mut c_char = ptr::null_mut();
    let code = unsafe {
        ts_humanize(
            &session,
            draft.as_ptr(),
            options.as_ptr(),
            Some(collect),
            &sink as *const Sink as *mut c_void,
            &mut out,
        )
    };
    assert_eq!(code, abi::TS_OK, "{}", last_error());
    let json = unsafe { CStr::from_ptr(out).to_str().unwrap().to_string() };
    unsafe { ts_string_free(out) };
    let result: serde_json::Value = serde_json::from_str(&json).unwrap();

    assert_eq!(result["stopReason"], "maxTokens");
    assert_eq!(result["newTokens"], 4);

    // THE RAW-COMPLETION PROOF. The prompt the engine saw is exactly the
    // byte-exact wrapper plus BOS and nothing else: a chat template would
    // add turn markers and move this count.
    let expected = fixture().encode(&build_humanize_prompt(DRAFT), true).len();
    assert_eq!(result["promptTokens"], expected);

    let streamed: String = sink
        .events
        .lock()
        .unwrap()
        .iter()
        .filter(|(k, _)| *k == TS_EVENT_CONTENT)
        .map(|(_, t)| t.as_str())
        .collect();
    assert_eq!(result["content"].as_str().unwrap(), streamed);
    assert!(!streamed.is_empty());

    // The rewrite ("hhhh") kept none of the draft's numbers.
    let audit = &result["audit"];
    assert_eq!(audit["droppedNumbers"], serde_json::json!(["2024", "37"]));
    assert_eq!(audit["inventedNumbers"], serde_json::json!([]));
    assert_eq!(audit["reuseFlagged"], false);
    let warnings = audit["warnings"].as_array().unwrap();
    assert!(
        warnings
            .iter()
            .any(|w| w.as_str().unwrap().contains("numbers dropped")),
        "{warnings:?}"
    );
}

#[test]
fn only_the_budget_and_seed_are_settable_and_the_rest_is_refused_by_name() {
    let session = endless_session("h", 8);
    let sink = sink();
    let (draft, options) = (c(DRAFT), c(r#"{"temperature":0.2}"#));
    let mut out: *mut c_char = ptr::null_mut();
    let code = unsafe {
        ts_humanize(
            &session,
            draft.as_ptr(),
            options.as_ptr(),
            Some(collect),
            &sink as *const Sink as *mut c_void,
            &mut out,
        )
    };
    assert_eq!(code, abi::TS_ERR_JSON);
    assert!(last_error().contains("temperature"), "{}", last_error());
    assert!(out.is_null());
    assert!(sink.events.lock().unwrap().is_empty());
}

#[test]
fn an_empty_draft_and_a_null_out_pointer_fail_before_any_decode() {
    let session = endless_session("h", 8);
    let sink = sink();
    let mut out: *mut c_char = ptr::null_mut();

    let blank = c("  \n ");
    let code = unsafe {
        ts_humanize(
            &session,
            blank.as_ptr(),
            ptr::null(),
            Some(collect),
            &sink as *const Sink as *mut c_void,
            &mut out,
        )
    };
    assert_eq!(code, abi::TS_ERR_GENERATE);
    assert!(last_error().contains("empty"), "{}", last_error());

    let draft = c(DRAFT);
    let code = unsafe {
        ts_humanize(
            &session,
            draft.as_ptr(),
            ptr::null(),
            Some(collect),
            &sink as *const Sink as *mut c_void,
            ptr::null_mut(),
        )
    };
    assert_eq!(code, abi::TS_ERR_INVALID_ARGUMENT);
    assert!(
        sink.events.lock().unwrap().is_empty(),
        "nothing may decode before the arguments are checked"
    );
}

/// The whole C path on the real install: open at the model's window, rewrite
/// a draft with numbers, check the stop, the stream and the audit.
///
/// Ignored: it maps a 12.7 GB install. Run with
/// `TURBOSPARK_HUMANIZER_INSTALL_DIR=~/.turbospark/models/text/humanizer-12b.gturbo
///  cargo test -p turbospark-ffi --test humanize_surface --release -- --ignored --nocapture`
#[test]
#[ignore = "needs the humanizer-12b install"]
fn the_real_model_rewrites_through_the_c_surface() {
    use turbospark_ffi::{ts_session_close, ts_session_open};

    let dir = std::env::var("TURBOSPARK_HUMANIZER_INSTALL_DIR")
        .expect("set TURBOSPARK_HUMANIZER_INSTALL_DIR");
    let dir = c(&dir);
    let open_options = c(r#"{"maxContext":8192}"#);
    let mut session: *mut Session = ptr::null_mut();
    let code = unsafe { ts_session_open(dir.as_ptr(), open_options.as_ptr(), &mut session) };
    assert_eq!(code, abi::TS_OK, "{}", last_error());

    let draft = "In conclusion, it is important to note that our team shipped 37 fixes across \
        12 services in March 2024. Moreover, the on-call load dropped by 40 percent, which is \
        truly a testament to the quality of the work.";
    let (draft_c, options) = (c(draft), c(r#"{"seed":3}"#));
    let sink = sink();
    let mut out: *mut c_char = ptr::null_mut();
    let code = unsafe {
        ts_humanize(
            session,
            draft_c.as_ptr(),
            options.as_ptr(),
            Some(collect),
            &sink as *const Sink as *mut c_void,
            &mut out,
        )
    };
    assert_eq!(code, abi::TS_OK, "{}", last_error());
    let json = unsafe { CStr::from_ptr(out).to_str().unwrap().to_string() };
    unsafe { ts_string_free(out) };
    let result: serde_json::Value = serde_json::from_str(&json).unwrap();
    eprintln!("{}", serde_json::to_string_pretty(&result).unwrap());

    // Ends on the model's own EOS, not the budget, and produced real text.
    assert_eq!(result["stopReason"], "eos");
    let content = result["content"].as_str().unwrap();
    assert!(content.split_whitespace().count() > 10, "{content:?}");
    assert!(
        !content.contains("### Rewritten"),
        "the separator leaked: {content:?}"
    );
    let streamed: String = sink
        .events
        .lock()
        .unwrap()
        .iter()
        .filter(|(k, _)| *k == TS_EVENT_CONTENT)
        .map(|(_, t)| t.as_str())
        .collect();
    assert_eq!(content, streamed);
    // The audit object is present and structured.
    assert!(result["audit"]["reuseRatio"].is_number());
    assert!(result["audit"]["warnings"].is_array());

    unsafe { ts_session_close(session) };
}
