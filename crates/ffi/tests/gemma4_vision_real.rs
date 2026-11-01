//! `ts_generate` answering an image prompt on a real Gemma 4 install with its
//! tower (`docs/GEMMA4_VISION_PHASE0.md`), through the C entry points the
//! app calls.
//!
//! ```sh
//! TURBOSPARK_GEMMA4_VISION_INSTALL_DIR=~/models/gemma4-26b-vision.gturbo \
//! TURBOSPARK_GEMMA4_VISION_PAGE=/tmp/gemma4-trunk-dump/image.png \
//!   cargo test -p turbospark-ffi --test gemma4_vision_real --release -- --ignored --nocapture
//! ```
//!
//! The image reads "TURBOSPARK 42" (`scripts/gemma4_vision_trunk_dump.py`
//! writes one), so the check is on the answer: a session that dropped the
//! image answers from the question alone.
#![cfg(target_os = "macos")]

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int, c_void};
use std::ptr;

use turbospark_ffi::{
    abi, ts_generate, ts_last_error, ts_session_close, ts_session_info_json, ts_session_open,
    ts_string_free, Session,
};

unsafe fn take(out: *mut c_char) -> String {
    let s = CStr::from_ptr(out).to_str().unwrap().to_string();
    ts_string_free(out);
    s
}

fn last_error() -> String {
    let mut buf = vec![0 as c_char; 4096];
    unsafe { ts_last_error(buf.as_mut_ptr(), buf.len()) };
    unsafe { CStr::from_ptr(buf.as_ptr()) }
        .to_string_lossy()
        .into_owned()
}

unsafe extern "C" fn ignore(_: *mut c_void, _: c_int, _: *const c_char, _: usize, _: u32, _: u32) {}

#[test]
#[ignore = "needs a real Gemma 4 vision install (TURBOSPARK_GEMMA4_VISION_INSTALL_DIR) and an \
            image (TURBOSPARK_GEMMA4_VISION_PAGE)"]
fn ts_generate_reads_an_image_on_a_gemma4_install() {
    let (Some(dir), Some(page)) = (
        std::env::var("TURBOSPARK_GEMMA4_VISION_INSTALL_DIR").ok(),
        std::env::var("TURBOSPARK_GEMMA4_VISION_PAGE").ok(),
    ) else {
        eprintln!("the Gemma 4 vision variables are not both set; skipping");
        return;
    };
    let dir_c = CString::new(dir).unwrap();
    let mut session: *mut Session = ptr::null_mut();
    let code = unsafe { ts_session_open(dir_c.as_ptr(), ptr::null(), &mut session) };
    assert_eq!(code, abi::TS_OK, "open: {}", last_error());

    let mut out: *mut c_char = ptr::null_mut();
    assert_eq!(
        unsafe { ts_session_info_json(session, &mut out) },
        abi::TS_OK
    );
    let info: serde_json::Value = serde_json::from_str(&unsafe { take(out) }).unwrap();
    assert_eq!(
        info["vision"]["active"], true,
        "the session must report vision as active: {}",
        info["vision"]
    );

    let messages = serde_json::json!([{
        "role": "user",
        "content": [
            {"type": "image", "path": page},
            {"type": "text", "text": "What text is written in this image? Answer with the text only."}
        ]
    }]);
    let messages = CString::new(messages.to_string()).unwrap();
    let options = CString::new(r#"{"maxNewTokens":16,"temperature":0}"#).unwrap();
    let mut result: *mut c_char = ptr::null_mut();
    let code = unsafe {
        ts_generate(
            session,
            messages.as_ptr(),
            options.as_ptr(),
            Some(ignore),
            ptr::null_mut(),
            &mut result,
        )
    };
    assert_eq!(code, abi::TS_OK, "generate: {}", last_error());
    let result: serde_json::Value = serde_json::from_str(&unsafe { take(result) }).unwrap();
    let content = result["content"].as_str().unwrap_or_default();
    eprintln!(
        "ts_generate: {content:?} ({} prompt tokens)",
        result["promptTokens"]
    );
    assert!(
        content.contains("TURBOSPARK") && content.contains("42"),
        "the model did not read the image; it answered {content:?}"
    );
    unsafe { ts_session_close(session) };
}
