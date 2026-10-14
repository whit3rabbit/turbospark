//! `ts_embedding_encode_request_json`: argument and request refusals, plus
//! an ignored real-model run against the Core ML EmbeddingGemma 2 export
//! (`TURBOSPARK_EMBEDDINGGEMMA2_COREML_MODEL`, default
//! `~/models/coreml/embeddinggemma-2-coreml`).

use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::ptr;

use turbospark_ffi::{abi, ts_embedding_encode_request_json, ts_last_error, ts_string_free};

fn c(s: &str) -> CString {
    CString::new(s).unwrap()
}

fn last_error() -> String {
    unsafe {
        let len = ts_last_error(ptr::null_mut(), 0);
        let mut buf = vec![0u8; len + 1];
        ts_last_error(buf.as_mut_ptr() as *mut c_char, buf.len());
        CStr::from_ptr(buf.as_ptr() as *const c_char)
            .to_string_lossy()
            .into_owned()
    }
}

fn call(model: &str, request: &str) -> (i32, Option<String>) {
    let (m, r) = (c(model), c(request));
    let mut out: *mut c_char = ptr::null_mut();
    let code = unsafe { ts_embedding_encode_request_json(m.as_ptr(), r.as_ptr(), &mut out) };
    let text = (!out.is_null()).then(|| unsafe {
        let s = CStr::from_ptr(out).to_str().unwrap().to_string();
        ts_string_free(out);
        s
    });
    (code, text)
}

#[test]
fn refuses_null_arguments_and_bad_requests() {
    let m = c("/nonexistent/model");
    let r = c("{}");
    let code = unsafe { ts_embedding_encode_request_json(m.as_ptr(), r.as_ptr(), ptr::null_mut()) };
    assert_eq!(code, abi::TS_ERR_INVALID_ARGUMENT);
    let mut out: *mut c_char = ptr::null_mut();
    let code = unsafe { ts_embedding_encode_request_json(ptr::null(), r.as_ptr(), &mut out) };
    assert_eq!(code, abi::TS_ERR_INVALID_ARGUMENT);
    let code = unsafe { ts_embedding_encode_request_json(m.as_ptr(), ptr::null(), &mut out) };
    assert_eq!(code, abi::TS_ERR_INVALID_ARGUMENT);

    #[cfg(target_os = "macos")]
    {
        assert_eq!(call("/nonexistent/model", "not json").0, abi::TS_ERR_JSON);
        // Unknown fields are refused rather than silently ignored.
        assert_eq!(
            call("/nonexistent/model", r#"{"textz": ["a"]}"#).0,
            abi::TS_ERR_JSON
        );
        assert_eq!(
            call("/nonexistent/model", r#"{"texts": ["a"]}"#).0,
            abi::TS_ERR_OPEN
        );
    }
    #[cfg(not(target_os = "macos"))]
    assert_eq!(call("/nonexistent/model", "{}").0, abi::TS_ERR_UNSUPPORTED);
}

#[test]
#[cfg(target_os = "macos")]
#[ignore = "needs the Core ML EmbeddingGemma 2 checkpoint"]
fn embeds_text_image_and_audio_in_one_request() {
    let dir = std::env::var("TURBOSPARK_EMBEDDINGGEMMA2_COREML_MODEL").unwrap_or_else(|_| {
        format!(
            "{}/models/coreml/embeddinggemma-2-coreml",
            std::env::var("HOME").unwrap_or_default()
        )
    });
    if !std::path::Path::new(&dir).is_dir() {
        eprintln!("skipping: {dir} not found");
        return;
    }
    let testdata = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../runtime/testdata/coreml_embeddinggemma2"
    );
    let request = serde_json::json!({
        "texts": ["What is the capital of France?", "Paris is in France."],
        "task": "search_query",
        "dimensions": 256,
        "image_paths": [format!("{testdata}/pattern.png")],
        "image_budget": 70,
        "audio_paths": [format!("{testdata}/chirp.wav")],
    });
    let (code, out) = call(&dir, &request.to_string());
    assert_eq!(code, abi::TS_OK, "{}", last_error());
    let v: serde_json::Value = serde_json::from_str(&out.unwrap()).unwrap();
    assert_eq!(v["texts"].as_array().unwrap().len(), 2);
    assert_eq!(v["texts"][0].as_array().unwrap().len(), 256);
    assert_eq!(v["images"][0].as_array().unwrap().len(), 256);
    let windows = v["audio"][0].as_array().unwrap();
    assert_eq!(windows.len(), 1);
    assert_eq!(windows[0]["embedding"].as_array().unwrap().len(), 256);
    // A Matryoshka size the model does not declare is refused.
    let (code, _) = call(&dir, r#"{"texts": ["a"], "dimensions": 300}"#);
    assert_eq!(code, abi::TS_ERR_GENERATE);
}
