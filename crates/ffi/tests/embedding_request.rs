//! `ts_embedding_encode_request_json`: argument and request refusals, plus
//! an ignored real-model run against the Core ML EmbeddingGemma 2 export
//! (`TURBOSPARK_EMBEDDINGGEMMA2_COREML_MODEL`, default
//! `~/models/coreml/embeddinggemma-2-coreml`).

use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::ptr;

use turbospark_ffi::{
    abi, ts_embedding_catalog_json, ts_embedding_encode_request_json, ts_embedding_install,
    ts_embedding_probe_json, ts_last_error, ts_string_free,
};

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

fn take_json(out: *mut c_char) -> serde_json::Value {
    assert!(!out.is_null());
    let v = unsafe { serde_json::from_str(CStr::from_ptr(out).to_str().unwrap()).unwrap() };
    unsafe { ts_string_free(out) };
    v
}

fn probe(arg: &str) -> serde_json::Value {
    let a = c(arg);
    let mut out: *mut c_char = ptr::null_mut();
    let code = unsafe { ts_embedding_probe_json(a.as_ptr(), &mut out) };
    assert_eq!(code, abi::TS_OK, "{}", last_error());
    take_json(out)
}

struct Tmp(std::path::PathBuf);
impl Drop for Tmp {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn encoder_dir(model_type: &str) -> Tmp {
    let d = std::env::temp_dir().join(format!(
        "ts-ffi-embed-probe-{model_type}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&d).unwrap();
    std::fs::write(
        d.join("config.json"),
        format!(r#"{{"model_type":"{model_type}"}}"#),
    )
    .unwrap();
    std::fs::write(d.join("model.safetensors"), b"x").unwrap();
    std::fs::write(d.join("tokenizer.json"), b"{}").unwrap();
    Tmp(d)
}

#[test]
fn probe_reports_family_verdicts_without_loading() {
    let mut out: *mut c_char = ptr::null_mut();
    assert_eq!(
        unsafe { ts_embedding_probe_json(ptr::null(), &mut out) },
        abi::TS_ERR_INVALID_ARGUMENT
    );
    let a = c("x");
    assert_eq!(
        unsafe { ts_embedding_probe_json(a.as_ptr(), ptr::null_mut()) },
        abi::TS_ERR_INVALID_ARGUMENT
    );

    let gte = encoder_dir("gte");
    let v = probe(gte.0.to_str().unwrap());
    assert_eq!(v["runnable"], false);
    assert_eq!(v["verified"], false);
    #[cfg(target_os = "macos")]
    {
        assert_eq!(v["family"], "gte");
        assert!(v["refused_because"].as_str().unwrap().contains("gte"));

        let bert = encoder_dir("bert");
        let v = probe(bert.0.to_str().unwrap());
        assert_eq!(v["runnable"], true, "{v}");
        assert_eq!(v["family"], "bert");
        assert_eq!(v["backend"], "encoder");
        // Runnable, but no catalog pin vouches for these bytes.
        assert_eq!(v["verified"], false);
        assert!(v["profile_alias"].is_null());
    }

    // The encode path applies the same verdict: a refused family never
    // reaches a loader.
    let (code, _) = call(gte.0.to_str().unwrap(), r#"{"texts":["hi"]}"#);
    #[cfg(target_os = "macos")]
    {
        assert_eq!(code, abi::TS_ERR_OPEN);
        assert!(last_error().contains("gte"), "{}", last_error());
    }
    #[cfg(not(target_os = "macos"))]
    assert_eq!(code, abi::TS_ERR_UNSUPPORTED);
}

#[test]
fn catalog_rows_carry_status_and_install_checks_the_revision() {
    let mut out: *mut c_char = ptr::null_mut();
    assert_eq!(
        unsafe { ts_embedding_catalog_json(&mut out) },
        abi::TS_OK,
        "{}",
        last_error()
    );
    let rows = take_json(out);
    let row = rows
        .as_array()
        .unwrap()
        .iter()
        .find(|r| r["alias"] == "embeddinggemma-2-coreml")
        .expect("pinned profile");
    assert_eq!(row["status"], "verified");
    assert!(!row["gates"].as_array().unwrap().is_empty());
    assert!(
        ["current", "stale", "foreign", "absent"].contains(&row["install_state"].as_str().unwrap()),
        "{row}"
    );
    assert_eq!(row["revision"].as_str().unwrap().len(), 40);

    // A revision the catalog does not pin is refused before any download.
    let alias = c("embeddinggemma-2-coreml");
    let wrong = c("main");
    let mut out: *mut c_char = ptr::null_mut();
    let code = unsafe {
        ts_embedding_install(
            alias.as_ptr(),
            wrong.as_ptr(),
            None,
            ptr::null_mut(),
            &mut out,
        )
    };
    assert_eq!(code, abi::TS_ERR_INVALID_ARGUMENT);
    assert!(last_error().contains("pinned at"), "{}", last_error());
    assert!(out.is_null());
    let code = unsafe {
        ts_embedding_install(alias.as_ptr(), ptr::null(), None, ptr::null_mut(), &mut out)
    };
    assert_eq!(code, abi::TS_ERR_INVALID_ARGUMENT);
}
