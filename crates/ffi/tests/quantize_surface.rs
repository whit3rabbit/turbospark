//! The GGUF quantizer C entry points, driven through the same function bodies
//! the header exposes. Model-free: every case here is a refusal, a shape, or a
//! state query, so it runs in milliseconds without a GGUF on disk. The
//! quantization itself is held to llama.cpp byte for byte by the repack
//! crate's tests, which is where that claim lives.

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int};
use std::path::PathBuf;
use std::ptr;

use turbospark_ffi::{
    abi, ts_last_error, ts_quantize_cancel, ts_quantize_check_updates_json, ts_quantize_gguf,
    ts_quantize_recipe_from_gguf_json, ts_quantize_types_json, ts_string_free,
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
            .to_str()
            .unwrap()
            .to_string()
    }
}

/// A path under the target dir's scratch space that does not exist.
fn absent(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("quantize_surface_{name}"))
}

/// Runs `ts_quantize_gguf` with no progress callback.
fn quantize(options: &str) -> (c_int, String) {
    let opts = c(options);
    let mut out: *mut c_char = ptr::null_mut();
    let status = unsafe { ts_quantize_gguf(opts.as_ptr(), None, ptr::null_mut(), &mut out) };
    // A failed call must not hand back a string the caller would have to free.
    if status != 0 {
        assert!(out.is_null(), "a failed call must leave the out-param null");
    } else {
        unsafe { ts_string_free(out) };
    }
    (status, last_error())
}

#[cfg(target_os = "macos")]
mod macos {
    use super::*;

    #[test]
    fn null_arguments_are_refused() {
        let opts = c("{}");
        let status =
            unsafe { ts_quantize_gguf(opts.as_ptr(), None, ptr::null_mut(), ptr::null_mut()) };
        assert_eq!(status, abi::TS_ERR_INVALID_ARGUMENT);
        let mut out: *mut c_char = ptr::null_mut();
        let status = unsafe { ts_quantize_gguf(ptr::null(), None, ptr::null_mut(), &mut out) };
        assert_eq!(status, abi::TS_ERR_INVALID_ARGUMENT);
        assert!(out.is_null());
        unsafe {
            assert_eq!(
                ts_quantize_types_json(ptr::null_mut()),
                abi::TS_ERR_INVALID_ARGUMENT
            );
            assert_eq!(
                ts_quantize_recipe_from_gguf_json(c("x").as_ptr(), ptr::null_mut()),
                abi::TS_ERR_INVALID_ARGUMENT
            );
            let mut out: *mut c_char = ptr::null_mut();
            assert_eq!(
                ts_quantize_recipe_from_gguf_json(ptr::null(), &mut out),
                abi::TS_ERR_INVALID_ARGUMENT
            );
            assert_eq!(
                ts_quantize_check_updates_json(ptr::null(), &mut out),
                abi::TS_ERR_INVALID_ARGUMENT
            );
            assert_eq!(
                ts_quantize_check_updates_json(c("x").as_ptr(), ptr::null_mut()),
                abi::TS_ERR_INVALID_ARGUMENT
            );
        }
    }

    #[test]
    fn malformed_options_are_json_errors() {
        assert_eq!(quantize("{not json").0, abi::TS_ERR_JSON);
        // `deny_unknown_fields`: a misspelt key is an error, not a silent no-op.
        let (status, msg) =
            quantize(r#"{"source":"a","destination":"b","recipe":"q4_k","recipeFromGGUF":"c"}"#);
        assert_eq!(status, abi::TS_ERR_JSON, "{msg}");
    }

    #[test]
    fn recipe_and_recipe_from_gguf_are_exclusive_and_required() {
        let (status, msg) =
            quantize(r#"{"source":"a","destination":"b","recipe":"q4_k","recipeFromGguf":"c"}"#);
        assert_eq!(status, abi::TS_ERR_INVALID_ARGUMENT);
        assert!(msg.contains("not both"), "{msg}");

        let (status, msg) = quantize(r#"{"source":"a","destination":"b"}"#);
        assert_eq!(status, abi::TS_ERR_INVALID_ARGUMENT);
        assert!(msg.contains("required"), "{msg}");
    }

    #[test]
    fn an_existing_destination_is_refused_and_left_alone() {
        let dst = absent("existing_dst");
        std::fs::write(&dst, b"keep me").unwrap();
        let opts = serde_json::json!({
            "source": absent("no_such_source").display().to_string(),
            "destination": dst.display().to_string(),
            "recipe": "q4_k",
        })
        .to_string();
        let (status, msg) = quantize(&opts);
        assert_eq!(status, abi::TS_ERR_INVALID_ARGUMENT);
        assert!(msg.contains("refusing to overwrite"), "{msg}");
        assert_eq!(std::fs::read(&dst).unwrap(), b"keep me");
        std::fs::remove_file(&dst).unwrap();
    }

    #[test]
    fn a_bad_tie_order_is_refused() {
        let (status, msg) = quantize(
            r#"{"source":"a","destination":"/nonexistent-dir/b","recipe":"q4_k","iq1TieOrder":"x"}"#,
        );
        assert_eq!(status, abi::TS_ERR_INVALID_ARGUMENT);
        assert!(msg.contains("iq1TieOrder"), "{msg}");
    }

    /// One test, because both halves touch the process-wide run slot: a
    /// missing source gets past the argument checks and briefly holds it, so a
    /// parallel `cancel` could see 1.
    #[test]
    fn a_missing_source_fails_cleanly_and_cancel_then_has_nothing_to_stop() {
        let dst = absent("missing_src_dst");
        let _ = std::fs::remove_file(&dst);
        let opts = serde_json::json!({
            "source": absent("missing_src").display().to_string(),
            "destination": dst.display().to_string(),
            "recipe": "q4_k",
        })
        .to_string();
        let (status, _) = quantize(&opts);
        assert_ne!(status, 0);
        assert!(!dst.exists(), "a failed run must leave no destination");
        assert!(
            !PathBuf::from(format!("{}.partial", dst.display())).exists(),
            "a failed run must leave no partial file"
        );
        // The run slot was released on the failure path: nothing is running.
        assert_eq!(unsafe { ts_quantize_cancel() }, 0);
        // ... and a second run is not refused as busy.
        let (again, msg) = quantize(&opts);
        assert_ne!(again, abi::TS_ERR_BUSY, "{msg}");
    }

    #[test]
    fn types_json_lists_the_producible_types() {
        let mut out: *mut c_char = ptr::null_mut();
        assert_eq!(unsafe { ts_quantize_types_json(&mut out) }, 0);
        let text = unsafe { CStr::from_ptr(out).to_str().unwrap().to_string() };
        unsafe { ts_string_free(out) };
        let rows: Vec<serde_json::Value> = serde_json::from_str(&text).unwrap();
        assert!(
            rows.len() >= 20,
            "expected the 20 producible types, got {}",
            rows.len()
        );
        let mut names = std::collections::HashSet::new();
        for r in &rows {
            let name = r["name"].as_str().expect("name");
            assert!(names.insert(name.to_string()), "duplicate type {name}");
            assert!(r["ggmlType"].is_u64(), "{r}");
            assert!(r["bitsPerWeight"].as_f64().unwrap() > 0.0, "{r}");
            assert!(r["blockElements"].as_u64().unwrap() > 0, "{r}");
            assert!(r["requiresImatrix"].is_boolean(), "{r}");
            assert!(r["executable"].is_boolean(), "{r}");
            assert!(r["readableByStockLlamaCpp"].is_boolean(), "{r}");
        }
        // Every producible type runs here, and only the three Unsloth fork
        // types are unreadable by a stock llama.cpp.
        assert_eq!(rows.len(), 31, "producible type count");
        assert!(rows.iter().all(|r| r["executable"] == true), "{rows:?}");
        let fork: Vec<&str> = rows
            .iter()
            .filter(|r| r["readableByStockLlamaCpp"] == false)
            .map(|r| r["name"].as_str().unwrap())
            .collect();
        assert_eq!(fork, ["iq1_xs", "iq1_xxs", "iq1_xxxs"]);
        let q4k = rows.iter().find(|r| r["name"] == "q4_k").expect("q4_k");
        assert_eq!(q4k["blockElements"], 256);
        assert_eq!(q4k["requiresImatrix"], false);
        let iq1 = rows.iter().find(|r| r["name"] == "iq1_s").expect("iq1_s");
        assert_eq!(iq1["requiresImatrix"], true);
    }

    #[test]
    fn recipe_and_provenance_reads_of_missing_files_fail() {
        let missing = c(absent("no_such.gguf").to_str().unwrap());
        let mut out: *mut c_char = ptr::null_mut();
        let status = unsafe { ts_quantize_recipe_from_gguf_json(missing.as_ptr(), &mut out) };
        assert_eq!(status, abi::TS_ERR_INVALID_ARGUMENT);
        assert!(out.is_null());
        assert!(!last_error().is_empty());

        let missing = c(absent("no_such.provenance.json").to_str().unwrap());
        let status = unsafe { ts_quantize_check_updates_json(missing.as_ptr(), &mut out) };
        assert_eq!(status, abi::TS_ERR_INVALID_ARGUMENT);
        assert!(out.is_null());
    }
}

#[cfg(not(target_os = "macos"))]
#[test]
fn every_entry_point_reports_unsupported_off_macos() {
    assert_eq!(quantize("{}").0, abi::TS_ERR_UNSUPPORTED);
    let mut out: *mut c_char = ptr::null_mut();
    assert_eq!(
        unsafe { ts_quantize_types_json(&mut out) },
        abi::TS_ERR_UNSUPPORTED
    );
    assert_eq!(unsafe { ts_quantize_cancel() }, 0);
}
