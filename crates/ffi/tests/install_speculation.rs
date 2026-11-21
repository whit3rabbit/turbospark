#![cfg(target_os = "macos")]
//! `ts_install_speculation_json` on real (synthetic) GGUF installs: the C
//! entry a host labels installed models with must give the open's answer
//! without opening anything.

use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::ptr;

use turbospark_ffi::{abi, ts_install_speculation_json, ts_string_free};

fn install(tag: &str, with_head: bool) -> std::path::PathBuf {
    let (bytes, _) = repack::build_synthetic_qwen35_dense_gguf(with_head);
    let header =
        repack::parse_gguf_header(&bytes, repack::GGUF_DEFAULT_MAX_HEADER_BYTES).expect("parse");
    let dir = std::env::temp_dir().join(format!(
        "turbospark-ffi-install-speculation-{tag}-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    repack::write_gguf_install_streamed(
        &dir,
        &header,
        &repack::MemoryRangeSource::new(&bytes),
        "qwen35-ffi-speculation",
        |_| {},
    )
    .expect("write install");
    dir
}

fn call(dir: &std::path::Path) -> (i32, serde_json::Value) {
    let path = CString::new(dir.to_str().unwrap()).unwrap();
    let mut out: *mut c_char = ptr::null_mut();
    let rc = unsafe { ts_install_speculation_json(path.as_ptr(), &mut out) };
    if out.is_null() {
        return (rc, serde_json::Value::Null);
    }
    let text = unsafe { CStr::from_ptr(out) }.to_str().unwrap().to_string();
    unsafe { ts_string_free(out) };
    (rc, serde_json::from_str(&text).expect("json"))
}

#[test]
fn a_gguf_install_with_its_head_reports_mtp_speculation() {
    let dir = install("head", true);
    let (rc, v) = call(&dir);
    std::fs::remove_dir_all(&dir).ok();
    assert_eq!(rc, abi::TS_OK);
    assert_eq!(v["available"], true, "{v}");
    assert_eq!(v["drafter"], "mtp", "{v}");
    assert!(v["reason"].is_null(), "{v}");
}

#[test]
fn a_headless_gguf_install_reports_the_engine_reason() {
    let dir = install("headless", false);
    let (rc, v) = call(&dir);
    std::fs::remove_dir_all(&dir).ok();
    assert_eq!(rc, abi::TS_OK);
    assert_eq!(v["available"], false, "{v}");
    assert!(v["drafter"].is_null(), "{v}");
    let reason = v["reason"].as_str().expect("a reason");
    assert!(reason.contains("multi-token-prediction head"), "{reason}");
}

#[test]
fn a_directory_with_no_manifest_is_an_invalid_argument() {
    let dir = std::env::temp_dir().join(format!(
        "turbospark-ffi-install-speculation-empty-{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let (rc, _) = call(&dir);
    std::fs::remove_dir_all(&dir).ok();
    assert_eq!(rc, abi::TS_ERR_INVALID_ARGUMENT);
}
