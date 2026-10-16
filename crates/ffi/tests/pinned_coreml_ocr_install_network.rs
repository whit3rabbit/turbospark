//! Network gate: the pinned Unlimited OCR profile installs from Hugging
//! Face into a scratch store through the C ABI, refuses a second install,
//! opens by alias, and reads a rendered page. Ignored by default (about
//! 4.1 GB of downloads, about 8 GB peak disk while staging).
//!
//!     cargo test --release -p turbospark-ffi --test pinned_coreml_ocr_install_network -- --ignored --nocapture
//!
//! A binary of its own because `TURBOSPARK_HOME` is process-wide.

#![cfg(target_os = "macos")]

use std::ffi::{c_void, CStr, CString};
use std::os::raw::c_char;
use std::ptr;

use turbospark_ffi::{
    abi, ts_last_error, ts_ocr_add_page_file, ts_ocr_catalog_json, ts_ocr_close, ts_ocr_install,
    ts_ocr_open, ts_ocr_read_json, ts_string_free,
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

unsafe fn take(out: *mut c_char) -> serde_json::Value {
    assert!(!out.is_null());
    let s = CStr::from_ptr(out).to_str().unwrap().to_string();
    ts_string_free(out);
    serde_json::from_str(&s).unwrap()
}

unsafe extern "C" fn no_progress(
    _: *mut c_void,
    _: i32,
    _: *const c_char,
    _: usize,
    _: u64,
    _: u64,
) {
}

#[test]
#[ignore = "downloads about 4.1 GB from Hugging Face"]
fn pinned_unlimited_ocr_installs_opens_by_alias_and_reads() {
    let home = std::env::temp_dir().join(format!(
        "ts-pinned-ocr-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&home).unwrap();
    std::env::set_var("TURBOSPARK_HOME", &home);
    let testdata =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../runtime/testdata/coreml_ocr");

    unsafe {
        let mut out: *mut c_char = ptr::null_mut();
        assert_eq!(
            ts_ocr_catalog_json(&mut out),
            abi::TS_OK,
            "{}",
            last_error()
        );
        let rows = take(out);
        let row = &rows[0];
        assert_eq!(row["alias"], "unlimited-ocr-coreml");
        assert!(row["installed_path"].is_null());

        let alias = c("unlimited-ocr-coreml");
        let revision = c(row["revision"].as_str().unwrap());
        let mut out: *mut c_char = ptr::null_mut();
        let code = ts_ocr_install(
            alias.as_ptr(),
            revision.as_ptr(),
            Some(no_progress),
            ptr::null_mut(),
            &mut out,
        );
        assert_eq!(code, abi::TS_OK, "{}", last_error());
        let installed = take(out);
        eprintln!("ocr installed at {}", installed["path"]);
        let mut again: *mut c_char = ptr::null_mut();
        assert_ne!(
            ts_ocr_install(
                alias.as_ptr(),
                revision.as_ptr(),
                None,
                ptr::null_mut(),
                &mut again
            ),
            abi::TS_OK
        );
        let mut out: *mut c_char = ptr::null_mut();
        assert_eq!(ts_ocr_catalog_json(&mut out), abi::TS_OK);
        assert_eq!(take(out)[0]["installed_path"], installed["path"]);

        // The bare alias opens the verified install.
        let mut session = ptr::null_mut();
        assert_eq!(
            ts_ocr_open(alias.as_ptr(), &mut session),
            abi::TS_OK,
            "{}",
            last_error()
        );
        let page = c(testdata.join("page1.png").to_str().unwrap());
        assert_eq!(
            ts_ocr_add_page_file(session, page.as_ptr()),
            abi::TS_OK,
            "{}",
            last_error()
        );
        let mut out: *mut c_char = ptr::null_mut();
        assert_eq!(
            ts_ocr_read_json(session, ptr::null(), &mut out),
            abi::TS_OK,
            "{}",
            last_error()
        );
        let result = take(out);
        let text = result["text"].as_str().unwrap();
        eprintln!("ocr text: {text}");
        let pages: serde_json::Value =
            serde_json::from_slice(&std::fs::read(testdata.join("pages.json")).unwrap()).unwrap();
        for line in pages["pages"][0]["lines"].as_array().unwrap() {
            let line = line.as_str().unwrap();
            assert!(text.contains(line), "missing {line:?} in {text:?}");
        }
        ts_ocr_close(session);
    }
    std::fs::remove_dir_all(&home).unwrap();
}
