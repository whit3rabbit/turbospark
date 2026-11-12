//! OCR C ABI: argument refusals, plus an ignored real-model read of the
//! rendered pages in `crates/runtime/testdata/coreml_ocr/`
//! (`TURBOSPARK_UNLIMITED_OCR_COREML_MODEL`, default
//! `~/models/coreml/unlimited-ocr-coreml`).

use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::ptr;

use turbospark_ffi::{
    abi, ts_last_error, ts_ocr_add_page_file, ts_ocr_add_page_rgb, ts_ocr_cancel,
    ts_ocr_catalog_json, ts_ocr_close, ts_ocr_install, ts_ocr_open, ts_ocr_read_json,
    ts_string_free, TsOcrSession,
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

#[test]
fn catalog_pins_ocr_and_install_checks_the_revision() {
    let mut out: *mut c_char = ptr::null_mut();
    assert_eq!(
        unsafe { ts_ocr_catalog_json(&mut out) },
        abi::TS_OK,
        "{}",
        last_error()
    );
    let rows: serde_json::Value =
        unsafe { serde_json::from_str(CStr::from_ptr(out).to_str().unwrap()).unwrap() };
    unsafe { ts_string_free(out) };
    let row = &rows[0];
    assert_eq!(row["alias"], "unlimited-ocr-coreml");
    assert_eq!(row["repository"], "FluidInference/unlimited-ocr-mlx-coreml");
    assert_eq!(row["revision"], "4568571eacd864e0a253c609b3c9d0e00450cebc");

    let alias = c("unlimited-ocr-coreml");
    let wrong = c("main");
    let mut out: *mut c_char = ptr::null_mut();
    let code = unsafe {
        ts_ocr_install(
            alias.as_ptr(),
            wrong.as_ptr(),
            None,
            ptr::null_mut(),
            &mut out,
        )
    };
    assert_eq!(code, abi::TS_ERR_INVALID_ARGUMENT);
    assert!(last_error().contains("pinned at"), "{}", last_error());
    let other = c("not-an-ocr-model");
    let rev = c("4568571eacd864e0a253c609b3c9d0e00450cebc");
    let code = unsafe {
        ts_ocr_install(
            other.as_ptr(),
            rev.as_ptr(),
            None,
            ptr::null_mut(),
            &mut out,
        )
    };
    assert_eq!(code, abi::TS_ERR_INVALID_ARGUMENT);
    assert!(out.is_null());
}

#[test]
fn refuses_null_and_missing_arguments() {
    let mut s: *mut TsOcrSession = ptr::null_mut();
    let m = c("/nonexistent/ocr");
    unsafe {
        assert_eq!(
            ts_ocr_open(m.as_ptr(), ptr::null_mut()),
            abi::TS_ERR_INVALID_ARGUMENT
        );
        assert_eq!(
            ts_ocr_open(ptr::null(), &mut s),
            abi::TS_ERR_INVALID_ARGUMENT
        );
        #[cfg(target_os = "macos")]
        assert_eq!(ts_ocr_open(m.as_ptr(), &mut s), abi::TS_ERR_OPEN);
        #[cfg(not(target_os = "macos"))]
        assert_eq!(ts_ocr_open(m.as_ptr(), &mut s), abi::TS_ERR_UNSUPPORTED);
        assert!(s.is_null());
        let rgb = [0u8; 12];
        assert_eq!(
            ts_ocr_add_page_rgb(ptr::null(), rgb.as_ptr(), 2, 2, 6),
            abi::TS_ERR_INVALID_ARGUMENT
        );
        let p = c("/x.png");
        assert_eq!(
            ts_ocr_add_page_file(ptr::null(), p.as_ptr()),
            abi::TS_ERR_INVALID_ARGUMENT
        );
        let mut out: *mut c_char = ptr::null_mut();
        assert_eq!(
            ts_ocr_read_json(ptr::null(), ptr::null(), &mut out),
            abi::TS_ERR_INVALID_ARGUMENT
        );
        // Null handles are no-ops.
        ts_ocr_cancel(ptr::null());
        ts_ocr_close(ptr::null_mut());
    }
}

#[test]
#[cfg(target_os = "macos")]
#[ignore = "needs the Core ML Unlimited OCR checkpoint (about 4.1 GB)"]
fn reads_pages_added_by_file_and_by_pixels() {
    let dir = std::env::var("TURBOSPARK_UNLIMITED_OCR_COREML_MODEL").unwrap_or_else(|_| {
        format!(
            "{}/models/coreml/unlimited-ocr-coreml",
            std::env::var("HOME").unwrap_or_default()
        )
    });
    if !std::path::Path::new(&dir).is_dir() {
        eprintln!("skipping: {dir} not found");
        return;
    }
    let testdata = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../runtime/testdata/coreml_ocr"
    );
    let m = c(&dir);
    let mut s: *mut TsOcrSession = ptr::null_mut();
    unsafe {
        assert_eq!(
            ts_ocr_open(m.as_ptr(), &mut s),
            abi::TS_OK,
            "{}",
            last_error()
        );
        let mut out: *mut c_char = ptr::null_mut();
        // Reading with no pages is refused.
        assert_eq!(
            ts_ocr_read_json(s, ptr::null(), &mut out),
            abi::TS_ERR_INVALID_ARGUMENT
        );

        // Page 1 by file, page 2 by pixels with padded rows.
        let p1 = c(&format!("{testdata}/page1.png"));
        assert_eq!(
            ts_ocr_add_page_file(s, p1.as_ptr()),
            abi::TS_OK,
            "{}",
            last_error()
        );
        let img = image_rgb(&format!("{testdata}/page2.png"));
        let (w, h) = (img.0, img.1);
        let stride = w * 3 + 16;
        let mut padded = vec![0u8; stride * h];
        for y in 0..h {
            padded[y * stride..y * stride + w * 3]
                .copy_from_slice(&img.2[y * w * 3..(y + 1) * w * 3]);
        }
        assert_eq!(
            ts_ocr_add_page_rgb(s, padded.as_ptr(), w, h, stride),
            abi::TS_OK
        );

        let opts = c(r#"{"max_new_tokens": 800}"#);
        assert_eq!(
            ts_ocr_read_json(s, opts.as_ptr(), &mut out),
            abi::TS_OK,
            "{}",
            last_error()
        );
        let json = CStr::from_ptr(out).to_str().unwrap().to_string();
        ts_string_free(out);
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(v["pages"], 2);
        assert_eq!(v["stopped_at_eos"], true);
        let text = v["text"].as_str().unwrap();
        assert!(text.contains("Quarterly Report"), "{text}");
        assert!(text.contains("Meeting Notes"), "{text}");
        assert!(v["raw_text"].as_str().unwrap().contains("<PAGE>"));

        // The pending list was consumed.
        assert_eq!(
            ts_ocr_read_json(s, ptr::null(), &mut out),
            abi::TS_ERR_INVALID_ARGUMENT
        );
        // Unknown options are refused.
        assert_eq!(ts_ocr_add_page_file(s, p1.as_ptr()), abi::TS_OK);
        let bad = c(r#"{"prompt": "x"}"#);
        assert_eq!(
            ts_ocr_read_json(s, bad.as_ptr(), &mut out),
            abi::TS_ERR_JSON
        );
        ts_ocr_close(s);
    }
}

#[cfg(target_os = "macos")]
fn image_rgb(path: &str) -> (usize, usize, Vec<u8>) {
    let img = turbospark_vision_io::decode_image_file(std::path::Path::new(path)).unwrap();
    (img.width, img.height, img.data)
}
