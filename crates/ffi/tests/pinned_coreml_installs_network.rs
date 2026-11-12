//! Network gate: the two pinned Core ML profiles install from Hugging Face
//! into a scratch store through the same C ABI the app uses, and the
//! installs then run. Ignored by default (about 1.7 GB of downloads).
//!
//!     cargo test --release -p turbospark-ffi --test pinned_coreml_installs_network -- --ignored --nocapture
//!
//! `TURBOSPARK_HOME` points the default store at a temporary directory, so
//! the real store is never touched. One test function, because the store
//! root is process-wide.

#![cfg(target_os = "macos")]

use std::ffi::{c_void, CStr, CString};
use std::os::raw::c_char;
use std::ptr;

use turbospark_ffi::{
    abi, ts_audio_catalog_json, ts_audio_install_with_options, ts_audio_job_append,
    ts_audio_job_close, ts_audio_job_open, ts_audio_job_run, ts_audio_session_close,
    ts_audio_session_open, ts_embedding_catalog_json, ts_embedding_encode_request_json,
    ts_embedding_install, ts_embedding_probe_json, ts_last_error, ts_string_free,
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

fn say_16k(text: &str, dir: &std::path::Path) -> Vec<f32> {
    let path = dir.join("clip.wav");
    let ok = std::process::Command::new("say")
        .arg("-o")
        .arg(&path)
        .args(["--data-format=LEI16@16000", text])
        .status()
        .unwrap()
        .success();
    assert!(ok, "`say` failed");
    audio::wav::read_wav_f32(&path).unwrap().samples
}

#[test]
#[ignore = "downloads about 1.7 GB from Hugging Face"]
fn pinned_parakeet_and_embeddinggemma2_install_and_run() {
    let home = std::env::temp_dir().join(format!(
        "ts-pinned-coreml-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&home).unwrap();
    std::env::set_var("TURBOSPARK_HOME", &home);

    unsafe {
        // --- EmbeddingGemma 2: listed, not installed, then installed by alias.
        let mut out: *mut c_char = ptr::null_mut();
        assert_eq!(
            ts_embedding_catalog_json(&mut out),
            abi::TS_OK,
            "{}",
            last_error()
        );
        let rows = take(out);
        let row = rows
            .as_array()
            .unwrap()
            .iter()
            .find(|r| r["alias"] == "embeddinggemma-2-coreml")
            .expect("pinned embedding profile");
        assert!(row["installed_path"].is_null());
        assert_eq!(row["can_run"], true);
        assert_eq!(row["status"], "verified");

        let alias = c("embeddinggemma-2-coreml");
        let revision = c(row["revision"].as_str().unwrap());
        let mut out: *mut c_char = ptr::null_mut();
        let code = ts_embedding_install(
            alias.as_ptr(),
            revision.as_ptr(),
            Some(no_progress),
            ptr::null_mut(),
            &mut out,
        );
        assert_eq!(code, abi::TS_OK, "{}", last_error());
        let installed = take(out);
        eprintln!("embedding installed at {}", installed["path"]);
        // A second install of the same alias at the same pin is refused.
        let mut again: *mut c_char = ptr::null_mut();
        assert_ne!(
            ts_embedding_install(
                alias.as_ptr(),
                revision.as_ptr(),
                None,
                ptr::null_mut(),
                &mut again
            ),
            abi::TS_OK
        );
        // The alias now probes as the verified install.
        let mut out: *mut c_char = ptr::null_mut();
        assert_eq!(
            ts_embedding_probe_json(alias.as_ptr(), &mut out),
            abi::TS_OK,
            "{}",
            last_error()
        );
        let verdict = take(out);
        assert_eq!(verdict["runnable"], true, "{verdict}");
        assert_eq!(verdict["verified"], true, "{verdict}");
        assert_eq!(verdict["family"], "embeddinggemma2");
        assert_eq!(verdict["profile_alias"], "embeddinggemma-2-coreml");
        let mut out: *mut c_char = ptr::null_mut();
        assert_eq!(ts_embedding_catalog_json(&mut out), abi::TS_OK);
        let rows = take(out);
        assert_eq!(rows[0]["installed_path"], installed["path"]);

        // The bare alias works where the app passes an embedding model.
        let req = c(r#"{"texts": ["hello"], "task": "search_query", "dimensions": 128}"#);
        let mut out: *mut c_char = ptr::null_mut();
        let code = ts_embedding_encode_request_json(alias.as_ptr(), req.as_ptr(), &mut out);
        assert_eq!(code, abi::TS_OK, "{}", last_error());
        assert_eq!(take(out)["texts"][0].as_array().unwrap().len(), 128);

        // --- Parakeet: the pinned audio profile installs and transcribes.
        let mut out: *mut c_char = ptr::null_mut();
        assert_eq!(
            ts_audio_catalog_json(&mut out),
            abi::TS_OK,
            "{}",
            last_error()
        );
        let profiles = take(out);
        let profile = profiles
            .as_array()
            .unwrap()
            .iter()
            .find(|p| p["identity"]["alias"] == "parakeet-redux-coreml")
            .expect("pinned parakeet profile");
        assert_eq!(profile["capabilities"]["can_run"], true);
        let identity = c(&profile["identity"].to_string());
        let mut out: *mut c_char = ptr::null_mut();
        let code = ts_audio_install_with_options(
            identity.as_ptr(),
            ptr::null(),
            Some(no_progress),
            ptr::null_mut(),
            &mut out,
        );
        assert_eq!(code, abi::TS_OK, "{}", last_error());
        let row = take(out);
        let path = row["path"].as_str().unwrap().to_string();
        eprintln!("parakeet installed at {path}");

        // Opening a managed install re-verifies it, then checks the family.
        let cpath = c(&path);
        let options = c(r#"{"task": "speech_to_text", "expected_family": "parakeet_tdt"}"#);
        let mut session = ptr::null_mut();
        let code = ts_audio_session_open(cpath.as_ptr(), options.as_ptr(), &mut session);
        assert_eq!(code, abi::TS_OK, "{}", last_error());
        let pcm = say_16k("The quick brown fox jumps over the lazy dog.", &home);
        let request = c(r#"{"task": "speech_to_text"}"#);
        let mut job = ptr::null_mut();
        assert_eq!(
            ts_audio_job_open(session, request.as_ptr(), &mut job),
            abi::TS_OK
        );
        for chunk in pcm.chunks(32_000) {
            assert_eq!(
                ts_audio_job_append(job, chunk.as_ptr(), chunk.len()),
                abi::TS_OK,
                "{}",
                last_error()
            );
        }
        let mut out: *mut c_char = ptr::null_mut();
        let code = ts_audio_job_run(job, None, ptr::null_mut(), &mut out);
        assert_eq!(code, abi::TS_OK, "{}", last_error());
        let result = take(out);
        let text = result["transcript"]["text"].as_str().unwrap();
        eprintln!("parakeet transcript: {text}");
        assert_eq!(text, "The quick brown fox jumps over the lazy dog.");
        assert_eq!(result["transcript"]["segments"][0]["timing"], "token");
        assert_eq!(ts_audio_job_close(job), abi::TS_OK);
        assert_eq!(ts_audio_session_close(session), abi::TS_OK);
    }
    std::fs::remove_dir_all(&home).unwrap();
}
