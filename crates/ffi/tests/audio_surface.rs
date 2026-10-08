use std::{
    ffi::{CStr, CString},
    ptr,
};
use turbospark_ffi::*;

// Model opening refuses busy admission; these tests intentionally own that gate.
static HEAVY_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn last_error_text() -> String {
    let mut buf = vec![0u8; 1024];
    let n = unsafe { ts_last_error(buf.as_mut_ptr().cast(), buf.len()) };
    String::from_utf8_lossy(&buf[..n.min(buf.len())]).into_owned()
}
#[test]
fn catalog_and_unknown_identity_are_guarded() {
    unsafe {
        assert_ne!(ts_audio_catalog_json(ptr::null_mut()), 0);
        let mut out = ptr::null_mut();
        assert_eq!(ts_audio_catalog_json(&mut out), 0);
        let rows: serde_json::Value =
            serde_json::from_str(CStr::from_ptr(out).to_str().unwrap()).unwrap();
        ts_string_free(out);
        let rows = rows.as_array().unwrap();
        assert!(!rows.is_empty());
        for row in rows {
            assert!(row["capabilities"]["can_run"].is_boolean());
            assert!(row["capabilities"]["cancellation"].is_string());
        }
        let mut stale = rows[0]["identity"].clone();
        stale["revision"] = "0000000000000000000000000000000000000000".into();
        let json = CString::new(stale.to_string()).unwrap();
        let mut out = ptr::dangling_mut();
        assert_ne!(ts_audio_resolve(json.as_ptr(), &mut out), 0);
        assert!(out.is_null());
    }
}
#[cfg(target_os = "macos")]
#[test]
fn failed_open_clears_handle_and_unknown_task_refuses() {
    let _serial = HEAVY_TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    unsafe {
        let mut out = ptr::dangling_mut();
        let path = CString::new("/missing/audio/model").unwrap();
        let options = CString::new(r#"{"task":"codec"}"#).unwrap();
        assert_ne!(
            ts_audio_session_open(path.as_ptr(), options.as_ptr(), &mut out),
            0
        );
        assert!(out.is_null());
        assert_eq!(ts_audio_job_cancel(ptr::null()), 0);
        assert_eq!(ts_audio_job_close(ptr::null_mut()), 0);
        assert_eq!(ts_audio_session_close(ptr::null_mut()), 0);
    }
}

#[test]
fn legacy_delete_is_guarded_and_refuses_unknown_identities() {
    unsafe {
        assert_ne!(ts_audio_delete_legacy(ptr::null()), 0, "null identity");
        let bogus = CString::new(
            r#"{"task":"music","alias":"nope","repository":"o/r","revision":"0","assetFingerprint":"f"}"#,
        )
        .unwrap();
        assert_ne!(
            ts_audio_delete_legacy(bogus.as_ptr()),
            0,
            "an identity not in the catalog must be refused, not silently succeed"
        );
    }
}

/// Pins the task names `ts_audio_session_open` accepts, because Swift's
/// `AudioTask.isRunnable` mirrors them and nothing else would catch drift. A
/// missing model path fails every open, so the assertion is on WHY: an
/// unaccepted task fails at deserialization ("unknown variant"), an accepted
/// one gets past it and fails on the path.
/// Opening while a host workload holds the device gate is BUSY, which a host
/// can retry, and is not an `OPEN` failure it should report as a broken model.
#[cfg(target_os = "macos")]
#[test]
fn opening_under_a_held_device_gate_is_busy_not_open() {
    let _serial = HEAVY_TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    unsafe {
        let mut permit = ptr::null_mut();
        for _ in 0..1000 {
            assert_eq!(ts_heavy_work_try_acquire(&mut permit), 0);
            if !permit.is_null() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        assert!(!permit.is_null(), "could not take the device gate");

        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../audio/testdata/minimax_music3/converted_plain");
        let path = CString::new(path.to_str().unwrap()).unwrap();
        let options = CString::new(r#"{"task":"music"}"#).unwrap();
        let mut session = ptr::null_mut();
        let status = ts_audio_session_open(path.as_ptr(), options.as_ptr(), &mut session);
        let why = last_error_text();
        assert_eq!(ts_heavy_work_release(permit), 0);
        assert_eq!(status, abi::TS_ERR_BUSY, "got {status}: {why}");
        assert!(session.is_null());
        assert!(why.contains("busy"), "{why}");
    }
}

/// A stop requested before the open begins is honoured immediately, with the
/// cancel code and no handle, and without touching the model path.
#[cfg(target_os = "macos")]
#[test]
fn a_cancelled_open_token_stops_the_open_with_the_cancel_code() {
    let _serial = HEAVY_TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    unsafe {
        let mut token = ptr::null_mut();
        assert_eq!(ts_audio_open_token_new(&mut token), 0);
        assert!(!token.is_null());
        assert_eq!(ts_audio_open_token_cancel(token), 0);

        let path = CString::new("/missing/audio/model").unwrap();
        let options = CString::new(r#"{"task":"music"}"#).unwrap();
        let mut session = ptr::dangling_mut();
        let status = ts_audio_session_open_cancellable(
            path.as_ptr(),
            options.as_ptr(),
            token,
            None,
            ptr::null_mut(),
            &mut session,
        );
        assert_eq!(status, abi::TS_ERR_CANCELLED, "{}", last_error_text());
        assert!(session.is_null(), "a stopped open leaves the handle NULL");

        // Null-safe, and idempotent to cancel twice.
        assert_eq!(ts_audio_open_token_cancel(token), 0);
        assert_eq!(ts_audio_open_token_cancel(ptr::null()), 0);
        assert_eq!(ts_audio_open_token_free(token), 0);
        assert_eq!(ts_audio_open_token_free(ptr::null_mut()), 0);
        assert_ne!(ts_audio_open_token_new(ptr::null_mut()), 0);
    }
}

/// An uncancelled token (and no token at all) opens exactly as
/// `ts_audio_session_open` does. The fixture folder is not a managed install,
/// so there is nothing to verify and the progress callback is never called.
#[cfg(target_os = "macos")]
#[test]
fn an_uncancelled_open_succeeds_and_reports_no_progress_for_an_unmanaged_folder() {
    let _serial = HEAVY_TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    static CALLS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    unsafe extern "C" fn count(
        _ud: *mut std::os::raw::c_void,
        _kind: std::os::raw::c_int,
        _json: *const std::os::raw::c_char,
        _len: usize,
        _pcm: *const f32,
        _n: usize,
    ) {
        CALLS.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    }
    unsafe {
        let mut token = ptr::null_mut();
        assert_eq!(ts_audio_open_token_new(&mut token), 0);
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../audio/testdata/minimax_music3/converted_plain");
        let path = CString::new(path.to_str().unwrap()).unwrap();
        let options = CString::new(r#"{"task":"music"}"#).unwrap();
        let mut session = ptr::null_mut();
        let status = ts_audio_session_open_cancellable(
            path.as_ptr(),
            options.as_ptr(),
            token,
            Some(count),
            ptr::null_mut(),
            &mut session,
        );
        assert_eq!(status, 0, "{}", last_error_text());
        assert!(!session.is_null());
        assert_eq!(CALLS.load(std::sync::atomic::Ordering::SeqCst), 0);
        assert_eq!(ts_audio_session_close(session), 0);

        // And with no token and no callback at all.
        let mut again = ptr::null_mut();
        let status = ts_audio_session_open_cancellable(
            path.as_ptr(),
            options.as_ptr(),
            ptr::null(),
            None,
            ptr::null_mut(),
            &mut again,
        );
        assert_eq!(status, 0, "{}", last_error_text());
        assert_eq!(ts_audio_session_close(again), 0);
        assert_eq!(ts_audio_open_token_free(token), 0);
    }
}

#[cfg(target_os = "macos")]
#[test]
fn open_accepts_exactly_the_runnable_task_names() {
    // Opens go through busy admission, which the native-job test also needs.
    let _serial = HEAVY_TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let reason = |task: &str| -> String {
        unsafe {
            let mut out = ptr::dangling_mut();
            let path = CString::new("/missing/audio/model").unwrap();
            let options = CString::new(format!(r#"{{"task":"{task}"}}"#)).unwrap();
            assert_ne!(
                ts_audio_session_open(path.as_ptr(), options.as_ptr(), &mut out),
                0
            );
            last_error_text()
        }
    };
    for task in ["speech_to_text", "text_to_speech", "music"] {
        let why = reason(task);
        assert!(
            !why.contains("unknown variant"),
            "{task} must be accepted, got: {why}"
        );
    }
    for task in [
        "enhancement",
        "separation",
        "alignment",
        "diarization",
        "speech_detection",
        "codec",
        "language_identification",
    ] {
        let why = reason(task);
        assert!(
            why.contains("unknown variant"),
            "{task} must be refused as an unknown task, got: {why}"
        );
    }
}

#[cfg(target_os = "macos")]
#[test]
fn native_job_ownership_pcm_copy_and_cancel() {
    let _serial = HEAVY_TEST_LOCK.lock().unwrap();
    use std::os::raw::{c_char, c_int, c_void};
    struct Sink {
        samples: Vec<f32>,
        largest: usize,
        cancel: *const TsAudioJob,
    }
    unsafe extern "C" fn callback(
        data: *mut c_void,
        kind: c_int,
        _json: *const c_char,
        _len: usize,
        pcm: *const f32,
        count: usize,
    ) {
        let sink = &mut *data.cast::<Sink>();
        if kind == 1 && !sink.cancel.is_null() {
            assert_ne!(
                ts_audio_job_close(sink.cancel.cast_mut()),
                0,
                "a running callback cannot free its job"
            );
            assert_eq!(ts_audio_job_cancel(sink.cancel), 0);
        }
        if kind == 2 {
            sink.largest = sink.largest.max(count);
            sink.samples
                .extend_from_slice(std::slice::from_raw_parts(pcm, count));
        }
    }
    unsafe {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../audio/testdata/minimax_music3/converted_plain");
        let path = CString::new(path.to_str().unwrap()).unwrap();
        let options = CString::new(r#"{"task":"music"}"#).unwrap();
        let mut session = ptr::null_mut();
        let status = ts_audio_session_open(path.as_ptr(), options.as_ptr(), &mut session);
        assert_eq!(status, 0, "open failed: {}", last_error_text());
        let request=CString::new(r#"{"task":"music","caption":"piano","lyrics":"[instrumental]","duration_seconds":0.12,"steps":1,"seed":null}"#).unwrap();
        let mut job = ptr::null_mut();
        assert_eq!(ts_audio_job_open(session, request.as_ptr(), &mut job), 0);
        assert_ne!(
            ts_audio_session_close(session),
            0,
            "model cannot close under a job"
        );
        let nan = f32::NAN;
        assert_ne!(ts_audio_job_append(job, &nan, 1), 0);
        let mut sink = Sink {
            samples: vec![],
            largest: 0,
            cancel: ptr::null(),
        };
        let mut result = ptr::null_mut();
        assert_eq!(
            ts_audio_job_run(
                job,
                Some(callback),
                (&mut sink as *mut Sink).cast(),
                &mut result
            ),
            0
        );
        let wire: serde_json::Value =
            serde_json::from_str(CStr::from_ptr(result).to_str().unwrap()).unwrap();
        ts_string_free(result);
        assert_eq!(
            wire["seed"], 0,
            "native defaults must return the resolved seed"
        );
        assert!(!sink.samples.is_empty());
        assert!(sink.samples.iter().all(|v| v.is_finite()));
        assert!(sink.largest <= 32000);
        assert_eq!(
            wire["sample_count"].as_u64().unwrap() as usize,
            sink.samples.len()
        );
        assert_ne!(
            ts_audio_job_run(job, None, ptr::null_mut(), &mut result),
            0,
            "run exactly once"
        );
        assert_eq!(ts_audio_job_status_json(job, &mut result), 0);
        assert!(CStr::from_ptr(result)
            .to_str()
            .unwrap()
            .contains("succeeded"));
        ts_string_free(result);
        assert_eq!(ts_audio_job_close(job), 0);
        assert_eq!(ts_audio_job_open(session, request.as_ptr(), &mut job), 0);
        sink.cancel = job;
        assert_eq!(
            ts_audio_job_run(
                job,
                Some(callback),
                (&mut sink as *mut Sink).cast(),
                &mut result
            ),
            abi::TS_ERR_CANCELLED,
            "a job the caller cancelled is cancelled, not a generation failure"
        );
        assert_eq!(ts_audio_job_status_json(job, &mut result), 0);
        assert!(CStr::from_ptr(result)
            .to_str()
            .unwrap()
            .contains("cancelled"));
        ts_string_free(result);
        assert_eq!(ts_audio_job_close(job), 0);
        assert_eq!(ts_audio_session_close(session), 0);
    }
}

#[test]
fn external_heavy_permit_serializes_and_releases() {
    let _serial = HEAVY_TEST_LOCK.lock().unwrap();
    unsafe {
        assert_ne!(ts_heavy_work_try_acquire(ptr::null_mut()), 0);
        let mut first = ptr::null_mut();
        // Other tests can own the process-wide device gate briefly.
        for _ in 0..1000 {
            assert_eq!(ts_heavy_work_try_acquire(&mut first), 0);
            if !first.is_null() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        assert!(!first.is_null());
        let mut second = ptr::dangling_mut();
        assert_eq!(ts_heavy_work_try_acquire(&mut second), 0);
        assert!(second.is_null(), "two host workloads cannot hold the gate");
        assert_eq!(ts_heavy_work_release(first), 0);
        assert_eq!(ts_heavy_work_release(ptr::null_mut()), 0);
    }
}
