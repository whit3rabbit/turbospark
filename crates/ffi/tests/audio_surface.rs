use std::{
    ffi::{CStr, CString},
    ptr,
};
use turbospark_ffi::*;

// Model opening refuses busy admission; these tests intentionally own that gate.
static HEAVY_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
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
        assert_eq!(
            ts_audio_session_open(path.as_ptr(), options.as_ptr(), &mut session),
            0
        );
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
        assert_ne!(
            ts_audio_job_run(
                job,
                Some(callback),
                (&mut sink as *mut Sink).cast(),
                &mut result
            ),
            0
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
