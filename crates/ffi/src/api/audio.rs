//! Additive audio catalog and worker/job ABI. Callback buffers are borrowed.
use crate::abi::{self, guard_result};
use crate::strings;
use catalog::audio_catalog::{AudioCatalog, AudioProfileIdentity};
use std::os::raw::{c_char, c_int, c_void};

type AbiResult<T> = Result<T, (c_int, String)>;
fn invalid(e: impl ToString) -> (c_int, String) {
    (abi::TS_ERR_INVALID_ARGUMENT, e.to_string())
}
unsafe fn identity(json: *const c_char) -> AbiResult<AudioProfileIdentity> {
    serde_json::from_str(strings::required(json, "identity").map_err(invalid)?).map_err(invalid)
}
unsafe fn clear_output(out: *mut *mut c_char) -> AbiResult<()> {
    if out.is_null() {
        return Err(invalid("out must not be null"));
    }
    *out = std::ptr::null_mut();
    Ok(())
}
unsafe fn output<T: serde::Serialize>(value: &T, out: *mut *mut c_char) -> AbiResult<()> {
    strings::emit(&serde_json::to_string(value).map_err(invalid)?, out).map_err(invalid)
}

#[no_mangle]
pub unsafe extern "C" fn ts_audio_catalog_json(out: *mut *mut c_char) -> c_int {
    guard_result(|| {
        clear_output(out)?;
        output(
            &AudioCatalog::embedded()
                .map_err(invalid)?
                .entries()
                .collect::<Vec<_>>(),
            out,
        )
    })
}
#[no_mangle]
pub unsafe extern "C" fn ts_audio_installed_json(out: *mut *mut c_char) -> c_int {
    guard_result(|| {
        clear_output(out)?;
        output(
            &AudioCatalog::embedded()
                .map_err(invalid)?
                .installed_report(&catalog::Store::default_store().map_err(invalid)?),
            out,
        )
    })
}
#[no_mangle]
pub unsafe extern "C" fn ts_audio_resolve(
    identity_json: *const c_char,
    out: *mut *mut c_char,
) -> c_int {
    guard_result(|| {
        clear_output(out)?;
        let identity = identity(identity_json)?;
        let path = AudioCatalog::embedded()
            .map_err(invalid)?
            .resolve(
                &catalog::Store::default_store().map_err(invalid)?,
                &identity,
            )
            .map_err(invalid)?;
        strings::emit(&path.to_string_lossy(), out).map_err(invalid)
    })
}
#[no_mangle]
pub unsafe extern "C" fn ts_audio_delete(identity_json: *const c_char) -> c_int {
    guard_result(|| {
        AudioCatalog::embedded()
            .map_err(invalid)?
            .delete(
                &catalog::Store::default_store().map_err(invalid)?,
                &identity(identity_json)?,
            )
            .map_err(invalid)
    })
}
#[no_mangle]
pub unsafe extern "C" fn ts_audio_adopt(
    identity_json: *const c_char,
    out: *mut *mut c_char,
) -> c_int {
    guard_result(|| {
        clear_output(out)?;
        let row = AudioCatalog::embedded()
            .map_err(invalid)?
            .adopt_legacy(
                &catalog::Store::default_store().map_err(invalid)?,
                &identity(identity_json)?,
                &catalog::CancelFlag::new(),
            )
            .map_err(invalid)?;
        output(&row, out)
    })
}
/// Uses the same owner/pause/cancel registry as existing model installations.
#[no_mangle]
pub unsafe extern "C" fn ts_audio_install(
    identity_json: *const c_char,
    callback: crate::TsInstallCallback,
    userdata: *mut c_void,
    out: *mut *mut c_char,
) -> c_int {
    guard_result(|| {
        if out.is_null() {
            return Err(invalid("out must not be null"));
        }
        *out = std::ptr::null_mut();
        let active = crate::models::ActiveInstall::register();
        let identity = identity(identity_json)?;
        let row = AudioCatalog::embedded()
            .map_err(invalid)?
            .install(
                &catalog::Client::new(),
                &catalog::Catalog::embedded().map_err(invalid)?,
                &catalog::Store::default_store().map_err(invalid)?,
                &identity,
                &mut |progress| {
                    if let Some(cb) = callback {
                        let stage = progress
                            .current_path
                            .as_deref()
                            .unwrap_or("audio download")
                            .to_string();
                        cb(
                            userdata,
                            crate::TS_INSTALL_BYTES,
                            stage.as_ptr().cast(),
                            stage.len(),
                            progress.completed_bytes,
                            progress.total_bytes,
                        );
                    }
                },
                active.cancel_flag(),
            )
            .map_err(invalid)?;
        output(&row, out)
    })
}

#[cfg(target_os = "macos")]
mod native {
    use super::*;
    use runtime::native_audio::{
        validate_pcm, AudioCancel, AudioEvent, AudioRequest, AudioSession, AudioTask,
    };
    use std::sync::{
        atomic::{AtomicU8, Ordering},
        Arc, Mutex,
    };
    pub struct TsAudioSession {
        session: Arc<AudioSession>,
    }
    struct JobState {
        samples: Vec<f32>,
        started: bool,
    }
    pub struct TsAudioJob {
        session: Arc<AudioSession>,
        request: AudioRequest,
        cancel: AudioCancel,
        state: Mutex<JobState>,
        status: AtomicU8,
    }
    #[derive(serde::Deserialize)]
    #[serde(deny_unknown_fields)]
    struct OpenOptions {
        task: AudioTask,
        #[serde(default)]
        allow_portable: bool,
        #[serde(default)]
        allow_experimental_metal: bool,
        #[serde(default)]
        expected_family: Option<String>,
    }
    pub type TsAudioCallback =
        Option<unsafe extern "C" fn(*mut c_void, c_int, *const c_char, usize, *const f32, usize)>;
    #[no_mangle]
    pub unsafe extern "C" fn ts_audio_session_open(
        path: *const c_char,
        options: *const c_char,
        out: *mut *mut TsAudioSession,
    ) -> c_int {
        guard_result(|| {
            if out.is_null() {
                return Err(invalid("out must not be null"));
            }
            *out = std::ptr::null_mut();
            let options: OpenOptions =
                serde_json::from_str(strings::required(options, "options").map_err(invalid)?)
                    .map_err(invalid)?;
            let path = strings::required(path, "path").map_err(invalid)?;
            let store = catalog::Store::default_store().map_err(invalid)?;
            let model_path = std::path::Path::new(path);
            if model_path.starts_with(store.audio_models_dir()) {
                let catalog = AudioCatalog::embedded().map_err(invalid)?;
                let profile = catalog
                    .entries()
                    .find(|profile| store.audio_install_path(&profile.identity.alias) == model_path)
                    .ok_or_else(|| invalid("managed audio model is not in the pinned catalog"))?;
                catalog.verify(&store, &profile.identity).map_err(invalid)?;
            }
            let session = AudioSession::open_with_options(
                path.into(),
                options.task,
                options.allow_portable,
                options.allow_experimental_metal,
            )
            .map_err(|e| (abi::TS_ERR_OPEN, e))?;
            if let Some(expected) = options.expected_family.as_deref() {
                if session.family() != expected {
                    return Err(invalid(format!(
                        "selected audio family {expected} does not match loaded family {}",
                        session.family()
                    )));
                }
            }
            *out = Box::into_raw(Box::new(TsAudioSession {
                session: Arc::new(session),
            }));
            Ok(())
        })
    }
    /// Refuses close while jobs reference the session. Close the job after run returns.
    #[no_mangle]
    pub unsafe extern "C" fn ts_audio_session_close(handle: *mut TsAudioSession) -> c_int {
        guard_result(|| {
            if let Some(session) = handle.as_ref() {
                if Arc::strong_count(&session.session) > 1 {
                    return Err(invalid("close all audio jobs before closing the session"));
                }
                drop(Box::from_raw(handle));
            }
            Ok(())
        })
    }
    #[no_mangle]
    pub unsafe extern "C" fn ts_audio_session_cancel(handle: *const TsAudioSession) -> c_int {
        guard_result(|| {
            if let Some(s) = handle.as_ref() {
                s.session.cancel();
            }
            Ok(())
        })
    }
    #[no_mangle]
    pub unsafe extern "C" fn ts_audio_job_open(
        handle: *const TsAudioSession,
        request: *const c_char,
        out: *mut *mut TsAudioJob,
    ) -> c_int {
        guard_result(|| {
            if out.is_null() {
                return Err(invalid("out must not be null"));
            }
            *out = std::ptr::null_mut();
            let session = handle
                .as_ref()
                .ok_or_else(|| invalid("session must not be null"))?;
            let request: AudioRequest =
                serde_json::from_str(strings::required(request, "request").map_err(invalid)?)
                    .map_err(invalid)?;
            *out = Box::into_raw(Box::new(TsAudioJob {
                session: session.session.clone(),
                request,
                cancel: AudioCancel::default(),
                status: AtomicU8::new(0),
                state: Mutex::new(JobState {
                    samples: Vec::new(),
                    started: false,
                }),
            }));
            Ok(())
        })
    }
    /// Appends 16 kHz mono f32 samples, 1..32000 per call, at most 30 minutes total.
    #[no_mangle]
    pub unsafe extern "C" fn ts_audio_job_append(
        handle: *const TsAudioJob,
        pcm: *const f32,
        count: usize,
    ) -> c_int {
        guard_result(|| {
            let job = handle
                .as_ref()
                .ok_or_else(|| invalid("job must not be null"))?;
            if pcm.is_null() || count == 0 || count > runtime::native_audio::MAX_PCM_CHUNK {
                return Err(invalid("invalid PCM chunk pointer or sample count"));
            }
            let samples = std::slice::from_raw_parts(pcm, count);
            let mut state = job
                .state
                .lock()
                .map_err(|_| invalid("audio job state poisoned"))?;
            if state.started {
                return Err(invalid("audio job already started"));
            }
            job.cancel.checkpoint().map_err(invalid)?;
            validate_pcm(samples, state.samples.len()).map_err(invalid)?;
            state.samples.extend_from_slice(samples);
            Ok(())
        })
    }
    /// Synchronous drain. kind 1 is UTF-8 progress JSON; kind 2 is borrowed PCM.
    #[no_mangle]
    pub unsafe extern "C" fn ts_audio_job_run(
        handle: *const TsAudioJob,
        callback: TsAudioCallback,
        userdata: *mut c_void,
        out: *mut *mut c_char,
    ) -> c_int {
        guard_result(|| {
            if out.is_null() {
                return Err(invalid("out must not be null"));
            }
            *out = std::ptr::null_mut();
            let job = handle
                .as_ref()
                .ok_or_else(|| invalid("job must not be null"))?;
            let samples = {
                let mut state = job
                    .state
                    .lock()
                    .map_err(|_| invalid("audio job state poisoned"))?;
                if state.started {
                    return Err(invalid("audio job already started"));
                }
                state.started = true;
                job.status.store(1, Ordering::Release);
                std::mem::take(&mut state.samples)
            };
            let result =
                job.session
                    .execute(job.request.clone(), samples, job.cancel.clone(), |event| {
                        if let Some(cb) = callback {
                            match event {
                                AudioEvent::Progress(progress) => {
                                    if let Ok(json) = serde_json::to_string(&progress) {
                                        cb(
                                            userdata,
                                            1,
                                            json.as_ptr().cast(),
                                            json.len(),
                                            std::ptr::null(),
                                            0,
                                        );
                                    }
                                }
                                AudioEvent::Pcm(pcm) => {
                                    cb(userdata, 2, std::ptr::null(), 0, pcm.as_ptr(), pcm.len())
                                }
                            }
                        }
                    });
            job.status.store(
                if result.is_ok() {
                    2
                } else if job.cancel.is_cancelled() {
                    3
                } else {
                    4
                },
                Ordering::Release,
            );
            output(&result.map_err(|e| (abi::TS_ERR_GENERATE, e))?, out)
        })
    }
    /// Independent of the job's operation mutex and worker. Safe during `run`.
    #[no_mangle]
    pub unsafe extern "C" fn ts_audio_job_cancel(handle: *const TsAudioJob) -> c_int {
        guard_result(|| {
            if let Some(job) = handle.as_ref() {
                job.cancel.cancel();
            }
            Ok(())
        })
    }
    #[no_mangle]
    pub unsafe extern "C" fn ts_audio_job_status_json(
        handle: *const TsAudioJob,
        out: *mut *mut c_char,
    ) -> c_int {
        guard_result(|| {
            clear_output(out)?;
            let job = handle
                .as_ref()
                .ok_or_else(|| invalid("job must not be null"))?;
            let state = match job.status.load(Ordering::Acquire) {
                0 if job.cancel.is_cancelled() => "cancelled",
                0 => "created",
                1 => "running",
                2 => "succeeded",
                3 => "cancelled",
                _ => "failed",
            };
            output(
                &serde_json::json!({"state":state,"cancellation_requested":job.cancel.is_cancelled()}),
                out,
            )
        })
    }
    /// Caller must first wait for run to return; closing cannot race any job call.
    #[no_mangle]
    pub unsafe extern "C" fn ts_audio_job_close(handle: *mut TsAudioJob) -> c_int {
        guard_result(|| {
            if let Some(job) = handle.as_ref() {
                if job.status.load(Ordering::Acquire) == 1 {
                    return Err(invalid("cancel and drain the running job before closing"));
                }
                drop(Box::from_raw(handle));
            }
            Ok(())
        })
    }
}
#[cfg(target_os = "macos")]
pub use native::*;

#[cfg(not(target_os = "macos"))]
mod unsupported {
    use super::*;
    type TsAudioSession = c_void;
    type TsAudioJob = c_void;
    type TsAudioCallback =
        Option<unsafe extern "C" fn(*mut c_void, c_int, *const c_char, usize, *const f32, usize)>;
    macro_rules! refused {($name:ident($($arg:ident:$ty:ty),*) $(,$out:ident)?)=>{
 #[no_mangle] pub unsafe extern "C" fn $name($($arg:$ty),*)->c_int {guard_result(||{let _=($($arg),*);$(if !$out.is_null(){*$out=std::ptr::null_mut();})?Err((abi::TS_ERR_UNSUPPORTED,"native audio inference requires macOS".into()))})}
 };}
    refused!(ts_audio_session_open(path:*const c_char,options:*const c_char,out:*mut *mut TsAudioSession),out);
    refused!(ts_audio_session_close(handle:*mut TsAudioSession));
    refused!(ts_audio_session_cancel(handle:*const TsAudioSession));
    refused!(ts_audio_job_open(handle:*const TsAudioSession,request:*const c_char,out:*mut *mut TsAudioJob),out);
    refused!(ts_audio_job_append(handle:*const TsAudioJob,pcm:*const f32,count:usize));
    refused!(ts_audio_job_run(handle:*const TsAudioJob,callback:TsAudioCallback,userdata:*mut c_void,out:*mut *mut c_char),out);
    refused!(ts_audio_job_status_json(handle:*const TsAudioJob,out:*mut *mut c_char),out);
    refused!(ts_audio_job_cancel(handle:*const TsAudioJob));
    refused!(ts_audio_job_close(handle:*mut TsAudioJob));
}
#[cfg(not(target_os = "macos"))]
pub use unsupported::*;

#[no_mangle]
pub unsafe extern "C" fn ts_audio_capabilities_json(out: *mut *mut c_char) -> c_int {
    guard_result(|| {
        clear_output(out)?;
        output(
            &catalog::audio_catalog::family_capabilities().map_err(invalid)?,
            out,
        )
    })
}

/// Host-owned permit for heavyweight work implemented outside Rust (such as MLX).
pub struct TsHeavyWorkPermit {
    _guard: runtime::heavy::HeavyWorkGuard,
}
#[no_mangle]
pub unsafe extern "C" fn ts_heavy_work_try_acquire(out: *mut *mut TsHeavyWorkPermit) -> c_int {
    guard_result(|| {
        if out.is_null() {
            return Err(invalid("out must not be null"));
        }
        *out = std::ptr::null_mut();
        if let Some(guard) = runtime::heavy::HeavyWorkGuard::try_acquire() {
            *out = Box::into_raw(Box::new(TsHeavyWorkPermit { _guard: guard }));
        }
        Ok(())
    })
}
#[no_mangle]
pub unsafe extern "C" fn ts_heavy_work_release(permit: *mut TsHeavyWorkPermit) -> c_int {
    guard_result(|| {
        if !permit.is_null() {
            drop(Box::from_raw(permit));
        }
        Ok(())
    })
}
