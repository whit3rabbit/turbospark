//! C ABI for the portable audio engine (`turbospark-audio`).
//!
//! Two halves. The DSP calls (capabilities, probe, peaks, convert, display
//! level) are stateless and portable: they run on any target, and the app
//! uses them for every waveform, normalization and trim. The session calls
//! (`ts_audio_session_*`, transcribe, synthesize) are the speech model
//! contract; every open is refused today with the engine's reason, which the
//! capabilities call reports up front so a host never has to try.

use std::os::raw::{c_char, c_int};
use std::path::Path;

use turbospark_audio::speech::{SynthesizeOptions, TranscribeOptions};
use turbospark_audio::{AudioError, ConvertOptions};

use crate::abi::{self, guard_result, parse_json_or_default};
use crate::audio_session::AudioSession;
use crate::strings;
use crate::TsAudioSession;

/// Maps an engine error onto the ABI's status codes. The message is the
/// error's own `Display`, which already says what to do about it.
fn code_for(error: &AudioError) -> c_int {
    match error {
        AudioError::Io(_) | AudioError::Decode(_) => abi::TS_ERR_OPEN,
        AudioError::Unsupported(_) | AudioError::NeedsModel(_) => abi::TS_ERR_UNSUPPORTED,
        AudioError::InvalidOption(_) | AudioError::EmptyRange => abi::TS_ERR_INVALID_ARGUMENT,
    }
}

fn audio_err(error: AudioError) -> (c_int, String) {
    (code_for(&error), error.to_string())
}

fn emit_json<T: serde::Serialize>(value: &T, out: *mut *mut c_char) -> Result<(), (c_int, String)> {
    let json = serde_json::to_string(value).map_err(|e| (abi::TS_ERR_JSON, e.to_string()))?;
    // SAFETY: `out` was null-checked by every caller before any work ran.
    unsafe { strings::emit(&json, out) }.map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))
}

fn require_out<T>(out: *mut *mut T) -> Result<(), (c_int, String)> {
    if out.is_null() {
        return Err((abi::TS_ERR_INVALID_ARGUMENT, "out must not be null".into()));
    }
    Ok(())
}

/// Writes the engine's audio capability table as JSON:
/// `{"decodeExtensions":[...],"refusedExtensions":[...],
///   "speechToText":{"active":false,"reason":"..."},"textToSpeech":{...},"music":{...}}`.
#[no_mangle]
pub unsafe extern "C" fn ts_audio_capabilities_json(out: *mut *mut c_char) -> c_int {
    guard_result(|| {
        require_out(out)?;
        emit_json(&turbospark_audio::capabilities(), out)
    })
}

/// Probes an audio file: `{"sampleRate","channels","frames","durationSeconds","codec"}`.
#[no_mangle]
pub unsafe extern "C" fn ts_audio_probe_json(path: *const c_char, out: *mut *mut c_char) -> c_int {
    guard_result(|| {
        require_out(out)?;
        let path =
            strings::required(path, "path").map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))?;
        let report = turbospark_audio::probe(Path::new(path)).map_err(audio_err)?;
        emit_json(&report, out)
    })
}

/// Waveform peaks: `{"peaks":[0..1 x buckets],"durationSeconds","sampleRate","channels"}`.
/// `buckets` must be 1..=8192.
#[no_mangle]
pub unsafe extern "C" fn ts_audio_peaks_json(
    path: *const c_char,
    buckets: usize,
    out: *mut *mut c_char,
) -> c_int {
    guard_result(|| {
        require_out(out)?;
        let path =
            strings::required(path, "path").map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))?;
        let report = turbospark_audio::peaks(Path::new(path), buckets).map_err(audio_err)?;
        emit_json(&report, out)
    })
}

/// Converts any supported file to WAV. `options_json` (nullable) is
/// `{"sampleRate","channels","sampleFormat":"int16"|"float32","startSeconds","endSeconds"}`
/// with every key optional; speech-model input is
/// `{"sampleRate":16000,"channels":1,"sampleFormat":"int16"}`. Writes
/// `{"sampleRate","channels","frames","durationSeconds"}` to `out`.
#[no_mangle]
pub unsafe extern "C" fn ts_audio_convert_json(
    source: *const c_char,
    destination: *const c_char,
    options_json: *const c_char,
    out: *mut *mut c_char,
) -> c_int {
    guard_result(|| {
        require_out(out)?;
        let source =
            strings::required(source, "source").map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))?;
        let destination = strings::required(destination, "destination")
            .map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))?;
        let options = parse_json_or_default::<ConvertOptions>(options_json, "options")?;
        let report = turbospark_audio::convert(Path::new(source), Path::new(destination), &options)
            .map_err(audio_err)?;
        emit_json(&report, out)
    })
}

/// Meter level in 0...1 for `len` samples (RMS over -50...0 dBFS).
///
/// Safe to call from a real-time audio thread: no allocation, no lock, no
/// I/O. Returns 0 for a null pointer, a zero length, or a caught panic.
#[no_mangle]
pub unsafe extern "C" fn ts_audio_display_level(samples: *const f32, len: usize) -> f32 {
    abi::guard_value(0.0, || {
        if samples.is_null() || len == 0 || len > isize::MAX as usize / std::mem::size_of::<f32>() {
            return 0.0;
        }
        // SAFETY: the caller promises `len` readable floats at `samples`,
        // and the bound above keeps the byte length inside `isize`.
        let slice = std::slice::from_raw_parts(samples, len);
        turbospark_audio::display_level(turbospark_audio::rms(slice))
    })
}

/// Opens an audio model install. Refused today for every path with
/// `TS_ERR_UNSUPPORTED` and the engine's reason (see
/// `ts_audio_capabilities_json`).
#[no_mangle]
pub unsafe extern "C" fn ts_audio_session_open(
    model_dir: *const c_char,
    out: *mut *mut TsAudioSession,
) -> c_int {
    guard_result(|| {
        require_out(out)?;
        *out = std::ptr::null_mut();
        let dir = strings::required(model_dir, "modelDir")
            .map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))?;
        let session = AudioSession::open(dir).map_err(audio_err)?;
        *out = Box::into_raw(Box::new(session));
        Ok(())
    })
}

/// Closes an audio session. Not while a call on it is in flight.
#[no_mangle]
pub unsafe extern "C" fn ts_audio_session_close(ptr: *mut TsAudioSession) {
    if !ptr.is_null() {
        drop(Box::from_raw(ptr));
    }
}

/// Transcribes any supported audio file. `options_json` (nullable) is
/// `{"language","timestamps"}`. Writes
/// `{"text","language","segments":[{"startSeconds","endSeconds","text"}]}`.
#[no_mangle]
pub unsafe extern "C" fn ts_audio_transcribe_json(
    ptr: *const TsAudioSession,
    audio_path: *const c_char,
    options_json: *const c_char,
    out: *mut *mut c_char,
) -> c_int {
    guard_result(|| {
        require_out(out)?;
        let session = ptr.as_ref().ok_or((
            abi::TS_ERR_INVALID_ARGUMENT,
            "audio session must not be null".to_string(),
        ))?;
        let path = strings::required(audio_path, "audioPath")
            .map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))?;
        let options = parse_json_or_default::<TranscribeOptions>(options_json, "options")?;
        let transcript = session
            .transcribe(Path::new(path), &options)
            .map_err(audio_err)?;
        emit_json(&transcript, out)
    })
}

/// Synthesizes `text` to a float WAV at `destination`. `options_json`
/// (nullable) is `{"voice","rate"}`. Writes
/// `{"sampleRate","frames","durationSeconds"}`.
#[no_mangle]
pub unsafe extern "C" fn ts_audio_synthesize_json(
    ptr: *const TsAudioSession,
    text: *const c_char,
    options_json: *const c_char,
    destination: *const c_char,
    out: *mut *mut c_char,
) -> c_int {
    guard_result(|| {
        require_out(out)?;
        let session = ptr.as_ref().ok_or((
            abi::TS_ERR_INVALID_ARGUMENT,
            "audio session must not be null".to_string(),
        ))?;
        let text =
            strings::required(text, "text").map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))?;
        let destination = strings::required(destination, "destination")
            .map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))?;
        let options = parse_json_or_default::<SynthesizeOptions>(options_json, "options")?;
        let report = session
            .synthesize(text, &options, Path::new(destination))
            .map_err(audio_err)?;
        emit_json(&report, out)
    })
}
