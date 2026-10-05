//! The `ts_audio_*` entry points, driven through the same bodies the header
//! exposes. Portable and model-free: fixtures are WAVs this test writes, and
//! the session paths use a fake model through `AudioSession::with_model`,
//! since every real `ts_audio_session_open` is refused today.

use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::path::PathBuf;
use std::ptr;

use turbospark_audio::speech::{
    SpeechToText, SynthesizeOptions, SynthesizedAudio, TextToSpeech, TranscribeOptions, Transcript,
    NO_AUDIO_MODEL_REASON,
};
use turbospark_audio::wav::write_mono;
use turbospark_audio::{AudioError, AudioModel, WavSampleFormat};
use turbospark_ffi::{
    abi, ts_audio_capabilities_json, ts_audio_convert_json, ts_audio_display_level,
    ts_audio_peaks_json, ts_audio_probe_json, ts_audio_session_close, ts_audio_session_open,
    ts_audio_synthesize_json, ts_audio_transcribe_json, ts_last_error, ts_string_free,
    AudioSession, TsAudioSession,
};

fn c(s: &str) -> CString {
    CString::new(s).unwrap()
}

unsafe fn take(out: *mut c_char) -> serde_json::Value {
    assert!(!out.is_null(), "a successful call must write its out-param");
    let s = CStr::from_ptr(out).to_str().unwrap().to_string();
    ts_string_free(out);
    serde_json::from_str(&s).expect("audio results are JSON")
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

struct Scratch(PathBuf);

impl Scratch {
    fn new(name: &str) -> Self {
        let dir = std::env::temp_dir().join(format!(
            "turbospark-ffi-audio-{name}-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        Self(dir)
    }

    fn path(&self, file: &str) -> PathBuf {
        self.0.join(file)
    }

    /// One second of 440 Hz at `rate`, mono, 16-bit.
    fn tone(&self, file: &str, rate: u32) -> PathBuf {
        let path = self.path(file);
        let samples: Vec<f32> = (0..rate)
            .map(|i| 0.5 * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / rate as f32).sin())
            .collect();
        write_mono(&path, &samples, rate, WavSampleFormat::Int16).unwrap();
        path
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn capabilities_report_decoders_and_refuse_every_model_task() {
    let mut out: *mut c_char = ptr::null_mut();
    assert_eq!(unsafe { ts_audio_capabilities_json(&mut out) }, abi::TS_OK);
    let caps = unsafe { take(out) };
    let decode: Vec<&str> = caps["decodeExtensions"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect();
    for ext in ["wav", "mp3", "m4a", "flac"] {
        assert!(decode.contains(&ext), "{ext} missing from {decode:?}");
    }
    for task in ["speechToText", "textToSpeech", "music"] {
        assert_eq!(caps[task]["active"], false, "{task}");
        assert_eq!(caps[task]["reason"], NO_AUDIO_MODEL_REASON, "{task}");
    }
}

#[test]
fn probe_and_peaks_answer_for_a_real_file() {
    let scratch = Scratch::new("probe");
    let path = c(scratch.tone("tone.wav", 22_050).to_str().unwrap());

    let mut out: *mut c_char = ptr::null_mut();
    assert_eq!(
        unsafe { ts_audio_probe_json(path.as_ptr(), &mut out) },
        abi::TS_OK
    );
    let probe = unsafe { take(out) };
    assert_eq!(probe["sampleRate"], 22_050);
    assert_eq!(probe["channels"], 1);
    assert_eq!(probe["durationSeconds"], 1.0);

    let mut out: *mut c_char = ptr::null_mut();
    assert_eq!(
        unsafe { ts_audio_peaks_json(path.as_ptr(), 40, &mut out) },
        abi::TS_OK
    );
    let peaks = unsafe { take(out) };
    assert_eq!(peaks["peaks"].as_array().unwrap().len(), 40);
}

#[test]
fn convert_writes_speech_shaped_wav_and_reports_it() {
    let scratch = Scratch::new("convert");
    let source = c(scratch.tone("tone.wav", 48_000).to_str().unwrap());
    let target_path = scratch.path("speech.wav");
    let target = c(target_path.to_str().unwrap());
    let options = c(r#"{"sampleRate":16000,"channels":1,"sampleFormat":"int16","endSeconds":0.5}"#);
    let mut out: *mut c_char = ptr::null_mut();
    let code = unsafe {
        ts_audio_convert_json(source.as_ptr(), target.as_ptr(), options.as_ptr(), &mut out)
    };
    assert_eq!(code, abi::TS_OK, "{}", last_error());
    let report = unsafe { take(out) };
    assert_eq!(report["sampleRate"], 16_000);
    assert_eq!(report["channels"], 1);
    assert_eq!(report["frames"], 8_000);
    assert!(target_path.exists());
}

#[test]
fn errors_map_to_codes_with_a_reason() {
    let scratch = Scratch::new("errors");
    let mut out: *mut c_char = ptr::null_mut();

    // Null out-param and null path are argument errors.
    assert_eq!(
        unsafe { ts_audio_probe_json(ptr::null(), ptr::null_mut()) },
        abi::TS_ERR_INVALID_ARGUMENT
    );
    assert_eq!(
        unsafe { ts_audio_probe_json(ptr::null(), &mut out) },
        abi::TS_ERR_INVALID_ARGUMENT
    );

    // A refused format is UNSUPPORTED and says to convert.
    let opus = scratch.path("voice.opus");
    std::fs::write(&opus, b"OggS").unwrap();
    let opus = c(opus.to_str().unwrap());
    assert_eq!(
        unsafe { ts_audio_probe_json(opus.as_ptr(), &mut out) },
        abi::TS_ERR_UNSUPPORTED
    );
    assert!(last_error().contains("convert"), "{}", last_error());
    assert!(out.is_null());

    // A missing file is OPEN.
    let missing = c(scratch.path("missing.wav").to_str().unwrap());
    assert_eq!(
        unsafe { ts_audio_probe_json(missing.as_ptr(), &mut out) },
        abi::TS_ERR_OPEN
    );

    // Malformed options JSON is JSON; an inverted range is an argument error.
    let source = c(scratch.tone("tone.wav", 8_000).to_str().unwrap());
    let target = c(scratch.path("out.wav").to_str().unwrap());
    let bad_json = c("{not json");
    assert_eq!(
        unsafe {
            ts_audio_convert_json(
                source.as_ptr(),
                target.as_ptr(),
                bad_json.as_ptr(),
                &mut out,
            )
        },
        abi::TS_ERR_JSON
    );
    let inverted = c(r#"{"startSeconds":0.8,"endSeconds":0.2}"#);
    assert_eq!(
        unsafe {
            ts_audio_convert_json(
                source.as_ptr(),
                target.as_ptr(),
                inverted.as_ptr(),
                &mut out,
            )
        },
        abi::TS_ERR_INVALID_ARGUMENT
    );

    // Zero buckets is an argument error.
    assert_eq!(
        unsafe { ts_audio_peaks_json(source.as_ptr(), 0, &mut out) },
        abi::TS_ERR_INVALID_ARGUMENT
    );
}

#[test]
fn display_level_is_safe_on_bad_input_and_maps_full_scale_to_one() {
    assert_eq!(unsafe { ts_audio_display_level(ptr::null(), 10) }, 0.0);
    let silence = [0.0f32; 64];
    assert_eq!(unsafe { ts_audio_display_level(silence.as_ptr(), 0) }, 0.0);
    assert_eq!(
        unsafe { ts_audio_display_level(silence.as_ptr(), silence.len()) },
        0.0
    );
    let square: Vec<f32> = (0..64)
        .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
        .collect();
    let level = unsafe { ts_audio_display_level(square.as_ptr(), square.len()) };
    assert!((level - 1.0).abs() < 1e-6, "{level}");
}

#[test]
fn every_session_open_is_refused_with_the_engine_reason() {
    let dir = c("/nonexistent/audio-model");
    // Non-null garbage, so the test proves the refusal CLEARS the handle.
    let mut session: *mut TsAudioSession = std::ptr::NonNull::dangling().as_ptr();
    let code = unsafe { ts_audio_session_open(dir.as_ptr(), &mut session) };
    assert_eq!(code, abi::TS_ERR_UNSUPPORTED);
    assert!(session.is_null(), "a refused open must clear the handle");
    assert_eq!(last_error(), NO_AUDIO_MODEL_REASON);
    // Closing null is a no-op, as for every other handle.
    unsafe { ts_audio_session_close(ptr::null_mut()) };
}

struct EchoStt;

impl SpeechToText for EchoStt {
    fn transcribe(
        &mut self,
        samples: &[f32],
        options: &TranscribeOptions,
    ) -> Result<Transcript, AudioError> {
        Ok(Transcript {
            text: format!("{} samples", samples.len()),
            language: options.language.clone(),
            segments: Vec::new(),
        })
    }
}

struct BeepTts;

impl TextToSpeech for BeepTts {
    fn synthesize(
        &mut self,
        text: &str,
        _options: &SynthesizeOptions,
    ) -> Result<SynthesizedAudio, AudioError> {
        Ok(SynthesizedAudio {
            samples: vec![0.25; text.len() * 100],
            sample_rate: 24_000,
        })
    }
}

#[test]
fn a_session_transcribes_at_16k_mono_and_synthesizes_to_wav() {
    let scratch = Scratch::new("session");
    let audio = c(scratch.tone("tone.wav", 44_100).to_str().unwrap());

    let stt = Box::into_raw(Box::new(AudioSession::with_model(
        AudioModel::SpeechToText(Box::new(EchoStt)),
    )));
    let options = c(r#"{"language":"en"}"#);
    let mut out: *mut c_char = ptr::null_mut();
    let code = unsafe { ts_audio_transcribe_json(stt, audio.as_ptr(), options.as_ptr(), &mut out) };
    assert_eq!(code, abi::TS_OK, "{}", last_error());
    let transcript = unsafe { take(out) };
    // One second at any source rate reaches the model as 16000 samples.
    assert_eq!(transcript["text"], "16000 samples");
    assert_eq!(transcript["language"], "en");

    // The wrong task on a session is an argument error, not a panic.
    let text = c("hello");
    let target = c(scratch.path("speech.wav").to_str().unwrap());
    let code = unsafe {
        ts_audio_synthesize_json(stt, text.as_ptr(), ptr::null(), target.as_ptr(), &mut out)
    };
    assert_eq!(code, abi::TS_ERR_INVALID_ARGUMENT);
    unsafe { ts_audio_session_close(stt) };

    let tts = Box::into_raw(Box::new(AudioSession::with_model(
        AudioModel::TextToSpeech(Box::new(BeepTts)),
    )));
    let code = unsafe {
        ts_audio_synthesize_json(tts, text.as_ptr(), ptr::null(), target.as_ptr(), &mut out)
    };
    assert_eq!(code, abi::TS_OK, "{}", last_error());
    let report = unsafe { take(out) };
    assert_eq!(report["sampleRate"], 24_000);
    assert_eq!(report["frames"], 500);
    let mut probe_out: *mut c_char = ptr::null_mut();
    assert_eq!(
        unsafe { ts_audio_probe_json(target.as_ptr(), &mut probe_out) },
        abi::TS_OK
    );
    assert_eq!(unsafe { take(probe_out) }["sampleRate"], 24_000);
    unsafe { ts_audio_session_close(tts) };
}
