//! Shared native audio execution. Model state never leaves its owning worker.
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    mpsc::{self, Receiver, SyncSender},
    Arc, Mutex,
};
use std::thread::JoinHandle;

pub const MAX_PCM_CHUNK: usize = 32_000;
pub const MAX_INPUT_SAMPLES: usize = 16_000 * 1800;
pub const MAX_OUTPUT_SAMPLES: usize = 44_100 * 2 * 360;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AudioTask {
    SpeechToText,
    TextToSpeech,
    Music,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AudioRequest {
    pub task: AudioTask,
    pub language: Option<String>,
    pub text: String,
    pub voice: String,
    pub speed: f32,
    pub caption: String,
    pub lyrics: String,
    pub duration_seconds: Option<f64>,
    pub steps: Option<usize>,
    pub seed: Option<u64>,
}
impl Default for AudioRequest {
    fn default() -> Self {
        Self {
            task: AudioTask::SpeechToText,
            language: None,
            text: String::new(),
            voice: "af_heart".into(),
            speed: 1.0,
            caption: String::new(),
            lyrics: String::new(),
            duration_seconds: Some(10.0),
            steps: Some(30),
            seed: Some(0),
        }
    }
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AudioSegment {
    pub start_seconds: Option<f64>,
    pub end_seconds: Option<f64>,
    pub text: String,
    pub timing: String,
    pub speaker: Option<String>,
    pub words: Option<Vec<AudioWord>>,
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AudioWord {
    pub text: String,
    pub start_seconds: f64,
    pub end_seconds: f64,
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AudioTranscript {
    pub text: String,
    pub language: Option<String>,
    pub segments: Vec<AudioSegment>,
    pub tags: Option<Vec<String>>,
    pub token_logprobs: Option<Vec<f32>>,
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AudioPcmFormat {
    pub sample_rate: u32,
    pub channels: u32,
    pub interleaved: bool,
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AudioResult {
    pub transcript: Option<AudioTranscript>,
    pub pcm_format: Option<AudioPcmFormat>,
    pub sample_count: usize,
    pub seed: Option<u64>,
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AudioProgress {
    pub stage: String,
    pub completed: usize,
    pub total: Option<usize>,
}
pub enum AudioEvent {
    Progress(AudioProgress),
    Pcm(Vec<f32>),
}

/// A job's cancel flag is independent of both the worker and result receiver.
#[derive(Clone, Default)]
pub struct AudioCancel(Arc<AtomicBool>);
impl AudioCancel {
    pub fn cancel(&self) {
        self.0.store(true, Ordering::Release);
    }
    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }
    pub fn checkpoint(&self) -> Result<(), String> {
        if self.is_cancelled() {
            Err("audio job cancelled".into())
        } else {
            Ok(())
        }
    }
}

pub fn validate_pcm(samples: &[f32], accumulated: usize) -> Result<(), String> {
    if samples.is_empty() || samples.len() > MAX_PCM_CHUNK {
        return Err("PCM chunks must contain 1 through 32000 samples".into());
    }
    if accumulated
        .checked_add(samples.len())
        .is_none_or(|n| n > MAX_INPUT_SAMPLES)
    {
        return Err("audio exceeds the 30-minute input limit".into());
    }
    if samples.iter().any(|v| !v.is_finite()) {
        return Err("PCM samples must be finite".into());
    }
    Ok(())
}

enum Message {
    Run {
        request: AudioRequest,
        samples: Vec<f32>,
        cancel: AudioCancel,
        events: SyncSender<AudioEvent>,
        reply: SyncSender<Result<AudioResult, String>>,
    },
}
/// Sendable commands only. The !Send engine is constructed and destroyed inside `worker`.
pub struct AudioSession {
    tx: Option<SyncSender<Message>>,
    worker: Option<JoinHandle<()>>,
    active: Mutex<Option<AudioCancel>>,
    task: AudioTask,
    family: String,
}
impl AudioSession {
    pub fn open(path: PathBuf, task: AudioTask, allow_portable: bool) -> Result<Self, String> {
        Self::open_with_options(path, task, allow_portable, false)
    }
    pub fn open_with_options(
        path: PathBuf,
        task: AudioTask,
        allow_portable: bool,
        allow_experimental_metal: bool,
    ) -> Result<Self, String> {
        let (tx, rx) = mpsc::sync_channel(1);
        let (ready, opened) = mpsc::sync_channel(1);
        let worker = std::thread::Builder::new()
            .name("native-audio".into())
            .spawn(move || {
                let engine = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    // No session handle exists yet to signal cancellation during teardown.
                    let _permit = crate::heavy::HeavyWorkGuard::try_acquire().ok_or(
                        "audio device is busy; retry opening the model after the current job finishes",
                    )?;
                    admit_memory(estimate_open_bytes(&path, task)?)?;
                    Engine::open(&path, task, allow_portable, allow_experimental_metal)
                }))
                .unwrap_or_else(|_| Err("audio model opening panicked".into()));
                match engine {
                    Ok(engine) => {
                        if ready.send(Ok(engine.family().to_string())).is_ok() {
                            run_worker(engine, rx);
                        }
                    }
                    Err(e) => {
                        let _ = ready.send(Err(e));
                    }
                }
            })
            .map_err(|e| e.to_string())?;
        let family = match opened
            .recv()
            .map_err(|_| "audio worker stopped while opening".to_string())?
        {
            Ok(family) => family,
            Err(e) => {
                let _ = worker.join();
                return Err(e);
            }
        };
        Ok(Self {
            tx: Some(tx),
            worker: Some(worker),
            active: Mutex::new(None),
            task,
            family,
        })
    }
    pub fn family(&self) -> &str {
        &self.family
    }
    pub fn cancel(&self) {
        if let Some(c) = self
            .active
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .as_ref()
        {
            c.cancel();
        }
    }
    /// Blocks until work stops, draining bounded events on this calling thread.
    pub fn execute(
        &self,
        request: AudioRequest,
        samples: Vec<f32>,
        cancel: AudioCancel,
        mut emit: impl FnMut(AudioEvent),
    ) -> Result<AudioResult, String> {
        if request.task != self.task {
            return Err("request task does not match the loaded model".into());
        }
        validate_request(&request, &samples)?;
        cancel.checkpoint()?;
        {
            let mut active = self.active.lock().unwrap_or_else(|p| p.into_inner());
            if active.is_some() {
                return Err("audio session is busy".into());
            }
            *active = Some(cancel.clone());
        }
        let (events, rx) = mpsc::sync_channel(2);
        let (reply, result) = mpsc::sync_channel(1);
        let outcome = match self
            .tx
            .as_ref()
            .ok_or("audio session is closed")
            .and_then(|tx| {
                tx.send(Message::Run {
                    request,
                    samples,
                    cancel,
                    events,
                    reply,
                })
                .map_err(|_| "audio worker stopped")
            }) {
            Ok(()) => {
                let mut callback_failed = false;
                while let Ok(event) = rx.recv() {
                    if !callback_failed
                        && std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| emit(event)))
                            .is_err()
                    {
                        self.cancel();
                        callback_failed = true;
                    }
                }
                let finished = result
                    .recv()
                    .unwrap_or_else(|_| Err("audio worker stopped without a result".into()));
                if callback_failed {
                    Err("audio event callback panicked".into())
                } else {
                    finished
                }
            }
            Err(e) => Err(e.into()),
        };
        *self.active.lock().unwrap_or_else(|p| p.into_inner()) = None;
        outcome
    }
}
impl Drop for AudioSession {
    fn drop(&mut self) {
        self.cancel();
        self.tx.take();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}
fn run_worker(engine: Engine, rx: Receiver<Message>) {
    while let Ok(Message::Run {
        request,
        samples,
        cancel,
        events,
        reply,
    }) = rx.recv()
    {
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _permit =
                crate::heavy::HeavyWorkGuard::acquire_cancellable(|| cancel.is_cancelled())
                    .ok_or("audio job cancelled")?;
            cancel.checkpoint()?;
            let _ = events.send(AudioEvent::Progress(AudioProgress {
                stage: "running".into(),
                completed: 0,
                total: None,
            }));
            admit_memory(estimate_run_bytes(&request))?;
            let outcome = engine.execute(&request, &samples, &cancel, &events);
            // Cancel discards unfinished output only after the backend actually stops.
            cancel.checkpoint()?;
            outcome
        }))
        .unwrap_or_else(|_| Err("audio inference panicked".into()));
        drop(events);
        let _ = reply.send(outcome);
    }
}
fn validate_request(request: &AudioRequest, samples: &[f32]) -> Result<(), String> {
    match request.task {
        AudioTask::SpeechToText => {
            if samples.is_empty()
                || samples.len() > MAX_INPUT_SAMPLES
                || samples.iter().any(|v| !v.is_finite())
            {
                return Err(
                    "transcription needs finite 16 kHz mono PCM, at most 30 minutes".into(),
                );
            }
        }
        AudioTask::TextToSpeech => {
            if !samples.is_empty()
                || request.text.trim().is_empty()
                || request.text.len() > 32_768
                || !request.speed.is_finite()
                || !(0.5..=2.0).contains(&request.speed)
            {
                return Err(
                    "speech needs 1 through 32768 text bytes and speed 0.5 through 2.0".into(),
                );
            }
            request
                .voice
                .parse::<audio::tts::kokoro::Voice>()
                .map_err(|e| e.to_string())?;
        }
        AudioTask::Music => {
            if !samples.is_empty() {
                return Err("music generation does not accept PCM".into());
            }
            music_request(request)
                .validate()
                .map_err(|e| e.to_string())?;
        }
    }
    Ok(())
}
fn music_request(r: &AudioRequest) -> audio::music::minimax_music3::TextGenerateRequest {
    audio::music::minimax_music3::TextGenerateRequest {
        caption: r.caption.clone(),
        lyrics: r.lyrics.clone(),
        duration_seconds: r.duration_seconds,
        steps: r.steps,
        seed: r.seed,
    }
}
enum Engine {
    Whisper(Box<crate::WhisperRunner>),
    Qwen(Box<crate::Qwen3AsrRunner>),
    QwenMetal(std::cell::RefCell<Box<crate::qwen3_asr_metal::Qwen3AsrMetalEngine>>),
    Kokoro(Box<audio::tts::kokoro::KokoroSynthesizer>),
    Music(Box<crate::Music3Runner>),
}
// The observer owns an event sender, so it must be removed even if a backend panics.
struct MusicTraceReset<'a>(&'a crate::Music3Runner);
impl Drop for MusicTraceReset<'_> {
    fn drop(&mut self) {
        self.0.clear_trace_observer();
    }
}
impl Engine {
    fn family(&self) -> &'static str {
        match self {
            Self::Whisper(_) => "whisper",
            Self::Qwen(_) | Self::QwenMetal(_) => "qwen3_asr",
            Self::Kokoro(_) => "kokoro",
            Self::Music(_) => "minimax_music3",
        }
    }
    fn open(
        path: &Path,
        task: AudioTask,
        allow_portable: bool,
        allow_experimental_metal: bool,
    ) -> Result<Self, String> {
        match task {
            AudioTask::SpeechToText => {
                let config: serde_json::Value = serde_json::from_slice(
                    &std::fs::read(path.join("config.json")).map_err(|e| e.to_string())?,
                )
                .map_err(|e| e.to_string())?;
                match config.get("model_type").and_then(|v|v.as_str()) {
                    Some("whisper") => {
                        let model = crate::WhisperRunner::open(path)?;
                        if !model.using_metal() && !allow_portable {
                            return Err("Whisper Metal initialization failed; CPU execution requires explicit opt-in".into());
                        }
                        Ok(Self::Whisper(Box::new(model)))
                    },
                    Some("qwen3_asr") if allow_experimental_metal => crate::qwen3_asr_metal::Qwen3AsrMetalEngine::open(path).map(|m|Self::QwenMetal(std::cell::RefCell::new(Box::new(m)))),
                    Some("qwen3_asr") if allow_portable => crate::Qwen3AsrRunner::open(path).map(|m|Self::Qwen(Box::new(m))),
                    Some("qwen3_asr") => Err("Qwen3-ASR native workspace Metal qualification is pending; portable execution requires explicit opt-in".into()),
                    family => Err(format!("unsupported transcription model_type {family:?}")),
                }
            }
            AudioTask::TextToSpeech if allow_portable => {
                audio::tts::kokoro::KokoroSynthesizer::open(path)
                    .map(|m| Self::Kokoro(Box::new(m)))
                    .map_err(|e| e.to_string())
            }
            AudioTask::TextToSpeech => Err(
                "Kokoro Metal runtime is pending; portable execution requires explicit opt-in"
                    .into(),
            ),
            AudioTask::Music => crate::Music3Runner::open(path)
                .map(|m| Self::Music(Box::new(m)))
                .map_err(|e| e.to_string()),
        }
    }
    fn execute(
        &self,
        r: &AudioRequest,
        samples: &[f32],
        cancel: &AudioCancel,
        events: &SyncSender<AudioEvent>,
    ) -> Result<AudioResult, String> {
        let mut result = AudioResult {
            transcript: None,
            pcm_format: None,
            sample_count: 0,
            seed: None,
        };
        match self {
            Self::Whisper(model) => {
                let t = model.transcribe_cancellable(samples, r.language.as_deref(), &|| {
                    cancel.is_cancelled()
                })?;
                result.transcript = Some(AudioTranscript {
                    text: t
                        .segments
                        .iter()
                        .map(|s| s.text.trim())
                        .filter(|s| !s.is_empty())
                        .collect::<Vec<_>>()
                        .join(" "),
                    language: Some(t.language).filter(|s| !s.is_empty()),
                    segments: t
                        .segments
                        .into_iter()
                        .map(|s| AudioSegment {
                            start_seconds: Some(s.start_seconds),
                            end_seconds: Some(s.end_seconds),
                            text: s.text,
                            timing: "window".into(),
                            speaker: None,
                            words: None,
                        })
                        .collect(),
                    tags: None,
                    token_logprobs: None,
                });
            }
            Self::Qwen(model) => {
                let language = r
                    .language
                    .as_deref()
                    .filter(|s| !s.is_empty() && *s != "auto")
                    .map(qwen_language_name);
                let t = model.transcribe_with_details(samples, language.as_deref())?;
                if t.token_logprobs.iter().any(|value| !value.is_finite()) {
                    return Err("Qwen3-ASR returned non-finite token confidence; retry the request or use another model".into());
                }
                result.transcript = Some(AudioTranscript {
                    text: t.text.clone(),
                    language: t.reported_language,
                    segments: vec![AudioSegment {
                        start_seconds: Some(0.0),
                        end_seconds: Some(samples.len() as f64 / 16000.0),
                        text: t.text,
                        timing: "clip".into(),
                        speaker: None,
                        words: None,
                    }],
                    tags: None,
                    token_logprobs: Some(t.token_logprobs),
                });
            }
            Self::QwenMetal(model) => {
                let language = r
                    .language
                    .as_deref()
                    .filter(|s| !s.is_empty() && *s != "auto")
                    .map(qwen_language_name);
                let t = model.borrow_mut().transcribe_with_details(
                    samples,
                    language.as_deref(),
                    512,
                    &|| cancel.is_cancelled(),
                )?;
                if t.token_logprobs.iter().any(|value| !value.is_finite()) {
                    return Err("Qwen3-ASR returned non-finite token confidence; retry the request or use another model".into());
                }
                result.transcript = Some(AudioTranscript {
                    text: t.text.clone(),
                    language: t.reported_language,
                    segments: vec![AudioSegment {
                        start_seconds: Some(0.0),
                        end_seconds: Some(samples.len() as f64 / 16000.0),
                        text: t.text,
                        timing: "clip".into(),
                        speaker: None,
                        words: None,
                    }],
                    tags: None,
                    token_logprobs: Some(t.token_logprobs),
                });
            }
            Self::Kokoro(model) => {
                let mut request = audio::tts::kokoro::SynthesisRequest::new(r.text.clone());
                request.speed = r.speed;
                let mut error = None;
                model
                    .synthesize(&request, |pcm| {
                        match emit_pcm(&pcm, cancel, events, &mut result.sample_count) {
                            Ok(()) => true,
                            Err(e) => {
                                error = Some(e);
                                false
                            }
                        }
                    })
                    .map_err(|e| e.to_string())?;
                if let Some(e) = error {
                    return Err(e);
                }
                result.pcm_format = Some(AudioPcmFormat {
                    sample_rate: 24000,
                    channels: 1,
                    interleaved: true,
                });
            }
            Self::Music(model) => {
                let tx = events.clone();
                let mut previous = String::new();
                model.set_trace_observer(move |trace| {
                    let stage = trace.name.split('.').next().unwrap_or("music");
                    if stage != previous {
                        previous = stage.to_string();
                        let _ = tx.send(AudioEvent::Progress(AudioProgress {
                            stage: previous.clone(),
                            completed: 0,
                            total: None,
                        }));
                    }
                });
                let reset = MusicTraceReset(model);
                let generated =
                    model.generate_text_cancellable(&music_request(r), cancel.0.clone());
                drop(reset);
                let generated = generated.map_err(|e| e.to_string())?;
                emit_pcm(
                    &generated.waveform,
                    cancel,
                    events,
                    &mut result.sample_count,
                )?;
                result.pcm_format = Some(AudioPcmFormat {
                    sample_rate: generated.sample_rate,
                    channels: 2,
                    interleaved: true,
                });
                result.seed = Some(r.seed.unwrap_or(0));
            }
        }
        Ok(result)
    }
}
fn emit_pcm(
    pcm: &[f32],
    cancel: &AudioCancel,
    events: &SyncSender<AudioEvent>,
    count: &mut usize,
) -> Result<(), String> {
    if pcm.iter().any(|v| !v.is_finite())
        || count
            .checked_add(pcm.len())
            .is_none_or(|n| n > MAX_OUTPUT_SAMPLES)
    {
        return Err("model output is non-finite or exceeds the six-minute output limit".into());
    }
    for chunk in pcm.chunks(MAX_PCM_CHUNK) {
        cancel.checkpoint()?;
        events
            .send(AudioEvent::Pcm(chunk.to_vec()))
            .map_err(|_| "audio consumer stopped".to_string())?;
    }
    *count += pcm.len();
    Ok(())
}

/// Shared ISO-code adapter for native and existing HTTP transcription contracts.
pub fn qwen_language_name(code: &str) -> String {
    const NAMES: [(&str, &str); 30] = [
        ("zh", "Chinese"),
        ("en", "English"),
        ("yue", "Cantonese"),
        ("ar", "Arabic"),
        ("de", "German"),
        ("fr", "French"),
        ("es", "Spanish"),
        ("pt", "Portuguese"),
        ("id", "Indonesian"),
        ("it", "Italian"),
        ("ko", "Korean"),
        ("ru", "Russian"),
        ("th", "Thai"),
        ("vi", "Vietnamese"),
        ("ja", "Japanese"),
        ("tr", "Turkish"),
        ("hi", "Hindi"),
        ("ms", "Malay"),
        ("nl", "Dutch"),
        ("sv", "Swedish"),
        ("da", "Danish"),
        ("fi", "Finnish"),
        ("pl", "Polish"),
        ("cs", "Czech"),
        ("fil", "Filipino"),
        ("fa", "Persian"),
        ("el", "Greek"),
        ("ro", "Romanian"),
        ("hu", "Hungarian"),
        ("mk", "Macedonian"),
    ];
    NAMES
        .iter()
        .find(|(c, _)| c.eq_ignore_ascii_case(code))
        .map_or_else(|| code.to_string(), |(_, name)| name.to_string())
}

fn estimate_open_bytes(path: &Path, task: AudioTask) -> Result<u64, String> {
    fn weights(path: &Path) -> Result<u64, String> {
        let mut total = 0u64;
        for entry in std::fs::read_dir(path).map_err(|e| e.to_string())? {
            let entry = entry.map_err(|e| e.to_string())?;
            let file = entry.path();
            if file.extension().is_some_and(|ext| ext == "safetensors") {
                total = total.saturating_add(entry.metadata().map_err(|e| e.to_string())?.len());
            }
        }
        Ok(total)
    }
    // Estimate CPU expansion during loading; packed Music weights stay packed.
    let multiplier = if task == AudioTask::Music { 2 } else { 4 };
    Ok(weights(path)?
        .saturating_mul(multiplier)
        .saturating_add(512 << 20))
}
fn estimate_run_bytes(request: &AudioRequest) -> u64 {
    match request.task {
        AudioTask::SpeechToText => (MAX_INPUT_SAMPLES as u64 * 4).saturating_add(512 << 20),
        AudioTask::TextToSpeech => 512 << 20,
        // Conservative duration-scaled reserve until full checkpoint peaks are qualified.
        AudioTask::Music => (2u64 << 30).saturating_add(
            (request.duration_seconds.unwrap_or(60.0).ceil() as u64).saturating_mul(64 << 20),
        ),
    }
}
fn admit_memory(additional: u64) -> Result<(), String> {
    let resident = gpu::process_footprint()
        .ok_or("cannot measure resident process memory for audio admission")?;
    let budget = crate::physical_memory()
        .saturating_sub(2 << 30)
        .saturating_mul(3)
        / 4;
    if resident.saturating_add(additional) > budget {
        return Err(format!("audio memory admission refused: {resident} resident bytes plus {additional} estimated bytes exceeds {budget}; unload another model or shorten the request"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn pcm_boundaries_refuse_nan_overflow_and_oversized_chunks() {
        assert!(validate_pcm(&[0.0], 0).is_ok());
        assert!(validate_pcm(&[f32::NAN], 0).is_err());
        assert!(validate_pcm(&[f32::INFINITY], 0).is_err());
        assert!(validate_pcm(&[0.0], usize::MAX).is_err());
        assert!(validate_pcm(&vec![0.0; MAX_PCM_CHUNK + 1], 0).is_err());
        assert!(validate_pcm(&[0.0], MAX_INPUT_SAMPLES).is_err());
    }
    #[test]
    fn request_validation_rejects_wrong_media_and_nonfinite_speed() {
        let mut request = AudioRequest {
            task: AudioTask::TextToSpeech,
            text: "test".into(),
            ..AudioRequest::default()
        };
        assert!(validate_request(&request, &[]).is_ok());
        assert!(validate_request(&request, &[0.0]).is_err());
        request.speed = f32::NAN;
        assert!(validate_request(&request, &[]).is_err());
        request.speed = 1.0;
        request.voice = "missing".into();
        assert!(validate_request(&request, &[]).is_err());
    }
    #[test]
    fn output_buffers_are_bounded_and_backpressure_keeps_producer_alive() {
        let (tx, rx) = mpsc::sync_channel(2);
        let (done, saw_done) = mpsc::channel();
        let cancel = AudioCancel::default();
        let c = cancel.clone();
        let worker = std::thread::spawn(move || {
            let mut count = 0;
            let result = emit_pcm(&vec![0.0; MAX_PCM_CHUNK * 4], &c, &tx, &mut count);
            done.send(result).unwrap();
        });
        assert!(saw_done
            .recv_timeout(std::time::Duration::from_millis(50))
            .is_err());
        cancel.cancel();
        while let Ok(AudioEvent::Pcm(chunk)) = rx.recv() {
            assert!(chunk.len() <= MAX_PCM_CHUNK);
        }
        assert!(saw_done.recv().unwrap().unwrap_err().contains("cancelled"));
        worker.join().unwrap();
    }
    #[test]
    fn cancellation_is_visible_while_heavy_permit_is_owned() {
        let permit = crate::heavy::HeavyWorkGuard::acquire();
        let cancel = AudioCancel::default();
        let signal = cancel.clone();
        std::thread::spawn(move || signal.cancel()).join().unwrap();
        assert!(cancel.checkpoint().is_err());
        let (tx, rx) = mpsc::channel();
        let waiter = std::thread::spawn(move || {
            let _permit = crate::heavy::HeavyWorkGuard::acquire();
            tx.send(()).unwrap();
        });
        assert!(rx
            .recv_timeout(std::time::Duration::from_millis(50))
            .is_err());
        drop(permit);
        rx.recv_timeout(std::time::Duration::from_secs(1)).unwrap();
        waiter.join().unwrap();
    }
    #[test]
    fn transcript_wire_preserves_unknown_metadata_and_timing_provenance() {
        let transcript = AudioTranscript {
            text: "hi".into(),
            language: Some("English".into()),
            segments: vec![AudioSegment {
                start_seconds: Some(0.0),
                end_seconds: Some(3.0),
                text: "hi".into(),
                timing: "clip".into(),
                speaker: None,
                words: None,
            }],
            tags: None,
            token_logprobs: Some(vec![-0.125]),
        };
        let json = serde_json::to_string(&transcript).unwrap();
        assert_eq!(
            serde_json::from_str::<AudioTranscript>(&json).unwrap(),
            transcript
        );
        assert!(json.contains("\"speaker\":null"));
        assert!(json.contains("\"timing\":\"clip\""));
    }
    #[test]
    #[ignore = "requires TURBOSPARK_AUDIO_WHISPER_DIR and TURBOSPARK_AUDIO_WAV"]
    fn real_whisper_worker_transcribes_and_cancels() {
        let path = std::env::var_os("TURBOSPARK_AUDIO_WHISPER_DIR").unwrap();
        let wav = std::env::var_os("TURBOSPARK_AUDIO_WAV").unwrap();
        let pcm = audio::read_wav_f32(Path::new(&wav)).unwrap();
        assert_eq!(pcm.sample_rate, 16000);
        assert_eq!(pcm.channels, 1);
        let session = AudioSession::open(path.into(), AudioTask::SpeechToText, false).unwrap();
        let result = session
            .execute(
                AudioRequest::default(),
                pcm.samples.clone(),
                AudioCancel::default(),
                |_| {},
            )
            .unwrap();
        assert!(!result.transcript.as_ref().unwrap().text.trim().is_empty());
        eprintln!("native transcript: {:?}", result.transcript);
        let cancel = AudioCancel::default();
        let signal = cancel.clone();
        let result = session.execute(AudioRequest::default(), pcm.samples, cancel, |event| {
            if matches!(event, AudioEvent::Progress(_)) {
                signal.cancel();
            }
        });
        assert!(result.unwrap_err().contains("cancelled"));
    }
    #[test]
    #[ignore = "requires TURBOSPARK_AUDIO_QWEN_DIR and TURBOSPARK_AUDIO_WAV"]
    fn real_qwen_metal_worker_preserves_language_and_confidence() {
        let path = std::env::var_os("TURBOSPARK_AUDIO_QWEN_DIR").unwrap();
        let wav = std::env::var_os("TURBOSPARK_AUDIO_WAV").unwrap();
        let pcm = audio::read_wav_f32(Path::new(&wav)).unwrap();
        let session =
            AudioSession::open_with_options(path.into(), AudioTask::SpeechToText, false, true)
                .unwrap();
        assert_eq!(session.family(), "qwen3_asr");
        let result = session
            .execute(
                AudioRequest::default(),
                pcm.samples,
                AudioCancel::default(),
                |_| {},
            )
            .unwrap();
        let transcript = result.transcript.unwrap();
        eprintln!("Qwen native transcript: {:?}", transcript);
        assert_eq!(
            transcript.text.trim(),
            "The quick brown fox jumps over the lazy dog."
        );
        assert_eq!(transcript.language.as_deref(), Some("English"));
        let probabilities = transcript.token_logprobs.unwrap();
        assert!(!probabilities.is_empty());
        assert!(probabilities.iter().all(|v| v.is_finite() && *v <= 0.0));
        assert_eq!(transcript.segments[0].timing, "clip");
    }
    #[test]
    #[ignore = "requires TURBOSPARK_AUDIO_KOKORO_DIR"]
    fn real_kokoro_portable_worker_generates_bounded_pcm() {
        let path = std::env::var_os("TURBOSPARK_AUDIO_KOKORO_DIR").unwrap();
        let session = AudioSession::open(path.into(), AudioTask::TextToSpeech, true).unwrap();
        let request = AudioRequest {
            task: AudioTask::TextToSpeech,
            text: "Hello.".into(),
            ..AudioRequest::default()
        };
        let mut pcm = Vec::new();
        let result = session
            .execute(request, vec![], AudioCancel::default(), |event| {
                if let AudioEvent::Pcm(chunk) = event {
                    assert!(chunk.len() <= MAX_PCM_CHUNK);
                    pcm.extend(chunk);
                }
            })
            .unwrap();
        assert_eq!(result.sample_count, pcm.len());
        assert!(pcm.iter().any(|v| v.abs() > 1e-6));
        assert!(pcm.iter().all(|v| v.is_finite()));
        assert_eq!(result.pcm_format.unwrap().sample_rate, 24000);
        eprintln!("Kokoro portable PCM: {} samples", pcm.len());
    }
    #[test]
    fn music_trace_sender_is_released_on_backend_unwind() {
        let model = crate::Music3Runner::open(
            &Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../audio/testdata/minimax_music3/converted_plain"),
        )
        .unwrap();
        let (tx, rx) = mpsc::channel::<()>();
        model.set_trace_observer(move |_| {
            let _ = tx.send(());
        });
        let failure = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _reset = MusicTraceReset(&model);
            panic!("forced backend unwind");
        }));
        assert!(failure.is_err());
        assert!(matches!(
            rx.recv_timeout(std::time::Duration::from_secs(1)),
            Err(mpsc::RecvTimeoutError::Disconnected)
        ));
    }
    #[test]
    fn opening_refuses_busy_device_instead_of_blocking_profile_teardown() {
        let permit = crate::heavy::HeavyWorkGuard::acquire();
        let (tx, rx) = mpsc::channel();
        let worker = std::thread::spawn(move || {
            let error = AudioSession::open(
                PathBuf::from("/missing/audio/checkpoint"),
                AudioTask::SpeechToText,
                false,
            )
            .err()
            .unwrap();
            tx.send(error).unwrap();
        });
        let reply = rx.recv_timeout(std::time::Duration::from_secs(1));
        drop(permit);
        worker.join().unwrap();
        assert!(reply.unwrap().contains("busy"));
    }
}
