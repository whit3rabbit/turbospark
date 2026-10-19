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
mod multitalker_stream;
mod stream;
pub use multitalker_stream::{MultitalkerStreamSession, SpeakerSegment, SpeakerStreamUpdate};
pub use stream::AudioStreamSession;

/// Parakeet Unified's encoder is a streaming-only export: a batch job would
/// have nothing to window it, so it is refused rather than run wrong.
pub const UNIFIED_IS_STREAMING_ONLY: &str =
    "Parakeet Unified is a streaming speech model; open it with ts_audio_stream_open";
pub const MAX_OUTPUT_SAMPLES: usize = 44_100 * 2 * 360;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AudioTask {
    SpeechToText,
    TextToSpeech,
    Music,
    /// Who spoke when: speaker turns as transcript segments with empty text.
    Diarization,
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
    pub instruction: Option<String>,
    pub reference_text: Option<String>,
    /// Voice cloning (MegaTTS3): intelligibility guidance weight.
    pub p_w: Option<f32>,
    /// Voice cloning (MegaTTS3): similarity guidance weight.
    pub t_w: Option<f32>,
    /// Drama 3 sampler settings. Unset keeps the reference recipe.
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<usize>,
    /// Drama 3: an upper bound on one chunk's frames, 1 through 1024.
    pub max_frames: Option<usize>,
    /// Rate of the reference audio carried as the job's PCM when cloning.
    pub reference_sample_rate: Option<u32>,
    /// The caller states it has the right to use the reference voice.
    /// Required for every cloning request.
    pub consent: bool,
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
            instruction: None,
            reference_text: None,
            p_w: None,
            t_w: None,
            temperature: None,
            top_p: None,
            top_k: None,
            max_frames: None,
            reference_sample_rate: None,
            consent: false,
        }
    }
}
impl AudioRequest {
    /// The request's Drama 3 sampler overrides.
    fn drama3_sampling(&self) -> crate::fish_drama3::Drama3Sampling {
        crate::fish_drama3::Drama3Sampling {
            temperature: self.temperature,
            top_p: self.top_p,
            top_k: self.top_k,
            max_frames: self.max_frames,
        }
    }

    /// Whether the caller set any Drama 3 sampler field.
    fn sampling_set(&self) -> bool {
        self.drama3_sampling() != crate::fish_drama3::Drama3Sampling::default()
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
    /// Drama 3: the take likely stopped speaking before its text ended (a
    /// chunk ran to its frame cap without EOS, or the audio held a silence no
    /// completed take has). The words were never generated, so a host can
    /// retake with another seed. False for every other model.
    #[serde(default)]
    pub stopped_early: bool,
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
        validate_request(&self.family, &request, &samples)?;
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
            admit_memory(engine.estimate_run_bytes(&request, &samples)?)?;
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
/// Speaker turns as a transcript: one segment per turn, no text, speakers
/// `S1`, `S2`, ... in order of first appearance. Hosts localize the label.
fn diarization_transcript(d: &crate::coreml::Diarization) -> AudioTranscript {
    AudioTranscript {
        text: String::new(),
        language: None,
        segments: d
            .turns
            .iter()
            .map(|t| AudioSegment {
                start_seconds: Some(t.start),
                end_seconds: Some(t.end),
                text: String::new(),
                // Frame-grid turn boundaries (16.875 ms), not word alignment.
                timing: "segment".into(),
                speaker: Some(format!("S{}", t.speaker + 1)),
                words: None,
            })
            .collect(),
        tags: None,
        token_logprobs: None,
    }
}
/// Persisted family string of the MegaTTS3 voice-cloning engine.
pub const MEGATTS3_FAMILY: &str = "megatts3";
/// Persisted family string of the Fish Audio Drama 3 engine.
pub const FISH_DRAMA3_FAMILY: &str = "fish_drama3";
/// Persisted family string of the OmniVoice engine (the checkpoint
/// config's `model_type`).
pub const OMNIVOICE_FAMILY: &str = "omnivoice";
/// Reference audio bounds for cloning: shorter clips align poorly, and the
/// pipeline uses at most 28 s.
const CLONE_REFERENCE_SECONDS: std::ops::RangeInclusive<f64> = 1.0..=30.0;
const CLONE_REFERENCE_RATES: std::ops::RangeInclusive<u32> = 8_000..=192_000;

/// The checks a session runs before queueing a job, for hosts that refuse a
/// request as the caller's mistake before submitting it. `family` is the
/// loaded engine's ([`AudioSession::family`]).
pub fn validate_request(
    family: &str,
    request: &AudioRequest,
    samples: &[f32],
) -> Result<(), String> {
    // One guard for every other engine: a setting a model ignores would look
    // like it worked.
    if family != FISH_DRAMA3_FAMILY && request.sampling_set() {
        return Err(
            "temperature, top_p, top_k and max_frames are Drama 3 sampler settings; this model does not take them"
                .into(),
        );
    }
    if request.task == AudioTask::TextToSpeech && family == MEGATTS3_FAMILY {
        return megatts3_request(request, samples).map(|_| ());
    }
    if request.task == AudioTask::TextToSpeech && family == FISH_DRAMA3_FAMILY {
        return fish_drama3_request(request, samples).map(|_| ());
    }
    if request.task == AudioTask::TextToSpeech && family == OMNIVOICE_FAMILY {
        return omnivoice_request(request, samples).map(|_| ());
    }
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
            kokoro_request(request)?
                .validate()
                .map_err(|e| e.to_string())?;
        }
        AudioTask::Diarization => {
            if samples.is_empty()
                || samples.len() > MAX_INPUT_SAMPLES
                || samples.iter().any(|v| !v.is_finite())
            {
                return Err(
                    "speaker diarization needs finite 16 kHz mono PCM, at most 30 minutes".into(),
                );
            }
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
/// Validates a cloning request and maps it onto the pipeline's parameters.
/// `voice` is ignored: the voice comes from the reference audio.
fn megatts3_request<'a>(
    r: &'a AudioRequest,
    samples: &'a [f32],
) -> Result<crate::megatts3::CloneRequest<'a>, String> {
    use audio::tts::megatts3::SynthParams;
    if !r.consent {
        return Err("voice cloning needs consent: set consent to true to confirm you have the right to use the reference voice".into());
    }
    let rate = r
        .reference_sample_rate
        .ok_or("voice cloning needs reference_sample_rate for the reference audio")?;
    if !CLONE_REFERENCE_RATES.contains(&rate) {
        return Err(format!(
            "reference_sample_rate {rate} is outside 8000 through 192000 Hz"
        ));
    }
    let seconds = samples.len() as f64 / f64::from(rate);
    if !CLONE_REFERENCE_SECONDS.contains(&seconds) || samples.iter().any(|v| !v.is_finite()) {
        return Err(format!(
            "voice cloning needs 1 through 30 seconds of finite reference audio, got {seconds:.2} s"
        ));
    }
    if r.text.trim().is_empty() || r.text.len() > 32_768 {
        return Err("speech needs 1 through 32768 text bytes".into());
    }
    if !r.speed.is_finite() || !(0.5..=2.0).contains(&r.speed) {
        return Err("speed must be 0.5 through 2.0".into());
    }
    if r.language
        .as_deref()
        .is_some_and(|l| !l.is_empty() && !l.to_ascii_lowercase().starts_with("en"))
    {
        return Err("MegaTTS3 speaks English only".into());
    }
    if r.instruction
        .as_deref()
        .is_some_and(|s| !s.trim().is_empty())
    {
        return Err("MegaTTS3 does not take an instruction".into());
    }
    if r.reference_text
        .as_deref()
        .is_some_and(|s| !s.trim().is_empty())
    {
        return Err("MegaTTS3 aligns the reference itself and does not take reference_text".into());
    }
    let defaults = SynthParams::default();
    let steps = r.steps.unwrap_or(defaults.steps);
    if !(1..=100).contains(&steps) {
        return Err(format!("steps {steps} is outside 1 through 100"));
    }
    let in_range = |name: &str, v: Option<f32>, default: f32, high: f32| -> Result<f32, String> {
        let v = v.unwrap_or(default);
        if v.is_finite() && (0.1..=high).contains(&v) {
            Ok(v)
        } else {
            Err(format!("{name} {v} is outside 0.1 through {high}"))
        }
    };
    let params = SynthParams {
        steps,
        p_w: in_range("p_w", r.p_w, defaults.p_w, 5.0)?,
        t_w: in_range("t_w", r.t_w, defaults.t_w, 10.0)?,
        // Durations scale with alpha, so a faster speed is a smaller alpha.
        dur_alpha: 1.0 / r.speed,
        ..defaults
    };
    audio::tts::megatts3::plan_chunks(&r.text, params.max_chunk_bytes)
        .map_err(|e| e.to_string())?;
    Ok(crate::megatts3::CloneRequest {
        text: &r.text,
        reference: samples,
        reference_rate: rate,
        seed: r.seed.unwrap_or(0),
        params,
    })
}

/// Validates a Drama 3 request. Speech without PCM speaks in a voice the model
/// picks; PCM makes it a clone, which needs consent and the reference's
/// transcript. `voice` and `language` are ignored (the voice comes from the
/// reference, and the model reads the language off the text).
fn fish_drama3_request<'a>(
    r: &'a AudioRequest,
    samples: &'a [f32],
) -> Result<crate::fish_drama3::Drama3Request<'a>, String> {
    use audio::tts::fish_drama3::chunking::validate_text;
    if r.text.trim().is_empty() || r.text.len() > 32_768 {
        return Err("speech needs 1 through 32768 text bytes".into());
    }
    validate_text(&r.text, "text").map_err(|e| e.to_string())?;
    if !r.speed.is_finite() || r.speed != 1.0 {
        return Err("Drama 3 has no speed control; leave speed at 1.0".into());
    }
    if r.instruction
        .as_deref()
        .is_some_and(|s| !s.trim().is_empty())
    {
        return Err(
            "Drama 3 takes delivery cues inline, such as [whispers], not an instruction".into(),
        );
    }
    if r.p_w.is_some() || r.t_w.is_some() {
        return Err("p_w and t_w are MegaTTS3 guidance weights; Drama 3 does not take them".into());
    }
    let sampling = r.drama3_sampling();
    // Refused here as the caller's mistake, before a job queues.
    sampling.options(0)?;
    let transcript = r.reference_text.as_deref().map(str::trim).unwrap_or("");
    let reference = if samples.is_empty() {
        if !transcript.is_empty() {
            return Err("reference_text needs reference audio".into());
        }
        None
    } else {
        if !r.consent {
            return Err("voice cloning needs consent: set consent to true to confirm you have the right to use the reference voice".into());
        }
        let rate = r
            .reference_sample_rate
            .ok_or("voice cloning needs reference_sample_rate for the reference audio")?;
        if !CLONE_REFERENCE_RATES.contains(&rate) {
            return Err(format!(
                "reference_sample_rate {rate} is outside 8000 through 192000 Hz"
            ));
        }
        let seconds = samples.len() as f64 / f64::from(rate);
        if !CLONE_REFERENCE_SECONDS.contains(&seconds) || samples.iter().any(|v| !v.is_finite()) {
            return Err(format!(
                "voice cloning needs 1 through 30 seconds of finite reference audio, got {seconds:.2} s"
            ));
        }
        if transcript.is_empty() {
            return Err(
                "Drama 3 voice cloning needs reference_text, the reference audio's transcript"
                    .into(),
            );
        }
        validate_text(transcript, "reference_text").map_err(|e| e.to_string())?;
        Some((samples, rate, transcript))
    };
    Ok(crate::fish_drama3::Drama3Request {
        text: &r.text,
        reference,
        seed: r.seed.unwrap_or(0),
        sampling,
    })
}

/// Validates an OmniVoice request. Speech without PCM speaks with the
/// model's auto voice; PCM makes it a clone, which needs consent.
/// `voice` is ignored (the model has no named voices); `p_w`/`t_w` are
/// MegaTTS3 guidance weights and refuse.
fn omnivoice_request<'a>(
    r: &'a AudioRequest,
    samples: &'a [f32],
) -> Result<crate::omnivoice::OmniVoiceRequest<'a>, String> {
    if r.text.trim().is_empty() || r.text.len() > 32_768 {
        return Err("speech needs 1 through 32768 text bytes".into());
    }
    if !r.speed.is_finite() || !(0.5..=2.0).contains(&r.speed) {
        return Err("speed must be 0.5 through 2.0".into());
    }
    if r.p_w.is_some() || r.t_w.is_some() {
        return Err(
            "p_w and t_w are MegaTTS3 guidance weights; OmniVoice does not take them".into(),
        );
    }
    let instruction = r
        .instruction
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty());
    if let Some(instruction) = instruction {
        crate::omnivoice::validate_instruction(instruction)?;
    }
    let language = r
        .language
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty() && *s != "auto");
    let steps = r
        .steps
        .unwrap_or(audio::tts::omnivoice::pipeline::GenerateParams::default().steps);
    if !(1..=100).contains(&steps) {
        return Err(format!("steps {steps} is outside 1 through 100"));
    }
    let reference = if samples.is_empty() {
        if r.reference_text
            .as_deref()
            .is_some_and(|s| !s.trim().is_empty())
        {
            return Err("reference_text needs reference audio".into());
        }
        None
    } else {
        if !r.consent {
            return Err("voice cloning needs consent: set consent to true to confirm you have the right to use the reference voice".into());
        }
        let rate = r
            .reference_sample_rate
            .ok_or("voice cloning needs reference_sample_rate for the reference audio")?;
        if !CLONE_REFERENCE_RATES.contains(&rate) {
            return Err(format!(
                "reference_sample_rate {rate} is outside 8000 through 192000 Hz"
            ));
        }
        let seconds = samples.len() as f64 / f64::from(rate);
        if !CLONE_REFERENCE_SECONDS.contains(&seconds) || samples.iter().any(|v| !v.is_finite()) {
            return Err(format!(
                "voice cloning needs 1 through 30 seconds of finite reference audio, got {seconds:.2} s"
            ));
        }
        Some(samples)
    };
    let reference_text = r
        .reference_text
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty());
    Ok(crate::omnivoice::OmniVoiceRequest {
        text: &r.text,
        language,
        instruction,
        reference,
        reference_rate: r.reference_sample_rate.unwrap_or(0),
        reference_text,
        steps,
        speed: r.speed,
        seed: r.seed.unwrap_or(0),
    })
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
    ParakeetCoreMl(Box<crate::coreml::ParakeetCoreMl>),
    DiarizationCoreMl(Box<crate::coreml::DiarizationCoreMl>),
    Kokoro(Box<crate::KokoroRunner>),
    KokoroPortable(Box<audio::tts::kokoro::KokoroSynthesizer>),
    MegaTts3(std::cell::RefCell<Box<crate::megatts3::MegaTts3Runner>>),
    FishDrama3(std::cell::RefCell<Box<crate::fish_drama3::FishDrama3Runner>>),
    OmniVoice(std::cell::RefCell<Box<crate::omnivoice::OmniVoiceRunner>>),
    Music(Box<crate::Music3Runner>),
}
// The observer owns an event sender, so it must be removed even if a backend panics.
#[cfg(test)]
struct MusicTraceReset<'a>(&'a crate::Music3Runner);
#[cfg(test)]
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
            Self::ParakeetCoreMl(_) => crate::coreml::CoreMlLayout::ParakeetTdt.family(),
            Self::DiarizationCoreMl(_) => crate::coreml::CoreMlLayout::SpeakerDiarization.family(),
            Self::Kokoro(_) | Self::KokoroPortable(_) => "kokoro",
            Self::MegaTts3(_) => MEGATTS3_FAMILY,
            Self::FishDrama3(_) => FISH_DRAMA3_FAMILY,
            Self::OmniVoice(_) => OMNIVOICE_FAMILY,
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
                // Core ML exports ship no `model_type`, so recognize them by
                // layout before reading config.json.
                let layout = crate::coreml::detect_layout(path);
                if layout == Some(crate::coreml::CoreMlLayout::ParakeetUnified) {
                    return Err(UNIFIED_IS_STREAMING_ONLY.into());
                }
                if layout == Some(crate::coreml::CoreMlLayout::ParakeetTdt) {
                    let opts = crate::coreml::ParakeetCoreMlOptions::for_model(path)?;
                    return crate::coreml::ParakeetCoreMl::open(path, &opts)
                        .map(|m| Self::ParakeetCoreMl(Box::new(m)));
                }
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
            // Declared family first: a MegaTTS3 directory must never fall
            // through to the Kokoro loader.
            AudioTask::TextToSpeech if tts_model_type(path).as_deref() == Some(MEGATTS3_FAMILY) => {
                let mut failure = String::new();
                for &device in megatts3_devices(allow_portable) {
                    match crate::megatts3::MegaTts3Runner::open(path, device) {
                        Ok(m) => {
                            if !failure.is_empty() {
                                eprintln!("MegaTTS3 runs on the CPU because Metal failed to open: {failure}");
                            }
                            return Ok(Self::MegaTts3(std::cell::RefCell::new(Box::new(m))));
                        }
                        Err(e) => failure = e.to_string(),
                    }
                }
                Err(failure)
            }
            // Metal only: the trunk is a 35B MoE whose routed experts stream
            // from disk, which has no portable path.
            AudioTask::TextToSpeech
                if tts_model_type(path).as_deref() == Some(FISH_DRAMA3_FAMILY) =>
            {
                crate::fish_drama3::FishDrama3Runner::open(
                    path,
                    model_io::ExpertCacheSlots::Fixed(crate::fish_drama3::DEFAULT_EXPERT_SLOTS),
                    model_io::ExpertResidency::Auto,
                    crate::fish_drama3::MAX_CONTEXT,
                )
                .map(|m| Self::FishDrama3(std::cell::RefCell::new(Box::new(m))))
            }
            // Declared family first: OmniVoice tries Metal, falling back to
            // CPU only when allow_portable is true.
            AudioTask::TextToSpeech
                if tts_model_type(path).as_deref() == Some(OMNIVOICE_FAMILY) =>
            {
                let mut failure = String::new();
                for &device in omnivoice_devices(allow_portable) {
                    match crate::omnivoice::OmniVoiceRunner::open(path, device) {
                        Ok(m) => {
                            if !failure.is_empty() {
                                eprintln!("OmniVoice runs on the CPU because Metal failed to open: {failure}");
                            }
                            return Ok(Self::OmniVoice(std::cell::RefCell::new(Box::new(m))));
                        }
                        Err(e) => failure = e,
                    }
                }
                Err(failure)
            }
            AudioTask::TextToSpeech if allow_portable => {
                audio::tts::kokoro::KokoroSynthesizer::open(path)
                    .map(|m| Self::KokoroPortable(Box::new(m)))
                    .map_err(|e| e.to_string())
            }
            AudioTask::TextToSpeech => crate::KokoroRunner::open(path)
                .map(|m| Self::Kokoro(Box::new(m)))
                .map_err(|e| e.to_string()),
            AudioTask::Music => crate::Music3Runner::open(path)
                .map(|m| Self::Music(Box::new(m)))
                .map_err(|e| e.to_string()),
            AudioTask::Diarization => {
                if crate::coreml::detect_layout(path)
                    != Some(crate::coreml::CoreMlLayout::SpeakerDiarization)
                {
                    return Err("unsupported speaker diarization model: expected the Core ML pyannote community-1 export".into());
                }
                let opts = crate::coreml::DiarizationOptions::from_env()?;
                crate::coreml::DiarizationCoreMl::open(path, &opts)
                    .map(|m| Self::DiarizationCoreMl(Box::new(m)))
            }
        }
    }
    fn estimate_run_bytes(&self, r: &AudioRequest, samples: &[f32]) -> Result<u64, String> {
        match self {
            Self::MegaTts3(_) => {
                let c = megatts3_request(r, samples)?;
                let longest = audio::tts::megatts3::plan_chunks(c.text, c.params.max_chunk_bytes)
                    .map_err(|e| e.to_string())?
                    .iter()
                    .map(String::len)
                    .max()
                    .unwrap_or(0);
                let prompt_24k = (samples.len() as u64
                    * u64::from(audio::tts::megatts3::SAMPLE_RATE)
                    / u64::from(c.reference_rate)) as usize;
                // The Metal path peaked no higher than the CPU path (its device
                // weights are counted when it opens).
                Ok(audio::tts::megatts3::activation_reserve_bytes(
                    prompt_24k,
                    longest,
                    c.params.dur_alpha,
                ))
            }
            Self::Kokoro(model) => model
                .estimate_run_bytes(&kokoro_request(r)?)
                .map_err(|e| e.to_string()),
            Self::KokoroPortable(model) => model
                .activation_reserve_bytes(&kokoro_request(r)?)
                .map_err(|e| e.to_string()),
            _ => Ok(estimate_run_bytes(r)),
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
            stopped_early: false,
        };
        match self {
            Self::Whisper(model) => {
                let t = model.transcribe_with_progress(
                    samples,
                    r.language.as_deref(),
                    &|| cancel.is_cancelled(),
                    |progress| {
                        let progress = match progress {
                            crate::WhisperProgress::Window { index, total } => AudioProgress {
                                stage: "window".into(),
                                completed: index,
                                total: Some(total),
                            },
                            crate::WhisperProgress::Token {
                                emitted, budget, ..
                            } => AudioProgress {
                                stage: "decode".into(),
                                completed: emitted,
                                total: Some(budget),
                            },
                        };
                        events.send(AudioEvent::Progress(progress)).is_ok()
                            && !cancel.is_cancelled()
                    },
                )?;
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
            Self::ParakeetCoreMl(model) => {
                let aligned = model.decode(
                    samples,
                    model.sample_rate(),
                    &|| cancel.is_cancelled(),
                    |index, total| {
                        events
                            .send(AudioEvent::Progress(AudioProgress {
                                stage: "window".into(),
                                completed: index,
                                total: Some(total),
                            }))
                            .is_ok()
                            && !cancel.is_cancelled()
                    },
                )?;
                result.transcript = Some(AudioTranscript {
                    text: aligned.text.clone(),
                    // Parakeet v3 identifies no language; it is reported
                    // as unknown rather than guessed.
                    language: None,
                    segments: aligned
                        .sentences
                        .iter()
                        .map(|s| AudioSegment {
                            start_seconds: Some(f64::from(s.start)),
                            end_seconds: Some(f64::from(s.end)),
                            text: s.text.trim().to_string(),
                            // Decoder-frame alignment (80 ms grid), not a
                            // window or clip extent.
                            timing: "token".into(),
                            speaker: None,
                            words: Some(
                                crate::coreml::parakeet::words(&s.tokens)
                                    .into_iter()
                                    .map(|(text, start, end)| AudioWord {
                                        text,
                                        start_seconds: f64::from(start),
                                        end_seconds: f64::from(end),
                                    })
                                    .collect(),
                            ),
                        })
                        .collect(),
                    tags: None,
                    token_logprobs: None,
                });
            }
            Self::DiarizationCoreMl(model) => {
                let d = model.diarize(
                    samples,
                    model.sample_rate(),
                    &|| cancel.is_cancelled(),
                    |done, total| {
                        events
                            .send(AudioEvent::Progress(AudioProgress {
                                stage: "window".into(),
                                completed: done,
                                total: Some(total),
                            }))
                            .is_ok()
                            && !cancel.is_cancelled()
                    },
                )?;
                result.transcript = Some(diarization_transcript(&d));
            }
            Self::Kokoro(model) => {
                synthesize_kokoro(
                    r,
                    cancel,
                    events,
                    &mut result,
                    |request, seed, checkpoint, emit| {
                        model.synthesize_controlled(request, seed, checkpoint, emit)
                    },
                )?;
            }
            Self::KokoroPortable(model) => {
                synthesize_kokoro(
                    r,
                    cancel,
                    events,
                    &mut result,
                    |request, seed, checkpoint, emit| {
                        model.synthesize_controlled(request, seed, checkpoint, emit)
                    },
                )?;
            }
            Self::FishDrama3(model) => {
                synthesize_fish_drama3(
                    r,
                    samples,
                    cancel,
                    events,
                    &mut result,
                    &mut model.borrow_mut(),
                )?;
            }
            Self::OmniVoice(model) => {
                synthesize_omnivoice(
                    r,
                    samples,
                    cancel,
                    events,
                    &mut result,
                    &mut model.borrow_mut(),
                )?;
            }
            Self::MegaTts3(model) => {
                synthesize_megatts3(
                    r,
                    samples,
                    cancel,
                    events,
                    &mut result,
                    &mut model.borrow_mut(),
                )?;
            }
            Self::Music(model) => {
                use audio::music::minimax_music3::{Control, Progress};
                let generated = model
                    .generate_text_controlled(&music_request(r), cancel.0.clone(), |progress| {
                        let (stage, completed, total) = match progress {
                            Progress::Tokenized { prompt_tokens } => {
                                ("tokenized", prompt_tokens, None)
                            }
                            Progress::ArFrame { emitted, target } => ("ar", emitted, Some(target)),
                            Progress::FlowChunk { index, total } => {
                                ("flow_chunk", index, Some(total))
                            }
                            Progress::DepthStep {
                                codebook, total, ..
                            } => ("depth", codebook - 1, Some(total)),
                            Progress::FlowStep { step, total, .. } => ("euler", step, Some(total)),
                            Progress::VocoderStage {
                                channel,
                                stage,
                                total,
                                ..
                            } => ("vocoder", channel * total + stage, Some(2 * total)),
                        };
                        if events
                            .send(AudioEvent::Progress(AudioProgress {
                                stage: stage.into(),
                                completed,
                                total,
                            }))
                            .is_ok()
                            && !cancel.is_cancelled()
                        {
                            Control::Continue
                        } else {
                            Control::Cancel
                        }
                    })
                    .map_err(|e| e.to_string())?;
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
fn kokoro_request(r: &AudioRequest) -> Result<audio::tts::kokoro::SynthesisRequest, String> {
    let mut request = audio::tts::kokoro::SynthesisRequest::new(r.text.clone());
    request.speed = r.speed;
    request.voice = r
        .voice
        .parse()
        .map_err(|e: audio::SpeechError| e.to_string())?;
    if let Some(language) = &r.language {
        request.language = language.clone();
    }
    Ok(request)
}

type SegmentCheckpoint<'a> = &'a mut dyn FnMut(usize, usize) -> audio::Result<()>;
type SegmentEmit<'a> = &'a mut dyn FnMut(Vec<f32>) -> bool;
fn synthesize_kokoro(
    r: &AudioRequest,
    cancel: &AudioCancel,
    events: &SyncSender<AudioEvent>,
    result: &mut AudioResult,
    synthesize: impl FnOnce(
        &audio::tts::kokoro::SynthesisRequest,
        u64,
        SegmentCheckpoint<'_>,
        SegmentEmit<'_>,
    ) -> audio::Result<()>,
) -> Result<(), String> {
    let request = kokoro_request(r)?;
    let seed = r.seed.unwrap_or(0);
    let mut error = None;
    let mut checkpoint = |index, total| {
        cancel
            .checkpoint()
            .map_err(|why| audio::SpeechError::Input { why })?;
        events
            .send(AudioEvent::Progress(AudioProgress {
                stage: "synthesis".into(),
                completed: index,
                total: Some(total),
            }))
            .map_err(|_| audio::SpeechError::Input {
                why: "audio event receiver closed".into(),
            })
    };
    let mut emit = |pcm: Vec<f32>| match emit_pcm(&pcm, cancel, events, &mut result.sample_count) {
        Ok(()) => true,
        Err(e) => {
            error = Some(e);
            false
        }
    };
    synthesize(&request, seed, &mut checkpoint, &mut emit).map_err(|e| e.to_string())?;
    if let Some(e) = error {
        return Err(e);
    }
    result.pcm_format = Some(AudioPcmFormat {
        sample_rate: 24000,
        channels: 1,
        interleaved: true,
    });
    result.seed = Some(seed);
    Ok(())
}
fn synthesize_megatts3(
    r: &AudioRequest,
    samples: &[f32],
    cancel: &AudioCancel,
    events: &SyncSender<AudioEvent>,
    result: &mut AudioResult,
    runner: &mut crate::megatts3::MegaTts3Runner,
) -> Result<(), String> {
    use crate::megatts3::CloneProgress;
    let request = megatts3_request(r, samples)?;
    let mut error = None;
    let mut progress = |p: CloneProgress| {
        cancel
            .checkpoint()
            .map_err(|why| audio::SpeechError::Input { why })?;
        let (stage, completed, total) = match p {
            CloneProgress::Reference => ("reference", 0, Some(1)),
            CloneProgress::Diffusion { completed, total } => ("diffusion", completed, Some(total)),
        };
        events
            .send(AudioEvent::Progress(AudioProgress {
                stage: stage.into(),
                completed,
                total,
            }))
            .map_err(|_| audio::SpeechError::Input {
                why: "audio event receiver closed".into(),
            })
    };
    let mut emit = |pcm: Vec<f32>| match emit_pcm(&pcm, cancel, events, &mut result.sample_count) {
        Ok(()) => true,
        Err(e) => {
            error = Some(e);
            false
        }
    };
    runner
        .synthesize_controlled(&request, &mut progress, &mut emit)
        .map_err(|e| e.to_string())?;
    if let Some(e) = error {
        return Err(e);
    }
    result.pcm_format = Some(AudioPcmFormat {
        sample_rate: audio::tts::megatts3::SAMPLE_RATE,
        channels: 1,
        interleaved: true,
    });
    result.seed = Some(request.seed);
    Ok(())
}
fn synthesize_fish_drama3(
    r: &AudioRequest,
    samples: &[f32],
    cancel: &AudioCancel,
    events: &SyncSender<AudioEvent>,
    result: &mut AudioResult,
    runner: &mut crate::fish_drama3::FishDrama3Runner,
) -> Result<(), String> {
    let request = fish_drama3_request(r, samples)?;
    let mut error = None;
    let mut count = 0usize;
    let stop = || cancel.is_cancelled();
    let take = runner.synthesize(&request, &stop, |pcm| {
        match emit_pcm(&pcm, cancel, events, &mut count) {
            Ok(()) => true,
            Err(e) => {
                error = Some(e);
                false
            }
        }
    })?;
    if let Some(e) = error {
        return Err(e);
    }
    cancel.checkpoint()?;
    result.stopped_early = take.stopped_early();
    result.sample_count = count;
    result.pcm_format = Some(AudioPcmFormat {
        sample_rate: runner.sample_rate(),
        channels: 1,
        interleaved: true,
    });
    result.seed = Some(request.seed);
    Ok(())
}
/// OmniVoice generates the whole utterance before any audio exists; the
/// diffusion steps report progress and cancellation, then the waveform
/// streams out in chunks like every other engine.
fn synthesize_omnivoice(
    r: &AudioRequest,
    samples: &[f32],
    cancel: &AudioCancel,
    events: &SyncSender<AudioEvent>,
    result: &mut AudioResult,
    runner: &mut crate::omnivoice::OmniVoiceRunner,
) -> Result<(), String> {
    let request = omnivoice_request(r, samples)?;
    let mut count = 0usize;
    let stop = || cancel.is_cancelled();
    let (_, audio) = runner
        .synthesize(&request, &mut |step, total| {
            if events
                .send(AudioEvent::Progress(AudioProgress {
                    stage: "diffusion_step".into(),
                    completed: step,
                    total: Some(total),
                }))
                .is_ok()
                && !stop()
            {
                true
            } else {
                false
            }
        })
        .map_err(|e| {
            if stop() || e.contains("cancelled") {
                "audio job cancelled".to_string()
            } else {
                e
            }
        })?;
    emit_pcm(&audio, cancel, events, &mut count)?;
    cancel.checkpoint()?;
    result.sample_count = count;
    result.pcm_format = Some(AudioPcmFormat {
        sample_rate: runner.sample_rate(),
        channels: 1,
        interleaved: true,
    });
    result.seed = Some(request.seed);
    Ok(())
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

/// Devices MegaTTS3 tries, in order. Metal is the default: it reproduces the
/// CPU reference to float rounding and is several times faster (family
/// README, Metal). The CPU path runs only as the portable opt-in's fallback
/// when the Metal device cannot open, so it is never chosen silently.
fn megatts3_devices(allow_portable: bool) -> &'static [crate::megatts3::Device] {
    use crate::megatts3::Device;
    if allow_portable {
        &[Device::Metal, Device::Cpu]
    } else {
        &[Device::Metal]
    }
}

/// Devices OmniVoice tries, in order. Metal is the default for accelerated
/// voice cloning. The CPU path runs only as the portable opt-in's fallback
/// when the Metal device cannot open, so it is never chosen silently.
fn omnivoice_devices(allow_portable: bool) -> &'static [crate::omnivoice::Device] {
    use crate::omnivoice::Device;
    if allow_portable {
        &[Device::Metal, Device::Cpu]
    } else {
        &[Device::Metal]
    }
}

/// `model_type` of a model directory's root `config.json`, if it declares one.
pub fn tts_model_type(path: &Path) -> Option<String> {
    let bytes = std::fs::read(path.join("config.json")).ok()?;
    let value: serde_json::Value = serde_json::from_slice(&bytes).ok()?;
    value.get("model_type")?.as_str().map(str::to_owned)
}
fn estimate_open_bytes(path: &Path, task: AudioTask) -> Result<u64, String> {
    if task == AudioTask::TextToSpeech && tts_model_type(path).as_deref() == Some(MEGATTS3_FAMILY) {
        // Metal is tried first, and its device copies are the larger estimate.
        return megatts3_open_bytes(path, true);
    }
    if task == AudioTask::TextToSpeech
        && tts_model_type(path).as_deref() == Some(FISH_DRAMA3_FAMILY)
    {
        return crate::fish_drama3::open_bytes(path);
    }
    if task == AudioTask::TextToSpeech && tts_model_type(path).as_deref() == Some(OMNIVOICE_FAMILY)
    {
        return crate::omnivoice::open_bytes(path);
    }
    if crate::coreml::detect_layout(path).is_some() {
        return coreml_open_bytes(path, task);
    }
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
/// Core ML exports keep their weights in `<program>.mlmodelc/weights/weight.bin`,
/// which the safetensors scan above never sees, so every export was admitted
/// at the 512 MiB floor. Same estimate as the catalog's `parakeet_tdt` row:
/// three times compiled weights, 30-minute PCM for speech, 256 MiB reserve.
fn coreml_open_bytes(path: &Path, task: AudioTask) -> Result<u64, String> {
    let pcm = if task == AudioTask::SpeechToText {
        (MAX_INPUT_SAMPLES as u64).saturating_mul(4)
    } else {
        0
    };
    Ok(coreml_weight_bytes(path)?
        .saturating_mul(3)
        .saturating_add(pcm)
        .saturating_add(256 << 20))
}
fn coreml_weight_bytes(path: &Path) -> Result<u64, String> {
    let mut total = 0u64;
    for entry in std::fs::read_dir(path).map_err(|e| e.to_string())? {
        let program = entry.map_err(|e| e.to_string())?.path();
        if program.extension().is_some_and(|ext| ext == "mlmodelc") {
            if let Ok(meta) = std::fs::metadata(program.join("weights").join("weight.bin")) {
                total = total.saturating_add(meta.len());
            }
        }
    }
    Ok(total)
}
/// MegaTTS3 keeps each component in its own subdirectory, loaded as f32. The
/// G2P checkpoint is bf16 on disk and doubles when widened. Measured resident
/// size on the CPU path is about 5.8 GB for 4.4 GB of safetensors. On Metal
/// the DiT's projections and the WavVAE decoder are uploaded once more as
/// device buffers while the host copies stay (the WavVAE file also holds the
/// encoder, so counting all of it is an upper bound).
fn megatts3_open_bytes(path: &Path, metal: bool) -> Result<u64, String> {
    let mut total = 0u64;
    for entry in std::fs::read_dir(path).map_err(|e| e.to_string())? {
        let dir = entry.map_err(|e| e.to_string())?.path();
        if !dir.is_dir() {
            continue;
        }
        for file in std::fs::read_dir(&dir).map_err(|e| e.to_string())? {
            let file = file.map_err(|e| e.to_string())?;
            if file
                .path()
                .extension()
                .is_some_and(|ext| ext == "safetensors")
            {
                let len = file.metadata().map_err(|e| e.to_string())?.len();
                let name = dir.file_name();
                let twice = name.is_some_and(|n| n == "g2p")
                    || (metal
                        && name.is_some_and(|n| n == "diffusion_transformer" || n == "wavvae"));
                let widen = if twice { 2 } else { 1 };
                total = total.saturating_add(len.saturating_mul(widen));
            }
        }
    }
    Ok(total.saturating_add(512 << 20))
}
fn estimate_run_bytes(request: &AudioRequest) -> u64 {
    match request.task {
        AudioTask::SpeechToText => (MAX_INPUT_SAMPLES as u64 * 4).saturating_add(512 << 20),
        AudioTask::TextToSpeech => 512 << 20,
        // Conservative duration-scaled reserve until full checkpoint peaks are qualified.
        AudioTask::Diarization => (MAX_INPUT_SAMPLES as u64 * 4).saturating_add(512 << 20),
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

    fn omnivoice_base() -> AudioRequest {
        AudioRequest {
            task: AudioTask::TextToSpeech,
            text: "Hello from OmniVoice.".into(),
            speed: 1.0,
            ..AudioRequest::default()
        }
    }

    #[test]
    fn omnivoice_request_validates_consent_bounds_and_params() {
        // Plain synthesis needs no consent and no reference rate.
        let request = omnivoice_base();
        assert!(omnivoice_request(&request, &[]).is_ok());

        // PCM present without consent refuses.
        let mut clone = omnivoice_base();
        clone.consent = false;
        assert!(omnivoice_request(&clone, &[0.1f32; 24_000]).is_err());
        // Consent alone is not enough: the rate and length bounds bind.
        clone.consent = true;
        assert!(omnivoice_request(&clone, &[0.1f32; 24_000]).is_err());
        clone.reference_sample_rate = Some(24_000);
        let short = omnivoice_request(&clone, &[0.1f32; 12_000]).is_err();
        let ok = omnivoice_request(&clone, &[0.1f32; 24_000]).is_ok();
        assert!(short && ok);
        // A clone refuses MegaTTS3's guidance weights.
        let mut guided = omnivoice_base();
        guided.reference_sample_rate = Some(24_000);
        guided.consent = true;
        guided.p_w = Some(1.4);
        assert!(omnivoice_request(&guided, &[0.1f32; 24_000]).is_err());
        // Steps outside 1..=100 refuse.
        let mut fast = omnivoice_base();
        fast.steps = Some(0);
        assert!(omnivoice_request(&fast, &[]).is_err());
        let mut slow = omnivoice_base();
        slow.steps = Some(101);
        assert!(omnivoice_request(&slow, &[]).is_err());
        // reference_text without audio refuses.
        let mut transcript_only = omnivoice_base();
        transcript_only.reference_text = Some("a transcript".into());
        assert!(omnivoice_request(&transcript_only, &[]).is_err());
        // Speed bounds.
        let mut fast_spoken = omnivoice_base();
        fast_spoken.speed = 2.5;
        assert!(omnivoice_request(&fast_spoken, &[]).is_err());
        // The "auto" language maps to the reference's unset marker.
        let mut auto = omnivoice_base();
        auto.language = Some("auto".into());
        let mapped = omnivoice_request(&auto, &[]).unwrap();
        assert!(mapped.language.is_none());
    }

    /// Reads a little-endian f32 `.npy` (the fixture layout the reference
    /// tools record).
    fn read_npy_f32(path: &Path) -> (Vec<usize>, Vec<f32>) {
        let bytes = std::fs::read(path).expect("npy reads");
        assert_eq!(&bytes[..6], b"\x93NUMPY", "{} is not npy", path.display());
        let header_len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
        let header = std::str::from_utf8(&bytes[10..10 + header_len]).expect("npy header");
        assert!(header.contains("<f4"), "{} must be f32", path.display());
        let start = header.find('(').expect("shape") + 1;
        let end = header[start..].find(')').expect("shape") + start;
        let shape: Vec<usize> = header[start..end]
            .split(',')
            .filter_map(|piece| piece.trim().parse::<usize>().ok())
            .collect();
        let data = bytes[10 + header_len..]
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        (shape, data)
    }

    /// Reads a little-endian int32 `.npy` as frames-major rows `[T, books]`.
    fn read_npy_i32_rows(path: &Path) -> Vec<Vec<i32>> {
        let bytes = std::fs::read(path).expect("npy reads");
        assert_eq!(&bytes[..6], b"\x93NUMPY", "{} is not npy", path.display());
        let header_len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
        let header = std::str::from_utf8(&bytes[10..10 + header_len]).expect("npy header");
        assert!(header.contains("<i4"), "{} must be int32", path.display());
        let start = header.find('(').expect("shape") + 1;
        let end = header[start..].find(')').expect("shape") + start;
        let shape: Vec<usize> = header[start..end]
            .split(',')
            .filter_map(|piece| piece.trim().parse::<usize>().ok())
            .collect();
        assert_eq!(shape.len(), 2, "{} must be [T, books]", path.display());
        let (frames, books) = (shape[0], shape[1]);
        bytes[10 + header_len..]
            .chunks_exact(4)
            .map(|chunk| i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect::<Vec<_>>()
            .chunks(books)
            .map(|row| row.to_vec())
            .take(frames)
            .collect()
    }

    /// Cross-engine encode parity: the Rust Higgs encode of the recorded
    /// 24 kHz samples must reproduce the Python reference tokenizer's
    /// codes exactly (the encode path is deterministic, no sampling).
    /// Opt-in via TURBOSPARK_TEST_OMNIVOICE (install dir) and
    /// TURBOSPARK_TEST_OMNIVOICE_REF (directory with reference_24k.npy
    /// and reference_codes.npy from tools run against the Python
    /// reference). Skips when unset.
    #[test]
    #[ignore = "requires TURBOSPARK_TEST_OMNIVOICE and TURBOSPARK_TEST_OMNIVOICE_REF"]
    fn real_checkpoint_encode_matches_the_python_reference() {
        let Ok(dir) = std::env::var("TURBOSPARK_TEST_OMNIVOICE") else {
            eprintln!("skipping: TURBOSPARK_TEST_OMNIVOICE is unset");
            return;
        };
        let Ok(reference_dir) = std::env::var("TURBOSPARK_TEST_OMNIVOICE_REF") else {
            eprintln!("skipping: TURBOSPARK_TEST_OMNIVOICE_REF is unset");
            return;
        };
        let mut runner =
            crate::omnivoice::OmniVoiceRunner::open(Path::new(&dir), crate::omnivoice::Device::Cpu)
                .expect("real checkpoint opens");
        let (shape, samples) = read_npy_f32(&Path::new(&reference_dir).join("reference_24k.npy"));
        assert_eq!(shape.len(), 1);
        let codes = runner.encode_reference(&samples, 24_000).expect("encode");
        let golden = read_npy_i32_rows(&Path::new(&reference_dir).join("reference_codes.npy"));
        assert_eq!(codes.len(), golden.len(), "frame count");
        for (frame, (mine, golden)) in codes.iter().zip(&golden).enumerate() {
            assert_eq!(mine, golden, "frame {frame}");
        }
        eprintln!(
            "encode parity: {} frames x {} codebooks exact",
            codes.len(),
            codes.first().map_or(0, Vec::len)
        );
    }

    /// Real-checkpoint gate; opt-in via TURBOSPARK_TEST_OMNIVOICE pointing
    /// at an installed OmniVoice directory. Skips when unset.
    #[test]
    #[ignore = "requires TURBOSPARK_TEST_OMNIVOICE and a real checkpoint"]
    fn real_checkpoint_gate() {
        let Ok(dir) = std::env::var("TURBOSPARK_TEST_OMNIVOICE") else {
            eprintln!("skipping: TURBOSPARK_TEST_OMNIVOICE is unset");
            return;
        };
        let started = std::time::Instant::now();
        let mut runner = crate::omnivoice::OmniVoiceRunner::open(
            Path::new(&dir),
            crate::omnivoice::Device::Metal,
        )
        .expect("real checkpoint opens");
        let open_seconds = started.elapsed().as_secs_f64();
        eprintln!(
            "footprint after open: {:?}",
            gpu::process_footprint().map(|bytes| (bytes as f64 / (1u64 << 30) as f64, "GiB"))
        );
        let request = crate::omnivoice::OmniVoiceRequest {
            text: "The quick brown fox jumps over the lazy dog.",
            language: Some("en"),
            instruction: None,
            reference: None,
            reference_rate: 0,
            reference_text: None,
            steps: 32,
            speed: 1.0,
            seed: 7,
        };
        let generated = std::time::Instant::now();
        let (token_cells, audio) = runner
            .synthesize(&request, &mut |step, total| {
                eprintln!("step {step}/{total}");
                true
            })
            .expect("real generation");
        let seconds = generated.elapsed().as_secs_f64();
        let frames = token_cells / 8;
        let audio_seconds = frames as f64 * 960.0 / f64::from(runner.sample_rate());
        eprintln!(
            "open {open_seconds:.1} s, generate {seconds:.1} s, {frames} frames = {audio_seconds:.2} s audio, rtf {:.2}",
            seconds / audio_seconds
        );
        eprintln!(
            "footprint after generate: {:?}",
            gpu::process_footprint().map(|bytes| (bytes as f64 / (1u64 << 30) as f64, "GiB"))
        );
        assert!(!audio.is_empty());
        assert!(audio.iter().all(|v| v.is_finite()));
        // Determinism: a second seeded run must be bit-identical.
        let (tokens_again, audio_again) = runner
            .synthesize(&request, &mut |_, _| true)
            .expect("second generation");
        assert_eq!(token_cells, tokens_again);
        assert_eq!(
            audio.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            audio_again.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        // A real clone when the reference fixture is available: the same
        // samples the Python reference encoded, with a transcript.
        if let Ok(reference_dir) = std::env::var("TURBOSPARK_TEST_OMNIVOICE_REF") {
            let (_, reference_samples) =
                read_npy_f32(&Path::new(&reference_dir).join("reference_24k.npy"));
            let mut clone = crate::omnivoice::OmniVoiceRequest {
                text: "Cross-engine cloning check.",
                language: Some("en"),
                instruction: None,
                reference: Some(&reference_samples),
                reference_rate: 24_000,
                reference_text: Some("The reference transcript."),
                steps: 32,
                speed: 1.0,
                seed: 7,
            };
            let (clone_cells, clone_audio) = runner
                .synthesize(&mut clone, &mut |_, _| true)
                .expect("real clone");
            assert!(!clone_audio.is_empty());
            assert!(clone_audio.iter().all(|v| v.is_finite()));
            let (clone_cells_again, clone_audio_again) = runner
                .synthesize(&mut clone, &mut |_, _| true)
                .expect("second clone");
            assert_eq!(clone_cells, clone_cells_again);
            assert_eq!(
                clone_audio.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                clone_audio_again
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>()
            );
            eprintln!("clone: {clone_cells} token cells, deterministic");
        }
        // Optional oracle dump for transcription checks.
        if let Ok(out) = std::env::var("TURBOSPARK_TEST_OMNIVOICE_OUT") {
            let wave = audio::Waveform {
                sample_rate: runner.sample_rate(),
                channels: 1,
                samples: audio,
            };
            let bytes = audio::wav::write_wav_f32(&wave);
            std::fs::write(std::path::Path::new(&out), bytes).expect("gate wav writes");
            eprintln!("wrote {}", out);
        }
    }
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
        assert!(validate_request("kokoro", &request, &[]).is_ok());
        assert!(validate_request("kokoro", &request, &[0.0]).is_err());
        request.speed = f32::NAN;
        assert!(validate_request("kokoro", &request, &[]).is_err());
        request.speed = 1.0;
        request.voice = "missing".into();
        assert!(validate_request("kokoro", &request, &[]).is_err());
    }
    #[test]
    fn kokoro_request_retains_language_and_refuses_unsupported_locale() {
        let mut request = AudioRequest {
            task: AudioTask::TextToSpeech,
            text: "Hello, world!".into(),
            ..Default::default()
        };
        assert!(validate_request("kokoro", &request, &[]).is_ok());
        request.language = Some("en-US".into());
        assert!(validate_request("kokoro", &request, &[]).is_ok());
        request.language = Some("fr-FR".into());
        assert!(validate_request("kokoro", &request, &[])
            .unwrap_err()
            .contains("en-US"));
    }
    fn clone_request() -> AudioRequest {
        AudioRequest {
            task: AudioTask::TextToSpeech,
            text: "Hello, world!".into(),
            reference_sample_rate: Some(16_000),
            consent: true,
            steps: Some(32),
            ..Default::default()
        }
    }
    #[test]
    fn fish_drama3_validation_requires_consent_and_a_transcript_to_clone() {
        let two_seconds = vec![0.1f32; 32_000];
        let clone = || AudioRequest {
            reference_text: Some("Transcript of the reference clip.".into()),
            steps: None,
            ..clone_request()
        };
        assert!(validate_request(FISH_DRAMA3_FAMILY, &clone(), &two_seconds).is_ok());
        // Without PCM it speaks in a voice of its own; no consent is needed.
        let plain = AudioRequest {
            task: AudioTask::TextToSpeech,
            text: "[sighs] Fine. If we miss it, it's on you.".into(),
            ..Default::default()
        };
        assert!(validate_request(FISH_DRAMA3_FAMILY, &plain, &[]).is_ok());

        let refused = |mutate: &dyn Fn(&mut AudioRequest), pcm: &[f32], needle: &str| {
            let mut r = clone();
            mutate(&mut r);
            let err = validate_request(FISH_DRAMA3_FAMILY, &r, pcm).unwrap_err();
            assert!(err.contains(needle), "{err:?} lacks {needle:?}");
        };
        refused(&|r| r.consent = false, &two_seconds, "consent");
        refused(&|r| r.reference_text = None, &two_seconds, "reference_text");
        refused(
            &|r| r.reference_text = Some("  ".into()),
            &two_seconds,
            "reference_text",
        );
        refused(&|_| {}, &[], "reference_text needs reference audio");
        refused(
            &|r| r.reference_sample_rate = None,
            &two_seconds,
            "reference_sample_rate",
        );
        refused(&|_| {}, &vec![0.1; 15_999], "1 through 30 seconds");
        refused(&|r| r.speed = 1.2, &two_seconds, "speed");
        refused(
            &|r| r.instruction = Some("whisper".into()),
            &two_seconds,
            "inline",
        );
        refused(&|r| r.p_w = Some(1.0), &two_seconds, "p_w");
        refused(
            &|r| r.text = "hi <|im_end|>".into(),
            &two_seconds,
            "<|im_end|>",
        );
        refused(
            &|r| r.reference_text = Some("<|speaker:16|> nope".into()),
            &two_seconds,
            "speaker",
        );
        refused(&|r| r.text = " ".into(), &two_seconds, "text bytes");
    }

    /// The four sampler fields reach Drama 3's request, out-of-range values
    /// are refused as the caller's mistake, and no other engine accepts them.
    #[test]
    fn fish_drama3_sampler_fields_validate_and_reach_the_request() {
        let speech = || AudioRequest {
            task: AudioTask::TextToSpeech,
            text: "[sighs] Fine. If we miss it, it's on you.".into(),
            ..Default::default()
        };
        // Unset: the recipe, so a seed keeps its codes.
        let plain = speech();
        let unset = fish_drama3_request(&plain, &[]).unwrap();
        assert_eq!(
            unset.sampling,
            crate::fish_drama3::Drama3Sampling::default()
        );

        let set = AudioRequest {
            temperature: Some(0.9),
            top_p: Some(0.8),
            top_k: Some(12),
            max_frames: Some(300),
            ..speech()
        };
        assert!(validate_request(FISH_DRAMA3_FAMILY, &set, &[]).is_ok());
        let mapped = fish_drama3_request(&set, &[]).unwrap();
        assert_eq!(
            mapped.sampling,
            crate::fish_drama3::Drama3Sampling {
                temperature: Some(0.9),
                top_p: Some(0.8),
                top_k: Some(12),
                max_frames: Some(300),
            }
        );

        let refused = |mutate: &dyn Fn(&mut AudioRequest), needle: &str| {
            let mut r = speech();
            mutate(&mut r);
            let err = validate_request(FISH_DRAMA3_FAMILY, &r, &[]).unwrap_err();
            assert!(err.contains(needle), "{err:?} lacks {needle:?}");
        };
        refused(&|r| r.temperature = Some(-1.0), "temperature");
        refused(&|r| r.top_p = Some(0.0), "top_p");
        refused(&|r| r.top_k = Some(0), "top_k");
        refused(&|r| r.max_frames = Some(0), "max_frames");
        refused(&|r| r.max_frames = Some(1025), "max_frames");

        // Every other engine refuses them, even a clone request that is valid
        // without them.
        for field in [
            |r: &mut AudioRequest| r.temperature = Some(0.7),
            |r: &mut AudioRequest| r.top_p = Some(0.7),
            |r: &mut AudioRequest| r.top_k = Some(30),
            |r: &mut AudioRequest| r.max_frames = Some(100),
        ] {
            let mut r = clone_request();
            assert!(validate_request(MEGATTS3_FAMILY, &r, &vec![0.1f32; 32_000]).is_ok());
            field(&mut r);
            let err = validate_request(MEGATTS3_FAMILY, &r, &vec![0.1f32; 32_000]).unwrap_err();
            assert!(err.contains("Drama 3"), "{err:?}");
            let mut kokoro = AudioRequest {
                task: AudioTask::TextToSpeech,
                text: "Hello, world!".into(),
                ..Default::default()
            };
            assert!(validate_request("kokoro", &kokoro, &[]).is_ok());
            field(&mut kokoro);
            assert!(validate_request("kokoro", &kokoro, &[]).is_err());
        }
    }

    #[test]
    fn megatts3_validation_requires_consent_and_a_bounded_reference() {
        let two_seconds = vec![0.1f32; 32_000];
        let ok = clone_request();
        assert!(validate_request(MEGATTS3_FAMILY, &ok, &two_seconds).is_ok());
        // Kokoro still refuses PCM for the same request.
        assert!(validate_request("kokoro", &ok, &two_seconds).is_err());

        let refused = |mutate: &dyn Fn(&mut AudioRequest), pcm: &[f32], needle: &str| {
            let mut r = clone_request();
            mutate(&mut r);
            let err = validate_request(MEGATTS3_FAMILY, &r, pcm).unwrap_err();
            assert!(err.contains(needle), "{err:?} lacks {needle:?}");
        };
        refused(&|r| r.consent = false, &two_seconds, "consent");
        refused(
            &|r| r.reference_sample_rate = None,
            &two_seconds,
            "reference_sample_rate",
        );
        refused(
            &|r| r.reference_sample_rate = Some(4_000),
            &two_seconds,
            "8000",
        );
        refused(&|_| {}, &[], "1 through 30 seconds");
        refused(&|_| {}, &vec![0.1; 15_999], "1 through 30 seconds");
        refused(&|_| {}, &vec![0.1; 16_000 * 30 + 1], "1 through 30 seconds");
        let mut nan = two_seconds.clone();
        nan[7] = f32::NAN;
        refused(&|_| {}, &nan, "finite");
        refused(&|r| r.text = " ".into(), &two_seconds, "text bytes");
        refused(
            &|r| r.text = "\u{4f60}\u{597d}".into(),
            &two_seconds,
            "English",
        );
        refused(&|r| r.language = Some("fr".into()), &two_seconds, "English");
        refused(&|r| r.speed = 3.0, &two_seconds, "speed");
        refused(&|r| r.steps = Some(0), &two_seconds, "steps");
        refused(&|r| r.steps = Some(101), &two_seconds, "steps");
        refused(&|r| r.p_w = Some(0.0), &two_seconds, "p_w");
        refused(&|r| r.t_w = Some(f32::NAN), &two_seconds, "t_w");
        refused(
            &|r| r.instruction = Some("whisper".into()),
            &two_seconds,
            "instruction",
        );
        refused(
            &|r| r.reference_text = Some("hi".into()),
            &two_seconds,
            "reference_text",
        );
        // Exactly 1 s and 30 s are accepted.
        assert!(validate_request(MEGATTS3_FAMILY, &ok, &vec![0.1; 16_000]).is_ok());
        assert!(validate_request(MEGATTS3_FAMILY, &ok, &vec![0.1; 16_000 * 30]).is_ok());
    }
    #[test]
    fn megatts3_request_maps_defaults_and_speed() {
        let pcm = vec![0.1f32; 32_000];
        let mut r = clone_request();
        r.steps = None;
        r.seed = Some(9);
        r.speed = 2.0;
        let c = megatts3_request(&r, &pcm).unwrap();
        let d = audio::tts::megatts3::SynthParams::default();
        assert_eq!(
            (c.params.steps, c.params.p_w, c.params.t_w),
            (d.steps, d.p_w, d.t_w)
        );
        assert_eq!(c.params.dur_alpha, 0.5, "twice the speed halves durations");
        assert_eq!((c.seed, c.reference_rate), (9, 16_000));
    }
    #[test]
    fn coreml_admission_counts_compiled_weights() {
        let dir = std::env::temp_dir().join(format!(
            "ts-native-coreml-estimate-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        for program in crate::coreml::parakeet::REQUIRED_DIRS {
            std::fs::create_dir_all(dir.join(program).join("weights")).unwrap();
        }
        std::fs::write(dir.join(crate::coreml::parakeet::VOCAB_FILE), b"{}").unwrap();
        std::fs::write(
            dir.join("Encoder.mlmodelc/weights/weight.bin"),
            vec![0u8; 1000],
        )
        .unwrap();
        std::fs::write(
            dir.join("Decoder.mlmodelc/weights/weight.bin"),
            vec![0u8; 100],
        )
        .unwrap();
        // A stray safetensors file is not what a Core ML export loads.
        std::fs::write(dir.join("extra.safetensors"), vec![0u8; 7]).unwrap();
        let pcm = (MAX_INPUT_SAMPLES as u64) * 4;
        assert_eq!(
            estimate_open_bytes(&dir, AudioTask::SpeechToText).unwrap(),
            3 * 1100 + pcm + (256 << 20)
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }
    #[test]
    fn tts_family_sniff_and_megatts3_admission_estimate() {
        let dir = std::env::temp_dir().join(format!("ts-megatts3-sniff-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("g2p")).unwrap();
        std::fs::create_dir_all(dir.join("diffusion_transformer")).unwrap();
        std::fs::create_dir_all(dir.join("wavvae")).unwrap();
        assert_eq!(tts_model_type(&dir), None);
        std::fs::write(dir.join("config.json"), r#"{"istftnet":{}}"#).unwrap();
        assert_eq!(
            tts_model_type(&dir),
            None,
            "a Kokoro-style config declares nothing"
        );
        std::fs::write(dir.join("config.json"), r#"{"model_type":"megatts3"}"#).unwrap();
        assert_eq!(tts_model_type(&dir).as_deref(), Some(MEGATTS3_FAMILY));
        std::fs::write(dir.join("g2p/model.safetensors"), vec![0u8; 1000]).unwrap();
        std::fs::write(
            dir.join("diffusion_transformer/model.safetensors"),
            vec![0u8; 300],
        )
        .unwrap();
        std::fs::write(dir.join("wavvae/model.safetensors"), vec![0u8; 50]).unwrap();
        // Subdirectory weights count, and the bf16 G2P doubles when widened.
        assert_eq!(
            megatts3_open_bytes(&dir, false).unwrap(),
            2 * 1000 + 300 + 50 + (512 << 20)
        );
        // Metal keeps the host DiT and WavVAE and uploads them again, and
        // admission sizes for it because Metal is tried first.
        let metal = 2 * 1000 + 2 * 300 + 2 * 50 + (512 << 20);
        assert_eq!(megatts3_open_bytes(&dir, true).unwrap(), metal);
        assert_eq!(
            estimate_open_bytes(&dir, AudioTask::TextToSpeech).unwrap(),
            metal
        );
        // No opt-in is needed: the open gets as far as loading the (empty)
        // weights instead of refusing up front.
        let err = Engine::open(&dir, AudioTask::TextToSpeech, false, false)
            .err()
            .unwrap();
        assert!(!err.contains("opt-in"), "{err}");
        std::fs::remove_dir_all(&dir).unwrap();
    }
    #[test]
    fn megatts3_opens_on_metal_and_falls_back_to_the_cpu_only_when_allowed() {
        use crate::megatts3::Device;
        assert_eq!(megatts3_devices(false), [Device::Metal]);
        assert_eq!(megatts3_devices(true), [Device::Metal, Device::Cpu]);
    }
    #[test]
    fn omnivoice_opens_on_metal_and_falls_back_to_the_cpu_only_when_allowed() {
        use crate::omnivoice::Device;
        assert_eq!(omnivoice_devices(false), [Device::Metal]);
        assert_eq!(omnivoice_devices(true), [Device::Metal, Device::Cpu]);
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
        let _serial = crate::heavy::serial_test_lock();
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
        let _serial = crate::heavy::serial_test_lock();
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
            if matches!(event, AudioEvent::Progress(AudioProgress { ref stage, completed, .. }) if stage == "decode" && completed > 0) {
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
        let _serial = crate::heavy::serial_test_lock();
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
        let _serial = crate::heavy::serial_test_lock();
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
    #[ignore = "requires TURBOSPARK_MEGATTS3_DIR and TURBOSPARK_MEGATTS3_REFERENCE"]
    fn real_megatts3_worker_clones_streams_and_cancels() {
        let path = std::env::var_os("TURBOSPARK_MEGATTS3_DIR").unwrap();
        let wav = std::env::var_os("TURBOSPARK_MEGATTS3_REFERENCE").unwrap();
        let wave = audio::read_wav_f32(Path::new(&wav)).unwrap();
        assert_eq!(wave.channels, 1);
        let _serial = crate::heavy::serial_test_lock();
        // Metal is the default: no portable or experimental opt-in.
        let session = AudioSession::open(path.into(), AudioTask::TextToSpeech, false).unwrap();
        assert_eq!(session.family(), MEGATTS3_FAMILY);
        let request = AudioRequest {
            task: AudioTask::TextToSpeech,
            text: "Hello world.".into(),
            reference_sample_rate: Some(wave.sample_rate),
            consent: true,
            steps: Some(2),
            seed: Some(1),
            ..AudioRequest::default()
        };
        let (mut pcm, mut stages) = (Vec::new(), Vec::new());
        let result = session
            .execute(
                request.clone(),
                wave.samples.clone(),
                AudioCancel::default(),
                |event| match event {
                    AudioEvent::Pcm(chunk) => pcm.extend(chunk),
                    AudioEvent::Progress(p) => stages.push(p.stage),
                },
            )
            .unwrap();
        assert_eq!(result.sample_count, pcm.len());
        assert!(pcm.iter().all(|v| v.is_finite()) && pcm.iter().any(|v| v.abs() > 1e-4));
        assert_eq!(result.pcm_format.unwrap().sample_rate, 24000);
        assert_eq!(result.seed, Some(1));
        assert!(stages.iter().any(|s| s == "reference"));
        assert!(stages.iter().any(|s| s == "diffusion"));
        eprintln!("MegaTTS3 PCM: {} samples, stages {stages:?}", pcm.len());

        // Missing consent never reaches the worker.
        let mut refused = request.clone();
        refused.consent = false;
        let err = session
            .execute(
                refused,
                wave.samples.clone(),
                AudioCancel::default(),
                |_| {},
            )
            .unwrap_err();
        assert!(err.contains("consent"), "{err}");

        // Cancelling at the first diffusion step stops the job.
        let cancel = AudioCancel::default();
        let signal = cancel.clone();
        let err = session
            .execute(request, wave.samples, cancel, |event| {
                if matches!(event, AudioEvent::Progress(AudioProgress { ref stage, completed, .. }) if stage == "diffusion" && completed >= 1)
                {
                    signal.cancel();
                }
            })
            .unwrap_err();
        assert!(err.contains("cancelled"), "{err}");
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
    fn cooperative_controls_worker_cancels_during_flow_and_replays_stereo() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../audio/testdata/minimax_music3/converted_plain");
        let _serial = crate::heavy::serial_test_lock();
        let session = AudioSession::open(path, AudioTask::Music, false).unwrap();
        let request = AudioRequest {
            task: AudioTask::Music,
            caption: "piano".into(),
            lyrics: "[instrumental]".into(),
            duration_seconds: Some(0.12),
            steps: Some(2),
            seed: Some(7),
            ..Default::default()
        };
        let mut expected_pcm = vec![];
        let expected = session
            .execute(request.clone(), vec![], AudioCancel::default(), |event| {
                if let AudioEvent::Pcm(samples) = event {
                    expected_pcm.extend(samples);
                }
            })
            .unwrap();
        assert!(expected.sample_count > 0 && expected.sample_count % 2 == 0);
        assert_eq!(
            expected.pcm_format,
            Some(AudioPcmFormat {
                sample_rate: 44100,
                channels: 2,
                interleaved: true
            })
        );
        let cancel = AudioCancel::default();
        let signal = cancel.clone();
        let mut reached = false;
        let error = session.execute(request.clone(), vec![], cancel, |event| {
            if matches!(event, AudioEvent::Progress(AudioProgress { ref stage, completed: 1, .. }) if stage == "euler") {
                reached = true;
                signal.cancel();
            }
        }).unwrap_err();
        assert!(reached && error.contains("cancelled"));
        let mut replay_pcm = vec![];
        let replay = session
            .execute(request, vec![], AudioCancel::default(), |event| {
                if let AudioEvent::Pcm(samples) = event {
                    replay_pcm.extend(samples);
                }
            })
            .unwrap();
        assert_eq!(replay, expected);
        assert_eq!(replay_pcm, expected_pcm);
    }
    #[test]
    fn opening_refuses_busy_device_instead_of_blocking_profile_teardown() {
        let _serial = crate::heavy::serial_test_lock();
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
    #[test]
    fn diarization_requests_need_bounded_finite_pcm() {
        let family = crate::coreml::diarization::FAMILY;
        let r = AudioRequest {
            task: AudioTask::Diarization,
            ..AudioRequest::default()
        };
        assert!(validate_request(family, &r, &[]).is_err());
        assert!(validate_request(family, &r, &[0.0, f32::NAN]).is_err());
        assert!(validate_request(family, &r, &[0.0; 16_000]).is_ok());
        let sampled = AudioRequest {
            temperature: Some(0.7),
            ..r.clone()
        };
        assert!(validate_request(family, &sampled, &[0.0; 16_000]).is_err());
        let task: AudioTask = serde_json::from_str("\"diarization\"").unwrap();
        assert_eq!(task, AudioTask::Diarization);
    }
    #[test]
    fn diarization_transcript_labels_turns_without_text() {
        use crate::coreml::{Diarization, SpeakerTurn};
        let d = Diarization {
            turns: vec![
                SpeakerTurn {
                    speaker: 0,
                    start: 0.03,
                    end: 2.5,
                },
                SpeakerTurn {
                    speaker: 1,
                    start: 2.5,
                    end: 4.0,
                },
            ],
            speakers: 2,
        };
        let t = diarization_transcript(&d);
        assert!(t.text.is_empty());
        let labels: Vec<_> = t.segments.iter().map(|s| s.speaker.as_deref()).collect();
        assert_eq!(labels, [Some("S1"), Some("S2")]);
        assert_eq!(t.segments[1].start_seconds, Some(2.5));
        assert!(t
            .segments
            .iter()
            .all(|s| s.text.is_empty() && s.words.is_none()));
    }
}
