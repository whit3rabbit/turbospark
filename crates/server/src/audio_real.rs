//! The standalone binary's [`AudioProvider`]: real STT, TTS and music models
//! on dedicated worker threads.
//!
//! One thread per loaded model, and the model is OPENED ON that thread. The
//! music runner holds `Rc` state (`!Send`) and the Whisper runner's Metal
//! engine is `RefCell`-based, so neither can be built here and moved in.
//! Requests reach the thread over a bounded channel: a full channel is
//! [`AudioError::Busy`], never an unbounded queue of multi-hundred-megabyte
//! sample buffers. A request whose HTTP future was dropped is skipped if it
//! has not started; an active request is cancelled at its next backend checkpoint.
//!
//! Native execution shares the process-wide heavyweight gate with the FFI host.
//! Dropped requests signal cancellation, while the worker retains the permit
//! until the model has stopped. HTTP task spellings remain adapter-owned.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{sync_channel, Receiver, SyncSender, TrySendError};
use std::sync::{Arc, Mutex};

use tokio::sync::{mpsc, oneshot};
use turbospark_audio::music::minimax_music3::TextGenerateRequest;
use turbospark_audio::tts::kokoro::Voice;

use crate::audio::{
    AudioError, AudioModelInfo, AudioProvider, AudioTask, GenerateRequest, GeneratedAudio,
    SpeechRequest, SpeechStream, TranscribeRequest, TranscribedSegment, Transcription,
};

type Fut<'a, T> = std::pin::Pin<Box<dyn std::future::Future<Output = T> + Send + 'a>>;

/// Pending requests per model before callers see 429. Music jobs are minutes
/// long, so its queue is shorter.
const STT_QUEUE: usize = 4;
const TTS_QUEUE: usize = 4;
const MUSIC_QUEUE: usize = 2;

type Engine = runtime::native_audio::AudioSession;
struct CancelOnDrop(runtime::native_audio::AudioCancel);
impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        self.0.cancel();
    }
}

enum Work {
    Transcribe {
        samples: Vec<f32>,
        language: Option<String>,
        cancel: runtime::native_audio::AudioCancel,
        reply: oneshot::Sender<Result<Transcription, AudioError>>,
    },
    Speak {
        request: runtime::native_audio::AudioRequest,
        /// The reference voice when cloning, else empty.
        samples: Vec<f32>,
        chunks: mpsc::Sender<Result<Vec<f32>, AudioError>>,
    },
    Generate {
        request: TextGenerateRequest,
        cancel: runtime::native_audio::AudioCancel,
        reply: oneshot::Sender<Result<GeneratedAudio, AudioError>>,
    },
}

struct Worker {
    info: AudioModelInfo,
    tx: SyncSender<Work>,
}

impl Worker {
    fn submit(&self, work: Work) -> Result<(), AudioError> {
        self.tx.try_send(work).map_err(|e| match e {
            TrySendError::Full(_) => AudioError::Busy,
            TrySendError::Disconnected(_) => {
                AudioError::Failed("the model worker stopped".to_string())
            }
        })
    }
}

fn open_engine(task: AudioTask, dir: &Path) -> Result<Engine, String> {
    // The standalone HTTP contract already permits the portable Kokoro/Qwen paths.
    Engine::open(dir.to_path_buf(), native_task(task), true)
}
fn native_task(task: AudioTask) -> runtime::native_audio::AudioTask {
    match task {
        AudioTask::SpeechToText => runtime::native_audio::AudioTask::SpeechToText,
        AudioTask::TextToSpeech => runtime::native_audio::AudioTask::TextToSpeech,
        AudioTask::Music => runtime::native_audio::AudioTask::Music,
    }
}

#[cfg(test)]
use runtime::native_audio::qwen_language_name;

fn failed(e: impl std::fmt::Display) -> AudioError {
    AudioError::Failed(e.to_string())
}

use runtime::native_audio::{FISH_DRAMA3_FAMILY, MEGATTS3_FAMILY, OMNIVOICE_FAMILY};

/// Drama 3 speaks at the Fish codec's 44.1 kHz. The runner only reports its
/// rate after it opens, and the route needs it before the first chunk, so
/// it is a constant here (the runtime test pins the codec to match).
const DRAMA3_SAMPLE_RATE: u32 = 44_100;

/// A Drama 3 request: reference audio is optional (its presence makes the
/// request a clone, which needs consent and the reference's transcript, both
/// checked by the engine's validator). The model has no steps, speed,
/// instruction or guidance weights, so none are forwarded; the route refuses
/// them by name first. `steps` stays `None` because the request default of
/// 30 would otherwise leak into an engine that refuses it.
async fn drama3_request(
    request: SpeechRequest,
) -> Result<(runtime::native_audio::AudioRequest, Vec<f32>, u32), AudioError> {
    let (samples, rate) = match request.reference_audio.clone() {
        Some(encoded) => {
            let (samples, rate) = tokio::task::spawn_blocking(move || decode_reference(&encoded))
                .await
                .map_err(|_| failed("decoding the reference audio stopped"))??;
            (samples, Some(rate))
        }
        None => (Vec::new(), None),
    };
    let native = runtime::native_audio::AudioRequest {
        task: native_task(AudioTask::TextToSpeech),
        text: request.text,
        steps: None,
        seed: Some(request.seed.unwrap_or(0)),
        reference_sample_rate: rate,
        consent: request.consent,
        reference_text: request.reference_text,
        temperature: request.temperature,
        top_p: request.top_p,
        top_k: request.top_k,
        max_frames: request.max_frames,
        ..runtime::native_audio::AudioRequest::default()
    };
    Ok((native, samples, DRAMA3_SAMPLE_RATE))
}

/// An OmniVoice request: reference audio is optional (its presence makes
/// the request a clone); the engine validates consent and bounds.
async fn omnivoice_request(
    request: SpeechRequest,
) -> Result<(runtime::native_audio::AudioRequest, Vec<f32>, u32), AudioError> {
    let (samples, rate) = match request.reference_audio.clone() {
        Some(encoded) => {
            let (samples, rate) = tokio::task::spawn_blocking(move || decode_reference(&encoded))
                .await
                .map_err(|_| failed("decoding the reference audio stopped"))??;
            (samples, Some(rate))
        }
        None => (Vec::new(), None),
    };
    let native = runtime::native_audio::AudioRequest {
        task: native_task(AudioTask::TextToSpeech),
        text: request.text,
        speed: request.speed,
        language: request.language,
        steps: Some(request.steps.unwrap_or(
            turbospark_audio::tts::omnivoice::pipeline::GenerateParams::default().steps,
        )),
        seed: Some(request.seed.unwrap_or(0)),
        reference_sample_rate: rate,
        consent: request.consent,
        instruction: request.instruction,
        reference_text: request.reference_text,
        ..runtime::native_audio::AudioRequest::default()
    };
    Ok((native, samples, 24_000))
}

/// A MegaTTS3 request: the reference voice decoded at its own rate (the
/// runtime resamples it), the model's step default, and the clone options.
async fn clone_request(
    request: SpeechRequest,
) -> Result<(runtime::native_audio::AudioRequest, Vec<f32>, u32), AudioError> {
    let encoded = request.reference_audio.ok_or_else(|| {
        AudioError::Invalid("voice cloning needs reference_audio (a base64 WAV)".into())
    })?;
    let (samples, rate) = tokio::task::spawn_blocking(move || decode_reference(&encoded))
        .await
        .map_err(|_| failed("decoding the reference audio stopped"))??;
    let defaults = turbospark_audio::tts::megatts3::SynthParams::default();
    let native = runtime::native_audio::AudioRequest {
        task: native_task(AudioTask::TextToSpeech),
        text: request.text,
        speed: request.speed,
        language: request.language,
        steps: Some(request.steps.unwrap_or(defaults.steps)),
        seed: Some(request.seed.unwrap_or(0)),
        p_w: request.p_w,
        t_w: request.t_w,
        reference_sample_rate: Some(rate),
        consent: request.consent,
        instruction: request.instruction,
        reference_text: request.reference_text,
        ..runtime::native_audio::AudioRequest::default()
    };
    Ok((
        native,
        samples,
        turbospark_audio::tts::megatts3::SAMPLE_RATE,
    ))
}

/// The native request, reference PCM and output rate for one speech request,
/// chosen by the loaded engine's family. Anything not named here is a Kokoro
/// style voice model: no reference, a named voice, 24 kHz.
async fn speech_request(
    family: &str,
    request: SpeechRequest,
) -> Result<(runtime::native_audio::AudioRequest, Vec<f32>, u32), AudioError> {
    if family == MEGATTS3_FAMILY {
        clone_request(request).await
    } else if family == OMNIVOICE_FAMILY {
        omnivoice_request(request).await
    } else if family == FISH_DRAMA3_FAMILY {
        drama3_request(request).await
    } else {
        if request.reference_audio.is_some() {
            return Err(AudioError::Invalid(format!(
                "model {:?} does not clone voices",
                request.model
            )));
        }
        request
            .voice
            .parse::<Voice>()
            .map_err(|e| AudioError::Invalid(e.to_string()))?;
        let native = runtime::native_audio::AudioRequest {
            task: native_task(AudioTask::TextToSpeech),
            text: request.text,
            voice: request.voice,
            speed: request.speed,
            ..runtime::native_audio::AudioRequest::default()
        };
        Ok((
            native,
            vec![],
            turbospark_audio::tts::kokoro::OUTPUT_SAMPLE_RATE,
        ))
    }
}

/// Base64 WAV (an optional `data:...;base64,` prefix is accepted) to mono f32
/// at the file's own rate.
fn decode_reference(encoded: &str) -> Result<(Vec<f32>, u32), AudioError> {
    use base64::Engine;
    let payload = match encoded.split_once(";base64,") {
        Some((prefix, rest)) if prefix.starts_with("data:") => rest,
        _ => encoded,
    };
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(payload.trim())
        .map_err(|e| AudioError::Invalid(format!("reference_audio is not valid base64: {e}")))?;
    let wave = turbospark_audio::wav::read_wav_f32_bytes(&bytes).map_err(|e| {
        AudioError::Invalid(format!(
            "reference_audio: {e}; only WAV (PCM or float) is supported"
        ))
    })?;
    let channels = usize::from(wave.channels).max(1);
    let mono = wave
        .samples
        .chunks_exact(channels)
        .map(|frame| frame.iter().sum::<f32>() / channels as f32)
        .collect();
    Ok((mono, wave.sample_rate))
}

fn run(engine: Engine, queue: Receiver<Work>) {
    use runtime::native_audio::{AudioEvent, AudioRequest};
    while let Ok(work) = queue.recv() {
        match work {
            Work::Transcribe {
                samples,
                language,
                reply,
                cancel,
            } => {
                if reply.is_closed() {
                    continue;
                }
                let result = engine
                    .execute(
                        AudioRequest {
                            task: native_task(AudioTask::SpeechToText),
                            language,
                            ..AudioRequest::default()
                        },
                        samples,
                        cancel,
                        |_| {},
                    )
                    .map_err(|error| {
                        if error.contains("language") {
                            AudioError::Invalid(error)
                        } else {
                            failed(error)
                        }
                    })
                    .and_then(|result| {
                        let t = result
                            .transcript
                            .ok_or_else(|| failed("missing native transcript"))?;
                        Ok(Transcription {
                            text: t.text,
                            language: t.language,
                            segments: t
                                .segments
                                .into_iter()
                                .map(|s| TranscribedSegment {
                                    start_seconds: s.start_seconds.unwrap_or(0.0),
                                    end_seconds: s.end_seconds.unwrap_or(0.0),
                                    text: s.text,
                                })
                                .collect(),
                        })
                    });
                let _ = reply.send(result);
            }
            Work::Speak {
                request,
                samples,
                chunks,
            } => {
                if chunks.is_closed() {
                    continue;
                }
                let cancel = runtime::native_audio::AudioCancel::default();
                let signal = cancel.clone();
                // Cloning runs long stretches with no audio, so a client that
                // went away is noticed at progress events too.
                let outcome = engine.execute(request, samples, cancel, |event| match event {
                    AudioEvent::Pcm(pcm) => {
                        if chunks.blocking_send(Ok(pcm)).is_err() {
                            signal.cancel();
                        }
                    }
                    AudioEvent::Progress(_) => {
                        if chunks.is_closed() {
                            signal.cancel();
                        }
                    }
                });
                // The request was validated before it was queued, so what is
                // left is the engine's failure, not the caller's.
                if let Err(e) = outcome {
                    let _ = chunks.blocking_send(Err(failed(e)));
                }
            }
            Work::Generate {
                request,
                reply,
                cancel,
            } => {
                if reply.is_closed() {
                    continue;
                }
                let mut samples = Vec::new();
                let result = engine
                    .execute(
                        AudioRequest {
                            task: native_task(AudioTask::Music),
                            caption: request.caption,
                            lyrics: request.lyrics,
                            duration_seconds: request.duration_seconds,
                            steps: request.steps,
                            seed: request.seed,
                            ..AudioRequest::default()
                        },
                        vec![],
                        cancel,
                        |event| {
                            if let AudioEvent::Pcm(pcm) = event {
                                samples.extend_from_slice(&pcm);
                            }
                        },
                    )
                    .map_err(failed)
                    .and_then(|r| {
                        let format = r
                            .pcm_format
                            .ok_or_else(|| failed("missing native PCM format"))?;
                        Ok(GeneratedAudio {
                            sample_rate: format.sample_rate,
                            channels: format.channels as u16,
                            samples,
                        })
                    });
                let _ = reply.send(result);
            }
        }
    }
}

fn spawn_worker(id: &str, task: AudioTask, dir: PathBuf) -> Result<Arc<Worker>, String> {
    let depth = match task {
        AudioTask::SpeechToText => STT_QUEUE,
        AudioTask::TextToSpeech => TTS_QUEUE,
        AudioTask::Music => MUSIC_QUEUE,
    };
    let (tx, rx) = sync_channel::<Work>(depth);
    let (ready_tx, ready_rx) = std::sync::mpsc::channel::<Result<String, String>>();
    std::thread::Builder::new()
        .name(format!("audio-{id}"))
        .spawn(move || match open_engine(task, &dir) {
            Ok(engine) => {
                let _ = ready_tx.send(Ok(engine.family().to_string()));
                run(engine, rx);
            }
            Err(e) => {
                let _ = ready_tx.send(Err(e));
            }
        })
        .map_err(|e| format!("could not start the audio worker: {e}"))?;
    let family = ready_rx
        .recv()
        .map_err(|_| "the audio worker exited while opening the model".to_string())??;
    Ok(Arc::new(Worker {
        info: AudioModelInfo {
            id: id.to_string(),
            task,
            family: Some(family),
        },
        tx,
    }))
}

/// A catalog alias is a plain file name. Refusing separators and dots keeps
/// HTTP callers inside the store's audio directory.
fn valid_alias(alias: &str) -> bool {
    !alias.is_empty()
        && alias.len() <= 128
        && !alias.starts_with('.')
        && alias
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
}

/// What an installed alias does, from the bundled catalogs. Aliases the
/// catalogs do not know are refused: a directory's contents are not trusted
/// to say what runner to start.
/// Refuses a catalog alias that only the TurboSpark app executes (FluidAudio
/// on Core ML). No Rust engine opens one, so attaching it here would fail
/// later with a less useful error.
fn refuse_app_only(alias: &str) -> Result<(), String> {
    match catalog::audio_catalog::AudioCatalog::embedded()
        .ok()
        .and_then(|c| c.get(alias).cloned())
    {
        Some(profile) => app_only_refusal(&profile),
        None => Ok(()),
    }
}

fn app_only_refusal(profile: &catalog::audio_catalog::AudioProfile) -> Result<(), String> {
    if profile.capabilities.http_api {
        return Ok(());
    }
    Err(format!(
        "{:?} runs only in the TurboSpark app; the HTTP server cannot attach it",
        profile.identity.alias
    ))
}

fn task_of_alias(alias: &str) -> Option<AudioTask> {
    use catalog::audio_catalog::{AudioCatalog, AudioTask as CatalogTask};
    if let Some(profile) = AudioCatalog::embedded()
        .ok()
        .and_then(|c| c.get(alias).cloned())
    {
        return Some(match profile.identity.task {
            CatalogTask::SpeechToText => AudioTask::SpeechToText,
            CatalogTask::TextToSpeech => AudioTask::TextToSpeech,
            CatalogTask::Music => AudioTask::Music,
            _ => return None,
        });
    }
    if catalog::speech::embedded_entry(alias).is_ok() {
        return Some(AudioTask::SpeechToText);
    }
    if catalog::music::embedded_music_entry(alias).is_ok() {
        return Some(AudioTask::Music);
    }
    None
}

pub struct RealAudioProvider {
    store: Option<catalog::Store>,
    workers: Arc<Mutex<HashMap<String, Arc<Worker>>>>,
}

impl RealAudioProvider {
    pub fn new() -> Self {
        Self {
            store: catalog::Store::default_store().ok(),
            workers: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Opens `spec` (an install directory, or an installed alias) as a `task`
    /// model. Startup only: this accepts paths, which the HTTP `load` never
    /// does. Blocks while the model opens.
    pub fn attach(&self, spec: &str, task: AudioTask) -> Result<AudioModelInfo, String> {
        let dir = self
            .store
            .as_ref()
            .and_then(|s| s.resolve_audio(spec))
            .unwrap_or_else(|| PathBuf::from(spec));
        if !dir.is_dir() {
            return Err(format!(
                "{spec:?} is neither a directory nor an installed audio alias"
            ));
        }
        let id = if Path::new(spec).is_dir() || spec.contains('/') {
            dir.file_name()
                .and_then(|n| n.to_str())
                .map(|n| n.strip_suffix(".gturbo").unwrap_or(n).to_string())
                .ok_or_else(|| format!("cannot name a model from {spec:?}"))?
        } else {
            spec.to_string()
        };
        refuse_app_only(&id)?;
        self.insert(&id, task, dir)
    }

    fn insert(&self, id: &str, task: AudioTask, dir: PathBuf) -> Result<AudioModelInfo, String> {
        if self
            .workers
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .contains_key(id)
        {
            return Err(format!("an audio model named {id:?} is already attached"));
        }
        let worker = spawn_worker(id, task, dir)?;
        let info = worker.info.clone();
        self.workers
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .insert(id.to_string(), worker);
        Ok(info)
    }

    /// Detaches a model by id.
    pub fn detach(&self, id: &str) -> Result<(), String> {
        let mut workers = self.workers.lock().unwrap_or_else(|p| p.into_inner());
        if workers.remove(id).is_some() {
            Ok(())
        } else {
            Err(format!("audio model {id:?} is not attached"))
        }
    }

    fn worker(&self, id: &str, task: AudioTask) -> Result<Arc<Worker>, AudioError> {
        let workers = self.workers.lock().unwrap_or_else(|p| p.into_inner());
        match workers.get(id) {
            Some(w) if w.info.task == task => Ok(Arc::clone(w)),
            Some(_) => Err(AudioError::Invalid(format!(
                "model {id:?} does not do that task"
            ))),
            None => Err(AudioError::NotFound(format!(
                "audio model {id:?} is not attached"
            ))),
        }
    }
}

impl Default for RealAudioProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl AudioProvider for RealAudioProvider {
    fn models(&self) -> Vec<AudioModelInfo> {
        let mut models: Vec<_> = self
            .workers
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .values()
            .map(|w| w.info.clone())
            .collect();
        models.sort_by(|a, b| a.id.cmp(&b.id));
        models
    }

    fn transcribe(&self, request: TranscribeRequest) -> Fut<'_, Result<Transcription, AudioError>> {
        Box::pin(async move {
            let worker = self.worker(&request.model, AudioTask::SpeechToText)?;
            let (reply, rx) = oneshot::channel();
            let cancellation = CancelOnDrop(runtime::native_audio::AudioCancel::default());
            worker.submit(Work::Transcribe {
                cancel: cancellation.0.clone(),
                samples: request.samples,
                language: request.language,
                reply,
            })?;
            // Dropping this future drops `rx`, which the worker checks
            // before it starts the request.
            rx.await.map_err(|_| AudioError::Cancelled)?
        })
    }

    fn speak(&self, request: SpeechRequest) -> Fut<'_, Result<SpeechStream, AudioError>> {
        Box::pin(async move {
            let worker = self.worker(&request.model, AudioTask::TextToSpeech)?;
            let family = worker.info.family.clone().unwrap_or_default();
            let (native, samples, sample_rate) = speech_request(&family, request).await?;
            // Refuse the caller's mistakes as 400 before a minutes-long job queues.
            runtime::native_audio::validate_request(&family, &native, &samples)
                .map_err(AudioError::Invalid)?;
            // Two chunks of look-ahead: synthesis stays a step ahead of the
            // socket without buffering a whole paragraph.
            let (chunks, rx) = mpsc::channel(2);
            worker.submit(Work::Speak {
                request: native,
                samples,
                chunks,
            })?;
            Ok(SpeechStream {
                sample_rate,
                chunks: rx,
            })
        })
    }

    fn generate(&self, request: GenerateRequest) -> Fut<'_, Result<GeneratedAudio, AudioError>> {
        Box::pin(async move {
            let worker = self.worker(&request.model, AudioTask::Music)?;
            let text = TextGenerateRequest {
                caption: request.caption,
                lyrics: request.lyrics,
                duration_seconds: request.duration_seconds,
                steps: request.steps,
                seed: request.seed,
            };
            // The runner validates ranges again; surface its refusal as the
            // caller's mistake before queueing a minutes-long job.
            text.validate()
                .map_err(|e| AudioError::Invalid(e.to_string()))?;
            let (reply, rx) = oneshot::channel();
            let cancellation = CancelOnDrop(runtime::native_audio::AudioCancel::default());
            worker.submit(Work::Generate {
                cancel: cancellation.0.clone(),
                request: text,
                reply,
            })?;
            rx.await.map_err(|_| AudioError::Cancelled)?
        })
    }

    fn load(&self, alias: String) -> Fut<'_, Result<AudioModelInfo, AudioError>> {
        Box::pin(async move {
            if !valid_alias(&alias) {
                return Err(AudioError::NotFound(format!(
                    "{alias:?} is not an installed audio model alias"
                )));
            }
            refuse_app_only(&alias).map_err(AudioError::Invalid)?;
            let Some(store) = &self.store else {
                return Err(AudioError::Failed(
                    "there is no model store on this machine".into(),
                ));
            };
            let dir = store.audio_install_path(&alias);
            let task = task_of_alias(&alias).filter(|_| dir.is_dir());
            let Some(task) = task else {
                return Err(AudioError::NotFound(format!(
                    "{alias:?} is not an installed audio model alias"
                )));
            };
            // Opening reads gigabytes; keep it off the async workers.
            let workers = Arc::clone(&self.workers);
            let id = alias.clone();
            tokio::task::spawn_blocking(move || {
                if workers
                    .lock()
                    .unwrap_or_else(|p| p.into_inner())
                    .contains_key(&id)
                {
                    return Err(AudioError::Invalid(format!("{id:?} is already attached")));
                }
                let worker = spawn_worker(&id, task, dir).map_err(AudioError::Failed)?;
                let info = worker.info.clone();
                workers
                    .lock()
                    .unwrap_or_else(|p| p.into_inner())
                    .insert(id, worker);
                Ok(info)
            })
            .await
            .map_err(|_| AudioError::Failed("model load was interrupted".into()))?
        })
    }

    fn unload(&self, id: String) -> Fut<'_, Result<(), AudioError>> {
        Box::pin(async move {
            // Dropping the last `Arc<Worker>` closes the channel; the thread
            // finishes what it already took and exits, freeing the model.
            match self
                .workers
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .remove(&id)
            {
                Some(_) => Ok(()),
                None => Err(AudioError::NotFound(format!(
                    "audio model {id:?} is not attached"
                ))),
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reference_decoding_downmixes_keeps_the_rate_and_refuses_garbage() {
        use base64::Engine;
        let stereo = turbospark_audio::Waveform::new(48_000, 2, vec![0.2, 0.4, -0.2, 0.0]).unwrap();
        let wav = turbospark_audio::write_wav_f32(&stereo);
        let plain = base64::engine::general_purpose::STANDARD.encode(&wav);
        let (mono, rate) = decode_reference(&plain).unwrap();
        assert_eq!(rate, 48_000);
        assert_eq!(mono.len(), 2);
        assert!((mono[0] - 0.3).abs() < 1e-6 && (mono[1] + 0.1).abs() < 1e-6);
        // A data URL prefix is accepted.
        let url = format!("data:audio/wav;base64,{plain}");
        assert_eq!(decode_reference(&url).unwrap().1, 48_000);
        for bad in ["not base64 !!", "aGVsbG8="] {
            assert!(
                matches!(decode_reference(bad), Err(AudioError::Invalid(_))),
                "{bad}"
            );
        }
    }

    fn drama3_speech(reference: Option<&str>) -> SpeechRequest {
        SpeechRequest {
            model: "drama".into(),
            text: "Hello there.".into(),
            voice: "af_heart".into(),
            speed: 1.0,
            language: None,
            reference_audio: reference.map(str::to_string),
            reference_text: None,
            instruction: None,
            consent: false,
            // The route default leaves steps unset; a value here must not
            // reach the engine.
            steps: Some(30),
            seed: None,
            p_w: None,
            t_w: None,
            temperature: None,
            top_p: None,
            top_k: None,
            max_frames: None,
        }
    }

    fn reference_wav(rate: u32, seconds: f32) -> String {
        use base64::Engine;
        let frames = (rate as f32 * seconds) as usize;
        let samples = (0..frames).map(|i| (i as f32 * 0.05).sin() * 0.3).collect();
        let wave = turbospark_audio::Waveform::new(rate, 1, samples).unwrap();
        base64::engine::general_purpose::STANDARD.encode(turbospark_audio::write_wav_f32(&wave))
    }

    /// The rate feeds `x-audio-sample-rate` and the WAV header, so a Drama
    /// model answering at the Kokoro default would play at the wrong speed.
    #[tokio::test]
    async fn drama3_requests_report_44100_and_forward_only_what_it_takes() {
        // Through the dispatch: a Drama model must not fall into the Kokoro
        // arm, which would report 24 kHz.
        let (_, _, kokoro_rate) = speech_request("kokoro", drama3_speech(None)).await.unwrap();
        assert_eq!(kokoro_rate, 24_000);
        let (_, _, rate) = speech_request(FISH_DRAMA3_FAMILY, drama3_speech(None))
            .await
            .unwrap();
        assert_eq!(rate, 44_100);
        let mut speech = drama3_speech(None);
        speech.temperature = Some(0.6);
        speech.top_p = Some(0.8);
        speech.top_k = Some(40);
        speech.max_frames = Some(100);
        let (native, samples, rate) = drama3_request(speech).await.unwrap();
        assert_eq!(rate, 44_100);
        assert!(samples.is_empty());
        assert_eq!(native.steps, None);
        assert_eq!(native.seed, Some(0));
        assert_eq!(
            (
                native.temperature,
                native.top_p,
                native.top_k,
                native.max_frames
            ),
            (Some(0.6), Some(0.8), Some(40), Some(100))
        );
        assert_eq!(native.reference_sample_rate, None);
        runtime::native_audio::validate_request(FISH_DRAMA3_FAMILY, &native, &samples).unwrap();
    }

    /// The route's own checks do not decode audio, so the duration, consent
    /// and transcript rules of a clone are the engine validator's.
    #[tokio::test]
    async fn drama3_clone_rules_come_from_the_runtime_validator() {
        let validate = |native: &runtime::native_audio::AudioRequest, samples: &[f32]| {
            runtime::native_audio::validate_request(FISH_DRAMA3_FAMILY, native, samples)
        };
        let two_seconds = reference_wav(24_000, 2.0);
        let mut speech = drama3_speech(Some(&two_seconds));
        speech.consent = true;
        speech.reference_text = Some("The reference.".into());
        let (native, samples, rate) = drama3_request(speech.clone()).await.unwrap();
        // The reference keeps its own rate; the runner resamples it.
        assert_eq!((rate, native.reference_sample_rate), (44_100, Some(24_000)));
        validate(&native, &samples).unwrap();

        let mut no_consent = speech.clone();
        no_consent.consent = false;
        let (native, samples, _) = drama3_request(no_consent).await.unwrap();
        assert!(validate(&native, &samples).unwrap_err().contains("consent"));

        let mut no_text = speech.clone();
        no_text.reference_text = None;
        let (native, samples, _) = drama3_request(no_text).await.unwrap();
        assert!(validate(&native, &samples)
            .unwrap_err()
            .contains("reference_text"));

        let short = reference_wav(24_000, 0.5);
        let mut too_short = speech;
        too_short.reference_audio = Some(short);
        let (native, samples, _) = drama3_request(too_short).await.unwrap();
        assert!(validate(&native, &samples)
            .unwrap_err()
            .contains("1 through 30 seconds"));
    }

    /// Every other family would ignore a sampler field, so the shared guard
    /// refuses it. The route relies on this for the providers it wraps.
    #[test]
    fn other_families_refuse_the_drama3_sampler_fields() {
        let native = runtime::native_audio::AudioRequest {
            task: native_task(AudioTask::TextToSpeech),
            text: "hi".into(),
            top_k: Some(10),
            ..runtime::native_audio::AudioRequest::default()
        };
        for family in ["kokoro", MEGATTS3_FAMILY, OMNIVOICE_FAMILY] {
            assert!(
                runtime::native_audio::validate_request(family, &native, &[]).is_err(),
                "{family}"
            );
        }
    }

    /// The HTTP load route reaches `valid_alias` first. A path-shaped name
    /// must never get as far as the store, which would happily open it.
    #[test]
    fn alias_validation_refuses_paths_and_dotfiles() {
        for good in [
            "whisper-base",
            "qwen3-asr-06b-8bit",
            "kokoro-82m-bf16",
            "a.b_c-1",
        ] {
            assert!(valid_alias(good), "{good}");
        }
        for bad in [
            "",
            "/etc",
            "../models/x",
            "a/b",
            "a\\b",
            ".hidden",
            "..",
            "name with space",
            "tab\tname",
            "nul\0byte",
            &"x".repeat(129),
        ] {
            assert!(!valid_alias(bad), "{bad:?}");
        }
    }

    /// Only aliases the bundled catalogs name get a runner. A directory's own
    /// contents never choose which model code runs.
    #[test]
    fn task_comes_from_the_catalog_not_the_directory() {
        assert_eq!(task_of_alias("whisper-base"), Some(AudioTask::SpeechToText));
        assert_eq!(
            task_of_alias("kokoro-82m-bf16"),
            Some(AudioTask::TextToSpeech)
        );
        assert_eq!(task_of_alias("minimax-music3-4bit"), Some(AudioTask::Music));
        assert_eq!(task_of_alias("not-in-any-catalog"), None);
    }

    /// A profile only the app runs is refused before a worker spawns. Every
    /// row the catalog ships today stays attachable.
    #[test]
    fn app_only_profiles_are_refused_and_current_rows_are_not() {
        let catalog = catalog::audio_catalog::AudioCatalog::embedded().unwrap();
        for profile in catalog.entries() {
            assert!(
                app_only_refusal(profile).is_ok(),
                "{}",
                profile.identity.alias
            );
            assert!(refuse_app_only(&profile.identity.alias).is_ok());
        }
        assert!(refuse_app_only("not-in-any-catalog").is_ok());
        let mut app_only = catalog.get("whisper-base").unwrap().clone();
        app_only.capabilities.http_api = false;
        let err = app_only_refusal(&app_only).unwrap_err();
        assert!(err.contains("runs only in the TurboSpark app"), "{err}");
    }

    /// The catalog labels Parakeet by where the runtime puts its encoder,
    /// which carries the run. Changing the default placement must change
    /// the label with it.
    #[test]
    fn parakeet_catalog_placement_matches_the_runtime_encoder_default() {
        let catalog = catalog::audio_catalog::AudioCatalog::embedded().unwrap();
        let defaults = [
            (
                "parakeet-redux-coreml",
                format!(
                    "{:?}",
                    runtime::coreml::ParakeetCoreMlOptions::default().encoder_units
                ),
            ),
            (
                "parakeet-unified-en-coreml",
                format!(
                    "{:?}",
                    runtime::coreml::UnifiedOptions::default().encoder_units
                ),
            ),
        ];
        for (alias, units) in defaults {
            let profile = catalog.get(alias).unwrap();
            let expected = match units.as_str() {
                "CpuAndGpu" => "gpu",
                "CpuAndNeuralEngine" => "ane",
                "CpuOnly" => "cpu",
                other => panic!("set the {alias} catalog compute for encoder units {other}"),
            };
            assert_eq!(
                profile.capabilities.compute.as_deref(),
                Some(expected),
                "{alias}"
            );
        }
    }

    #[test]
    fn qwen_gets_language_names_for_iso_codes() {
        assert_eq!(qwen_language_name("en"), "English");
        assert_eq!(qwen_language_name("ZH"), "Chinese");
        assert_eq!(qwen_language_name("fil"), "Filipino");
        // Names and unknown codes are left for the model to accept or refuse.
        assert_eq!(qwen_language_name("English"), "English");
        assert_eq!(qwen_language_name("xx"), "xx");
    }

    #[test]
    fn attaching_a_missing_directory_fails_without_a_worker() {
        let provider = RealAudioProvider {
            store: None,
            workers: Arc::new(Mutex::new(HashMap::new())),
        };
        let err = provider
            .attach("/definitely/not/a/model/dir", AudioTask::SpeechToText)
            .unwrap_err();
        assert!(
            err.contains("neither a directory nor an installed audio alias"),
            "{err}"
        );
        assert!(provider.models().is_empty());
    }

    #[test]
    fn a_directory_that_is_not_a_model_fails_to_open_and_is_not_registered() {
        let dir = std::env::temp_dir().join("turbospark-audio-real-empty-dir");
        std::fs::create_dir_all(&dir).unwrap();
        let provider = RealAudioProvider {
            store: None,
            workers: Arc::new(Mutex::new(HashMap::new())),
        };
        for task in [
            AudioTask::SpeechToText,
            AudioTask::TextToSpeech,
            AudioTask::Music,
        ] {
            let err = provider.attach(dir.to_str().unwrap(), task).unwrap_err();
            assert!(!err.is_empty());
            assert!(
                provider.models().is_empty(),
                "{task:?} left a worker behind"
            );
        }
    }

    #[test]
    fn an_unknown_model_is_not_found_not_a_panic() {
        let provider = RealAudioProvider {
            store: None,
            workers: Arc::new(Mutex::new(HashMap::new())),
        };
        let rt = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        let err = rt
            .block_on(provider.transcribe(TranscribeRequest {
                model: "nope".into(),
                samples: vec![0.0; 16],
                language: None,
            }))
            .unwrap_err();
        assert!(matches!(err, AudioError::NotFound(_)), "{err:?}");
    }
}
