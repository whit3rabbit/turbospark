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
        text: String,
        speed: f32,
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
                text,
                speed,
                chunks,
            } => {
                let cancel = runtime::native_audio::AudioCancel::default();
                let signal = cancel.clone();
                let outcome = engine.execute(
                    AudioRequest {
                        task: native_task(AudioTask::TextToSpeech),
                        text,
                        speed,
                        ..AudioRequest::default()
                    },
                    vec![],
                    cancel,
                    |event| {
                        if let AudioEvent::Pcm(pcm) = event {
                            if chunks.blocking_send(Ok(pcm)).is_err() {
                                signal.cancel();
                            }
                        }
                    },
                );
                if let Err(e) = outcome {
                    let _ = chunks.blocking_send(Err(AudioError::Invalid(e)));
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
    let (ready_tx, ready_rx) = std::sync::mpsc::channel::<Result<(), String>>();
    std::thread::Builder::new()
        .name(format!("audio-{id}"))
        .spawn(move || match open_engine(task, &dir) {
            Ok(engine) => {
                let _ = ready_tx.send(Ok(()));
                run(engine, rx);
            }
            Err(e) => {
                let _ = ready_tx.send(Err(e));
            }
        })
        .map_err(|e| format!("could not start the audio worker: {e}"))?;
    ready_rx
        .recv()
        .map_err(|_| "the audio worker exited while opening the model".to_string())??;
    Ok(Arc::new(Worker {
        info: AudioModelInfo {
            id: id.to_string(),
            task,
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
            request
                .voice
                .parse::<Voice>()
                .map_err(|e| AudioError::Invalid(e.to_string()))?;
            // Two chunks of look-ahead: synthesis stays a step ahead of the
            // socket without buffering a whole paragraph.
            let (chunks, rx) = mpsc::channel(2);
            worker.submit(Work::Speak {
                text: request.text,
                speed: request.speed,
                chunks,
            })?;
            Ok(SpeechStream {
                sample_rate: turbospark_audio::tts::kokoro::OUTPUT_SAMPLE_RATE,
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
