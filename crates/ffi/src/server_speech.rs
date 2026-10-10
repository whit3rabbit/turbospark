//! Bounded asynchronous handoff from the HTTP runtime to the Swift audio speech / voice cloning worker.

use std::collections::{HashMap, VecDeque};
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use tokio::sync::oneshot;
use turbospark_server::{
    AudioError, AudioModelInfo, AudioProvider, AudioTask, GenerateRequest, GeneratedAudio,
    SpeechRequest, SpeechStream, TranscribeRequest, Transcription,
};

#[derive(serde::Serialize, serde::Deserialize, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct ServerSpeechRequest {
    pub model: String,
    pub text: String,
    pub voice: String,
    pub speed: f32,
    pub language: Option<String>,
    pub reference_audio: Option<String>,
    pub reference_text: Option<String>,
    pub instruction: Option<String>,
}

#[derive(serde::Serialize)]
#[serde(tag = "kind", rename_all = "camelCase")]
pub enum SpeechEvent {
    Start {
        id: u64,
        request: ServerSpeechRequest,
    },
    Cancel {
        id: u64,
    },
}

const MAX_QUEUED_TEXT_BYTES: usize = 1024 * 1024;

struct Pending {
    reply: oneshot::Sender<Result<(Vec<f32>, u32), AudioError>>,
}

#[derive(Default)]
struct State {
    model: Option<String>,
    next_id: u64,
    events: VecDeque<SpeechEvent>,
    pending: HashMap<u64, Pending>,
    stopped: bool,
}

#[derive(Clone, Default)]
pub struct SpeechBridge {
    state: Arc<Mutex<State>>,
}

struct CancelOnDrop {
    state: Arc<Mutex<State>>,
    id: u64,
}

impl State {
    fn withdraw(&mut self, id: u64) {
        let queued = self.events.iter().position(
            |event| matches!(event, SpeechEvent::Start { id: queued, .. } if *queued == id),
        );
        match queued {
            Some(index) => {
                self.events.remove(index);
            }
            None => self.events.push_back(SpeechEvent::Cancel { id }),
        }
    }

    fn queued_text_bytes(&self) -> usize {
        self.events
            .iter()
            .map(|event| match event {
                SpeechEvent::Start { request, .. } => request.text.len(),
                SpeechEvent::Cancel { .. } => 0,
            })
            .sum()
    }
}

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if state.pending.remove(&self.id).is_some() && !state.stopped {
            state.withdraw(self.id);
        }
    }
}

impl SpeechBridge {
    pub fn attach(&self, model: String) -> Result<(), String> {
        if model.trim().is_empty() {
            return Err("speech model id must not be empty".into());
        }
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if state.stopped {
            return Err("server has stopped".into());
        }
        if state.model.is_some() {
            return Err("detach the current speech model first".into());
        }
        state.model = Some(model);
        Ok(())
    }

    pub fn detach(&self) -> Result<(), String> {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if state.model.take().is_none() {
            return Err("no speech model is attached".into());
        }
        Self::cancel_all(&mut state);
        Ok(())
    }

    pub fn stop(&self) {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        state.stopped = true;
        state.model = None;
        Self::cancel_all(&mut state);
    }

    fn cancel_all(state: &mut State) {
        let ids: Vec<_> = state.pending.keys().copied().collect();
        for id in ids {
            if let Some(pending) = state.pending.remove(&id) {
                let _ = pending.reply.send(Err(AudioError::Cancelled));
            }
            state.withdraw(id);
        }
    }

    pub fn drain(&self, max: usize) -> Vec<SpeechEvent> {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let count = if max == 0 {
            state.events.len()
        } else {
            max.min(state.events.len())
        };
        state.events.drain(..count).collect()
    }

    pub fn complete(
        &self,
        id: u64,
        result: Result<(Vec<f32>, u32), AudioError>,
    ) -> Result<(), String> {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let pending = state
            .pending
            .remove(&id)
            .ok_or("speech request is no longer active")?;
        pending
            .reply
            .send(result)
            .map_err(|_| "speech client disconnected".to_string())
    }

    pub fn model(&self) -> Option<String> {
        self.state
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .model
            .clone()
    }
}

impl AudioProvider for SpeechBridge {
    fn models(&self) -> Vec<AudioModelInfo> {
        self.model()
            .map(|id| AudioModelInfo {
                id,
                task: AudioTask::TextToSpeech,
            })
            .into_iter()
            .collect()
    }

    fn transcribe(
        &self,
        _request: TranscribeRequest,
    ) -> Pin<Box<dyn Future<Output = Result<Transcription, AudioError>> + Send + '_>> {
        Box::pin(async {
            Err(AudioError::Unsupported(
                "transcription is not supported by speech bridge".into(),
            ))
        })
    }

    fn generate(
        &self,
        _request: GenerateRequest,
    ) -> Pin<Box<dyn Future<Output = Result<GeneratedAudio, AudioError>> + Send + '_>> {
        Box::pin(async {
            Err(AudioError::Unsupported(
                "music generation is not supported by speech bridge".into(),
            ))
        })
    }

    fn speak(
        &self,
        request: SpeechRequest,
    ) -> Pin<Box<dyn Future<Output = Result<SpeechStream, AudioError>> + Send + '_>> {
        let state = Arc::clone(&self.state);
        Box::pin(async move {
            let (reply, receive) = oneshot::channel();
            let id = {
                let mut locked = state.lock().unwrap_or_else(|e| e.into_inner());
                if locked.stopped || locked.model.as_deref() != Some(request.model.as_str()) {
                    return Err(AudioError::NotFound(format!(
                        "speech model {:?} is not attached",
                        request.model
                    )));
                }
                if locked.pending.len() >= 4
                    || locked.events.len() >= 64
                    || locked.queued_text_bytes() + request.text.len() > MAX_QUEUED_TEXT_BYTES
                {
                    return Err(AudioError::Busy);
                }
                locked.next_id = locked.next_id.wrapping_add(1);
                let id = locked.next_id;
                locked.pending.insert(id, Pending { reply });
                locked.events.push_back(SpeechEvent::Start {
                    id,
                    request: ServerSpeechRequest {
                        model: request.model,
                        text: request.text,
                        voice: request.voice,
                        speed: request.speed,
                        language: request.language,
                        reference_audio: request.reference_audio,
                        reference_text: request.reference_text,
                        instruction: request.instruction,
                    },
                });
                id
            };
            let guard = CancelOnDrop { state, id };
            let result = receive.await.unwrap_or(Err(AudioError::Cancelled))?;
            drop(guard);
            let (samples, sample_rate) = result;
            let (chunks_tx, chunks_rx) = tokio::sync::mpsc::channel(2);
            let _ = chunks_tx.send(Ok(samples)).await;
            Ok(SpeechStream {
                sample_rate,
                chunks: chunks_rx,
            })
        })
    }
}
