//! Bounded asynchronous handoff from the HTTP runtime to the Swift image worker.

use std::collections::{HashMap, VecDeque};
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use tokio::sync::oneshot;
use turbospark_server::{ImageError, ImageGenerateRequest, ImageProvider};

#[derive(serde::Serialize)]
#[serde(tag = "kind", rename_all = "camelCase")]
pub enum ImageEvent {
    Start {
        id: u64,
        request: ImageGenerateRequest,
    },
    Cancel {
        id: u64,
    },
}

struct Pending {
    reply: oneshot::Sender<Result<Vec<u8>, ImageError>>,
}

#[derive(Default)]
struct State {
    model: Option<String>,
    next_id: u64,
    events: VecDeque<ImageEvent>,
    pending: HashMap<u64, Pending>,
    stopped: bool,
}

#[derive(Clone, Default)]
pub struct ImageBridge {
    state: Arc<Mutex<State>>,
}

struct CancelOnDrop {
    state: Arc<Mutex<State>>,
    id: u64,
}

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if state.pending.remove(&self.id).is_some() && !state.stopped {
            state.events.push_back(ImageEvent::Cancel { id: self.id });
        }
    }
}

impl ImageBridge {
    pub fn attach(&self, model: String) -> Result<(), String> {
        if model.trim().is_empty() {
            return Err("image model id must not be empty".into());
        }
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if state.stopped {
            return Err("server has stopped".into());
        }
        if state.model.is_some() {
            return Err("detach the current image model first".into());
        }
        state.model = Some(model);
        Ok(())
    }

    pub fn detach(&self) -> Result<(), String> {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if state.model.take().is_none() {
            return Err("no image model is attached".into());
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
                let _ = pending.reply.send(Err(ImageError::Cancelled));
            }
            state.events.push_back(ImageEvent::Cancel { id });
        }
    }

    pub fn drain(&self, max: usize) -> Vec<ImageEvent> {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let count = if max == 0 {
            state.events.len()
        } else {
            max.min(state.events.len())
        };
        state.events.drain(..count).collect()
    }

    pub fn complete(&self, id: u64, result: Result<Vec<u8>, ImageError>) -> Result<(), String> {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let pending = state
            .pending
            .remove(&id)
            .ok_or("image request is no longer active")?;
        pending
            .reply
            .send(result)
            .map_err(|_| "image client disconnected".to_string())
    }

    pub fn model(&self) -> Option<String> {
        self.state
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .model
            .clone()
    }
}

impl ImageProvider for ImageBridge {
    fn models(&self) -> Vec<String> {
        self.model().into_iter().collect()
    }

    fn generate(
        &self,
        request: ImageGenerateRequest,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<u8>, ImageError>> + Send + '_>> {
        let state = Arc::clone(&self.state);
        Box::pin(async move {
            let (reply, receive) = oneshot::channel();
            let id = {
                let mut locked = state.lock().unwrap_or_else(|e| e.into_inner());
                if locked.stopped || locked.model.as_deref() != Some(request.model.as_str()) {
                    return Err(ImageError::Cancelled);
                }
                // A disconnected client can leave a Start and Cancel pair
                // until Swift next polls. Bound that backlog as well as work.
                if locked.pending.len() >= 4 || locked.events.len() >= 64 {
                    return Err(ImageError::Busy);
                }
                locked.next_id = locked.next_id.wrapping_add(1);
                let id = locked.next_id;
                locked.pending.insert(id, Pending { reply });
                locked.events.push_back(ImageEvent::Start { id, request });
                id
            };
            let guard = CancelOnDrop { state, id };
            let result = receive.await.unwrap_or(Err(ImageError::Cancelled));
            drop(guard);
            result
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request() -> ImageGenerateRequest {
        ImageGenerateRequest {
            model: "z-test".into(),
            prompt: "a fox".into(),
            width: 512,
            height: 512,
            seed: 7,
        }
    }

    #[tokio::test]
    async fn queue_is_bounded_and_detach_cancels_waiters() {
        let bridge = ImageBridge::default();
        bridge.attach("z-test".into()).unwrap();
        let mut jobs = Vec::new();
        for _ in 0..4 {
            let worker = bridge.clone();
            jobs.push(tokio::spawn(
                async move { worker.generate(request()).await },
            ));
        }
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        assert_eq!(bridge.drain(0).len(), 4);
        assert_eq!(bridge.generate(request()).await, Err(ImageError::Busy));
        bridge.detach().unwrap();
        assert_eq!(bridge.drain(0).len(), 4);
        for job in jobs {
            assert_eq!(job.await.unwrap(), Err(ImageError::Cancelled));
        }
        assert!(bridge.models().is_empty());
    }

    #[tokio::test]
    async fn dropped_client_emits_cancel_and_stop_refuses_work() {
        let bridge = ImageBridge::default();
        bridge.attach("z-test".into()).unwrap();
        let worker = bridge.clone();
        let job = tokio::spawn(async move { worker.generate(request()).await });
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        let id = match bridge.drain(0).remove(0) {
            ImageEvent::Start { id, .. } => id,
            ImageEvent::Cancel { .. } => panic!("expected start"),
        };
        job.abort();
        let _ = job.await;
        assert!(
            matches!(bridge.drain(0).as_slice(), [ImageEvent::Cancel { id: cancelled }] if *cancelled == id)
        );
        assert!(bridge.complete(id, Ok(vec![1])).is_err());
        bridge.stop();
        assert_eq!(bridge.generate(request()).await, Err(ImageError::Cancelled));
    }

    #[tokio::test]
    async fn undrained_events_refuse_more_work() {
        let bridge = ImageBridge::default();
        bridge.attach("z-test".into()).unwrap();
        {
            let mut state = bridge.state.lock().unwrap();
            for id in 0..64 {
                state.events.push_back(ImageEvent::Cancel { id });
            }
        }
        assert_eq!(bridge.generate(request()).await, Err(ImageError::Busy));
    }
}
