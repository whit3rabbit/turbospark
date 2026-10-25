//! Live transcription over a streaming speech model (Parakeet Unified on
//! Core ML), for `ts_audio_stream_*`.
//!
//! Separate from [`super::AudioSession`] on purpose. That session takes the
//! process-wide `HeavyWorkGuard` at open and for every run, which chat
//! generation also holds for a whole reply. A recording can last hours, and
//! holding the guard that long would block chat; taking it per step would
//! stall live text behind any queued job. This model never touches Metal:
//! the log-mel and the RNNT steps run on the CPU and the int8 encoder on the
//! Neural Engine (GPU placement is refused at open). Memory is admitted once
//! at open, and the stream state is bounded to about one window of audio.

use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

use turbospark_coreml::Tensor;

use crate::coreml::parakeet_unified::{ParakeetUnifiedCoreMl, UnifiedOptions};
use crate::coreml::parakeet_unified_stream::{StreamUpdate, StreamingModel, UnifiedStream};
use crate::coreml::{detect_layout, CoreMlLayout};

pub struct AudioStreamSession {
    model: ParakeetUnifiedCoreMl,
    stream: Mutex<UnifiedStream<(Tensor, Tensor)>>,
    cancelled: AtomicBool,
}

impl std::fmt::Debug for AudioStreamSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AudioStreamSession")
            .field("model", &self.model)
            .finish()
    }
}

impl AudioStreamSession {
    /// Opens a Parakeet Unified install. Anything else is refused before
    /// memory is admitted or a model is loaded.
    pub fn open(path: &Path) -> Result<Self, String> {
        if detect_layout(path) != Some(CoreMlLayout::ParakeetUnified) {
            return Err(format!(
                "{} is not a Parakeet Unified streaming install",
                path.display()
            ));
        }
        // Three times compiled weights plus a working reserve, as the
        // catalog row estimates; no 30-minute PCM term, the stream keeps
        // about one window.
        super::admit_memory(
            super::coreml_weight_bytes(path)?
                .saturating_mul(3)
                .saturating_add(256 << 20),
        )?;
        let model = ParakeetUnifiedCoreMl::open(path, &UnifiedOptions::from_env()?)?;
        let stream = UnifiedStream::new(model.geometry());
        Ok(Self {
            model,
            stream: Mutex::new(stream),
            cancelled: AtomicBool::new(false),
        })
    }

    pub fn family(&self) -> &'static str {
        CoreMlLayout::ParakeetUnified.family()
    }

    /// Chunk plus look-ahead: how far committed text trails the audio,
    /// before compute time.
    pub fn latency_seconds(&self) -> f64 {
        self.model.geometry().latency_seconds()
    }

    /// Adds 16 kHz mono audio and returns what it committed. One call runs
    /// at a time; a concurrent one is refused rather than queued, so a
    /// caller that falls behind learns it instead of building a backlog.
    pub fn append(&self, pcm: &[f32]) -> Result<StreamUpdate, String> {
        let mut stream = self.lock()?;
        stream.append(&self.model, pcm, &|| self.cancelled.load(Ordering::Acquire))
    }

    /// Commits the held-back tail and closes the last word. The stream then
    /// starts again at time zero, so a caller can begin a new segment after
    /// a gap in its audio.
    pub fn finish(&self) -> Result<StreamUpdate, String> {
        let mut stream = self.lock()?;
        stream.finish(&self.model, &|| self.cancelled.load(Ordering::Acquire))
    }

    /// Stops an append or finish in flight before its next encoder step.
    /// Terminal: the stream refuses further work, because a cancelled step
    /// leaves it partway through a window.
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
    }

    fn lock(&self) -> Result<std::sync::MutexGuard<'_, UnifiedStream<(Tensor, Tensor)>>, String> {
        if self.cancelled.load(Ordering::Acquire) {
            return Err("stream was cancelled".into());
        }
        self.stream.try_lock().map_err(|e| match e {
            std::sync::TryLockError::WouldBlock => {
                "stream is busy: another append or finish is running".to_string()
            }
            std::sync::TryLockError::Poisoned(_) => "stream failed during a previous call".into(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn open_refuses_a_directory_that_is_not_unified() {
        let dir = std::env::temp_dir().join(format!(
            "ts-audio-stream-open-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let err = AudioStreamSession::open(&dir).unwrap_err();
        assert!(err.contains("not a Parakeet Unified"), "{err}");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn session_is_shareable_across_threads() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<AudioStreamSession>();
    }
}
