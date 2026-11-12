//! Live meeting transcription with Frozen E: Nemotron-3 diarization coupled
//! to Multitalker Parakeet, for the `ts_multitalker_stream_*` calls.
//!
//! Like [`super::AudioStreamSession`] and for the same reason, this is not an
//! [`super::AudioSession`]: a recording can last hours, so it must not hold
//! the process-wide heavy-work guard that chat generation also holds, and it
//! takes none. It runs on the CPU (the Metal encoders stay opt-in until their
//! gates pass) and never touches the GPU.
//!
//! The models and the [`LiveSession`] that borrows them live on one worker
//! thread for the stream's life, and calls reach it over a channel. That
//! keeps the borrow out of the public type, lets `cancel` work from any
//! thread, and means a panic in a step ends the worker instead of unwinding
//! across the caller. Memory is admitted once at open from the two
//! checkpoints' sizes. The estimate is not a measurement: no peak footprint
//! has been recorded for this pipeline yet.

use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, SyncSender};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use audio::models::stt::multitalker_asr::scheduler::{CoupledConfig, LiveSession};
use audio::models::stt::multitalker_asr::MultitalkerAsr;
use audio::models::vad::nemotron_diarization::NemotronDiarization;
use serde::Serialize;

/// One speaker segment that is new or changed. A consumer keys on
/// `(speaker, index)` and replaces what it holds: a speaker's last segment
/// grows while they keep talking.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SpeakerSegment {
    pub speaker: String,
    pub index: usize,
    pub start_seconds: f64,
    pub end_seconds: f64,
    pub text: String,
}

/// What one `append` or `finish` changed.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SpeakerStreamUpdate {
    /// Segments new or changed by this call, in speaker order.
    pub segments: Vec<SpeakerSegment>,
    /// Speakers the last step decoded (the diarizer's active set).
    pub active_speakers: Vec<usize>,
    /// Audio received so far.
    pub received_seconds: f64,
    /// Audio stepped through the models so far. It trails
    /// `received_seconds` by up to one chunk plus the frontend's right context.
    pub processed_seconds: f64,
    pub is_final: bool,
}

enum Op {
    Append(Vec<f32>),
    Finish,
}

struct Request {
    op: Op,
    reply: SyncSender<Result<SpeakerStreamUpdate, String>>,
}

pub struct MultitalkerStreamSession {
    requests: Mutex<Option<SyncSender<Request>>>,
    worker: Mutex<Option<JoinHandle<()>>>,
    /// One call at a time; a concurrent one is refused, not queued.
    busy: AtomicBool,
    cancelled: Arc<AtomicBool>,
    latency_seconds: f64,
}

impl std::fmt::Debug for MultitalkerStreamSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MultitalkerStreamSession")
            .field("latency_seconds", &self.latency_seconds)
            .finish()
    }
}

/// Resets `busy` however the call ends.
struct Busy<'a>(&'a AtomicBool);

impl Drop for Busy<'_> {
    fn drop(&mut self) {
        self.0.store(false, Ordering::Release);
    }
}

fn safetensors_bytes(dir: &Path) -> Result<u64, String> {
    std::fs::metadata(dir.join("model.safetensors"))
        .map(|m| m.len())
        .map_err(|e| format!("{}: no model.safetensors: {e}", dir.display()))
}

impl MultitalkerStreamSession {
    /// Opens the converted Multitalker snapshot (`asr_dir`) and the
    /// Nemotron-3 diarization snapshot (`diar_dir`). Both load f32 weights
    /// onto the host, so the admission estimate is their file sizes plus a
    /// quarter for tables and caches and a fixed working reserve.
    pub fn open(asr_dir: &Path, diar_dir: &Path) -> Result<Self, String> {
        let weights = safetensors_bytes(asr_dir)?.saturating_add(safetensors_bytes(diar_dir)?);
        super::admit_memory(
            weights
                .saturating_add(weights / 4)
                .saturating_add(512 << 20),
        )?;
        let cancelled = Arc::new(AtomicBool::new(false));
        let (requests, inbox) = mpsc::sync_channel::<Request>(0);
        let (ready_tx, ready_rx) = mpsc::sync_channel::<Result<f64, String>>(1);
        let (asr_dir, diar_dir) = (asr_dir.to_path_buf(), diar_dir.to_path_buf());
        let worker = {
            let cancelled = Arc::clone(&cancelled);
            std::thread::Builder::new()
                .name("multitalker-stream".into())
                .spawn(move || work(&asr_dir, &diar_dir, cancelled, ready_tx, inbox))
                .map_err(|e| format!("cannot start the stream worker: {e}"))?
        };
        let latency_seconds = ready_rx
            .recv()
            .map_err(|_| "the stream worker stopped while loading".to_string())??;
        Ok(Self {
            requests: Mutex::new(Some(requests)),
            worker: Mutex::new(Some(worker)),
            busy: AtomicBool::new(false),
            cancelled,
            latency_seconds,
        })
    }

    pub fn family(&self) -> &'static str {
        "multitalker_asr"
    }

    /// Chunk plus the frontend's right context: how far committed text
    /// trails the audio, before compute time.
    pub fn latency_seconds(&self) -> f64 {
        self.latency_seconds
    }

    /// Adds 16 kHz mono audio, 1 to [`super::MAX_PCM_CHUNK`] samples, and
    /// returns what it changed.
    pub fn append(&self, pcm: &[f32]) -> Result<SpeakerStreamUpdate, String> {
        if pcm.is_empty() || pcm.len() > super::MAX_PCM_CHUNK {
            return Err(format!(
                "a PCM chunk holds 1 to {} samples, got {}",
                super::MAX_PCM_CHUNK,
                pcm.len()
            ));
        }
        self.call(Op::Append(pcm.to_vec()))
    }

    /// Steps the held-back tail and ends the recording. The session is then
    /// closed: open a new one for the next recording.
    pub fn finish(&self) -> Result<SpeakerStreamUpdate, String> {
        self.call(Op::Finish)
    }

    /// Stops a call in flight before its next step, from any thread.
    /// Terminal: the models are partway through a recording.
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
    }

    fn call(&self, op: Op) -> Result<SpeakerStreamUpdate, String> {
        if self.cancelled.load(Ordering::Acquire) {
            return Err("cancelled".into());
        }
        if self
            .busy
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return Err("stream is busy: another append or finish is running".into());
        }
        let _busy = Busy(&self.busy);
        let (reply, answer) = mpsc::sync_channel(1);
        let sender = self
            .requests
            .lock()
            .map_err(|_| "stream failed during a previous call".to_string())?
            .clone()
            .ok_or("stream is closed")?;
        sender
            .send(Request { op, reply })
            .map_err(|_| "the stream worker stopped".to_string())?;
        answer
            .recv()
            .map_err(|_| "the stream worker stopped during the call".to_string())?
    }
}

impl Drop for MultitalkerStreamSession {
    fn drop(&mut self) {
        // Cancel first so a call in flight on another thread returns, then
        // close the channel so the worker's loop ends.
        self.cancelled.store(true, Ordering::Release);
        if let Ok(mut requests) = self.requests.lock() {
            requests.take();
        }
        if let Ok(mut worker) = self.worker.lock() {
            if let Some(handle) = worker.take() {
                let _ = handle.join();
            }
        }
    }
}

fn work(
    asr_dir: &Path,
    diar_dir: &Path,
    cancelled: Arc<AtomicBool>,
    ready: SyncSender<Result<f64, String>>,
    inbox: Receiver<Request>,
) {
    let loaded = MultitalkerAsr::open(asr_dir)
        .map_err(|e| format!("Multitalker: {e}"))
        .and_then(|asr| {
            NemotronDiarization::load(diar_dir)
                .map(|diar| (asr, diar))
                .map_err(|e| format!("Nemotron-3 diarization: {e}"))
        });
    let (asr, diar) = match loaded {
        Ok(models) => models,
        Err(e) => {
            let _ = ready.send(Err(e));
            return;
        }
    };
    let setup = diar
        .coupled(diar.config().modules.multitalker_coupled(asr.chunk_rows()))
        .map_err(|e| format!("Nemotron-3 diarization: {e}"))
        .and_then(|coupled| {
            let cancel = Arc::clone(&cancelled);
            LiveSession::new(&asr, coupled, CoupledConfig::frozen_e())
                .map(|s| s.with_cancel(Box::new(move || cancel.load(Ordering::Acquire))))
                .map_err(|e| format!("Multitalker: {e}"))
        });
    let mut session = match setup {
        Ok(session) => session,
        Err(e) => {
            let _ = ready.send(Err(e));
            return;
        }
    };
    let rate = f64::from(asr.config().sample_rate);
    let hop_seconds = asr.config().hop_samples as f64 / rate;
    let chunk_seconds = asr.shift_frames() as f64 * hop_seconds;
    let right_context = (asr.config().n_fft / 2) as f64 / rate;
    let _ = ready.send(Ok(chunk_seconds + right_context));

    while let Ok(Request { op, reply }) = inbox.recv() {
        let is_final = matches!(op, Op::Finish);
        let stepped = match op {
            Op::Append(pcm) => session.append(&pcm),
            Op::Finish => session.finish(),
        };
        let result = stepped
            .map(|reports| SpeakerStreamUpdate {
                segments: session
                    .take_segment_updates()
                    .into_iter()
                    .map(|u| SpeakerSegment {
                        speaker: u.speaker,
                        index: u.index,
                        start_seconds: u.start_time,
                        end_seconds: u.end_time,
                        text: u.words,
                    })
                    .collect(),
                active_speakers: reports.last().map(|r| r.active.clone()).unwrap_or_default(),
                received_seconds: session.samples_received() as f64 / rate,
                processed_seconds: session.processed_frames() as f64 * hop_seconds,
                is_final,
            })
            .map_err(|e| {
                let message = e.to_string();
                // The ABI maps this exact string to its cancelled status.
                if message.ends_with("cancelled") {
                    "cancelled".to_string()
                } else {
                    message
                }
            });
        let _ = reply.send(result);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn session_is_shareable_across_threads() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<MultitalkerStreamSession>();
    }

    #[test]
    fn open_refuses_a_directory_without_weights_before_admitting_memory() {
        let dir = std::env::temp_dir().join(format!("ts-mt-stream-open-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let err = MultitalkerStreamSession::open(&dir, &dir).unwrap_err();
        assert!(err.contains("no model.safetensors"), "{err}");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// The whole Frozen E pipeline through the session API on the traced IS1004a
    /// clip, in uneven PCM calls: the segments, applied as a consumer would,
    /// equal NeMo's SegLST for that clip. Gated on the three checkpoint
    /// variables of the audio crate's tests.
    #[test]
    fn streams_the_traced_clip_to_nemo_seglst() {
        let (Ok(asr), Ok(diar), Ok(trace)) = (
            std::env::var("TURBOSPEECH_MULTITALKER_MODEL"),
            std::env::var("TURBOSPEECH_NEMOTRON_MODEL"),
            std::env::var("TURBOSPEECH_FROZEN_E_TRACE"),
        ) else {
            eprintln!("skipping: checkpoint variables unset");
            return;
        };
        let trace = std::path::PathBuf::from(trace);
        let meta: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(trace.join("meta.json")).unwrap())
                .unwrap();
        let wav = audio::read_wav_f32(Path::new(meta["wav"].as_str().unwrap())).unwrap();
        let session = MultitalkerStreamSession::open(Path::new(&asr), Path::new(&diar)).unwrap();
        assert!((session.latency_seconds() - 1.136).abs() < 1e-9);

        let mut held = std::collections::BTreeMap::new();
        let mut apply = |mut update: SpeakerStreamUpdate| {
            for s in std::mem::take(&mut update.segments) {
                held.insert(
                    (s.speaker, s.index),
                    (s.start_seconds, s.end_seconds, s.text),
                );
            }
            update
        };
        let (mut at, mut sizes) = (0, [7_919usize, 160, 32_000, 1].iter().cycle());
        let mut last = None;
        while at < wav.samples.len() {
            let n = (*sizes.next().unwrap()).min(wav.samples.len() - at);
            last = Some(apply(session.append(&wav.samples[at..at + n]).unwrap()));
            at += n;
        }
        let last = last.unwrap();
        assert!(!last.is_final);
        assert!(last.processed_seconds <= last.received_seconds);
        let fin = apply(session.finish().unwrap());
        assert!(fin.is_final);
        assert!((fin.received_seconds - wav.samples.len() as f64 / 16_000.0).abs() < 1e-9);
        assert!(
            session.append(&[0.0]).is_err(),
            "a finished stream is closed"
        );

        let want: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(trace.join("seglst.json")).unwrap())
                .unwrap();
        let mut want: Vec<(String, f64, f64, String)> = want
            .as_array()
            .unwrap()
            .iter()
            .map(|e| {
                (
                    e["speaker"].as_str().unwrap().to_string(),
                    e["start_time"].as_f64().unwrap(),
                    e["end_time"].as_f64().unwrap(),
                    e["words"].as_str().unwrap().to_string(),
                )
            })
            .collect();
        let mut got: Vec<(String, f64, f64, String)> = held
            .into_iter()
            .map(|((speaker, _), (start, end, text))| (speaker, start, end, text))
            .collect();
        want.sort_by(|a, b| a.partial_cmp(b).unwrap());
        got.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(got, want);
    }

    /// A second call while one runs is refused, and cancel is terminal.
    #[test]
    fn concurrent_call_is_refused_and_cancel_is_terminal() {
        let (Ok(asr), Ok(diar)) = (
            std::env::var("TURBOSPEECH_MULTITALKER_MODEL"),
            std::env::var("TURBOSPEECH_NEMOTRON_MODEL"),
        ) else {
            eprintln!("skipping: checkpoint variables unset");
            return;
        };
        let session = MultitalkerStreamSession::open(Path::new(&asr), Path::new(&diar)).unwrap();
        session.busy.store(true, Ordering::Release);
        let err = session.append(&[0.0; 160]).unwrap_err();
        assert!(err.contains("is busy"), "{err}");
        session.busy.store(false, Ordering::Release);
        assert!(session.append(&[]).is_err());
        session.cancel();
        assert_eq!(session.append(&[0.0; 160]).unwrap_err(), "cancelled");
    }
}
