//! Live voice isolation: a realtime checkpoint streamed as audio arrives.
//!
//! The network runs on a worker thread that owns it (the device backends are
//! not `Send`); a push hands its samples over and waits for the released
//! output. Two networks sit behind the stream's analysis and synthesis:
//!
//! - the port (`denoiser_realtime.safetensors`) on Metal or the CPU, every
//!   projection and scan a device call of its own;
//! - a stateful Core ML step (`VoiceIsolatorStep.mlmodelc` or `.mlpackage`
//!   beside the weights, two frames a call, KV caches and Mamba states held
//!   in `MLState`), exported by `crates/audio/tools/export_voice_isolator_coreml.py`.
//!
//! A stream takes no heavy-work guard, like live transcription. Every push
//! reports how long it took against the audio it carried, so a host can stop
//! feeding cleaned audio onward when the engine falls behind real time.

use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver, SyncSender};
use std::thread::JoinHandle;
use std::time::Instant;

use audio::fft::ComplexF32;
use audio::sts::voice_isolator::stream::StreamStep;
use audio::sts::{VoiceIsolator, VoiceIsolatorStream};
use audio::{Result as AudioResult, SpeechError};
use turbospark_coreml::{ComputeUnits, CoreMlModel, CoreMlState, LoadOptions, Tensor};

use crate::voice_isolator::{self as port, Device};

/// The realtime checkpoint's file.
pub const REALTIME_WEIGHTS: &str = "denoiser_realtime.safetensors";
/// The Core ML step's program names, compiled first.
pub const COREML_PROGRAMS: [&str; 2] =
    ["VoiceIsolatorStep.mlmodelc", "VoiceIsolatorStep.mlpackage"];
/// Frames the Core ML step takes per call (its fixed input shape).
const COREML_FRAMES: usize = 2;
/// Past key slots the Core ML step holds (the trained window minus one).
const COREML_PAST: usize = 1199;
/// The trained attention window of the realtime checkpoint.
const WINDOW: usize = 1200;
const BINS: usize = 1025;
const HEAD_DIM: usize = 64;

/// Which network a live stream runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Backend {
    /// Core ML when the step is installed and the OS supports state, else Metal.
    Auto,
    CoreMl,
    Metal,
    Cpu,
}

/// One push's result.
#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct LivePush {
    /// Output samples released (48 kHz mono), handed to the caller separately.
    #[serde(skip)]
    pub samples: Vec<f32>,
    pub released: usize,
    /// Wall time of the push.
    pub processing_ms: f64,
    /// The push took longer than the audio it carried.
    pub late: bool,
}

/// The stateful Core ML step (the program's contract is in the export
/// script): `spectrum [1, 2, 1025, 2]`, `rope_cos`/`rope_sin [2, 32]`,
/// `mask [2, 1201]` over 1199 cached slots then the 2 new frames, output
/// `raw [2, 1025, 4]`.
pub struct CoreMlStep {
    model: CoreMlModel,
    state: CoreMlState,
}

impl CoreMlStep {
    pub fn open(program: &Path, units: ComputeUnits) -> Result<Self, String> {
        let dir = program.parent().unwrap_or(Path::new("."));
        let cache = crate::coreml::cache_root(dir);
        let model = CoreMlModel::open(program, &cache, &LoadOptions::new(units))
            .map_err(|e| e.to_string())?;
        let state = model.new_state().map_err(|e| e.to_string())?;
        Ok(Self { model, state })
    }

    /// Rotary tables of absolute frames `first..first + 2`, in f64 so a long
    /// session keeps exact angles.
    fn rope(first: usize) -> (Vec<f32>, Vec<f32>) {
        let half = HEAD_DIM / 2;
        let mut cos = Vec::with_capacity(COREML_FRAMES * half);
        let mut sin = Vec::with_capacity(COREML_FRAMES * half);
        for t in first..first + COREML_FRAMES {
            for d in 0..half {
                let angle = t as f64 * 10_000f64.powf(-(d as f64) / half as f64);
                cos.push(angle.cos() as f32);
                sin.push(angle.sin() as f32);
            }
        }
        (cos, sin)
    }

    /// The additive mask: cached slot `j` holds frame `first - 1199 + j`; a
    /// new frame sees cached frames that exist and fall inside the window,
    /// and the new frames up to itself.
    fn mask(first: usize) -> Vec<f32> {
        let keys = COREML_PAST + COREML_FRAMES;
        let mut mask = vec![-1e4f32; COREML_FRAMES * keys];
        for i in 0..COREML_FRAMES {
            let q = first + i;
            for j in 0..COREML_PAST {
                let frame = first as isize - COREML_PAST as isize + j as isize;
                if frame >= 0 && (q as isize - frame) < WINDOW as isize {
                    mask[i * keys + j] = 0.0;
                }
            }
            for jj in 0..=i {
                mask[i * keys + COREML_PAST + jj] = 0.0;
            }
        }
        mask
    }
}

impl StreamStep for CoreMlStep {
    fn frames_per_call(&self) -> usize {
        COREML_FRAMES
    }

    fn run(&mut self, first: usize, spectra: &[Vec<ComplexF32>]) -> AudioResult<Vec<f32>> {
        let err = |e: turbospark_coreml::CoreMlError| SpeechError::Unsupported {
            why: format!("Core ML step: {e}"),
        };
        let mut spectrum = Vec::with_capacity(COREML_FRAMES * BINS * 2);
        for frame in spectra {
            for bin in frame {
                spectrum.push(bin.re);
                spectrum.push(bin.im);
            }
        }
        let (cos, sin) = Self::rope(first);
        let inputs = [
            (
                "spectrum",
                Tensor::f32(vec![1, COREML_FRAMES, BINS, 2], spectrum).map_err(err)?,
            ),
            (
                "rope_cos",
                Tensor::f32(vec![COREML_FRAMES, HEAD_DIM / 2], cos).map_err(err)?,
            ),
            (
                "rope_sin",
                Tensor::f32(vec![COREML_FRAMES, HEAD_DIM / 2], sin).map_err(err)?,
            ),
            (
                "mask",
                Tensor::f32(
                    vec![COREML_FRAMES, COREML_PAST + COREML_FRAMES],
                    Self::mask(first),
                )
                .map_err(err)?,
            ),
        ];
        let named: Vec<(&str, &Tensor)> = inputs.iter().map(|(n, t)| (*n, t)).collect();
        let out = self
            .model
            .predict_with_state(&mut self.state, &named)
            .map_err(err)?;
        let raw = out.get("raw").and_then(Tensor::to_f32_vec).ok_or_else(|| {
            SpeechError::Unsupported {
                why: "Core ML step returned no float raw output".into(),
            }
        })?;
        if raw.len() != COREML_FRAMES * BINS * 4 || raw.iter().any(|v| !v.is_finite()) {
            return Err(SpeechError::Unsupported {
                why: "Core ML step output has the wrong size or is not finite".into(),
            });
        }
        Ok(raw)
    }
}

/// The network on the worker thread.
enum Engine {
    Port(Box<VoiceIsolator>, VoiceIsolatorStream),
    CoreMl(Box<CoreMlStep>, VoiceIsolatorStream),
}

impl Engine {
    fn open(dir: &Path, backend: Backend) -> Result<(Self, &'static str), String> {
        let weights = dir.join(REALTIME_WEIGHTS);
        let program = COREML_PROGRAMS
            .iter()
            .map(|p| dir.join(p))
            .find(|p| p.exists());
        let want_coreml = match backend {
            Backend::CoreMl => true,
            Backend::Auto => program.is_some() && turbospark_coreml::os_version().0 >= 15,
            Backend::Metal | Backend::Cpu => false,
        };
        if want_coreml {
            let program =
                program.ok_or("no VoiceIsolatorStep Core ML program beside the weights")?;
            let step = CoreMlStep::open(&program, ComputeUnits::CpuAndNeuralEngine)?;
            // The stream's STFT and bound come from the checkpoint the step was
            // exported from; read its config when the weights are present.
            let (n_fft, hop, bound, lookahead) = match VoiceIsolator::open_file(&weights) {
                Ok(m) => (
                    m.config.n_fft,
                    m.config.hop_length,
                    m.config.correction_bound,
                    m.lookahead_frames(),
                ),
                Err(_) => (2048, 480, 0.5, 2),
            };
            let stream = VoiceIsolatorStream::external(n_fft, hop, bound, lookahead, &step)
                .map_err(|e| e.to_string())?;
            return Ok((Self::CoreMl(Box::new(step), stream), "coreml"));
        }
        let device = if backend == Backend::Cpu {
            Device::Cpu
        } else {
            Device::Metal
        };
        let only = single_file_dir(&weights)?;
        let model = port::open(&only, device);
        let _ = std::fs::remove_dir_all(&only);
        let model = model?;
        let stream = VoiceIsolatorStream::new(&model, 2, None).map_err(|e| e.to_string())?;
        Ok((
            Self::Port(Box::new(model), stream),
            if device == Device::Metal {
                "metal"
            } else {
                "cpu"
            },
        ))
    }

    fn push(&mut self, samples: &[f32]) -> AudioResult<Vec<f32>> {
        match self {
            Engine::Port(m, s) => s.push(m, samples),
            Engine::CoreMl(step, s) => s.push_external(step.as_mut(), samples),
        }
    }

    fn flush(&mut self) -> AudioResult<Vec<f32>> {
        match self {
            Engine::Port(m, s) => s.flush(m),
            Engine::CoreMl(step, s) => s.flush_external(step.as_mut()),
        }
    }
}

/// A directory holding only the realtime weights (symlinked), so the port's
/// `open` cannot pick the offline file beside it.
fn single_file_dir(weights: &Path) -> Result<PathBuf, String> {
    if !weights.is_file() {
        return Err(format!("{} is missing", weights.display()));
    }
    let dir = std::env::temp_dir().join(format!(
        "ts-vi-live-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos())
    ));
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    std::os::unix::fs::symlink(weights, dir.join(REALTIME_WEIGHTS)).map_err(|e| e.to_string())?;
    Ok(dir)
}

enum Message {
    Push(Vec<f32>, SyncSender<Result<Vec<f32>, String>>),
    Flush(SyncSender<Result<Vec<f32>, String>>),
}

/// A live isolation stream at 48 kHz mono.
pub struct LiveDenoiser {
    tx: Option<SyncSender<Message>>,
    worker: Option<JoinHandle<()>>,
    backend: &'static str,
}

impl LiveDenoiser {
    /// Opens the realtime install at `dir` on `backend`.
    pub fn open(dir: &Path, backend: Backend) -> Result<Self, String> {
        let (tx, rx) = mpsc::sync_channel::<Message>(1);
        let (ready, opened) = mpsc::sync_channel(1);
        let dir = dir.to_path_buf();
        let worker = std::thread::Builder::new()
            .name("voice-isolator-live".into())
            .spawn(move || {
                let engine = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    Engine::open(&dir, backend)
                }))
                .unwrap_or_else(|_| Err("opening the live isolator panicked".into()));
                match engine {
                    Ok((engine, name)) => {
                        if ready.send(Ok(name)).is_ok() {
                            run(engine, rx);
                        }
                    }
                    Err(e) => {
                        let _ = ready.send(Err(e));
                    }
                }
            })
            .map_err(|e| e.to_string())?;
        match opened
            .recv()
            .map_err(|_| "live isolator worker stopped while opening".to_string())?
        {
            Ok(backend) => Ok(Self {
                tx: Some(tx),
                worker: Some(worker),
                backend,
            }),
            Err(e) => {
                let _ = worker.join();
                Err(e)
            }
        }
    }

    /// The network in use: `coreml`, `metal` or `cpu`.
    pub fn backend(&self) -> &'static str {
        self.backend
    }

    fn call(
        &self,
        message: impl FnOnce(SyncSender<Result<Vec<f32>, String>>) -> Message,
    ) -> Result<Vec<f32>, String> {
        let (reply, result) = mpsc::sync_channel(1);
        self.tx
            .as_ref()
            .ok_or("the live isolator is closed")?
            .send(message(reply))
            .map_err(|_| "the live isolator worker stopped")?;
        result
            .recv()
            .map_err(|_| "the live isolator worker stopped without a result")?
    }

    /// Feeds 48 kHz mono samples; returns the released output and timing.
    pub fn push(&self, samples: &[f32]) -> Result<LivePush, String> {
        if samples.is_empty() || samples.iter().any(|v| !v.is_finite()) {
            return Err("live isolation needs finite 48 kHz mono samples".into());
        }
        let started = Instant::now();
        let out = self.call(|reply| Message::Push(samples.to_vec(), reply))?;
        let processing_ms = started.elapsed().as_secs_f64() * 1e3;
        Ok(LivePush {
            released: out.len(),
            samples: out,
            processing_ms,
            late: processing_ms > samples.len() as f64 / 48.0,
        })
    }

    /// Ends the stream and returns the remaining output.
    pub fn flush(&self) -> Result<Vec<f32>, String> {
        self.call(Message::Flush)
    }
}

impl Drop for LiveDenoiser {
    fn drop(&mut self) {
        self.tx.take();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

fn run(mut engine: Engine, rx: Receiver<Message>) {
    while let Ok(message) = rx.recv() {
        let (result, reply) = match message {
            Message::Push(samples, reply) => (catch(|| engine.push(&samples)), reply),
            Message::Flush(reply) => (catch(|| engine.flush()), reply),
        };
        let _ = reply.send(result);
    }
}

fn catch(f: impl FnOnce() -> AudioResult<Vec<f32>>) -> Result<Vec<f32>, String> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(f))
        .unwrap_or_else(|_| {
            Err(SpeechError::Input {
                why: "live isolation panicked".into(),
            })
        })
        .map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coreml_mask_follows_the_window_and_hides_empty_slots() {
        let keys = COREML_PAST + COREML_FRAMES;
        // First call: no past at all, new frames causal.
        let m = CoreMlStep::mask(0);
        assert!(m[..COREML_PAST].iter().all(|&v| v < 0.0));
        assert_eq!(m[COREML_PAST], 0.0);
        assert!(m[COREML_PAST + 1] < 0.0, "frame 0 must not see frame 1");
        assert_eq!(m[keys + COREML_PAST], 0.0);
        assert_eq!(m[keys + COREML_PAST + 1], 0.0);
        // Long into a session: every cached slot is inside the window for
        // the first new frame; for the second, the oldest slot falls out.
        let m = CoreMlStep::mask(5_000);
        assert!(m[..COREML_PAST].iter().all(|&v| v == 0.0));
        assert!(
            m[keys] < 0.0,
            "q = first + 1 sees 1199 frames back, not 1200"
        );
        assert!(m[keys + 1..keys + COREML_PAST].iter().all(|&v| v == 0.0));
        // Exactly the frames that exist after 3 calls.
        let m = CoreMlStep::mask(6);
        let visible = m[..COREML_PAST].iter().filter(|&&v| v == 0.0).count();
        assert_eq!(visible, 6);
    }

    #[test]
    fn coreml_rope_matches_the_port_tables() {
        let (cos, sin) = CoreMlStep::rope(1000);
        let (pc, ps) = audio::ops::rope_tables(1002, HEAD_DIM, 10_000.0);
        let half = HEAD_DIM / 2;
        for i in 0..COREML_FRAMES * half {
            assert!((cos[i] - pc[1000 * half + i]).abs() < 2e-4);
            assert!((sin[i] - ps[1000 * half + i]).abs() < 2e-4);
        }
    }

    /// The port backends through the worker on the tiny causal fixture:
    /// output released as it arrives, the same as the stream run directly.
    #[test]
    fn live_port_backends_stream_the_fixture() {
        let _serial = crate::heavy::serial_test_lock();
        let fixture =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../audio/testdata/voice_isolator/causal");
        let dir = std::env::temp_dir().join(format!("ts-vi-live-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::copy(
            fixture.join("model.safetensors"),
            dir.join(REALTIME_WEIGHTS),
        )
        .unwrap();
        let audio: Vec<f32> = (0..4_800).map(|i| (i as f32 * 0.05).sin() * 0.2).collect();
        let direct = {
            let m = VoiceIsolator::open_file(&dir.join(REALTIME_WEIGHTS)).unwrap();
            let mut s = VoiceIsolatorStream::new(&m, 2, None).unwrap();
            let mut out = Vec::new();
            for piece in audio.chunks(320) {
                out.extend(s.push(&m, piece).unwrap());
            }
            out.extend(s.flush(&m).unwrap());
            out
        };
        for backend in [Backend::Cpu, Backend::Metal, Backend::Auto] {
            let live = LiveDenoiser::open(&dir, backend).unwrap();
            assert_ne!(
                live.backend(),
                "coreml",
                "no Core ML program in the fixture"
            );
            let mut out = Vec::new();
            for piece in audio.chunks(320) {
                let push = live.push(piece).unwrap();
                assert_eq!(push.released, push.samples.len());
                out.extend(push.samples);
            }
            out.extend(live.flush().unwrap());
            assert_eq!(out.len(), audio.len());
            let diff = out
                .iter()
                .zip(&direct)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(diff < 1e-5, "{backend:?}: {diff}");
        }
        assert!(
            LiveDenoiser::open(&dir, Backend::CoreMl).is_err(),
            "Core ML refuses without a program"
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
