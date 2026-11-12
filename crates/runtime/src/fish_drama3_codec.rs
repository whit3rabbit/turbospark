//! The Fish S1 DAC for Drama 3 on Metal, on a thread of its own.
//!
//! The codec's convolutions, transposed convolutions, ConvNeXt pointwise
//! linears and Snake run on `gpu::Music3Device` through
//! `audio::codec::fish_s1_dac::DacExec` (F32 weights and activations, the
//! route MegaTTS3's WavVAE decoder takes). Code lookups, the transformers and
//! the norms stay on the CPU. The device and its uploaded weights are tied to
//! the thread that made them, so [`CodecWorker`] owns the codec and its exec
//! for the runner's lifetime and takes jobs over a channel: a reference
//! encode, or a streamed decode fed frame by frame.

use std::path::Path;
use std::rc::Rc;
use std::sync::mpsc::{channel, Receiver, Sender};

use audio::backend::{
    AttentionShape, ComputeBackend, ConvShape, DType, DeviceWeight, WeightData, WeightEncoding,
};
use audio::codec::fish_s1_dac::{DacExec, FishS1Dac};
use audio::{Result as AudioResult, SpeechError};
use gpu::{Music3Device, Music3Encoding, Music3Weight};

fn metal_error(e: gpu::GpuError) -> SpeechError {
    SpeechError::Unsupported {
        why: format!("Fish S1 DAC Metal: {e}"),
    }
}

/// F32 weights only, the tiled float convolution and GEMM, and Snake.
/// Anything else is refused rather than approximated.
struct MetalDac {
    device: Music3Device,
}

struct MetalDacWeight {
    inner: Music3Weight,
}

impl ComputeBackend for MetalDac {
    fn load_weight(&self, data: WeightData<'_>) -> AudioResult<Rc<dyn DeviceWeight>> {
        if data.encoding != WeightEncoding::F32 || data.dtype != DType::F32 {
            return Err(SpeechError::Unsupported {
                why: "the Fish S1 DAC on Metal loads F32 weights only".into(),
            });
        }
        let inner = self
            .device
            .load_weight(data.shape, data.bytes, Music3Encoding::F32, &[], &[], &[])
            .map_err(metal_error)?;
        Ok(Rc::new(MetalDacWeight { inner }))
    }

    fn attention(
        &self,
        _: &[f32],
        _: &[f32],
        _: &[f32],
        _: AttentionShape,
        _: DType,
    ) -> AudioResult<Vec<f32>> {
        Err(SpeechError::Unsupported {
            why: "the Fish S1 DAC transformers run on the CPU".into(),
        })
    }

    fn snake(
        &self,
        x: &[f32],
        alpha: &[f32],
        channels: usize,
        frames: usize,
        dtype: DType,
    ) -> AudioResult<Vec<f32>> {
        if dtype != DType::F32 {
            return Err(SpeechError::Unsupported {
                why: "the Fish S1 DAC Snake is F32".into(),
            });
        }
        self.device
            .snake(x, alpha, channels, frames, gpu::Music3DType::F32)
            .map_err(metal_error)
    }
}

impl DeviceWeight for MetalDacWeight {
    fn linear(
        &self,
        x: &[f32],
        bias: Option<&[f32]>,
        rows: usize,
        input: usize,
        output: usize,
        dtype: DType,
    ) -> AudioResult<Vec<f32>> {
        if dtype != DType::F32 {
            return Err(SpeechError::Unsupported {
                why: "the Fish S1 DAC linears are F32".into(),
            });
        }
        self.inner
            .linear_f32(x, bias, rows, input, output)
            .map_err(metal_error)
    }

    fn embedding(&self, _: &[i32], _: usize) -> AudioResult<Vec<f32>> {
        Err(SpeechError::Unsupported {
            why: "the Fish S1 DAC has no device embeddings".into(),
        })
    }

    fn convolution(
        &self,
        x: &[f32],
        bias: Option<&[f32]>,
        shape: ConvShape,
        dtype: DType,
    ) -> AudioResult<Vec<f32>> {
        self.convolution_grouped(x, bias, shape, 1, dtype)
    }

    fn convolution_grouped(
        &self,
        x: &[f32],
        bias: Option<&[f32]>,
        s: ConvShape,
        groups: usize,
        dtype: DType,
    ) -> AudioResult<Vec<f32>> {
        if dtype != DType::F32 {
            return Err(SpeechError::Unsupported {
                why: "the Fish S1 DAC convolutions are F32".into(),
            });
        }
        self.inner
            .convolution_f32_grouped(
                x,
                bias,
                s.input_channels,
                s.output_channels,
                s.kernel,
                s.stride,
                s.padding,
                s.dilation,
                s.transpose,
                groups,
            )
            .map_err(metal_error)
    }
}

/// A [`DacExec`] on a new Metal device. It and its weights stay on the
/// calling thread.
pub fn metal_dac_exec() -> AudioResult<DacExec> {
    let device = Music3Device::new().map_err(metal_error)?;
    Ok(DacExec::device(Rc::new(MetalDac { device })))
}

/// Codes per book and the code length, as `FishS1Dac::encode` returns them.
type Encoded = (Vec<Vec<i32>>, usize);

enum Job {
    Encode {
        samples: Vec<f32>,
        reply: Sender<Result<Encoded, String>>,
    },
    /// Decode frames as they arrive, `first` frames then `every` frames per
    /// piece, until `None` or a closed channel. Each piece's samples go to
    /// `pcm`, and a closed `pcm` ends the job.
    Stream {
        frames: Receiver<Option<Vec<i32>>>,
        pcm: Sender<Result<Vec<f32>, String>>,
        first: usize,
        every: usize,
    },
}

/// The codec and its Metal exec on one thread (see the module doc).
pub struct CodecWorker {
    jobs: Option<Sender<Job>>,
    thread: Option<std::thread::JoinHandle<()>>,
    sample_rate: u32,
}

impl CodecWorker {
    /// Loads the codec at `weights` on a new thread with a Metal exec, or the
    /// host when `metal` is false. Returns once the codec is ready.
    pub fn open(weights: &Path, metal: bool) -> Result<Self, String> {
        let weights = weights.to_path_buf();
        let (ready_tx, ready_rx) = channel::<Result<u32, String>>();
        let (jobs_tx, jobs_rx) = channel::<Job>();
        let thread = std::thread::Builder::new()
            .name("fish-drama3-codec".into())
            .spawn(move || {
                let opened = FishS1Dac::open_weights(&weights).and_then(|codec| {
                    let exec = if metal {
                        metal_dac_exec()?
                    } else {
                        DacExec::host()
                    };
                    Ok((codec, exec))
                });
                let (codec, exec) = match opened {
                    Ok(pair) => pair,
                    Err(e) => {
                        let _ = ready_tx.send(Err(e.to_string()));
                        return;
                    }
                };
                let _ = ready_tx.send(Ok(codec.config.sample_rate));
                drop(ready_tx);
                for job in jobs_rx {
                    match job {
                        Job::Encode { samples, reply } => {
                            let _ = reply
                                .send(codec.encode_on(&exec, &samples).map_err(|e| e.to_string()));
                        }
                        Job::Stream {
                            frames,
                            pcm,
                            first,
                            every,
                        } => stream(&codec, &exec, &frames, &pcm, first, every),
                    }
                }
            })
            .map_err(|e| format!("codec thread: {e}"))?;
        let sample_rate = ready_rx
            .recv()
            .map_err(|_| "the codec thread exited before loading".to_string())??;
        Ok(CodecWorker {
            jobs: Some(jobs_tx),
            thread: Some(thread),
            sample_rate,
        })
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    fn send(&self, job: Job) -> Result<(), String> {
        self.jobs
            .as_ref()
            .and_then(|jobs| jobs.send(job).ok())
            .ok_or_else(|| "the codec thread has stopped".to_string())
    }

    /// `FishS1Dac::encode` on the worker.
    pub fn encode(&self, samples: Vec<f32>) -> Result<Encoded, String> {
        let (reply, result) = channel();
        self.send(Job::Encode { samples, reply })?;
        result
            .recv()
            .map_err(|_| "the codec thread stopped mid-encode".to_string())?
    }

    /// Starts a streamed decode (see [`Job::Stream`]). Returns the frame
    /// sender and the PCM receiver.
    #[allow(clippy::type_complexity)]
    pub fn stream(
        &self,
        first: usize,
        every: usize,
    ) -> Result<(Sender<Option<Vec<i32>>>, Receiver<Result<Vec<f32>, String>>), String> {
        let (frames_tx, frames) = channel();
        let (pcm, pcm_rx) = channel();
        self.send(Job::Stream {
            frames,
            pcm,
            first,
            every,
        })?;
        Ok((frames_tx, pcm_rx))
    }
}

impl Drop for CodecWorker {
    fn drop(&mut self) {
        // Closing the job channel ends the thread's loop.
        self.jobs.take();
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

fn stream(
    codec: &FishS1Dac,
    exec: &DacExec,
    frames: &Receiver<Option<Vec<i32>>>,
    pcm: &Sender<Result<Vec<f32>, String>>,
    first: usize,
    every: usize,
) {
    let books = codec.config.total_books();
    let mut decoder = codec.stream_on(exec);
    let mut pending: Vec<Vec<i32>> = vec![Vec::new(); books];
    let mut piece = first.max(1);
    // False once the caller has gone or the decode failed.
    let mut flush = |pending: &mut Vec<Vec<i32>>| -> bool {
        if pending[0].is_empty() {
            return true;
        }
        let result = decoder.push(pending).map_err(|e| e.to_string());
        pending.iter_mut().for_each(Vec::clear);
        let failed = result.is_err();
        pcm.send(result).is_ok() && !failed
    };
    while let Ok(Some(frame)) = frames.recv() {
        if frame.len() != books {
            let _ = pcm.send(Err(format!(
                "a frame of {} codes for a {books}-book codec",
                frame.len()
            )));
            return;
        }
        for (book, code) in pending.iter_mut().zip(frame) {
            book.push(code);
        }
        if pending[0].len() >= piece {
            piece = every.max(1);
            if !flush(&mut pending) {
                return;
            }
        }
    }
    flush(&mut pending);
}
