//! MiniMax Music 3 runner. Uses the portable pipeline with packed resident
//! Metal weights, typed attention, and device vocoder convolutions.
use audio::music::minimax_music3::backend::{
    AttentionShape, ComputeBackend, ConvShape, DeviceWeight, RopeShape, WeightData, WeightEncoding,
};
use audio::music::minimax_music3::precision::DType;
use audio::music::minimax_music3::{
    GenerateRequest, Generation, Model, ModelConfig, Music3Precision, SamplingTrace,
    TextGenerateRequest,
};
use audio::{Result, SpeechError};
use gpu::{Music3DType, Music3Device, Music3Encoding, Music3Weight};
use std::cell::RefCell;
use std::path::Path;
use std::rc::Rc;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
type Cancellation = Rc<RefCell<Option<Arc<AtomicBool>>>>;
fn checkpoint(cancel: &Cancellation) -> Result<()> {
    if cancel
        .borrow()
        .as_ref()
        .is_some_and(|flag| flag.load(Ordering::Acquire))
    {
        Err(SpeechError::Input {
            why: "audio job cancelled".into(),
        })
    } else {
        Ok(())
    }
}

fn device_error(error: gpu::GpuError) -> SpeechError {
    SpeechError::Unsupported {
        why: format!("Music 3 Metal: {error}"),
    }
}
/// Intermediate values retain their logical dtype even though storage is f32.
pub struct Music3Stage<'a> {
    pub name: &'a str,
    pub values: &'a [f32],
    pub dtype: DType,
    pub shape: &'a [usize],
}
type Observer = Rc<RefCell<Option<Box<dyn FnMut(Music3Stage<'_>)>>>>;
struct Backend {
    device: Music3Device,
    observer: Observer,
    cancellation: Cancellation,
}
fn dtype(d: DType) -> Music3DType {
    match d {
        DType::F32 => Music3DType::F32,
        DType::F16 => Music3DType::F16,
        DType::Bf16 => Music3DType::Bf16,
    }
}
struct Weight(Music3Weight, Cancellation);
impl ComputeBackend for Backend {
    fn load_weight(&self, data: WeightData<'_>) -> Result<Rc<dyn DeviceWeight>> {
        let encoding = match data.encoding {
            WeightEncoding::F32 => Music3Encoding::F32,
            WeightEncoding::F16 => Music3Encoding::F16,
            WeightEncoding::Bf16 => Music3Encoding::Bf16,
            WeightEncoding::Affine { bits, group_size } => {
                Music3Encoding::Affine { bits, group_size }
            }
            WeightEncoding::MxFp4 => Music3Encoding::MxFp4,
            WeightEncoding::MxFp8 => Music3Encoding::MxFp8,
            WeightEncoding::NvFp4 => Music3Encoding::NvFp4,
        };
        let weight = self
            .device
            .load_weight(
                data.shape,
                data.bytes,
                encoding,
                data.scales,
                data.offsets,
                data.block_scales,
            )
            .map_err(|error| SpeechError::Tensor {
                name: data.name.into(),
                why: error.to_string(),
            })?;
        Ok(Rc::new(Weight(weight, self.cancellation.clone())))
    }
    fn attention(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        s: AttentionShape,
        d: DType,
    ) -> Result<Vec<f32>> {
        checkpoint(&self.cancellation)?;
        self.device
            .attention_typed(
                q,
                k,
                v,
                s.batch,
                s.queries,
                s.keys,
                s.heads,
                s.kv_heads,
                s.dim,
                s.kv_time_major,
                s.causal,
                s.offset,
                dtype(d),
            )
            .map_err(device_error)
    }
    fn rms_norm(
        &self,
        x: &[f32],
        w: &[f32],
        rows: usize,
        cols: usize,
        eps: f32,
        d: DType,
    ) -> Result<Vec<f32>> {
        checkpoint(&self.cancellation)?;
        self.device
            .rms_norm(x, w, rows, cols, eps, dtype(d))
            .map_err(device_error)
    }
    fn layer_norm(
        &self,
        x: &[f32],
        w: &[f32],
        bias: Option<&[f32]>,
        rows: usize,
        cols: usize,
        eps: f32,
        d: DType,
    ) -> Result<Vec<f32>> {
        checkpoint(&self.cancellation)?;
        self.device
            .layer_norm(x, w, bias, rows, cols, eps, dtype(d))
            .map_err(device_error)
    }
    fn rope(&self, x: &[f32], s: RopeShape, d: DType) -> Result<Vec<f32>> {
        checkpoint(&self.cancellation)?;
        self.device
            .rope(
                x,
                s.batch,
                s.seq,
                s.heads,
                s.dim,
                s.offset,
                s.theta,
                dtype(d),
            )
            .map_err(device_error)
    }
    fn normal_from_uniform(&self, x: &[f32], d: DType) -> Result<Vec<f32>> {
        checkpoint(&self.cancellation)?;
        self.device
            .normal_from_uniform(x, dtype(d))
            .map_err(device_error)
    }
    fn rotary_tables(&self, seq: usize, dim: usize, theta: f32) -> Result<(Vec<f32>, Vec<f32>)> {
        checkpoint(&self.cancellation)?;
        self.device
            .rotary_tables(seq, dim, theta)
            .map_err(device_error)
    }
    fn snake(
        &self,
        x: &[f32],
        alpha: &[f32],
        channels: usize,
        frames: usize,
        d: DType,
    ) -> Result<Vec<f32>> {
        checkpoint(&self.cancellation)?;
        self.device
            .snake(x, alpha, channels, frames, dtype(d))
            .map_err(device_error)
    }
    fn trace(&self, stage: &str, data: &[f32], d: DType, shape: &[usize]) {
        if let Some(f) = self.observer.borrow_mut().as_mut() {
            f(Music3Stage {
                name: stage,
                values: data,
                dtype: d,
                shape,
            });
        }
    }
}
impl DeviceWeight for Weight {
    fn linear(
        &self,
        input: &[f32],
        bias: Option<&[f32]>,
        rows: usize,
        inn: usize,
        out: usize,
        d: DType,
    ) -> Result<Vec<f32>> {
        checkpoint(&self.1)?;
        self.0
            .linear_typed(input, bias, rows, inn, out, dtype(d))
            .map_err(device_error)
    }
    fn embedding(&self, ids: &[i32], width: usize) -> Result<Vec<f32>> {
        checkpoint(&self.1)?;
        self.0.embedding(ids, width).map_err(device_error)
    }
    fn convolution(
        &self,
        input: &[f32],
        bias: Option<&[f32]>,
        s: ConvShape,
        d: DType,
    ) -> Result<Vec<f32>> {
        checkpoint(&self.1)?;
        self.0
            .convolution_typed(
                input,
                bias,
                s.input_channels,
                s.output_channels,
                s.kernel,
                s.stride,
                s.padding,
                s.dilation,
                s.transpose,
                dtype(d),
            )
            .map_err(device_error)
    }
}

/// A converted MiniMax Music 3 checkpoint, resident on the Metal device.
/// Each request creates fresh AR caches and flow overlap state.
pub struct Music3Runner {
    model: Model,
    device: Music3Device,
    observer: Observer,
    cancellation: Cancellation,
}
impl Music3Runner {
    pub fn open(path: &Path) -> Result<Self> {
        Self::open_with_precision(path, Music3Precision::Checkpoint)
    }
    pub fn open_with_precision(path: &Path, precision: Music3Precision) -> Result<Self> {
        let device = Music3Device::new().map_err(device_error)?;
        let observer = Rc::new(RefCell::new(None));
        let cancellation = Rc::new(RefCell::new(None));
        let backend = Rc::new(Backend {
            device: device.clone(),
            observer: observer.clone(),
            cancellation: cancellation.clone(),
        });
        let model = Model::load_converted_with_backend_and_precision(path, backend, precision)?;
        Ok(Self {
            model,
            device,
            observer,
            cancellation,
        })
    }
    pub fn set_trace_observer(&self, observer: impl FnMut(Music3Stage<'_>) + 'static) {
        *self.observer.borrow_mut() = Some(Box::new(observer));
    }
    pub fn clear_trace_observer(&self) {
        *self.observer.borrow_mut() = None;
    }
    pub fn config(&self) -> &ModelConfig {
        self.model.config()
    }
    pub fn resident_weight_bytes(&self) -> usize {
        self.device.resident_weight_bytes()
    }
    /// Stops at a native compute boundary; the caller retains its permit until return.
    pub fn generate_text_cancellable(
        &self,
        request: &TextGenerateRequest,
        cancel: Arc<AtomicBool>,
    ) -> Result<Generation> {
        *self.cancellation.borrow_mut() = Some(cancel);
        struct Reset(Cancellation);
        impl Drop for Reset {
            fn drop(&mut self) {
                self.0.borrow_mut().take();
            }
        }
        let _reset = Reset(self.cancellation.clone());
        checkpoint(&self.cancellation)?;
        self.model.generate_text(request)
    }
    pub fn generate_text(&self, request: &TextGenerateRequest) -> Result<Generation> {
        self.model.generate_text(request)
    }
    pub fn generate(&self, request: &GenerateRequest) -> Result<Generation> {
        self.model.generate(request)
    }
    /// Decode precomputed AR frames, useful for independent stage parity.
    pub fn run_flow(
        &self,
        hiddens: &[f32],
        frames: usize,
        steps: usize,
        seed: u64,
    ) -> Result<Vec<f32>> {
        self.model.run_flow(hiddens, frames, steps, seed)
    }
    /// Replay supplied channel-major noise chunks for independent stage probes.
    pub fn run_flow_with_noise(
        &self,
        hiddens: &[f32],
        frames: usize,
        steps: usize,
        noise_chunks: &[Vec<f32>],
    ) -> Result<Vec<f32>> {
        self.model
            .run_flow_with_noise(hiddens, frames, steps, noise_chunks)
    }
    pub fn generate_frame_hiddens(
        &self,
        ids: &[i32],
        frames: usize,
        seed: u64,
    ) -> Result<(Vec<f32>, Vec<Vec<i32>>)> {
        self.model.generate_frame_hiddens(ids, frames, seed)
    }
    pub fn generate_frame_hiddens_traced(
        &self,
        ids: &[i32],
        frames: usize,
        seed: u64,
        observe: impl FnMut(SamplingTrace<'_>),
    ) -> Result<(Vec<f32>, Vec<Vec<i32>>)> {
        self.model
            .generate_frame_hiddens_traced(ids, frames, seed, observe)
    }
}
