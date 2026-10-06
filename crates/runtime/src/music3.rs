//! MiniMax Music 3 runner. Uses the portable pipeline with packed resident
//! Metal weights, typed attention, and device vocoder convolutions.
use audio::music::minimax_music3::backend::{
    AttentionShape, ComputeBackend, ConvShape, DeviceWeight, RopeShape, WeightData, WeightEncoding,
};
use audio::music::minimax_music3::precision::DType;
use audio::music::minimax_music3::{
    GenerateRequest, Generation, Model, ModelConfig, Music3Precision, SamplingTrace, StageTimings,
    TextGenerateRequest,
};
use audio::{Result, SpeechError};
use gpu::{Music3DType, Music3Device, Music3Encoding, Music3Weight};
use std::cell::RefCell;
use std::collections::HashMap;
use std::path::Path;
use std::rc::Rc;
use std::time::{Duration, Instant};

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
/// Wall time and call count of one device operation at one shape, as the
/// pipeline sees it (buffer setup, dispatch, wait, and readback included).
#[derive(Debug, Clone, PartialEq)]
pub struct DispatchStat {
    pub op: &'static str,
    /// Operation geometry: linear is `[rows, input, output]`, attention is
    /// `[queries, keys, heads]`, norms are `[rows, cols]`, and so on.
    pub shape: Vec<usize>,
    pub calls: u64,
    pub total: Duration,
}

type Profile = Rc<RefCell<HashMap<(&'static str, Vec<usize>), (u64, Duration)>>>;

fn record(profile: &Profile, op: &'static str, shape: &[usize], started: Instant) {
    let mut table = profile.borrow_mut();
    let entry = table.entry((op, shape.to_vec())).or_default();
    entry.0 += 1;
    entry.1 += started.elapsed();
}

struct Backend {
    device: Music3Device,
    observer: Observer,
    profile: Profile,
}
fn dtype(d: DType) -> Music3DType {
    match d {
        DType::F32 => Music3DType::F32,
        DType::F16 => Music3DType::F16,
        DType::Bf16 => Music3DType::Bf16,
    }
}
struct Weight(Music3Weight, Profile);
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
        Ok(Rc::new(Weight(weight, self.profile.clone())))
    }
    fn attention(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        s: AttentionShape,
        d: DType,
    ) -> Result<Vec<f32>> {
        let started = Instant::now();
        let out = self
            .device
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
            .map_err(device_error);
        record(
            &self.profile,
            "attention",
            &[s.batch, s.queries, s.keys, s.heads, s.dim],
            started,
        );
        out
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
        let started = Instant::now();
        let out = self
            .device
            .rms_norm(x, w, rows, cols, eps, dtype(d))
            .map_err(device_error);
        record(&self.profile, "rms_norm", &[rows, cols], started);
        out
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
        let started = Instant::now();
        let out = self
            .device
            .layer_norm(x, w, bias, rows, cols, eps, dtype(d))
            .map_err(device_error);
        record(&self.profile, "layer_norm", &[rows, cols], started);
        out
    }
    fn rope(&self, x: &[f32], s: RopeShape, d: DType) -> Result<Vec<f32>> {
        let started = Instant::now();
        let out = self
            .device
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
            .map_err(device_error);
        record(
            &self.profile,
            "rope",
            &[s.batch, s.seq, s.heads, s.dim],
            started,
        );
        out
    }
    fn normal_from_uniform(&self, x: &[f32], d: DType) -> Result<Vec<f32>> {
        self.device
            .normal_from_uniform(x, dtype(d))
            .map_err(device_error)
    }
    fn rotary_tables(&self, seq: usize, dim: usize, theta: f32) -> Result<(Vec<f32>, Vec<f32>)> {
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
        let started = Instant::now();
        let out = self
            .device
            .snake(x, alpha, channels, frames, dtype(d))
            .map_err(device_error);
        record(&self.profile, "snake", &[channels, frames], started);
        out
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
        let started = Instant::now();
        let result = self
            .0
            .linear_typed(input, bias, rows, inn, out, dtype(d))
            .map_err(device_error);
        record(&self.1, "linear", &[rows, inn, out], started);
        result
    }
    fn embedding(&self, ids: &[i32], width: usize) -> Result<Vec<f32>> {
        let started = Instant::now();
        let result = self.0.embedding(ids, width).map_err(device_error);
        record(&self.1, "embedding", &[ids.len(), width], started);
        result
    }
    fn convolution(
        &self,
        input: &[f32],
        bias: Option<&[f32]>,
        s: ConvShape,
        d: DType,
    ) -> Result<Vec<f32>> {
        let started = Instant::now();
        let result = self
            .0
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
            .map_err(device_error);
        record(
            &self.1,
            if s.transpose {
                "conv_transpose"
            } else {
                "conv"
            },
            &[s.input_channels, s.output_channels, s.kernel],
            started,
        );
        result
    }
}

/// A converted MiniMax Music 3 checkpoint, resident on the Metal device.
/// Each request creates fresh AR caches and flow overlap state.
pub struct Music3Runner {
    model: Model,
    device: Music3Device,
    observer: Observer,
    profile: Profile,
}
impl Music3Runner {
    pub fn open(path: &Path) -> Result<Self> {
        Self::open_with_precision(path, Music3Precision::Checkpoint)
    }
    pub fn open_with_precision(path: &Path, precision: Music3Precision) -> Result<Self> {
        let device = Music3Device::new().map_err(device_error)?;
        let observer = Rc::new(RefCell::new(None));
        let profile = Profile::default();
        let backend = Rc::new(Backend {
            device: device.clone(),
            observer: observer.clone(),
            profile: profile.clone(),
        });
        let model = Model::load_converted_with_backend_and_precision(path, backend, precision)?;
        Ok(Self {
            model,
            device,
            observer,
            profile,
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
    /// Per-operation call counts and wall time since the last reset,
    /// slowest total first. Weight upload during `open` is not included.
    pub fn dispatch_profile(&self) -> Vec<DispatchStat> {
        let mut stats: Vec<DispatchStat> = self
            .profile
            .borrow()
            .iter()
            .map(|((op, shape), (calls, total))| DispatchStat {
                op,
                shape: shape.clone(),
                calls: *calls,
                total: *total,
            })
            .collect();
        stats.sort_by(|a, b| b.total.cmp(&a.total));
        stats
    }
    pub fn reset_dispatch_profile(&self) {
        self.profile.borrow_mut().clear();
    }
    pub fn resident_weight_bytes(&self) -> usize {
        self.device.resident_weight_bytes()
    }
    pub fn generate_text(&self, request: &TextGenerateRequest) -> Result<Generation> {
        self.model.generate_text(request)
    }
    /// [`Self::generate_text`] plus a per-stage wall-time breakdown.
    pub fn generate_text_timed(
        &self,
        request: &TextGenerateRequest,
    ) -> Result<(Generation, StageTimings)> {
        self.model.generate_text_timed(request)
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
