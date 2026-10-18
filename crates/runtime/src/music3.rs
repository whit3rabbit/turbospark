//! MiniMax Music 3 runner. Uses the portable pipeline with packed resident
//! Metal weights, typed attention, and device vocoder convolutions.
use audio::music::minimax_music3::backend::{
    AttentionShape, ComputeBackend, ConvShape, DeviceBuffer, DeviceWeight, RopeShape, Sequence,
    Slot, WeightData, WeightEncoding,
};
use audio::music::minimax_music3::precision::{bf16_silu_table, DType};
use audio::music::minimax_music3::{
    Control, FlowProfile, GenerateRequest, Generation, Model, ModelConfig, Music3Precision,
    Progress, SamplingTrace, StageTimings, TextGenerateRequest,
};
use audio::{Result, SpeechError};
use gpu::{
    Music3Buffer, Music3DType, Music3Device, Music3Encoding, Music3Sequence, Music3Slot,
    Music3Table, Music3Weight,
};
use std::cell::RefCell;
use std::collections::HashMap;
use std::path::Path;
use std::rc::Rc;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::time::{Duration, Instant};

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
    cancellation: Cancellation,
    profile: Profile,
    /// BF16 SiLU of every 16-bit pattern, uploaded on first use.
    silu: RefCell<Option<Rc<Music3Table>>>,
}

impl Backend {
    fn silu_table(&self) -> Result<Rc<Music3Table>> {
        if let Some(table) = self.silu.borrow().as_ref() {
            return Ok(table.clone());
        }
        let table = Rc::new(
            self.device
                .table(&bf16_silu_table())
                .map_err(device_error)?,
        );
        *self.silu.borrow_mut() = Some(table.clone());
        Ok(table)
    }
}

/// A persistent Metal buffer (the AR KV cache) behind [`DeviceBuffer`].
struct KvBuffer(Music3Buffer);
impl DeviceBuffer for KvBuffer {
    fn write(&self, first: usize, data: &[f32]) -> Result<()> {
        self.0.write(first, data).map_err(device_error)
    }
    fn read(&self, first: usize, len: usize) -> Result<Vec<f32>> {
        self.0.read(first, len).map_err(device_error)
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// A Metal [`Sequence`]: one command buffer per step. Cancellation is
/// checked at every recorded operation and before commit; a cancelled
/// sequence is dropped uncommitted. One `dispatch_profile` row per finish.
struct MetalSequence<'a> {
    backend: &'a Backend,
    inner: Music3Sequence,
    slots: Vec<Music3Slot>,
    op: &'static str,
    shape: Vec<usize>,
    started: Instant,
}

impl MetalSequence<'_> {
    fn slot(&self, slot: Slot) -> Result<Music3Slot> {
        self.slots
            .get(slot.0)
            .copied()
            .ok_or_else(|| SpeechError::Input {
                why: "Music 3 sequence slot does not exist".into(),
            })
    }
    fn push(&mut self, result: std::result::Result<Music3Slot, gpu::GpuError>) -> Result<Slot> {
        self.slots.push(result.map_err(device_error)?);
        Ok(Slot(self.slots.len() - 1))
    }
}

fn require_bf16(op: &str, dtypes: &[DType]) -> Result<()> {
    if dtypes.iter().all(|d| *d == DType::Bf16) {
        Ok(())
    } else {
        Err(SpeechError::Unsupported {
            why: format!("Music 3 Metal sequence {op} reproduces BF16 rounding only"),
        })
    }
}

impl Sequence for MetalSequence<'_> {
    fn upload(&mut self, data: &[f32]) -> Result<Slot> {
        checkpoint(&self.backend.cancellation)?;
        let result = self.inner.upload(data);
        self.push(result)
    }
    fn bind(&mut self, buffer: &dyn DeviceBuffer) -> Result<Slot> {
        checkpoint(&self.backend.cancellation)?;
        let buffer =
            buffer
                .as_any()
                .downcast_ref::<KvBuffer>()
                .ok_or_else(|| SpeechError::Unsupported {
                    why: "Music 3 Metal sequence needs a Metal-resident buffer".into(),
                })?;
        let result = self.inner.bind(&buffer.0);
        self.push(result)
    }
    fn view(&mut self, slot: Slot, first: usize, len: usize) -> Result<Slot> {
        let slot = self.slot(slot)?;
        let result = self.inner.view(slot, first, len);
        self.push(result)
    }
    fn copy(&mut self, src: Slot, dst: Slot) -> Result<()> {
        checkpoint(&self.backend.cancellation)?;
        let (src, dst) = (self.slot(src)?, self.slot(dst)?);
        self.inner.copy(src, dst).map_err(device_error)
    }
    fn rope(&mut self, x: Slot, s: RopeShape, d: DType) -> Result<Slot> {
        checkpoint(&self.backend.cancellation)?;
        let x = self.slot(x)?;
        let result = self.inner.rope(
            x,
            s.batch,
            s.seq,
            s.heads,
            s.dim,
            s.offset,
            s.theta,
            dtype(d),
        );
        self.push(result)
    }
    fn linear(
        &mut self,
        x: Slot,
        weight: &dyn DeviceWeight,
        bias: Option<&[f32]>,
        rows: usize,
        input_dim: usize,
        output_dim: usize,
        d: DType,
    ) -> Result<Slot> {
        checkpoint(&self.backend.cancellation)?;
        let weight = weight
            .as_any()
            .and_then(|w| w.downcast_ref::<Weight>())
            .ok_or_else(|| SpeechError::Unsupported {
                why: "Music 3 Metal sequence needs a Metal-resident weight".into(),
            })?;
        let x = self.slot(x)?;
        let result = self
            .inner
            .linear(x, &weight.0, bias, rows, input_dim, output_dim, dtype(d));
        self.push(result)
    }
    fn convolution(
        &mut self,
        x: Slot,
        weight: &dyn DeviceWeight,
        bias: Option<&[f32]>,
        s: ConvShape,
        batch: usize,
        d: DType,
    ) -> Result<Slot> {
        checkpoint(&self.backend.cancellation)?;
        let weight = weight
            .as_any()
            .and_then(|w| w.downcast_ref::<Weight>())
            .ok_or_else(|| SpeechError::Unsupported {
                why: "Music 3 Metal sequence needs a Metal-resident weight".into(),
            })?;
        let x = self.slot(x)?;
        let result = self.inner.convolution(
            x,
            &weight.0,
            bias,
            s.input_channels,
            s.output_channels,
            s.kernel,
            s.stride,
            s.padding,
            s.dilation,
            s.transpose,
            dtype(d),
            batch,
        );
        self.push(result)
    }
    fn snake(
        &mut self,
        x: Slot,
        alpha: &[f32],
        channels: usize,
        frames: usize,
        d: DType,
    ) -> Result<Slot> {
        checkpoint(&self.backend.cancellation)?;
        let x = self.slot(x)?;
        let result = self.inner.snake(x, alpha, channels, frames, dtype(d));
        self.push(result)
    }
    fn layer_norm(
        &mut self,
        x: Slot,
        w: &[f32],
        bias: Option<&[f32]>,
        rows: usize,
        cols: usize,
        eps: f32,
        d: DType,
    ) -> Result<Slot> {
        checkpoint(&self.backend.cancellation)?;
        let x = self.slot(x)?;
        let result = self.inner.layer_norm(x, w, bias, rows, cols, eps, dtype(d));
        self.push(result)
    }
    fn partial_rope(
        &mut self,
        x: Slot,
        cos: Slot,
        sin: Slot,
        len: usize,
        heads: usize,
        head_dim: usize,
        rotary: usize,
        d: DType,
    ) -> Result<Slot> {
        checkpoint(&self.backend.cancellation)?;
        require_bf16("partial rotary", &[d])?;
        let (x, cos, sin) = (self.slot(x)?, self.slot(cos)?, self.slot(sin)?);
        let result = self
            .inner
            .partial_rope_bf16(x, cos, sin, len, heads, head_dim, rotary);
        self.push(result)
    }
    fn gated_silu(&mut self, x: Slot, rows: usize, half: usize, d: DType) -> Result<Slot> {
        checkpoint(&self.backend.cancellation)?;
        require_bf16("gated SiLU", &[d])?;
        let table = self.backend.silu_table()?;
        let x = self.slot(x)?;
        let result = self.inner.gated_silu_bf16(x, rows, half, &table);
        self.push(result)
    }
    fn rms_norm(
        &mut self,
        x: Slot,
        w: &[f32],
        rows: usize,
        cols: usize,
        eps: f32,
        d: DType,
    ) -> Result<Slot> {
        checkpoint(&self.backend.cancellation)?;
        let x = self.slot(x)?;
        let result = self.inner.rms_norm(x, w, rows, cols, eps, dtype(d));
        self.push(result)
    }
    fn attention(
        &mut self,
        q: Slot,
        k: Slot,
        v: Slot,
        s: AttentionShape,
        d: DType,
    ) -> Result<Slot> {
        checkpoint(&self.backend.cancellation)?;
        let (q, k, v) = (self.slot(q)?, self.slot(k)?, self.slot(v)?);
        let result = self.inner.attention(
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
        );
        self.push(result)
    }
    fn add_round(&mut self, a: Slot, b: Slot, d: DType) -> Result<Slot> {
        checkpoint(&self.backend.cancellation)?;
        require_bf16("residual add", &[d])?;
        let (a, b) = (self.slot(a)?, self.slot(b)?);
        let result = self.inner.add_round_bf16(a, b);
        self.push(result)
    }
    fn swiglu(&mut self, g: Slot, u: Slot, gate: DType, fused: DType) -> Result<Slot> {
        checkpoint(&self.backend.cancellation)?;
        require_bf16("SwiGLU", &[gate, fused])?;
        let table = self.backend.silu_table()?;
        let (g, u) = (self.slot(g)?, self.slot(u)?);
        let result = self.inner.swiglu_bf16(g, u, &table);
        self.push(result)
    }
    fn finish(self: Box<Self>, read: &[Slot]) -> Result<Option<Vec<Vec<f32>>>> {
        checkpoint(&self.backend.cancellation)?;
        let read = read
            .iter()
            .map(|slot| self.slot(*slot))
            .collect::<Result<Vec<_>>>()?;
        let Self {
            backend,
            inner,
            op,
            shape,
            started,
            ..
        } = *self;
        let result = inner.finish(&read).map_err(device_error);
        record(&backend.profile, op, &shape, started);
        result
    }
}
fn dtype(d: DType) -> Music3DType {
    match d {
        DType::F32 => Music3DType::F32,
        DType::F16 => Music3DType::F16,
        DType::Bf16 => Music3DType::Bf16,
    }
}
struct Weight(Music3Weight, Cancellation, Profile);
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
        Ok(Rc::new(Weight(
            weight,
            self.cancellation.clone(),
            self.profile.clone(),
        )))
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
        checkpoint(&self.cancellation)?;
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
        checkpoint(&self.cancellation)?;
        let started = Instant::now();
        let out = self
            .device
            .layer_norm(x, w, bias, rows, cols, eps, dtype(d))
            .map_err(device_error);
        record(&self.profile, "layer_norm", &[rows, cols], started);
        out
    }
    fn rope(&self, x: &[f32], s: RopeShape, d: DType) -> Result<Vec<f32>> {
        checkpoint(&self.cancellation)?;
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
    fn sequence(&self, op: &'static str, shape: &[usize]) -> Option<Box<dyn Sequence + '_>> {
        // Trace observers need host values between operations, so observed
        // runs keep the per-operation path.
        if self.observer.borrow().is_some() {
            return None;
        }
        Some(Box::new(MetalSequence {
            backend: self,
            inner: self.device.sequence(op),
            slots: Vec::new(),
            op,
            shape: shape.to_vec(),
            started: Instant::now(),
        }))
    }
    fn buffer(&self, len: usize) -> Option<Result<Rc<dyn DeviceBuffer>>> {
        Some(
            self.device
                .buffer(len)
                .map(|buffer| Rc::new(KvBuffer(buffer)) as Rc<dyn DeviceBuffer>)
                .map_err(device_error),
        )
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
        let started = Instant::now();
        let result = self
            .0
            .linear_typed(input, bias, rows, inn, out, dtype(d))
            .map_err(device_error);
        record(&self.2, "linear", &[rows, inn, out], started);
        result
    }
    fn embedding(&self, ids: &[i32], width: usize) -> Result<Vec<f32>> {
        checkpoint(&self.1)?;
        let started = Instant::now();
        let result = self.0.embedding(ids, width).map_err(device_error);
        record(&self.2, "embedding", &[ids.len(), width], started);
        result
    }
    fn convolution(
        &self,
        input: &[f32],
        bias: Option<&[f32]>,
        s: ConvShape,
        d: DType,
    ) -> Result<Vec<f32>> {
        self.convolution_batched(input, bias, s, 1, d)
    }
    fn convolution_batched(
        &self,
        input: &[f32],
        bias: Option<&[f32]>,
        s: ConvShape,
        batch: usize,
        d: DType,
    ) -> Result<Vec<f32>> {
        checkpoint(&self.1)?;
        let started = Instant::now();
        let result = self
            .0
            .convolution_batched_typed(
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
                batch,
            )
            .map_err(device_error);
        record(
            &self.2,
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
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }
}

/// A converted MiniMax Music 3 checkpoint, resident on the Metal device.
/// Each request creates fresh AR caches and flow overlap state.
pub struct Music3Runner {
    model: Model,
    device: Music3Device,
    observer: Observer,
    cancellation: Cancellation,
    profile: Profile,
}
impl Music3Runner {
    pub fn open(path: &Path) -> Result<Self> {
        Self::open_with_precision(path, Music3Precision::Checkpoint)
    }
    pub fn open_with_precision(path: &Path, precision: Music3Precision) -> Result<Self> {
        let device = Music3Device::new().map_err(device_error)?;
        let observer = Rc::new(RefCell::new(None));
        let cancellation = Rc::new(RefCell::new(None));
        let profile = Profile::default();
        let backend = Rc::new(Backend {
            device: device.clone(),
            observer: observer.clone(),
            cancellation: cancellation.clone(),
            profile: profile.clone(),
            silu: RefCell::new(None),
        });
        let model = Model::load_converted_with_backend_and_precision(path, backend, precision)?;
        // The runner serves the app, server, FFI and CLI, so it opens in the
        // faster profile. The MLX parity gates call `set_flow_profile(Exact)`.
        model.set_flow_profile(FlowProfile::Fast);
        Ok(Self {
            model,
            device,
            observer,
            cancellation,
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
    /// Select the named flow profile. The runner opens in
    /// [`FlowProfile::Fast`]; [`FlowProfile::Exact`] restores the MLX
    /// reference trajectory (Euler, both CFG branches, 30 default steps).
    pub fn set_flow_profile(&self, profile: FlowProfile) {
        self.model.set_flow_profile(profile);
    }
    /// Override the flow unconditional CFG schedule (the profile sets it by
    /// default; `FlowGuidance::EXACT` evaluates both branches every step).
    pub fn set_flow_guidance(
        &self,
        guidance: audio::music::minimax_music3::FlowGuidance,
    ) -> Result<()> {
        self.model.set_flow_guidance(guidance)
    }
    /// Override the flow integrator (the profile sets it by default;
    /// `FlowSolver::Euler` is the reference trajectory).
    pub fn set_flow_solver(&self, solver: audio::music::minimax_music3::FlowSolver) {
        self.model.set_flow_solver(solver);
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
        stats.sort_by_key(|stat| std::cmp::Reverse(stat.total));
        stats
    }
    pub fn reset_dispatch_profile(&self) {
        self.profile.borrow_mut().clear();
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
        self.generate_text_controlled(request, cancel, |_| Control::Continue)
    }

    /// Atomic cancellation remains independent of callbacks and native model state.
    pub fn generate_text_controlled(
        &self,
        request: &TextGenerateRequest,
        cancel: Arc<AtomicBool>,
        mut on_progress: impl FnMut(Progress) -> Control,
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
        let result = self.model.generate_text_with_progress(request, |event| {
            if checkpoint(&self.cancellation).is_err() {
                Control::Cancel
            } else {
                on_progress(event)
            }
        })?;
        checkpoint(&self.cancellation)?;
        result
            .map(|(generation, _)| generation)
            .ok_or_else(|| SpeechError::Input {
                why: "audio job cancelled".into(),
            })
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
    /// [`Self::generate_text_timed`] with milestone callbacks. Return
    /// [`Control::Cancel`] to stop at the next frame or chunk boundary; the
    /// result is then `Ok(None)` and the runner is ready for the next request.
    pub fn generate_text_with_progress(
        &self,
        request: &TextGenerateRequest,
        on_progress: impl FnMut(Progress) -> Control,
    ) -> Result<Option<(Generation, StageTimings)>> {
        self.model.generate_text_with_progress(request, on_progress)
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
