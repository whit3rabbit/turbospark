//! Kokoro Metal runner. Rc model/device state stays on the native audio worker.
use audio::backend::{
    AttentionShape, ComputeBackend, ConvShape, DType, DeviceWeight, WeightData, WeightEncoding,
};
use audio::tts::kokoro::{KokoroSynthesizer, SynthesisRequest};
use audio::{Result, SpeechError};
use gpu::{Music3Device, Music3Encoding, Music3Weight};
use std::cell::RefCell;
use std::collections::HashMap;
use std::path::Path;
use std::rc::Rc;

type Counts = Rc<RefCell<HashMap<&'static str, usize>>>;
type Observer = Rc<RefCell<Option<Box<dyn FnMut(&str, &[f32], &[usize])>>>>;
fn error(e: gpu::GpuError) -> SpeechError {
    SpeechError::Unsupported {
        why: format!("Kokoro Metal: {e}"),
    }
}
fn count(counts: &Counts, op: &'static str) {
    *counts.borrow_mut().entry(op).or_default() += 1;
}
struct Backend {
    device: Music3Device,
    counts: Counts,
    observer: Observer,
}
struct Weight {
    inner: Music3Weight,
    counts: Counts,
}
impl ComputeBackend for Backend {
    fn load_weight(&self, data: WeightData<'_>) -> Result<Rc<dyn DeviceWeight>> {
        if data.encoding != WeightEncoding::F32 || data.dtype != DType::F32 {
            return Err(SpeechError::Unsupported {
                why: "Kokoro Metal supports the verified F32 checkpoint only".into(),
            });
        }
        let inner = self
            .device
            .load_weight(data.shape, data.bytes, Music3Encoding::F32, &[], &[], &[])
            .map_err(error)?;
        Ok(Rc::new(Weight {
            inner,
            counts: self.counts.clone(),
        }))
    }
    fn attention(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        s: AttentionShape,
        d: DType,
    ) -> Result<Vec<f32>> {
        if d != DType::F32
            || s.batch != 1
            || s.queries != s.keys
            || s.heads != 1
            || s.kv_heads != 1
            || s.kv_time_major
            || s.causal
            || s.offset != 0
        {
            return Err(SpeechError::Unsupported {
                why: "Kokoro attention requires one noncausal F32 head with matching lengths"
                    .into(),
            });
        }
        let out = self
            .device
            .kokoro_attention_f32(q, k, v, s.queries, s.dim)
            .map_err(error)?;
        count(&self.counts, "attention");
        Ok(out)
    }
    fn normalize(
        &self,
        x: &[f32],
        rows: usize,
        cols: usize,
        eps: f32,
        layout: audio::backend::NormalizationLayout,
    ) -> Result<Vec<f32>> {
        self.device
            .kokoro_normalize_f32(
                x,
                rows,
                cols,
                eps,
                layout == audio::backend::NormalizationLayout::Columns,
            )
            .map_err(error)
    }
    fn sine(&self, x: &[f32]) -> Result<Vec<f32>> {
        self.device.kokoro_unary_f32(x, true).map_err(error)
    }
    fn tanh(&self, x: &[f32]) -> Result<Vec<f32>> {
        self.device.kokoro_unary_f32(x, false).map_err(error)
    }
    fn gelu_erf(&self, x: &[f32]) -> Result<Vec<f32>> {
        self.device.kokoro_gelu_f32(x).map_err(error)
    }
    fn weight_norm(
        &self,
        x: &[f32],
        g: &[f32],
        rows: usize,
        input: usize,
        kernel: usize,
    ) -> Result<Vec<f32>> {
        self.device
            .kokoro_weight_norm_f32(x, g, rows, input, kernel)
            .map_err(error)
    }
    fn layer_norm(
        &self,
        x: &[f32],
        w: &[f32],
        b: Option<&[f32]>,
        rows: usize,
        cols: usize,
        eps: f32,
        d: DType,
    ) -> Result<Vec<f32>> {
        if d != DType::F32 {
            return Err(SpeechError::Unsupported {
                why: "Kokoro LayerNorm must remain F32".into(),
            });
        }
        self.device
            .kokoro_layer_norm_f32(x, w, b, rows, cols, eps)
            .map_err(error)
    }
    fn normal_from_uniform(&self, x: &[f32], d: DType) -> Result<Vec<f32>> {
        if d != DType::F32 {
            return Err(SpeechError::Unsupported {
                why: "Kokoro RNG must remain F32".into(),
            });
        }
        self.device.kokoro_normal_from_uniform_f32(x).map_err(error)
    }
    fn cumulative_sum(&self, x: &[f32], rows: usize, columns: usize) -> Result<Vec<f32>> {
        self.device
            .cumulative_sum_f32(x, rows, columns)
            .map_err(error)
    }
    fn stft_magnitude_phase(&self, x: &[f32], window: &[f32], hop: usize) -> Result<Vec<f32>> {
        self.device
            .stft20_magnitude_phase(x, window, hop)
            .map_err(error)
    }
    fn trace(&self, stage: &str, values: &[f32], _: DType, shape: &[usize]) {
        if let Some(observer) = self.observer.borrow_mut().as_mut() {
            observer(stage, values, shape);
        }
    }
}
impl DeviceWeight for Weight {
    fn linear(
        &self,
        x: &[f32],
        bias: Option<&[f32]>,
        rows: usize,
        input: usize,
        output: usize,
        d: DType,
    ) -> Result<Vec<f32>> {
        if d != DType::F32 {
            return Err(SpeechError::Unsupported {
                why: "Kokoro linear must remain F32".into(),
            });
        }
        let out = self
            .inner
            .linear_f32(x, bias, rows, input, output)
            .map_err(error)?;
        count(&self.counts, "linear");
        Ok(out)
    }
    fn embedding(&self, ids: &[i32], width: usize) -> Result<Vec<f32>> {
        let out = self.inner.embedding(ids, width).map_err(error)?;
        count(&self.counts, "embedding");
        Ok(out)
    }
    fn convolution(
        &self,
        x: &[f32],
        bias: Option<&[f32]>,
        s: ConvShape,
        d: DType,
    ) -> Result<Vec<f32>> {
        self.convolution_grouped(x, bias, s, 1, d)
    }
    fn convolution_grouped(
        &self,
        x: &[f32],
        bias: Option<&[f32]>,
        s: ConvShape,
        groups: usize,
        d: DType,
    ) -> Result<Vec<f32>> {
        if d != DType::F32 {
            return Err(SpeechError::Unsupported {
                why: "Kokoro convolution must remain F32".into(),
            });
        }
        let out = self
            .inner
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
            .map_err(error)?;
        count(&self.counts, "convolution");
        Ok(out)
    }
    fn lstm_recurrence(
        &self,
        projection: &[f32],
        hidden: usize,
        backward: bool,
    ) -> Result<Vec<f32>> {
        let out = self
            .inner
            .lstm_recurrence(projection, hidden, backward)
            .map_err(error)?;
        count(&self.counts, "lstm");
        Ok(out)
    }
}

pub struct KokoroRunner {
    model: KokoroSynthesizer,
    device: Music3Device,
    counts: Counts,
    observer: Observer,
}
impl KokoroRunner {
    pub fn open(path: &Path) -> Result<Self> {
        let device = Music3Device::new().map_err(error)?;
        let counts = Counts::default();
        let observer = Observer::default();
        let backend = Rc::new(Backend {
            device: device.clone(),
            counts: counts.clone(),
            observer: observer.clone(),
        });
        let model = KokoroSynthesizer::open_with_backend(path, Some(backend))?;
        if !model.using_device() {
            return Err(SpeechError::Unsupported {
                why: "Kokoro device backend was not selected".into(),
            });
        }
        Ok(Self {
            model,
            device,
            counts,
            observer,
        })
    }
    pub fn estimate_run_bytes(&self, request: &SynthesisRequest) -> Result<u64> {
        Ok(self
            .model
            .activation_reserve_bytes(request)?
            .saturating_add(
                self.model
                    .device_weight_reserve_bytes()
                    .saturating_sub(self.resident_weight_bytes() as u64),
            ))
    }
    #[doc(hidden)]
    pub fn diagnostic_vocoder(
        &self,
        features: &[f32],
        style: &[f32],
        f0: &[f32],
        seed: u64,
        source: Option<&[f32]>,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        gpu::autorelease_pool(|| {
            self.model
                .diagnostic_vocoder(features, style, f0, seed, source)
        })
    }
    pub fn using_metal(&self) -> bool {
        self.model.using_device()
    }
    pub fn resident_weight_bytes(&self) -> usize {
        self.device.resident_weight_bytes()
    }
    pub fn dispatch_counts(&self) -> HashMap<&'static str, usize> {
        self.counts.borrow().clone()
    }
    pub fn set_trace_observer(&self, observer: impl FnMut(&str, &[f32], &[usize]) + 'static) {
        *self.observer.borrow_mut() = Some(Box::new(observer));
    }
    pub fn clear_trace_observer(&self) {
        self.observer.borrow_mut().take();
    }
    pub fn synthesize_controlled(
        &self,
        request: &SynthesisRequest,
        seed: u64,
        checkpoint: impl FnMut(usize, usize) -> Result<()>,
        emit: impl FnMut(Vec<f32>) -> bool,
    ) -> Result<()> {
        gpu::autorelease_pool(|| {
            self.model
                .synthesize_controlled(request, seed, checkpoint, emit)
        })
    }
}
