//! Inworld Voice Isolator v1: a band-split dual-path network over the 48 kHz
//! STFT that keeps the voice and removes everything else.
//!
//! Ported from the inworld reference (`src/inworld/voice_isolation/model.py`
//! at `c73b3fab81a870410cbac50d2f05f25edc4d10aa`, Apache-2.0). Each frame's
//! spectrum is cut into bands, every band becomes one token, blocks of time
//! attention (rotary positions, exclusive self attention) and dual-path Mamba
//! refine the tokens, and per-band heads predict a bounded complex mask and a
//! bounded residual per bin. The output is the input plus the iSTFT of that
//! correction.
//!
//! The architecture is read from the checkpoint's safetensors metadata
//! (`config`), never guessed; fields this port does not implement are
//! refused. Both released files load: `denoiser.safetensors` (offline,
//! two-sided) and `denoiser_realtime.safetensors` (causal, trained look-ahead
//! positions).

mod layers;
pub mod stream;
#[cfg(test)]
mod tests;

pub use stream::VoiceIsolatorStream;

use std::cell::RefCell;
use std::collections::HashSet;
use std::path::Path;
use std::rc::Rc;

use turbospark_model_io::safetensors::SafetensorsFile;

use crate::backend::ComputeBackend;
use crate::fft::ComplexF32;
use crate::nn::{self, LayerNorm, Linear};
use crate::stft::{self, StftOptions, StftPaddingMode, StftWindowPlacement};
use crate::{Result, SpeechError};

use layers::{Block, Decoder, Exec, LN_EPS};

/// Weight files [`VoiceIsolator::open`] looks for, in order: the released
/// offline file, the released realtime file, a generic name.
pub const WEIGHT_FILES: [&str; 3] = [
    "denoiser.safetensors",
    "denoiser_realtime.safetensors",
    "model.safetensors",
];

/// Architecture of a checkpoint (`VoiceIsolatorConfig`).
#[derive(Clone, Debug, PartialEq)]
pub struct VoiceIsolatorConfig {
    pub width: usize,
    /// `true` for a transformer block, `false` for a Mamba block.
    pub attention_blocks: Vec<bool>,
    pub num_heads: usize,
    pub n_fft: usize,
    pub hop_length: usize,
    pub sample_rate: u32,
    pub band_plan: Vec<(f64, f64)>,
    pub correction_bound: f32,
    pub chunk_seconds: f64,
    pub overlap_seconds: f64,
    pub exclusive_time: bool,
    pub exclusive_band: bool,
    pub mamba_state: usize,
    pub mamba_conv: usize,
    pub mamba_expand: usize,
    pub causal: bool,
    pub lookahead_frames: usize,
    pub lookahead_frames_choices: Vec<usize>,
    pub attention_window_frames: usize,
    pub offline_passes: usize,
}

/// Fields of a checkpoint config that only training reads.
const TRAINING_ONLY: &[&str] = &[
    "activation_checkpointing",
    "checkpoint_every",
    "attention_backend",
    "offline_mode_prob",
    "dual_distill",
    "hint_sha256",
    "hint_bf16",
    "grid_width",
    "grid_hidden",
    "grid_kernel",
    "grid_stride",
    "grid_heads",
    "grid_qk_channels",
    "grid_output",
    "grid_normalize_input",
    "decision_head",
];
const KNOWN: &[&str] = &[
    "width",
    "block_types",
    "num_heads",
    "n_fft",
    "hop_length",
    "sample_rate",
    "band_plan",
    "correction_bound",
    "chunk_seconds",
    "overlap_seconds",
    "exclusive_self_attention",
    "exclusive_axes",
    "mamba_state",
    "mamba_conv",
    "mamba_expand",
    "causal",
    "lookahead_frames",
    "lookahead_frames_choices",
    "dual_mode",
    "mode_film",
    "attention_window_frames",
    "offline_passes",
];

fn bad(field: &str, why: impl std::fmt::Display) -> SpeechError {
    nn::bad_config(field, why)
}

impl VoiceIsolatorConfig {
    /// Parses the architecture dictionary the reference embeds (the
    /// `config` metadata, or its `modeling.compact_family` entry), with the
    /// reference's defaults and refusals.
    pub fn from_json(raw: &serde_json::Value) -> Result<Self> {
        let raw = match raw.get("modeling").and_then(|m| m.get("compact_family")) {
            Some(inner) => inner,
            None => raw,
        };
        let obj = raw
            .as_object()
            .filter(|o| o.contains_key("width"))
            .ok_or_else(|| bad("config", "no embedded architecture config"))?;
        let falsy = |v: &serde_json::Value| match v {
            serde_json::Value::Null => true,
            serde_json::Value::Bool(b) => !b,
            serde_json::Value::Number(n) => n.as_f64() == Some(0.0),
            serde_json::Value::String(s) => s.is_empty(),
            serde_json::Value::Array(a) => a.is_empty(),
            serde_json::Value::Object(o) => o.is_empty(),
        };
        for (name, expected) in [
            ("representation", Some("bands")),
            ("decoder_type", Some("complex_mask_residual")),
            ("hint_checkpoint", None),
        ] {
            if let Some(v) = obj.get(name) {
                let ok = match expected {
                    Some(e) => v.as_str() == Some(e),
                    None => v.is_null(),
                };
                if !ok {
                    return Err(bad(
                        name,
                        format!("this port implements {expected:?}, not {v}"),
                    ));
                }
            }
        }
        let unknown: Vec<&str> = obj
            .iter()
            .filter(|(k, v)| {
                !KNOWN.contains(&k.as_str())
                    && !TRAINING_ONLY.contains(&k.as_str())
                    && !["representation", "decoder_type", "hint_checkpoint"].contains(&k.as_str())
                    && !falsy(v)
            })
            .map(|(k, _)| k.as_str())
            .collect();
        if !unknown.is_empty() {
            return Err(bad(
                "config",
                format!("fields this port does not implement: {unknown:?}"),
            ));
        }
        let uint = |name: &str, default: usize| -> Result<usize> {
            match obj.get(name) {
                None => Ok(default),
                Some(v) => v
                    .as_u64()
                    .map(|n| n as usize)
                    .ok_or_else(|| bad(name, "expected a non-negative integer")),
            }
        };
        let float = |name: &str, default: f64| -> Result<f64> {
            match obj.get(name) {
                None => Ok(default),
                Some(v) => v.as_f64().ok_or_else(|| bad(name, "expected a number")),
            }
        };
        let boolean = |name: &str, default: bool| -> Result<bool> {
            match obj.get(name) {
                None => Ok(default),
                Some(v) => v.as_bool().ok_or_else(|| bad(name, "expected a boolean")),
            }
        };
        if boolean("dual_mode", false)? {
            return Err(bad("dual_mode", "dual-mode checkpoints are not supported"));
        }
        if boolean("mode_film", false)? {
            return Err(bad("mode_film", "mode FiLM checkpoints are not supported"));
        }
        let attention_blocks = match obj.get("block_types") {
            None => vec![true, false, true, false, true, false, true],
            Some(v) => v
                .as_array()
                .ok_or_else(|| bad("block_types", "expected a list"))?
                .iter()
                .map(|kind| match kind.as_str() {
                    Some("transformer") => Ok(true),
                    Some("mamba") => Ok(false),
                    _ => Err(bad("block_types", format!("unknown block {kind}"))),
                })
                .collect::<Result<Vec<_>>>()?,
        };
        let band_plan = match obj.get("band_plan") {
            None => vec![
                (100.0, 1000.0),
                (250.0, 4000.0),
                (500.0, 8000.0),
                (1000.0, 16000.0),
                (2000.0, 24000.0),
            ],
            Some(v) => v
                .as_array()
                .ok_or_else(|| bad("band_plan", "expected a list"))?
                .iter()
                .map(|pair| {
                    let p = pair.as_array().filter(|p| p.len() == 2);
                    match p.map(|p| (p[0].as_f64(), p[1].as_f64())) {
                        Some((Some(w), Some(u))) => Ok((w, u)),
                        _ => Err(bad("band_plan", "expected [width_hz, upto_hz] pairs")),
                    }
                })
                .collect::<Result<Vec<_>>>()?,
        };
        let axes = obj
            .get("exclusive_axes")
            .and_then(|v| v.as_str())
            .unwrap_or("time");
        if !["both", "time", "band"].contains(&axes) {
            return Err(bad("exclusive_axes", "expected both, time or band"));
        }
        let exclusive = boolean("exclusive_self_attention", true)?;
        let choices: Vec<usize> = match obj.get("lookahead_frames_choices") {
            None => vec![],
            Some(v) => v
                .as_array()
                .ok_or_else(|| bad("lookahead_frames_choices", "expected a list"))?
                .iter()
                .map(|n| {
                    n.as_u64()
                        .map(|n| n as usize)
                        .ok_or_else(|| bad("lookahead_frames_choices", "expected integers"))
                })
                .collect::<Result<Vec<_>>>()?,
        };
        let config = Self {
            width: uint("width", 320)?,
            attention_blocks,
            num_heads: uint("num_heads", 8)?,
            n_fft: uint("n_fft", 2048)?,
            hop_length: uint("hop_length", 480)?,
            sample_rate: uint("sample_rate", 48_000)? as u32,
            band_plan,
            correction_bound: float("correction_bound", 0.5)? as f32,
            chunk_seconds: float("chunk_seconds", 12.0)?,
            overlap_seconds: float("overlap_seconds", 2.0)?,
            exclusive_time: exclusive && (axes == "both" || axes == "time"),
            exclusive_band: exclusive && (axes == "both" || axes == "band"),
            mamba_state: uint("mamba_state", 16)?,
            mamba_conv: uint("mamba_conv", 4)?,
            mamba_expand: uint("mamba_expand", 2)?,
            causal: boolean("causal", false)?,
            lookahead_frames: uint("lookahead_frames", 0)?,
            lookahead_frames_choices: choices,
            attention_window_frames: uint("attention_window_frames", 0)?,
            offline_passes: uint("offline_passes", 1)?,
        };
        config.validate()?;
        Ok(config)
    }

    /// The reference's `__post_init__` checks.
    fn validate(&self) -> Result<()> {
        if self.width == 0 || self.num_heads == 0 || self.width % (2 * self.num_heads) != 0 {
            return Err(bad(
                "width",
                "must be a positive multiple of twice num_heads",
            ));
        }
        if self.attention_blocks.is_empty() {
            return Err(bad("block_types", "at least one block"));
        }
        if self.n_fft < 4
            || self.n_fft % 2 != 0
            || self.hop_length == 0
            || self.hop_length > self.n_fft / 2
        {
            return Err(bad(
                "n_fft",
                "n_fft must be even and hop_length within (0, n_fft / 2]",
            ));
        }
        if self.sample_rate == 0
            || self.band_plan.is_empty()
            || self.band_plan.iter().any(|&(w, u)| w < 1.0 || u < 1.0)
        {
            return Err(bad("band_plan", "positive widths and limits"));
        }
        if !(self.correction_bound > 0.0 && self.correction_bound < 1.0) {
            return Err(bad("correction_bound", "must lie in (0, 1)"));
        }
        if self.chunk_seconds * f64::from(self.sample_rate) < self.n_fft as f64
            || !(self.overlap_seconds > 0.0 && self.overlap_seconds <= self.chunk_seconds / 2.0)
        {
            return Err(bad(
                "chunk_seconds",
                "chunk must cover one window, overlap in (0, chunk / 2]",
            ));
        }
        if self.mamba_state == 0 || self.mamba_conv == 0 || self.mamba_expand == 0 {
            return Err(bad("mamba_state", "Mamba dimensions must be positive"));
        }
        let c = &self.lookahead_frames_choices;
        if c.windows(2).any(|w| w[0] >= w[1]) {
            return Err(bad("lookahead_frames_choices", "distinct and ascending"));
        }
        if !c.is_empty() && !c.contains(&self.lookahead_frames) {
            return Err(bad(
                "lookahead_frames",
                "must be one of lookahead_frames_choices",
            ));
        }
        if (self.lookahead_frames > 0 || !c.is_empty()) && !self.causal {
            return Err(bad("lookahead_frames", "a look-ahead needs a causal model"));
        }
        if self.attention_window_frames > 0 && !self.causal {
            return Err(bad("attention_window_frames", "needs a causal model"));
        }
        if self.offline_passes == 0 || (self.offline_passes > 1 && self.causal) {
            return Err(bad(
                "offline_passes",
                "at least 1, and 1 for a causal model",
            ));
        }
        Ok(())
    }

    /// Bin ranges `[start, end)` of the bands, covering every bin once
    /// (`band_edges`).
    pub fn band_edges(&self) -> Vec<(usize, usize)> {
        let bins = self.n_fft / 2 + 1;
        let hz_per_bin = f64::from(self.sample_rate) / self.n_fft as f64;
        let mut edges = Vec::new();
        let mut start = 0usize;
        for &(width_hz, upto_hz) in &self.band_plan {
            let width = (round_half_even(width_hz / hz_per_bin) as usize).max(1);
            let upto = bins.min(round_half_even(upto_hz / hz_per_bin) as usize);
            while start < upto {
                let end = (start + width).min(upto);
                edges.push((start, end));
                start = end;
            }
        }
        if start < bins {
            edges.push((start, bins));
        }
        edges
    }

    /// The render window and its crossfade in samples (`window_samples`).
    pub fn window_samples(&self) -> (usize, usize) {
        let rate = f64::from(self.sample_rate);
        let chunk = (self.chunk_seconds * rate) as usize;
        let overlap = ((self.overlap_seconds * rate) as usize).min(chunk / 2);
        (chunk, overlap)
    }
}

/// Python's `round`: halves go to the even neighbour.
fn round_half_even(x: f64) -> f64 {
    let r = x.round();
    if (x - x.trunc()).abs() == 0.5 && r % 2.0 != 0.0 {
        r - x.signum()
    } else {
        r
    }
}

/// Loads tensors and records every name taken, so the model can refuse a
/// checkpoint that carries learned weights it never read.
pub(crate) struct Loader<'a> {
    file: &'a SafetensorsFile,
    taken: RefCell<HashSet<String>>,
}

impl<'a> Loader<'a> {
    fn new(file: &'a SafetensorsFile) -> Self {
        Self {
            file,
            taken: RefCell::new(HashSet::new()),
        }
    }

    pub(crate) fn tensor(&self, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
        self.taken.borrow_mut().insert(name.to_owned());
        nn::load_tensor(self.file, name, shape)
    }

    pub(crate) fn linear(
        &self,
        name: &str,
        input: usize,
        output: usize,
        bias: bool,
    ) -> Result<Linear> {
        self.linear_named(
            &format!("{name}.weight"),
            bias.then(|| format!("{name}.bias")).as_deref(),
            input,
            output,
        )
    }

    pub(crate) fn linear_named(
        &self,
        weight: &str,
        bias: Option<&str>,
        input: usize,
        output: usize,
    ) -> Result<Linear> {
        let w = self.tensor(weight, &[output, input])?;
        let b = bias.map(|b| self.tensor(b, &[output])).transpose()?;
        Ok(Linear::new(w, b, input, output))
    }

    pub(crate) fn norm(&self, name: &str, width: usize) -> Result<LayerNorm> {
        Ok(LayerNorm::new(
            self.tensor(&format!("{name}.weight"), &[width])?,
            Some(self.tensor(&format!("{name}.bias"), &[width])?),
            LN_EPS,
        ))
    }

    /// Names in the file the network never read, outside the reference's
    /// training-only prefixes.
    fn unread(&self) -> Vec<String> {
        const TRAINING: &[&str] = &[
            "discriminator.",
            "disc.",
            "ema.",
            "optimizer.",
            "scheduler.",
            "enhance_decoder.",
            "decision_head.",
        ];
        let taken = self.taken.borrow();
        self.file
            .tensor_names()
            .filter(|n| !taken.contains(*n) && !TRAINING.iter().any(|p| n.starts_with(p)))
            .map(str::to_owned)
            .collect()
    }
}

/// Progress of a render: one step per model pass.
pub type Progress<'a> = &'a mut dyn FnMut(usize, usize) -> bool;

/// The loaded network.
pub struct VoiceIsolator {
    pub config: VoiceIsolatorConfig,
    edges: Vec<(usize, usize)>,
    encoders: Vec<(LayerNorm, Linear)>,
    band_embedding: Vec<f32>,
    /// `[positions, width]` for the trained look-ahead choices.
    lookahead_embedding: Option<Vec<f32>>,
    lookahead_index: usize,
    blocks: Vec<Block>,
    decoder: Decoder,
    window: Vec<f32>,
    exec: Exec,
}

impl VoiceIsolator {
    /// Loads the first of [`WEIGHT_FILES`] present in `dir`.
    pub fn open(dir: &Path) -> Result<Self> {
        let name = WEIGHT_FILES
            .iter()
            .find(|name| dir.join(name).is_file())
            .ok_or_else(|| {
                bad(
                    "weights",
                    format!("no {} in {}", WEIGHT_FILES.join(" or "), dir.display()),
                )
            })?;
        Self::open_file(&dir.join(name))
    }

    /// Loads one checkpoint file.
    pub fn open_file(path: &Path) -> Result<Self> {
        let file = SafetensorsFile::open(path)
            .map_err(|e| bad("weights", format!("{}: {e}", path.display())))?;
        Self::load(&file)
    }

    /// Builds the network from an open checkpoint whose metadata carries the
    /// architecture.
    pub fn load(file: &SafetensorsFile) -> Result<Self> {
        let text = file
            .metadata()
            .get("config")
            .ok_or_else(|| bad("config", "safetensors metadata lacks the 'config' entry"))?;
        let raw: serde_json::Value = serde_json::from_str(text).map_err(|e| bad("config", e))?;
        let config = VoiceIsolatorConfig::from_json(&raw)?;
        let l = Loader::new(file);
        let width = config.width;
        let edges = config.band_edges();
        let mut encoders = Vec::with_capacity(edges.len());
        for (i, &(start, end)) in edges.iter().enumerate() {
            let n = 2 * (end - start);
            encoders.push((
                l.norm(&format!("encoders.{i}.0"), n)?,
                l.linear(&format!("encoders.{i}.1"), n, width, true)?,
            ));
        }
        let band_embedding = l.tensor("band_embedding", &[edges.len(), width])?;
        let (lookahead_embedding, lookahead_index) = if config.lookahead_frames_choices.is_empty() {
            (None, 0)
        } else {
            let positions = config.lookahead_frames_choices.len();
            let table = l.tensor("lookahead_embedding.weight", &[positions, width])?;
            let index = config
                .lookahead_frames_choices
                .iter()
                .position(|&c| c == config.lookahead_frames)
                .expect("validated");
            (Some(table), index)
        };
        let mut blocks = Vec::with_capacity(config.attention_blocks.len());
        for (i, &attention) in config.attention_blocks.iter().enumerate() {
            let name = format!("blocks.{i}");
            blocks.push(if attention {
                Block::Attention {
                    time: layers::TimeBlock::load(
                        &l,
                        &format!("{name}.time_transformer"),
                        width,
                        config.num_heads,
                        config.exclusive_time,
                        config.attention_window_frames,
                    )?,
                    band: layers::BandBlock::load(
                        &l,
                        &format!("{name}.band_transformer"),
                        width,
                        config.num_heads,
                        config.exclusive_band,
                    )?,
                }
            } else {
                let m = (config.mamba_state, config.mamba_conv, config.mamba_expand);
                Block::Mamba {
                    time: layers::BiMamba::load(
                        &l,
                        &format!("{name}.time_mamba"),
                        width,
                        m.0,
                        m.1,
                        m.2,
                        config.causal,
                    )?,
                    band: layers::BiMamba::load(
                        &l,
                        &format!("{name}.freq_mamba"),
                        width,
                        m.0,
                        m.1,
                        m.2,
                        false,
                    )?,
                }
            });
        }
        let decoder = Decoder::load(&l, width, &edges)?;
        let unread = l.unread();
        if !unread.is_empty() {
            return Err(bad(
                "weights",
                format!("tensors this architecture does not use: {unread:?}"),
            ));
        }
        let n = config.n_fft;
        // torch.hann_window: periodic.
        let window = (0..n)
            .map(|i| (0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / n as f64).cos()) as f32)
            .collect();
        Ok(Self {
            config,
            edges,
            encoders,
            band_embedding,
            lookahead_embedding,
            lookahead_index,
            blocks,
            decoder,
            window,
            exec: Exec::default(),
        })
    }

    /// Attaches an optional device backend for the projections and the
    /// non-causal attention. `None` keeps everything on the host.
    pub fn set_backend(&mut self, backend: Option<Rc<dyn ComputeBackend>>) {
        self.exec = Exec::new(backend);
    }

    /// Whether a device backend is attached.
    pub fn on_device(&self) -> bool {
        self.exec.on_device()
    }

    /// Device calls that fell back to the host since the backend was set.
    pub fn device_fallbacks(&self) -> usize {
        self.exec.fallbacks()
    }

    pub fn sample_rate(&self) -> u32 {
        self.config.sample_rate
    }

    /// Whether every frame sees its whole chunk.
    pub fn offline(&self) -> bool {
        !self.config.causal
    }

    /// The look-ahead in frames in force (zero offline).
    pub fn lookahead_frames(&self) -> usize {
        if self.config.lookahead_frames_choices.is_empty() {
            self.config.lookahead_frames
        } else {
            self.config.lookahead_frames_choices[self.lookahead_index]
        }
    }

    /// Moves a causal checkpoint to one of its trained look-ahead positions.
    pub fn set_lookahead(&mut self, frames: usize) -> Result<()> {
        let choices = &self.config.lookahead_frames_choices;
        if choices.is_empty() {
            if frames != self.config.lookahead_frames || !self.config.causal {
                return Err(bad(
                    "lookahead_frames",
                    "this checkpoint has no streaming positions",
                ));
            }
            return Ok(());
        }
        self.lookahead_index = choices.iter().position(|&c| c == frames).ok_or_else(|| {
            bad(
                "lookahead_frames",
                format!("{frames} is not among the trained positions {choices:?}"),
            )
        })?;
        Ok(())
    }

    pub fn bands(&self) -> usize {
        self.edges.len()
    }

    fn stft_options(&self) -> StftOptions {
        StftOptions {
            fft_size: self.config.n_fft,
            hop: self.config.hop_length,
            window: self.window.clone(),
            center: true,
        }
    }

    /// Complex spectra `[frames][bins]` (torch.stft, center, reflect unless
    /// the clip is too short to reflect).
    pub(crate) fn stft(&self, audio: &[f32]) -> Result<Vec<Vec<ComplexF32>>> {
        let mode = if audio.len() > self.config.n_fft / 2 {
            StftPaddingMode::Reflect
        } else {
            StftPaddingMode::Constant
        };
        stft::stft_with_modes(audio, &self.stft_options(), mode, StftWindowPlacement::Left)
            .map_err(|e| SpeechError::Input { why: e.to_string() })
    }

    /// Band tokens `[bands, frames, width]`.
    pub(crate) fn encode(&self, spectra: &[Vec<ComplexF32>]) -> Vec<f32> {
        let frames = spectra.len();
        let width = self.config.width;
        let mut out = vec![0.0f32; self.edges.len() * frames * width];
        for (b, ((norm, linear), &(start, end))) in
            self.encoders.iter().zip(&self.edges).enumerate()
        {
            let n = 2 * (end - start);
            let mut x = Vec::with_capacity(frames * n);
            for frame in spectra {
                for bin in &frame[start..end] {
                    x.push(bin.re);
                    x.push(bin.im);
                }
            }
            let h = layers::layer_norm(norm, &x, frames);
            let y = self.exec.linear(linear, &h, frames);
            let dst = &mut out[b * frames * width..(b + 1) * frames * width];
            let emb = &self.band_embedding[b * width..(b + 1) * width];
            let look = self
                .lookahead_embedding
                .as_ref()
                .map(|t| &t[self.lookahead_index * width..(self.lookahead_index + 1) * width]);
            for (t, row) in dst.chunks_exact_mut(width).enumerate() {
                for (j, v) in row.iter_mut().enumerate() {
                    *v = y[t * width + j] + emb[j] + look.map_or(0.0, |l| l[j]);
                }
            }
        }
        out
    }

    /// Backbone features through every block, calling `observe(i, x)` after
    /// block `i` (tests record the per-block traces through it).
    pub(crate) fn features(
        &self,
        mut x: Vec<f32>,
        frames: usize,
        mut observe: impl FnMut(usize, &[f32]),
    ) -> Vec<f32> {
        for (i, block) in self.blocks.iter().enumerate() {
            block.forward(
                &self.exec,
                &mut x,
                self.edges.len(),
                frames,
                self.config.causal,
            );
            observe(i, &x);
        }
        x
    }

    /// Decoder output `[frames, bins, 4]`, read ahead for a causal model.
    pub(crate) fn decode(&self, features: &[f32], frames: usize) -> Vec<f32> {
        let raw = self
            .decoder
            .forward(&self.exec, features, frames, &self.edges);
        let ahead = if self.config.causal {
            self.lookahead_frames()
        } else {
            0
        };
        if ahead == 0 || frames == 0 {
            return raw;
        }
        let stride = raw.len() / frames;
        let mut out = vec![0.0f32; raw.len()];
        for t in 0..frames {
            let src = (t + ahead).min(frames - 1);
            out[t * stride..(t + 1) * stride]
                .copy_from_slice(&raw[src * stride..(src + 1) * stride]);
        }
        out
    }

    /// The input plus the iSTFT of the decoded correction.
    pub(crate) fn synthesize(
        &self,
        audio: &[f32],
        spectra: &[Vec<ComplexF32>],
        raw: &[f32],
    ) -> Result<Vec<f32>> {
        let bins = self.config.n_fft / 2 + 1;
        let bound = self.config.correction_bound;
        let corrected: Vec<Vec<ComplexF32>> = spectra
            .iter()
            .enumerate()
            .map(|(t, frame)| {
                let mut row: Vec<ComplexF32> = frame
                    .iter()
                    .enumerate()
                    .map(|(f, s)| {
                        correction(*s, &raw[(t * bins + f) * 4..(t * bins + f + 1) * 4], bound)
                    })
                    .collect();
                // torch.istft's irfft ignores the imaginary part of the DC and
                // Nyquist bins; the correction gives them one.
                row[0].im = 0.0;
                row[bins - 1].im = 0.0;
                row
            })
            .collect();
        let mut delta = stft::istft(&corrected, &self.stft_options(), audio.len())
            .map_err(|e| SpeechError::Input { why: e.to_string() })?;
        // torch.istft(length=...) zero-fills past the last frame.
        delta.resize(audio.len(), 0.0);
        Ok(audio.iter().zip(&delta).map(|(a, d)| a + d).collect())
    }

    /// One forward pass over a whole clip at the model rate (`forward`).
    pub fn forward(&self, audio: &[f32]) -> Result<Vec<f32>> {
        if audio.is_empty() || audio.iter().any(|v| !v.is_finite()) {
            return Err(SpeechError::Input {
                why: "expected nonempty finite mono audio".into(),
            });
        }
        let spectra = self.stft(audio)?;
        let frames = spectra.len();
        let tokens = self.encode(&spectra);
        let features = self.features(tokens, frames, |_, _| {});
        let raw = self.decode(&features, frames);
        self.synthesize(audio, &spectra, &raw)
    }

    /// Isolates mono audio of any length at the model rate in overlapping
    /// chunks (`enhance`), each further pass over the previous output.
    pub fn isolate(&self, audio: &[f32]) -> Result<Vec<f32>> {
        self.isolate_controlled(audio, &mut |_, _| true)
    }

    /// The model passes [`isolate`](Self::isolate) runs for `samples`.
    pub fn passes(&self, samples: usize) -> usize {
        let (chunk, overlap) = self.config.window_samples();
        let per = if samples <= chunk {
            1
        } else {
            (samples - chunk).div_ceil(chunk - overlap) + 1
        };
        per * self.config.offline_passes
    }

    /// [`isolate`](Self::isolate), calling `progress(done, total)` before
    /// each model pass and once after the last; false stops before the next
    /// pass. The output is identical to the uncontrolled call.
    pub fn isolate_controlled(&self, audio: &[f32], progress: Progress<'_>) -> Result<Vec<f32>> {
        let total = self.passes(audio.len());
        let mut done = 0usize;
        let mut out = audio.to_vec();
        for _ in 0..self.config.offline_passes {
            out = self.render(&out, &mut done, total, progress)?;
        }
        step(progress, done, total)?;
        Ok(out)
    }

    fn render(
        &self,
        audio: &[f32],
        done: &mut usize,
        total: usize,
        progress: Progress<'_>,
    ) -> Result<Vec<f32>> {
        let (chunk, overlap) = self.config.window_samples();
        if chunk < self.config.hop_length {
            return Err(bad("chunk_seconds", "the window is shorter than one hop"));
        }
        let n = audio.len();
        if n <= chunk {
            step(progress, *done, total)?;
            *done += 1;
            return self.forward(audio);
        }
        let mut starts: Vec<usize> = (0..n - chunk).step_by(chunk - overlap).collect();
        starts.push(n - chunk);
        let mut out = vec![0.0f32; n];
        let mut weight = vec![0.0f32; n];
        for (index, &start) in starts.iter().enumerate() {
            step(progress, *done, total)?;
            let end = start + chunk;
            let mut w = vec![1.0f32; chunk];
            let ramp = |i: usize| {
                if overlap > 1 {
                    i as f32 / (overlap - 1) as f32
                } else {
                    0.0
                }
            };
            if index > 0 {
                let fade_in = starts[index - 1] + chunk - overlap - start;
                w[..fade_in].fill(0.0);
                for i in 0..overlap {
                    w[fade_in + i] = ramp(i);
                }
            }
            if overlap > 0 && end < n {
                for i in 0..overlap {
                    w[chunk - overlap + i] = 1.0 - ramp(i);
                }
            }
            let y = self.forward(&audio[start..end])?;
            for i in 0..chunk {
                out[start + i] += y[i] * w[i];
                weight[start + i] += w[i];
            }
            *done += 1;
        }
        for (o, w) in out.iter_mut().zip(&weight) {
            *o /= w.max(1e-6);
        }
        Ok(out)
    }
}

/// `spectrum * mask + residual` for one bin from its four raw values
/// (`correction` and `bounded_correction`).
fn correction(s: ComplexF32, raw: &[f32], bound: f32) -> ComplexF32 {
    let (mr, mi) = (2.0 * raw[0].tanh(), 2.0 * raw[1].tanh());
    let (c0, c1) = (raw[2].tanh(), raw[3].tanh());
    let norm = (c0 * c0 + c1 * c1).sqrt().max(1.0);
    let magnitude = bound * (s.re * s.re + s.im * s.im).sqrt();
    ComplexF32::new(
        s.re * mr - s.im * mi + c0 / norm * magnitude,
        s.re * mi + s.im * mr + c1 / norm * magnitude,
    )
}

fn step(progress: Progress<'_>, done: usize, total: usize) -> Result<()> {
    if progress(done, total) {
        Ok(())
    } else {
        Err(SpeechError::Input {
            why: "voice isolation cancelled".into(),
        })
    }
}
