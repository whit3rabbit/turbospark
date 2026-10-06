//! MiniMax Music 3: hierarchical autoregressive music generation with a
//! flow-matching latent decoder and a stereo 44.1 kHz vocoder.
//!
//! Reference: `mlx_audio/music/models/minimax_music3/` at mlx-audio
//! commit `feb25a37b07923bae556e59111995071d66afa0d` (upstream
//! attribution: `mikolaj92/minimax-music3-mlx`, Apache-2.0). The
//! pipeline: a Qwen3 AR backbone emits one semantic code per frame, an
//! RVQ depth decoder expands it into residual codebooks, the fused
//! per-frame hiddens are conditioned through a flow-matching
//! transformer (Euler-integrated in overlapping 200-frame chunks), and
//! a DAC-style vocoder decodes each chunk to stereo audio that is
//! crop-stitched at the overlaps.
//!
//! Sampling uses the MLX-compatible RNG in [`rng`]: uniform draws and
//! top-k categorical decisions are bit-matched to the reference, so
//! recorded AR traces reproduce exactly. The flow-stage initial noise
//! uses an f64 `erfinv` that agrees with Metal's to f32 rounding only
//! (see [`rng`] docs); parity tests therefore replay recorded noise.
//!
//! Official checkpoints use the Qwen tokenizer from the `tokenizer`
//! directory of the converted tree. [`Model::generate_text`] assembles
//! and tokenizes caption plus lyrics; [`Model::generate`] remains the
//! low-level pre-tokenized reference path.

pub mod backend;
pub mod precision;
pub use output_stats::{output_stats, OutputStats};
pub use precision::Music3Precision;
pub use stage_timings::StageTimings;
mod conv;
mod depth;
mod dit;
mod euler;
mod fusion;
mod output_stats;
mod prompt;
mod qwen3;
mod rng;
mod stage_timings;
mod vocoder;
mod weights;

#[cfg(test)]
mod stress;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod timing;

use std::cell::RefCell;
use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

use tokenizers::Tokenizer;
use turbospark_model_io::safetensors::SafetensorsFile;

use crate::quant::QuantScheme;
use crate::Result;
use crate::SpeechError;

use backend::{ComputeBackend, Weight};
use depth::{DepthDecoder, DepthDims};
use dit::{DitDims, FlowMatchingTransformer};
use fusion::{ConditionDims, ConditionEncoder};
use qwen3::{KvCache, Qwen3, Qwen3Dims};
use rng::Key;
use std::rc::Rc;
use vocoder::{Vocoder, VocoderDims};
use weights::{open_converted_shards, LinearQuantization, Tensor, WeightStore};

/// Classifier-free guidance scale of both AR sampling sites.
pub const AR_CFG_SCALE: f32 = 1.5;
/// Top-k applied to the conditional logits when masking the guided
/// semantic distribution.
pub const AR_CFG_TOP_K: usize = 50;
/// Top-k applied inside the top-k sampler itself.
pub const AR_SAMPLING_TOP_K: usize = 50;
/// Classifier-free guidance scale of the flow-matching velocity.
pub const DIT_CFG_SCALE: f32 = 1.7;
/// AR frames per second of audio.
pub const FRAME_RATE: f64 = 25.0;
/// Frame cap of one request (360 s at 25 fps).
pub const MAX_AUDIO_FRAMES: usize = 9_000;
/// Prompt token cap before the AR prefill.
pub const MAX_PROMPT_TOKENS: usize = 5_000;
/// Flow chunk window and hop in AR frames.
pub const CHUNK_FRAMES: usize = 200;
pub const CHUNK_HOP: usize = 100;
/// Vocoder output samples per latent step.
pub const LATENT_HOP_LENGTH: usize = 512;
/// Overlap carried between flow chunks, in latent steps.
pub const OVERLAP_LATENT_LENGTH: usize = 172;
/// Waveform crops that remove the chunk overlap after stitching.
pub const CROP_LEFT_LATENT: usize = 86;
pub const CROP_RIGHT_LATENT: usize = 344 - CROP_LEFT_LATENT;
/// Output sample rate.
pub const SAMPLING_RATE: u32 = 44_100;

/// Checkpoint contract and architecture sizes.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub model_type: String,
    pub hidden_size: usize,
    pub num_codebooks: usize,
    pub audio_vocab_size: usize,
    pub semantic_vocab_size: usize,
    pub vocab_size: usize,
    pub audio_code_offset: usize,
    pub audio_end_token_id: usize,
    pub audio_cfg_token_id: usize,
    pub num_hidden_layers: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f32,
    pub max_position_embeddings: usize,
    pub rope_theta: f32,
    pub tie_word_embeddings: bool,
    pub depth_num_layers: usize,
    pub depth_num_heads: usize,
    pub depth_intermediate_size: usize,
    pub depth_max_position_embeddings: usize,
    pub condition_out_dim: usize,
    pub num_condition_layers: usize,
    pub input_sampling_rate: usize,
    pub input_hop_length: usize,
    pub output_sampling_rate: usize,
    pub output_hop_length: usize,
    pub dit_in_channels: usize,
    pub dit_num_layers: usize,
    pub dit_num_heads: usize,
    pub dit_head_dim: usize,
    pub dit_ff_inner_dim: usize,
    pub dit_rotary_dim: usize,
    pub dit_fourier_dim: usize,
    pub vocoder_input_dim: usize,
    pub vocoder_hidden_dim: usize,
    pub vocoder_upsampling_ratios: Vec<usize>,
    pub sample_rate: u32,
    pub frame_rate: f64,
}

impl Default for ModelConfig {
    fn default() -> Self {
        ModelConfig {
            model_type: "minimax_music3".to_string(),
            hidden_size: 4096,
            num_codebooks: 8,
            audio_vocab_size: 1024,
            semantic_vocab_size: 16_384,
            vocab_size: 200_000,
            audio_code_offset: 151_675,
            audio_end_token_id: 151_670,
            audio_cfg_token_id: 151_654,
            num_hidden_layers: 36,
            intermediate_size: 12_288,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            head_dim: 128,
            rms_norm_eps: 1e-6,
            max_position_embeddings: 10_240,
            rope_theta: 1_000_000.0,
            tie_word_embeddings: false,
            depth_num_layers: 4,
            depth_num_heads: 16,
            depth_intermediate_size: 6144,
            depth_max_position_embeddings: 16,
            condition_out_dim: 2048,
            num_condition_layers: 8,
            input_sampling_rate: 24_000,
            input_hop_length: 960,
            output_sampling_rate: SAMPLING_RATE as usize,
            output_hop_length: LATENT_HOP_LENGTH,
            dit_in_channels: 128,
            dit_num_layers: 36,
            dit_num_heads: 32,
            dit_head_dim: 64,
            dit_ff_inner_dim: 8192,
            dit_rotary_dim: 32,
            dit_fourier_dim: 256,
            vocoder_input_dim: 1024,
            vocoder_hidden_dim: 1536,
            vocoder_upsampling_ratios: vec![8, 8, 4, 2],
            sample_rate: SAMPLING_RATE,
            frame_rate: FRAME_RATE,
        }
    }
}

impl ModelConfig {
    /// The scaled-down configuration used by the upstream tests and
    /// this crate's fixtures.
    pub fn tiny() -> ModelConfig {
        ModelConfig {
            hidden_size: 64,
            num_codebooks: 8,
            audio_vocab_size: 32,
            semantic_vocab_size: 64,
            vocab_size: 512,
            audio_code_offset: 200,
            audio_end_token_id: 199,
            audio_cfg_token_id: 198,
            num_hidden_layers: 2,
            intermediate_size: 128,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 16,
            max_position_embeddings: 256,
            rope_theta: 10_000.0,
            depth_num_layers: 2,
            depth_num_heads: 4,
            depth_intermediate_size: 128,
            condition_out_dim: 32,
            num_condition_layers: 8,
            dit_in_channels: 16,
            dit_num_layers: 2,
            dit_num_heads: 4,
            dit_head_dim: 16,
            dit_ff_inner_dim: 64,
            dit_rotary_dim: 8,
            dit_fourier_dim: 16,
            vocoder_input_dim: 16,
            vocoder_hidden_dim: 32,
            vocoder_upsampling_ratios: vec![4, 2],
            ..Default::default()
        }
    }

    fn from_json(value: &serde_json::Value) -> ModelConfig {
        let mut config = ModelConfig::default();
        macro_rules! field_usize {
            ($key:expr, $slot:expr) => {
                if let Some(v) = value.get($key).and_then(|v| v.as_u64()) {
                    $slot = v as usize;
                }
            };
        }
        field_usize!("hidden_size", config.hidden_size);
        field_usize!("num_codebooks", config.num_codebooks);
        field_usize!("audio_vocab_size", config.audio_vocab_size);
        field_usize!("semantic_vocab_size", config.semantic_vocab_size);
        field_usize!("vocab_size", config.vocab_size);
        field_usize!("audio_code_offset", config.audio_code_offset);
        field_usize!("audio_end_token_id", config.audio_end_token_id);
        field_usize!("audio_cfg_token_id", config.audio_cfg_token_id);
        field_usize!("num_hidden_layers", config.num_hidden_layers);
        field_usize!("intermediate_size", config.intermediate_size);
        field_usize!("num_attention_heads", config.num_attention_heads);
        field_usize!("num_key_value_heads", config.num_key_value_heads);
        field_usize!("head_dim", config.head_dim);
        field_usize!("max_position_embeddings", config.max_position_embeddings);
        field_usize!("depth_num_layers", config.depth_num_layers);
        field_usize!("depth_num_heads", config.depth_num_heads);
        field_usize!("depth_intermediate_size", config.depth_intermediate_size);
        field_usize!(
            "depth_max_position_embeddings",
            config.depth_max_position_embeddings
        );
        field_usize!("condition_out_dim", config.condition_out_dim);
        field_usize!("num_condition_layers", config.num_condition_layers);
        field_usize!("input_sampling_rate", config.input_sampling_rate);
        field_usize!("input_hop_length", config.input_hop_length);
        field_usize!("output_sampling_rate", config.output_sampling_rate);
        field_usize!("output_hop_length", config.output_hop_length);
        field_usize!("dit_in_channels", config.dit_in_channels);
        field_usize!("dit_num_layers", config.dit_num_layers);
        field_usize!("dit_num_heads", config.dit_num_heads);
        field_usize!("dit_head_dim", config.dit_head_dim);
        field_usize!("dit_ff_inner_dim", config.dit_ff_inner_dim);
        field_usize!("dit_rotary_dim", config.dit_rotary_dim);
        field_usize!("dit_fourier_dim", config.dit_fourier_dim);
        field_usize!("vocoder_input_dim", config.vocoder_input_dim);
        field_usize!("vocoder_hidden_dim", config.vocoder_hidden_dim);
        if let Some(v) = value.get("rms_norm_eps").and_then(|v| v.as_f64()) {
            config.rms_norm_eps = v as f32;
        }
        if let Some(v) = value.get("rope_theta").and_then(|v| v.as_f64()) {
            config.rope_theta = v as f32;
        }
        if let Some(v) = value.get("tie_word_embeddings").and_then(|v| v.as_bool()) {
            config.tie_word_embeddings = v;
        }
        if let Some(v) = value.get("sample_rate").and_then(|v| v.as_u64()) {
            config.sample_rate = v as u32;
        }
        if let Some(v) = value.get("frame_rate").and_then(|v| v.as_f64()) {
            config.frame_rate = v;
        }
        if let Some(v) = value
            .get("vocoder_upsampling_ratios")
            .and_then(|v| v.as_array())
        {
            let ratios: Vec<usize> = v
                .iter()
                .filter_map(|r| r.as_u64().map(|r| r as usize))
                .collect();
            if !ratios.is_empty() {
                config.vocoder_upsampling_ratios = ratios;
            }
        }
        config
    }

    fn validate(&self) -> Result<()> {
        let dimensions = [
            self.hidden_size,
            self.num_hidden_layers,
            self.intermediate_size,
            self.num_attention_heads,
            self.num_key_value_heads,
            self.head_dim,
            self.vocab_size,
            self.audio_vocab_size,
            self.semantic_vocab_size,
            self.depth_num_layers,
            self.depth_num_heads,
            self.depth_intermediate_size,
            self.depth_max_position_embeddings,
            self.condition_out_dim,
            self.num_condition_layers,
            self.input_sampling_rate,
            self.input_hop_length,
            self.output_sampling_rate,
            self.output_hop_length,
            self.dit_in_channels,
            self.dit_num_layers,
            self.dit_num_heads,
            self.dit_head_dim,
            self.dit_ff_inner_dim,
            self.dit_fourier_dim,
            self.vocoder_input_dim,
            self.vocoder_hidden_dim,
        ];
        if dimensions.contains(&0)
            || self.num_codebooks < 2
            || self.num_codebooks != self.num_condition_layers
            || self.num_attention_heads % self.num_key_value_heads != 0
            || self.hidden_size % self.depth_num_heads != 0
            || self.head_dim % 2 != 0
            || self.dit_rotary_dim % 2 != 0
            || self.dit_rotary_dim == 0
            || self.dit_rotary_dim > self.dit_head_dim
            || self.dit_fourier_dim % 2 != 0
            || self.audio_cfg_token_id >= self.vocab_size
            || self.audio_end_token_id >= self.vocab_size
            || self
                .audio_code_offset
                .checked_add(self.semantic_vocab_size)
                .is_none_or(|end| end > self.vocab_size)
            || !self.rms_norm_eps.is_finite()
            || self.rms_norm_eps <= 0.0
            || !self.rope_theta.is_finite()
            || self.rope_theta <= 0.0
            || self.vocoder_upsampling_ratios.is_empty()
            || self.vocoder_upsampling_ratios.contains(&0)
            || self
                .vocoder_upsampling_ratios
                .iter()
                .any(|&r| r.checked_mul(2).is_none())
            || self.vocoder_upsampling_ratios.len() >= usize::BITS as usize - 1
            || self.vocoder_hidden_dim / (1usize << self.vocoder_upsampling_ratios.len()) == 0
        {
            return Err(SpeechError::BadConfig {
                field: "geometry".into(),
                why: "inconsistent Music 3 dimensions, token range, or numeric parameters".into(),
            });
        }
        if self.sample_rate != SAMPLING_RATE {
            return Err(SpeechError::BadConfig {
                field: "sample_rate".to_string(),
                why: format!(
                    "MiniMax Music 3 outputs 44.1 kHz audio, config requested {} Hz",
                    self.sample_rate
                ),
            });
        }
        // Reject corrupt dimensions before component loaders multiply them or
        // allocate request buffers. The bound is Vec<f32>'s addressable size.
        let products: &[&[usize]] = &[
            &[self.vocab_size, self.hidden_size],
            &[self.num_codebooks, self.hidden_size, MAX_AUDIO_FRAMES],
            &[
                self.audio_vocab_size,
                self.num_codebooks - 1,
                self.hidden_size,
            ],
            &[self.depth_max_position_embeddings, self.hidden_size],
            &[self.hidden_size, self.hidden_size],
            &[self.hidden_size, self.intermediate_size],
            &[self.hidden_size, self.depth_intermediate_size],
            &[self.num_attention_heads, self.head_dim, self.hidden_size],
            &[self.dit_num_heads, self.dit_head_dim, self.dit_ff_inner_dim],
            &[self.vocoder_hidden_dim, self.vocoder_hidden_dim, 7],
        ];
        if products.iter().any(|dims| {
            dims.iter()
                .try_fold(1usize, |n, &d| n.checked_mul(d))
                .is_none_or(|n| n > isize::MAX as usize / 4)
        }) {
            return Err(SpeechError::BadConfig {
                field: "geometry".into(),
                why: "Music 3 dimension product exceeds addressable memory".into(),
            });
        }
        if !self.frame_rate.is_finite() || self.frame_rate <= 0.0 {
            return Err(SpeechError::BadConfig {
                field: "frame_rate".to_string(),
                why: "frame rate must be a finite positive number".to_string(),
            });
        }
        Ok(())
    }
}

/// One generation request. `text_ids` is the conditional row (BOS,
/// prompt, EOS as produced by the checkpoint's tokenizer or
/// [`prompt::encode_tiny_text_pair`]); the unconditional CFG row is
/// derived from it.
#[derive(Debug, Clone)]
pub struct GenerateRequest {
    pub text_ids: Vec<i32>,
    pub frames: usize,
    pub steps: usize,
    pub seed: u64,
}

/// Borrowed AR sampling inputs for independent stage diagnostics. Frame zero
/// is the warmup frame; semantic samples are vocabulary IDs, residuals local IDs.
pub struct SamplingTrace<'a> {
    pub frame: usize,
    pub codebook: usize,
    pub key: (u32, u32),
    pub hiddens: &'a [f32],
    pub guided_logits: &'a [f32],
    pub sampled: usize,
}

/// Caption-and-lyrics request matching mlx-audio's MiniMax Music 3 API.
/// Duration, steps, and seed default to 60 seconds, 30, and 0.
#[derive(Debug, Clone, PartialEq)]
pub struct TextGenerateRequest {
    pub caption: String,
    pub lyrics: String,
    pub duration_seconds: Option<f64>,
    pub steps: Option<usize>,
    pub seed: Option<u64>,
}

impl TextGenerateRequest {
    pub fn new(caption: impl Into<String>, lyrics: impl Into<String>) -> Self {
        Self {
            caption: caption.into(),
            lyrics: lyrics.into(),
            duration_seconds: None,
            steps: None,
            seed: None,
        }
    }

    /// Check request values without loading a model or tokenizing the prompt.
    pub fn validate(&self) -> Result<()> {
        self.resolve().map(|_| ())
    }

    fn resolve(&self) -> Result<(f64, usize, u64)> {
        if self.caption.trim().is_empty() {
            return Err(SpeechError::Input {
                why: "A music description is required".to_string(),
            });
        }
        if self.lyrics.trim().is_empty() {
            return Err(SpeechError::Input {
                why: "Lyrics are required; use [instrumental] explicitly".to_string(),
            });
        }
        let duration = self.duration_seconds.unwrap_or(60.0);
        if !duration.is_finite() || duration <= 0.0 || duration > 360.0 {
            return Err(SpeechError::Input {
                why: "duration must be greater than 0 and no more than 360 seconds".to_string(),
            });
        }
        let steps = self.steps.unwrap_or(30);
        if !(1..=30).contains(&steps) {
            return Err(SpeechError::Input {
                why: "steps must be between 1 and 30".to_string(),
            });
        }
        Ok((duration, steps, self.seed.unwrap_or(0)))
    }
}

#[derive(Debug, Clone)]
pub struct Generation {
    /// Interleaved stereo samples `[S, 2]` flattened.
    pub waveform: Vec<f32>,
    pub samples: usize,
    pub sample_rate: u32,
    pub frames: usize,
}

pub struct Model {
    config: ModelConfig,
    lm: Qwen3,
    depth: DepthDecoder,
    condition: ConditionEncoder,
    transformer: FlowMatchingTransformer,
    vocoder: Vocoder,
    tokenizer: Option<Tokenizer>,
    backend: Option<Rc<dyn ComputeBackend>>,
    precision: Music3Precision,
    // Reset at the start of each timed request; `Model` is already
    // single-threaded (`Rc` backend), so a `RefCell` adds no constraint.
    stage: RefCell<StageTimings>,
}

/// Chunk windows over `num_frames` AR frames: 200-frame windows hopping
/// by 100, single-window for short inputs.
pub(crate) fn chunk_starts(num_frames: usize) -> Vec<usize> {
    if num_frames <= CHUNK_FRAMES {
        return vec![0];
    }
    (0..num_frames - CHUNK_HOP).step_by(CHUNK_HOP).collect()
}

/// Assemble prompt text from caption and lyrics.
pub fn assemble_prompt(caption: &str, lyrics: &str) -> String {
    prompt::assemble_prompt(caption, lyrics)
}

/// The deterministic tiny text-pair encoder (no tokenizer needed).
pub fn encode_tiny_text_pair(text: &str, audio_cfg_token_id: i32) -> Vec<i32> {
    prompt::encode_tiny_text_pair(text, audio_cfg_token_id, 32)
}

/// The reference's quantization predicate: only the large generation
/// linears quantize; heads, embeddings, and convs stay dense.
pub fn model_quant_predicate(path: &str, is_linear: bool) -> bool {
    if !is_linear {
        return false;
    }
    if path.ends_with("lm_head") || path.contains(".audio_heads.") {
        return false;
    }
    path.starts_with("language_model.model.layers.")
        || path.starts_with("rvq_depth_decoder.layers.")
        || path.starts_with("transformer.proj_in")
        || path.starts_with("transformer.proj_out")
        || path.starts_with("transformer.transformer_blocks.")
}

impl Model {
    pub fn precision(&self) -> Music3Precision {
        self.precision
    }
    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    /// Load a converted MLX tree (`config.json` plus safetensors
    /// shards). Affine-quantized linears dequantize at load; supported
    /// floating-point and MXFP formats are parsed by the same strict loader.
    pub fn load_converted(dir: &Path) -> Result<Model> {
        Self::load_converted_with_precision(dir, Music3Precision::Float32)
    }

    pub fn load_converted_with_precision(dir: &Path, precision: Music3Precision) -> Result<Model> {
        let config_path = dir.join("config.json");
        if !config_path.is_file() {
            return Err(SpeechError::BadConfig {
                field: "config.json".to_string(),
                why: "missing from the converted tree".to_string(),
            });
        }
        let value = crate::quant::read_json(&config_path)?;
        let config = ModelConfig::from_json(&value);
        let scheme = parse_quant_scheme(&value)?;
        let files = open_converted_shards(dir)?;
        let mut store = WeightStore::from_files(files, scheme).with_precision(precision);
        let mut model = Model::from_store(&mut store, &config)?;
        store.finish()?;
        model.tokenizer = load_tokenizer(dir)?;
        Ok(model)
    }

    /// Load converted weights directly into a device backend without CPU
    /// expansion of packed matrices. The same pipeline owns both paths.
    pub fn load_converted_with_backend(
        dir: &Path,
        backend: Rc<dyn ComputeBackend>,
    ) -> Result<Model> {
        Self::load_converted_with_backend_and_precision(dir, backend, Music3Precision::Float32)
    }

    pub fn load_converted_with_backend_and_precision(
        dir: &Path,
        backend: Rc<dyn ComputeBackend>,
        precision: Music3Precision,
    ) -> Result<Model> {
        let value = crate::quant::read_json(&dir.join("config.json"))?;
        let config = ModelConfig::from_json(&value);
        let scheme = parse_quant_scheme(&value)?;
        let files = open_converted_shards(dir)?;
        let mut store =
            WeightStore::from_files_with_backend(files, scheme, backend).with_precision(precision);
        let mut model = Model::from_store(&mut store, &config)?;
        store.finish()?;
        model.tokenizer = load_tokenizer(dir)?;
        Ok(model)
    }

    /// Load the official modular Diffusers checkpoint (five component
    /// directories): flatten the component configs, fuse vocoder
    /// weight-norm pairs, and remap the PyTorch conv layouts.
    pub fn load_official_tree(dir: &Path) -> Result<Model> {
        Self::load_official_tree_with_precision(dir, Music3Precision::Float32)
    }
    pub fn load_official_tree_with_precision(
        dir: &Path,
        precision: Music3Precision,
    ) -> Result<Model> {
        let config = prepare_official_config(dir)?;
        let tensors = load_official_tensors(dir)?;
        let mut store = WeightStore::from_map(tensors).with_precision(precision);
        let mut model = Model::from_store(&mut store, &config)?;
        store.finish()?;
        model.tokenizer = load_tokenizer(dir)?;
        Ok(model)
    }

    fn from_store(store: &mut WeightStore, config: &ModelConfig) -> Result<Model> {
        config.validate()?;
        let lm = Qwen3::load(
            store,
            "language_model",
            Qwen3Dims {
                hidden: config.hidden_size,
                layers: config.num_hidden_layers,
                intermediate: config.intermediate_size,
                heads: config.num_attention_heads,
                kv_heads: config.num_key_value_heads,
                head_dim: config.head_dim,
                vocab: config.vocab_size,
                eps: config.rms_norm_eps,
                rope_theta: config.rope_theta,
                tie_embeddings: config.tie_word_embeddings,
            },
        )?;
        let depth = DepthDecoder::load(
            store,
            "rvq_depth_decoder",
            DepthDims {
                hidden: config.hidden_size,
                layers: config.depth_num_layers,
                heads: config.depth_num_heads,
                intermediate: config.depth_intermediate_size,
                eps: config.rms_norm_eps,
                audio_vocab: config.audio_vocab_size,
                num_codebooks: config.num_codebooks,
                max_positions: config.depth_max_position_embeddings,
            },
        )?;
        let condition = ConditionEncoder::load(
            store,
            "condition_encoder",
            ConditionDims {
                hidden: config.hidden_size,
                num_layers: config.num_condition_layers,
                input_sampling_rate: config.input_sampling_rate,
                input_hop_length: config.input_hop_length,
                output_sampling_rate: config.output_sampling_rate,
                output_hop_length: config.output_hop_length,
            },
        )?;
        let transformer = FlowMatchingTransformer::load(
            store,
            "transformer",
            DitDims {
                in_channels: config.dit_in_channels,
                num_layers: config.dit_num_layers,
                heads: config.dit_num_heads,
                head_dim: config.dit_head_dim,
                ff_inner: config.dit_ff_inner_dim,
                rotary_dim: config.dit_rotary_dim,
                fourier: config.dit_fourier_dim,
                condition_dim: config.condition_out_dim,
            },
        )?;
        let vocoder = Vocoder::load(
            store,
            "vocoder",
            VocoderDims {
                latent_channels: config.dit_in_channels,
                input_dim: config.vocoder_input_dim,
                hidden_dim: config.vocoder_hidden_dim,
                upsampling_ratios: config.vocoder_upsampling_ratios.clone(),
            },
        )?;
        Ok(Model {
            config: config.clone(),
            lm,
            depth,
            condition,
            transformer,
            vocoder,
            tokenizer: None,
            backend: store.backend(),
            precision: store.precision(),
            stage: RefCell::new(StageTimings::default()),
        })
    }

    /// Generate from the public caption-and-lyrics contract using the
    /// tokenizer shipped with the converted checkpoint.
    pub fn generate_text(&self, request: &TextGenerateRequest) -> Result<Generation> {
        self.generate_text_timed(request)
            .map(|(generation, _)| generation)
    }

    /// [`Model::generate_text`] plus a per-stage wall-time breakdown. The
    /// generated audio is identical; only the timings are extra.
    pub fn generate_text_timed(
        &self,
        request: &TextGenerateRequest,
    ) -> Result<(Generation, StageTimings)> {
        let started = Instant::now();
        *self.stage.borrow_mut() = StageTimings::default();
        let (duration, steps, seed) = request.resolve()?;
        let frames = ((duration * self.config.frame_rate).trunc() as usize).max(1);
        let prompt_text = assemble_prompt(&request.caption, &request.lyrics);
        let text_ids = match &self.tokenizer {
            Some(tokenizer) => prompt::encode_official_text(tokenizer, &prompt_text)?,
            None if self.config.vocab_size <= 512 && self.config.hidden_size <= 64 => {
                prompt::encode_tiny_ids(&prompt_text, 32)
            }
            None => {
                return Err(SpeechError::Unsupported {
                    why: "the checkpoint is missing tokenizer/tokenizer.json; MiniMax Music 3 text generation requires its official tokenizer".to_string(),
                });
            }
        };
        if text_ids.len() > MAX_PROMPT_TOKENS {
            return Err(SpeechError::Input {
                why: format!(
                    "The assembled prompt has {} tokens; the maximum is {MAX_PROMPT_TOKENS}",
                    text_ids.len()
                ),
            });
        }
        self.stage.borrow_mut().tokenize = started.elapsed();
        let generation = self.generate(&GenerateRequest {
            text_ids,
            frames,
            steps,
            seed,
        })?;
        let mut timings = self.stage.borrow().clone();
        timings.total = started.elapsed();
        Ok((generation, timings))
    }

    /// Full generation: AR frame hiddens, flow decoding, and stereo
    /// output clipped to [-1, 1].
    pub fn generate(&self, request: &GenerateRequest) -> Result<Generation> {
        if request.frames == 0 || request.frames > MAX_AUDIO_FRAMES {
            return Err(SpeechError::Input {
                why: format!("frames must be in 1..={MAX_AUDIO_FRAMES}"),
            });
        }
        if request.steps == 0 || request.steps > 30 {
            return Err(SpeechError::Input {
                why: "steps must be in 1..=30".to_string(),
            });
        }
        if request.text_ids.len() > MAX_PROMPT_TOKENS {
            return Err(SpeechError::Input {
                why: format!(
                    "prompt has {} tokens; the maximum is {MAX_PROMPT_TOKENS}",
                    request.text_ids.len()
                ),
            });
        }
        let (hiddens, _codes) =
            self.generate_frame_hiddens(&request.text_ids, request.frames, request.seed)?;
        // The end token can stop the AR stage before `frames`; the
        // flow stage consumes the emitted count, matching the
        // reference, which derives chunking from the hiddens' own
        // length.
        let fused = self.config.num_codebooks * self.config.hidden_size;
        let emitted = hiddens.len() / fused;
        let stereo = self.run_flow(&hiddens, emitted, request.steps, request.seed)?;
        let interleave_started = Instant::now();
        let samples = stereo.len() / 2;
        // Interleave planar stereo into the reference's [S, 2] frame
        // order and clip to the unit range.
        let mut waveform = Vec::with_capacity(stereo.len());
        for t in 0..samples {
            for c in 0..2 {
                let clipped = stereo[c * samples + t].clamp(-1.0, 1.0);
                if !clipped.is_finite() {
                    return Err(SpeechError::Input {
                        why: "MiniMax Music 3 produced non-finite audio".to_string(),
                    });
                }
                waveform.push(clipped);
            }
        }
        self.stage.borrow_mut().stitch += interleave_started.elapsed();
        Ok(Generation {
            waveform,
            samples,
            sample_rate: self.config.sample_rate,
            frames: emitted,
        })
    }

    /// The AR stage: prefill the text pair, then emit up to
    /// `max_frames` fused frame hiddens `[frames, codebooks * hidden]`
    /// plus the sampled `[semantic, residuals...]` codebook per frame.
    pub fn generate_frame_hiddens(
        &self,
        text_ids: &[i32],
        max_frames: usize,
        seed: u64,
    ) -> Result<(Vec<f32>, Vec<Vec<i32>>)> {
        self.generate_frame_hiddens_traced(text_ids, max_frames, seed, |_| {})
    }

    /// Generate while observing the exact inputs of each sampling decision.
    /// The ordinary generation path does not retain diagnostic tensors.
    pub fn generate_frame_hiddens_traced(
        &self,
        text_ids: &[i32],
        max_frames: usize,
        seed: u64,
        mut observe: impl FnMut(SamplingTrace<'_>),
    ) -> Result<(Vec<f32>, Vec<Vec<i32>>)> {
        let config = &self.config;
        let hidden = config.hidden_size;
        let vocab = config.vocab_size;
        let prompt_len = text_ids.len();
        if prompt_len == 0 {
            return Err(SpeechError::Input {
                why: "empty prompt".to_string(),
            });
        }
        if max_frames == 0 || max_frames > MAX_AUDIO_FRAMES {
            return Err(SpeechError::Input {
                why: format!("frames must be in 1..={MAX_AUDIO_FRAMES}"),
            });
        }
        if let Some(id) = text_ids
            .iter()
            .find(|id| **id < 0 || **id as usize >= vocab)
        {
            return Err(SpeechError::Input {
                why: format!("prompt token id {id} is outside the model vocabulary {vocab}"),
            });
        }
        if prompt_len > MAX_PROMPT_TOKENS {
            return Err(SpeechError::Input {
                why: "prompt exceeds Music 3 token limit".into(),
            });
        }
        // Build the conditional / unconditional pair.
        let mut pair = Vec::with_capacity(2 * prompt_len);
        pair.extend_from_slice(text_ids);
        if prompt_len > 3 {
            pair.push(text_ids[0]);
            pair.extend(std::iter::repeat_n(
                config.audio_cfg_token_id as i32,
                prompt_len - 3,
            ));
            pair.extend_from_slice(&text_ids[prompt_len - 2..]);
        } else {
            pair.extend_from_slice(text_ids);
        }
        let prefill_started = Instant::now();
        let embeddings = self.lm.embed_ids(&pair, 2, prompt_len)?;
        let mut cache: Vec<KvCache> = (0..config.num_hidden_layers)
            .map(|_| KvCache::new())
            .collect();
        let mut hidden_state = self
            .lm
            .hidden_forward(&embeddings, 2, prompt_len, &mut cache)?;
        let mut last_hidden: Vec<f32> = Vec::with_capacity(2 * hidden);
        for b in 0..2 {
            let base = (b * prompt_len + prompt_len - 1) * hidden;
            last_hidden.extend_from_slice(&hidden_state[base..base + hidden]);
        }
        self.stage.borrow_mut().prefill += prefill_started.elapsed();

        let mut key = Key::new(seed);
        let mut frames: Vec<f32> = Vec::with_capacity(max_frames * config.num_codebooks * hidden);
        let mut frame_codes: Vec<Vec<i32>> = Vec::with_capacity(max_frames);
        let mut emitted = 0usize;
        for frame_index in 0..=max_frames {
            if emitted >= max_frames {
                break;
            }
            let (next_key, subkey) = rng::split(key);
            key = next_key;
            let lap = Instant::now();
            let logits = self.lm.logits(&last_hidden)?;
            self.stage.borrow_mut().lm_head += lap.elapsed();
            let lap = Instant::now();
            // Semantic mask: audio codes plus the end token.
            let mut guided = vec![0.0f32; vocab];
            let mut conditional_row = vec![0.0f32; vocab];
            for v in 0..vocab {
                let allowed = (v >= config.audio_code_offset
                    && v < config.audio_code_offset + config.semantic_vocab_size)
                    || v == config.audio_end_token_id;
                let conditional = if allowed { logits[v] } else { -1e9 };
                let unconditional = if allowed { logits[vocab + v] } else { -1e9 };
                conditional_row[v] = conditional;
                guided[v] = unconditional + AR_CFG_SCALE * (conditional - unconditional);
            }
            let threshold = rng::kth_largest(&conditional_row, AR_CFG_TOP_K.min(vocab));
            for v in 0..vocab {
                let allowed = (v >= config.audio_code_offset
                    && v < config.audio_code_offset + config.semantic_vocab_size)
                    || v == config.audio_end_token_id;
                if conditional_row[v] < threshold || !allowed {
                    guided[v] = -1e9;
                }
            }
            let (sampled, mut rng_key) = rng::sample_top_k(&guided, subkey, AR_SAMPLING_TOP_K);
            self.stage.borrow_mut().sampling += lap.elapsed();
            observe(SamplingTrace {
                frame: frame_index,
                codebook: 0,
                key: (subkey.0, subkey.1),
                hiddens: &last_hidden,
                guided_logits: &guided,
                sampled,
            });
            if sampled == config.audio_end_token_id {
                break;
            }
            let semantic_code = sampled - config.audio_code_offset;

            // Depth expansion over both CFG rows.
            let lap = Instant::now();
            let mut sequence: Vec<f32> = Vec::new();
            for b in 0..2 {
                sequence.extend(ops_linear_no_bias(
                    &last_hidden[b * hidden..(b + 1) * hidden],
                    &self.depth.projection,
                    hidden,
                    hidden,
                    self.lm.output_dtype(),
                )?);
            }
            // Reorder the two rows into [2, 1, hidden].
            let semantic_embed = self.lm.embed_ids(
                &[
                    (semantic_code + config.audio_code_offset) as i32,
                    (semantic_code + config.audio_code_offset) as i32,
                ],
                2,
                1,
            )?;
            let mut semantic_projected = Vec::with_capacity(2 * hidden);
            for b in 0..2 {
                semantic_projected.extend(ops_linear_no_bias(
                    &semantic_embed[b * hidden..(b + 1) * hidden],
                    &self.depth.projection,
                    hidden,
                    hidden,
                    self.lm.dtype(),
                )?);
            }
            let residual = config.num_codebooks - 1;
            let mut codes = vec![semantic_code];
            let mut hidden_parts: Vec<f32> = Vec::with_capacity(residual * hidden);
            // Projected residual embeddings appended to the depth
            // sequence; identical for both CFG rows.
            let mut extra_embeddings: Vec<f32> = Vec::new();
            let mut depth_input_dtype = self
                .depth
                .projection
                .output_dtype(self.lm.output_dtype())
                .promote(self.depth.projection.output_dtype(self.lm.dtype()));
            for index in 1..config.num_codebooks {
                let seq_len = 2 + (index - 1);
                let mut depth_in = vec![0.0f32; 2 * seq_len * hidden];
                for b in 0..2 {
                    depth_in[b * seq_len * hidden..b * seq_len * hidden + hidden]
                        .copy_from_slice(&sequence[b * hidden..(b + 1) * hidden]);
                    depth_in[b * seq_len * hidden + hidden..b * seq_len * hidden + 2 * hidden]
                        .copy_from_slice(&semantic_projected[b * hidden..(b + 1) * hidden]);
                    for extra in 0..(index - 1) {
                        let src = extra * hidden;
                        let dst = b * seq_len * hidden + (2 + extra) * hidden;
                        depth_in[dst..dst + hidden]
                            .copy_from_slice(&extra_embeddings[src..src + hidden]);
                    }
                }
                let depth_out =
                    self.depth
                        .forward_typed(&depth_in, 2, seq_len, depth_input_dtype)?;
                // Last-position rows for both CFG halves.
                let mut last_rows = Vec::with_capacity(2 * hidden);
                for b in 0..2 {
                    let base = (b * seq_len + seq_len - 1) * hidden;
                    last_rows.extend_from_slice(&depth_out[base..base + hidden]);
                }
                let head = &self.depth.audio_heads[index - 1];
                let mut head_logits = vec![0.0f32; 2 * config.audio_vocab_size];
                for b in 0..2 {
                    let out = ops_linear_no_bias(
                        &last_rows[b * hidden..(b + 1) * hidden],
                        head,
                        hidden,
                        config.audio_vocab_size,
                        self.depth.output_dtype(depth_input_dtype),
                    )?;
                    head_logits[b * config.audio_vocab_size..(b + 1) * config.audio_vocab_size]
                        .copy_from_slice(&out);
                }
                let mut guided_d = vec![0.0f32; config.audio_vocab_size];
                for v in 0..config.audio_vocab_size {
                    let conditional = head_logits[v];
                    let unconditional = head_logits[config.audio_vocab_size + v];
                    guided_d[v] = unconditional + AR_CFG_SCALE * (conditional - unconditional);
                }
                let (code, next_rng) = rng::sample_top_k(&guided_d, rng_key, AR_SAMPLING_TOP_K);
                observe(SamplingTrace {
                    frame: frame_index,
                    codebook: index,
                    key: (rng_key.0, rng_key.1),
                    hiddens: &last_rows,
                    guided_logits: &guided_d,
                    sampled: code,
                });
                rng_key = next_rng;
                codes.push(code);
                hidden_parts.extend_from_slice(&last_rows[..hidden]);
                if index < config.num_codebooks - 1 {
                    depth_input_dtype = depth_input_dtype
                        .promote(self.depth.projection.output_dtype(self.depth.dtype()));
                    let embed = self.depth.embed_code(code, index - 1);
                    extra_embeddings.extend(ops_linear_no_bias(
                        embed,
                        &self.depth.projection,
                        hidden,
                        hidden,
                        self.depth.dtype(),
                    )?);
                }
            }

            self.stage.borrow_mut().depth += lap.elapsed();
            if frame_index > 0 {
                let mut frame = Vec::with_capacity(config.num_codebooks * hidden);
                frame.extend_from_slice(&last_hidden[..hidden]);
                frame.extend_from_slice(&hidden_parts);
                frames.extend_from_slice(&frame);
                frame_codes.push(codes.iter().map(|c| *c as i32).collect());
                emitted += 1;
            }

            // Feedback embedding for the LM step (both rows identical).
            let lap = Instant::now();
            let mut feedback_row =
                self.lm
                    .embed_ids(&[(semantic_code + config.audio_code_offset) as i32], 1, 1)?;
            let residual_dtype = self.depth.dtype();
            let residual_values: Vec<f32> = codes[1..]
                .iter()
                .enumerate()
                .flat_map(|(c, code)| self.depth.embed_code(*code, c).to_vec())
                .collect();
            let mut residual_sum =
                precision::sum_rows(&residual_values, codes.len() - 1, hidden, residual_dtype);
            let feedback_dtype = self.lm.dtype();
            feedback_dtype.round_slice(&mut residual_sum);
            let scale = feedback_dtype.round((config.num_codebooks as f32).powf(-0.5));
            for d in 0..hidden {
                feedback_row[d] = feedback_dtype
                    .round(feedback_dtype.round(feedback_row[d] + residual_sum[d]) * scale);
            }
            let feedback = [feedback_row.clone(), feedback_row];
            hidden_state = self
                .lm
                .hidden_forward(&feedback.concat(), 2, 1, &mut cache)?;
            last_hidden.clear();
            for b in 0..2 {
                last_hidden.extend_from_slice(&hidden_state[b * hidden..(b + 1) * hidden]);
            }
            self.stage.borrow_mut().lm_decode += lap.elapsed();
        }
        self.stage.borrow_mut().frames = emitted;
        if frames.is_empty() {
            return Err(SpeechError::Input {
                why: "MiniMax Music 3 generated zero audio frames".to_string(),
            });
        }
        Ok((frames, frame_codes))
    }

    // Concatenation promotes across every residual codebook, including
    // embeddings appended after the first depth-decoder call.
    fn frame_hidden_dtype(&self) -> precision::DType {
        let mut fused = self.lm.output_dtype();
        let mut depth_input = self
            .depth
            .projection
            .output_dtype(fused)
            .promote(self.depth.projection.output_dtype(self.lm.dtype()));
        for index in 1..self.config.num_codebooks {
            fused = fused.promote(self.depth.output_dtype(depth_input));
            if index < self.config.num_codebooks - 1 {
                depth_input =
                    depth_input.promote(self.depth.projection.output_dtype(self.depth.dtype()));
            }
        }
        fused
    }

    /// Condition, denoise overlapping chunks, decode, and crop-stitch.
    /// Returns the planar stereo waveform `[2, S]` flattened.
    pub fn run_flow(
        &self,
        frame_hiddens: &[f32],
        frames: usize,
        steps: usize,
        seed: u64,
    ) -> Result<Vec<f32>> {
        self.run_flow_internal(frame_hiddens, frames, steps, seed, None)
    }

    /// Replay flow with supplied channel-major initial noise per chunk.
    /// Each chunk is cast to the conditioning tensor's logical dtype.
    pub fn run_flow_with_noise(
        &self,
        frame_hiddens: &[f32],
        frames: usize,
        steps: usize,
        noise_chunks: &[Vec<f32>],
    ) -> Result<Vec<f32>> {
        self.run_flow_internal(frame_hiddens, frames, steps, 0, Some(noise_chunks))
    }

    fn run_flow_internal(
        &self,
        frame_hiddens: &[f32],
        frames: usize,
        steps: usize,
        seed: u64,
        supplied_noise: Option<&[Vec<f32>]>,
    ) -> Result<Vec<f32>> {
        if frames == 0 || frames > MAX_AUDIO_FRAMES || !(1..=30).contains(&steps) {
            return Err(SpeechError::Input {
                why: "flow requires 1..=9000 frames and 1..=30 steps".into(),
            });
        }
        let config = &self.config;
        let fused = config.num_codebooks * config.hidden_size;
        if frame_hiddens.len() != frames * fused {
            return Err(SpeechError::Input {
                why: format!(
                    "frame hiddens {} do not match {frames}x{fused}",
                    frame_hiddens.len()
                ),
            });
        }
        let starts = chunk_starts(frames);
        if let Some(noise) = supplied_noise {
            if noise.len() != starts.len() {
                return Err(SpeechError::Input {
                    why: format!(
                        "supplied noise has {} chunks, expected {}",
                        noise.len(),
                        starts.len()
                    ),
                });
            }
            if noise.iter().flatten().any(|v| !v.is_finite()) {
                return Err(SpeechError::Input {
                    why: "supplied flow noise must contain finite values".into(),
                });
            }
        }
        let mut waves: Vec<Vec<f32>> = Vec::with_capacity(starts.len());
        let mut previous_latent: Option<Vec<f32>> = None;
        let mut previous_condition: Option<Vec<f32>> = None;
        // Keys are 64 bits, including the flow stream's fixed seed offset.
        let mut noise_sequence = rng::KeySequence::new(seed.wrapping_add(7));
        for (chunk_index, start) in starts.into_iter().enumerate() {
            let end = (start + CHUNK_FRAMES).min(frames);
            let chunk_frames = end - start;
            let lap = Instant::now();
            let condition = self.condition.forward_typed(
                &frame_hiddens[start * fused..end * fused],
                chunk_frames,
                self.frame_hidden_dtype(),
            )?;
            let cond_dim = config.condition_out_dim;
            let target = condition.len() / cond_dim;
            let hidden_dtype = self.frame_hidden_dtype();
            let condition_dtype = self.condition.dtype(hidden_dtype);
            backend::trace(
                &self.backend,
                "flow.condition",
                &condition,
                condition_dtype,
                &[1, target, cond_dim],
            );
            let noise = if let Some(supplied) = supplied_noise {
                let mut noise = supplied[chunk_index].clone();
                let expected = config.dit_in_channels * target;
                if noise.len() != expected {
                    return Err(SpeechError::Input {
                        why: format!(
                            "supplied noise chunk {chunk_index} has {} values, expected {expected}",
                            noise.len()
                        ),
                    });
                }
                condition_dtype.round_slice(&mut noise);
                if noise.iter().any(|v| !v.is_finite()) {
                    return Err(SpeechError::Input {
                        why: format!("supplied noise chunk {chunk_index} is non-finite after dtype conversion"),
                    });
                }
                noise
            } else {
                let noise_key = noise_sequence.next();
                if self.precision == Music3Precision::Float32 {
                    rng::normal(noise_key, config.dit_in_channels * target)
                } else {
                    let uniforms: Vec<f32> =
                        rng::uniform01(noise_key, config.dit_in_channels * target)
                            .into_iter()
                            .map(|u| {
                                let above_minus_one = -(1.0f32 - f32::EPSILON / 2.0);
                                (1.0 - above_minus_one) * u + above_minus_one
                            })
                            .collect();
                    match &self.backend {
                        Some(b) => b.normal_from_uniform(&uniforms, condition_dtype)?,
                        None => rng::normal_from_uniform(&uniforms, condition_dtype),
                    }
                }
            };
            backend::trace(
                &self.backend,
                "flow.noise",
                &noise,
                condition_dtype,
                &[config.dit_in_channels, target],
            );
            self.stage.borrow_mut().condition += lap.elapsed();
            let lap = Instant::now();
            let (latents, cond_out) = euler::denoise_chunk_typed(
                &self.transformer,
                &noise,
                &condition_row_major_to_channel(&condition, cond_dim, target),
                config.dit_in_channels,
                cond_dim,
                target,
                steps,
                DIT_CFG_SCALE,
                previous_latent.as_deref(),
                previous_condition.as_deref(),
                condition_dtype,
            )?;
            self.stage.borrow_mut().dit += lap.elapsed();
            let carry_start = target.saturating_sub(2 * OVERLAP_LATENT_LENGTH);
            let carry_end = carry_start.max(target.saturating_sub(OVERLAP_LATENT_LENGTH));
            previous_latent = Some(slice_channel_rows(
                &latents,
                config.dit_in_channels,
                target,
                carry_start,
                carry_end,
            ));
            previous_condition = Some(slice_channel_rows(
                &cond_out,
                cond_dim,
                target,
                carry_start,
                carry_end,
            ));
            let mut latent_dtype = condition_dtype;
            for _ in 0..steps {
                latent_dtype = latent_dtype
                    .promote(self.transformer.output_dtype(latent_dtype, condition_dtype));
            }
            let lap = Instant::now();
            waves.push(self.vocoder.forward_typed(&latents, target, latent_dtype)?);
            let mut stage = self.stage.borrow_mut();
            stage.vocoder += lap.elapsed();
            stage.chunks += 1;
        }
        let stitch_started = Instant::now();
        // Crop-stitch along the sample axis per channel: the reference
        // concatenates the cropped `[1, 2, S_i]` waves on the last
        // axis, so left and right accumulate separately.
        let mut out_left: Vec<f32> = Vec::new();
        let mut out_right: Vec<f32> = Vec::new();
        let num_waves = waves.len();
        for (index, wave) in waves.iter().enumerate() {
            let samples = wave.len() / 2;
            let left = if index == 0 {
                0
            } else {
                CROP_LEFT_LATENT * LATENT_HOP_LENGTH
            };
            let right = if index == num_waves - 1 {
                0
            } else {
                CROP_RIGHT_LATENT * LATENT_HOP_LENGTH
            };
            // A short final chunk can be smaller than the left crop;
            // the reference keeps it whole in that case.
            let end = samples.saturating_sub(right);
            let (lo, hi) = if end <= left {
                (0, samples)
            } else {
                (left, end)
            };
            out_left.extend_from_slice(&wave[lo..hi]);
            out_right.extend_from_slice(&wave[samples + lo..samples + hi]);
        }
        out_left.extend_from_slice(&out_right);
        self.stage.borrow_mut().stitch += stitch_started.elapsed();
        Ok(out_left)
    }
}

fn load_tokenizer(dir: &Path) -> Result<Option<Tokenizer>> {
    let nested = dir.join("tokenizer/tokenizer.json");
    let flat = dir.join("tokenizer.json");
    let path = if nested.is_file() {
        nested
    } else if flat.is_file() {
        flat
    } else {
        return Ok(None);
    };
    Tokenizer::from_file(&path)
        .map(Some)
        .map_err(|error| SpeechError::BadConfig {
            field: path.display().to_string(),
            why: format!("loading MiniMax Music 3 tokenizer: {error}"),
        })
}

/// `x [1, rows] @ w.T` with no bias, `w` stored `[out, in]`.
fn ops_linear_no_bias(
    x: &[f32],
    w: &Weight,
    inn: usize,
    out: usize,
    dtype: precision::DType,
) -> Result<Vec<f32>> {
    backend::linear(x, w, None, 1, inn, out, dtype)
}

/// `[1, target, dim]` row-major -> `[dim, target]` channel-major.
fn condition_row_major_to_channel(x: &[f32], dim: usize, target: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; dim * target];
    for t in 0..target {
        for c in 0..dim {
            out[c * target + t] = x[t * dim + c];
        }
    }
    out
}

/// Slice `[dim, len]` channel-major rows to `[dim, end - start]`.
fn slice_channel_rows(x: &[f32], dim: usize, len: usize, start: usize, end: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(dim * (end - start));
    for c in 0..dim {
        out.extend_from_slice(&x[c * len + start..c * len + end]);
    }
    out
}

fn parse_quant_scheme(value: &serde_json::Value) -> Result<LinearQuantization> {
    let block = match value.get("quantization") {
        Some(block) => block,
        None => return Ok(LinearQuantization::Dense),
    };
    let mode = block
        .get("mode")
        .and_then(|m| m.as_str())
        .unwrap_or("affine");
    let bits = block.get("bits").and_then(|b| b.as_u64()).unwrap_or(0);
    let group_size = block
        .get("group_size")
        .and_then(|g| g.as_u64())
        .unwrap_or(0);
    match mode {
        "affine" => {
            if bits == 0
                || group_size == 0
                || bits > u32::MAX as u64
                || group_size > usize::MAX as u64
            {
                return Err(SpeechError::BadConfig {
                    field: "quantization".to_string(),
                    why: "affine quantization requires representable bits and group_size"
                        .to_string(),
                });
            }
            let scheme = QuantScheme {
                bits: bits as u32,
                group_size: group_size as usize,
            };
            if !matches!(scheme.bits, 2 | 3 | 4 | 5 | 6 | 8) {
                return Err(SpeechError::Unsupported {
                    why: format!(
                        "affine quantization with {} bits is unsupported",
                        scheme.bits
                    ),
                });
            }
            Ok(LinearQuantization::Affine(scheme))
        }
        "mxfp4" if bits == 4 && group_size == 32 => Ok(LinearQuantization::MxFp4),
        "mxfp8" if bits == 8 && group_size == 32 => Ok(LinearQuantization::MxFp8),
        "nvfp4" if bits == 4 && group_size == 16 => Ok(LinearQuantization::NvFp4),
        "mxfp4" | "mxfp8" | "nvfp4" => Err(SpeechError::BadConfig {
            field: "quantization".to_string(),
            why: format!("{mode} requires its upstream bits and group_size geometry"),
        }),
        _ => Err(SpeechError::Unsupported {
            why: format!("quantization mode {mode} is not supported for MiniMax Music 3"),
        }),
    }
}

fn read_component_json(dir: &Path, component: &str) -> Result<serde_json::Value> {
    let path = dir.join(component).join("config.json");
    if !path.is_file() {
        return Err(SpeechError::BadConfig {
            field: format!("{component}/config.json"),
            why: "MiniMax Music 3 component config missing".to_string(),
        });
    }
    crate::quant::read_json(&path)
}

fn config_usize(value: &serde_json::Value, key: &str) -> Option<usize> {
    value.get(key).and_then(|v| v.as_u64()).map(|v| v as usize)
}

fn config_usize_or(value: &serde_json::Value, key: &str, default: usize) -> usize {
    config_usize(value, key).unwrap_or(default)
}

fn config_f32_or(value: &serde_json::Value, key: &str, default: f32) -> f32 {
    value
        .get(key)
        .and_then(|v| v.as_f64())
        .map(|v| v as f32)
        .unwrap_or(default)
}

/// Flatten the five official component configs into a ModelConfig,
/// with the reference's cross-component consistency checks.
fn prepare_official_config(dir: &Path) -> Result<ModelConfig> {
    let lm = read_component_json(dir, "language_model")?;
    let depth = read_component_json(dir, "rvq_depth_decoder")?;
    let condition = read_component_json(dir, "condition_encoder")?;
    let transformer = read_component_json(dir, "transformer")?;
    let vocoder = read_component_json(dir, "vocoder")?;
    let required = |value: &serde_json::Value, key: &str, component: &str| -> Result<usize> {
        config_usize(value, key).ok_or_else(|| SpeechError::BadConfig {
            field: format!("{component}/{key}"),
            why: "missing from the official component config".to_string(),
        })
    };
    let rope_theta = lm
        .get("rope_parameters")
        .and_then(|r| r.get("rope_theta"))
        .and_then(|t| t.as_f64())
        .map(|t| t as f32)
        .unwrap_or(1_000_000.0);
    let mut config = ModelConfig {
        model_type: "minimax_music3".to_string(),
        hidden_size: required(&lm, "hidden_size", "language_model")?,
        vocab_size: required(&lm, "vocab_size", "language_model")?,
        num_hidden_layers: required(&lm, "num_hidden_layers", "language_model")?,
        intermediate_size: required(&lm, "intermediate_size", "language_model")?,
        num_attention_heads: required(&lm, "num_attention_heads", "language_model")?,
        num_key_value_heads: required(&lm, "num_key_value_heads", "language_model")?,
        head_dim: required(&lm, "head_dim", "language_model")?,
        max_position_embeddings: required(&lm, "max_position_embeddings", "language_model")?,
        rms_norm_eps: config_f32_or(&lm, "rms_norm_eps", 1e-6),
        rope_theta,
        tie_word_embeddings: lm
            .get("tie_word_embeddings")
            .and_then(|t| t.as_bool())
            .unwrap_or(false),
        audio_vocab_size: required(&depth, "audio_vocab_size", "rvq_depth_decoder")?,
        num_codebooks: required(&depth, "num_codebooks", "rvq_depth_decoder")?,
        depth_num_layers: required(&depth, "num_layers", "rvq_depth_decoder")?,
        depth_num_heads: required(&depth, "num_attention_heads", "rvq_depth_decoder")?,
        depth_intermediate_size: required(&depth, "intermediate_size", "rvq_depth_decoder")?,
        depth_max_position_embeddings: config_usize_or(&depth, "max_position_embeddings", 16),
        condition_out_dim: required(&condition, "out_dim", "condition_encoder")?,
        num_condition_layers: required(&condition, "num_condition_layers", "condition_encoder")?,
        input_sampling_rate: config_usize_or(&condition, "input_sampling_rate", 24_000),
        input_hop_length: config_usize_or(&condition, "input_hop_length", 960),
        output_sampling_rate: config_usize_or(&condition, "output_sampling_rate", 44_100),
        output_hop_length: config_usize_or(&condition, "output_hop_length", 512),
        dit_in_channels: required(&transformer, "in_channels", "transformer")?,
        dit_num_layers: required(&transformer, "num_layers", "transformer")?,
        dit_num_heads: required(&transformer, "num_attention_heads", "transformer")?,
        dit_head_dim: required(&transformer, "attention_head_dim", "transformer")?,
        dit_ff_inner_dim: required(&transformer, "ff_inner_dim", "transformer")?,
        dit_rotary_dim: required(&transformer, "rotary_dim", "transformer")?,
        dit_fourier_dim: required(&transformer, "fourier_embedding_dim", "transformer")?,
        vocoder_input_dim: required(&vocoder, "decoder_input_dim", "vocoder")?,
        vocoder_hidden_dim: required(&vocoder, "decoder_hidden_dim", "vocoder")?,
        vocoder_upsampling_ratios: vocoder
            .get("upsampling_ratios")
            .and_then(|r| r.as_array())
            .map(|rows| {
                rows.iter()
                    .filter_map(|v| v.as_u64().map(|v| v as usize))
                    .collect::<Vec<usize>>()
            })
            .unwrap_or_default(),
        sample_rate: required(&vocoder, "sampling_rate", "vocoder")? as u32,
        frame_rate: FRAME_RATE,
        ..Default::default()
    };
    if config.vocoder_upsampling_ratios.is_empty() {
        return Err(SpeechError::BadConfig {
            field: "vocoder/upsampling_ratios".to_string(),
            why: "empty upsampling ratios".to_string(),
        });
    }
    // The reference's prepare_config pins the audio token contract to
    // the official constants. An official config may carry its own
    // values (the fixture tree does, to stay consistent with a small
    // vocab); accept them when present.
    if let Some(v) = config_usize(&lm, "audio_code_offset") {
        config.audio_code_offset = v;
    }
    if let Some(v) = config_usize(&lm, "audio_end_token_id") {
        config.audio_end_token_id = v;
    }
    if let Some(v) = config_usize(&lm, "audio_cfg_token_id") {
        config.audio_cfg_token_id = v;
    }
    if let Some(v) = config_usize(&lm, "semantic_vocab_size") {
        config.semantic_vocab_size = v;
    }
    if config_usize(&depth, "hidden_size").unwrap_or(config.hidden_size) != config.hidden_size {
        return Err(SpeechError::BadConfig {
            field: "rvq_depth_decoder/hidden_size".to_string(),
            why: "does not match the language model".to_string(),
        });
    }
    if config_usize(&condition, "condition_hidden_dim").unwrap_or(config.hidden_size)
        != config.hidden_size
    {
        return Err(SpeechError::BadConfig {
            field: "condition_encoder/condition_hidden_dim".to_string(),
            why: "does not match the language model".to_string(),
        });
    }
    if config_usize(&transformer, "condition_dim").unwrap_or(config.condition_out_dim)
        != config.condition_out_dim
    {
        return Err(SpeechError::BadConfig {
            field: "transformer/condition_dim".to_string(),
            why: "does not match the condition encoder".to_string(),
        });
    }
    if config_usize(&vocoder, "latent_channels").unwrap_or(config.dit_in_channels)
        != config.dit_in_channels
    {
        return Err(SpeechError::BadConfig {
            field: "vocoder/latent_channels".to_string(),
            why: "does not match the flow transformer".to_string(),
        });
    }
    Ok(config)
}

/// Load and remap the official component shards into the converted
/// tensor namespace.
fn load_official_tensors(dir: &Path) -> Result<HashMap<String, Tensor>> {
    let components = [
        "language_model",
        "rvq_depth_decoder",
        "condition_encoder",
        "transformer",
        "vocoder",
    ];
    let mut out: HashMap<String, Tensor> = HashMap::new();
    for component in components {
        let component_dir = dir.join(component);
        let mut shard_names: Vec<String> = Vec::new();
        if let Ok(entries) = std::fs::read_dir(&component_dir) {
            for entry in entries.flatten() {
                let name = entry.file_name().to_string_lossy().to_string();
                if name.ends_with(".safetensors") {
                    shard_names.push(name);
                }
            }
        }
        shard_names.sort();
        if shard_names.is_empty() {
            return Err(SpeechError::Tensor {
                name: format!("{component}/*.safetensors"),
                why: "no safetensors found".to_string(),
            });
        }
        let mut raw: Vec<(String, Tensor)> = Vec::new();
        for shard in &shard_names {
            let file = SafetensorsFile::open(&component_dir.join(shard))?;
            for name in file.tensor_names() {
                let shape = file
                    .descriptor(name)
                    .map(|d| d.shape.clone())
                    .unwrap_or_default();
                let data = file.load_as_f32(name)?;
                raw.push((
                    name.to_string(),
                    Tensor {
                        data,
                        shape,
                        dtype: precision::DType::from_checkpoint(
                            &file.descriptor(name).unwrap().dtype,
                        )?,
                    },
                ));
            }
        }
        let mut state: HashMap<String, Tensor> = raw.into_iter().collect();
        if component == "vocoder" {
            state = fuse_weight_norm_pairs(state);
        }
        for (key, tensor) in state {
            let Some(key) = sanitize_official_key(&key) else {
                continue;
            };
            let full = format!("{component}.{key}");
            let tensor = remap_official_tensor(&key, tensor);
            if out.insert(full, tensor).is_some() {
                return Err(SpeechError::Tensor {
                    name: format!("{component}.{key}"),
                    why: "duplicate converted tensor".to_string(),
                });
            }
        }
    }
    Ok(out)
}

/// Collapse PyTorch `weight_g` / `weight_v` pairs into a fused weight.
fn fuse_weight_norm_pairs(state: HashMap<String, Tensor>) -> HashMap<String, Tensor> {
    let mut out = HashMap::new();
    let mut keys: Vec<String> = state.keys().cloned().collect();
    keys.sort();
    for key in keys {
        if key.ends_with(".weight_v") {
            let prefix = &key[..key.len() - ".weight_v".len()];
            let g_key = format!("{prefix}.weight_g");
            if let Some(g) = state.get(&g_key) {
                let v = &state[&key];
                let per = v.data.len() / v.shape[0];
                let mut fused = Vec::with_capacity(v.data.len());
                for o in 0..v.shape[0] {
                    let norm = v.data[o * per..(o + 1) * per]
                        .iter()
                        .map(|x| x * x)
                        .sum::<f32>()
                        .sqrt();
                    let norm = norm.max(1e-12);
                    for value in &v.data[o * per..(o + 1) * per] {
                        fused.push(g.data[o] * value / norm);
                    }
                }
                out.insert(
                    format!("{prefix}.weight"),
                    Tensor {
                        data: fused,
                        shape: v.shape.clone(),
                        // Upstream conversion explicitly casts both operands
                        // to float32 and leaves the fused weight in float32.
                        dtype: precision::DType::F32,
                    },
                );
                continue;
            }
        }
        if key.ends_with(".weight_g")
            && state.contains_key(&format!(
                "{}{}",
                &key[..key.len() - ".weight_g".len()],
                ".weight_v"
            ))
        {
            continue;
        }
        let tensor = state.get(&key).unwrap();
        out.insert(
            key,
            Tensor {
                data: tensor.data.clone(),
                shape: tensor.shape.clone(),
                dtype: tensor.dtype,
            },
        );
    }
    out
}

/// The official-tree key sanitizer; `None` drops the tensor.
fn sanitize_official_key(key: &str) -> Option<String> {
    if key.contains("rotary_emb") || key.ends_with(".inv_freq") {
        return None;
    }
    if key.contains("transformer_blocks")
        && (key.contains(".to_out.1.") || key.ends_with(".to_out.1"))
    {
        return None;
    }
    if key.contains("transformer_blocks")
        && (key.ends_with(".to_out.weight") || key.ends_with(".to_out.bias"))
    {
        let suffix = if key.ends_with(".weight") {
            ".weight"
        } else {
            ".bias"
        };
        return Some(format!(
            "{}{}{}",
            &key[..key.len() - suffix.len()],
            ".0",
            suffix
        ));
    }
    Some(key.to_string())
}

fn remap_official_tensor(key: &str, tensor: Tensor) -> Tensor {
    if tensor.shape.len() != 3 || !key.ends_with(".weight") {
        return tensor;
    }
    let [a, b, c] = [tensor.shape[0], tensor.shape[1], tensor.shape[2]];
    let mut data = Vec::with_capacity(tensor.data.len());
    if is_conv_transpose_key(key) {
        // PyTorch [in=a, out=b, K=c] -> MLX [out, K, in].
        for o in 0..b {
            for k in 0..c {
                for i in 0..a {
                    data.push(tensor.data[(i * b + o) * c + k]);
                }
            }
        }
        Tensor {
            data,
            shape: vec![b, c, a],
            dtype: tensor.dtype,
        }
    } else {
        // PyTorch [out=a, in=b, K=c] -> MLX [out, K, in].
        for o in 0..a {
            for k in 0..c {
                for i in 0..b {
                    data.push(tensor.data[(o * b + i) * c + k]);
                }
            }
        }
        Tensor {
            data,
            shape: vec![a, c, b],
            dtype: tensor.dtype,
        }
    }
}

/// Port of the reference's conv-transpose key regex
/// `(conv_t\d+|conv_transpose|convtr[^a-z]|deconv)`, case-insensitive.
fn is_conv_transpose_key(key: &str) -> bool {
    let lower = key.to_lowercase();
    if lower.contains("conv_transpose") || lower.contains("deconv") {
        return true;
    }
    // conv_t\d+
    let bytes = lower.as_bytes();
    let mut search = 0usize;
    while let Some(pos) = lower[search..].find("conv_t") {
        let after = search + pos + "conv_t".len();
        if after < bytes.len() && bytes[after].is_ascii_digit() {
            return true;
        }
        search += pos + 1;
    }
    let bytes = lower.as_bytes();
    let mut search = 0usize;
    while let Some(pos) = lower[search..].find("convtr") {
        let after = search + pos + "convtr".len();
        if after >= bytes.len() || !bytes[after].is_ascii_lowercase() {
            return true;
        }
        search += pos + 1;
    }
    false
}
