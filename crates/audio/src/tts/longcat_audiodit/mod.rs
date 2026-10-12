//! LongCat-AudioDiT family (meituan-longcat LongCat-AudioDiT).
//!
//! Port of `mlx_audio/tts/models/longcat_audiodit/` (longcat_audiodit.py,
//! config.py, dit.py, text_encoder.py, vae.py) at mlx-audio
//! [`e1b19b9054bf163f5d812221a54fcc346f1890e9`](https://github.com/Blaizzy/mlx-audio/tree/e1b19b9054bf163f5d812221a54fcc346f1890e9/mlx_audio/tts/models/longcat_audiodit):
//! conditional flow matching over a latent space. A frozen UMT5 encoder
//! (relative-bias blocks, gated exact-GELU feed forwards) conditions a
//! CrossDiT transformer (global AdaLN, Qwen2-style rope over both the
//! latent and text axes, ConvNeXtV2 text processing, long skip with the
//! reference's mid-stack add at the repa layer, latent-condition
//! embedders). The Euler ODE integrates the velocity field from pure
//! noise with classifier-free guidance (or APG) against a zero-text
//! null branch, and the resulting latent decodes through the WAV-VAE
//! convolutional codec at 24 kHz.
//!
//! Scope of this port, matching the reference single-prompt path:
//! batch 1, non-streaming (the streaming chunk decoder is out of
//! scope), and caller-owned text tokenization (the reference downloads
//! the `google/umt5-base` tokenizer inside generate). Noise comes from
//! a [`Noise`] provider so fixture tests replay the recorded reference
//! draws and standalone runs can seed the crate-local threefry RNG.
//! See [README.md](README.md) for the verification evidence and open
//! items.

use serde_json::Value;

use self::dit::AudioDiTTransformer;
use self::text_encoder::Umt5Encoder;
use self::vae::AudioDiTVae;
use crate::lm::Weights;
use crate::music::minimax_music3::rng;
use crate::{quant, Result, SpeechError};

mod dit;
mod text_encoder;
mod vae;

#[cfg(test)]
mod tests;

/// Pinned mlx-audio commit this port was transcribed from.
pub const REFERENCE_COMMIT: &str = "e1b19b9054bf163f5d812221a54fcc346f1890e9";

/// `EN_DUR_PER_CHAR`: English seconds per alphabetic character.
pub const EN_DUR_PER_CHAR: f64 = 0.082;
/// `ZH_DUR_PER_CHAR`: Chinese seconds per CJK character.
pub const ZH_DUR_PER_CHAR: f64 = 0.21;

/// `VaeConfig` with the reference defaults.
#[derive(Debug, Clone)]
pub struct VaeConfig {
    pub in_channels: usize,
    pub channels: usize,
    pub c_mults: Vec<usize>,
    pub strides: Vec<usize>,
    pub latent_dim: usize,
    pub encoder_latent_dim: usize,
    pub use_snake: bool,
    pub downsample_shortcut: String,
    pub upsample_shortcut: String,
    pub out_shortcut: String,
    pub in_shortcut: String,
    pub final_tanh: bool,
    pub downsampling_ratio: usize,
    pub sample_rate: u32,
    pub scale: f32,
}

impl Default for VaeConfig {
    fn default() -> Self {
        Self {
            in_channels: 1,
            channels: 128,
            c_mults: vec![1, 2, 4, 8, 16],
            strides: vec![2, 4, 4, 8, 8],
            latent_dim: 64,
            encoder_latent_dim: 128,
            use_snake: true,
            downsample_shortcut: "averaging".into(),
            upsample_shortcut: "duplicating".into(),
            out_shortcut: "averaging".into(),
            in_shortcut: "duplicating".into(),
            final_tanh: false,
            downsampling_ratio: 2048,
            sample_rate: 24000,
            scale: 0.71,
        }
    }
}

impl VaeConfig {
    /// One stride per encoder/decoder stage. A short `strides` list
    /// panics on indexing in the VAE load; a long one silently changes
    /// nothing, which hides a config that does not match the checkpoint.
    pub(crate) fn validate(&self) -> Result<()> {
        if self.c_mults.is_empty() {
            return Err(SpeechError::BadConfig {
                field: "vae_config.c_mults".into(),
                why: "must not be empty".into(),
            });
        }
        if self.strides.len() != self.c_mults.len() {
            return Err(SpeechError::BadConfig {
                field: "vae_config.strides".into(),
                why: format!(
                    "expected {} entries to match c_mults, got {}",
                    self.c_mults.len(),
                    self.strides.len()
                ),
            });
        }
        Ok(())
    }

    fn from_json(value: &Value) -> Self {
        let mut config = Self::default();
        let uint = |key: &str, default: usize| -> usize {
            value
                .get(key)
                .and_then(Value::as_u64)
                .map(|n| n as usize)
                .unwrap_or(default)
        };
        config.in_channels = uint("in_channels", config.in_channels);
        config.channels = uint("channels", config.channels);
        if let Some(list) = usize_list(value, "c_mults") {
            config.c_mults = list;
        }
        if let Some(list) = usize_list(value, "strides") {
            config.strides = list;
        }
        config.latent_dim = uint("latent_dim", config.latent_dim);
        config.encoder_latent_dim = uint("encoder_latent_dim", config.encoder_latent_dim);
        if let Some(v) = value.get("use_snake").and_then(Value::as_bool) {
            config.use_snake = v;
        }
        for (key, field) in [
            ("downsample_shortcut", &mut config.downsample_shortcut),
            ("upsample_shortcut", &mut config.upsample_shortcut),
            ("out_shortcut", &mut config.out_shortcut),
            ("in_shortcut", &mut config.in_shortcut),
        ] {
            if let Some(v) = value.get(key).and_then(Value::as_str) {
                *field = v.to_owned();
            }
        }
        if let Some(v) = value.get("final_tanh").and_then(Value::as_bool) {
            config.final_tanh = v;
        }
        config.downsampling_ratio = uint("downsampling_ratio", config.downsampling_ratio);
        config.sample_rate = uint("sample_rate", config.sample_rate as usize) as u32;
        if let Some(v) = value.get("scale").and_then(Value::as_f64) {
            config.scale = v as f32;
        }
        config
    }
}

/// `TextEncoderConfig` with the reference defaults.
#[derive(Debug, Clone)]
pub struct TextEncoderConfig {
    pub vocab_size: usize,
    pub d_model: usize,
    pub d_kv: usize,
    pub d_ff: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub relative_attention_num_buckets: usize,
    pub relative_attention_max_distance: usize,
    pub layer_norm_epsilon: f64,
    pub is_gated_act: bool,
    pub dense_act_fn: String,
}

impl Default for TextEncoderConfig {
    fn default() -> Self {
        Self {
            vocab_size: 256384,
            d_model: 768,
            d_kv: 64,
            d_ff: 2048,
            num_layers: 12,
            num_heads: 12,
            relative_attention_num_buckets: 32,
            relative_attention_max_distance: 128,
            layer_norm_epsilon: 1e-6,
            is_gated_act: true,
            dense_act_fn: "gelu_new".into(),
        }
    }
}

impl TextEncoderConfig {
    fn from_json(value: &Value) -> Self {
        let mut config = Self::default();
        let uint = |key: &str, default: usize| -> usize {
            value
                .get(key)
                .and_then(Value::as_u64)
                .map(|n| n as usize)
                .unwrap_or(default)
        };
        config.vocab_size = uint("vocab_size", config.vocab_size);
        config.d_model = uint("d_model", config.d_model);
        config.d_kv = uint("d_kv", config.d_kv);
        config.d_ff = uint("d_ff", config.d_ff);
        config.num_layers = uint("num_layers", config.num_layers);
        config.num_heads = uint("num_heads", config.num_heads);
        config.relative_attention_num_buckets = uint(
            "relative_attention_num_buckets",
            config.relative_attention_num_buckets,
        );
        config.relative_attention_max_distance = uint(
            "relative_attention_max_distance",
            config.relative_attention_max_distance,
        );
        if let Some(v) = value.get("layer_norm_epsilon").and_then(Value::as_f64) {
            config.layer_norm_epsilon = v;
        }
        if let Some(v) = value.get("is_gated_act").and_then(Value::as_bool) {
            config.is_gated_act = v;
        }
        if let Some(v) = value.get("dense_act_fn").and_then(Value::as_str) {
            config.dense_act_fn = v.to_owned();
        }
        config
    }
}

/// `ModelConfig` with the reference defaults.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub dit_dim: usize,
    pub dit_depth: usize,
    pub dit_heads: usize,
    pub dit_ff_mult: f64,
    pub dit_text_dim: usize,
    pub dit_bias: bool,
    pub dit_adaln_type: String,
    pub dit_long_skip: bool,
    pub dit_text_conv: bool,
    pub dit_qk_norm: bool,
    pub dit_eps: f64,
    pub dit_use_latent_condition: bool,
    pub repa_dit_layer: usize,
    pub latent_dim: usize,
    pub sampling_rate: u32,
    pub latent_hop: usize,
    pub max_wav_duration: f64,
    pub text_add_embed: bool,
    pub text_norm_feat: bool,
    pub vae_config: VaeConfig,
    pub text_encoder_config: TextEncoderConfig,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            dit_dim: 1536,
            dit_depth: 24,
            dit_heads: 24,
            dit_ff_mult: 4.0,
            dit_text_dim: 768,
            dit_bias: true,
            dit_adaln_type: "global".into(),
            dit_long_skip: true,
            dit_text_conv: true,
            dit_qk_norm: true,
            dit_eps: 1e-6,
            dit_use_latent_condition: true,
            repa_dit_layer: 8,
            latent_dim: 64,
            sampling_rate: 24000,
            latent_hop: 2048,
            max_wav_duration: 30.0,
            text_add_embed: true,
            text_norm_feat: true,
            vae_config: VaeConfig::default(),
            text_encoder_config: TextEncoderConfig::default(),
        }
    }
}

impl ModelConfig {
    /// `ModelConfig.from_dict` with the nested section parsing.
    pub fn from_json(value: &Value) -> Self {
        let mut config = Self::default();
        let uint = |key: &str, default: usize| -> usize {
            value
                .get(key)
                .and_then(Value::as_u64)
                .map(|n| n as usize)
                .unwrap_or(default)
        };
        config.dit_dim = uint("dit_dim", config.dit_dim);
        config.dit_depth = uint("dit_depth", config.dit_depth);
        config.dit_heads = uint("dit_heads", config.dit_heads);
        if let Some(v) = value.get("dit_ff_mult").and_then(Value::as_f64) {
            config.dit_ff_mult = v;
        }
        config.dit_text_dim = uint("dit_text_dim", config.dit_text_dim);
        if let Some(v) = value.get("dit_bias").and_then(Value::as_bool) {
            config.dit_bias = v;
        }
        if let Some(v) = value.get("dit_adaln_type").and_then(Value::as_str) {
            config.dit_adaln_type = v.to_owned();
        }
        if let Some(v) = value.get("dit_long_skip").and_then(Value::as_bool) {
            config.dit_long_skip = v;
        }
        if let Some(v) = value.get("dit_text_conv").and_then(Value::as_bool) {
            config.dit_text_conv = v;
        }
        if let Some(v) = value.get("dit_qk_norm").and_then(Value::as_bool) {
            config.dit_qk_norm = v;
        }
        if let Some(v) = value.get("dit_eps").and_then(Value::as_f64) {
            config.dit_eps = v;
        }
        if let Some(v) = value
            .get("dit_use_latent_condition")
            .and_then(Value::as_bool)
        {
            config.dit_use_latent_condition = v;
        }
        config.repa_dit_layer = uint("repa_dit_layer", config.repa_dit_layer);
        config.latent_dim = uint("latent_dim", config.latent_dim);
        config.sampling_rate = uint("sampling_rate", config.sampling_rate as usize) as u32;
        config.latent_hop = uint("latent_hop", config.latent_hop);
        if let Some(v) = value.get("max_wav_duration").and_then(Value::as_f64) {
            config.max_wav_duration = v;
        }
        if let Some(v) = value.get("text_add_embed").and_then(Value::as_bool) {
            config.text_add_embed = v;
        }
        if let Some(v) = value.get("text_norm_feat").and_then(Value::as_bool) {
            config.text_norm_feat = v;
        }
        if let Some(vae) = value.get("vae_config") {
            if vae.is_object() {
                config.vae_config = VaeConfig::from_json(vae);
            }
        }
        if let Some(te) = value.get("text_encoder_config") {
            if te.is_object() {
                config.text_encoder_config = TextEncoderConfig::from_json(te);
            }
        }
        config
    }
}

fn usize_list(value: &Value, key: &str) -> Option<Vec<usize>> {
    value.get(key).and_then(Value::as_array).map(|list| {
        list.iter()
            .filter_map(|v| v.as_u64().map(|n| n as usize))
            .collect()
    })
}

/// `_normalize_text`: lowercase, smart quotes to spaces, whitespace
/// collapsed, trimmed.
pub fn normalize_text(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    // The reference class is ["\u2018\u2019] (plus a duplicated ASCII
    // quote); the curly double quotes survive.
    for c in text.to_lowercase().chars() {
        if matches!(c, '"' | '\u{2018}' | '\u{2019}') {
            out.push(' ');
        } else {
            out.push(c);
        }
    }
    out.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// `_approx_duration`: seconds from the CJK / alphabetic / other
/// character counts, capped at `max_duration`.
pub fn approx_duration(text: &str, max_duration: f64) -> f64 {
    let squeezed: String = text.chars().filter(|c| !c.is_whitespace()).collect();
    let mut num_zh = 0usize;
    let mut num_en = 0usize;
    let mut num_other = 0usize;
    for c in squeezed.chars() {
        if ('\u{4e00}'..='\u{9fff}').contains(&c) {
            num_zh += 1;
        } else if c.is_alphabetic() {
            num_en += 1;
        } else {
            num_other += 1;
        }
    }
    if num_zh > num_en {
        num_zh += num_other;
    } else {
        num_en += num_other;
    }
    (num_zh as f64 * ZH_DUR_PER_CHAR + num_en as f64 * EN_DUR_PER_CHAR).min(max_duration)
}

/// The Euler schedule is `linspace(0, 1, steps)`, i.e. `steps - 1`
/// intervals. Below 2 steps the loop runs zero times and the raw
/// initial noise would come back as audio.
pub(crate) fn check_steps(steps: usize) -> Result<()> {
    if steps < 2 {
        return Err(SpeechError::Input {
            why: format!("steps must be at least 2, got {steps}"),
        });
    }
    Ok(())
}

/// `total_duration` is clamped to `max_frames`. A voice prompt at or
/// over the cap leaves no room for generated frames, and the
/// `total_duration - prompt_dur` span would underflow.
pub(crate) fn check_prompt_fits(prompt_dur: usize, max_frames: usize) -> Result<()> {
    if prompt_dur >= max_frames {
        return Err(SpeechError::Input {
            why: format!(
                "voice prompt is {prompt_dur} latent frames, which leaves no room under the \
                 {max_frames}-frame output cap"
            ),
        });
    }
    Ok(())
}

/// The flow-matching noise source: one standard-normal draw of `n`
/// values per call, in reference draw order (the VAE reparameterization
/// draw of a voice prompt, then the initial latent noise).
pub trait Noise {
    fn normal(&mut self, n: usize) -> Vec<f32>;
}

/// Seeded crate-local threefry noise for standalone runs.
pub struct SeededNoise {
    keys: rng::KeySequence,
}

impl SeededNoise {
    pub fn new(seed: u64) -> Self {
        Self {
            keys: rng::KeySequence::new(seed),
        }
    }
}

impl Noise for SeededNoise {
    fn normal(&mut self, n: usize) -> Vec<f32> {
        rng::normal(self.keys.next(), n)
    }
}

/// The text the caller tokenizes: the normalized reference transcript
/// prepended to the normalized input text (`f"{prompt_text} {text}"`).
pub fn prompt_text(text: &str, ref_text: Option<&str>) -> String {
    let text = normalize_text(text);
    match ref_text {
        Some(ref_text) => format!("{} {text}", normalize_text(ref_text)),
        None => text,
    }
}

/// The APG momentum buffer (`_MomentumBuffer`).
struct MomentumBuffer {
    momentum: f32,
    running_average: Vec<f32>,
}

impl MomentumBuffer {
    fn update(&mut self, value: &[f32]) {
        if self.running_average.len() != value.len() {
            self.running_average = vec![0.0; value.len()];
        }
        for (avg, v) in self.running_average.iter_mut().zip(value) {
            *avg = *avg * self.momentum + v;
        }
    }
}

/// `_apg_forward`: the adaptive projected guidance update. The
/// projection decomposes the (momentum-smoothed) difference into
/// components parallel and orthogonal to the conditional prediction.
fn apg_forward(
    pred_cond: &[f32],
    pred_uncond: &[f32],
    guidance_scale: f32,
    mut buffer: Option<&mut MomentumBuffer>,
    eta: f32,
) -> Vec<f32> {
    let mut diff: Vec<f32> = pred_cond
        .iter()
        .zip(pred_uncond)
        .map(|(c, u)| c - u)
        .collect();
    if let Some(buffer) = buffer.as_mut() {
        buffer.update(&diff);
        diff = buffer.running_average.clone();
    }
    // norm_threshold is 0.0 in the reference call, so the norm clamp
    // never applies.
    // v1_norm = pred_cond / (sqrt(sum(pred_cond^2)) + 1e-8), reduced
    // over every axis (a scalar here).
    let norm_sq: f32 = pred_cond.iter().map(|v| v * v).sum();
    let norm = norm_sq.sqrt() + 1e-8;
    // v0_parallel = (diff . v1_norm) * v1_norm.
    let dot: f32 = diff.iter().zip(pred_cond).map(|(d, c)| d * c / norm).sum();
    let mut out = Vec::with_capacity(pred_cond.len());
    for ((d, c), _u) in diff.iter().zip(pred_cond).zip(pred_uncond) {
        let parallel = dot * c / norm;
        let orthogonal = d - parallel;
        out.push(c + guidance_scale * (orthogonal + eta * parallel));
    }
    out
}

/// `generate` parameters with the reference defaults.
#[derive(Debug, Clone)]
pub struct GenerateParams {
    pub steps: usize,
    pub cfg_strength: f32,
    /// `"cfg"` or `"apg"`.
    pub guidance_method: String,
}

impl Default for GenerateParams {
    fn default() -> Self {
        Self {
            steps: 16,
            cfg_strength: 4.0,
            guidance_method: "cfg".into(),
        }
    }
}

/// One generation result.
#[derive(Debug, Clone)]
pub struct Generated {
    pub waveform: Vec<f32>,
    pub sample_rate: u32,
    /// The ODE result frames (`pred_latent`).
    pub latent_frames: usize,
    pub latent: Vec<f32>,
}

/// The loaded LongCat-AudioDiT model.
pub struct Model {
    pub config: ModelConfig,
    text_encoder: Umt5Encoder,
    transformer: AudioDiTTransformer,
    vae: AudioDiTVae,
}

impl Model {
    /// Opens a checkpoint directory (`config.json` + safetensors).
    pub fn open(dir: &std::path::Path) -> Result<Self> {
        let config_value = quant::read_json(&dir.join("config.json"))?;
        let config = ModelConfig::from_json(&config_value);
        let weights = Weights::open(dir)?;
        Self::load(&weights, config)
    }

    /// Loads from opened weights and a parsed config. The checkpoint
    /// tensor names sit under `text_encoder.`, `transformer.`, and
    /// `vae.` (the reference sanitize output).
    pub fn load(weights: &Weights, config: ModelConfig) -> Result<Self> {
        let file = weights
            .shards()
            .first()
            .ok_or_else(|| SpeechError::Tensor {
                name: "model.safetensors".into(),
                why: "checkpoint has no shards".into(),
            })?;
        Ok(Self {
            text_encoder: Umt5Encoder::load(file, "text_encoder", &config.text_encoder_config)?,
            transformer: AudioDiTTransformer::load(file, "transformer", &config)?,
            vae: AudioDiTVae::load(file, "vae", &config.vae_config)?,
            config,
        })
    }

    pub fn sample_rate(&self) -> u32 {
        self.config.sampling_rate
    }

    /// `encode_text`: the UMT5 pass plus the `text_norm_feat` /
    /// `text_add_embed` post-processing.
    fn encode_text(&self, input_ids: &[i32], attention_mask: &[f32]) -> Result<Vec<f32>> {
        let (mut last_hidden, mut initial_embed) =
            self.text_encoder.forward(input_ids, attention_mask)?;
        let d_model = self.config.text_encoder_config.d_model;
        if self.config.text_norm_feat {
            dit::layer_norm_in_place(&mut last_hidden, d_model, 1e-6);
        }
        if self.config.text_add_embed {
            if self.config.text_norm_feat {
                dit::layer_norm_in_place(&mut initial_embed, d_model, 1e-6);
            }
            for (dst, src) in last_hidden.iter_mut().zip(&initial_embed) {
                *dst += src;
            }
        }
        Ok(last_hidden)
    }

    /// `encode_prompt_audio`: right-pad the mono prompt to the hop
    /// multiple plus `3 * hop`, encode, and drop the 3 tail frames.
    pub fn encode_prompt_audio(
        &self,
        wav: &[f32],
        noise: &mut dyn Noise,
    ) -> Result<(Vec<f32>, usize)> {
        let hop = self.config.latent_hop;
        let off = 3usize;
        let mut padded = wav.to_vec();
        let remainder = padded.len() % hop;
        if remainder != 0 {
            padded.resize(padded.len() + (hop - remainder), 0.0);
        }
        padded.resize(padded.len() + hop * off, 0.0);
        let mut latent = self.vae.encode(&padded, noise)?;
        let frames = latent.len() / self.config.latent_dim;
        let keep = frames - off;
        latent.truncate(keep * self.config.latent_dim);
        Ok((latent, keep))
    }

    /// The reference `generate` for batch 1, non-streaming.
    /// `input_ids`/`attention_mask` come from the caller's tokenizer.
    /// `ref_audio` is mono samples at the checkpoint sample rate;
    /// `ref_text` is its transcript (raw text, normalized inside).
    #[allow(
        clippy::too_many_arguments,
        reason = "the reference generate signature carries these parameters"
    )]
    pub fn generate(
        &mut self,
        text: &str,
        input_ids: &[i32],
        attention_mask: &[f32],
        params: &GenerateParams,
        noise: &mut dyn Noise,
        ref_audio: Option<&[f32]>,
        ref_text: Option<&str>,
    ) -> Result<Generated> {
        let sr = self.config.sampling_rate as f64;
        let hop = self.config.latent_hop as f64;
        let max_duration = self.config.max_wav_duration;
        let latent_dim = self.config.latent_dim;
        check_steps(params.steps)?;

        let text = normalize_text(text);
        let no_prompt = ref_audio.is_none();

        let text_condition = self.encode_text(input_ids, attention_mask)?;
        let text_condition_len: f32 = attention_mask.iter().sum();
        let cond_seq = input_ids.len();

        // Prompt audio encoding.
        let (prompt_latent, prompt_dur) = match ref_audio {
            Some(wav) => self.encode_prompt_audio(wav, noise)?,
            None => (Vec::new(), 0usize),
        };

        // Duration estimation. The prompt must fit under the cap before
        // any subtraction below uses `max_duration - prompt_time`.
        let max_frames = (max_duration * sr / hop) as usize;
        check_prompt_fits(prompt_dur, max_frames)?;
        let prompt_time = prompt_dur as f64 * hop / sr;
        let mut dur_sec = approx_duration(&text, max_duration - prompt_time);
        if let (false, Some(ref_text)) = (no_prompt, ref_text) {
            let approx_pd = approx_duration(ref_text, max_duration);
            let ratio = (prompt_time / approx_pd.max(1e-6)).clamp(1.0, 1.5);
            dur_sec *= ratio;
        }
        let duration = (dur_sec * sr / hop) as usize;
        let total_duration = (duration + prompt_dur).min(max_frames);
        if total_duration == 0 {
            return Err(SpeechError::Input {
                why: "estimated duration rounds to zero frames".into(),
            });
        }

        // Masks.
        let mask = vec![true; total_duration];
        let text_mask: Vec<bool> = (0..cond_seq)
            .map(|i| (i as f32) < text_condition_len)
            .collect();

        // Conditioning.
        let latent_cond: Vec<f32> = if !no_prompt {
            let gen_len = total_duration - prompt_dur;
            let mut padded = prompt_latent.clone();
            padded.resize(padded.len() + gen_len * latent_dim, 0.0);
            padded
        } else {
            vec![0.0; total_duration * latent_dim]
        };
        let empty_latent_cond = vec![0.0; total_duration * latent_dim];

        // APG buffer.
        let mut apg_buffer = if params.guidance_method == "apg" {
            Some(MomentumBuffer {
                momentum: -0.3,
                running_average: Vec::new(),
            })
        } else {
            None
        };

        // Initial noise (draw 2 after a voice prompt's VAE draw).
        let mut y = noise.normal(total_duration * latent_dim);
        // The reference captures the prompt frames of the ORIGINAL
        // noise and re-interpolates from them every step.
        let prompt_noise: Vec<f32> = y[..prompt_dur * latent_dim].to_vec();

        // Euler ODE integration over linspace(0, 1, steps).
        let steps = params.steps;
        let repa_layer = self.config.repa_dit_layer;
        for i in 0..steps.saturating_sub(1) {
            let t_val = i as f64 / (steps - 1) as f64;
            let dt = 1.0f64 / (steps - 1) as f64;
            if prompt_dur > 0 {
                // y_prompt = prompt_noise * (1 - t) + latent_cond[:, :pd] * t.
                for f in 0..prompt_dur * latent_dim {
                    y[f] = (prompt_noise[f] as f64 * (1.0 - t_val) + latent_cond[f] as f64 * t_val)
                        as f32;
                }
            }
            let mut pred = self.transformer.forward(
                &y,
                total_duration,
                &text_condition,
                cond_seq,
                t_val as f32,
                Some(&mask),
                Some(&text_mask),
                Some(repa_layer),
                &latent_cond,
            );
            if params.cfg_strength >= 1e-5 {
                let mut y_null = y.clone();
                if prompt_dur > 0 {
                    for v in &mut y_null[..prompt_dur * latent_dim] {
                        *v = 0.0;
                    }
                }
                let null_pred = self.transformer.forward(
                    &y_null,
                    total_duration,
                    &vec![0.0; text_condition.len()],
                    cond_seq,
                    t_val as f32,
                    Some(&mask),
                    Some(&text_mask),
                    Some(repa_layer),
                    &empty_latent_cond,
                );
                if params.guidance_method == "apg" {
                    // Guidance over the non-prompt slice only.
                    let width = (total_duration - prompt_dur) * latent_dim;
                    let x_s: Vec<f32> = y[prompt_dur * latent_dim..].to_vec();
                    let pred_s: Vec<f32> = pred[prompt_dur * latent_dim..].to_vec();
                    let null_s: Vec<f32> = null_pred[prompt_dur * latent_dim..].to_vec();
                    let pred_sample: Vec<f32> = x_s
                        .iter()
                        .zip(&pred_s)
                        .map(|(x, p)| *x + (1.0 - t_val) as f32 * p)
                        .collect();
                    let null_sample: Vec<f32> = x_s
                        .iter()
                        .zip(&null_s)
                        .map(|(x, n)| *x + (1.0 - t_val) as f32 * n)
                        .collect();
                    let buffer = apg_buffer.as_mut();
                    let out =
                        apg_forward(&pred_sample, &null_sample, params.cfg_strength, buffer, 0.5);
                    let recovered: Vec<f32> = out
                        .iter()
                        .zip(&x_s)
                        .map(|(o, x)| (*o - *x) / (1.0 - t_val) as f32)
                        .collect();
                    let mut combined = vec![0.0f32; total_duration * latent_dim];
                    combined[..prompt_dur * latent_dim]
                        .copy_from_slice(&pred[..prompt_dur * latent_dim]);
                    combined[prompt_dur * latent_dim..]
                        .copy_from_slice(&recovered[..width.min(recovered.len())]);
                    pred = combined;
                } else {
                    let guided: Vec<f32> = pred
                        .iter()
                        .zip(&null_pred)
                        .map(|(c, n)| c + (c - n) * params.cfg_strength)
                        .collect();
                    pred = guided;
                }
            }
            for (v, p) in y.iter_mut().zip(&pred) {
                *v += p * dt as f32;
            }
        }

        let pred_latent = if no_prompt {
            y
        } else {
            y[prompt_dur * latent_dim..].to_vec()
        };
        let latent_frames = pred_latent.len() / latent_dim;
        let waveform = self.vae.decode(&pred_latent, latent_frames);
        Ok(Generated {
            waveform,
            sample_rate: self.config.sampling_rate,
            latent_frames,
            latent: pred_latent,
        })
    }
}
