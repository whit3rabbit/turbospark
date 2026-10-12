//! Confucius4-TTS: multilingual zero-shot voice cloning over a
//! w2v-BERT semantic encoder, an ECAPA-style prefix, a GPT-2
//! text-to-semantic decoder, an S2A flow-matching mel estimator, and
//! the BigVGAN vocoder.
//!
//! Port of `mlx_audio/tts/models/confucius4/` at mlx-audio
//! [`e1b19b9054bf163f5d812221a54fcc346f1890e9`](https://github.com/Blaizzy/mlx-audio/tree/e1b19b9054bf163f5d812221a54fcc346f1890e9/mlx_audio/tts/models/confucius4),
//! single batch, portable f32.
//!
//! Checkpoint layout (one directory): `w2vbert_mlx.safetensors`,
//! `t2s_model.safetensors`, `s2a_mlx.safetensors`,
//! `bigvgan_mlx.safetensors`, `campplus.safetensors`,
//! `w2v_stats.npz` (mean/std for the condition vector),
//! `fbank_filters.npz` (the precomputed mel matrix + povey window).
//! The reference reads those with `mx.load`/`np.load`; the port opens
//! the safetensors files directly and the npz archives through the
//! model_io npz reader.
//!
//! Caller-owned intake: the text tokenizer (a `tokenizers` JSON file in
//! the checkpoint) and the reference-audio decode are caller concerns;
//! [`Model::generate`] takes 16 kHz mono samples and text token ids.
//! The language token prefix and the wrapper sentence are assembled by
//! the caller too ([`language_token`] carries the reference map).

pub mod campplus;
pub mod dsp;
pub mod prefix;
pub mod s2a;
pub mod t2s;
pub mod w2vbert;

use std::path::Path;

use serde_json::Value;

use crate::codec::bigvgan::{Bigvgan, BigvganConfig};
use crate::{Result, SpeechError};

use campplus::{CampPlus, CampPlusConfig};
use prefix::{Prefix, PrefixConfig};
use s2a::{S2AConfig, S2A};
use t2s::{T2SConfig, T2S};
use w2vbert::{W2VBert, W2VBertConfig};

/// The reference language-token subset (`LANGUAGE_TOKEN`).
pub fn language_token(lang: &str) -> &'static str {
    match lang {
        "zh" => "请用中文朗读接下来的文字",
        "en" => "请用英文朗读接下来的文字",
        "vi" => "请用越南语朗读接下来的文字",
        "ja" => "请用日语朗读接下来的文字",
        "ko" => "请用韩语朗读接下来的文字",
        "th" => "请用泰语朗读接下来的文字",
        _ => "请用英文朗读接下来的文字",
    }
}

/// The pipeline configuration parsed from the checkpoint `config.json`
/// (tiny fixture checkpoints override every dimension).
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub model_type: String,
    pub sample_rate: u32,
    pub w2vbert: W2VBertConfig,
    pub t2s: T2SConfig,
    pub s2a: S2AConfig,
    pub prefix: PrefixConfig,
    pub campplus: CampPlusConfig,
    pub bigvgan: Value,
}

impl Default for ModelConfig {
    fn default() -> Self {
        ModelConfig {
            model_type: "confucius4".into(),
            sample_rate: 22050,
            w2vbert: W2VBertConfig::default(),
            t2s: T2SConfig::default(),
            s2a: S2AConfig::default(),
            prefix: PrefixConfig::default(),
            campplus: CampPlusConfig::default(),
            bigvgan: serde_json::json!({
                "num_mels": 80,
                "upsample_rates": [4, 4, 2, 2, 2, 2],
                "upsample_kernel_sizes": [8, 8, 4, 4, 4, 4],
                "upsample_initial_channel": 1536,
                "resblock_kernel_sizes": [3, 7, 11],
                "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                "activation": "snakebeta",
                "snake_logscale": true,
                "use_bias_at_final": false,
                "use_tanh_at_final": false,
            }),
        }
    }
}

impl ModelConfig {
    pub fn from_json(value: &Value) -> Self {
        let mut config = Self::default();
        if let Some(v) = value.get("model_type").and_then(Value::as_str) {
            config.model_type = v.to_owned();
        }
        if let Some(v) = value.get("sample_rate").and_then(Value::as_u64) {
            config.sample_rate = v as u32;
        }
        if let Some(section) = value.get("w2vbert") {
            if let Some(v) = opt_usize(section, "hidden") {
                config.w2vbert.hidden = v;
            }
            if let Some(v) = opt_usize(section, "num_heads") {
                config.w2vbert.num_heads = v;
            }
            if let Some(v) = opt_usize(section, "head_dim") {
                config.w2vbert.head_dim = v;
            }
            if let Some(v) = opt_usize(section, "layers") {
                config.w2vbert.layers = v;
            }
            if let Some(v) = opt_usize(section, "left") {
                config.w2vbert.left = v as i64;
            }
            if let Some(v) = opt_usize(section, "right") {
                config.w2vbert.right = v as i64;
            }
        }
        if let Some(section) = value.get("t2s") {
            if let Some(v) = opt_usize(section, "d_model") {
                config.t2s.d_model = v;
            }
            if let Some(v) = opt_usize(section, "n_heads") {
                config.t2s.n_heads = v;
            }
            if let Some(v) = opt_usize(section, "n_layers") {
                config.t2s.n_layers = v;
            }
            if let Some(v) = opt_usize(section, "semantic_vocab") {
                config.t2s.semantic_vocab = v;
            }
            if let Some(v) = opt_usize(section, "max_positions") {
                config.t2s.max_positions = v;
            }
        }
        if let Some(section) = value.get("s2a") {
            if let Some(v) = opt_usize(section, "hidden") {
                config.s2a.hidden = v;
            }
            if let Some(v) = opt_usize(section, "num_heads") {
                config.s2a.num_heads = v;
            }
            if let Some(v) = opt_usize(section, "head_dim") {
                config.s2a.head_dim = v;
            }
            if let Some(v) = opt_usize(section, "depth") {
                config.s2a.depth = v;
            }
            if let Some(v) = opt_usize(section, "wavenet_layers") {
                config.s2a.wavenet_layers = v;
            }
            if let Some(v) = opt_usize(section, "freqs_positions") {
                config.s2a.freqs_positions = v;
            }
            if let Some(v) = opt_usize(section, "codebook_size") {
                config.s2a.codebook_size = v;
            }
            if let Some(v) = opt_usize(section, "codebook_dim") {
                config.s2a.codebook_dim = v;
            }
            if let Some(v) = opt_usize(section, "speaker_dim") {
                config.s2a.speaker_dim = v;
            }
            if let Some(v) = opt_usize(section, "encoder_dim") {
                config.s2a.encoder_dim = v;
            }
            if let Some(v) = opt_usize(section, "latent_dim") {
                config.s2a.latent_dim = v;
            }
            if let Some(v) = opt_usize(section, "mels") {
                config.s2a.mels = v;
            }
        }
        if let Some(section) = value.get("prefix") {
            if let Some(v) = opt_usize(section, "text_vocab") {
                config.prefix.text_vocab = v;
            }
            if let Some(v) = opt_usize(section, "text_embed_dim") {
                config.prefix.text_embed_dim = v;
            }
            if let Some(v) = opt_usize(section, "hidden") {
                config.prefix.hidden = v;
            }
            if let Some(v) = opt_usize(section, "cond_dim") {
                config.prefix.cond_dim = v;
            }
            if let Some(v) = opt_usize(section, "max_positions") {
                config.prefix.max_positions = v;
            }
        }
        if let Some(section) = value.get("campplus") {
            if let Some(v) = opt_usize(section, "embedding_size") {
                config.campplus.embedding_size = v;
            }
            if let Some(v) = opt_usize(section, "m_channels") {
                config.campplus.m_channels = v;
            }
            if let Some(v) = opt_usize(section, "init_channels") {
                config.campplus.init_channels = v;
            }
            if let Some(v) = opt_usize(section, "growth_rate") {
                config.campplus.growth_rate = v;
            }
            if let Some(v) = opt_usize(section, "bn_size") {
                config.campplus.bn_size = v;
            }
            if let Some(layers) = section.get("num_layers").and_then(Value::as_array) {
                for (index, v) in layers.iter().enumerate() {
                    if index < 3 {
                        if let Some(n) = v.as_u64() {
                            config.campplus.num_layers[index] = n as usize;
                        }
                    }
                }
            }
        }
        if let Some(section) = value.get("bigvgan") {
            config.bigvgan = section.clone();
        }
        config
    }
}

fn opt_usize(value: &Value, key: &str) -> Option<usize> {
    value.get(key).and_then(Value::as_u64).map(|n| n as usize)
}

/// `compute_condition` indexes both stats vectors by `index % cond_dim`,
/// so a short tensor would panic mid-request. Longer tensors are accepted,
/// matching the reference indexing.
fn check_stats_len(mean_len: usize, std_len: usize, cond_dim: usize) -> Result<()> {
    if mean_len < cond_dim || std_len < cond_dim {
        return Err(SpeechError::BadConfig {
            field: "w2v_stats".into(),
            why: format!(
                "mean/std must hold at least cond_dim={cond_dim} values, got {mean_len}/{std_len}"
            ),
        });
    }
    Ok(())
}

/// The loaded pipeline.
pub struct Model {
    pub config: ModelConfig,
    w2vbert: W2VBert,
    prefix: Prefix,
    t2s: T2S,
    s2a: S2A,
    vocoder: Bigvgan,
    campplus: CampPlus,
    stats_mean: Vec<f32>,
    stats_std: Vec<f32>,
    fbank_mel: Vec<f32>,
    fbank_window: Vec<f32>,
}

impl Model {
    /// Opens the checkpoint directory.
    pub fn open(dir: &Path) -> Result<Self> {
        let config_value = crate::quant::read_json(&dir.join("config.json"))?;
        Self::open_with_config(dir, config_value)
    }

    /// [`Self::open`] with a caller-supplied config document.
    pub fn open_with_config(dir: &Path, config_value: Value) -> Result<Self> {
        let config = ModelConfig::from_json(&config_value);
        let open = |name: &str| -> Result<turbospark_model_io::safetensors::SafetensorsFile> {
            turbospark_model_io::safetensors::SafetensorsFile::open(&dir.join(name))
                .map_err(crate::SpeechError::from)
        };
        let w2vbert = W2VBert::load(&open("w2vbert_mlx.safetensors")?, config.w2vbert.clone())?;
        let t2s_file = open("t2s_model.safetensors")?;
        let prefix = Prefix::load(&t2s_file, config.prefix.clone())?;
        let t2s = T2S::load(&t2s_file, config.t2s.clone())?;
        let s2a = S2A::load(&open("s2a_mlx.safetensors")?, config.s2a.clone())?;
        let vocoder = Bigvgan::load(
            BigvganConfig::from_json(&config.bigvgan)?,
            &open("bigvgan_mlx.safetensors")?,
        )?;
        let campplus = CampPlus::load(&open("campplus.safetensors")?, config.campplus.clone())?;
        // w2v_stats.npz + fbank_filters.npz: minimal npz readers over
        // the model_io zip reader.
        // The reference reads `w2v_stats.npz` and `fbank_filters.npz`;
        // the port reads the same-key safetensors files the conversion
        // writes alongside them (no zip reader in the portable crate).
        let stats = turbospark_model_io::safetensors::SafetensorsFile::open(
            &dir.join("w2v_stats.safetensors"),
        )
        .map_err(crate::SpeechError::from)?;
        let stats_mean = stats
            .load_as_f32("mean")
            .map_err(crate::SpeechError::from)?;
        let stats_std = stats.load_as_f32("std").map_err(crate::SpeechError::from)?;
        // compute_condition indexes both vectors by `index % cond_dim`, so
        // a short stats tensor would panic mid-request. Longer tensors are
        // accepted, matching the indexing the reference performs.
        check_stats_len(stats_mean.len(), stats_std.len(), config.prefix.cond_dim)?;
        let fbank = turbospark_model_io::safetensors::SafetensorsFile::open(
            &dir.join("fbank_filters.safetensors"),
        )
        .map_err(crate::SpeechError::from)?;
        let fbank_mel = fbank.load_as_f32("mel").map_err(crate::SpeechError::from)?;
        let fbank_window = fbank
            .load_as_f32("window")
            .map_err(crate::SpeechError::from)?;
        Ok(Model {
            config,
            w2vbert,
            prefix,
            t2s,
            s2a,
            vocoder,
            campplus,
            stats_mean,
            stats_std,
            fbank_mel,
            fbank_window,
        })
    }

    pub fn sample_rate(&self) -> u32 {
        self.config.sample_rate
    }

    /// Stage 1-2 of the reference generate: the normalized w2v condition
    /// vector, the CAMPPlus style embedding, the reference mel, and its
    /// frame count. `audio_16k` is 16 kHz mono.
    pub fn compute_condition(
        &self,
        audio_16k: &[f32],
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>, usize)> {
        let feats = dsp::fbank_160(audio_16k, &self.fbank_mel, &self.fbank_window)?;
        let frames = feats.len() / 160;
        let h17 = self.w2vbert.hidden17(&feats, frames)?;
        let cond_dim = self.config.prefix.cond_dim;
        let mut cond_vec = vec![0.0f32; h17.len()];
        for (index, value) in h17.iter().enumerate() {
            let col = index % cond_dim;
            cond_vec[index] = (value - self.stats_mean[col]) / self.stats_std[col];
        }
        let camp_fbank = dsp::kaldi_fbank(audio_16k, 16000, 80)?;
        let style = self
            .campplus
            .inference(&mean_normalize(&camp_fbank), camp_fbank.len() / 80);
        let ref_mel = dsp::ref_mel(audio_16k)?;
        let t_ref = ref_mel.len() / 80;
        Ok((cond_vec, style, ref_mel, t_ref))
    }

    /// Stage 3: the prefix embeddings ([cond_emb 1 x d], [text_emb T x d]).
    pub fn prefix_embeddings(
        &self,
        cond_vec: &[f32],
        text_ids: &[i32],
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let cond_emb = self
            .prefix
            .cond_emb(cond_vec, cond_vec.len() / self.config.prefix.cond_dim);
        let text_emb = self.prefix.text_emb(text_ids)?;
        Ok((cond_emb, text_emb))
    }

    /// Stage 4: the T2S generation (`T2S::generate`).
    #[allow(clippy::too_many_arguments)]
    pub fn t2s_generate(
        &self,
        cond_emb: &[f32],
        text_emb: &[f32],
        max_new: usize,
        temperature: f32,
        top_k: usize,
        top_p: f32,
        rep_pen: f32,
        greedy: bool,
        seed: u64,
    ) -> Result<(Vec<i32>, Vec<f32>)> {
        let key = crate::music::minimax_music3::rng::Key::new(seed);
        self.t2s.generate(
            cond_emb,
            text_emb,
            max_new,
            temperature,
            top_k,
            top_p,
            rep_pen,
            greedy,
            key,
        )
    }

    /// Stage 4b: the teacher-forced T2S step logits (fixture parity
    /// path for the head outputs the greedy trajectory saw).
    pub fn t2s_step_logits(
        &self,
        cond_emb: &[f32],
        text_emb: &[f32],
        codes: &[i32],
    ) -> Result<Vec<Vec<f32>>> {
        // Errors propagate: unwrap_or_default used to turn a T2S failure
        // into empty logits that looked like a valid (empty) trajectory.
        self.t2s.step_logits(cond_emb, text_emb, codes)
    }

    /// Stage 5: `build_mu` plus the Euler CFG solve with a caller-owned
    /// initial noise plane `[mels, total]` (the reference seeds
    /// `mx.random`; the fixtures replay recorded draws).
    pub fn s2a_solve(
        &self,
        codes: &[i32],
        latent: &[f32],
        style: &[f32],
        ref_mel: &[f32],
        t_ref: usize,
        z: &[f32],
    ) -> Result<Vec<f32>> {
        let mu = self.s2a.build_mu(codes, latent, t_ref)?;
        let total = mu.len() / self.config.s2a.hidden;
        Ok(self.s2a.solve_euler(
            z,
            ref_mel,
            &mu,
            style,
            &linspace(0.0, 1.0, 26),
            0.7,
            total,
            t_ref,
        ))
    }

    /// Stage 6: crop the target mel tail and vocode.
    pub fn vocode(&self, mel: &[f32], total: usize, t_ref: usize) -> Result<Vec<f32>> {
        let mels = self.config.s2a.mels;
        let target_frames = total - t_ref;
        let mel_target = &mel[t_ref * mels..];
        let mut mel_cm = vec![0.0f32; target_frames * mels];
        for f in 0..target_frames {
            for m in 0..mels {
                mel_cm[m * target_frames + f] = mel_target[f * mels + m];
            }
        }
        self.vocoder.forward(&mel_cm, target_frames)
    }

    /// The reference generate pipeline over all stages; the initial
    /// noise comes from the crate-local threefry RNG seeded with
    /// `seed + 1` (the reference seeds `mx.random`).
    #[allow(clippy::too_many_arguments)]
    pub fn generate(
        &self,
        audio_16k: &[f32],
        text_ids: &[i32],
        max_new: usize,
        temperature: f32,
        top_k: usize,
        top_p: f32,
        rep_pen: f32,
        greedy: bool,
        seed: u64,
    ) -> Result<Generated> {
        let (cond_vec, style, ref_mel, t_ref) = self.compute_condition(audio_16k)?;
        let (cond_emb, text_emb) = self.prefix_embeddings(&cond_vec, text_ids)?;
        let (codes, latent) = self.t2s_generate(
            &cond_emb,
            &text_emb,
            max_new,
            temperature,
            top_k,
            top_p,
            rep_pen,
            greedy,
            seed,
        )?;
        let mu = self.s2a.build_mu(&codes, &latent, t_ref)?;
        let total = mu.len() / self.config.s2a.hidden;
        // The draw fills the channel-major [mels, total] plane in
        // reference consumption order.
        let z = crate::music::minimax_music3::rng::normal(
            crate::music::minimax_music3::rng::Key::new(seed.wrapping_add(1)),
            total * 80,
        );
        let mel = self.s2a_solve(&codes, &latent, &style, &ref_mel, t_ref, &z)?;
        let samples = self.vocode(&mel, total, t_ref)?;
        Ok(Generated {
            samples,
            sample_rate: self.config.sample_rate,
            codes: codes.len(),
        })
    }
}

/// One generated utterance.
#[derive(Debug, Clone)]
pub struct Generated {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    /// The number of semantic codes the T2S stage produced.
    pub codes: usize,
}

fn mean_normalize(fbank: &[f32]) -> Vec<f32> {
    let bins = 80;
    let frames = fbank.len() / bins;
    let mut out = fbank.to_vec();
    for d in 0..bins {
        let mut mean = 0.0f32;
        for f in 0..frames {
            mean += fbank[f * bins + d];
        }
        mean /= frames as f32;
        for f in 0..frames {
            out[f * bins + d] -= mean;
        }
    }
    out
}

fn linspace(start: f32, end: f32, steps: usize) -> Vec<f32> {
    (0..steps)
        .map(|i| start + (end - start) * i as f32 / (steps - 1) as f32)
        .collect()
}

#[cfg(test)]
mod tests;
