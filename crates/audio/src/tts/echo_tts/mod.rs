//! Echo-TTS: a flow-matching DiT over Fish S1 DAC latents with PCA
//! compression and independent text/speaker classifier-free guidance.
//!
//! Port of `mlx_audio/tts/models/echo_tts/` (echo_tts.py, model.py,
//! sampling.py, audio.py, text.py, config.py) at mlx-audio
//! [`e1b19b9054bf163f5d812221a54fcc346f1890e9`](https://github.com/Blaizzy/mlx-audio/tree/e1b19b9054bf163f5d812221a54fcc346f1890e9/mlx_audio/tts/models/echo_tts),
//! single batch, non-streaming (the reference refuses streaming too),
//! portable f32.
//!
//! Structure: the byte-level text intake is family-local
//! ([`text`]), the denoiser is [`dit`], the euler samplers are
//! [`sampler`], and the PCA/codec bridge is [`audio`]. The waveform
//! path decodes through the ported Fish S1 DAC
//! ([`crate::codec::fish_s1_dac`]) continuous-latent API plus the
//! checkpoint's PCA state; both are separate downloads from the DiT
//! checkpoint (the reference `post_load_hook` fetches the codec from
//! `fish_codec_repo`), so [`Model::open`] loads the DiT and PCA only
//! and the caller installs the codec with [`Model::set_codec`].
//! [`Model::generate`] refuses to run without them, like the
//! reference.

pub mod audio;
pub mod config;
pub mod dit;
pub mod sampler;
pub mod text;

use crate::codec::fish_s1_dac::FishS1Dac;
use crate::lm::{Bias, Weights};
use crate::nn::Linear;
use crate::{Result, SpeechError};

use config::{EchoDiTConfig, ModelConfig};
use dit::{
    DecoderBlock, EchoDiT, EncoderBlock, JointAttention, LowRankAdaLN, Mlp, RmsNorm, SelfAttention,
    SpeakerEncoder, TextEncoder,
};
use sampler::{GaussianNoise, SamplerParams};

/// One generated utterance.
#[derive(Debug, Clone)]
pub struct Generated {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    /// Byte-level text token count (BOS included).
    pub token_count: usize,
    /// The flattening point found on the latent (latent frames kept).
    pub flattening_point: usize,
    /// Latent frames the sampler produced.
    pub latent_frames: usize,
}

/// The generate-path controls: the sampler config (defaults come from
/// the checkpoint), the optional blockwise split, and the optional
/// continuation latent for the blockwise path.
#[derive(Debug, Clone)]
pub struct GenerateControls {
    pub sampler: SamplerParams,
    pub sequence_length: Option<usize>,
    pub block_sizes: Option<Vec<usize>>,
    /// `(rows x latent, rows)` prefix for the blockwise path.
    pub continuation_latent: Option<(Vec<f32>, usize)>,
    /// Overrides the reference `640 * audio_downsample_factor` speaker
    /// encode chunk. The real checkpoints never need this; the tiny
    /// fixture codecs have a much shorter frame length and would blow
    /// the speaker encoder's causal mask up to the padded length.
    pub speaker_audio_chunk_size: Option<usize>,
}

impl GenerateControls {
    /// The checkpoint sampler defaults.
    pub fn from_config(config: &ModelConfig) -> Self {
        let sampler = &config.sampler;
        GenerateControls {
            sampler: SamplerParams {
                num_steps: sampler.num_steps,
                cfg_scale_text: sampler.cfg_scale_text,
                cfg_scale_speaker: sampler.cfg_scale_speaker,
                cfg_min_t: sampler.cfg_min_t,
                cfg_max_t: sampler.cfg_max_t,
                truncation_factor: sampler.truncation_factor,
                rescale_k: sampler.rescale_k,
                rescale_sigma: sampler.rescale_sigma,
                speaker_kv_scale: sampler.speaker_kv_scale,
                speaker_kv_max_layers: sampler.speaker_kv_max_layers,
                speaker_kv_min_t: sampler.speaker_kv_min_t,
            },
            sequence_length: Some(sampler.sequence_length),
            block_sizes: None,
            continuation_latent: None,
            speaker_audio_chunk_size: None,
        }
    }
}

/// The loaded family model.
pub struct Model {
    pub config: ModelConfig,
    dit: EchoDiT,
    codec: Option<FishS1Dac>,
    pca: Option<audio::PcaState>,
}

impl Model {
    /// Opens a checkpoint directory: `config.json`, `model.safetensors`
    /// through the reference `sanitize` naming (`model.` prefix,
    /// `cond_module.N` renamed to `cond_module.layers.N`, the PCA and
    /// blockwise-skip tensors ignored), plus the PCA state file from
    /// `pca_filename` when present. The Fish codec is a separate
    /// download; install it with [`Self::set_codec`].
    pub fn open(dir: &std::path::Path) -> Result<Self> {
        let config_value = crate::quant::read_json(&dir.join("config.json"))?;
        Self::open_with_config(dir, config_value)
    }

    /// [`Self::open`] with a caller-supplied config document (the
    /// checkpoint weights are unchanged; the config decides which
    /// modules load). This is how a `delete_blockwise_modules` checkpoint
    /// shares a weight file with the full variant.
    pub fn open_with_config(
        dir: &std::path::Path,
        config_value: serde_json::Value,
    ) -> Result<Self> {
        let config = ModelConfig::from_json(&config_value)?;
        let weights = Weights::open(dir)?.with_prefix("model.");
        let dit = load_dit(&weights, &config.dit)?;
        let pca_path = dir.join(&config.pca_filename);
        let pca = if pca_path.exists() {
            Some(audio::PcaState::open(&pca_path)?)
        } else {
            None
        };
        Ok(Model {
            config,
            dit,
            codec: None,
            pca,
        })
    }

    /// Installs the Fish S1 DAC (the reference `post_load_hook`'s
    /// `FishS1DAC.from_pretrained`).
    pub fn set_codec(&mut self, codec: FishS1Dac) {
        self.codec = Some(codec);
    }

    /// Whether the codec and PCA are installed (generate needs both).
    pub fn waveform_ready(&self) -> bool {
        self.codec.is_some() && self.pca.is_some()
    }

    pub fn sample_rate(&self) -> u32 {
        self.config.sample_rate
    }

    pub fn latent_size(&self) -> usize {
        self.config.dit.latent_size
    }

    /// `generate_latents`: run the euler sampler for the whole text.
    /// `speaker_latent` is `[rows, latent]` with a per-sample mask; the
    /// default is a zero latent of one patch with an all-false mask.
    pub fn generate_latents(
        &self,
        text: &str,
        speaker_latent: Option<(&[f32], usize, &[bool])>,
        controls: &GenerateControls,
        noise: &mut dyn GaussianNoise,
    ) -> Result<Vec<f32>> {
        let batch = text::get_text_input_ids_and_mask(
            &[text],
            Some(self.config.max_text_length),
            self.config.normalize_text,
            false,
        );
        let (speaker, speaker_rows, speaker_mask) = match speaker_latent {
            Some((latent, rows, mask)) => (latent.to_vec(), rows, mask.to_vec()),
            None => {
                let patch = self.config.dit.speaker_patch_size;
                (
                    vec![0.0f32; patch * self.config.dit.latent_size],
                    patch,
                    vec![false; patch],
                )
            }
        };
        let params = &controls.sampler;
        match &controls.block_sizes {
            None => sampler::sample_euler_cfg_independent_guidances(
                &self.dit,
                &speaker,
                speaker_rows,
                &speaker_mask,
                &batch.ids,
                &batch.mask,
                params,
                controls.sequence_length.unwrap_or(640),
                noise,
            ),
            Some(block_sizes) => {
                if self.config.delete_blockwise_modules {
                    return Err(SpeechError::BadConfig {
                        field: "delete_blockwise_modules".into(),
                        why: "blockwise generation requires latent-prefix modules".into(),
                    });
                }
                let continuation = controls
                    .continuation_latent
                    .as_ref()
                    .map(|(latent, rows)| (latent.as_slice(), *rows));
                sampler::sample_blockwise_euler_cfg_independent_guidances(
                    &self.dit,
                    &speaker,
                    speaker_rows,
                    &speaker_mask,
                    &batch.ids,
                    &batch.mask,
                    block_sizes,
                    params,
                    continuation,
                    noise,
                )
            }
        }
    }

    /// The reference `generate`: latents, PCA decode, codec decode,
    /// flattening-point crop. Non-streaming; refuses a missing codec or
    /// PCA state.
    pub fn generate(
        &self,
        text: &str,
        ref_audio: Option<&[f32]>,
        controls: &GenerateControls,
        noise: &mut dyn GaussianNoise,
    ) -> Result<Generated> {
        let codec = self.codec.as_ref().ok_or_else(|| SpeechError::Input {
            why: "Echo-TTS requires the Fish S1 DAC loaded; call set_codec".into(),
        })?;
        let pca = self.pca.as_ref().ok_or_else(|| SpeechError::Input {
            why: "Echo-TTS requires the PCA state loaded from the checkpoint".into(),
        })?;
        let batch = text::get_text_input_ids_and_mask(
            &[text],
            Some(self.config.max_text_length),
            self.config.normalize_text,
            false,
        );
        let token_count = batch.width;

        let (speaker, speaker_rows, speaker_mask) = match ref_audio {
            Some(audio) => {
                if self.config.dit.speaker_patch_size == 0 {
                    return Err(SpeechError::BadConfig {
                        field: "dit.speaker_patch_size".into(),
                        why: "must be positive".into(),
                    });
                }
                audio::get_speaker_latent_and_mask(
                    codec,
                    pca,
                    audio,
                    self.config.max_speaker_latent_length,
                    self.config.audio_downsample_factor,
                    controls
                        .speaker_audio_chunk_size
                        .unwrap_or(self.config.audio_downsample_factor * 640),
                    false,
                    self.config.dit.speaker_patch_size,
                )?
            }
            None => {
                let patch = self.config.dit.speaker_patch_size;
                (
                    vec![0.0f32; patch * self.config.dit.latent_size],
                    patch,
                    vec![false; patch],
                )
            }
        };

        let latent_out = self.generate_latents(
            text,
            Some((&speaker, speaker_rows, &speaker_mask)),
            controls,
            noise,
        )?;
        let latent_frames = latent_out.len() / self.config.dit.latent_size;
        let wav = audio::ae_decode(codec, pca, &latent_out, latent_frames)?;
        let flattening_point = audio::find_flattening_point(
            &latent_out,
            latent_frames,
            self.config.dit.latent_size,
            0.0,
            20,
            0.05,
        );
        let samples = audio::crop_audio_to_flattening_point(&wav, flattening_point);
        Ok(Generated {
            samples,
            sample_rate: self.config.sample_rate,
            token_count,
            flattening_point,
            latent_frames,
        })
    }
}

/// Loads the denoiser under the `model.` prefix.
fn load_dit(weights: &Weights, config: &EchoDiTConfig) -> Result<EchoDiT> {
    let latent = config.latent_size;
    let model = config.model_size;
    let heads = config.num_heads;
    let head_dim = model / heads;
    let text_model = config.text_model_size;
    let text_heads = config.text_num_heads;
    let text_head_dim = text_model / text_heads;
    let speaker_model = config.speaker_model_size;
    let speaker_heads = config.speaker_num_heads;
    let speaker_head_dim = speaker_model / speaker_heads;
    let eps = config.norm_eps;

    // Text encoder.
    let text_embedding = weights.embedding(
        "text_encoder.text_embedding",
        config.text_vocab_size,
        text_model,
    )?;
    let mut text_blocks = Vec::with_capacity(config.text_num_layers);
    for index in 0..config.text_num_layers {
        text_blocks.push(load_encoder_block(
            weights,
            &format!("text_encoder.blocks.{index}"),
            text_model,
            text_heads,
            text_head_dim,
            config.text_intermediate_size,
            eps,
            false,
        )?);
    }
    let text_encoder = TextEncoder {
        embedding: text_embedding,
        vocab_size: config.text_vocab_size,
        model_size: text_model,
        blocks: text_blocks,
        head_dim: text_head_dim,
    };

    // Speaker encoder.
    let speaker_encoder = load_speaker_encoder(
        weights,
        "speaker_encoder",
        latent,
        config.speaker_patch_size,
        speaker_model,
        config.speaker_num_layers,
        speaker_heads,
        speaker_head_dim,
        config.speaker_intermediate_size,
        eps,
    )?;
    let latent_encoder = if config.enable_blockwise_modules {
        Some(load_speaker_encoder(
            weights,
            "latent_encoder",
            latent,
            config.speaker_patch_size,
            speaker_model,
            config.speaker_num_layers,
            speaker_heads,
            speaker_head_dim,
            config.speaker_intermediate_size,
            eps,
        )?)
    } else {
        None
    };

    let latent_norm = if config.enable_blockwise_modules {
        Some(RmsNorm::new(
            weights.tensor("latent_norm", &[speaker_model])?,
            1,
            speaker_model,
            eps,
        ))
    } else {
        None
    };
    let text_norm = RmsNorm::new(
        weights.tensor("text_norm", &[text_model])?,
        1,
        text_model,
        eps,
    );
    let speaker_norm = RmsNorm::new(
        weights.tensor("speaker_norm", &[speaker_model])?,
        1,
        speaker_model,
        eps,
    );

    // cond_module: Linear, SiLU, Linear, SiLU, Linear; the Sequential
    // indices count the parameterless SiLU layers (0, 2, 4).
    let cond_module = [
        load_cond_linear(weights, 0, config.timestep_embed_size, model)?,
        load_cond_linear(weights, 2, model, model)?,
        load_cond_linear(weights, 4, model, model * 3)?,
    ];

    let in_proj = weights.linear("in_proj", latent, model, Bias::Required)?;
    let mut blocks = Vec::with_capacity(config.num_layers);
    for index in 0..config.num_layers {
        let base = format!("blocks.{index}");
        let attention = JointAttention {
            wq: weights.linear(&format!("{base}.attention.wq"), model, model, Bias::None)?,
            wk: weights.linear(&format!("{base}.attention.wk"), model, model, Bias::None)?,
            wv: weights.linear(&format!("{base}.attention.wv"), model, model, Bias::None)?,
            gate: weights.linear(&format!("{base}.attention.gate"), model, model, Bias::None)?,
            wo: weights.linear(&format!("{base}.attention.wo"), model, model, Bias::None)?,
            wk_text: weights.linear(
                &format!("{base}.attention.wk_text"),
                text_model,
                model,
                Bias::None,
            )?,
            wv_text: weights.linear(
                &format!("{base}.attention.wv_text"),
                text_model,
                model,
                Bias::None,
            )?,
            wk_speaker: weights.linear(
                &format!("{base}.attention.wk_speaker"),
                speaker_model,
                model,
                Bias::None,
            )?,
            wv_speaker: weights.linear(
                &format!("{base}.attention.wv_speaker"),
                speaker_model,
                model,
                Bias::None,
            )?,
            wk_latent: if config.enable_blockwise_modules {
                Some(weights.linear(
                    &format!("{base}.attention.wk_latent"),
                    speaker_model,
                    model,
                    Bias::None,
                )?)
            } else {
                None
            },
            wv_latent: if config.enable_blockwise_modules {
                Some(weights.linear(
                    &format!("{base}.attention.wv_latent"),
                    speaker_model,
                    model,
                    Bias::None,
                )?)
            } else {
                None
            },
            q_norm: load_headed_norm(
                weights,
                &format!("{base}.attention.q_norm"),
                heads,
                head_dim,
                eps,
            )?,
            k_norm: load_headed_norm(
                weights,
                &format!("{base}.attention.k_norm"),
                heads,
                head_dim,
                eps,
            )?,
            heads,
            head_dim,
            speaker_patch_size: config.speaker_patch_size,
        };
        let mlp = load_mlp(
            weights,
            &format!("{base}.mlp"),
            model,
            config.intermediate_size,
        )?;
        let attention_adaln = load_adaln(
            weights,
            &format!("{base}.attention_adaln"),
            model,
            config.adaln_rank,
            eps,
        )?;
        let mlp_adaln = load_adaln(
            weights,
            &format!("{base}.mlp_adaln"),
            model,
            config.adaln_rank,
            eps,
        )?;
        blocks.push(DecoderBlock {
            attention,
            mlp,
            attention_adaln,
            mlp_adaln,
        });
    }

    let out_norm = RmsNorm::new(weights.tensor("out_norm", &[model])?, 1, model, eps);
    let out_proj = weights.linear("out_proj", model, latent, Bias::Required)?;

    Ok(EchoDiT::from_parts(
        config.clone(),
        text_encoder,
        speaker_encoder,
        latent_encoder,
        latent_norm,
        text_norm,
        speaker_norm,
        cond_module,
        in_proj,
        blocks,
        out_norm,
        out_proj,
    ))
}

fn load_speaker_encoder(
    weights: &Weights,
    base: &str,
    latent: usize,
    patch: usize,
    model: usize,
    num_layers: usize,
    heads: usize,
    head_dim: usize,
    intermediate: usize,
    eps: f32,
) -> Result<SpeakerEncoder> {
    let in_proj = weights.linear(
        &format!("{base}.in_proj"),
        latent * patch,
        model,
        Bias::Required,
    )?;
    let mut blocks = Vec::with_capacity(num_layers);
    for index in 0..num_layers {
        blocks.push(load_encoder_block(
            weights,
            &format!("{base}.blocks.{index}"),
            model,
            heads,
            head_dim,
            intermediate,
            eps,
            true,
        )?);
    }
    Ok(SpeakerEncoder {
        in_proj,
        blocks,
        patch_size: patch,
        head_dim,
    })
}

fn load_encoder_block(
    weights: &Weights,
    base: &str,
    model: usize,
    heads: usize,
    head_dim: usize,
    intermediate: usize,
    eps: f32,
    is_causal: bool,
) -> Result<EncoderBlock> {
    let attention = SelfAttention {
        wq: weights.linear(&format!("{base}.attention.wq"), model, model, Bias::None)?,
        wk: weights.linear(&format!("{base}.attention.wk"), model, model, Bias::None)?,
        wv: weights.linear(&format!("{base}.attention.wv"), model, model, Bias::None)?,
        wo: weights.linear(&format!("{base}.attention.wo"), model, model, Bias::None)?,
        gate: weights.linear(&format!("{base}.attention.gate"), model, model, Bias::None)?,
        q_norm: load_headed_norm(
            weights,
            &format!("{base}.attention.q_norm"),
            heads,
            head_dim,
            eps,
        )?,
        k_norm: load_headed_norm(
            weights,
            &format!("{base}.attention.k_norm"),
            heads,
            head_dim,
            eps,
        )?,
        heads,
        head_dim,
        is_causal,
    };
    Ok(EncoderBlock {
        attention,
        mlp: load_mlp(weights, &format!("{base}.mlp"), model, intermediate)?,
        attention_norm: RmsNorm::new(
            weights.tensor(&format!("{base}.attention_norm"), &[model])?,
            1,
            model,
            eps,
        ),
        mlp_norm: RmsNorm::new(
            weights.tensor(&format!("{base}.mlp_norm"), &[model])?,
            1,
            model,
            eps,
        ),
    })
}

fn load_mlp(weights: &Weights, base: &str, model: usize, intermediate: usize) -> Result<Mlp> {
    Ok(Mlp {
        w1: weights.linear(&format!("{base}.w1"), model, intermediate, Bias::None)?,
        w2: weights.linear(&format!("{base}.w2"), intermediate, model, Bias::None)?,
        w3: weights.linear(&format!("{base}.w3"), model, intermediate, Bias::None)?,
    })
}

fn load_adaln(
    weights: &Weights,
    base: &str,
    model: usize,
    rank: usize,
    eps: f32,
) -> Result<LowRankAdaLN> {
    Ok(LowRankAdaLN {
        shift_down: weights.linear(&format!("{base}.shift_down"), model, rank, Bias::None)?,
        scale_down: weights.linear(&format!("{base}.scale_down"), model, rank, Bias::None)?,
        gate_down: weights.linear(&format!("{base}.gate_down"), model, rank, Bias::None)?,
        shift_up: weights.linear(&format!("{base}.shift_up"), rank, model, Bias::Required)?,
        scale_up: weights.linear(&format!("{base}.scale_up"), rank, model, Bias::Required)?,
        gate_up: weights.linear(&format!("{base}.gate_up"), rank, model, Bias::Required)?,
        model_size: model,
        eps,
    })
}

/// The per-head q/k norm weights store as `[heads, head_dim]`.
fn load_headed_norm(
    weights: &Weights,
    name: &str,
    heads: usize,
    head_dim: usize,
    eps: f32,
) -> Result<RmsNorm> {
    Ok(RmsNorm::new(
        weights.tensor(name, &[heads, head_dim])?,
        heads,
        head_dim,
        eps,
    ))
}

/// The reference `sanitize` rename: checkpoints may carry
/// `cond_module.N.weight` (renamed to `cond_module.layers.N`) or the
/// already-renamed form.
fn load_cond_linear(
    weights: &Weights,
    index: usize,
    input: usize,
    output: usize,
) -> Result<Linear> {
    let layered = format!("cond_module.layers.{index}");
    if weights.contains(&layered) {
        return weights.linear(&layered, input, output, Bias::None);
    }
    weights.linear(&format!("cond_module.{index}"), input, output, Bias::None)
}

/// Re-exported for the tests' rotary-table checks.
pub use dit::{precompute_freqs_cis, timestep_embedding};

#[cfg(test)]
mod tests;
