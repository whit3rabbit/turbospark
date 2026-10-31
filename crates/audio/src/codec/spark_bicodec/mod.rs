//! Spark-TTS BiCodec decoder (semantic + global tokens to waveform).
//!
//! Reference: `mlx_audio/tts/models/spark/` (bicodec.py, modules/) at
//! mlx-audio
//! [`e1b19b9054bf163f5d812221a54fcc346f1890e9`](https://github.com/Blaizzy/mlx-audio/tree/e1b19b9054bf163f5d812221a54fcc346f1890e9/mlx_audio/tts/models/spark).
//! This port covers the decode direction the TTS family needs,
//! `BiCodec.detokenize`: the factorized vector-quantizer code lookup
//! through a kernel-1 weight-norm projection, the FSQ global-token
//! speaker embedding, a Vocos-backbone prenet conditioned on that
//! speaker vector, and the DAC-style WaveGenerator. The encode direction
//! (Wav2Vec2 features, the BiCodec encoder, the ECAPA/Perceiver speaker
//! path) is an open item recorded in the family README.
//!
//! Checkpoint notes: the published module tree keeps the MLX conv
//! layouts (`[out, K, in]`), so conv loads here go through the shared
//! MLX-layout loaders. The reference sanitize chain (torch-layout
//! `weight_v` transposes, snake alpha `[1, C, 1]` -> `[1, 1, C]`, the
//! WaveGenerator `block` -> `block.layers` rename, and the transposed
//! conv `transpose(1, 2, 0)`) is applied to raw torch-layout tensors
//! when the MLX layout is absent, so both the converted-for-mlx
//! checkpoints and raw PyTorch-conversion exports load.

use std::path::Path;

use turbospark_model_io::safetensors::SafetensorsFile;

use crate::codec::conv::{self, Conv1d, ConvTranspose1d, WnFold};
use crate::codec::dac::add_cropped_residual;
use crate::codec::vocos::{self, ConvLayout, VocosBackboneConfig};
use crate::codec::wnconv::{load_f32_shaped, snake1d};
use crate::nn::Linear;
use crate::{Result, SpeechError};

/// Pinned mlx-audio commit this port was transcribed from.
pub const REFERENCE_COMMIT: &str = "e1b19b9054bf163f5d812221a54fcc346f1890e9";

/// `audio_tokenizer.quantizer` geometry.
#[derive(Debug, Clone)]
pub struct QuantizerConfig {
    pub input_dim: usize,
    pub codebook_size: usize,
    pub codebook_dim: usize,
}

/// `audio_tokenizer.speaker_encoder` geometry.
#[derive(Debug, Clone)]
pub struct SpeakerEncoderConfig {
    pub out_dim: usize,
    pub latent_dim: usize,
    pub token_num: usize,
    pub fsq_levels: Vec<usize>,
    pub fsq_num_quantizers: usize,
}

/// `audio_tokenizer.prenet` geometry (the feat `Decoder`; the postnet
/// shares the shape and stays on the unported encode path).
#[derive(Debug, Clone)]
pub struct FeatDecoderConfig {
    pub input_channels: usize,
    pub vocos_dim: usize,
    pub vocos_intermediate_dim: usize,
    pub vocos_num_layers: usize,
    pub out_channels: usize,
    pub condition_dim: Option<usize>,
    pub sample_ratios: Vec<usize>,
    pub use_tanh_at_final: bool,
}

/// `audio_tokenizer.decoder` geometry (the WaveGenerator).
#[derive(Debug, Clone)]
pub struct WaveGeneratorConfig {
    pub input_channel: usize,
    pub channels: usize,
    pub rates: Vec<usize>,
    pub kernel_sizes: Vec<usize>,
    pub d_out: usize,
}

/// Parsed BiCodec `config.yaml` (the `audio_tokenizer` subset the
/// decode direction reads).
#[derive(Debug, Clone)]
pub struct BiCodecConfig {
    pub quantizer: QuantizerConfig,
    pub speaker_encoder: SpeakerEncoderConfig,
    pub prenet: FeatDecoderConfig,
    pub decoder: WaveGeneratorConfig,
}

impl BiCodecConfig {
    /// Parses the `config.yaml` text. The `mel_params`, `encoder`, and
    /// `postnet` sections belong to the unported encode direction and
    /// are not read.
    pub fn from_yaml(text: &str) -> Result<Self> {
        let root = flow_lists(vocos::parse_yaml_mapping(text)?);
        let section = root
            .get("audio_tokenizer")
            .ok_or_else(|| SpeechError::BadConfig {
                field: "audio_tokenizer".to_string(),
                why: "missing audio_tokenizer section in config.yaml".to_string(),
            })?;
        let quantizer = section
            .get("quantizer")
            .ok_or_else(|| SpeechError::BadConfig {
                field: "audio_tokenizer.quantizer".to_string(),
                why: "missing section in config.yaml".to_string(),
            })?;
        let speaker = section
            .get("speaker_encoder")
            .ok_or_else(|| SpeechError::BadConfig {
                field: "audio_tokenizer.speaker_encoder".to_string(),
                why: "missing section in config.yaml".to_string(),
            })?;
        let prenet = section
            .get("prenet")
            .ok_or_else(|| SpeechError::BadConfig {
                field: "audio_tokenizer.prenet".to_string(),
                why: "missing section in config.yaml".to_string(),
            })?;
        let decoder = section
            .get("decoder")
            .ok_or_else(|| SpeechError::BadConfig {
                field: "audio_tokenizer.decoder".to_string(),
                why: "missing section in config.yaml".to_string(),
            })?;
        let uint = |value: &serde_json::Value, key: &str| -> usize {
            value
                .get(key)
                .and_then(serde_json::Value::as_u64)
                .unwrap_or_default() as usize
        };
        let uint_list = |value: &serde_json::Value, key: &str| -> Vec<usize> {
            value
                .get(key)
                .and_then(serde_json::Value::as_array)
                .map(|entries| {
                    entries
                        .iter()
                        .filter_map(|v| v.as_u64().map(|n| n as usize))
                        .collect()
                })
                .unwrap_or_default()
        };
        let condition_dim = uint(prenet, "condition_dim");
        let ratios = {
            let ratios = uint_list(prenet, "sample_ratios");
            if ratios.is_empty() {
                vec![1, 1]
            } else {
                ratios
            }
        };
        let config = BiCodecConfig {
            quantizer: QuantizerConfig {
                input_dim: uint(quantizer, "input_dim"),
                codebook_size: uint(quantizer, "codebook_size"),
                codebook_dim: uint(quantizer, "codebook_dim"),
            },
            speaker_encoder: SpeakerEncoderConfig {
                out_dim: uint(speaker, "out_dim"),
                latent_dim: uint(speaker, "latent_dim"),
                token_num: uint(speaker, "token_num"),
                fsq_levels: uint_list(speaker, "fsq_levels"),
                fsq_num_quantizers: uint(speaker, "fsq_num_quantizers"),
            },
            prenet: FeatDecoderConfig {
                input_channels: uint(prenet, "input_channels"),
                vocos_dim: uint(prenet, "vocos_dim"),
                vocos_intermediate_dim: uint(prenet, "vocos_intermediate_dim"),
                vocos_num_layers: uint(prenet, "vocos_num_layers"),
                out_channels: uint(prenet, "out_channels"),
                condition_dim: if condition_dim == 0 {
                    None
                } else {
                    Some(condition_dim)
                },
                sample_ratios: ratios,
                use_tanh_at_final: prenet
                    .get("use_tanh_at_final")
                    .and_then(serde_json::Value::as_bool)
                    .unwrap_or(false),
            },
            decoder: WaveGeneratorConfig {
                input_channel: uint(decoder, "input_channel"),
                channels: uint(decoder, "channels"),
                rates: uint_list(decoder, "rates"),
                kernel_sizes: uint_list(decoder, "kernel_sizes"),
                d_out: uint(decoder, "d_out").max(1),
            },
        };
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> Result<()> {
        if self.quantizer.codebook_size == 0 || self.quantizer.codebook_dim == 0 {
            return Err(SpeechError::BadConfig {
                field: "quantizer".into(),
                why: "codebook_size and codebook_dim must be positive".into(),
            });
        }
        if self.speaker_encoder.fsq_levels.is_empty() {
            return Err(SpeechError::BadConfig {
                field: "speaker_encoder".into(),
                why: "fsq_levels must not be empty".into(),
            });
        }
        if self.speaker_encoder.token_num == 0 || self.speaker_encoder.latent_dim == 0 {
            return Err(SpeechError::BadConfig {
                field: "speaker_encoder".into(),
                why: "token_num and latent_dim must be positive".into(),
            });
        }
        if self.decoder.rates.len() != self.decoder.kernel_sizes.len() {
            return Err(SpeechError::BadConfig {
                field: "decoder".into(),
                why: format!(
                    "rates ({}) and kernel_sizes ({}) lengths differ",
                    self.decoder.rates.len(),
                    self.decoder.kernel_sizes.len()
                ),
            });
        }
        if self.decoder.input_channel != self.prenet.out_channels {
            return Err(SpeechError::BadConfig {
                field: "decoder".into(),
                why: format!(
                    "input_channel {} does not match the prenet out_channels {}",
                    self.decoder.input_channel, self.prenet.out_channels
                ),
            });
        }
        // The detokenize broadcast x + d_vector[..., None] needs the
        // prenet output width to carry the speaker vector. The published
        // 0.5B checkpoint uses 1024 for both; the mlx-audio dataclass
        // defaults (256/512) cannot broadcast and are rejected here.
        if self.speaker_encoder.out_dim != self.prenet.out_channels {
            return Err(SpeechError::BadConfig {
                field: "prenet".into(),
                why: format!(
                    "out_channels {} must equal the speaker out_dim {} for the \
                     detokenize condition broadcast",
                    self.prenet.out_channels, self.speaker_encoder.out_dim
                ),
            });
        }
        if self.prenet.input_channels != self.quantizer.input_dim {
            return Err(SpeechError::BadConfig {
                field: "prenet".into(),
                why: format!(
                    "input_channels {} does not match the quantizer input_dim {}",
                    self.prenet.input_channels, self.quantizer.input_dim
                ),
            });
        }
        if self.prenet.sample_ratios.iter().any(|&r| r != 1) {
            return Err(SpeechError::Unsupported {
                why: "prenet sample_ratios above 1 ride the unported encode \
                      direction; the published decode path uses [1, 1]"
                    .to_string(),
            });
        }
        Ok(())
    }
}

/// A channel-major linear helper: `[out, frames] = weight @ x + bias`
/// for row-major `weight [out, in]` and channel-major `x [in, frames]`.
fn linear_cm(
    weight: &[f32],
    bias: Option<&[f32]>,
    in_ch: usize,
    out_ch: usize,
    x: &[f32],
    frames: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; out_ch * frames];
    for (o, row) in out.chunks_mut(frames).enumerate() {
        let w = &weight[o * in_ch..(o + 1) * in_ch];
        for (f, slot) in row.iter_mut().enumerate() {
            let mut acc = bias.map_or(0.0, |b| b[o]);
            for (c, &wv) in w.iter().enumerate() {
                acc += wv * x[c * frames + f];
            }
            *slot = acc;
        }
    }
    out
}

/// A weight-norm conv1d in either checkpoint layout: MLX
/// `[out, K, in/groups]` (the published module tree) or torch
/// `[out, in/groups, K]` (a raw conversion export). The fold runs per
/// output channel on the contiguous row either way.
#[allow(clippy::too_many_arguments)]
fn load_wn_conv1d(
    file: &SafetensorsFile,
    prefix: &str,
    in_ch: usize,
    out_ch: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> Result<Conv1d> {
    if let Ok(loaded) = Conv1d::load(
        file, prefix, in_ch, out_ch, kernel, stride, padding, dilation, 1,
    ) {
        return Ok(loaded);
    }
    let in_g = in_ch;
    let mut v = load_f32_shaped(file, &format!("{prefix}.weight_v"), &[out_ch, in_g, kernel])?;
    let g = load_f32_shaped(file, &format!("{prefix}.weight_g"), &[out_ch])?;
    conv::wn_fold_in_place(&mut v, &g, in_g * kernel, WnFold::Div);
    let bias = conv::load_bias(
        file,
        &format!("{prefix}.bias"),
        out_ch,
        conv::BiasMode::Optional,
    )?;
    Ok(Conv1d {
        in_ch,
        out_ch,
        kernel,
        stride,
        padding,
        dilation,
        groups: 1,
        weight: v,
        bias,
    })
}

/// A weight-norm transposed conv in either checkpoint layout: MLX
/// `[out, K, in]` (the published tree) or torch `[in, out, K]`.
fn load_wn_convt(
    file: &SafetensorsFile,
    prefix: &str,
    in_ch: usize,
    out_ch: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
) -> Result<ConvTranspose1d> {
    if let Ok(loaded) =
        ConvTranspose1d::load_out_first(file, prefix, in_ch, out_ch, kernel, stride, padding)
    {
        return Ok(loaded);
    }
    let mut v = load_f32_shaped(
        file,
        &format!("{prefix}.weight_v"),
        &[in_ch, out_ch, kernel],
    )?;
    let g = load_f32_shaped(file, &format!("{prefix}.weight_g"), &[in_ch])?;
    // The torch row per input channel is contiguous; fold in place.
    conv::wn_fold_in_place(&mut v, &g, out_ch * kernel, WnFold::Div);
    let bias = conv::load_bias(
        file,
        &format!("{prefix}.bias"),
        out_ch,
        conv::BiasMode::Optional,
    )?;
    Ok(ConvTranspose1d {
        in_ch,
        out_ch,
        kernel,
        stride,
        padding,
        output_padding: 0,
        groups: 1,
        weight: v,
        bias,
    })
}

/// Snake1d alpha in either layout: `[1, 1, C]` (the reference module
/// holds this) or `[1, C, 1]` torch.
fn load_alpha(file: &SafetensorsFile, name: &str, channels: usize) -> Result<Vec<f32>> {
    if let Ok(alpha) = load_f32_shaped(file, name, &[1, 1, channels]) {
        return Ok(alpha);
    }
    load_f32_shaped(file, name, &[1, channels, 1])
}

/// The raw checkpoint keeps `block.` names; the loaded module tree uses
/// `block.layers.`. Try the module-tree name first, then the raw one
/// (the reference sanitize rename applied to every occurrence).
fn wave_name(name: &str) -> String {
    name.replace("block.layers", "block")
}

/// The factorized vector quantizer decode side (`detokenize`): the
/// codebook lookup plus the kernel-1 weight-norm out projection. The
/// in projection only feeds the unported encode path.
struct FactorizedVectorQuantize {
    codebook: Vec<f32>,
    out_project: Conv1d,
    input_dim: usize,
    codebook_size: usize,
    codebook_dim: usize,
}

impl FactorizedVectorQuantize {
    fn load(file: &SafetensorsFile, config: &QuantizerConfig) -> Result<Self> {
        let prefix = "quantizer";
        Ok(Self {
            codebook: file.load_as_f32(&format!("{prefix}.codebook.weight"))?,
            out_project: load_wn_conv1d(
                file,
                &format!("{prefix}.out_project"),
                config.codebook_dim,
                config.input_dim,
                1,
                1,
                0,
                1,
            )?,
            input_dim: config.input_dim,
            codebook_size: config.codebook_size,
            codebook_dim: config.codebook_dim,
        })
    }

    /// `decode_code` + `out_project` for `[frames]` indices: each frame
    /// contributes `out_project_linear @ codebook[idx] + b`. The
    /// reference routes the lookup through length-1 MLX conv calls, so
    /// the kernel-1 projection is per frame regardless.
    fn detokenize(&self, indices: &[i32]) -> Result<Vec<f32>> {
        if indices.is_empty() {
            // The reference empty-input branch.
            return Ok(vec![0.0; self.input_dim]);
        }
        let mut zq = vec![0.0f32; self.codebook_dim * indices.len()];
        for (frame, &index) in indices.iter().enumerate() {
            let row = usize::try_from(index).map_err(|_| SpeechError::Input {
                why: format!("negative semantic token {index}"),
            })?;
            if row >= self.codebook_size {
                return Err(SpeechError::Input {
                    why: format!(
                        "semantic token {row} outside the {}-entry codebook",
                        self.codebook_size
                    ),
                });
            }
            let source = &self.codebook[row * self.codebook_dim..(row + 1) * self.codebook_dim];
            for (d, &value) in source.iter().enumerate() {
                zq[d * indices.len() + frame] = value;
            }
        }
        // Channel-major [codebook_dim, frames] -> [input_dim, frames]
        // through the kernel-1 conv.
        Ok(self.out_project.forward(&zq))
    }
}

/// The FSQ global-token speaker path (`SpeakerEncoder.detokenize`): the
/// implicit FSQ codebook is derived from the levels (no learned
/// weights), then one linear re-projects and one linear flattens to the
/// d-vector.
struct SpeakerDecoder {
    /// `[codebook_size, codebook_dim]` from `_indices_to_codes`.
    implicit_codebook: Vec<f32>,
    codebook_dim: usize,
    codebook_size: usize,
    project_out: Linear,
    project: Linear,
    token_num: usize,
}

impl SpeakerDecoder {
    fn load(file: &SafetensorsFile, config: &SpeakerEncoderConfig) -> Result<Self> {
        let prefix = "speaker_encoder.quantizer";
        let levels = &config.fsq_levels;
        let codebook_dim = levels.len();
        // basis = cumprod([1, levels[:-1]]); half_width = levels // 2.
        let mut basis = vec![1usize; codebook_dim];
        for d in 1..codebook_dim {
            basis[d] = basis[d - 1] * levels[d - 1];
        }
        let codebook_size: usize = levels.iter().product();
        let half: Vec<f32> = levels.iter().map(|l| (*l / 2) as f32).collect();
        let mut implicit = vec![0.0f32; codebook_size * codebook_dim];
        for index in 0..codebook_size {
            for (d, &base) in basis.iter().enumerate() {
                let level_index = (index / base) % levels[d];
                implicit[index * codebook_dim + d] = (level_index as f32 - half[d]) / half[d];
            }
        }
        // The decode path sums ONE implicit codebook (the per-quantizer
        // scales are (levels - 1) ** 0 = 1 for the first quantizer).
        if config.fsq_num_quantizers != 1 {
            return Err(SpeechError::Unsupported {
                why: format!(
                    "fsq_num_quantizers {} is not 1; the decode path sums one \
                     implicit codebook",
                    config.fsq_num_quantizers
                ),
            });
        }
        let latent = config.latent_dim;
        let linear = |weight_name: String, input: usize, output: usize| -> Result<Linear> {
            Ok(Linear::new(
                file.load_as_f32(&weight_name)?,
                Some(file.load_as_f32(&format!(
                    "{}.bias",
                    weight_name.strip_suffix(".weight").unwrap_or(&weight_name)
                ))?),
                input,
                output,
            ))
        };
        Ok(Self {
            implicit_codebook: implicit,
            codebook_dim,
            codebook_size,
            // project_out lives under the quantizer; the flatten
            // projection is a direct SpeakerEncoder member.
            project_out: linear(format!("{prefix}.project_out.weight"), codebook_dim, latent)?,
            project: linear(
                "speaker_encoder.project.weight".to_string(),
                latent * config.token_num,
                config.out_dim,
            )?,
            token_num: config.token_num,
        })
    }

    /// `detokenize` for `[token_num]` global tokens: implicit-codebook
    /// gather, the project_out linear, then the flatten-and-project.
    fn detokenize(&self, tokens: &[i32]) -> Result<Vec<f32>> {
        if tokens.len() != self.token_num {
            return Err(SpeechError::Input {
                why: format!(
                    "expected {} global tokens, got {}",
                    self.token_num,
                    tokens.len()
                ),
            });
        }
        let dims = self.codebook_dim;
        let mut codes = vec![0.0f32; tokens.len() * dims];
        for (frame, &token) in tokens.iter().enumerate() {
            let index = usize::try_from(token).map_err(|_| SpeechError::Input {
                why: format!("negative global token {token}"),
            })?;
            if index >= self.codebook_size {
                return Err(SpeechError::Input {
                    why: format!("global token {index} outside the FSQ codebook"),
                });
            }
            let source = &self.implicit_codebook[index * dims..(index + 1) * dims];
            codes[frame * dims..(frame + 1) * dims].copy_from_slice(source);
        }
        let projected = self.project_out.forward(&codes, tokens.len());
        // The reference swapaxes(-1, -2) + reshape flattens dim-major
        // (time slowest): [d0_t0, d0_t1, ..., d1_t0, ...].
        let latent = projected.len() / tokens.len();
        let mut flat = vec![0.0f32; latent * tokens.len()];
        for (t, row) in projected.chunks_exact(latent).enumerate() {
            for (d, &value) in row.iter().enumerate() {
                flat[d * tokens.len() + t] = value;
            }
        }
        Ok(self.project.forward(&flat, 1))
    }
}

/// The `SamplingBlock` identity path: with both scales at 1 the block
/// has no weights and its three skip adds multiply the signal by 3 (the
/// reference identity branch, load-bearing for the published prenet
/// `sample_ratios [1, 1]`).
struct SamplingBlock {
    dim: usize,
}

impl SamplingBlock {
    fn forward(&self, x: &[f32], frames: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; self.dim * frames];
        for (slot, &value) in out.iter_mut().zip(x) {
            *slot = 3.0 * value;
        }
        out
    }
}

/// The feat `Decoder` (prenet): linear pre, per-ratio SamplingBlock +
/// 2-layer Vocos backbone, the conditioned main Vocos backbone, and the
/// output linear. Activations run channel-major `[ch, frames]`.
struct FeatDecoder {
    linear_pre: Linear,
    stages: Vec<(SamplingBlock, vocos::Backbone)>,
    vocos_backbone: vocos::Backbone,
    linear: Linear,
    config: FeatDecoderConfig,
}

impl FeatDecoder {
    fn load(file: &SafetensorsFile, prefix: &str, config: &FeatDecoderConfig) -> Result<Self> {
        let dim = config.vocos_dim;
        let linear_pre = Linear::new(
            file.load_as_f32(&format!("{prefix}.linear_pre.weight"))?,
            Some(file.load_as_f32(&format!("{prefix}.linear_pre.bias"))?),
            config.input_channels,
            dim,
        );
        let mut stages = Vec::with_capacity(config.sample_ratios.len());
        for (index, _ratio) in config.sample_ratios.iter().enumerate() {
            let stage_prefix = format!("{prefix}.downsample.{index}");
            let sampling = SamplingBlock { dim };
            let backbone = load_vocos_backbone(
                file,
                &format!("{stage_prefix}.1"),
                dim,
                dim,
                2,
                None,
                config.vocos_intermediate_dim,
            )?;
            stages.push((sampling, backbone));
        }
        let vocos_backbone = load_vocos_backbone(
            file,
            &format!("{prefix}.vocos_backbone"),
            dim,
            dim,
            config.vocos_num_layers,
            config.condition_dim,
            config.vocos_intermediate_dim,
        )?;
        Ok(Self {
            linear_pre,
            stages,
            vocos_backbone,
            linear: Linear::new(
                file.load_as_f32(&format!("{prefix}.linear.weight"))?,
                Some(file.load_as_f32(&format!("{prefix}.linear.bias"))?),
                dim,
                config.out_channels,
            ),
            config: config.clone(),
        })
    }

    /// `[input_channels, frames]` + the `[condition_dim]` d-vector to
    /// `[out_channels, frames]` (the `bandwidth_id` condition). The
    /// pre-transpose, per-stage transposes, and final transpose of the
    /// reference collapse in the channel-major convention here.
    fn forward(&self, x: &[f32], frames: usize, condition: &[f32]) -> Vec<f32> {
        let dim = self.config.vocos_dim;
        let mut h = linear_cm(
            &self.linear_pre.weight,
            self.linear_pre.bias.as_deref(),
            self.config.input_channels,
            dim,
            x,
            frames,
        );
        for (sampling, backbone) in &self.stages {
            h = sampling.forward(&h, frames);
            h = backbone.forward(&h, frames, None);
        }
        let h = self.vocos_backbone.forward(&h, frames, Some(condition));
        let mut out = linear_cm(
            &self.linear.weight,
            self.linear.bias.as_deref(),
            dim,
            self.config.out_channels,
            &h,
            frames,
        );
        if self.config.use_tanh_at_final {
            for value in &mut out {
                *value = value.tanh();
            }
        }
        out
    }
}

fn load_vocos_backbone(
    file: &SafetensorsFile,
    prefix: &str,
    input_channels: usize,
    dim: usize,
    num_layers: usize,
    adanorm_num_embeddings: Option<usize>,
    intermediate_dim: usize,
) -> Result<vocos::Backbone> {
    let config = VocosBackboneConfig {
        input_channels,
        dim,
        intermediate_dim,
        num_layers,
        layer_scale_init_value: None,
        adanorm_num_embeddings,
        bias: true,
        input_kernel_size: 7,
        dw_kernel_size: 7,
    };
    vocos::load_backbone(file, &config, prefix, ConvLayout::Mlx)
}

/// A residual unit block (snake, dilated conv, snake, 1x1 conv).
struct WaveResidualUnit {
    snake1: Vec<f32>,
    conv1: Conv1d,
    snake2: Vec<f32>,
    conv2: Conv1d,
    ch: usize,
}

impl WaveResidualUnit {
    fn forward(&self, x: &[f32]) -> Vec<f32> {
        let ch = self.ch;
        let seq = x.len() / ch;
        let mut h = x.to_vec();
        snake1d(&mut h, &self.snake1, ch, seq);
        let h2 = self.conv1.forward(&h);
        let seq2 = h2.len() / ch;
        let mut h2 = h2;
        snake1d(&mut h2, &self.snake2, ch, seq2);
        let h3 = self.conv2.forward(&h2);
        // The reference crops the residual by half the length
        // difference; dilation paddings keep the lengths equal here.
        let out_seq = h3.len() / ch;
        let pad = seq.saturating_sub(out_seq) / 2;
        add_cropped_residual(x, seq, h3, ch, pad)
    }
}

/// A WaveGenerator `DecoderBlock`: snake, strided transposed conv, then
/// three residual units with dilations 1, 3, 9.
struct WaveDecoderBlock {
    snake: Vec<f32>,
    up: ConvTranspose1d,
    units: [WaveResidualUnit; 3],
}

impl WaveDecoderBlock {
    fn forward(&self, x: &[f32]) -> Vec<f32> {
        let seq = x.len() / self.up.in_ch;
        let mut h = x.to_vec();
        snake1d(&mut h, &self.snake, self.up.in_ch, seq);
        let h = self.up.forward(&h);
        let h = self.units[0].forward(&h);
        let h = self.units[1].forward(&h);
        self.units[2].forward(&h)
    }
}

/// The DAC-style WaveGenerator: the stem conv, per-rate blocks, then
/// the final snake + conv + tanh.
struct WaveGenerator {
    stem: Conv1d,
    blocks: Vec<WaveDecoderBlock>,
    snake: Vec<f32>,
    out_conv: Conv1d,
    out_dim: usize,
    d_out: usize,
    total_scale: usize,
}

impl WaveGenerator {
    fn load(file: &SafetensorsFile, prefix: &str, config: &WaveGeneratorConfig) -> Result<Self> {
        let stem = load_wn_conv1d_any_prefix(
            file,
            &format!("{prefix}.model.0"),
            config.input_channel,
            config.channels,
            7,
            1,
            3,
        )?;
        let mut blocks = Vec::with_capacity(config.rates.len());
        for (index, (&kernel, &stride)) in config.kernel_sizes.iter().zip(&config.rates).enumerate()
        {
            let in_dim = config.channels / 2usize.pow(index as u32);
            let out_dim = config.channels / 2usize.pow(index as u32 + 1);
            blocks.push(load_wave_decoder_block(
                file,
                &format!("{prefix}.model.{}", index + 1),
                in_dim,
                out_dim,
                kernel,
                stride,
            )?);
        }
        // Sequential indices: stem 0, blocks 1..N, final snake N + 1,
        // final conv N + 2.
        let snake_index = config.rates.len() + 1;
        let out_dim = config.channels / 2usize.pow(config.rates.len() as u32);
        let snake = load_alpha(
            file,
            &wave_name(&format!("{prefix}.model.{snake_index}.alpha")),
            out_dim,
        )?;
        let out_conv = load_wn_conv1d_any_prefix(
            file,
            &format!("{prefix}.model.{}", snake_index + 1),
            out_dim,
            config.d_out,
            7,
            1,
            3,
        )?;
        Ok(Self {
            stem,
            blocks,
            snake,
            out_conv,
            out_dim,
            d_out: config.d_out,
            total_scale: config.rates.iter().product::<usize>().max(1),
        })
    }

    /// `[input_channel, frames]` to `[d_out, frames * prod(rates)]`.
    fn forward(&self, x: &[f32], frames: usize) -> Result<Vec<f32>> {
        let mut h = self.stem.forward(x);
        for block in &self.blocks {
            h = block.forward(&h);
        }
        let frames_out = h.len() / self.out_dim;
        snake1d(&mut h, &self.snake, self.out_dim, frames_out);
        let mut h = self.out_conv.forward(&h);
        for value in &mut h {
            *value = value.tanh();
        }
        if h.len() != frames * self.d_out * self.total_scale {
            return Err(SpeechError::Tensor {
                name: "wave_generator output".into(),
                why: format!(
                    "expected {} samples for {frames} frames, got {}",
                    frames * self.d_out * self.total_scale,
                    h.len()
                ),
            });
        }
        Ok(h)
    }
}

/// WaveGenerator convs may carry the raw `block.` names; try both.
fn load_wn_conv1d_any_prefix(
    file: &SafetensorsFile,
    name: &str,
    in_ch: usize,
    out_ch: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
) -> Result<Conv1d> {
    if file.contains_tensor(&format!("{name}.weight_v")) {
        load_wn_conv1d(file, name, in_ch, out_ch, kernel, stride, padding, 1)
    } else {
        load_wn_conv1d(
            file,
            &wave_name(name),
            in_ch,
            out_ch,
            kernel,
            stride,
            padding,
            1,
        )
    }
}

fn load_wave_decoder_block(
    file: &SafetensorsFile,
    prefix: &str,
    in_dim: usize,
    out_dim: usize,
    kernel: usize,
    stride: usize,
) -> Result<WaveDecoderBlock> {
    let snake_tree = format!("{prefix}.block.layers.0.alpha");
    let snake = load_alpha(file, &snake_name(file, &snake_tree), in_dim)?;
    let padding = kernel.saturating_sub(stride) / 2;
    let convt_tree = format!("{prefix}.block.layers.1");
    let convt_name = if file.contains_tensor(&format!("{convt_tree}.weight_v")) {
        convt_tree
    } else {
        wave_name(&convt_tree)
    };
    let up = load_wn_convt(file, &convt_name, in_dim, out_dim, kernel, stride, padding)?;
    let mut units: Vec<WaveResidualUnit> = Vec::with_capacity(3);
    for (unit_index, dilation) in [1usize, 3, 9].into_iter().enumerate() {
        let unit_tree = format!("{prefix}.block.layers.{}", unit_index + 2);
        let unit_prefix = if file.contains_tensor(&format!("{unit_tree}.block.layers.0.alpha")) {
            unit_tree
        } else {
            wave_name(&unit_tree)
        };
        units.push(load_residual_unit(file, &unit_prefix, out_dim, dilation)?);
    }
    Ok(WaveDecoderBlock {
        snake,
        up,
        units: units.try_into().map_err(|_| SpeechError::Input {
            why: "wave decoder block needs three residual units".to_string(),
        })?,
    })
}

fn snake_name(file: &SafetensorsFile, tree: &str) -> String {
    if file.contains_tensor(tree) {
        tree.to_string()
    } else {
        wave_name(tree)
    }
}

fn load_residual_unit(
    file: &SafetensorsFile,
    prefix: &str,
    dim: usize,
    dilation: usize,
) -> Result<WaveResidualUnit> {
    Ok(WaveResidualUnit {
        snake1: load_alpha(
            file,
            &snake_name(file, &format!("{prefix}.block.layers.0.alpha")),
            dim,
        )?,
        conv1: load_wn_conv1d(
            file,
            &resolve_name(file, &format!("{prefix}.block.layers.1")),
            dim,
            dim,
            7,
            1,
            3 * dilation,
            dilation,
        )?,
        snake2: load_alpha(
            file,
            &snake_name(file, &format!("{prefix}.block.layers.2.alpha")),
            dim,
        )?,
        conv2: load_wn_conv1d(
            file,
            &resolve_name(file, &format!("{prefix}.block.layers.3")),
            dim,
            dim,
            1,
            1,
            0,
            1,
        )?,
        ch: dim,
    })
}

fn resolve_name(file: &SafetensorsFile, tree: &str) -> String {
    if file.contains_tensor(&format!("{tree}.weight_v")) {
        tree.to_string()
    } else {
        wave_name(tree)
    }
}

/// The loaded BiCodec decode stack.
pub struct BiCodec {
    pub config: BiCodecConfig,
    quantizer: FactorizedVectorQuantize,
    speaker: SpeakerDecoder,
    prenet: FeatDecoder,
    decoder: WaveGenerator,
}

impl BiCodec {
    /// Opens a BiCodec directory: `config.yaml` plus `model.safetensors`
    /// (the reference `load_from_checkpoint` layout).
    pub fn open(dir: &Path) -> Result<BiCodec> {
        let text = std::fs::read_to_string(dir.join("config.yaml")).map_err(|e| {
            SpeechError::BadConfig {
                field: "config.yaml".to_string(),
                why: e.to_string(),
            }
        })?;
        let config = BiCodecConfig::from_yaml(&text)?;
        let file = SafetensorsFile::open(&dir.join("model.safetensors"))?;
        BiCodec::load(config, &file)
    }

    /// Loads from a parsed config and a safetensors file.
    pub fn load(config: BiCodecConfig, file: &SafetensorsFile) -> Result<Self> {
        Ok(Self {
            quantizer: FactorizedVectorQuantize::load(file, &config.quantizer)?,
            speaker: SpeakerDecoder::load(file, &config.speaker_encoder)?,
            prenet: FeatDecoder::load(file, "prenet", &config.prenet)?,
            decoder: WaveGenerator::load(file, "decoder", &config.decoder)?,
            config,
        })
    }

    /// `BiCodec.detokenize` for one batch row: `[frames]` semantic
    /// tokens and `[token_num]` global tokens to mono samples. The
    /// sample count is `frames * prod(decoder rates)` (the published
    /// checkpoint upsamples 320x at 16 kHz). Empty semantic tokens are
    /// rejected: the reference placeholder shape never survives its own
    /// prenet transpose, so the observable upstream behavior is an
    /// error, not audio.
    pub fn detokenize(&self, semantic_tokens: &[i32], global_tokens: &[i32]) -> Result<Vec<f32>> {
        if semantic_tokens.is_empty() {
            return Err(SpeechError::Input {
                why: "empty semantic tokens; the reference detokenize crashes on \
                      its zero-frame placeholder"
                    .to_string(),
            });
        }
        let frames = semantic_tokens.len();
        let zq = self.quantizer.detokenize(semantic_tokens)?;
        let d_vector = self.speaker.detokenize(global_tokens)?;
        let mut x = self.prenet.forward(&zq, frames, &d_vector);
        // x = x + d_vector[..., None]: broadcast the speaker vector
        // over time.
        let channels = self.config.prenet.out_channels;
        for frame in 0..frames {
            for (c, &d) in d_vector.iter().enumerate().take(channels) {
                x[c * frames + frame] += d;
            }
        }
        self.decoder.forward(&x, frames)
    }
}

/// The shared YAML parser reads only scalars; the spark config carries
/// flow lists (`[1, 1]`, `[3, 3, 3, 3]`), which are valid JSON, so a
/// pass rewrites bracketed strings into arrays.
fn flow_lists(value: serde_json::Value) -> serde_json::Value {
    match value {
        serde_json::Value::Object(map) => {
            serde_json::Value::Object(map.into_iter().map(|(k, v)| (k, flow_lists(v))).collect())
        }
        serde_json::Value::String(text) if text.starts_with('[') && text.ends_with(']') => {
            serde_json::from_str(&text).unwrap_or(serde_json::Value::String(text))
        }
        other => other,
    }
}

#[cfg(test)]
mod tests;
