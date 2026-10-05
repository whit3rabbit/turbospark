//! Mega-ASR audio-quality router.
//!
//! Transcribed from `mlx_audio/stt/models/mega_asr/router.py` at the pinned
//! mlx-audio 0.5.7 commit (`e1b19b9054bf163f5d812221a54fcc346f1890e9`): an
//! 80-band Slaney log-mel frontend (no whisper peak clamp, no tail-frame
//! drop), a two-stage strided Conv1d frontend with BatchNorm and GELU, a
//! sinusoidal-position single-layer transformer encoder, attention pooling,
//! and a two-class classifier head. `route` degrades a clip when the softmax
//! probability of the degraded class reaches 0.5, which is the argmax rule.
//!
//! Weight loading mirrors `AudioQualityRouter::from_converted`: the geometry
//! is inferred from tensor shapes, the packed `in_proj` attention tensors are
//! split into per-head projections, and the head count stays at the verified
//! 4.

use std::collections::BTreeMap;

use turbospark_model_io::safetensors::SafetensorsFile;

use crate::mel::MelFilterbank;
use crate::ops;
use crate::stft::StftOptions;
use crate::whisper::{whisper_mel_filterbank, WHISPER_HOP, WHISPER_N_FFT};
use crate::{Result, SpeechError};

use super::config::RouterSettings;

const LOG_MEL_FLOOR: f32 = 1.0e-10;
const LOG_MEL_OFFSET: f32 = 4.0;
const BATCH_NORM_EPS: f32 = 1.0e-5;
const LAYER_NORM_EPS: f32 = 1.0e-5;
const CONV_KERNEL: usize = 3;
const CONV_STRIDE: usize = 2;
const CONV_PADDING: usize = 1;
const NHEAD: usize = 4;

/// One router evaluation with its intermediate stage tensors.
#[derive(Debug, Clone, PartialEq)]
pub struct RouterStages {
    /// Router log-mel, `[frames, 80]` row-major.
    pub logmel: Vec<f32>,
    pub logmel_frames: usize,
    /// Conv frontend output, `[steps, d_model]` row-major.
    pub frontend_hidden: Vec<f32>,
    /// Transformer encoder output, `[steps, d_model]` row-major.
    pub transformer_hidden: Vec<f32>,
    /// Attention-pooled clip vector, `[d_model]`.
    pub pooled: Vec<f32>,
    /// Raw two-class logits `[clean, degraded]`.
    pub logits: [f32; 2],
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RouteDecision {
    pub degraded_prob: f32,
    pub use_lora: bool,
}

struct Linear {
    weight: Vec<f32>,
    bias: Vec<f32>,
    input: usize,
    output: usize,
}

impl Linear {
    fn forward(&self, x: &[f32], rows: usize) -> Vec<f32> {
        ops::linear(
            x,
            &self.weight,
            Some(&self.bias),
            rows,
            self.input,
            self.output,
        )
    }
}

struct LayerNorm {
    weight: Vec<f32>,
    bias: Vec<f32>,
    width: usize,
}

impl LayerNorm {
    fn apply(&self, values: &mut [f32], rows: usize) {
        ops::layernorm(
            values,
            rows,
            self.width,
            &self.weight,
            Some(&self.bias),
            LAYER_NORM_EPS,
        );
    }
}

/// Two strided convolutions with inference-mode BatchNorm and GELU, exactly
/// `ConvFrontend` from the reference.
struct ConvFrontend {
    conv1: Linear,
    conv2: Linear,
    bn1: BatchNorm,
    bn2: BatchNorm,
}

struct BatchNorm {
    weight: Vec<f32>,
    bias: Vec<f32>,
    running_mean: Vec<f32>,
    running_var: Vec<f32>,
}

impl BatchNorm {
    fn apply(&self, channels: &mut [Vec<f32>]) {
        for (channel, values) in channels.iter_mut().enumerate() {
            let scale = self.weight[channel] / (self.running_var[channel] + BATCH_NORM_EPS).sqrt();
            let shift = self.bias[channel] - self.running_mean[channel] * scale;
            for value in values.iter_mut() {
                *value = *value * scale + shift;
            }
        }
    }
}

struct EncoderLayer {
    norm1: LayerNorm,
    attention: Attention,
    norm2: LayerNorm,
    linear1: Linear,
    linear2: Linear,
}

struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    heads: usize,
    head_dim: usize,
}

impl Attention {
    /// Full self-attention over the time axis, matching
    /// `MultiHeadSelfAttention`: scaled `q @ k^T`, softmax, then `@ v`.
    fn forward(&self, x: &[f32], steps: usize) -> Vec<f32> {
        let queries = self.q_proj.forward(x, steps);
        let keys = self.k_proj.forward(x, steps);
        let values = self.v_proj.forward(x, steps);
        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let mut attended = vec![0.0f32; steps * self.heads * self.head_dim];
        for step in 0..steps {
            for head in 0..self.heads {
                let offset = head * self.head_dim;
                let query = &queries[step * self.d_model() + offset
                    ..step * self.d_model() + offset + self.head_dim];
                let mut scores = vec![0.0f32; steps];
                for (other, score) in scores.iter_mut().enumerate() {
                    let key = &keys[other * self.d_model() + offset
                        ..other * self.d_model() + offset + self.head_dim];
                    let dot: f32 = query.iter().zip(key).map(|(&q, &k)| q * k).sum::<f32>();
                    *score = dot * scale;
                }
                ops::softmax_row(&mut scores);
                let target = step * self.heads * self.head_dim + offset;
                for (other, &weight) in scores.iter().enumerate() {
                    let value = &values[other * self.d_model() + offset
                        ..other * self.d_model() + offset + self.head_dim];
                    for (index, slot) in attended[target..target + self.head_dim]
                        .iter_mut()
                        .enumerate()
                    {
                        *slot += weight * value[index];
                    }
                }
            }
        }
        self.out_proj.forward(&attended, steps)
    }

    fn d_model(&self) -> usize {
        self.heads * self.head_dim
    }
}

/// The loaded router. Geometry follows `from_converted`: every dimension is
/// inferred from the tensor shapes, and the head count stays at the verified
/// 4.
pub struct AudioQualityRouter {
    frontend: ConvFrontend,
    pe: Vec<f32>,
    max_len: usize,
    layers: Vec<EncoderLayer>,
    final_norm: LayerNorm,
    pooling: Linear,
    classifier1: Linear,
    classifier2: Linear,
    d_model: usize,
    n_mels: usize,
}

fn tensor<'map>(
    map: &'map BTreeMap<String, (Vec<usize>, Vec<f32>)>,
    name: &str,
    shape: &[usize],
) -> Result<&'map [f32]> {
    let (found, values) = map.get(name).ok_or_else(|| SpeechError::Tensor {
        name: name.to_owned(),
        why: "required router tensor is missing".into(),
    })?;
    if found.as_slice() != shape {
        return Err(SpeechError::Tensor {
            name: name.to_owned(),
            why: format!("expected shape {shape:?}, got {found:?}"),
        });
    }
    Ok(values)
}

fn linear(
    map: &BTreeMap<String, (Vec<usize>, Vec<f32>)>,
    weight: &str,
    bias: &str,
    input: usize,
    output: usize,
) -> Result<Linear> {
    Ok(Linear {
        weight: tensor(map, weight, &[output, input])?.to_vec(),
        bias: tensor(map, bias, &[output])?.to_vec(),
        input,
        output,
    })
}

fn layer_norm(
    map: &BTreeMap<String, (Vec<usize>, Vec<f32>)>,
    prefix: &str,
    width: usize,
) -> Result<LayerNorm> {
    Ok(LayerNorm {
        weight: tensor(map, &format!("{prefix}.weight"), &[width])?.to_vec(),
        bias: tensor(map, &format!("{prefix}.bias"), &[width])?.to_vec(),
        width,
    })
}

/// Converts an MLX Conv1d weight `[out, kernel, in]` to the shared kernel's
/// PyTorch `[out, in, kernel]` layout at load time.
fn conv_weight(raw: &[f32], out: usize, kernel: usize, input: usize) -> Vec<f32> {
    let mut converted = vec![0.0f32; raw.len()];
    for o in 0..out {
        for i in 0..input {
            for k in 0..kernel {
                let source = (o * kernel + k) * input + i;
                let target = (o * input + i) * kernel + k;
                converted[target] = raw[source];
            }
        }
    }
    converted
}

impl AudioQualityRouter {
    /// Infers the geometry from tensor shapes exactly like
    /// `AudioQualityRouter.from_converted`, then loads every tensor the
    /// forward pass consumes. `settings`, when present (a dynamic-profile
    /// `router_config`), must agree with the inferred geometry.
    pub fn from_tensors(
        map: &BTreeMap<String, (Vec<usize>, Vec<f32>)>,
        settings: Option<&RouterSettings>,
    ) -> Result<Self> {
        let d_model = {
            let (shape, _) =
                map.get("transformer.norm.weight")
                    .ok_or_else(|| SpeechError::Tensor {
                        name: "transformer.norm.weight".into(),
                        why: "required router tensor is missing".into(),
                    })?;
            if shape.len() != 1 {
                return Err(bad_shape("transformer.norm.weight", shape));
            }
            shape[0]
        };
        let num_layers = {
            let mut count = 0usize;
            while map.contains_key(&format!("transformer.layers.{count}.norm1.weight")) {
                count += 1;
            }
            count
        };
        if num_layers == 0 {
            return Err(SpeechError::Tensor {
                name: "transformer.layers.0.norm1.weight".into(),
                why: "the router needs at least one transformer layer".into(),
            });
        }
        let (conv0_shape, _) =
            map.get("frontend.conv.0.weight")
                .ok_or_else(|| SpeechError::Tensor {
                    name: "frontend.conv.0.weight".into(),
                    why: "required router tensor is missing".into(),
                })?;
        if conv0_shape.len() != 3 {
            return Err(bad_shape("frontend.conv.0.weight", conv0_shape));
        }
        let frontend_hidden_dim = conv0_shape[0];
        let n_mels = conv0_shape[2];
        let dim_feedforward = map
            .get("transformer.layers.0.linear1.weight")
            .map(|(shape, _)| shape[0])
            .ok_or_else(|| SpeechError::Tensor {
                name: "transformer.layers.0.linear1.weight".into(),
                why: "required router tensor is missing".into(),
            })?;
        let classifier_hidden_dim = map
            .get("classifier.0.weight")
            .map(|(shape, _)| shape[0])
            .ok_or_else(|| SpeechError::Tensor {
                name: "classifier.0.weight".into(),
                why: "required router tensor is missing".into(),
            })?;
        let pe_shape = map
            .get("pos_encoder.pe")
            .map(|(shape, _)| shape.clone())
            .ok_or_else(|| SpeechError::Tensor {
                name: "pos_encoder.pe".into(),
                why: "required router tensor is missing".into(),
            })?;
        if pe_shape.len() != 3 || pe_shape[0] != 1 {
            return Err(bad_shape("pos_encoder.pe", &pe_shape));
        }
        let max_len = pe_shape[1];
        if d_model % NHEAD != 0 {
            return Err(SpeechError::Unsupported {
                why: format!("router d_model {d_model} does not divide across 4 heads"),
            });
        }
        if let Some(settings) = settings {
            let inferred = (
                d_model,
                NHEAD,
                dim_feedforward,
                num_layers,
                n_mels,
                frontend_hidden_dim,
                classifier_hidden_dim,
                max_len,
            );
            let declared = (
                settings.d_model,
                settings.nhead,
                settings.dim_feedforward,
                settings.num_layers,
                settings.n_mels,
                settings.frontend_hidden_dim,
                settings.classifier_hidden_dim,
                settings.max_len,
            );
            if inferred != declared {
                return Err(SpeechError::Unsupported {
                    why: format!(
                        "router_config {declared:?} disagrees with the weight-derived geometry {inferred:?}"
                    ),
                });
            }
        }

        // The router log-mel frontend is the verified 80-band Slaney bank.
        if n_mels != 80 {
            return Err(SpeechError::Unsupported {
                why: format!(
                    "Mega-ASR router supports the verified 80-band log-mel frontend, found {n_mels}"
                ),
            });
        }

        let head_dim = d_model / NHEAD;
        let mut layers = Vec::with_capacity(num_layers);
        for index in 0..num_layers {
            let prefix = format!("transformer.layers.{index}");
            let in_proj_weight = tensor(
                map,
                &format!("{prefix}.self_attn.in_proj_weight"),
                &[3 * d_model, d_model],
            )?
            .to_vec();
            let in_proj_bias = tensor(
                map,
                &format!("{prefix}.self_attn.in_proj_bias"),
                &[3 * d_model],
            )?
            .to_vec();
            let split = |part: usize| {
                in_proj_weight[part * d_model * d_model..(part + 1) * d_model * d_model].to_vec()
            };
            let split_bias =
                |part: usize| in_proj_bias[part * d_model..(part + 1) * d_model].to_vec();
            layers.push(EncoderLayer {
                norm1: layer_norm(map, &format!("{prefix}.norm1"), d_model)?,
                attention: Attention {
                    q_proj: Linear {
                        weight: split(0),
                        bias: split_bias(0),
                        input: d_model,
                        output: d_model,
                    },
                    k_proj: Linear {
                        weight: split(1),
                        bias: split_bias(1),
                        input: d_model,
                        output: d_model,
                    },
                    v_proj: Linear {
                        weight: split(2),
                        bias: split_bias(2),
                        input: d_model,
                        output: d_model,
                    },
                    out_proj: linear(
                        map,
                        &format!("{prefix}.self_attn.out_proj.weight"),
                        &format!("{prefix}.self_attn.out_proj.bias"),
                        d_model,
                        d_model,
                    )?,
                    heads: NHEAD,
                    head_dim,
                },
                norm2: layer_norm(map, &format!("{prefix}.norm2"), d_model)?,
                linear1: linear(
                    map,
                    &format!("{prefix}.linear1.weight"),
                    &format!("{prefix}.linear1.bias"),
                    d_model,
                    dim_feedforward,
                )?,
                linear2: linear(
                    map,
                    &format!("{prefix}.linear2.weight"),
                    &format!("{prefix}.linear2.bias"),
                    dim_feedforward,
                    d_model,
                )?,
            });
        }

        let frontend = ConvFrontend {
            conv1: Linear {
                weight: conv_weight(
                    tensor(
                        map,
                        "frontend.conv.0.weight",
                        &[frontend_hidden_dim, CONV_KERNEL, n_mels],
                    )?,
                    frontend_hidden_dim,
                    CONV_KERNEL,
                    n_mels,
                ),
                bias: tensor(map, "frontend.conv.0.bias", &[frontend_hidden_dim])?.to_vec(),
                input: n_mels,
                output: frontend_hidden_dim,
            },
            bn1: BatchNorm {
                weight: tensor(map, "frontend.conv.1.weight", &[frontend_hidden_dim])?.to_vec(),
                bias: tensor(map, "frontend.conv.1.bias", &[frontend_hidden_dim])?.to_vec(),
                running_mean: tensor(map, "frontend.conv.1.running_mean", &[frontend_hidden_dim])?
                    .to_vec(),
                running_var: tensor(map, "frontend.conv.1.running_var", &[frontend_hidden_dim])?
                    .to_vec(),
            },
            conv2: Linear {
                weight: conv_weight(
                    tensor(
                        map,
                        "frontend.conv.4.weight",
                        &[d_model, CONV_KERNEL, frontend_hidden_dim],
                    )?,
                    d_model,
                    CONV_KERNEL,
                    frontend_hidden_dim,
                ),
                bias: tensor(map, "frontend.conv.4.bias", &[d_model])?.to_vec(),
                input: frontend_hidden_dim,
                output: d_model,
            },
            bn2: BatchNorm {
                weight: tensor(map, "frontend.conv.5.weight", &[d_model])?.to_vec(),
                bias: tensor(map, "frontend.conv.5.bias", &[d_model])?.to_vec(),
                running_mean: tensor(map, "frontend.conv.5.running_mean", &[d_model])?.to_vec(),
                running_var: tensor(map, "frontend.conv.5.running_var", &[d_model])?.to_vec(),
            },
        };

        Ok(Self {
            frontend,
            pe: tensor(map, "pos_encoder.pe", &[1, max_len, d_model])?.to_vec(),
            max_len,
            layers,
            final_norm: layer_norm(map, "transformer.norm", d_model)?,
            pooling: linear(
                map,
                "pooling.query.weight",
                "pooling.query.bias",
                d_model,
                1,
            )?,
            classifier1: linear(
                map,
                "classifier.0.weight",
                "classifier.0.bias",
                d_model,
                classifier_hidden_dim,
            )?,
            classifier2: linear(
                map,
                "classifier.3.weight",
                "classifier.3.bias",
                classifier_hidden_dim,
                2,
            )?,
            d_model,
            n_mels,
        })
    }

    /// Loads a converted router from an `extras/router.safetensors` file.
    pub fn from_safetensors(
        file: &SafetensorsFile,
        settings: Option<&RouterSettings>,
    ) -> Result<Self> {
        let mut map = BTreeMap::new();
        for name in file.tensor_names() {
            let descriptor = file.descriptor(name).ok_or_else(|| SpeechError::Tensor {
                name: name.to_owned(),
                why: "descriptor vanished while reading router weights".into(),
            })?;
            map.insert(
                name.to_owned(),
                (descriptor.shape.clone(), file.load_as_f32(name)?),
            );
        }
        Self::from_tensors(&map, settings)
    }

    /// The reference `LogMel80`: power log-mel with the 1e-10 floor and the
    /// `(x + 4) / 4` normalization. Unlike the whisper frontend there is no
    /// peak-relative clamp and the final centered frame is kept.
    pub fn logmel(&self, samples: &[f32]) -> Result<(Vec<f32>, usize)> {
        let options = StftOptions {
            fft_size: WHISPER_N_FFT,
            hop: WHISPER_HOP,
            window: crate::dsp::hann_window(WHISPER_N_FFT),
            center: true,
        };
        let filterbank: MelFilterbank = whisper_mel_filterbank(self.n_mels)?;
        let spectra = crate::stft::stft(samples, &options)?;
        let mut out = Vec::with_capacity(spectra.len() * self.n_mels);
        for spectrum in &spectra {
            let power: Vec<f32> = spectrum.iter().map(|c| c.re * c.re + c.im * c.im).collect();
            let projected = filterbank.project(&power)?;
            for value in projected {
                out.push((value.max(LOG_MEL_FLOOR).log10() + LOG_MEL_OFFSET) / LOG_MEL_OFFSET);
            }
        }
        Ok((out, spectra.len()))
    }

    /// Full router evaluation with every stage tensor, mirroring
    /// `AudioQualityRouter.logits` and its internals.
    pub fn forward(&self, samples: &[f32]) -> Result<RouterStages> {
        if samples.is_empty() || samples.iter().any(|s| !s.is_finite()) {
            return Err(SpeechError::Input {
                why: "router audio must be non-empty and finite".into(),
            });
        }
        let (logmel, logmel_frames) = self.logmel(samples)?;

        // Conv frontend over channel-major `[channels, steps]` buffers.
        let mut channels: Vec<Vec<f32>> = (0..self.n_mels)
            .map(|mel| {
                (0..logmel_frames)
                    .map(|frame| logmel[frame * self.n_mels + mel])
                    .collect()
            })
            .collect();
        let mut steps = logmel_frames;
        for (conv, norm) in [
            (&self.frontend.conv1, &self.frontend.bn1),
            (&self.frontend.conv2, &self.frontend.bn2),
        ] {
            let planar: Vec<f32> = channels.iter().flatten().copied().collect();
            let output = ops::conv1d(
                &planar,
                &conv.weight,
                Some(&conv.bias),
                conv.input,
                conv.output,
                CONV_KERNEL,
                CONV_STRIDE,
                CONV_PADDING,
                1,
                1,
            );
            steps = (steps + 2 * CONV_PADDING - CONV_KERNEL) / CONV_STRIDE + 1;
            channels = (0..conv.output)
                .map(|channel| output[channel * steps..(channel + 1) * steps].to_vec())
                .collect();
            norm.apply(&mut channels);
            for channel in &mut channels {
                ops::gelu_erf(channel);
            }
        }
        let mut frontend_hidden = vec![0.0f32; steps * self.d_model];
        for step in 0..steps {
            for (dim, channel) in channels.iter().enumerate() {
                frontend_hidden[step * self.d_model + dim] = channel[step];
            }
        }

        if steps > self.max_len {
            return Err(SpeechError::Input {
                why: format!(
                    "router frontend produced {steps} steps beyond the {max}-step positional table",
                    max = self.max_len
                ),
            });
        }
        let mut hidden = frontend_hidden.clone();
        for (step, row) in hidden.chunks_mut(self.d_model).enumerate() {
            for (dim, value) in row.iter_mut().enumerate() {
                *value += self.pe[step * self.d_model + dim];
            }
        }
        for layer in &self.layers {
            let mut normalized = hidden.clone();
            layer.norm1.apply(&mut normalized, steps);
            let attended = layer.attention.forward(&normalized, steps);
            for (value, add) in hidden.iter_mut().zip(attended) {
                *value += add;
            }
            let mut normalized = hidden.clone();
            layer.norm2.apply(&mut normalized, steps);
            let mut fed = layer.linear1.forward(&normalized, steps);
            ops::gelu_erf(&mut fed);
            let fed = layer.linear2.forward(&fed, steps);
            for (value, add) in hidden.iter_mut().zip(fed) {
                *value += add;
            }
        }
        self.final_norm.apply(&mut hidden, steps);
        let transformer_hidden = hidden;

        // Attention pooling: softmax over time of the projected query.
        let query = self.pooling.forward(&transformer_hidden, steps);
        let mut weights = query;
        ops::softmax_row(&mut weights);
        let mut pooled = vec![0.0f32; self.d_model];
        for (step, &weight) in weights.iter().enumerate() {
            for (dim, slot) in pooled.iter_mut().enumerate() {
                *slot += weight * transformer_hidden[step * self.d_model + dim];
            }
        }

        let classified = self.classifier1.forward(&pooled, 1);
        let mut classified = classified;
        ops::gelu_erf(&mut classified);
        let logits = self.classifier2.forward(&classified, 1);
        Ok(RouterStages {
            logmel,
            logmel_frames,
            frontend_hidden,
            transformer_hidden,
            pooled,
            logits: [logits[0], logits[1]],
        })
    }

    /// Router width inferred from the weights; the transformer, pooling, and
    /// classifier all share it.
    pub fn d_model(&self) -> usize {
        self.d_model
    }

    /// The reference `route`: degraded when the softmax probability of the
    /// degraded class reaches 0.5, which is the argmax rule.
    pub fn route(&self, samples: &[f32]) -> Result<(RouterStages, RouteDecision)> {
        let stages = self.forward(samples)?;
        let max = stages.logits[0].max(stages.logits[1]);
        let exp0 = (stages.logits[0] - max).exp();
        let exp1 = (stages.logits[1] - max).exp();
        let degraded_prob = exp1 / (exp0 + exp1);
        Ok((
            stages,
            RouteDecision {
                degraded_prob,
                use_lora: degraded_prob >= 0.5,
            },
        ))
    }
}

fn bad_shape(name: &str, shape: &[usize]) -> SpeechError {
    SpeechError::Tensor {
        name: name.to_owned(),
        why: format!("unexpected router tensor shape {shape:?}"),
    }
}
