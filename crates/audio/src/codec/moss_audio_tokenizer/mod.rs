//! MOSS audio tokenizer (MOSS-Audio-Tokenizer, tensor2struct line).
//!
//! Reference: `mlx_audio/codec/models/moss_audio_tokenizer/moss_audio_tokenizer.py`
//! at mlx-audio
//! [`e1b19b9054bf163f5d812221a54fcc346f1890e9`](https://github.com/Blaizzy/mlx-audio/tree/e1b19b9054bf163f5d812221a54fcc346f1890e9/mlx_audio/codec/models/moss_audio_tokenizer).
//! A config-driven stack of patching and projected-transformer modules
//! around a residual lookup-free quantizer (L2-normalized lookups,
//! per-book in/out projections), with channel interleaving for stereo
//! input. Convolutions in the quantizer are all kernel-1 and the
//! checkpoints store PyTorch `parametrizations.weight.original0/1`
//! tensors, so the weight-norm fold happens at load.
//!
//! Scope: the batch-of-one `encode_audio` / `decode_audio_codes`
//! contract. The streaming `step` path (AttentionStepCache) and the
//! batch>1 orchestration are not ported. Attention masks use
//! `finfo.min` exactly like the reference; invalid query rows are
//! zeroed before the output projection.

use std::path::Path;

use turbospark_model_io::safetensors::SafetensorsFile;

use crate::codec::wnconv::load_f32_shaped;
use crate::ops;
use crate::{Result, SpeechError};

/// Pinned mlx-audio commit this port was transcribed from.
pub const REFERENCE_COMMIT: &str = "e1b19b9054bf163f5d812221a54fcc346f1890e9";

/// One encoder/decoder stack entry from `encoder_kwargs` /
/// `decoder_kwargs`.
#[derive(Debug, Clone)]
pub enum MossModuleConfig {
    /// `PatchedPretransform`: fold `patch_size` frames into channels
    /// (encode) or unfold (decode).
    Patched {
        patch_size: usize,
        is_downsample: bool,
    },
    /// `ProjectedTransformer` over a causal/windowed transformer.
    Transformer {
        input_dimension: usize,
        output_dimension: usize,
        d_model: usize,
        num_heads: usize,
        num_layers: usize,
        dim_feedforward: usize,
        causal: bool,
        /// Attention context band (`delta < context`), computed from
        /// `context_duration` by the reference constructor.
        context: Option<usize>,
        /// "none", "rope", "sin", or "sin_rope".
        positional_embedding: String,
        max_period: f32,
        positional_scale: f32,
        layer_scale: Option<f32>,
    },
}

/// Residual LFQ geometry (`quantizer_kwargs`).
#[derive(Debug, Clone)]
pub struct MossRlfqConfig {
    pub input_dim: usize,
    pub rvq_dim: usize,
    pub output_dim: usize,
    pub num_quantizers: usize,
    pub codebook_size: usize,
    pub codebook_dim: usize,
}

/// Tokenizer geometry, one-to-one with `AudioTokenizerConfig`.
#[derive(Debug, Clone)]
pub struct MossConfig {
    pub sample_rate: u32,
    pub downsample_rate: usize,
    pub number_channels: usize,
    pub enable_channel_interleave: bool,
    pub causal_transformer_context_duration: f32,
    pub encoder: Vec<MossModuleConfig>,
    pub decoder: Vec<MossModuleConfig>,
    pub quantizer_type: String,
    pub quantizer: MossRlfqConfig,
}

impl MossConfig {
    /// Parses a `config.json`-style object with the reference
    /// `from_dict` defaults, computing each transformer's context band
    /// from `context_duration` (or the shared default duration) exactly
    /// like the reference constructor: the frame rate starts at
    /// `sample_rate * channels` (interleave) and divides/multiplies by
    /// each module's downsample ratio.
    pub fn from_json(value: &serde_json::Value) -> Result<Self> {
        let get = |field: &str| value.get(field);
        let u = |field: &str, default: u64| -> usize {
            get(field).and_then(|v| v.as_u64()).unwrap_or(default) as usize
        };
        let sample_rate = match get("sample_rate").or_else(|| get("sampling_rate")) {
            Some(v) => v.as_u64().unwrap_or(48_000) as u32,
            None => 48_000,
        };
        let channels = get("number_channels").and_then(|v| v.as_u64()).unwrap_or(1) as usize;
        let interleave = get("enable_channel_interleave")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);
        let default_duration = get("causal_transformer_context_duration")
            .and_then(|v| v.as_f64())
            .unwrap_or(10.0) as f32;
        let channel_factor = if interleave && channels > 1 {
            channels
        } else {
            1
        };
        let parse_module = |kwargs: &serde_json::Value,
                            frame_rate: &mut f32|
         -> Result<MossModuleConfig> {
            let module_type = kwargs
                .get("module_type")
                .and_then(|v| v.as_str())
                .ok_or_else(|| SpeechError::BadConfig {
                    field: "module_type".to_string(),
                    why: "missing module_type".to_string(),
                })?;
            match module_type {
                "PatchedPretransform" => {
                    let patch_size = kwargs
                        .get("patch_size")
                        .and_then(|v| v.as_u64())
                        .ok_or_else(|| SpeechError::BadConfig {
                            field: "patch_size".to_string(),
                            why: "missing patch_size".to_string(),
                        })? as usize;
                    *frame_rate /= patch_size as f32;
                    Ok(MossModuleConfig::Patched {
                        patch_size,
                        is_downsample: true,
                    })
                }
                "Transformer" => {
                    let input_dimension = kwargs
                        .get("input_dimension")
                        .and_then(|v| v.as_u64())
                        .ok_or_else(|| SpeechError::BadConfig {
                            field: "input_dimension".to_string(),
                            why: "missing input_dimension".to_string(),
                        })? as usize;
                    let output_dimension = kwargs
                        .get("output_dimension")
                        .and_then(|v| v.as_u64())
                        .ok_or_else(|| SpeechError::BadConfig {
                        field: "output_dimension".to_string(),
                        why: "missing output_dimension".to_string(),
                    })? as usize;
                    let d_model =
                        kwargs
                            .get("d_model")
                            .and_then(|v| v.as_u64())
                            .ok_or_else(|| SpeechError::BadConfig {
                                field: "d_model".to_string(),
                                why: "missing d_model".to_string(),
                            })? as usize;
                    let duration = kwargs
                        .get("context_duration")
                        .and_then(|v| v.as_f64())
                        .unwrap_or(default_duration as f64)
                        as f32;
                    let context = Some(((*frame_rate * duration) as f64).round() as usize);
                    *frame_rate /= 1.0; // Transformer keeps the rate.
                    let num_layers = kwargs
                        .get("num_layers")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0) as usize;
                    Ok(MossModuleConfig::Transformer {
                        input_dimension,
                        output_dimension,
                        d_model,
                        num_heads: kwargs
                            .get("num_heads")
                            .and_then(|v| v.as_u64())
                            .unwrap_or(1) as usize,
                        num_layers,
                        dim_feedforward: kwargs
                            .get("dim_feedforward")
                            .and_then(|v| v.as_u64())
                            .unwrap_or(0) as usize,
                        causal: kwargs
                            .get("causal")
                            .and_then(|v| v.as_bool())
                            .unwrap_or(false),
                        context,
                        positional_embedding: kwargs
                            .get("positional_embedding")
                            .and_then(|v| v.as_str())
                            .unwrap_or("none")
                            .to_string(),
                        max_period: kwargs
                            .get("max_period")
                            .and_then(|v| v.as_f64())
                            .unwrap_or(10000.0) as f32,
                        positional_scale: kwargs
                            .get("position_scale")
                            .or_else(|| kwargs.get("positional_scale"))
                            .and_then(|v| v.as_f64())
                            .unwrap_or(1.0) as f32,
                        layer_scale: kwargs
                            .get("layer_scale")
                            .and_then(|v| v.as_f64())
                            .map(|v| v as f32),
                    })
                }
                other => Err(SpeechError::BadConfig {
                    field: "module_type".to_string(),
                    why: format!("unsupported module_type {other}"),
                }),
            }
        };
        let mut frame_rate = sample_rate as f32 * channel_factor as f32;
        let mark_side = |mut module: MossModuleConfig, encoder_side: bool| {
            if let MossModuleConfig::Patched {
                ref mut is_downsample,
                ..
            } = module
            {
                *is_downsample = encoder_side;
            }
            module
        };
        let mut encoder = Vec::new();
        if let Some(modules) = get("encoder_kwargs").and_then(|v| v.as_array()) {
            for kwargs in modules {
                encoder.push(mark_side(parse_module(kwargs, &mut frame_rate)?, true));
            }
        }
        let mut decoder = Vec::new();
        if let Some(modules) = get("decoder_kwargs").and_then(|v| v.as_array()) {
            for kwargs in modules {
                decoder.push(mark_side(parse_module(kwargs, &mut frame_rate)?, false));
            }
        }
        let q = get("quantizer_kwargs");
        let qv = q.cloned().unwrap_or(serde_json::json!({}));
        let qu = |field: &str, default: u64| -> usize {
            qv.get(field).and_then(|v| v.as_u64()).unwrap_or(default) as usize
        };
        let input_dim = qu("input_dim", 1024);
        let quantizer = MossRlfqConfig {
            input_dim,
            rvq_dim: qu("rvq_dim", input_dim as u64),
            output_dim: qu("output_dim", input_dim as u64),
            num_quantizers: qu("num_quantizers", 32),
            codebook_size: qu("codebook_size", 1024),
            codebook_dim: qu("codebook_dim", 8),
        };
        Ok(MossConfig {
            sample_rate,
            downsample_rate: u("downsample_rate", 3840),
            number_channels: channels,
            enable_channel_interleave: interleave,
            causal_transformer_context_duration: default_duration,
            encoder,
            decoder,
            quantizer_type: get("quantizer_type")
                .and_then(|v| v.as_str())
                .unwrap_or("rlfq")
                .to_string(),
            quantizer,
        })
    }
}

/// Moss's weight-norm kernel-1 conv folded at load: the checkpoint
/// stores `parametrizations.weight.original0` `[out, 1, 1]` and
/// `.original1` `[out, in, K]` in PyTorch layout; the norm runs over
/// every axis but the output channel. K is 1 everywhere here, so the
/// folded weight is a linear map.
#[derive(Debug, Clone)]
struct MossPointwise {
    in_dim: usize,
    out_dim: usize,
    weight: Vec<f32>,
    bias: Vec<f32>,
}

impl MossPointwise {
    /// `prefix` points at the conv module (its `.bias` sibling is
    /// loaded too).
    fn load(file: &SafetensorsFile, prefix: &str) -> Result<Self> {
        let g_name = format!("{prefix}.parametrizations.weight.original0");
        let v_name = format!("{prefix}.parametrizations.weight.original1");
        let g_desc = file
            .descriptor(&g_name)
            .ok_or_else(|| SpeechError::Tensor {
                name: g_name.clone(),
                why: "missing weight-norm gain".to_string(),
            })?;
        let v_desc = file
            .descriptor(&v_name)
            .ok_or_else(|| SpeechError::Tensor {
                name: v_name.clone(),
                why: "missing weight-norm direction".to_string(),
            })?;
        if g_desc.shape != vec![v_desc.shape[0], 1, 1] {
            return Err(SpeechError::Tensor {
                name: g_name.clone(),
                why: format!("expected [out, 1, 1], got {:?}", g_desc.shape),
            });
        }
        let (out_dim, in_dim, kernel) = (v_desc.shape[0], v_desc.shape[1], v_desc.shape[2]);
        let g = load_f32_shaped(file, &g_name, &[out_dim, 1, 1])?;
        let v = load_f32_shaped(file, &v_name, &[out_dim, in_dim, kernel])?;
        let bias = load_f32_shaped(file, &format!("{prefix}.bias"), &[out_dim])?;
        let mut weight = vec![0.0f32; v.len()];
        for oc in 0..out_dim {
            let mut acc = 0.0f32;
            for i in 0..in_dim {
                for k in 0..kernel {
                    let value = v[oc * in_dim * kernel + i * kernel + k];
                    acc += value * value;
                }
            }
            let norm = acc.sqrt();
            let gain = g[oc];
            for i in 0..in_dim {
                for k in 0..kernel {
                    weight[oc * in_dim * kernel + i * kernel + k] =
                        gain * v[oc * in_dim * kernel + i * kernel + k] / norm;
                }
            }
        }
        if kernel == 1 {
            // Already a pointwise map.
        } else {
            return Err(SpeechError::Tensor {
                name: v_name,
                why: format!("kernel {kernel} unsupported (only k=1 in the quantizer)"),
            });
        }
        Ok(MossPointwise {
            in_dim,
            out_dim,
            weight,
            bias,
        })
    }

    /// Channel-major `[in_dim, frames]` -> `[out_dim, frames]`.
    fn forward_channels(&self, x: &[f32], frames: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; self.out_dim * frames];
        for t in 0..frames {
            for o in 0..self.out_dim {
                let w = &self.weight[o * self.in_dim..(o + 1) * self.in_dim];
                let mut acc = self.bias[o];
                for i in 0..self.in_dim {
                    acc += x[i * frames + t] * w[i];
                }
                out[o * frames + t] = acc;
            }
        }
        out
    }
}

/// Fused-QKV pre-norm attention with finfo-min masks, interleaved
/// RoPE, and optional sinusoidal content handled by the parent
/// Transformer. Invalid query rows are zeroed before the output
/// projection.
#[derive(Debug, Clone)]
struct MossAttention {
    heads: usize,
    head_dim: usize,
    causal: bool,
    context: Option<usize>,
    max_period: f32,
    use_rope: bool,
    /// `[3 * dim, dim]` fused in-projection, no bias.
    in_proj: Vec<f32>,
    out_proj: Vec<f32>,
    dim: usize,
}

impl MossAttention {
    #[allow(clippy::too_many_arguments)]
    fn load(
        file: &SafetensorsFile,
        prefix: &str,
        dim: usize,
        heads: usize,
        causal: bool,
        context: Option<usize>,
        max_period: f32,
        use_rope: bool,
    ) -> Result<Self> {
        let in_proj = load_f32_shaped(file, &format!("{prefix}.in_proj.weight"), &[3 * dim, dim])?;
        let out_proj = load_f32_shaped(file, &format!("{prefix}.out_proj.weight"), &[dim, dim])?;
        Ok(MossAttention {
            heads,
            head_dim: dim / heads,
            causal,
            context,
            max_period,
            use_rope,
            in_proj,
            out_proj,
            dim,
        })
    }

    fn forward(&self, x: &[f32], seq: usize, input_len: usize) -> Vec<f32> {
        let dim = self.dim;
        // qkv rows [seq, 3 * dim].
        let qkv = ops::linear(x, &self.in_proj, None, seq, dim, 3 * dim);
        let mut qh = vec![0.0f32; self.heads * seq * self.head_dim];
        let mut kh = qh.clone();
        let mut vh = qh.clone();
        for h in 0..self.heads {
            for t in 0..seq {
                for d in 0..self.head_dim {
                    qh[(h * seq + t) * self.head_dim + d] =
                        qkv[t * 3 * dim + h * self.head_dim + d];
                    kh[(h * seq + t) * self.head_dim + d] =
                        qkv[t * 3 * dim + dim + h * self.head_dim + d];
                    vh[(h * seq + t) * self.head_dim + d] =
                        qkv[t * 3 * dim + 2 * dim + h * self.head_dim + d];
                }
            }
        }
        if self.use_rope {
            let (cos, sin) = ops::rope_tables(seq, self.head_dim, self.max_period);
            rope_interleaved_seq(&mut qh, self.heads, seq, self.head_dim, &cos, &sin);
            rope_interleaved_seq(&mut kh, self.heads, seq, self.head_dim, &cos, &sin);
        }
        // Additive mask: 0 or f32::MIN; rows are queries.
        let mut mask = vec![0.0f32; seq * seq];
        let mut has_mask = false;
        for i in 0..seq {
            for j in 0..seq {
                let mut allowed = j < input_len;
                if self.causal {
                    allowed &= i >= j;
                }
                if let Some(context) = self.context {
                    allowed &= i < j + context;
                }
                if !allowed {
                    mask[i * seq + j] = f32::MIN;
                    has_mask = true;
                }
            }
        }
        let scale = (self.head_dim as f32).powf(-0.5);
        let mut out = vec![0.0f32; qh.len()];
        for h in 0..self.heads {
            let plane = h * seq * self.head_dim;
            let o = ops::sdpa(
                &qh[plane..plane + seq * self.head_dim],
                &kh[plane..plane + seq * self.head_dim],
                &vh[plane..plane + seq * self.head_dim],
                if has_mask { Some(&mask) } else { None },
                seq,
                seq,
                self.head_dim,
                self.head_dim,
                scale,
            );
            out[plane..plane + o.len()].copy_from_slice(&o);
        }
        let mut merged = vec![0.0f32; seq * dim];
        for h in 0..self.heads {
            for t in 0..seq {
                let src = (h * seq + t) * self.head_dim;
                merged[t * dim + h * self.head_dim..t * dim + (h + 1) * self.head_dim]
                    .copy_from_slice(&out[src..src + self.head_dim]);
            }
        }
        // Zero invalid query rows before the output projection.
        for t in input_len..seq {
            merged[t * dim..(t + 1) * dim].fill(0.0);
        }
        ops::linear(&merged, &self.out_proj, None, seq, dim, dim)
    }
}

#[derive(Debug, Clone)]
struct MossLayerScale {
    scale: Vec<f32>,
}

/// Pre-norm transformer layer with optional LayerScale and a bias-free
/// exact-GELU FFN.
#[derive(Debug, Clone)]
struct MossLayer {
    self_attn: MossAttention,
    norm1: (Vec<f32>, Vec<f32>),
    norm2: (Vec<f32>, Vec<f32>),
    ffn_in: Vec<f32>,
    ffn_out: Vec<f32>,
    ffn_dim: usize,
    layer_scale_1: Option<MossLayerScale>,
    layer_scale_2: Option<MossLayerScale>,
    dim: usize,
}

impl MossLayer {
    #[allow(clippy::too_many_arguments)]
    fn load(
        file: &SafetensorsFile,
        prefix: &str,
        dim: usize,
        heads: usize,
        ffn_dim: usize,
        causal: bool,
        context: Option<usize>,
        positional_embedding: &str,
        max_period: f32,
        layer_scale: Option<f32>,
    ) -> Result<Self> {
        let ln = |name: &str| -> Result<(Vec<f32>, Vec<f32>)> {
            Ok((
                load_f32_shaped(file, &format!("{prefix}.{name}.weight"), &[dim])?,
                load_f32_shaped(file, &format!("{prefix}.{name}.bias"), &[dim])?,
            ))
        };
        // LayerScale vectors are checkpoint parameters (the config's
        // layer_scale flag only decides whether the module exists); a
        // missing tensor is the Identity case.
        let load_scale = |name: &str| -> Result<Option<MossLayerScale>> {
            Ok(if layer_scale.is_some() {
                Some(MossLayerScale {
                    scale: load_f32_shaped(file, &format!("{prefix}.{name}.scale"), &[dim])?,
                })
            } else {
                None
            })
        };
        Ok(MossLayer {
            self_attn: MossAttention::load(
                file,
                &format!("{prefix}.self_attn"),
                dim,
                heads,
                causal,
                context,
                max_period,
                matches!(positional_embedding, "rope" | "sin_rope"),
            )?,
            norm1: ln("norm1")?,
            norm2: ln("norm2")?,
            ffn_in: load_f32_shaped(file, &format!("{prefix}.ffn.0.weight"), &[ffn_dim, dim])?,
            ffn_out: load_f32_shaped(file, &format!("{prefix}.ffn.2.weight"), &[dim, ffn_dim])?,
            ffn_dim,
            layer_scale_1: load_scale("layer_scale_1")?,
            layer_scale_2: load_scale("layer_scale_2")?,
            dim,
        })
    }

    fn forward(&self, x: &mut [f32], seq: usize, input_len: usize) {
        let mut normed = x.to_vec();
        ops::layernorm(
            &mut normed,
            seq,
            self.dim,
            &self.norm1.0,
            Some(&self.norm1.1),
            1e-5,
        );
        let attn = self.self_attn.forward(&normed, seq, input_len);
        // LayerScale is per-channel over [seq, dim].
        let attn = match &self.layer_scale_1 {
            Some(ls) => attn
                .iter()
                .enumerate()
                .map(|(i, v)| v * ls.scale[i % self.dim])
                .collect(),
            None => attn,
        };
        for (v, a) in x.iter_mut().zip(attn) {
            *v += a;
        }
        let mut normed = x.to_vec();
        ops::layernorm(
            &mut normed,
            seq,
            self.dim,
            &self.norm2.0,
            Some(&self.norm2.1),
            1e-5,
        );
        let mut h = ops::linear(&normed, &self.ffn_in, None, seq, self.dim, self.ffn_dim);
        ops::gelu_erf(&mut h);
        let h = ops::linear(&h, &self.ffn_out, None, seq, self.ffn_dim, self.dim);
        let h = match &self.layer_scale_2 {
            Some(ls) => h
                .iter()
                .enumerate()
                .map(|(i, v)| v * ls.scale[i % self.dim])
                .collect(),
            None => h,
        };
        for (v, m) in x.iter_mut().zip(h) {
            *v += m;
        }
    }
}

/// ProjectedTransformer: optional input/output projections plus the
/// sinusoidal content embedding, over rows `[seq, d_model]`.
#[derive(Debug, Clone)]
struct MossTransformer {
    input_proj: Option<MimoStyleLinear>,
    output_proj: Option<MimoStyleLinear>,
    layers: Vec<MossLayer>,
    d_model: usize,
    positional_embedding: String,
    max_period: f32,
    positional_scale: f32,
}

/// Bias-free linear with `[out, in]` weight (nn.Linear).
#[derive(Debug, Clone)]
struct MimoStyleLinear {
    in_dim: usize,
    out_dim: usize,
    weight: Vec<f32>,
}

impl MimoStyleLinear {
    fn load(file: &SafetensorsFile, prefix: &str, in_dim: usize, out_dim: usize) -> Result<Self> {
        Ok(MimoStyleLinear {
            in_dim,
            out_dim,
            weight: load_f32_shaped(file, &format!("{prefix}.weight"), &[out_dim, in_dim])?,
        })
    }

    fn forward(&self, x: &[f32], rows: usize) -> Vec<f32> {
        ops::linear(x, &self.weight, None, rows, self.in_dim, self.out_dim)
    }
}

impl MossTransformer {
    #[allow(clippy::too_many_arguments)]
    fn load(
        file: &SafetensorsFile,
        prefix: &str,
        spec: &MossModuleConfig,
        force_input: bool,
        force_output: bool,
    ) -> Result<Self> {
        let MossModuleConfig::Transformer {
            input_dimension,
            output_dimension,
            d_model,
            num_heads,
            num_layers,
            dim_feedforward,
            causal,
            context,
            positional_embedding,
            max_period,
            positional_scale,
            layer_scale,
        } = spec
        else {
            return Err(SpeechError::BadConfig {
                field: "module".to_string(),
                why: "expected a Transformer module".to_string(),
            });
        };
        let input_proj = if force_input || input_dimension != d_model {
            Some(MimoStyleLinear::load(
                file,
                &format!("{prefix}.input_proj"),
                *input_dimension,
                *d_model,
            )?)
        } else {
            None
        };
        let output_proj = if force_output || output_dimension != d_model {
            Some(MimoStyleLinear::load(
                file,
                &format!("{prefix}.output_proj"),
                *d_model,
                *output_dimension,
            )?)
        } else {
            None
        };
        let mut layers = Vec::with_capacity(*num_layers);
        for i in 0..*num_layers {
            layers.push(MossLayer::load(
                file,
                &format!("{prefix}.transformer.layers.{i}"),
                *d_model,
                *num_heads,
                *dim_feedforward,
                *causal,
                *context,
                positional_embedding,
                *max_period,
                *layer_scale,
            )?);
        }
        Ok(MossTransformer {
            input_proj,
            output_proj,
            layers,
            d_model: *d_model,
            positional_embedding: positional_embedding.clone(),
            max_period: *max_period,
            positional_scale: *positional_scale,
        })
    }

    /// Rows `[seq, in_dim]` -> `[seq, out_dim]`.
    fn forward(&self, x: &[f32], seq: usize, input_len: usize) -> Vec<f32> {
        let mut h = match &self.input_proj {
            Some(proj) => proj.forward(x, seq),
            None => x.to_vec(),
        };
        if matches!(self.positional_embedding.as_str(), "sin" | "sin_rope") {
            add_sin_embedding(
                &mut h,
                seq,
                self.d_model,
                self.max_period,
                self.positional_scale,
                0,
            );
        }
        for layer in &self.layers {
            layer.forward(&mut h, seq, input_len);
        }
        match &self.output_proj {
            Some(proj) => proj.forward(&h, seq),
            None => h,
        }
    }
}

/// The Transformer sinusoidal content embedding: `concat(cos, sin)` of
/// `positions / max_period^(arange(half) / max(half - 1, 1))`.
fn add_sin_embedding(
    x: &mut [f32],
    seq: usize,
    dim: usize,
    max_period: f32,
    scale: f32,
    offset: usize,
) {
    let half = dim / 2;
    let mut emb = vec![0.0f32; seq * dim];
    for t in 0..seq {
        for i in 0..half {
            let scale_i = max_period.powf(i as f32 / (half.saturating_sub(1)).max(1) as f32);
            let phase = (offset + t) as f32 / scale_i;
            emb[t * dim + i] = phase.cos();
            emb[t * dim + half + i] = phase.sin();
        }
    }
    for (v, e) in x.iter_mut().zip(emb) {
        *v += scale * e;
    }
}

/// One LFQ book: L2-normalized lookup against L2-normalized codebook
/// rows, emitting the RAW rows (the out projection runs outside).
#[derive(Debug, Clone)]
struct MossLfq {
    in_proj: MossPointwise,
    out_proj: MossPointwise,
    /// `[codebook_size, codebook_dim]`.
    codebook: Vec<f32>,
    codebook_norm: Vec<f32>,
    codebook_size: usize,
    codebook_dim: usize,
}

impl MossLfq {
    fn load(
        file: &SafetensorsFile,
        prefix: &str,
        rvq_dim: usize,
        size: usize,
        dim: usize,
    ) -> Result<Self> {
        let in_proj = MossPointwise::load(file, &format!("{prefix}.in_proj"))?;
        if in_proj.in_dim != rvq_dim || in_proj.out_dim != dim {
            return Err(SpeechError::Tensor {
                name: format!("{prefix}.in_proj"),
                why: format!(
                    "expected {} -> {}, got {} -> {}",
                    rvq_dim, dim, in_proj.in_dim, in_proj.out_dim
                ),
            });
        }
        let out_proj = MossPointwise::load(file, &format!("{prefix}.out_proj"))?;
        let codebook = load_f32_shaped(file, &format!("{prefix}.codebook.weight"), &[size, dim])?;
        let mut codebook_norm = vec![0.0f32; size];
        for (r, norm) in codebook_norm.iter_mut().enumerate() {
            let row = &codebook[r * dim..(r + 1) * dim];
            *norm = row.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-12);
        }
        Ok(MossLfq {
            in_proj,
            out_proj,
            codebook,
            codebook_norm,
            codebook_size: size,
            codebook_dim: dim,
        })
    }

    /// Latents channel-major `[rvq_dim, frames]` -> raw codebook
    /// vectors channel-major `[codebook_dim, frames]` and indices.
    /// Both sides are L2-normalized (floor 1e-12), so the nearest
    /// lookup maximizes the dot product.
    fn decode_latents(&self, latents: &[f32], frames: usize) -> (Vec<f32>, Vec<i32>) {
        let mut indices = vec![0i32; frames];
        let mut zq = vec![0.0f32; self.codebook_dim * frames];
        for t in 0..frames {
            let mut col = vec![0.0f32; self.codebook_dim];
            for (d, slot) in col.iter_mut().enumerate() {
                *slot = latents[d * frames + t];
            }
            let norm = col.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-12);
            let mut best = f32::NEG_INFINITY;
            let mut best_idx = 0usize;
            for c in 0..self.codebook_size {
                let row = &self.codebook[c * self.codebook_dim..(c + 1) * self.codebook_dim];
                let mut dot = 0.0f32;
                for (a, &b) in col.iter().zip(row) {
                    dot += (a / norm) * (b / self.codebook_norm[c]);
                }
                if dot > best {
                    best = dot;
                    best_idx = c;
                }
            }
            indices[t] = best_idx as i32;
            for d in 0..self.codebook_dim {
                zq[d * frames + t] = self.codebook[best_idx * self.codebook_dim + d];
            }
        }
        (zq, indices)
    }
}

/// Residual LFQ over the books, channel-major throughout
/// (`[dim, frames]`, matching the reference's `(B, C, T)`).
#[derive(Debug, Clone)]
struct MossRlfq {
    input_proj: MossPointwise,
    output_proj: MossPointwise,
    quantizers: Vec<MossLfq>,
    rvq_dim: usize,
}

impl MossRlfq {
    fn load(config: &MossConfig, file: &SafetensorsFile) -> Result<Self> {
        let q = &config.quantizer;
        let input_proj = MossPointwise::load(file, "quantizer.input_proj")?;
        if input_proj.in_dim != q.input_dim || input_proj.out_dim != q.rvq_dim {
            return Err(SpeechError::Tensor {
                name: "quantizer.input_proj".to_string(),
                why: format!(
                    "expected {} -> {}, got {} -> {}",
                    q.input_dim, q.rvq_dim, input_proj.in_dim, input_proj.out_dim
                ),
            });
        }
        let output_proj = MossPointwise::load(file, "quantizer.output_proj")?;
        let mut quantizers = Vec::with_capacity(q.num_quantizers);
        for i in 0..q.num_quantizers {
            quantizers.push(MossLfq::load(
                file,
                &format!("quantizer.quantizers.{i}"),
                q.rvq_dim,
                q.codebook_size,
                q.codebook_dim,
            )?);
        }
        Ok(MossRlfq {
            input_proj,
            output_proj,
            quantizers,
            rvq_dim: q.rvq_dim,
        })
    }

    /// Hidden `[input_dim, frames]` with `input_len` valid frames ->
    /// quantized `[output_dim, frames]` plus per-book indices. Invalid
    /// frames are masked out of both the residual stream and the
    /// accumulation, exactly like the reference's `update_mask`.
    fn encode(
        &self,
        hidden: &[f32],
        frames: usize,
        input_len: usize,
        n_quantizers: Option<usize>,
    ) -> Result<(Vec<f32>, Vec<Vec<i32>>)> {
        let count = n_quantizers
            .unwrap_or(self.quantizers.len())
            .min(self.quantizers.len());
        let projected = self.input_proj.forward_channels(hidden, frames);
        let mut residual = projected;
        for d in 0..self.rvq_dim {
            for t in input_len..frames {
                residual[d * frames + t] = 0.0;
            }
        }
        let mut quantized = vec![0.0f32; self.rvq_dim * frames];
        let mut indices = Vec::with_capacity(count);
        for book in &self.quantizers[..count] {
            // LFQ.__call__: project the residual to codebook space,
            // look up, then project the raw rows back out.
            let z_e = book.in_proj.forward_channels(&residual, frames);
            let (zq_raw, idx) = book.decode_latents(&z_e, frames);
            let zq = book.out_proj.forward_channels(&zq_raw, frames);
            for d in 0..self.rvq_dim {
                for t in 0..frames {
                    let value = if t < input_len {
                        zq[d * frames + t]
                    } else {
                        0.0
                    };
                    quantized[d * frames + t] += value;
                    residual[d * frames + t] -= value;
                }
            }
            indices.push(idx);
        }
        Ok((
            self.output_proj.forward_channels(&quantized, frames),
            indices,
        ))
    }

    /// Codes per book -> latents `[output_dim, frames]`. Each book
    /// contributes `out_proj(raw codebook rows)`; the shared
    /// `output_proj` runs once over the sum.
    fn decode_codes(&self, codes: &[Vec<i32>], frames: usize) -> Result<Vec<f32>> {
        if codes.is_empty() || codes.len() > self.quantizers.len() {
            return Err(SpeechError::Input {
                why: "codes must carry 1..=num_quantizers books".to_string(),
            });
        }
        let mut emb = vec![0.0f32; frames * self.rvq_dim];
        for (q, book_codes) in codes.iter().enumerate() {
            let book = &self.quantizers[q];
            let mut raw = vec![0.0f32; frames * book.codebook_dim];
            for (t, &c) in book_codes.iter().enumerate() {
                let idx = usize::try_from(c).map_err(|_| SpeechError::Input {
                    why: format!("negative code {c}"),
                })?;
                if idx >= book.codebook_size {
                    return Err(SpeechError::Input {
                        why: format!("code {c} outside book {q} of {}", book.codebook_size),
                    });
                }
                for d in 0..book.codebook_dim {
                    raw[d * frames + t] = book.codebook[idx * book.codebook_dim + d];
                }
            }
            let projected = book.out_proj.forward_channels(&raw, frames);
            for (v, p) in emb.iter_mut().zip(projected) {
                *v += p;
            }
        }
        Ok(self.output_proj.forward_channels(&emb, frames))
    }
}

/// One encoder/decoder stack module.
#[derive(Debug, Clone)]
enum MossModule {
    Patched { patch: usize, is_downsample: bool },
    Transformer(MossTransformer),
}

/// Loaded MOSS audio tokenizer, batch-of-one.
pub struct MossAudioTokenizer {
    pub config: MossConfig,
    encoder: Vec<MossModule>,
    decoder: Vec<MossModule>,
    rvq: MossRlfq,
}

impl MossAudioTokenizer {
    /// Opens a checkpoint directory with `config.json` plus
    /// safetensors weights (`model.safetensors` or shards).
    pub fn open(dir: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(dir.join("config.json")).map_err(|e| {
            SpeechError::BadConfig {
                field: "config.json".to_string(),
                why: e.to_string(),
            }
        })?;
        let value: serde_json::Value =
            serde_json::from_str(&text).map_err(|e| SpeechError::BadConfig {
                field: "config.json".to_string(),
                why: e.to_string(),
            })?;
        let config = MossConfig::from_json(&value)?;
        let file = open_checkpoint(dir)?;
        Self::load(config, &file)
    }

    /// Loads from a parsed config and a safetensors file. Projection
    /// presence is taken from the checkpoint keys (the reference
    /// `projection_keys` mechanism).
    pub fn load(config: MossConfig, file: &SafetensorsFile) -> Result<Self> {
        if config.quantizer_type != "rlfq" && config.quantizer_type != "random_prefix_rlfq" {
            return Err(SpeechError::BadConfig {
                field: "quantizer_type".to_string(),
                why: format!("unsupported MOSS quantizer_type {}", config.quantizer_type),
            });
        }
        let load_modules = |modules: &[MossModuleConfig], side: &str| -> Result<Vec<MossModule>> {
            let mut out = Vec::with_capacity(modules.len());
            for (i, spec) in modules.iter().enumerate() {
                match spec {
                    MossModuleConfig::Patched { patch_size, .. } => {
                        out.push(MossModule::Patched {
                            patch: *patch_size,
                            is_downsample: side == "encoder",
                        });
                    }
                    transformer @ MossModuleConfig::Transformer { .. } => {
                        let base = format!("{side}.{i}");
                        let force_input = file
                            .descriptor(&format!("{base}.input_proj.weight"))
                            .is_some();
                        let force_output = file
                            .descriptor(&format!("{base}.output_proj.weight"))
                            .is_some();
                        out.push(MossModule::Transformer(MossTransformer::load(
                            file,
                            &base,
                            transformer,
                            force_input,
                            force_output,
                        )?));
                    }
                }
            }
            Ok(out)
        };
        let encoder = load_modules(&config.encoder, "encoder")?;
        let decoder = load_modules(&config.decoder, "decoder")?;
        let rvq = MossRlfq::load(&config, file)?;
        Ok(MossAudioTokenizer {
            config,
            encoder,
            decoder,
            rvq,
        })
    }

    /// Mono samples at `sample_rate` -> codes per book plus the code
    /// length. Port of `encode_audio` for one channel-mono stream (the
    /// reference repeats mono to the configured channel count).
    pub fn encode(&self, samples: &[f32]) -> Result<(Vec<Vec<i32>>, usize)> {
        let channels = self.config.number_channels;
        let mut input = vec![0.0f32; channels * samples.len()];
        for c in 0..channels {
            input[c * samples.len()..(c + 1) * samples.len()].copy_from_slice(samples);
        }
        let len = samples.len();
        let (codes, code_len) = self.encode_frame(&input, len, None)?;
        Ok((codes, code_len))
    }

    /// The reference `_encode_frame`: pad to a `downsample_rate`
    /// multiple, interleave channels, run the encoder stack, quantize.
    /// Input is channel-major `[channels, samples]`.
    pub fn encode_frame(
        &self,
        input: &[f32],
        input_len: usize,
        n_quantizers: Option<usize>,
    ) -> Result<(Vec<Vec<i32>>, usize)> {
        let c = &self.config;
        let channels = c.number_channels;
        let mut hidden = input.to_vec();
        let mut len = input_len;
        let total = input.len() / channels;
        let pad = (c.downsample_rate - total % c.downsample_rate) % c.downsample_rate;
        if pad > 0 {
            let mut padded = vec![0.0f32; (total + pad) * channels];
            for ch in 0..channels {
                padded[ch * (total + pad)..ch * (total + pad) + total]
                    .copy_from_slice(&hidden[ch * total..ch * total + total]);
            }
            hidden = padded;
            len += pad;
        }
        if channels > 1 && c.enable_channel_interleave {
            // (C, T) -> interleaved (1, T*C): position p carries
            // channel p % C of frame p / C.
            let interleaved_len = len * channels;
            let mut flat = vec![0.0f32; interleaved_len];
            for t in 0..len {
                for ch in 0..channels {
                    flat[t * channels + ch] = hidden[ch * len + t];
                }
            }
            hidden = flat;
            len = interleaved_len;
        }
        for module in &self.encoder {
            (hidden, len) = self.run_module(module, &hidden, len)?;
        }
        let frames = hidden.len() / self.rvq.input_dim();
        let (quantized, indices) = self.rvq.encode(&hidden, frames, len, n_quantizers)?;
        let _ = quantized;
        Ok((indices, len))
    }

    /// Codes per book -> time-major interleaved samples
    /// (`samples[t * channels + ch]`, the reference
    /// `decode_audio_codes` contract).
    pub fn decode(&self, codes: &[Vec<i32>]) -> Result<Vec<f32>> {
        let (audio, len) = self.decode_frame(codes)?;
        let channels = self.config.number_channels;
        if channels == 1 || !self.config.enable_channel_interleave {
            return Ok(audio);
        }
        let mut out = vec![0.0f32; audio.len()];
        for ch in 0..channels {
            for t in 0..len {
                out[t * channels + ch] = audio[ch * len + t];
            }
        }
        Ok(out)
    }

    /// The reference `_decode_frame`: quantizer decode, decoder stack,
    /// channel restore. Returns channel-major `[channels, samples]`.
    pub fn decode_frame(&self, codes: &[Vec<i32>]) -> Result<(Vec<f32>, usize)> {
        let c = &self.config;
        let frames = codes.first().map_or(0, |book| book.len());
        if frames == 0 {
            return Ok((Vec::new(), 0));
        }
        let mut hidden = self.rvq.decode_codes(codes, frames)?;
        let mut len = frames;
        for module in &self.decoder {
            (hidden, len) = self.run_module(module, &hidden, len)?;
        }
        // Restore channels from the interleaved stream.
        if c.number_channels > 1 && c.enable_channel_interleave {
            let deinter = vec![0.0f32; hidden.len()];
            let _ = deinter;
            let channels = c.number_channels;
            let stream = hidden.len(); // 1 channel * len * channels
            let total = stream / channels;
            let mut out = vec![0.0f32; stream];
            for t in 0..total {
                for ch in 0..channels {
                    out[ch * total + t] = hidden[t * channels + ch];
                }
            }
            return Ok((out, total));
        }
        Ok((hidden, len))
    }

    fn run_module(&self, module: &MossModule, x: &[f32], len: usize) -> Result<(Vec<f32>, usize)> {
        match module {
            MossModule::Patched {
                patch,
                is_downsample,
            } => Ok(self.run_patched(*patch, *is_downsample, x, len)),
            MossModule::Transformer(t) => {
                let channels = x.len() / len;
                let mut rows = vec![0.0f32; x.len()];
                for time in 0..len {
                    for d in 0..channels {
                        rows[time * channels + d] = x[d * len + time];
                    }
                }
                let out = t.forward(&rows, len, len);
                let out_dim = out.len() / len;
                let mut cm = vec![0.0f32; out.len()];
                for time in 0..len {
                    for d in 0..out_dim {
                        cm[d * len + time] = out[time * out_dim + d];
                    }
                }
                Ok((cm, len))
            }
        }
    }

    /// PatchedPretransform: the encode direction folds `patch` frames
    /// per channel into new channels (`C*patch, T/patch`); the decode
    /// direction unfolds (`C/patch... i.e. channels are C*patch from
    /// the fold, output C, T*patch`). Both are pure reshapes.
    fn run_patched(
        &self,
        patch: usize,
        is_downsample: bool,
        x: &[f32],
        len: usize,
    ) -> (Vec<f32>, usize) {
        let channels = x.len() / len;
        if is_downsample {
            let frames = len / patch;
            let mut out = vec![0.0f32; channels * patch * frames];
            for ch in 0..channels {
                for t in 0..frames {
                    for p in 0..patch {
                        out[(ch * patch + p) * frames + t] = x[ch * len + t * patch + p];
                    }
                }
            }
            (out, frames)
        } else {
            let true_channels = channels / patch;
            let mut out = vec![0.0f32; true_channels * len * patch];
            for ch in 0..true_channels {
                for t in 0..len {
                    for p in 0..patch {
                        out[ch * (len * patch) + t * patch + p] = x[(ch * patch + p) * len + t];
                    }
                }
            }
            (out, len * patch)
        }
    }
}

/// GPT-J-style interleaved rope over full sequences: pairs are
/// `(x[2i], x[2i+1])` and the cos/sin tables are `[seq, dim / 2]`
/// (the `ops::rope_*` helpers are decode-step sized).
fn rope_interleaved_seq(
    x: &mut [f32],
    heads: usize,
    seq: usize,
    dim: usize,
    cos: &[f32],
    sin: &[f32],
) {
    let half = dim / 2;
    assert!(cos.len() >= seq * half);
    for h in 0..heads {
        for t in 0..seq {
            let base = (h * seq + t) * dim;
            for d in 0..half {
                let a = x[base + 2 * d];
                let b = x[base + 2 * d + 1];
                let c = cos[t * half + d];
                let s = sin[t * half + d];
                x[base + 2 * d] = a * c - b * s;
                x[base + 2 * d + 1] = b * c + a * s;
            }
        }
    }
}

fn open_checkpoint(dir: &Path) -> Result<SafetensorsFile> {
    let canonical = dir.join("model.safetensors");
    if canonical.exists() {
        return SafetensorsFile::open(&canonical).map_err(SpeechError::from);
    }
    let mut entries: Vec<_> = std::fs::read_dir(dir)
        .map_err(|e| SpeechError::BadConfig {
            field: dir.display().to_string(),
            why: e.to_string(),
        })?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|ext| ext == "safetensors"))
        .collect();
    entries.sort();
    let first = match entries.len() {
        1 => entries.remove(0),
        0 => {
            return Err(SpeechError::BadConfig {
                field: dir.display().to_string(),
                why: "no safetensors checkpoint found".to_string(),
            })
        }
        _ => {
            // Shards: load the lexicographically first shard's keys
            // through the same file handle; shard support here covers
            // the single-shard fixtures and refuses true multi-shard
            // checkpoints rather than mis-loading them.
            return Err(SpeechError::BadConfig {
                field: dir.display().to_string(),
                why: "multiple safetensors shards are not supported; merge first".to_string(),
            });
        }
    };
    SafetensorsFile::open(first.as_path()).map_err(SpeechError::from)
}

impl MossRlfq {
    fn input_dim(&self) -> usize {
        self.input_proj.in_dim
    }
}

#[cfg(test)]
mod tests;
