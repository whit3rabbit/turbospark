//! Canary configuration parsing with refusals.
//!
//! Reference: `mlx_audio/stt/models/canary/config.py` at mlx-audio 0.5.7,
//! commit `e1b19b9054bf163f5d812221a54fcc346f1890e9`. The reference dataclasses
//! default every unrecognized field into oblivion; this parser instead refuses
//! any checkpoint whose graph switches fall outside the one layout the port
//! implements, so a similar but different encoder never loads silently.

use serde_json::Value;
use turbospark_audio::nemo_mel::{NemoMelNormalization, NemoMelOptions};
use turbospark_audio::stft::StftOptions;

use crate::nn::json::{bool_field, required, text_field, usize_field};
use crate::nn::symmetric_hann;
use crate::quant::QuantScheme;
use crate::vad::sortformer::FcEncoderConfig;
use crate::{Result, SpeechError};

/// Transformer decoder configuration (`transf_decoder` in `config.json`).
#[derive(Debug, Clone)]
pub struct DecoderConfig {
    pub num_layers: usize,
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub inner_size: usize,
    pub max_sequence_length: usize,
}

/// Parsed Canary model configuration.
#[derive(Debug, Clone)]
pub struct CanaryConfig {
    pub mel: NemoMelOptions,
    pub encoder: FcEncoderConfig,
    pub decoder: DecoderConfig,
    pub vocab_size: usize,
    /// The pinned groupwise affine scheme every decoder and encoder linear
    /// dequantizes through.
    pub quant: QuantScheme,
}

fn bad(field: impl Into<String>, why: impl Into<String>) -> SpeechError {
    SpeechError::BadConfig {
        field: field.into(),
        why: why.into(),
    }
}

impl CanaryConfig {
    /// Parses the checkpoint's `config.json`. Unsupported graph switches
    /// fail during open rather than silently selecting a similar but wrong
    /// model.
    pub fn from_json(v: &Value) -> Result<Self> {
        let pre = required(v, "preprocessor")?;
        let enc = required(v, "encoder")?;
        let dec = required(v, "transf_decoder")?;

        let sample_rate = u32::try_from(usize_field(pre, "sample_rate")?)
            .map_err(|_| bad("preprocessor.sample_rate", "does not fit u32"))?;
        if sample_rate != 16_000 {
            return Err(bad("preprocessor.sample_rate", "only 16 kHz is supported"));
        }
        for (key, expected) in [("window", "hann"), ("normalize", "per_feature")] {
            if text_field(pre, key)? != expected {
                return Err(bad(
                    format!("preprocessor.{key}"),
                    format!("only {expected} is supported"),
                ));
            }
        }
        // The reference PreprocessArgs carries `dither` but
        // log_mel_spectrogram never reads it, so the port also ignores it
        // (as it ignores every config key outside the reference dataclass).
        let feature_size = usize_field(pre, "features")?;
        let n_fft = usize_field(pre, "n_fft")?;
        let window_samples = ((required(pre, "window_size")?
            .as_f64()
            .ok_or_else(|| bad("preprocessor.window_size", "must be numeric"))?
            * f64::from(sample_rate)) as usize)
            .max(1);
        let hop = ((required(pre, "window_stride")?
            .as_f64()
            .ok_or_else(|| bad("preprocessor.window_stride", "must be numeric"))?
            * f64::from(sample_rate)) as usize)
            .max(1);
        if n_fft < window_samples || n_fft % 2 != 0 {
            return Err(bad(
                "preprocessor.n_fft",
                "must be even and at least the analysis-window length",
            ));
        }
        let mel = NemoMelOptions {
            stft: StftOptions {
                fft_size: n_fft,
                hop,
                window: symmetric_hann(window_samples),
                center: true,
            },
            sample_rate,
            num_mels: feature_size,
            normalize: NemoMelNormalization::PerFeature,
            preemphasis: pre.get("preemph").and_then(Value::as_f64).unwrap_or(0.97) as f32,
            log_zero_guard_value: pre
                .get("log_zero_guard_value")
                .and_then(Value::as_f64)
                .unwrap_or(2.0f64.powi(-24)) as f32,
            pad_to: pre.get("pad_to").and_then(Value::as_u64).unwrap_or(0) as usize,
            pad_value: pre.get("pad_value").and_then(Value::as_f64).unwrap_or(0.0) as f32,
            normalize_valid_frames: pre
                .get("normalize_valid_frames")
                .and_then(Value::as_bool)
                .unwrap_or(false),
        };

        if text_field(enc, "subsampling")? != "dw_striding"
            || text_field(enc, "self_attention_model")? != "rel_pos"
            || enc.get("causal_downsampling").and_then(Value::as_bool) == Some(true)
        {
            return Err(SpeechError::Unsupported {
                why: "Canary requires the non-causal depthwise-striding relative-position \
                      FastConformer encoder"
                    .into(),
            });
        }
        let hidden_size = usize_field(enc, "d_model")?;
        let heads = usize_field(enc, "n_heads")?;
        let ff_expansion = usize_field(enc, "ff_expansion_factor")?;
        let conv_kernel_size = usize_field(enc, "conv_kernel_size")?;
        let subsampling_factor = usize_field(enc, "subsampling_factor")?;
        let encoder = FcEncoderConfig {
            hidden_size,
            num_hidden_layers: usize_field(enc, "n_layers")?,
            num_attention_heads: heads,
            intermediate_size: hidden_size
                .checked_mul(ff_expansion)
                .ok_or_else(|| bad("encoder.ff_expansion_factor", "dimension overflow"))?,
            num_mel_bins: feature_size,
            conv_kernel_size,
            subsampling_conv_channels: usize_field(enc, "subsampling_conv_channels")?,
            subsampling_conv_kernel_size: 3,
            subsampling_conv_stride: 2,
            attention_bias: bool_field(enc, "use_bias")?,
            scale_input: bool_field(enc, "xscaling")?,
        };
        if subsampling_factor != 8
            || heads == 0
            || hidden_size % heads != 0
            || conv_kernel_size % 2 == 0
            || usize_field(enc, "feat_in")? != encoder.num_mel_bins
        {
            return Err(SpeechError::Unsupported {
                why: "only an 8x, odd-kernel FastConformer whose feat_in matches the mel \
                      feature count is supported"
                    .into(),
            });
        }
        // The reference projects the encoder output to enc_output_dim when it
        // differs from d_model; the pinned conversion ships no projection
        // tensor, so the port refuses that geometry instead of guessing.
        let enc_output_dim = v
            .get("enc_output_dim")
            .and_then(Value::as_u64)
            .map(|n| usize::try_from(n).map_err(|_| bad("enc_output_dim", "does not fit usize")))
            .transpose()?
            .unwrap_or(hidden_size);
        if enc_output_dim != hidden_size {
            return Err(SpeechError::Unsupported {
                why: "an encoder output projection is required when enc_output_dim differs \
                      from encoder d_model; this port implements only the identity path the \
                      pinned checkpoint ships"
                    .into(),
            });
        }

        for (field, expected) in [
            ("pre_ln", true),
            ("pre_ln_final_layer_norm", true),
            ("learn_positional_encodings", false),
        ] {
            if bool_field(dec, field)? != expected {
                return Err(bad(
                    format!("transf_decoder.{field}"),
                    format!(
                        "the port implements the reference decoder only with {field}={expected}"
                    ),
                ));
            }
        }
        if text_field(dec, "hidden_act")? != "relu" {
            return Err(bad(
                "transf_decoder.hidden_act",
                "only the relu feed-forward is supported",
            ));
        }
        let vocab_size = usize_field(dec, "vocab_size")?;
        let decoder = DecoderConfig {
            num_layers: usize_field(dec, "num_layers")?,
            hidden_size: usize_field(dec, "hidden_size")?,
            num_attention_heads: usize_field(dec, "num_attention_heads")?,
            inner_size: usize_field(dec, "inner_size")?,
            max_sequence_length: usize_field(dec, "max_sequence_length")?,
        };
        if decoder.hidden_size != hidden_size {
            return Err(bad(
                "transf_decoder.hidden_size",
                "must equal the encoder d_model (the reference feeds encoder states to the \
                 cross-attention of the same width)",
            ));
        }
        if decoder.num_attention_heads == 0
            || decoder.hidden_size % decoder.num_attention_heads != 0
        {
            return Err(bad(
                "transf_decoder.num_attention_heads",
                "must divide hidden_size",
            ));
        }
        if let Some(head) = v.get("head") {
            let classes = usize_field(head, "num_classes")?;
            if classes != vocab_size {
                return Err(bad(
                    "head.num_classes",
                    "disagrees with transf_decoder.vocab_size",
                ));
            }
        }

        let quantization = required(v, "quantization")?;
        let bits = u32::try_from(
            required(quantization, "bits")?
                .as_u64()
                .ok_or_else(|| bad("quantization.bits", "must be an integer"))?,
        )
        .map_err(|_| bad("quantization.bits", "does not fit u32"))?;
        // Only the pinned 8-bit groupwise affine scheme is verified; anything
        // else must refuse rather than silently mis-decode.
        if bits != 8 {
            return Err(SpeechError::Unsupported {
                why: format!(
                    "only the verified 8-bit quantization is supported, config says {bits} bits"
                ),
            });
        }
        let group_size = quantization
            .get("group_size")
            .and_then(Value::as_u64)
            .map(|n| {
                usize::try_from(n).map_err(|_| bad("quantization.group_size", "does not fit usize"))
            })
            .transpose()?
            .unwrap_or(64);
        if group_size != 64 {
            return Err(SpeechError::Unsupported {
                why: format!(
                    "only the verified group size 64 is supported, config says {group_size}"
                ),
            });
        }

        Ok(Self {
            mel,
            encoder,
            decoder,
            vocab_size,
            quant: QuantScheme { bits, group_size },
        })
    }
}
