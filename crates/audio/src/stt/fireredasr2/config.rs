use serde_json::Value;

use crate::{Result, SpeechError};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FireRedAsr2Config {
    pub input_dim: usize,
    pub vocab_size: usize,
    pub model_dim: usize,
    pub start_token: usize,
    pub end_token: usize,
    pub pad_token: usize,
    pub encoder_layers: usize,
    pub encoder_heads: usize,
    pub encoder_kernel: usize,
    pub encoder_position_limit: usize,
    pub decoder_layers: usize,
    pub decoder_heads: usize,
    pub decoder_position_limit: usize,
}

fn bad(field: &str, why: impl Into<String>) -> SpeechError {
    SpeechError::BadConfig {
        field: field.into(),
        why: why.into(),
    }
}

fn positive(value: &Value, field: &str) -> Result<usize> {
    value
        .get(field)
        .and_then(Value::as_u64)
        .and_then(|number| usize::try_from(number).ok())
        .filter(|&number| number > 0)
        .ok_or_else(|| bad(field, "must be a positive integer"))
}

fn token(value: &Value, field: &str) -> Result<usize> {
    value
        .get(field)
        .and_then(Value::as_u64)
        .and_then(|number| usize::try_from(number).ok())
        .ok_or_else(|| bad(field, "must be a non-negative integer"))
}

impl FireRedAsr2Config {
    pub fn from_json(root: &Value) -> Result<Self> {
        if root.get("model_type").and_then(Value::as_str) != Some("fireredasr2") {
            return Err(SpeechError::Unsupported {
                why: "config is not a FireRedASR2 checkpoint".into(),
            });
        }
        let encoder = root
            .get("encoder")
            .ok_or_else(|| bad("encoder", "missing from config.json"))?;
        let decoder = root
            .get("decoder")
            .ok_or_else(|| bad("decoder", "missing from config.json"))?;
        let config = Self {
            input_dim: positive(root, "idim")?,
            vocab_size: positive(root, "odim")?,
            model_dim: positive(root, "d_model")?,
            start_token: token(root, "sos_id")?,
            end_token: token(root, "eos_id")?,
            pad_token: token(root, "pad_id")?,
            encoder_layers: positive(encoder, "n_layers")?,
            encoder_heads: positive(encoder, "n_head")?,
            encoder_kernel: positive(encoder, "kernel_size")?,
            encoder_position_limit: positive(encoder, "pe_maxlen")?,
            decoder_layers: positive(decoder, "n_layers")?,
            decoder_heads: positive(decoder, "n_head")?,
            decoder_position_limit: positive(decoder, "pe_maxlen")?,
        };
        if config.input_dim != 80
            || config.vocab_size != 8667
            || config.model_dim != 1280
            || config.encoder_layers != 16
            || config.decoder_layers != 16
            || config.encoder_heads != 20
            || config.decoder_heads != 20
            || config.encoder_kernel != 33
            || config.encoder_position_limit < 5000
            || config.decoder_position_limit < 5000
        {
            return Err(SpeechError::Unsupported {
                why: "FireRedASR2 supports the pinned 80-bin, 16-block, width-1280 profile".into(),
            });
        }
        if [config.start_token, config.end_token, config.pad_token]
            .iter()
            .any(|&id| id >= config.vocab_size)
        {
            return Err(bad("token ids", "must be within the configured vocabulary"));
        }
        if config.model_dim % config.encoder_heads != 0
            || config.model_dim % config.decoder_heads != 0
            || config.encoder_kernel % 2 == 0
        {
            return Err(bad(
                "model dimensions",
                "unsupported attention or convolution geometry",
            ));
        }
        Ok(config)
    }
}
