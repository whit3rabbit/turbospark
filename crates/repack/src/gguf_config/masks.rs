//! Layer attention kind mask builders across GGUF model families.

use crate::gguf_header::GgufValue;

use super::meta::Meta;
use super::GgufConfigError;

/// `gpt-oss`'s alternating window (ROADMAP M5), derived rather than read.
///
/// This is the third spelling of "which layers slide" in this file and the
/// only one the file does not state directly. Gemma ships a per-layer BOOL
/// ARRAY; Qwen 3.6 a period; `gpt-oss` ships `attention.sliding_window` and
/// NOTHING ELSE, and llama.cpp's loader supplies the rest: an absent
/// `attention.sliding_window_pattern` means period 2, and
/// `llama_hparams::set_swa_pattern(2, dense_first = false)` computes
/// `is_swa[il] = (il % 2) < 1`. So EVEN layers slide.
///
/// AGENTS.md Gotcha 39's rule applies twice here. The default belongs to the
/// FORMAT (llama.cpp's 2), not to a neighbouring family, and a file that DOES
/// publish a period must be believed over it -- which is why the key is read
/// rather than assumed even though the shipped checkpoints omit it. Getting
/// the phase inverted yields a model wrong only past 128 tokens of context.
pub fn gpt_oss_layer_mask(m: &Meta<'_>, num_layers: usize) -> Result<Vec<u8>, GgufConfigError> {
    let window = m.i64("attention.sliding_window")?;
    if window <= 0 {
        return Err(GgufConfigError::BadValue {
            key: m.key("attention.sliding_window"),
            detail: format!("{window} is not a usable sliding window"),
        });
    }
    let period = m.opt_i64("attention.sliding_window_pattern").unwrap_or(2);
    if period < 1 {
        return Err(GgufConfigError::BadValue {
            key: m.key("attention.sliding_window_pattern"),
            detail: format!("period {period} is not usable"),
        });
    }
    Ok((0..num_layers)
        .map(|i| u8::from(i as i64 % period >= period - 1))
        .collect())
}

/// Layer kinds, matching `ArchConfig::full_attention_layer_mask`:
/// 0 = sliding-window, 1 = full attention, 2 = gated-DeltaNet linear.
pub fn gemma4_layer_mask(m: &Meta<'_>, num_layers: usize) -> Result<Vec<u8>, GgufConfigError> {
    let key = m.key("attention.sliding_window_pattern");
    let pattern = m
        .opt("attention.sliding_window_pattern")
        .and_then(GgufValue::as_array)
        .ok_or(GgufConfigError::MissingKey { key: key.clone() })?;
    if pattern.len() != num_layers {
        return Err(GgufConfigError::BadValue {
            key,
            detail: format!("{} entries for {num_layers} layers", pattern.len()),
        });
    }
    pattern
        .iter()
        .map(|v| {
            // true means "this layer slides", i.e. mask 0. The inversion is
            // the whole content of this function and is easy to get exactly
            // backwards, which would silently swap every layer's attention
            // kind and its rope base.
            v.as_bool()
                .map(|sliding| u8::from(!sliding))
                .ok_or_else(|| GgufConfigError::BadValue {
                    key: key.clone(),
                    detail: "entries are not booleans".to_string(),
                })
        })
        .collect()
}

pub fn qwen_gdn_moe_layer_mask(
    m: &Meta<'_>,
    num_layers: usize,
) -> Result<Vec<u8>, GgufConfigError> {
    let interval = m.u64("full_attention_interval")? as usize;
    if interval == 0 {
        return Err(GgufConfigError::BadValue {
            key: m.key("full_attention_interval"),
            detail: "must be non-zero".to_string(),
        });
    }
    // Every `interval`-th layer counting from one: 3, 7, 11, ... at 4.
    Ok((0..num_layers)
        .map(|i| if (i + 1) % interval == 0 { 1u8 } else { 2u8 })
        .collect())
}
