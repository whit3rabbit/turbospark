//! GGUF KV heads and linear attention configuration extraction.

use model_io::LinearAttentionConfig;

use super::meta::Meta;
use super::GgufConfigError;

/// Gemma 4 publishes `head_count_kv` as a PER-LAYER ARRAY, because its
/// sliding-window and global layers differ (8 vs 2). Qwen publishes a
/// scalar. Returns `(sliding, full)`.
pub fn kv_heads(m: &Meta<'_>, mask: &[u8]) -> Result<(i64, i64), GgufConfigError> {
    let key = m.key("attention.head_count_kv");
    let value = m
        .opt("attention.head_count_kv")
        .ok_or(GgufConfigError::MissingKey { key: key.clone() })?;

    if let Some(scalar) = value.as_u64() {
        return Ok((scalar as i64, scalar as i64));
    }
    let per_layer = value.as_array().ok_or_else(|| GgufConfigError::BadValue {
        key: key.clone(),
        detail: "neither an integer nor an array".to_string(),
    })?;
    if per_layer.len() != mask.len() {
        return Err(GgufConfigError::BadValue {
            key,
            detail: format!("{} entries for {} layers", per_layer.len(), mask.len()),
        });
    }
    let at = |want: u8| -> Option<i64> {
        mask.iter()
            .position(|k| *k == want)
            .and_then(|i| per_layer[i].as_u64())
            .map(|v| v as i64)
    };
    // A model with only one layer kind reuses that kind's count for both,
    // which is what the Qwen scalar path does too.
    let sliding = at(0);
    let full = at(1);
    match (sliding, full) {
        (Some(s), Some(f)) => Ok((s, f)),
        (Some(s), None) => Ok((s, s)),
        (None, Some(f)) => Ok((f, f)),
        (None, None) => Err(GgufConfigError::BadValue {
            key,
            detail: "no layer kind matched the per-layer array".to_string(),
        }),
    }
}

/// Qwen's gated-DeltaNet dimensions, which GGUF publishes under the `ssm.`
/// keys it borrows the tensor slots from.
pub fn linear_attention(m: &Meta<'_>) -> Result<LinearAttentionConfig, GgufConfigError> {
    let inner = m.i64("ssm.inner_size")?;
    let num_v_heads = m.i64("ssm.time_step_rank")?;
    if num_v_heads == 0 {
        return Err(GgufConfigError::BadValue {
            key: m.key("ssm.time_step_rank"),
            detail: "must be non-zero".to_string(),
        });
    }
    Ok(LinearAttentionConfig {
        num_k_heads: m.i64("ssm.group_count")?,
        num_v_heads,
        key_head_dim: m.i64("ssm.state_size")?,
        // `inner_size` is the whole value stream, so the per-head width is
        // implied rather than published.
        value_head_dim: inner / num_v_heads,
        conv_kernel_size: m.i64("ssm.conv_kernel")?,
        output_gate_sigmoid: false,
    })
}
