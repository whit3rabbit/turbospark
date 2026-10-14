//! Sentence-transformers pooling over per-token hidden states.
//!
//! Mirrors `sentence_transformers.models.Pooling.forward` (5.1.2) for one
//! sequence at a time: `hidden [seq, dim]` and an attention `mask [seq]`
//! (`1` real token, `0` padding). The CPU path runs each text unpadded, but
//! the mask is still honored so a caller that pads (or drops the prompt
//! with `include_prompt = false`) gets the reference result.
//!
//! When several modes are enabled the outputs concatenate in the
//! reference order: cls, max, mean, mean_sqrt_len, weighted_mean,
//! last_token.

use crate::error::{input_error, EmbeddingError};
use crate::st_config::PoolingConfig;

/// Lower clamp on the mask sum (`torch.clamp(sum_mask, min=1e-9)`), so an
/// all-padding row divides by a tiny number instead of zero.
pub const MASK_SUM_FLOOR: f32 = 1e-9;

/// Value written into padded positions before max pooling (`-1e9` in the
/// reference). An all-padding row therefore pools to `-1e9`, not `-inf`.
pub const MAX_POOL_PAD: f32 = -1e9;

/// One pooling reduction. [`PoolingConfig`] carries the set the checkpoint
/// enables.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PoolingMode {
    /// First token (`token_embeddings[:, 0]`), mask ignored.
    Cls,
    /// Per-dimension max over real tokens.
    Max,
    /// Mask-weighted mean.
    Mean,
    /// Masked sum divided by `sqrt(token count)`.
    MeanSqrtLen,
    /// Position-weighted mean with weights `1..=seq`.
    WeightedMean,
    /// Last real token (last `1` in the mask).
    LastToken,
}

fn check(hidden: &[f32], mask: &[u32], dim: usize) -> Result<usize, EmbeddingError> {
    if dim == 0 {
        return Err(input_error("pooling dim is zero"));
    }
    if hidden.len() % dim != 0 {
        return Err(input_error(format!(
            "hidden length {} is not a multiple of dim {dim}",
            hidden.len()
        )));
    }
    let seq = hidden.len() / dim;
    if mask.len() != seq {
        return Err(input_error(format!(
            "mask length {} != sequence length {seq}",
            mask.len()
        )));
    }
    Ok(seq)
}

/// Applies one pooling mode to `hidden [seq, dim]` under `mask [seq]`.
pub fn pool_one(
    mode: PoolingMode,
    hidden: &[f32],
    mask: &[u32],
    dim: usize,
) -> Result<Vec<f32>, EmbeddingError> {
    let seq = check(hidden, mask, dim)?;
    let row = |t: usize| &hidden[t * dim..(t + 1) * dim];
    let on = |t: usize| if mask[t] != 0 { 1.0f32 } else { 0.0 };
    Ok(match mode {
        PoolingMode::Cls => {
            if seq == 0 {
                return Err(input_error("cls pooling over an empty sequence"));
            }
            row(0).to_vec()
        }
        PoolingMode::Max => {
            let mut out = vec![f32::NEG_INFINITY; dim];
            for (t, &m) in mask.iter().enumerate() {
                for (o, &v) in out.iter_mut().zip(row(t)) {
                    let v = if m != 0 { v } else { MAX_POOL_PAD };
                    *o = o.max(v);
                }
            }
            out
        }
        PoolingMode::Mean | PoolingMode::MeanSqrtLen => {
            let mut sum = vec![0.0f32; dim];
            let mut count = 0.0f32;
            for t in 0..seq {
                let w = on(t);
                count += w;
                for (s, &v) in sum.iter_mut().zip(row(t)) {
                    *s += v * w;
                }
            }
            let count = count.max(MASK_SUM_FLOOR);
            let denom = if mode == PoolingMode::Mean {
                count
            } else {
                count.sqrt()
            };
            sum.iter().map(|s| s / denom).collect()
        }
        PoolingMode::WeightedMean => {
            let mut sum = vec![0.0f32; dim];
            let mut weight_sum = 0.0f32;
            for t in 0..seq {
                let w = on(t) * (t + 1) as f32;
                weight_sum += w;
                for (s, &v) in sum.iter_mut().zip(row(t)) {
                    *s += v * w;
                }
            }
            let weight_sum = weight_sum.max(MASK_SUM_FLOOR);
            sum.iter().map(|s| s / weight_sum).collect()
        }
        PoolingMode::LastToken => {
            if seq == 0 {
                return Err(input_error("last-token pooling over an empty sequence"));
            }
            // The reference gathers the last `1`; with no `1` at all it
            // gathers index 0 and then masks it, giving zeros.
            match (0..seq).rev().find(|&t| mask[t] != 0) {
                Some(t) => row(t).to_vec(),
                None => vec![0.0; dim],
            }
        }
    })
}

/// Applies every mode `config` enables, concatenated in reference order.
/// When `config.include_prompt` is false, the first `prompt_len` mask
/// entries are zeroed first, as the reference does.
pub fn pool(
    config: &PoolingConfig,
    hidden: &[f32],
    mask: &[u32],
    prompt_len: usize,
) -> Result<Vec<f32>, EmbeddingError> {
    let dim = config.word_embedding_dimension;
    let modes = config.modes();
    if modes.is_empty() {
        return Err(EmbeddingError::Unsupported {
            why: "pooling config enables no pooling mode".into(),
        });
    }
    let mut owned;
    let mask = if !config.include_prompt && prompt_len > 0 {
        owned = mask.to_vec();
        let n = prompt_len.min(owned.len());
        owned[..n].fill(0);
        &owned[..]
    } else {
        mask
    };
    let mut out = Vec::with_capacity(dim * modes.len());
    for mode in modes {
        out.extend(pool_one(mode, hidden, mask, dim)?);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    // hidden [3, 2]: rows (1,2), (3,4), (5,6); mask drops the last row.
    const H: [f32; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    const M: [u32; 3] = [1, 1, 0];

    #[test]
    fn mean_ignores_padding() {
        let v = pool_one(PoolingMode::Mean, &H, &M, 2).unwrap();
        assert_eq!(v, vec![2.0, 3.0]);
    }

    #[test]
    fn mean_sqrt_len_divides_by_root_count() {
        let v = pool_one(PoolingMode::MeanSqrtLen, &H, &M, 2).unwrap();
        let r = 2.0f32.sqrt();
        assert_eq!(v, vec![4.0 / r, 6.0 / r]);
    }

    #[test]
    fn max_uses_pad_floor() {
        let v = pool_one(PoolingMode::Max, &H, &M, 2).unwrap();
        assert_eq!(v, vec![3.0, 4.0]);
        let all_pad = pool_one(PoolingMode::Max, &H, &[0, 0, 0], 2).unwrap();
        assert_eq!(all_pad, vec![MAX_POOL_PAD, MAX_POOL_PAD]);
    }

    #[test]
    fn weighted_mean_uses_positions() {
        // weights 1, 2 (third masked): (1*1+3*2)/3, (2*1+4*2)/3
        let v = pool_one(PoolingMode::WeightedMean, &H, &M, 2).unwrap();
        assert_eq!(v, vec![7.0 / 3.0, 10.0 / 3.0]);
    }

    #[test]
    fn last_token_finds_last_real_row() {
        let v = pool_one(PoolingMode::LastToken, &H, &M, 2).unwrap();
        assert_eq!(v, vec![3.0, 4.0]);
        // Left padding: last real row is still the last 1.
        let v = pool_one(PoolingMode::LastToken, &H, &[0, 1, 1], 2).unwrap();
        assert_eq!(v, vec![5.0, 6.0]);
        let v = pool_one(PoolingMode::LastToken, &H, &[0, 0, 0], 2).unwrap();
        assert_eq!(v, vec![0.0, 0.0]);
    }

    #[test]
    fn cls_ignores_mask() {
        let v = pool_one(PoolingMode::Cls, &H, &[0, 1, 1], 2).unwrap();
        assert_eq!(v, vec![1.0, 2.0]);
    }

    #[test]
    fn mean_of_all_padding_is_zero_not_nan() {
        let v = pool_one(PoolingMode::Mean, &H, &[0, 0, 0], 2).unwrap();
        assert_eq!(v, vec![0.0, 0.0]);
    }

    #[test]
    fn exclude_prompt_masks_prefix() {
        let mut config = PoolingConfig::mean(2);
        config.include_prompt = false;
        let v = pool(&config, &H, &[1, 1, 1], 1).unwrap();
        assert_eq!(v, vec![4.0, 5.0]);
        config.include_prompt = true;
        let v = pool(&config, &H, &[1, 1, 1], 1).unwrap();
        assert_eq!(v, vec![3.0, 4.0]);
    }

    #[test]
    fn multiple_modes_concatenate_in_reference_order() {
        let mut config = PoolingConfig::mean(2);
        config.cls = true;
        config.max = true;
        let v = pool(&config, &H, &M, 0).unwrap();
        // cls, max, mean
        assert_eq!(v, vec![1.0, 2.0, 3.0, 4.0, 2.0, 3.0]);
    }

    #[test]
    fn shape_errors_are_reported() {
        assert!(pool_one(PoolingMode::Mean, &H, &[1, 1], 2).is_err());
        assert!(pool_one(PoolingMode::Mean, &H[..5], &M, 2).is_err());
        assert!(pool_one(PoolingMode::Mean, &H, &M, 0).is_err());
    }
}
