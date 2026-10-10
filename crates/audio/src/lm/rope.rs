//! Rotary embeddings for the shared LM backbones.
//!
//! Port of `mlx_audio/lm/models/rope_utils.py` (vendored from mlx-lm) at the
//! pinned mlx-audio commit. MLX `mx.fast.rope` treats a passed `freqs` vector
//! as the per-pair angle denominator (`angle = position / freqs[d]`), verified
//! against the Python reference: an `inf` tail leaves the unrotated features
//! untouched, and `Llama3RoPE`'s stored vector reproduces
//! `cos(position / freqs[0])` for the first pair. The plain `nn.RoPE` path
//! (no scaling config) keeps `angle = position * base^(-2 d / dims)` via the
//! crate rope tables, which the music and Kokoro families already verified
//! against MLX.

use serde_json::Value;

use crate::nn::json as cfg;
use crate::ops;
use crate::{Result, SpeechError};

/// A rotary embedding: per-pair frequencies in the denominator convention,
/// a pre-rotation scale for the YaR/LongRoPE mscale, an angle multiplier for
/// the linear path, and the pair convention.
#[derive(Debug, Clone)]
pub struct Rope {
    pub traditional: bool,
    /// Feature count rotated by every application (`dims` in the reference).
    pub dims: usize,
    /// Denominator per rotated pair, length `dims / 2`. The plain path
    /// stores `base ** (+2 d / dims)` so the same division covers it.
    freqs: Vec<f32>,
    /// Multiplier applied to the features before rotation
    /// (`mscale` for YaRN, `_scale` for SuScaled, 1.0 otherwise).
    prescale: f32,
    /// Multiplier applied to every angle (the linear path divides the
    /// positions by `factor`; folding it here keeps one division site).
    angle_scale: f32,
}

impl Rope {
    /// Port of `initialize_rope`. `scaling` is the checkpoint's
    /// `rope_scaling` object; `max_position_embeddings` feeds the llama3 and
    /// YaR constructions exactly as the reference passes them.
    pub fn initialize(
        dims: usize,
        base: f32,
        traditional: bool,
        scaling: Option<&Value>,
        max_position_embeddings: Option<usize>,
    ) -> Result<Self> {
        let rope_type = scaling
            .map(|s| {
                s.get("type")
                    .or_else(|| s.get("rope_type"))
                    .and_then(Value::as_str)
                    .unwrap_or("default")
                    .to_owned()
            })
            .unwrap_or_else(|| "default".to_owned());
        let scaling = match scaling {
            Some(v) => v,
            None => &Value::Null,
        };
        match rope_type.as_str() {
            "default" => Ok(Self::plain(dims, base, traditional, 1.0)),
            "linear" => {
                let factor = cfg::required(scaling, "factor")?
                    .as_f64()
                    .ok_or_else(|| bad("factor", "must be numeric"))?;
                if factor <= 0.0 {
                    return Err(bad("factor", "must be positive"));
                }
                Ok(Self::plain(
                    dims,
                    base,
                    traditional,
                    (factor as f32).recip(),
                ))
            }
            "llama3" => {
                let factor = f32_field(scaling, "factor")?;
                let low_freq_factor = f32_field_or(scaling, "low_freq_factor", 1.0);
                let high_freq_factor = f32_field_or(scaling, "high_freq_factor", 4.0);
                let old_context_len = scaling
                    .get("original_max_position_embeddings")
                    .and_then(Value::as_f64)
                    .unwrap_or(8192.0) as f32;
                // Port of Llama3RoPE: wavelengths from the unscaled theta
                // values, low band scaled by factor, medium band smoothed,
                // high band untouched.
                let thetas = growing_freqs(dims, base);
                let low_freq_wavelen = old_context_len / low_freq_factor;
                let high_freq_wavelen = old_context_len / high_freq_factor;
                let mut freqs = Vec::with_capacity(dims / 2);
                for &theta in &thetas {
                    let wavelen = 2.0 * std::f32::consts::PI * theta;
                    let scaled = if wavelen > low_freq_wavelen {
                        theta * factor
                    } else if wavelen > high_freq_wavelen {
                        let smooth = (old_context_len / wavelen - low_freq_factor)
                            / (high_freq_factor - low_freq_factor);
                        theta / ((1.0 - smooth) / factor + smooth)
                    } else {
                        theta
                    };
                    freqs.push(scaled);
                }
                Ok(Self {
                    traditional,
                    dims,
                    freqs,
                    prescale: 1.0,
                    angle_scale: 1.0,
                })
            }
            "yarn" | "deepseek_yarn" | "telechat3-yarn" => {
                let scaling_factor = f32_field(scaling, "factor")?;
                let original_max_position_embeddings = scaling
                    .get("original_max_position_embeddings")
                    .and_then(Value::as_u64)
                    .unwrap_or(4096) as f32;
                let beta_fast = f32_field_or(scaling, "beta_fast", 32.0);
                let beta_slow = f32_field_or(scaling, "beta_slow", 1.0);
                let mscale = f32_field_or(scaling, "mscale", 1.0);
                let mscale_all_dim = f32_field_or(scaling, "mscale_all_dim", 0.0);
                // yarn_find_correction_range with the reference clamps.
                let correction = |num_rotations: f32| -> f32 {
                    dims as f32
                        * ((original_max_position_embeddings
                            / (num_rotations * 2.0 * std::f32::consts::PI))
                            .ln())
                        / (2.0 * (base).ln())
                };
                let low = correction(beta_fast).floor().max(0.0);
                let high = correction(beta_slow).ceil().min(dims as f32 - 1.0);
                let yarn_mscale = |scale: f32, m: f32| -> f32 {
                    if scale <= 1.0 {
                        1.0
                    } else {
                        0.1 * m * scale.ln() + 1.0
                    }
                };
                let prescale = yarn_mscale(scaling_factor, mscale)
                    / yarn_mscale(scaling_factor, mscale_all_dim);
                let thetas = growing_freqs(dims, base);
                let freqs = thetas
                    .iter()
                    .enumerate()
                    .map(|(d, &extra)| {
                        let inter = scaling_factor * extra;
                        // yarn_linear_ramp_mask over dims / 2 pairs; the
                        // singularity guard adds 0.001 to max.
                        let (min, max) = if (low - high).abs() < f32::EPSILON {
                            (low, high + 0.001)
                        } else {
                            (low, high)
                        };
                        let ramp = ((d as f32 - min) / (max - min)).clamp(0.0, 1.0);
                        let mask = 1.0 - ramp;
                        (inter * extra) / (inter * mask + extra * (1.0 - mask))
                    })
                    .collect();
                Ok(Self {
                    traditional,
                    dims,
                    freqs,
                    prescale,
                    angle_scale: 1.0,
                })
            }
            "longrope" => {
                // SuScaledRoPE: the reference always uses the long factor and
                // the long mscale; the short fields are accepted and unused,
                // exactly as in the Python.
                let original = cfg::required(scaling, "original_max_position_embeddings")?
                    .as_u64()
                    .ok_or_else(|| bad("original_max_position_embeddings", "must be an integer"))?
                    as f32;
                let long_factor = vector_field(scaling, "long_factor", dims / 2)?;
                let long_mscale = scaling
                    .get("long_mscale")
                    .and_then(Value::as_f64)
                    .map(|v| v as f32);
                let thetas = growing_freqs(dims, base);
                let freqs = thetas
                    .iter()
                    .zip(&long_factor)
                    .map(|(&theta, &factor)| theta * factor)
                    .collect();
                let factor = max_position_embeddings
                    .map(|m| m as f32 / original)
                    .unwrap_or(1.0);
                let prescale = long_mscale.unwrap_or_else(|| {
                    if factor <= 1.0 {
                        1.0
                    } else {
                        (1.0 + factor.ln() / original.ln()).sqrt()
                    }
                });
                Ok(Self {
                    traditional,
                    dims,
                    freqs,
                    prescale,
                    angle_scale: 1.0,
                })
            }
            "proportional" => {
                let partial = scaling
                    .get("partial_rotary_factor")
                    .and_then(Value::as_f64)
                    .unwrap_or(1.0);
                let factor = f32_field_or(scaling, "factor", 1.0);
                let rotated = (dims as f64 * partial) as usize;
                if rotated > dims || rotated % 2 != 0 {
                    return Err(bad(
                        "rope_scaling.partial_rotary_factor",
                        "must rotate an even count of features",
                    ));
                }
                // Exponents run over the full `dims` even when only the
                // leading `rotated` features rotate; an infinite denominator
                // leaves the tail untouched (verified: cos(0)).
                let thetas: Vec<f32> = (0..rotated / 2)
                    .map(|d| base.powf(2.0 * d as f32 / dims as f32))
                    .collect();
                let mut freqs = thetas
                    .iter()
                    .map(|&theta| theta / factor)
                    .collect::<Vec<f32>>();
                freqs.resize(dims / 2, f32::INFINITY);
                Ok(Self {
                    traditional,
                    dims,
                    freqs,
                    prescale: 1.0,
                    angle_scale: 1.0,
                })
            }
            "mrope" => {
                // The reference asserts three sections and then builds a
                // plain nn.RoPE; the sections are not used.
                let sections = cfg::required(scaling, "mrope_section")?
                    .as_array()
                    .ok_or_else(|| bad("mrope_section", "must be a list"))?;
                if sections.len() != 3 {
                    return Err(bad(
                        "rope_scaling.mrope_section",
                        "must have exactly 3 entries",
                    ));
                }
                Ok(Self::plain(dims, base, traditional, 1.0))
            }
            other => Err(SpeechError::Unsupported {
                why: format!("rope type {other}"),
            }),
        }
    }

    /// The plain `nn.RoPE` path: `angle = position * angle_scale /
    /// base ** (+2 d / dims)` which equals `position * base ** (-2 d /
    /// dims)` at scale 1.
    fn plain(dims: usize, base: f32, traditional: bool, angle_scale: f32) -> Self {
        Self {
            traditional,
            dims,
            freqs: growing_freqs(dims, base),
            prescale: 1.0,
            angle_scale,
        }
    }

    /// Rotates `x [heads, seq, >= dims]` in place for positions
    /// `offset .. offset + seq` using the crate rope kernels.
    pub fn apply(&self, x: &mut [f32], heads: usize, seq: usize, offset: usize) {
        let half = self.dims / 2;
        let mut cos = vec![0.0f32; seq * half];
        let mut sin = vec![0.0f32; seq * half];
        for row in 0..seq {
            let position = (offset + row) as f32 * self.angle_scale;
            for (d, &freq) in self.freqs.iter().enumerate() {
                let angle = if freq.is_infinite() {
                    0.0
                } else {
                    position / freq
                };
                cos[row * half + d] = angle.cos();
                sin[row * half + d] = angle.sin();
            }
        }
        if self.prescale != 1.0 {
            // The reference scales only the first `dims` features of every
            // row before rotating.
            let dim = x.len() / (heads * seq).max(1);
            for h in 0..heads {
                for t in 0..seq {
                    let base = (h * seq + t) * dim;
                    for v in &mut x[base..base + self.dims] {
                        *v *= self.prescale;
                    }
                }
            }
        }
        if self.traditional {
            ops::rope_interleaved(x, heads, seq, self.dims, &cos, &sin);
        } else {
            ops::rope_neox(x, heads, seq, self.dims, &cos, &sin);
        }
    }
}

/// `base ** (+2 d / dims)` for `d in 0..dims / 2`, the denominator-form
/// frequencies every reference construction starts from.
fn growing_freqs(dims: usize, base: f32) -> Vec<f32> {
    (0..dims / 2)
        .map(|d| base.powf(2.0 * d as f32 / dims as f32))
        .collect()
}

fn f32_field(v: &Value, key: &str) -> Result<f32> {
    cfg::required(v, key)?
        .as_f64()
        .map(|n| n as f32)
        .ok_or_else(|| bad(key, "must be numeric"))
}

fn f32_field_or(v: &Value, key: &str, default: f32) -> f32 {
    v.get(key)
        .and_then(Value::as_f64)
        .map(|n| n as f32)
        .unwrap_or(default)
}

fn vector_field(v: &Value, key: &str, len: usize) -> Result<Vec<f32>> {
    let list = cfg::required(v, key)?
        .as_array()
        .ok_or_else(|| bad(key, "must be a list"))?;
    if list.len() != len {
        return Err(bad(
            key,
            &format!("must have {len} entries for {len} rotated pairs"),
        ));
    }
    list.iter()
        .map(|item| {
            item.as_f64()
                .map(|n| n as f32)
                .ok_or_else(|| bad(key, "must contain numbers"))
        })
        .collect()
}

fn bad(field: &str, why: &str) -> SpeechError {
    SpeechError::BadConfig {
        field: format!("rope_scaling.{field}"),
        why: why.to_owned(),
    }
}
