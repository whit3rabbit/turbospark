//! Fluid Speech Pro: Parakeet TDT 0.6B v3 (via moondream/parakeet-ultra)
//! with one extra encoder block and an optional "clean" adapter.
//!
//! Reference: `fluid_speech_mlx.py` in `FluidVoice-Labs/fluid-speech-pro-mlx`
//! at `aeb1d4e43487586b15d86c0a1dd4511e1cf6b7a0` (parakeet-mlx 0.7.0).
//!
//! - `FrameCleaner` sits at `encoder.pre_encode.fv_insert`, after the
//!   subsampling projection and before the first Conformer layer:
//!   `x + up(gelu(depthwise_conv(down(layer_norm(x)))))`. The depthwise
//!   conv looks `kernel / 2` frames ahead, so the model is offline only.
//! - Clean mode adds `alpha / rank * B @ A` to each adapted encoder linear
//!   and folds two vectors into biases: `decoder_input` through the first
//!   LSTM layer's `Wx`, and `joint_hidden` into `joint.pred.bias`. The
//!   reference keeps quantized layers quantized and adds the low-rank term at
//!   run time; here every weight is f32 after load, so the term is folded.

use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use serde_json::Value;
use turbospark_model_io::safetensors::SafetensorsFile;

use crate::models::vad::sortformer::ParakeetWeights;
use crate::ops;
use crate::{Result, SpeechError};

/// MLX `nn.LayerNorm` default epsilon, which the reference block uses.
const LAYER_NORM_EPS: f32 = 1e-5;
/// File name of the clean-mode adapter beside the base weights.
pub const CLEAN_MODE_FILE: &str = "clean-mode.safetensors";

/// The `fluid_speech` block of `config.json`.
#[derive(Debug, Clone, PartialEq)]
pub struct FluidSpeechConfig {
    pub insert_dim: usize,
    pub insert_kernel: usize,
    pub clean_rank: usize,
    pub clean_alpha: f32,
}

fn bad(field: &str, why: impl Into<String>) -> SpeechError {
    SpeechError::BadConfig {
        field: format!("fluid_speech.{field}"),
        why: why.into(),
    }
}

impl FluidSpeechConfig {
    /// Parses the block. Only the published geometry is accepted: a
    /// different insert position or kernel is a different graph.
    pub fn parse(v: &Value, hidden_size: usize) -> Result<Self> {
        if v.get("insert").and_then(Value::as_str) != Some("pre") {
            return Err(bad("insert", "only the pre-Conformer insert is supported"));
        }
        let int = |key: &str| -> Result<usize> {
            v.get(key)
                .and_then(Value::as_u64)
                .and_then(|x| usize::try_from(x).ok())
                .ok_or_else(|| bad(key, "must be an unsigned integer"))
        };
        let insert_dim = int("insert_dim")?;
        let insert_kernel = int("insert_kernel")?;
        if insert_dim != hidden_size || insert_kernel != 15 {
            return Err(SpeechError::Unsupported {
                why: format!(
                    "Fluid Speech Pro FrameCleaner must be {hidden_size} wide with kernel 15, \
                     found {insert_dim} and {insert_kernel}"
                ),
            });
        }
        let clean = v
            .get("clean_mode")
            .ok_or_else(|| bad("clean_mode", "missing"))?;
        let clean_rank = clean
            .get("rank")
            .and_then(Value::as_u64)
            .and_then(|x| usize::try_from(x).ok())
            .filter(|&x| x > 0)
            .ok_or_else(|| bad("clean_mode.rank", "must be a positive integer"))?;
        let clean_alpha = clean
            .get("alpha")
            .and_then(Value::as_f64)
            .filter(|x| x.is_finite())
            .ok_or_else(|| bad("clean_mode.alpha", "must be a finite number"))?
            as f32;
        Ok(Self {
            insert_dim,
            insert_kernel,
            clean_rank,
            clean_alpha,
        })
    }
}

/// The residual block between subsampling and the Conformer stack.
pub(super) struct FrameCleaner {
    ln_w: Vec<f32>,
    ln_b: Vec<f32>,
    down_w: Vec<f32>,
    down_b: Vec<f32>,
    /// Depthwise kernel `[dim, kernel]` (MLX stores `[dim, kernel, 1]`).
    mix_w: Vec<f32>,
    mix_b: Vec<f32>,
    up_w: Vec<f32>,
    up_b: Vec<f32>,
    width: usize,
    dim: usize,
    kernel: usize,
}

const PREFIX: &str = "encoder.pre_encode.fv_insert";

fn tensor(file: &SafetensorsFile, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
    let desc = file.descriptor(name).ok_or_else(|| SpeechError::Tensor {
        name: name.to_string(),
        why: "missing from safetensors".into(),
    })?;
    if desc.shape != shape {
        return Err(SpeechError::Tensor {
            name: name.to_string(),
            why: format!("expected shape {shape:?}, found {:?}", desc.shape),
        });
    }
    file.load_as_f32(name).map_err(|e| SpeechError::Tensor {
        name: name.to_string(),
        why: format!("load failed: {e}"),
    })
}

fn required_bias(bias: Option<Vec<f32>>, name: &str, len: usize) -> Result<Vec<f32>> {
    match bias {
        Some(b) if b.len() == len => Ok(b),
        Some(b) => Err(SpeechError::Tensor {
            name: name.into(),
            why: format!("expected {len} values, found {}", b.len()),
        }),
        None => Err(SpeechError::Tensor {
            name: name.into(),
            why: "missing from safetensors".into(),
        }),
    }
}

impl FrameCleaner {
    pub(super) fn load(weights: &ParakeetWeights<'_>, cfg: &FluidSpeechConfig) -> Result<Self> {
        let (width, dim, kernel) = (cfg.insert_dim, cfg.insert_dim, cfg.insert_kernel);
        let file = weights.file;
        let linear = |name: &str, out: usize, inp: usize| -> Result<(Vec<f32>, Vec<f32>)> {
            let base = format!("{PREFIX}.{name}");
            let (w, b) = weights.linear(&base)?;
            if w.len() != out * inp {
                return Err(SpeechError::Tensor {
                    name: format!("{base}.weight"),
                    why: format!("expected {out}x{inp} values, found {}", w.len()),
                });
            }
            Ok((w, required_bias(b, &format!("{base}.bias"), out)?))
        };
        let (down_w, down_b) = linear("down", dim, width)?;
        let (up_w, up_b) = linear("up", width, dim)?;
        Ok(Self {
            ln_w: tensor(file, &format!("{PREFIX}.ln.weight"), &[width])?,
            ln_b: tensor(file, &format!("{PREFIX}.ln.bias"), &[width])?,
            down_w,
            down_b,
            mix_w: tensor(file, &format!("{PREFIX}.mix.weight"), &[dim, kernel, 1])?,
            mix_b: tensor(file, &format!("{PREFIX}.mix.bias"), &[dim])?,
            up_w,
            up_b,
            width,
            dim,
            kernel,
        })
    }

    /// Applies the block in place to t-major `[frames, width]` features.
    pub(super) fn forward(&self, x: &mut [f32], frames: usize) -> Result<()> {
        if x.len() != frames * self.width {
            return Err(SpeechError::Input {
                why: format!(
                    "FrameCleaner input has {} values, expected {frames}x{}",
                    x.len(),
                    self.width
                ),
            });
        }
        let mut h = x.to_vec();
        ops::layernorm(
            &mut h,
            frames,
            self.width,
            &self.ln_w,
            Some(&self.ln_b),
            LAYER_NORM_EPS,
        );
        let down = ops::linear(
            &h,
            &self.down_w,
            Some(&self.down_b),
            frames,
            self.width,
            self.dim,
        );
        let mut mixed = depthwise_conv_time(
            &down,
            &self.mix_w,
            &self.mix_b,
            frames,
            self.dim,
            self.kernel,
        );
        ops::gelu_erf(&mut mixed);
        let up = ops::linear(
            &mixed,
            &self.up_w,
            Some(&self.up_b),
            frames,
            self.dim,
            self.width,
        );
        for (v, u) in x.iter_mut().zip(up) {
            *v += u;
        }
        Ok(())
    }
}

/// Depthwise conv over time on t-major `[frames, dim]` input with zero
/// padding `kernel / 2` on both sides (same output length).
fn depthwise_conv_time(
    x: &[f32],
    w: &[f32],
    b: &[f32],
    frames: usize,
    dim: usize,
    kernel: usize,
) -> Vec<f32> {
    let pad = kernel / 2;
    let mut out = vec![0.0f32; frames * dim];
    for t in 0..frames {
        let row = &mut out[t * dim..(t + 1) * dim];
        row.copy_from_slice(b);
        for j in 0..kernel {
            let Some(src) = (t + j).checked_sub(pad).filter(|&s| s < frames) else {
                continue;
            };
            let input = &x[src * dim..(src + 1) * dim];
            for c in 0..dim {
                row[c] += w[c * kernel + j] * input[c];
            }
        }
    }
    out
}

/// The clean-mode adapter: per-module `A [rank, in]` and `B [out, rank]`,
/// plus the two bias vectors. `used` records which modules were folded so
/// open can refuse an adapter that names a module the loader never saw.
pub(super) struct CleanAdapter {
    pairs: BTreeMap<String, (Vec<f32>, Vec<f32>)>,
    rank: usize,
    scale: f32,
    pub(super) decoder_input: Vec<f32>,
    pub(super) joint_hidden: Vec<f32>,
    used: RefCell<BTreeSet<String>>,
}

impl CleanAdapter {
    pub(super) fn open(model_dir: &Path, cfg: &FluidSpeechConfig) -> Result<Self> {
        let path = model_dir.join(CLEAN_MODE_FILE);
        let file = SafetensorsFile::open(&path)?;
        let rank = cfg.clean_rank;
        let mut pairs = BTreeMap::new();
        let mut decoder_input = None;
        let mut joint_hidden = None;
        let mut names: Vec<String> = file.tensor_names().map(str::to_owned).collect();
        names.sort();
        for name in &names {
            match name.as_str() {
                "decoder_input" => decoder_input = Some(file.load_as_f32(name)?),
                "joint_hidden" => joint_hidden = Some(file.load_as_f32(name)?),
                other => {
                    let Some(module) = other.strip_suffix(".A") else {
                        if other.ends_with(".B") {
                            continue;
                        }
                        return Err(SpeechError::Tensor {
                            name: other.into(),
                            why: "unexpected tensor in the clean-mode adapter".into(),
                        });
                    };
                    let b_name = format!("{module}.B");
                    let a_desc = file.descriptor(other).ok_or_else(|| SpeechError::Tensor {
                        name: other.into(),
                        why: "missing descriptor".into(),
                    })?;
                    let b_desc = file
                        .descriptor(&b_name)
                        .ok_or_else(|| SpeechError::Tensor {
                            name: b_name.clone(),
                            why: "missing the matching .B tensor".into(),
                        })?;
                    if a_desc.shape.len() != 2
                        || b_desc.shape.len() != 2
                        || a_desc.shape[0] != rank
                        || b_desc.shape[1] != rank
                    {
                        return Err(SpeechError::Tensor {
                            name: module.into(),
                            why: format!(
                                "expected A [{rank}, in] and B [out, {rank}], found {:?} and {:?}",
                                a_desc.shape, b_desc.shape
                            ),
                        });
                    }
                    pairs.insert(
                        format!("encoder.{module}"),
                        (file.load_as_f32(other)?, file.load_as_f32(&b_name)?),
                    );
                }
            }
        }
        // Every .B must have had an .A.
        for name in &names {
            if let Some(module) = name.strip_suffix(".B") {
                if !pairs.contains_key(&format!("encoder.{module}")) {
                    return Err(SpeechError::Tensor {
                        name: name.clone(),
                        why: "missing the matching .A tensor".into(),
                    });
                }
            }
        }
        let missing = |what: &str| SpeechError::Tensor {
            name: what.into(),
            why: "missing from the clean-mode adapter".into(),
        };
        Ok(Self {
            pairs,
            rank,
            scale: cfg.clean_alpha / rank as f32,
            decoder_input: decoder_input.ok_or_else(|| missing("decoder_input"))?,
            joint_hidden: joint_hidden.ok_or_else(|| missing("joint_hidden"))?,
            used: RefCell::new(BTreeSet::new()),
        })
    }

    /// Folds `scale * B @ A` into the dense `[out, in]` weight of `module`
    /// when the adapter covers it; other modules are left unchanged.
    pub(super) fn fold(&self, module: &str, weight: &mut [f32]) -> Result<()> {
        let Some((a, b)) = self.pairs.get(module) else {
            return Ok(());
        };
        let r = self.rank;
        let inp = a.len() / r;
        let out = b.len() / r;
        if weight.len() != out * inp {
            return Err(SpeechError::Tensor {
                name: module.into(),
                why: format!(
                    "clean-mode update is {out}x{inp}, the weight has {} values",
                    weight.len()
                ),
            });
        }
        for o in 0..out {
            let b_row = &b[o * r..(o + 1) * r];
            let w_row = &mut weight[o * inp..(o + 1) * inp];
            for (k, &bk) in b_row.iter().enumerate() {
                let coeff = self.scale * bk;
                for (w, &av) in w_row.iter_mut().zip(&a[k * inp..(k + 1) * inp]) {
                    *w += coeff * av;
                }
            }
        }
        self.used.borrow_mut().insert(module.to_owned());
        Ok(())
    }

    /// Refuses an adapter with modules the encoder loader never asked for:
    /// a renamed module would otherwise be dropped silently.
    pub(super) fn check_all_used(&self) -> Result<()> {
        let used = self.used.borrow();
        if let Some(unused) = self.pairs.keys().find(|k| !used.contains(*k)) {
            return Err(SpeechError::Tensor {
                name: unused.clone(),
                why: "clean-mode module does not match any encoder layer".into(),
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn depthwise_conv_matches_direct_sum() {
        // frames 4, dim 2, kernel 3: out[t][c] = b[c] + sum_j w[c][j] x[t+j-1][c].
        let x = [1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0];
        let w = [1.0, 2.0, 3.0, 0.5, 0.0, -0.5];
        let b = [0.25, -1.0];
        let out = depthwise_conv_time(&x, &w, &b, 4, 2, 3);
        let want = [
            0.25 + 2.0 * 1.0 + 3.0 * 2.0,
            -1.0 - 0.5 * 20.0,
            0.25 + 1.0 + 2.0 * 2.0 + 3.0 * 3.0,
            -1.0 + 0.5 * 10.0 - 0.5 * 30.0,
            0.25 + 2.0 + 2.0 * 3.0 + 3.0 * 4.0,
            -1.0 + 0.5 * 20.0 - 0.5 * 40.0,
            0.25 + 3.0 + 2.0 * 4.0,
            -1.0 + 0.5 * 30.0,
        ];
        assert_eq!(out, want);
    }

    #[test]
    fn config_refuses_other_geometry() {
        let ok = serde_json::json!({
            "insert": "pre", "insert_dim": 1024, "insert_kernel": 15,
            "clean_mode": {"rank": 32, "alpha": 64.0}
        });
        let cfg = FluidSpeechConfig::parse(&ok, 1024).unwrap();
        assert_eq!(cfg.clean_rank, 32);
        assert_eq!(cfg.clean_alpha, 64.0);
        for (key, value) in [
            ("insert", serde_json::json!("post")),
            ("insert_kernel", serde_json::json!(9)),
            ("insert_dim", serde_json::json!(512)),
        ] {
            let mut v = ok.clone();
            v[key] = value;
            assert!(FluidSpeechConfig::parse(&v, 1024).is_err(), "{key}");
        }
    }
}
