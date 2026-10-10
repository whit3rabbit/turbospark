//! The VibeVoice diffusion prediction head kugelaudio reuses
//! (`mlx_audio/tts/models/vibevoice/diffusion_head.py`).
//!
//! Reference: mlx-audio at commit
//! [`e1b19b9054bf163f5d812221a54fcc346f1890e9`](https://github.com/Blaizzy/mlx-audio/tree/e1b19b9054bf163f5d812221a54fcc346f1890e9/mlx_audio/tts/models/vibevoice/diffusion_head.py).
//! `noisy_images_proj` lifts the noisy latent, `cond_proj` projects the
//! conditioning hidden state, and the sinusoidal timestep embedding adds
//! to it; `head_layers` AdaLN blocks (shift/scale/gate modulation of an
//! RMSNorm before a SwiGLU feed forward) end in a final AdaLN layer
//! over an affine-free norm. All linears are bias-free.

use crate::nn::{Linear, RmsNorm};
use crate::ops;
use crate::{Result, SpeechError};
use turbospark_model_io::safetensors::SafetensorsFile;

fn load_sharded(files: &[SafetensorsFile], name: &str, shape: &[usize]) -> Result<Vec<f32>> {
    let file = files
        .iter()
        .find(|file| file.contains_tensor(name))
        .ok_or_else(|| SpeechError::Tensor {
            name: name.to_owned(),
            why: "tensor is missing".into(),
        })?;
    let descriptor = file.descriptor(name).ok_or_else(|| SpeechError::Tensor {
        name: name.to_owned(),
        why: "tensor is missing".into(),
    })?;
    if descriptor.shape != shape {
        return Err(SpeechError::Tensor {
            name: name.to_owned(),
            why: format!("expected shape {shape:?}, got {:?}", descriptor.shape),
        });
    }
    Ok(file.load_as_f32(name)?)
}

/// Loads a bias-free linear under both converted key spellings (the
/// converted checkpoints carry the sanitize-renamed `mlp.layers.{i}` /
/// `adaLN_modulation.layers.{i}` names; raw saves keep the numeric
/// indices).
fn load_head_linear(
    files: &[SafetensorsFile],
    name: &str,
    input: usize,
    output: usize,
) -> Result<Linear> {
    let weight = load_sharded(files, &format!("{name}.weight"), &[output, input])?;
    Ok(Linear::new(weight, None, input, output))
}

fn load_layer_linear(
    files: &[SafetensorsFile],
    name: &str,
    input: usize,
    output: usize,
) -> Result<Linear> {
    if files
        .iter()
        .any(|file| file.contains_tensor(&format!("{name}.weight")))
    {
        load_head_linear(files, name, input, output)
    } else {
        // `prediction_head.t_embedder.mlp.0.weight` and
        // `prediction_head.layers.{i}.adaLN_modulation.1.weight` style
        // Sequential indices.
        let renamed = name
            .replace("mlp.layers.", "mlp.")
            .replace("adaLN_modulation.layers.", "adaLN_modulation.");
        load_head_linear(files, &renamed, input, output)
    }
}

/// `TimestepEmbedder.timestep_embedding`: half the frequencies as
/// `exp(-ln(max_period) * d / half)`, then `concat(cos, sin)` over the
/// scaled positions (the dim is even for the pinned 256).
fn timestep_embedding(t: f32, dim: usize, max_period: f32) -> Vec<f32> {
    let half = dim / 2;
    let mut out = vec![0.0f32; dim];
    for (d, slot) in out[..half].iter_mut().enumerate() {
        let freq = (-max_period.ln() * d as f32 / half as f32).exp();
        *slot = (t * freq).cos();
    }
    for (d, slot) in out[half..].iter_mut().enumerate() {
        let freq = (-max_period.ln() * d as f32 / half as f32).exp();
        *slot = (t * freq).sin();
    }
    out
}

pub(crate) struct TimestepEmbedder {
    linear1: Linear,
    linear2: Linear,
    frequency_embedding_size: usize,
}

impl TimestepEmbedder {
    fn load(files: &[SafetensorsFile], name: &str, hidden: usize) -> Result<Self> {
        let frequency_embedding_size = 256;
        Ok(Self {
            linear1: load_layer_linear(
                files,
                &format!("{name}.mlp.layers.0"),
                frequency_embedding_size,
                hidden,
            )?,
            linear2: load_layer_linear(files, &format!("{name}.mlp.layers.2"), hidden, hidden)?,
            frequency_embedding_size,
        })
    }

    /// `t [batch] -> [batch, hidden]`.
    pub(crate) fn forward(&self, timesteps: &[f32], rows: usize) -> Vec<f32> {
        let mut freqs = Vec::with_capacity(rows * self.frequency_embedding_size);
        for &t in timesteps {
            freqs.extend(timestep_embedding(
                t,
                self.frequency_embedding_size,
                10000.0,
            ));
        }
        let mut hidden = self.linear1.forward(&freqs, rows);
        ops::silu(&mut hidden);
        self.linear2.forward(&hidden, rows)
    }
}

/// One AdaLN block: `x + gate * ffn(modulate(norm(x), shift, scale))`
/// with the modulation linear producing shift, scale, gate thirds.
pub(crate) struct HeadLayer {
    pub(crate) ffn_gate: Linear,
    pub(crate) ffn_up: Linear,
    pub(crate) ffn_down: Linear,
    pub(crate) norm: RmsNorm,
    pub(crate) adaln: Linear,
    embed_dim: usize,
}

impl HeadLayer {
    fn load(
        files: &[SafetensorsFile],
        name: &str,
        embed_dim: usize,
        ffn_dim: usize,
        cond_dim: usize,
        norm_eps: f32,
    ) -> Result<Self> {
        Ok(Self {
            ffn_gate: load_layer_linear(
                files,
                &format!("{name}.ffn.gate_proj"),
                embed_dim,
                ffn_dim,
            )?,
            ffn_up: load_layer_linear(files, &format!("{name}.ffn.up_proj"), embed_dim, ffn_dim)?,
            ffn_down: load_layer_linear(
                files,
                &format!("{name}.ffn.down_proj"),
                ffn_dim,
                embed_dim,
            )?,
            norm: RmsNorm::new(
                load_sharded(files, &format!("{name}.norm.weight"), &[embed_dim])?,
                norm_eps,
            ),
            adaln: load_layer_linear(
                files,
                &format!("{name}.adaLN_modulation.layers.1"),
                cond_dim,
                3 * embed_dim,
            )?,
            embed_dim,
        })
    }

    /// `x [rows, embed] , c [rows, cond] -> [rows, embed]`. The
    /// `adaLN_modulation` Sequential applies SiLU to `c` before the
    /// linear.
    fn forward(&self, x: &mut [f32], c: &[f32], rows: usize) {
        let mut silu_c = c.to_vec();
        ops::silu(&mut silu_c);
        let modulation = self.adaln.forward(&silu_c, rows);
        let width = self.embed_dim;
        let mut fed = x.to_vec();
        self.norm.apply(&mut fed, rows);
        for row in 0..rows {
            let shift = &modulation[row * 3 * width..row * 3 * width + width];
            let scale = &modulation[row * 3 * width + width..row * 3 * width + 2 * width];
            let base = row * width;
            for (col, value) in fed[base..base + width].iter_mut().enumerate() {
                *value = *value * (1.0 + scale[col]) + shift[col];
            }
        }
        // SwiGLU: down(silu(gate(x)) * up(x)).
        let gate = self.ffn_gate.forward(&fed, rows);
        let up = self.ffn_up.forward(&fed, rows);
        let mut fused = vec![0.0f32; gate.len()];
        for ((fused_slot, &g), &u) in fused.iter_mut().zip(&gate).zip(&up) {
            *fused_slot = g * (1.0 / (1.0 + (-g).exp())) * u;
        }
        let out = self.ffn_down.forward(&fused, rows);
        for row in 0..rows {
            let gate = &modulation[row * 3 * width + 2 * width..row * 3 * width + 3 * width];
            let base = row * width;
            for col in 0..width {
                x[base + col] += out[base + col] * gate[col];
            }
        }
    }
}

/// The final AdaLN layer over an affine-free norm.
struct FinalLayer {
    norm: RmsNorm,
    linear: Linear,
    adaln: Linear,
    hidden: usize,
}

impl FinalLayer {
    fn load(
        files: &[SafetensorsFile],
        name: &str,
        hidden: usize,
        output: usize,
        cond_dim: usize,
        norm_eps: f32,
    ) -> Result<Self> {
        Ok(Self {
            // elementwise_affine=False: the norm runs on ones.
            norm: RmsNorm::new(vec![1.0f32; hidden], norm_eps),
            linear: load_layer_linear(files, &format!("{name}.linear"), hidden, output)?,
            adaln: load_layer_linear(
                files,
                &format!("{name}.adaLN_modulation.layers.1"),
                cond_dim,
                2 * hidden,
            )?,
            hidden,
        })
    }

    fn forward(&self, x: &mut [f32], c: &[f32], rows: usize) -> Vec<f32> {
        let mut silu_c = c.to_vec();
        ops::silu(&mut silu_c);
        let modulation = self.adaln.forward(&silu_c, rows);
        let width = self.hidden;
        self.norm.apply(x, rows);
        for row in 0..rows {
            let shift = &modulation[row * 2 * width..row * 2 * width + width];
            let scale = &modulation[row * 2 * width + width..row * 2 * width + 2 * width];
            let base = row * width;
            for (col, value) in x[base..base + width].iter_mut().enumerate() {
                *value = *value * (1.0 + scale[col]) + shift[col];
            }
        }
        self.linear.forward(x, rows)
    }
}

/// The diffusion prediction head (`DiffusionHead`): noise/velocity for
/// the latent diffusion. `noisy [rows, latent]`, `timesteps [rows]`,
/// `condition [rows, hidden]` -> `[rows, latent]`.
pub(crate) struct DiffusionHead {
    pub(crate) noisy_images_proj: Linear,
    pub(crate) cond_proj: Linear,
    pub(crate) t_embedder: TimestepEmbedder,
    pub(crate) layers: Vec<HeadLayer>,
    final_layer: FinalLayer,
    hidden: usize,
    latent_size: usize,
}

impl DiffusionHead {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn load(
        files: &[SafetensorsFile],
        prefix: &str,
        hidden: usize,
        head_layers: usize,
        head_ffn_ratio: f32,
        norm_eps: f32,
        latent_size: usize,
    ) -> Result<Self> {
        let ffn_dim = (hidden as f32 * head_ffn_ratio) as usize;
        let cond_dim = hidden;
        let mut layers = Vec::with_capacity(head_layers);
        for index in 0..head_layers {
            layers.push(HeadLayer::load(
                files,
                &format!("{prefix}.layers.{index}"),
                hidden,
                ffn_dim,
                cond_dim,
                norm_eps,
            )?);
        }
        Ok(Self {
            noisy_images_proj: load_head_linear(
                files,
                &format!("{prefix}.noisy_images_proj"),
                latent_size,
                hidden,
            )?,
            cond_proj: load_head_linear(files, &format!("{prefix}.cond_proj"), hidden, cond_dim)?,
            t_embedder: TimestepEmbedder::load(files, &format!("{prefix}.t_embedder"), hidden)?,
            layers,
            final_layer: FinalLayer::load(
                files,
                &format!("{prefix}.final_layer"),
                hidden,
                latent_size,
                cond_dim,
                norm_eps,
            )?,
            hidden,
            latent_size,
        })
    }

    pub(crate) fn forward(
        &self,
        noisy: &[f32],
        timesteps: &[f32],
        condition: &[f32],
        rows: usize,
    ) -> Result<Vec<f32>> {
        if noisy.len() != rows * self.latent_size || condition.len() != rows * self.hidden {
            return Err(SpeechError::Input {
                why: "diffusion head input shapes do not match the batch".into(),
            });
        }
        let x = self.noisy_images_proj.forward(noisy, rows);
        let t = self.t_embedder.forward(timesteps, rows);
        let condition = self.cond_proj.forward(condition, rows);
        let mut c = vec![0.0f32; condition.len()];
        for ((slot, &a), &b) in c.iter_mut().zip(&condition).zip(&t) {
            *slot = a + b;
        }
        let mut current = x;
        for layer in &self.layers {
            layer.forward(&mut current, &c, rows);
        }
        Ok(self.final_layer.forward(&mut current, &c, rows))
    }
}

#[cfg(test)]
mod tests {
    use super::timestep_embedding;

    #[test]
    fn timestep_embedding_matches_the_cos_sin_concat() {
        // dim 8, max period 10000: freq d is exp(-ln(10000) d / 4).
        let got = timestep_embedding(3.0, 8, 10000.0);
        for d in 0..4 {
            let freq = (-10000.0f32.ln() * d as f32 / 4.0).exp();
            assert!((got[d] - (3.0f32 * freq).cos()).abs() < 1e-6);
            assert!((got[4 + d] - (3.0f32 * freq).sin()).abs() < 1e-6);
        }
    }
}
