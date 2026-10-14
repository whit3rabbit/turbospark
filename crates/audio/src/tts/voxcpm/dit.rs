//! The VoxCPM local DiT denoiser and the UnifiedCFM Euler flow solver.
//!
//! Port of `mlx_audio/tts/models/voxcpm/dit.py` at the pinned
//! mlx-audio commit: sinusoidal timestep embeddings (scale 1000,
//! sin-then-cos halves), two biased MLP towers summed into the start
//! token together with `mu`, full (non-causal) attention over
//! `[start | cond | x]`, and the Euler solver with sway-sampled
//! timesteps, the CFG zero-star trick (steps past the 4% warmup window
//! double the batch and rescale by the conditional/unconditional dot
//! product), and one recorded `mx.random.normal` draw per `sample`.

use crate::lm::{Bias, Weights};
use crate::nn::Linear;
use crate::tts::voxcpm::minicpm::{self, Mask, MiniCpmModel};
use crate::{Result, SpeechError};

use super::FlowNoise;

/// `SinusoidalPosEmb(dim)` with the reference scale of 1000.
pub(crate) struct SinusoidalPosEmb {
    dim: usize,
}

impl SinusoidalPosEmb {
    pub(crate) fn new(dim: usize) -> Self {
        assert!(dim % 2 == 0, "sinusoidal embedding dim must be even");
        Self { dim }
    }

    /// The embedding of one scalar timestep, `[dim]`:
    /// `concat(sin(1000 * t * f), cos(1000 * t * f))` with
    /// `f = exp(arange(half) * -log(10000) / (half - 1))`.
    pub(crate) fn forward(&self, t: f32) -> Vec<f32> {
        let half = self.dim / 2;
        let step = (10000.0f64).ln() / (half - 1) as f64;
        let mut out = vec![0.0f32; self.dim];
        for i in 0..half {
            let freq = (-(i as f32) * step as f32).exp();
            let angle = 1000.0 * t * freq;
            out[i] = angle.sin();
            out[half + i] = angle.cos();
        }
        out
    }
}

/// `TimestepEmbedding`: `linear_2(silu(linear_1(x)))`, both biased.
pub(crate) struct TimestepEmbedding {
    linear_1: Linear,
    linear_2: Linear,
}

impl TimestepEmbedding {
    pub(crate) fn load(weights: &Weights, prefix: &str, dim: usize) -> Result<Self> {
        Ok(Self {
            linear_1: weights.linear(&format!("{prefix}linear_1"), dim, dim, Bias::Required)?,
            linear_2: weights.linear(&format!("{prefix}linear_2"), dim, dim, Bias::Required)?,
        })
    }

    pub(crate) fn forward(&self, x: &[f32]) -> Vec<f32> {
        let mut h = self.linear_1.forward(x, 1);
        for value in &mut h {
            *value *= 1.0 / (1.0 + (-*value).exp());
        }
        self.linear_2.forward(&h, 1)
    }
}

/// `VoxCPMLocDiT`: the patch denoiser over a vocabulary-free MiniCPM
/// backbone with full (non-causal) attention.
pub struct VoxCpmLocDiT {
    in_channels: usize,
    hidden: usize,
    in_proj: Linear,
    cond_proj: Linear,
    out_proj: Linear,
    time_mlp: TimestepEmbedding,
    delta_time_mlp: TimestepEmbedding,
    time_emb: SinusoidalPosEmb,
    decoder: MiniCpmModel,
}

impl VoxCpmLocDiT {
    fn load(
        weights: &Weights,
        prefix: &str,
        in_channels: usize,
        decoder_config: &minicpm::Config,
    ) -> Result<Self> {
        let hidden = decoder_config.hidden_size;
        Ok(Self {
            in_channels,
            hidden,
            in_proj: weights.linear(
                &format!("{prefix}in_proj"),
                in_channels,
                hidden,
                Bias::Required,
            )?,
            cond_proj: weights.linear(
                &format!("{prefix}cond_proj"),
                in_channels,
                hidden,
                Bias::Required,
            )?,
            out_proj: weights.linear(
                &format!("{prefix}out_proj"),
                hidden,
                in_channels,
                Bias::Required,
            )?,
            time_mlp: TimestepEmbedding::load(weights, &format!("{prefix}time_mlp."), hidden)?,
            delta_time_mlp: TimestepEmbedding::load(
                weights,
                &format!("{prefix}delta_time_mlp."),
                hidden,
            )?,
            time_emb: SinusoidalPosEmb::new(hidden),
            decoder: MiniCpmModel::load(
                weights,
                &format!("{prefix}decoder."),
                decoder_config.clone(),
            )?,
        })
    }

    /// One estimator call for a single batch row: `x` and `cond` are
    /// `[C, T]` / `[C, T']` channel-major (the reference `(N, C, T)`
    /// with N = 1), `mu` is `[hidden]`, `t` and `dt` scalars. The
    /// reference passes `dt = 0` for every step (`mean_mode` false),
    /// which still routes through the delta-time embedding tower.
    /// Returns the prediction `[C, T]`.
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &mut self,
        x: &[f32],
        x_t: usize,
        mu: &[f32],
        t: f32,
        cond: &[f32],
        cond_t: usize,
        dt: f32,
    ) -> Result<Vec<f32>> {
        let hidden = self.hidden;
        let channels = self.in_channels;

        // x: (N, C, T) -> (N, T, C) -> in_proj.
        let x_rows = super::transpose(x, channels, x_t);
        let x_proj = self.in_proj.forward(&x_rows, x_t);

        let cond_rows = super::transpose(cond, channels, cond_t);
        let cond_proj = self.cond_proj.forward(&cond_rows, cond_t);

        let t_emb = self.time_mlp.forward(&self.time_emb.forward(t));
        let dt_emb = self.delta_time_mlp.forward(&self.time_emb.forward(dt));
        let mut start = vec![0.0f32; hidden];
        for i in 0..hidden {
            start[i] = mu[i] + t_emb[i] + dt_emb[i];
        }

        // [start | cond_proj | x_proj], full bidirectional attention.
        let mut sequence = start;
        sequence.extend_from_slice(&cond_proj);
        sequence.extend_from_slice(&x_proj);
        let total = 1 + cond_t + x_t;
        let out = self.decoder.forward(&sequence, total, None, Mask::Full)?;

        // Slice off the start token and the conditioning prefix.
        let tail = &out[(1 + cond_t) * hidden..];
        let projected = self.out_proj.forward(tail, x_t);
        Ok(super::transpose(&projected, x_t, channels))
    }
}

/// `UnifiedCFM`: the Euler flow sampler around the DiT estimator.
pub struct UnifiedCfm {
    pub in_channels: usize,
    estimator: VoxCpmLocDiT,
}

impl UnifiedCfm {
    pub(super) fn load(
        weights: &Weights,
        prefix: &str,
        in_channels: usize,
        decoder_config: &minicpm::Config,
    ) -> Result<Self> {
        Ok(Self {
            in_channels,
            estimator: VoxCpmLocDiT::load(
                weights,
                &format!("{prefix}estimator."),
                in_channels,
                decoder_config,
            )?,
        })
    }

    /// Test hook: one raw estimator call (see
    /// `VoxCpmLocDiT::forward`); used by the fixture tests only.
    #[allow(clippy::too_many_arguments)]
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn estimator_call(
        &mut self,
        x: &[f32],
        x_t: usize,
        mu: &[f32],
        t: f32,
        cond: &[f32],
        cond_t: usize,
        dt: f32,
    ) -> Result<Vec<f32>> {
        self.estimator.forward(x, x_t, mu, t, cond, cond_t, dt)
    }

    /// The sway-sampled t-span (`mx.linspace(1, 0, steps + 1)` plus the
    /// cosine sway adjustment), returned step-major.
    pub fn t_span(steps: usize) -> Vec<f32> {
        let count = steps + 1;
        let step = -1.0f32 / steps as f32;
        (0..count)
            .map(|i| {
                let t = 1.0 + i as f32 * step;
                t + (std::f32::consts::FRAC_PI_2 * t).cos() - 1.0 + t
            })
            .collect()
    }

    /// `UnifiedCFM.sample`: one noise draw of `[C, patch_size]`
    /// (reference `(B, C, T)` at batch 1), then the Euler solve from
    /// the sway t-span. `mu` is the DiT conditioning `[hidden]`,
    /// `cond` the previous patch `[C, patch]` channel-major, `cfg_value`
    /// the guidance scale. `t_scheduler` / `solver` names are validated
    /// by the caller.
    pub fn sample(
        &mut self,
        mu: &[f32],
        n_timesteps: usize,
        patch_size: usize,
        cond: &[f32],
        cfg_value: f32,
        noise: &mut dyn FlowNoise,
    ) -> Result<Vec<f32>> {
        if n_timesteps < 1 {
            return Err(SpeechError::Input {
                why: "n_timesteps must be at least 1".into(),
            });
        }
        let channels = self.in_channels;
        let mut current_x: Vec<f32> = noise.normal(channels * patch_size);

        let t_span = Self::t_span(n_timesteps);
        let mut t = t_span[0];
        let mut dt = t_span[0] - t_span[1];
        // `max(1, int(len(t_span) * 0.04))` warmup steps with a zero
        // derivative.
        let zero_init_steps = ((t_span.len() as f64) * 0.04).floor().max(1.0) as usize;

        for step in 1..t_span.len() {
            if step <= zero_init_steps {
                // dphi_dt = 0: the state does not move, only time.
            } else {
                // The reference doubles the batch (conditional mu first,
                // zeros-mu unconditional second); the rows are
                // independent sequences, so the port runs them one at a
                // time.
                let dphi = self
                    .estimator
                    .forward(&current_x, patch_size, mu, t, cond, patch_size, 0.0)?;
                let cfg_dphi = self.estimator.forward(
                    &current_x,
                    patch_size,
                    &vec![0.0; mu.len()],
                    t,
                    cond,
                    patch_size,
                    0.0,
                )?;
                // CFG zero-star: st* = <cond, uncond> / (||uncond||^2 + eps),
                // then uncond * st* + cfg * (cond - uncond * st*).
                let mut dot = 0.0f32;
                let mut sq_norm = 0.0f32;
                for (positive, negative) in dphi.iter().zip(&cfg_dphi) {
                    dot += positive * negative;
                    sq_norm += negative * negative;
                }
                let st_star = dot / (sq_norm + 1e-8);
                for (value, (positive, negative)) in
                    current_x.iter_mut().zip(dphi.iter().zip(&cfg_dphi))
                {
                    let (positive, negative) = (*positive, *negative);
                    *value -=
                        dt * (negative * st_star + cfg_value * (positive - negative * st_star));
                }
            }
            t -= dt;
            if step < t_span.len() - 1 {
                dt = t - t_span[step + 1];
            }
        }
        Ok(current_x)
    }
}
