//! The VoxCPM2 DiT denoiser (`VoxCPMLocDiTV2`) and flow solver.
//!
//! Port of `mlx_audio/tts/models/voxcpm2/dit.py` at the pinned
//! mlx-audio commit. The V2 estimator takes a multi-token `mu`: the
//! caller concatenates the two conditioning projections into
//! `[H_dit, H_dit]`, which the estimator reshapes into a `(2, H)`
//! mu-token block prepended before the delta-time token:
//! `[mu_1 | mu_2 | t | cond | x]`, with the output sliced after the
//! mu tokens, the time token, and the conditioning prefix. The
//! solver is the voxcpm Euler flow with the `mean_mode` switch (live
//! delta-time values instead of zeros) and the same sway t-span,
//! zero-star CFG, and recorded noise draw.

use crate::lm::Weights;
use crate::tts::voxcpm::minicpm::{self, Mask, MiniCpmModel};
use crate::tts::voxcpm::FlowNoise;
use crate::{Result, SpeechError};

use super::super::voxcpm::dit::SinusoidalPosEmb as V1SinusoidalPosEmb;
use super::super::voxcpm::dit::TimestepEmbedding as V1TimestepEmbedding;

/// `VoxCPMLocDiTV2`: the patch denoiser with the mu-token block.
pub struct VoxCpmLocDiTv2 {
    in_channels: usize,
    hidden: usize,
    in_proj: crate::nn::Linear,
    cond_proj: crate::nn::Linear,
    out_proj: crate::nn::Linear,
    time_mlp: V1TimestepEmbedding,
    delta_time_mlp: V1TimestepEmbedding,
    time_emb: V1SinusoidalPosEmb,
    decoder: MiniCpmModel,
}

impl VoxCpmLocDiTv2 {
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
                crate::lm::Bias::Required,
            )?,
            cond_proj: weights.linear(
                &format!("{prefix}cond_proj"),
                in_channels,
                hidden,
                crate::lm::Bias::Required,
            )?,
            out_proj: weights.linear(
                &format!("{prefix}out_proj"),
                hidden,
                in_channels,
                crate::lm::Bias::Required,
            )?,
            time_mlp: V1TimestepEmbedding::load(weights, &format!("{prefix}time_mlp."), hidden)?,
            delta_time_mlp: V1TimestepEmbedding::load(
                weights,
                &format!("{prefix}delta_time_mlp."),
                hidden,
            )?,
            time_emb: V1SinusoidalPosEmb::new(hidden),
            decoder: MiniCpmModel::load(
                weights,
                &format!("{prefix}decoder."),
                decoder_config.clone(),
            )?,
        })
    }

    /// One estimator call for a single batch row: `x` and `cond` are
    /// `[C, T]` / `[C, T']` channel-major, `mu` is `[2 * hidden]`
    /// (the two conditioning tokens), `t` and `dt` scalars.
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
        if mu.len() != hidden * 2 {
            return Err(SpeechError::Input {
                why: format!("mu is {} wide, expected 2 x {hidden}", mu.len()),
            });
        }

        let x_rows = super::super::voxcpm::transpose(x, channels, x_t);
        let x_proj = self.in_proj.forward(&x_rows, x_t);
        let cond_rows = super::super::voxcpm::transpose(cond, channels, cond_t);
        let cond_proj = self.cond_proj.forward(&cond_rows, cond_t);

        let t_emb = self.time_mlp.forward(&self.time_emb.forward(t));
        let dt_emb = self.delta_time_mlp.forward(&self.time_emb.forward(dt));
        let mut t_comb = vec![0.0f32; hidden];
        for i in 0..hidden {
            t_comb[i] = t_emb[i] + dt_emb[i];
        }

        // [mu_1 | mu_2 | t | cond | x], full bidirectional attention.
        let mut sequence = mu.to_vec();
        sequence.extend_from_slice(&t_comb);
        sequence.extend_from_slice(&cond_proj);
        sequence.extend_from_slice(&x_proj);
        let total = 2 + 1 + cond_t + x_t;
        let out = self.decoder.forward(&sequence, total, None, Mask::Full)?;

        // Slice off the mu tokens, the time token, and the prefix.
        let tail = &out[(2 + 1 + cond_t) * hidden..];
        let projected = self.out_proj.forward(tail, x_t);
        Ok(super::super::voxcpm::transpose(&projected, x_t, channels))
    }
}

/// `UnifiedCFM` (V2): the Euler flow sampler with the `mean_mode`
/// delta-time switch.
pub struct UnifiedCfm {
    pub in_channels: usize,
    mean_mode: bool,
    estimator: VoxCpmLocDiTv2,
}

impl UnifiedCfm {
    pub(super) fn load(
        weights: &Weights,
        prefix: &str,
        in_channels: usize,
        decoder_config: &minicpm::Config,
        mean_mode: bool,
    ) -> Result<Self> {
        Ok(Self {
            in_channels,
            mean_mode,
            estimator: VoxCpmLocDiTv2::load(
                weights,
                &format!("{prefix}estimator."),
                in_channels,
                decoder_config,
            )?,
        })
    }

    /// The sway t-span, identical to the voxcpm one.
    pub fn t_span(steps: usize) -> Vec<f32> {
        super::super::voxcpm::dit::UnifiedCfm::t_span(steps)
    }

    /// `UnifiedCFM.sample`: one noise draw, then the Euler solve.
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
        let zero_init_steps = ((t_span.len() as f64) * 0.04).floor().max(1.0) as usize;

        for step in 1..t_span.len() {
            if step <= zero_init_steps {
                // dphi_dt = 0 during the warmup window.
            } else {
                let dt_val = if self.mean_mode { dt } else { 0.0 };
                let dphi = self
                    .estimator
                    .forward(&current_x, patch_size, mu, t, cond, patch_size, dt_val)?;
                let cfg_dphi = self.estimator.forward(
                    &current_x,
                    patch_size,
                    &vec![0.0; mu.len()],
                    t,
                    cond,
                    patch_size,
                    dt_val,
                )?;
                // CFG zero-star.
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
