//! Flow-matching Euler integrator and chunk scheduler.
//!
//! Reference: `mlx_audio/music/models/minimax_music3/euler.py`. The
//! sigma schedule walks 0 -> 1 in `steps` increments; each Euler step
//! evaluates the velocity field twice (conditional and CFG-uncond) and
//! advances `x` by `(sigma_next - sigma) * v`. Chunk overlap blends the
//! previous chunk's tail latents back in so consecutive chunks stitch.

use crate::Result;
use crate::SpeechError;

use super::dit::FlowMatchingTransformer;
use super::precision::DType;

/// `(1.0 - 1e-6)` evaluated in f64 then narrowed, the way the Python
/// scalar reaches the f32 blend in the reference.
const ONE_MINUS_EPS: f32 = (1.0f64 - 1e-6) as f32;

/// Sigmas for `steps` Euler steps: `linspace(1, 1/steps, steps)`
/// subtracted from one, with a final 1.0.
pub(crate) fn sigma_schedule(steps: usize) -> Result<Vec<f32>> {
    if steps == 0 {
        return Err(SpeechError::Input {
            why: "num_inference_steps must be at least one".to_string(),
        });
    }
    let mut sigmas = Vec::with_capacity(steps + 1);
    // numpy linspace(1, 1/steps, steps) in float32: the step is
    // (stop - start) / (num - 1) computed in f64, then index * step
    // and + start round through f32. A single-step schedule is just
    // the start value.
    let step = if steps > 1 {
        ((1.0f64 / steps as f64) - 1.0) / (steps - 1) as f64
    } else {
        0.0
    };
    for i in 0..steps {
        let value = (i as f32) * (step as f32) + 1.0f32;
        sigmas.push(1.0f32 - value);
    }
    sigmas.push(1.0);
    Ok(sigmas)
}

/// How often the unconditional CFG branch is re-evaluated inside one chunk.
///
/// The exact reference evaluates both branches every step. With
/// `uncond_interval > 1`, the first `uncond_warmup` steps still do, then the
/// unconditional forward runs on every `uncond_interval`-th step and the
/// guidance correction `guided - conditional` it produced is reused (added to
/// the fresh conditional velocity) in between. That changes the trajectory, so
/// it is opt-in and the default is the exact path.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FlowGuidance {
    pub uncond_interval: usize,
    pub uncond_warmup: usize,
}

impl FlowGuidance {
    /// Both branches every step: the reference trajectory.
    pub const EXACT: Self = Self {
        uncond_interval: 1,
        uncond_warmup: 2,
    };

    pub(crate) fn validate(self) -> Result<()> {
        if self.uncond_interval == 0 {
            return Err(SpeechError::Input {
                why: "flow uncond interval must be at least one".to_string(),
            });
        }
        Ok(())
    }

    fn evaluates_uncond(self, step: usize) -> bool {
        self.uncond_interval <= 1
            || step < self.uncond_warmup
            || (step - self.uncond_warmup) % self.uncond_interval == 0
    }

    /// Unconditional forwards a chunk of `steps` Euler steps performs.
    pub fn uncond_forwards(self, steps: usize) -> usize {
        (0..steps).filter(|s| self.evaluates_uncond(*s)).count()
    }
}

impl Default for FlowGuidance {
    fn default() -> Self {
        Self::EXACT
    }
}

pub(crate) fn guided_velocity(
    unconditional: f32,
    conditional: f32,
    guidance: f32,
    dtype: DType,
) -> f32 {
    dtype.round(
        unconditional
            + dtype.round(dtype.round(guidance) * dtype.round(conditional - unconditional)),
    )
}

pub(crate) fn update(
    latent: f32,
    velocity: f32,
    delta: f32,
    dtype: DType,
    velocity_dtype: DType,
) -> f32 {
    dtype
        .promote(velocity_dtype)
        .round(latent + velocity_dtype.round(velocity_dtype.round(delta) * velocity))
}

pub(crate) fn overlap_blend(noise: f32, previous: f32, sigma: f32, dtype: DType) -> f32 {
    if dtype == DType::F32 {
        return (1.0f32 - ONE_MINUS_EPS * sigma) * noise + sigma * previous;
    }
    let blend = dtype.round((1.0f64 - (1.0f64 - 1e-6) * sigma as f64) as f32);
    dtype.round(dtype.round(blend * noise) + dtype.round(dtype.round(sigma) * previous))
}

/// Euler-denoise one chunk. `latents` and the returned buffers are
/// channel-major `[1, in_channels, len]`; `condition` is
/// `[1, condition_dim, len]` and may be spliced with
/// `previous_condition` at the front, in which case the returned
/// condition covers the spliced length.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn denoise_chunk(
    transformer: &FlowMatchingTransformer,
    latents: &[f32],
    condition: &[f32],
    channels: usize,
    cond_dim: usize,
    len: usize,
    steps: usize,
    guidance_scale: f32,
    previous_latent: Option<&[f32]>,
    previous_condition: Option<&[f32]>,
) -> Result<(Vec<f32>, Vec<f32>)> {
    denoise_chunk_typed(
        transformer,
        latents,
        condition,
        channels,
        cond_dim,
        len,
        steps,
        guidance_scale,
        previous_latent,
        previous_condition,
        transformer.dtype(),
    )
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn denoise_chunk_typed(
    transformer: &FlowMatchingTransformer,
    latents: &[f32],
    condition: &[f32],
    channels: usize,
    cond_dim: usize,
    len: usize,
    steps: usize,
    guidance_scale: f32,
    previous_latent: Option<&[f32]>,
    previous_condition: Option<&[f32]>,
    condition_dtype: DType,
) -> Result<(Vec<f32>, Vec<f32>)> {
    denoise_chunk_controlled(
        transformer,
        latents,
        condition,
        channels,
        cond_dim,
        len,
        steps,
        guidance_scale,
        FlowGuidance::EXACT,
        previous_latent,
        previous_condition,
        condition_dtype,
        &mut |_| super::Control::Continue,
    )?
    .ok_or_else(super::never_cancelled)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn denoise_chunk_controlled(
    transformer: &FlowMatchingTransformer,
    latents: &[f32],
    condition: &[f32],
    channels: usize,
    cond_dim: usize,
    len: usize,
    steps: usize,
    guidance_scale: f32,
    guidance: FlowGuidance,
    previous_latent: Option<&[f32]>,
    previous_condition: Option<&[f32]>,
    condition_dtype: DType,
    progress: &mut dyn FnMut(usize) -> super::Control,
) -> Result<Option<(Vec<f32>, Vec<f32>)>> {
    let mut dtype = condition_dtype;
    let mut condition = condition.to_vec();
    let mut overlap = 0usize;
    if let (Some(prev_lat), Some(prev_cond)) = (previous_latent, previous_condition) {
        overlap = (prev_lat.len() / channels).min(len);
        if overlap > 0 {
            // The spliced condition is previous[:overlap] ++ condition[overlap:].
            let mut spliced = Vec::with_capacity(cond_dim * len);
            for c in 0..cond_dim {
                spliced.extend_from_slice(&prev_cond[c * overlap..(c + 1) * overlap]);
                spliced.extend_from_slice(&condition[c * len + overlap..(c + 1) * len]);
            }
            condition = spliced;
        }
    }
    let sigmas = sigma_schedule(steps)?;
    let mut x = latents.to_vec();
    // The pre-blend overlap region of the starting noise, kept in the
    // `[channels, overlap]` layout the blend loop indexes; a flat head
    // slice would cut through the channel-major buffer.
    let noise_prompt: Vec<f32> = if overlap > 0 {
        (0..channels)
            .flat_map(|c| x[c * len..c * len + overlap].to_vec())
            .collect()
    } else {
        Vec::new()
    };
    let zeros = vec![0.0f32; condition.len()];
    // `guided - conditional` from the last step that ran both branches; only
    // read when `guidance.uncond_interval > 1`.
    let mut cached_delta: Vec<f32> = Vec::new();
    for index in 0..steps {
        if progress(index) == super::Control::Cancel {
            return Ok(None);
        }
        let sigma = sigmas[index];
        let sigma_next = sigmas[index + 1];
        if overlap > 0 {
            let prev = previous_latent.unwrap();
            for c in 0..channels {
                for i in 0..overlap {
                    x[c * len + i] = overlap_blend(
                        noise_prompt[c * overlap + i],
                        prev[c * overlap + i],
                        sigma,
                        dtype,
                    );
                }
            }
        }
        transformer.trace(&format!("euler.{index}.input"), &x, dtype, &[channels, len]);
        let conditional =
            transformer.forward_typed(&x, sigma, &condition, len, dtype, condition_dtype)?;
        let velocity_dtype = transformer.output_dtype(dtype, condition_dtype);
        transformer.trace(
            &format!("euler.{index}.conditional"),
            &conditional,
            velocity_dtype,
            &[channels, len],
        );
        let velocity = if guidance_scale == 1.0 {
            conditional
        } else if !guidance.evaluates_uncond(index) {
            // Reuse the last correction; `cached_delta` was filled by an
            // earlier step because step 0 always evaluates both branches.
            let mut guided = Vec::with_capacity(conditional.len());
            for (c, d) in conditional.iter().zip(&cached_delta) {
                guided.push(velocity_dtype.round(*c + *d));
            }
            guided
        } else {
            let unconditional =
                transformer.forward_typed(&x, sigma, &zeros, len, dtype, condition_dtype)?;
            transformer.trace(
                &format!("euler.{index}.unconditional"),
                &unconditional,
                velocity_dtype,
                &[channels, len],
            );
            let mut guided = Vec::with_capacity(unconditional.len());
            for (u, c) in unconditional.iter().zip(&conditional) {
                guided.push(guided_velocity(*u, *c, guidance_scale, velocity_dtype));
            }
            if guidance.uncond_interval > 1 {
                cached_delta = guided
                    .iter()
                    .zip(&conditional)
                    .map(|(g, c)| g - c)
                    .collect();
            }
            guided
        };
        // (sigma_next - sigma) is computed in Python floats (f64) and
        // multiplies the f32 velocity field.
        let delta = (sigma_next as f64 - sigma as f64) as f32;
        transformer.trace(
            &format!("euler.{index}.velocity"),
            &velocity,
            velocity_dtype,
            &[channels, len],
        );
        let update_dtype = dtype.promote(velocity_dtype);
        for (v, vel) in x.iter_mut().zip(velocity) {
            *v = update(*v, vel, delta, dtype, velocity_dtype);
        }
        dtype = update_dtype;
        transformer.trace(
            &format!("euler.{index}.output"),
            &x,
            dtype,
            &[channels, len],
        );
    }
    if overlap > 0 {
        let prev = previous_latent.unwrap();
        for c in 0..channels {
            for i in 0..overlap {
                x[c * len + i] = prev[c * overlap + i];
            }
        }
    }
    Ok(Some((x, condition)))
}
