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

/// Euler-denoise one chunk. `latents` and the returned buffers are
/// channel-major `[1, in_channels, len]`; `condition` is
/// `[1, condition_dim, len]` and may be spliced with
/// `previous_condition` at the front, in which case the returned
/// condition covers the spliced length.
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
    for index in 0..steps {
        let sigma = sigmas[index];
        let sigma_next = sigmas[index + 1];
        if overlap > 0 {
            let prev = previous_latent.unwrap();
            for c in 0..channels {
                for i in 0..overlap {
                    x[c * len + i] = (1.0f32 - ONE_MINUS_EPS * sigma)
                        * noise_prompt[c * overlap + i]
                        + sigma * prev[c * overlap + i];
                }
            }
        }
        let conditional = transformer.forward(&x, sigma, &condition, len)?;
        let velocity = if guidance_scale == 1.0 {
            conditional
        } else {
            let unconditional = transformer.forward(&x, sigma, &zeros, len)?;
            let mut guided = Vec::with_capacity(unconditional.len());
            for (u, c) in unconditional.iter().zip(&conditional) {
                guided.push(u + guidance_scale * (c - u));
            }
            guided
        };
        // (sigma_next - sigma) is computed in Python floats (f64) and
        // multiplies the f32 velocity field.
        let delta = (sigma_next as f64 - sigma as f64) as f32;
        for (v, vel) in x.iter_mut().zip(velocity) {
            *v += delta * vel;
        }
    }
    if overlap > 0 {
        let prev = previous_latent.unwrap();
        for c in 0..channels {
            for i in 0..overlap {
                x[c * len + i] = prev[c * overlap + i];
            }
        }
    }
    Ok((x, condition))
}
