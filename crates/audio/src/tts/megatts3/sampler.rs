//! The MegaTTS3 latent sampler: an Euler flow-matching loop with F5-TTS sway
//! sampling and the AMO overshoot step (`Diffusion.inference` in
//! `tts/modules/llm_dit/dit.py`).
//!
//! Every step draws fresh Gaussian noise, so the loop is stochastic. The
//! draws come from a caller-supplied filler and are taken in the reference's
//! order (the initial latent, then one tensor per step, including the final
//! step whose noise weight is zero) so a recorded stream reproduces PyTorch.

use super::dit::{Dit, DitCondition};
use crate::{Result, SpeechError};

const MAX_STEPS: usize = 1000;
/// Overshoot factor `c` of Algorithm 1 in the AMO paper (hardcoded upstream).
const AMO_C: f32 = 3.0;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SamplerParams {
    /// Euler steps (`time_step`, 32 by default upstream).
    pub steps: usize,
    /// Intelligibility weight (`p_w`).
    pub p_w: f32,
    /// Similarity weight (`t_w`).
    pub t_w: f32,
}

impl Default for SamplerParams {
    fn default() -> Self {
        Self {
            steps: 32,
            p_w: 1.4,
            t_w: 3.0,
        }
    }
}

/// `torch.linspace(0, 1, steps + 1)` in f32: the first half counts up from 0
/// and the second half counts down from 1, as the kernel does.
fn linspace01(steps: usize) -> Vec<f32> {
    let step = 1.0f32 / steps as f32;
    (0..=steps)
        .map(|i| {
            // torch: halfway = (steps + 1) / 2 elements count up from 0.
            if i < steps.div_ceil(2) {
                step * i as f32
            } else {
                1.0 - step * (steps - i) as f32
            }
        })
        .collect()
}

/// Timestep schedule with sway coefficient c = -1: `t + c * (cos(pi t / 2) - 1 + t)`,
/// which is `1 - cos(pi t / 2)`.
pub fn sway_schedule(steps: usize) -> Vec<f32> {
    linspace01(steps)
        .into_iter()
        .map(|t| t - ((std::f32::consts::FRAC_PI_2 * t).cos() - 1.0 + t))
        .collect()
}

/// One AMO step: overshoot to `o`, then re-noise back to `t_next`.
/// `z`, `v` and `noise` share a shape.
pub fn amo_step(z: &[f32], t: f32, t_next: f32, v: &[f32], noise: &[f32]) -> Vec<f32> {
    let s = t_next;
    let o = (t_next + AMO_C * (t_next - t)).min(1.0);
    let a = s / o;
    // Rounding can push the radicand a hair below zero when o ~ s; torch
    // would produce NaN there, which is never a meaningful latent.
    let b = ((1.0 - s).powi(2) - (a * (1.0 - o)).powi(2))
        .max(0.0)
        .sqrt();
    z.iter()
        .zip(v)
        .zip(noise)
        .map(|((&zi, &vi), &ni)| a * (zi + (o - t) * vi) + b * ni)
        .collect()
}

/// Runs the full sampler. `noise` fills a slice with standard normal draws.
pub fn sample(
    dit: &Dit,
    cond: &DitCondition,
    params: SamplerParams,
    noise: &mut dyn FnMut(&mut [f32]),
) -> Result<Vec<f32>> {
    if params.steps == 0 || params.steps > MAX_STEPS {
        return Err(SpeechError::Input {
            why: format!("steps must be in 1..={MAX_STEPS}, got {}", params.steps),
        });
    }
    if !(params.p_w.is_finite() && params.t_w.is_finite()) {
        return Err(SpeechError::Input {
            why: "p_w and t_w must be finite".to_string(),
        });
    }
    let n = cond.frames * dit.config().channels;
    let schedule = sway_schedule(params.steps);
    let mut x = vec![0.0f32; n];
    noise(&mut x);
    let mut eps = vec![0.0f32; n];
    for i in 0..params.steps {
        let v = dit.velocity(cond, &x, schedule[i], params.p_w, params.t_w)?;
        noise(&mut eps);
        x = amo_step(&x, schedule[i], schedule[i + 1], &v, &eps);
    }
    Ok(x)
}
