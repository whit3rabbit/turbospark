//! The DramaBox flow-matching scheduler, guidance, and latent tools
//! (`mlx_audio/tts/models/dramabox/{scheduler,guidance,latent,sampling}.py`).
//!
//! Reference: mlx-audio at commit
//! [`e1b19b9054bf163f5d812221a54fcc346f1890e9`](https://github.com/Blaizzy/mlx-audio/tree/e1b19b9054bf163f5d812221a54fcc346f1890e9/mlx_audio/tts/models/dramabox).
//! The LTX2 sigma schedule (token-count-shifted logistic sigmas with
//! the terminal stretch), the x0-model Euler loop with the multi-modal
//! guider (cfg + STG + modality terms and the std-rescale), the patch
//! layout `[tokens, channels * mel_bins]` with causal time bounds, and
//! the long-clip silence prior patch. Single batch.

use crate::{Result, SpeechError};

// ---------------------------------------------------------------------------
// Scheduler
// ---------------------------------------------------------------------------

const BASE_SHIFT_ANCHOR: f64 = 1024.0;
const MAX_SHIFT_ANCHOR: f64 = 4096.0;

/// `ltx2_sigmas`: the linspace of sigmas, the token-count-dependent
/// logistic shift, and the terminal stretch towards `terminal`.
pub(crate) fn ltx2_sigmas(steps: usize, latent_tokens_size: usize) -> Vec<f32> {
    let tokens = latent_tokens_size as f64;
    let max_shift = 2.05f64;
    let base_shift = 0.95f64;
    let terminal = 0.1f64;
    // mx.linspace(1.0, 0.0, steps + 1)
    let mut sigmas: Vec<f64> = (0..=steps).map(|i| 1.0 - i as f64 / steps as f64).collect();
    let slope = (max_shift - base_shift) / (MAX_SHIFT_ANCHOR - BASE_SHIFT_ANCHOR);
    let intercept = base_shift - slope * BASE_SHIFT_ANCHOR;
    let sigma_shift = tokens * slope + intercept;
    let exp_shift = sigma_shift.exp();
    for sigma in &mut sigmas {
        let shifted = exp_shift / (exp_shift + (1.0 / *sigma - 1.0));
        *sigma = if *sigma != 0.0 { shifted } else { 0.0 };
    }
    // Stretch the non-terminal sigmas so the last non-zero sigma sits
    // at `terminal`.
    let last_nonzero = sigmas[sigmas.len() - 2];
    let scale_factor = (1.0 - last_nonzero) / (1.0 - terminal);
    let stretched: Vec<f64> = sigmas[..sigmas.len() - 1]
        .iter()
        .map(|s| 1.0 - ((1.0 - s) / scale_factor))
        .collect();
    let mut out: Vec<f32> = stretched.iter().map(|s| *s as f32).collect();
    out.push(*sigmas.last().expect("sigma") as f32);
    out
}

/// `to_velocity`.
pub(crate) fn to_velocity(sample: &[f32], sigma: f32, denoised: &[f32]) -> Result<Vec<f32>> {
    if sigma == 0.0 {
        return Err(SpeechError::Input {
            why: "sigma can't be 0.0".into(),
        });
    }
    Ok(sample
        .iter()
        .zip(denoised)
        .map(|(&s, &d)| (s - d) / sigma)
        .collect())
}

/// `to_denoised` (the x0 model wraps its velocity output with this).
pub(crate) fn to_denoised(sample: &[f32], velocity: &[f32], sigma: f32) -> Vec<f32> {
    sample
        .iter()
        .zip(velocity)
        .map(|(&s, &v)| s - v * sigma)
        .collect()
}

/// `euler_step`.
pub(crate) fn euler_step(
    sample: &[f32],
    denoised_sample: &[f32],
    sigmas: &[f32],
    step_index: usize,
) -> Result<Vec<f32>> {
    let sigma = sigmas[step_index];
    let sigma_next = sigmas[step_index + 1];
    let dt = sigma_next - sigma;
    let velocity = to_velocity(sample, sigma, denoised_sample)?;
    Ok(sample
        .iter()
        .zip(&velocity)
        .map(|(&s, &v)| s + v * dt)
        .collect())
}

// ---------------------------------------------------------------------------
// Guidance
// ---------------------------------------------------------------------------

/// `auto_rescale_for_cfg`.
pub(crate) fn auto_rescale_for_cfg(cfg: f32) -> f32 {
    if cfg <= 2.0 {
        0.0
    } else if cfg <= 3.0 {
        0.6 * (cfg - 2.0)
    } else if cfg <= 4.0 {
        0.6 + 0.2 * (cfg - 3.0)
    } else if cfg <= 8.0 {
        0.8
    } else {
        (0.8 + 0.1 * (cfg - 8.0)).min(1.0)
    }
}

/// The guider parameters (`MultiModalGuiderParams`).
#[derive(Debug, Clone)]
pub(crate) struct GuiderParams {
    pub cfg_scale: f32,
    pub stg_scale: f32,
    pub rescale_scale: f32,
    pub modality_scale: f32,
}

/// `calculate_guided_prediction` over the flat `[tokens, width]`
/// predictions; the std rescale is over ALL elements (mx.std over the
/// whole array) and cfg_clamp_scale stays 0 in the shipped defaults.
pub(crate) fn calculate_guided_prediction(
    cond: &[f32],
    uncond_text: &[f32],
    uncond_perturbed: &[f32],
    uncond_modality: &[f32],
    params: &GuiderParams,
) -> Vec<f32> {
    let mut pred = vec![0.0f32; cond.len()];
    for index in 0..cond.len() {
        pred[index] = cond[index]
            + (params.cfg_scale - 1.0) * (cond[index] - uncond_text[index])
            + params.stg_scale * (cond[index] - uncond_perturbed[index])
            + (params.modality_scale - 1.0) * (cond[index] - uncond_modality[index]);
    }
    if params.rescale_scale != 0.0 {
        let std = |values: &[f32]| -> f32 {
            let mean = values.iter().sum::<f32>() / values.len() as f32;
            let var = values
                .iter()
                .map(|v| {
                    let d = v - mean;
                    d * d
                })
                .sum::<f32>()
                / values.len() as f32;
            var.sqrt()
        };
        let factor = params.rescale_scale * (std(cond) / std(&pred)) + (1.0 - params.rescale_scale);
        for value in &mut pred {
            *value *= factor;
        }
    }
    pred
}

// ---------------------------------------------------------------------------
// Latent tools
// ---------------------------------------------------------------------------

/// The patchifier grid bounds (`get_patch_grid_bounds`): per token the
/// causal `[start, end]` time in seconds,
/// `(max(f * downsample + 1 - downsample, 0)) * hop / sample_rate`.
pub(crate) fn patch_grid_bounds(
    tokens: usize,
    downsample: usize,
    hop: usize,
    sample_rate: usize,
) -> Vec<f32> {
    let mut bounds = vec![0.0f32; tokens * 2];
    for token in 0..tokens {
        // Causal mel frame: `arange(k) * downsample` then
        // `max(frame + 1 - downsample, 0)` = `max(d * (k - 1) + 1, 0)`.
        let t = |latent_frame: usize| -> f32 {
            let mel = (latent_frame * downsample + 1).saturating_sub(downsample);
            (mel * hop) as f32 / sample_rate as f32
        };
        bounds[token * 2] = t(token);
        bounds[token * 2 + 1] = t(token + 1);
    }
    bounds
}

/// `patch_long_clip_silence_prior`: frames 512 and 513 of a clip
/// longer than 513 frames are replaced by the linear interpolation of
/// frames 511 and 514. `latent [1, channels, frames, mel_bins]`.
pub(crate) fn patch_long_clip_silence_prior(
    latent: &mut [f32],
    channels: usize,
    mut frames: usize,
    mel_bins: usize,
) {
    if frames <= 513 {
        return;
    }
    let (f0, f1) = (511usize, 514usize);
    let denom = (f1 - f0) as f32;
    let frame_stride = channels * mel_bins;
    let total = latent.len();
    frames = frames.max(total / frame_stride);
    let _ = frames;
    for frame in [512usize, 513usize] {
        let t = (frame - f0) as f32 / denom;
        for index in 0..frame_stride {
            let a = latent[f0 * frame_stride + index];
            let b = latent[f1 * frame_stride + index];
            latent[frame * frame_stride + index] = (1.0 - t) * a + t * b;
        }
    }
}
