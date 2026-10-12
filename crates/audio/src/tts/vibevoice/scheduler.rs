//! The deterministic DPM-Solver++ multistep scheduler VibeVoice uses
//! (`mlx_audio/tts/models/vibevoice/scheduler.py`).
//!
//! Reference: mlx-audio at commit
//! [`e1b19b9054bf163f5d812221a54fcc346f1890e9`](https://github.com/Blaizzy/mlx-audio/tree/e1b19b9054bf163f5d812221a54fcc346f1890e9/mlx_audio/tts/models/vibevoice/scheduler.py).
//! The cosine beta schedule (`betas_for_alpha_bar`), the f32 cumulative
//! product of the alphas exactly as the reference `mx.cumprod` computes
//! it, the `(num_train - 1) * (1 - i / steps)` timestep grid with Python
//! half-to-even rounding, the cached per-inference-step alpha/sigma/
//! lambda values recomputed from the f32 schedule (plus the terminal
//! `1.0 / 0.0 / +inf` row), the v-prediction x0 conversion, and the
//! deterministic first- and second-order updates. Unlike the kugelaudio
//! SDE variant there is no per-step noise: the second-order step uses
//! the caller's previous x0 prediction. The first step and the final
//! step (when `lower_order_final` and fewer than 15 steps, or
//! `final_sigmas_type == "zero"`) run first order; the middle steps run
//! second order.

use crate::{Result, SpeechError};

/// `betas_for_alpha_bar` with the cosine transform; the betas are
/// computed in f64 and stored f32 like `mx.array(betas, dtype=float32)`.
fn betas_for_alpha_bar(num_diffusion_timesteps: usize, max_beta: f64) -> Vec<f32> {
    let alpha_bar_fn = |t: f64| -> f64 {
        ((t + 0.008) / 1.008 * std::f64::consts::FRAC_PI_2)
            .cos()
            .powi(2)
    };
    let mut betas = Vec::with_capacity(num_diffusion_timesteps);
    for i in 0..num_diffusion_timesteps {
        let t1 = i as f64 / num_diffusion_timesteps as f64;
        let t2 = (i + 1) as f64 / num_diffusion_timesteps as f64;
        betas.push((1.0 - alpha_bar_fn(t2) / alpha_bar_fn(t1)).min(max_beta) as f32);
    }
    betas
}

/// Python `int(round(x))`: half to even.
fn python_round(x: f64) -> i64 {
    let floor = x.floor();
    let diff = x - floor;
    let round_up = diff > 0.5 || (diff == 0.5 && floor as i64 % 2 != 0);
    if round_up {
        floor as i64 + 1
    } else {
        floor as i64
    }
}

/// The DPM-Solver++ multistep scheduler for one batch of `[rows, dim]`
/// samples (both halves of the guided pair stay in lockstep, so the
/// port steps one half and the reference's doubled batch would match).
pub struct DpmSolver {
    prediction_type: String,
    solver_order: usize,
    lower_order_final: bool,
    final_sigmas_type: String,
    num_train_timesteps: usize,
    /// `sqrt(alphas_cumprod)` per training timestep, f32 like the
    /// reference `mx.sqrt` over the f32 cumprod.
    alpha_t: Vec<f32>,
    num_inference_steps: usize,
    timesteps: Vec<i64>,
    cached_alpha_t: Vec<f32>,
    cached_sigma_t: Vec<f32>,
    cached_lambda: Vec<f64>,
    model_outputs: Vec<Option<Vec<f32>>>,
    lower_order_nums: usize,
    step_index: Option<usize>,
}

impl DpmSolver {
    /// Builds the scheduler (`DPMSolverMultistepScheduler` with the
    /// diffusion head's ddpm fields; the deployed settings are
    /// `solver_order = 2`, `lower_order_final = true`,
    /// `final_sigmas_type = "zero"`, and the cosine schedule family).
    pub fn new(
        num_train_timesteps: usize,
        beta_schedule: &str,
        prediction_type: &str,
    ) -> Result<Self> {
        if beta_schedule != "cosine"
            && beta_schedule != "scaled_linear"
            && beta_schedule != "squaredcos_cap_v2"
        {
            return Err(SpeechError::Unsupported {
                why: format!("beta_schedule {beta_schedule}: only the cosine family is ported"),
            });
        }
        // f32 betas, then the f32 cumulative product chain
        // (`mx.cumprod` over the f32 array).
        let betas = betas_for_alpha_bar(num_train_timesteps, 0.999);
        let mut alphas_cumprod = Vec::with_capacity(betas.len());
        let mut product = 1.0f32;
        for &beta in &betas {
            product *= 1.0 - beta;
            alphas_cumprod.push(product);
        }
        let alpha_t: Vec<f32> = alphas_cumprod.iter().map(|a| a.sqrt()).collect();
        Ok(Self {
            prediction_type: prediction_type.to_owned(),
            solver_order: 2,
            lower_order_final: true,
            final_sigmas_type: "zero".to_owned(),
            num_train_timesteps,
            alpha_t,
            num_inference_steps: 0,
            timesteps: Vec::new(),
            cached_alpha_t: Vec::new(),
            cached_sigma_t: Vec::new(),
            cached_lambda: Vec::new(),
            model_outputs: Vec::new(),
            lower_order_nums: 0,
            step_index: None,
        })
    }

    /// `set_timesteps`: the inference grid and the cached per-step
    /// values, resetting the solver state.
    pub fn set_timesteps(&mut self, num_inference_steps: usize) {
        self.num_inference_steps = num_inference_steps;
        let mut timesteps = Vec::with_capacity(num_inference_steps);
        for i in 0..num_inference_steps {
            let t = (self.num_train_timesteps as f64 - 1.0)
                * (1.0 - i as f64 / num_inference_steps as f64);
            timesteps.push(python_round(t));
        }
        self.timesteps = timesteps;

        // The cached values are recomputed from the f32 schedule values
        // read back into f64 (`np.array(self.alpha_t.tolist())`).
        self.cached_alpha_t.clear();
        self.cached_sigma_t.clear();
        self.cached_lambda.clear();
        for &t in &self.timesteps {
            let t = t.max(0) as usize;
            let alpha = self.alpha_t[t] as f64;
            let sigma = ((1.0 - alpha * alpha) / (alpha * alpha)).sqrt();
            let alpha_val = 1.0 / (sigma * sigma + 1.0).sqrt();
            let sigma_val = sigma * alpha_val;
            self.cached_alpha_t.push(alpha_val as f32);
            self.cached_sigma_t.push(sigma_val as f32);
            self.cached_lambda.push(alpha_val.ln() - sigma_val.ln());
        }
        // The final step values: alpha 1, sigma 0, lambda +inf.
        self.cached_alpha_t.push(1.0);
        self.cached_sigma_t.push(0.0);
        self.cached_lambda.push(f64::INFINITY);

        self.model_outputs = vec![None; self.solver_order];
        self.lower_order_nums = 0;
        self.step_index = None;
    }

    /// `reset` for a new generation.
    pub fn reset(&mut self) {
        self.model_outputs = vec![None; self.solver_order];
        self.lower_order_nums = 0;
        self.step_index = None;
    }

    pub fn timesteps(&self) -> &[i64] {
        &self.timesteps
    }

    /// `_convert_model_output`: the v-prediction x0
    /// (`alpha_t * sample - sigma_t * model_output`; the epsilon and
    /// sample prediction types are ported for completeness).
    fn convert_model_output(
        &self,
        model_output: &[f32],
        sample: &[f32],
        step_idx: usize,
    ) -> Vec<f32> {
        let alpha = self.cached_alpha_t[step_idx];
        let sigma = self.cached_sigma_t[step_idx];
        model_output
            .iter()
            .zip(sample)
            .map(|(&m, &s)| match self.prediction_type.as_str() {
                "epsilon" => (s - sigma * m) / alpha,
                "v_prediction" => alpha * s - sigma * m,
                "sample" => m,
                other => panic!("unknown prediction_type {other}"),
            })
            .collect()
    }

    /// The deterministic first-order update
    /// `sigma_ratio * sample - alpha_t * (exp(-h) - 1) * x0`.
    fn first_order_update(&self, x0_pred: &[f32], sample: &[f32], step_idx: usize) -> Vec<f32> {
        let alpha_next = self.cached_alpha_t[step_idx + 1] as f64;
        let sigma_next = self.cached_sigma_t[step_idx + 1] as f64;
        let sigma_curr = self.cached_sigma_t[step_idx] as f64;
        let h = self.cached_lambda[step_idx + 1] - self.cached_lambda[step_idx];
        let sigma_ratio = if sigma_curr > 0.0 {
            sigma_next / sigma_curr
        } else {
            0.0
        };
        let exp_neg_h = (-h).exp();
        // The reference computes the coefficient products in Python f64
        // and multiplies f32 arrays, so each coefficient casts to f32.
        let sample_coef = sigma_ratio as f32;
        let x0_coef = (alpha_next * (exp_neg_h - 1.0)) as f32;
        x0_pred
            .iter()
            .zip(sample)
            .map(|(&x0, &s)| sample_coef * s - x0_coef * x0)
            .collect()
    }

    /// The deterministic second-order update with the `prev_x0`
    /// difference term
    /// `sigma_ratio * sample - a * D0 - 0.5 * a * D1` with
    /// `a = alpha_t * (exp(-h) - 1)` and `D1 = (x0 - prev_x0) / r0`.
    fn second_order_update(
        &self,
        x0_pred: &[f32],
        prev_x0: &[f32],
        sample: &[f32],
        step_idx: usize,
    ) -> Vec<f32> {
        let alpha_next = self.cached_alpha_t[step_idx + 1] as f64;
        let sigma_next = self.cached_sigma_t[step_idx + 1] as f64;
        let sigma_curr = self.cached_sigma_t[step_idx] as f64;
        let lambda_t = self.cached_lambda[step_idx + 1];
        let lambda_s0 = self.cached_lambda[step_idx];
        let lambda_s1 = if step_idx > 0 {
            self.cached_lambda[step_idx - 1]
        } else {
            lambda_s0
        };
        let h = lambda_t - lambda_s0;
        let h0 = lambda_s0 - lambda_s1;
        let r0 = if h != 0.0 { h0 / h } else { 1.0 };
        let sigma_ratio = if sigma_curr > 0.0 {
            sigma_next / sigma_curr
        } else {
            0.0
        };
        let exp_neg_h = (-h).exp();
        let sample_coef = sigma_ratio as f32;
        let d0_coef = (alpha_next * (exp_neg_h - 1.0)) as f32;
        let d1_coef = (0.5 * alpha_next * (exp_neg_h - 1.0)) as f32;
        x0_pred
            .iter()
            .zip(prev_x0)
            .zip(sample)
            .map(|((&x0, &prev), &s)| {
                let d1 = if r0 != 0.0 {
                    (x0 - prev) / r0 as f32
                } else {
                    0.0
                };
                sample_coef * s - d0_coef * x0 - d1_coef * d1
            })
            .collect()
    }

    /// One solver step: x0 conversion, the output history shift, the
    /// order choice (first when `lower_order_nums < 1` or on the final
    /// step under the lower-order flag, second otherwise), and the
    /// update. `prev_sample` and `x0_pred` come back.
    pub fn step(
        &mut self,
        model_output: &[f32],
        sample: &[f32],
        prev_x0: Option<&[f32]>,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let step_idx = match self.step_index {
            None => {
                self.step_index = Some(0);
                0
            }
            Some(index) => index,
        };
        let x0_pred = self.convert_model_output(model_output, sample, step_idx);

        for i in (1..self.solver_order).rev() {
            self.model_outputs[i] = self.model_outputs[i - 1].clone();
        }
        self.model_outputs[0] = Some(x0_pred.clone());

        let lower_order_final_flag = step_idx == self.num_inference_steps - 1
            && ((self.lower_order_final && self.num_inference_steps < 15)
                || self.final_sigmas_type == "zero");

        let prev_sample = if self.lower_order_nums < 1 || lower_order_final_flag {
            self.first_order_update(&x0_pred, sample, step_idx)
        } else {
            match prev_x0 {
                Some(prev) => self.second_order_update(&x0_pred, prev, sample, step_idx),
                None => match self.model_outputs.get(1).and_then(|slot| slot.clone()) {
                    Some(prev) => self.second_order_update(&x0_pred, &prev, sample, step_idx),
                    None => self.first_order_update(&x0_pred, sample, step_idx),
                },
            }
        };

        if self.lower_order_nums < self.solver_order - 1 {
            self.lower_order_nums += 1;
        }
        self.step_index = Some(step_idx + 1);
        Ok((prev_sample, x0_pred))
    }
}

#[cfg(test)]
mod tests {
    use super::{python_round, DpmSolver};

    #[test]
    fn python_round_is_half_to_even() {
        assert_eq!(python_round(9.5), 10);
        assert_eq!(python_round(4.5), 4);
        assert_eq!(python_round(14.25), 14);
        assert_eq!(python_round(4.75), 5);
        assert_eq!(python_round(-0.5), 0);
        assert_eq!(python_round(-1.5), -2);
    }

    #[test]
    fn timesteps_grid_and_terminal_values_match_the_reference() {
        // 20 train steps, 4 inference steps: (20-1) * (1 - i/4).
        let mut scheduler = DpmSolver::new(20, "cosine", "v_prediction").unwrap();
        scheduler.set_timesteps(4);
        assert_eq!(scheduler.timesteps(), vec![19, 14, 10, 5]);
        // Terminal row: alpha 1, sigma 0, lambda +inf.
        let last = scheduler.cached_alpha_t.len() - 1;
        assert_eq!(scheduler.cached_alpha_t[last], 1.0);
        assert_eq!(scheduler.cached_sigma_t[last], 0.0);
        assert!(scheduler.cached_lambda[last].is_infinite());
        assert!(scheduler.cached_lambda[0] < scheduler.cached_lambda[3]);
    }

    #[test]
    fn final_step_collapses_to_the_x0_prediction() {
        // At the terminal row the sigma ratio is 0 and exp(-inf) is 0,
        // so the first-order update is exactly x0_pred.
        let mut scheduler = DpmSolver::new(20, "cosine", "v_prediction").unwrap();
        scheduler.set_timesteps(4);
        let sample = vec![0.3f32, -0.4];
        let output = vec![0.1f32, 0.2];
        let mut prev = sample.clone();
        let mut prev_x0 = None;
        for step in 0..4 {
            let (next, x0_pred) = scheduler.step(&output, &prev, prev_x0.as_deref()).unwrap();
            if step == 3 {
                let diff: f32 = next
                    .iter()
                    .zip(&x0_pred)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0, f32::max);
                assert!(diff < 1e-6, "final step {diff}");
            }
            prev = next;
            prev_x0 = Some(x0_pred);
        }
    }

    #[test]
    fn non_cosine_schedules_are_refused() {
        assert!(DpmSolver::new(20, "linear", "v_prediction").is_err());
    }
}
