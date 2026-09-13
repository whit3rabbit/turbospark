/// FlowMatchEuler discrete scheduler for Z-Image.
///
/// Implements the pinned FlowMatchEulerDiscreteScheduler contract:
/// - 1000 training timesteps
/// - static shift 3.0: 3.0 * sigma / (1.0 + 2.0 * sigma)
/// - normalized transformer time: (1000.0 - timestep) / 1000.0 = 1.0 - sigma
/// - model output negated before Euler step:
///   sample_next = sample + (sigma_next - sigma) * model_output
#[derive(Debug, Clone, PartialEq)]
pub struct FlowMatchEulerScheduler {
    pub num_train_timesteps: f32,
    pub shift: f32,
    pub timesteps: Vec<f32>,
    pub sigmas: Vec<f32>,
}

impl Default for FlowMatchEulerScheduler {
    fn default() -> Self {
        Self::new(1000.0, 3.0)
    }
}

impl FlowMatchEulerScheduler {
    pub fn new(num_train_timesteps: f32, shift: f32) -> Self {
        Self {
            num_train_timesteps,
            shift,
            timesteps: Vec::new(),
            sigmas: Vec::new(),
        }
    }

    /// Set timesteps and sigmas for N requested inference steps.
    ///
    /// Formula from pinned diffusers ZImagePipeline:
    /// sigmas = linspace(1.0, 1.0 / N, N)
    /// shifted_sigmas = shift * sigmas / (1.0 + (shift - 1.0) * sigmas)
    /// timesteps = shifted_sigmas * num_train_timesteps
    /// terminal sigma 0.0 appended to sigmas schedule
    pub fn set_timesteps(&mut self, num_inference_steps: usize) {
        assert!(num_inference_steps > 0, "num_inference_steps must be > 0");

        let n = num_inference_steps;
        let mut raw_sigmas = Vec::with_capacity(n);
        if n == 1 {
            raw_sigmas.push(1.0f32);
        } else {
            let start = 1.0f64;
            let end = 1.0f64 / (n as f64);
            let step = (end - start) / ((n - 1) as f64);
            for i in 0..n {
                raw_sigmas.push((start + (i as f64) * step) as f32);
            }
        }

        let mut shifted_sigmas = Vec::with_capacity(n + 1);
        let mut timesteps = Vec::with_capacity(n);

        for &sigma in &raw_sigmas {
            let shifted = self.shift * sigma / (1.0 + (self.shift - 1.0) * sigma);
            shifted_sigmas.push(shifted);
            timesteps.push(shifted * self.num_train_timesteps);
        }

        shifted_sigmas.push(0.0f32);

        self.sigmas = shifted_sigmas;
        self.timesteps = timesteps;
    }

    /// Set timesteps from explicit custom sigmas, applying shift.
    pub fn set_timesteps_from_sigmas(&mut self, custom_sigmas: &[f32]) {
        assert!(!custom_sigmas.is_empty(), "custom_sigmas cannot be empty");
        let n = custom_sigmas.len();
        let mut shifted_sigmas = Vec::with_capacity(n + 1);
        let mut timesteps = Vec::with_capacity(n);

        for &sigma in custom_sigmas {
            let shifted = self.shift * sigma / (1.0 + (self.shift - 1.0) * sigma);
            shifted_sigmas.push(shifted);
            timesteps.push(shifted * self.num_train_timesteps);
        }
        shifted_sigmas.push(0.0f32);

        self.sigmas = shifted_sigmas;
        self.timesteps = timesteps;
    }

    /// Compute normalized time for transformer block modulation:
    /// (1000.0 - timestep) / 1000.0 = 1.0 - sigma.
    pub fn normalized_time(&self, step_index: usize) -> f32 {
        let timestep = self.timesteps[step_index];
        (self.num_train_timesteps - timestep) / self.num_train_timesteps
    }

    /// Perform Euler step:
    /// dt = sigma_next - sigma
    /// prev_sample = sample + dt * model_output
    ///
    /// Note: `model_output` is the negated noise prediction (-pred),
    /// exactly as in diffusers: `scheduler.step(-noise_pred, t, latents)`.
    pub fn step(&self, model_output: &[f32], step_index: usize, sample: &[f32]) -> Vec<f32> {
        assert_eq!(
            model_output.len(),
            sample.len(),
            "model_output and sample dimensions must match"
        );
        assert!(
            step_index < self.timesteps.len(),
            "step_index {step_index} out of bounds ({})",
            self.timesteps.len()
        );

        let sigma = self.sigmas[step_index];
        let sigma_next = self.sigmas[step_index + 1];
        let dt = sigma_next - sigma;

        let mut prev = Vec::with_capacity(sample.len());
        for i in 0..sample.len() {
            prev.push(sample[i] + dt * model_output[i]);
        }
        prev
    }
}
