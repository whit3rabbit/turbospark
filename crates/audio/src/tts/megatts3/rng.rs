//! Randomness for the MegaTTS3 pipeline.
//!
//! The reference draws Gaussian noise (the WavVAE posterior sample and the
//! sampler's initial latent and per-step noise) and uniform noise (duration
//! jitter) from torch's global generator. The pipeline takes a [`Randomness`]
//! so a caller can supply a seeded generator for reproducible output, or a
//! recorded stream to reproduce a PyTorch run draw for draw. [`SeededRng`] is
//! a small self-contained generator; it does not reproduce torch's stream.

/// Source of the pipeline's random draws.
///
/// Draw order is part of the contract (it is the reference's order): one
/// normal draw for the prompt encoder, then for each text chunk two uniform
/// draws (duration jitter) followed by `1 + steps` normal draws (sampler).
pub trait Randomness {
    /// Fills `out` with standard normal draws.
    fn normal(&mut self, out: &mut [f32]);
    /// Fills `out` with uniform draws in `[0, 1)`.
    fn uniform(&mut self, out: &mut [f32]);
}

/// SplitMix64 with Box-Muller normals.
#[derive(Debug, Clone)]
pub struct SeededRng {
    state: u64,
    spare: Option<f64>,
}

impl SeededRng {
    pub fn new(seed: u64) -> Self {
        Self {
            state: seed,
            spare: None,
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[0, 1)` with 53 bits.
    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn gaussian(&mut self) -> f64 {
        if let Some(z) = self.spare.take() {
            return z;
        }
        // u1 in (0, 1] so the logarithm is finite.
        let u1 = 1.0 - self.unit();
        let u2 = self.unit();
        let r = (-2.0 * u1.ln()).sqrt();
        let (s, c) = (2.0 * std::f64::consts::PI * u2).sin_cos();
        self.spare = Some(r * s);
        r * c
    }
}

impl Randomness for SeededRng {
    fn normal(&mut self, out: &mut [f32]) {
        for v in out {
            *v = self.gaussian() as f32;
        }
    }

    fn uniform(&mut self, out: &mut [f32]) {
        for v in out {
            // 24 bits are exactly representable in f32, so 1.0 is never reached.
            *v = (self.next_u64() >> 40) as f32 / (1u32 << 24) as f32;
        }
    }
}

/// Replays recorded draws in order, for reproducing a reference run.
/// Running out of draws, or a draw of the wrong size, panics: a mismatch means
/// the pipeline's draw order drifted from the reference's.
#[derive(Debug, Clone, Default)]
pub struct ReplayRng {
    normals: std::collections::VecDeque<Vec<f32>>,
    uniforms: std::collections::VecDeque<Vec<f32>>,
}

impl ReplayRng {
    pub fn new(normals: Vec<Vec<f32>>, uniforms: Vec<Vec<f32>>) -> Self {
        Self {
            normals: normals.into(),
            uniforms: uniforms.into(),
        }
    }

    /// True once every recorded draw has been consumed.
    pub fn is_exhausted(&self) -> bool {
        self.normals.is_empty() && self.uniforms.is_empty()
    }
}

impl Randomness for ReplayRng {
    fn normal(&mut self, out: &mut [f32]) {
        let next = self
            .normals
            .pop_front()
            .expect("replay: no normal draws left");
        assert_eq!(next.len(), out.len(), "replay: normal draw size mismatch");
        out.copy_from_slice(&next);
    }

    fn uniform(&mut self, out: &mut [f32]) {
        let next = self
            .uniforms
            .pop_front()
            .expect("replay: no uniform draws left");
        assert_eq!(next.len(), out.len(), "replay: uniform draw size mismatch");
        out.copy_from_slice(&next);
    }
}
