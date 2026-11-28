//! DCW: differential correction in the wavelet domain, a sampler-side fix
//! for the SNR-t bias of flow-matching samplers (Yu et al., CVPR 2026,
//! arXiv:2604.16044).
//!
//! Reference: upstream `acestep/models/common/dcw_correction.py` and
//! `dcw_primitives.py` at the pinned commit, over pytorch-wavelets 1.3.0
//! `DWT1DForward(J=1, mode="zero", wave="haar")` along the time axis.
//! Upstream's turbo sampler applies it after every step by default
//! (`dcw_enabled=True`, mode `double`, scaler 0.05, high scaler 0.02),
//! with `denoised = x_before_step - v * t`.
//!
//! A one-level Haar transform pairs frames `(2k, 2k + 1)` (an odd last
//! frame pairs with a zero pad that the inverse trims). Per pair and
//! channel, with `h = 1 / sqrt(2)`:
//!
//! ```text
//! low  = (x0 + x1) h      high = (x0 - x1) h
//! low' = low + sl (low - low_y)      high' = high + sh (high - high_y)
//! x0'  = (low' + high') h            x1'   = (low' - high') h
//! ```
//!
//! `sl = t * scaler` and `sh = (1 - t) * high_scaler` in `double` mode;
//! `low` uses `t * scaler` on the low band only, `high` uses
//! `(1 - t) * scaler` on the high band only, `pix` adds
//! `scaler * (x - y)` with no transform. Only the Haar basis is ported.

use crate::{Result, SpeechError};

/// Which bands DCW corrects (`VALID_DCW_MODES`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DcwMode {
    Low,
    High,
    Double,
    Pix,
}

/// One DCW configuration (`DCWCorrector`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Dcw {
    pub mode: DcwMode,
    pub scaler: f32,
    pub high_scaler: f32,
}

impl Dcw {
    /// Upstream's turbo default: `double`, 0.05, 0.02, Haar.
    pub const TURBO: Dcw = Dcw {
        mode: DcwMode::Double,
        scaler: 0.05,
        high_scaler: 0.02,
    };

    /// `DCWCorrector.is_active`.
    pub fn is_active(&self) -> bool {
        match self.mode {
            DcwMode::Double => self.scaler != 0.0 || self.high_scaler != 0.0,
            _ => self.scaler != 0.0,
        }
    }

    /// Corrects `x [frames, channels]` (the latent after a sampler step)
    /// away from `denoised [frames, channels]` at timestep `t`.
    pub fn apply(
        &self,
        x: &mut [f32],
        denoised: &[f32],
        frames: usize,
        channels: usize,
        t: f32,
    ) -> Result<()> {
        if x.len() != frames * channels || denoised.len() != x.len() {
            return Err(SpeechError::Input {
                why: format!(
                    "DCW expects {frames} x {channels}, got {} and {}",
                    x.len(),
                    denoised.len()
                ),
            });
        }
        if !self.is_active() {
            return Ok(());
        }
        // `float(t_curr)` and the schedules are Python float arithmetic.
        let t = t as f64;
        let (low_s, high_s) = match self.mode {
            DcwMode::Low => (t * self.scaler as f64, 0.0),
            DcwMode::High => (0.0, (1.0 - t) * self.scaler as f64),
            DcwMode::Double => (t * self.scaler as f64, (1.0 - t) * self.high_scaler as f64),
            DcwMode::Pix => {
                let s = self.scaler;
                for (xv, yv) in x.iter_mut().zip(denoised) {
                    *xv += s * (*xv - yv);
                }
                return Ok(());
            }
        };
        // The primitives short-circuit when the active scale is zero
        // (`low` at t = 0, `high` at t = 1).
        if low_s == 0.0 && high_s == 0.0 {
            return Ok(());
        }
        let (sl, sh) = (low_s as f32, high_s as f32);
        let h = std::f32::consts::FRAC_1_SQRT_2;
        for k in 0..frames.div_ceil(2) {
            let (i0, i1) = (2 * k, 2 * k + 1);
            for c in 0..channels {
                let x0 = x[i0 * channels + c];
                let y0 = denoised[i0 * channels + c];
                let (x1, y1) = if i1 < frames {
                    (x[i1 * channels + c], denoised[i1 * channels + c])
                } else {
                    (0.0, 0.0)
                };
                let (xl, xh) = ((x0 + x1) * h, (x0 - x1) * h);
                let (yl, yh) = ((y0 + y1) * h, (y0 - y1) * h);
                let xl = if sl != 0.0 { xl + sl * (xl - yl) } else { xl };
                let xh = if sh != 0.0 { xh + sh * (xh - yh) } else { xh };
                x[i0 * channels + c] = (xl + xh) * h;
                if i1 < frames {
                    x[i1 * channels + c] = (xl - xh) * h;
                }
            }
        }
        Ok(())
    }
}
