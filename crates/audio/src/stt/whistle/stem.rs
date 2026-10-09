//! Three stages of strided convolution subsample time and mel bins by eight.

use super::cact::{Archive, CqMatrix};
use super::layers::{floats, quantized};
use crate::Result;

pub(super) fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

pub(super) struct Stem {
    kernels: [Vec<f32>; 3],
    pointwise: [CqMatrix; 2],
    projection: CqMatrix,
}

impl Stem {
    pub fn load(a: &Archive) -> Result<Self> {
        Ok(Self {
            kernels: [
                floats(a, 673, &[9, 128])?,
                floats(a, 674, &[9, 128])?,
                floats(a, 676, &[9, 128])?,
            ],
            pointwise: [
                quantized(a, 675, &[128, 128])?,
                quantized(a, 677, &[128, 128])?,
            ],
            projection: quantized(a, 678, &[512, 1280])?,
        })
    }

    pub fn forward(&self, features: &[f32]) -> Vec<f32> {
        let mut x = features.to_vec();
        let (mut time, mut freq, mut channels) = (features.len() / 80, 80usize, 1usize);
        for stage in 0..3 {
            let (nt, nf) = (time.div_ceil(2), freq.div_ceil(2));
            let mut y = vec![0.0; nt * nf * 128];
            for t in 0..nt {
                for f in 0..nf {
                    for c in 0..128 {
                        let mut sum = 0.0;
                        for kt in 0..3 {
                            for kf in 0..3 {
                                let it = (2 * t + kt) as isize - 1;
                                let iff = (2 * f + kf) as isize - 1;
                                if it >= 0
                                    && iff >= 0
                                    && (it as usize) < time
                                    && (iff as usize) < freq
                                {
                                    let ic = if channels == 1 { 0 } else { c };
                                    sum += x[(it as usize * freq + iff as usize) * channels + ic]
                                        * self.kernels[stage][(kt * 3 + kf) * 128 + c];
                                }
                            }
                        }
                        y[(t * nf + f) * 128 + c] = if stage == 0 { silu(sum) } else { sum };
                    }
                }
            }
            if stage > 0 {
                y = self.pointwise[stage - 1].project(&y);
                y.iter_mut().for_each(|v| *v = silu(*v));
            }
            x = y;
            time = nt;
            freq = nf;
            channels = 128;
        }
        // The projection consumes channels followed by frequency bins.
        let mut flat = vec![0.0; x.len()];
        for t in 0..time {
            for c in 0..128 {
                for f in 0..freq {
                    flat[(t * 128 + c) * freq + f] = x[(t * freq + f) * 128 + c];
                }
            }
        }
        self.projection.project(&flat)
    }
}
