//! Bidirectional Conformer encoder with two half-residual Hadamard blocks.

use super::attention::Attention;
use super::cact::{Archive, CqMatrix};
use super::layers::{floats, norm, quantized, sigmoid, HadamardMlp, Mhc, DIM, LANES};
use super::stem::silu;
use crate::Result;

struct Block {
    norms: [Vec<f32>; 6],
    first: HadamardMlp,
    second: HadamardMlp,
    attention: Attention,
    attention_gate: f32,
    pw1: CqMatrix,
    pw2: CqMatrix,
    taps: Vec<f32>,
}

impl Block {
    fn load(a: &Archive, start: usize) -> Result<Self> {
        Ok(Self {
            norms: [
                floats(a, start, &[DIM])?,
                floats(a, start + 1, &[DIM])?,
                floats(a, start + 2, &[DIM])?,
                floats(a, start + 3, &[DIM])?,
                floats(a, start + 4, &[DIM])?,
                floats(a, start + 5, &[DIM])?,
            ],
            attention: Attention {
                q: quantized(a, start + 6, &[384, DIM])?,
                k: quantized(a, start + 7, &[96, DIM])?,
                v: quantized(a, start + 8, &[128, DIM])?,
                gate: quantized(a, start + 9, &[DIM, DIM])?,
                out: quantized(a, start + 10, &[DIM, DIM])?,
                q_norm: floats(a, start + 11, &[48])?,
                k_norm: floats(a, start + 12, &[48])?,
                kv_heads: 2,
            },
            attention_gate: floats(a, start + 13, &[1])?[0],
            pw1: quantized(a, start + 14, &[1024, DIM])?,
            pw2: quantized(a, start + 15, &[DIM, DIM])?,
            taps: floats(a, start + 16, &[9, DIM])?,
            first: HadamardMlp::load(a, start + 17)?,
            second: HadamardMlp::load(a, start + 30)?,
        })
    }

    fn forward(&self, mut x: Vec<f32>, permutations: &[Vec<usize>; 2], theta: f32) -> Vec<f32> {
        let y = self.first.forward(
            &norm(&x, &self.norms[0]),
            &permutations[0],
            &permutations[1],
        );
        for (x, y) in x.iter_mut().zip(y) {
            *x += 0.5 * y;
        }
        let y = norm(
            &self.attention.forward(&norm(&x, &self.norms[1]), theta),
            &self.norms[2],
        );
        let gate = sigmoid(self.attention_gate);
        for (x, y) in x.iter_mut().zip(y) {
            *x += gate * y;
        }
        let projected = self.pw1.project(&norm(&x, &self.norms[3]));
        let frames = x.len() / DIM;
        let mut glu = vec![0.0; x.len()];
        for t in 0..frames {
            for c in 0..DIM {
                glu[t * DIM + c] =
                    projected[t * DIM * 2 + c] * sigmoid(projected[t * DIM * 2 + DIM + c]);
            }
        }
        let mut conv = vec![0.0; x.len()];
        for t in 0..frames {
            for tap in 0..9 {
                let it = (t + tap) as isize - 4;
                if it >= 0 && (it as usize) < frames {
                    for c in 0..DIM {
                        conv[t * DIM + c] += glu[it as usize * DIM + c] * self.taps[tap * DIM + c];
                    }
                }
            }
        }
        let mut conv = norm(&conv, &self.norms[4]);
        conv.iter_mut().for_each(|v| *v = silu(*v));
        let y = self.pw2.project(&conv);
        for (x, y) in x.iter_mut().zip(y) {
            *x += y;
        }
        let y = self.second.forward(
            &norm(&x, &self.norms[5]),
            &permutations[0],
            &permutations[1],
        );
        for (x, y) in x.iter_mut().zip(y) {
            *x += 0.5 * y;
        }
        x
    }
}

pub(super) struct Encoder {
    blocks: Vec<Block>,
    mhc: Mhc,
    final_norm: Vec<f32>,
    theta: f32,
}

impl Encoder {
    pub fn load(a: &Archive) -> Result<Self> {
        Ok(Self {
            blocks: (0..8)
                .map(|i| Block::load(a, 319 + i * 43))
                .collect::<Result<_>>()?,
            mhc: Mhc::load(a, 663)?,
            final_norm: floats(a, 672, &[DIM])?,
            theta: a.header.rope_theta,
        })
    }

    pub fn forward(&self, x: &[f32], permutations: &[Vec<usize>; 2]) -> Vec<f32> {
        let mut streams = Vec::with_capacity(x.len() * LANES);
        for row in x.chunks_exact(DIM) {
            for _ in 0..LANES {
                streams.extend_from_slice(row);
            }
        }
        for (i, block) in self.blocks.iter().enumerate() {
            let (mixed, state) = self.mhc.read(&streams, i);
            let output = block.forward(mixed, permutations, self.theta);
            streams = self.mhc.write(state, &output, i);
        }
        let mut mean = vec![0.0; x.len()];
        for (t, frame) in streams.chunks_exact(LANES * DIM).enumerate() {
            for lane in 0..LANES {
                for i in 0..DIM {
                    mean[t * DIM + i] += frame[lane * DIM + i] / LANES as f32;
                }
            }
        }
        norm(&mean, &self.final_norm)
    }
}
