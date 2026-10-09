//! Gated grouped-query attention with per-head A8 query and KV rounding.

use super::cact::CqMatrix;
use super::layers::{a8, norm, sigmoid, softmax, DIM};

pub(super) struct Attention {
    pub q: CqMatrix,
    pub k: CqMatrix,
    pub v: CqMatrix,
    pub gate: CqMatrix,
    pub out: CqMatrix,
    pub q_norm: Vec<f32>,
    pub k_norm: Vec<f32>,
    pub kv_heads: usize,
}

pub(super) struct Memory {
    pub keys: Vec<f32>,
    pub values: Vec<f32>,
    pub frames: usize,
}

pub(super) fn rope(x: &mut [f32], heads: usize, position: usize, theta: f32) {
    for (t, row) in x.chunks_exact_mut(heads * 48).enumerate() {
        for head in row.chunks_exact_mut(48) {
            for i in 0..24 {
                let angle = (position + t) as f32 * theta.powf(-((2 * i) as f32) / 48.0);
                let (sin, cos) = angle.sin_cos();
                let (a, b) = (head[i], head[i + 24]);
                head[i] = a * cos - b * sin;
                head[i + 24] = b * cos + a * sin;
            }
        }
    }
}

impl Attention {
    pub fn memory(&self, x: &[f32], theta: Option<f32>) -> Memory {
        let mut keys = norm(&self.k.project(x), &self.k_norm);
        let mut values = self.v.project(x);
        if let Some(theta) = theta {
            rope(&mut keys, self.kv_heads, 0, theta);
        }
        keys.chunks_exact_mut(48).for_each(a8);
        values.chunks_exact_mut(64).for_each(a8);
        Memory {
            keys,
            values,
            frames: x.len() / DIM,
        }
    }

    pub fn attend(
        &self,
        x: &[f32],
        mut query: Vec<f32>,
        memory: &Memory,
        theta: Option<f32>,
        position: usize,
    ) -> Vec<f32> {
        query = norm(&query, &self.q_norm);
        if let Some(theta) = theta {
            rope(&mut query, 8, position, theta);
        }
        query.chunks_exact_mut(48).for_each(a8);
        let gates = self.gate.project(x);
        let mut attended = vec![0.0; x.len()];
        let mut scores = vec![0.0; memory.frames];
        for t in 0..x.len() / DIM {
            for h in 0..8 {
                let kh = h / (8 / self.kv_heads);
                let q = &query[(t * 8 + h) * 48..(t * 8 + h + 1) * 48];
                for (j, score) in scores.iter_mut().enumerate() {
                    let k = &memory.keys
                        [(j * self.kv_heads + kh) * 48..(j * self.kv_heads + kh + 1) * 48];
                    *score = q.iter().zip(k).map(|(q, k)| q * k).sum::<f32>() / 48.0f32.sqrt();
                }
                softmax(&mut scores);
                for i in 0..64 {
                    let value = scores
                        .iter()
                        .enumerate()
                        .map(|(j, s)| s * memory.values[(j * self.kv_heads + kh) * 64 + i])
                        .sum::<f32>();
                    attended[t * DIM + h * 64 + i] = value * sigmoid(gates[t * DIM + h * 64 + i]);
                }
            }
        }
        self.out.project(&attended)
    }

    pub fn forward(&self, x: &[f32], theta: f32) -> Vec<f32> {
        let memory = self.memory(x, Some(theta));
        self.attend(x, self.q.project(x), &memory, Some(theta), 0)
    }
}
