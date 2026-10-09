//! Autoregressive text decoder and its audio cross-attention memory.

use super::attention::{rope, Attention, Memory};
use super::cact::{Archive, CqMatrix};
use super::layers::{a8, floats, norm, quantized, rms, sigmoid, HadamardMlp, Mhc, DIM, LANES};
use crate::Result;
use turbospark_core::TokenId;

struct Cross {
    attention: Attention,
    pre: Vec<f32>,
    post: Vec<f32>,
    gate: f32,
}
struct Block {
    attention: Attention,
    pre: Vec<f32>,
    post: Vec<f32>,
    gate: f32,
    taps: [Vec<f32>; 3],
    mlp: HadamardMlp,
    pre_mlp: Vec<f32>,
    cross: Cross,
}

impl Block {
    fn load(a: &Archive, layer: usize) -> Result<Self> {
        let s = 1 + layer * 27;
        let c = 238 + layer * 10;
        Ok(Self {
            pre: floats(a, s, &[DIM])?,
            attention: Attention {
                q: quantized(a, s + 1, &[384, DIM])?,
                k: quantized(a, s + 2, &[96, DIM])?,
                v: quantized(a, s + 3, &[128, DIM])?,
                gate: quantized(a, s + 9, &[DIM, DIM])?,
                out: quantized(a, s + 10, &[DIM, DIM])?,
                q_norm: floats(a, s + 7, &[48])?,
                k_norm: floats(a, s + 8, &[48])?,
                kv_heads: 2,
            },
            taps: [
                floats(a, s + 4, &[3, 384])?,
                floats(a, s + 5, &[3, 96])?,
                floats(a, s + 6, &[3, 128])?,
            ],
            post: floats(a, s + 11, &[DIM])?,
            gate: floats(a, s + 12, &[1])?[0],
            pre_mlp: floats(a, s + 13, &[DIM])?,
            mlp: HadamardMlp::load(a, s + 14)?,
            cross: Cross {
                pre: floats(a, c, &[DIM])?,
                attention: Attention {
                    q: quantized(a, c + 1, &[384, DIM])?,
                    k: quantized(a, c + 2, &[384, DIM])?,
                    v: quantized(a, c + 3, &[DIM, DIM])?,
                    gate: quantized(a, c + 4, &[DIM, DIM])?,
                    out: quantized(a, c + 5, &[DIM, DIM])?,
                    q_norm: floats(a, c + 6, &[48])?,
                    k_norm: floats(a, c + 7, &[48])?,
                    kv_heads: 8,
                },
                post: floats(a, c + 8, &[DIM])?,
                gate: floats(a, c + 9, &[1])?[0],
            },
        })
    }

    fn forward(
        &self,
        mut x: Vec<f32>,
        state: &mut LayerState,
        audio: &Memory,
        position: usize,
        theta: f32,
        p: &[Vec<usize>; 2],
    ) -> Vec<f32> {
        let nx = norm(&x, &self.pre);
        let mut q = state.q.push(self.attention.q.project(&nx), &self.taps[0]);
        let mut k = norm(
            &state.k.push(self.attention.k.project(&nx), &self.taps[1]),
            &self.attention.k_norm,
        );
        let mut v = state.v.push(self.attention.v.project(&nx), &self.taps[2]);
        rope(&mut k, 2, position, theta);
        k.chunks_exact_mut(48).for_each(a8);
        v.chunks_exact_mut(64).for_each(a8);
        state.memory.keys.extend(k);
        state.memory.values.extend(v);
        state.memory.frames += 1;
        let attended = self.attention.attend(
            &nx,
            std::mem::take(&mut q),
            &state.memory,
            Some(theta),
            position,
        );
        let attended = norm(&attended, &self.post);
        for (x, y) in x.iter_mut().zip(attended) {
            *x += sigmoid(self.gate) * y;
        }
        let nx = norm(&x, &self.cross.pre);
        let y =
            self.cross
                .attention
                .attend(&nx, self.cross.attention.q.project(&nx), audio, None, 0);
        let y = norm(&y, &self.cross.post);
        for (x, y) in x.iter_mut().zip(y) {
            *x += sigmoid(self.cross.gate) * y;
        }
        let y = self.mlp.forward(&norm(&x, &self.pre_mlp), &p[0], &p[1]);
        for (x, y) in x.iter_mut().zip(y) {
            *x += y;
        }
        x
    }
}

struct Temporal {
    history: Vec<Vec<f32>>,
}

impl Temporal {
    fn push(&mut self, row: Vec<f32>, taps: &[f32]) -> Vec<f32> {
        let width = row.len();
        self.history.insert(0, row);
        self.history.truncate(3);
        let mut out = vec![0.0; width];
        for (j, row) in self.history.iter().enumerate() {
            for i in 0..width {
                out[i] += row[i] * taps[j * width + i];
            }
        }
        out
    }
}

struct LayerState {
    q: Temporal,
    k: Temporal,
    v: Temporal,
    memory: Memory,
}
impl LayerState {
    fn new() -> Self {
        Self {
            q: Temporal {
                history: Vec::new(),
            },
            k: Temporal {
                history: Vec::new(),
            },
            v: Temporal {
                history: Vec::new(),
            },
            memory: Memory {
                keys: Vec::new(),
                values: Vec::new(),
                frames: 0,
            },
        }
    }
}

struct Engram {
    table: CqMatrix,
    key: CqMatrix,
    value: CqMatrix,
    taps: Vec<f32>,
}
impl Engram {
    fn load(a: &Archive, start: usize) -> Result<Self> {
        Ok(Self {
            table: quantized(a, start, &[73728, 128])?,
            key: quantized(a, start + 1, &[DIM, DIM])?,
            value: quantized(a, start + 2, &[DIM, DIM])?,
            taps: floats(a, start + 3, &[4, DIM])?,
        })
    }

    fn lookup(&self, history: &[TokenId]) -> (Vec<f32>, Vec<f32>) {
        let mut x = vec![0.0; DIM];
        for (oi, order) in [2, 3].into_iter().enumerate() {
            if history.len() < order {
                continue;
            }
            for head in 0..2 {
                let table = oi * 2 + head;
                let mut hash = 0x9e37_79b9u32.wrapping_mul((table + 1) as u32);
                for j in 0..order {
                    hash = (hash ^ history[history.len() - 1 - j] as u32).wrapping_mul(0x0100_0193);
                }
                hash ^= hash >> 15;
                let row = self.table.row(table * 18432 + hash as usize % 18432);
                x[table * 128..(table + 1) * 128].copy_from_slice(&row);
            }
        }
        (self.key.project(&x), self.value.project(&x))
    }

    fn apply(&self, x: &mut [f32], history: &[TokenId], values: &mut Vec<Vec<f32>>) {
        let (key, value) = self.lookup(history);
        values.push(value);
        let gate = sigmoid(
            x.iter().zip(&key).map(|(x, k)| x * k).sum::<f32>() * rms(x) * rms(&key)
                / (DIM as f32).sqrt(),
        );
        for j in 0..4 {
            if let Some(t) = values.len().checked_sub(1 + j * 3) {
                for i in 0..DIM {
                    x[i] += gate * self.taps[j * DIM + i] * values[t][i];
                }
            }
        }
    }
}

pub(super) struct Decoder {
    embedding: CqMatrix,
    blocks: Vec<Block>,
    mhc: Mhc,
    final_norm: Vec<f32>,
    engrams: [Engram; 2],
    pe_gate: f32,
    theta: f32,
}
pub(super) struct DecodeState {
    layers: Vec<LayerState>,
    audio: Vec<Memory>,
    history: Vec<TokenId>,
    engrams: [Vec<Vec<f32>>; 2],
}

impl Decoder {
    pub fn load(a: &Archive) -> Result<Self> {
        Ok(Self {
            embedding: quantized(a, 0, &[8199, DIM])?,
            blocks: (0..8).map(|i| Block::load(a, i)).collect::<Result<_>>()?,
            mhc: Mhc::load(a, 217)?,
            final_norm: floats(a, 236, &[DIM])?,
            engrams: [Engram::load(a, 228)?, Engram::load(a, 232)?],
            pe_gate: floats(a, 318, &[1])?[0],
            theta: a.header.rope_theta,
        })
    }

    pub fn state(&self, encoded: &[f32]) -> DecodeState {
        let mut positioned = encoded.to_vec();
        let gate = sigmoid(self.pe_gate);
        let coefficient = -10000.0f32.ln() / 255.0;
        for (t, row) in positioned.chunks_exact_mut(DIM).enumerate() {
            for i in 0..DIM / 2 {
                let angle = t as f32 * (i as f32 * coefficient).exp();
                let (sin, cos) = angle.sin_cos();
                row[i] += gate * sin;
                row[i + DIM / 2] += gate * cos;
            }
        }
        DecodeState {
            layers: (0..8).map(|_| LayerState::new()).collect(),
            audio: self
                .blocks
                .iter()
                .map(|b| b.cross.attention.memory(&positioned, None))
                .collect(),
            history: Vec::new(),
            engrams: [Vec::new(), Vec::new()],
        }
    }

    pub fn step(&self, token: TokenId, state: &mut DecodeState, p: &[Vec<usize>; 2]) -> Vec<f32> {
        let position = state.history.len();
        state.history.push(token);
        let mut embedding = self.embedding.row(token as usize);
        // The trained decoder scales input embeddings, but its tied output
        // projection consumes the original weights.
        embedding.iter_mut().for_each(|v| *v *= (DIM as f32).sqrt());
        let mut streams = Vec::with_capacity(DIM * LANES);
        for _ in 0..LANES {
            streams.extend_from_slice(&embedding);
        }
        for (i, block) in self.blocks.iter().enumerate() {
            let (mut x, mhc_state) = self.mhc.read(&streams, i);
            if let Some(engram) = [3, 7].iter().position(|&layer| layer == i) {
                self.engrams[engram].apply(&mut x, &state.history, &mut state.engrams[engram]);
            }
            let output = block.forward(
                x,
                &mut state.layers[i],
                &state.audio[i],
                position,
                self.theta,
                p,
            );
            streams = self.mhc.write(mhc_state, &output, i);
        }
        let mut mean = vec![0.0; DIM];
        for lane in 0..LANES {
            for i in 0..DIM {
                mean[i] += streams[lane * DIM + i] / LANES as f32;
            }
        }
        self.embedding.project(&norm(&mean, &self.final_norm))
    }
}

#[cfg(test)]
mod tests {
    use super::Temporal;

    #[test]
    fn temporal_filter_is_causal_current_first_and_bounded() {
        let mut filter = Temporal {
            history: Vec::new(),
        };
        let taps = [1.0, 2.0, 10.0, 20.0, 100.0, 200.0];
        assert_eq!(filter.push(vec![1.0, 2.0], &taps), [1.0, 4.0]);
        assert_eq!(filter.push(vec![3.0, 4.0], &taps), [13.0, 48.0]);
        assert_eq!(filter.push(vec![5.0, 6.0], &taps), [135.0, 492.0]);
        assert_eq!(filter.push(vec![7.0, 8.0], &taps), [357.0, 936.0]);
        assert_eq!(filter.history.len(), 3);
    }
}
