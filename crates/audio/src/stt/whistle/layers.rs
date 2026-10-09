//! Shared normalization, Hadamard feed-forward blocks, and lane mixing.

use super::cact::Archive;
use super::stem::silu;
use crate::{Result, SpeechError};

pub(super) const DIM: usize = 512;
pub(super) const LANES: usize = 4;

pub(super) fn floats(a: &Archive, index: usize, shape: &[usize]) -> Result<Vec<f32>> {
    if a.shape(index)? != shape {
        return Err(SpeechError::Tensor {
            name: format!("whistle.tensor.{index}"),
            why: format!("expected shape {shape:?}, got {:?}", a.shape(index)?),
        });
    }
    a.floats(index)
}

pub(super) fn quantized(
    a: &Archive,
    index: usize,
    shape: &[usize],
) -> Result<super::cact::CqMatrix> {
    if a.shape(index)? != shape {
        return Err(SpeechError::Tensor {
            name: format!("whistle.tensor.{index}"),
            why: format!("expected shape {shape:?}, got {:?}", a.shape(index)?),
        });
    }
    a.quantized(index)
}

pub(super) fn rms(x: &[f32]) -> f32 {
    (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32 + 1e-6)
        .sqrt()
        .recip()
}

pub(super) fn norm(x: &[f32], weights: &[f32]) -> Vec<f32> {
    let mut out = Vec::with_capacity(x.len());
    for row in x.chunks_exact(weights.len()) {
        let scale = rms(row);
        out.extend(
            row.iter()
                .zip(weights)
                .map(|(&v, &w)| v * scale * (1.0 + w)),
        );
    }
    out
}

pub(super) fn softmax(x: &mut [f32]) {
    let peak = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;
    for v in x.iter_mut() {
        *v = (*v - peak).exp();
        sum += *v;
    }
    for v in x {
        *v /= sum;
    }
}

pub(super) fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

pub(super) fn a8(x: &mut [f32]) {
    let peak = x.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let scale = if peak > 0.0 { peak / 127.0 } else { 1.0 };
    for v in x {
        *v = (*v / scale).round().clamp(-127.0, 127.0) * scale;
    }
}

pub(super) struct HadamardMlp {
    diagonals: [Vec<f32>; 5],
    factors: [Vec<f32>; 6],
    v: Vec<f32>,
    u: Vec<f32>,
}

impl HadamardMlp {
    pub fn load(a: &Archive, start: usize) -> Result<Self> {
        Ok(Self {
            diagonals: [
                floats(a, start, &[DIM])?,
                floats(a, start + 1, &[DIM])?,
                floats(a, start + 2, &[DIM])?,
                floats(a, start + 3, &[DIM])?,
                floats(a, start + 4, &[DIM])?,
            ],
            factors: [
                floats(a, start + 5, &[16, 16])?,
                floats(a, start + 6, &[32, 32])?,
                floats(a, start + 7, &[16, 16])?,
                floats(a, start + 8, &[32, 32])?,
                floats(a, start + 9, &[16, 16])?,
                floats(a, start + 10, &[32, 32])?,
            ],
            v: floats(a, start + 11, &[DIM, 8])?,
            u: floats(a, start + 12, &[8, DIM])?,
        })
    }

    pub fn forward(&self, x: &[f32], p1: &[usize], p2: &[usize]) -> Vec<f32> {
        let mut output = Vec::with_capacity(x.len());
        for x in x.chunks_exact(DIM) {
            let mut c = [0.0; 8];
            for (i, &v) in x.iter().enumerate() {
                for (j, c) in c.iter_mut().enumerate() {
                    *c += v * self.v[i * 8 + j];
                }
            }
            softmax(&mut c);
            let cond: Vec<f32> = (0..DIM)
                .map(|i| 1.0 + (0..8).map(|j| c[j] * self.u[j * DIM + i]).sum::<f32>())
                .collect();
            let z: Vec<f32> = (0..DIM).map(|i| x[i] * self.diagonals[0][i]).collect();
            let y = kron(&z, &self.factors[0], &self.factors[1]);
            let z: Vec<f32> = (0..DIM)
                .map(|i| silu(self.diagonals[1][i] * cond[i] * y[p1[i]] + self.diagonals[2][i]))
                .collect();
            let y = kron(&z, &self.factors[2], &self.factors[3]);
            let z: Vec<f32> = (0..DIM).map(|i| self.diagonals[3][i] * y[p2[i]]).collect();
            let y = kron(&z, &self.factors[4], &self.factors[5]);
            output.extend((0..DIM).map(|i| self.diagonals[4][i] * y[i]));
        }
        output
    }
}

fn kron(x: &[f32], a: &[f32], b: &[f32]) -> Vec<f32> {
    let mut temp = vec![0.0; DIM];
    for k in 0..16 {
        for j in 0..32 {
            for i in 0..16 {
                temp[k * 32 + j] += x[i * 32 + j] * a[i * 16 + k];
            }
        }
    }
    let mut out = vec![0.0; DIM];
    for k in 0..16 {
        for l in 0..32 {
            for j in 0..32 {
                out[k * 32 + l] += temp[k * 32 + j] * b[j * 32 + l];
            }
        }
    }
    out
}

pub(super) fn permutation(a: &Archive, index: usize) -> Result<Vec<usize>> {
    let values = floats(a, index, &[DIM])?;
    let mut seen = [false; DIM];
    let mut p = Vec::with_capacity(DIM);
    for v in values {
        if v < 0.0 || v >= DIM as f32 || v.fract() != 0.0 || seen[v as usize] {
            return Err(SpeechError::Tensor {
                name: format!("whistle.tensor.{index}"),
                why: "invalid Hadamard permutation".into(),
            });
        }
        seen[v as usize] = true;
        p.push(v as usize);
    }
    Ok(p)
}

struct Phi {
    rows: Vec<Vec<f32>>,
}

impl Phi {
    fn load(a: &Archive, index: usize, rows: usize) -> Result<Self> {
        // The released lane mixer rounds reconstructed Phi rows to A8 once,
        // then rounds each normalized stream frame before its ordinary dot.
        let matrix = quantized(a, index, &[rows, LANES * DIM])?;
        let rows = (0..rows)
            .map(|i| {
                let mut row = matrix.row(i);
                a8(&mut row);
                row
            })
            .collect();
        Ok(Self { rows })
    }
    fn apply(&self, x: &[f32], start: usize, count: usize) -> Vec<f32> {
        self.rows[start..start + count]
            .iter()
            .map(|w| x.iter().zip(w).map(|(x, w)| x * w).sum())
            .collect()
    }
}

pub(super) struct Mhc {
    a_pre: Vec<f32>,
    a_post: Vec<f32>,
    a_res: Vec<f32>,
    b_pre: Vec<f32>,
    b_post: Vec<f32>,
    b_res: Vec<f32>,
    pre: Phi,
    post: Phi,
    res: Phi,
}

pub(super) struct MhcState {
    normalized: Vec<f32>,
    original: Vec<f32>,
    mixed: Vec<f32>,
}

impl Mhc {
    pub fn load(a: &Archive, start: usize) -> Result<Self> {
        Ok(Self {
            a_pre: floats(a, start, &[8])?,
            a_post: floats(a, start + 1, &[8])?,
            a_res: floats(a, start + 2, &[8])?,
            b_pre: floats(a, start + 3, &[8, LANES])?,
            b_post: floats(a, start + 4, &[8, LANES])?,
            b_res: floats(a, start + 5, &[8, LANES, LANES])?,
            pre: Phi::load(a, start + 6, 8 * LANES)?,
            post: Phi::load(a, start + 7, 8 * LANES)?,
            res: Phi::load(a, start + 8, 8 * LANES * LANES)?,
        })
    }

    pub fn read(&self, streams: &[f32], layer: usize) -> (Vec<f32>, MhcState) {
        let mut normalized = Vec::with_capacity(streams.len());
        let mut mixed = Vec::with_capacity(streams.len() / LANES);
        for frame in streams.chunks_exact(LANES * DIM) {
            let scale = rms(frame);
            let mut nx: Vec<f32> = frame.iter().map(|v| v * scale).collect();
            a8(&mut nx);
            let dots = self.pre.apply(&nx, layer * LANES, LANES);
            let mut row = vec![0.0; DIM];
            for lane in 0..LANES {
                let offset = if lane == layer % LANES { 4.0 } else { -4.0 };
                let gate = sigmoid(
                    self.a_pre[layer] * dots[lane] + self.b_pre[layer * LANES + lane] + offset,
                );
                for i in 0..DIM {
                    row[i] += gate * frame[lane * DIM + i];
                }
            }
            normalized.extend(nx);
            mixed.extend(row);
        }
        (
            mixed.clone(),
            MhcState {
                normalized,
                original: streams.to_vec(),
                mixed,
            },
        )
    }

    pub fn write(&self, state: MhcState, output: &[f32], layer: usize) -> Vec<f32> {
        let mut streams = vec![0.0; state.original.len()];
        for (t, x) in state.original.chunks_exact(LANES * DIM).enumerate() {
            let nx = &state.normalized[t * LANES * DIM..(t + 1) * LANES * DIM];
            let post = self.post.apply(nx, layer * LANES, LANES);
            let res = self.res.apply(nx, layer * LANES * LANES, LANES * LANES);
            let logits: Vec<f32> = res
                .iter()
                .enumerate()
                .map(|(i, &v)| self.a_res[layer] * v + self.b_res[layer * LANES * LANES + i])
                .collect();
            let mix = sinkhorn(&logits);
            for lane in 0..LANES {
                let offset = if lane == layer % LANES { 0.0 } else { -4.0 };
                let gate = 2.0
                    * sigmoid(
                        self.a_post[layer] * post[lane]
                            + self.b_post[layer * LANES + lane]
                            + offset,
                    );
                for i in 0..DIM {
                    let residual = (0..LANES)
                        .map(|j| mix[lane * LANES + j] * x[j * DIM + i])
                        .sum::<f32>();
                    streams[(t * LANES + lane) * DIM + i] =
                        residual + gate * (output[t * DIM + i] - state.mixed[t * DIM + i]);
                }
            }
        }
        streams
    }
}

fn sinkhorn(logits: &[f32]) -> Vec<f32> {
    let mut logs = logits.to_vec();
    for _ in 0..20 {
        for row in logs.chunks_exact_mut(LANES) {
            log_normalize(row);
        }
        for c in 0..LANES {
            let mut col: Vec<f32> = (0..LANES).map(|r| logs[r * LANES + c]).collect();
            log_normalize(&mut col);
            for r in 0..LANES {
                logs[r * LANES + c] = col[r];
            }
        }
    }
    logs.into_iter().map(f32::exp).collect()
}

fn log_normalize(x: &mut [f32]) {
    let peak = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let logsum = peak + x.iter().map(|v| (*v - peak).exp()).sum::<f32>().ln();
    for v in x {
        *v -= logsum;
    }
}
