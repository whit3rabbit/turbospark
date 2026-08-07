//! FP32 references for `utility.metal`'s three Qwen 3.6 gating kernels:
//! the full-attention output gate, the shared-expert scalar gate, and the
//! packed-q_proj split those two sit around.

use crate::gdn::sigmoid;

/// `out[i] *= sigmoid(gate[i])` (`sigmoid_gate_mul_fp16`). Qwen's
/// full-attention layers gate the attention output per element.
pub fn sigmoid_gate_mul(out: &[f32], gate: &[f32]) -> Vec<f32> {
    assert_eq!(out.len(), gate.len());
    out.iter()
        .zip(gate)
        .map(|(&o, &g)| o * sigmoid(g))
        .collect()
}

/// `y[i] *= sigmoid(gate)` (`sigmoid_scalar_mul_fp16`). One scalar logit
/// gates the whole shared-expert output.
pub fn sigmoid_scalar_mul(y: &[f32], gate: f32) -> Vec<f32> {
    let s = sigmoid(gate);
    y.iter().map(|&v| v * s).collect()
}

/// Splits a `[heads, 2 * dim]` packed projection into contiguous
/// `[heads, dim]` query and gate halves (`split_q_gate_fp16`). Qwen's
/// `q_proj` emits per-head `[query; gate]` pairs, which the per-head norm,
/// RoPE, and attention kernels cannot consume interleaved.
pub fn split_q_gate(packed: &[f32], heads: usize, dim: usize) -> (Vec<f32>, Vec<f32>) {
    assert_eq!(packed.len(), heads * 2 * dim);
    let mut q = vec![0.0f32; heads * dim];
    let mut gate = vec![0.0f32; heads * dim];
    for head in 0..heads {
        for i in 0..dim {
            q[head * dim + i] = packed[head * 2 * dim + i];
            gate[head * dim + i] = packed[head * 2 * dim + dim + i];
        }
    }
    (q, gate)
}
