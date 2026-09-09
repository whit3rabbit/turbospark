//! FP32 references for `utility.metal`'s gating kernels: Qwen 3.6's
//! full-attention output gate, shared-expert scalar gate, and packed-q_proj
//! split, plus the Spark-X2.5 port-local additions beside them (per-head
//! output gate, exact-erf gated GELU, fused qkv split).

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

/// `out[i] *= sigmoid(gate[i / head_dim])` (`sigmoid_head_gate_mul_fp16`).
/// Spark-X2.5's attention output gate is per HEAD: one logit per head scales
/// that head's whole slice of the projection, where Qwen's gate above is as
/// long as the row.
pub fn sigmoid_head_gate_mul(out: &[f32], gate: &[f32], head_dim: usize) -> Vec<f32> {
    assert!(head_dim > 0, "head_dim must be non-zero");
    assert_eq!(
        out.len() % head_dim,
        0,
        "out length must be a multiple of head_dim"
    );
    assert_eq!(out.len() / head_dim, gate.len(), "one gate per head");
    out.iter()
        .enumerate()
        .map(|(i, &o)| o * sigmoid(gate[i / head_dim]))
        .collect()
}

/// `y[i] *= sigmoid(gate)` (`sigmoid_scalar_mul_fp16`). One scalar logit
/// gates the whole shared-expert output.
pub fn sigmoid_scalar_mul(y: &[f32], gate: f32) -> Vec<f32> {
    let s = sigmoid(gate);
    y.iter().map(|&v| v * s).collect()
}

/// `out[i] = gelu_erf(gate[i]) * up[i]` (`gelu_erf_mul_fp16`). Spark-X2.5's
/// MLP activation, the EXACT-erf GELU where Gemma's GeGLU uses the tanh
/// approximation. The erf itself is `crate::vision::gelu_erf`'s, reused
/// rather than duplicated: one Abramowitz-Stegun evaluation, already f64 and
/// already the contract of the tower's erf kernel, is the only definition
/// this port should carry.
pub fn gelu_erf_mul(gate: &[f32], up: &[f32]) -> Vec<f32> {
    assert_eq!(gate.len(), up.len());
    crate::vision::gelu_erf(gate)
        .iter()
        .zip(up)
        .map(|(&g, &u)| g * u)
        .collect()
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

/// Splits a fused `[q (q_elems) | k (kv_elems) | v (kv_elems)]` projection
/// output into its three ranges (`split_qkv_fp16`). Spark's `q_k_v_proj` is
/// one weight, so the three attention operands arrive as one contiguous row.
/// A pure copy, which is why the GPU test asserts exact bits.
pub fn split_qkv(src: &[f32], q_elems: usize, kv_elems: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    assert_eq!(src.len(), q_elems + 2 * kv_elems);
    (
        src[..q_elems].to_vec(),
        src[q_elems..q_elems + kv_elems].to_vec(),
        src[q_elems + kv_elems..].to_vec(),
    )
}
