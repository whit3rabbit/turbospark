//! FP32 fused MoE FFN reference: top-k routed experts, each with gate/up/down
//! projections, GeGLU activation, and a weighted combine into the residual
//! stream. Ported from `Support/Reference/MoE/Moe.swift`.

use crate::quant::{dequant_int4_gemv, Int4AffineRow};

const GELU_COEFF: f32 = 0.797_884_6; // sqrt(2 / pi)
const GELU_CUBIC: f32 = 0.044715;

/// Per-element `gelu_pytorch_tanh(x) = 0.5 * x * (1 + tanh(c * (x + k * x^3)))`.
pub fn gelu_tanh(x: &[f32]) -> Vec<f32> {
    x.iter()
        .map(|&xv| {
            let inner = GELU_COEFF * (xv + GELU_CUBIC * xv * xv * xv);
            0.5 * xv * (1.0 + inner.tanh())
        })
        .collect()
}

/// Runs one FFN block: `down(gelu(gate(x)) * up(x))`.
/// `gate_rows` / `up_rows` are F-by-D affine INT4; `down_rows` is D-by-F.
pub fn run_ffn(
    gate_rows: &[Int4AffineRow],
    up_rows: &[Int4AffineRow],
    down_rows: &[Int4AffineRow],
    x: &[f32],
    d: usize,
    f: usize,
) -> Vec<f32> {
    assert_eq!(gate_rows.len(), f, "gate_rows must be F={f}");
    assert_eq!(up_rows.len(), f, "up_rows must be F={f}");
    assert_eq!(down_rows.len(), d, "down_rows must be D={d}");
    assert_eq!(x.len(), d);

    let gate_out = dequant_int4_gemv(gate_rows, x, d);
    let up_out = dequant_int4_gemv(up_rows, x, d);
    let gated = gelu_tanh(&gate_out);
    let act: Vec<f32> = gated
        .iter()
        .zip(up_out.iter())
        .map(|(g, u)| g * u)
        .collect();
    dequant_int4_gemv(down_rows, &act, f)
}

/// Routed-only half of Gemma 4's parallel-MoE block (dense MLP and routed
/// branches are computed separately and summed elsewhere).
/// `y = residual + sum_slot(weight[slot] * routed_expert(x))`.
#[allow(clippy::too_many_arguments)]
pub fn apply_streamed_routed(
    x: &[f32],
    residual: &[f32],
    routed_gate: &[Vec<Int4AffineRow>],
    routed_up: &[Vec<Int4AffineRow>],
    routed_down: &[Vec<Int4AffineRow>],
    indices: &[usize],
    routing_weights: &[f32],
    d: usize,
    f: usize,
) -> Vec<f32> {
    assert_eq!(indices.len(), routing_weights.len());
    assert_eq!(residual.len(), d);

    let mut y = residual.to_vec();
    for (slot, &e) in indices.iter().enumerate() {
        let w = routing_weights[slot];
        let out = run_ffn(&routed_gate[e], &routed_up[e], &routed_down[e], x, d, f);
        for (yv, ov) in y.iter_mut().zip(out.iter()) {
            *yv += w * ov;
        }
    }
    y
}
