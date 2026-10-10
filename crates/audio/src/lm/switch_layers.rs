//! Switched (per-token routed) feed-forward layers.
//!
//! Port of the `mlx_audio/lm/models/switch_layers.py` subset bailing_moe
//! uses (vendored from mlx-lm) at the pinned mlx-audio commit: the
//! `SwitchLinear` gather matmul and `SwitchGLU`. Single batch. The
//! reference sorts token indices with `_gather_sort` once `indices.size >=
//! 64` so expert access is ordered; that is a memory-access optimization
//! around `mx.gather_mm` and does not change any per-token arithmetic, so
//! this port computes each (token, slot) pair directly in router order.
//! `SwitchMLP` and the quantized switch linears stay unported until a
//! family needs them.

use crate::{Result, SpeechError};

/// `SwitchGLU`: per-slot SwiGLU experts with weights stored
/// `[num_experts, out, in]` (the reference `SwitchLinear` layout; note the
/// checkpoint tensors are not HF `[out, in]` pairs but a stacked expert
/// axis, saved after the model sanitize joins `mlp.experts.*`).
#[derive(Debug, Clone)]
pub struct SwitchGlu {
    /// `[num_experts, hidden_dims, input_dims]`.
    gate_proj: Vec<f32>,
    /// `[num_experts, hidden_dims, input_dims]`.
    up_proj: Vec<f32>,
    /// `[num_experts, input_dims, hidden_dims]`.
    down_proj: Vec<f32>,
    /// Optional `[num_experts, hidden_dims]` shifts, present when the
    /// checkpoint trains `use_bias`.
    gate_bias: Option<Vec<f32>>,
    up_bias: Option<Vec<f32>>,
    down_bias: Option<Vec<f32>>,
    num_experts: usize,
    input_dims: usize,
    hidden_dims: usize,
}

impl SwitchGlu {
    #[allow(
        clippy::too_many_arguments,
        reason = "the three expert stacks carry their biases and dims explicitly"
    )]
    pub fn new(
        gate_proj: Vec<f32>,
        up_proj: Vec<f32>,
        down_proj: Vec<f32>,
        gate_bias: Option<Vec<f32>>,
        up_bias: Option<Vec<f32>>,
        down_bias: Option<Vec<f32>>,
        num_experts: usize,
        input_dims: usize,
        hidden_dims: usize,
    ) -> Self {
        Self {
            gate_proj,
            up_proj,
            down_proj,
            gate_bias,
            up_bias,
            down_bias,
            num_experts,
            input_dims,
            hidden_dims,
        }
    }

    /// The per-expert output width (the down projection's output, equal to
    /// the block input width).
    pub fn output_dims(&self) -> usize {
        self.input_dims
    }

    /// Runs the routed experts for `x [len, input_dims]` and
    /// `indices [len, slots]` (the router output, one expert id per slot).
    /// Returns `[len, slots, input_dims]`, slot-major per token in the
    /// given router order: slot order is reduction order for the
    /// aggregation that follows.
    pub fn forward(&self, x: &[f32], len: usize, indices: &[i32]) -> Result<Vec<f32>> {
        let slots = indices.len() / len;
        let mut out = vec![0.0f32; len * slots * self.input_dims];
        for token in 0..len {
            let row = &x[token * self.input_dims..(token + 1) * self.input_dims];
            for slot in 0..slots {
                let expert = usize::try_from(indices[token * slots + slot]).map_err(|_| {
                    SpeechError::Input {
                        why: format!("negative expert index {}", indices[token * slots + slot]),
                    }
                })?;
                if expert >= self.num_experts {
                    return Err(SpeechError::Input {
                        why: format!(
                            "expert index {expert} outside {} routed experts",
                            self.num_experts
                        ),
                    });
                }
                let mut gate = self.expert_proj(&self.gate_proj, expert, row);
                if let Some(bias) = &self.gate_bias {
                    let bias = &bias[expert * self.hidden_dims..(expert + 1) * self.hidden_dims];
                    for (value, bias) in gate.iter_mut().zip(bias) {
                        *value += bias;
                    }
                }
                let mut up = self.expert_proj(&self.up_proj, expert, row);
                if let Some(bias) = &self.up_bias {
                    let bias = &bias[expert * self.hidden_dims..(expert + 1) * self.hidden_dims];
                    for (value, bias) in up.iter_mut().zip(bias) {
                        *value += bias;
                    }
                }
                let mut fused = vec![0.0f32; self.hidden_dims];
                for d in 0..self.hidden_dims {
                    fused[d] = gate[d] * (1.0 / (1.0 + (-gate[d]).exp())) * up[d];
                }
                let hidden = &self.down_proj[expert * self.input_dims * self.hidden_dims
                    ..(expert + 1) * self.input_dims * self.hidden_dims];
                let dst = (token * slots + slot) * self.input_dims;
                for (o, out_value) in out[dst..dst + self.input_dims].iter_mut().enumerate() {
                    let mut dot = 0.0f32;
                    for d in 0..self.hidden_dims {
                        // down_proj stores [out = input_dims, in =
                        // hidden_dims] per expert; gather_mm applies the
                        // transpose.
                        dot += fused[d] * hidden[o * self.hidden_dims + d];
                    }
                    if let Some(bias) = &self.down_bias {
                        dot += bias[expert * self.input_dims + o];
                    }
                    *out_value = dot;
                }
            }
        }
        Ok(out)
    }

    /// One expert matmul: `weight [out, in]` slice of the stacked
    /// `[num_experts, out, in]` tensor times `row`.
    fn expert_proj(&self, weight: &[f32], expert: usize, row: &[f32]) -> Vec<f32> {
        let base = expert * self.hidden_dims * self.input_dims;
        let mut out = vec![0.0f32; self.hidden_dims];
        for (o, out_value) in out.iter_mut().enumerate() {
            let mut dot = 0.0f32;
            for (i, &value) in row.iter().enumerate() {
                dot += value * weight[base + o * self.input_dims + i];
            }
            *out_value = dot;
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn two_expert_switch() -> SwitchGlu {
        // Two experts over input 2, hidden 2. Expert identity: the down
        // projection emits the summed inputs so expert effects are easy
        // to tell apart.
        SwitchGlu::new(
            // gate_proj [E, hidden, in]: expert e row-major.
            vec![1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0],
            // up_proj: identity for both experts.
            vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0],
            // down_proj [E, out, hidden]: sums the hidden dims.
            vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            None,
            None,
            None,
            2,
            2,
            2,
        )
    }

    #[test]
    fn experts_dispatch_by_index_and_slot_order() {
        let switch = two_expert_switch();
        // Token 0 -> expert 0, token 1 -> expert 1, one slot.
        let out = switch.forward(&[1.0, 2.0, 3.0, 4.0], 2, &[0, 1]).unwrap();
        assert_eq!(out.len(), 4);
        // Expert 0: gate and up are the identity, so the fused value is
        // silu(x) * x; the down projection sums the hidden dims.
        let silu = |v: f32| v / (1.0 + (-v).exp());
        let want0 = silu(1.0) + silu(2.0) * 2.0;
        assert!((out[0] - want0).abs() < 1e-6, "{} vs {want0}", out[0]);
        assert!((out[1] - want0).abs() < 1e-6, "{} vs {want0}", out[1]);
        // Token 1 routed to expert 1: gate = 2x, up = x.
        let want1 = silu(6.0) * 3.0 + silu(8.0) * 4.0;
        assert!((out[2] - want1).abs() < 1e-6, "{} vs {want1}", out[2]);
        assert!((out[3] - want1).abs() < 1e-6, "{} vs {want1}", out[3]);
    }

    #[test]
    fn expert_index_out_of_range_is_an_input_error() {
        let switch = two_expert_switch();
        let error = switch.forward(&[1.0, 2.0], 1, &[2]).unwrap_err();
        assert!(
            error.to_string().contains("outside 2 routed experts"),
            "{error}"
        );
        let error = switch.forward(&[1.0, 2.0], 1, &[-1]).unwrap_err();
        assert!(
            error.to_string().contains("negative expert index"),
            "{error}"
        );
    }
}
