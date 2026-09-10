//! Family-specific router arithmetic shared by decode and chunked prefill.
use super::{layer_tensor, state::RealLlamaState};
use crate::real_forward::RealForwardError;
use crate::real_forward_utils::entry;
use model_io::ResidentIndex;

pub(crate) fn fp32_tensor(name: &str) -> bool {
    name.starts_with("language_model.model.layers.")
        && (name.ends_with(".mlp.gate.weight") || name.ends_with(".mlp.e_score_correction_bias"))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn encode(
    context: &mut gpu::MetalContext,
    pass: &gpu::PassEncoder,
    weights: &gpu::ResidentGpuWeights,
    index: &ResidentIndex,
    state: &RealLlamaState,
    layer: usize,
    token: usize,
    hidden: usize,
    experts: usize,
) -> Result<(), RealForwardError> {
    let name = layer_tensor(layer, "mlp.gate.weight");
    let router = entry(index, &name)?;
    let fp32 = !state.correction_bias.is_empty();
    let dtype = if fp32 { 3 } else { 5 };
    let size = experts * hidden * if fp32 { 4 } else { 1 };
    if router.dtype != dtype || router.size_bytes as usize != size {
        return Err(RealForwardError::Unsupported(format!(
            "{name}: expected dtype {dtype} and {size} bytes"
        )));
    }
    let base = index.header.index_size;
    let view = |off| (weights.buffer(), weights.gpu_offset(off - base));
    let x = (&state.moe_x, (token * hidden * 2) as u64);
    let out = (&state.router_logits_f32, (token * experts * 4) as u64);
    if fp32 {
        gpu::encode_minimax_router(
            context,
            pass,
            view(router.file_offset),
            x,
            out,
            experts as u32,
            hidden as u32,
        )
        .map_err(RealForwardError::Gpu)?;
    } else {
        gpu::encode_router_gemv_gemma4(
            context,
            pass,
            view(router.file_offset),
            view(router.scale_offset),
            view(router.bias_offset),
            x,
            (&state.router_ones, 0),
            out,
            experts as u32,
            hidden as u32,
        )
        .map_err(RealForwardError::Gpu)?;
    }
    Ok(())
}

pub(crate) fn sigmoid_topk(logits: &[f32], bias: &[f32], k: usize) -> (Vec<usize>, Vec<f32>) {
    assert_eq!(logits.len(), bias.len());
    let scores: Vec<f32> = logits
        .iter()
        .map(|&x| {
            if x >= 0.0 {
                1.0 / (1.0 + (-x).exp())
            } else {
                let e = x.exp();
                e / (1.0 + e)
            }
        })
        .collect();
    let mut order: Vec<usize> = (0..scores.len()).collect();
    order.sort_by(|&a, &b| {
        (scores[b] + bias[b])
            .total_cmp(&(scores[a] + bias[a]))
            .then(a.cmp(&b))
    });
    order.truncate(k);
    let sum = order.iter().map(|&i| scores[i]).sum::<f32>() + 1e-20;
    let weights = order.iter().map(|&i| scores[i] / sum).collect();
    (order, weights)
}

#[cfg(test)]
mod tests {
    use super::sigmoid_topk;
    #[test]
    fn correction_selects_but_does_not_weight() {
        let (ids, w) = sigmoid_topk(&[0.0, 3f32.ln(), -3f32.ln()], &[0.0, 0.0, 1.0], 2);
        assert_eq!(ids, [2, 1]);
        assert!((w[0] - 0.25).abs() < 1e-7);
        assert!((w[1] - 0.75).abs() < 1e-7);
        // Exercise the correction on both sides of the sort comparator.
        let (ids, w) = sigmoid_topk(&[-3f32.ln(), 3f32.ln(), 0.0], &[1.0, 0.0, 0.0], 2);
        assert_eq!(ids, [0, 1]);
        assert!((w[0] - 0.25).abs() < 1e-7);
    }
    #[test]
    fn sigmoid_weights_are_not_softmax_and_ties_are_stable() {
        let (ids, w) = sigmoid_topk(&[0.0, 0.0, 3f32.ln()], &[0.0; 3], 3);
        assert_eq!(ids, [2, 0, 1]);
        assert!((w[0] - 3.0 / 7.0).abs() < 1e-7);
        assert!((w[2] - 2.0 / 7.0).abs() < 1e-7);
        assert_eq!(sigmoid_topk(&[0.0; 3], &[0.0; 3], 2).0, [0, 1]);
    }
}
