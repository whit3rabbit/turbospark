//! Per-layer real-checkpoint Gemma 4 state (effective router scales,
//! expert scales, and layer scalars) built once at open time.

use model_io::{ArchConfig, ResidentIndex};

use crate::real_forward_types::{RealForwardError, MAX_PREFILL_BATCH};
use crate::real_forward_utils::{layer_tensor, read_bf16_host};

/// Per-layer real-checkpoint state built once at open.
pub(crate) struct RealGemmaState {
    /// BF16 `[hidden]` per layer: the checkpoint's `router.scale` with
    /// `1/sqrt(hidden)` pre-folded (the Swift effective-scale buffers).
    pub(crate) effective_scale: Vec<gpu::MetalBuffer>,
    /// Decoded `router.per_expert_scale` per layer.
    pub(crate) per_expert_scale: Vec<Vec<f32>>,
    /// Decoded `layer_scalar` per layer.
    pub(crate) layer_scalar: Vec<f32>,
    /// FP32 router logits (the INT8 router kernel writes float), one
    /// `[num_experts]` row per token of a prefill micro-batch. All M rows
    /// are read back together, after the ONE command buffer that produced
    /// them; that readback is what the chunk driver exists to amortize.
    pub(crate) router_logits_f32: gpu::MetalBuffer,
    /// `[hidden]` FP16 branch scratch, one row per token of a prefill
    /// micro-batch: both are written in the attention half of a layer and
    /// read in the routed half, which the driver runs as two separate
    /// passes over the chunk.
    pub(crate) dense_x: gpu::MetalBuffer,
    pub(crate) routed_x: gpu::MetalBuffer,
    /// `[hidden]` FP16, ONE row: written and read inside a single token's
    /// dispatch run, so a serial encoder makes reuse across the chunk safe.
    pub(crate) router_x: gpu::MetalBuffer,
    /// `[hidden]` FP16, ONE row. The FFN halves are written and read by the
    /// GPU alone, and command buffers on one queue run in commit order, so
    /// the tokens of a chunk cannot overlap on them (see `ROUTED_BANKS`).
    pub(crate) h1: gpu::MetalBuffer,
    pub(crate) h2: gpu::MetalBuffer,
}

impl RealGemmaState {
    pub(crate) fn build(
        context: &mut gpu::MetalContext,
        weights: &gpu::ResidentGpuWeights,
        index: &ResidentIndex,
        arch: &ArchConfig,
    ) -> Result<Self, RealForwardError> {
        if !arch.ffn_sandwich_norms || !arch.router_scaled {
            return Err(RealForwardError::Unsupported(
                "real-checkpoint flow currently implements Gemma 4 only \
                 (ffn_sandwich_norms + router_scaled)"
                    .to_string(),
            ));
        }
        if arch.num_experts <= 0 || arch.top_k_experts <= 0 {
            return Err(RealForwardError::Unsupported(
                "real Gemma 4 installs are MoE; num_experts/top_k must be positive".to_string(),
            ));
        }
        if arch.top_k_experts as usize > gpu::MAX_STREAMED_EXPERTS {
            return Err(RealForwardError::Unsupported(format!(
                "top_k {} exceeds the {}-slot MoE kernels",
                arch.top_k_experts,
                gpu::MAX_STREAMED_EXPERTS
            )));
        }

        let hidden = arch.hidden_size as usize;
        let num_layers = arch.num_layers as usize;
        let num_experts = arch.num_experts as usize;
        let inv_sqrt_d = 1.0f32 / (hidden as f32).sqrt();

        let mut effective_scale = Vec::with_capacity(num_layers);
        let mut per_expert_scale = Vec::with_capacity(num_layers);
        let mut layer_scalar = Vec::with_capacity(num_layers);
        for layer in 0..num_layers {
            // Pre-fold 1/sqrt(D) into router.scale once per open, as the
            // Swift runner does per generation.
            let scale = read_bf16_host(weights, index, &layer_tensor(layer, "router.scale"))?;
            if scale.len() != hidden {
                return Err(RealForwardError::Unsupported(format!(
                    "router.scale layer {layer}: {} elements, expected {hidden}",
                    scale.len()
                )));
            }
            let folded: Vec<u8> = scale
                .iter()
                .flat_map(|&v| compute::f32_to_bf16(v * inv_sqrt_d).to_le_bytes())
                .collect();
            let buffer = context.new_output_buffer(folded.len() as u64);
            gpu::write_buffer_bytes(&buffer, 0, &folded);
            effective_scale.push(buffer);

            let pes = read_bf16_host(
                weights,
                index,
                &layer_tensor(layer, "router.per_expert_scale"),
            )?;
            if pes.len() != num_experts {
                return Err(RealForwardError::Unsupported(format!(
                    "per_expert_scale layer {layer}: {} elements, expected {num_experts}",
                    pes.len()
                )));
            }
            per_expert_scale.push(pes);

            let scalar = read_bf16_host(weights, index, &layer_tensor(layer, "layer_scalar"))?;
            layer_scalar.push(*scalar.first().ok_or_else(|| {
                RealForwardError::Unsupported(format!("layer_scalar layer {layer} is empty"))
            })?);
        }

        let halfs = |n: usize| context.new_output_buffer((n.max(1) * 2) as u64);
        Ok(Self {
            router_logits_f32: context
                .new_output_buffer((num_experts * 4 * MAX_PREFILL_BATCH) as u64),
            dense_x: halfs(hidden * MAX_PREFILL_BATCH),
            routed_x: halfs(hidden * MAX_PREFILL_BATCH),
            router_x: halfs(hidden),
            h1: halfs(hidden),
            h2: halfs(hidden),
            effective_scale,
            per_expert_scale,
            layer_scalar,
        })
    }
}
