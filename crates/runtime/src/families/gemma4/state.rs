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
    /// The chunk driver's batched routed scratch, allocated ON FIRST USE
    /// rather than at open. See [`BatchedPrefillScratch`].
    pub(crate) batched: Option<BatchedPrefillScratch>,
}

/// Batched routed-expert scratch (`docs/BATCHED_PREFILL.md` steps 2
/// and 3), all sized for [`MAX_PREFILL_BATCH`] rows and used only by
/// the chunk driver's batched routed half:
/// - `batch_acts`: `[M * top_k, moe_inter]` FP16, written by the
///   route-list phase 1;
/// - `batch_y`: `[M, hidden]` FP16, the fused phase-2 output the
///   sandwich tail reads per token;
/// - `batch_h1`: `[M, hidden]` FP16, the shared-expert branch's
///   output rows (the batched tail consumes all M tokens' shared
///   outputs after they have all run, so the single-row `h1` above
///   cannot be reused per token the way the per-token path does);
/// - `batch_routing_w`: `[M * top_k]` FP16, one HOST write per layer;
/// - `batch_routes`: the encoded route list, likewise host-written.
///   Both are single-banked: the driver retires the previous layer's
///   routed command buffer before this layer's routed half begins,
///   so no in-flight dispatch reads them while they are rewritten.
///
/// **ALLOCATED ON FIRST USE, so an install that never chunks its prefill
/// allocates none of it.** That is the rule this engine follows everywhere
/// else and this struct was the one place breaking it: `RealForwardRunner`
/// holds `real_mtp` and `real_dflash` as `Option`s for exactly this reason,
/// and `families/qwen/batched_scratch.rs` states it outright -- an
/// unasked-for feature must allocate nothing, so that a frozen memory-oracle
/// row keeps describing the engine that shipped before the feature existed.
///
/// The stake is small and saying so is part of the point: on the real Gemma 4
/// install (hidden 2816, moe_inter 704, top_k 8, `MAX_PREFILL_BATCH` 16) the
/// six buffers come to ~354 KiB, well inside the 77 MiB run-to-run spread of
/// the oracle's own peak. This is consistency with a rule, not a memory win,
/// and a future `MAX_PREFILL_BATCH` or a wider model is what would make it
/// one.
///
/// **Lazily and not from a flag at open**, because there is no open-time
/// signal to key on: `prefill_chunk` is a `LogitProducer` method any caller
/// may reach at any point in a run, and the CLI's own route to it is an
/// environment variable read per generation (`crates/cli/CLAUDE.md`
/// Gotcha 7), not an option the runner is opened with.
pub(crate) struct BatchedPrefillScratch {
    pub(crate) batch_acts: gpu::MetalBuffer,
    pub(crate) batch_y: gpu::MetalBuffer,
    pub(crate) batch_h1: gpu::MetalBuffer,
    pub(crate) batch_routing_w: gpu::MetalBuffer,
    pub(crate) batch_routes: gpu::MetalBuffer,
    /// The wide expert-blob argument buffer the batched pair reads
    /// through (up to [`gpu::MAX_PREFILL_EXPERT_BINDINGS`] pointers, one
    /// per cache slot, bound once per layer).
    pub(crate) wide_blobs: gpu::RoutedBlobsWideBuffer,
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
            batched: None,
            effective_scale,
            per_expert_scale,
            layer_scalar,
        })
    }

    /// Allocates [`BatchedPrefillScratch`] the first time the chunk driver
    /// asks for it, and is a no-op afterwards.
    ///
    /// Idempotent rather than "call once", because the entry point that calls
    /// it (`prefill_chunk_real_gemma4`) is reached once per CHUNK, not once
    /// per run.
    pub(crate) fn ensure_batched(
        &mut self,
        context: &mut gpu::MetalContext,
        arch: &ArchConfig,
    ) -> Result<(), RealForwardError> {
        if self.batched.is_some() {
            return Ok(());
        }
        let use_silu = arch.hidden_activation.contains("silu");
        let wide_blobs =
            gpu::RoutedBlobsWideBuffer::new(context, use_silu).map_err(RealForwardError::Gpu)?;
        let top_k = arch.top_k_experts as usize;
        let hidden = arch.hidden_size as usize;
        let moe_inter = arch.moe_intermediate_size.max(1) as usize;
        let halfs = |n: usize| context.new_output_buffer((n.max(1) * 2) as u64);
        let batch_rows = |per_token: usize| halfs(MAX_PREFILL_BATCH * per_token);
        self.batched = Some(BatchedPrefillScratch {
            batch_acts: batch_rows(top_k * moe_inter),
            batch_y: batch_rows(hidden),
            batch_h1: batch_rows(hidden),
            batch_routing_w: batch_rows(top_k),
            // 16 bytes per encoded route (token, rank, slot, reserved),
            // matching `MoePrefillRoute::bytes` and the shader struct.
            batch_routes: context.new_output_buffer((MAX_PREFILL_BATCH * top_k * 16) as u64),
            wide_blobs,
        });
        Ok(())
    }

    /// The batched scratch, which every caller reaches only from inside the
    /// chunk driver -- downstream of the `ensure_batched` at its entry point,
    /// so the `expect` is a structural invariant rather than a hope.
    pub(crate) fn batched(&self) -> &BatchedPrefillScratch {
        self.batched
            .as_ref()
            .expect("prefill_chunk_real_gemma4 calls ensure_batched before any layer runs")
    }
}
