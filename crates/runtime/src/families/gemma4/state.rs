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
    /// Destination for the one-layer-ahead router probe
    /// (`TURBOSPARK_PILOT_PROBE`): layer L+1's router run early, on layer L's
    /// post-attention residual. Allocated unconditionally (one token of
    /// f32s) but only ever WRITTEN when the probe is on, so the default
    /// decode path dispatches exactly the kernels it always did.
    pub(crate) pilot_logits_f32: gpu::MetalBuffer,
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
/// - `batch_zero`: `[M, hidden]` FP16 of ZEROS, written once here and
///   never again. It is the fused phase 2's accumulator seed, the batched
///   twin of `DecodeScratch::zero_hidden` -- this family adds its shared
///   expert in the sandwich tail after a norm, so the routed sum starts
///   from nothing, exactly as the decode path's phase 2 does. Filled
///   explicitly rather than trusting a fresh `MTLBuffer` to be zeroed,
///   which is `zero_hidden`'s own precedent.
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
/// install (hidden 2816, ffn_intermediate 2112, moe_inter 704, top_k 8, 16
/// heads at full_head_dim 512, `MAX_PREFILL_BATCH` 16) the routed seven come
/// to ~442 KiB and the batched-GEMV seven to ~0.87 MiB, together well inside
/// the 77 MiB run-to-run spread of the oracle's own peak. This is consistency
/// with a rule, not a memory win, and a future `MAX_PREFILL_BATCH` or a wider
/// model is what would make it one.
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
    pub(crate) batch_zero: gpu::MetalBuffer,
    /// The wide expert-blob argument buffer the batched pair reads
    /// through (up to [`gpu::MAX_PREFILL_EXPERT_BINDINGS`] pointers, one
    /// per cache slot, bound once per layer).
    pub(crate) wide_blobs: gpu::RoutedBlobsWideBuffer,
    /// The M-row siblings of the `DecodeScratch` buffers the batched
    /// RESIDENT GEMVs write (`TURBOSPARK_BATCHED_GEMV`). They live here
    /// rather than in a second lazily-allocated struct because the two
    /// seams share one allocation point and neither is reachable outside
    /// the chunk driver.
    ///
    /// `batch_q` and `batch_attn_out` are sized at the LARGEST `q_dim`
    /// this model has, `num_heads * max(head_dim, full_head_dim)`: Gemma 4
    /// gives its five full layers a 512-wide head against the sliding
    /// window's 256, so one buffer serves both and the narrow layers
    /// simply use a prefix of each row. That is the same thing
    /// `DecodeScratch` does with its single-row `q`.
    ///
    /// `x`, `dense_x`, `routed_x`, `router_logits_f32` and `batch_h1` are
    /// NOT duplicated here -- all five already hold `MAX_PREFILL_BATCH`
    /// rows, because the residual stream crosses layers and the routed
    /// half reads its inputs back after a commit.
    pub(crate) batch_normed: gpu::MetalBuffer,
    pub(crate) batch_q: gpu::MetalBuffer,
    pub(crate) batch_attn_out: gpu::MetalBuffer,
    pub(crate) batch_o: gpu::MetalBuffer,
    pub(crate) batch_ffn_gate: gpu::MetalBuffer,
    pub(crate) batch_ffn_up: gpu::MetalBuffer,
    pub(crate) batch_ffn_act: gpu::MetalBuffer,
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
            // One token's worth only: the probe is a decode-path diagnostic
            // and the batched prefill encoders never fill it.
            pilot_logits_f32: context.new_output_buffer((num_experts * 4) as u64),
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
        let inter = arch.intermediate_size as usize;
        // The widest `q_dim` any layer of this model uses. Gemma 4's five
        // full layers carry a 512-wide head against the sliding window's
        // 256, so sizing from `head_dim` alone would under-allocate every
        // full layer's batched q by half.
        let q_dim = (arch.num_heads * arch.head_dim.max(arch.full_head_dim)) as usize;
        let halfs = |n: usize| context.new_output_buffer((n.max(1) * 2) as u64);
        let batch_rows = |per_token: usize| halfs(MAX_PREFILL_BATCH * per_token);
        self.batched = Some(BatchedPrefillScratch {
            batch_acts: batch_rows(top_k * moe_inter),
            batch_y: batch_rows(hidden),
            batch_h1: batch_rows(hidden),
            batch_routing_w: batch_rows(top_k),
            batch_routes: context.new_output_buffer(
                (MAX_PREFILL_BATCH * top_k * gpu::MoePrefillRoute::STRIDE_BYTES) as u64,
            ),
            batch_zero: {
                let bytes = MAX_PREFILL_BATCH * hidden.max(1) * 2;
                let buffer = context.new_output_buffer(bytes as u64);
                gpu::write_buffer_bytes(&buffer, 0, &vec![0u8; bytes]);
                buffer
            },
            wide_blobs,
            batch_normed: batch_rows(hidden),
            batch_q: batch_rows(q_dim),
            batch_attn_out: batch_rows(q_dim),
            batch_o: batch_rows(hidden),
            batch_ffn_gate: batch_rows(inter),
            batch_ffn_up: batch_rows(inter),
            batch_ffn_act: batch_rows(inter),
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
