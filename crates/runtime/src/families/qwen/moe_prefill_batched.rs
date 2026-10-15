//! The batched arm of the qwen MoE chunked prefill driver
//! (`moe_prefill.rs`): `docs/BATCHED_PREFILL.md` step 6 for the MoE half of
//! the qwen linear-attention family, the dense driver's batched arm
//! (`prefill.rs`) carried over to the half that has routed experts.
//!
//! **What batches.** Each layer's `cb1` reads every resident matrix ONCE per
//! micro-batch instead of once per token: the GDN and full-attention
//! projections through the verify pass's own M-row encoders
//! (`verify_layers::encode_qwen_layer_attn_and_norms_batched`), and the
//! shared expert's three projections (`moe::encode_qwen_shared_expert_batched`).
//! Their GEMMs take the vendored Steel kernel from 32 rows when the scratch
//! allows it, at a 64-row micro-batch.
//!
//! **What does not.** The router stays one INT8 GEMV per row (its FP32
//! output has no GEMM sibling, and the host reads all rows back after the one
//! wait either way). The routed experts stay the per-token pipelined loop,
//! fed from this arm's M-row buffers through `moe::MoeRowSources`: the expert
//! `pread` does not batch (`BATCHED_PREFILL.md`), and the union-capped
//! batched routed pair would run sub-batches of about two tokens at this
//! family's 256 experts and top-8.
//!
//! **Numerics.** Without Steel the projections are bit-exact against the
//! per-token GEMVs, and the synthetic fixture reads byte-identical logits.
//! On a real install the M-row GDN and attention blocks are not (the dense
//! arm measured 6.2e-8 to 1.5e-5 nats, `prefill.rs`), and Steel rounds
//! weights to FP16. So, as for the dense arm, the per-token arm
//! (`TURBOSPARK_BATCHED_GEMV=0`) stays the byte-identical reference and this
//! arm is held to a KL probe (`crates/bench/tests/qwen36_prefill_arm_probe.rs`)
//! and the family's quality gate.

use model_io::{ArchConfig, ResidentIndex};

use super::moe::encode_qwen_shared_expert_batched;
use super::verify_layers::{encode_qwen_layer_attn_and_norms_batched, encode_router_gemv_batched};
use super::{layer_tensor, RealQwenState};
use crate::real_forward::RealForwardRunner;
use crate::real_forward_types::{DecodeScratch, RealForwardError};
use crate::real_forward_utils::entry;

impl RealForwardRunner {
    /// Why the MoE driver's batched arm cannot serve this install and
    /// prompt, or `None` when it can. Install-level refusals (hadamard,
    /// shared-read overlap, an open drafter) are `qwen_moe_chunked_refusal`'s
    /// and are checked before this. The forced arm turns a reason into an
    /// error; auto takes the per-token arm instead.
    pub(crate) fn qwen_moe_batched_prefill_refusal(
        &self,
    ) -> Result<Option<String>, RealForwardError> {
        let arch = &self.arch;
        // INT4-affine resident projections only: the M-row GEMM has no BF16
        // kernel (MiniCPM-V 4.7 ships its GDN projections and head as BF16
        // and widens its mxfp4 attention to BF16 at repack), and looping the
        // per-token GEMVs under this arm's label would measure the unbatched
        // engine (crate Gotcha 22). One probe per block kind; a mixed-width
        // install past them still meets `encode_gemm_any`'s own refusal.
        let probes = [
            (0..arch.num_layers as usize)
                .find(|&l| arch.layer_is_linear(l))
                .map(|l| layer_tensor(l, "linear_attn.in_proj_qkv.weight")),
            (0..arch.num_layers as usize)
                .find(|&l| !arch.layer_is_linear(l))
                .map(|l| layer_tensor(l, "self_attn.q_proj.weight")),
        ];
        for probe in probes.into_iter().flatten() {
            let dtype = entry(&self.index, &probe)?.dtype;
            if dtype != 4 {
                return Ok(Some(format!(
                    "TURBOSPARK_BATCHED_GEMV needs INT4-affine (dtype 4) resident weights and \
                     {probe} is dtype {dtype}: the M-row GEMM has no kernel at this width"
                )));
            }
        }
        // The shared expert runs through the same GEMM dispatcher, which
        // serves INT4 (4) and INT8 (5) affine, and 1-bit (15) and 2-bit (16).
        let shared = layer_tensor(0, "mlp.shared_expert.gate_proj.weight");
        let dtype = entry(&self.index, &shared)?.dtype;
        if !matches!(dtype, 4 | 5 | 15 | 16) {
            return Ok(Some(format!(
                "TURBOSPARK_BATCHED_GEMV has no M-row GEMM for {shared} at dtype {dtype}"
            )));
        }
        // THE ANGLE, as in the dense arm: the batched attention block rotates
        // at the raw position and takes no mRoPE triple.
        if self.prompt_vision.is_some() {
            return Ok(Some(
                "TURBOSPARK_BATCHED_GEMV cannot serve an image prompt: the batched \
                 attention block rotates at the raw position and takes no mRoPE triple, \
                 so every image position would get the wrong angle. Unset it to chunk \
                 this prompt, or prefill sequentially"
                    .to_string(),
            ));
        }
        // The M-row K/V projections write the cache directly and are not
        // wired to `--kv-bits` (`batched_layers.rs` refuses it too, deeper).
        if (0..arch.num_layers as usize)
            .any(|layer| !arch.layer_is_linear(layer) && self.kv.layer_quant(layer).is_some())
        {
            return Ok(Some(
                "--kv-bits is not yet supported with TURBOSPARK_BATCHED_GEMV".to_string(),
            ));
        }
        Ok(None)
    }
}

/// One layer's `cb1` on the batched arm, for all `m` rows of the
/// micro-batch, into `pass`: the attention half and both norms
/// (post-attention norm into `batched.moe_x`), one router GEMV per row into
/// `batch_router_logits_f32`, and the gated shared expert into `batch_h1`.
/// The caller commits and waits once, then runs the routed loop from those
/// three buffers.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_qwen_moe_cb1_batched(
    context: &mut gpu::MetalContext,
    pass: &gpu::PassEncoder,
    weights: &gpu::ResidentGpuWeights,
    index: &ResidentIndex,
    scratch: &DecodeScratch,
    kv: &mut gpu::KvCacheManager,
    qwen: &RealQwenState,
    arch: &ArchConfig,
    input_norm: (&gpu::MetalBuffer, u64),
    post_attn_norm: (&gpu::MetalBuffer, u64),
    layer: usize,
    hidden: usize,
    inter: usize,
    num_experts: usize,
    use_silu: bool,
    start_position: usize,
    m: usize,
) -> Result<(), RealForwardError> {
    let batched = qwen.batched_prefill();
    encode_qwen_layer_attn_and_norms_batched(
        context,
        pass,
        weights,
        index,
        scratch,
        batched,
        qwen,
        kv,
        arch,
        input_norm,
        post_attn_norm,
        layer,
        hidden,
        start_position,
        m,
        // No tape: a prefill is never rolled back.
        None,
    )?;
    let router_name = layer_tensor(layer, "mlp.gate.weight");
    for t in 0..m {
        encode_router_gemv_batched(
            context,
            pass,
            weights,
            index,
            &router_name,
            qwen,
            batched,
            t,
            hidden,
            num_experts,
        )?;
    }
    encode_qwen_shared_expert_batched(
        context, pass, weights, index, batched, layer, hidden, inter, use_silu, m,
    )
}
