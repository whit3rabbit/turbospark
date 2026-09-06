//! Write-path helpers for a TurboQuant-quantized layer
//! (`model_io::KvQuant`), used by every family's attention block instead
//! of `kv.k_slot` / `kv.v_slot` / `gpu::encode_attention_decode` directly.
//!
//! **A quantized layer's cache row is never an in-place write target.**
//! Every family projects K/V straight into the cache slot and norms/RoPEs
//! it there; a quantized layer instead projects into a FP16 STAGING row
//! ([`crate::real_forward_types::KvStage`]), norms/RoPEs the staging row
//! in place exactly as an FP16 layer's slot would be, then
//! [`encode_kv_commit`] quantizes staging into the real slot.
//!
//! **Chunked prefill gets this for free where a family's driver reuses its
//! own per-token attention block** (`llama`, `gemma4`, and by the same
//! pattern `gpt-oss`/`muse_glimmer`/dense `qwen`'s drivers, all of which
//! keep attention itself per-token and unbatched -- see each family's own
//! `CLAUDE.md` entry): wiring the sequential function wires every caller
//! of it. **What is NOT wired is the M-ROW BATCHED GEMM path**
//! (`TURBOSPARK_BATCHED_GEMV`'s widened K/V projections in
//! `attn_batch.rs`/`batched_layers.rs`, and the MTP/DFlash2 verify's own
//! M-row forward), which writes K/V straight into the cache via a GEMM and
//! bypasses this module entirely -- each such site must refuse a
//! quantized layer by name until it is (`docs/TRUBOQUANT.md`'s follow-ups).

use crate::real_forward_types::{DecodeScratch, RealForwardError};

/// Which half of a K/V pair a call concerns.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum KvHalf {
    K,
    V,
}

/// The write target for one layer's K or V projection at `position`: the
/// cache slot itself on an FP16 layer (unchanged from before this
/// feature), the staging row on a TurboQuant-quantized one. A quantized
/// layer's own cache slot is reached only by [`encode_kv_commit`], never
/// by a projection GEMV directly -- see this module's doc.
pub(crate) fn kv_write_target<'a>(
    kv: &'a gpu::KvCacheManager,
    scratch: &'a DecodeScratch,
    half: KvHalf,
    layer: usize,
    position: usize,
) -> (&'a gpu::MetalBuffer, u64) {
    if kv.layer_quant(layer).is_some() {
        let stage = scratch
            .kv_stage
            .as_ref()
            .expect("kv_stage is allocated whenever any layer is TurboQuant-quantized");
        match half {
            KvHalf::K => (&stage.k, 0),
            KvHalf::V => (&stage.v, 0),
        }
    } else {
        let (buf, off) = match half {
            KvHalf::K => kv.k_slot(layer, position),
            KvHalf::V => kv.v_slot(layer, position),
        };
        (buf, off as u64)
    }
}

/// No-op on an FP16 layer. On a TurboQuant-quantized one, quantizes the
/// staging row (already normed and RoPE'd by the caller, exactly as an
/// FP16 layer's cache slot would have been) into the real cache slot at
/// `position`, for both K and V.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_kv_commit(
    context: &mut gpu::MetalContext,
    pass: &gpu::PassEncoder,
    kv: &gpu::KvCacheManager,
    scratch: &DecodeScratch,
    layer: usize,
    position: usize,
    head_dim: u32,
    num_kv_heads: u32,
) -> Result<(), RealForwardError> {
    if kv.layer_quant(layer).is_none() {
        return Ok(());
    }
    let tables = kv
        .quant_tables()
        .expect("quant_tables is Some whenever a layer's layer_quant is Some");
    let stage = scratch
        .kv_stage
        .as_ref()
        .expect("kv_stage is allocated whenever any layer is TurboQuant-quantized");
    let row_elems = num_kv_heads * head_dim;

    let (k_buf, k_off) = kv.k_slot(layer, position);
    gpu::encode_kv_quantize_tq(
        context,
        pass,
        (&stage.k, 0),
        row_elems,
        (k_buf, k_off as u64),
        &tables.k,
        head_dim,
        num_kv_heads,
        1,
    )
    .map_err(RealForwardError::Gpu)?;

    let (v_buf, v_off) = kv.v_slot(layer, position);
    gpu::encode_kv_quantize_tq(
        context,
        pass,
        (&stage.v, 0),
        row_elems,
        (v_buf, v_off as u64),
        &tables.v,
        head_dim,
        num_kv_heads,
        1,
    )
    .map_err(RealForwardError::Gpu)?;
    Ok(())
}

/// Dispatches decode attention over `layer`'s cache, forking on whether
/// that layer is TurboQuant-quantized. `position` locates the layer's
/// underlying K/V buffer (the returned offset is unused here -- attention
/// always reads from row 0 of the buffer under linear addressing).
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_attention_any(
    context: &mut gpu::MetalContext,
    pass: &gpu::PassEncoder,
    q: (&gpu::MetalBuffer, u64),
    kv: &gpu::KvCacheManager,
    scratch: &DecodeScratch,
    out: (&gpu::MetalBuffer, u64),
    layer: usize,
    position: usize,
    head_dim: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
    seq_len: u32,
    kv_start: u32,
    ring_capacity: u32,
    scale: f32,
    sinks: Option<(&gpu::MetalBuffer, u64)>,
) -> Result<(), RealForwardError> {
    let (k_buf, _) = kv.k_slot(layer, position);
    let (v_buf, _) = kv.v_slot(layer, position);
    if kv.layer_quant(layer).is_some() {
        assert_eq!(
            ring_capacity, 0,
            "a TurboQuant-quantized layer is never a sliding-window ring"
        );
        let tables = kv
            .quant_tables()
            .expect("quant_tables is Some whenever a layer's layer_quant is Some");
        let tq_scratch = scratch
            .tq_attn
            .as_ref()
            .expect("tq_attn is allocated whenever any layer is TurboQuant-quantized");
        gpu::encode_attention_decode_tq(
            context,
            pass,
            q,
            k_buf,
            v_buf,
            tq_scratch,
            out,
            head_dim,
            num_q_heads,
            num_kv_heads,
            seq_len,
            kv_start,
            scale,
            tables,
            sinks,
        )
        .map_err(RealForwardError::Gpu)
    } else {
        gpu::encode_attention_decode(
            context,
            pass,
            q,
            k_buf,
            v_buf,
            &scratch.attn,
            out,
            head_dim,
            num_q_heads,
            num_kv_heads,
            seq_len,
            kv_start,
            ring_capacity,
            scale,
            sinks,
        )
        .map_err(RealForwardError::Gpu)
    }
}
