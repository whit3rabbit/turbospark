//! The MLA attention block: projections, in-place latent norm + rope inside
//! the cache row, absorb, MQA attention, v-combine, and the output
//! projection. `docs/DEEPSEEK2_PHASE0.md` carries the design; the kernels'
//! contracts are `compute::mla`.

use model_io::{ArchConfig, ResidentIndex};

use crate::families::deepseek2::{layer_tensor, RealDeepseek2State};
use crate::real_forward::RealForwardError;
use crate::real_forward_dispatch::encode_gemv_any;
use crate::real_forward_types::DecodeScratch;
use crate::real_forward_utils::norm_view;

/// The q projection's rows and the kv_b rows the flow reads, as the
/// resident names spell them.
pub(crate) const Q_PROJ: &str = "self_attn.q_proj.weight";
pub(crate) const KV_A_PROJ: &str = "self_attn.kv_a_proj.weight";
pub(crate) const KV_A_NORM: &str = "self_attn.kv_a_norm.weight";
pub(crate) const KV_B_PROJ: &str = "self_attn.kv_b_proj.weight";
pub(crate) const O_PROJ: &str = "self_attn.o_proj.weight";

/// One layer's MLA attention, decode shape (single token at `position`).
/// The caller has already run the input norm into `scratch.normed`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_mla_attention_block(
    context: &mut gpu::MetalContext,
    pass: &gpu::PassEncoder,
    weights: &gpu::ResidentGpuWeights,
    index: &ResidentIndex,
    arch: &ArchConfig,
    state: &RealDeepseek2State,
    scratch: &DecodeScratch,
    kv: &mut gpu::KvCacheManager,
    layer: usize,
    position: usize,
    hidden: usize,
) -> Result<(), RealForwardError> {
    let gpu_err = RealForwardError::Gpu;
    let (k_buf, k_off) = kv.k_slot(layer, position);

    // q: [heads * (nope + rope)]. The pe windows are roped in place BEFORE
    // the absorb kernel gathers them into the fused query rows.
    encode_gemv_any(
        context,
        pass,
        weights,
        index,
        &layer_tensor(layer, Q_PROJ),
        (state.heads * state.head_dim) as usize,
        hidden,
        (&scratch.normed, 0),
        (&state.q, 0),
    )?;
    gpu::encode_mla_rope_q_pe(
        context,
        pass,
        (&state.q, 0),
        (&state.rope_frequencies, 0),
        1,
        state.heads,
        state.head_dim,
        state.nope,
        state.rope_dim,
        position as u32,
        state.rope_mscale,
    )
    .map_err(gpu_err)?;

    // kv_a writes STRAIGHT INTO the cache row: the norm and the rope then
    // run in place there, and no staging copy exists. This is the
    // compressed row `[latent ; k_pe]` the whole cache is.
    encode_gemv_any(
        context,
        pass,
        weights,
        index,
        &layer_tensor(layer, KV_A_PROJ),
        state.cache_row as usize,
        hidden,
        (&scratch.normed, 0),
        (k_buf, k_off as u64),
    )?;
    let kv_a_norm = norm_view(
        weights,
        index,
        &layer_tensor(layer, KV_A_NORM),
        state.kv_lora as usize,
    )?;
    gpu::encode_mla_kv_norm(
        context,
        pass,
        (k_buf, k_off as u64),
        kv_a_norm,
        1,
        state.kv_lora,
        state.cache_row,
        state.rms_eps,
    )
    .map_err(gpu_err)?;
    // The pe tail of the cache row: one head of rope_dim, from byte offset
    // kv_lora * 2. `rope_neox_freqs` pairs `(i, rope_dim/2 + i)` within the
    // head, which is exactly the pe window's own pairing.
    gpu::encode_rope_neox_freqs(
        context,
        pass,
        (k_buf, (k_off + state.kv_lora as usize * 2) as u64),
        position as u32,
        1,
        state.rope_dim,
        state.rope_dim / 2,
        (&state.rope_frequencies, 0),
        state.rope_mscale,
    )
    .map_err(gpu_err)?;

    // Absorb: fused `[q' ; q_pe]` rows from the full q buffer.
    gpu::encode_mla_absorb_q(
        context,
        pass,
        weights_gpu_view(weights, index, layer, KV_B_PROJ)?,
        (&state.q, 0),
        (&state.q_abs, 0),
        state.heads,
        state.nope,
        state.kv_lora,
        state.v_dim,
        state.rope_dim,
    )
    .map_err(gpu_err)?;

    // MQA attention over the compressed rows. V is the row's first
    // kv_lora halves, read inside the kernel; scale is the manifest's
    // (mscale^2 / sqrt(key_dim), folded at repack).
    let (k_base_buf, _) = kv.k_slot(layer, 0);
    gpu::encode_mla_attention_decode(
        context,
        pass,
        (&state.q_abs, 0),
        (k_base_buf, 0),
        (&state.attn_c, 0),
        state.heads,
        state.cache_row,
        state.kv_lora,
        (position + 1) as u32,
        // Folded at repack: (1 + 0.1 * mscale * ln factor)^2 / sqrt(key_dim).
        // The NO_YARN debug switch pairs plain rope with the plain scale.
        if let Ok(v) = std::env::var("TURBOSPARK_DSV2_SCALE") {
            v.parse().unwrap()
        } else if std::env::var("TURBOSPARK_DSV2_NO_YARN").is_ok() {
            1.0 / (arch.mla.key_head_dim() as f32).sqrt()
        } else {
            arch.attention_scale as f32
        },
    )
    .map_err(gpu_err)?;

    // v-combine then the output projection into scratch.o; the caller adds
    // the residual.
    gpu::encode_mla_v_combine(
        context,
        pass,
        weights_gpu_view(weights, index, layer, KV_B_PROJ)?,
        (&state.attn_c, 0),
        (&state.mla_out, 0),
        state.heads,
        state.nope,
        state.kv_lora,
        state.v_dim,
    )
    .map_err(gpu_err)?;
    encode_gemv_any(
        context,
        pass,
        weights,
        index,
        &layer_tensor(layer, O_PROJ),
        hidden,
        (state.heads * state.v_dim) as usize,
        (&state.mla_out, 0),
        (&scratch.o, 0),
    )?;
    Ok(())
}

/// The resident (buffer, byte-offset) view of a whole tensor, for kernels
/// that read Q8_0 weights in place.
fn weights_gpu_view<'a>(
    weights: &'a gpu::ResidentGpuWeights,
    index: &ResidentIndex,
    layer: usize,
    suffix: &str,
) -> Result<(&'a gpu::MetalBuffer, u64), RealForwardError> {
    let name = layer_tensor(layer, suffix);
    let e = crate::real_forward_utils::entry(index, &name)?;
    let base = index.header.index_size;
    Ok((weights.buffer(), weights.gpu_offset(e.file_offset - base)))
}
