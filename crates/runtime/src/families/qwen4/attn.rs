//! The two sublayer branches a `qwen4_exp` layer's `attn_hc` output feeds:
//! gated DeltaNet (mask-2 layers, 36 of 48) or QSA-as-dense-attention
//! (mask-1, 12 of 48). Both read `mixed` (`hidden`-wide) and write
//! `scratch.o` (`hidden`-wide), matching `families/qwen/attn.rs`'s shape
//! closely enough that this is a copy-and-adapt rather than a new design --
//! see `mod.rs`'s "## GDN" / "## QSA-as-dense-attention" sections for what
//! differs and why neither is shared code with that file.

use std::time::Instant;

use model_io::{ArchConfig, ResidentIndex};

use crate::families::qwen4::state::RealQwen4State;
use crate::families::qwen4::{layer_tensor, RMS_EPS};
use crate::kv_write::{encode_attention_any, encode_kv_commit, kv_write_target, KvHalf};
use crate::real_forward_dispatch::encode_gemv_any;
use crate::real_forward_types::{DecodeScratch, PhaseCounters, RealForwardError};
use crate::real_forward_utils::{entry, norm_view, resident_matrix};

/// Mask-2 layer: gated DeltaNet. Identical dataflow to
/// `families/qwen/attn.rs::encode_linear_block`, differing in the state
/// type and in the gated output norm's SIGMOID variant
/// (`docs/QWEN4_PHASE0.md` item 0 finding 4, `linear_attention.output_gate_sigmoid`).
/// `mixed` is the `attn_hc` output (`hidden`-wide); this family's own
/// causal conv is UNDILATED (`dilation = 1`), unlike PLE's separate
/// dilated one -- the two are different tensors (`linear_attn.conv1d` vs
/// `ple.conv1d`) and different call sites.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_linear_block(
    context: &mut gpu::MetalContext,
    pass: &gpu::PassEncoder,
    weights: &gpu::ResidentGpuWeights,
    index: &ResidentIndex,
    arch: &ArchConfig,
    qwen4: &RealQwen4State,
    mixed: (&gpu::MetalBuffer, u64),
    scratch: &DecodeScratch,
    layer: usize,
) -> Result<(), RealForwardError> {
    let gpu_err = RealForwardError::Gpu;
    let hidden = arch.hidden_size as usize;
    let shape = qwen4.shape;
    let qkv_dim = shape.qkv_dim() as usize;
    let value_dim = shape.value_dim() as usize;
    let v_heads = shape.num_v_heads as usize;

    let name = |suffix: &str| layer_tensor(layer, &format!("linear_attn.{suffix}"));
    let in_proj = [
        ("in_proj_qkv.weight", qkv_dim, &qwen4.gdn_qkv_raw),
        ("in_proj_z.weight", value_dim, &qwen4.gdn_z),
        ("in_proj_a.weight", v_heads, &qwen4.gdn_a),
        ("in_proj_b.weight", v_heads, &qwen4.gdn_b),
    ];
    if entry(index, &name(in_proj[0].0))?.dtype == 4 {
        let projection = |suffix: &str, rows: usize| {
            resident_matrix(weights, index, &name(suffix), rows, hidden)
        };
        let (qkv_w, z_w, a_w, b_w) = (
            projection(in_proj[0].0, qkv_dim)?,
            projection(in_proj[1].0, value_dim)?,
            projection(in_proj[2].0, v_heads)?,
            projection(in_proj[3].0, v_heads)?,
        );
        gpu::encode_gdn_in_proj(
            context,
            pass,
            &qkv_w,
            &z_w,
            &a_w,
            &b_w,
            mixed,
            (&qwen4.gdn_qkv_raw, 0),
            (&qwen4.gdn_z, 0),
            (&qwen4.gdn_a, 0),
            (&qwen4.gdn_b, 0),
        )
        .map_err(gpu_err)?;
    } else {
        for (suffix, rows, out) in in_proj {
            encode_gemv_any(
                context,
                pass,
                weights,
                index,
                &name(suffix),
                rows,
                hidden,
                mixed,
                (out, 0),
            )?;
        }
    }

    let conv_w = norm_view(
        weights,
        index,
        &name("conv1d.weight"),
        qkv_dim * shape.conv_kernel_size as usize,
    )?;
    gpu::encode_gdn_conv_decode(
        context,
        pass,
        shape,
        (qwen4.gdn.conv_tail_buffer(layer), 0),
        (&qwen4.gdn_qkv_raw, 0),
        conv_w,
        (&qwen4.gdn_conv_out, 0),
        1, // this chain's own conv is undilated; PLE's is the dilated one.
    )
    .map_err(gpu_err)?;
    gpu::encode_gdn_qk_norm(context, pass, shape, (&qwen4.gdn_conv_out, 0), 1).map_err(gpu_err)?;

    let a_log = norm_view(weights, index, &name("A_log"), v_heads)?;
    let dt_bias = norm_view(weights, index, &name("dt_bias"), v_heads)?;
    gpu::encode_gdn_delta_decode(
        context,
        pass,
        shape,
        (&qwen4.gdn_conv_out, 0),
        (&qwen4.gdn_a, 0),
        (&qwen4.gdn_b, 0),
        a_log,
        dt_bias,
        qwen4.gdn.state_buffer(layer),
        (&qwen4.gdn_y, 0),
    )
    .map_err(gpu_err)?;

    let gated_norm = norm_view(
        weights,
        index,
        &name("norm.weight"),
        shape.value_head_dim as usize,
    )?;
    // SIGMOID variant: `qwen4_exp`'s `output_gate_sigmoid`, unlike
    // `qwen3_5`/`qwen3_6`'s silu (`crates/gpu/CLAUDE.md` Gotcha 12).
    gpu::encode_gdn_gated_norm_sigmoid(
        context,
        pass,
        shape,
        (&qwen4.gdn_y, 0),
        (&qwen4.gdn_z, 0),
        gated_norm,
        (&qwen4.gdn_out, 0),
        1,
    )
    .map_err(gpu_err)?;
    encode_gemv_any(
        context,
        pass,
        weights,
        index,
        &name("out_proj.weight"),
        hidden,
        value_dim,
        (&qwen4.gdn_out, 0),
        (&scratch.o, 0),
    )
}

/// Mask-1 layer: QSA (query-sparse attention), `docs/QWEN4_PHASE0.md`
/// section 5. Packed `[query; gate]` in `q_proj`, per-head `q_norm`/`k_norm`
/// CENTERED (unconditionally -- unlike `families/qwen/attn.rs`, this family
/// has no plain-norm sibling tensor sharing this call, so there is no
/// `QkNormConvention` parameter), `rope_neox_subdim` at `rotary_dim = 64`,
/// text-only (`RopePosition` is always sequential; no mRoPE, since vision
/// is out of scope for this cut).
///
/// **THE INDEXER RUNS EVERY TOKEN; SELECTION RUNS ONLY ABOVE BUDGET.**
/// Every token projects `index_qk_proj`, stores its raw key in the
/// indexer cache and pools/norms/ropes whichever blocks just completed, so
/// the pooled-block cache is current the moment the budget is crossed. At
/// or below `index_top_k` complete blocks (`visible <= 2051` on the real
/// checkpoint) nothing is scored and the trunk's dispatch stream is what it
/// was before the indexer existed: dense causal attention, byte for byte,
/// which is what keeps every frozen digest where it is. Above it: the
/// indexer query is normed and roped, `qsa_score_blocks_fp16` scores every
/// complete block, the pass is COMMITTED AND WAITED ON mid-layer for the
/// score readback (the MoE router's own precedent, one layer down), the
/// host runs `select_blocks`, and `attention_decode_indexed_partial`
/// attends over the selected positions in place of the dense kernel.
///
/// `pass` is `&mut` for that mid-layer commit: the caller's encoder is
/// swapped for a fresh one and keeps encoding into it afterwards.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_full_attention_block(
    context: &mut gpu::MetalContext,
    pass: &mut gpu::PassEncoder,
    weights: &gpu::ResidentGpuWeights,
    index: &ResidentIndex,
    arch: &ArchConfig,
    qwen4: &mut RealQwen4State,
    mixed: (&gpu::MetalBuffer, u64),
    scratch: &DecodeScratch,
    kv: &gpu::KvCacheManager,
    phases: &mut PhaseCounters,
    layer: usize,
    position: usize,
) -> Result<(), RealForwardError> {
    let gpu_err = RealForwardError::Gpu;
    let hidden = arch.hidden_size as usize;
    let num_heads = arch.num_heads as u32;
    let num_kv = arch.num_full_kv_heads as u32;
    let head_dim = arch.full_head_dim as u32;
    let q_dim = (num_heads * head_dim) as usize;
    let kv_dim = (num_kv * head_dim) as usize;
    let theta = arch.full_rope_theta as f32;
    let name = |suffix: &str| layer_tensor(layer, &format!("self_attn.{suffix}"));

    // --- The indexer, every token: raw key into the cache, new blocks pooled.
    let idx_heads = qwen4.idx_heads;
    let idx_dim = qwen4.idx_head_dim;
    let compress = qwen4.idx_compress;
    let visible = position + 1;
    let complete_blocks = visible / compress as usize;
    encode_gemv_any(
        context,
        pass,
        weights,
        index,
        &name("indexer.index_qk_proj.weight"),
        ((idx_heads + 1) * idx_dim) as usize,
        hidden,
        mixed,
        (&qwen4.idx_qk, 0),
    )?;
    // The raw (un-normed, un-roped) key is the projection's LAST head. A
    // plain strided row copy; the kernel's name says what first needed it.
    let (raw_buf, raw_off) = qwen4.qsa.raw_key_slot(layer, position);
    gpu::encode_dflash_copy_rows(
        context,
        pass,
        (&qwen4.idx_qk, (idx_heads * idx_dim) as u64 * 2),
        (raw_buf, raw_off as u64),
        1,
        idx_dim,
        idx_dim,
    )
    .map_err(gpu_err)?;
    let pooled_count = qwen4.qsa.pooled_block_count(layer);
    if complete_blocks > pooled_count {
        let k_norm = norm_view(
            weights,
            index,
            &name("indexer.k_layernorm.weight"),
            idx_dim as usize,
        )?;
        gpu::encode_qsa_advance_blocks(
            context,
            pass,
            (qwen4.qsa.raw_keys_view(layer), 0),
            (qwen4.qsa.pooled_blocks_buffer(layer), 0),
            k_norm,
            compress,
            idx_dim,
            qwen4.rotary_dim,
            theta,
            RMS_EPS,
            pooled_count as u32,
            (complete_blocks - pooled_count) as u32,
            0,
        )
        .map_err(gpu_err)?;
        // Recorded at encode time: the dispatch precedes every reader of
        // these rows in this pass, and a `?` that drops the pass uncommitted
        // fails the whole token, after which `reset()` rewinds this cursor.
        qwen4.qsa.advance_pooled_blocks(layer, complete_blocks);
    }

    // --- Attention proper: projections, norms, RoPE, KV write.
    let (k_buf, k_off) = kv_write_target(kv, scratch, KvHalf::K, layer, position);
    let (v_buf, v_off) = kv_write_target(kv, scratch, KvHalf::V, layer, position);
    encode_gemv_any(
        context,
        pass,
        weights,
        index,
        &name("q_proj.weight"),
        2 * q_dim,
        hidden,
        mixed,
        (&qwen4.q_packed, 0),
    )?;
    gpu::encode_split_q_gate(
        context,
        pass,
        (&qwen4.q_packed, 0),
        (&scratch.q, 0),
        (&qwen4.attn_gate, 0),
        num_heads,
        head_dim,
    )
    .map_err(gpu_err)?;
    for (suffix, out) in [
        ("k_proj.weight", (k_buf, k_off)),
        ("v_proj.weight", (v_buf, v_off)),
    ] {
        encode_gemv_any(
            context,
            pass,
            weights,
            index,
            &name(suffix),
            kv_dim,
            hidden,
            mixed,
            out,
        )?;
    }

    for (suffix, data, heads) in [
        ("q_norm.weight", (&scratch.q, 0u64), num_heads),
        ("k_norm.weight", (k_buf, k_off), num_kv),
    ] {
        let weight = norm_view(weights, index, &name(suffix), head_dim as usize)?;
        gpu::encode_rms_norm_bf16w_perhead_centered(
            context, pass, data, weight, data, heads, head_dim, RMS_EPS,
        )
        .map_err(gpu_err)?;
    }

    for (data, heads) in [((&scratch.q, 0u64), num_heads), ((k_buf, k_off), num_kv)] {
        gpu::encode_rope_neox_subdim(
            context,
            pass,
            data,
            position as u32,
            heads,
            head_dim,
            qwen4.rotary_dim,
            theta,
        )
        .map_err(gpu_err)?;
    }

    // On a TurboQuant-quantized layer, quantize the staging row into the
    // real cache slot now that it is normed and RoPE'd -- a no-op on an
    // FP16 layer (`crate::kv_write`'s doc). Placed before block selection
    // since both the dense and the indexed attention kernels below read
    // from the committed cache slot, never from the staging row.
    encode_kv_commit(
        context, pass, kv, scratch, layer, position, head_dim, num_kv,
    )?;

    // --- Block selection, above budget only.
    let sparse = complete_blocks > qwen4.idx_block_topk && !qwen4.qsa_force_dense;
    if sparse {
        // The indexer query: per-head centered norm, then the trunk's own
        // RoPE at the current position (one shared object in the reference,
        // `crates/compute/src/qsa_indexer.rs`'s module doc).
        let q_norm = norm_view(
            weights,
            index,
            &name("indexer.q_layernorm.weight"),
            idx_dim as usize,
        )?;
        gpu::encode_rms_norm_bf16w_perhead_centered(
            context,
            pass,
            (&qwen4.idx_qk, 0),
            q_norm,
            (&qwen4.idx_qk, 0),
            idx_heads,
            idx_dim,
            RMS_EPS,
        )
        .map_err(gpu_err)?;
        gpu::encode_rope_neox_subdim(
            context,
            pass,
            (&qwen4.idx_qk, 0),
            position as u32,
            idx_heads,
            idx_dim,
            qwen4.rotary_dim,
            theta,
        )
        .map_err(gpu_err)?;
        gpu::encode_qsa_score_blocks(
            context,
            pass,
            (&qwen4.idx_qk, 0),
            (qwen4.qsa.pooled_blocks_buffer(layer), 0),
            (&qwen4.qsa_scores, 0),
            idx_heads,
            idx_dim,
            complete_blocks as u32,
        )
        .map_err(gpu_err)?;

        // Top-k is a host round trip, as it is for the MoE router: commit
        // what has been encoded so far and wait for the scores. The fresh
        // encoder keeps the caller's label so the phase report reads the
        // same; its GPU time lands in the same `cb1` bucket.
        let done = std::mem::replace(pass, context.begin_pass_labeled("cb1 (attn+router)"));
        let t_wait = Instant::now();
        phases.cb1_gpu_nanos += (done.commit().wait_with_gpu_time() * 1e9) as u64;
        phases.gpu_wait_nanos += t_wait.elapsed().as_nanos() as u64;

        let scores = gpu::read_f32_buffer(&qwen4.qsa_scores, complete_blocks);
        // AGENTS.md Gotcha 59: a NaN score would read as a top-ranked block
        // in any sort; refuse it by name before `select_blocks` sees it.
        if let Some(bad) = scores.iter().position(|s| !s.is_finite()) {
            return Err(RealForwardError::Unsupported(format!(
                "layer {layer} position {position}: QSA indexer score for block {bad} is not \
                 finite ({}); the sparse path cannot select over it",
                scores[bad]
            )));
        }
        let mask =
            compute::select_blocks(&scores, visible, compress as usize, qwen4.idx_block_topk);
        let positions: Vec<u32> = mask
            .iter()
            .enumerate()
            .filter(|(_, &selected)| selected)
            .map(|(p, _)| p as u32)
            .collect();
        debug_assert!(positions.iter().all(|&p| (p as usize) < visible));
        let capacity = (qwen4.qsa_positions.length() / 4) as usize;
        assert!(
            !positions.is_empty() && positions.len() <= capacity,
            "QSA selected {} positions against a {capacity}-entry buffer",
            positions.len()
        );
        // ONE position buffer for all QSA layers, written from the host
        // while the previous layer's indexed attention may still be
        // encoded: safe only because `produce.rs` commits and WAITS on
        // every layer's pass at its router readback before the next layer
        // encodes anything, so the dispatch that read the old list has
        // completed by the time this overwrites it. A driver that stops
        // waiting per layer (chunked prefill, a pipelined router) needs a
        // buffer per QSA layer instead.
        let bytes: Vec<u8> = positions.iter().flat_map(|p| p.to_le_bytes()).collect();
        gpu::write_buffer_bytes(&qwen4.qsa_positions, 0, &bytes);

        // TurboQuant fork, matching `crate::kv_write::encode_attention_any`'s
        // dense-path fork -- kept separate here because the QSA sparse
        // kernel is a fourth shape (`crate::kv_write`'s doc only covers the
        // two dense variants) rather than a case that helper can express.
        match kv.layer_quant(layer) {
            Some(_) => {
                let tables = kv
                    .quant_tables()
                    .expect("quant_tables is Some whenever a layer's layer_quant is Some");
                let tq_scratch = scratch
                    .tq_attn
                    .as_ref()
                    .expect("tq_attn is allocated whenever any layer is TurboQuant-quantized");
                gpu::encode_attention_decode_indexed_tq(
                    context,
                    pass,
                    (&scratch.q, 0),
                    k_buf,
                    v_buf,
                    (&qwen4.qsa_positions, 0),
                    positions.len() as u32,
                    tq_scratch,
                    (&scratch.attn_out, 0),
                    head_dim,
                    num_heads,
                    num_kv,
                    arch.attention_scale as f32,
                    tables,
                )
                .map_err(gpu_err)?;
            }
            None => {
                gpu::encode_attention_decode_indexed(
                    context,
                    pass,
                    (&scratch.q, 0),
                    k_buf,
                    v_buf,
                    (&qwen4.qsa_positions, 0),
                    positions.len() as u32,
                    &scratch.attn,
                    (&scratch.attn_out, 0),
                    head_dim,
                    num_heads,
                    num_kv,
                    arch.attention_scale as f32,
                )
                .map_err(gpu_err)?;
            }
        }
    } else {
        encode_attention_any(
            context,
            pass,
            (&scratch.q, 0),
            kv,
            scratch,
            (&scratch.attn_out, 0),
            layer,
            position,
            head_dim,
            num_heads,
            num_kv,
            visible as u32,
            0,
            0,
            arch.attention_scale as f32,
            None,
        )?;
    }
    gpu::encode_sigmoid_gate_mul(
        context,
        pass,
        (&scratch.attn_out, 0),
        (&qwen4.attn_gate, 0),
        q_dim as u32,
    )
    .map_err(gpu_err)?;
    encode_gemv_any(
        context,
        pass,
        weights,
        index,
        &name("o_proj.weight"),
        hidden,
        q_dim,
        (&scratch.attn_out, 0),
        (&scratch.o, 0),
    )
}
