//! Inkling 2 decode state: per-cache-slot KV and conv tails, activation
//! buffers, the routed-expert offsets and the routed-blob argument buffer.
//!
//! The shared `KvCacheManager` is sized per PHYSICAL layer (24), and Inkling
//! runs 48 layer applications per token, each with its own KV and conv state
//! (llama.cpp `inkling2-loop.cpp`: "each cache slot keeps its own KV and conv
//! state"). So this state owns its 48 slots and uses the shared manager only
//! for the position. The manager still allocates its own 24 layers of KV,
//! which is unused memory (`docs/INKLING2_PHASE0.md`, Stage B).

use model_io::inkling2_config::Inkling2Config;
use model_io::{ArchConfig, ModelFamily, ResidentIndex};

use crate::real_forward::RealForwardError;
use crate::real_forward_utils::entry;

/// Resident dtype tags this flow reads (`repack::resident_writer`).
pub(crate) const TAG_BF16: u8 = 1;
pub(crate) const TAG_F32: u8 = 3;
/// MLX affine 4-bit and 8-bit at group 64 (`resident_writer` tags 4 and 5).
pub(crate) const TAG_INT4_AFFINE: u8 = 4;
pub(crate) const TAG_INT8_AFFINE: u8 = 5;

/// The GGUF library function the routed argument buffer reflects (every GGUF
/// pair binds the same `RoutedBlobs`).
pub(crate) const GGUF_REFLECT_FUNCTION: &str = "moe_phase1_gate_up_act_q8_0";

/// The most block states a depth mix sees (blocks plus the current state).
pub(crate) const MAX_STATES: usize = gpu::INKLING2_DEPTH_MAX_STATES;

/// One physical layer's routed-expert format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ExpertKind {
    /// The checkpoint's own NVFP4 / MXFP8 rows (`gpu::moe_inkling2`).
    Native(gpu::Inkling2ExpertType),
    /// GGUF blocks, gate-up and down each their own type, run by the GGUF
    /// pairs' Inkling mode (`gpu::encode_inkling2_gguf_phase{1,2}`).
    Gguf {
        gate_up: gpu::Inkling2GgufExpertType,
        down: gpu::Inkling2GgufExpertType,
    },
}

pub(crate) fn name(layer: usize, suffix: &str) -> String {
    format!("model.llm.layers.{layer}.{suffix}")
}

/// One cache slot's KV and conv state.
pub(crate) struct SlotState {
    /// K/V rows: position `p` lives at row `p % rows`. A sliding-window slot
    /// keeps a window-sized ring; a full slot keeps every position.
    pub(crate) rows: usize,
    pub(crate) k: gpu::MetalBuffer,
    pub(crate) v: gpu::MetalBuffer,
    pub(crate) tail_k: gpu::MetalBuffer,
    pub(crate) tail_v: gpu::MetalBuffer,
    pub(crate) tail_o: gpu::MetalBuffer,
    pub(crate) tail_f: gpu::MetalBuffer,
}

pub(crate) struct RealInkling2State {
    pub(crate) cfg: Inkling2Config,
    pub(crate) max_ctx: usize,
    pub(crate) slots: Vec<SlotState>,
    /// Per physical layer: expert sub-tensor offsets, with `gate_s` / `up_s`
    /// / `down_s` pointing into the blob's outer-scale record.
    pub(crate) offsets: Vec<gpu::MoeExpertOffsets>,
    pub(crate) kinds: Vec<ExpertKind>,
    pub(crate) routed: gpu::RoutedBlobsBuffer,
    /// Set when the open resolved `ResolvedExpertResidency::Resident`: every
    /// expert of every layer resident, so routing and the MoE run on the GPU
    /// and a token is one command buffer (`mod.rs`).
    pub(crate) resident: Option<ResidentExperts>,
    /// A LookAtMyQuant install's coded linears, embedding, head and experts
    /// (`lamq.rs`). Shared so the flow's GEMV helper can hold it beside the
    /// mutable state.
    pub(crate) lamq: Option<std::sync::Arc<super::lamq::LamqRuntime>>,
    /// `[DSPARK_TAP_ROWS, hidden]` f16, the last token's DSpark context taps
    /// (`TURBOSPARK_INKLING2_TAPS=1`; `RealForwardRunner::inkling2_taps`).
    pub(crate) taps: Option<gpu::MetalBuffer>,
    /// The speculative verify's per-row scratch, allocated by the first
    /// verify (`verify.rs`), so plain decode allocates none of it.
    pub(crate) batch: Option<super::verify::BatchScratch>,
    /// What the last verify fed, for its rollback; cleared by every
    /// single-token pass.
    pub(crate) verify: Option<super::verify::InklingVerifyRecord>,
    /// The DSpark block drafter, open only when a caller asked for it
    /// (`DflashDraftPolicy::Fixed` at open, `dspark.rs`).
    pub(crate) dspark: Option<super::dspark::DsparkState>,
    /// Per physical layer, host copies of the F32 router bias.
    pub(crate) router_bias: Vec<Vec<f32>>,
    /// Per core layer, the four message queries [Q, K, V, FFN] as F32 rows.
    pub(crate) msg_queries: Vec<gpu::MetalBuffer>,
    /// Rows of `bank` in use on the current pass (previous blocks + output).
    pub(crate) bank_states: usize,
    // Activations (FP16 unless noted).
    pub(crate) cur: gpu::MetalBuffer,
    pub(crate) attn_out: gpu::MetalBuffer,
    pub(crate) prelude: gpu::MetalBuffer,
    pub(crate) states: gpu::MetalBuffer,
    pub(crate) bank: gpu::MetalBuffer,
    pub(crate) mixed: gpu::MetalBuffer,
    pub(crate) msgs: gpu::MetalBuffer,
    pub(crate) inputs: gpu::MetalBuffer,
    pub(crate) normed: gpu::MetalBuffer,
    pub(crate) cat: gpu::MetalBuffer,
    pub(crate) gate_r: gpu::MetalBuffer,
    pub(crate) gain: gpu::MetalBuffer,
    pub(crate) forget: gpu::MetalBuffer,
    pub(crate) q: gpu::MetalBuffer,
    pub(crate) k_pre: gpu::MetalBuffer,
    pub(crate) v_pre: gpu::MetalBuffer,
    pub(crate) r: gpu::MetalBuffer,
    pub(crate) sinks: gpu::MetalBuffer,
    pub(crate) gate_attn: gpu::MetalBuffer,
    pub(crate) dk: gpu::MetalBuffer,
    pub(crate) dv: gpu::MetalBuffer,
    /// F32 `[heads, extent]`.
    pub(crate) table: gpu::MetalBuffer,
    pub(crate) attn_o: gpu::MetalBuffer,
    pub(crate) o2: gpu::MetalBuffer,
    pub(crate) m: gpu::MetalBuffer,
    pub(crate) mn: gpu::MetalBuffer,
    /// F32 router logits.
    pub(crate) router_logits: gpu::MetalBuffer,
    pub(crate) acts: gpu::MetalBuffer,
    pub(crate) routing_w: gpu::MetalBuffer,
    pub(crate) y: gpu::MetalBuffer,
    pub(crate) y2: gpu::MetalBuffer,
}

fn f32_host(
    weights: &gpu::ResidentGpuWeights,
    index: &ResidentIndex,
    name: &str,
    n: usize,
) -> Result<Vec<f32>, RealForwardError> {
    let e = entry(index, name)?;
    if e.dtype != TAG_F32 || e.size_bytes as usize != n * 4 {
        return Err(RealForwardError::Unsupported(format!(
            "{name}: expected F32 with {n} elements, found dtype {} and {} bytes",
            e.dtype, e.size_bytes
        )));
    }
    let local = (e.file_offset - index.header.index_size) as usize;
    Ok(weights.data()[local..local + n * 4]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

/// Checks a resident tensor's dtype and byte size at open, so a wrong install
/// fails here instead of decoding garbage.
pub(crate) fn require(
    index: &ResidentIndex,
    name: &str,
    dtype: u8,
    elems: usize,
) -> Result<(), RealForwardError> {
    let e = entry(index, name)?;
    let width = if dtype == TAG_F32 { 4 } else { 2 };
    if e.dtype != dtype || e.size_bytes as usize != elems * width {
        return Err(RealForwardError::Unsupported(format!(
            "{name}: expected dtype {dtype} with {elems} elements, found dtype {} and {} bytes",
            e.dtype, e.size_bytes
        )));
    }
    Ok(())
}

/// A dense matrix the flow runs a GEMV over (or looks rows up in): raw BF16,
/// MLX affine 4/8-bit at group 64 (an MLX-sourced install), or a LAMQ coded
/// matrix (tag 36, whose aux and table `lamq.rs` checks at build), checked
/// against its logical `rows x cols`.
pub(crate) fn require_matrix(
    index: &ResidentIndex,
    name: &str,
    rows: usize,
    cols: usize,
) -> Result<(), RealForwardError> {
    let e = entry(index, name)?;
    match e.dtype {
        TAG_BF16 => require(index, name, TAG_BF16, rows * cols),
        TAG_INT4_AFFINE | TAG_INT8_AFFINE
            if (e.shape.0 as usize, e.shape.1 as usize) == (rows, cols) && cols % 64 == 0 =>
        {
            Ok(())
        }
        super::lamq::TAG_LAMQ if (e.shape.0 as usize, e.shape.1 as usize) == (rows, cols) => Ok(()),
        // A GGUF install's block-quantized matrices, run by `encode_gemv_any`
        // and `encode_embed_any` (each refuses a type it has no kernel for).
        tag if crate::real_forward_layout::EXECUTABLE_GGUF_DTYPES.contains(&tag)
            && (e.shape.0 as usize, e.shape.1 as usize) == (rows, cols) =>
        {
            Ok(())
        }
        other => Err(RealForwardError::Unsupported(format!(
            "{name}: expected a BF16 or affine 4/8-bit [{rows}, {cols}] matrix, found dtype \
             {other} shaped {:?}",
            (e.shape.0, e.shape.1)
        ))),
    }
}

impl RealInkling2State {
    pub(crate) fn build(
        context: &mut gpu::MetalContext,
        weights: &gpu::ResidentGpuWeights,
        index: &ResidentIndex,
        arch: &ArchConfig,
        install_dir: &std::path::Path,
        max_ctx: usize,
        resident_experts: bool,
    ) -> Result<Self, RealForwardError> {
        let unsupported = RealForwardError::Unsupported;
        if arch.family != ModelFamily::Inkling2Loop {
            return Err(unsupported(format!(
                "{:?} is not inkling2_loop",
                arch.family
            )));
        }
        let cfg =
            Inkling2Config::load(install_dir, arch).map_err(|e| unsupported(e.to_string()))?;
        let (h, heads, kvh, hd) = (
            arch.hidden_size as usize,
            arch.num_heads as usize,
            arch.num_kv_heads as usize,
            arch.head_dim as usize,
        );
        let (layers, experts, top_k, ffn) = (
            arch.num_layers as usize,
            arch.num_experts as usize,
            arch.top_k_experts as usize,
            arch.moe_intermediate_size as usize,
        );
        let taps = arch.short_conv_kernel_size as usize;
        let (d_rel, extent, rank) = (
            cfg.d_rel as usize,
            cfg.rel_extent as usize,
            cfg.gate_rank as usize,
        );
        let kvd = kvh * hd;
        if hd > gpu::INKLING2_ATTN_MAX_HEAD_DIM as usize || top_k > gpu::INKLING2_MAX_SLOTS {
            return Err(unsupported(format!(
                "head_dim {hd} or top_k {top_k} is past the Inkling kernels' limits"
            )));
        }
        if max_ctx == 0 {
            return Err(unsupported("zero context capacity".into()));
        }

        // A LAMQ install's coded entries, parsed and checked once; its
        // embedding and head are bucket orders, checked there.
        let mut lamq = super::lamq::LamqRuntime::build(
            context,
            weights,
            index,
            h,
            ffn,
            arch.vocab_size as usize,
            cfg.unpadded_vocab as usize,
        )?;
        // Every tensor the flow reads, checked once.
        if lamq.is_none() {
            require_matrix(index, "model.llm.embed.weight", arch.vocab_size as usize, h)?;
            require_matrix(
                index,
                "model.llm.unembed.weight",
                arch.vocab_size as usize,
                h,
            )?;
        }
        require(index, "model.llm.embed_norm.weight", TAG_BF16, h)?;
        require(index, "model.llm.norm.weight", TAG_BF16, h)?;
        require(index, "model.llm.attn_res.output_query", TAG_F32, h)?;
        let mut router_bias = Vec::with_capacity(layers);
        for (l, &mask) in arch
            .full_attention_layer_mask
            .iter()
            .enumerate()
            .take(layers)
        {
            let full = mask == 1;
            let a = |s: &str| name(l, &format!("attn.{s}"));
            for (s, rows, cols) in [
                ("wq_du.weight", heads * hd, h),
                ("wk_dv.weight", kvd, h),
                ("wv_dv.weight", kvd, h),
                ("wr_du.weight", heads * d_rel, h),
                ("wo_ud.weight", h, heads * hd),
                ("gate_attn.weight", heads * hd, h),
                ("attn_sink_proj.weight", heads, h),
            ] {
                require_matrix(index, &a(s), rows, cols)?;
            }
            for (s, n) in [
                ("q_l2_clip.scale", heads * hd),
                ("rel_l2_clip.scale", heads * d_rel),
                ("rel_logits_proj.proj", d_rel * extent),
                ("k_sconv.weight", kvd * taps),
                ("v_sconv.weight", kvd * taps),
            ] {
                require(index, &a(s), TAG_BF16, n)?;
            }
            if full {
                for s in ["k_sconv.weight_proj", "v_sconv.weight_proj"] {
                    require_matrix(index, &a(s), kvh * taps, h)?;
                }
                for s in ["q_input_norm", "k_input_norm", "v_input_norm"] {
                    require(index, &name(l, &format!("{s}.weight")), TAG_BF16, h)?;
                }
            } else {
                require(index, &name(l, "attn_norm.weight"), TAG_BF16, h)?;
            }
            for s in ["attn_sconv.weight", "mlp_sconv.weight"] {
                require(index, &name(l, s), TAG_BF16, h * taps)?;
            }
            require(index, &name(l, "mlp_norm.weight"), TAG_BF16, h)?;
            require(index, &name(l, "mlp.gate.weight"), TAG_F32, experts * h)?;
            router_bias.push(f32_host(
                weights,
                index,
                &name(l, "mlp.gate.bias"),
                experts,
            )?);
            if l > 0 {
                let q = format!("model.llm.attn_res.attn_queries.{}", l - 1);
                require(index, &q, TAG_F32, if full { 3 * h } else { h })?;
            }
            require(
                index,
                &format!("model.llm.attn_res.mlp_queries.{l}"),
                TAG_F32,
                h,
            )?;
            if (cfg.core_start as usize..cfg.core_end as usize).contains(&l) {
                for c in ["attn", "k", "v", "mlp"] {
                    let p = name(l, &format!("controllers.{c}"));
                    // GEMV weights (`Ctx::control`), so any matrix form the
                    // GEMV helper runs: BF16, or a GGUF install's blocks.
                    require_matrix(index, &format!("{p}.down_weight"), rank, 2 * h)?;
                    require_matrix(index, &format!("{p}.gain_weight"), h, rank)?;
                    require_matrix(index, &format!("{p}.forget_weight"), h, rank)?;
                }
            }
        }

        // Message queries per core layer: [q[2i], kv[2i], kv[2i+1], q[2i+1]].
        let core = (cfg.core_end - cfg.core_start) as usize;
        let qs = f32_host(
            weights,
            index,
            "model.llm.middle_cycle_cross_pass_queries",
            2 * core * h,
        )?;
        let kvs = f32_host(
            weights,
            index,
            "model.llm.middle_cycle_cross_pass_kv_queries",
            2 * core * h,
        )?;
        let row = |v: &[f32], i: usize| v[i * h..(i + 1) * h].to_vec();
        let msg_queries = (0..core)
            .map(|i| {
                let rows = [
                    row(&qs, 2 * i),
                    row(&kvs, 2 * i),
                    row(&kvs, 2 * i + 1),
                    row(&qs, 2 * i + 1),
                ]
                .concat();
                let bytes: Vec<u8> = rows.iter().flat_map(|x| x.to_le_bytes()).collect();
                context.new_buffer_with_data(&bytes)
            })
            .collect();

        // Expert offsets and schemes from the layout (one entry per layer).
        let layout = model_io::load_packed_experts_layout(install_dir, 1 << 26)
            .map_err(RealForwardError::Model)?;
        if layout.layers.len() != layers {
            return Err(unsupported(format!(
                "packed layout has {} layers, expected {layers}",
                layout.layers.len()
            )));
        }
        let mut offsets = Vec::with_capacity(layers);
        let mut kinds = Vec::with_capacity(layers);
        // LAMQ blobs open with a format record; their experts mix codes, so
        // `lamq.rs` checks every record and the LAMQ pair reads them.
        let lamq_experts = layout
            .layers
            .first()
            .and_then(|l| l.experts.first())
            .is_some_and(|e| e.sub_tensors.contains_key("lamq_fmt"));
        if lamq_experts {
            let rt = lamq.as_mut().ok_or_else(|| {
                unsupported("LAMQ expert blobs in an install without LAMQ resident entries".into())
            })?;
            let luts = super::lamq::read_luts(weights, index)?;
            rt.attach_experts(context, weights, index, install_dir, &layout, h, ffn, &luts)?;
        } else if lamq.is_some() {
            return Err(unsupported(
                "a LAMQ install without LAMQ expert blobs".into(),
            ));
        }
        for (l, ll) in layout.layers.iter().enumerate() {
            if lamq_experts {
                break;
            }
            let first = ll
                .experts
                .first()
                .ok_or_else(|| unsupported(format!("layer {l} has no experts")))?;
            let sub = |role: &str| {
                first
                    .sub_tensors
                    .get(role)
                    .ok_or_else(|| unsupported(format!("layer {l} expert blob has no {role}")))
            };
            let (outer, gate, up, down) = (sub("outer")?, sub("gate")?, sub("up")?, sub("down")?);
            let kind = match gate.dtype.as_str() {
                "inkling2_nvfp4" => ExpertKind::Native(gpu::Inkling2ExpertType::Nvfp4),
                "inkling2_mxfp8" => ExpertKind::Native(gpu::Inkling2ExpertType::Mxfp8),
                other => match (
                    gpu::Inkling2GgufExpertType::from_dtype(other),
                    gpu::Inkling2GgufExpertType::from_dtype(&down.dtype),
                ) {
                    (Some(gate_up), Some(down)) => ExpertKind::Gguf { gate_up, down },
                    _ => {
                        return Err(unsupported(format!(
                            "layer {l}: expert dtypes {other} / {} have no Inkling pair",
                            down.dtype
                        )))
                    }
                },
            };
            // Gate and up share a type; down has its own in a GGUF layer.
            let row_bytes = |role: &str, cols: usize| match kind {
                ExpertKind::Native(k) => k.row_bytes(cols),
                ExpertKind::Gguf { gate_up, down } => {
                    if role == "down" {
                        down.row_bytes(cols)
                    } else {
                        gate_up.row_bytes(cols)
                    }
                }
            };
            let tiles = |role: &str, cols: usize| match kind {
                ExpertKind::Native(k) => cols % k.block_elems() == 0,
                ExpertKind::Gguf { gate_up, down } => {
                    cols % if role == "down" {
                        down.block_elems()
                    } else {
                        gate_up.block_elems()
                    } == 0
                }
            };
            for (role, s, rows, cols) in [
                ("gate", gate, ffn, h),
                ("up", up, ffn, h),
                ("down", down, h, ffn),
            ] {
                let same_type = if role == "down" {
                    true
                } else {
                    s.dtype == gate.dtype
                };
                let native_type = !matches!(kind, ExpertKind::Native(_)) || s.dtype == gate.dtype;
                if !same_type
                    || !native_type
                    || !tiles(role, cols)
                    || s.size as usize != rows * row_bytes(role, cols)
                {
                    return Err(unsupported(format!(
                        "layer {l}: {role} blob size, dtype or block tiling mismatch"
                    )));
                }
            }
            if outer.size != 16 || outer.offset % 4 != 0 {
                return Err(unsupported(format!(
                    "layer {l}: malformed outer-scale record"
                )));
            }
            // Every expert of a layer must share the first one's sub-offsets,
            // since one uniform describes all the bound blobs.
            if ll.experts.iter().any(|e| {
                ["outer", "gate", "up", "down"].iter().any(|r| {
                    e.sub_tensors.get(*r).map(|s| s.offset)
                        != first.sub_tensors.get(*r).map(|s| s.offset)
                })
            }) {
                return Err(unsupported(format!(
                    "layer {l}: experts disagree on sub-tensor offsets"
                )));
            }
            // The NVFP4 kernel loads code bytes as 32-bit words.
            if kind == ExpertKind::Native(gpu::Inkling2ExpertType::Nvfp4)
                && [gate, up, down].iter().any(|s| s.offset % 4 != 0)
            {
                return Err(unsupported(format!(
                    "layer {l}: NVFP4 expert rows are not 4-byte aligned"
                )));
            }
            let o = outer.offset as u32;
            offsets.push(gpu::MoeExpertOffsets {
                gate_w: gate.offset as u32,
                gate_s: o,
                gate_b: 0,
                up_w: up.offset as u32,
                up_s: o + 4,
                up_b: 0,
                down_w: down.offset as u32,
                down_s: o + 8,
                down_b: 0,
            });
            kinds.push(kind);
        }
        let resident = if resident_experts {
            Some(ResidentExperts::load(
                context,
                install_dir,
                &layout,
                &router_bias,
                experts,
            )?)
        } else {
            None
        };
        let gguf_experts = kinds.iter().any(|k| matches!(k, ExpertKind::Gguf { .. }));
        if gguf_experts && kinds.iter().any(|k| matches!(k, ExpertKind::Native(_))) {
            return Err(unsupported(
                "an install mixing GGUF and native expert layers".into(),
            ));
        }
        let (source, function) = if lamq_experts {
            (
                gpu::moe_inkling2_lamq_source(),
                gpu::MOE_INKLING2_LAMQ_REFLECT_FUNCTION,
            )
        } else if gguf_experts {
            (gpu::moe_gguf_source(), GGUF_REFLECT_FUNCTION)
        } else {
            (
                gpu::moe_inkling2_source(),
                gpu::MOE_INKLING2_REFLECT_FUNCTION,
            )
        };
        let routed = gpu::RoutedBlobsBuffer::new_for(context, source, function, false)
            .map_err(RealForwardError::Gpu)?;

        let half = |context: &gpu::MetalContext, n: usize| {
            context.new_output_buffer((n.max(1) * 2) as u64)
        };
        let zeros = |context: &mut gpu::MetalContext, n: usize| {
            context.new_buffer_with_data(&vec![0u8; n * 2])
        };
        let n_slots = cfg.slots(layers as u32) as usize;
        let window = arch.sliding_window as usize;
        if window == 0 {
            return Err(unsupported("zero sliding window".into()));
        }
        let slots = (0..n_slots)
            .map(|slot| {
                let ip = compute::inkling2::layer_phys(
                    slot,
                    layers,
                    cfg.core_start as usize,
                    cfg.core_end as usize,
                );
                // A window slot keeps `window + VERIFY_SLACK` rows so the rows
                // of a speculative verify block never overwrite a position an
                // earlier row of the block still reads. Attention reads row
                // `p % rows`, so decode is the same with or without the slack.
                let rows = if arch.full_attention_layer_mask[ip] == 1 {
                    max_ctx
                } else {
                    max_ctx.min(window + super::verify::VERIFY_SLACK)
                };
                SlotState {
                    rows,
                    k: half(context, rows * kvd),
                    v: half(context, rows * kvd),
                    tail_k: zeros(context, (taps - 1) * kvd),
                    tail_v: zeros(context, (taps - 1) * kvd),
                    tail_o: zeros(context, (taps - 1) * h),
                    tail_f: zeros(context, (taps - 1) * h),
                }
            })
            .collect();
        let tap_buffer = (std::env::var("TURBOSPARK_INKLING2_TAPS").as_deref() == Ok("1"))
            .then(|| context.new_buffer_with_data(&vec![0u8; super::DSPARK_TAP_ROWS * h * 2]));
        Ok(RealInkling2State {
            taps: tap_buffer,
            batch: None,
            verify: None,
            dspark: None,
            resident,
            lamq: lamq.map(std::sync::Arc::new),
            max_ctx,
            slots,
            offsets,
            kinds,
            routed,
            router_bias,
            msg_queries,
            bank_states: 0,
            cur: half(context, h),
            attn_out: half(context, h),
            prelude: half(context, h),
            states: half(context, MAX_STATES * h),
            bank: half(context, MAX_STATES * h),
            mixed: half(context, 4 * h),
            msgs: half(context, 4 * h),
            inputs: half(context, 3 * h),
            normed: half(context, 3 * h),
            cat: half(context, 2 * h),
            gate_r: half(context, rank),
            gain: half(context, h),
            forget: half(context, h),
            q: half(context, heads * hd),
            k_pre: half(context, kvd),
            v_pre: half(context, kvd),
            r: half(context, heads * d_rel),
            sinks: half(context, heads),
            gate_attn: half(context, heads * hd),
            dk: half(context, kvh * taps),
            dv: half(context, kvh * taps),
            table: context.new_output_buffer((heads * extent * 4) as u64),
            attn_o: half(context, heads * hd),
            o2: half(context, h),
            m: half(context, h),
            mn: half(context, h),
            router_logits: context.new_output_buffer((experts * 4) as u64),
            acts: zeros(context, gpu::MAX_STREAMED_EXPERTS * ffn),
            routing_w: zeros(context, gpu::MAX_STREAMED_EXPERTS),
            y: half(context, h),
            y2: half(context, h),
            cfg,
        })
    }
}

impl RealInkling2State {
    /// The checkpoint's own NVFP4 / MXFP8 expert type of every layer, or
    /// the reason there is none. The batched verify (and so the DSpark
    /// drafter) has multi-row kernels for those rows only: a LookAtMyQuant
    /// install codes its matrices and experts, and a GGUF install's experts
    /// run the GGUF pairs, so both decode without speculation.
    /// Why this install cannot run the multi-row forward (speculative verify,
    /// chunked prefill), or `None`. A LookAtMyQuant install runs it with its
    /// coded GEMVs, embedding, head and expert pair called per row (the same
    /// calls decode makes, so the same bits); a GGUF install's experts have no
    /// multi-row form.
    /// Why this install has no multi-row forward (verify, chunked prefill,
    /// DSpark), or `None`. Native and GGUF experts both have one (GGUF rows
    /// run the decode pair over per-row blob tables); a LookAtMyQuant
    /// install needs its coded experts.
    pub(crate) fn multi_row_refusal(&self) -> Option<String> {
        let l = self.lamq.as_ref()?;
        l.experts.is_none().then(|| {
            "a LookAtMyQuant install without coded experts has no multi-row forward".to_string()
        })
    }

    /// Zeroes every slot's conv tails and drops the verify record: a new
    /// generation starts from an empty context. KV rows need no clearing
    /// (attention never reads past the cursor).
    pub(crate) fn reset(&mut self) {
        for s in &self.slots {
            for t in [&s.tail_k, &s.tail_v, &s.tail_o, &s.tail_f] {
                gpu::write_buffer_bytes(t, 0, &vec![0u8; t.length() as usize]);
            }
        }
        self.verify = None;
        self.bank_states = 0;
        if let Some(d) = self.dspark.as_mut() {
            d.reset();
        }
    }
}

/// Every routed expert resident in one GPU buffer per physical layer, copied
/// from the packed layer file as written (expert `e` at `e * stride`), plus
/// the router biases on the GPU and the selection the router writes. About
/// 15.7 GB on the Mini, on top of whatever slot cache the open allocated.
pub(crate) struct ResidentExperts {
    pub(crate) layers: Vec<gpu::MetalBuffer>,
    pub(crate) strides: Vec<u64>,
    pub(crate) bias: Vec<gpu::MetalBuffer>,
    /// `u32[top_k]` ranked expert ids, written by the GPU router each
    /// application and read by the indexed MoE pair in the same buffer.
    pub(crate) sel: gpu::MetalBuffer,
    /// One GGUF blob table (`gpu::encode_inkling2_gguf_bind_rows`) for
    /// decode on a GGUF install's resident experts.
    pub(crate) table: gpu::MetalBuffer,
    pub(crate) n_experts: u32,
}

impl ResidentExperts {
    fn load(
        context: &mut gpu::MetalContext,
        install_dir: &std::path::Path,
        layout: &model_io::PackedExpertsLayout,
        router_bias: &[Vec<f32>],
        experts: usize,
    ) -> Result<Self, RealForwardError> {
        let unsupported = RealForwardError::Unsupported;
        let mut layers = Vec::with_capacity(layout.layers.len());
        let mut strides = Vec::with_capacity(layout.layers.len());
        for (l, ll) in layout.layers.iter().enumerate() {
            let stride = ll.expert_stride;
            // The kernels address expert `e` at `e * stride`: refuse a layer
            // file that does not lay its experts out that way.
            if ll.experts.len() != experts
                || ll
                    .experts
                    .iter()
                    .enumerate()
                    .any(|(i, e)| e.expert != i || e.offset != i as u64 * stride)
            {
                return Err(unsupported(format!(
                    "resident experts: layer {l} does not place expert e at e * {stride}"
                )));
            }
            let path = install_dir.join("packed_experts").join(&ll.file);
            let mut bytes = std::fs::read(&path).map_err(|e| {
                unsupported(format!("resident experts: read {}: {e}", path.display()))
            })?;
            let want = stride as usize * experts;
            if bytes.len() < want {
                return Err(unsupported(format!(
                    "resident experts: {} is {} B, expected {want}",
                    path.display(),
                    bytes.len()
                )));
            }
            bytes.truncate(want);
            layers.push(context.new_buffer_with_data(&bytes));
            strides.push(stride);
        }
        let bias = router_bias
            .iter()
            .map(|b| {
                let bytes: Vec<u8> = b.iter().flat_map(|v| v.to_le_bytes()).collect();
                context.new_buffer_with_data(&bytes)
            })
            .collect();
        let sel = context.new_buffer_with_data(&[0u8; 4 * gpu::INKLING2_ROUTER_TOP_K]);
        let table = context.new_output_buffer(gpu::INKLING2_GGUF_TABLE_BYTES);
        Ok(Self {
            layers,
            strides,
            bias,
            sel,
            table,
            n_experts: experts as u32,
        })
    }
}
