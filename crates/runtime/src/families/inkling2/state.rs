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

/// The most block states a depth mix sees (blocks plus the current state).
pub(crate) const MAX_STATES: usize = gpu::INKLING2_DEPTH_MAX_STATES;

pub(crate) fn name(layer: usize, suffix: &str) -> String {
    format!("model.llm.layers.{layer}.{suffix}")
}

/// One cache slot's KV and conv state.
pub(crate) struct SlotState {
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
    pub(crate) kinds: Vec<gpu::Inkling2ExpertType>,
    pub(crate) routed: gpu::RoutedBlobsBuffer,
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

impl RealInkling2State {
    pub(crate) fn build(
        context: &mut gpu::MetalContext,
        weights: &gpu::ResidentGpuWeights,
        index: &ResidentIndex,
        arch: &ArchConfig,
        install_dir: &std::path::Path,
        max_ctx: usize,
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

        // Every tensor the flow reads, checked once.
        require(
            index,
            "model.llm.embed.weight",
            TAG_BF16,
            arch.vocab_size as usize * h,
        )?;
        require(
            index,
            "model.llm.unembed.weight",
            TAG_BF16,
            arch.vocab_size as usize * h,
        )?;
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
            for (s, n) in [
                ("wq_du.weight", heads * hd * h),
                ("wk_dv.weight", kvd * h),
                ("wv_dv.weight", kvd * h),
                ("wr_du.weight", heads * d_rel * h),
                ("wo_ud.weight", h * heads * hd),
                ("gate_attn.weight", heads * hd * h),
                ("attn_sink_proj.weight", heads * h),
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
                    require(index, &a(s), TAG_BF16, kvh * taps * h)?;
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
                    require(index, &format!("{p}.down_weight"), TAG_BF16, rank * 2 * h)?;
                    require(index, &format!("{p}.gain_weight"), TAG_BF16, h * rank)?;
                    require(index, &format!("{p}.forget_weight"), TAG_BF16, h * rank)?;
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
        for (l, ll) in layout.layers.iter().enumerate() {
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
                "inkling2_nvfp4" => gpu::Inkling2ExpertType::Nvfp4,
                "inkling2_mxfp8" => gpu::Inkling2ExpertType::Mxfp8,
                other => return Err(unsupported(format!("layer {l}: expert dtype {other}"))),
            };
            for (role, s, rows, cols) in [
                ("gate", gate, ffn, h),
                ("up", up, ffn, h),
                ("down", down, h, ffn),
            ] {
                if s.dtype != gate.dtype || s.size as usize != rows * kind.row_bytes(cols) {
                    return Err(unsupported(format!(
                        "layer {l}: {role} blob size or dtype mismatch"
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
        let routed = gpu::RoutedBlobsBuffer::new_for(
            context,
            gpu::moe_inkling2_source(),
            gpu::MOE_INKLING2_REFLECT_FUNCTION,
            false,
        )
        .map_err(RealForwardError::Gpu)?;

        let half = |context: &gpu::MetalContext, n: usize| {
            context.new_output_buffer((n.max(1) * 2) as u64)
        };
        let zeros = |context: &mut gpu::MetalContext, n: usize| {
            context.new_buffer_with_data(&vec![0u8; n * 2])
        };
        let n_slots = cfg.slots(layers as u32) as usize;
        let slots = (0..n_slots)
            .map(|_| SlotState {
                k: half(context, max_ctx * kvd),
                v: half(context, max_ctx * kvd),
                tail_k: zeros(context, (taps - 1) * kvd),
                tail_v: zeros(context, (taps - 1) * kvd),
                tail_o: zeros(context, (taps - 1) * h),
                tail_f: zeros(context, (taps - 1) * h),
            })
            .collect();
        Ok(RealInkling2State {
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
