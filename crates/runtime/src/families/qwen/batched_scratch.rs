//! Metal buffer allocation for batched Qwen forward verification passes.

use model_io::ArchConfig;

/// The M-row siblings of the buffers `DecodeScratch` and `RealQwenState`
/// hold one row of.
///
/// **Allocated only when a draft depth was asked for**, beside [`super::mtp::MtpState`]
/// and for its reason: `TURBOSPARK_MTP_DRAFT` unset must allocate nothing, so
/// that `qwen38_memory_oracle`'s frozen row keeps describing the engine that
/// shipped before this module existed. At the block sizes that pay this is
/// ~10 MiB, almost all of it `logits` (`batch * vocab` halfs, and this
/// family's vocab is 248,320).
///
/// `x` is NOT here: `DecodeScratch::x` already holds `MAX_PREFILL_BATCH`
/// rows, because the residual stream crosses layers and the chunk driver
/// needed the same thing first.
pub(crate) struct BatchedScratch {
    /// Rows this was sized for. A verify of block M needs `M + 1`.
    pub(crate) batch: usize,
    pub(crate) normed: gpu::MetalBuffer,
    pub(crate) q_packed: gpu::MetalBuffer,
    pub(crate) q: gpu::MetalBuffer,
    pub(crate) attn_gate: gpu::MetalBuffer,
    pub(crate) attn_out: gpu::MetalBuffer,
    pub(crate) o: gpu::MetalBuffer,
    pub(crate) gdn_qkv_raw: gpu::MetalBuffer,
    pub(crate) gdn_conv_out: gpu::MetalBuffer,
    pub(crate) gdn_z: gpu::MetalBuffer,
    pub(crate) gdn_a: gpu::MetalBuffer,
    pub(crate) gdn_b: gpu::MetalBuffer,
    pub(crate) gdn_y: gpu::MetalBuffer,
    pub(crate) gdn_out: gpu::MetalBuffer,
    pub(crate) moe_x: gpu::MetalBuffer,
    pub(crate) ffn_gate: gpu::MetalBuffer,
    pub(crate) ffn_up: gpu::MetalBuffer,
    pub(crate) ffn_act: gpu::MetalBuffer,
    pub(crate) h2: gpu::MetalBuffer,
    pub(crate) logits: gpu::MetalBuffer,
    /// The routed half's M-row buffers, `None` on a DENSE install.
    ///
    /// Gated on `num_experts != 0` for the same reason the whole struct is
    /// gated on a draft depth: a dense install must allocate exactly what it
    /// allocated before the routed batched path existed, so that
    /// `qwen38_memory_oracle`'s frozen row keeps describing it. A dense
    /// checkpoint also has no `moe_intermediate_size` (it encodes 0 there,
    /// per `crates/runtime` Gotcha 12's note on the two widths), so the
    /// sizes below are not merely wasted, they are meaningless.
    pub(crate) routed: Option<BatchedRoutedScratch>,
    /// Per-linear-layer copies of the row inputs the GDN state advance
    /// consumed (`qkv_raw`, `a`, `b`), `[layer ordinal][batch][dim]` FP16,
    /// `Some` only on the VERIFY scratch the drafters own. This is the tape
    /// `RealForwardRunner::rollback_retaining` replays the accepted prefix
    /// over; the chunked-prefill driver's own `BatchedScratch`
    /// (`RealQwenState::batched_prefill`) passes `false` and holds `None`,
    /// because a prefill is never rolled back and the bytes would be dead
    /// weight on every chunked prompt.
    pub(crate) verify_tape: Option<BatchedVerifyTape>,
    /// Whether the INT4 GEMMs encoded against this scratch may run the
    /// vendored Steel kernel (`real_forward_dispatch::encode_gemm_rows`).
    /// Set only on the chunked-prefill driver's instance: Steel is not
    /// bit-exact against the scalar GEMV, and the drafters' verify pass must
    /// reproduce sequential decode exactly, so their scratch keeps `false`.
    pub(crate) steel_gemm: bool,
    /// The multi-row GQA attention partials
    /// (`gpu::encode_attention_decode_gqa_rows`), `Some` only on the
    /// chunked-prefill driver's instance with `TURBOSPARK_PREFILL_ROWS_ATTN`
    /// on (`RealQwenState::ensure_batched_prefill`). It reads each K/V tile
    /// once per group of rows instead of once per row, and like Steel it is
    /// not bit-exact against the one-row kernel, so the drafters' verify
    /// scratch never carries it.
    pub(crate) rows_attention: Option<gpu::GqaRowsScratch>,
    /// The tree of the verify pass being encoded, `Some` only for the
    /// duration of a `RealForwardRunner::verify_tree` call on the DFlash2
    /// verify scratch (`TURBOSPARK_DFLASH_TREE`). Carried here rather than
    /// threaded through the layer encoders because they are shared with
    /// the chunked prefill and the MTP verify, which never set it; the
    /// encoders that read it are the attention core (RoPE positions and the
    /// ancestor-mapped attention) and the linear core (the tree conv and
    /// delta kernels).
    pub(crate) tree: std::cell::RefCell<Option<TreeSpec>>,
}

/// A tree verify's shape over its `rows` slots: slot 0 is the anchor,
/// `parent[s] < s` (-1 at the anchor), `depth[s]` the slot's depth below the
/// anchor and `paths[s]` its root path, anchor first.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct TreeSpec {
    pub(crate) parent: Vec<i32>,
    pub(crate) depth: Vec<u32>,
    pub(crate) paths: Vec<Vec<u32>>,
}

impl TreeSpec {
    /// Refuses a parent array that is not a tree rooted at slot 0 with
    /// parents before children.
    pub(crate) fn new(parent: &[i32]) -> Result<Self, String> {
        if parent.first() != Some(&-1) {
            return Err(format!(
                "a tree's slot 0 is the anchor (parent -1): {parent:?}"
            ));
        }
        let mut depth = Vec::with_capacity(parent.len());
        let mut paths: Vec<Vec<u32>> = Vec::with_capacity(parent.len());
        for (s, &p) in parent.iter().enumerate() {
            if s == 0 {
                depth.push(0);
                paths.push(vec![0]);
                continue;
            }
            if p < 0 || p as usize >= s {
                return Err(format!("slot {s}'s parent {p} must precede it: {parent:?}"));
            }
            depth.push(depth[p as usize] + 1);
            let mut path = paths[p as usize].clone();
            path.push(s as u32);
            paths.push(path);
        }
        Ok(Self {
            parent: parent.to_vec(),
            depth,
            paths,
        })
    }
}

/// The tape buffers. One `qkv_raw` row set and one `a`/`b` row set per
/// linear layer; the recurrent state after `C` rows is a function of the
/// block-start snapshot and these rows for `C` only, which is what makes a
/// retaining rollback a kernel replay instead of a forward pass. `z` is
/// deliberately not taped: the state advance never reads it (it feeds the
/// gated norm, whose output feeds the residual, which the next pass
/// recomputes from embeddings anyway).
///
/// With `TURBOSPARK_GDN_LAZY_TAPE` on (the default), the tape is LAZY: the
/// verify leaves the live state at the block start, and `rank1` carries what
/// the round-end fold needs instead of a replay -- the post-qk-norm `k`
/// slice and the per-row fp32 `delta` the verify kernel recorded in
/// registers. The eager mode (flag `0`) leaves `rank1` absent and replays
/// over `qkv_raw`/`a`/`b` from a snapshot instead.
pub(crate) struct BatchedVerifyTape {
    pub(crate) qkv_raw: gpu::MetalBuffer,
    pub(crate) a: gpu::MetalBuffer,
    pub(crate) b: gpu::MetalBuffer,
    pub(crate) rank1: Option<Rank1Tape>,
}

/// The lazy tape's rank-1 record, `[linear layer ordinal][row][...]`, k as
/// FP16 (`Hk * Dk`, the post-norm slice `gdn_delta_step_prefill` consumes)
/// and delta as FP32 (`Hv * Dv`). `g` is not recorded: the fold recomputes
/// it from taped `a` with the same expression the delta kernel runs.
pub(crate) struct Rank1Tape {
    pub(crate) k: gpu::MetalBuffer,
    pub(crate) delta: gpu::MetalBuffer,
}

/// The M-row siblings of the buffers the per-token routed pass
/// (`families/qwen/moe.rs`) holds one row of.
///
/// `ffn_gate` / `ffn_up` / `ffn_act` are NOT here: the shared expert's
/// width is `intermediate_size`, which is what the dense FFN buffers above
/// are already sized for, and the two never run in the same layer.
pub(crate) struct BatchedRoutedScratch {
    /// `[M * top_k, moe_inter]` FP16, phase 1's output over the route list.
    pub(crate) batch_acts: gpu::MetalBuffer,
    /// `[M, hidden]` FP16, the fused phase 2's output.
    pub(crate) batch_y: gpu::MetalBuffer,
    /// `[M, hidden]` FP16, the GATED shared-expert output per token, and
    /// phase 2's accumulator SEED. It cannot be the single-row `h1` the
    /// per-token path reuses: all M tokens' shared branches are encoded
    /// before any of them is consumed.
    pub(crate) batch_h1: gpu::MetalBuffer,
    /// `[M]` FP16, one `shared_expert_gate` logit per token.
    pub(crate) batch_gate_logit: gpu::MetalBuffer,
    /// `[M * top_k]` FP16 in PAIR order (token-major, rank within), which
    /// is the layout the fused phase 2 looks its routes up by -- not the
    /// decode path's slot-indexed `MAX_STREAMED_EXPERTS` row.
    pub(crate) batch_routing_w: gpu::MetalBuffer,
    /// The encoded route list, 16 bytes per `MoePrefillRoute`.
    pub(crate) batch_routes: gpu::MetalBuffer,
    /// `[M, num_experts]` FP32 router logits, read back for the WHOLE
    /// batch in one host wait. `RealQwenState::router_logits_f32` holds a
    /// single row and is not widened: that buffer is allocated by every
    /// MoE install whether or not a drafter is open, and this one is not.
    pub(crate) batch_router_logits_f32: gpu::MetalBuffer,
    /// The wide expert-blob argument buffer, bound once per layer with
    /// every cache slot.
    pub(crate) wide_blobs: gpu::RoutedBlobsWideBuffer,
    /// A second one, so the routed prefill's read-ahead can bind the next
    /// sub-batch while the GPU still reads the first
    /// (`TURBOSPARK_ROUTED_BATCH_OVERLAP`). Unused otherwise.
    pub(crate) wide_blobs_alt: gpu::RoutedBlobsWideBuffer,
}

impl BatchedScratch {
    /// `record_tape` is `true` only for the verify scratch the drafters
    /// own (`MtpState`, `DflashState`); the chunked-prefill driver's
    /// instance passes `false`.
    pub(crate) fn new(
        context: &mut gpu::MetalContext,
        arch: &ArchConfig,
        qwen_shape: gpu::GdnShape,
        batch: usize,
        record_tape: bool,
    ) -> Result<Self, gpu::GpuError> {
        let hidden = arch.hidden_size as u64;
        let q_dim = (arch.num_heads * arch.full_head_dim) as u64;
        let inter = arch.intermediate_size as u64;
        let vocab = arch.vocab_size as u64;
        let qkv_dim = qwen_shape.qkv_dim() as u64;
        let value_dim = qwen_shape.value_dim() as u64;
        let v_heads = qwen_shape.num_v_heads as u64;
        let b = batch as u64;
        // The argument buffer needs `context` MUTABLY where every plain
        // allocation below needs it immutably through `halfs`, so it is
        // built first and its borrow ends here.
        let wide_blobs = if arch.num_experts == 0 {
            None
        } else {
            let silu = arch.hidden_activation.contains("silu");
            Some((
                gpu::RoutedBlobsWideBuffer::new(context, silu)?,
                gpu::RoutedBlobsWideBuffer::new(context, silu)?,
            ))
        };
        let halfs = |n: u64| context.new_output_buffer(n.max(1) * b * 2);
        // One row set per LINEAR layer, addressed by linear-layer ordinal
        // (`arch.layer_is_linear`), not by layer index -- the same ordinal
        // both the recording blit and the replay walk compute.
        let verify_tape = if record_tape {
            let linear_layers = (0..arch.num_layers as usize)
                .filter(|&l| arch.layer_is_linear(l))
                .count() as u64;
            // The lazy tape's extra buffers ride the same per-slot stride as
            // the row tapes: a slot holds `batch` rows of each record.
            let rank1 = if crate::real_forward_rollback::gdn_lazy_tape() {
                let k_width = (qwen_shape.num_k_heads * qwen_shape.key_head_dim) as u64;
                let delta_width = (qwen_shape.num_v_heads * qwen_shape.value_head_dim) as u64;
                Some(Rank1Tape {
                    k: context.new_output_buffer(linear_layers * b * k_width * 2),
                    delta: context.new_output_buffer(linear_layers * b * delta_width * 4),
                })
            } else {
                None
            };
            Some(BatchedVerifyTape {
                qkv_raw: halfs(linear_layers * qkv_dim),
                a: halfs(linear_layers * v_heads),
                b: halfs(linear_layers * v_heads),
                rank1,
            })
        } else {
            None
        };
        let routed = wide_blobs.map(|(wide_blobs, wide_blobs_alt)| {
            let top_k = arch.top_k_experts as u64;
            let moe_inter = arch.moe_intermediate_size.max(1) as u64;
            BatchedRoutedScratch {
                batch_acts: halfs(top_k * moe_inter),
                batch_y: halfs(hidden),
                batch_h1: halfs(hidden),
                batch_gate_logit: halfs(1),
                batch_routing_w: halfs(top_k),
                batch_routes: context
                    .new_output_buffer(b * top_k * gpu::MoePrefillRoute::STRIDE_BYTES as u64),
                batch_router_logits_f32: context
                    .new_output_buffer(b * arch.num_experts.max(1) as u64 * 4),
                wide_blobs,
                wide_blobs_alt,
            }
        });
        Ok(Self {
            batch,
            normed: halfs(hidden),
            q_packed: halfs(2 * q_dim),
            q: halfs(q_dim),
            attn_gate: halfs(q_dim),
            attn_out: halfs(q_dim),
            o: halfs(hidden),
            gdn_qkv_raw: halfs(qkv_dim),
            gdn_conv_out: halfs(qkv_dim),
            gdn_z: halfs(value_dim),
            gdn_a: halfs(v_heads),
            gdn_b: halfs(v_heads),
            gdn_y: halfs(value_dim),
            gdn_out: halfs(value_dim),
            moe_x: halfs(hidden),
            ffn_gate: halfs(inter),
            ffn_up: halfs(inter),
            ffn_act: halfs(inter),
            h2: halfs(hidden),
            // Only the verify pass reads M rows of logits (`batched.rs`). The
            // chunked-prefill instance (no tape) runs its head as a single-row
            // GEMV into `DecodeScratch::logits`, so one row is enough there,
            // and at the Steel micro-batch the full plane would be ~32 MiB of
            // a 248K vocabulary nobody reads.
            logits: context.new_output_buffer(vocab * if record_tape { b } else { 1 } * 2),
            routed,
            verify_tape,
            steel_gemm: false,
            rows_attention: None,
            tree: std::cell::RefCell::new(None),
        })
    }
}
