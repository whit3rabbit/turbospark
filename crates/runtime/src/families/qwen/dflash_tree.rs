//! The runner's half of a DFlash2 tree round (`TURBOSPARK_DFLASH_TREE`,
//! `docs/KAPITSA_KERNELS.md` lever 4): the switch, the tree verify, and the
//! compaction that turns the accepted path back into the chain layout every
//! round-end helper reads.
//!
//! After a tree verify the accepted path sits at its node SLOTS, not at rows
//! `0..keep`. `commit_pending`, `rollback_retaining`, the drafter's next
//! context write and the fold all read rows `0..keep` consecutively, so the
//! path is gathered into them first, in place (`gpu::encode_gather_rows_fp16`,
//! race-free because a path's slots are strictly increasing with
//! `path[i] >= i`). Every per-row buffer the round-end reads is gathered:
//! each attention layer's K and V, the lazy tape's inputs to the fold
//! (`qkv_raw`, `a`, `k` and fp32 `delta`), and the drafter's capture rows.
//! The eager tape's `b` is not: the tree runs only on the lazy tape, whose
//! fold never reads it. Of the residual rows, only the one
//! `rollback_retaining` reads (`keep - 1`) is moved, on the host.
//!
//! The gather pass is committed WITHOUT a wait. Every reader of what it
//! writes is later GPU work on the same queue (the fold, the next context
//! write, the next verify), and buffers on one queue execute in commit order
//! (`crates/gpu` Gotcha 8). The one host reader, the rollback's copy of
//! residual row `keep - 1`, is served by the host move below instead, so it
//! never reads a row the GPU has not written yet. A node at slot `s` was rotated at `start +
//! depth[s]`, so after compaction row `i` carries position `start + i`, as
//! the chain's does. The machinery after that runs unchanged.

use foundation::{LogitValue, TokenId};

use super::batched_scratch::TreeSpec;
use crate::real_forward::{RealForwardError, RealForwardRunner};
use crate::real_forward_rollback::gdn_lazy_tape;

/// Greedy DFlash2 rounds verify a best-first tree instead of the
/// selector's chain. Default on; `TURBOSPARK_DFLASH_TREE=0` keeps the chain.
/// Read once.
pub(crate) fn dflash_tree() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("TURBOSPARK_DFLASH_TREE").as_deref() != Ok("0"))
}

impl RealForwardRunner {
    /// Nodes per tree round when this runner can verify one, `None` when it
    /// cannot: the switch is on, the DFlash2 drafter alone is open, the GDN
    /// tape is lazy (the tree kernels never write the state back), the
    /// trunk's decode attention is the GQA-shared kernel (the tree has no
    /// barrier-kernel variant), and the round fits the tree kernels' 8
    /// rows. The node count is the drafter's block, so a tree round costs
    /// the chain round's rows.
    pub fn dflash_tree_nodes(&self) -> Option<usize> {
        let d = self.real_dflash.as_ref()?;
        let qwen = self.real_qwen.as_ref()?;
        let arch = &self.arch;
        let ready = dflash_tree()
            && self.real_mtp.is_none()
            && gdn_lazy_tape()
            && gpu::gdn_tree_supported(qwen.shape)
            && self.scratch.attn.decode_kernel() == gpu::DecodeAttentionKernel::GqaShared
            && gpu::attention_decode_gqa_supported(
                arch.full_head_dim as u32,
                arch.num_heads as u32,
                arch.num_full_kv_heads as u32,
            )
            && d.block < gpu::GDN_TREE_MAX_ROWS as usize
            && d.block < gpu::ATTENTION_TREE_MAX_DEPTH;
        ready.then_some(d.block)
    }

    /// The tree verify: [`Self::produce_batched`] over `tokens` at their
    /// slots, where slot `s`'s row continues from `parent[s]`'s
    /// (`parent[0] == -1`, the anchor). Each slot's logits are the chain
    /// verify's over that slot's root path.
    pub fn dflash_verify_tree(
        &mut self,
        tokens: &[TokenId],
        start_position: usize,
        parent: &[i32],
        logits: &mut [LogitValue],
    ) -> Result<(), RealForwardError> {
        if self.dflash_tree_nodes().is_none() {
            return Err(RealForwardError::Unsupported(
                "this runner cannot verify a DFlash2 tree (see dflash_tree_nodes)".to_string(),
            ));
        }
        if parent.len() != tokens.len() {
            return Err(RealForwardError::Unsupported(format!(
                "a tree of {} slots verified with {} tokens",
                parent.len(),
                tokens.len()
            )));
        }
        let spec = TreeSpec::new(parent).map_err(RealForwardError::Unsupported)?;
        foundation::engaged!("dflash_tree");
        let d = self.real_dflash.as_ref().expect("checked above");
        *d.batched.tree.borrow_mut() = Some(spec);
        let result = self.produce_batched(tokens, start_position, logits);
        let d = self.real_dflash.as_ref().expect("checked above");
        *d.batched.tree.borrow_mut() = None;
        result
    }

    /// Gathers the accepted path's slots (`path[0] == 0`, strictly
    /// increasing) into rows `0..path.len()` of every per-row buffer the
    /// round-end reads. A no-op when the path is already `0..keep`. Must run
    /// after the tree verify and before `rollback_retaining` /
    /// `commit_pending` / the next draft.
    pub fn dflash_tree_compact(
        &mut self,
        start_position: usize,
        path: &[u32],
    ) -> Result<(), RealForwardError> {
        let tape = self.batched_tape.ok_or_else(|| {
            RealForwardError::Unsupported(
                "tree compaction needs the tree verify's tape; the last pass was not one"
                    .to_string(),
            )
        })?;
        if path.first() != Some(&0)
            || path.windows(2).any(|w| w[0] >= w[1])
            || path.last().is_some_and(|&s| s as usize >= tape.rows)
            || tape.start_position != start_position
        {
            return Err(RealForwardError::Unsupported(format!(
                "tree compaction path {path:?} is not an increasing path from slot 0 inside the \
                 {}-row verify at {}",
                tape.rows, tape.start_position
            )));
        }
        if path.iter().enumerate().all(|(i, &s)| s as usize == i) {
            return Ok(());
        }
        let rows = tape.rows;
        let arch = &self.arch;
        let hidden = arch.hidden_size as usize;
        let d = self.real_dflash.as_ref().expect("a tape implies a drafter");
        let qwen = self
            .real_qwen
            .as_ref()
            .ok_or_else(|| RealForwardError::Unsupported("not a Qwen install".to_string()))?;
        let shape = qwen.shape;
        let gdn_tape = d.batched.verify_tape.as_ref().ok_or_else(|| {
            RealForwardError::Unsupported("the DFlash2 verify scratch records no tape".to_string())
        })?;
        let rank1 = gdn_tape.rank1.as_ref().ok_or_else(|| {
            RealForwardError::Unsupported("tree compaction needs the lazy GDN tape".to_string())
        })?;
        // The residual row the rollback copies to row 0, moved on the host:
        // the verify that wrote it has been waited on, and no queued GPU work
        // touches `scratch.x`. No DFlash2 reader depends on it (only the
        // step-wise MTP head reads row 0, and the tree never runs with it
        // open), so dropping this line fails no test; it is kept so that
        // `rollback_retaining`'s row-0 contract holds whatever reads it later.
        // Rows `1..keep - 1` are read by nothing before the next pass.
        let keep = path.len();
        let row_bytes = hidden * 2;
        let last = path[keep - 1] as usize;
        let residual = gpu::read_buffer_bytes(&self.scratch.x, last * row_bytes, row_bytes);
        gpu::write_buffer_bytes(&self.scratch.x, (keep - 1) * row_bytes, &residual);

        let gpu_err = RealForwardError::Gpu;
        let context = &mut self.context;
        let kv = &self.kv;
        let pass = context.begin_pass_labeled("dflash tree compact");
        // One region of `rows` rows of `width` halfs at `offset`, in place.
        let gather = |context: &mut gpu::MetalContext,
                      buf: &gpu::MetalBuffer,
                      offset: u64,
                      width: usize|
         -> Result<(), RealForwardError> {
            gpu::encode_gather_rows_fp16(
                context,
                &pass,
                (buf, offset),
                (buf, offset),
                path,
                width as u32,
                width as u32,
                width as u32,
            )
            .map_err(gpu_err)
        };
        let (qkv_dim, v_heads) = (shape.qkv_dim() as usize, shape.num_v_heads as usize);
        let k_width = (shape.num_k_heads * shape.key_head_dim) as usize;
        let value_dim = shape.value_dim() as usize;
        let mut linear = 0usize;
        for layer in 0..arch.num_layers as usize {
            if arch.layer_is_linear(layer) {
                // The tape's per-layer stride is the verify's `rows`
                // (`encode_linear_core_batched`'s `slot * batch * dim`).
                let at = |dim: usize, bytes: usize| (linear * rows * dim * bytes) as u64;
                gather(context, &gdn_tape.qkv_raw, at(qkv_dim, 2), qkv_dim)?;
                gather(context, &gdn_tape.a, at(v_heads, 2), v_heads)?;
                gather(context, &rank1.k, at(k_width, 2), k_width)?;
                // fp32 rows, moved as twice as many halfs: a bit copy.
                gather(context, &rank1.delta, at(value_dim, 4), 2 * value_dim)?;
                linear += 1;
                continue;
            }
            let (k_buf, k_off) = kv.k_slot(layer, start_position);
            let (v_buf, v_off) = kv.v_slot(layer, start_position);
            gather(context, k_buf, k_off as u64, kv.k_stride(layer) / 2)?;
            gather(context, v_buf, v_off as u64, kv.v_stride(layer) / 2)?;
        }
        gather(context, &d.capture, 0, d.shape.aux_count * hidden)?;
        // Not waited (module header). Dropping the handle does not wait.
        drop(pass.commit());
        Ok(())
    }
}
