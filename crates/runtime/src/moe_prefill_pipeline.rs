//! The per-token routed-MoE pipelining pattern a chunked-prefill driver
//! needs, shared across every family whose routed half requires a host
//! round trip (a router top-k the GPU cannot decide on its own). Landed
//! first for Gemma 4 (`families/gemma4/moe.rs`, `prefill.rs`) and moved
//! here once `families/llama/` and `families/gptoss/` needed the identical
//! struct and retire logic rather than a second and third copy of it.

use std::time::Instant;

use crate::real_forward::RealForwardRunner;
use crate::real_forward_types::ROUTED_BANKS;

/// Which token of a prefill micro-batch a routed pass is for, and which
/// bank of per-token routed resources it may use. Both are 0 on the
/// sequential decode path, which is why that path's bytes cannot move.
#[derive(Debug, Clone)]
pub(crate) struct RoutedSlot {
    /// Index inside the micro-batch: selects the `x`, `routed_x` and
    /// router-logits rows.
    pub(crate) token: usize,
    /// Index inside [`crate::real_forward_types::ROUTED_BANKS`]: selects
    /// the two resources the HOST writes per token, `routing_w` and the
    /// routed argument buffer. The GPU-only intermediates are not banked;
    /// see `ROUTED_BANKS` for why that is safe and this is not.
    pub(crate) bank: usize,
    /// Expert slots a command buffer still in flight is reading, which this
    /// token's plan may not evict. Empty on the sequential path and for the
    /// first token of a micro-batch.
    pub(crate) protect: std::collections::HashSet<usize>,
}

impl RoutedSlot {
    /// The sequential decode path: token 0, bank 0, nothing in flight.
    pub(crate) fn sequential() -> Self {
        Self {
            token: 0,
            bank: 0,
            protect: std::collections::HashSet::new(),
        }
    }
}

/// How many tokens' routed command buffers may be in flight at once.
/// Pipelining costs a plan that must AVOID the previous token's slots, so
/// the cache needs room for those plus this token's misses; below
/// `2 * top_k` slots it cannot guarantee that and `ExpertCache::plan` aborts
/// the process rather than degrading, so a small cache falls back to
/// retiring before it encodes (`banks == 1`).
pub(crate) fn routed_pipeline_banks(expert_cache_slots: usize, top_k: usize) -> usize {
    if expert_cache_slots >= 2 * top_k {
        ROUTED_BANKS
    } else {
        1
    }
}

impl RealForwardRunner {
    /// Waits out a pipelined routed command buffer and books its GPU time,
    /// leaving `pending` empty. A no-op when nothing is in flight.
    pub(crate) fn retire_routed(&mut self, pending: &mut Option<gpu::CommittedPass>) {
        if let Some(committed) = pending.take() {
            let t_retire = Instant::now();
            self.phases.routed_cb_gpu_nanos += (committed.wait_with_gpu_time() * 1e9) as u64;
            self.phases.pipeline_wait_nanos += t_retire.elapsed().as_nanos() as u64;
        }
    }
}
