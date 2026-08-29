//! The per-token routed-MoE pipelining pattern a chunked-prefill driver
//! needs, shared across every family whose routed half requires a host
//! round trip (a router top-k the GPU cannot decide on its own). Landed
//! first for Gemma 4 (`families/gemma4/moe.rs`, `prefill.rs`) and moved
//! here once `families/llama/` and `families/gptoss/` needed the identical
//! struct and retire logic rather than a second and third copy of it.
//!
//! The BATCHED routed half's union-bounded sub-batch planner lives here
//! for the same reason, extracted once `families/gemma4/`, `families/qwen/`
//! and `families/gptoss/` held three near-verbatim copies of the greedy
//! shrink, the first-seen union, the plan-and-pread round and the
//! pair-order route assembly. The family files keep what actually differs:
//! kernels, bias handling, phase-2 seeding, and the tail.

use std::collections::{HashMap, HashSet};
use std::time::Instant;

use half::f16;

use crate::real_forward::RealForwardRunner;
use crate::real_forward_types::{PhaseCounters, RealForwardError, ROUTED_BANKS};

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
///
/// THE `banks == 1` FALLBACK DOES NOT MAKE EVERY `expert_cache_slots` VALUE
/// SAFE (AGENTS.md Gotcha 64). Every per-token routed prefill loop that
/// calls this (`families/{gemma4,gptoss,llama}/prefill.rs`) still carries
/// the previous token's `used` slots into the NEXT token's `protect` set
/// unconditionally, regardless of `banks` -- the retire-before-encode the
/// comment above describes makes that reservation stale (the buffer is
/// provably no longer in flight) rather than removing it. At
/// `expert_cache_slots == top_k` exactly, a stale reservation of one
/// token's worth of slots can leave zero room for the next token's misses,
/// which panics `ExpertCache::plan` on the first multi-token prefill of a
/// real prompt rather than degrading. `--expert-cache-slots 8` on a
/// top_k=8 model (real Gemma 4) hits this deterministically; 16 does not,
/// because `16 >= 2 * 8` takes the pipelined branch instead.
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

    /// Resolves the layer's expert streamer (`family` names it in the
    /// refusal) and runs one plan-and-pread round for a sub-batch's union
    /// through [`plan_and_stream_union`]. The method form serves the two
    /// batched drivers that are `RealForwardRunner` methods; the qwen
    /// verify, a free function, calls the helper directly.
    pub(crate) fn plan_routed_union(
        &mut self,
        layer: usize,
        family: &str,
        union: &[usize],
    ) -> Result<HashMap<usize, usize>, RealForwardError> {
        let streamer = self.streamers[layer].as_mut().ok_or_else(|| {
            RealForwardError::Unsupported(format!(
                "{family} layer {layer} has no packed-expert streamer"
            ))
        })?;
        plan_and_stream_union(streamer, &mut self.phases, union)
    }
}

/// One union-bounded sub-batch of a batched routed prefill pass: how many
/// tokens from `sub_start` it covers, and their distinct experts in
/// FIRST-SEEN order (the order the plan binds them in).
pub(crate) struct RoutedSubBatch {
    /// First token of the sub-batch, an index into `selected_all`.
    pub(crate) sub_start: usize,
    /// Tokens covered; at least 1.
    pub(crate) sub_len: usize,
    /// The sub-batch's expert union in first-seen order.
    pub(crate) union: Vec<usize>,
}

/// The greedy shrink: as many tokens from `sub_start` as the union of
/// their routes fits `cap` slots. The bound exists because a sub-batch
/// keeps its whole union resident at once and `ExpertCache::plan` ASSERTS
/// `experts.len() <= slot_count` rather than degrading (AGENTS.md Gotcha
/// 54); it is computed per layer because it is a property of the ROUTING,
/// so a layer whose routing is unusually scattered shrinks further. The
/// union is built inside the shrink, so first-seen order falls out for
/// free rather than needing a second dedup pass.
pub(crate) fn next_routed_sub_batch(
    selected_all: &[Vec<usize>],
    sub_start: usize,
    cap: usize,
    layer: usize,
    top_k: usize,
) -> Result<RoutedSubBatch, RealForwardError> {
    let mut union: Vec<usize> = Vec::new();
    let mut seen: HashSet<usize> = HashSet::new();
    for &e in &selected_all[sub_start] {
        if seen.insert(e) {
            union.push(e);
        }
    }
    if union.len() > cap {
        return Err(RealForwardError::Unsupported(format!(
            "layer {layer}: one token routes {top_k} experts against {cap} slots"
        )));
    }
    let mut sub_len = 1usize;
    while sub_start + sub_len < selected_all.len() {
        let selected = &selected_all[sub_start + sub_len];
        let new = selected.iter().filter(|e| !seen.contains(*e)).count();
        if union.len() + new > cap {
            break;
        }
        for &e in selected {
            if seen.insert(e) {
                union.push(e);
            }
        }
        sub_len += 1;
    }
    Ok(RoutedSubBatch {
        sub_start,
        sub_len,
        union,
    })
}

/// One expert-cache plan and one pread round for a sub-batch's union, with
/// the phase counters booked, returning the expert-to-slot map the route
/// assembly needs. The plan order cannot move bytes: the fused phase 2
/// reduces over RANKS per token, never over slots, so cache-slot
/// assignment is numerically invisible here -- a stronger property than
/// the decode kernel has (AGENTS.md Gotcha 27).
pub(crate) fn plan_and_stream_union(
    streamer: &mut streaming::PreadExpertStreamer,
    phases: &mut PhaseCounters,
    union: &[usize],
) -> Result<HashMap<usize, usize>, RealForwardError> {
    let t_io = Instant::now();
    let plan = streamer.plan_experts_cached(union, &HashSet::new());
    phases.expert_requests += plan.experts.len() as u64;
    phases.expert_hits += plan.hits as u64;
    let slots = streamer
        .execute_expert_cache_plan(&plan)
        .map_err(|e| RealForwardError::Unsupported(format!("expert stream: {e}")))?;
    phases.expert_io_nanos += t_io.elapsed().as_nanos() as u64;
    Ok(union.iter().copied().zip(slots).collect())
}

/// The route list and FP16 routing weights for one sub-batch, in PAIR
/// order (token-major, rank within): the fused phase 2 looks its routes up
/// BY PAIR, so a blob-locality sort would break it -- and each blob is
/// megabytes against caches far smaller, so the sort would buy nothing
/// anyway.
pub(crate) fn encode_routes(
    selected_all: &[Vec<usize>],
    weights_all: &[Vec<f32>],
    sub: &RoutedSubBatch,
    slot_of: &HashMap<usize, usize>,
    top_k: usize,
) -> (Vec<gpu::MoePrefillRoute>, Vec<f16>) {
    let mut routes = Vec::with_capacity(sub.sub_len * top_k);
    let mut routing16 = Vec::with_capacity(sub.sub_len * top_k);
    for i in 0..sub.sub_len {
        for r in 0..top_k {
            let expert = selected_all[sub.sub_start + i][r];
            routes.push(gpu::MoePrefillRoute {
                token: i as u32,
                rank: r as u32,
                slot: *slot_of
                    .get(&expert)
                    .expect("plan assigned every union expert a slot") as u32,
            });
            routing16.push(f16::from_f32(weights_all[sub.sub_start + i][r]));
        }
    }
    (routes, routing16)
}
