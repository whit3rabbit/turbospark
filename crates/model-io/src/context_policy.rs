//! How large a context window to open, and how `Auto` resolves.
//!
//! Two bounds decide this and they are not the same kind of thing.
//!
//! The MODEL's bound is its trained context, which is a quality claim: past
//! it, RoPE is extrapolating into positions the checkpoint never saw, and
//! output degrades gradually rather than failing. So exceeding it WARNS.
//!
//! The MACHINE's bound is arithmetic: `KvCacheManager::new` allocates every
//! layer's K and V buffers up front, so a window that does not fit is a
//! failed allocation or a swapping machine, and neither says which number
//! caused it. So exceeding it is REFUSED, with the subtraction shown.
//!
//! Deliberately portable, following [`crate::expert_cache_policy`]: no
//! `gpu`, no `#[cfg(target_os = "macos")]`, and no OS probe of its own.
//! Every input arrives as a parameter, so the whole policy is unit-tested on
//! any platform rather than needing a Mac with an install on disk.
//!
//! **HERE RATHER THAN IN `crates/runtime`, WHERE IT WAS WRITTEN.** It moved
//! when `crates/catalog` needed the same arithmetic to answer "would this fit
//! before I spend twenty minutes streaming it": that crate builds on every
//! platform and `runtime` does not (its `model_io` dependency is macOS-only),
//! so a `runtime` dependency would have made a portable question answerable
//! only on a Mac. The alternative was a second copy of
//! [`kv_bytes_for_context`] in a crate that could not see this one, which is
//! precisely the rot AGENTS.md Gotcha 38 is about -- and the formula's own
//! doc explains why a second copy would get the ring wrong. `runtime`
//! re-exports every name here, so `runtime::MaxContext` still resolves.

use std::path::Path;

use crate::ArchConfig;

/// Bytes per FP16 KV element. `KvCacheManager` stores K and V as FP16.
const FP16_SIZE: u64 = 2;

/// The prefill chunk budget `RealForwardRunner` passes to
/// `KvCacheManager::new`, which is added to the sliding window to size a
/// ring. Mirrored rather than imported because that constant is inside the
/// macOS-only module and this file compiles everywhere; the two are pinned
/// against each other by `the_ring_slack_matches_the_runners_chunk_budget`.
const MAX_PREFILL_CHUNK_TOKENS: u64 = 128;

/// Held back from the context budget for everything that is not KV: the
/// process baseline, the Metal command-buffer churn, the host sampler's
/// per-token scratch, and whatever else the user is running. The same
/// figure and the same reasoning as [`crate::HEADROOM_RESERVE_BYTES`],
/// which that module's doc describes as covering KV -- it was written when
/// the slot cache was the only thing being sized and KV was a fixed cost.
/// It is now the other sized term, so the two share the reserve rather than
/// each subtracting one.
pub const CONTEXT_RESERVE_BYTES: u64 = crate::expert_cache_policy::HEADROOM_RESERVE_BYTES;

/// The share of the remaining pool that `Auto` will spend on KV.
///
/// A quarter, the same as [`crate::HEADROOM_FRACTION`], and for the same
/// reason: the two budgets are siblings drawn from one pool, so together
/// they commit half of it and leave half for everything else. Note this is
/// the SUGGESTION's fraction only. An explicit `--max-context` is checked
/// against the whole pool, because a user naming a number has decided how
/// to spend their own machine and only needs to be stopped when it cannot
/// work at all.
pub const CONTEXT_BUDGET_FRACTION: f64 = 0.25;

/// Context sizing, as it reaches `runtime::RealForwardRunner`.
///
/// The mirror of `turbospark_invocation::MaxContext`, spelled again here
/// for the reason [`crate::ExpertCacheSlots`] and `runtime::PowerProfile`
/// are: this crate does not depend on the pure argument parser, and
/// `crates/cli` maps between the two.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MaxContext {
    /// Exactly this many tokens, whatever the machine and the checkpoint
    /// look like. Only a machine that cannot hold it refuses it.
    Fixed(u32),
    /// Size against the checkpoint and the machine at open.
    #[default]
    Auto,
}

/// Why a requested context cannot be opened.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContextTooLarge {
    /// The context that was asked for.
    pub requested: u32,
    /// KV bytes that context needs.
    pub needs: u64,
    /// KV bytes available after the install's own weights and the reserve.
    pub available: u64,
    /// Physical memory on this machine.
    pub physical: u64,
    /// What the install already commits before any KV: its mapped weight
    /// region (pinned for a streamed MoE install, AGENTS.md Gotcha 19) plus
    /// the largest slot cache `Auto` could pick. See [`committed_bytes`].
    pub committed: u64,
    /// The largest context that does fit `available`.
    pub largest_fitting: u32,
}

impl std::fmt::Display for ContextTooLarge {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "--max-context {} needs {} of KV cache; {} available \
             ({} physical - {} weights and expert cache - {} reserve). \
             Largest context that fits: {}",
            self.requested,
            gib(self.needs),
            gib(self.available),
            gib(self.physical),
            gib(self.committed),
            gib(CONTEXT_RESERVE_BYTES),
            self.largest_fitting,
        )
    }
}

/// A resolved context window, with the arithmetic that produced it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContextPlan {
    /// The window to open.
    pub resolved: u32,
    /// What `Auto` would have chosen, reported even when the caller named a
    /// number so the two can be compared in one startup line.
    pub suggested: u32,
    /// The checkpoint's trained context, or `None` when the install does not
    /// declare one (every install written before that field existed).
    pub trained: Option<u32>,
    /// KV bytes [`Self::resolved`] will allocate.
    pub kv_bytes: u64,
    /// True when [`Self::resolved`] is past what the checkpoint was trained
    /// on, which is a quality warning and never an error.
    pub past_trained: bool,
}

/// KV bytes `KvCacheManager::new` will allocate for `context`.
///
/// Mirrors that constructor exactly, including the parts that are easy to
/// miss. A linear or compressed layer keeps no KV at all and gets a
/// one-page placeholder, so it contributes nothing. A sliding-window layer
/// is a RING capped at `sliding_window + MAX_PREFILL_CHUNK_TOKENS`, so past
/// that cap it stops growing with the window -- which is why a model's KV
/// cost per token is decided by its FULL layers alone, and why Gemma 4 (5
/// full layers of 30) costs 20 KiB per token where a dense 7B costs 128.
pub fn kv_bytes_for_context(arch: &ArchConfig, context: u32) -> u64 {
    let context = context as u64;
    let swa_stride = arch.num_kv_heads as u64 * arch.head_dim as u64 * FP16_SIZE;
    let full_stride = arch.num_full_kv_heads as u64 * arch.full_head_dim as u64 * FP16_SIZE;
    let ring_capacity = (arch.sliding_window as u64 + MAX_PREFILL_CHUNK_TOKENS).max(1);

    // A compressed-attention install gives EVERY layer a placeholder, not
    // just its compressed ones, so this is checked once rather than per
    // layer -- the same shape `KvCacheManager::new`'s `all_placeholder` has.
    if arch.has_compressed_attention_layers() {
        return 0;
    }

    let mut bytes = 0u64;
    for &mask in &arch.full_attention_layer_mask {
        let (capacity, stride) = match mask {
            2 => continue,
            0 => (context.min(ring_capacity), swa_stride),
            _ => (context, full_stride),
        };
        // K and V, hence the doubling.
        bytes = bytes.saturating_add(capacity.saturating_mul(stride).saturating_mul(2));
    }
    bytes
}

/// The largest context whose KV fits `budget`, rounded down to a multiple of
/// [`CONTEXT_GRANULARITY`].
///
/// `kv_bytes_for_context` is monotone non-decreasing in the context, so this
/// is a bisection rather than a formula -- which keeps it correct without
/// restating the per-layer case analysis a second time and getting one of
/// the two copies wrong. Capped at [`MAX_SUPPORTED_CONTEXT`], because a
/// model whose layers are ALL sliding-window has a KV cost that stops
/// growing and would otherwise have no largest fitting context at all.
pub fn largest_context_within(arch: &ArchConfig, budget: u64) -> u32 {
    if kv_bytes_for_context(arch, CONTEXT_GRANULARITY) > budget {
        return 0;
    }
    let (mut lo, mut hi) = (CONTEXT_GRANULARITY, MAX_SUPPORTED_CONTEXT);
    if kv_bytes_for_context(arch, hi) <= budget {
        return hi;
    }
    // Invariant: `lo` fits and `hi` does not.
    while hi - lo > CONTEXT_GRANULARITY {
        let mid = lo + (hi - lo) / 2;
        let mid = mid - (mid % CONTEXT_GRANULARITY);
        if mid <= lo {
            break;
        }
        if kv_bytes_for_context(arch, mid) <= budget {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    lo
}

/// The step [`largest_context_within`] reports in. A KV allocation is
/// per-token, so any value is legal; a number ending in 1024 is legible
/// where the exact maximum (say 147,441) reads as noise.
pub const CONTEXT_GRANULARITY: u32 = 1024;

/// The ceiling on any resolved or suggested context.
///
/// Two jobs. It bounds the bisection above for a model whose KV stops
/// growing, and it stops `Auto` proposing a window no checkpoint here was
/// trained for when an install declares no trained context and the machine
/// has room to spare.
pub const MAX_SUPPORTED_CONTEXT: u32 = 1 << 20;

/// Resolve a context request against the checkpoint and the machine.
///
/// `trained` is the checkpoint's own context length, absent on any install
/// written before the manifest carried one. `committed` is what the install
/// already spends -- see [`committed_bytes`], which computes it.
///
/// **`Auto` with no declared trained context resolves to `default_context`,
/// never to what memory allows.** Silence means the install does not know,
/// and inventing a window from free RAM alone would take every install
/// already on disk from its documented 4,096 to whatever that machine
/// happens to have free -- a footprint change nobody asked for, on the
/// strength of no information at all. That is AGENTS.md Gotcha 39's rule: a
/// default is a claim about what silence MEANS.
pub fn resolve_max_context(
    request: MaxContext,
    arch: &ArchConfig,
    trained: Option<u32>,
    default_context: u32,
    physical: u64,
    committed: u64,
) -> Result<ContextPlan, ContextTooLarge> {
    // **`physical == 0` means the probe is unavailable, not that the machine
    // has no memory**, which is what `runtime::physical_memory` answers off
    // macOS. Reading it as an empty budget would refuse every explicit
    // window and resolve every `Auto` to a context of ZERO -- a plan that
    // admits no prompt at all, arrived at with no information. An unknown
    // machine therefore imposes no bound, exactly as an unknown trained
    // context imposes no ceiling.
    let known_machine = physical > 0;
    let available = physical.saturating_sub(committed.saturating_add(CONTEXT_RESERVE_BYTES));
    let comfortable = (available as f64 * CONTEXT_BUDGET_FRACTION) as u64;

    let ceiling = trained
        .unwrap_or(default_context)
        .min(MAX_SUPPORTED_CONTEXT);
    let suggested = if known_machine {
        largest_context_within(arch, comfortable).min(ceiling)
    } else {
        ceiling
    };

    let resolved = match request {
        MaxContext::Fixed(n) => n,
        MaxContext::Auto => suggested,
    };

    let kv_bytes = kv_bytes_for_context(arch, resolved);
    if known_machine && kv_bytes > available {
        return Err(ContextTooLarge {
            requested: resolved,
            needs: kv_bytes,
            available,
            physical,
            committed,
            largest_fitting: largest_context_within(arch, available),
        });
    }

    Ok(ContextPlan {
        resolved,
        suggested,
        trained,
        kv_bytes,
        past_trained: trained.is_some_and(|t| resolved > t),
    })
}

/// Bytes an install commits before any KV is allocated: the mapped resident
/// weight region, plus the LARGEST routed-expert slot cache
/// [`crate::ExpertCacheSlots::Auto`] could choose.
///
/// **The slot term is why this is not just the weight file's size.** On a
/// streamed MoE install the two are far apart: Gemma 4's `model_weights.bin`
/// is 1.26 GiB while its expert table is 12 GB, of which the cache pins
/// `slots x sum(expert_stride)` -- about 3.0 GiB at the top of
/// `ALLOWED_CACHE_SLOTS`. Counting the mapped file alone lets an explicit
/// `--max-context` claim memory the slot cache is about to take, and the two
/// policies then both spend it. On a DENSE install there is no layout file
/// and the term is zero, which is correct: nothing streams.
///
/// The WORST case rather than the resolved count, deliberately. The slot
/// policy resolves inside `RealForwardRunner::open`, after this has already
/// decided whether the window fits, so the only safe assumption is that
/// `Auto` climbs as far as it can.
///
/// **The one function in this module that touches the filesystem.** It reads
/// the INSTALL rather than the machine, and its result is a parameter to
/// [`resolve_max_context`], so every case in this module's tests is still a
/// pure computation with no directory on disk.
pub fn committed_bytes(model_dir: &Path) -> u64 {
    let resident = std::fs::metadata(model_dir.join("model_weights.bin"))
        .map(|m| m.len())
        .unwrap_or(0);
    let max_slots = foundation::runtime_config::ALLOWED_CACHE_SLOTS
        .iter()
        .copied()
        .max()
        .unwrap_or(0) as u64;
    // A missing or unreadable layout means a dense install, which has no
    // routed experts and therefore no slot cache. Reading the PER-LAYER
    // strides and not `layers * max(stride)`, for the reason
    // `expert_cache_policy` sums them too: a mixed sub-4-bit install has one
    // layer 1.6x its siblings, and the model-wide maximum over-states a
    // slot's cost by 35% (`model-io` Gotcha 2).
    let slot_cache = crate::load_packed_experts_layout(
        model_dir,
        crate::PACKED_EXPERTS_LAYOUT_DEFAULT_MAX_BYTES,
    )
    .map(|layout| {
        layout
            .layers
            .iter()
            .map(|l| l.expert_stride)
            .sum::<u64>()
            .saturating_mul(max_slots)
    })
    .unwrap_or(0);
    resident.saturating_add(slot_cache)
}

/// Bytes as GiB to one decimal, for the messages above.
fn gib(bytes: u64) -> String {
    format!("{:.1} GiB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
}

#[cfg(test)]
#[path = "context_policy_tests.rs"]
mod tests;
