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

use crate::{ArchConfig, LoadPolicy};

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
    /// What the guard held back, which is the term the user can move by
    /// changing tiers. Carried rather than read off [`CONTEXT_RESERVE_BYTES`]
    /// at format time, because that constant is only one tier's answer and a
    /// message quoting it under `strict` would name a number that did not
    /// produce this refusal.
    pub reserve: u64,
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
            gib(self.reserve),
            self.largest_fitting,
        )
    }
}

/// Why an AUTOMATIC context resolution was refused for landing below the
/// caller's floor.
///
/// A sibling of [`ContextTooLarge`] rather than a variant of it, because the
/// two are different kinds of failure and the fix differs: that one says the
/// machine cannot hold what was asked for, this one says the machine (or the
/// checkpoint) cannot offer what was required. [`Self::capped_by`] is what
/// tells them apart, and it is the whole value of the message -- lowering a
/// floor, changing a guard tier and picking a different checkpoint are three
/// different actions and only one of them helps.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContextFloorUnmet {
    /// The minimum the caller required.
    pub floor: u32,
    /// What `Auto` actually resolved to.
    pub resolved: u32,
    /// Which bound produced [`Self::resolved`].
    pub capped_by: ContextCap,
    /// The largest window this machine affords at all, ignoring the
    /// comfortable fraction. Reported because it is the number that says
    /// whether a looser GUARD would clear the floor, which
    /// [`Self::resolved`] alone cannot.
    pub largest_fitting: u32,
}

/// Which of the three bounds on `Auto` was the binding one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContextCap {
    /// The checkpoint's own trained context. No guard tier helps; only a
    /// different checkpoint does.
    Trained,
    /// What a quarter (or the tier's fraction) of the pool affords. A looser
    /// guard, or freeing memory, raises this.
    Memory,
    /// [`DEFAULT_MAX_CONTEXT`]'s stand-in for an install that declares no
    /// trained context. Naming an explicit `--max-context` is the answer.
    ///
    /// [`DEFAULT_MAX_CONTEXT`]: foundation::runtime_config::DEFAULT_MAX_CONTEXT
    Undeclared,
}

impl ContextCap {
    /// What to change, in the imperative. Kept beside the enum so the three
    /// arms cannot drift from the three diagnoses above.
    fn remedy(self) -> &'static str {
        match self {
            Self::Trained => {
                "the checkpoint was not trained past this; a different checkpoint \
                              or an explicit --max-context is the only way up"
            }
            Self::Memory => {
                "a looser --load-guard, fewer expert-cache slots, or a smaller \
                             install would raise it"
            }
            Self::Undeclared => {
                "this install declares no trained context, so `auto` falls back \
                                 to the default window; name an explicit --max-context"
            }
        }
    }
}

impl std::fmt::Display for ContextFloorUnmet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "--min-auto-context {} was not met: `auto` resolved {} \
             (largest this machine affords: {}). {}",
            self.floor,
            self.resolved,
            self.largest_fitting,
            self.capped_by.remedy(),
        )
    }
}

/// Either reason [`resolve_max_context`] declines.
///
/// One `Display` per arm and one delegating `Display` here, so the three call
/// sites keep their existing `.map_err(|e| e.to_string())` unchanged.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContextRefused {
    /// The window does not fit the machine.
    TooLarge(ContextTooLarge),
    /// `Auto` landed below the caller's floor.
    FloorUnmet(ContextFloorUnmet),
}

impl std::fmt::Display for ContextRefused {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TooLarge(e) => e.fmt(f),
            Self::FloorUnmet(e) => e.fmt(f),
        }
    }
}

impl std::error::Error for ContextRefused {}

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

/// Bytes `GdnStateManager::new` will allocate for one session's recurrent
/// state, on installs that carry gated-DeltaNet linear-attention layers.
///
/// Mirrors that constructor exactly, for `kv_bytes_for_context`'s own
/// reason: this crate cannot depend on `crates/gpu`, so a second copy of the
/// arithmetic is a second place to get it wrong. Unlike KV, this state is
/// FIXED-size regardless of context length (`crates/gpu/CLAUDE.md`'s
/// `gdn_state.rs` entry): a delta-rule state `S` (FP32) plus a causal-conv
/// tail (FP16) per linear layer, and zero for every family without one.
pub fn gdn_state_bytes(arch: &ArchConfig) -> u64 {
    let la = &arch.linear_attention;
    let state_bytes =
        la.num_v_heads as u64 * la.value_head_dim as u64 * la.key_head_dim as u64 * FP32_SIZE;
    let conv_tail_bytes = (la.conv_kernel_size.max(1) - 1) as u64 * la.qkv_dim() as u64 * FP16_SIZE;
    let linear_layers = arch
        .full_attention_layer_mask
        .iter()
        .filter(|&&mask| mask == 2)
        .count() as u64;
    linear_layers.saturating_mul(state_bytes.saturating_add(conv_tail_bytes))
}

/// Bytes per FP32 element, mirroring `GdnStateManager::new`'s state buffer.
const FP32_SIZE: u64 = 4;

/// Extra bytes `--session-slots N` commits beyond the ONE live session
/// [`resolve_max_context`] already sizes.
///
/// The pool holds `session_slots - 1` PARKED slots (the Nth is always the
/// runner's own live `kv`/`gdn` fields, sized by the existing
/// `resolve_max_context` call), each an independent `KvCacheManager` plus,
/// on a GDN family, an independent `GdnStateManager`. `session_slots <= 1`
/// therefore commits nothing extra, which is what keeps the default
/// (`--session-slots 1`) a true no-op on this budget.
pub fn session_pool_bytes(arch: &ArchConfig, context: u32, session_slots: u32) -> u64 {
    let extra_slots = u64::from(session_slots.saturating_sub(1));
    extra_slots
        .saturating_mul(kv_bytes_for_context(arch, context).saturating_add(gdn_state_bytes(arch)))
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
/// **`policy` CARRIES THE GUARD TIER AND THE FLOOR, AND `LoadPolicy::default()`
/// REPRODUCES THIS FUNCTION AS IT BEHAVED BEFORE EITHER EXISTED.** That is
/// asserted rather than claimed: every case in `context_policy_tests.rs`
/// passes the default and is otherwise unchanged, so the file staying green
/// IS the no-behaviour-change proof (`load_guard`'s own header explains what
/// a moved default would quietly invalidate).
///
/// The two knobs act on different arms. The GUARD moves the reserve and the
/// comfortable fraction, so it changes what `Auto` suggests and what an
/// explicit window is refused against; [`LoadGuard::Off`] additionally
/// declines to refuse at all. The FLOOR acts on `Auto` alone, for the reason
/// [`ContextFloorUnmet`] gives.
///
/// [`LoadGuard::Off`]: crate::LoadGuard::Off
pub fn resolve_max_context(
    request: MaxContext,
    arch: &ArchConfig,
    trained: Option<u32>,
    default_context: u32,
    physical: u64,
    committed: u64,
    policy: &LoadPolicy,
) -> Result<ContextPlan, ContextRefused> {
    // **`physical == 0` means the probe is unavailable, not that the machine
    // has no memory**, which is what `runtime::physical_memory` answers off
    // macOS. Reading it as an empty budget would refuse every explicit
    // window and resolve every `Auto` to a context of ZERO -- a plan that
    // admits no prompt at all, arrived at with no information. An unknown
    // machine therefore imposes no bound, exactly as an unknown trained
    // context imposes no ceiling.
    let known_machine = physical > 0;
    let budget = policy.guard.budget();
    let available = policy.guard.available(physical, committed);
    let comfortable = (available as f64 * budget.budget_fraction) as u64;

    let ceiling = trained
        .unwrap_or(default_context)
        .min(MAX_SUPPORTED_CONTEXT);
    let by_memory = largest_context_within(arch, comfortable);
    let suggested = if known_machine {
        by_memory.min(ceiling)
    } else {
        ceiling
    };

    let resolved = match request {
        MaxContext::Fixed(n) => n,
        MaxContext::Auto => suggested,
    };

    let kv_bytes = kv_bytes_for_context(arch, resolved);
    // `refuses` is false on `Off` alone. A budget without a refusal is the
    // whole content of that tier: the arithmetic still runs and still sizes
    // `Auto`, and a caller who names a window too large for the machine gets
    // the allocation failure they asked for rather than a message.
    if known_machine && budget.refuses && kv_bytes > available {
        return Err(ContextRefused::TooLarge(ContextTooLarge {
            requested: resolved,
            needs: kv_bytes,
            available,
            reserve: budget.reserve_bytes,
            physical,
            committed,
            largest_fitting: largest_context_within(arch, available),
        }));
    }

    // The floor applies to `Auto` ALONE, and it applies under every tier
    // including `Off`: it is not a memory precaution the guard could relax,
    // it is the caller stating a requirement of their own workload. A
    // `Fixed` request has already named its number and is not asking to be
    // told it is small.
    if request == MaxContext::Auto && policy.min_auto_context > resolved {
        let capped_by = if known_machine && by_memory < ceiling {
            ContextCap::Memory
        } else if trained.is_some() {
            ContextCap::Trained
        } else {
            ContextCap::Undeclared
        };
        return Err(ContextRefused::FloorUnmet(ContextFloorUnmet {
            floor: policy.min_auto_context,
            resolved,
            capped_by,
            largest_fitting: if known_machine {
                largest_context_within(arch, available)
            } else {
                MAX_SUPPORTED_CONTEXT
            },
        }));
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
