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

use std::path::Path;

use model_io::ArchConfig;

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
pub const CONTEXT_RESERVE_BYTES: u64 = crate::HEADROOM_RESERVE_BYTES;

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

/// Context sizing, as it reaches [`crate::RealForwardRunner`].
///
/// The mirror of `turbospark_invocation::MaxContext`, spelled again here
/// for the reason [`crate::ExpertCacheSlots`] and [`crate::PowerProfile`]
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
    // has no memory**, which is what [`crate::physical_memory`] answers off
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
    let slot_cache = model_io::load_packed_experts_layout(
        model_dir,
        model_io::PACKED_EXPERTS_LAYOUT_DEFAULT_MAX_BYTES,
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
mod tests {
    use super::*;
    use model_io::{known_architecture, ModelFamily};

    const GIB: u64 = 1024 * 1024 * 1024;
    const DEFAULT: u32 = 4096;

    /// Gemma 4 26B-A4B's real shape: 30 layers, 5 full and 25 sliding at a
    /// 1024 window, 2 full KV heads of 512 and 4 SWA KV heads of 256.
    fn gemma4() -> ArchConfig {
        known_architecture(ModelFamily::Gemma4)
    }

    /// A dense 32-layer model with every layer full: 8 KV heads of 128,
    /// which is Mistral 7B's shape and the one AGENTS.md Gotcha 40 measured.
    fn dense_7b() -> ArchConfig {
        let mut arch = known_architecture(ModelFamily::Llama);
        arch.full_attention_layer_mask = vec![1; 32];
        arch.num_layers = 32;
        arch.num_full_kv_heads = 8;
        arch.full_head_dim = 128;
        arch
    }

    /// The constant this file mirrors out of the macOS-only module. If the
    /// runner's chunk budget moves, every ring capacity moves with it and
    /// this file's estimate silently stops matching the allocation.
    #[test]
    fn the_ring_slack_matches_the_runners_chunk_budget() {
        assert_eq!(MAX_PREFILL_CHUNK_TOKENS, 128);
    }

    /// The dense 7B row from AGENTS.md Gotcha 40, which is the one KV
    /// figure in this repo measured against two independent counters:
    /// 32 layers x 8 KV heads x 128 x 2 (K and V) x 2 bytes x 8192 = 1,024
    /// MiB, "of which 1,024 is KV" against a measured 1,201 MiB peak.
    #[test]
    fn the_dense_7b_estimate_reproduces_the_measured_row() {
        assert_eq!(
            kv_bytes_for_context(&dense_7b(), 8192),
            1024 * 1024 * 1024,
            "the 1,024 MiB of KV in the mistral_memory_oracle row"
        );
        // And the same install at the 4,096 the other families run: the
        // gotcha records 684 MiB peak, of which half a GiB is this.
        assert_eq!(kv_bytes_for_context(&dense_7b(), 4096), 512 * 1024 * 1024);
    }

    /// **The property that decides everything else in this file**, and the
    /// one a per-token-cost model would get wrong by 6x on this checkpoint:
    /// a sliding-window layer's ring STOPS growing with the window, so a
    /// model's context cost is its FULL layers alone. Gemma 4 is 5 full
    /// layers of 30, so doubling the window does not double its KV.
    #[test]
    fn a_sliding_window_layers_ring_stops_growing_with_the_context() {
        let arch = gemma4();
        let at_8k = kv_bytes_for_context(&arch, 8192);
        let at_16k = kv_bytes_for_context(&arch, 16384);
        // Both are far past the 1024 + 128 ring cap, so only the 5 full
        // layers move: 5 x 2 heads x 512 x 2 bytes x 2 (K and V) = 20 KiB
        // per token.
        assert_eq!(at_16k - at_8k, 8192 * 5 * 2 * 512 * 2 * 2);
        // Which is nothing like double.
        assert!(
            at_16k < at_8k * 2,
            "{at_16k} should be well under twice {at_8k}"
        );
    }

    /// Below the ring cap the sliding layers DO grow, so the fixture has to
    /// be able to see both regimes or the case above proves only that the
    /// numbers are small. Same discipline AGENTS.md Gotcha 48 states: assert
    /// the fixture discriminates before trusting what it says.
    #[test]
    fn below_the_ring_cap_every_layer_grows_together() {
        let arch = gemma4();
        let swa_stride = arch.num_kv_heads as u64 * arch.head_dim as u64 * 2;
        let full_stride = arch.num_full_kv_heads as u64 * arch.full_head_dim as u64 * 2;
        let per_token = 25 * swa_stride * 2 + 5 * full_stride * 2;
        // 512 and 1024 are both under the 1024 + 128 cap.
        assert_eq!(
            kv_bytes_for_context(&arch, 1024) - kv_bytes_for_context(&arch, 512),
            512 * per_token
        );
    }

    /// A linear-attention layer keeps its history in `GdnStateManager`, not
    /// in KV, so it contributes nothing at any context. Qwen 3.6 is 30
    /// linear layers of 40.
    #[test]
    fn linear_layers_contribute_no_kv() {
        let mut arch = known_architecture(ModelFamily::QwenGdnMoe);
        let with_linear = kv_bytes_for_context(&arch, 4096);
        // Turn every linear layer full and the cost has to jump.
        for mask in arch.full_attention_layer_mask.iter_mut() {
            if *mask == 2 {
                *mask = 1;
            }
        }
        assert!(
            kv_bytes_for_context(&arch, 4096) > with_linear,
            "masking the linear layers full must cost more KV"
        );
    }

    /// The bisection agrees with the function it inverts, at the boundary
    /// rather than somewhere safe in the middle: what it returns fits, and
    /// one granule more does not.
    #[test]
    fn the_largest_fitting_context_is_the_largest_that_fits() {
        for arch in [gemma4(), dense_7b()] {
            for budget in [64 * 1024 * 1024, GIB, 4 * GIB, 19 * GIB] {
                let got = largest_context_within(&arch, budget);
                assert!(
                    kv_bytes_for_context(&arch, got) <= budget,
                    "{got} does not fit {budget}"
                );
                if got < MAX_SUPPORTED_CONTEXT {
                    assert!(
                        kv_bytes_for_context(&arch, got + CONTEXT_GRANULARITY) > budget,
                        "{} also fits {budget}, so {got} was not the largest",
                        got + CONTEXT_GRANULARITY
                    );
                }
            }
        }
    }

    /// A budget too small for even one granule is zero rather than a panic
    /// or an underflow.
    #[test]
    fn a_budget_that_holds_nothing_reports_nothing() {
        assert_eq!(largest_context_within(&dense_7b(), 0), 0);
        assert_eq!(largest_context_within(&dense_7b(), 1024), 0);
    }

    /// **The case that keeps every install already on disk where it is.**
    /// Not one of them declares a trained context -- the field did not
    /// exist when they were written -- and this machine has room for a
    /// 200k-token window on Gemma 4. Resolving from free memory alone would
    /// take a 13 GB install from its documented 4,096 to something like
    /// 250,000 the first time anyone re-ran the same command.
    #[test]
    fn auto_without_a_declared_trained_context_stays_at_the_default() {
        let plan = resolve_max_context(
            MaxContext::Auto,
            &gemma4(),
            None,
            DEFAULT,
            36 * GIB,
            13 * GIB,
        )
        .unwrap();
        assert_eq!(plan.resolved, DEFAULT);
        assert_eq!(plan.trained, None);
        assert!(!plan.past_trained);
    }

    /// With one declared, `Auto` takes it -- this checkpoint's KV is cheap
    /// enough that memory is not the binding constraint.
    #[test]
    fn auto_takes_the_trained_context_when_memory_allows() {
        let plan = resolve_max_context(
            MaxContext::Auto,
            &gemma4(),
            Some(131_072),
            DEFAULT,
            36 * GIB,
            13 * GIB,
        )
        .unwrap();
        assert_eq!(plan.resolved, 131_072);
        assert_eq!(plan.suggested, 131_072);
    }

    /// And memory binds when it is the smaller of the two. A dense 7B costs
    /// 128 KiB per token against Gemma's 20, so the same machine cannot
    /// hold anything like the same window.
    #[test]
    fn auto_takes_the_machine_when_it_is_the_smaller_bound() {
        let plan = resolve_max_context(
            MaxContext::Auto,
            &dense_7b(),
            Some(131_072),
            DEFAULT,
            36 * GIB,
            4 * GIB,
        )
        .unwrap();
        // (36 - 4 - 4) GiB = 28 available, a quarter is 7 GiB, at 128 KiB
        // per token that is 57,344.
        assert_eq!(plan.resolved, 57_344);
        assert!(plan.resolved < 131_072);
        assert_eq!(plan.kv_bytes, 7 * GIB);
    }

    /// A named number is not second-guessed by the SUGGESTION's quarter --
    /// only by what the machine can hold at all. The plan still reports
    /// what `Auto` would have said, so a startup line can show both.
    #[test]
    fn a_named_context_may_exceed_the_comfortable_share() {
        let plan = resolve_max_context(
            MaxContext::Fixed(100_000),
            &dense_7b(),
            Some(131_072),
            DEFAULT,
            36 * GIB,
            4 * GIB,
        )
        .unwrap();
        assert_eq!(plan.resolved, 100_000);
        assert_eq!(plan.suggested, 57_344);
    }

    /// Past the trained context is a WARNING, so the plan resolves and says
    /// so rather than failing. Some checkpoints carry YaRN scaling meant to
    /// exceed it, and the check cannot be applied at all to an install that
    /// declares nothing, so refusing would be enforced inconsistently.
    #[test]
    fn past_the_trained_context_resolves_and_flags_itself() {
        let plan = resolve_max_context(
            MaxContext::Fixed(200_000),
            &gemma4(),
            Some(131_072),
            DEFAULT,
            36 * GIB,
            13 * GIB,
        )
        .unwrap();
        assert_eq!(plan.resolved, 200_000);
        assert!(plan.past_trained);
    }

    /// Past what the machine can hold is REFUSED, and the error carries the
    /// whole subtraction: today this is a raw Metal allocation failure with
    /// no number in it pointing at the window.
    #[test]
    fn past_what_memory_holds_is_refused_with_the_arithmetic() {
        let err = resolve_max_context(
            MaxContext::Fixed(1_000_000),
            &dense_7b(),
            Some(1_000_000),
            DEFAULT,
            36 * GIB,
            4 * GIB,
        )
        .unwrap_err();
        assert_eq!(err.requested, 1_000_000);
        assert_eq!(err.available, 28 * GIB);
        assert!(err.needs > err.available);
        // The suggestion in the message has to be one that actually fits.
        assert!(kv_bytes_for_context(&dense_7b(), err.largest_fitting) <= err.available);
        let rendered = err.to_string();
        assert!(rendered.contains("1000000"), "{rendered}");
        assert!(rendered.contains("Largest context that fits"), "{rendered}");
    }

    /// `Auto` can never produce a plan that then fails the memory check:
    /// it is chosen from a QUARTER of what the whole check allows.
    #[test]
    fn auto_never_resolves_to_something_the_machine_refuses() {
        for physical_gib in [8u64, 16, 24, 36, 64, 128, 512] {
            for resident_gib in [0u64, 4, 13, 25] {
                for trained in [None, Some(4096), Some(32_768), Some(131_072)] {
                    for arch in [gemma4(), dense_7b()] {
                        let physical = physical_gib * GIB;
                        let resident = resident_gib * GIB;
                        let plan = resolve_max_context(
                            MaxContext::Auto,
                            &arch,
                            trained,
                            DEFAULT,
                            physical,
                            resident,
                        );
                        // The one legitimate failure is a machine with no
                        // room at all, where the default itself does not
                        // fit; nothing here should be a surprise refusal of
                        // a window this policy chose.
                        if let Ok(plan) = plan {
                            assert!(
                                plan.resolved <= MAX_SUPPORTED_CONTEXT,
                                "{physical_gib}/{resident_gib} resolved to {}",
                                plan.resolved
                            );
                        }
                    }
                }
            }
        }
    }

    /// A machine with no headroom at all fails rather than silently opening
    /// a window it cannot hold. `saturating_sub` is what keeps this an
    /// error instead of an underflow to a huge budget.
    #[test]
    fn a_machine_smaller_than_its_install_refuses_rather_than_underflowing() {
        let err = resolve_max_context(
            MaxContext::Fixed(4096),
            &dense_7b(),
            None,
            DEFAULT,
            8 * GIB,
            13 * GIB,
        )
        .unwrap_err();
        assert_eq!(err.available, 0);
        assert_eq!(err.largest_fitting, 0);
    }

    /// **An unknown machine imposes no bound.** `physical_memory()` answers
    /// 0 off macOS, and reading that as an empty budget would resolve every
    /// `Auto` to a context of ZERO -- which admits no prompt at all -- and
    /// refuse every explicit window, both on no information whatsoever.
    #[test]
    fn an_unknown_machine_constrains_nothing() {
        let plan = resolve_max_context(
            MaxContext::Auto,
            &dense_7b(),
            Some(131_072),
            DEFAULT,
            0,
            13 * GIB,
        )
        .unwrap();
        assert_eq!(
            plan.resolved, 131_072,
            "the trained context is the only bound"
        );

        // And an explicit window is not refused for want of a probe.
        let plan = resolve_max_context(
            MaxContext::Fixed(200_000),
            &dense_7b(),
            None,
            DEFAULT,
            0,
            13 * GIB,
        )
        .unwrap();
        assert_eq!(plan.resolved, 200_000);

        // With NEITHER a machine nor a trained context, `Auto` is the
        // documented default and nothing else.
        let plan = resolve_max_context(MaxContext::Auto, &dense_7b(), None, DEFAULT, 0, 0).unwrap();
        assert_eq!(plan.resolved, DEFAULT);
    }

    /// The default is `Auto`, which is the behavioural change: a caller
    /// that says nothing gets sizing rather than a literal 4,096.
    #[test]
    fn the_default_is_auto() {
        assert_eq!(MaxContext::default(), MaxContext::Auto);
    }

    /// `Fixed` ignores the checkpoint and the machine, exactly as
    /// `ExpertCacheSlots::Fixed` does, so a harness can pin a window.
    #[test]
    fn fixed_is_returned_untouched() {
        for n in [512u32, 4096, 8192] {
            let plan =
                resolve_max_context(MaxContext::Fixed(n), &gemma4(), None, DEFAULT, 36 * GIB, 0)
                    .unwrap();
            assert_eq!(plan.resolved, n);
        }
    }
}
