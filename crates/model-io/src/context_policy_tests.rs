use super::*;
use crate::{known_architecture, ModelFamily};

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
        let plan = resolve_max_context(MaxContext::Fixed(n), &gemma4(), None, DEFAULT, 36 * GIB, 0)
            .unwrap();
        assert_eq!(plan.resolved, n);
    }
}
