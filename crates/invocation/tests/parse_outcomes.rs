//! Accepted-outcome scenarios for the argument-translation contract.
//!
//! Corresponds to behavior-spec test-001, test-002, test-004, test-010,
//! test-012, test-014, test-016, and test-017.

use foundation::runtime_config::{ALLOWED_CACHE_SLOTS, ALLOWED_CHUNK_SIZES, DEFAULT_CHUNK_SIZE};
use turbospark_invocation::{
    parse, ExpertCacheSlots, ExpertResidency, InvocationRequest, KvBits, MaxContext, Mode,
    ParseOutcome, PowerProfile, PrefillChunk, ReasoningEffort, Speculation,
};

fn tok(items: &[&str]) -> Vec<String> {
    items.iter().map(|s| s.to_string()).collect()
}

fn expect_success(outcome: ParseOutcome) -> InvocationRequest {
    match outcome {
        ParseOutcome::Success(req) => req,
        other => panic!("expected success, got {other:?}"),
    }
}

#[test]
fn minimal_prompt_invocation_uses_documented_defaults() {
    let req = expect_success(parse(&tok(&["--model", "m.bin", "--prompt", "hello"])));
    assert_eq!(req.model, "m.bin");
    assert_eq!(req.mode, Mode::Prompt("hello".to_string()));
    assert_eq!(req.max_new, 1024);
    assert_eq!(req.top_k, 64);
}

#[test]
fn explicit_generation_and_sampling_values_round_trip() {
    let req = expect_success(parse(&tok(&[
        "--model",
        "m.bin",
        "--prompt",
        "hello",
        "--max-new",
        "50",
        "--max-context",
        "2048",
        "--temperature",
        "0.7",
        "--top-k",
        "10",
        "--top-p",
        "1.0",
        "--repetition-penalty",
        "1.2",
        "--seed",
        "42",
        "--stop",
        "a",
        "--stop",
        "b",
        "--quiet",
    ])));
    assert_eq!(req.max_new, 50);
    assert_eq!(req.max_context, MaxContext::Fixed(2048));
    assert_eq!(req.temperature, 0.7);
    assert_eq!(req.top_k, 10);
    assert_eq!(req.top_p, 1.0);
    assert_eq!(req.repetition_penalty, 1.2);
    assert_eq!(req.seed, Some(42));
    assert_eq!(req.stop, vec!["a".to_string(), "b".to_string()]);
    assert!(req.quiet);
}

#[test]
fn top_k_disabled_reports_the_dedicated_disabled_state() {
    let req = expect_success(parse(&tok(&[
        "--model", "m", "--prompt", "hi", "--top-k", "0",
    ])));
    assert_eq!(req.top_k, 0);
}

#[test]
fn messages_file_mode_leaves_prompt_unset() {
    let req = expect_success(parse(&tok(&[
        "--model",
        "m",
        "--messages-file",
        "chat.json",
    ])));
    assert_eq!(req.mode, Mode::MessagesFile("chat.json".to_string()));
}

#[test]
fn chat_mode_selects_interactive_with_no_prompt_or_file() {
    let req = expect_success(parse(&tok(&["--model", "m", "--chat"])));
    assert_eq!(req.mode, Mode::Chat);
}

#[test]
fn repeated_system_messages_combine_in_supplied_order() {
    let req = expect_success(parse(&tok(&[
        "--model", "m", "--chat", "--system", "first", "--system", "second",
    ])));
    let combined = req.system.expect("system message expected");
    assert!(combined.find("first").unwrap() < combined.find("second").unwrap());
}

#[test]
fn every_documented_chunk_size_is_accepted() {
    for &size in ALLOWED_CHUNK_SIZES.iter() {
        let req = expect_success(parse(&tok(&[
            "--model",
            "m",
            "--chat",
            "--prefill-chunk",
            &size.to_string(),
        ])));
        assert_eq!(req.prefill_chunk, PrefillChunk::Fixed(size));
    }
}

#[test]
fn automatic_chunk_sizing_keyword_selects_auto() {
    let req = expect_success(parse(&tok(&[
        "--model",
        "m",
        "--chat",
        "--prefill-chunk",
        "auto",
    ])));
    assert_eq!(req.prefill_chunk, PrefillChunk::Auto);
    assert_eq!(req.prefill_chunk.resolved(), DEFAULT_CHUNK_SIZE);
}

/// The slot flag now takes the same `auto`-or-a-member grammar
/// `--prefill-chunk` does, and unlike that one `auto` is also the DEFAULT
/// (`request_defaults.rs`). Every allowed count must still round-trip, since
/// pinning one is how every harness in the repo keeps its frozen rows
/// comparable.
#[test]
fn every_documented_slot_count_is_accepted_and_auto_is_a_value() {
    for &slots in ALLOWED_CACHE_SLOTS.iter() {
        let req = expect_success(parse(&tok(&[
            "--model",
            "m",
            "--chat",
            "--expert-cache-slots",
            &slots.to_string(),
        ])));
        assert_eq!(req.expert_cache_slots, ExpertCacheSlots::Fixed(slots));
    }
    let req = expect_success(parse(&tok(&[
        "--model",
        "m",
        "--chat",
        "--expert-cache-slots",
        "auto",
    ])));
    assert_eq!(req.expert_cache_slots, ExpertCacheSlots::Auto);
}

#[test]
fn expert_residency_round_trips_and_defaults_to_auto() {
    let default = expect_success(parse(&tok(&["--model", "m", "--prompt", "hi"]))).expert_residency;
    assert_eq!(default, ExpertResidency::Auto);
    for (spelling, expected) in [
        ("auto", ExpertResidency::Auto),
        ("streamed", ExpertResidency::Streamed),
        ("mapped", ExpertResidency::Mapped),
    ] {
        let req = expect_success(parse(&tok(&[
            "--model",
            "m",
            "--prompt",
            "hi",
            "--expert-residency",
            spelling,
        ])));
        assert_eq!(req.expert_residency, expected);
    }
}

/// The THIRD flag on the `auto`-or-a-value grammar, and the second whose
/// `auto` is the default. Unlike the slot count, this one takes an arbitrary
/// positive integer rather than a member of a published set: a context window
/// is a per-token allocation, so there is no allowed set to check against and
/// the only real bound is what memory holds -- which this pure crate cannot
/// look at, and `crates/runtime`'s policy checks instead.
#[test]
fn the_context_window_takes_a_count_or_the_auto_keyword() {
    for tokens in [4096u32, 1, 131_072] {
        let req = expect_success(parse(&tok(&[
            "--model",
            "m",
            "--chat",
            "--max-context",
            &tokens.to_string(),
        ])));
        assert_eq!(req.max_context, MaxContext::Fixed(tokens));
    }
    let req = expect_success(parse(&tok(&[
        "--model",
        "m",
        "--chat",
        "--max-context",
        "auto",
    ])));
    assert_eq!(req.max_context, MaxContext::Auto);
}

/// Zero is refused rather than treated as `auto`. A window of zero admits no
/// prompt at all, and the flag already has a spelling for "you decide".
#[test]
fn a_zero_or_unparsable_context_window_is_refused() {
    for bad in ["0", "-1", "many", "4096tokens", ""] {
        let outcome = parse(&tok(&["--model", "m", "--chat", "--max-context", bad]));
        assert!(
            matches!(outcome, ParseOutcome::Failure(_)),
            "--max-context {bad:?} should be refused"
        );
    }
}

/// Every documented level round-trips, and the default is the one that
/// changes no rendered byte.
///
/// The UNION is accepted here on purpose, `high` and `xhigh` both, even
/// though no single checkpoint takes both: this crate may not look at the
/// install, and the template that can validate says so by name. A typo is
/// still rejected, which is the line between "not this crate's question" and
/// "no question at all".
#[test]
fn reasoning_levels_translate_to_their_validated_values() {
    let req = expect_success(parse(&tok(&["--model", "m", "--chat"])));
    assert_eq!(
        req.reasoning,
        ReasoningEffort::Off,
        "the default must be the level that renders what every earlier release rendered"
    );

    for spelling in ["off", "low", "medium", "high", "xhigh"] {
        let req = expect_success(parse(&tok(&[
            "--model",
            "m",
            "--chat",
            "--reasoning",
            spelling,
        ])));
        assert_eq!(
            req.reasoning.as_str(),
            spelling,
            "reasoning spelling must round-trip"
        );
    }

    assert!(matches!(
        parse(&tok(&["--model", "m", "--chat", "--reasoning", "maximum"])),
        ParseOutcome::Failure(_)
    ));
    // `true` reads like a plausible spelling for a knob that used to be a
    // boolean everywhere upstream, and is not one here.
    assert!(matches!(
        parse(&tok(&["--model", "m", "--chat", "--reasoning", "true"])),
        ParseOutcome::Failure(_)
    ));
}

/// Every documented `--kv-bits` spelling round-trips into the right
/// key/value width pair, and an out-of-set spelling is a typed failure
/// rather than a silent fallback to `Off` -- a caller asking for a width
/// this crate does not recognize should not measure the unquantized engine
/// and believe it asked for something else.
#[test]
fn kv_bits_spellings_translate_to_their_validated_widths() {
    let req = expect_success(parse(&tok(&["--model", "m", "--chat"])));
    assert_eq!(
        req.kv_bits,
        KvBits::Off,
        "the default must be the value that renders what every earlier release rendered"
    );

    for (spelling, expected) in [
        ("off", KvBits::Off),
        (
            "2",
            KvBits::TurboQuant {
                k_bits: 2,
                v_bits: 2,
            },
        ),
        (
            "3",
            KvBits::TurboQuant {
                k_bits: 3,
                v_bits: 3,
            },
        ),
        (
            "3.5",
            KvBits::TurboQuant {
                k_bits: 3,
                v_bits: 4,
            },
        ),
        (
            "4",
            KvBits::TurboQuant {
                k_bits: 4,
                v_bits: 4,
            },
        ),
    ] {
        let req = expect_success(parse(&tok(&[
            "--model",
            "m",
            "--chat",
            "--kv-bits",
            spelling,
        ])));
        assert_eq!(
            req.kv_bits, expected,
            "spelling {spelling} must map to {expected:?}"
        );
    }

    assert!(matches!(
        parse(&tok(&["--model", "m", "--chat", "--kv-bits", "5"])),
        ParseOutcome::Failure(_)
    ));
    // `2.5` is not one of mlx-vlm's fractional widths -- only `3.5` (K3/V4)
    // is defined -- so it must not silently floor or round to a neighbor.
    assert!(matches!(
        parse(&tok(&["--model", "m", "--chat", "--kv-bits", "2.5"])),
        ParseOutcome::Failure(_)
    ));
}

#[test]
fn power_profile_and_rate_cap_translate_to_their_validated_values() {
    let req = expect_success(parse(&tok(&[
        "--model",
        "m",
        "--chat",
        "--power-profile",
        "efficiency",
        "--max-tokens-per-sec",
        "8.5",
    ])));
    assert_eq!(req.power_profile, Some(PowerProfile::Efficiency));
    assert_eq!(req.max_tokens_per_sec, Some(8.5));

    // The two are independent: a cap without a profile is the documented
    // way to pace without changing thermal behavior.
    let rate_only = expect_success(parse(&tok(&[
        "--model",
        "m",
        "--chat",
        "--max-tokens-per-sec",
        "3",
    ])));
    assert_eq!(rate_only.power_profile, None);
    assert_eq!(rate_only.max_tokens_per_sec, Some(3.0));

    for spelling in ["performance", "balanced", "efficiency"] {
        let req = expect_success(parse(&tok(&[
            "--model",
            "m",
            "--chat",
            "--power-profile",
            spelling,
        ])));
        assert_eq!(
            req.power_profile.map(PowerProfile::as_str),
            Some(spelling),
            "profile spelling must round-trip"
        );
    }
}

/// `--speculative`, and the three values are three different ANSWERS rather
/// than a spectrum: see `Speculation`'s doc comment. What this pins is the
/// spelling and the default; what a named block does when the model cannot
/// serve it belongs to `crates/cli`, which is the layer that can see an
/// install.
#[test]
fn a_speculation_policy_parses_from_its_three_spellings() {
    let spec = |v: &str| {
        expect_success(parse(&tok(&[
            "--model",
            "m",
            "--prompt",
            "p",
            "--speculative",
            v,
        ])))
        .speculation
    };
    assert_eq!(spec("auto"), Speculation::Auto);
    assert_eq!(spec("off"), Speculation::Off);
    assert_eq!(spec("2"), Speculation::Block(2));
    // The bound is the batched verify's row cap: a round of block B verifies
    // B + 1 rows against MAX_BATCH_ROWS = 16.
    assert_eq!(spec("15"), Speculation::Block(15));

    // And the DEFAULT is Auto, which is the value that decides whether an
    // install carrying a head speculates without anyone asking.
    let default = expect_success(parse(&tok(&["--model", "m", "--prompt", "p"])));
    assert_eq!(default.speculation, Speculation::Auto);
}

#[test]
fn an_out_of_range_or_misspelled_speculation_block_is_refused() {
    for bad in ["0", "16", "99", "yes", "on", "-1", ""] {
        let outcome = parse(&tok(&[
            "--model",
            "m",
            "--prompt",
            "p",
            "--speculative",
            bad,
        ]));
        assert!(
            !matches!(outcome, ParseOutcome::Success(_)),
            "--speculative {bad} must be refused rather than silently defaulted"
        );
    }
}

// --- directional steering (docs/OBLITERATION.md) --------------------------

#[test]
fn steering_defaults_to_off_and_reads_no_file() {
    let req = expect_success(parse(&tok(&["--model", "m.bin", "--prompt", "hi"])));
    assert!(req.steering.is_empty());
    assert!(req.steering_mode.is_empty());
    assert!(req.steering_scale.is_empty());
    assert!(req.steering_layers.is_empty());
    assert_eq!(req.steering_target, 0.0);
    assert_eq!(req.steering_gate, 0.0);
}

#[test]
fn steering_flags_round_trip() {
    let req = expect_success(parse(&tok(&[
        "--model",
        "m.bin",
        "--prompt",
        "hi",
        "--steering",
        "/tmp/d.gguf",
        "--steering-mode",
        "clamp",
        "--steering-scale",
        "0.75",
        "--steering-layers",
        "30:40",
        "--steering-target",
        "2.5",
        "--steering-gate",
        "0.1",
    ])));
    assert_eq!(req.steering, vec!["/tmp/d.gguf".to_string()]);
    assert_eq!(
        req.steering_mode,
        vec![turbospark_invocation::SteeringMode::Clamp]
    );
    assert_eq!(req.steering_scale, vec![0.75]);
    assert_eq!(req.steering_layers, vec![(30, 40)]);
    assert_eq!(req.steering_target, 2.5);
    assert_eq!(req.steering_gate, 0.1);
}

/// The path stays an OPAQUE string here. This crate is pure and reads no
/// file, so a missing or malformed vector is the front end's error to
/// report, not a parse failure.
#[test]
fn a_steering_path_is_not_validated_by_the_parser() {
    let req = expect_success(parse(&tok(&[
        "--model",
        "m.bin",
        "--prompt",
        "hi",
        "--steering",
        "/does/not/exist.gguf",
    ])));
    assert_eq!(req.steering, vec!["/does/not/exist.gguf".to_string()]);
}

/// Each mode must parse, or a flag silently narrows to a subset of the edits
/// while still accepting the names of the rest.
///
/// A list rather than a loop over an enum, deliberately: this asserts the
/// SPELLINGS the CLI, the server and a vector file's `declared_mode` metadata
/// all share, and an exhaustive-match helper would only restate the enum.
#[test]
fn every_steering_mode_parses() {
    for (name, want) in [
        ("ablate", turbospark_invocation::SteeringMode::Ablate),
        ("add", turbospark_invocation::SteeringMode::Add),
        ("clamp", turbospark_invocation::SteeringMode::Clamp),
        ("renorm", turbospark_invocation::SteeringMode::Renorm),
    ] {
        // `--steering` is carried because a parameter without it is refused
        // as an orphan. This crate is pure and opens nothing, so the path is
        // an opaque string and need not exist.
        let req = expect_success(parse(&tok(&[
            "--model",
            "m.bin",
            "--prompt",
            "hi",
            "--steering",
            "d.gguf",
            "--steering-mode",
            name,
        ])));
        assert_eq!(req.steering_mode, vec![want], "mode {name}");
    }
}

/// A single-layer range is `N:N`, and it must be accepted: steering one
/// layer is a normal thing to ask for, and an off-by-one in the inclusivity
/// check would reject it.
#[test]
fn a_single_layer_range_is_accepted() {
    let req = expect_success(parse(&tok(&[
        "--model",
        "m.bin",
        "--prompt",
        "hi",
        "--steering",
        "d.gguf",
        "--steering-layers",
        "31:31",
    ])));
    assert_eq!(req.steering_layers, vec![(31, 31)]);
}

/// `--image` accumulates IN ORDER (ROADMAP M-V7).
///
/// The nth path pairs with the nth marker run the template renders, so a
/// container that reordered or deduplicated would silently pair each picture
/// with the wrong span. Two DISTINCT paths plus a repeat, because a set would
/// pass a two-path test and fail this one.
#[test]
fn image_paths_accumulate_in_supplied_order() {
    let request = expect_success(parse(&tok(&[
        "--model", "m", "--prompt", "p", "--image", "b.png", "--image", "a.png", "--image", "b.png",
    ])));
    assert_eq!(request.images, vec!["b.png", "a.png", "b.png"]);
    assert!(!request.image_batch);
}

#[test]
fn image_batch_is_a_valueless_flag() {
    let request = expect_success(parse(&tok(&[
        "--model",
        "m",
        "--prompt",
        "p",
        "--image-batch",
    ])));
    assert!(request.image_batch);
    assert!(request.images.is_empty());
}

#[test]
fn no_image_flag_leaves_the_list_empty() {
    let request = expect_success(parse(&tok(&["--model", "m", "--prompt", "p"])));
    assert!(request.images.is_empty());
    assert!(!request.image_batch);
}

// --- vision sidecar (vision memory sidecar, Part A4) -----------------------

#[test]
fn vision_sidecar_defaults_to_none() {
    let req = expect_success(parse(&tok(&["--model", "m.bin", "--prompt", "hi"])));
    assert_eq!(req.vision_sidecar, None);
}

/// The path stays an OPAQUE string here, exactly as `--steering`'s does: this
/// crate is pure and reads no directory, so a missing or malformed sidecar
/// is the front end's error to report, not a parse failure.
#[test]
fn a_vision_sidecar_path_is_not_validated_by_the_parser() {
    let req = expect_success(parse(&tok(&[
        "--model",
        "m.bin",
        "--prompt",
        "hi",
        "--vision-sidecar",
        "/does/not/exist.gturbo-vision",
    ])));
    assert_eq!(
        req.vision_sidecar.as_deref(),
        Some("/does/not/exist.gturbo-vision")
    );
}

/// `"auto"` is NOT a keyword this part resolves -- catalog-based sidecar
/// resolution is a later part -- so it must round-trip as an ordinary,
/// literal path rather than being special-cased or rejected.
#[test]
fn the_literal_value_auto_is_read_as_a_path_not_a_keyword() {
    let req = expect_success(parse(&tok(&[
        "--model",
        "m.bin",
        "--prompt",
        "hi",
        "--vision-sidecar",
        "auto",
    ])));
    assert_eq!(req.vision_sidecar.as_deref(), Some("auto"));
}

// --- multi-direction steering: repeatability and positional pairing --------

/// Two vectors with two of each per-vector knob resolve positionally: the
/// i-th knob occurrence configures the i-th path, in the order the paths
/// were supplied. This is the contract `runtime::SteeringState::build` and
/// the server's own resolver both consume, so the parser is where the
/// ordering is pinned.
#[test]
fn steering_flags_are_repeatable_and_pair_positionally() {
    let req = expect_success(parse(&tok(&[
        "--model",
        "m.bin",
        "--prompt",
        "hi",
        "--steering",
        "/tmp/first.gguf",
        "--steering-scale",
        "0.4",
        "--steering",
        "/tmp/second.gguf",
        "--steering-scale",
        "0.8",
        "--steering-mode",
        "add",
        "--steering-mode",
        "renorm",
        "--steering-layers",
        "10:12",
        "--steering-layers",
        "40:44",
    ])));
    assert_eq!(
        req.steering,
        vec![
            "/tmp/first.gguf".to_string(),
            "/tmp/second.gguf".to_string()
        ]
    );
    assert_eq!(req.steering_scale, vec![0.4, 0.8]);
    assert_eq!(
        req.steering_mode,
        vec![
            turbospark_invocation::SteeringMode::Add,
            turbospark_invocation::SteeringMode::Renorm,
        ]
    );
    assert_eq!(req.steering_layers, vec![(10, 12), (40, 44)]);
}

/// One knob value against several paths extends by its last value: a single
/// `--steering-scale` steers EVERY vector at that strength. This is also
/// exactly what any pre-multi-vector invocation spelling resolves to, which
/// is why the rule is "last value" and not "first" or "error".
#[test]
fn a_shorter_knob_list_extends_by_its_last_value() {
    let req = expect_success(parse(&tok(&[
        "--model",
        "m.bin",
        "--prompt",
        "hi",
        "--steering-scale",
        "0.3",
        "--steering",
        "/tmp/a.gguf",
        "--steering",
        "/tmp/b.gguf",
    ])));
    assert_eq!(req.steering.len(), 2);
    assert_eq!(req.steering_scale, vec![0.3]);
}

/// `steering_knob` is the pairing rule's one definition; both front ends
/// call it, so pinning it here pins both.
#[test]
fn steering_knob_pairs_by_index_and_extends_by_last() {
    use turbospark_invocation::steering_knob;
    let knobs = [10.0f32, 20.0, 30.0];
    assert_eq!(steering_knob(&knobs, 0), Some(10.0));
    assert_eq!(steering_knob(&knobs, 2), Some(30.0));
    // Index past the end extends by the LAST value, never wraps.
    assert_eq!(steering_knob(&knobs, 5), Some(30.0));
    assert_eq!(steering_knob::<f32>(&[], 0), None);
}
