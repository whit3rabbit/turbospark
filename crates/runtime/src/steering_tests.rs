use super::*;

/// THE PER-LAYER BLOCKS MUST PARTITION THE BUFFER, and nothing about the
/// generated logits can tell you whether they do.
///
/// A batched pass writes M floats from `coeff_offset(layer, k, vectors)`. If
/// the stride were narrower than a block -- one float per layer, which is
/// what it was before the batched path existed -- layer L's write would
/// land inside layer L+1's block, and at the LAST layer it would run off
/// the end of the buffer entirely. Neither shows up downstream: the
/// coefficients are an output-only measurement surface, every later
/// layer's own write happens to overwrite the spill, and a short GPU
/// buffer overrun lands in page slack rather than faulting.
///
/// So this is the only place the layout can be checked at all, and it is
/// arithmetic: no GPU, no install, microseconds.
#[test]
fn the_coefficient_blocks_partition_the_buffer() {
    for layers in [1usize, 2, 4, 64] {
        for vectors in [1usize, 2, 4] {
            let size = SteeringState::coeff_bytes(layers, vectors);
            for l in 0..layers {
                for k in 0..vectors {
                    let base = SteeringState::coeff_offset(l, k, vectors);
                    assert!(
                        base + (MAX_STEER_ROWS * 4) as u64 <= size,
                        "a full block at layer {l} vector {k} of {layers}x{vectors} runs past \
                         the {size}-byte buffer; a batched steer there would write off the end \
                         of it"
                    );
                    if k + 1 < vectors {
                        assert_eq!(
                            SteeringState::coeff_offset(l, k + 1, vectors) - base,
                            (MAX_STEER_ROWS * 4) as u64,
                            "layer {l} vector {k}'s block overlaps vector {}'s, so one vector's \
                             coefficients would be read as another's",
                            k + 1
                        );
                    }
                }
                if l + 1 < layers {
                    assert_eq!(
                        SteeringState::coeff_offset(l + 1, 0, vectors)
                            - SteeringState::coeff_offset(l, 0, vectors),
                        (vectors * MAX_STEER_ROWS * 4) as u64,
                        "layer {l}'s blocks overlap layer {}'s, so a batched write of M rows \
                         would land in the next layer's coefficients",
                        l + 1
                    );
                }
            }
        }
    }
}

/// **THE REFUSAL AND THE CAPABILITY REPORT ARE ONE STRING, and this is the
/// only thing that says so.**
///
/// `real_forward_open` refuses a direction set on a family that cannot
/// dispatch the edit, and `crates/ffi` reports the same sentence as
/// `steering.reason` BEFORE anyone opens anything. Two spellings would let a
/// GUI tell a user a family steers while the open refuses it, which reads as
/// a broken engine rather than an unsupported family -- and the two would
/// drift the way the edit's own family gate and the capture's did before they
/// were merged into one predicate (see `family_dispatches_steering`).
///
/// It is also the ONLY coverage the family refusal has: no test in this
/// repository opens an unsupported install with a vector, because reaching
/// that arm needs a correctly-SIZED vector for a family this port refuses,
/// and `SteeringSet::validate` rejects a width mismatch first (measured on
/// the real `qwen4exp` install: 2560 expected against a 5120 vector, so the
/// width check fires and the family check is never reached).
#[test]
fn the_reason_is_present_exactly_when_the_family_cannot_steer() {
    use model_io::ModelFamily as F;
    for family in [
        F::Gemma4,
        F::QwenGdnMoe,
        F::QwenGdnDense,
        F::Llama,
        F::Qwen3Moe,
        F::GptOss,
        F::MuseGlimmer,
        F::DeepseekV4Flash,
        F::Qwen4Exp,
    ] {
        let dispatches = family_dispatches_steering(family);
        let reason = steering_unsupported_reason(family);
        assert_eq!(
            dispatches,
            reason.is_none(),
            "{family:?}: a family that steers must name no reason, and one that does not must \
             name one"
        );
        if let Some(reason) = reason {
            // It has to NAME the family, or a GUI shows a user a sentence
            // that could be about any model they have.
            assert!(
                reason.contains(&format!("{family:?}")),
                "{family:?} is not named in its own refusal: {reason}"
            );
            assert!(
                reason.contains("silent no-op"),
                "the refusal must say WHY refusing beats loading and ignoring: {reason}"
            );
        }
    }
}

/// Exactly two families cannot steer today, and both are refused for reasons
/// recorded beside the predicate. Pinned so ADDING a family is a deliberate
/// edit here rather than something a `_ =>` arm could absorb.
#[test]
fn the_two_unsteerable_families_are_the_expected_ones() {
    use model_io::ModelFamily as F;
    let unsupported: Vec<F> = [
        F::Gemma4,
        F::QwenGdnMoe,
        F::QwenGdnDense,
        F::Llama,
        F::Qwen3Moe,
        F::GptOss,
        F::MuseGlimmer,
        F::DeepseekV4Flash,
        F::Qwen4Exp,
    ]
    .into_iter()
    .filter(|f| !family_dispatches_steering(*f))
    .collect();
    assert_eq!(unsupported, vec![F::DeepseekV4Flash, F::Qwen4Exp]);
}

fn one_layer_set(hidden: usize, values: Vec<f32>) -> model_io::SteeringSet {
    model_io::SteeringSet {
        layers: vec![Some(model_io::LayerDirection::new(values))],
        hidden,
        declared_mode: None,
        declared_arch: None,
    }
}

#[test]
fn a_zero_norm_steering_direction_is_refused_naming_the_layer() {
    let context = gpu::MetalContext::new().expect("Metal device");
    let arch = turbospark_repack::tiny_gemma4_arch(128, 1);
    let hidden = arch.hidden_size as usize;
    let policy = SteeringPolicy::single(
        one_layer_set(hidden, vec![0.0; hidden]),
        foundation::SteeringMode::Add,
        1.0,
        0.0,
        0.0,
    );
    let res = SteeringState::build(&context, &arch, &policy);
    let err = match res {
        Err(e) => e,
        Ok(_) => panic!("expected zero norm steering direction to be refused"),
    };
    match err {
        RealForwardError::Unsupported(msg) => {
            assert!(msg.contains("layer 0 has zero norm"), "msg: {msg}");
        }
        other => panic!("expected Unsupported error, got {other:?}"),
    }
}

/// With two vectors loaded, a zero norm in the SECOND one must name the
/// vector it came from: "layer 0 has zero norm" alone would send the caller
/// debugging the wrong file.
#[test]
fn a_zero_norm_in_the_second_vector_names_that_vector() {
    let context = gpu::MetalContext::new().expect("Metal device");
    let arch = turbospark_repack::tiny_gemma4_arch(128, 1);
    let hidden = arch.hidden_size as usize;
    let policy = SteeringPolicy {
        vectors: vec![
            SteeringVector {
                set: one_layer_set(hidden, vec![1.0; hidden]),
                mode: foundation::SteeringMode::Add,
                alpha: 1.0,
            },
            SteeringVector {
                set: one_layer_set(hidden, vec![0.0; hidden]),
                mode: foundation::SteeringMode::Add,
                alpha: 1.0,
            },
        ],
        target: 0.0,
        gate_threshold: 0.0,
    };
    match SteeringState::build(&context, &arch, &policy) {
        Err(RealForwardError::Unsupported(msg)) => {
            assert!(
                msg.contains("vector 1") && msg.contains("layer 0 has zero norm"),
                "the refusal must name the offending vector and layer: {msg}"
            );
        }
        Err(other) => panic!("expected Unsupported error, got {other:?}"),
        Ok(_) => panic!("expected zero norm steering direction to be refused"),
    }
}

#[test]
fn nan_steering_parameters_are_refused() {
    let context = gpu::MetalContext::new().expect("Metal device");
    let arch = turbospark_repack::tiny_gemma4_arch(128, 1);
    let hidden = arch.hidden_size as usize;
    let make_policy = |alpha: f32, target: f32, gate: f32| {
        SteeringPolicy::single(
            one_layer_set(hidden, vec![1.0; hidden]),
            foundation::SteeringMode::Add,
            alpha,
            target,
            gate,
        )
    };

    for (alpha, target, gate) in [
        (f32::NAN, 0.0, 0.0),
        (1.0, f32::NAN, 0.0),
        (1.0, 0.0, f32::NAN),
    ] {
        let policy = make_policy(alpha, target, gate);
        let res = SteeringState::build(&context, &arch, &policy);
        match res {
            Err(RealForwardError::Unsupported(msg)) => {
                assert!(
                    msg.contains("steering parameters must be finite")
                        || msg.contains("alpha must be finite"),
                    "unexpected error message: {msg}"
                );
            }
            Err(other) => panic!("expected Unsupported error for NaN parameter, got {other:?}"),
            Ok(_) => panic!("expected Unsupported error for NaN parameter, got Ok"),
        }
    }
}

/// A NaN alpha on the SECOND vector must be caught too: the shared-scalar
/// check predates per-vector alphas, and a check that only reads `policy[0]`
/// would pass vector 1's NaN into the kernel.
#[test]
fn a_nan_alpha_on_the_second_vector_is_refused() {
    let context = gpu::MetalContext::new().expect("Metal device");
    let arch = turbospark_repack::tiny_gemma4_arch(128, 1);
    let hidden = arch.hidden_size as usize;
    let policy = SteeringPolicy {
        vectors: vec![
            SteeringVector {
                set: one_layer_set(hidden, vec![1.0; hidden]),
                mode: foundation::SteeringMode::Add,
                alpha: 1.0,
            },
            SteeringVector {
                set: one_layer_set(hidden, vec![1.0; hidden]),
                mode: foundation::SteeringMode::Add,
                alpha: f32::NAN,
            },
        ],
        target: 0.0,
        gate_threshold: 0.0,
    };
    match SteeringState::build(&context, &arch, &policy) {
        Err(RealForwardError::Unsupported(msg)) => {
            assert!(
                msg.contains("vector 1") && msg.contains("finite"),
                "the refusal must name the offending vector: {msg}"
            );
        }
        Err(other) => panic!("expected Unsupported error, got {other:?}"),
        Ok(_) => panic!("expected NaN alpha to be refused"),
    }
}

#[test]
fn steering_state_build_computes_inv_norm_from_f16_values_and_packs_offsets() {
    let context = gpu::MetalContext::new().expect("Metal device");
    let arch = turbospark_repack::tiny_gemma4_arch(128, 2);
    let hidden = arch.hidden_size as usize;

    let mut dir0 = vec![0.0f32; hidden];
    dir0[0] = 3.0;
    dir0[1] = 4.0;

    let mut set = one_layer_set(hidden, dir0);
    set.layers.push(None); // layer 1 unsteered

    let policy = SteeringPolicy::single(set, foundation::SteeringMode::Ablate, 1.0, 0.0, 0.0);

    let state = SteeringState::build(&context, &arch, &policy)
        .expect("build should succeed")
        .expect("state must be Some");

    assert_eq!(state.layers.len(), 2);
    let l0 = state.layer(0);
    assert_eq!(l0.len(), 1, "one slot per vector");
    let e0 = l0[0].expect("vector 0 covers layer 0");
    assert_eq!(e0.offset, 0);
    // sqrt(3^2 + 4^2) = 5.0, 1 / 5 = 0.2
    assert!((e0.inv_norm - 0.2).abs() < 1e-5);
    assert!(
        state.layer(1).iter().all(|s| s.is_none()),
        "layer 1 was not in set and must hold no entries"
    );
}

/// The multi-vector packing: two vectors over disjoint layers, each keeping
/// its own mode and alpha, with the second vector's directions packed AFTER
/// the first's so neither offset collides.
#[test]
fn two_vectors_keep_their_own_params_and_distinct_offsets() {
    let context = gpu::MetalContext::new().expect("Metal device");
    let arch = turbospark_repack::tiny_gemma4_arch(128, 2);
    let hidden = arch.hidden_size as usize;

    let set_a = one_layer_set(hidden, vec![1.0; hidden]); // covers layer 0
    let mut set_b = model_io::SteeringSet {
        layers: vec![None],
        hidden,
        declared_mode: None,
        declared_arch: None,
    };
    set_b
        .layers
        .push(Some(model_io::LayerDirection::new(vec![2.0; hidden]))); // covers layer 1

    let policy = SteeringPolicy {
        vectors: vec![
            SteeringVector {
                set: set_a,
                mode: foundation::SteeringMode::Ablate,
                alpha: 0.4,
            },
            SteeringVector {
                set: set_b,
                mode: foundation::SteeringMode::Add,
                alpha: 0.8,
            },
        ],
        target: 0.0,
        gate_threshold: 0.0,
    };

    let state = SteeringState::build(&context, &arch, &policy)
        .expect("build should succeed")
        .expect("state must be Some");

    assert_eq!(state.coeff_stride, 2);
    assert_eq!(state.covered_layers(), 2);

    let l0 = state.layer(0);
    assert_eq!(l0.len(), 2, "one slot per vector");
    assert!(l0[1].is_none(), "layer 0 is covered by vector 0 only");
    let e0 = l0[0].expect("vector 0 covers layer 0");
    assert_eq!(e0.offset, 0);
    assert_eq!(e0.mode, foundation::SteeringMode::Ablate);
    assert!((e0.alpha - 0.4).abs() < 1e-6);

    let l1 = state.layer(1);
    assert!(l1[0].is_none(), "layer 1 is covered by vector 1 only");
    let e1 = l1[1].expect("vector 1 covers layer 1");
    assert_eq!(
        e1.offset,
        (hidden * 2) as u64,
        "vector 1's direction packs after vector 0's, not at offset 0 where it would \
         steer layer 1 with vector 0's direction"
    );
    assert_eq!(e1.mode, foundation::SteeringMode::Add);
    assert!((e1.alpha - 0.8).abs() < 1e-6);

    // The readback shape: one slot per (layer, vector), None where a vector
    // does not cover.
    let coeffs = state.coefficients();
    assert_eq!(coeffs.len(), 2);
    assert_eq!(coeffs[0].len(), 2);
    assert!(coeffs[0][0].is_some(), "vector 0 covers layer 0");
    assert!(coeffs[0][1].is_none(), "vector 1 does not cover layer 0");
    assert!(coeffs[1][0].is_none(), "vector 0 does not cover layer 1");
    assert!(coeffs[1][1].is_some(), "vector 1 covers layer 1");
}

/// Two vectors covering the SAME layer must get distinct coefficient slots:
/// a shared slot would make vector 1's dispatch overwrite vector 0's
/// measurement, and the trace would report only the last edit as if it were
/// the only one.
#[test]
fn two_vectors_on_one_layer_take_distinct_coefficient_slots() {
    let context = gpu::MetalContext::new().expect("Metal device");
    let arch = turbospark_repack::tiny_gemma4_arch(128, 1);
    let hidden = arch.hidden_size as usize;

    let policy = SteeringPolicy {
        vectors: vec![
            SteeringVector {
                set: one_layer_set(hidden, vec![1.0; hidden]),
                mode: foundation::SteeringMode::Add,
                alpha: 0.2,
            },
            SteeringVector {
                set: one_layer_set(hidden, {
                    let mut v = vec![0.0f32; hidden];
                    v[0] = 1.0;
                    v
                }),
                mode: foundation::SteeringMode::Add,
                alpha: 0.4,
            },
        ],
        target: 0.0,
        gate_threshold: 0.0,
    };

    let state = SteeringState::build(&context, &arch, &policy)
        .expect("build should succeed")
        .expect("state must be Some");

    let l0 = state.layer(0);
    assert_eq!(l0.len(), 2, "one slot per vector");
    let (a, b) = (
        l0[0].as_ref().expect("vector 0 covers layer 0"),
        l0[1].as_ref().expect("vector 1 covers layer 0"),
    );
    assert_ne!(
        a.offset, b.offset,
        "two directions in one layer must pack at distinct offsets, or the second edit \
         would steer with the first vector's direction"
    );
    assert_eq!(
        SteeringState::coeff_offset(0, 0, state.coeff_stride),
        0,
        "vector 0's block starts the layer's region"
    );
    assert_eq!(
        SteeringState::coeff_offset(0, 1, state.coeff_stride),
        (MAX_STEER_ROWS * 4) as u64,
        "vector 1's block starts one full block in, so a batched write by vector 0 \
         cannot land in vector 1's coefficients"
    );
}

/// One vector keeps the summary line every earlier release printed, because
/// that line is the Swift capability surface's `summary` and the startup
/// line people have read for months.
#[test]
fn the_single_vector_summary_line_is_unchanged() {
    let context = gpu::MetalContext::new().expect("Metal device");
    let arch = turbospark_repack::tiny_gemma4_arch(128, 1);
    let hidden = arch.hidden_size as usize;
    let policy = SteeringPolicy::single(
        one_layer_set(hidden, vec![1.0; hidden]),
        foundation::SteeringMode::Ablate,
        0.4,
        0.0,
        0.5,
    );
    let state = SteeringState::build(&context, &arch, &policy)
        .expect("build should succeed")
        .expect("state must be Some");
    assert_eq!(
        state.summary(),
        "steering: ablate at alpha 0.4 over 1 of 1 layers, gated at |c| >= 0.5"
    );
}

/// The multi-vector summary names the count, the modes with their alphas in
/// application order, and the gate.
#[test]
fn the_multi_vector_summary_lists_each_edit_in_order() {
    let context = gpu::MetalContext::new().expect("Metal device");
    let arch = turbospark_repack::tiny_gemma4_arch(128, 1);
    let hidden = arch.hidden_size as usize;
    let policy = SteeringPolicy {
        vectors: vec![
            SteeringVector {
                set: one_layer_set(hidden, vec![1.0; hidden]),
                mode: foundation::SteeringMode::Ablate,
                alpha: 0.4,
            },
            SteeringVector {
                set: one_layer_set(hidden, vec![2.0; hidden]),
                mode: foundation::SteeringMode::Add,
                alpha: 0.8,
            },
        ],
        target: 0.0,
        gate_threshold: 0.0,
    };
    let state = SteeringState::build(&context, &arch, &policy)
        .expect("build should succeed")
        .expect("state must be Some");
    let line = state.summary();
    assert!(
        line.starts_with("steering: 2 vectors over 1 of 1 layers:"),
        "unexpected summary: {line}"
    );
    assert!(
        line.contains("ablate@0.4 (1 layers) then add@0.8 (1 layers)"),
        "the summary must list one clause per vector, in application order, with \
         its coverage: {line}"
    );
}

/// `off` builds no state and is inactive; this is the property the frozen
/// qwen38 memory rows stand on, restated for the new shape.
#[test]
fn an_empty_policy_is_inactive_and_builds_nothing() {
    assert!(!SteeringPolicy::off().is_active());
    let context = gpu::MetalContext::new().expect("Metal device");
    let arch = turbospark_repack::tiny_gemma4_arch(128, 1);
    assert!(
        SteeringState::build(&context, &arch, &SteeringPolicy::off())
            .expect("off builds clean")
            .is_none()
    );
}
