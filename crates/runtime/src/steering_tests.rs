use super::*;

/// THE PER-LAYER BLOCKS MUST PARTITION THE BUFFER, and nothing about the
/// generated logits can tell you whether they do.
///
/// A batched pass writes M floats from `coeff_offset(layer)`. If the
/// stride were narrower than a block -- one float per layer, which is
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
        let size = SteeringState::coeff_bytes(layers);
        for l in 0..layers {
            let base = SteeringState::coeff_offset(l);
            assert!(
                base + (MAX_STEER_ROWS * 4) as u64 <= size,
                "a full block at layer {l} of {layers} runs past the {size}-byte buffer; \
                 a batched steer there would write off the end of it"
            );
            if l + 1 < layers {
                assert_eq!(
                    SteeringState::coeff_offset(l + 1) - base,
                    (MAX_STEER_ROWS * 4) as u64,
                    "layer {l}'s block overlaps layer {}'s, so a batched write of M rows \
                     would land in the next layer's coefficients",
                    l + 1
                );
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
/// a broken engine rather than as an unsupported family -- and the two would
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
