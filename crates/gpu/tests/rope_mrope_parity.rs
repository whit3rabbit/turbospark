//! `rope_mrope_interleaved` on real Metal, against
//! `turbospark_compute::rope_mrope_interleaved` and against the kernel it has
//! to degenerate into (ROADMAP M-V5, `docs/VISION_PHASE0.md` item 2).
//!
//! # The three cases answer three different questions
//!
//! The DIVERGENT case is the ordinary parity check: does the GPU compute what
//! the CPU reference says, when `(t, h, w)` actually differ.
//!
//! The DEGENERATE case is the one the milestone rests on, and it asserts BIT
//! equality rather than a tolerance. `get_rope_index` gives every text token
//! `t == h == w`, including the text between and after images, so the trunk
//! keeps dispatching `rope_neox_subdim` there and a mixed prompt's text
//! positions stay byte-exact against the pre-vision engine. Both kernels call
//! one `apply_neox_pair`, which is what makes exactness structural; a
//! rewrite that precomputes cos/sin on the host would fail this case, and
//! that is the point of stating it.
//!
//! The CLAMP case is the only thing that reaches the two `min()`s. On the
//! real family they never bind -- `[11, 11, 10]` tiles `freq_dim` 32 exactly,
//! which is why the selector collapses to `i % 3` there -- so without a
//! section triple that does NOT tile, the clamps are untested code and the
//! collapse would pass every other case in this file.
#![cfg(target_os = "macos")]

use half::f16;
use turbospark_gpu::{rope_mrope_interleaved, rope_neox_subdim, MetalContext};

/// The real checkpoint's geometry, scaled down where the scaling cannot
/// change which branch runs. `head_dim` and `rotary_dim` are NOT scaled: they
/// are what decides `freq_dim`, and `[11, 11, 10]` tiling 32 exactly is the
/// property under test.
const HEAD_DIM: u32 = 256;
const ROTARY_DIM: u32 = 64;
const SECTION: (u32, u32, u32) = (11, 11, 10);
const THETA: f32 = 10_000_000.0;

fn ramp(len: usize) -> Vec<f32> {
    (0..len)
        .map(|i| (i as f32 - len as f32 / 2.0) * 0.05)
        .collect()
}

fn to_f16(values: &[f32]) -> Vec<f16> {
    values.iter().map(|&v| f16::from_f32(v)).collect()
}

/// **SCALE-AWARE, on purpose.** The first draft of this file compared
/// per-element against an absolute `2e-3` and failed a correct kernel at
/// `cpu 16.882074, gpu 16.875` -- a gap of 0.007, which is UNDER one FP16 ULP
/// at that magnitude (the quantum near 16 is 0.0078125). That is
/// `docs/VISION.md`'s recorded trap on this tower arriving one crate over:
/// before believing a red instrument, check whether the statistic knows the
/// scale of what it is measuring. `rel_error` normalizes by the reference's
/// own maximum, and `FP16_REDUCTION` is what the neighbouring
/// `rope_parity.rs` holds the same kernel family to.
fn assert_matches(gpu: &[f16], cpu: &[f32]) {
    let gpu_f32: Vec<f32> = gpu.iter().map(|v| v.to_f32()).collect();
    // Finiteness at the point the measurement is taken (AGENTS.md Gotcha 59):
    // `rel_error`'s `max_abs_diff` folds with `f32::max`, which silently
    // returns the non-NaN operand -- so a NaN entry is invisible to the
    // comparison below rather than failing it, exactly the "NaN reads as a
    // perfect score" shape.
    assert!(
        gpu_f32.iter().all(|v| v.is_finite()),
        "this port produced a non-finite value"
    );
    assert!(
        cpu.iter().all(|v| v.is_finite()),
        "the cpu reference produced a non-finite value"
    );
    let err = turbospark_compute::rel_error(&gpu_f32, cpu);
    assert!(
        err < turbospark_compute::Tolerance::FP16_REDUCTION,
        "rel_error = {err}"
    );
}

#[test]
fn a_divergent_triple_matches_the_cpu_reference() {
    let mut context = MetalContext::new().expect("Metal device available on this machine");

    let num_tokens = 1u32;
    let num_heads = 2u32;
    let len = (num_tokens * num_heads * HEAD_DIM) as usize;
    let input_f32 = ramp(len);

    // Deliberately three different numbers, in the shape an image block
    // produces: one clock for the block, distinct h and w offsets inside it.
    let positions = (40u32, 43u32, 47u32);

    let cpu = turbospark_compute::rope_mrope_interleaved(
        &input_f32,
        num_tokens as usize,
        num_heads as usize,
        HEAD_DIM as usize,
        ROTARY_DIM as usize,
        [
            positions.0 as usize,
            positions.1 as usize,
            positions.2 as usize,
        ],
        [SECTION.0 as usize, SECTION.1 as usize, SECTION.2 as usize],
        THETA,
    );
    let gpu = rope_mrope_interleaved(
        &mut context,
        &to_f16(&input_f32),
        positions,
        num_tokens,
        num_heads,
        HEAD_DIM,
        ROTARY_DIM,
        SECTION,
        THETA,
    )
    .expect("mrope dispatch");

    assert_matches(&gpu, &cpu);
}

/// The whole milestone's invariant. If this reddens, a mixed prompt's TEXT
/// tokens have stopped being byte-exact and the trunk's dispatch condition is
/// no longer safe -- that is the bug, not a tolerance to widen.
#[test]
fn at_t_equals_h_equals_w_it_is_bit_identical_to_rope_neox_subdim() {
    let mut context = MetalContext::new().expect("Metal device available on this machine");

    let num_tokens = 3u32;
    let num_heads = 4u32;
    let len = (num_tokens * num_heads * HEAD_DIM) as usize;
    let input = to_f16(&ramp(len));

    // Several positions, because pair 0 has angle 0 at position 0 and every
    // rope is the identity there -- one position could agree by degeneracy.
    for position in [0u32, 1, 7, 133] {
        let mrope = rope_mrope_interleaved(
            &mut context,
            &input,
            (position, position, position),
            num_tokens,
            num_heads,
            HEAD_DIM,
            ROTARY_DIM,
            SECTION,
            THETA,
        )
        .expect("mrope dispatch");
        let subdim = rope_neox_subdim(
            &mut context,
            &input,
            position,
            num_tokens,
            num_heads,
            HEAD_DIM,
            ROTARY_DIM,
            THETA,
        )
        .expect("subdim dispatch");

        for (i, (a, b)) in mrope.iter().zip(subdim.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "position {position}, element {i}: mrope {a}, subdim {b}"
            );
        }
    }
}

/// Proves the case above can FAIL, so its exactness is evidence rather than a
/// property of an input nothing rotates. Without this, a kernel that ignored
/// its positions entirely would pass the degenerate case perfectly.
#[test]
fn the_degenerate_case_discriminates() {
    let mut context = MetalContext::new().expect("Metal device available on this machine");

    let num_tokens = 1u32;
    let num_heads = 2u32;
    let input = to_f16(&ramp((num_tokens * num_heads * HEAD_DIM) as usize));

    let same = rope_mrope_interleaved(
        &mut context,
        &input,
        (12, 12, 12),
        num_tokens,
        num_heads,
        HEAD_DIM,
        ROTARY_DIM,
        SECTION,
        THETA,
    )
    .expect("mrope dispatch");
    let diverged = rope_mrope_interleaved(
        &mut context,
        &input,
        (12, 13, 14),
        num_tokens,
        num_heads,
        HEAD_DIM,
        ROTARY_DIM,
        SECTION,
        THETA,
    )
    .expect("mrope dispatch");

    assert!(
        same.iter().zip(diverged.iter()).any(|(a, b)| a != b),
        "diverging h and w changed nothing; the kernel is not reading them"
    );
}

/// The two `min()` clamps. `[1, 1, 1]` gives h only pair 1 and w only pair 2,
/// leaving every other pair on `t` -- which the `i % 3` collapse would hand to
/// h and w instead.
///
/// **THE SECTION HAS TO CLAMP AT A LOW PAIR INDEX, and the first draft chose
/// one that did not.** At `[4, 4, 4]` the clamp first bites at pair 13, whose
/// frequency is `theta^(-26/64)` = 0.0014, so dropping the clamp moves those
/// angles by 0.0043 rad and the outputs by ~3e-3 relative -- UNDER the FP16
/// bar this file has to allow. Deleting the h clamp from the shader survived
/// all five cases. At `[1, 1, 1]` the first wrongly-claimed pair is 4, at
/// frequency 0.133, and the same mutation reddens. The rule generalises past
/// this file: a rope fixture can only see a change to a pair whose FREQUENCY
/// is large, so a case testing which position drives which pair has to place
/// the disagreement near pair 0.
#[test]
fn a_section_that_does_not_tile_freq_dim_leaves_the_tail_on_t() {
    let mut context = MetalContext::new().expect("Metal device available on this machine");

    let section = (1u32, 1u32, 1u32);

    // Tolerance-free, and the assertion that survives any FP16 argument: with
    // the clamps, exactly one pair each goes to h and w.
    let selector =
        turbospark_compute::mrope_component_selector([1, 1, 1], (ROTARY_DIM / 2) as usize);
    assert_eq!(selector[1], 1, "h claims pair 1");
    assert_eq!(selector[2], 2, "w claims pair 2");
    assert!(
        selector[3..].iter().all(|&s| s == 0),
        "every pair past 2 belongs to t once the sections are exhausted"
    );

    let num_tokens = 1u32;
    let num_heads = 2u32;
    let len = (num_tokens * num_heads * HEAD_DIM) as usize;
    let input_f32 = ramp(len);
    let positions = (40u32, 43u32, 47u32);

    let cpu = turbospark_compute::rope_mrope_interleaved(
        &input_f32,
        num_tokens as usize,
        num_heads as usize,
        HEAD_DIM as usize,
        ROTARY_DIM as usize,
        [
            positions.0 as usize,
            positions.1 as usize,
            positions.2 as usize,
        ],
        [section.0 as usize, section.1 as usize, section.2 as usize],
        THETA,
    );
    let gpu = rope_mrope_interleaved(
        &mut context,
        &to_f16(&input_f32),
        positions,
        num_tokens,
        num_heads,
        HEAD_DIM,
        ROTARY_DIM,
        section,
        THETA,
    )
    .expect("mrope dispatch");

    assert_matches(&gpu, &cpu);

    // And that the clamp is OBSERVABLE: a clamped section must not agree with
    // the tiling one, or this case is checking the same code path twice.
    let tiling = rope_mrope_interleaved(
        &mut context,
        &to_f16(&input_f32),
        positions,
        num_tokens,
        num_heads,
        HEAD_DIM,
        ROTARY_DIM,
        SECTION,
        THETA,
    )
    .expect("mrope dispatch");
    assert!(
        gpu.iter().zip(tiling.iter()).any(|(a, b)| a != b),
        "the clamped section computed what the tiling one did; the min()s are dead"
    );
}

/// The internal cross-check that made the selector a derivation rather than a
/// plausible transcription: the reference's own six lines, run on this
/// family's geometry, must reproduce the section it was given.
#[test]
fn the_selector_reproduces_the_declared_section_counts() {
    let freq_dim = (ROTARY_DIM / 2) as usize;
    assert_eq!(freq_dim, 32);
    let selector = turbospark_compute::mrope_component_selector([11, 11, 10], freq_dim);

    let count = |c: u8| selector.iter().filter(|&&s| s == c).count();
    assert_eq!([count(0), count(1), count(2)], [11, 11, 10]);
}
