//! The two kernels Qwen2.5-VL adds, on real Metal (`docs/QWEN25_VL_VISION_PHASE0.md`):
//! the tower's SiLU and the trunk's CHUNKED mRoPE.
//!
//! As in `rope_mrope_parity.rs` and `gemma4_vision_kernels.rs`, each parity
//! case is paired with one that shows the fixture can tell the kernel from
//! its plausible wrong twin: the chunked selector from the interleaved one
//! (same inputs, same sections, different pairs on different streams), and
//! SiLU from the GELU that looks like it.
#![cfg(target_os = "macos")]

use half::f16;
use turbospark_gpu::{
    encode_rope_mrope_chunked, encode_vision_silu, rope_mrope_interleaved, rope_neox_subdim,
    MetalContext,
};

const HEAD_DIM: u32 = 128;
const SECTION: (u32, u32, u32) = (16, 24, 24);
const THETA: f32 = 1_000_000.0;

fn read_f16(buffer: &metal::Buffer, len: usize) -> Vec<f16> {
    let ptr = buffer.contents() as *const u16;
    // The buffer holds at least `len` halves and its pass has been waited on.
    let bits = unsafe { std::slice::from_raw_parts(ptr, len) };
    bits.iter().map(|&b| f16::from_bits(b)).collect()
}

fn ramp(len: usize) -> Vec<f16> {
    (0..len)
        .map(|i| f16::from_f32((i as f32 - len as f32 / 2.0) * 0.013))
        .collect()
}

fn run_chunked(
    context: &mut MetalContext,
    input: &[f16],
    positions: (u32, u32, u32),
    heads: u32,
    section: (u32, u32, u32),
) -> Vec<f16> {
    let buffer = context.new_buffer_with_data(input);
    let pass = context.begin_pass();
    encode_rope_mrope_chunked(
        context,
        &pass,
        (&buffer, 0),
        positions,
        heads,
        HEAD_DIM,
        HEAD_DIM,
        section,
        THETA,
    )
    .expect("chunked mrope dispatch");
    pass.commit_and_wait();
    read_f16(&buffer, input.len())
}

/// The reference selector written out from the pair index, independent of the
/// kernel: pairs [0, t) take t, [t, t+h) take h, the rest take w.
fn cpu_chunked(input: &[f16], heads: usize, positions: (u32, u32, u32)) -> Vec<f32> {
    let half = HEAD_DIM as usize / 2;
    let mut out: Vec<f32> = input.iter().map(|v| v.to_f32()).collect();
    for head in 0..heads {
        let base = head * HEAD_DIM as usize;
        for pair in 0..half {
            let position = if (pair as u32) < SECTION.0 {
                positions.0
            } else if (pair as u32) < SECTION.0 + SECTION.1 {
                positions.1
            } else {
                positions.2
            };
            let angle = position as f32 * THETA.powf(-(2.0 * pair as f32) / HEAD_DIM as f32);
            let (sin, cos) = angle.sin_cos();
            let x0 = input[base + pair].to_f32();
            let x1 = input[base + half + pair].to_f32();
            out[base + pair] = x0 * cos - x1 * sin;
            out[base + half + pair] = x0 * sin + x1 * cos;
        }
    }
    out
}

fn worst_relative(gpu: &[f16], cpu: &[f32]) -> f32 {
    assert_eq!(gpu.len(), cpu.len());
    let scale = cpu.iter().fold(1.0f32, |m, v| m.max(v.abs()));
    gpu.iter().zip(cpu).fold(0.0f32, |m, (g, c)| {
        let g = g.to_f32();
        assert!(g.is_finite() && c.is_finite(), "non-finite value");
        m.max((g - c).abs() / scale)
    })
}

/// The invariant that keeps every text token of a mixed prompt byte-exact.
#[test]
fn at_t_equals_h_equals_w_it_is_bit_identical_to_rope_neox_subdim() {
    let mut context = MetalContext::new().expect("Metal device");
    let heads = 3u32;
    let input = ramp((heads * HEAD_DIM) as usize);
    // Several positions: position 0 is the identity for every rope.
    for position in [0u32, 1, 7, 133, 4096] {
        let chunked = run_chunked(
            &mut context,
            &input,
            (position, position, position),
            heads,
            SECTION,
        );
        let neox = rope_neox_subdim(
            &mut context,
            &input,
            position,
            1,
            heads,
            HEAD_DIM,
            HEAD_DIM,
            THETA,
        )
        .expect("neox dispatch");
        let a: Vec<u16> = chunked.iter().map(|v| v.to_bits()).collect();
        let b: Vec<u16> = neox.iter().map(|v| v.to_bits()).collect();
        assert_eq!(a, b, "position {position}");
    }
}

#[test]
fn a_divergent_triple_matches_the_cpu_selector() {
    let mut context = MetalContext::new().expect("Metal device");
    let heads = 2u32;
    let input = ramp((heads * HEAD_DIM) as usize);
    // The shape an image block produces: one clock, distinct h and w offsets.
    let positions = (40u32, 43u32, 47u32);
    let gpu = run_chunked(&mut context, &input, positions, heads, SECTION);
    let cpu = cpu_chunked(&input, heads as usize, positions);
    let err = worst_relative(&gpu, &cpu);
    assert!(err < 2e-3, "worst relative error {err}");
}

/// Without this the case above could pass against a kernel that implemented
/// Qwen3-VL's interleaved selector: same sections, same positions, and the
/// two disagree on which stream most pairs read.
#[test]
fn the_chunked_selector_is_not_the_interleaved_one() {
    let mut context = MetalContext::new().expect("Metal device");
    let heads = 2u32;
    let input = ramp((heads * HEAD_DIM) as usize);
    let positions = (40u32, 43u32, 47u32);
    let chunked = run_chunked(&mut context, &input, positions, heads, SECTION);
    let interleaved = rope_mrope_interleaved(
        &mut context,
        &input,
        positions,
        1,
        heads,
        HEAD_DIM,
        HEAD_DIM,
        SECTION,
        THETA,
    )
    .expect("interleaved dispatch");
    let cpu_interleaved: Vec<f32> = interleaved.iter().map(|v| v.to_f32()).collect();
    let gap = worst_relative(&chunked, &cpu_interleaved);
    assert!(
        gap > 1e-2,
        "the two selectors agree to {gap}; the fixture cannot tell them apart"
    );
}

#[test]
fn sections_that_do_not_cover_the_rotary_pairs_are_refused() {
    let mut context = MetalContext::new().expect("Metal device");
    let buffer = context.new_buffer_with_data(&ramp(HEAD_DIM as usize));
    let pass = context.begin_pass();
    let err = encode_rope_mrope_chunked(
        &mut context,
        &pass,
        (&buffer, 0),
        (1, 2, 3),
        1,
        HEAD_DIM,
        HEAD_DIM,
        (16, 24, 23),
        THETA,
    );
    assert!(
        err.is_err(),
        "a split summing to 63 of 64 pairs must be refused"
    );
    pass.commit_and_wait();
}

fn run_silu(context: &mut MetalContext, values: &[f32]) -> Vec<f32> {
    let halves: Vec<f16> = values.iter().map(|&v| f16::from_f32(v)).collect();
    let buffer = context.new_buffer_with_data(&halves);
    let pass = context.begin_pass();
    encode_vision_silu(context, &pass, (&buffer, 0), values.len() as u32).expect("silu dispatch");
    pass.commit_and_wait();
    read_f16(&buffer, values.len())
        .iter()
        .map(|v| v.to_f32())
        .collect()
}

#[test]
fn silu_matches_the_definition_and_is_not_gelu() {
    let mut context = MetalContext::new().expect("Metal device");
    // Negative, zero, small and large magnitudes, and a length that is not a
    // multiple of the threadgroup width.
    let values: Vec<f32> = (0..1000).map(|i| (i as f32 - 500.0) * 0.02).collect();
    let rounded: Vec<f32> = values.iter().map(|&v| f16::from_f32(v).to_f32()).collect();
    let got = run_silu(&mut context, &values);
    let mut worst_silu = 0.0f32;
    let mut worst_gelu = 0.0f32;
    for (x, g) in rounded.iter().zip(&got) {
        assert!(g.is_finite());
        let silu = x / (1.0 + (-x).exp());
        let inner = 0.797_884_6 * (x + 0.044_715 * x * x * x);
        let gelu = 0.5 * x * (1.0 + inner.tanh());
        worst_silu = worst_silu.max((g - silu).abs() / silu.abs().max(1.0));
        worst_gelu = worst_gelu.max((g - gelu).abs() / gelu.abs().max(1.0));
    }
    assert!(
        worst_silu < 2e-3,
        "worst relative error against silu {worst_silu}"
    );
    assert!(
        worst_gelu > 1e-2,
        "silu is indistinguishable from gelu here ({worst_gelu})"
    );
}
