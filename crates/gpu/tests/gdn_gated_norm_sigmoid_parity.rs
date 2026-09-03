//! Parity test for `gdn_gated_norm`'s SIGMOID-baked pipeline
//! (`FC_GDN_GATE_SIGMOID`, `qwen4_exp`'s `output_gate_type: sigmoid`)
//! against `turbospark_compute::gated_norm_sigmoid`.
//!
//! **THE DEFAULT (SILU) PATH IS UNTOUCHED BY THIS FILE.** `crates/gpu`
//! CLAUDE.md Gotcha 12's mandate ("a family that needs sigmoid owes a
//! function constant AND a parity case that reddens under the wrong one")
//! is met here without changing `encode_gdn_gated_norm`'s signature or
//! either of its two production call sites
//! (`families/qwen/attn.rs`, `families/qwen/batched_layers.rs`) at all --
//! `gdn_parity.rs`'s existing `decode_chain_matches_cpu_reference` already
//! covers that path unchanged, which is the point: this file adds a new,
//! separately-keyed pipeline rather than touching the one two real families
//! already dispatch.

#![cfg(target_os = "macos")]

use half::f16;
use turbospark_compute::{bf16_to_f32, f32_to_bf16, gated_norm_sigmoid};
use turbospark_gpu::{
    encode_gdn_gated_norm, encode_gdn_gated_norm_sigmoid, read_buffer_f16, GdnShape, MetalContext,
};

const HV: usize = 4;
const DV: usize = 32;

fn shape() -> GdnShape {
    GdnShape {
        num_k_heads: 2,
        num_v_heads: HV as u32,
        key_head_dim: 32,
        value_head_dim: DV as u32,
        conv_kernel_size: 4,
    }
}

fn deterministic(seed: u64, n: usize, scale: f32, center: f32) -> Vec<f32> {
    let mut state = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
    (0..n)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            center + (((state % 2000) as f32 / 1000.0) - 1.0) * scale
        })
        .collect()
}

fn f16_pair(seed: u64, n: usize, scale: f32, center: f32) -> (Vec<f16>, Vec<f32>) {
    let halves: Vec<f16> = deterministic(seed, n, scale, center)
        .iter()
        .map(|&v| f16::from_f32(v))
        .collect();
    let values = halves.iter().map(|h| h.to_f32()).collect();
    (halves, values)
}

fn bf16_pair(seed: u64, n: usize, scale: f32, center: f32) -> (Vec<u16>, Vec<f32>) {
    let bits: Vec<u16> = deterministic(seed, n, scale, center)
        .iter()
        .map(|&v| f32_to_bf16(v))
        .collect();
    let values = bits.iter().map(|&b| bf16_to_f32(b)).collect();
    (bits, values)
}

fn half_bytes(v: &[f16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_bits().to_le_bytes()).collect()
}

fn u16_bytes(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// Runs one gated-norm kernel over `y16`/`z16`/`w_bits` (`[Hv*Dv]`,
/// `[Hv*Dv]`, `[Dv]`), returning FP32. `sigmoid` picks
/// `encode_gdn_gated_norm_sigmoid` over `encode_gdn_gated_norm`.
fn run_gated_norm(
    context: &mut MetalContext,
    y16: &[f16],
    z16: &[f16],
    w_bits: &[u16],
    sigmoid: bool,
) -> Vec<f32> {
    let total = y16.len();
    let y_buf = context.new_buffer_with_data(&half_bytes(y16));
    let z_buf = context.new_buffer_with_data(&half_bytes(z16));
    let w_buf = context.new_buffer_with_data(&u16_bytes(w_bits));
    let out_buf = context.new_output_buffer((total * 2) as u64);

    let pass = context.begin_pass();
    let encode = if sigmoid {
        encode_gdn_gated_norm_sigmoid
    } else {
        encode_gdn_gated_norm
    };
    encode(
        context,
        &pass,
        shape(),
        (&y_buf, 0),
        (&z_buf, 0),
        (&w_buf, 0),
        (&out_buf, 0),
        1,
    )
    .expect("encode");
    pass.commit_and_wait();
    read_buffer_f16(&out_buf, 0, total)
        .iter()
        .map(|h| h.to_f32())
        .collect()
}

/// The sigmoid pipeline against the CPU reference, at real proportions
/// (`HV = 4` value heads). Every head's `y`/`z` slice is distinct, so a
/// mis-derived head stride reads a wrong number rather than a plausible one.
#[test]
fn gdn_gated_norm_sigmoid_matches_cpu_reference() {
    let mut context = MetalContext::new().expect("Metal device");
    let total = HV * DV;

    let (y16, y32) = f16_pair(11, total, 0.6, 0.0);
    let (z16, z32) = f16_pair(13, total, 2.5, 0.0);
    let (w_bits, w32) = bf16_pair(17, DV, 0.5, 1.0);

    let expected = gated_norm_sigmoid(&y32, &z32, &w32, HV);
    let got = run_gated_norm(&mut context, &y16, &z16, &w_bits, true);

    assert_eq!(got.len(), expected.len());
    for i in 0..total {
        let diff = (got[i] - expected[i]).abs();
        assert!(
            diff <= 2e-3_f32.max(expected[i].abs() * 1e-2),
            "i={i} (head {}): got {} want {}",
            i / DV,
            got[i],
            expected[i]
        );
    }
}

/// **THE FIXTURE MUST DISCRIMINATE SIGMOID FROM SILU, per `crates/gpu`
/// CLAUDE.md Gotcha 12's own mandate.** `z` sits at large NEGATIVE
/// magnitude, where the two gates diverge sharply and even in SIGN:
/// `silu(-6) ~ -0.0148` while `sigmoid(-6) ~ 0.00247` -- so a kernel that
/// silently fell back to silu under the sigmoid pipeline disagrees with the
/// CPU reference by more than rounding, not merely by a different curve
/// shape near zero where the two gates nearly coincide.
#[test]
fn the_sigmoid_and_silu_gates_are_different_functions() {
    let mut context = MetalContext::new().expect("Metal device");
    let total = HV * DV;

    let (y16, _y32) = f16_pair(23, total, 0.6, 0.0);
    // Large negative z: silu(-6) ~ -0.0148, sigmoid(-6) ~ 0.00247 -- opposite
    // sign, not just a different magnitude.
    let z16: Vec<f16> = vec![f16::from_f32(-6.0); total];
    let (w_bits, _w32) = bf16_pair(29, DV, 0.5, 1.0);

    let silu_out = run_gated_norm(&mut context, &y16, &z16, &w_bits, false);
    let sigmoid_out = run_gated_norm(&mut context, &y16, &z16, &w_bits, true);

    let max_gap = silu_out
        .iter()
        .zip(&sigmoid_out)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_gap > 1e-3,
        "the fixture cannot tell the two gates apart (max gap {max_gap}); \
         a parity test on it would pass against either kernel"
    );
    // The two must even DISAGREE IN SIGN on at least one output: at z=-6,
    // silu is negative and sigmoid is positive, so `normed * gate` flips
    // sign between the two unless `normed` itself is exactly zero.
    let sign_flip = silu_out
        .iter()
        .zip(&sigmoid_out)
        .any(|(&a, &b)| a.is_sign_negative() != b.is_sign_negative() && a != 0.0 && b != 0.0);
    assert!(
        sign_flip,
        "expected at least one output to flip sign between silu and sigmoid at z=-6"
    );
}
