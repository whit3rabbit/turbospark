//! Parity for `silu_fp16`/`sigmoid_fp16` (`crates/gpu/src/utility.rs`),
//! the plain UNARY siblings `qwen4_exp`'s hyper-connection mix needs
//! (`docs/QWEN4_PHASE0.md` section 3) -- every other silu/sigmoid call
//! site in this port gates a SECOND buffer, so neither existing kernel
//! fits a bare `silu(x)` or `sigmoid(x)`. Against
//! `turbospark_compute::{silu, sigmoid}`, the same scalar functions
//! `GdnReference` already uses.

#![cfg(target_os = "macos")]

use half::f16;
use turbospark_compute::{sigmoid, silu};
use turbospark_gpu::{encode_sigmoid, encode_silu, MetalContext};

const N: usize = 37; // not a multiple of THREADS_PER_GROUP, deliberately

fn deterministic(seed: u64, n: usize) -> Vec<f32> {
    let mut state = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
    (0..n)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state % 4000) as f32 / 1000.0) - 2.0 // roughly [-2, 2)
        })
        .collect()
}

fn half_bytes(v: &[f16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_bits().to_le_bytes()).collect()
}

#[test]
fn silu_matches_the_cpu_reference() {
    let mut context = MetalContext::new().expect("Metal device");
    let input = deterministic(1, N);
    let f16s: Vec<f16> = input.iter().map(|&v| f16::from_f32(v)).collect();
    let buf = context.new_buffer_with_data(&half_bytes(&f16s));

    let pass = context.begin_pass();
    encode_silu(&mut context, &pass, (&buf, 0), N as u32).expect("encode");
    pass.commit_and_wait();

    let got: Vec<f32> = turbospark_gpu::read_buffer_f16(&buf, 0, N)
        .iter()
        .map(|h| h.to_f32())
        .collect();
    for (i, (&g, &v)) in got.iter().zip(f16s.iter()).enumerate() {
        let expected = silu(v.to_f32());
        let diff = (g - expected).abs();
        assert!(
            diff <= 2e-3_f32.max(expected.abs() * 1e-2),
            "elem {i}: got {g} want {expected}"
        );
    }
}

#[test]
fn sigmoid_matches_the_cpu_reference() {
    let mut context = MetalContext::new().expect("Metal device");
    let input = deterministic(2, N);
    let f16s: Vec<f16> = input.iter().map(|&v| f16::from_f32(v)).collect();
    let buf = context.new_buffer_with_data(&half_bytes(&f16s));

    let pass = context.begin_pass();
    encode_sigmoid(&mut context, &pass, (&buf, 0), N as u32).expect("encode");
    pass.commit_and_wait();

    let got: Vec<f32> = turbospark_gpu::read_buffer_f16(&buf, 0, N)
        .iter()
        .map(|h| h.to_f32())
        .collect();
    for (i, (&g, &v)) in got.iter().zip(f16s.iter()).enumerate() {
        let expected = sigmoid(v.to_f32());
        let diff = (g - expected).abs();
        assert!(diff <= 2e-3, "elem {i}: got {g} want {expected}");
    }
}

/// **DISCRIMINATION.** `silu` and `sigmoid` differ substantially away from
/// zero (`silu(2) = 1.7616`, `sigmoid(2) = 0.8808`), so a fixture that
/// happened to sit near zero (where `silu(x) ~= x/2` and both curves are
/// close) could not tell a swapped kernel from a correct one. This case
/// pins both at a fixed input away from zero and checks they disagree.
#[test]
fn silu_and_sigmoid_are_different_functions_away_from_zero() {
    let mut context = MetalContext::new().expect("Metal device");
    let value = f16::from_f32(2.0);
    let buf_a = context.new_buffer_with_data(&half_bytes(&[value; 4]));
    let buf_b = context.new_buffer_with_data(&half_bytes(&[value; 4]));

    let pass = context.begin_pass();
    encode_silu(&mut context, &pass, (&buf_a, 0), 4).expect("encode silu");
    encode_sigmoid(&mut context, &pass, (&buf_b, 0), 4).expect("encode sigmoid");
    pass.commit_and_wait();

    let a = turbospark_gpu::read_buffer_f16(&buf_a, 0, 4)[0].to_f32();
    let b = turbospark_gpu::read_buffer_f16(&buf_b, 0, 4)[0].to_f32();
    assert!(
        (a - b).abs() > 0.5,
        "silu({a}) and sigmoid({b}) should differ substantially"
    );
}
