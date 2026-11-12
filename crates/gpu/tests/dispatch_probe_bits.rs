#![cfg(target_os = "macos")]
//! The L1 dispatch probe (`TURBOSPARK_DISPATCH_PROBE`, evaluator item L1 in
//! `docs/MLX_SERVE_KERNELS.md`) inserts N in-place `scalar_mul_fp16` by 1.0
//! dispatches on a layer's FFN chain. It prices a chained dispatch only if
//! it changes no output bit, so this pins it on every f16 bit pattern:
//! subnormals (a fast-math flush would zero them), both zeros, infinities,
//! and the largest finite values. NaNs are excluded: a multiply may quiet a
//! signaling payload, and no layer output is NaN on a path that completes.

use turbospark_gpu::{encode_scalar_mul, read_buffer_bytes, MetalContext};

fn all_non_nan_patterns() -> Vec<u16> {
    (0..=u16::MAX)
        .filter(|&b| !half::f16::from_bits(b).is_nan())
        .collect()
}

fn run_probe(scalar: f32, n: usize) -> (Vec<u16>, Vec<u16>) {
    let mut context = MetalContext::new().expect("Metal device");
    let input = all_non_nan_patterns();
    let bytes: Vec<u8> = input.iter().flat_map(|b| b.to_le_bytes()).collect();
    let buf = context.new_buffer_with_data(&bytes);
    let pass = context.begin_pass();
    for _ in 0..n {
        encode_scalar_mul(&mut context, &pass, (&buf, 0), scalar, input.len() as u32)
            .expect("encode");
    }
    pass.commit_and_wait_with_gpu_time();
    let out: Vec<u16> = read_buffer_bytes(&buf, 0, bytes.len())
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect();
    (input, out)
}

#[test]
fn eight_chained_multiplies_by_one_keep_every_bit() {
    let (input, out) = run_probe(1.0, 8);
    let changed: Vec<(u16, u16)> = input
        .iter()
        .zip(&out)
        .filter(|(a, b)| a != b)
        .map(|(a, b)| (*a, *b))
        .take(8)
        .collect();
    assert!(
        changed.is_empty(),
        "first changed patterns (in, out): {changed:04x?}"
    );
}

/// The test's teeth: a multiply that is not by exactly one changes bits,
/// so the comparison above can see a probe that perturbs its input.
#[test]
fn a_scale_that_is_not_one_is_caught() {
    let (input, out) = run_probe(1.0 + 1.0 / 1024.0, 1);
    assert!(input.iter().zip(&out).any(|(a, b)| a != b));
}
