//! Runs `whisper_conv1d3_gelu` on real Metal hardware against the CPU
//! reference in `turbospark_compute::whisper::conv1d3_gelu`, for both
//! front-end shapes (conv1: stride 1 pad 1; conv2: stride 2 pad 1).
#![cfg(target_os = "macos")]

use turbospark_gpu::{whisper_conv1d3_gelu, MetalContext};

/// Deterministic, INDEPENDENT pseudo-random values in `[-1, 1)` (splitmix64
/// on the flat index, the same fixture family as
/// `dequant_int4_gemv_parity.rs`): no period the kernel's channel/time
/// strides can alias against, bounded magnitude so f32 accumulation error
/// stays predictable.
fn unit(i: usize, salt: u64) -> f32 {
    let mut z = (i as u64)
        .wrapping_add(salt)
        .wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    ((z >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
}

fn run_case(name: &str, in_ch: usize, t_len: usize, out_ch: usize, stride: usize, pad: usize) {
    let mut context = MetalContext::new().expect("Metal device available on this machine");

    let input: Vec<f32> = (0..in_ch * t_len).map(|i| unit(i, 11)).collect();
    let weight: Vec<f32> = (0..out_ch * in_ch * 3).map(|i| unit(i, 22)).collect();
    let bias: Vec<f32> = (0..out_ch).map(|i| unit(i, 33) * 0.1).collect();

    let cpu = turbospark_compute::whisper::conv1d3_gelu(
        &input, in_ch, t_len, &weight, &bias, out_ch, stride, pad,
    );
    let gpu = whisper_conv1d3_gelu(
        &mut context,
        &input,
        &weight,
        &bias,
        in_ch,
        t_len,
        out_ch,
        stride,
        pad,
    )
    .expect("GPU dispatch succeeds");

    let out_t = (t_len + 2 * pad - 3) / stride + 1;
    assert_eq!(cpu.len(), out_ch * out_t);
    assert_eq!(gpu.len(), cpu.len());
    let mut max_err = 0.0f32;
    for (a, b) in cpu.iter().zip(&gpu) {
        max_err = max_err.max((a - b).abs());
    }
    // Error budget, measured: the CPU evaluates erf in f64 (A&S 7.1.26),
    // the GPU in f32, over a 3-term-per-channel f32 accumulation -- the
    // 384-channel conv2 case measures 6.9e-5 max. Every structural bug
    // this test exists to catch (wrong index, stride, pad sign, missing
    // sqrt-2 scaling) measured 0.08 or worse on the same fixtures, so
    // 2e-4 splits them by three orders of magnitude.
    assert!(max_err < 2e-4, "{name}: max abs diff {max_err}");
}

#[test]
fn parity_conv1_shape_stride1_pad1() {
    run_case("conv1", 80, 3000, 384, 1, 1);
}

#[test]
fn parity_conv2_shape_stride2_pad1() {
    run_case("conv2", 384, 3000, 384, 2, 1);
}

#[test]
fn parity_small_and_odd_shapes() {
    // Small enough to hand-check, odd lengths so the time tail is
    // exercised, and an in/out channel mismatch so no broadcasting
    // accident can pass.
    run_case("tiny", 3, 7, 2, 1, 1);
    run_case("odd", 5, 13, 4, 2, 1);
}
