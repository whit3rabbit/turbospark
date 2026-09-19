//! Runs `bf16_gemv_rows` on real Metal hardware against a CPU reference that
//! reads the SAME bf16 bit patterns -- the unquantized-matrix GEMV the
//! Bonsai-2 line's F32-shipped `in_proj_a`/`in_proj_b` needed. Proves the
//! shader compiles and that a resident BF16 tag is executable, which no
//! install before this line ever exercised.
#![cfg(target_os = "macos")]

use half::f16;
use turbospark_gpu::{
    encode_bf16_gemv_resident, read_buffer_f16, Bf16ResidentMatrix, MetalContext,
};

/// f32 -> bf16 by truncation; both sides of the parity read the SAME bits,
/// so the rounding mode is the test's choice, not a variable.
fn bf16_bits(v: f32) -> u16 {
    ((v.to_bits() >> 16) & 0xFFFF) as u16
}

fn f32_of_bits(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

#[test]
fn matches_cpu_reference_reading_the_same_bits() {
    let mut context = MetalContext::new().expect("Metal device available on this machine");

    let (m, n) = (48usize, 5120usize);
    let weights: Vec<u16> = (0..m * n)
        .map(|i| bf16_bits(((i % 37) as f32 - 18.0) * 0.11))
        .collect();
    let x: Vec<f16> = (0..n)
        .map(|i| f16::from_f32(((i % 23) as f32 - 11.0) * 0.07))
        .collect();

    let mut weight_bytes = Vec::with_capacity(m * n * 2);
    for w in &weights {
        weight_bytes.extend_from_slice(&w.to_le_bytes());
    }
    let w_buf = context.new_buffer_with_data(&weight_bytes);
    let x_buf = context.new_buffer_with_data(&x);
    let y_buf = context.new_output_buffer((m * 2) as u64);

    let matrix = Bf16ResidentMatrix {
        buffer: &w_buf,
        weights_offset: 0,
        rows: m,
        cols: n,
    };
    let pass = context.begin_pass();
    encode_bf16_gemv_resident(&mut context, &pass, &matrix, (&x_buf, 0), (&y_buf, 0))
        .expect("dispatch succeeds");
    pass.commit_and_wait();

    let gpu = read_buffer_f16(&y_buf, 0, m);
    let gpu: Vec<f32> = gpu.iter().map(|v| v.to_f32()).collect();
    let cpu: Vec<f32> = (0..m)
        .map(|r| {
            (0..n)
                .map(|c| f32_of_bits(weights[r * n + c]) * x[c].to_f32())
                .sum()
        })
        .collect();
    assert_eq!(gpu.len(), cpu.len());
    let err = turbospark_compute::max_abs_diff(&gpu, &cpu);
    assert!(
        err < turbospark_compute::Tolerance::FP16_REDUCTION,
        "err = {err}"
    );
}

/// A non-multiple-of-threadgroup row count, so the guard clause and the
/// partial last threadgroup are both on real hardware.
#[test]
fn handles_a_row_count_that_is_not_a_threadgroup_multiple() {
    let mut context = MetalContext::new().expect("Metal device available on this machine");

    let (m, n) = (7usize, 256usize);
    let weights: Vec<u16> = (0..m * n)
        .map(|i| bf16_bits(((i % 9) as f32 - 4.0) * 0.5))
        .collect();
    let x: Vec<f16> = (0..n)
        .map(|i| f16::from_f32(((i % 5) as f32 - 2.0) * 0.25))
        .collect();

    let mut weight_bytes = Vec::with_capacity(m * n * 2);
    for w in &weights {
        weight_bytes.extend_from_slice(&w.to_le_bytes());
    }
    let w_buf = context.new_buffer_with_data(&weight_bytes);
    let x_buf = context.new_buffer_with_data(&x);
    let y_buf = context.new_output_buffer((m * 2) as u64);

    let matrix = Bf16ResidentMatrix {
        buffer: &w_buf,
        weights_offset: 0,
        rows: m,
        cols: n,
    };
    let pass = context.begin_pass();
    encode_bf16_gemv_resident(&mut context, &pass, &matrix, (&x_buf, 0), (&y_buf, 0))
        .expect("dispatch succeeds");
    pass.commit_and_wait();

    let gpu = read_buffer_f16(&y_buf, 0, m);
    for (r, row) in gpu.iter().enumerate() {
        let cpu: f32 = (0..n)
            .map(|c| f32_of_bits(weights[r * n + c]) * x[c].to_f32())
            .sum();
        assert!(
            (row.to_f32() - cpu).abs() < turbospark_compute::Tolerance::FP16_REDUCTION,
            "row {r}: {} vs {cpu}",
            row.to_f32()
        );
    }
}
