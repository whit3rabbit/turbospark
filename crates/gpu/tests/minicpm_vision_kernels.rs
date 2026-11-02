//! The MiniCPM-V window merger's residual kernel on real Metal
//! (`docs/MINICPMV_VISION_PHASE0.md`): `y[g] += mean of rows 4g .. 4g+3 of x`.
#![cfg(target_os = "macos")]

use half::f16;
use turbospark_gpu::{encode_vision_window_mean4_add, MetalContext};

fn read_f16(buffer: &metal::Buffer, len: usize) -> Vec<f32> {
    let ptr = buffer.contents() as *const u16;
    // The buffer holds at least `len` halves and its pass has been waited on.
    let bits = unsafe { std::slice::from_raw_parts(ptr, len) };
    bits.iter().map(|&b| f16::from_bits(b).to_f32()).collect()
}

fn half_of(values: &[f32]) -> Vec<f16> {
    values.iter().map(|&v| f16::from_f32(v)).collect()
}

fn rounded(values: &[f32]) -> Vec<f32> {
    half_of(values).iter().map(|v| v.to_f32()).collect()
}

fn run(groups: usize, width: usize, y: &[f32], x: &[f32]) -> Vec<f32> {
    let mut context = MetalContext::new().expect("Metal device");
    let yb = context.new_buffer_with_data(&half_of(y));
    let xb = context.new_buffer_with_data(&half_of(x));
    let pass = context.begin_pass();
    encode_vision_window_mean4_add(
        &mut context,
        &pass,
        (&yb, 0),
        (&xb, 0),
        groups as u32,
        width as u32,
    )
    .expect("dispatch");
    pass.commit_and_wait();
    read_f16(&yb, groups * width)
}

fn wave(n: usize, phase: f32) -> Vec<f32> {
    (0..n)
        .map(|i| ((i as f32) * 0.37 + phase).sin() * 3.0)
        .collect()
}

#[test]
fn each_window_adds_the_mean_of_its_own_four_rows() {
    // A width that is not a multiple of the threadgroup and a group count that
    // does not fill one either.
    let (groups, width) = (37usize, 1152usize);
    let y = wave(groups * width, 0.1);
    let x = wave(4 * groups * width, 0.9);
    let (yr, xr) = (rounded(&y), rounded(&x));
    let got = run(groups, width, &y, &x);
    let mut worst = 0.0f32;
    for g in 0..groups {
        for c in 0..width {
            let mean: f32 = (0..4).map(|k| xr[(4 * g + k) * width + c]).sum::<f32>() / 4.0;
            let want = yr[g * width + c] + mean;
            worst = worst.max((got[g * width + c] - want).abs());
        }
    }
    assert!(worst < 4e-3, "worst absolute error {worst}");
}

/// The fixture must tell the right grouping from the wrong one: a kernel that
/// averaged rows `g, g + groups, ...` (a transposed view) would also be finite
/// and plausible.
#[test]
fn the_window_is_four_consecutive_rows_not_a_strided_set() {
    let (groups, width) = (8usize, 16usize);
    let y = vec![0.0f32; groups * width];
    let x = wave(4 * groups * width, 0.4);
    let xr = rounded(&x);
    let got = run(groups, width, &y, &x);
    let strided: Vec<f32> = (0..groups * width)
        .map(|i| {
            let (g, c) = (i / width, i % width);
            (0..4)
                .map(|k| xr[(g + k * groups) * width + c])
                .sum::<f32>()
                / 4.0
        })
        .collect();
    let gap = got
        .iter()
        .zip(&strided)
        .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
    assert!(
        gap > 0.1,
        "the strided grouping agrees to {gap}; the fixture cannot tell them apart"
    );
}
