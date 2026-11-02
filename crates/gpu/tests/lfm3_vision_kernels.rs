//! The LFM3-VL tower's two new Metal kernels on real hardware: the gated
//! depthwise 3x3 conv block core and SiLU.
//!
//! The conv reference is written the way the MLX model is: un-permute the
//! window-ordered tokens to a `height x width` grid, run an explicit zero
//! padded depthwise 3x3 correlation, and permute back. The kernel instead
//! gathers neighbours straight from the window-ordered rows, so the two share
//! no indexing code. Each case is paired with a check that the fixture can
//! tell the kernel from its plausible wrong twin (a flipped kernel).
#![cfg(target_os = "macos")]

use half::f16;
use turbospark_gpu::{encode_vision_gated_dwconv3x3, encode_vision_silu, MetalContext};

fn half_of(values: &[f32]) -> Vec<f16> {
    values.iter().map(|&v| f16::from_f32(v)).collect()
}

fn rounded(values: &[f32]) -> Vec<f32> {
    half_of(values).iter().map(|v| v.to_f32()).collect()
}

fn wave(n: usize, phase: f32, amp: f32) -> Vec<f32> {
    (0..n)
        .map(|i| ((i as f32) * 0.37 + phase).sin() * amp)
        .collect()
}

fn read_f16(buffer: &metal::Buffer, len: usize) -> Vec<f32> {
    let ptr = buffer.contents() as *const u16;
    // The buffer holds at least `len` halves and its pass has been waited on.
    let bits = unsafe { std::slice::from_raw_parts(ptr, len) };
    bits.iter().map(|&b| f16::from_bits(b).to_f32()).collect()
}

/// Grid cell `(row, col)` of window-ordered token `t`.
fn cell(t: usize, width: usize, merge: usize) -> (usize, usize) {
    let (win, inside) = (t / (merge * merge), t % (merge * merge));
    let cols = width / merge;
    (
        (win / cols) * merge + inside / merge,
        (win % cols) * merge + inside % merge,
    )
}

/// The reference. `flip` reverses the taps (the wrong twin).
fn reference(
    proj: &[f32],
    w: &[f32],
    height: usize,
    width: usize,
    merge: usize,
    channels: usize,
    flip: bool,
) -> Vec<f32> {
    let seq = height * width;
    let stride = 3 * channels;
    // Un-permute b*v onto the grid.
    let mut grid = vec![0.0f32; seq * channels];
    for t in 0..seq {
        let (r, c) = cell(t, width, merge);
        for ch in 0..channels {
            grid[(r * width + c) * channels + ch] =
                proj[t * stride + ch] * proj[t * stride + 2 * channels + ch];
        }
    }
    let mut out = vec![0.0f32; seq * channels];
    for t in 0..seq {
        let (r, c) = cell(t, width, merge);
        for ch in 0..channels {
            let mut acc = 0.0f32;
            for dr in 0..3usize {
                for dc in 0..3usize {
                    let (rr, cc) = (r as i64 + dr as i64 - 1, c as i64 + dc as i64 - 1);
                    if rr < 0 || cc < 0 || rr >= height as i64 || cc >= width as i64 {
                        continue;
                    }
                    let tap = if flip {
                        (2 - dr) * 3 + (2 - dc)
                    } else {
                        dr * 3 + dc
                    };
                    acc +=
                        w[ch * 9 + tap] * grid[(rr as usize * width + cc as usize) * channels + ch];
                }
            }
            out[t * channels + ch] = proj[t * stride + channels + ch] * acc;
        }
    }
    out
}

fn run(proj: &[f32], w: &[f32], height: usize, width: usize, merge: usize, ch: usize) -> Vec<f32> {
    let mut context = MetalContext::new().expect("Metal device");
    let pb = context.new_buffer_with_data(&half_of(proj));
    let wb = context.new_buffer_with_data(&half_of(w));
    let ob = context.new_output_buffer((height * width * ch * 2) as u64);
    let pass = context.begin_pass();
    encode_vision_gated_dwconv3x3(
        &mut context,
        &pass,
        (&ob, 0),
        (&pb, 0),
        (&wb, 0),
        height as u32,
        width as u32,
        merge as u32,
        ch as u32,
    )
    .expect("encode");
    pass.commit_and_wait();
    read_f16(&ob, height * width * ch)
}

fn worst(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).fold(0.0f32, |m, (x, y)| {
        assert!(x.is_finite() && y.is_finite(), "non-finite value");
        m.max((x - y).abs())
    })
}

fn case(height: usize, width: usize, merge: usize, ch: usize) {
    let seq = height * width;
    let proj = wave(seq * 3 * ch, 0.2, 1.5);
    let w = wave(ch * 9, 1.1, 0.8);
    let (pr, wr) = (rounded(&proj), rounded(&w));
    let got = run(&proj, &w, height, width, merge, ch);
    let want = reference(&pr, &wr, height, width, merge, ch, false);
    let scale = want.iter().fold(1.0f32, |m, v| m.max(v.abs()));
    let err = worst(&got, &want);
    assert!(
        err <= 2e-3 * scale,
        "{height}x{width} merge {merge} ch {ch}: worst {err} against bound {}",
        2e-3 * scale
    );
    // The fixture distinguishes the kernel from a flipped-tap twin.
    let flipped = reference(&pr, &wr, height, width, merge, ch, true);
    assert!(
        worst(&got, &flipped) > 20.0 * err.max(1e-6),
        "{height}x{width}: the fixture cannot tell a flipped kernel from the real one"
    );
}

#[test]
fn conv_matches_the_grid_reference_on_a_square_grid() {
    case(8, 8, 2, 16);
}

#[test]
fn conv_matches_the_grid_reference_on_a_tall_grid() {
    case(12, 4, 2, 8);
}

#[test]
fn conv_matches_the_grid_reference_on_a_wide_grid_with_odd_block_count() {
    case(4, 14, 2, 24);
}

#[test]
fn a_grid_off_the_merge_window_is_refused() {
    let mut context = MetalContext::new().expect("Metal device");
    let b = context.new_output_buffer(1024);
    let pass = context.begin_pass();
    let err =
        encode_vision_gated_dwconv3x3(&mut context, &pass, (&b, 0), (&b, 0), (&b, 0), 5, 4, 2, 4);
    assert!(err.is_err());
    pass.commit_and_wait();
}

#[test]
fn silu_matches_the_formula_and_differs_from_gelu() {
    let n = 1000usize;
    let x = wave(n, 0.3, 6.0);
    let mut context = MetalContext::new().expect("Metal device");
    let xb = context.new_buffer_with_data(&half_of(&x));
    let pass = context.begin_pass();
    encode_vision_silu(&mut context, &pass, (&xb, 0), n as u32).expect("encode");
    pass.commit_and_wait();
    let got = read_f16(&xb, n);
    let want: Vec<f32> = rounded(&x).iter().map(|v| v / (1.0 + (-v).exp())).collect();
    let err = worst(&got, &want);
    assert!(err < 4e-3, "silu worst error {err}");
    // A tanh-GELU twin differs by far more than the bound on this range.
    let gelu: Vec<f32> = rounded(&x)
        .iter()
        .map(|v| 0.5 * v * (1.0 + (0.797_884_6 * (v + 0.044_715 * v * v * v)).tanh()))
        .collect();
    assert!(worst(&got, &gelu) > 10.0 * err);
}
