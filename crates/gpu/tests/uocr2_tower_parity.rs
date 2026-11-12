#![cfg(target_os = "macos")]
//! The Unlimited-OCR-2 tower kernels against `compute::uocr2_tower`. The
//! real-model stages are `crates/runtime/tests/unlimited_ocr2_tower_real.rs`.

use half::f16;
use turbospark_compute::uocr2_tower::{conv2d, quick_gelu, rel_pos_table, relpos_attention};
use turbospark_gpu::{
    encode_vision_attention_relpos, encode_vision_conv2d, encode_vision_quick_gelu, MetalContext,
    VisionConvShape,
};

fn noise(n: usize, seed: u32, range: f32) -> Vec<f32> {
    let mut x = seed.wrapping_mul(2_654_435_761) | 1;
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            ((x as f32 / u32::MAX as f32) * 2.0 - 1.0) * range
        })
        .collect()
}

fn buffer(ctx: &MetalContext, v: &[f32]) -> metal::Buffer {
    let bytes: Vec<u8> = v
        .iter()
        .flat_map(|x| f16::from_f32(*x).to_bits().to_le_bytes())
        .collect();
    ctx.new_buffer_with_data(&bytes)
}

/// The same FP16 rounding the kernel sees, so the reference shares it.
fn round(v: &[f32]) -> Vec<f32> {
    v.iter().map(|x| f16::from_f32(*x).to_f32()).collect()
}

fn read(b: &metal::Buffer, n: usize) -> Vec<f32> {
    turbospark_gpu::read_half_buffer(b, n)
        .iter()
        .map(|x| x.to_f32())
        .collect()
}

fn assert_close(got: &[f32], want: &[f32], tol: f32, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(g.is_finite(), "{what}[{i}] is not finite");
        assert!((g - w).abs() <= tol, "{what}[{i}]: got {g}, want {w}");
    }
}

#[test]
fn quick_gelu_matches() {
    let mut ctx = MetalContext::new().unwrap();
    let x = noise(1000, 7, 6.0);
    let y = buffer(&ctx, &x);
    let pass = ctx.begin_pass();
    encode_vision_quick_gelu(&mut ctx, &pass, (&y, 0), x.len() as u32).unwrap();
    pass.commit_and_wait();
    let want: Vec<f32> = round(&x).iter().map(|v| quick_gelu(*v)).collect();
    assert_close(&read(&y, x.len()), &want, 2e-3, "quick_gelu");
}

fn conv_case(shape: VisionConvShape, with_bias: bool) {
    let mut ctx = MetalContext::new().unwrap();
    let VisionConvShape {
        h,
        w,
        c_in,
        c_out,
        kh,
        kw,
        stride,
        pad,
    } = shape;
    let n = (h * w * c_in) as usize;
    let x = noise(n, 11, 1.0);
    let weight = noise((c_out * kh * kw * c_in) as usize, 13, 0.5);
    let bias = noise(c_out as usize, 17, 0.5);
    let (xb, wb, bb) = (buffer(&ctx, &x), buffer(&ctx, &weight), buffer(&ctx, &bias));
    let (oh, ow) = shape.output();
    let out = ctx.new_output_buffer(u64::from(oh * ow * c_out) * 2);
    let pass = ctx.begin_pass();
    encode_vision_conv2d(
        &mut ctx,
        &pass,
        (&xb, 0),
        (&wb, 0),
        with_bias.then_some((&bb, 0)),
        (&out, 0),
        shape,
    )
    .unwrap();
    pass.commit_and_wait();
    let rounded_bias = round(&bias);
    let (want, roh, row) = conv2d(
        &round(&x),
        (h as usize, w as usize, c_in as usize),
        &round(&weight),
        (c_out as usize, kh as usize, kw as usize),
        stride as usize,
        pad as usize,
        with_bias.then_some(&rounded_bias[..]),
    );
    assert_eq!((roh as u32, row as u32), (oh, ow));
    assert_close(&read(&out, want.len()), &want, 0.02, "conv2d");
}

#[test]
fn the_three_real_conv_shapes_match() {
    // Patch embed: stride equal to the kernel, with a bias.
    conv_case(
        VisionConvShape {
            h: 32,
            w: 48,
            c_in: 3,
            c_out: 8,
            kh: 16,
            kw: 16,
            stride: 16,
            pad: 0,
        },
        true,
    );
    // The neck's 3x3, and the downsample's stride 2, both zero padded.
    conv_case(
        VisionConvShape {
            h: 9,
            w: 7,
            c_in: 6,
            c_out: 5,
            kh: 3,
            kw: 3,
            stride: 1,
            pad: 1,
        },
        false,
    );
    conv_case(
        VisionConvShape {
            h: 8,
            w: 10,
            c_in: 6,
            c_out: 12,
            kh: 3,
            kw: 3,
            stride: 2,
            pad: 1,
        },
        false,
    );
}

#[test]
fn relpos_attention_matches_across_windows() {
    let mut ctx = MetalContext::new().unwrap();
    let (side, heads, head_dim, windows) = (6usize, 12usize, 64usize, 3usize);
    let seq = side * side * windows;
    let n = seq * heads * head_dim;
    let (q, k, v) = (noise(n, 1, 1.0), noise(n, 2, 1.0), noise(n, 3, 1.0));
    let rel_h = rel_pos_table(&noise((2 * side - 1) * head_dim, 4, 0.2), head_dim, side);
    let rel_w = rel_pos_table(&noise((2 * side - 1) * head_dim, 5, 0.2), head_dim, side);
    let scale = (head_dim as f32).powf(-0.5);
    let buffers = [&q, &k, &v, &rel_h, &rel_w].map(|b| buffer(&ctx, b));
    let out = ctx.new_output_buffer(n as u64 * 2);
    let pass = ctx.begin_pass();
    encode_vision_attention_relpos(
        &mut ctx,
        &pass,
        (&buffers[0], 0),
        (&buffers[1], 0),
        (&buffers[2], 0),
        (&buffers[3], 0),
        (&buffers[4], 0),
        (&out, 0),
        seq as u32,
        heads as u32,
        head_dim as u32,
        side as u32,
        scale,
    )
    .unwrap();
    pass.commit_and_wait();
    let want = relpos_attention(
        &round(&q),
        &round(&k),
        &round(&v),
        &round(&rel_h),
        &round(&rel_w),
        side,
        heads,
        head_dim,
        scale,
    );
    assert_close(&read(&out, n), &want, 2e-3, "relpos attention");
    // A bias the kernel ignored would pass only if it were too small to matter.
    let zero = vec![0.0; rel_h.len()];
    let unbiased = relpos_attention(
        &round(&q),
        &round(&k),
        &round(&v),
        &zero,
        &zero,
        side,
        heads,
        head_dim,
        scale,
    );
    let moved = want
        .iter()
        .zip(&unbiased)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(moved > 0.02, "the bias only moved the output {moved}");
}

#[test]
fn a_resampled_table_keeps_its_ends() {
    let rel: Vec<f32> = (0..127).flat_map(|i| [i as f32, -(i as f32)]).collect();
    let table = rel_pos_table(&rel, 2, 40);
    // (query 39, key 0) is the far end of the axis, (0, 39) the near end.
    let at = |q: usize, k: usize| (q * 40 + k) * 2;
    assert!(table[at(39, 0)] > 124.0, "{}", table[at(39, 0)]);
    assert!(table[at(0, 39)] < 2.5, "{}", table[at(0, 39)]);
    assert!(
        (table[at(7, 7)] - 63.0).abs() < 0.01,
        "the zero offset is the centre"
    );
}
