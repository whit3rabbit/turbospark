//! The three kernels DeepSeek-OCR's SAM encoder added to `vision.metal`
//! (`docs/DEEPSEEK_OCR_VISION_PHASE0.md`), against naive CPU references written
//! here, on real hardware: the zero-sentinel row gather, im2col, and the SAM
//! attention with its decomposed relative-position bias.
//!
//! Each parity case has a DISCRIMINATION case: a fixture that cannot tell the
//! kernel from the plausible wrong kernel (a bias that is zero, a table whose
//! height and width halves are interchangeable) passes against both.
#![cfg(target_os = "macos")]

use half::f16;
use turbospark_gpu::{
    encode_vision_gather_rows, encode_vision_im2col, encode_vision_matmul,
    encode_vision_sam_attention, MetalContext, VISION_GATHER_ZERO,
};

fn to_le(values: &[f16]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|v| v.to_bits().to_le_bytes())
        .collect()
}

fn u32_le(values: &[u32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn read_f16(buffer: &metal::Buffer, len: usize) -> Vec<f32> {
    let ptr = buffer.contents() as *const u16;
    let bits = unsafe { std::slice::from_raw_parts(ptr, len) };
    bits.iter().map(|&b| f16::from_bits(b).to_f32()).collect()
}

/// A deterministic, non-flat spread, already rounded to FP16 so the CPU
/// reference and the kernel read the same numbers.
fn wave(n: usize, phase: f32, amp: f32) -> Vec<f32> {
    (0..n)
        .map(|i| f16::from_f32(((i as f32) * 0.37 + phase).sin() * amp).to_f32())
        .collect()
}

fn f16_of(values: &[f32]) -> Vec<f16> {
    values.iter().map(|&v| f16::from_f32(v)).collect()
}

fn worst(a: &[f32], b: &[f32]) -> (f32, usize) {
    let mut best = (0.0f32, 0usize);
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        let d = (x - y).abs();
        if d > best.0 {
            best = (d, i);
        }
    }
    best
}

// ------------------------------------------------------------------- gather

#[test]
fn gather_rows_copies_by_index_and_zeroes_the_sentinel() {
    let mut context = MetalContext::new().expect("Metal device");
    let (src_rows, cols) = (7usize, 13usize);
    let src = wave(src_rows * cols, 0.4, 3.0);
    let index: Vec<u32> = vec![6, VISION_GATHER_ZERO, 0, 3, VISION_GATHER_ZERO, 3, 5];
    let rows = index.len();

    let src_buf = context.new_buffer_with_data(&to_le(&f16_of(&src)));
    let idx_buf = context.new_buffer_with_data(&u32_le(&index));
    let out_buf = context.new_output_buffer((rows * cols * 2) as u64);
    let pass = context.begin_pass();
    encode_vision_gather_rows(
        &mut context,
        &pass,
        (&src_buf, 0),
        (&idx_buf, 0),
        (&out_buf, 0),
        rows as u32,
        cols as u32,
    )
    .expect("encode");
    pass.commit_and_wait();
    let gpu = read_f16(&out_buf, rows * cols);

    for (r, &i) in index.iter().enumerate() {
        for c in 0..cols {
            let want = if i == VISION_GATHER_ZERO {
                0.0
            } else {
                src[i as usize * cols + c]
            };
            assert_eq!(gpu[r * cols + c], want, "row {r} col {c}");
        }
    }
    // Discrimination: the zero rows really are rows a copy would have filled
    // with nonzero data (a buffer left at zero would also read as zero).
    assert!(src[..cols].iter().any(|v| *v != 0.0));
    assert!(gpu[cols..2 * cols].iter().all(|v| *v == 0.0));
}

// ------------------------------------------------------------------- im2col

/// Direct NHWC convolution, weight `[O, k, k, I]`, no bias.
fn conv_direct(
    img: &[f32],
    (h, w, c): (usize, usize, usize),
    weight: &[f32],
    o: usize,
    k: usize,
    stride: usize,
    pad: usize,
) -> Vec<f32> {
    let oh = (h + 2 * pad - k) / stride + 1;
    let ow = (w + 2 * pad - k) / stride + 1;
    let mut out = vec![0.0f32; oh * ow * o];
    for y in 0..oh {
        for x in 0..ow {
            for oc in 0..o {
                let mut acc = 0.0f32;
                for a in 0..k {
                    for b in 0..k {
                        let iy = (y * stride + a) as isize - pad as isize;
                        let ix = (x * stride + b) as isize - pad as isize;
                        if iy < 0 || ix < 0 || iy >= h as isize || ix >= w as isize {
                            continue;
                        }
                        for ic in 0..c {
                            acc += img[(iy as usize * w + ix as usize) * c + ic]
                                * weight[((oc * k + a) * k + b) * c + ic];
                        }
                    }
                }
                out[(y * ow + x) * o + oc] = acc;
            }
        }
    }
    out
}

fn run_conv_through_im2col(
    context: &mut MetalContext,
    img: &[f32],
    dims: (usize, usize, usize),
    weight: &[f32],
    o: usize,
    k: usize,
    stride: usize,
    pad: usize,
) -> Vec<f32> {
    let (h, w, c) = dims;
    let img_buf = context.new_buffer_with_data(&to_le(&f16_of(img)));
    let w_buf = context.new_buffer_with_data(&to_le(&f16_of(weight)));
    let oh = (h + 2 * pad - k) / stride + 1;
    let ow = (w + 2 * pad - k) / stride + 1;
    let cols_buf = context.new_output_buffer((oh * ow * k * k * c * 2) as u64);
    let out_buf = context.new_output_buffer((oh * ow * o * 2) as u64);
    let pass = context.begin_pass();
    let (got_h, got_w) = encode_vision_im2col(
        context,
        &pass,
        (&img_buf, 0),
        (&cols_buf, 0),
        h as u32,
        w as u32,
        c as u32,
        k as u32,
        stride as u32,
        pad as u32,
    )
    .expect("encode im2col");
    assert_eq!((got_h as usize, got_w as usize), (oh, ow));
    encode_vision_matmul(
        context,
        &pass,
        (&cols_buf, 0),
        (&w_buf, 0),
        None,
        (&out_buf, 0),
        (oh * ow) as u32,
        (k * k * c) as u32,
        o as u32,
    )
    .expect("encode matmul");
    pass.commit_and_wait();
    read_f16(&out_buf, oh * ow * o)
}

#[test]
fn a_conv_through_im2col_matches_a_direct_convolution() {
    let mut context = MetalContext::new().expect("Metal device");
    // The three convs the encoder runs, scaled down: 16-wide patchify (stride
    // = kernel, no pad), 3x3 pad 1, and 3x3 stride 2 pad 1, on a non-square
    // image so a height/width swap shows.
    for &(k, stride, pad) in &[(4usize, 4usize, 0usize), (3, 1, 1), (3, 2, 1)] {
        let (h, w, c, o) = (12usize, 8usize, 5usize, 6usize);
        let img = wave(h * w * c, 0.3, 2.0);
        let weight = wave(o * k * k * c, 1.1, 0.5);
        let want = conv_direct(&img, (h, w, c), &weight, o, k, stride, pad);
        let got =
            run_conv_through_im2col(&mut context, &img, (h, w, c), &weight, o, k, stride, pad);
        let (err, at) = worst(&got, &want);
        let bound = 4e-3 * want.iter().fold(1.0f32, |m, v| m.max(v.abs()));
        assert!(
            err <= bound,
            "k {k} stride {stride} pad {pad}: worst {err} at {at}: gpu {} cpu {}",
            got[at],
            want[at]
        );
    }
}

#[test]
fn im2col_discriminates_a_transposed_weight_layout() {
    // The matmul side of the contract: a weight stored [O, I, k, k] (the
    // PyTorch order) read as [O, k, k, I] gives a different answer, so the
    // case above is not satisfiable by a layout-agnostic kernel.
    let mut context = MetalContext::new().expect("Metal device");
    let (h, w, c, o, k) = (6usize, 6usize, 4usize, 3usize, 3usize);
    let img = wave(h * w * c, 0.9, 2.0);
    let weight = wave(o * k * k * c, 0.2, 0.7);
    let right = run_conv_through_im2col(&mut context, &img, (h, w, c), &weight, o, k, 1, 1);
    // Reinterpret the same bytes as [O, I, k, k] and permute them into the
    // [O, k, k, I] order a wrong reader would have assumed.
    let mut permuted = vec![0.0f32; weight.len()];
    for oc in 0..o {
        for ic in 0..c {
            for a in 0..k {
                for b in 0..k {
                    permuted[((oc * k + a) * k + b) * c + ic] =
                        weight[((oc * c + ic) * k + a) * k + b];
                }
            }
        }
    }
    let wrong = run_conv_through_im2col(&mut context, &img, (h, w, c), &permuted, o, k, 1, 1);
    let (gap, _) = worst(&right, &wrong);
    assert!(
        gap > 0.05,
        "layouts are indistinguishable on this fixture: {gap}"
    );
}

// ---------------------------------------------------------- SAM attention

/// Naive f32 reference of `vision_sam_attention_fp16`: per window, per head,
/// `softmax(scale * q.k + q.Rh[ih - jh + gh - 1] + q.Rw[iw - jw + gw - 1]) v`
/// with the bias from the unscaled query.
#[allow(clippy::too_many_arguments)]
fn sam_attention_ref(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    rel_h: &[f32],
    rel_w: &[f32],
    gh: usize,
    gw: usize,
    windows: usize,
    heads: usize,
    scale: f32,
) -> Vec<f32> {
    let d = 64usize;
    let keys = gh * gw;
    let width = heads * d;
    let mut out = vec![0.0f32; windows * keys * width];
    for win in 0..windows {
        for head in 0..heads {
            for i in 0..keys {
                let (ih, iw) = (i / gw, i % gw);
                let qrow = &q[((win * keys + i) * width + head * d)..][..d];
                let mut scores = vec![0.0f32; keys];
                for j in 0..keys {
                    let (jh, jw) = (j / gw, j % gw);
                    let krow = &k[((win * keys + j) * width + head * d)..][..d];
                    let dot: f32 = qrow.iter().zip(krow).map(|(a, b)| a * b).sum();
                    let rh = &rel_h[((ih + gh - 1 - jh) * d)..][..d];
                    let rw = &rel_w[((iw + gw - 1 - jw) * d)..][..d];
                    let bh: f32 = qrow.iter().zip(rh).map(|(a, b)| a * b).sum();
                    let bw: f32 = qrow.iter().zip(rw).map(|(a, b)| a * b).sum();
                    scores[j] = dot * scale + bh + bw;
                }
                let m = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0f32;
                for s in scores.iter_mut() {
                    *s = (*s - m).exp();
                    sum += *s;
                }
                for dd in 0..d {
                    let mut acc = 0.0f32;
                    for j in 0..keys {
                        acc += scores[j] * v[(win * keys + j) * width + head * d + dd];
                    }
                    out[(win * keys + i) * width + head * d + dd] = acc / sum;
                }
            }
        }
    }
    out
}

struct SamCase {
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    rel_h: Vec<f32>,
    rel_w: Vec<f32>,
}

fn sam_case(gh: usize, gw: usize, windows: usize, heads: usize, rel_amp: f32) -> SamCase {
    let n = windows * gh * gw * heads * 64;
    SamCase {
        q: wave(n, 0.1, 1.0),
        k: wave(n, 1.3, 1.0),
        v: wave(n, 2.2, 1.0),
        rel_h: wave((2 * gh - 1) * 64, 3.1, rel_amp),
        rel_w: wave((2 * gw - 1) * 64, 4.7, rel_amp),
    }
}

#[allow(clippy::too_many_arguments)]
fn run_sam_attention(
    context: &mut MetalContext,
    c: &SamCase,
    gh: usize,
    gw: usize,
    windows: usize,
    heads: usize,
    scale: f32,
) -> Vec<f32> {
    let n = windows * gh * gw * heads * 64;
    let buf = |ctx: &mut MetalContext, x: &[f32]| ctx.new_buffer_with_data(&to_le(&f16_of(x)));
    let (qb, kb, vb) = (buf(context, &c.q), buf(context, &c.k), buf(context, &c.v));
    let (rhb, rwb) = (buf(context, &c.rel_h), buf(context, &c.rel_w));
    let out = context.new_output_buffer((n * 2) as u64);
    let pass = context.begin_pass();
    encode_vision_sam_attention(
        context,
        &pass,
        (&qb, 0),
        (&kb, 0),
        (&vb, 0),
        (&rhb, 0),
        (&rwb, 0),
        (&out, 0),
        gh as u32,
        gw as u32,
        windows as u32,
        heads as u32,
        scale,
    )
    .expect("encode");
    pass.commit_and_wait();
    read_f16(&out, n)
}

fn assert_close(got: &[f32], want: &[f32], what: &str) {
    let (err, at) = worst(got, want);
    let bound = 3e-3 * want.iter().fold(1.0f32, |m, v| m.max(v.abs()));
    assert!(
        err <= bound,
        "{what}: worst {err} at {at}: gpu {} cpu {}",
        got[at],
        want[at]
    );
}

#[test]
fn sam_attention_matches_the_reference_on_non_square_windows() {
    let mut context = MetalContext::new().expect("Metal device");
    // Unequal sides so an h/w mix-up between the two tables shows; two windows
    // so a window-offset error shows; two heads so a head-stride error shows.
    let (gh, gw, windows, heads) = (3usize, 5usize, 2usize, 2usize);
    let scale = 64f32.powf(-0.5);
    let c = sam_case(gh, gw, windows, heads, 0.6);
    let want = sam_attention_ref(
        &c.q, &c.k, &c.v, &c.rel_h, &c.rel_w, gh, gw, windows, heads, scale,
    );
    let got = run_sam_attention(&mut context, &c, gh, gw, windows, heads, scale);
    assert_close(&got, &want, "3x5 windows");
}

#[test]
fn sam_attention_matches_the_reference_on_a_full_global_grid() {
    let mut context = MetalContext::new().expect("Metal device");
    // The global blocks: one window of 64 x 64 = 4096 keys, the score row's
    // limit, with the 127-row tables.
    let (gh, gw, windows, heads) = (64usize, 64usize, 1usize, 2usize);
    let scale = 64f32.powf(-0.5);
    let c = sam_case(gh, gw, windows, heads, 0.4);
    let want = sam_attention_ref(
        &c.q, &c.k, &c.v, &c.rel_h, &c.rel_w, gh, gw, windows, heads, scale,
    );
    let got = run_sam_attention(&mut context, &c, gh, gw, windows, heads, scale);
    assert_close(&got, &want, "64x64 global");
}

#[test]
fn sam_attention_discriminates_the_bias_and_its_orientation() {
    let mut context = MetalContext::new().expect("Metal device");
    let (gh, gw, windows, heads) = (3usize, 5usize, 1usize, 1usize);
    let scale = 64f32.powf(-0.5);
    let c = sam_case(gh, gw, windows, heads, 0.9);
    let right = run_sam_attention(&mut context, &c, gh, gw, windows, heads, scale);

    // No bias at all: the kernel must not be passing on the plain softmax.
    let mut zero = sam_case(gh, gw, windows, heads, 0.9);
    zero.rel_h = vec![0.0; zero.rel_h.len()];
    zero.rel_w = vec![0.0; zero.rel_w.len()];
    let plain = run_sam_attention(&mut context, &zero, gh, gw, windows, heads, scale);
    assert!(
        worst(&right, &plain).0 > 0.02,
        "the bias does not move the output"
    );

    // Reversed relative index (`jh - ih` instead of `ih - jh`): the table read
    // backwards. Build it by reversing both tables' rows, which is the same
    // as indexing them from the other end.
    let mut flipped = sam_case(gh, gw, windows, heads, 0.9);
    flipped.rel_h = flipped.rel_h.chunks(64).rev().flatten().copied().collect();
    flipped.rel_w = flipped.rel_w.chunks(64).rev().flatten().copied().collect();
    let wrong = run_sam_attention(&mut context, &flipped, gh, gw, windows, heads, scale);
    assert!(
        worst(&right, &wrong).0 > 0.02,
        "a reversed relative index is indistinguishable on this fixture"
    );

    // The bias is added after the scale: scaling q.k by a different `scale`
    // while leaving the bias must change the output (a kernel that folded the
    // scale over the bias too would not).
    let rescaled = run_sam_attention(&mut context, &c, gh, gw, windows, heads, scale * 2.0);
    assert!(worst(&right, &rescaled).0 > 0.02);
}

#[test]
fn sam_attention_refuses_a_grid_it_cannot_hold() {
    let mut context = MetalContext::new().expect("Metal device");
    let c = sam_case(2, 2, 1, 1, 0.1);
    let buf = |ctx: &mut MetalContext, x: &[f32]| ctx.new_buffer_with_data(&to_le(&f16_of(x)));
    let (qb, kb, vb) = (
        buf(&mut context, &c.q),
        buf(&mut context, &c.k),
        buf(&mut context, &c.v),
    );
    let (rhb, rwb) = (buf(&mut context, &c.rel_h), buf(&mut context, &c.rel_w));
    let out = context.new_output_buffer(2 * 4 * 64);
    for &(gh, gw) in &[(65u32, 1u32), (1, 65), (0, 4), (64, 65)] {
        let pass = context.begin_pass();
        let result = encode_vision_sam_attention(
            &mut context,
            &pass,
            (&qb, 0),
            (&kb, 0),
            (&vb, 0),
            (&rhb, 0),
            (&rwb, 0),
            (&out, 0),
            gh,
            gw,
            1,
            1,
            0.125,
        );
        assert!(result.is_err(), "{gh}x{gw} must be refused, not truncated");
        pass.commit_and_wait();
    }
}
