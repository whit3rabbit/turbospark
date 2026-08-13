//! Runs `dequant_int1_gemv_simd` and `dequant_int1_gemv_symmetric_simd` on
//! real Metal hardware against the CPU reference in
//! `turbospark_compute::quant_1bit` (ROADMAP's 1-bit entry, step 2). The
//! kernel rule: no quant kernel is trusted before this file exists and
//! passes.
//!
//! The reference on the other side of these comparisons is itself pinned to
//! an MLX oracle (`crates/compute/tests/quant_1bit.rs`), so parity against
//! it transitively pins the kernel's bit order and companion dtype -- but
//! ONLY on cases where a wrong reading would move the answer. At one bit
//! that is not automatic: a wrong bit order permutes elements within an
//! 8-element run and leaves every magnitude, every group scale and the total
//! popcount untouched. So the two cases written for those traps construct
//! data that discriminates and then ASSERT that it discriminates, rather
//! than trusting a general tolerance to notice.
#![cfg(target_os = "macos")]

use half::f16;
use turbospark_compute::quant_1bit::{
    dequantize_int1_affine, f32_to_f16, quantize_int1_affine_symmetric, Int1AffineRow,
    BONSAI_GROUP_SIZE,
};
use turbospark_gpu::{
    dequant_int1_gemv, dequant_int1_gemv_resident, dequant_int1_gemv_symmetric, int1_row_bytes,
    Int1AffineRowGpu, Int1ResidentMatrix, Int1SymmetricRowGpu, MetalContext,
};

fn pseudo(n: usize, seed: u32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((s >> 8) as f32 / (1u32 << 23) as f32) - 1.0
        })
        .collect()
}

fn to_f16(v: &[f32]) -> Vec<f16> {
    v.iter().map(|&x| f16::from_f32(x)).collect()
}

fn gpu_rows(rows: &[Int1AffineRow]) -> Vec<Int1AffineRowGpu<'_>> {
    rows.iter()
        .map(|r| Int1AffineRowGpu {
            packed: &r.packed,
            scales: &r.scales,
            biases: &r.biases,
        })
        .collect()
}

/// The CPU reference and the GPU are handed the SAME already-quantized
/// bytes, so quantization error cancels and what is left is FP16 rounding of
/// the result plus reduction order (the kernel factors the affine form per
/// byte and reduces with `simd_sum`; the reference multiplies out in element
/// order). Anything above that is a kernel bug.
fn assert_matches(label: &str, gpu: &[f16], cpu: &[f32], n: usize) {
    assert_eq!(gpu.len(), cpu.len());
    let gpu_f32: Vec<f32> = gpu.iter().map(|v| v.to_f32()).collect();
    let err = turbospark_compute::max_abs_diff(&gpu_f32, cpu);
    let scale = cpu.iter().fold(0f32, |m, &v| m.max(v.abs())).max(1.0);
    let bound = scale * 1e-2 + (n as f32) * 1e-4;
    assert!(err < bound, "{label}: err = {err}, bound = {bound}");
}

#[test]
fn matches_cpu_reference() {
    let mut context = MetalContext::new().expect("Metal device available on this machine");

    let g = BONSAI_GROUP_SIZE;
    let n = 4 * g; // four groups, so a group-stride bug is reachable
    let m = 5usize; // not a multiple of 8: exercises the early-return guard
    let rows: Vec<Int1AffineRow> = (0..m)
        .map(|r| quantize_int1_affine_symmetric(&pseudo(n, 7 + r as u32), g))
        .collect();

    let x_f32 = pseudo(n, 99);
    let cpu = turbospark_compute::dequant_int1_gemv(&rows, &x_f32, n);
    let gpu = dequant_int1_gemv(&mut context, &gpu_rows(&rows), &to_f16(&x_f32), n, g)
        .expect("GPU dispatch succeeds");
    assert_matches("m=5,n=512,g=128", &gpu, &cpu, n);
}

/// The bit order inside a byte is LSB-first, and this case is built so that
/// reversing it moves the answer decisively.
///
/// A row's weights are only ever `+/- scale/2`, so a permutation within an
/// 8-element run changes nothing about their multiset. What it changes is
/// WHICH activation each one multiplies. The row here sets exactly the low
/// four bits of every byte, and `x` alternates a large positive value over
/// the low half of each run with a large negative one over the high half, so
/// LSB-first and MSB-first land on opposite signs. The reference's own bit
/// order is pinned to MLX by `crates/compute/tests/quant_1bit.rs`.
#[test]
fn the_bit_order_is_lsb_first() {
    let mut context = MetalContext::new().expect("Metal device available on this machine");

    let g = BONSAI_GROUP_SIZE;
    let n = 2 * g;
    let row = Int1AffineRow {
        // 0x0F: bits 0..3 set, 4..7 clear.
        packed: vec![0x0Fu8; n / 8],
        scales: vec![f32_to_f16(1.0); n / g],
        biases: vec![f32_to_f16(-0.5); n / g],
        group_size: g,
    };
    // +1 on the four elements a low bit addresses, -1 on the other four.
    let x_f32: Vec<f32> = (0..n)
        .map(|i| if i % 8 < 4 { 1.0f32 } else { -1.0 })
        .collect();

    let cpu = turbospark_compute::dequant_int1_gemv(std::slice::from_ref(&row), &x_f32, n);

    let mut flipped = row.clone();
    flipped.packed = flipped.packed.iter().map(|b| b.reverse_bits()).collect();
    let cpu_flipped = turbospark_compute::dequant_int1_gemv(&[flipped], &x_f32, n);
    assert!(
        (cpu[0] - cpu_flipped[0]).abs() > 0.5 * cpu[0].abs().max(1.0),
        "the fixture cannot tell the two bit orders apart: {} vs {}",
        cpu[0],
        cpu_flipped[0]
    );

    let gpu = dequant_int1_gemv(&mut context, &gpu_rows(&[row]), &to_f16(&x_f32), n, g)
        .expect("GPU dispatch succeeds");
    assert_matches("lsb-first", &gpu, &cpu, n);
}

/// The companions are read as FP16, not BF16.
///
/// The two planes are the same width, so nothing structural catches a
/// misread and the magnitudes are what tell them apart: this checkpoint's
/// scales sit near 0.027, whose FP16 bit pattern read as BF16 is ~1.7e-16.
/// So a kernel binding `device const bfloat*` returns approximately zero
/// here, which the tolerance alone would accept on a small result. The
/// answer is asserted decisively non-zero as well as matching.
#[test]
fn the_companions_are_read_as_fp16_not_bf16() {
    let mut context = MetalContext::new().expect("Metal device available on this machine");

    let g = BONSAI_GROUP_SIZE;
    let n = 2 * g;
    let scale = 0.0271f32; // the real checkpoint's magnitude
    let row = Int1AffineRow {
        packed: vec![0xFFu8; n / 8], // every weight at +scale/2
        scales: vec![f32_to_f16(scale); n / g],
        biases: vec![f32_to_f16(-scale / 2.0); n / g],
        group_size: g,
    };
    let x_f32 = vec![1.0f32; n];

    let as_bf16 = f32::from_bits((row.scales[0] as u32) << 16);
    assert!(
        as_bf16 < 1e-10,
        "the fixture's scale reads the same either way: {as_bf16}"
    );

    let cpu = turbospark_compute::dequant_int1_gemv(std::slice::from_ref(&row), &x_f32, n);
    let gpu = dequant_int1_gemv(&mut context, &gpu_rows(&[row]), &to_f16(&x_f32), n, g)
        .expect("GPU dispatch succeeds");

    assert!(
        cpu[0] > 3.0,
        "the reference is not decisively non-zero: {}",
        cpu[0]
    );
    assert!(
        gpu[0].to_f32() > 1.0,
        "the kernel read the FP16 companions as BF16: got {} against {}",
        gpu[0].to_f32(),
        cpu[0]
    );
    assert_matches("fp16 companions", &gpu, &cpu, n);
}

/// Every group is decoded with its own scale and bias.
///
/// A kernel that resolves the pair once per row, or strides the group index
/// wrongly, passes the single-group case and fails here: the groups differ
/// in magnitude by orders of magnitude rather than by rounding, which is
/// also what makes the discriminating assertion below non-vacuous.
#[test]
fn per_group_scales_are_not_hoisted() {
    let mut context = MetalContext::new().expect("Metal device available on this machine");

    let g = BONSAI_GROUP_SIZE;
    let n = 8 * g;
    let m = 8usize;
    let rows: Vec<Int1AffineRow> = (0..m)
        .map(|r| {
            let src: Vec<f32> = pseudo(n, 41 + r as u32)
                .iter()
                .enumerate()
                .map(|(i, &w)| w * 10f32.powi((i / g) as i32 % 5 - 2))
                .collect();
            quantize_int1_affine_symmetric(&src, g)
        })
        .collect();

    let spread = rows[0]
        .scales
        .iter()
        .map(|&s| turbospark_compute::quant_1bit::f16_to_f32(s))
        .fold((f32::MAX, 0f32), |(lo, hi), s| (lo.min(s), hi.max(s)));
    assert!(
        spread.1 > 100.0 * spread.0,
        "the fixture's group scales are too close to catch a hoist: {spread:?}"
    );

    let x_f32 = pseudo(n, 5);
    let cpu = turbospark_compute::dequant_int1_gemv(&rows, &x_f32, n);
    let gpu = dequant_int1_gemv(&mut context, &gpu_rows(&rows), &to_f16(&x_f32), n, g)
        .expect("GPU dispatch succeeds");
    assert_matches("8 groups, varying scales", &gpu, &cpu, n);
}

/// The group size is a parameter, not the 128 the real checkpoint declares.
///
/// The kernel takes it as a runtime uniform (see the shader header for why
/// it is not a function constant), so this dispatches the same shape at 64
/// and asserts the two group sizes do not agree -- which is what says the
/// uniform is actually read.
#[test]
fn the_group_size_is_a_parameter_not_a_constant() {
    let mut context = MetalContext::new().expect("Metal device available on this machine");

    let n = 256usize; // two groups at 128, four at 64
    let src = pseudo(n, 21);
    let at_64 = quantize_int1_affine_symmetric(&src, 64);
    let x_f32 = pseudo(n, 22);

    let cpu = turbospark_compute::dequant_int1_gemv(std::slice::from_ref(&at_64), &x_f32, n);
    let gpu = dequant_int1_gemv(
        &mut context,
        &gpu_rows(std::slice::from_ref(&at_64)),
        &to_f16(&x_f32),
        n,
        64,
    )
    .expect("GPU dispatch succeeds");
    assert_matches("g=64", &gpu, &cpu, n);

    // The same bytes read at the wrong group size take each pair of groups'
    // scales from the first of the pair, which is a different answer.
    let mut mislabelled = at_64.clone();
    mislabelled.group_size = 128;
    mislabelled.scales.truncate(2);
    mislabelled.biases.truncate(2);
    let wrong = turbospark_compute::dequant_int1_gemv(&[mislabelled], &x_f32, n);
    assert!(
        (wrong[0] - cpu[0]).abs() > 1e-3 * cpu[0].abs().max(1.0),
        "the fixture reads the same at both group sizes: {} vs {}",
        wrong[0],
        cpu[0]
    );
}

/// The `+/-1` kernel computes what the affine one does on a symmetric row,
/// to FP32 rounding rather than exactly (factoring the scale out of the
/// group reassociates the sum).
#[test]
fn the_symmetric_form_agrees_with_the_affine_one() {
    let mut context = MetalContext::new().expect("Metal device available on this machine");

    let g = BONSAI_GROUP_SIZE;
    let n = 4 * g;
    let m = 6usize;
    let rows: Vec<Int1AffineRow> = (0..m)
        .map(|r| quantize_int1_affine_symmetric(&pseudo(n, 61 + r as u32), g))
        .collect();
    assert!(rows
        .iter()
        .all(turbospark_compute::quant_1bit::is_symmetric));

    let x_f32 = pseudo(n, 62);
    let cpu = turbospark_compute::dequant_int1_gemv_symmetric(&rows, &x_f32, n);

    let sym: Vec<Int1SymmetricRowGpu<'_>> = rows
        .iter()
        .map(|r| Int1SymmetricRowGpu {
            packed: &r.packed,
            scales: &r.scales,
        })
        .collect();
    let gpu = dequant_int1_gemv_symmetric(&mut context, &sym, &to_f16(&x_f32), n, g)
        .expect("GPU dispatch succeeds");
    assert_matches("symmetric vs affine", &gpu, &cpu, n);

    // ...and it agrees with the GENERAL kernel on the same bytes, which is
    // the property that licenses selecting it per tensor.
    let affine = dequant_int1_gemv(&mut context, &gpu_rows(&rows), &to_f16(&x_f32), n, g)
        .expect("GPU dispatch succeeds");
    let affine_f32: Vec<f32> = affine.iter().map(|v| v.to_f32()).collect();
    assert_matches("symmetric vs general kernel", &gpu, &affine_f32, n);
}

/// The `+/-1` kernel is a claim about the DATA, not a general fast path.
///
/// It ignores the bias plane entirely, so on a row whose groups are not
/// `bias == -scale/2` it returns a different answer from the affine kernel.
/// Asserted rather than assumed, because the reason
/// [`Int1SymmetricRowGpu`] carries no bias field at all is exactly this: a
/// caller must establish the symmetry before it can reach this kernel.
#[test]
fn the_symmetric_form_is_not_a_general_fast_path() {
    let mut context = MetalContext::new().expect("Metal device available on this machine");

    let g = BONSAI_GROUP_SIZE;
    let n = 2 * g;
    let mut row = quantize_int1_affine_symmetric(&pseudo(n, 71), g);
    // Move one group's bias off `-scale/2`: legal in the container, and the
    // two values it now represents are same-signed.
    row.biases[1] = f32_to_f16(0.0);
    assert_eq!(
        turbospark_compute::quant_1bit::asymmetric_group_count(&row),
        1
    );

    let x_f32 = vec![1.0f32; n];
    let affine = dequant_int1_gemv(
        &mut context,
        &gpu_rows(&[row.clone()]),
        &to_f16(&x_f32),
        n,
        g,
    )
    .expect("GPU dispatch succeeds");
    let sym = dequant_int1_gemv_symmetric(
        &mut context,
        &[Int1SymmetricRowGpu {
            packed: &row.packed,
            scales: &row.scales,
        }],
        &to_f16(&x_f32),
        n,
        g,
    )
    .expect("GPU dispatch succeeds");

    let (a, s) = (affine[0].to_f32(), sym[0].to_f32());
    assert!(
        (a - s).abs() > 1e-3 * a.abs().max(1.0),
        "the two kernels agreed on an asymmetric row: {a} vs {s}"
    );
    // The affine one is the correct answer here; the CPU reference agrees.
    let cpu = turbospark_compute::dequant_int1_gemv(&[row], &x_f32, n);
    assert_matches("affine on an asymmetric row", &affine, &cpu, n);
}

/// The offset-bound form has to land on the same answer as the copying one.
/// Its whole purpose is reading weights in place out of the resident
/// mapping, and an offset bug there reads a neighbouring tensor: finite,
/// plausible, and wrong. The three planes are at three different offsets
/// inside one buffer, none of them zero.
#[test]
fn the_resident_form_matches_the_copying_form() {
    let mut context = MetalContext::new().expect("Metal device available on this machine");

    let g = BONSAI_GROUP_SIZE;
    let n = 4 * g;
    let m = 4usize;
    let rows: Vec<Int1AffineRow> = (0..m)
        .map(|r| quantize_int1_affine_symmetric(&pseudo(n, 13 + r as u32), g))
        .collect();

    let pad = 4096usize;
    let mut blob = vec![0xABu8; pad];
    let weights_offset = blob.len() as u64;
    for r in &rows {
        blob.extend_from_slice(&r.packed);
    }
    blob.extend_from_slice(&[0xCDu8; 256]);
    let scales_offset = blob.len() as u64;
    for r in &rows {
        for &s in &r.scales {
            blob.extend_from_slice(&s.to_le_bytes());
        }
    }
    let biases_offset = blob.len() as u64;
    for r in &rows {
        for &b in &r.biases {
            blob.extend_from_slice(&b.to_le_bytes());
        }
    }
    let buffer = context.new_buffer_with_data(&blob);

    let x_f32 = pseudo(n, 77);
    let cpu = turbospark_compute::dequant_int1_gemv(&rows, &x_f32, n);
    let gpu = dequant_int1_gemv_resident(
        &mut context,
        &Int1ResidentMatrix {
            buffer: &buffer,
            weights_offset,
            scales_offset,
            biases_offset,
            rows: m,
            cols: n,
            group_size: g,
        },
        &to_f16(&x_f32),
    )
    .expect("GPU dispatch succeeds");

    assert_eq!(int1_row_bytes(n) * m, (scales_offset - 256) as usize - pad);
    assert_matches("resident at three offsets", &gpu, &cpu, n);
}

/// A row shorter than the 32 bytes a SIMD group has lanes for.
///
/// One group of 128 elements is 16 bytes, so half the lanes find nothing to
/// do and must contribute exactly zero rather than reading past the row.
/// The dequantized weights are checked against the reference too, so a
/// kernel that read a neighbouring row's bytes would move the answer.
#[test]
fn a_row_shorter_than_the_simd_width_is_handled() {
    let mut context = MetalContext::new().expect("Metal device available on this machine");

    let g = BONSAI_GROUP_SIZE;
    let n = g; // 16 bytes against 32 lanes
    let rows: Vec<Int1AffineRow> = (0..3)
        .map(|r| quantize_int1_affine_symmetric(&pseudo(n, 91 + r as u32), g))
        .collect();
    assert_eq!(int1_row_bytes(n), 16);

    let x_f32 = pseudo(n, 92);
    let cpu = turbospark_compute::dequant_int1_gemv(&rows, &x_f32, n);
    let gpu = dequant_int1_gemv(&mut context, &gpu_rows(&rows), &to_f16(&x_f32), n, g)
        .expect("GPU dispatch succeeds");
    assert_matches("n=128, one group", &gpu, &cpu, n);

    // Non-vacuous: the rows decode to genuinely different values, so an
    // out-of-range lane reading row r+1 would show up.
    let w0 = dequantize_int1_affine(&rows[0], n);
    let w1 = dequantize_int1_affine(&rows[1], n);
    assert_ne!(w0, w1);
}
