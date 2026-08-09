//! Runs `dequant_q6_k_gemv_simd` on real Metal hardware against the CPU
//! reference in `turbospark_compute::dequant_q6_k_gemv` (ROADMAP Phase G
//! Stage 2). The kernel rule: no quant kernel is trusted before this file
//! exists and passes.
//!
//! The cases here are chosen against Q6_K's own failure modes rather than
//! copied from the Q4_K file. Three of them stay finite and plausible when
//! wrong: dropping the two high bits in `qh` (values collapse to a sixteenth
//! of their range), dropping the bias of 32 (every weight turns positive),
//! and reading the sixteen sub-block scales as unsigned (whole 16-element
//! runs flip sign). A fourth, striding the sixteen scales by one instead of
//! two per quarter, mixes scales between quarters and is invisible unless the
//! sub-blocks differ in magnitude.
#![cfg(target_os = "macos")]

use half::f16;
use turbospark_gpu::{
    dequant_q6_k_gemv, dequant_q6_k_gemv_resident, q6_k_row_bytes, MetalContext, Q6KResidentMatrix,
};

const SUPERBLOCK: usize = 256;
const SUB: usize = 16;

fn weights(n: usize, seed: u32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((s >> 8) as f32 / (1u32 << 23) as f32) - 1.0
        })
        .collect()
}

/// The CPU reference and the GPU are handed the SAME already-quantized bytes,
/// so quantization error cancels and what is left is FP16 rounding and
/// reduction order. Anything above that is a kernel bug.
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

    let n = 2 * SUPERBLOCK;
    let m = 5usize; // not a multiple of 8: exercises the early-return guard
    let rows: Vec<Vec<u8>> = (0..m)
        .map(|r| turbospark_compute::quantize_q6_k(&weights(n, 7 + r as u32)))
        .collect();
    let refs: Vec<&[u8]> = rows.iter().map(|r| r.as_slice()).collect();

    let x_f32 = weights(n, 99);
    let x_f16: Vec<f16> = x_f32.iter().map(|&v| f16::from_f32(v)).collect();

    let cpu = turbospark_compute::dequant_q6_k_gemv(&refs, &x_f32, n);
    let gpu = dequant_q6_k_gemv(&mut context, &refs, &x_f16, n).expect("GPU dispatch succeeds");
    assert_matches("m=5,n=512", &gpu, &cpu, n);
}

/// Weights pushed hard against the ends of the 6-bit range, so most levels
/// need their two `qh` bits. A kernel reading only `ql` returns a row whose
/// every element is inside a sixteenth of the true range: finite, correctly
/// signed for the low quarter, and wrong everywhere else.
#[test]
fn the_high_two_bits_come_from_the_second_run() {
    let mut context = MetalContext::new().expect("Metal device available on this machine");

    let n = SUPERBLOCK;
    let m = 8usize;
    let rows: Vec<Vec<u8>> = (0..m)
        .map(|r| {
            // A sawtooth across the whole sub-block range visits every level
            // from -32 to 31, so three quarters of them carry a non-zero
            // high bit pair.
            let w: Vec<f32> = (0..n)
                .map(|e| {
                    let level = ((e + r) % 64) as f32 - 32.0;
                    level / 32.0
                })
                .collect();
            turbospark_compute::quantize_q6_k(&w)
        })
        .collect();
    let refs: Vec<&[u8]> = rows.iter().map(|r| r.as_slice()).collect();

    let x_f32 = weights(n, 5);
    let x_f16: Vec<f16> = x_f32.iter().map(|&v| f16::from_f32(v)).collect();

    let cpu = turbospark_compute::dequant_q6_k_gemv(&refs, &x_f32, n);
    let gpu = dequant_q6_k_gemv(&mut context, &refs, &x_f16, n).expect("GPU dispatch succeeds");
    assert_matches("full 6-bit range", &gpu, &cpu, n);
}

/// A row of all-negative weights against all-positive activations. The exact
/// result is decisively negative, and it can only be negative through the
/// BIAS: the stored level is unsigned and 32 is subtracted from it. A kernel
/// that drops the bias returns a positive number, which no tolerance on
/// magnitude catches, so the sign is asserted outright.
#[test]
fn the_bias_of_thirty_two_is_what_carries_the_sign() {
    let mut context = MetalContext::new().expect("Metal device available on this machine");

    let n = SUPERBLOCK;
    let w: Vec<f32> = (0..n).map(|e| -1.0 + (e % SUB) as f32 / 128.0).collect();
    let row = turbospark_compute::quantize_q6_k(&w);
    let refs: Vec<&[u8]> = vec![&row];
    let x_f32 = vec![1.0f32; n];
    let x_f16: Vec<f16> = x_f32.iter().map(|&v| f16::from_f32(v)).collect();

    let cpu = turbospark_compute::dequant_q6_k_gemv(&refs, &x_f32, n);
    let gpu = dequant_q6_k_gemv(&mut context, &refs, &x_f16, n).expect("GPU dispatch succeeds");

    assert!(
        cpu[0] < -200.0,
        "reference is not decisively negative: {}",
        cpu[0]
    );
    assert!(
        gpu[0].to_f32() < 0.0,
        "the kernel dropped the bias: got {} against {}",
        gpu[0].to_f32(),
        cpu[0]
    );
    assert_matches("all-negative row", &gpu, &cpu, n);
}

/// Sixteen sub-blocks of very different magnitude inside one superblock, so
/// the sixteen int8 scales come out spread rather than all near the maximum.
/// This is the case that separates the scales striding by TWO per quarter
/// from striding by one: with a flat magnitude profile the two schemes read
/// nearly the same numbers.
#[test]
fn the_sixteen_sub_block_scales_stride_by_two_per_quarter() {
    let mut context = MetalContext::new().expect("Metal device available on this machine");

    const FACTORS: [f32; 16] = [
        1.0, 0.25, 4.0, 2.0, 0.5, 8.0, 1.0, 0.125, 3.0, 0.5, 6.0, 1.5, 0.25, 5.0, 2.0, 0.75,
    ];
    let n = SUPERBLOCK;
    let m = 8usize;
    let rows: Vec<Vec<u8>> = (0..m)
        .map(|r| {
            let scaled: Vec<f32> = weights(n, 41 + r as u32)
                .iter()
                .enumerate()
                .map(|(e, &v)| v * FACTORS[(e / SUB) % 16])
                .collect();
            turbospark_compute::quantize_q6_k(&scaled)
        })
        .collect();
    let refs: Vec<&[u8]> = rows.iter().map(|r| r.as_slice()).collect();

    let x_f32 = weights(n, 5);
    let x_f16: Vec<f16> = x_f32.iter().map(|&v| f16::from_f32(v)).collect();

    let cpu = turbospark_compute::dequant_q6_k_gemv(&refs, &x_f32, n);
    let gpu = dequant_q6_k_gemv(&mut context, &refs, &x_f16, n).expect("GPU dispatch succeeds");
    assert_matches("16 sub-blocks, 64x spread", &gpu, &cpu, n);
}

/// Sub-blocks whose extremes alternate in sign, so ggml's negative `iscale`
/// puts both signs among the sixteen stored scale bytes. A kernel reading
/// them as `uchar` turns every negative one into a large positive scale:
/// finite, and wrong by two orders of magnitude on half the row.
#[test]
fn the_sub_block_scales_are_read_as_signed() {
    let mut context = MetalContext::new().expect("Metal device available on this machine");

    let n = SUPERBLOCK;
    let w: Vec<f32> = (0..n)
        .map(|e| {
            let sign = if (e / SUB) % 2 == 0 { 1.0 } else { -1.0 };
            sign * (1.0 + (e % SUB) as f32 / 16.0)
        })
        .collect();
    let row = turbospark_compute::quantize_q6_k(&w);
    let stored: Vec<i8> = row[192..208].iter().map(|&b| b as i8).collect();
    assert!(
        stored.iter().any(|&s| s < 0) && stored.iter().any(|&s| s > 0),
        "fixture does not exercise both signs: {stored:?}"
    );

    let refs: Vec<&[u8]> = vec![&row];
    let x_f32 = weights(n, 23);
    let x_f16: Vec<f16> = x_f32.iter().map(|&v| f16::from_f32(v)).collect();

    let cpu = turbospark_compute::dequant_q6_k_gemv(&refs, &x_f32, n);
    let gpu = dequant_q6_k_gemv(&mut context, &refs, &x_f16, n).expect("GPU dispatch succeeds");
    assert_matches("signed sub-block scales", &gpu, &cpu, n);
}

/// Many superblocks per row, each with a different super-scale. A kernel that
/// hoisted `d` out of the superblock loop, or strided the 210-byte block
/// wrongly, passes the two-superblock case above and fails here. 210 is not a
/// multiple of 4, so a stride bug is likelier than in either sibling.
#[test]
fn per_superblock_scales_are_not_hoisted() {
    let mut context = MetalContext::new().expect("Metal device available on this machine");

    let n = 16 * SUPERBLOCK;
    let m = 8usize;
    let rows: Vec<Vec<u8>> = (0..m)
        .map(|r| {
            let scaled: Vec<f32> = weights(n, 61 + r as u32)
                .iter()
                .enumerate()
                .map(|(e, &w)| w * 10f32.powi((e / SUPERBLOCK) as i32 % 5 - 2))
                .collect();
            turbospark_compute::quantize_q6_k(&scaled)
        })
        .collect();
    let refs: Vec<&[u8]> = rows.iter().map(|r| r.as_slice()).collect();

    let x_f32 = weights(n, 17);
    let x_f16: Vec<f16> = x_f32.iter().map(|&v| f16::from_f32(v)).collect();

    let cpu = turbospark_compute::dequant_q6_k_gemv(&refs, &x_f32, n);
    let gpu = dequant_q6_k_gemv(&mut context, &refs, &x_f16, n).expect("GPU dispatch succeeds");
    assert_matches("16 superblocks, varying scales", &gpu, &cpu, n);
}

/// The offset-bound form has to land on the same answer as the copying one.
/// Its whole purpose is reading weights in place out of the resident mapping,
/// and an offset bug there reads a neighbouring tensor: finite, plausible,
/// and wrong.
#[test]
fn the_resident_form_matches_the_copying_form() {
    let mut context = MetalContext::new().expect("Metal device available on this machine");

    let n = SUPERBLOCK;
    let m = 4usize;
    let rows: Vec<Vec<u8>> = (0..m)
        .map(|r| turbospark_compute::quantize_q6_k(&weights(n, 13 + r as u32)))
        .collect();
    let refs: Vec<&[u8]> = rows.iter().map(|r| r.as_slice()).collect();

    // A non-zero offset with garbage in front, so an offset of zero cannot
    // pass by accident. 4098 rather than 4096: a Q6_K row is 210 bytes and
    // nothing in this path may assume a 4-byte-aligned base.
    let row_bytes = q6_k_row_bytes(n);
    let pad = 4098usize;
    let mut blob = vec![0xABu8; pad];
    for r in &rows {
        blob.extend_from_slice(r);
    }
    let buffer = context.new_buffer_with_data(&blob);

    let x_f32 = weights(n, 77);
    let x_f16: Vec<f16> = x_f32.iter().map(|&v| f16::from_f32(v)).collect();

    let cpu = turbospark_compute::dequant_q6_k_gemv(&refs, &x_f32, n);
    let gpu = dequant_q6_k_gemv_resident(
        &mut context,
        &Q6KResidentMatrix {
            buffer: &buffer,
            weights_offset: pad as u64,
            rows: m,
            cols: n,
        },
        &x_f16,
    )
    .expect("GPU dispatch succeeds");

    assert_eq!(row_bytes * m + pad, blob.len());
    assert_matches("resident at offset 4098", &gpu, &cpu, n);
}

/// `embed_lookup_q6_k` against the CPU dequant of the same row (ROADMAP
/// Phase S).
///
/// This kernel did not exist when Q6_K landed, deliberately: the only real
/// file using the type put it in `output.weight`, which needs a GEMV and
/// nothing else. Phase S's candidate puts `token_embd` in Q6_K and ties the
/// head to it, so it is now the lookup that install runs on every token.
///
/// Three bugs are in scope and none faults. The row stride is `D / 256 * 210`
/// bytes rather than `D`, so using the element count reads a neighbouring
/// token. The lookup walks ELEMENTS where the GEMV walks lanes, so it has to
/// rebuild the quarter/half addressing itself, and getting the scale stride
/// wrong (one instead of two per quarter) mixes scales between quarters. And
/// dropping the bias of 32 turns every value positive. The token looked up is
/// deliberately not row 0, and the comparison is element by element, because
/// each of those returns a plausible row.
#[test]
fn embed_lookup_q6_k_reads_the_right_row_and_scales_it() {
    let mut context = MetalContext::new().expect("Metal device available on this machine");

    let (vocab, d) = (7usize, 2 * SUPERBLOCK);
    let rows: Vec<Vec<f32>> = (0..vocab).map(|t| weights(d, 400 + t as u32)).collect();
    let mut table = Vec::new();
    for r in &rows {
        table.extend_from_slice(&turbospark_compute::quantize_q6_k(r));
    }
    let table_buffer = context.new_buffer_with_data(&table);
    let out = context.new_output_buffer((d * std::mem::size_of::<u16>()) as u64);

    let token = 5u32;
    let out_scale = 4.0f32;
    let pass = context.begin_pass();
    turbospark_gpu::encode_embed_lookup_q6_k(
        &mut context,
        &pass,
        (&table_buffer, 0),
        (&out, 0),
        token,
        d as u32,
        out_scale,
    )
    .expect("GPU dispatch succeeds");
    pass.commit_and_wait();

    let row_bytes = q6_k_row_bytes(d);
    let want: Vec<f32> = turbospark_compute::dequantize_q6_k(
        &table[token as usize * row_bytes..(token as usize + 1) * row_bytes],
        d,
    )
    .iter()
    .map(|v| v * out_scale)
    .collect();

    let got: Vec<f32> = {
        let ptr = out.contents() as *const u16;
        let bits = unsafe { std::slice::from_raw_parts(ptr, d) };
        bits.iter().map(|&b| f16::from_bits(b).to_f32()).collect()
    };
    assert_matches(
        "q6_k embed row 5, scale 4",
        &got.iter().map(|&v| f16::from_f32(v)).collect::<Vec<_>>(),
        &want,
        d,
    );
    // The scale is not cosmetic: without it every value is 4x too small,
    // which the tolerance above would tolerate on a near-zero row.
    let peak = want.iter().fold(0f32, |m, &v| m.max(v.abs()));
    assert!(
        peak > 1.0,
        "the test row is too small to prove the scale: {peak}"
    );
}
