//! Runs `dequant_q3_k_gemv_simd` on real Metal hardware against the CPU
//! reference in `turbospark_compute::dequant_q3_k_gemv` (Dense Qwen2 roadmap
//! item). The kernel rule: no quant kernel is trusted before this file
//! exists and passes.
//!
//! The reference itself is held to ggml by the generated oracle
//! (`q3_k_decodes_exactly_what_ggml_decodes` in `crates/compute`), so the
//! layout is anchored twice: this file pins the KERNEL to the reference, and
//! the oracle pins the reference to the format.
//!
//! Q3_K needs cases its Q4_K sibling does not, because four of its
//! "obvious" readings are finite, byte-aligned and length-correct: the
//! super-scale that trails the block, the high-bit run whose byte index is
//! `e % 32` while its BIT index is `e / 32`, the sixteen 6-bit scales
//! shuffled out of twelve bytes, and signed 2-bit levels. Each trap below
//! loads the case where dropping the corresponding rule still returns
//! numbers of the right magnitude.
#![cfg(target_os = "macos")]

use half::f16;
use turbospark_gpu::{
    dequant_q3_k_gemv, dequant_q3_k_gemv_resident, q3_k_row_bytes, MetalContext, Q3KResidentMatrix,
};

const SUPERBLOCK: usize = 256;
const GROUP: usize = 16;

fn weights(n: usize, seed: u32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((s >> 8) as f32 / (1u32 << 23) as f32) - 1.0
        })
        .collect()
}

/// The CPU reference and the GPU are handed the SAME already-quantized
/// bytes, so quantization error cancels and what is left is FP16 rounding
/// and reduction order. Anything above that is a kernel bug.
fn assert_matches(label: &str, gpu: &[f16], cpu: &[f32], n: usize) {
    assert_eq!(gpu.len(), cpu.len());
    let gpu_f32: Vec<f32> = gpu.iter().map(|v| v.to_f32()).collect();
    let err = turbospark_compute::max_abs_diff(&gpu_f32, cpu);
    let scale = cpu.iter().fold(0f32, |m, &v| m.max(v.abs())).max(1.0);
    // FP16 has 10 mantissa bits, and the kernel rounds once at the end while
    // accumulating in FP32, so the bound scales with the result magnitude
    // and only weakly with N.
    let bound = scale * 1e-2 + (n as f32) * 1e-4;
    assert!(err < bound, "{label}: err = {err}, bound = {bound}");
}

#[test]
fn matches_cpu_reference() {
    let mut context = MetalContext::new().expect("Metal device available on this machine");

    let n = 2 * SUPERBLOCK;
    let m = 5usize; // not a multiple of 8: exercises the early-return guard
    let rows: Vec<Vec<u8>> = (0..m)
        .map(|r| turbospark_compute::quantize_q3_k(&weights(n, 7 + r as u32)))
        .collect();
    let refs: Vec<&[u8]> = rows.iter().map(|r| r.as_slice()).collect();

    let x_f32 = weights(n, 99);
    let x_f16: Vec<f16> = x_f32.iter().map(|&v| f16::from_f32(v)).collect();

    let cpu = turbospark_compute::dequant_q3_k_gemv(&refs, &x_f32, n);
    let gpu = dequant_q3_k_gemv(&mut context, &refs, &x_f16, n).expect("GPU dispatch succeeds");
    assert_matches("m=5,n=512", &gpu, &cpu, n);
}

/// Groups whose ranges differ by 16x inside one superblock, with the
/// LARGEST at group 12. That puts the maximum 6-bit scale in the second
/// half of the sixteen, where its low nibble lives in the HIGH nibble of a
/// byte and its top two bits in a packed field a quarter-block away. A
/// kernel that reads only the low nibbles of `packed[0..8]` still returns
/// finite, correctly signed numbers here; they are merely scaled wrongly per
/// group, which is what the tolerance below rejects.
#[test]
fn six_bit_scales_are_shuffled_across_all_twelve_bytes() {
    let mut context = MetalContext::new().expect("Metal device available on this machine");

    const FACTORS: [f32; 16] = [
        1.0, 0.5, 4.0, 2.0, 0.5, 8.0, 1.0, 2.0, 1.0, 0.25, 16.0, 0.5, 1.0, 2.0, 4.0, 0.5,
    ];
    let n = SUPERBLOCK;
    let m = 8usize;
    let rows: Vec<Vec<u8>> = (0..m)
        .map(|r| {
            let scaled: Vec<f32> = weights(n, 41 + r as u32)
                .iter()
                .enumerate()
                .map(|(e, &v)| v * FACTORS[(e / GROUP) % 16])
                .collect();
            turbospark_compute::quantize_q3_k(&scaled)
        })
        .collect();
    let refs: Vec<&[u8]> = rows.iter().map(|r| r.as_slice()).collect();

    let x_f32 = weights(n, 5);
    let x_f16: Vec<f16> = x_f32.iter().map(|&v| f16::from_f32(v)).collect();

    let cpu = turbospark_compute::dequant_q3_k_gemv(&refs, &x_f32, n);
    let gpu = dequant_q3_k_gemv(&mut context, &refs, &x_f16, n).expect("GPU dispatch succeeds");
    assert_matches("16 groups, 64x spread", &gpu, &cpu, n);
}

/// A row of all-negative weights against all-positive activations. The exact
/// result is strongly negative, and it can only be negative through the
/// CLEARED high bit shifting the stored level down by 4. A kernel that reads
/// the 2-bit levels unsigned (the Q6_K habit) returns a non-negative number
/// instead, which no tolerance on magnitude would catch, so the sign is
/// asserted outright.
#[test]
fn the_high_bit_run_is_what_carries_the_sign() {
    let mut context = MetalContext::new().expect("Metal device available on this machine");

    let n = SUPERBLOCK;
    // Not a constant row: a flat one quantizes to a single level and would
    // not exercise the level arithmetic on both signs.
    let w: Vec<f32> = (0..n).map(|e| -1.0 + (e % GROUP) as f32 / 64.0).collect();
    let row = turbospark_compute::quantize_q3_k(&w);
    let refs: Vec<&[u8]> = vec![&row];
    let x_f32 = vec![1.0f32; n];
    let x_f16: Vec<f16> = x_f32.iter().map(|&v| f16::from_f32(v)).collect();

    let cpu = turbospark_compute::dequant_q3_k_gemv(&refs, &x_f32, n);
    let gpu = dequant_q3_k_gemv(&mut context, &refs, &x_f16, n).expect("GPU dispatch succeeds");

    assert!(
        cpu[0] < -100.0,
        "reference is not decisively negative: {}",
        cpu[0]
    );
    assert!(
        gpu[0].to_f32() < 0.0,
        "the kernel read the levels unsigned: got {} against {}",
        gpu[0].to_f32(),
        cpu[0]
    );
    assert_matches("all-negative row", &gpu, &cpu, n);
}

/// Many superblocks per row, each at a different magnitude. A kernel that
/// hoisted `d` out of the superblock loop, or strided the 110-byte block
/// wrongly (the 110/144 confusion with Q4_K is byte-aligned and reads the
/// next block's fields), passes the two-superblock case above and fails
/// here.
#[test]
fn per_superblock_scales_are_not_hoisted() {
    let mut context = MetalContext::new().expect("Metal device available on this machine");

    let n = 16 * SUPERBLOCK;
    let m = 8usize;
    let rows: Vec<Vec<u8>> = (0..m)
        .map(|r| {
            // Each superblock spans a different magnitude, so the 16
            // super-scales within a row differ by orders of magnitude rather
            // than by rounding.
            let scaled: Vec<f32> = weights(n, 61 + r as u32)
                .iter()
                .enumerate()
                .map(|(e, &w)| w * 10f32.powi((e / SUPERBLOCK) as i32 % 5 - 2))
                .collect();
            turbospark_compute::quantize_q3_k(&scaled)
        })
        .collect();
    let refs: Vec<&[u8]> = rows.iter().map(|r| r.as_slice()).collect();

    let x_f32 = weights(n, 17);
    let x_f16: Vec<f16> = x_f32.iter().map(|&v| f16::from_f32(v)).collect();

    let cpu = turbospark_compute::dequant_q3_k_gemv(&refs, &x_f32, n);
    let gpu = dequant_q3_k_gemv(&mut context, &refs, &x_f16, n).expect("GPU dispatch succeeds");
    assert_matches("16 superblocks, varying scales", &gpu, &cpu, n);
}

/// The offset-bound form has to land on the same answer as the copying one.
/// Its whole purpose is reading weights in place out of the resident
/// mapping, and an offset bug there reads a neighbouring tensor: finite,
/// plausible, and wrong.
#[test]
fn the_resident_form_matches_the_copying_form() {
    let mut context = MetalContext::new().expect("Metal device available on this machine");

    let n = SUPERBLOCK;
    let m = 4usize;
    let rows: Vec<Vec<u8>> = (0..m)
        .map(|r| turbospark_compute::quantize_q3_k(&weights(n, 13 + r as u32)))
        .collect();
    let refs: Vec<&[u8]> = rows.iter().map(|r| r.as_slice()).collect();

    // Put the matrix at a non-zero offset inside a bigger buffer, with
    // garbage in front of it, so an offset of zero cannot pass by accident.
    let row_bytes = q3_k_row_bytes(n);
    let pad = 4096usize;
    let mut blob = vec![0xABu8; pad];
    for r in &rows {
        blob.extend_from_slice(r);
    }
    let buffer = context.new_buffer_with_data(&blob);

    let x_f32 = weights(n, 77);
    let x_f16: Vec<f16> = x_f32.iter().map(|&v| f16::from_f32(v)).collect();

    let cpu = turbospark_compute::dequant_q3_k_gemv(&refs, &x_f32, n);
    let gpu = dequant_q3_k_gemv_resident(
        &mut context,
        &Q3KResidentMatrix {
            buffer: &buffer,
            weights_offset: pad as u64,
            rows: m,
            cols: n,
        },
        &x_f16,
    )
    .expect("GPU dispatch succeeds");

    assert_eq!(row_bytes * m + pad, blob.len());
    assert_matches("resident at offset 4096", &gpu, &cpu, n);
}

/// `embed_lookup_q3_k` against the CPU dequant of the same row. No real file
/// in this port's orbit keeps `token_embd` at Q3_K; this holds the lookup to
/// the same contract its GEMV sibling has, for the day one does (the Q6_K
/// precedent). The row stride is `D / 256 * 110` bytes rather than `D`, and
/// the element mapping is the GEMV's walked from the other direction, so the
/// token looked up is deliberately not row 0 and the comparison is element
/// by element.
#[test]
fn embed_lookup_reads_the_right_row_and_scales_it() {
    let mut context = MetalContext::new().expect("Metal device available on this machine");

    let (vocab, d) = (7usize, 2 * SUPERBLOCK);
    let rows: Vec<Vec<f32>> = (0..vocab).map(|t| weights(d, 300 + t as u32)).collect();
    let mut table = Vec::new();
    for r in &rows {
        table.extend_from_slice(&turbospark_compute::quantize_q3_k(r));
    }
    let table_buffer = context.new_buffer_with_data(&table);
    let out = context.new_output_buffer((d * std::mem::size_of::<u16>()) as u64);

    let token = 5u32;
    let out_scale = 4.0f32;
    let pass = context.begin_pass();
    turbospark_gpu::encode_embed_lookup_q3_k(
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

    let row_bytes = q3_k_row_bytes(d);
    let want: Vec<f32> = turbospark_compute::dequantize_q3_k(
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
    let got_f16: Vec<f16> = got.iter().map(|&v| f16::from_f32(v)).collect();
    assert_matches("embed lookup, token 5", &got_f16, &want, d);
    // A wrong token would still produce values of the right magnitude, so
    // one element differing from a NEIGHBOURING token's is asserted too.
    let other: Vec<f32> =
        turbospark_compute::dequantize_q3_k(&table[2 * row_bytes..3 * row_bytes], d)
            .iter()
            .map(|v| v * out_scale)
            .collect();
    assert_ne!(got[0], other[0]);
}
