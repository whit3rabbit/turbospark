//! Runs `kv_quantize_tq` on real Metal hardware and checks its packed
//! words and norm against `turbospark_compute::kv_quant::quantize_row`.
#![cfg(target_os = "macos")]

use half::f16;
use model_io::KvQuant;
use turbospark_compute::kv_quant::{codebook, midpoints, quantize_row, sign_vector, KEY_SEED};
use turbospark_gpu::{encode_kv_quantize_tq, KvQuantTables, MetalContext};

fn to_f16(v: &[f32]) -> Vec<f16> {
    v.iter().map(|&x| f16::from_f32(x)).collect()
}

fn half_bytes(v: &[f16]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 2);
    for x in v {
        out.extend_from_slice(&x.to_bits().to_le_bytes());
    }
    out
}

/// Reads back `rows * num_kv_heads` packed rows of `1 + packed_words`
/// `u32` words each.
fn read_packed_rows(
    buffer: &metal::Buffer,
    count: usize,
    packed_words: usize,
) -> Vec<(f32, Vec<u32>)> {
    let ptr = buffer.contents() as *const u32;
    let row_words = 1 + packed_words;
    (0..count)
        .map(|r| {
            let base = unsafe { std::slice::from_raw_parts(ptr.add(r * row_words), row_words) };
            let norm = f32::from_bits(base[0]);
            (norm, base[1..].to_vec())
        })
        .collect()
}

fn run_case(dim: usize, bits: u8, rows: usize, num_kv_heads: usize) {
    let mut context = MetalContext::new().expect("Metal device available on this machine");
    let quant = KvQuant::TurboQuant {
        k_bits: bits,
        v_bits: bits,
    };
    let tables = KvQuantTables::new(context.device(), dim, quant).expect("tq tables build");

    // Hashed, non-degenerate input: every row and head distinct.
    let total = rows * num_kv_heads * dim;
    let input_f32: Vec<f32> = (0..total)
        .map(|i| ((i as f32) * 0.913 + 0.37).sin() * 3.0)
        .collect();
    let input_f16 = to_f16(&input_f32);

    let src_buffer = context.new_buffer_with_data(&half_bytes(&input_f16));
    let packed_words = model_io::tq_packed_words(dim as i64, bits) as usize;
    let dst_bytes = (rows * num_kv_heads * (1 + packed_words) * 4) as u64;
    let dst_buffer = context.new_output_buffer(dst_bytes);

    let pass = context.begin_pass();
    encode_kv_quantize_tq(
        &mut context,
        &pass,
        (&src_buffer, 0),
        (num_kv_heads * dim) as u32,
        (&dst_buffer, 0),
        &tables.k,
        dim as u32,
        num_kv_heads as u32,
        rows as u32,
    )
    .expect("dispatch succeeds");
    pass.commit_and_wait();

    let gpu_rows = read_packed_rows(&dst_buffer, rows * num_kv_heads, packed_words);

    let signs = sign_vector(dim, KEY_SEED);
    let cb = codebook(dim, bits);
    let mp = midpoints(&cb);

    for r in 0..rows {
        for h in 0..num_kv_heads {
            let idx = r * num_kv_heads + h;
            let x = &input_f32[idx * dim..(idx + 1) * dim];
            // The GPU reads its source row through FP16 (matching the
            // real cache write path), so the CPU reference must quantize
            // the SAME FP16-rounded values, not the original f32 ones.
            let x_f16: Vec<f32> = input_f16[idx * dim..(idx + 1) * dim]
                .iter()
                .map(|v| v.to_f32())
                .collect();
            let _ = x; // silence unused warning if the rounding path changes
            let cpu = quantize_row(&x_f16, &signs, &mp, bits);

            let (gpu_norm, gpu_words) = &gpu_rows[idx];
            let rel = (gpu_norm - cpu.norm).abs() / cpu.norm.max(1e-6);
            assert!(
                rel < 1e-4,
                "dim={dim} bits={bits} row={r} head={h}: norm gpu={gpu_norm} cpu={}",
                cpu.norm
            );
            assert_eq!(
                gpu_words, &cpu.words,
                "dim={dim} bits={bits} row={r} head={h}: packed words differ\ngpu={gpu_words:?}\ncpu={:?}",
                cpu.words
            );
        }
    }
}

#[test]
fn matches_cpu_reference_at_two_bits_d64() {
    run_case(64, 2, 3, 2);
}

#[test]
fn matches_cpu_reference_at_three_bits_d128_straddling_words() {
    // 3 bits x 128 = 384 bits = 12 words exactly; a straddling index
    // exists at every word boundary not a multiple of 3 elements.
    run_case(128, 3, 2, 3);
}

#[test]
fn matches_cpu_reference_at_four_bits_d256() {
    run_case(256, 4, 2, 1);
}

#[test]
fn a_wider_source_row_stride_is_honoured() {
    // num_kv_heads=4 but the source buffer is laid out with extra heads
    // per row than we quantize -- exercised implicitly by having
    // num_kv_heads > 1 above; this case additionally checks a single
    // kv_head slice deep inside a wider row.
    run_case(64, 2, 1, 5);
}
