//! `embed_lookup_bf16` on real Metal against a CPU reference that reads the
//! same BF16 bits, compared with `==`: the lookup does one multiply and one
//! FP16 rounding, so there is no reduction order to allow for.
//!
//! The table sits at a nonzero offset inside its buffer and rows differ, so
//! an offset slip or a row-stride slip shows (both mutation-checked, as was
//! reading the table as `half`). The `gid >= D` guard is NOT covered: the
//! encoder dispatches exactly `D` threads, so removing it changes nothing
//! this test can see.
#![cfg(target_os = "macos")]

use half::f16;
use turbospark_gpu::{encode_embed_lookup_bf16, read_buffer_f16, MetalContext};

const ROWS: usize = 5;
const D: usize = 200;
/// Bytes of junk ahead of the table, as a resident weight file has.
const LEAD: usize = 4096;

fn bf16_bits(v: f32) -> u16 {
    (v.to_bits() >> 16) as u16
}

fn f32_of_bits(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

fn table() -> Vec<u16> {
    (0..ROWS * D)
        .map(|i| {
            let row = i / D;
            let col = i % D;
            // Distinct per row, signed, mostly sub-unit like a real table.
            bf16_bits((col as f32 - 97.0) * 0.013 + row as f32 * 1.75)
        })
        .collect()
}

fn lookup(table: &[u16], token: u32, scale: f32) -> Vec<f16> {
    let mut context = MetalContext::new().expect("Metal device available on this machine");
    let mut bytes = vec![0xA5u8; LEAD];
    for v in table {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    let table_buf = context.new_buffer_with_data(&bytes);
    let out = context.new_output_buffer((D * 2) as u64);
    let pass = context.begin_pass();
    encode_embed_lookup_bf16(
        &mut context,
        &pass,
        (&table_buf, LEAD as u64),
        (&out, 0),
        token,
        D as u32,
        scale,
    )
    .expect("dispatch succeeds");
    pass.commit_and_wait();
    read_buffer_f16(&out, 0, D)
}

#[test]
fn every_row_matches_the_cpu_bit_for_bit() {
    let t = table();
    for token in 0..ROWS as u32 {
        for scale in [1.0f32, 0.5, 45.254_833] {
            let gpu = lookup(&t, token, scale);
            let cpu: Vec<f16> = t[token as usize * D..(token as usize + 1) * D]
                .iter()
                .map(|&b| f16::from_f32(f32_of_bits(b) * scale))
                .collect();
            let gpu_bits: Vec<u16> = gpu.iter().map(|v| v.to_bits()).collect();
            let cpu_bits: Vec<u16> = cpu.iter().map(|v| v.to_bits()).collect();
            assert_eq!(gpu_bits, cpu_bits, "token {token} scale {scale}");
        }
    }
}

#[test]
fn a_value_past_fp16_range_becomes_infinity() {
    let mut t = table();
    t[D + 3] = bf16_bits(1.0e5);
    t[D + 4] = bf16_bits(-1.0e5);
    let gpu = lookup(&t, 1, 1.0);
    assert_eq!(gpu[3], f16::INFINITY);
    assert_eq!(gpu[4], f16::NEG_INFINITY);
}
