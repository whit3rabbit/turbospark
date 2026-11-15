//! Q1_0 decoded against ggml's own `to_float` (scripts/ggml_dequant_fixture.c, type 41).

use turbospark_compute::{dequant_q1_0_gemv, dequantize_q1_0, Q1_0_BLOCK_BYTES, Q1_0_BLOCK_ELEMS};

#[test]
fn q1_0_block_geometry_matches_ggml() {
    // ggml_blck_size(41) = 128, ggml_type_size(41) = 18 (Homebrew ggml probe).
    assert_eq!(Q1_0_BLOCK_ELEMS, 128);
    assert_eq!(Q1_0_BLOCK_BYTES, 18);
}

#[test]
fn q1_0_set_bit_is_plus_d_lsb_first() {
    let mut block = vec![0u8; Q1_0_BLOCK_BYTES];
    block[..2].copy_from_slice(&0x3800u16.to_le_bytes()); // f16 0.5
    block[2] = 0x01; // element 0
    block[3] = 0x80; // element 15
    let got = dequantize_q1_0(&block, Q1_0_BLOCK_ELEMS);
    for (i, v) in got.iter().enumerate() {
        let want = if i == 0 || i == 15 { 0.5 } else { -0.5 };
        assert_eq!(*v, want, "element {i}");
    }
}

#[test]
fn q1_0_gemv_matches_explicit_decode_then_dot() {
    let n = 2 * Q1_0_BLOCK_ELEMS;
    let rows: Vec<Vec<u8>> = (0..3)
        .map(|seed| {
            let mut row = vec![0u8; 2 * Q1_0_BLOCK_BYTES];
            for (block, bytes) in row.chunks_exact_mut(Q1_0_BLOCK_BYTES).enumerate() {
                let d_bits = 0x3000u16 + (seed + block) as u16 * 0x100;
                bytes[..2].copy_from_slice(&d_bits.to_le_bytes());
                for (i, byte) in bytes[2..].iter_mut().enumerate() {
                    *byte = (seed as u8).wrapping_mul(37).wrapping_add((i * 53) as u8);
                }
            }
            row
        })
        .collect();
    let x: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.17).sin()).collect();
    let refs: Vec<&[u8]> = rows.iter().map(Vec::as_slice).collect();
    let got = dequant_q1_0_gemv(&refs, &x, n);
    for (i, row) in rows.iter().enumerate() {
        let want: f32 = dequantize_q1_0(row, n)
            .iter()
            .zip(&x)
            .map(|(w, x)| w * x)
            .sum();
        assert_eq!(got[i], want);
    }
}
