//! Q2_0 fixture decoded against ggml's published block formula.

use turbospark_compute::{dequant_q2_0_gemv, dequantize_q2_0, Q2_0_BLOCK_BYTES, Q2_0_BLOCK_ELEMS};

#[test]
fn q2_0_uses_lsb_first_signed_levels_and_f16_scale() {
    assert_eq!(Q2_0_BLOCK_ELEMS, 64);
    assert_eq!(Q2_0_BLOCK_BYTES, 18);
    let mut block = vec![0u8; Q2_0_BLOCK_BYTES];
    block[..2].copy_from_slice(&0x3800u16.to_le_bytes()); // d = 0.5
    for (i, byte) in block[2..].iter_mut().enumerate() {
        *byte = ((i % 4) as u8)
            | ((((i + 1) % 4) as u8) << 2)
            | ((((i + 2) % 4) as u8) << 4)
            | ((((i + 3) % 4) as u8) << 6);
    }

    let got = dequantize_q2_0(&block, Q2_0_BLOCK_ELEMS);
    for i in 0..Q2_0_BLOCK_ELEMS {
        let q = (block[2 + i / 4] >> (2 * (i % 4))) & 3;
        assert_eq!(got[i], (i32::from(q) - 1) as f32 * 0.5, "element {i}");
    }
    assert_eq!(&got[..4], &[-0.5, 0.0, 0.5, 1.0]);
}

#[test]
fn q2_0_gemv_matches_explicit_decode_then_dot() {
    let n = 2 * Q2_0_BLOCK_ELEMS;
    let rows: Vec<Vec<u8>> = (0..3)
        .map(|seed| {
            let mut row = vec![0u8; 2 * Q2_0_BLOCK_BYTES];
            for (block, bytes) in row.chunks_exact_mut(Q2_0_BLOCK_BYTES).enumerate() {
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
    let got = dequant_q2_0_gemv(&refs, &x, n);

    for (i, row) in rows.iter().enumerate() {
        let want: f32 = dequantize_q2_0(row, n)
            .iter()
            .zip(&x)
            .map(|(w, x)| w * x)
            .sum();
        assert_eq!(got[i], want);
    }
}
