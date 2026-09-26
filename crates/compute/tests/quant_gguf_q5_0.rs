use turbospark_compute::{dequantize_q5_0, Q5_0_BLOCK_BYTES, Q5_0_BLOCK_ELEMS};

#[test]
fn q5_0_unpacks_low_and_high_halves_and_signed_offset() {
    let mut block = vec![0u8; Q5_0_BLOCK_BYTES];
    block[..2].copy_from_slice(&0x3800u16.to_le_bytes()); // d = 0.5
    let qh = (1u32 << 0) | (1u32 << 16);
    block[2..6].copy_from_slice(&qh.to_le_bytes());
    block[6..].fill(0x10);
    block[6] = 0;

    let values = dequantize_q5_0(&block, Q5_0_BLOCK_ELEMS);
    assert_eq!(values.len(), Q5_0_BLOCK_ELEMS);
    assert_eq!(values[0], 0.0, "first low nibble plus qh high bit");
    assert_eq!(values[1], -8.0, "next low nibble has no high bit");
    assert_eq!(values[16], 0.0, "first high nibble plus qh bit 16");
    assert_eq!(values[17], -7.5, "next high nibble has no high bit");
}
