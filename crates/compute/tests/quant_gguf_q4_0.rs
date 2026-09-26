use turbospark_compute::{dequantize_q4_0, Q4_0_BLOCK_BYTES, Q4_0_BLOCK_ELEMS};

#[test]
fn q4_0_unpacks_low_then_high_signed_nibbles() {
    let mut block = vec![0u8; Q4_0_BLOCK_BYTES];
    block[..2].copy_from_slice(&0x3800u16.to_le_bytes()); // d = 0.5
    block[2] = 0x90;
    block[3] = 0xf0;

    let values = dequantize_q4_0(&block, Q4_0_BLOCK_ELEMS);
    assert_eq!(values[0], -4.0);
    assert_eq!(values[1], -4.0);
    assert_eq!(values[16], 0.5);
    assert_eq!(values[17], 3.5);
}
