//! Gray-code time codec (`mlx_audio/tts/models/tada/gray_code.py`).
//!
//! Reference: mlx-audio at commit
//! [`e1b19b9054bf163f5d812221a54fcc346f1890e9`](https://github.com/Blaizzy/mlx-audio/tree/e1b19b9054bf163f5d812221a54fcc346f1890e9/mlx_audio/tts/models/tada/gray_code.py).
//! Durations ride the flow-matching output as two `num_time_bits`
//! blocks of +/-1 bits (MSB first); a bit decodes to 1 when positive.
//! The gray-to-int walk is the standard xor chain.

/// `int_to_gray_code`: MSB-first +/-1 bits of `n ^ (n >> 1)`. The
/// Rust generation loop only decodes; the encoder side of the gray
/// code rides the reference duration pipeline and the unit tests.
#[allow(dead_code)]
pub fn int_to_gray_code(n: i32, num_bits: usize) -> Vec<f32> {
    let gray = n ^ (n >> 1);
    let mut bits = Vec::with_capacity(num_bits);
    for i in (0..num_bits).rev() {
        bits.push(if (gray >> i) & 1 == 1 { 1.0 } else { -1.0 });
    }
    bits
}

/// `gray_code_to_int`: positive bits are 1; the xor chain unwinds the
/// gray code.
pub fn gray_code_to_int(bits: &[f32]) -> i32 {
    let mut gray = 0i32;
    for &bit in bits {
        gray = (gray << 1) | i32::from(bit > 0.0);
    }
    let mut n = gray;
    let mut mask = n >> 1;
    while mask != 0 {
        n ^= mask;
        mask >>= 1;
    }
    n
}

/// `decode_gray_code_to_time` for the `[rows, num_bits]` layout the
/// generation loop uses: one decoded integer per row.
pub fn decode_gray_code_to_time(gray_code_bits: &[f32], num_bits: usize) -> Vec<i32> {
    gray_code_bits
        .chunks(num_bits)
        .map(gray_code_to_int)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{decode_gray_code_to_time, gray_code_to_int, int_to_gray_code};

    #[test]
    fn gray_code_round_trips() {
        for n in [0i32, 1, 2, 3, 4, 5, 15, 16, 100, 255] {
            let bits = int_to_gray_code(n, 8);
            assert_eq!(gray_code_to_int(&bits), n, "round trip {n}");
        }
    }

    #[test]
    fn gray_bits_are_plus_minus_one_msb_first() {
        // 1 ^ (1 >> 1) = 1 -> MSB-first over 4 bits: [-1, -1, -1, 1].
        assert_eq!(int_to_gray_code(1, 4), vec![-1.0, -1.0, -1.0, 1.0]);
        // 3 ^ 1 = 2 = binary 0010 over 4 bits -> [-1, -1, 1, -1].
        assert_eq!(int_to_gray_code(3, 4), vec![-1.0, -1.0, 1.0, -1.0]);
    }

    #[test]
    fn decode_walks_rows() {
        let bits = int_to_gray_code(7, 4)
            .into_iter()
            .chain(int_to_gray_code(2, 4))
            .collect::<Vec<_>>();
        assert_eq!(decode_gray_code_to_time(&bits, 4), vec![7, 2]);
    }
}
