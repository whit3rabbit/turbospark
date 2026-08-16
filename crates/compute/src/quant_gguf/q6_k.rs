use foundation::LogitValue as F16;

/// Elements in one Q6_K superblock.
pub const Q6_K_BLOCK_ELEMS: usize = 256;

/// Elements sharing one Q6_K scale. Sixteen of these tile a superblock, half
/// the sub-block size Q4_K uses.
pub const Q6_K_SUB_ELEMS: usize = 16;

/// Bytes in one Q6_K superblock: 128 bytes of low nibbles, 64 bytes of high
/// bit-pairs, 16 SIGNED sub-block scales, then the f16 super-scale. Matches
/// `ggml_type_block(14)` in `turbospark_repack::gguf_header`, held to it by
/// `crates/repack`'s tests rather than by an import.
pub const Q6_K_BLOCK_BYTES: usize = 210;

/// Byte offset of the high-bit run inside a superblock.
const Q6_K_QH_AT: usize = 128;
/// Byte offset of the 16 signed sub-block scales.
const Q6_K_SCALES_AT: usize = 192;
/// Byte offset of the f16 super-scale.
const Q6_K_D_AT: usize = 208;
/// Elements covered by one (ql, qh, scales) stride: a superblock is two.
const Q6_K_HALF_ELEMS: usize = 128;

/// Dequantize `n` elements of a Q6_K byte run to FP32.
///
/// Q6_K is the third layout in this module and shares its shape with neither
/// sibling. Four things about it are silently wrong rather than a fault if
/// carried over from Q4_K by habit:
///
/// 1. The six bits of an element are SPLIT ACROSS TWO RUNS: four low bits in
///    `ql`, two high bits in `qh`. Reading only `ql` gives values in `0..16`
///    where `0..64` was meant, which is finite and merely compressed.
/// 2. The quants are BIASED, not unsigned: the stored 6-bit value has 32
///    SUBTRACTED from it, so the reconstruction is symmetric and there is no
///    per-sub-block min. Dropping the bias shifts every weight positive.
/// 3. The sub-block scales are SIGNED int8, plain bytes rather than the 6-bit
///    packing Q4_K uses, and real files do carry negative ones (ggml derives
///    them through a negative `iscale`). Reading them as `u8` mirrors the
///    sign of whole 16-element runs.
/// 4. A superblock is walked in two halves of 128 elements. Within a half,
///    the four values a `qh` byte serves are 32 elements apart and each uses
///    a DIFFERENT scale pair (`is`, `is + 2`, `is + 4`, `is + 6`).
pub fn dequantize_q6_k(blocks: &[u8], n: usize) -> Vec<f32> {
    assert!(
        n % Q6_K_BLOCK_ELEMS == 0,
        "n ({n}) is not a whole number of {Q6_K_BLOCK_ELEMS}-element Q6_K superblocks"
    );
    let n_blocks = n / Q6_K_BLOCK_ELEMS;
    assert!(
        blocks.len() >= n_blocks * Q6_K_BLOCK_BYTES,
        "need {} bytes for {n} elements, got {}",
        n_blocks * Q6_K_BLOCK_BYTES,
        blocks.len()
    );

    let mut out = vec![0f32; n];
    for b in 0..n_blocks {
        let base = b * Q6_K_BLOCK_BYTES;
        let d = f32::from(F16::from_bits(u16::from_le_bytes([
            blocks[base + Q6_K_D_AT],
            blocks[base + Q6_K_D_AT + 1],
        ])));
        for h in 0..Q6_K_BLOCK_ELEMS / Q6_K_HALF_ELEMS {
            let ql = &blocks[base + h * 64..base + h * 64 + 64];
            let qh = &blocks[base + Q6_K_QH_AT + h * 32..base + Q6_K_QH_AT + h * 32 + 32];
            let sc = &blocks[base + Q6_K_SCALES_AT + h * 8..base + Q6_K_SCALES_AT + h * 8 + 8];
            let at = b * Q6_K_BLOCK_ELEMS + h * Q6_K_HALF_ELEMS;
            for l in 0..32 {
                let is = l / Q6_K_SUB_ELEMS;
                let q = [
                    ((ql[l] & 0xF) | ((qh[l] & 3) << 4)) as i32 - 32,
                    ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) as i32 - 32,
                    ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i32 - 32,
                    ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i32 - 32,
                ];
                for (k, &qk) in q.iter().enumerate() {
                    out[at + l + k * 32] = d * (sc[is + k * 2] as i8) as f32 * qk as f32;
                }
            }
        }
    }
    out
}

/// Quantize a row to Q6_K bytes, following ggml's `quantize_row_q6_K_ref`
/// with a plain largest-magnitude fit per 16-element sub-block in place of
/// its `make_qx_quants` search.
///
/// Same standing as the other two quantizers here: this port never encodes a
/// real GGUF (bytes arrive quantized and are copied through verbatim), so
/// what has to match ggml exactly is the BYTE LAYOUT and the reconstruction
/// formula, not the encoder's search quality. ggml's negative `iscale` is
/// kept rather than simplified away, because it is what makes real files
/// carry negative sub-block scales, and a quantizer that only ever emitted
/// positive ones would leave the decoder's signed read untested.
pub fn quantize_q6_k(row: &[f32]) -> Vec<u8> {
    assert!(
        row.len() % Q6_K_BLOCK_ELEMS == 0,
        "row length {} is not a whole number of {Q6_K_BLOCK_ELEMS}-element superblocks",
        row.len()
    );
    let subs = Q6_K_BLOCK_ELEMS / Q6_K_SUB_ELEMS;
    let mut out = Vec::with_capacity(row.len() / Q6_K_BLOCK_ELEMS * Q6_K_BLOCK_BYTES);

    for block in row.chunks_exact(Q6_K_BLOCK_ELEMS) {
        // Per sub-block: the SIGNED element of largest magnitude sets the
        // scale, so a sub-block whose extreme is negative gets a negative
        // scale, exactly as ggml's `make_qx_quants` does.
        let mut scales = [0f32; 16];
        let mut max_scale = 0f32;
        for (j, scale) in scales.iter_mut().enumerate().take(subs) {
            let sub = &block[j * Q6_K_SUB_ELEMS..(j + 1) * Q6_K_SUB_ELEMS];
            let mut extreme = 0f32;
            for &v in sub {
                if v.abs() > extreme.abs() {
                    extreme = v;
                }
            }
            *scale = -extreme / 32.0;
            if scale.abs() > max_scale.abs() {
                max_scale = *scale;
            }
        }

        let mut sc = [0i8; 16];
        let d = if max_scale == 0.0 {
            F16::from_f32(0.0)
        } else {
            let iscale = -128.0 / max_scale;
            for (j, s) in sc.iter_mut().enumerate().take(subs) {
                *s = ((iscale * scales[j]).round() as i32).clamp(-128, 127) as i8;
            }
            // Round the super-scale through f16 before deriving quants, so a
            // decode reproduces the integers chosen here. Same discipline as
            // the Q8_0 and Q4_K siblings.
            F16::from_f32(1.0 / iscale)
        };
        let d_f32 = f32::from(d);

        // Stored levels are the quant PLUS 32, which is what the packing
        // holds and what the decoder subtracts back off.
        let mut levels = [32u8; Q6_K_BLOCK_ELEMS];
        for j in 0..subs {
            let step = d_f32 * sc[j] as f32;
            if step == 0.0 {
                continue;
            }
            for (ii, &w) in block[j * Q6_K_SUB_ELEMS..(j + 1) * Q6_K_SUB_ELEMS]
                .iter()
                .enumerate()
            {
                let q = (w / step).round().clamp(-32.0, 31.0) as i32;
                levels[j * Q6_K_SUB_ELEMS + ii] = (q + 32) as u8;
            }
        }

        let mut ql = [0u8; 128];
        let mut qh = [0u8; 64];
        for h in 0..Q6_K_BLOCK_ELEMS / Q6_K_HALF_ELEMS {
            let at = h * Q6_K_HALF_ELEMS;
            for l in 0..32 {
                let v = [
                    levels[at + l],
                    levels[at + l + 32],
                    levels[at + l + 64],
                    levels[at + l + 96],
                ];
                ql[h * 64 + l] = (v[0] & 0xF) | ((v[2] & 0xF) << 4);
                ql[h * 64 + l + 32] = (v[1] & 0xF) | ((v[3] & 0xF) << 4);
                qh[h * 32 + l] =
                    (v[0] >> 4) | ((v[1] >> 4) << 2) | ((v[2] >> 4) << 4) | ((v[3] >> 4) << 6);
            }
        }

        out.extend_from_slice(&ql);
        out.extend_from_slice(&qh);
        out.extend(sc.iter().map(|&s| s as u8));
        out.extend_from_slice(&d.to_bits().to_le_bytes());
    }
    out
}

/// FP32 reference for the Q6_K GEMV `y = W * x`, one byte run per output row.
pub fn dequant_q6_k_gemv(weight_rows: &[&[u8]], x: &[f32], n: usize) -> Vec<f32> {
    assert!(!weight_rows.is_empty());
    assert_eq!(x.len(), n);
    weight_rows
        .iter()
        .map(|row| {
            dequantize_q6_k(row, n)
                .iter()
                .zip(x.iter())
                .map(|(w, xv)| w * xv)
                .sum()
        })
        .collect()
}
