//! Holds the batched GGUF Q4_K, Q6_K and Q8_0 kernels to their parity
//! contract on real Metal: for every batch width the verify pass can ask
//! for, B rows of the batched kernel must be BIT-IDENTICAL to B calls of the
//! matching GEMV.
//!
//! Bit-exactness is the gate that admits these kernels, as it was for the
//! affine batched ones (`dequant_1bit_gemm_parity.rs`): a K-quant trunk
//! decodes through these GEMVs, so a bit-identical verify keeps speculation
//! lossless against plain decode. The GEMVs are held to the CPU references
//! by their own parity files, so the chain is transitive.
//!
//! ONE context serves every width and every type, on purpose. The pipeline
//! cache keys on (source address, name, key); if B were missing from the key
//! the second width would be served the first width's pipeline, and only a
//! shared context can see that.
#![cfg(target_os = "macos")]

use half::f16;
use turbospark_gpu::{
    dequant_kquant_gemm_resident, dequant_q4_k_gemv_resident, dequant_q5_k_gemv_resident,
    dequant_q6_k_gemv_resident, dequant_q8_0_gemv_resident, KQuantBatchType, KQuantResidentMatrix,
    MetalContext, Q4KResidentMatrix, Q5KResidentMatrix, Q6KResidentMatrix, Q8_0ResidentMatrix,
    MAX_BATCH_ROWS,
};

fn lcg(seed: u32) -> impl FnMut() -> u32 {
    let mut s = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
    move || {
        s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        s
    }
}

fn activations(n: usize, seed: u32) -> Vec<f16> {
    let mut next = lcg(seed);
    (0..n)
        .map(|_| f16::from_f32(((next() >> 8) as f32 / (1u32 << 23) as f32) - 1.0))
        .collect()
}

/// A small finite f16 super-scale, so no block decodes to Inf or NaN (random
/// bytes would, and a NaN compares unequal to itself bit for bit only by
/// luck).
fn scale_bytes(next: &mut impl FnMut() -> u32) -> [u8; 2] {
    let v = 0.001 + (next() % 1000) as f32 * 0.00002;
    f16::from_f32(v).to_le_bytes()
}

/// `m` rows of random K-quant blocks with sane super-scales patched in.
/// Every other byte (sub-scales, mins, quants) is uniformly random, which
/// covers the split 6-bit scales of Q4_K and the signed scales of Q6_K.
fn random_rows(kind: KQuantBatchType, m: usize, n: usize, seed: u32) -> Vec<u8> {
    let (block_bytes, block_elems, scale_at): (usize, usize, &[usize]) = match kind {
        KQuantBatchType::Q4K => (144, 256, &[0, 2]),
        KQuantBatchType::Q5K => (176, 256, &[0, 2]),
        KQuantBatchType::Q6K => (210, 256, &[208]),
        KQuantBatchType::Q8_0 => (34, 32, &[0]),
    };
    let mut next = lcg(seed);
    let blocks = m * n / block_elems;
    let mut out = Vec::with_capacity(blocks * block_bytes);
    for _ in 0..blocks {
        let mut blk: Vec<u8> = (0..block_bytes).map(|_| next() as u8).collect();
        for &at in scale_at {
            blk[at..at + 2].copy_from_slice(&scale_bytes(&mut next));
        }
        out.extend_from_slice(&blk);
    }
    assert_eq!(out.len(), m * kind.row_bytes(n));
    out
}

fn gemv(
    context: &mut MetalContext,
    kind: KQuantBatchType,
    buffer: &metal::Buffer,
    offset: u64,
    m: usize,
    n: usize,
    x: &[f16],
) -> Vec<f16> {
    match kind {
        KQuantBatchType::Q4K => {
            let w = Q4KResidentMatrix {
                buffer,
                weights_offset: offset,
                rows: m,
                cols: n,
            };
            dequant_q4_k_gemv_resident(context, &w, x)
        }
        KQuantBatchType::Q5K => {
            let w = Q5KResidentMatrix {
                buffer,
                weights_offset: offset,
                rows: m,
                cols: n,
            };
            dequant_q5_k_gemv_resident(context, &w, x)
        }
        KQuantBatchType::Q6K => {
            let w = Q6KResidentMatrix {
                buffer,
                weights_offset: offset,
                rows: m,
                cols: n,
            };
            dequant_q6_k_gemv_resident(context, &w, x)
        }
        KQuantBatchType::Q8_0 => {
            let w = Q8_0ResidentMatrix {
                buffer,
                weights_offset: offset,
                rows: m,
                cols: n,
            };
            dequant_q8_0_gemv_resident(context, &w, x)
        }
    }
    .expect("gemv dispatch")
}

/// `m` is not a multiple of eight rows per threadgroup, so the early return
/// runs, and `n` spans two Q4_K/Q6_K superblocks (sixteen Q8_0 blocks) so a
/// block-stride bug is reachable. The matrix sits at a nonzero offset behind
/// filler bytes, the resident form the runtime dispatches.
fn assert_every_width_bit_identical(context: &mut MetalContext, kind: KQuantBatchType) {
    let m = 67usize;
    let n = 4096usize;
    let rows = random_rows(kind, m, n, 7);
    let pad = 4096usize;
    let mut blob = vec![0xABu8; pad];
    blob.extend_from_slice(&rows);
    blob.extend_from_slice(&[0xCDu8; 256]);
    let buffer = context.new_buffer_with_data(&blob);
    let w = KQuantResidentMatrix {
        buffer: &buffer,
        weights_offset: pad as u64,
        rows: m,
        cols: n,
        kind,
    };

    for batch in 1..=MAX_BATCH_ROWS {
        let x: Vec<Vec<f16>> = (0..batch).map(|b| activations(n, 500 + b as u32)).collect();
        let mut expected = Vec::with_capacity(batch * m);
        for xb in &x {
            expected.extend(gemv(context, kind, &buffer, pad as u64, m, n, xb));
        }
        let flat: Vec<f16> = x.iter().flatten().copied().collect();
        let batched =
            dequant_kquant_gemm_resident(context, &w, &flat, batch).expect("batch dispatch");
        assert_eq!(batched.len(), expected.len());
        assert!(
            expected.iter().any(|v| v.to_f32() != 0.0),
            "{kind:?}: the fixture produced all-zero outputs, so this comparison proves nothing"
        );
        let differing = batched
            .iter()
            .zip(expected.iter())
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        assert_eq!(
            differing,
            0,
            "{kind:?} batch {batch}: {differing} of {} outputs differ from B GEMV calls. The \
             batched kernel mirrors the GEMV's per-row arithmetic and both compile with \
             the same flags, so any difference is a batching bug, not rounding.",
            batched.len()
        );
    }
}

#[test]
fn every_kquant_batch_width_is_bit_identical_to_b_gemv_calls() {
    let mut context = MetalContext::new().expect("Metal device available on this machine");
    for kind in [
        KQuantBatchType::Q4K,
        KQuantBatchType::Q5K,
        KQuantBatchType::Q6K,
        KQuantBatchType::Q8_0,
    ] {
        assert_every_width_bit_identical(&mut context, kind);
    }
}
