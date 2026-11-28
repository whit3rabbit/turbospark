//! The wide Q4_K pair (KERN-04) with `TURBOSPARK_GGUF_KERNEL=wide`, in its own
//! test binary because the selection is read once per process.
//!
//! Two contracts, both required before the wide kernels may be a default:
//! the batched verify GEMM is BIT-IDENTICAL to B calls of the GEMV at every
//! width (what keeps speculation lossless against plain decode, as
//! `dequant_kquant_gemm_parity.rs` holds for the current pair), and the GEMV
//! matches the CPU decoder. One context serves every width, so a pipeline key
//! missing B would show here too.
#![cfg(target_os = "macos")]

use half::f16;
use turbospark_gpu::{
    dequant_kquant_gemm_resident, dequant_q4_k_gemv_resident, q4_k_wide_selected, KQuantBatchType,
    KQuantResidentMatrix, MetalContext, Q4KResidentMatrix, MAX_BATCH_ROWS,
};

fn values(n: usize, seed: u32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((s >> 8) as f32 / (1u32 << 23) as f32) - 1.0
        })
        .collect()
}

#[test]
fn wide_pair_is_bit_identical_across_widths_and_matches_cpu() {
    // Before any kernel runs: the selection is a process-wide OnceLock.
    std::env::set_var("TURBOSPARK_GGUF_KERNEL", "wide");
    assert!(
        q4_k_wide_selected(),
        "the variable must select the wide pair"
    );

    let mut context = MetalContext::new().expect("Metal device");
    let (m, n) = (19usize, 4 * 256);
    let rows: Vec<Vec<u8>> = (0..m)
        .map(|r| turbospark_compute::quantize_q4_k(&values(n, 40 + r as u32)))
        .collect();
    // Behind 256 bytes of garbage: an offset of zero cannot pass by accident.
    let pad = 256usize;
    let mut blob = vec![0xCDu8; pad];
    for r in &rows {
        blob.extend_from_slice(r);
    }
    let buffer = context.new_buffer_with_data(&blob);
    let gemv_w = Q4KResidentMatrix {
        buffer: &buffer,
        weights_offset: pad as u64,
        rows: m,
        cols: n,
    };
    let gemm_w = KQuantResidentMatrix {
        buffer: &buffer,
        weights_offset: pad as u64,
        rows: m,
        cols: n,
        kind: KQuantBatchType::Q4K,
    };

    // GEMV against the CPU decoder, same bytes.
    let x0 = values(n, 7);
    let x0_h: Vec<f16> = x0.iter().map(|&v| f16::from_f32(v)).collect();
    let refs: Vec<&[u8]> = rows.iter().map(|r| r.as_slice()).collect();
    let cpu = turbospark_compute::dequant_q4_k_gemv(&refs, &x0, n);
    let gpu = dequant_q4_k_gemv_resident(&mut context, &gemv_w, &x0_h).expect("gemv");
    let scale = cpu.iter().fold(0f32, |a, &v| a.max(v.abs())).max(1.0);
    for (i, (&g, &c)) in gpu.iter().zip(&cpu).enumerate() {
        let bound = scale * 1e-2 + n as f32 * 1e-4;
        assert!((g.to_f32() - c).abs() < bound, "row {i}: gpu {g} cpu {c}");
    }

    for batch in 1..=MAX_BATCH_ROWS {
        let xs: Vec<Vec<f16>> = (0..batch)
            .map(|b| {
                values(n, 1000 + b as u32)
                    .into_iter()
                    .map(f16::from_f32)
                    .collect()
            })
            .collect();
        let flat: Vec<f16> = xs.iter().flatten().copied().collect();
        let batched = dequant_kquant_gemm_resident(&mut context, &gemm_w, &flat, batch)
            .expect("batched gemm");
        for (b, x) in xs.iter().enumerate() {
            let single = dequant_q4_k_gemv_resident(&mut context, &gemv_w, x).expect("gemv");
            let got: Vec<u16> = batched[b * m..(b + 1) * m]
                .iter()
                .map(|v| v.to_bits())
                .collect();
            let want: Vec<u16> = single.iter().map(|v| v.to_bits()).collect();
            assert_eq!(
                got, want,
                "batch {batch}, token {b}: verify differs from decode"
            );
        }
    }
}
