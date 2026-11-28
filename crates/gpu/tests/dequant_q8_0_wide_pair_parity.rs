//! The wide Q8_0 pair (KERN-04) with `TURBOSPARK_GGUF_KERNEL=wide`, in its own
//! test binary because the selection is read once per process.
//!
//! Same two contracts as `dequant_q4_k_wide_parity.rs`: the batched verify
//! GEMM is BIT-IDENTICAL to B calls of the GEMV at every width (what keeps
//! speculation lossless against plain decode), and the GEMV matches the CPU
//! decoder. The matrix sits at a 2-byte, not 4-byte, aligned offset, which is
//! where 34-byte blocks put half of all block starts, and its block count is
//! not a multiple of the 4 blocks a simdgroup takes per step.
#![cfg(target_os = "macos")]

use half::f16;
use turbospark_gpu::{
    dequant_kquant_gemm_resident, dequant_q8_0_gemv, dequant_q8_0_gemv_resident,
    q8_0_wide_selected, KQuantBatchType, KQuantResidentMatrix, MetalContext, Q8_0ResidentMatrix,
    MAX_BATCH_ROWS,
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
fn wide_q8_0_pair_is_bit_identical_across_widths_and_matches_cpu() {
    // Before any kernel runs: the selection is a process-wide OnceLock.
    std::env::set_var("TURBOSPARK_GGUF_KERNEL", "wide");
    assert!(
        q8_0_wide_selected(),
        "the variable must select the wide pair"
    );

    let mut context = MetalContext::new().expect("Metal device");
    let (m, blocks) = (2048usize, 133usize);
    let n = blocks * 32;
    let rows: Vec<Vec<u8>> = (0..m)
        .map(|r| turbospark_compute::quantize_q8_0(&values(n, 70 + r as u32)))
        .collect();
    let pad = 2usize;
    let mut blob = vec![0x5Au8; pad];
    for r in &rows {
        blob.extend_from_slice(r);
    }
    let buffer = context.new_buffer_with_data(&blob);
    let gemv_w = Q8_0ResidentMatrix {
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
        kind: KQuantBatchType::Q8_0,
    };

    let x0 = values(n, 9);
    let x0_h: Vec<f16> = x0.iter().map(|&v| f16::from_f32(v)).collect();
    let refs: Vec<&[u8]> = rows.iter().map(|r| r.as_slice()).collect();
    let cpu = turbospark_compute::dequant_q8_0_gemv(&refs, &x0, n);
    let gpu = dequant_q8_0_gemv_resident(&mut context, &gemv_w, &x0_h).expect("gemv");
    let scale = cpu.iter().fold(0f32, |a, &v| a.max(v.abs())).max(1.0);
    for (i, (&g, &c)) in gpu.iter().zip(&cpu).enumerate() {
        let bound = scale * 1e-2 + n as f32 * 1e-4;
        assert!((g.to_f32() - c).abs() < bound, "row {i}: gpu {g} cpu {c}");
    }

    // The fixture must be able to tell the two reductions apart: on small
    // inputs the half rounding of y can hide them, and then a verify kernel
    // left on the current reduction would pass this test. `dequant_q8_0_gemv`
    // (the copying form) always runs the current kernel.
    let current = dequant_q8_0_gemv(&mut context, &refs, &x0_h, n).expect("current gemv");
    assert!(
        current
            .iter()
            .zip(&gpu)
            .any(|(c, w)| c.to_bits() != w.to_bits()),
        "fixture too small: the wide and current reductions give identical bits"
    );

    for batch in 1..=MAX_BATCH_ROWS {
        let xs: Vec<Vec<f16>> = (0..batch)
            .map(|b| {
                values(n, 2000 + b as u32)
                    .into_iter()
                    .map(f16::from_f32)
                    .collect()
            })
            .collect();
        let flat: Vec<f16> = xs.iter().flatten().copied().collect();
        let batched = dequant_kquant_gemm_resident(&mut context, &gemm_w, &flat, batch)
            .expect("batched gemm");
        for (b, x) in xs.iter().enumerate() {
            let single = dequant_q8_0_gemv_resident(&mut context, &gemv_w, x).expect("gemv");
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
