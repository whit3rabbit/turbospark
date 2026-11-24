#![cfg(target_os = "macos")]
//! A GGUF NVFP4 expert blob IS a native Inkling 2 NVFP4 blob: ggml
//! `block_nvfp4` rows (36 bytes per 64) behind the same 16-byte outer-scale
//! record (`docs/INKLING2_PHASE0.md`, "Checkpoint quantization"). So the
//! runtime opens a GGUF layer whose gate-up and down are both NVFP4 as the
//! native kind. This checks the claim on real Metal hardware: the SAME
//! bytes through the native pair (`encode_inkling2_moe_phase1/2`) and the
//! GGUF Inkling pair (`encode_inkling2_gguf_phase1/2`, `MoeExtType::Nvfp4`)
//! give the same output to FP16 rounding, at the Mini's dims and top 8.

use half::f16;
use turbospark_compute::inkling2::nvfp4_row_to_blocks;
use turbospark_gpu::{
    encode_inkling2_gguf_phase1, encode_inkling2_gguf_phase2, encode_inkling2_moe_phase1,
    encode_inkling2_moe_phase2, moe_gguf_source, moe_inkling2_source, Inkling2ExpertType,
    Inkling2GgufExpertType, MetalContext, MoeExpertOffsets, MoeExtType, RoutedBlobsBuffer,
    INKLING2_OUTER_SCALE_BYTES, MAX_STREAMED_EXPERTS, MOE_INKLING2_REFLECT_FUNCTION,
};

const D: usize = 2304;
const F: usize = 1152;
const TOP_K: usize = 8;
const GGUF_REFLECT: &str = "moe_phase1_gate_up_act_q8_0";

fn lcg(s: &mut u32) -> u32 {
    *s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    *s
}

/// One NVFP4 row of `cols` in ggml block layout, scales E4M3 in a small range.
fn row(cols: usize, s: &mut u32) -> Vec<u8> {
    let packed: Vec<u8> = (0..cols / 2).map(|_| (lcg(s) >> 8) as u8).collect();
    let scales: Vec<u8> = (0..cols / 16)
        .map(|_| {
            let r = lcg(s) >> 8;
            (((1 + r % 3) << 3) | ((r >> 4) & 7)) as u8
        })
        .collect();
    nvfp4_row_to_blocks(&packed, &scales).unwrap()
}

fn read_f16(b: &metal::Buffer, n: usize) -> Vec<f32> {
    unsafe { std::slice::from_raw_parts(b.contents() as *const u16, n) }
        .iter()
        .map(|&v| f16::from_bits(v).to_f32())
        .collect()
}

#[test]
fn gguf_nvfp4_blobs_run_on_the_native_pair() {
    let mut context = MetalContext::new().expect("Metal device");
    let mut s = 41u32;
    let mut blobs = Vec::new();
    let mut at = [0u32; 3];
    for e in 0..TOP_K {
        let mut blob = Vec::new();
        for o in [
            0.8 + 0.03 * e as f32,
            0.8 + 0.03 * e as f32,
            1.2 - 0.02 * e as f32,
        ] {
            blob.extend_from_slice(&o.to_le_bytes());
        }
        blob.resize(INKLING2_OUTER_SCALE_BYTES, 0);
        for (i, (rows, cols)) in [(F, D), (F, D), (D, F)].into_iter().enumerate() {
            at[i] = blob.len() as u32;
            for _ in 0..rows {
                blob.extend_from_slice(&row(cols, &mut s));
            }
        }
        blobs.push(context.new_buffer_with_data(&blob));
    }
    let offsets = MoeExpertOffsets {
        gate_w: at[0],
        gate_s: 0,
        gate_b: 0,
        up_w: at[1],
        up_s: 4,
        up_b: 0,
        down_w: at[2],
        down_s: 8,
        down_b: 0,
    };
    let x: Vec<u8> = (0..D)
        .flat_map(|i| {
            f16::from_f32(((i as f32) * 0.37).sin())
                .to_bits()
                .to_le_bytes()
        })
        .collect();
    let rw: Vec<u8> = (0..MAX_STREAMED_EXPERTS)
        .flat_map(|k| {
            let w = if k < TOP_K {
                0.2 + 0.05 * k as f32
            } else {
                0.0
            };
            f16::from_f32(w).to_bits().to_le_bytes()
        })
        .collect();
    let x_buf = context.new_buffer_with_data(&x);
    let rw_buf = context.new_buffer_with_data(&rw);
    let refs: Vec<(&metal::Buffer, u64)> = blobs.iter().map(|b| (b, 0)).collect();

    let mut run = |native: bool| -> Vec<f32> {
        let acts = context.new_output_buffer((MAX_STREAMED_EXPERTS * F * 2) as u64);
        let y = context.new_output_buffer((D * 2) as u64);
        let (source, function) = if native {
            (moe_inkling2_source(), MOE_INKLING2_REFLECT_FUNCTION)
        } else {
            (moe_gguf_source(), GGUF_REFLECT)
        };
        let routed = RoutedBlobsBuffer::new_for(&mut context, source, function, false).unwrap();
        routed
            .bind_for(&mut context, source, function, false, &refs)
            .unwrap();
        let pass = context.begin_pass();
        for b in &blobs {
            pass.use_read_buffer(b);
        }
        if native {
            let k = Inkling2ExpertType::Nvfp4;
            encode_inkling2_moe_phase1(
                &mut context,
                &pass,
                k,
                &routed,
                &offsets,
                (&x_buf, 0),
                (&acts, 0),
                D as u32,
                F as u32,
                TOP_K as u32,
                4.0,
                25.0,
            )
            .unwrap();
            encode_inkling2_moe_phase2(
                &mut context,
                &pass,
                k,
                &routed,
                &offsets,
                (&acts, 0),
                (&rw_buf, 0),
                (&y, 0),
                D as u32,
                F as u32,
                TOP_K as u32,
            )
            .unwrap();
        } else {
            let k = Inkling2GgufExpertType::Ext(MoeExtType::Nvfp4);
            encode_inkling2_gguf_phase1(
                &mut context,
                &pass,
                k,
                &routed,
                &offsets,
                (&x_buf, 0),
                (&acts, 0),
                D as u32,
                F as u32,
                TOP_K as u32,
                4.0,
                25.0,
            )
            .unwrap();
            encode_inkling2_gguf_phase2(
                &mut context,
                &pass,
                k,
                &routed,
                &offsets,
                (&acts, 0),
                (&rw_buf, 0),
                (&y, 0),
                D as u32,
                F as u32,
                TOP_K as u32,
            )
            .unwrap();
        }
        pass.commit_and_wait();
        read_f16(&y, D)
    };
    let (native, gguf) = (run(true), run(false));
    let rms = (gguf.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / D as f64).sqrt();
    assert!(rms > 0.05, "output too small to prove anything: {rms}");
    let worst = native
        .iter()
        .zip(&gguf)
        .map(|(a, b)| (*a as f64 - *b as f64).abs())
        .fold(0f64, f64::max)
        / rms;
    println!("native vs GGUF NVFP4 pair, same bytes: worst |diff| / rms {worst:.2e}");
    // FP16 output rounding of a sum of eight expert terms; a layout
    // mismatch (nibble order, scale position) would be order 1.
    assert!(
        worst < 5e-3,
        "the two pairs disagree on the same bytes: {worst:.2e}"
    );
}
