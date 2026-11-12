//! Bit-exactness tests, on real Metal hardware, for the fused FFN kernels
//! (`dequant_int4_fused.metal`, docs/KAPITSA_KERNELS.md items 4 and 5). The
//! whole point of the fusion is that it must be INVISIBLE: every output half
//! has to match the unfused dispatch chain bit for bit, because the fused
//! path feeds the same frozen digests the unfused one froze. A tolerance
//! here would be a bug report, not a test.
#![cfg(target_os = "macos")]

use half::f16;
use turbospark_compute::quantize_int4_affine;
use turbospark_gpu::{
    encode_dequant_int4_gemv_resident, encode_down_epi_gemv, encode_gate_up_act_gemv,
    encode_gelu_mul, encode_residual_add, encode_sigmoid_scalar_mul, encode_silu_mul,
    read_buffer_f16, write_buffer_bytes, DownEpilogue, Int4ResidentMatrix, MetalBuffer,
    MetalContext,
};

fn deterministic(seed: u64, n: usize, scale: f32) -> Vec<f32> {
    let mut state = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
    (0..n)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (((state % 2000) as f32 / 1000.0) - 1.0) * scale
        })
        .collect()
}

fn half_bytes(v: &[f16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_bits().to_le_bytes()).collect()
}

fn u16_bytes(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn zeroed(context: &MetalContext, bytes: usize) -> MetalBuffer {
    let buffer = context.new_output_buffer(bytes as u64);
    write_buffer_bytes(&buffer, 0, &vec![0u8; bytes]);
    buffer
}

/// One packed INT4 projection at a 2-byte-aligned pad (the resident layout),
/// with the three offsets into its buffer.
fn packed_projection(
    context: &MetalContext,
    rows: usize,
    n: usize,
    pad: usize,
    seed: u64,
) -> (MetalBuffer, u64, u64, u64) {
    let mut weights = Vec::new();
    let mut scales = Vec::new();
    let mut biases = Vec::new();
    for r in 0..rows {
        let q = quantize_int4_affine(&deterministic(seed + r as u64 * 97 + 1, n, 0.5));
        weights.extend_from_slice(&q.packed);
        scales.extend_from_slice(&q.scales);
        biases.extend_from_slice(&q.biases);
    }
    let weights_offset = pad;
    let scale_offset = weights_offset + weights.len();
    let bias_offset = scale_offset + scales.len() * 2;
    let mut bytes = vec![0u8; bias_offset + biases.len() * 2];
    bytes[weights_offset..weights_offset + weights.len()].copy_from_slice(&weights);
    bytes[scale_offset..scale_offset + scales.len() * 2].copy_from_slice(&u16_bytes(&scales));
    bytes[bias_offset..bias_offset + biases.len() * 2].copy_from_slice(&u16_bytes(&biases));
    (
        context.new_buffer_with_data(&bytes),
        weights_offset as u64,
        scale_offset as u64,
        bias_offset as u64,
    )
}

fn resident<'a>(
    buffer: &'a MetalBuffer,
    w: u64,
    s: u64,
    b: u64,
    rows: usize,
    cols: usize,
) -> Int4ResidentMatrix<'a> {
    Int4ResidentMatrix {
        buffer,
        weights_offset: w,
        scales_offset: s,
        biases_offset: b,
        rows,
        cols,
    }
}

fn assert_bits_equal(got: &[f16], want: &[f16], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        assert!(
            g.to_bits() == w.to_bits(),
            "{what}: element {i} differs (got {g:?}, want {w:?})"
        );
    }
}

#[test]
fn fused_gate_up_act_is_bit_identical_to_two_gemvs_and_the_activation() {
    for &use_gelu in &[false, true] {
        for &(rows, cols) in &[(192usize, 256usize), (8, 4096), (37, 1792)] {
            let mut context = MetalContext::new().expect("Metal device");
            let (gate_buf, gw, gs, gb) = packed_projection(&context, rows, cols, 2, 0x11_000);
            let (up_buf, uw, us, ub) = packed_projection(&context, rows, cols, 4, 0x22_000);
            let gate = resident(&gate_buf, gw, gs, gb, rows, cols);
            let up = resident(&up_buf, uw, us, ub, rows, cols);
            let x_halfs: Vec<f16> = deterministic(0x33_000, cols, 1.0)
                .iter()
                .map(|&v| f16::from_f32(v))
                .collect();
            let x = context.new_buffer_with_data(&half_bytes(&x_halfs));

            // Reference: two GEMVs, then the elementwise activation.
            let ref_gate = zeroed(&context, rows * 2);
            let ref_up = zeroed(&context, rows * 2);
            let ref_act = zeroed(&context, rows * 2);
            let pass = context.begin_pass();
            encode_dequant_int4_gemv_resident(&mut context, &pass, &gate, (&x, 0), (&ref_gate, 0))
                .expect("gate gemv");
            encode_dequant_int4_gemv_resident(&mut context, &pass, &up, (&x, 0), (&ref_up, 0))
                .expect("up gemv");
            if use_gelu {
                encode_gelu_mul(
                    &mut context,
                    &pass,
                    (&ref_gate, 0),
                    (&ref_up, 0),
                    (&ref_act, 0),
                    rows as u32,
                )
                .expect("gelu");
            } else {
                encode_silu_mul(
                    &mut context,
                    &pass,
                    (&ref_gate, 0),
                    (&ref_up, 0),
                    (&ref_act, 0),
                    rows as u32,
                )
                .expect("silu");
            }
            pass.commit_and_wait();

            // Fused: one dispatch.
            let fused = zeroed(&context, rows * 2);
            let pass = context.begin_pass();
            encode_gate_up_act_gemv(
                &mut context,
                &pass,
                &gate,
                &up,
                (&x, 0),
                (&fused, 0),
                use_gelu,
            )
            .expect("fused gate_up_act");
            pass.commit_and_wait();

            let want = read_buffer_f16(&ref_act, 0, rows);
            let got = read_buffer_f16(&fused, 0, rows);
            assert_bits_equal(
                &got,
                &want,
                &format!("gate_up_act (gelu={use_gelu}, {rows}x{cols})"),
            );
        }
    }
}

#[test]
fn fused_down_epilogues_are_bit_identical_to_the_separate_dispatches() {
    for &(rows, cols) in &[(128usize, 512usize), (7, 2816)] {
        let mut context = MetalContext::new().expect("Metal device");
        let (down_buf, dw, ds, db) = packed_projection(&context, rows, cols, 2, 0x44_000);
        let down = resident(&down_buf, dw, ds, db, rows, cols);
        let act_halfs: Vec<f16> = deterministic(0x55_000, cols, 1.0)
            .iter()
            .map(|&v| f16::from_f32(v))
            .collect();
        let act = context.new_buffer_with_data(&half_bytes(&act_halfs));

        // -- Plain store: the fused Plain epilogue must equal the GEMV.
        let ref_plain = zeroed(&context, rows * 2);
        let pass = context.begin_pass();
        encode_dequant_int4_gemv_resident(&mut context, &pass, &down, (&act, 0), (&ref_plain, 0))
            .expect("down gemv");
        pass.commit_and_wait();
        let fused_plain = zeroed(&context, rows * 2);
        let pass = context.begin_pass();
        encode_down_epi_gemv(
            &mut context,
            &pass,
            &down,
            (&act, 0),
            (&fused_plain, 0),
            (&ref_plain, 0),
            DownEpilogue::Plain,
        )
        .expect("down plain");
        pass.commit_and_wait();
        assert_bits_equal(
            &read_buffer_f16(&fused_plain, 0, rows),
            &read_buffer_f16(&ref_plain, 0, rows),
            &format!("down plain ({rows}x{cols})"),
        );

        // -- Residual add in place: reference writes h2 then adds into the
        // stream copy; the fused epilogue adds into its own copy.
        let logit_halfs: Vec<f16> = vec![f16::from_f32(0.75)];
        let logit = context.new_buffer_with_data(&half_bytes(&logit_halfs));
        let ref_h2 = zeroed(&context, rows * 2);
        let ref_stream_halfs: Vec<f16> = deterministic(0x66_000, rows, 1.0)
            .iter()
            .map(|&v| f16::from_f32(v))
            .collect();
        let ref_stream = context.new_buffer_with_data(&half_bytes(&ref_stream_halfs));
        let pass = context.begin_pass();
        encode_dequant_int4_gemv_resident(&mut context, &pass, &down, (&act, 0), (&ref_h2, 0))
            .expect("down gemv");
        encode_residual_add(
            &mut context,
            &pass,
            (&ref_stream, 0),
            (&ref_h2, 0),
            rows as u32,
        )
        .expect("residual add");
        pass.commit_and_wait();

        // DIAGNOSTIC: residual epilogue over a ZERO stream must equal the
        // plain store, isolating epilogue math from any loop recompile drift.
        let zero_stream = zeroed(&context, rows * 2);
        let pass = context.begin_pass();
        encode_down_epi_gemv(
            &mut context,
            &pass,
            &down,
            (&act, 0),
            (&zero_stream, 0),
            (&logit, 0),
            DownEpilogue::Residual,
        )
        .expect("down residual zero");
        pass.commit_and_wait();
        let z = read_buffer_f16(&zero_stream, 0, rows);
        let p_ref = read_buffer_f16(&ref_plain, 0, rows);
        let zdiff = z
            .iter()
            .zip(&p_ref)
            .position(|(&a, &b)| a.to_bits() != b.to_bits());
        eprintln!("DIAG zero-stream residual vs plain: first diff = {zdiff:?}");

        let fused_stream = context.new_buffer_with_data(&half_bytes(&ref_stream_halfs));
        let pass = context.begin_pass();
        encode_down_epi_gemv(
            &mut context,
            &pass,
            &down,
            (&act, 0),
            (&fused_stream, 0),
            (&logit, 0),
            DownEpilogue::Residual,
        )
        .expect("down residual");
        pass.commit_and_wait();
        assert_bits_equal(
            &read_buffer_f16(&fused_stream, 0, rows),
            &read_buffer_f16(&ref_stream, 0, rows),
            &format!("down residual ({rows}x{cols})"),
        );

        // -- Shared-expert scalar gate: reference applies
        // `sigmoid_scalar_mul_fp16` to the GEMV output.
        let ref_h1 = zeroed(&context, rows * 2);
        let pass = context.begin_pass();
        encode_dequant_int4_gemv_resident(&mut context, &pass, &down, (&act, 0), (&ref_h1, 0))
            .expect("down gemv");
        encode_sigmoid_scalar_mul(&mut context, &pass, (&ref_h1, 0), (&logit, 0), rows as u32)
            .expect("sigmoid gate");
        pass.commit_and_wait();

        let fused_h1 = zeroed(&context, rows * 2);
        let pass = context.begin_pass();
        encode_down_epi_gemv(
            &mut context,
            &pass,
            &down,
            (&act, 0),
            (&fused_h1, 0),
            (&logit, 0),
            DownEpilogue::SigmoidGate,
        )
        .expect("down sigmoid");
        pass.commit_and_wait();
        assert_bits_equal(
            &read_buffer_f16(&fused_h1, 0, rows),
            &read_buffer_f16(&ref_h1, 0, rows),
            &format!("down sigmoid gate ({rows}x{cols})"),
        );
    }
}
