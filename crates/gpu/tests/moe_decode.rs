#![cfg(target_os = "macos")]
//! Parity test for the vendored `moe.metal` decode pair
//! (`moe_phase1_gate_up_act_u16load` + `moe_phase2_down_reduce_k8`)
//! against the `turbospark_compute` CPU MoE reference: same INT4-affine
//! expert weights, same gated activation, same weighted combine, on real
//! Metal hardware, reading the expert blobs through a real argument
//! buffer exactly as the runtime does.

use half::f16;
use turbospark_compute::quant::Int4AffineRow;
use turbospark_gpu::{MetalContext, MoeExpertOffsets, RoutedBlobsBuffer};

const D: usize = 64;
const F: usize = 64;

fn deterministic_row(seed: u64, n: usize) -> Vec<f32> {
    let mut state = seed.wrapping_mul(2_654_435_761).wrapping_add(0x9E37_79B9);
    (0..n)
        .map(|i| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state = state.wrapping_add(i as u64);
            ((state % 2000) as f32 / 1000.0) - 1.0
        })
        .collect()
}

fn quantized_rows(rows: usize, cols: usize, seed: u64) -> Vec<Int4AffineRow> {
    (0..rows)
        .map(|r| {
            let row = deterministic_row(seed.wrapping_add(r as u64 * 97 + 1), cols);
            turbospark_compute::quantize_int4_affine(&row)
        })
        .collect()
}

fn u16_le(v: &[u16]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 2);
    for x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

/// Packs gate/up/down row sets into one expert blob laid out like the
/// synthetic streamed install: per role, packed then scales then biases.
fn build_blob(
    gate: &[Int4AffineRow],
    up: &[Int4AffineRow],
    down: &[Int4AffineRow],
) -> (Vec<u8>, MoeExpertOffsets) {
    let mut blob = Vec::new();
    let mut offsets = [0u32; 9];
    let mut push = |i: usize, bytes: Vec<u8>, blob: &mut Vec<u8>| {
        offsets[i] = blob.len() as u32;
        blob.extend_from_slice(&bytes);
    };
    for (base, rows) in [(0usize, gate), (3, up), (6, down)] {
        let packed: Vec<u8> = rows.iter().flat_map(|r| r.packed.clone()).collect();
        let scales: Vec<u16> = rows.iter().flat_map(|r| r.scales.clone()).collect();
        let biases: Vec<u16> = rows.iter().flat_map(|r| r.biases.clone()).collect();
        push(base, packed, &mut blob);
        push(base + 1, u16_le(&scales), &mut blob);
        push(base + 2, u16_le(&biases), &mut blob);
    }
    (
        blob,
        MoeExpertOffsets {
            gate_w: offsets[0],
            gate_s: offsets[1],
            gate_b: offsets[2],
            up_w: offsets[3],
            up_s: offsets[4],
            up_b: offsets[5],
            down_w: offsets[6],
            down_s: offsets[7],
            down_b: offsets[8],
        },
    )
}

#[test]
fn moe_phase1_phase2_match_cpu_reference() {
    let mut context = MetalContext::new().expect("Metal device");
    let top_k = 2usize;

    let mut expert_rows = Vec::new();
    for e in 0..top_k {
        let seed = 5000 + 100 * e as u64;
        expert_rows.push((
            quantized_rows(F, D, seed + 1),
            quantized_rows(F, D, seed + 2),
            quantized_rows(D, F, seed + 3),
        ));
    }

    let x32: Vec<f32> = (0..D).map(|i| ((i as f32) * 0.21).sin()).collect();
    let x16: Vec<f16> = x32.iter().map(|&v| f16::from_f32(v)).collect();
    let residual32: Vec<f32> = (0..D).map(|i| ((i as f32) * 0.11).cos() * 0.5).collect();
    let residual16: Vec<f16> = residual32.iter().map(|&v| f16::from_f32(v)).collect();
    let weights = [0.6f32, 0.4f32];

    // CPU reference: residual + sum_e w_e * run_ffn_e(x16-rounded x).
    let x32_rounded: Vec<f32> = x16.iter().map(|v| v.to_f32()).collect();
    let mut expected: Vec<f32> = residual16.iter().map(|v| v.to_f32()).collect();
    for (e, (gate, up, down)) in expert_rows.iter().enumerate() {
        let out = turbospark_compute::run_ffn(gate, up, down, &x32_rounded, D, F);
        for (dst, o) in expected.iter_mut().zip(out.iter()) {
            *dst += weights[e] * o;
        }
    }

    // GPU: blobs bound through the argument buffer.
    let mut blobs = Vec::new();
    let mut offsets = None;
    for (gate, up, down) in &expert_rows {
        let (blob, off) = build_blob(gate, up, down);
        offsets.get_or_insert(off);
        blobs.push(context.new_buffer_with_data(&blob));
    }
    let offsets = offsets.unwrap();

    let to_le = |v: &[f16]| -> Vec<u8> {
        let mut out = Vec::with_capacity(v.len() * 2);
        for x in v {
            out.extend_from_slice(&x.to_bits().to_le_bytes());
        }
        out
    };
    let x_buf = context.new_buffer_with_data(&to_le(&x16));
    let residual_buf = context.new_buffer_with_data(&to_le(&residual16));
    let acts_buf = context.new_output_buffer((top_k * F * 2) as u64);
    let y_buf = context.new_output_buffer((D * 2) as u64);
    let mut routing = vec![f16::from_f32(0.0); turbospark_gpu::MAX_STREAMED_EXPERTS];
    routing[0] = f16::from_f32(weights[0]);
    routing[1] = f16::from_f32(weights[1]);
    let routing_buf = context.new_buffer_with_data(&to_le(&routing));

    let routed = RoutedBlobsBuffer::new(&mut context, false).expect("arg buffer");
    let blob_refs: Vec<(&metal::Buffer, u64)> = blobs.iter().map(|b| (b, 0u64)).collect();
    routed
        .bind(&mut context, false, &blob_refs)
        .expect("bind blobs");

    let pass = context.begin_pass();
    for blob in &blobs {
        pass.use_read_buffer(blob);
    }
    turbospark_gpu::encode_moe_phase1(
        &mut context,
        &pass,
        &routed,
        &offsets,
        (&x_buf, 0),
        (&acts_buf, 0),
        D as u32,
        F as u32,
        top_k as u32,
        false,
    )
    .expect("phase1");
    turbospark_gpu::encode_moe_phase2(
        &mut context,
        &pass,
        &routed,
        &offsets,
        (&acts_buf, 0),
        (&routing_buf, 0),
        (&residual_buf, 0),
        (&y_buf, 0),
        D as u32,
        F as u32,
        false,
    )
    .expect("phase2");
    pass.commit_and_wait();

    let got: Vec<f32> = {
        let ptr = y_buf.contents() as *const u16;
        let bits = unsafe { std::slice::from_raw_parts(ptr, D) };
        bits.iter().map(|&b| f16::from_bits(b).to_f32()).collect()
    };

    for d in 0..D {
        let want = expected[d];
        let diff = (got[d] - want).abs();
        let tol = 5e-2_f32.max(want.abs() * 3e-2);
        assert!(
            diff <= tol,
            "d={d}: got {} want {want} (diff {diff})",
            got[d]
        );
    }
}
