//! Semantics tests, on real Metal hardware, for the concurrent compute
//! encoder (`MetalContext::begin_pass_concurrent_labeled`,
//! docs/KAPITSA_KERNELS.md item 2). The contract a production flow would
//! rely on: a concurrent pass whose dependent dispatch groups are separated
//! by `PassEncoder::memory_barrier_with_buffers` produces BIT-identical
//! results to the same dispatch chain in a serial pass, and independent
//! dispatches stay independent. No production flow is wired yet -- the
//! per-flow dependency audit is the remaining work and the A/B that would
//! justify it is pending -- so this file is the proof the plumbing does what
//! its doc claims, and the audit's starting point.
#![cfg(target_os = "macos")]

use half::f16;
use turbospark_gpu::{
    encode_residual_add, encode_rms_norm_bf16w, read_buffer_f16, write_buffer_bytes, MetalBuffer,
    MetalContext,
};

const N: usize = 2048;

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

fn zeroed(context: &MetalContext, bytes: usize) -> MetalBuffer {
    let buffer = context.new_output_buffer(bytes as u64);
    write_buffer_bytes(&buffer, 0, &vec![0u8; bytes]);
    buffer
}

/// x -> rmsnorm(x; wA) -> a; rmsnorm(x; wB) -> b; then the residual add
/// folds `b` INTO `a` in place (`residual_add_fp16` writes its hidden
/// argument). Two independent norms feeding one dependent in-place add:
/// exactly the shape a production dependency group has. Returns the normed
/// pair AFTER the chain, `a` holding the sum.
fn run_chain(context: &mut MetalContext, concurrent: bool) -> (Vec<f16>, Vec<f16>) {
    let x_halfs: Vec<f16> = deterministic(0xC0_000, N, 1.0)
        .iter()
        .map(|&v| f16::from_f32(v))
        .collect();
    let wa_halfs: Vec<f16> = deterministic(0xC0_001, N, 0.5)
        .iter()
        .map(|&v| f16::from_f32(v))
        .collect();
    let wb_halfs: Vec<f16> = deterministic(0xC0_002, N, 0.5)
        .iter()
        .map(|&v| f16::from_f32(v))
        .collect();
    let x = context.new_buffer_with_data(&half_bytes(&x_halfs));
    let wa = context.new_buffer_with_data(&half_bytes(&wa_halfs));
    let wb = context.new_buffer_with_data(&half_bytes(&wb_halfs));
    let normed_a = zeroed(context, N * 2);
    let normed_b = zeroed(context, N * 2);

    let pass = if concurrent {
        context.begin_pass_concurrent_labeled("concurrent chain")
    } else {
        context.begin_pass_labeled("serial chain")
    };
    encode_rms_norm_bf16w(
        context,
        &pass,
        (&x, 0),
        (&wa, 0),
        (&normed_a, 0),
        N as u32,
        1e-6,
    )
    .expect("norm a");
    encode_rms_norm_bf16w(
        context,
        &pass,
        (&x, 0),
        (&wb, 0),
        (&normed_b, 0),
        N as u32,
        1e-6,
    )
    .expect("norm b");
    if concurrent {
        // The add reads what BOTH norms wrote: the barrier is the whole
        // point of this test.
        pass.memory_barrier_with_buffers(&[&normed_a, &normed_b]);
    }
    encode_residual_add(context, &pass, (&normed_a, 0), (&normed_b, 0), N as u32).expect("add");
    pass.commit_and_wait();

    (
        read_buffer_f16(&normed_a, 0, N),
        read_buffer_f16(&normed_b, 0, N),
    )
}

#[test]
fn a_barriered_concurrent_pass_matches_the_serial_pass_bit_for_bit() {
    let mut context = MetalContext::new().expect("Metal device");
    let (serial_sum, serial_b) = run_chain(&mut context, false);
    let (con_sum, con_b) = run_chain(&mut context, true);

    for (i, (&g, &w)) in con_sum.iter().zip(&serial_sum).enumerate() {
        assert!(
            g.to_bits() == w.to_bits(),
            "concurrent chain diverged from serial at {i} (got {g:?}, want {w:?})"
        );
    }
    for (i, (&g, &w)) in con_b.iter().zip(&serial_b).enumerate() {
        assert!(
            g.to_bits() == w.to_bits(),
            "the barrier-free second norm diverged at {i}"
        );
    }
    assert!(
        serial_sum.iter().any(|&v| v.to_f32() != 0.0),
        "the chain must produce non-trivial output for the identity to mean anything"
    );
}

#[test]
fn many_independent_dispatches_survive_a_concurrent_pass() {
    // Eight independent norm dispatches into eight sinks in ONE concurrent
    // pass, each bit-identical to its serial counterpart: more in-flight
    // dispatches than a serial encoder can ever overlap, none corrupting
    // another's sink.
    let mut context = MetalContext::new().expect("Metal device");
    const LANES: usize = 8;
    let x_halfs: Vec<f16> = deterministic(0xD0_000, N, 1.0)
        .iter()
        .map(|&v| f16::from_f32(v))
        .collect();
    let x = context.new_buffer_with_data(&half_bytes(&x_halfs));
    let weights: Vec<MetalBuffer> = (0..LANES)
        .map(|i| {
            let w: Vec<f16> = deterministic(0xD1_000 + i as u64, N, 0.5)
                .iter()
                .map(|&v| f16::from_f32(v))
                .collect();
            context.new_buffer_with_data(&half_bytes(&w))
        })
        .collect();

    let mut serial: Vec<Vec<f16>> = Vec::with_capacity(LANES);
    for lane_weights in weights.iter().take(LANES) {
        let sink = zeroed(&context, N * 2);
        let pass = context.begin_pass_labeled("serial lane");
        encode_rms_norm_bf16w(
            &mut context,
            &pass,
            (&x, 0),
            (lane_weights, 0),
            (&sink, 0),
            N as u32,
            1e-6,
        )
        .expect("serial norm");
        pass.commit_and_wait();
        serial.push(read_buffer_f16(&sink, 0, N));
    }

    let sinks: Vec<MetalBuffer> = (0..LANES).map(|_| zeroed(&context, N * 2)).collect();
    let sink_refs: Vec<&metal::Buffer> = sinks.iter().collect();
    let pass = context.begin_pass_concurrent_labeled("concurrent fan");
    for lane in 0..LANES {
        encode_rms_norm_bf16w(
            &mut context,
            &pass,
            (&x, 0),
            (&weights[lane], 0),
            (&sinks[lane], 0),
            N as u32,
            1e-6,
        )
        .expect("concurrent norm");
    }
    pass.memory_barrier_with_buffers(&sink_refs);
    pass.commit_and_wait();

    for lane in 0..LANES {
        let got = read_buffer_f16(&sinks[lane], 0, N);
        for (i, (&g, &w)) in got.iter().zip(&serial[lane]).enumerate() {
            assert!(
                g.to_bits() == w.to_bits(),
                "lane {lane} diverged at {i} (got {g:?}, want {w:?})"
            );
        }
    }
}
