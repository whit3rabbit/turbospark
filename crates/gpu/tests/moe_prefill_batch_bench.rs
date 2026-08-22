#![cfg(target_os = "macos")]
//! Kernel-level `c(M)` for the BATCHED routed-expert pair at the real
//! Gemma 4 shape (D=2816, F=704, top_k=8): the per-token cost of one
//! route-list dispatch pair against M sequential decode-pair passes.
//!
//! This is the measurement `docs/BATCHED_PREFILL.md` steps 2 and 3 exist
//! to produce: its `fully batched` column carries a 0.287 PROXY for the
//! routed pair (a resident INT4 GEMV measured at a routed expert's
//! shape) because no batched form existed to measure. Whatever this
//! reads replaces that number.
//!
//! Same conventions as `gemv_bandwidth_bench.rs`: ratios, never absolute
//! microseconds; a discarded warmup arm per configuration (the first run
//! after a build executes at low DVFS clocks); ROUTES ARE RAGGED, with a
//! union above eight experts so the wide binding does real work. The
//! sequential arm pays the same two dispatches per token the decode path
//! pays; the batched arm pays one pair per M tokens. Host-side work
//! (bind, route build) is deliberately outside both arms -- it belongs
//! to the end-to-end A/B, not to the kernels.

use half::f16;
use metal::Buffer;
use turbospark_gpu::{
    encode_moe_phase1, encode_moe_phase2, encode_moe_prefill_phase1,
    encode_moe_prefill_phase2_fused, MetalContext, MoeExpertOffsets, MoePrefillRoute,
    RoutedBlobsBuffer, RoutedBlobsWideBuffer, MAX_STREAMED_EXPERTS,
};

const D: usize = 2816;
const F: usize = 704;
const TOP_K: usize = 8;
/// Distinct experts per (M, union) cell, from the measured prefill union
/// table (`docs/BATCHED_PREFILL.md`): 13.2 / 20.4 / 29.9 at
/// M = 2 / 4 / 8. M=16 (union 41.5) is deliberately absent: the engine
/// caps a routed sub-batch at `union <= slot_count <= 32`, so an M=16
/// batch is unreachable by construction and a c(16) here would measure
/// a configuration the runtime cannot dispatch.
const UNIONS: [(usize, usize); 3] = [(2, 13), (4, 20), (8, 30)];
const ITERATIONS: usize = 50;
const WARMUPS: usize = 3;

fn to_le(v: &[f16]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 2);
    for x in v {
        out.extend_from_slice(&x.to_bits().to_le_bytes());
    }
    out
}

/// Blob layout mirroring `build_blob` in the parity test: per role
/// (gate, up, down) the packed run, then BF16 scales, then BF16 biases,
/// contiguously. Gate/up are `[F, D]`; down is `[D, F]`.
struct BlobLayout {
    offsets: MoeExpertOffsets,
    bytes: usize,
}

fn blob_layout() -> BlobLayout {
    let gate_p = F * D / 2;
    let gate_s = F * (D / 64) * 2;
    let up_p = F * D / 2;
    let up_s = F * (D / 64) * 2;
    let down_p = D * F / 2;
    let down_s = D * (F / 64) * 2;
    let mut at = 0usize;
    let mut place = |len: usize| {
        let off = at;
        at += len;
        off as u32
    };
    let gate_w = place(gate_p);
    let gate_s_off = place(gate_s);
    let gate_b = place(gate_s);
    let up_w = place(up_p);
    let up_s_off = place(up_s);
    let up_b = place(up_s);
    let down_w = place(down_p);
    let down_s_off = place(down_s);
    let down_b = place(down_s);
    BlobLayout {
        offsets: MoeExpertOffsets {
            gate_w,
            gate_s: gate_s_off,
            gate_b,
            up_w,
            up_s: up_s_off,
            up_b,
            down_w,
            down_s: down_s_off,
            down_b,
        },
        bytes: at,
    }
}

/// One blob of plausible INT4-affine bytes. Values do not matter to
/// timing; FINITENESS does, and 0x11-everywhere is finite in both the
/// packed nibbles and the BF16 scale/bias planes.
fn blob_bytes(len: usize) -> Vec<u8> {
    vec![0x11u8; len]
}

#[test]
#[ignore = "benchmark: needs a real Metal device, reports rather than asserts"]
fn c_of_m_for_the_batched_routed_pair() {
    let mut context = MetalContext::new().expect("Metal device");
    let layout = blob_layout();
    let offsets = layout.offsets;
    let blob = blob_bytes(layout.bytes);

    println!(
        "\nrouted pair, D={D} F={F} top_k={TOP_K}: c(M) per token, against M sequential decode pairs"
    );
    for (m, experts) in UNIONS {
        assert!(experts <= turbospark_gpu::MAX_PREFILL_EXPERT_BINDINGS);
        let blobs: Vec<Buffer> = (0..experts)
            .map(|_| context.new_buffer_with_data(&blob))
            .collect();
        let blob_refs: Vec<(&Buffer, u64)> = blobs.iter().map(|b| (b, 0u64)).collect();

        // Ragged routes: token t uses experts (t * TOP_K + r) % experts,
        // distinct within a token, sharing across tokens like real routing.
        let x: Vec<f16> = (0..m * D)
            .map(|i| f16::from_f32(((i as f32) * 0.13).sin() * 0.7))
            .collect();
        let x_buf = context.new_buffer_with_data(&to_le(&x));
        let weights: Vec<f16> = (0..m * TOP_K)
            .map(|i| f16::from_f32(0.05 + 0.08 * ((i % 7) as f32)))
            .collect();
        let routing_buf = context.new_buffer_with_data(&to_le(&weights));

        // The two arms INTERLEAVE at the round level, the same discipline
        // the end-to-end A/Bs use: absolute GPU times wander run to run
        // with clock state, and measuring one arm entirely before the
        // other lets that wander masquerade as a c(M) (it moved c(2) from
        // 0.44 to 1.07 between two back-to-back runs before this).
        let decode_routed = RoutedBlobsBuffer::new(&mut context, false).expect("arg buffer");
        // Output scratch is allocated ONCE per config and reused: fresh
        // buffers per round put first-touch page faults inside the
        // measured GPU interval, a cost the engine never pays (its
        // scratch is preallocated at open).
        let seq_acts = context.new_output_buffer((MAX_STREAMED_EXPERTS * F * 2) as u64);
        let seq_y = context.new_output_buffer((D * 2) as u64);
        let seq_zero = context.new_buffer_with_data(&vec![0u8; D * 2]);
        let seq_round = |context: &mut MetalContext| {
            let mut gpu = 0.0;
            for t in 0..m {
                let mut routing16 = vec![f16::from_f32(0.0); MAX_STREAMED_EXPERTS];
                for r in 0..TOP_K {
                    routing16[r] = weights[t * TOP_K + r];
                }
                let routing_buf = context.new_buffer_with_data(&to_le(&routing16));
                let refs: Vec<(&Buffer, u64)> = (0..TOP_K)
                    .map(|r| {
                        let e = (t * TOP_K + r) % experts;
                        (&blobs[e], 0u64)
                    })
                    .collect();
                decode_routed
                    .bind(context, false, &refs)
                    .expect("bind decode blobs");
                let pass = context.begin_pass();
                for b in &blobs {
                    pass.use_read_buffer(b);
                }
                encode_moe_phase1(
                    context,
                    &pass,
                    &decode_routed,
                    &offsets,
                    (&x_buf, (t * D * 2) as u64),
                    (&seq_acts, 0),
                    D as u32,
                    F as u32,
                    TOP_K as u32,
                    false,
                )
                .expect("decode phase1");
                encode_moe_phase2(
                    context,
                    &pass,
                    &decode_routed,
                    &offsets,
                    (&seq_acts, 0),
                    (&routing_buf, 0),
                    (&seq_zero, 0),
                    (&seq_y, 0),
                    D as u32,
                    F as u32,
                    false,
                )
                .expect("decode phase2");
                gpu += pass.commit_and_wait_with_gpu_time();
            }
            gpu
        };

        // Batched arm: one route-list pair for all M tokens.
        let routes: Vec<MoePrefillRoute> = (0..m)
            .flat_map(|t| {
                (0..TOP_K).map(move |r| MoePrefillRoute {
                    token: t as u32,
                    rank: r as u32,
                    slot: ((t * TOP_K + r) % experts) as u32,
                })
            })
            .collect();
        let routes_buf = context.new_buffer_with_data(&MoePrefillRoute::bytes(&routes));
        let wide = RoutedBlobsWideBuffer::new(&mut context, false).expect("wide arg buffer");
        wide.bind(&mut context, false, &blob_refs)
            .expect("bind wide blobs");
        let bat_acts = context.new_output_buffer((m * TOP_K * F * 2) as u64);
        let bat_y = context.new_output_buffer((m * D * 2) as u64);
        // Zero seed: this bench times the routed pair the way the families
        // without a shared expert dispatch it, which is the arithmetic it
        // has always timed.
        let bat_zero = context.new_buffer_with_data(&vec![0u8; m * D * 2]);
        let batched_round = |context: &mut MetalContext| {
            let pass = context.begin_pass();
            for b in &blobs {
                pass.use_read_buffer(b);
            }
            encode_moe_prefill_phase1(
                context,
                &pass,
                &wide,
                &offsets,
                (&x_buf, 0),
                (&bat_acts, 0),
                (&routes_buf, 0),
                D as u32,
                F as u32,
                TOP_K as u32,
                (m * TOP_K) as u32,
                false,
            )
            .expect("batched phase1");
            encode_moe_prefill_phase2_fused(
                context,
                &pass,
                &wide,
                &offsets,
                (&bat_acts, 0),
                (&routing_buf, 0),
                (&routes_buf, 0),
                (&bat_zero, 0),
                (&bat_y, 0),
                D as u32,
                F as u32,
                TOP_K as u32,
                m as u32,
                false,
            )
            .expect("batched phase2");
            pass.commit_and_wait_with_gpu_time()
        };

        for _ in 0..WARMUPS {
            seq_round(&mut context);
            batched_round(&mut context);
        }
        let (mut seq, mut batched) = (0.0f64, 0.0f64);
        for _ in 0..ITERATIONS {
            seq += seq_round(&mut context);
            batched += batched_round(&mut context);
        }
        let (seq, batched) = (seq / ITERATIONS as f64, batched / ITERATIONS as f64);

        println!(
            "  M={m:>2} (union {experts:>2}): sequential {:>6.3} ms/token, batched {:>6.3} ms/token, c({m}) = {:>5.3}x",
            seq * 1e3 / m as f64,
            batched * 1e3 / m as f64,
            batched / seq,
        );
    }
    println!(
        "\nRead against the routed row of `docs/BATCHED_PREFILL.md`'s fully-batched\n\
         column, whose 0.287 was a resident-GEMV proxy at this shape. Host-side\n\
         costs (argument-buffer bind, route build, router readback) are outside\n\
         both arms by construction; the end-to-end A/B owns those.\n"
    );
}
