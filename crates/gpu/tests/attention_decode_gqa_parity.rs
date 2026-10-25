//! Parity for the GQA-shared one-row decode partial
//! (`attention_decode_gqa_partial`, reached through
//! `encode_attention_decode_gqa`) on real Metal hardware, against an FP64
//! CPU reference and against the barrier kernel behind
//! `encode_attention_decode`.
//!
//! The kernel's failure modes are quiet: a wrong KV head per Q head, a
//! simdgroup whose state is merged without its rescale, or an empty
//! simdgroup or chunk weighted as if it saw a position all still produce a
//! normalized, finite, plausible output. So the cases pin the shapes where
//! each of those would differ: Qwen3.8's 6:1 grouping at head_dim 256,
//! head_dim 128 at groups 1, 2 and 8, more simdgroups and chunks than
//! positions, a windowed `kv_start`, a wrapped ring, and sinks.
#![cfg(target_os = "macos")]

use half::{bf16, f16};
use turbospark_gpu::{
    autorelease_pool, encode_attention_decode, encode_attention_decode_gqa, read_half_buffer,
    AttentionScratch, DecodeAttentionKernel, MetalContext,
};

struct Case {
    head_dim: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    seq_len: usize,
    kv_start: usize,
    /// 0 = linear layout; otherwise logical position p lives in row p % ring.
    ring: usize,
    sinks: bool,
}

struct Fixture {
    q: Vec<f16>,
    k: Vec<f16>,
    v: Vec<f16>,
    sinks: Vec<bf16>,
}

/// Deterministic values with a wide spread of scores, so the softmax is far
/// from uniform and a misweighted position moves the output.
fn fixture(case: &Case) -> Fixture {
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = move || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((state >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
    };
    let rows = if case.ring > 0 {
        case.ring
    } else {
        case.seq_len
    };
    let kv_width = case.num_kv_heads * case.head_dim;
    let q = (0..case.num_q_heads * case.head_dim)
        .map(|_| f16::from_f32(next() * 1.5))
        .collect();
    let k = (0..rows * kv_width)
        .map(|_| f16::from_f32(next() * 1.2))
        .collect();
    let v = (0..rows * kv_width)
        .map(|_| f16::from_f32(next()))
        .collect();
    let sinks = (0..case.num_q_heads)
        .map(|h| bf16::from_f32(if h % 2 == 0 { 2.5 } else { -1.0 }))
        .collect();
    Fixture { q, k, v, sinks }
}

fn reference(case: &Case, fx: &Fixture, scale: f32) -> Vec<f64> {
    let group = case.num_q_heads / case.num_kv_heads;
    let hd = case.head_dim;
    let mut out = vec![0.0f64; case.num_q_heads * hd];
    for h in 0..case.num_q_heads {
        let kvh = h / group;
        let q = &fx.q[h * hd..(h + 1) * hd];
        let row_of = |p: usize| {
            let slot = if case.ring > 0 { p % case.ring } else { p };
            (slot * case.num_kv_heads + kvh) * hd
        };
        let scores: Vec<f64> = (case.kv_start..case.seq_len)
            .map(|p| {
                let r = row_of(p);
                let dot: f64 = (0..hd).map(|i| q[i].to_f64() * fx.k[r + i].to_f64()).sum();
                dot * scale as f64
            })
            .collect();
        let sink = case.sinks.then(|| fx.sinks[h].to_f64());
        let max = scores
            .iter()
            .copied()
            .chain(sink)
            .fold(f64::NEG_INFINITY, f64::max);
        let weights: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
        let denom: f64 = weights.iter().sum::<f64>() + sink.map_or(0.0, |s| (s - max).exp());
        for (j, p) in (case.kv_start..case.seq_len).enumerate() {
            let r = row_of(p);
            for i in 0..hd {
                out[h * hd + i] += weights[j] / denom * fx.v[r + i].to_f64();
            }
        }
    }
    out
}

fn bytes(v: &[f16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_bits().to_le_bytes()).collect()
}

/// Runs the GQA kernel at (`simdgroups`, `num_chunks`, `unroll`), or
/// `encode_attention_decode` with a scratch selecting `routed` when `gqa`
/// is `None`, and returns the FP16 output.
fn run(
    context: &mut MetalContext,
    case: &Case,
    fx: &Fixture,
    scale: f32,
    gqa: Option<(u32, u32, u32)>,
    routed: DecodeAttentionKernel,
) -> Vec<f16> {
    let q = context.new_buffer_with_data(&bytes(&fx.q));
    let k = context.new_buffer_with_data(&bytes(&fx.k));
    let v = context.new_buffer_with_data(&bytes(&fx.v));
    let sink_bytes: Vec<u8> = fx
        .sinks
        .iter()
        .flat_map(|x| x.to_bits().to_le_bytes())
        .collect();
    let sinks = context.new_buffer_with_data(&sink_bytes);
    let out = context.new_output_buffer((case.num_q_heads * case.head_dim * 2) as u64);
    let scratch = AttentionScratch::new(context, case.num_q_heads as u32, case.head_dim as u32)
        .with_decode_kernel(routed);
    let sinks_arg = case.sinks.then_some((&sinks, 0u64));
    autorelease_pool(|| {
        let pass = context.begin_pass();
        match gqa {
            Some((simdgroups, num_chunks, unroll)) => encode_attention_decode_gqa(
                context,
                &pass,
                (&q, 0),
                &k,
                &v,
                &scratch,
                (&out, 0),
                case.head_dim as u32,
                case.num_q_heads as u32,
                case.num_kv_heads as u32,
                case.seq_len as u32,
                case.kv_start as u32,
                case.ring as u32,
                scale,
                sinks_arg,
                simdgroups,
                num_chunks,
                unroll,
            ),
            None => encode_attention_decode(
                context,
                &pass,
                (&q, 0),
                &k,
                &v,
                &scratch,
                (&out, 0),
                case.head_dim as u32,
                case.num_q_heads as u32,
                case.num_kv_heads as u32,
                case.seq_len as u32,
                case.kv_start as u32,
                case.ring as u32,
                scale,
                sinks_arg,
            ),
        }
        .expect("encode");
        pass.commit_and_wait();
    });
    read_half_buffer(&out, case.num_q_heads * case.head_dim)
}

/// Max |a - b| over the output. The tolerance is FP16 output rounding plus
/// FP32 reassociation: outputs are convex combinations of V in [-1, 1], so
/// one FP16 ulp near 1 is ~1e-3.
const TOL: f64 = 3e-3;

fn check(case: Case, configs: &[(u32, u32, u32)]) {
    let mut context = MetalContext::new().expect("Metal device");
    let scale = 1.0 / (case.head_dim as f32).sqrt();
    let fx = fixture(&case);
    let want = reference(&case, &fx, scale);
    let barrier = run(
        &mut context,
        &case,
        &fx,
        scale,
        None,
        DecodeAttentionKernel::Barrier,
    );
    let barrier_err = max_err(&barrier, &want);
    assert!(
        barrier_err < TOL,
        "barrier kernel vs reference: {barrier_err}"
    );
    for &config in configs {
        let got = run(
            &mut context,
            &case,
            &fx,
            scale,
            Some(config),
            DecodeAttentionKernel::Barrier,
        );
        assert!(
            got.iter().all(|x| x.to_f32().is_finite()),
            "non-finite output at {config:?}"
        );
        let err = max_err(&got, &want);
        assert!(err < TOL, "gqa {config:?} vs reference: {err}");
        let vs_barrier = got
            .iter()
            .zip(&barrier)
            .map(|(a, b)| (a.to_f64() - b.to_f64()).abs())
            .fold(0.0, f64::max);
        assert!(
            vs_barrier < TOL,
            "gqa {config:?} vs barrier kernel: {vs_barrier}"
        );
    }
}

fn max_err(got: &[f16], want: &[f64]) -> f64 {
    assert_eq!(got.len(), want.len());
    got.iter()
        .zip(want)
        .map(|(a, b)| (a.to_f64() - b).abs())
        .fold(0.0, f64::max)
}

#[test]
fn qwen38_shape_matches_reference_across_lengths() {
    for seq_len in [1, 7, 300, 4097] {
        check(
            Case {
                head_dim: 256,
                num_q_heads: 24,
                num_kv_heads: 4,
                seq_len,
                kv_start: 0,
                ring: 0,
                sinks: false,
            },
            &[(8, 1, 1), (8, 16, 4), (16, 16, 2), (4, 8, 3)],
        );
    }
}

/// More chunks and simdgroups than positions: most simdgroup states and
/// most chunks are empty and must contribute exactly nothing.
#[test]
fn empty_simdgroups_and_chunks_contribute_nothing() {
    check(
        Case {
            head_dim: 256,
            num_q_heads: 24,
            num_kv_heads: 4,
            seq_len: 3,
            kv_start: 0,
            ring: 0,
            sinks: false,
        },
        &[(16, 16, 4), (16, 1, 4), (1, 16, 1), (1, 1, 4)],
    );
}

#[test]
fn head_dim_128_at_groups_one_two_and_eight() {
    for (num_q_heads, num_kv_heads) in [(8, 8), (16, 8), (32, 4)] {
        check(
            Case {
                head_dim: 128,
                num_q_heads,
                num_kv_heads,
                seq_len: 777,
                kv_start: 0,
                ring: 0,
                sinks: false,
            },
            &[(8, 16, 4), (16, 16, 2), (2, 4, 1)],
        );
    }
}

#[test]
fn windowed_range_and_wrapped_ring() {
    check(
        Case {
            head_dim: 256,
            num_q_heads: 16,
            num_kv_heads: 8,
            seq_len: 900,
            kv_start: 388,
            ring: 0,
            sinks: false,
        },
        &[(8, 8, 4), (16, 16, 2)],
    );
    // Window of 512 over a 512-row ring after it has wrapped.
    check(
        Case {
            head_dim: 256,
            num_q_heads: 16,
            num_kv_heads: 8,
            seq_len: 1500,
            kv_start: 988,
            ring: 512,
            sinks: false,
        },
        &[(8, 8, 4), (16, 16, 2)],
    );
}

#[test]
fn sinks_go_through_the_shared_combine() {
    check(
        Case {
            head_dim: 128,
            num_q_heads: 16,
            num_kv_heads: 2,
            seq_len: 200,
            kv_start: 0,
            ring: 0,
            sinks: true,
        },
        &[(8, 4, 4), (16, 16, 1)],
    );
}

/// Flint's shape: 32 Q over 4 KV heads at head_dim 128, learned sinks on
/// every layer, full layers linear and sliding layers a 513-position window
/// over the 641-row ring (513 + one 128-token chunk). The cases above cover
/// sinks and a wrapped ring only separately, and never at group 8. (8, 8, 4)
/// is `gqa_launch`'s production config for 4 KV heads below 16K positions.
#[test]
fn flint_shape_sinks_with_window_and_ring() {
    let flint = |seq_len, kv_start, ring| Case {
        head_dim: 128,
        num_q_heads: 32,
        num_kv_heads: 4,
        seq_len,
        kv_start,
        ring,
        sinks: true,
    };
    let configs = [(8, 8, 4), (8, 16, 4), (16, 16, 2)];
    // Full attention, linear layout.
    check(flint(3000, 0, 0), &configs);
    // Sliding layer before the window fills: the whole prefix, unwrapped.
    check(flint(300, 0, 641), &configs);
    // Sliding layer after the ring has wrapped twice.
    check(flint(1500, 1500 - 513, 641), &configs);
}

/// `encode_attention_decode` honours the scratch's kernel choice: a
/// `GqaShared` scratch produces exactly the GQA kernel's output at the
/// production launch (8 simdgroups, 8 chunks for 4 KV heads below 16K, 4
/// positions in flight), and a default scratch exactly the barrier
/// kernel's. Bit equality, because routing must not change the arithmetic
/// of either path. Skipped when the process-wide override is set, which
/// replaces the scratch's choice by design.
#[test]
fn encoder_routes_on_the_scratch_kernel_choice() {
    if std::env::var_os("TURBOSPARK_ATTN_DECODE_KERNEL").is_some() {
        eprintln!("TURBOSPARK_ATTN_DECODE_KERNEL is set; skipping the routing check");
        return;
    }
    let case = Case {
        head_dim: 256,
        num_q_heads: 24,
        num_kv_heads: 4,
        seq_len: 4097,
        kv_start: 0,
        ring: 0,
        sinks: false,
    };
    let mut context = MetalContext::new().expect("Metal device");
    let scale = 1.0 / (case.head_dim as f32).sqrt();
    let fx = fixture(&case);
    let explicit_gqa = run(
        &mut context,
        &case,
        &fx,
        scale,
        Some((8, 8, 4)),
        DecodeAttentionKernel::Barrier,
    );
    let routed_gqa = run(
        &mut context,
        &case,
        &fx,
        scale,
        None,
        DecodeAttentionKernel::GqaShared,
    );
    let routed_barrier = run(
        &mut context,
        &case,
        &fx,
        scale,
        None,
        DecodeAttentionKernel::Barrier,
    );
    let bits = |v: &[f16]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert_eq!(
        bits(&routed_gqa),
        bits(&explicit_gqa),
        "GqaShared scratch did not run the GQA launch"
    );
    assert_ne!(
        bits(&routed_gqa),
        bits(&routed_barrier),
        "the two kernels agreed bit for bit, so this case cannot tell which one ran"
    );
    let want = reference(&case, &fx, scale);
    assert!(max_err(&routed_gqa, &want) < TOL);
}
