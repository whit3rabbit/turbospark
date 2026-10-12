//! Parity for the GQA-shared multi-row partial
//! (`attention_decode_gqa_rows_partial`, reached through
//! `encode_attention_decode_gqa_rows`) on real Metal hardware, against an
//! FP64 CPU reference per row and against the one-row GQA kernel called once
//! per row, which is what the prefill micro-batch ran before it.
//!
//! Row `r` attends over `[kv_start, first + r]`. The failure modes are quiet:
//! a row that sees one position past its causal end, a dead row's simdgroup
//! writing a partial, a tile staged from the wrong KV head, or a chunk split
//! that drops the union's tail all still give a normalized, finite output.
//! The cases pin the shapes where each would differ: more rows than one
//! threadgroup holds and a ragged last group, rows whose range ends inside a
//! staged tile, more chunks than positions, head_dim 128 at three groupings,
//! a windowed `kv_start`, a wrapped ring, and output rows past `live_rows`
//! that must stay untouched.
#![cfg(target_os = "macos")]

use half::f16;
use turbospark_gpu::{
    autorelease_pool, encode_attention_decode, encode_attention_decode_gqa_rows,
    encode_attention_decode_gqa_rows_with, read_half_buffer, AttentionScratch,
    DecodeAttentionKernel, GqaRowsScratch, MetalContext,
};

#[derive(Clone, Copy)]
struct Case {
    head_dim: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    /// Row 0's own position; row `r` sits at `first + r`.
    first: usize,
    rows: usize,
    kv_start: usize,
    /// 0 = linear layout; otherwise logical position p lives in row p % ring.
    ring: usize,
}

struct Fixture {
    q: Vec<f16>,
    k: Vec<f16>,
    v: Vec<f16>,
}

/// Deterministic values with a wide spread of scores, so the softmax is far
/// from uniform and a misweighted position moves the output. Alternate rows
/// get opposite Q, so a row reading its neighbour's Q is visible.
fn fixture(case: &Case) -> Fixture {
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    let mut next = move || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((state >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
    };
    let stored = if case.ring > 0 {
        case.ring
    } else {
        case.first + case.rows
    };
    let q_width = case.num_q_heads * case.head_dim;
    let kv_width = case.num_kv_heads * case.head_dim;
    let q = (0..case.rows * q_width)
        .map(|i| {
            let sign = if (i / q_width) % 2 == 0 { 1.0 } else { -1.0 };
            f16::from_f32(sign * next() * 1.5)
        })
        .collect();
    let k = (0..stored * kv_width)
        .map(|_| f16::from_f32(next() * 1.2))
        .collect();
    let v = (0..stored * kv_width)
        .map(|_| f16::from_f32(next()))
        .collect();
    Fixture { q, k, v }
}

fn reference(case: &Case, fx: &Fixture, scale: f32) -> Vec<f64> {
    let group = case.num_q_heads / case.num_kv_heads;
    let hd = case.head_dim;
    let q_width = case.num_q_heads * hd;
    let mut out = vec![0.0f64; case.rows * q_width];
    for r in 0..case.rows {
        let end = case.first + r + 1;
        for h in 0..case.num_q_heads {
            let kvh = h / group;
            let q = &fx.q[r * q_width + h * hd..r * q_width + (h + 1) * hd];
            let row_of = |p: usize| {
                let slot = if case.ring > 0 { p % case.ring } else { p };
                (slot * case.num_kv_heads + kvh) * hd
            };
            let scores: Vec<f64> = (case.kv_start..end)
                .map(|p| {
                    let k = row_of(p);
                    let dot: f64 = (0..hd).map(|i| q[i].to_f64() * fx.k[k + i].to_f64()).sum();
                    dot * scale as f64
                })
                .collect();
            let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let weights: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
            let denom: f64 = weights.iter().sum();
            for (j, p) in (case.kv_start..end).enumerate() {
                let v = row_of(p);
                for i in 0..hd {
                    out[r * q_width + h * hd + i] += weights[j] / denom * fx.v[v + i].to_f64();
                }
            }
        }
    }
    out
}

fn bytes(v: &[f16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_bits().to_le_bytes()).collect()
}

/// A value no attention output can take, so an untouched row is visible.
const SENTINEL: f32 = 7.5;
/// Output rows past `live_rows`, filled with the sentinel.
const SPARE_ROWS: usize = 2;

enum Arm {
    /// The production launch.
    Rows,
    /// (rows per threadgroup, heads per simdgroup, chunks, unroll, tile).
    RowsWith(u32, u32, u32, u32, u32),
    /// The one-row GQA kernel, once per row.
    PerRow,
}

fn run(context: &mut MetalContext, case: &Case, fx: &Fixture, scale: f32, arm: &Arm) -> Vec<f16> {
    let q_width = case.num_q_heads * case.head_dim;
    let q = context.new_buffer_with_data(&bytes(&fx.q));
    let k = context.new_buffer_with_data(&bytes(&fx.k));
    let v = context.new_buffer_with_data(&bytes(&fx.v));
    let filled = vec![f16::from_f32(SENTINEL); (case.rows + SPARE_ROWS) * q_width];
    let out = context.new_buffer_with_data(&bytes(&filled));
    let (hd, nq, nkv) = (
        case.head_dim as u32,
        case.num_q_heads as u32,
        case.num_kv_heads as u32,
    );
    let rows_scratch = GqaRowsScratch::new(context, case.rows as u32, nq, hd, 16);
    let row_scratch =
        AttentionScratch::new(context, nq, hd).with_decode_kernel(DecodeAttentionKernel::GqaShared);
    autorelease_pool(|| {
        let pass = context.begin_pass();
        match *arm {
            Arm::Rows => encode_attention_decode_gqa_rows(
                context,
                &pass,
                (&q, 0),
                &k,
                &v,
                &rows_scratch,
                (&out, 0),
                hd,
                nq,
                nkv,
                case.first as u32,
                case.rows as u32,
                case.kv_start as u32,
                case.ring as u32,
                scale,
            )
            .expect("encode rows"),
            Arm::RowsWith(group_rows, heads, chunks, unroll, tile) => {
                encode_attention_decode_gqa_rows_with(
                    context,
                    &pass,
                    (&q, 0),
                    &k,
                    &v,
                    &rows_scratch,
                    (&out, 0),
                    hd,
                    nq,
                    nkv,
                    case.first as u32,
                    case.rows as u32,
                    case.kv_start as u32,
                    case.ring as u32,
                    scale,
                    group_rows,
                    heads,
                    chunks,
                    unroll,
                    tile,
                )
                .expect("encode rows")
            }
            Arm::PerRow => {
                for r in 0..case.rows {
                    let off = (r * q_width * 2) as u64;
                    encode_attention_decode(
                        context,
                        &pass,
                        (&q, off),
                        &k,
                        &v,
                        &row_scratch,
                        (&out, off),
                        hd,
                        nq,
                        nkv,
                        (case.first + r + 1) as u32,
                        case.kv_start as u32,
                        case.ring as u32,
                        scale,
                        None,
                    )
                    .expect("encode per row");
                }
            }
        }
        pass.commit_and_wait();
    });
    read_half_buffer(&out, (case.rows + SPARE_ROWS) * q_width)
}

/// FP16 output rounding plus FP32 reassociation, as the one-row GQA parity
/// test: outputs are convex combinations of V in [-1, 1].
const TOL: f64 = 3e-3;

fn max_err(got: &[f16], want: &[f64]) -> f64 {
    got.iter()
        .zip(want)
        .map(|(a, b)| (a.to_f64() - b).abs())
        .fold(0.0, f64::max)
}

fn check(case: Case, launches: &[(u32, u32, u32, u32, u32)]) {
    let mut context = MetalContext::new().expect("Metal device");
    let scale = 1.0 / (case.head_dim as f32).sqrt();
    let fx = fixture(&case);
    let want = reference(&case, &fx, scale);
    let live = case.rows * case.num_q_heads * case.head_dim;
    let per_row = run(&mut context, &case, &fx, scale, &Arm::PerRow);
    let per_row_err = max_err(&per_row[..live], &want);
    assert!(per_row_err < TOL, "per-row GQA vs reference: {per_row_err}");
    let arms: Vec<(String, Arm)> = std::iter::once(("production".to_string(), Arm::Rows))
        .chain(launches.iter().map(|&(r, h, c, u, t)| {
            (
                format!("{:?}", (r, h, c, u, t)),
                Arm::RowsWith(r, h, c, u, t),
            )
        }))
        .collect();
    for (label, arm) in &arms {
        let got = run(&mut context, &case, &fx, scale, arm);
        assert!(
            got[..live].iter().all(|x| x.to_f32().is_finite()),
            "non-finite output at {label}"
        );
        assert!(
            got[live..].iter().all(|x| x.to_f32() == SENTINEL),
            "{label} wrote output rows past live_rows"
        );
        let err = max_err(&got[..live], &want);
        assert!(err < TOL, "rows {label} vs reference: {err}");
        let vs_per_row = got[..live]
            .iter()
            .zip(&per_row[..live])
            .map(|(a, b)| (a.to_f64() - b.to_f64()).abs())
            .fold(0.0, f64::max);
        assert!(
            vs_per_row < TOL,
            "rows {label} vs per-row GQA: {vs_per_row}"
        );
    }
}

const QWEN38: (usize, usize, usize) = (256, 24, 4);

fn qwen38(first: usize, rows: usize) -> Case {
    Case {
        head_dim: QWEN38.0,
        num_q_heads: QWEN38.1,
        num_kv_heads: QWEN38.2,
        first,
        rows,
        kv_start: 0,
        ring: 0,
    }
}

/// The prefill shapes: a 64-row Steel micro-batch from the prompt start and
/// deep into it, the scalar arm's 16 rows, and ragged tails.
#[test]
fn qwen38_shape_matches_reference_and_per_row() {
    let launches = [
        (2, 1, 2, 3, 12),
        (8, 6, 8, 4, 16),
        (4, 6, 4, 1, 7),
        (16, 6, 16, 2, 16),
        (1, 6, 1, 3, 1),
        (8, 3, 1, 4, 16),
        (5, 2, 4, 4, 16),
        (5, 1, 8, 4, 9),
    ];
    for (first, rows) in [(0, 64), (0, 1), (5, 3), (300, 64), (1000, 16), (4000, 17)] {
        check(qwen38(first, rows), &launches);
    }
}

/// More chunks than positions and more rows per threadgroup than live rows:
/// most chunks are empty for most rows and must contribute nothing, and the
/// dead rows' simdgroups must write nothing.
#[test]
fn empty_chunks_and_dead_rows_contribute_nothing() {
    check(
        qwen38(0, 3),
        &[(16, 6, 16, 4, 16), (8, 3, 16, 1, 2), (16, 6, 1, 4, 16)],
    );
}

#[test]
fn head_dim_128_at_groups_one_two_and_eight() {
    for (num_q_heads, num_kv_heads) in [(8, 8), (16, 8), (32, 4)] {
        let group = (num_q_heads / num_kv_heads) as u32;
        check(
            Case {
                head_dim: 128,
                num_q_heads,
                num_kv_heads,
                first: 777,
                rows: 21,
                kv_start: 0,
                ring: 0,
            },
            &[(8, group, 8, 4, 16), (2, 1, 4, 1, 5)],
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
            first: 900,
            rows: 12,
            kv_start: 388,
            ring: 0,
        },
        &[(8, 2, 8, 4, 16), (8, 1, 16, 2, 3)],
    );
    // Window over a 1024-row ring after it has wrapped.
    check(
        Case {
            head_dim: 256,
            num_q_heads: 16,
            num_kv_heads: 8,
            first: 1500,
            rows: 20,
            kv_start: 600,
            ring: 1024,
        },
        &[(8, 2, 8, 4, 16), (8, 1, 16, 2, 3)],
    );
}
