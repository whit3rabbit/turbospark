//! What would a tree verify (kapitsa's KFlash) cost on top of the chain
//! verify, for the two pieces that are kernel work rather than bookkeeping?
//! (`docs/KAPITSA_KERNELS.md`, KFlash section, "Costing the tree".)
//!
//! 1. **GDN rows that start from their parent's state.** The shipped verify
//!    (`gdn_delta_step_prefill`, lazy build) walks its rows in one register
//!    loop, each row from the one before. A thread holds only `Dk / 32`
//!    floats of state (4 at the qwen38 shape), so a tree build can keep
//!    every node's state in registers and start node `t` from slot
//!    `parent[t]`. `tv_gdn_tree` below is that build, bench-only: the
//!    shipped kernel's math in the shipped order plus a parent array. Its
//!    cost against the shipped kernel is the GDN piece's cost.
//! 2. **KV compaction.** After a tree round the accepted path's K/V rows
//!    sit at their node slots, not at consecutive positions. Timed here as
//!    the worst case: one single-row copy per accepted node, per K and V
//!    buffer, per attention layer.
//!
//! The parity tests pin the tree build to the shipped kernel: with chain
//! parents its y is bit-identical, and on a branching tree each node's y is
//! bit-identical to the shipped kernel run over that node's root path. So
//! the timed kernel computes what a tree verify needs, not something
//! cheaper.
//!
//! Per-row RoPE positions and the ancestor mask are not timed here: the
//! first is a position array in place of `start + index`, and the second
//! touches at most 8 keys per row of a context that is hundreds long
//! (`tslab kernel --bench attn_decode` gives the whole attention's cost).
//!
//! ```sh
//! cargo test -p turbospark-gpu --test tree_verify_cost_bench --release -- --nocapture
//! cargo test -p turbospark-gpu --test tree_verify_cost_bench --release -- --ignored --nocapture
//! ```
#![cfg(target_os = "macos")]

use half::f16;
use metal::FunctionConstantValues;
use turbospark_compute::f32_to_bf16;
use turbospark_gpu::{
    autorelease_pool, encode_copy_rows_fp16_strided, encode_gdn_delta_prefill_lazy,
    read_buffer_f16, write_buffer_bytes, GdnShape, MetalBuffer, MetalContext,
};

/// qwen38-27B's linear-attention geometry (`gdn_prefill_share_bench.rs`).
const HK: usize = 16;
const HV: usize = 48;
const DK: usize = 128;
const DV: usize = 128;
const GDN_LAYERS: usize = 48;
const ATTN_LAYERS: usize = 16;
/// Full-attention K/V width per row: 4 KV heads of 256.
const KV_WIDTH: u32 = 4 * 256;
/// Verify rows at DFlash2 depth 7: the anchor plus 7 nodes.
const ROWS: usize = 8;
const REPS: usize = 256;
const ROUNDS: usize = 5;

const SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

static inline float tv_softplus(float x) {
    return (x > 20.0f) ? x : log(1.0f + exp(x));
}

// gdn_delta_step_prefill's lazy build (RECORD on, no state write-back) with
// one change: row t starts from the state row parent[t] left, or from the
// block-start state when parent[t] < 0. Every node's state stays in
// registers (Dk = 128: 4 floats a lane, 8 nodes). Same expressions in the
// same order as the shipped kernel, so a chain of parents is bit-identical.
kernel void tv_gdn_tree(
    device const half*   conv_out [[buffer(0)]],
    device const half*   a_proj   [[buffer(1)]],
    device const half*   b_proj   [[buffer(2)]],
    device const bfloat* A_log    [[buffer(3)]],
    device const bfloat* dt_bias  [[buffer(4)]],
    device const float*  state    [[buffer(5)]],
    device half*         y        [[buffer(6)]],
    constant uint&       kHeads   [[buffer(7)]],
    constant uint&       vHeads   [[buffer(8)]],
    constant uint&       keyDim   [[buffer(9)]],
    constant uint&       valueDim [[buffer(10)]],
    constant uint&       rows     [[buffer(11)]],
    constant uint&       rowStride [[buffer(12)]],
    device float*        delta_tape [[buffer(13)]],
    device const int*    parent   [[buffer(14)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint2 tpos [[thread_position_in_threadgroup]]
) {
    const uint Hk = kHeads;
    const uint Hv = vHeads;
    const uint Dk = keyDim;
    const uint Dv = valueDim;
    const uint T = rows;
    const uint C = rowStride;
    const uint h = tg.x;
    const uint dv = tg.y * 4u + tpos.y;
    const uint lane = tpos.x;
    if (h >= Hv || dv >= Dv || Dk != 128u || T > 8u) return;

    const uint hk = h / (Hv / Hk);
    const float expA = exp(float(A_log[h]));
    const float dtb = float(dt_bias[h]);
    device const float* srow = state + (uint(h) * Dv + dv) * Dk;

    float s0[4];
    for (uint i = 0; i < 4u; ++i) {
        s0[i] = srow[lane * 4u + i];
    }
    float sn[8][4];

    for (uint t = 0; t < T; ++t) {
        device const half* q = conv_out + t * C + hk * Dk;
        device const half* k = conv_out + t * C + Hk * Dk + hk * Dk;
        device const half* v = conv_out + t * C + 2u * Hk * Dk + h * Dv;
        const int p = parent[t];

        float s[4];
        for (uint i = 0; i < 4u; ++i) {
            s[i] = (p < 0) ? s0[i] : sn[p][i];
        }

        const float g = exp(-expA * tv_softplus(float(a_proj[t * Hv + h]) + dtb));
        const float beta = 1.0f / (1.0f + exp(-float(b_proj[t * Hv + h])));

        float kv = 0.0f;
        for (uint i = 0; i < 4u; ++i) {
            const uint idx = lane * 4u + i;
            s[i] *= g;
            kv = fma(s[i], float(k[idx]), kv);
        }
        kv = simd_sum(kv);

        const float delta = (float(v[dv]) - kv) * beta;
        if (lane == 0) {
            delta_tape[t * Hv * Dv + h * Dv + dv] = delta;
        }

        float out = 0.0f;
        for (uint i = 0; i < 4u; ++i) {
            const uint idx = lane * 4u + i;
            s[i] = fma(float(k[idx]), delta, s[i]);
            out = fma(s[i], float(q[idx]), out);
        }
        out = simd_sum(out);
        if (lane == 0) y[t * Hv * Dv + h * Dv + dv] = half(out);

        for (uint i = 0; i < 4u; ++i) {
            sn[t][i] = s[i];
        }
    }
}
// Control arm: tv_gdn_tree's specialization (Dk = 128 unrolled, no state
// write-back) walking a plain chain with no parent array, so tree against
// chain4 is the tree's own cost and chain4 against the shipped kernel is the
// specialization's.
kernel void tv_gdn_chain4(
    device const half*   conv_out [[buffer(0)]],
    device const half*   a_proj   [[buffer(1)]],
    device const half*   b_proj   [[buffer(2)]],
    device const bfloat* A_log    [[buffer(3)]],
    device const bfloat* dt_bias  [[buffer(4)]],
    device const float*  state    [[buffer(5)]],
    device half*         y        [[buffer(6)]],
    constant uint&       kHeads   [[buffer(7)]],
    constant uint&       vHeads   [[buffer(8)]],
    constant uint&       keyDim   [[buffer(9)]],
    constant uint&       valueDim [[buffer(10)]],
    constant uint&       rows     [[buffer(11)]],
    constant uint&       rowStride [[buffer(12)]],
    device float*        delta_tape [[buffer(13)]],
    device const int*    parent   [[buffer(14)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint2 tpos [[thread_position_in_threadgroup]]
) {
    const uint Hk = kHeads;
    const uint Hv = vHeads;
    const uint Dk = keyDim;
    const uint Dv = valueDim;
    const uint T = rows;
    const uint C = rowStride;
    const uint h = tg.x;
    const uint dv = tg.y * 4u + tpos.y;
    const uint lane = tpos.x;
    if (h >= Hv || dv >= Dv || Dk != 128u || T > 8u) return;

    const uint hk = h / (Hv / Hk);
    const float expA = exp(float(A_log[h]));
    const float dtb = float(dt_bias[h]);
    device const float* srow = state + (uint(h) * Dv + dv) * Dk;

    float s[4];
    for (uint i = 0; i < 4u; ++i) {
        s[i] = srow[lane * 4u + i];
    }

    for (uint t = 0; t < T; ++t) {
        device const half* q = conv_out + t * C + hk * Dk;
        device const half* k = conv_out + t * C + Hk * Dk + hk * Dk;
        device const half* v = conv_out + t * C + 2u * Hk * Dk + h * Dv;

        const float g = exp(-expA * tv_softplus(float(a_proj[t * Hv + h]) + dtb));
        const float beta = 1.0f / (1.0f + exp(-float(b_proj[t * Hv + h])));

        float kv = 0.0f;
        for (uint i = 0; i < 4u; ++i) {
            const uint idx = lane * 4u + i;
            s[i] *= g;
            kv = fma(s[i], float(k[idx]), kv);
        }
        kv = simd_sum(kv);

        const float delta = (float(v[dv]) - kv) * beta;
        if (lane == 0) {
            delta_tape[t * Hv * Dv + h * Dv + dv] = delta;
        }

        float out = 0.0f;
        for (uint i = 0; i < 4u; ++i) {
            const uint idx = lane * 4u + i;
            s[i] = fma(float(k[idx]), delta, s[i]);
            out = fma(s[i], float(q[idx]), out);
        }
        out = simd_sum(out);
        if (lane == 0) y[t * Hv * Dv + h * Dv + dv] = half(out);

    }
}
"#;

fn shape() -> GdnShape {
    GdnShape {
        num_k_heads: HK as u32,
        num_v_heads: HV as u32,
        key_head_dim: DK as u32,
        value_head_dim: DV as u32,
        conv_kernel_size: 4,
    }
}

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

fn half_bytes(v: &[f32]) -> Vec<u8> {
    v.iter()
        .flat_map(|&x| f16::from_f32(x).to_bits().to_le_bytes())
        .collect()
}

fn bf16_bytes(v: &[f32]) -> Vec<u8> {
    v.iter()
        .flat_map(|&x| f32_to_bf16(x).to_le_bytes())
        .collect()
}

/// Per-row inputs as host values, so a path can be gathered into its own
/// contiguous buffer for the reference run.
struct HostRows {
    conv: Vec<f32>,
    a: Vec<f32>,
    b: Vec<f32>,
}

impl HostRows {
    fn new(rows: usize) -> Self {
        let c = shape().qkv_dim() as usize;
        Self {
            // Post-qk-norm scale keeps the recurrence well inside FP16.
            conv: deterministic(0xD1CE, rows * c, 0.25),
            a: deterministic(0xA000, rows * HV, 1.5),
            b: deterministic(0xB000, rows * HV, 1.0),
        }
    }

    fn gather(&self, path: &[usize]) -> HostRows {
        let c = shape().qkv_dim() as usize;
        let pick = |v: &[f32], w: usize| -> Vec<f32> {
            path.iter()
                .flat_map(|&r| v[r * w..(r + 1) * w].iter().copied())
                .collect()
        };
        HostRows {
            conv: pick(&self.conv, c),
            a: pick(&self.a, HV),
            b: pick(&self.b, HV),
        }
    }
}

struct Bufs {
    conv_out: MetalBuffer,
    a: MetalBuffer,
    b: MetalBuffer,
    a_log: MetalBuffer,
    dt_bias: MetalBuffer,
    state: MetalBuffer,
    y: MetalBuffer,
    tape: MetalBuffer,
    parent: MetalBuffer,
    rows: usize,
}

impl Bufs {
    fn new(context: &MetalContext, host: &HostRows, rows: usize, parent: &[i32]) -> Self {
        let s = shape();
        let state = deterministic(0x5EED, HV * DV * DK, 0.05);
        let parents: Vec<u8> = parent.iter().flat_map(|p| p.to_le_bytes()).collect();
        let bufs = Self {
            conv_out: context.new_buffer_with_data(&half_bytes(&host.conv)),
            a: context.new_buffer_with_data(&half_bytes(&host.a)),
            b: context.new_buffer_with_data(&half_bytes(&host.b)),
            a_log: context.new_buffer_with_data(&bf16_bytes(&deterministic(0xA100, HV, 1.25))),
            dt_bias: context.new_buffer_with_data(&bf16_bytes(&deterministic(0xA200, HV, 0.5))),
            state: context.new_buffer_with_data(
                &state
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<u8>>(),
            ),
            y: context.new_output_buffer((rows * s.value_dim() as usize * 2) as u64),
            tape: context.new_output_buffer((rows * s.value_dim() as usize * 4) as u64),
            parent: context.new_buffer_with_data(if parents.is_empty() {
                &[0u8; 4]
            } else {
                &parents
            }),
            rows,
        };
        write_buffer_bytes(&bufs.y, 0, &vec![0u8; rows * s.value_dim() as usize * 2]);
        bufs
    }

    fn shipped(&self, context: &mut MetalContext, pass: &turbospark_gpu::PassEncoder) {
        encode_gdn_delta_prefill_lazy(
            context,
            pass,
            shape(),
            (&self.conv_out, 0),
            (&self.a, 0),
            (&self.b, 0),
            (&self.a_log, 0),
            (&self.dt_bias, 0),
            &self.state,
            (&self.y, 0),
            (&self.tape, 0),
            self.rows as u32,
        )
        .expect("shipped lazy gdn");
    }

    fn tree(&self, context: &mut MetalContext, pass: &turbospark_gpu::PassEncoder) {
        self.bench_kernel(context, pass, "tv_gdn_tree");
    }

    fn chain4(&self, context: &mut MetalContext, pass: &turbospark_gpu::PassEncoder) {
        self.bench_kernel(context, pass, "tv_gdn_chain4");
    }

    fn bench_kernel(
        &self,
        context: &mut MetalContext,
        pass: &turbospark_gpu::PassEncoder,
        name: &'static str,
    ) {
        let p = context
            .pipeline(SOURCE, name, &FunctionConstantValues::new(), b"")
            .expect("bench pipeline");
        let s = shape();
        let rows = self.rows as u32;
        let c = s.qkv_dim();
        let words = [
            (s.num_k_heads, 7u64),
            (s.num_v_heads, 8),
            (s.key_head_dim, 9),
            (s.value_head_dim, 10),
            (rows, 11),
            (c, 12),
        ];
        let raw: Vec<([u8; 4], u64)> = words.iter().map(|&(v, i)| (v.to_le_bytes(), i)).collect();
        let bytes: Vec<(&[u8], u64)> = raw.iter().map(|(v, i)| (&v[..], *i)).collect();
        pass.encode_threadgroups_3d(
            &p,
            &[
                (&self.conv_out, 0, 0),
                (&self.a, 1, 0),
                (&self.b, 2, 0),
                (&self.a_log, 3, 0),
                (&self.dt_bias, 4, 0),
                (&self.state, 5, 0),
                (&self.y, 6, 0),
                (&self.tape, 13, 0),
                (&self.parent, 14, 0),
            ],
            &bytes,
            (s.num_v_heads as u64, (s.value_head_dim / 4) as u64, 1),
            (32, 4, 1),
        );
    }

    fn y_row(&self, row: usize) -> Vec<u16> {
        let w = shape().value_dim() as usize;
        read_buffer_f16(&self.y, row * w * 2, w)
            .iter()
            .map(|v| v.to_bits())
            .collect()
    }
}

fn run(
    context: &mut MetalContext,
    f: impl FnOnce(&mut MetalContext, &turbospark_gpu::PassEncoder),
) {
    autorelease_pool(|| {
        let pass = context.begin_pass();
        f(context, &pass);
        pass.commit_and_wait();
    })
}

/// A best-first-shaped tree: a 4-deep spine with a runner-up at depth 0
/// carrying its own child, and a sibling at depth 2. Row 0 is the anchor.
const BRANCHING: [i32; ROWS] = [-1, 0, 1, 2, 3, 0, 5, 2];

fn path(parent: &[i32], node: usize) -> Vec<usize> {
    let mut p = vec![node];
    let mut at = parent[node];
    while at >= 0 {
        p.push(at as usize);
        at = parent[at as usize];
    }
    p.reverse();
    p
}

#[test]
fn the_tree_build_with_chain_parents_is_the_shipped_kernel() {
    let mut context = MetalContext::new().expect("Metal device");
    let host = HostRows::new(ROWS);
    let chain: Vec<i32> = (0..ROWS as i32).map(|t| t - 1).collect();
    let shipped = Bufs::new(&context, &host, ROWS, &chain);
    let tree = Bufs::new(&context, &host, ROWS, &chain);
    run(&mut context, |c, p| shipped.shipped(c, p));
    let chain4 = Bufs::new(&context, &host, ROWS, &chain);
    run(&mut context, |c, p| tree.tree(c, p));
    run(&mut context, |c, p| chain4.chain4(c, p));
    for r in 0..ROWS {
        assert_eq!(shipped.y_row(r), tree.y_row(r), "row {r}");
        assert_eq!(shipped.y_row(r), chain4.y_row(r), "chain4 row {r}");
    }
}

#[test]
fn each_tree_node_matches_the_shipped_kernel_on_its_root_path() {
    let mut context = MetalContext::new().expect("Metal device");
    let host = HostRows::new(ROWS);
    let tree = Bufs::new(&context, &host, ROWS, &BRANCHING);
    run(&mut context, |c, p| tree.tree(c, p));
    for node in 0..ROWS {
        let rows = path(&BRANCHING, node);
        let gathered = host.gather(&rows);
        let reference = Bufs::new(&context, &gathered, rows.len(), &[]);
        run(&mut context, |c, p| reference.shipped(c, p));
        assert_eq!(
            reference.y_row(rows.len() - 1),
            tree.y_row(node),
            "node {node} on path {rows:?}"
        );
    }
}

fn best_of(context: &mut MetalContext, mut f: impl FnMut(&mut MetalContext) -> f64) -> f64 {
    f(context);
    (0..ROUNDS)
        .map(|_| f(context))
        .fold(f64::INFINITY, f64::min)
}

fn timed(
    context: &mut MetalContext,
    reps: usize,
    f: &dyn Fn(&mut MetalContext, &turbospark_gpu::PassEncoder),
) -> f64 {
    autorelease_pool(|| {
        let pass = context.begin_pass();
        for _ in 0..reps {
            f(context, &pass);
        }
        pass.commit_and_wait_with_gpu_time()
    })
}

#[test]
#[ignore = "benchmark: needs a real Metal device, reports rather than asserts"]
fn tree_verify_kernel_overhead_at_eight_rows() {
    let mut context = MetalContext::new().expect("Metal device");
    let host = HostRows::new(ROWS);
    let chain: Vec<i32> = (0..ROWS as i32).map(|t| t - 1).collect();
    let shipped = Bufs::new(&context, &host, ROWS, &chain);
    let tree_chain = Bufs::new(&context, &host, ROWS, &chain);
    let tree_branch = Bufs::new(&context, &host, ROWS, &BRANCHING);
    let chain4 = Bufs::new(&context, &host, ROWS, &chain);

    // Interleave the arms inside every round so clock drift hits all four.
    let mut best = [f64::INFINITY; 4];
    for round in 0..=ROUNDS {
        let t = [
            timed(&mut context, REPS, &|c, p| shipped.shipped(c, p)),
            timed(&mut context, REPS, &|c, p| chain4.chain4(c, p)),
            timed(&mut context, REPS, &|c, p| tree_chain.tree(c, p)),
            timed(&mut context, REPS, &|c, p| tree_branch.tree(c, p)),
        ];
        if round > 0 {
            for i in 0..4 {
                best[i] = best[i].min(t[i] / REPS as f64);
            }
        }
    }

    println!(
        "\nGDN verify recurrence, {ROWS} rows, qwen38 shape, x {GDN_LAYERS} layers per round\n"
    );
    println!("arm                          per call us   per round ms   vs shipped   vs chain4");
    for (label, v) in [
        ("shipped lazy (chain)", best[0]),
        ("chain4 (specialized chain)", best[1]),
        ("tree build, chain parents", best[2]),
        ("tree build, branching", best[3]),
    ] {
        println!(
            "{label:<27}  {:>11.2}   {:>12.3}   {:>9.3}x   {:>8.3}x",
            v * 1e6,
            v * GDN_LAYERS as f64 * 1e3,
            v / best[0],
            v / best[1]
        );
    }

    // KV compaction, worst case: every accepted node (at most 7) moved by
    // its own single-row copy, for K and V, in every attention layer. A
    // gather kernel would be one dispatch per buffer; this bounds it above.
    let k = context.new_output_buffer(64 * KV_WIDTH as u64 * 2);
    let v = context.new_output_buffer(64 * KV_WIDTH as u64 * 2);
    let compaction = best_of(&mut context, |c| {
        autorelease_pool(|| {
            let pass = c.begin_pass();
            for _layer in 0..ATTN_LAYERS {
                for buf in [&k, &v] {
                    for node in 0..7u64 {
                        let src = (40 + node) * KV_WIDTH as u64 * 2;
                        let dst = (32 + node) * KV_WIDTH as u64 * 2;
                        encode_copy_rows_fp16_strided(
                            c,
                            &pass,
                            (buf, src),
                            (buf, dst),
                            1,
                            KV_WIDTH,
                            KV_WIDTH,
                            KV_WIDTH,
                        )
                        .expect("copy row");
                    }
                }
            }
            pass.commit_and_wait_with_gpu_time()
        })
    });
    println!(
        "\nKV compaction, {} single-row copies (7 nodes x K,V x {ATTN_LAYERS} layers): {:.3} ms a round\n",
        7 * 2 * ATTN_LAYERS,
        compaction * 1e3
    );
}
