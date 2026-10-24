//! What does an 8-row verify pay on the SMALL INT4 projections?
//! (`docs/KAPITSA_KERNELS.md`, lever 9.)
//!
//! The batched verify profile of `qwen38-27b-dflash2` with the 8-row tile on
//! shows the shipped scalar GEMM (`dequant_int4_gemm_simd`) at 9.2% of the
//! pass on 128 dispatches: the attention K/V projections (1024 x 5120, 32
//! dispatches) and the GDN `in_proj_a`/`in_proj_b` (48 x 5120, 96 dispatches).
//! Their weights are tiny, about 95 MB together, which is under 0.3 ms of
//! bandwidth, so the 6 ms is launch and occupancy latency and not data.
//!
//! This prices that on the real shapes, per dispatch, with a pool of distinct
//! weight copies so no call finds its weights in the cache. The arms are the
//! ones the engine can choose between today:
//!
//!   - the 1-row GEMV (what plain decode runs),
//!   - 8 sequential GEMVs (the obvious fallback),
//!   - the shipped batched GEMM at 8 rows (what verify runs on these shapes),
//!   - the 8-row MMA tile (what the large shapes run).
//!
//! ```sh
//! cargo test -p turbospark-gpu --test small_n_gemm_bench --release -- --ignored --nocapture
//! ```
//!
//! Needs the tree that carries `dequant_int4_mma8` (an uncommitted addition
//! when this was written). Ratios and microseconds per dispatch only: the host
//! is shared, so absolute numbers move.

#![cfg(target_os = "macos")]

use turbospark_gpu::{
    autorelease_pool, encode_dequant_int4_gemm_resident, encode_dequant_int4_gemv_resident,
    encode_dequant_int4_mma8_resident, Int4ResidentMatrix, MetalContext,
};

/// `(label, rows, cols, dispatches per verify in the profile)`.
const SHAPES: [(&str, usize, usize, usize); 3] = [
    ("gdn in_proj_a/b   48x5120", 48, 5120, 96),
    ("attn k/v        1024x5120", 1024, 5120, 32),
    ("(ref) o_proj    5120x6144", 5120, 6144, 16),
];

/// Distinct weight copies per shape, so successive calls read cold weights.
const POOL: usize = 64;
const ROWS: usize = 8;
const ROUNDS: usize = 5;

struct Pool {
    weights: metal::Buffer,
    x: metal::Buffer,
    y: metal::Buffer,
    stride: u64,
    scales: u64,
    biases: u64,
    rows: usize,
    cols: usize,
}

impl Pool {
    fn new(context: &MetalContext, rows: usize, cols: usize) -> Self {
        let packed = (rows * cols / 2) as u64;
        let sb = (rows * cols / 64 * 2) as u64;
        let align = |v: u64| v.div_ceil(256) * 256;
        let scales = align(packed);
        let biases = align(scales + sb);
        let stride = align(biases + sb);
        Self {
            weights: context.new_output_buffer(stride * POOL as u64),
            x: context.new_output_buffer((ROWS * cols * 2) as u64),
            y: context.new_output_buffer((ROWS * rows * 2) as u64),
            stride,
            scales,
            biases,
            rows,
            cols,
        }
    }

    fn matrix(&self, i: usize) -> Int4ResidentMatrix<'_> {
        let base = self.stride * (i % POOL) as u64;
        Int4ResidentMatrix {
            buffer: &self.weights,
            weights_offset: base,
            scales_offset: base + self.scales,
            biases_offset: base + self.biases,
            rows: self.rows,
            cols: self.cols,
        }
    }
}

#[derive(Clone, Copy)]
enum Arm {
    Gemv1,
    Gemv8,
    Gemm8,
    Mma8,
}

impl Arm {
    fn name(self) -> &'static str {
        match self {
            Arm::Gemv1 => "gemv x1 row",
            Arm::Gemv8 => "gemv x8 (8 dispatches)",
            Arm::Gemm8 => "shipped gemm, 8 rows",
            Arm::Mma8 => "mma8 tile, 8 rows",
        }
    }
}

/// Microseconds of GPU time per logical call (one projection of 1 or 8 rows).
fn time_arm(context: &mut MetalContext, pool: &Pool, arm: Arm) -> f64 {
    let calls = POOL * 4;
    let seconds = autorelease_pool(|| {
        let pass = context.begin_pass();
        for i in 0..calls {
            let w = pool.matrix(i);
            match arm {
                Arm::Gemv1 => encode_dequant_int4_gemv_resident(
                    context,
                    &pass,
                    &w,
                    (&pool.x, 0),
                    (&pool.y, 0),
                )
                .unwrap(),
                Arm::Gemv8 => {
                    for r in 0..ROWS {
                        encode_dequant_int4_gemv_resident(
                            context,
                            &pass,
                            &w,
                            (&pool.x, (r * pool.cols * 2) as u64),
                            (&pool.y, (r * pool.rows * 2) as u64),
                        )
                        .unwrap();
                    }
                }
                Arm::Gemm8 => encode_dequant_int4_gemm_resident(
                    context,
                    &pass,
                    &w,
                    (&pool.x, 0),
                    (&pool.y, 0),
                    ROWS,
                )
                .unwrap(),
                Arm::Mma8 => encode_dequant_int4_mma8_resident(
                    context,
                    &pass,
                    &w,
                    (&pool.x, 0),
                    (&pool.y, 0),
                    ROWS,
                )
                .unwrap(),
            }
        }
        pass.commit_and_wait_with_gpu_time()
    });
    seconds * 1e6 / calls as f64
}

#[test]
#[ignore = "benchmark: needs a real Metal device, reports rather than asserts"]
fn small_projection_costs_at_eight_rows() {
    let mut context = MetalContext::new().expect("Metal device");
    let pools: Vec<Pool> = SHAPES
        .iter()
        .map(|(_, r, c, _)| Pool::new(&context, *r, *c))
        .collect();
    let arms = [Arm::Gemv1, Arm::Gemv8, Arm::Gemm8, Arm::Mma8];

    // Warmup and fault-in, then interleaved rounds with rotated arm order.
    for pool in &pools {
        for arm in arms {
            time_arm(&mut context, pool, arm);
        }
    }
    let mut samples = vec![vec![Vec::new(); arms.len()]; SHAPES.len()];
    for round in 0..ROUNDS {
        for (s, pool) in pools.iter().enumerate() {
            for k in 0..arms.len() {
                let a = (k + round) % arms.len();
                samples[s][a].push(time_arm(&mut context, pool, arms[a]));
            }
        }
    }
    let median = |v: &Vec<f64>| {
        let mut v = v.clone();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    };

    println!("\nmicroseconds of GPU time per projection, median of {ROUNDS} rounds");
    for (s, (label, _, _, per_verify)) in SHAPES.iter().enumerate() {
        println!("\n{label}   ({per_verify} per verify)");
        let base = median(&samples[s][2]);
        for (a, arm) in arms.iter().enumerate() {
            let us = median(&samples[s][a]);
            println!(
                "  {:<24} {:>8.1} us   {:>5.2}x the shipped gemm   {:>6.2} ms over {per_verify}",
                arm.name(),
                us,
                us / base,
                us * *per_verify as f64 / 1000.0
            );
        }
    }
    println!(
        "\nThe last column is what that many projections cost in one verify at\n\
         this arm's per-call time. Weights are zero-filled (the kernels read\n\
         every byte whatever it holds) and the pool defeats the cache.\n"
    );
}
