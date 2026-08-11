//! Is `dequant_int4_gemv_simd` bandwidth-bound at the shapes decode uses?
//!
//! ROADMAP's speculative-decoding item needs a batched (M-row) verify pass
//! to be materially cheaper than M sequential decode passes, and whether
//! that is possible is decided by ONE property of the existing kernel. If
//! it already saturates memory at decode's shapes, batching cannot help:
//! the weight bytes have to move either way and they are the cost. If the
//! shapes decode actually dispatches read far less than the same kernel
//! reaches when it is given enough rows to fill the GPU, then it is
//! occupancy-bound, one weight load can serve M rows instead of 1, and a
//! batched verify is worth building.
//!
//! The reference is THE SAME KERNEL at a large row count, not a second
//! kernel and not a spec sheet. A second kernel measures its own
//! shortcomings (the first draft of this used `residual_add_fp16` and the
//! GEMV beat it by 2x, because a scalar f16 elementwise kernel is itself
//! poor); a spec number has never been hit here by anything. The large row
//! is a LOWER bound on what the device can do, which is the conservative
//! direction: it can only understate the headroom.
//!
//! Two biases, both making the small shapes look BETTER than they are in
//! production, so they do not threaten the conclusion:
//!   - the repeats hit one matrix, so anything under a few MiB is served
//!     warm, where a real token walks 1.29 GiB and revisits nothing;
//!   - there is no dependency stall, no host encode, and no other work in
//!     the buffer.
//!
//! The RATIO is the deliverable. Absolute GiB/s moves with clocks, so it
//! is not comparable across power states or sessions (Gotchas 20, 22, 28);
//! every arm runs in one process, back to back, after a discarded warmup.
//!
//! ```sh
//! cargo test -p turbospark-gpu --test gemv_bandwidth_bench --release -- --ignored --nocapture
//! ```

#![cfg(target_os = "macos")]

use metal::{FunctionConstantValues, MTLDataType};
use turbospark_gpu::{
    autorelease_pool, encode_dequant_int4_gemm_resident, encode_dequant_int4_gemv_resident,
    Int4ResidentMatrix, MetalContext, MAX_BATCH_ROWS,
};

const INT4_SOURCE: &str = include_str!("../src/shaders/dequant_int4.metal");

/// Mirrors the private `unused_function_constants` in
/// `dequant_int4_gemv.rs`: M/N stay runtime arguments.
fn int4_constants() -> FunctionConstantValues {
    let values = FunctionConstantValues::new();
    let zero: u32 = 0;
    let use_fc = false;
    values.set_constant_value_at_index((&zero as *const u32).cast(), MTLDataType::UInt, 20);
    values.set_constant_value_at_index((&zero as *const u32).cast(), MTLDataType::UInt, 21);
    values.set_constant_value_at_index((&use_fc as *const bool).cast(), MTLDataType::Bool, 22);
    values
}

/// `(label, rows, cols)`. Qwen 3.6 35B-A3B is 2048 wide, so `cols` is
/// fixed at its hidden size and `rows` sweeps what decode dispatches: a
/// routed expert's projection (512), the KV projections (512), `q_proj`
/// with its output gate (4096), and `o_proj` (2048). The last row is not a
/// real shape -- it is the reference, sized past any cache and past the
/// point where threadgroup count limits occupancy.
const SHAPES: [(&str, usize, usize); 5] = [
    ("expert       512x2048", 512, 2048),
    ("o_proj      2048x2048", 2048, 2048),
    ("q_proj      4096x2048", 4096, 2048),
    ("stacked     8192x2048", 8192, 2048),
    ("reference  32768x2048", 32768, 2048),
];

/// Weight bytes each timed call must move. Repeats are chosen per shape to
/// hit it, which is the fix for the second failure this bench had: with a
/// fixed repeat count the small shapes ran for tens of microseconds and
/// the large one for a few milliseconds, so every number was DVFS ramp
/// rather than steady-state throughput. An early draft read 311 GiB/s for
/// the reference shape purely because a heavy arm had run just before it
/// and left the clocks high. Equal bytes per call means equal duration,
/// which means one clock state across the table.
const TARGET_BYTES: u64 = 2 << 30;
/// Three timed rounds per shape, interleaved shape by shape rather than
/// run as three consecutive batches, so thermal drift cannot land on one
/// shape (the CLAUDE.local.md A/B rule, applied inside one process).
const ROUNDS: usize = 3;

fn gib_per_sec(bytes: u64, seconds: f64) -> f64 {
    bytes as f64 / seconds / (1024.0 * 1024.0 * 1024.0)
}

/// Bytes one INT4-affine GEMV must read: the packed nibbles plus a BF16
/// scale and bias per group of 64. The `x` vector and `y` output are
/// negligible beside them and are left out on purpose, so the number is
/// "weight bytes moved", which is exactly what batching amortizes.
fn weight_bytes(rows: usize, cols: usize) -> u64 {
    (rows * cols / 2 + 2 * (rows * cols / 64) * 2) as u64
}

/// Buffers allocated ONCE per shape and reused across every round. This is
/// load-bearing, not tidiness: a freshly allocated Metal buffer is zero
/// pages that fault in on the GPU's first touch, so allocating inside the
/// timed call measures the fault, not the kernel. The first draft did, and
/// read 35 then 68 GiB/s for one shape in consecutive rounds.
struct ShapeBuffers {
    weights: metal::Buffer,
    x: metal::Buffer,
    y: metal::Buffer,
    scales_offset: u64,
    biases_offset: u64,
    rows: usize,
    cols: usize,
    repeats: usize,
}

impl ShapeBuffers {
    fn new(context: &MetalContext, rows: usize, cols: usize) -> Self {
        let scales_offset = (rows * cols / 2) as u64;
        let biases_offset = scales_offset + (rows * cols / 64 * 2) as u64;
        let total = biases_offset + (rows * cols / 64 * 2) as u64;
        Self {
            repeats: (TARGET_BYTES / weight_bytes(rows, cols)).max(1) as usize,
            // Zero-filled: the kernel reads every byte whatever it holds,
            // and all-zero scales keep the output finite without staging
            // real weights.
            weights: context.new_output_buffer(total),
            x: context.new_output_buffer((cols * 2) as u64),
            y: context.new_output_buffer((rows * 2) as u64),
            scales_offset,
            biases_offset,
            rows,
            cols,
        }
    }
}

/// One timed command buffer holding `REPEATS` back-to-back dispatches,
/// returning the achieved weight-byte rate. The repeats share one buffer
/// so the number is device time, not commit overhead.
fn rate(context: &mut MetalContext, shape: &ShapeBuffers) -> f64 {
    let seconds = autorelease_pool(|| {
        let pass = context.begin_pass();
        for _ in 0..shape.repeats {
            let matrix = Int4ResidentMatrix {
                buffer: &shape.weights,
                weights_offset: 0,
                scales_offset: shape.scales_offset,
                biases_offset: shape.biases_offset,
                rows: shape.rows,
                cols: shape.cols,
            };
            encode_dequant_int4_gemv_resident(
                context,
                &pass,
                &matrix,
                (&shape.x, 0),
                (&shape.y, 0),
            )
            .unwrap();
        }
        pass.commit_and_wait_with_gpu_time()
    });
    gib_per_sec(
        weight_bytes(shape.rows, shape.cols) * shape.repeats as u64,
        seconds,
    )
}

#[test]
#[ignore = "benchmark: needs a real Metal device, reports rather than asserts"]
fn int4_gemv_headroom_across_decode_shapes() {
    let mut context = MetalContext::new().expect("Metal device");
    let shapes: Vec<ShapeBuffers> = SHAPES
        .iter()
        .map(|(_, rows, cols)| ShapeBuffers::new(&context, *rows, *cols))
        .collect();

    // Gotcha 20: the first dispatch after a build runs at low DVFS clocks.
    // This also faults every buffer in, which is the larger effect here.
    for shape in &shapes {
        rate(&mut context, shape);
    }

    let mut rounds = vec![Vec::with_capacity(ROUNDS); SHAPES.len()];
    for _ in 0..ROUNDS {
        for (i, shape) in shapes.iter().enumerate() {
            rounds[i].push(rate(&mut context, shape));
        }
    }

    let best = |v: &Vec<f64>| v.iter().cloned().fold(f64::MIN, f64::max);
    let reference = best(&rounds[SHAPES.len() - 1]);

    println!("\nshape                     GiB/s (3 rounds)          of reference");
    for (i, (label, _, _)) in SHAPES.iter().enumerate() {
        let r = &rounds[i];
        println!(
            "{label}   {:>6.1} {:>6.1} {:>6.1}   {:>6.2}x   {:>5.1}x headroom",
            r[0],
            r[1],
            r[2],
            best(r) / reference,
            reference / best(r)
        );
    }
    println!(
        "\nHeadroom is what an M-row batched verify could reclaim on that\n\
         shape, since M rows share one weight load. Headroom near 1.0x means\n\
         the shape is already bandwidth-bound and batching buys nothing\n\
         there. Both biases above inflate the small shapes, so a headroom\n\
         reading is a LOWER bound.\n"
    );
}

/// Phase D2's first question, and the one that decides whether the phase
/// needs a kernel at all: how much does a batch of M tokens actually cost?
///
/// A batched verify does not have to mean an M-row GEMM. The same GEMV
/// dispatched M times back to back on ONE matrix reads that matrix once
/// from memory and M times from cache, where M separate forward passes
/// walk every other layer in between and come back to a cold matrix. If
/// the cache does the amortizing, `c(M)` is already small and the phase
/// needs no MSL.
///
/// The two arms differ ONLY in dispatch order:
///   - `sequential`: M dispatches over M DIFFERENT matrices, which is what
///     M forward passes do (a pool far larger than any cache, so each is
///     cold, the way a real token's 1.29 GiB walk is);
///   - `batched`: M dispatches over ONE matrix before moving on.
/// Same kernel, same dispatch count, same bytes of allocation touched.
/// `c(M)` is batched / sequential, and the ideal is 1/M.
const POOL_BYTES: u64 = 512 << 20;
const BATCH_SIZES: [usize; 4] = [2, 4, 8, 16];

struct MatrixPool {
    weights: Vec<metal::Buffer>,
    x: metal::Buffer,
    y: metal::Buffer,
    x_batch: metal::Buffer,
    y_batch: metal::Buffer,
    scales_offset: u64,
    biases_offset: u64,
    rows: usize,
    cols: usize,
}

impl MatrixPool {
    fn new(context: &MetalContext, rows: usize, cols: usize) -> Self {
        let scales_offset = (rows * cols / 2) as u64;
        let biases_offset = scales_offset + (rows * cols / 64 * 2) as u64;
        let total = biases_offset + (rows * cols / 64 * 2) as u64;
        let count = (POOL_BYTES / total).max(BATCH_SIZES[BATCH_SIZES.len() - 1] as u64) as usize;
        Self {
            weights: (0..count)
                .map(|_| context.new_output_buffer(total))
                .collect(),
            x: context.new_output_buffer((cols * 2) as u64),
            y: context.new_output_buffer((rows * 2) as u64),
            x_batch: context.new_output_buffer((MAX_BATCH_ROWS * cols * 2) as u64),
            y_batch: context.new_output_buffer((MAX_BATCH_ROWS * rows * 2) as u64),
            scales_offset,
            biases_offset,
            rows,
            cols,
        }
    }

    fn matrix(&self, i: usize) -> Int4ResidentMatrix<'_> {
        Int4ResidentMatrix {
            buffer: &self.weights[i % self.weights.len()],
            weights_offset: 0,
            scales_offset: self.scales_offset,
            biases_offset: self.biases_offset,
            rows: self.rows,
            cols: self.cols,
        }
    }

    /// Same work as [`Self::time`], encoded into a CONCURRENT compute
    /// encoder so Metal does not barrier between dispatches. Built by hand
    /// rather than through `PassEncoder`, which only makes serial ones.
    fn time_concurrent(&self, context: &mut MetalContext, groups: usize, batch: usize) -> f64 {
        autorelease_pool(|| {
            let command_buffer = context.queue().new_command_buffer().to_owned();
            let encoder = command_buffer
                .compute_command_encoder_with_dispatch_type(metal::MTLDispatchType::Concurrent)
                .to_owned();
            let pipeline = context
                .pipeline(
                    INT4_SOURCE,
                    "dequant_int4_gemv_simd",
                    &int4_constants(),
                    b"",
                )
                .expect("pipeline");
            for group in 0..groups {
                let matrix = self.matrix(group);
                for _ in 0..batch {
                    encoder.set_compute_pipeline_state(&pipeline);
                    encoder.set_buffer(0, Some(matrix.buffer), 0);
                    encoder.set_buffer(1, Some(matrix.buffer), self.scales_offset);
                    encoder.set_buffer(2, Some(matrix.buffer), self.biases_offset);
                    encoder.set_buffer(3, Some(&self.x), 0);
                    encoder.set_buffer(4, Some(&self.y), 0);
                    let m = self.rows as u32;
                    let n = self.cols as u32;
                    encoder.set_bytes(5, 4, (&m as *const u32).cast());
                    encoder.set_bytes(6, 4, (&n as *const u32).cast());
                    encoder.dispatch_thread_groups(
                        metal::MTLSize::new(self.rows.div_ceil(8) as u64, 1, 1),
                        metal::MTLSize::new(256, 1, 1),
                    );
                }
            }
            encoder.end_encoding();
            // Wall clock rather than GPUStartTime/GPUEndTime: this arm
            // builds its command buffer by hand and both arms of the
            // comparison run milliseconds of work, so the commit overhead
            // is noise. The serial arm's own number below is taken the
            // same way for that reason.
            let started = std::time::Instant::now();
            command_buffer.commit();
            command_buffer.wait_until_completed();
            started.elapsed().as_secs_f64()
        })
    }

    /// `groups` dispatches of the BATCHED kernel, each covering `batch`
    /// tokens against one matrix.
    fn time_gemm(&self, context: &mut MetalContext, groups: usize, batch: usize) -> f64 {
        autorelease_pool(|| {
            let pass = context.begin_pass();
            for group in 0..groups {
                encode_dequant_int4_gemm_resident(
                    context,
                    &pass,
                    &self.matrix(group),
                    (&self.x_batch, 0),
                    (&self.y_batch, 0),
                    batch,
                )
                .unwrap();
            }
            pass.commit_and_wait_with_gpu_time()
        })
    }

    /// [`Self::time`] measured on the wall clock, so it compares
    /// like-for-like against the hand-built concurrent arm.
    fn time_wall(&self, context: &mut MetalContext, groups: usize, batch: usize) -> f64 {
        let started = std::time::Instant::now();
        self.time(context, groups, batch);
        started.elapsed().as_secs_f64()
    }

    /// `groups * batch` dispatches. `batch = 1` is the sequential arm.
    fn time(&self, context: &mut MetalContext, groups: usize, batch: usize) -> f64 {
        autorelease_pool(|| {
            let pass = context.begin_pass();
            for group in 0..groups {
                for slot in 0..batch {
                    // Sequential walks the pool one matrix per dispatch;
                    // batched holds one matrix for `batch` dispatches.
                    let index = if batch == 1 {
                        group
                    } else {
                        group * batch + slot * (self.weights.len() / batch.max(1))
                    };
                    let matrix = if batch == 1 {
                        self.matrix(index)
                    } else {
                        self.matrix(group)
                    };
                    encode_dequant_int4_gemv_resident(
                        context,
                        &pass,
                        &matrix,
                        (&self.x, 0),
                        (&self.y, 0),
                    )
                    .unwrap();
                }
            }
            pass.commit_and_wait_with_gpu_time()
        })
    }
}

#[test]
#[ignore = "benchmark: needs a real Metal device, reports rather than asserts"]
fn batching_m_tokens_through_one_matrix_amortizes_the_weight_read() {
    let mut context = MetalContext::new().expect("Metal device");

    for (label, rows, cols) in SHAPES {
        if rows > 8192 {
            continue; // the reference shape is not a real projection
        }
        let pool = MatrixPool::new(&context, rows, cols);
        // Equal DISPATCH count across arms, sized so every arm runs long
        // enough to be past the DVFS ramp (see TARGET_BYTES above).
        let dispatches = ((TARGET_BYTES / weight_bytes(rows, cols)).max(64) as usize).max(256);

        pool.time(&mut context, dispatches, 1); // warmup, Gotcha 20
        let sequential = pool.time(&mut context, dispatches, 1);
        let per_token_sequential = sequential / dispatches as f64;

        print!("{label}  seq {:>7.1} GiB/s |", {
            gib_per_sec(weight_bytes(rows, cols) * dispatches as u64, sequential)
        });
        for batch in BATCH_SIZES {
            let groups = (dispatches / batch).max(1);
            let batched = pool.time(&mut context, groups, batch);
            let per_token = batched / (groups * batch) as f64;
            print!("  M={batch} {:>5.2}x", per_token / per_token_sequential);
        }
        println!();
    }
    println!(
        "\nc(M) per dispatch against the sequential arm. 1.00x means a batch of M\n\
         costs M separate passes (no win); 1/M would be free. Anything well under\n\
         1.00 means the cache already amortizes the weight read and a batched\n\
         verify needs no new kernel.\n"
    );
}

/// The batching arm above says M dispatches cost ~0.65M, not ~1. That is
/// not the cache failing; it is the ENCODER. Every pass in this engine
/// uses `newComputeCommandEncoder`, whose default dispatch type is SERIAL:
/// Metal inserts a barrier between consecutive dispatches, so M
/// independent GEMVs run one after another and each pays the full
/// threadgroup-launch and drain latency. A 512-row projection dispatches
/// 64 threadgroups, which does not fill this GPU on its own.
///
/// The M tokens of a verify block are independent at every projection
/// (different `x`, different `y`, same weights), so they may legally run
/// concurrently. This arm asks what that is worth before anyone writes an
/// M-row kernel: same kernel, same dispatches, one encoder built with
/// `MTLDispatchTypeConcurrent` instead.
#[test]
#[ignore = "benchmark: needs a real Metal device, reports rather than asserts"]
fn a_concurrent_encoder_is_what_batching_actually_needs() {
    let mut context = MetalContext::new().expect("Metal device");

    println!("\nshape                     serial      concurrent    speedup");
    for (label, rows, cols) in SHAPES {
        if rows > 8192 {
            continue;
        }
        let pool = MatrixPool::new(&context, rows, cols);
        let dispatches = ((TARGET_BYTES / weight_bytes(rows, cols)).max(64) as usize).max(256);
        let batch = 8;
        let groups = (dispatches / batch).max(1);

        pool.time_wall(&mut context, groups, batch);
        let serial = pool.time_wall(&mut context, groups, batch);
        pool.time_concurrent(&mut context, groups, batch);
        let concurrent = pool.time_concurrent(&mut context, groups, batch);

        let bytes = weight_bytes(rows, cols) * groups as u64;
        println!(
            "{label}  {:>7.1} GiB/s  {:>7.1} GiB/s   {:>5.2}x",
            gib_per_sec(bytes, serial),
            gib_per_sec(bytes, concurrent),
            serial / concurrent
        );
    }
    println!(
        "\nBytes counted ONCE per group of {} on both arms, since that is what a\n\
         batched verify reads. A speedup near {} means the serial barrier was the\n\
         whole cost and no M-row kernel is needed.\n",
        8, 8
    );
}

/// `c(M)`: what one batched-verify pass costs against M sequential decode
/// passes, per token, with the real `dequant_int4_gemm_simd`.
///
/// The sequential arm walks a pool far larger than any cache, one matrix
/// per dispatch, which is what M forward passes do -- a token's 1.29 GiB
/// walk revisits nothing. The batched arm is ONE dispatch per matrix
/// covering M tokens. `1/M` would be free; `1.00` would mean batching
/// bought nothing.
#[test]
#[ignore = "benchmark: needs a real Metal device, reports rather than asserts"]
fn c_of_m_for_the_batched_kernel() {
    let mut context = MetalContext::new().expect("Metal device");

    println!("\nshape                     c(M) per token, against M sequential passes");
    for (label, rows, cols) in SHAPES {
        if rows > 8192 {
            continue;
        }
        let pool = MatrixPool::new(&context, rows, cols);
        let dispatches = ((TARGET_BYTES / weight_bytes(rows, cols)).max(64) as usize).max(256);

        pool.time(&mut context, dispatches, 1);
        let sequential = pool.time(&mut context, dispatches, 1) / dispatches as f64;

        print!("{label}  ");
        for batch in BATCH_SIZES {
            let groups = (dispatches / batch).max(1);
            pool.time_gemm(&mut context, groups, batch);
            let batched = pool.time_gemm(&mut context, groups, batch) / (groups * batch) as f64;
            print!("  M={batch} {:>5.2}x", batched / sequential);
        }
        println!();
    }
    println!(
        "\nMultiply by the 0.75 compute share of a decode step, add 0.25 x the\n\
         expert-union breakeven, and the result is the verify step's cost in\n\
         decode-steps. It pays when the accepted-token count exceeds that.\n"
    );
}
