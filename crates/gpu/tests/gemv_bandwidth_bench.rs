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
    autorelease_pool, encode_dequant_int4_gemm_mma_resident, encode_dequant_int4_gemm_resident,
    encode_dequant_int4_gemm_resident_blocked, encode_dequant_int4_gemv_resident,
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

/// `(label, rows, cols)` for the DENSE `qwen3_5` 27B (Qwen3.8-27B, hidden
/// 5120, FFN 17408, INT4 group 64). A dense token walks every one of these
/// once, so together they are ~12.3 GB of the token's weight reads: the
/// three FFN projections on all 64 layers, the fused GDN in_proj on the 48
/// linear layers, the packed q (with output gate) on the 16 full layers,
/// and the full-vocab head once. The reference row is the same kernel past
/// any cache and past the occupancy limit, as above.
const QWEN38_SHAPES: [(&str, usize, usize); 7] = [
    ("gate/up    17408x5120", 17408, 5120),
    ("down       5120x17408", 5120, 17408),
    ("gdn_inproj 16480x5120", 16480, 5120),
    ("packed_q   12288x5120", 12288, 5120),
    ("o_proj      5120x6144", 5120, 6144),
    ("head      248320x5120", 248320, 5120),
    ("reference  32768x5120", 32768, 5120),
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

/// The same headroom question at the DENSE 27B's shapes (qwen38). Decode
/// there is ~234 GB/s effective against this kernel's measured saturation,
/// so the question is which shapes are leaving throughput on the table at
/// M=1. Ratios only, same discipline as above.
#[test]
#[ignore = "benchmark: needs a real Metal device, reports rather than asserts"]
fn int4_gemv_headroom_at_qwen38_shapes() {
    let mut context = MetalContext::new().expect("Metal device");
    let shapes: Vec<ShapeBuffers> = QWEN38_SHAPES
        .iter()
        .map(|(_, rows, cols)| ShapeBuffers::new(&context, *rows, *cols))
        .collect();

    for shape in &shapes {
        rate(&mut context, shape); // warmup + fault-in, Gotcha 20
    }

    let mut rounds = vec![Vec::with_capacity(ROUNDS); QWEN38_SHAPES.len()];
    for _ in 0..ROUNDS {
        for (i, shape) in shapes.iter().enumerate() {
            rounds[i].push(rate(&mut context, shape));
        }
    }

    let best = |v: &Vec<f64>| v.iter().cloned().fold(f64::MIN, f64::max);
    let reference = best(&rounds[QWEN38_SHAPES.len() - 1]);

    println!("\nshape                     GiB/s (3 rounds)          of reference");
    for (i, (label, _, _)) in QWEN38_SHAPES.iter().enumerate() {
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
        "\nEvery row here is walked once per dense token, so a shape far under\n\
         the reference is decode throughput lost to occupancy at M=1, not a\n\
         batching question. Both header biases apply: small shapes read warm\n\
         and stall-free, so a low reading is a LOWER bound on the gap.\n"
    );
}

/// Does baking M/N in as function constants move the GEMV at all? The
/// production dispatch always passes `use_fc = false`, so before anyone
/// specializes a dispatch site (the GDN FC 90-94 item), this asks the
/// kernel in isolation. A saturated kernel cannot be helped by hoisting
/// bounds arithmetic, so the expected answer is a null; measured rather
/// than assumed because the dispatch-site change is hours and this is
/// seconds. Distinct pipeline-cache keys per arm -- constants that miss
/// the key silently reuse one pipeline for both (crate Gotcha 1).
#[test]
#[ignore = "benchmark: needs a real Metal device, reports rather than asserts"]
fn baking_m_n_as_function_constants_is_measured_not_assumed() {
    let mut context = MetalContext::new().expect("Metal device");

    let fc_constants = |rows: u32, cols: u32| {
        let values = FunctionConstantValues::new();
        let use_fc = true;
        values.set_constant_value_at_index((&rows as *const u32).cast(), MTLDataType::UInt, 20);
        values.set_constant_value_at_index((&cols as *const u32).cast(), MTLDataType::UInt, 21);
        values.set_constant_value_at_index((&use_fc as *const bool).cast(), MTLDataType::Bool, 22);
        values
    };

    let time = |context: &mut MetalContext,
                shape: &ShapeBuffers,
                constants: &FunctionConstantValues,
                key: &[u8]|
     -> f64 {
        let seconds = autorelease_pool(|| {
            let pipeline = context
                .pipeline(INT4_SOURCE, "dequant_int4_gemv_simd", constants, key)
                .expect("pipeline");
            let pass = context.begin_pass();
            let m = (shape.rows as u32).to_ne_bytes();
            let n = (shape.cols as u32).to_ne_bytes();
            for _ in 0..shape.repeats {
                pass.encode_threadgroups(
                    &pipeline,
                    &[
                        (&shape.weights, 0, 0),
                        (&shape.weights, 1, shape.scales_offset),
                        (&shape.weights, 2, shape.biases_offset),
                        (&shape.x, 3, 0),
                        (&shape.y, 4, 0),
                    ],
                    &[(&m, 5), (&n, 6)],
                    shape.rows.div_ceil(8) as u64,
                    256,
                );
            }
            pass.commit_and_wait_with_gpu_time()
        });
        gib_per_sec(
            weight_bytes(shape.rows, shape.cols) * shape.repeats as u64,
            seconds,
        )
    };

    println!("\nshape                     no-FC GiB/s   FC(M,N) GiB/s   delta");
    for (label, rows, cols) in QWEN38_SHAPES {
        if rows > 32767 {
            continue; // head + reference: FC probe reads the per-layer shapes
        }
        let shape = ShapeBuffers::new(&context, rows, cols);
        let plain = int4_constants();
        let baked = fc_constants(rows as u32, cols as u32);
        // The key MUST carry the shape: baked M/N differ per shape, and a
        // shared key silently reuses the first-compiled pipeline for every
        // later shape -- the first run of this very test did exactly that,
        // and the wrong-N pipeline read a third of each row and printed
        // +322% "speedup" (impossible: past DRAM bandwidth). Gotcha 1.
        let key_on = format!("fc-on-{rows}x{cols}").into_bytes();
        // Warmup both arms (fault-in + DVFS, Gotcha 20), then interleave.
        time(&mut context, &shape, &plain, b"fc-off");
        time(&mut context, &shape, &baked, &key_on);
        let mut off = f64::MIN;
        let mut on = f64::MIN;
        for _ in 0..ROUNDS {
            off = off.max(time(&mut context, &shape, &plain, b"fc-off"));
            on = on.max(time(&mut context, &shape, &baked, &key_on));
        }
        println!(
            "{label}   {:>9.1}   {:>11.1}   {:>+6.1}%",
            off,
            on,
            (on / off - 1.0) * 100.0
        );
    }
    println!(
        "\nWARM arms above repeat one matrix, so anything under ~50 MiB is\n\
         served from cache and a large delta there is cache bandwidth the\n\
         dynamic-bounds loop cannot exploit -- NOT a production prediction.\n\
         Production reads every matrix cold, once per token; the COLD arms\n\
         below walk a pool past any cache, which is the decision-relevant\n\
         number for specializing dispatch sites.\n"
    );

    let time_pool = |context: &mut MetalContext,
                     pool: &MatrixPool,
                     constants: &FunctionConstantValues,
                     key: &[u8],
                     dispatches: usize|
     -> f64 {
        autorelease_pool(|| {
            let pipeline = context
                .pipeline(INT4_SOURCE, "dequant_int4_gemv_simd", constants, key)
                .expect("pipeline");
            let pass = context.begin_pass();
            let m = (pool.rows as u32).to_ne_bytes();
            let n = (pool.cols as u32).to_ne_bytes();
            for i in 0..dispatches {
                let matrix = pool.matrix(i);
                pass.encode_threadgroups(
                    &pipeline,
                    &[
                        (matrix.buffer, 0, 0),
                        (matrix.buffer, 1, pool.scales_offset),
                        (matrix.buffer, 2, pool.biases_offset),
                        (&pool.x, 3, 0),
                        (&pool.y, 4, 0),
                    ],
                    &[(&m, 5), (&n, 6)],
                    pool.rows.div_ceil(8) as u64,
                    256,
                );
            }
            pass.commit_and_wait_with_gpu_time()
        })
    };

    println!("shape                     COLD no-FC GiB/s   COLD FC GiB/s   delta");
    for (label, rows, cols) in QWEN38_SHAPES {
        if rows > 32767 {
            continue;
        }
        let pool = MatrixPool::new(&context, rows, cols);
        let dispatches = ((TARGET_BYTES / weight_bytes(rows, cols)).max(64) as usize).max(256);
        let plain = int4_constants();
        let baked = fc_constants(rows as u32, cols as u32);
        let key_on = format!("fc-on-{rows}x{cols}").into_bytes(); // per-shape, see above
        time_pool(&mut context, &pool, &plain, b"fc-off", dispatches);
        time_pool(&mut context, &pool, &baked, &key_on, dispatches);
        let mut off = f64::MIN;
        let mut on = f64::MIN;
        for _ in 0..ROUNDS {
            off = off.max(gib_per_sec(
                weight_bytes(rows, cols) * dispatches as u64,
                time_pool(&mut context, &pool, &plain, b"fc-off", dispatches),
            ));
            on = on.max(gib_per_sec(
                weight_bytes(rows, cols) * dispatches as u64,
                time_pool(&mut context, &pool, &baked, &key_on, dispatches),
            ));
        }
        println!(
            "{label}   {:>14.1}   {:>13.1}   {:>+6.1}%",
            off,
            on,
            (on / off - 1.0) * 100.0
        );
    }
    println!(
        "\nA cold delta inside the run-to-run spread means specialization\n\
         cannot pay in production and the dispatch-site FC items stay closed.\n"
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
///
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
            x_batch: context.new_output_buffer((64 * cols * 2) as u64),
            y_batch: context.new_output_buffer((64 * rows * 2) as u64),
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

    /// The same, at `row_block` output rows per SIMD group. Interchangeable
    /// with `time_gemm` in every sense that matters: the two produce
    /// bit-identical output (`dequant_int4_gemm_parity.rs`), so the ratio
    /// between them is pure throughput and needs no numerics caveat.
    fn time_gemm_blocked(
        &self,
        context: &mut MetalContext,
        groups: usize,
        batch: usize,
        row_block: usize,
    ) -> f64 {
        autorelease_pool(|| {
            let pass = context.begin_pass();
            for group in 0..groups {
                encode_dequant_int4_gemm_resident_blocked(
                    context,
                    &pass,
                    &self.matrix(group),
                    (&self.x_batch, 0),
                    (&self.y_batch, 0),
                    batch,
                    row_block,
                )
                .unwrap();
            }
            pass.commit_and_wait_with_gpu_time()
        })
    }

    /// The same, through the MATRIX-hardware kernel. Not interchangeable
    /// with `time_gemm`: that one is bit-exact against the GEMV and this
    /// one reassociates the K reduction, so the two answer different
    /// questions and the ratio between them is the price of the trade.
    fn time_gemm_mma(&self, context: &mut MetalContext, groups: usize, batch: usize) -> f64 {
        autorelease_pool(|| {
            let pass = context.begin_pass();
            for group in 0..groups {
                encode_dequant_int4_gemm_mma_resident(
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

    /// The matrix kernel with `x` staged through threadgroup memory
    /// (`FC_MMA_STAGE_X`). Bit-identical to the arm above
    /// (`dequant_int4_mma_parity.rs`), so the ratio between them is pure
    /// throughput and needs no numerics caveat.
    fn time_gemm_mma_staged(&self, context: &mut MetalContext, groups: usize, batch: usize) -> f64 {
        autorelease_pool(|| {
            let pass = context.begin_pass();
            for group in 0..groups {
                turbospark_gpu::encode_dequant_int4_gemm_mma_resident_staged(
                    context,
                    &pass,
                    &self.matrix(group),
                    (&self.x_batch, 0),
                    (&self.y_batch, 0),
                    batch,
                    true,
                )
                .unwrap();
            }
            pass.commit_and_wait_with_gpu_time()
        })
    }

    /// The matrix kernel with the DEQUANT DELETED (`FC_MMA_SKIP_DEQUANT`).
    /// Output is meaningless; the time is the matrix path's cost alone.
    fn time_gemm_mma_no_dequant(
        &self,
        context: &mut MetalContext,
        groups: usize,
        batch: usize,
    ) -> f64 {
        autorelease_pool(|| {
            let pass = context.begin_pass();
            for group in 0..groups {
                turbospark_gpu::encode_dequant_int4_gemm_mma_resident_skip_dequant(
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

/// The same `c(M)`, at the DENSE `qwen3_5` 27B's shapes rather than Qwen
/// 3.6's (`docs/MTP_SPECULATIVE.md`, stage 0). This is the term that decides
/// whether an MTP verify pass pays, and it could not be borrowed from the
/// table above: those shapes are hidden 2048, these are hidden 5120 with a
/// 17408 FFN, so every matrix here is 4 to 8 times larger.
///
/// **THE DIRECTION MATTERS AND IT IS NOT OBVIOUS.**
/// `docs/SPECULATIVE_DECODING.md` explains its own disappointing `c(M)` by
/// noting that the SEQUENTIAL arm is not bandwidth-bound either -- a cold
/// matrix reads at 66 GiB/s against the kernel's 355 GiB/s saturation -- so
/// batching divides a term that was never the cost. That argument is a
/// statement about SIZE, not about the kernel: these matrices are tens of
/// megabytes and cannot sit in any cache, so the sequential arm here should
/// be closer to bandwidth-bound and `c(M)` correspondingly better. Measured
/// rather than assumed, because the opposite reading (bigger rows spill more
/// accumulator state) is equally plausible from the armchair.
///
/// The full-vocab head is EXCLUDED and that is a memory limit rather than a
/// judgement: at 248320x5120 one INT4 matrix plus companions is ~715 MB and
/// `MatrixPool` insists on at least `max(BATCH_SIZES)` of them, i.e. 11.4 GB
/// of Metal buffers. It is ~2.8% of profiled decode compute and one dispatch
/// per token; account for it separately.
#[test]
#[ignore = "benchmark: needs a real Metal device, reports rather than asserts"]
fn c_of_m_at_qwen38_shapes() {
    let mut context = MetalContext::new().expect("Metal device");

    println!("\nshape                     c(M) per token, against M sequential passes");
    for (label, rows, cols) in QWEN38_SHAPES {
        // Skips the head (see the doc above) and keeps the reference row,
        // which is a saturation control rather than a decode shape but is
        // cheap and says whether the table was taken at one clock state.
        if rows > 32768 {
            println!("{label}    SKIPPED: pool would need 11.4 GB of Metal buffers");
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
        "\nCompose against THIS family's dispatch split, not the MoE one: the\n\
         INT4/INT2 GEMV is 93.1% of profiled decode compute here against the\n\
         MoE family's 52.7%, there is no expert-union term at all, and only\n\
         6.4% (norms, elementwise, the GDN recurrent step) cannot amortize\n\
         against the MoE family's 19%. Break-even at block M is M x c(M),\n\
         plus a draft cost of ~0.015 per proposal -- the MTP head is 1.4% of\n\
         the trunk's weight bytes, which is the whole point of using it.\n"
    );
}

/// `c(M)` for the MATRIX-hardware kernel, beside the exact one, on the
/// dense 27B shapes. This is the measurement that prices
/// `docs/MTP_SPECULATIVE.md`'s one remaining lever: `simdgroup_matrix` is
/// the only thing left that can move `c(M)`, and it costs bit-exactness
/// against a sequential decode (AGENTS.md Gotcha 27), so the question is
/// not whether it is faster but whether it is faster ENOUGH to buy that.
///
/// Both arms are timed in one process against the same matrices, so the
/// third column is a within-session ratio and is the number to read.
#[test]
#[ignore = "benchmark: needs a real Metal device, reports rather than asserts"]
fn c_of_m_matrix_against_exact_at_qwen38_shapes() {
    let mut context = MetalContext::new().expect("Metal device");

    println!("\nshape                    M    exact   matrix   matrix/exact");
    for (label, rows, cols) in QWEN38_SHAPES {
        if rows > 32768 {
            continue;
        }
        let pool = MatrixPool::new(&context, rows, cols);
        let dispatches = ((TARGET_BYTES / weight_bytes(rows, cols)).max(64) as usize).max(256);

        pool.time(&mut context, dispatches, 1);
        let sequential = pool.time(&mut context, dispatches, 1) / dispatches as f64;

        // Past MAX_BATCH_ROWS only the matrix kernel can run, which is the
        // point of going there: the SIMD kernel's cap is a register-array
        // limit and this one's accumulators are not a register array.
        for batch in [2usize, 4, 8, 16, 32, 64] {
            let groups = (dispatches / batch).max(1);
            let exact = if batch <= MAX_BATCH_ROWS {
                pool.time_gemm(&mut context, groups, batch);
                Some(pool.time_gemm(&mut context, groups, batch) / (groups * batch) as f64)
            } else {
                None
            };
            pool.time_gemm_mma(&mut context, groups, batch);
            let mma = pool.time_gemm_mma(&mut context, groups, batch) / (groups * batch) as f64;
            match exact {
                Some(e) => println!(
                    "{label}  {batch:>3}   {:>5.2}x   {:>5.2}x   {:>8.2}x",
                    e / sequential,
                    mma / sequential,
                    mma / e
                ),
                None => println!(
                    "{label}  {batch:>3}       --   {:>5.2}x         --  (past the SIMD cap)",
                    mma / sequential
                ),
            }
        }
    }
    println!(
        "\nThe third column is what matters. Below 1.00 the matrix kernel is\n\
         faster and the question is whether the margin buys giving up a\n\
         provably-lossless verify; at or above 1.00 there is nothing to buy.\n"
    );
}

/// **`c(R, B)`: THE ONE COMMAND THE AC SESSION RUNS.**
///
/// `docs/BENCHMARKS.md` decomposes this port's 4.94x prefill gap against
/// mlx-lm into 2.3x of micro-batch WIDTH and 2.2x of KERNEL quality, and
/// `FC_GEMM_R` is the first attempt at the second term: `row_block` rows per
/// SIMD group divides the per-block activation loads by R and hoists
/// `e0 + ... + e7` out of the row loop, which is 7 of ~17 inner ALU ops.
/// Output is bit-identical at every width, so this is a pure throughput
/// question and the only instrument that can answer it is a clock.
///
/// **READ THE `R=1` COLUMN FIRST.** It must reproduce
/// `dequant_int4_batch.metal`'s `count(4)` row (0.50 / 0.55 / 0.46 / 0.44 at
/// M=2/4/8/16 on gate/up). If it does not, the 2-D `acc` declaration stopped
/// collapsing and the shipped shape regressed -- which no parity case can
/// see, and which no static instrument on this device can see either
/// (`dequant_int4_gemm_parity.rs`'s
/// `pipeline_reflection_cannot_see_this_kernels_register_pressure`).
///
/// **AND READ THE `R=1` ROW ACROSS B BEFORE COSTING A WIDER `MAX_BATCH_ROWS`.**
/// That row is FLAT in the frozen table, which is what a compute-bound
/// kernel looks like: if it is still flat, widening M amortizes weight bytes
/// that were never the cost, and a cap raise buys footprint and no
/// throughput. The published "width target M=64" is derived from mlx-lm's
/// 1.32 ms compute floor, which needs FP16 matrix hardware this scalar-FP32
/// kernel does not use, so it is not this kernel's crossover.
///
/// **RUN CONDITIONS, and they are not optional here.** AC power, a quiet
/// machine, `--release`. Discard the first pass on an idle GPU (Gotcha 20:
/// cold vs warm is 53% on this machine) -- the sweep does one untimed run
/// per cell for that. Effects under a few percent cannot be measured on
/// battery at all (Gotcha 28), and the widths being compared here may well
/// differ by less than that.
#[test]
#[ignore = "benchmark: needs a real Metal device on AC, reports rather than asserts"]
fn c_of_r_and_m_for_the_batched_kernel() {
    let mut context = MetalContext::new().expect("Metal device");
    const ROW_BLOCKS: [usize; 3] = [1, 2, 4];

    println!("\nshape                     R   c(M) per token, against M sequential passes");
    for (label, rows, cols) in QWEN38_SHAPES {
        if rows > 32768 {
            println!("{label}    SKIPPED: pool would need 11.4 GB of Metal buffers");
            continue;
        }
        let pool = MatrixPool::new(&context, rows, cols);
        let dispatches = ((TARGET_BYTES / weight_bytes(rows, cols)).max(64) as usize).max(256);

        pool.time(&mut context, dispatches, 1);
        let sequential = pool.time(&mut context, dispatches, 1) / dispatches as f64;

        for row_block in ROW_BLOCKS {
            print!("{label}  {row_block:>2}  ");
            for batch in BATCH_SIZES {
                let groups = (dispatches / batch).max(1);
                // Untimed warmup per cell, then the reading. Interleaving
                // the R arms WITHIN one batch would be better still; this
                // orders them R-major because each cell compiles its own
                // pipeline and a cold compile inside a timed region is a
                // bigger error than the drift between adjacent cells.
                pool.time_gemm_blocked(&mut context, groups, batch, row_block);
                let batched = pool.time_gemm_blocked(&mut context, groups, batch, row_block)
                    / (groups * batch) as f64;
                print!("  M={batch} {:>5.2}x", batched / sequential);
            }
            println!();
        }
    }
    println!(
        "\nR=1 must reproduce the frozen count(4) row before any other column is\n\
         read. A row that is FLAT in M is compute-bound, and says the width term\n\
         is already spent on this kernel whatever mlx-lm's crossover is.\n"
    );
}

/// **DOES STAGING `x` EXPLAIN THE MATRIX KERNEL'S PLATEAU?**
///
/// `dequant_int4_mma.metal` plateaus at `c ~ 0.5` past M=16 and its header
/// attributes that to dequant work being independent of B. That attribution
/// is scoped to its tile (`kMmaTile = 8`, ONE SIMD group); MLX's
/// `qmm_t_impl` reaches 0.145 at M=32 on the same shapes
/// (`scripts/mlx_qmm_reference.py`, `docs/BENCHMARKS.md`) at `WM = WN = 2`,
/// `BM` up to 128, with BOTH operands staged.
///
/// This prices the FIRST of the three differences on its own. The un-staged
/// arm `simdgroup_load`s `x` transposed straight from device with row stride
/// N, once per `(n0, kt)` -- a strided device gather in the innermost loop.
/// The staged arm reads the same rows once per `n0` block, coalesced, and
/// transposes out of threadgroup memory.
///
/// **Both arms are BIT-IDENTICAL** (`dequant_int4_mma_parity.rs`,
/// mutation-checked), so the third column is pure throughput. They are timed
/// in ONE process against the same matrices and interleaved width by width,
/// which is what makes a few-percent difference readable at all (AGENTS.md
/// Gotchas 22 and 28). AC, quiet machine, `--release`, warmup discarded.
///
/// **READ IT AGAINST THE EXACT KERNEL, NOT ONLY AGAINST ITSELF.** Staging
/// could win handily and still leave this kernel behind
/// `dequant_int4_gemm_simd`, which is the only comparison that decides
/// whether the matrix line is worth re-opening.
#[test]
#[ignore = "benchmark: needs a real Metal device on AC, reports rather than asserts"]
fn c_of_m_matrix_staged_against_unstaged() {
    let mut context = MetalContext::new().expect("Metal device");

    println!("\nshape                    M   exact   mma   mma+stageX   staged/plain");
    for (label, rows, cols) in QWEN38_SHAPES {
        if rows > 32768 {
            continue;
        }
        let pool = MatrixPool::new(&context, rows, cols);
        let dispatches = ((TARGET_BYTES / weight_bytes(rows, cols)).max(64) as usize).max(256);

        pool.time(&mut context, dispatches, 1);
        let sequential = pool.time(&mut context, dispatches, 1) / dispatches as f64;

        for batch in [2usize, 4, 8, 16, 32, 64] {
            let groups = (dispatches / batch).max(1);
            let exact = if batch <= MAX_BATCH_ROWS {
                pool.time_gemm_blocked(
                    &mut context,
                    groups,
                    batch,
                    turbospark_gpu::best_row_block(batch),
                );
                Some(
                    pool.time_gemm_blocked(
                        &mut context,
                        groups,
                        batch,
                        turbospark_gpu::best_row_block(batch),
                    ) / (groups * batch) as f64,
                )
            } else {
                None
            };
            // Interleaved: warm each arm, then time each, so thermal drift
            // lands on both rather than on whichever ran last.
            pool.time_gemm_mma(&mut context, groups, batch);
            pool.time_gemm_mma_staged(&mut context, groups, batch);
            let plain = pool.time_gemm_mma(&mut context, groups, batch) / (groups * batch) as f64;
            let staged =
                pool.time_gemm_mma_staged(&mut context, groups, batch) / (groups * batch) as f64;
            let exact_col = match exact {
                Some(e) => format!("{:>5.2}x", e / sequential),
                None => "    --".to_string(),
            };
            println!(
                "{label}  {batch:>3}  {exact_col}  {:>5.2}x  {:>9.2}x  {:>11.2}x",
                plain / sequential,
                staged / sequential,
                staged / plain
            );
        }
    }
    println!(
        "\nLast column below 1.00 means staging helped. The question it \
         answers is\nwhether the strided device gather of `x` is what \
         plateaus this kernel;\nthe `exact` column is what says whether \
         any of it is worth wiring.\n"
    );
}

/// **WHICH HALF OF THE MATRIX KERNEL IS THE COST: the dequant, or the
/// matrix path?**
///
/// The kernel's header ends at an unidentified mechanism. Its own
/// explanation (dequant work independent of B) is refuted by arithmetic --
/// dequant per output is `N / B` here and `K / BM` in MLX, the same number
/// at the same width -- and staging `x`, the cheapest structural
/// difference, lost 3.3x to 5.9x. Total dequant work is `N * K` in both
/// engines and MLX spreads it over FEWER threads, so parallelism is not it
/// either.
///
/// This splits the cost by DELETION. `FC_MMA_SKIP_DEQUANT` fills the weight
/// tile with a constant and leaves every barrier, `simdgroup_load` and
/// `simdgroup_multiply_accumulate` exactly where they were, so the third
/// column is the matrix path with the unpack removed. **Its output is
/// meaningless and it is never dispatched outside this bench.**
///
/// Read the last column. Near 1.00 means the dequant is nearly free and the
/// matrix path is the whole cost, so a re-tile must change the matrix side
/// (threadgroup width, accumulator reuse). Near 0 means the dequant
/// dominates and the loader is what to fix. Anything in between splits it.
#[test]
#[ignore = "benchmark: needs a real Metal device on AC, reports rather than asserts"]
fn c_of_m_matrix_with_and_without_the_dequant() {
    let mut context = MetalContext::new().expect("Metal device");

    println!("\nshape                    M   exact     mma   mma-no-dequant   nodq/mma");
    for (label, rows, cols) in QWEN38_SHAPES {
        if rows > 32768 {
            continue;
        }
        let pool = MatrixPool::new(&context, rows, cols);
        let dispatches = ((TARGET_BYTES / weight_bytes(rows, cols)).max(64) as usize).max(256);

        pool.time(&mut context, dispatches, 1);
        let sequential = pool.time(&mut context, dispatches, 1) / dispatches as f64;

        for batch in [2usize, 8, 16, 32, 64] {
            let groups = (dispatches / batch).max(1);
            let exact = if batch <= MAX_BATCH_ROWS {
                let r = turbospark_gpu::best_row_block(batch);
                pool.time_gemm_blocked(&mut context, groups, batch, r);
                Some(
                    pool.time_gemm_blocked(&mut context, groups, batch, r)
                        / (groups * batch) as f64,
                )
            } else {
                None
            };
            pool.time_gemm_mma(&mut context, groups, batch);
            pool.time_gemm_mma_no_dequant(&mut context, groups, batch);
            let full = pool.time_gemm_mma(&mut context, groups, batch) / (groups * batch) as f64;
            let nodq = pool.time_gemm_mma_no_dequant(&mut context, groups, batch)
                / (groups * batch) as f64;
            let exact_col = match exact {
                Some(e) => format!("{:>5.2}x", e / sequential),
                None => "    --".to_string(),
            };
            println!(
                "{label}  {batch:>3}  {exact_col}  {:>6.2}x  {:>13.2}x  {:>9.2}",
                full / sequential,
                nodq / sequential,
                nodq / full
            );
        }
    }
    println!(
        "\nLast column: the fraction of this kernel's time that is NOT the\n\
         dequant. Near 1.00 the matrix path is the whole cost and the loader\n\
         is not worth fixing; near 0 the dequant is, and the threadgroup\n\
         width is the wrong lever.\n"
    );
}
