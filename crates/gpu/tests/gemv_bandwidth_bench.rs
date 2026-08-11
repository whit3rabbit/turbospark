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

use turbospark_gpu::{
    autorelease_pool, encode_dequant_int4_gemv_resident, Int4ResidentMatrix, MetalContext,
};

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
