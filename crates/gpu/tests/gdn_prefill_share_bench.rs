//! What share of a `qwen3_5` prefill micro-batch does the gated-DeltaNet
//! recurrence actually hold?
//!
//! **WHY THIS EXISTS RATHER THAN A PROFILER RUN.** `TURBOSPARK_DISPATCH_PROFILE=1`
//! is the only surface in this repo that attributes GPU time BY KERNEL NAME
//! (`PhaseCounters` has no GDN bucket at all), and it "waits on every command
//! buffer at commit" -- its own module doc -- which serializes exactly the
//! pipelining that makes prefill fast. Measured 2026-08-29 on the real
//! `qwen38-27b` install: 2,940 tokens ran 17.5 min without reaching the
//! report, 582 tokens 17.5 min, and ~150 tokens over 12 min. The cost does
//! NOT fall with prompt length, so shortening the prompt is not the fix and
//! the instrument is unusable inside a working session.
//!
//! This bench answers the same question in seconds, with no model and no
//! install, by timing the two kernels against each other at the real shapes
//! and weighting them by the real layer counts. It is a narrower question
//! than the profiler's (two kernels, not all of them) and that is the point:
//! the profiler already named `dequant_int4_gemm_simd` at 85.4% of prefill
//! and left GDN unnamed inside the residue, so what is missing is one ratio.
//!
//! **THE QUESTION IT SETTLES.** `jundot/omlx`'s `qwen35_prefill/gdn.py`
//! names the stock GDN prefill shape as re-reading k/q from device "once per
//! (Dv/4)-slice threadgroup => 32x redundant traffic (~13 GB per 16k-token
//! layer)", and this port dispatches `gdn_delta_step_prefill` at exactly
//! `(Hv, Dv/4)` threadgroups. Their fix is a `DB=32` Dv block plus
//! threadgroup staging. Whether that is worth building here depends entirely
//! on the share, and `docs/BATCHED_PREFILL.md` currently carries an
//! ARITHMETIC bound (under 5% before cache) rather than a measurement.
//!
//! **READ THE ABSOLUTE ms COLUMN, NOT ONLY THE SHARE.** A share is a ratio
//! of two kernels that both scale with the micro-batch, so it is stable, but
//! it is only a bound on what optimizing GDN could buy: the denominator here
//! is GEMM + GDN and excludes norms, conv, RoPE, attention and the gated
//! norm, so the true prefill share is SMALLER than what this prints. That
//! direction is deliberate -- an upper bound is what closes an item.
#![cfg(target_os = "macos")]

use turbospark_gpu::{
    autorelease_pool, best_row_block, encode_dequant_int4_gemm_resident_blocked,
    encode_gdn_delta_prefill, GdnShape, Int4ResidentMatrix, MetalContext,
};

/// `Qwen/Qwen3.8-27B`'s linear-attention geometry, from
/// `model_io::arch_baselines::qwen`'s `qwen_gdn_dense_27b`. Not a fixture
/// shape: the whole point is to time the kernel where it really runs, and
/// the `(Hv, Dv/4)` threadgroup count this bench is about is a function of
/// exactly these two numbers.
const HK: u32 = 16;
const HV: u32 = 48;
const DK: u32 = 128;
const DV: u32 = 128;

/// 64 layers, every 4th one full attention, so 48 gated-DeltaNet and 16
/// attention (`qwen_hybrid_layer_mask(64)`).
const GDN_LAYERS: usize = 48;
const ATTN_LAYERS: usize = 16;
const LAYERS: usize = GDN_LAYERS + ATTN_LAYERS;

/// The micro-batch width the prefill driver actually uses:
/// `MAX_PREFILL_BATCH == gpu::MAX_BATCH_ROWS`.
const BATCH: usize = 16;

/// Every INT4 matrix one micro-batch walks, and how many times. These are
/// `gemv_bandwidth_bench.rs`'s QWEN38_SHAPES with the layer counts attached;
/// the head is excluded because prefill runs it once per PROMPT rather than
/// per micro-batch (`logits_last_only`'s reason), so including it would
/// inflate the denominator and flatter the GDN share.
const MICRO_BATCH_GEMMS: [(&str, usize, usize, usize); 5] = [
    ("gate/up    17408x5120", 17408, 5120, LAYERS),
    ("down       5120x17408", 5120, 17408, LAYERS),
    ("gdn_inproj 16480x5120", 16480, 5120, GDN_LAYERS),
    ("packed_q   12288x5120", 12288, 5120, ATTN_LAYERS),
    ("o_proj      5120x6144", 5120, 6144, ATTN_LAYERS),
];

/// Three timed rounds, best taken, interleaved across the arms rather than
/// run as consecutive batches (CLAUDE.local.md's A/B rule inside one
/// process). One untimed pass per arm first: cold-vs-warm GPU on this
/// machine is 53% (AGENTS.md Gotcha 20).
const ROUNDS: usize = 3;

fn shape() -> GdnShape {
    GdnShape {
        num_k_heads: HK,
        num_v_heads: HV,
        key_head_dim: DK,
        value_head_dim: DV,
        conv_kernel_size: 4,
    }
}

/// One `gdn_delta_step_prefill` over `BATCH` tokens, repeated `reps` times
/// in ONE command buffer so the number is device time rather than commit
/// overhead. The state buffer is shared across reps deliberately: the
/// kernel reads it, updates registers and writes back once, so reusing it
/// reproduces the real dependency chain instead of a best case.
struct GdnBufs {
    conv_out: metal::Buffer,
    a: metal::Buffer,
    b: metal::Buffer,
    a_log: metal::Buffer,
    dt_bias: metal::Buffer,
    state: metal::Buffer,
    y: metal::Buffer,
}

impl GdnBufs {
    fn new(context: &MetalContext) -> Self {
        let s = shape();
        let rows = BATCH as u64;
        Self {
            conv_out: context.new_output_buffer(rows * s.qkv_dim() as u64 * 2),
            a: context.new_output_buffer(rows * HV as u64 * 2),
            b: context.new_output_buffer(rows * HV as u64 * 2),
            a_log: context.new_output_buffer(HV as u64 * 2),
            dt_bias: context.new_output_buffer(HV as u64 * 2),
            state: context.new_output_buffer(HV as u64 * DV as u64 * DK as u64 * 4),
            y: context.new_output_buffer(rows * s.value_dim() as u64 * 2),
        }
    }

    fn time(&self, context: &mut MetalContext, reps: usize) -> f64 {
        autorelease_pool(|| {
            let pass = context.begin_pass();
            for _ in 0..reps {
                encode_gdn_delta_prefill(
                    context,
                    &pass,
                    shape(),
                    (&self.conv_out, 0),
                    (&self.a, 0),
                    (&self.b, 0),
                    (&self.a_log, 0),
                    (&self.dt_bias, 0),
                    &self.state,
                    (&self.y, 0),
                    BATCH as u32,
                )
                .expect("gdn delta prefill");
            }
            pass.commit_and_wait_with_gpu_time()
        })
    }
}

/// One INT4 matrix plus its activation and output slabs, at the shipped
/// `best_row_block(BATCH)` so the GEMM arm measures what prefill runs today
/// rather than the pre-`FC_GEMM_R` shape.
struct GemmBufs {
    w: metal::Buffer,
    x: metal::Buffer,
    y: metal::Buffer,
    scales_offset: u64,
    biases_offset: u64,
    rows: usize,
    cols: usize,
}

impl GemmBufs {
    fn new(context: &MetalContext, rows: usize, cols: usize) -> Self {
        let scales_offset = (rows * cols / 2) as u64;
        let biases_offset = scales_offset + (rows * cols / 64 * 2) as u64;
        let total = biases_offset + (rows * cols / 64 * 2) as u64;
        Self {
            w: context.new_output_buffer(total),
            x: context.new_output_buffer((BATCH * cols * 2) as u64),
            y: context.new_output_buffer((BATCH * rows * 2) as u64),
            scales_offset,
            biases_offset,
            rows,
            cols,
        }
    }

    fn time(&self, context: &mut MetalContext, reps: usize) -> f64 {
        autorelease_pool(|| {
            let pass = context.begin_pass();
            for _ in 0..reps {
                let m = Int4ResidentMatrix {
                    buffer: &self.w,
                    weights_offset: 0,
                    scales_offset: self.scales_offset,
                    biases_offset: self.biases_offset,
                    rows: self.rows,
                    cols: self.cols,
                };
                encode_dequant_int4_gemm_resident_blocked(
                    context,
                    &pass,
                    &m,
                    (&self.x, 0),
                    (&self.y, 0),
                    BATCH,
                    best_row_block(BATCH),
                )
                .expect("int4 gemm");
            }
            pass.commit_and_wait_with_gpu_time()
        })
    }
}

fn best_of(context: &mut MetalContext, mut run: impl FnMut(&mut MetalContext) -> f64) -> f64 {
    run(context);
    let mut best = f64::INFINITY;
    for _ in 0..ROUNDS {
        best = best.min(run(context));
    }
    best
}

#[test]
#[ignore = "benchmark: needs a real Metal device, reports rather than asserts"]
fn gdn_delta_share_of_a_prefill_micro_batch() {
    let mut context = MetalContext::new().expect("Metal device");

    // Reps chosen so each timed region runs long enough for one clock
    // state; the GDN kernel is far cheaper per dispatch than a 17408x5120
    // GEMM, so it needs many more of them.
    const GDN_REPS: usize = 256;
    const GEMM_REPS: usize = 16;

    let gdn = GdnBufs::new(&context);
    let gdn_once = best_of(&mut context, |c| gdn.time(c, GDN_REPS)) / GDN_REPS as f64;

    println!(
        "\nqwen3_5 dense 27B, micro-batch of {BATCH} tokens, \
         row_block {}\n",
        best_row_block(BATCH)
    );
    println!("kernel                        per call ms   x layers   total ms");
    println!(
        "gdn_delta_step_prefill          {:>9.4}   {:>8}   {:>8.3}",
        gdn_once * 1e3,
        GDN_LAYERS,
        gdn_once * GDN_LAYERS as f64 * 1e3
    );

    let mut gemm_total = 0.0;
    for (label, rows, cols, count) in MICRO_BATCH_GEMMS {
        let bufs = GemmBufs::new(&context, rows, cols);
        let once = best_of(&mut context, |c| bufs.time(c, GEMM_REPS)) / GEMM_REPS as f64;
        gemm_total += once * count as f64;
        println!(
            "{label}          {:>9.4}   {:>8}   {:>8.3}",
            once * 1e3,
            count,
            once * count as f64 * 1e3
        );
    }

    let gdn_total = gdn_once * GDN_LAYERS as f64;
    let share = gdn_total / (gdn_total + gemm_total);
    println!(
        "\nGDN recurrence is {:.2}% of (GEMM + GDN) device time in one \
         micro-batch.\nThat is an UPPER BOUND on its share of prefill: the \
         denominator omits\nnorms, the conv pair, RoPE, attention and the \
         gated norm, all of which\nare real prefill cost this bench does not \
         encode.\n",
        share * 100.0
    );
    println!(
        "Under ~5% the oMLX threadgroup-staging rewrite \
         (`docs/BATCHED_PREFILL.md`,\n\"Step 7 candidates\") is closed by \
         measurement rather than by arithmetic.\n"
    );
}
