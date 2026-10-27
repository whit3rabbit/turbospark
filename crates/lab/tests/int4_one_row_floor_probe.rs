//! Why does one 48-row INT4 GEMV read about three times its share of an
//! 8-GEMV pass in the `int4_matmul` bench? (`docs/KAPITSA_KERNELS.md`,
//! "Measured: the kernel-modes tooling".)
//!
//! The bench times a pass of `copies * sweeps` calls and divides the command
//! buffer's GPU interval by the call count. At M = 1 a call is one GEMV
//! dispatch, at M = 8 it is eight, so the one-row pass holds 64 dispatches
//! where the 8-row pass holds 512. Earlier ledgers (`km-retro`, `km-retro2`)
//! already said the per-dispatch time does not depend on M: an 8-GEMV pass
//! of 64 dispatches read the same 13.5 us per GEMV as the one-row pass, and
//! only the 512-dispatch pass read 3.9 us. Output ordering (a separate y per
//! copy) and cache warmth (one weight copy) were ruled out before this probe.
//!
//! What it found (M4 Max, AC power, 2026-10-25, other sessions building and
//! measuring beside it): none of the pass-structure arms moves the number
//! while the device is warm, and the bench's own `run_shape` called three
//! times in one process read 13.6, 3.8 and 3.0 us for the one row. From a
//! cold device (3 s idle) a 64-dispatch pass reads 18.5 us per dispatch, a
//! 16384-dispatch pass stays slow for its whole 280 ms, primers of 1024 and
//! 4096 dispatches queued ahead of it do not help and 16384 (about 50 ms)
//! brings it to 3.2 us. The device's speed is set by its state when a pass
//! starts, and only tens of milliseconds of continuous work ahead of the
//! pass raise it. M = 1 was the bench's first measurement, with host work
//! between passes, so it read that state; the fix is `support::Primer`.
//!
//! The first test's arms, which separated the hypotheses:
//!
//! - `dispatches`: the one-row GEMV with 8 to 2048 dispatches in the timed
//!   pass, nothing else changed.
//! - `offsets`: 64 dispatches that rotate through eight x and y offsets, the
//!   8-row path's layout one row at a time.
//! - `primed`: the same 64-dispatch pass with a separate command buffer of
//!   1024 dispatches committed right before it and not timed.
//! - `x8`: the 8-row path as the bench runs it (512 dispatches).
//! - `gap`: the 64-dispatch pass after the host sleeps 1 ms or 5 ms.
//! - `after scalar`, `after mma8`: the 64-dispatch pass right after a
//!   waited-on pass of the other arm the bench interleaves it with.
//! - `gap, primed`: the 5 ms gap, then the primer, then the timed pass.
//!
//! The second test interleaves the bench's `run_shape` with those arms in
//! one process; the third watches a fresh process and the return from idle;
//! the fourth is the cold-device primer and pass-length comparison that
//! decided the fix.
//!
//! One measuring session per machine: the probe takes the lab's machine lock.
//!
//! ```sh
//! cargo test -p turbospark-lab --release --test int4_one_row_floor_probe -- --ignored --nocapture
//! ```
#![cfg(target_os = "macos")]

use gpu::{
    autorelease_pool, encode_dequant_int4_gemm_resident, encode_dequant_int4_gemv_resident,
    encode_dequant_int4_mma8_resident, Int4ResidentMatrix, MetalContext,
};
use turbospark_lab::kernel::support::{f16_activations, int4_blob};
use turbospark_lab::stats::median;

const SHAPES: [(usize, usize); 2] = [(48, 5120), (1024, 5120)];
const COPIES: usize = 8;
const ROUNDS: usize = 5;
/// Dispatches in a primer buffer: enough GPU work to be well past any ramp.
const PRIMER_DISPATCHES: usize = 1024;

struct Mats {
    bufs: Vec<(gpu::MetalBuffer, u64, u64)>,
    x: gpu::MetalBuffer,
    y: gpu::MetalBuffer,
    rows: usize,
    cols: usize,
}

impl Mats {
    fn new(context: &MetalContext, rows: usize, cols: usize) -> Self {
        let bufs = (0..COPIES)
            .map(|i| {
                let (data, scales, biases) = int4_blob(rows, cols, 0xC0FF_EE00 + i as u64);
                (context.new_buffer_with_data(&data), scales, biases)
            })
            .collect();
        Self {
            bufs,
            x: context.new_buffer_with_data(&f16_activations(8 * cols, 7)),
            y: context.new_output_buffer((8 * rows * 2) as u64),
            rows,
            cols,
        }
    }

    fn matrix(&self, i: usize) -> Int4ResidentMatrix<'_> {
        let (buffer, scales, biases) = &self.bufs[i % COPIES];
        Int4ResidentMatrix {
            buffer,
            weights_offset: 0,
            scales_offset: *scales,
            biases_offset: *biases,
            rows: self.rows,
            cols: self.cols,
        }
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Arm {
    /// One-row GEMVs, `n` dispatches per pass, x and y at offset 0.
    Dispatches(usize),
    /// 64 one-row GEMVs rotating through eight row offsets.
    Offsets,
    /// 64 one-row GEMVs after an untimed primer buffer.
    Primed,
    /// The bench's 8-row path: 64 calls of eight GEMVs (512 dispatches).
    X8,
    /// 64 one-row GEMVs after the host idles for this many milliseconds.
    Gap(u64),
    /// 64 one-row GEMVs after a waited-on 64-call scalar GEMM pass at M = 1.
    AfterScalar,
    /// 64 one-row GEMVs after a waited-on 64-call MMA8 pass at M = 1.
    AfterMma8,
    /// A 5 ms gap, then the primer, then 64 one-row GEMVs.
    GapPrimed,
}

impl Arm {
    fn label(self) -> String {
        match self {
            Arm::Dispatches(n) => format!("one row, {n} dispatches"),
            Arm::Offsets => "one row, 64 dispatches, 8 offsets".into(),
            Arm::Primed => "one row, 64 dispatches, primed".into(),
            Arm::X8 => "8 rows, 64 calls (512 dispatches)".into(),
            Arm::Gap(ms) => format!("one row, 64 dispatches, {ms} ms gap"),
            Arm::AfterScalar => "one row, 64 dispatches, after scalar pass".into(),
            Arm::AfterMma8 => "one row, 64 dispatches, after mma8 pass".into(),
            Arm::GapPrimed => "one row, 64 dispatches, 5 ms gap, primed".into(),
        }
    }

    /// GEMV dispatches the timed pass holds.
    fn dispatches(self) -> usize {
        match self {
            Arm::Dispatches(n) => n,
            Arm::Offsets | Arm::Primed | Arm::Gap(_) | Arm::AfterScalar | Arm::AfterMma8 => 64,
            Arm::GapPrimed => 64,
            Arm::X8 => 512,
        }
    }
}

fn encode_row(
    context: &mut MetalContext,
    pass: &gpu::PassEncoder,
    mats: &Mats,
    copy: usize,
    row: usize,
) {
    let w = mats.matrix(copy);
    encode_dequant_int4_gemv_resident(
        context,
        pass,
        &w,
        (&mats.x, (row * mats.cols * 2) as u64),
        (&mats.y, (row * mats.rows * 2) as u64),
    )
    .expect("gemv encodes");
}

/// Microseconds of GPU time per GEMV dispatch in the timed pass.
fn time_arm(context: &mut MetalContext, mats: &Mats, arm: Arm) -> f64 {
    autorelease_pool(|| {
        match arm {
            Arm::Gap(ms) => std::thread::sleep(std::time::Duration::from_millis(ms)),
            Arm::GapPrimed => std::thread::sleep(std::time::Duration::from_millis(5)),
            Arm::AfterScalar | Arm::AfterMma8 => {
                let p = context.begin_pass();
                for i in 0..64 {
                    let w = mats.matrix(i);
                    if arm == Arm::AfterScalar {
                        encode_dequant_int4_gemm_resident(
                            context,
                            &p,
                            &w,
                            (&mats.x, 0),
                            (&mats.y, 0),
                            1,
                        )
                        .expect("gemm encodes");
                    } else {
                        encode_dequant_int4_mma8_resident(
                            context,
                            &p,
                            &w,
                            (&mats.x, 0),
                            (&mats.y, 0),
                            1,
                        )
                        .expect("mma8 encodes");
                    }
                }
                p.commit_and_wait();
            }
            _ => {}
        }
        let primer = matches!(arm, Arm::Primed | Arm::GapPrimed).then(|| {
            let p = context.begin_pass();
            for i in 0..PRIMER_DISPATCHES {
                encode_row(context, &p, mats, i, i % 8);
            }
            p.commit()
        });
        let pass = context.begin_pass();
        match arm {
            Arm::Dispatches(n) => {
                for i in 0..n {
                    encode_row(context, &pass, mats, i, 0);
                }
            }
            Arm::Offsets => {
                for i in 0..64 {
                    encode_row(context, &pass, mats, i, i % 8);
                }
            }
            Arm::Primed | Arm::Gap(_) | Arm::AfterScalar | Arm::AfterMma8 | Arm::GapPrimed => {
                for i in 0..64 {
                    encode_row(context, &pass, mats, i, 0);
                }
            }
            Arm::X8 => {
                for call in 0..64 {
                    for r in 0..8 {
                        encode_row(context, &pass, mats, call, r);
                    }
                }
            }
        }
        let seconds = pass.commit_and_wait_with_gpu_time();
        drop(primer);
        seconds * 1e6 / arm.dispatches() as f64
    })
}

#[test]
#[ignore = "benchmark: needs a real Metal device, reports rather than asserts"]
fn one_row_gemv_floor_by_pass_length_offsets_and_primer() {
    let lab = turbospark_lab::paths::lab_dir();
    let _lock = turbospark_lab::lock::MachineLock::acquire(&lab, "int4_one_row_floor_probe")
        .expect("machine lock");
    let mut context = MetalContext::new().expect("Metal device");
    let arms = [
        Arm::Dispatches(8),
        Arm::Dispatches(64),
        Arm::Dispatches(128),
        Arm::Dispatches(256),
        Arm::Dispatches(512),
        Arm::Dispatches(1024),
        Arm::Dispatches(2048),
        Arm::Offsets,
        Arm::Primed,
        Arm::X8,
        Arm::Gap(1),
        Arm::Gap(5),
        Arm::AfterScalar,
        Arm::AfterMma8,
        Arm::GapPrimed,
    ];
    println!(
        "\nus of GPU time per GEMV dispatch, median of {ROUNDS} rounds, first round discarded"
    );
    for (rows, cols) in SHAPES {
        let mats = Mats::new(&context, rows, cols);
        for arm in arms {
            time_arm(&mut context, &mats, arm);
        }
        let mut samples = vec![Vec::new(); arms.len()];
        for round in 0..ROUNDS {
            for k in 0..arms.len() {
                let a = (k + round) % arms.len();
                samples[a].push(time_arm(&mut context, &mats, arms[a]));
            }
        }
        println!("\n{rows}x{cols}");
        for (a, arm) in arms.iter().enumerate() {
            let us = median(&samples[a]);
            println!(
                "  {:<40} {us:>7.2} us   pass {:>8.0} us",
                arm.label(),
                us * arm.dispatches() as f64
            );
        }
    }
}

/// The bench's own `run_shape` on the 48-row shape, interleaved with this
/// probe's one-row and 8-row arms in one process, so a difference between
/// the two cannot be the host's state at the time.
#[test]
#[ignore = "benchmark: needs a real Metal device, reports rather than asserts"]
fn bench_run_shape_against_the_probe_arms_in_one_process() {
    use turbospark_lab::kernel::{int4_matmul, BenchOpts};
    let lab = turbospark_lab::paths::lab_dir();
    let _lock = turbospark_lab::lock::MachineLock::acquire(&lab, "int4_one_row_floor_probe")
        .expect("machine lock");
    let mut context = MetalContext::new().expect("Metal device");
    let (rows, cols) = (48usize, 5120usize);
    let mats = Mats::new(&context, rows, cols);
    let opts = BenchOpts {
        rows: vec![1, 8],
        contexts: vec![],
        reps: 5,
        hash: false,
        verify_from: None,
        ane_programs: None,
    };
    println!("\nbench run_shape against probe arms, 48x5120, three alternations");
    for _ in 0..3 {
        let r = int4_matmul::run_shape(&mut context, "48x5120", rows, cols, &opts).unwrap();
        let gemv = &r.arms[0];
        let m8 = gemv.points.iter().find(|p| p.x == 8.0).unwrap();
        println!(
            "  bench run_shape: one row {:>6.2} us, 8-row pass {:>6.2} us per GEMV",
            gemv.points[0].ms * 1e3,
            m8.ms * 1e3 / 8.0
        );
        for arm in [Arm::Dispatches(64), Arm::X8] {
            let mut v = Vec::new();
            for _ in 0..=ROUNDS {
                v.push(time_arm(&mut context, &mats, arm));
            }
            println!("  probe {:<36} {:>6.2} us", arm.label(), median(&v[1..]));
        }
    }
}

/// In a fresh process: one-row passes back to back from the first GPU use,
/// printed against elapsed time, then again after a few seconds of GPU
/// idle. Says how long the slow window lasts and whether idle brings it back.
#[test]
#[ignore = "benchmark: needs a real Metal device, reports rather than asserts"]
fn fresh_process_decay_curve_and_return_after_idle() {
    let lab = turbospark_lab::paths::lab_dir();
    let _lock = turbospark_lab::lock::MachineLock::acquire(&lab, "int4_one_row_floor_probe")
        .expect("machine lock");
    let mut context = MetalContext::new().expect("Metal device");
    let mats = Mats::new(&context, 48, 5120);
    let report = |context: &mut MetalContext, label: &str| {
        let start = std::time::Instant::now();
        let mut v = Vec::new();
        for _ in 0..600 {
            let t = start.elapsed().as_secs_f64();
            v.push((t, time_arm(context, &mats, Arm::Dispatches(64))));
        }
        println!("\n{label}: us per one-row GEMV dispatch over the first passes");
        let mut next_print = 0.0;
        for (t, us) in &v {
            if *t >= next_print {
                println!("  t = {t:>6.3} s  {us:>6.2} us");
                next_print += if *t < 0.5 { 0.05 } else { 0.25 };
            }
        }
        let (last_t, last) = v[v.len() - 1];
        println!("  t = {last_t:>6.3} s  {last:>6.2} us (last)");
    };
    report(&mut context, "fresh process");
    std::thread::sleep(std::time::Duration::from_secs(3));
    report(&mut context, "after 3 s idle");
    std::thread::sleep(std::time::Duration::from_secs(10));
    report(&mut context, "after 10 s idle");
}

/// From a cold device (3 s idle before every sample): the 64-dispatch pass
/// alone, primed by 1024 or 4096 untimed dispatches committed right before
/// it, and long passes alone. Says what brings a tiny pass to the speed a
/// busy decode step has.
#[test]
#[ignore = "benchmark: needs a real Metal device, reports rather than asserts"]
fn cold_device_primer_and_pass_length() {
    let lab = turbospark_lab::paths::lab_dir();
    let _lock = turbospark_lab::lock::MachineLock::acquire(&lab, "int4_one_row_floor_probe")
        .expect("machine lock");
    let mut context = MetalContext::new().expect("Metal device");
    let mats = Mats::new(&context, 48, 5120);
    // (label, primer dispatches, timed dispatches)
    let arms: [(&str, usize, usize); 6] = [
        ("64 alone", 0, 64),
        ("64 primed by 1024", 1024, 64),
        ("64 primed by 4096", 4096, 64),
        ("64 primed by 16384", 16384, 64),
        ("4096 alone", 0, 4096),
        ("16384 alone", 0, 16384),
    ];
    let mut samples = vec![Vec::new(); arms.len()];
    for round in 0..3 {
        for k in 0..arms.len() {
            let a = (k + round) % arms.len();
            let (_, primer_n, timed_n) = arms[a];
            std::thread::sleep(std::time::Duration::from_secs(3));
            let us = autorelease_pool(|| {
                let primer = (primer_n > 0).then(|| {
                    let p = context.begin_pass();
                    for i in 0..primer_n {
                        encode_row(&mut context, &p, &mats, i, i % 8);
                    }
                    p.commit()
                });
                let pass = context.begin_pass();
                for i in 0..timed_n {
                    encode_row(&mut context, &pass, &mats, i, 0);
                }
                let s = pass.commit_and_wait_with_gpu_time();
                drop(primer);
                s * 1e6 / timed_n as f64
            });
            samples[a].push(us);
        }
    }
    println!("\ncold device (3 s idle before each sample), 48x5120, us per one-row GEMV dispatch, 3 rounds");
    for (a, (label, _, _)) in arms.iter().enumerate() {
        println!(
            "  {:<22} median {:>6.2} us   samples {:?}",
            label,
            median(&samples[a]),
            samples[a]
                .iter()
                .map(|v| (v * 100.0).round() / 100.0)
                .collect::<Vec<_>>()
        );
    }
}
