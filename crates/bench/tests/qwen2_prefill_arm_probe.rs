#![cfg(target_os = "macos")]
//! The quality gate and paired timing of the llama-flow matrix prefill arm
//! (Steel INT4 GEMMs and the multi-row GQA partial at a 64-row micro-batch,
//! `families/llama/prefill_batched.rs`) on the REAL Qwen2.5-7B MLX INT4
//! install.
//!
//! The reference is the per-token loop. For prompts of 64 to 3000 tokens, fed
//! in 128-token chunks as the CLI and server do, it compares the next-token
//! distribution at the prompt's end and then at 32 teacher-forced decode
//! steps on the reference's own greedy ids. The decode steps are the same
//! code in every arm, so what they measure is the K/V cache the prefill wrote.
//!
//! Bars, fixed before the first run: mean KL(reference || arm) under 1e-3
//! nats (the Qwen3.8 Steel probe's bound) and top-1 agreement of at least
//! 99%, each over the prompt-end positions and the decode positions
//! separately. The exact arm must equal the reference bit for bit.
//!
//! ```sh
//! TURBOSPARK_QWEN2_DENSE_INSTALL_DIR=~/.turbospark/models/text/qwen25-7b-4bit.gturbo \
//!   cargo test -p turbospark-bench --release --test qwen2_prefill_arm_probe -- --ignored --nocapture
//! ```

#[allow(dead_code)]
mod qwen2_dense_common;

use foundation::LogitValue;
use runtime::{ChunkedPrefillRunner, LogitProducer, RealForwardRunner};
use turbospark_bench::logit_compare;
use turbospark_bench::real_model::open_model_runner_with_context;

const CHUNK: usize = 128;
const DECODE_STEPS: usize = 32;
/// (prompt length, how many different corpus slices of it).
const CASES: [(usize, usize); 6] = [(64, 3), (200, 3), (500, 3), (1000, 3), (2000, 1), (3000, 1)];
/// Peak footprint ceilings per arm, MiB, on the M4 Max at 8192 context. The
/// batched path has its own row because the memory oracle never takes it:
/// 662.7 (exact) and 666.2 (matrix) measured 2026-11-26, with the family's
/// usual ~6% headroom. Accepted as the price of the matrix arm's speed; the
/// oracle's per-token ceiling is unchanged.
const PER_TOKEN_CEILING_MIB: f64 = 660.0;
const BATCHED_CEILING_MIB: f64 = 706.0;
const MEAN_KL_BOUND: f64 = 1e-3;
const TOP1_FLOOR: f64 = 0.99;

#[derive(Clone, Copy)]
enum Arm {
    PerToken,
    Exact,
    Matrix,
    MatrixRowByRow,
}

fn select(runner: &mut RealForwardRunner, arm: Arm) {
    runner.set_batched_gemv_prefill(!matches!(arm, Arm::PerToken));
    runner.set_prefill_steel(matches!(arm, Arm::Matrix | Arm::MatrixRowByRow));
    runner.set_prefill_rows_attention(matches!(arm, Arm::Matrix));
}

fn prefill(runner: &mut RealForwardRunner, ids: &[i32], vocab: usize) -> Vec<LogitValue> {
    runner.reset();
    let mut logits = vec![LogitValue::from_f32(0.0); vocab];
    for (i, span) in ids.chunks(CHUNK).enumerate() {
        runner
            .prefill_chunk(span, i * CHUNK, &mut logits)
            .expect("chunked prefill");
    }
    logits
}

fn wide(logits: &[LogitValue]) -> Vec<f64> {
    logits.iter().map(|v| v.to_f32() as f64).collect()
}

fn greedy(logits: &[LogitValue]) -> i32 {
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.to_f32().total_cmp(&b.1.to_f32()))
        .expect("vocabulary")
        .0 as i32
}

/// The prompt-end logits, then one per decode step. `forced` replays those
/// token ids; without it the arm chooses greedily and returns what it chose.
fn run(
    runner: &mut RealForwardRunner,
    prompt: &[i32],
    vocab: usize,
    forced: Option<&[i32]>,
) -> (Vec<Vec<LogitValue>>, Vec<i32>) {
    let mut logits = prefill(runner, prompt, vocab);
    let mut outputs = vec![logits.clone()];
    let mut ids = Vec::new();
    for step in 0..DECODE_STEPS {
        let id = forced.map_or_else(|| greedy(&logits), |f| f[step]);
        ids.push(id);
        runner
            .produce(id, prompt.len() + step, &mut logits)
            .expect("decode");
        outputs.push(logits.clone());
    }
    (outputs, ids)
}

#[derive(Default)]
struct Tally {
    kl: Vec<f64>,
    agree: usize,
}

impl Tally {
    fn add(&mut self, reference: &[LogitValue], arm: &[LogitValue]) {
        let c = logit_compare::compare(&wide(reference), &wide(arm));
        self.kl.push(c.kl);
        self.agree += usize::from(c.top1_agree);
    }

    fn report(&self, label: &str) -> (f64, f64) {
        let mut sorted = self.kl.clone();
        sorted.sort_by(f64::total_cmp);
        let n = sorted.len();
        let mean = sorted.iter().sum::<f64>() / n as f64;
        let agreement = self.agree as f64 / n as f64;
        println!(
            "{label}: {n} positions, mean KL {mean:.3e}, p99 {:.3e}, max {:.3e}, top-1 {}/{n}",
            sorted[(n * 99 / 100).min(n - 1)],
            sorted[n - 1],
            self.agree
        );
        (mean, agreement)
    }
}

#[test]
#[ignore = "needs the pinned Qwen2.5 7B MLX INT4 install and a release build"]
fn qwen2_matrix_prefill_quality_and_timing() {
    for var in [
        "TURBOSPARK_BATCHED_GEMV",
        "TURBOSPARK_PREFILL_STEEL",
        "TURBOSPARK_PREFILL_ROWS_ATTN",
        "TURBOSPARK_ATTN_DECODE_KERNEL",
    ] {
        assert!(
            std::env::var_os(var).is_none(),
            "{var} is set; this probe chooses the arms itself"
        );
    }
    let dir = qwen2_dense_common::install_dir();
    let (mut runner, tokenizer) = open_model_runner_with_context(&dir, 8, 4096).expect("open");
    let vocab = runner.vocab_size();
    let docs = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../docs");
    let mut files: Vec<_> = std::fs::read_dir(&docs)
        .expect("docs dir")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "md"))
        .collect();
    files.sort();
    let text: String = files
        .iter()
        .take(12)
        .map(|f| std::fs::read_to_string(f).unwrap_or_default() + "\n\n")
        .collect();
    let ids = tokenizer.encode(&text, false);

    let prompts: Vec<Vec<i32>> = CASES
        .iter()
        .flat_map(|&(n, slices)| (0..slices).map(move |s| (n, s)))
        .enumerate()
        .map(|(i, (n, _))| {
            assert!(ids.len() > n, "corpus too short");
            let start = (i * 1999) % (ids.len() - n);
            ids[start..start + n].to_vec()
        })
        .collect();

    let arms = [
        ("matrix", Arm::Matrix),
        ("matrix, one-row attention", Arm::MatrixRowByRow),
    ];
    let mut tallies: Vec<(Tally, Tally)> = arms.iter().map(|_| Default::default()).collect();
    for (case, prompt) in prompts.iter().enumerate() {
        select(&mut runner, Arm::PerToken);
        let (reference, forced) = run(&mut runner, prompt, vocab, None);
        if case % 4 == 0 {
            select(&mut runner, Arm::Exact);
            let (exact, _) = run(&mut runner, prompt, vocab, Some(&forced));
            assert!(exact == reference, "case {case}: the exact arm moved a bit");
        }
        for ((_, arm), (ends, steps)) in arms.iter().zip(&mut tallies) {
            select(&mut runner, *arm);
            let (out, _) = run(&mut runner, prompt, vocab, Some(&forced));
            ends.add(&reference[0], &out[0]);
            for (r, o) in reference[1..].iter().zip(&out[1..]) {
                steps.add(r, o);
            }
        }
        println!("case {case}: {} tokens done", prompt.len());
    }
    let mut results = Vec::new();
    for ((label, _), (ends, steps)) in arms.iter().zip(&tallies) {
        results.push((format!("{label}, prompt end"), ends.report(label)));
        results.push((
            format!("{label}, decode"),
            steps.report(&format!("{label}, decode")),
        ));
    }

    for (label, (mean, agreement)) in results {
        assert!(mean < MEAN_KL_BOUND, "{label}: mean KL {mean}");
        assert!(agreement >= TOP1_FLOOR, "{label}: top-1 {agreement}");
    }
}

/// Paired prefill time of the four arms. Separate from the quality gate so a
/// loaded host does not cost a second 15-minute quality run.
#[test]
#[ignore = "needs the pinned Qwen2.5 7B MLX INT4 install and a release build"]
fn qwen2_prefill_arm_timing() {
    let dir = qwen2_dense_common::install_dir();
    let (mut runner, _) = open_model_runner_with_context(&dir, 8, 4096).expect("open");
    let vocab = runner.vocab_size();
    // Ids only need to be in range: nothing here reads what they say.
    let ids: Vec<i32> = (0..1117u32)
        .map(|i| (100 + (i * 7919) % 20_000) as i32)
        .collect();
    // Round 0 is warmup; arms interleave.
    let timed = &ids[..];
    for round in 0..4 {
        let mut line = format!("round {round}:");
        let mut base = 0.0;
        for (label, arm) in [
            ("per-token", Arm::PerToken),
            ("exact", Arm::Exact),
            ("matrix", Arm::Matrix),
            ("matrix one-row attn", Arm::MatrixRowByRow),
        ] {
            select(&mut runner, arm);
            let started = std::time::Instant::now();
            prefill(&mut runner, timed, vocab);
            let rate = timed.len() as f64 / started.elapsed().as_secs_f64();
            if base == 0.0 {
                base = rate;
            }
            line += &format!(" {label} {rate:.1} tok/s ({:.2}x)", rate / base);
        }
        println!("{line}");
    }
}

/// Peak footprint of one arm, sampled as the memory oracle samples it, at
/// the oracle's 8192 context. One arm per process
/// (`TURBOSPARK_PROBE_ARM=per-token|exact|matrix`), so no arm inherits
/// another's allocations. The memory oracle itself prefills per token and so
/// never allocates a batched scratch.
#[test]
#[ignore = "needs the pinned Qwen2.5 7B MLX INT4 install and a release build"]
fn qwen2_prefill_arm_footprint() {
    let name = std::env::var("TURBOSPARK_PROBE_ARM").expect("TURBOSPARK_PROBE_ARM");
    let arm = match name.as_str() {
        "per-token" => Arm::PerToken,
        "exact" => Arm::Exact,
        "matrix" => Arm::Matrix,
        other => panic!("unknown arm {other}"),
    };
    let dir = qwen2_dense_common::install_dir();
    let (mut runner, _) = open_model_runner_with_context(&dir, 8, 8192).expect("open");
    let vocab = runner.vocab_size();
    let ids: Vec<i32> = (0..2859u32)
        .map(|i| (100 + (i * 7919) % 20_000) as i32)
        .collect();
    select(&mut runner, arm);
    let mut sampler = turbospark_bench::memory::AppMemorySampler::new();
    sampler.sample();
    for _ in 0..2 {
        let mut logits = prefill(&mut runner, &ids, vocab);
        sampler.sample();
        for step in 0..32 {
            let id = greedy(&logits);
            runner
                .produce(id, ids.len() + step, &mut logits)
                .expect("decode");
            sampler.sample();
        }
    }
    let peak = sampler.peak_bytes().expect("footprint");
    let mib = peak as f64 / (1024.0 * 1024.0);
    let ceiling = match arm {
        Arm::PerToken => PER_TOKEN_CEILING_MIB,
        _ => BATCHED_CEILING_MIB,
    };
    println!("{name}: peak {mib:.1} MiB, ceiling {ceiling} MiB");
    // The ceilings are one chip's reading; elsewhere this only reports.
    if is_m4_max() {
        assert!(mib <= ceiling, "{name}: {mib:.1} MiB over {ceiling} MiB");
    }
}

fn is_m4_max() -> bool {
    std::process::Command::new("sysctl")
        .args(["-n", "machdep.cpu.brand_string"])
        .output()
        .is_ok_and(|o| String::from_utf8_lossy(&o.stdout).contains("Apple M4 Max"))
}
