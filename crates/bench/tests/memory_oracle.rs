#![cfg(target_os = "macos")]
//! Memory oracle for the real Gemma 4 install: peak `phys_footprint` across
//! the frozen benchmark protocol's three cases, next to a static accounting
//! of what the design says should be resident. The point is attribution --
//! whatever the three cases spend that the accounting does not explain is
//! the number worth chasing.
//!
//! `phys_footprint` (not RSS) is the comparable metric: it is what
//! `/usr/bin/time -l` reports as "peak memory footprint" and what
//! Mference's `docs/BENCHMARKS.md` publishes.
//!
//! Measured fact worth knowing before reading the numbers: the resident
//! weight mapping DOES count. A plain read-only `mmap` would not (clean
//! file-backed pages are excluded), but wrapping it in an `MTLBuffer` via
//! `newBufferWithBytesNoCopy` makes Metal pin the range, and pinned pages
//! land in the footprint. `after open` minus the KV allocation minus the
//! process baseline comes out at the mapping's size, every run. So the
//! accounting below counts it.
//!
//! Ignored by default: needs the real 13 GiB install. Run with
//!
//! ```text
//! MREFRUST_GEMMA4_INSTALL_DIR=~/models/gemma4.gturbo \
//!   cargo test -p mrefrust-bench --test memory_oracle --release -- --ignored --nocapture
//! ```

use std::path::PathBuf;

use foundation::LogitValue;
use runtime::{arch_from_manifest_dir, LogitProducer, RealForwardRunner};
use tokenizer::MfTokenizer;

const MIB: f64 = 1024.0 * 1024.0;
/// `RealForwardRunner`'s hardcoded per-layer expert slot count.
const EXPERT_CACHE_SLOTS: u64 = 16;
/// The SWA ring depth `KvCacheManager` allocates per sliding-window layer:
/// `sliding_window + max_prefill_chunk_tokens`, and the runner prefills one
/// token at a time.
const SWA_PREFILL_CHUNK_ROWS: u64 = 1;
const MAX_CONTEXT: u64 = 4096;
const NEW_TOKENS: usize = 64;
/// Filler-word counts for the three protocol cases: no context, a few
/// hundred prompt tokens, and enough to cross a thousand.
const CASE_WORDS: [usize; 3] = [0, 300, 1200];
/// Allowance for the steady-state guard. Covers malloc jitter around the
/// per-token host `Vec`s (a few MiB at V=262144), nothing more. Observed
/// spread on a healthy run is 0.00-0.52 MiB per replay; the autorelease
/// leak this guard was written to catch was 15.8 MiB per replay.
const STEADY_STATE_SLACK: u64 = 4 * 1024 * 1024;
/// How many replays the steady-state guard gives the expert slot cache to
/// stop dirtying new pages before it calls the growth a leak.
const STEADY_STATE_ROUNDS: usize = 3;

/// Current and lifetime-peak `phys_footprint`, in bytes.
fn footprint() -> (u64, u64) {
    let mut info = std::mem::MaybeUninit::<libc::rusage_info_v4>::zeroed();
    // SAFETY: `proc_pid_rusage` fills a caller-owned `rusage_info_v4` when
    // asked for RUSAGE_INFO_V4; the buffer is sized by its own type.
    let rc = unsafe {
        libc::proc_pid_rusage(
            std::process::id() as libc::c_int,
            libc::RUSAGE_INFO_V4,
            &mut info as *mut _ as *mut libc::rusage_info_t,
        )
    };
    assert_eq!(rc, 0, "proc_pid_rusage failed");
    // SAFETY: a successful call initialized the struct.
    let info = unsafe { info.assume_init() };
    (info.ri_phys_footprint, info.ri_lifetime_max_phys_footprint)
}

fn mib(bytes: u64) -> f64 {
    bytes as f64 / MIB
}

fn install_dir() -> Option<PathBuf> {
    let raw = std::env::var("MREFRUST_GEMMA4_INSTALL_DIR").ok()?;
    let expanded = if let Some(rest) = raw.strip_prefix("~/") {
        PathBuf::from(std::env::var("HOME").expect("HOME")).join(rest)
    } else {
        PathBuf::from(raw)
    };
    Some(expanded)
}

/// A prompt of roughly `words` filler words inside the Gemma 4 turn
/// markers, so each case exercises a different prefill length.
fn case_prompt(words: usize) -> String {
    const FILLER: [&str; 8] = [
        "coastal", "wetland", "marsh", "mangrove", "sediment", "surge", "tide", "estuary",
    ];
    let body: String = (0..words)
        .map(|i| FILLER[i % FILLER.len()])
        .collect::<Vec<_>>()
        .join(" ");
    format!(
        "<|turn>user\nExplain how coastal wetlands reduce flood damage. \
         Context: {body}<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"
    )
}

struct CaseReport {
    prompt_tokens: usize,
    after_prefill: u64,
    after_decode: u64,
    peak: u64,
}

fn run_case(runner: &mut RealForwardRunner, tokenizer: &MfTokenizer, words: usize) -> CaseReport {
    let prompt_ids = tokenizer.encode(&case_prompt(words), true);
    let vocab = runner.arch().vocab_size as usize;
    let mut logits = vec![LogitValue::from_f32(0.0); vocab];

    runner.reset();
    let mut position = 0usize;
    for &token in &prompt_ids {
        runner
            .produce(token, position, &mut logits)
            .expect("prefill");
        position += 1;
    }
    let after_prefill = footprint().0;

    // Greedy continuation: the decode hot path, without a sampler in the
    // way. Token choice does not matter here, only that experts get routed.
    let mut token = argmax(&logits);
    for _ in 0..NEW_TOKENS {
        runner
            .produce(token, position, &mut logits)
            .expect("decode");
        position += 1;
        token = argmax(&logits);
    }
    let (after_decode, peak) = footprint();

    CaseReport {
        prompt_tokens: prompt_ids.len(),
        after_prefill,
        after_decode,
        peak,
    }
}

fn argmax(logits: &[LogitValue]) -> i32 {
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.to_f32().total_cmp(&b.1.to_f32()))
        .map(|(i, _)| i as i32)
        .unwrap()
}

#[test]
#[ignore = "needs the real Gemma 4 install; set MREFRUST_GEMMA4_INSTALL_DIR"]
fn peak_footprint_is_explained_by_static_accounting() {
    let Some(dir) = install_dir() else {
        panic!("set MREFRUST_GEMMA4_INSTALL_DIR to a real .gturbo install");
    };
    let arch = arch_from_manifest_dir(&dir).expect("manifest resolves an arch");
    let tokenizer = MfTokenizer::load_from_dir(&dir).expect("tokenizer loads");

    let baseline = footprint().0;
    let mut runner = RealForwardRunner::open(&dir, arch.clone()).expect("install opens");
    let after_open = footprint().0;

    // Static accounting, from the shapes alone.
    let layers = arch.num_layers as u64;
    let expert_stride = runner.expert_stride().expect("streamed experts");
    let slot_capacity = EXPERT_CACHE_SLOTS * layers * expert_stride;
    let swa_layers = arch
        .full_attention_layer_mask
        .iter()
        .filter(|&&m| m == 0)
        .count() as u64;
    let full_layers = layers - swa_layers;
    let swa_row = 2 * (arch.num_kv_heads * arch.head_dim) as u64 * 2;
    let full_row = 2 * (arch.num_full_kv_heads * arch.full_head_dim) as u64 * 2;
    let swa_rows = MAX_CONTEXT.min(arch.sliding_window as u64 + SWA_PREFILL_CHUNK_ROWS);
    let kv_bytes = swa_layers * swa_rows * swa_row + full_layers * MAX_CONTEXT * full_row;
    let resident_bytes = runner.resident_bytes();

    println!("\n=== static accounting ===");
    println!("  expert slot capacity  {:>9.0} MiB  ({EXPERT_CACHE_SLOTS} slots x {layers} layers x {expert_stride} B)", mib(slot_capacity));
    println!("  KV cache              {:>9.0} MiB  ({swa_layers} SWA x {swa_rows} rows + {full_layers} full x {MAX_CONTEXT} rows)", mib(kv_bytes));
    println!(
        "  resident weights      {:>9.0} MiB  (mmap'd, pinned by the MTLBuffer wrap)",
        mib(resident_bytes)
    );
    let dirty_accounted = slot_capacity + kv_bytes + resident_bytes;
    println!("  accounted             {:>9.0} MiB", mib(dirty_accounted));

    println!("\n=== footprint ===");
    println!("  baseline              {:>9.0} MiB", mib(baseline));
    println!("  after open            {:>9.0} MiB", mib(after_open));

    let mut peak = after_open;
    let mut longest_after_decode = 0u64;
    for (index, words) in CASE_WORDS.into_iter().enumerate() {
        let report = run_case(&mut runner, &tokenizer, words);
        peak = peak.max(report.peak);
        longest_after_decode = report.after_decode;
        println!(
            "  case {index} ({:>4} prompt tokens): after prefill {:>7.0} MiB, \
             after decode {:>7.0} MiB, lifetime peak {:>7.0} MiB",
            report.prompt_tokens,
            mib(report.after_prefill),
            mib(report.after_decode),
            mib(report.peak),
        );
    }

    // Steady state. Replaying ONE case must converge to zero growth: its
    // KV was sized at open, and its experts land in the same slots every
    // time once the cache has stopped shuffling them. Growth is allowed
    // for a round or two -- the varied case lengths above left the slot
    // cache holding other layers' experts, and re-routing dirties pages
    // that were never touched -- but it must stop. Anything that
    // accumulates per token or per prompt token never converges, which
    // the absolute ceiling below would hide under the unused slot
    // capacity until the leak exceeded ~500 MiB.
    let mut previous = longest_after_decode;
    let mut steady_growth = u64::MAX;
    let mut rounds = 0usize;
    while rounds < STEADY_STATE_ROUNDS && steady_growth > STEADY_STATE_SLACK {
        let repeat = run_case(&mut runner, &tokenizer, CASE_WORDS[0]);
        peak = peak.max(repeat.peak);
        steady_growth = repeat.after_decode.saturating_sub(previous);
        previous = repeat.after_decode;
        rounds += 1;
        println!(
            "  replay {rounds} of case 0:                 after decode {:>7.0} MiB, \
             growth {:>7.2} MiB, gpu buffers {}",
            mib(repeat.after_decode),
            mib(steady_growth),
            runner.gpu_buffer_allocations(),
        );
    }

    let ceiling = dirty_accounted + baseline;
    println!(
        "\n  peak {:.0} MiB, ceiling {:.0} MiB (accounted {:.0} + baseline {:.0}), \
         headroom {:.0} MiB",
        mib(peak),
        mib(ceiling),
        mib(dirty_accounted),
        mib(baseline),
        mib(ceiling.saturating_sub(peak)),
    );

    // Guard 1, absolute: every dirty byte at peak must be one the static
    // accounting names. No slack -- the three terms are exact allocations,
    // and slot capacity is already an upper bound (only routed experts
    // ever dirty their pages), so a healthy run lands well under this.
    // Catches an allocation the design does not describe at all.
    assert!(
        peak <= ceiling,
        "peak footprint {:.0} MiB exceeds the accounted ceiling {:.0} MiB \
         by {:.0} MiB -- something is resident that the static accounting \
         does not describe",
        mib(peak),
        mib(ceiling),
        mib(peak.saturating_sub(ceiling)),
    );

    // Guard 2, steady state: replaying a case whose allocations are all
    // already warm must be free. The slack covers allocator jitter only
    // (the host sampler churns a few MiB of `Vec`s per token, which
    // malloc may or may not have returned to the OS when we sample);
    // anything that accumulates is orders of magnitude above it.
    assert!(
        steady_growth <= STEADY_STATE_SLACK,
        "replaying the same case {STEADY_STATE_ROUNDS} times never stopped \
         growing the footprint (last round +{:.2} MiB, slack {:.0} MiB) -- \
         something accumulates per token or per prompt token",
        mib(steady_growth),
        mib(STEADY_STATE_SLACK),
    );
}
