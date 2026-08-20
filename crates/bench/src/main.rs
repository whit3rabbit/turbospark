//! `turbospark-bench`: throughput benchmark harness, replicating the frozen
//! community benchmark protocol's structure — three fixed prompts, a fixed
//! seed, and a discarded warmup run per prompt before the measured run.
//!
//! No real model weights exist for this port to load (see
//! `turbospark-runtime`'s and this crate's own module docs), so every
//! generation here runs through a scripted producer that always emits the
//! same fixed token, at a fixed token count, rather than real inference.
//! The numbers this prints are therefore a measurement of this port's
//! prefill+decode *loop* overhead (tokenizer, sampler, detokenizer, stop
//! matcher) — not a Rust-vs-Swift inference throughput comparison. That
//! comparison needs a real forward pass; see `DEVIATIONS.md`.
//!
//! "Fresh processes" (the third leg of the frozen protocol) is a
//! process-level concern this binary does not orchestrate itself: run it
//! once per fresh process externally (e.g. a shell loop) if that isolation
//! matters for a given measurement.
//!
//! Usage: `turbospark-bench <tokenizer-dir> [--real]`
//!    or: `turbospark-bench --model <install-dir> [--case <id>]`.
//!
//! `--real` (macOS only): instead of the scripted producer, builds a
//! small synthetic dense `.gturbo` install (deterministic INT4 weights,
//! vocab sized to the tokenizer) and drives the same three prompts
//! through `RealForwardRunner` — a real GPU forward pass per token
//! (zero-copy resident weights, persistent KV, one command buffer per
//! token). Still not a Swift-comparison number (the synthetic model is
//! tiny), but it measures the real dispatch path, not just loop overhead.
//!
//! `--model <install-dir>` (macOS only): THE Swift-comparison mode. Opens
//! a real `.gturbo` install (tokenizer bundled in the same directory) and
//! runs the frozen community-protocol cases (real-generation-v1: frozen
//! prompts, seeds 20260721-23, temp 0.2, top-k 64, top-p 0.95), one
//! discarded warmup then one measured run per
//! case. Prints per case the split prefill/decode seconds (from
//! `RawDecodeResult`, unlike the scripted mode's wall-clock lump), tok/s,
//! and the peak `phys_footprint` in MiB — the exact counter and cadence
//! the published Swift baselines were measured with — plus the
//! Swift-format `[stop=...]` footer on stderr for `grep -h '^\[stop='`
//! parity with `docs/COMMUNITY_BENCHMARKS.md`.
//!
//! THE CONTEXT WINDOW AND THE GENERATION BUDGET ARE PER-FAMILY, resolved
//! from the opened install's own manifest by
//! `real_model::protocol_parameters` and printed in the header beside the
//! slot count. Four families run the shared 4,096/1,024; the dense `llama`
//! half needs 8,192 (its tokenizer makes `long-synthesis` 3,444 tokens) and
//! `gpt-oss` needs 8,192/3,072 (Harmony's reasoning channel). Read a peak
//! or a tok/s row WITH those two numbers -- a row taken at one window says
//! nothing about another.
//!
//! `--case <id>` restricts that mode to one protocol case, which is how
//! the protocol's fresh-process leg is run (Swift's CLI launches once per
//! case, so a cross-engine comparison has to match that). `scripts/parity.sh`
//! drives both engines this way.

use std::path::PathBuf;
use std::time::Instant;

use foundation::runtime_config::ALLOWED_CACHE_SLOTS;
use foundation::LogitValue;
use runtime::{
    rate_control_for, run_raw_completion, GenerationConfig, PowerProfile, RateControl,
    RawDecodeProgress, ScriptedLogitProducer,
};
use selection::ShapingConfig;
use tokenizer::MfTokenizer;
use turbospark_bench::protocol::PROTOCOL_EXPERT_CACHE_SLOTS;

/// Three fixed prompts and a fixed seed, matching the frozen protocol's
/// shape. Content is arbitrary (no real model is being measured); only the
/// count (three) and the fact that they are fixed across runs matters.
const FIXED_PROMPTS: [&str; 3] = [
    "Explain the difference between a mutex and a semaphore.",
    "Write a short story about a lighthouse keeper.",
    "Summarize the plot of a three-act play in one paragraph.",
];
const FIXED_SEED: u64 = 42;
const FIXED_MAX_NEW_TOKENS: u32 = 64;

/// Wall clock, for the `[power-window ...]` markers. `Instant` is
/// deliberately not used: the marker has to be comparable against a
/// timeline `powermetrics` builds in another process.
#[cfg(target_os = "macos")]
fn unix_millis() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

struct RunStats {
    tokens: usize,
    decode_seconds: f64,
}

impl RunStats {
    fn tokens_per_second(&self) -> f64 {
        if self.decode_seconds <= 0.0 {
            0.0
        } else {
            self.tokens as f64 / self.decode_seconds
        }
    }
}

fn main() -> std::process::ExitCode {
    let mut args = std::env::args().skip(1);
    let Some(first) = args.next() else {
        eprintln!(
            "usage: turbospark-bench <tokenizer-dir> [--real] | --model <install-dir> [--case <id>]"
        );
        return std::process::ExitCode::from(2);
    };
    if first == "--model" {
        const USAGE: &str = "usage: turbospark-bench --model <install-dir> [--case <id>] [--expert-cache-slots N] [--power-profile performance|balanced|efficiency] [--max-tokens-per-sec R]";
        let Some(install_dir) = args.next() else {
            eprintln!("{USAGE}");
            return std::process::ExitCode::from(2);
        };
        let mut case_filter: Option<String> = None;
        let mut slots = PROTOCOL_EXPERT_CACHE_SLOTS;
        // ROADMAP Phase P2. Deliberately NOT defaulted from Low Power Mode
        // the way the CLI and server are: a measurement tool has to be
        // explicit, or an LPM-enabled machine would silently cap the
        // `performance` arm of a power A/B and manufacture the very gain
        // the A/B exists to measure.
        let mut power_profile = PowerProfile::Performance;
        let mut max_tokens_per_sec: Option<f64> = None;
        // `None` is off; `Some(0)` is "the drafter's own default block".
        let mut speculative: Option<usize> = None;
        // `None` reads the install's index; `Some(true)` forces dflash.
        let mut drafter: Option<bool> = None;
        // A SEPARATE axis from speculation, so a spec A/B holds it fixed.
        let mut shaping = turbospark_bench::real_model::ProtocolShaping::Sampled;
        while let Some(flag) = args.next() {
            match flag.as_str() {
                "--case" => match args.next() {
                    Some(id) => case_filter = Some(id),
                    None => {
                        eprintln!("--case needs a case id");
                        return std::process::ExitCode::from(2);
                    }
                },
                // Slot count is the one runtime control a Swift comparison
                // has to be able to match; MferenceCLI takes the same set.
                "--expert-cache-slots" => match args.next().map(|v| v.parse::<u32>()) {
                    Some(Ok(n)) if ALLOWED_CACHE_SLOTS.contains(&n) => slots = n as usize,
                    _ => {
                        eprintln!("--expert-cache-slots needs one of {ALLOWED_CACHE_SLOTS:?}");
                        return std::process::ExitCode::from(2);
                    }
                },
                "--power-profile" => match args.next().as_deref().map(PowerProfile::parse) {
                    Some(Some(profile)) => power_profile = profile,
                    _ => {
                        eprintln!("--power-profile needs performance, balanced or efficiency");
                        return std::process::ExitCode::from(2);
                    }
                },
                // DEFAULTS OFF, EXPLICITLY. AGENTS.md Gotcha 35's rule, and
                // the reason is sharper here than for the power profile: a
                // drafter changes the shaping to greedy (see
                // `run_protocol_case_speculating`), so a bench that
                // speculated by default would silently retire every sampled
                // row this harness has ever produced.
                "--speculative" => match args.next().as_deref() {
                    Some("off") => speculative = None,
                    Some("auto") => speculative = Some(0),
                    Some(v) => match v.parse::<usize>() {
                        Ok(n) if n > 0 => speculative = Some(n),
                        _ => {
                            eprintln!("--speculative needs off, auto or a block above 0");
                            return std::process::ExitCode::from(2);
                        }
                    },
                    None => {
                        eprintln!("--speculative needs off, auto or a block above 0");
                        return std::process::ExitCode::from(2);
                    }
                },
                "--shaping" => match args.next().as_deref() {
                    Some("protocol") => {
                        shaping = turbospark_bench::real_model::ProtocolShaping::Sampled
                    }
                    Some("greedy") => {
                        shaping = turbospark_bench::real_model::ProtocolShaping::Greedy
                    }
                    _ => {
                        eprintln!("--shaping needs protocol or greedy");
                        return std::process::ExitCode::from(2);
                    }
                },
                "--speculative-drafter" => match args.next().as_deref() {
                    Some("auto") => drafter = None,
                    Some("mtp") => drafter = Some(false),
                    Some("dflash") => drafter = Some(true),
                    _ => {
                        eprintln!("--speculative-drafter needs auto, mtp or dflash");
                        return std::process::ExitCode::from(2);
                    }
                },
                "--max-tokens-per-sec" => match args.next().map(|v| v.parse::<f64>()) {
                    Some(Ok(r)) if r.is_finite() && r > 0.0 => max_tokens_per_sec = Some(r),
                    _ => {
                        eprintln!("--max-tokens-per-sec needs a number greater than 0");
                        return std::process::ExitCode::from(2);
                    }
                },
                other => {
                    eprintln!("unexpected argument {other:?}; {USAGE}");
                    return std::process::ExitCode::from(2);
                }
            }
        }
        let rate = rate_control_for(power_profile, max_tokens_per_sec);
        return run_model_mode(
            &install_dir,
            case_filter.as_deref(),
            slots,
            power_profile,
            rate,
            speculative,
            drafter,
            shaping,
        );
    }
    let tokenizer_dir = first;
    let real_mode = args.next().as_deref() == Some("--real");

    let tok = match MfTokenizer::load_from_dir(&PathBuf::from(tokenizer_dir)) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("failed to load tokenizer: {e}");
            return std::process::ExitCode::from(1);
        }
    };

    if real_mode {
        return run_real_mode(&tok);
    }

    println!(
        "turbospark-bench: {} fixed prompts, seed {FIXED_SEED}, scripted producer (see module docs)",
        FIXED_PROMPTS.len()
    );
    println!(
        "{:<10} {:>12} {:>14} {:>16}",
        "prompt", "tokens", "decode_secs", "tokens_per_sec"
    );

    let mut measured: Vec<RunStats> = Vec::with_capacity(FIXED_PROMPTS.len());
    for (index, prompt) in FIXED_PROMPTS.iter().enumerate() {
        // Warmup run: discarded, matching the frozen protocol.
        let _ = run_once(&tok, prompt);
        // Measured run.
        match run_once(&tok, prompt) {
            Ok(stats) => {
                println!(
                    "{:<10} {:>12} {:>14.4} {:>16.1}",
                    index,
                    stats.tokens,
                    stats.decode_seconds,
                    stats.tokens_per_second()
                );
                measured.push(stats);
            }
            Err(e) => {
                eprintln!("prompt {index} failed: {e}");
                return std::process::ExitCode::from(1);
            }
        }
    }

    let total_tokens: usize = measured.iter().map(|r| r.tokens).sum();
    let total_seconds: f64 = measured.iter().map(|r| r.decode_seconds).sum();
    let aggregate = if total_seconds > 0.0 {
        total_tokens as f64 / total_seconds
    } else {
        0.0
    };
    println!("aggregate: {total_tokens} tokens in {total_seconds:.4}s = {aggregate:.1} tokens/sec");

    std::process::ExitCode::SUCCESS
}

fn run_once(tok: &MfTokenizer, prompt: &str) -> Result<RunStats, String> {
    let vocab_size = tok.vocab_size;
    let h_id = tok.token_to_id("h").unwrap_or(0) as usize;
    let prompt_ids = tok.encode(prompt, true);

    let steps: Vec<Vec<LogitValue>> = (0..prompt_ids.len() + FIXED_MAX_NEW_TOKENS as usize)
        .map(|_| one_hot(vocab_size, h_id))
        .collect();
    let mut producer = ScriptedLogitProducer::new(steps);
    let config = GenerationConfig {
        shaping: ShapingConfig::new(0.0, 0, None, 1.0, Some(FIXED_SEED))
            .map_err(|e| e.to_string())?,
        max_new_tokens: FIXED_MAX_NEW_TOKENS,
        stop_strings: Vec::new(),
        extra_stop_tokens: Vec::new(),
        rate: Default::default(),
    };

    let start = Instant::now();
    let mut tokens = 0usize;
    let result = run_raw_completion(
        &mut producer,
        tok,
        &prompt_ids,
        &config,
        8192,
        vocab_size,
        |e| {
            if let RawDecodeProgress::Token { .. } = e {
                tokens += 1;
            }
        },
    )
    .map_err(|e| e.to_string())?;
    let decode_seconds = start.elapsed().as_secs_f64();

    Ok(RunStats {
        tokens: result.new_tokens.max(tokens),
        decode_seconds,
    })
}

#[cfg(target_os = "macos")]
fn run_real_mode(tok: &MfTokenizer) -> std::process::ExitCode {
    let dir = std::env::temp_dir().join(format!("turbospark-bench-real-{}", std::process::id()));
    if let Err(e) = std::fs::create_dir_all(&dir) {
        eprintln!("failed to create temp install dir: {e}");
        return std::process::ExitCode::from(1);
    }
    let arch = match repack::build_synthetic_gemma4_install(&dir, tok.vocab_size as i64, 2, "bench")
    {
        Ok(a) => a,
        Err(e) => {
            eprintln!("failed to build synthetic install: {e}");
            return std::process::ExitCode::from(1);
        }
    };
    let mut runner = match runtime::RealForwardRunner::open(&dir, arch) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("failed to open RealForwardRunner: {e}");
            return std::process::ExitCode::from(1);
        }
    };

    println!(
        "turbospark-bench: {} fixed prompts, seed {FIXED_SEED}, REAL forward pass \
         (synthetic tiny dense model; measures the real GPU dispatch path, \
         not production throughput)",
        FIXED_PROMPTS.len()
    );
    println!(
        "{:<10} {:>12} {:>14} {:>16}",
        "prompt", "tokens", "decode_secs", "tokens_per_sec"
    );

    let mut measured: Vec<RunStats> = Vec::with_capacity(FIXED_PROMPTS.len());
    for (index, prompt) in FIXED_PROMPTS.iter().enumerate() {
        let _ = run_once_real(&mut runner, tok, prompt);
        match run_once_real(&mut runner, tok, prompt) {
            Ok(stats) => {
                println!(
                    "{:<10} {:>12} {:>14.4} {:>16.1}",
                    index,
                    stats.tokens,
                    stats.decode_seconds,
                    stats.tokens_per_second()
                );
                measured.push(stats);
            }
            Err(e) => {
                eprintln!("prompt {index} failed: {e}");
                return std::process::ExitCode::from(1);
            }
        }
    }

    let total_tokens: usize = measured.iter().map(|r| r.tokens).sum();
    let total_seconds: f64 = measured.iter().map(|r| r.decode_seconds).sum();
    let aggregate = if total_seconds > 0.0 {
        total_tokens as f64 / total_seconds
    } else {
        0.0
    };
    println!("aggregate: {total_tokens} tokens in {total_seconds:.4}s = {aggregate:.1} tokens/sec");
    let _ = std::fs::remove_dir_all(&dir);
    std::process::ExitCode::SUCCESS
}

#[cfg(target_os = "macos")]
fn run_once_real(
    runner: &mut runtime::RealForwardRunner,
    tok: &MfTokenizer,
    prompt: &str,
) -> Result<RunStats, String> {
    let vocab_size = tok.vocab_size;
    let prompt_ids = tok.encode(prompt, true);
    let config = GenerationConfig {
        shaping: ShapingConfig::new(0.0, 0, None, 1.0, Some(FIXED_SEED))
            .map_err(|e| e.to_string())?,
        max_new_tokens: FIXED_MAX_NEW_TOKENS,
        stop_strings: Vec::new(),
        extra_stop_tokens: Vec::new(),
        rate: Default::default(),
    };

    let start = Instant::now();
    let mut tokens = 0usize;
    let result = run_raw_completion(runner, tok, &prompt_ids, &config, 4096, vocab_size, |e| {
        if let RawDecodeProgress::Token { .. } = e {
            tokens += 1;
        }
    })
    .map_err(|e| e.to_string())?;
    let decode_seconds = start.elapsed().as_secs_f64();

    Ok(RunStats {
        tokens: result.new_tokens.max(tokens),
        decode_seconds,
    })
}

#[cfg(not(target_os = "macos"))]
fn run_real_mode(_tok: &MfTokenizer) -> std::process::ExitCode {
    eprintln!("--real requires macOS (Metal)");
    std::process::ExitCode::from(2)
}

/// The real-install protocol run (see module docs). One shared footprint
/// sampler across warmups and measured runs: the number that matters is
/// the process peak under the whole workload, which is what the Swift
/// baselines report.
#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn run_model_mode(
    install_dir: &str,
    case_filter: Option<&str>,
    slots: usize,
    profile: PowerProfile,
    rate: RateControl,
    speculative: Option<usize>,
    drafter: Option<bool>,
    shaping: turbospark_bench::real_model::ProtocolShaping,
) -> std::process::ExitCode {
    use turbospark_bench::memory::AppMemorySampler;
    use turbospark_bench::protocol::{swift_footer, PROTOCOL_CASES};
    use turbospark_bench::real_model::{
        open_model_runner_for_protocol_speculative, run_protocol_case_speculating,
    };

    // `--case` runs exactly one case in this process, which is the frozen
    // protocol's fresh-process leg (Swift launches its CLI once per case).
    // The default stays all three in one process: the memory oracle wants
    // the whole session's peak on one runner.
    let cases: Vec<_> = match case_filter {
        None => PROTOCOL_CASES.iter().collect(),
        Some(id) => {
            let selected: Vec<_> = PROTOCOL_CASES.iter().filter(|c| c.id == id).collect();
            if selected.is_empty() {
                eprintln!("unknown case {id:?}; valid ids:");
                for case in &PROTOCOL_CASES {
                    eprintln!("  {}", case.id);
                }
                return std::process::ExitCode::from(2);
            }
            selected
        }
    };

    // The protocol's window and budget are PER-FAMILY (see
    // `real_model::protocol_parameters`), and they are resolved from the
    // install's own manifest rather than taken from the shared constants:
    // `gpt-oss` stops two of three cases on maxTokens at 1,024, and the dense
    // `llama` half cannot fit `long-synthesis` in 4,096 at all.
    // REFUSED BEFORE THE OPEN, not at the first case. The open is ~20 s on
    // a real install and the answer does not depend on it, so checking late
    // spends that for a message it already had -- `scripts/power.sh`
    // validates ARMS before `sudo` for the same reason.
    if speculative.is_some() && shaping != turbospark_bench::real_model::ProtocolShaping::Greedy {
        eprintln!(
            "--speculative needs --shaping greedy: acceptance is exact only at temperature 0, \
             and the frozen protocol samples (temperature 0.2, top-k 64, top-p 0.95)"
        );
        return std::process::ExitCode::from(2);
    }

    // Which drafter, resolved the way the CLI resolves it: an explicit
    // `--speculative-drafter` is a promise, and `auto` reads the install's
    // own resident index rather than guessing. Both are decided BEFORE the
    // open, because the policies name exactly one drafter and opening two is
    // the bug that verified a 9-row block against a 3-row scratch.
    let use_dflash = drafter.unwrap_or_else(|| {
        model_io::load_resident_index(&std::path::Path::new(install_dir).join("model_weights.bin"))
            .map(|ix| !runtime::install_has_mtp_head(&ix) && runtime::install_has_dflash(&ix))
            .unwrap_or(false)
    });
    let policies = match (speculative, use_dflash) {
        (None, _) => runtime::DraftPolicies::off(),
        (Some(0), true) => runtime::DraftPolicies {
            mtp: runtime::MtpDraftPolicy::Off,
            dflash: runtime::DflashDraftPolicy::Auto,
        },
        (Some(0), false) => runtime::DraftPolicies::mtp(runtime::MtpDraftPolicy::Auto),
        (Some(n), true) => runtime::DraftPolicies {
            mtp: runtime::MtpDraftPolicy::Off,
            dflash: runtime::DflashDraftPolicy::Fixed(n),
        },
        (Some(n), false) => runtime::DraftPolicies::mtp(runtime::MtpDraftPolicy::Fixed(n)),
    };
    let (mut runner, tok, params) = match open_model_runner_for_protocol_speculative(
        std::path::Path::new(install_dir),
        slots,
        policies,
    ) {
        Ok(triple) => triple,
        Err(e) => {
            eprintln!("failed to open {install_dir}: {e}");
            return std::process::ExitCode::from(1);
        }
    };
    if let Some(brand) = turbospark_bench::memory::chip_brand_string() {
        println!("turbospark-bench: real install {install_dir} on {brand}, frozen protocol real-generation-v1");
    } else {
        println!(
            "turbospark-bench: real install {install_dir}, frozen protocol real-generation-v1"
        );
    }
    // Printed beside the numbers because a peak or a tok/s row measured at
    // one window and budget says nothing about another (crate Gotchas 11
    // and 12). The oracles print theirs beside the ceiling for the same
    // reason.
    println!(
        "  family={} context={} max_new={} expert_cache_slots={}",
        params.family.as_str(),
        params.max_context,
        params.max_new,
        slots
    );
    // The RESOLVED power pair, for the same reason and one worse: an arm of
    // a `scripts/power.sh` A/B is named entirely outside this process, so
    // without this line a `--power-profile efficiency` run and an uncapped
    // one differ only in the tok/s column -- a label with no tell in the
    // artifact it labels. That is how `COOLING=max` was once passed to a
    // script that ignored it and reported success. BOTH values are printed
    // because neither implies the other: an explicit `--max-tokens-per-sec`
    // overrides the profile's own cap without changing whether the thermal
    // ladder runs, so `performance` at 15 tok/s and `efficiency` at 15
    // tok/s are different runs that agree on every other column.
    // The RESOLVED speculation, for exactly the reason the power pair below
    // is printed: an arm of a `scripts/power.sh` A/B is named outside this
    // process, and a `spec` arm that silently ran non-speculative would
    // differ from `nospec` only in the tok/s column. `auto` also carries no
    // number, and the two drafters' defaults differ (2 against 8).
    //
    // THE SHAPING IS ON THIS LINE and not implied, because a speculative run
    // is GREEDY where the frozen protocol samples: its joules-per-token is
    // not comparable to any published row, and the line that says so has to
    // be in the artifact rather than in a doc.
    let resolved_block = speculative.map(|n| {
        if n > 0 {
            n
        } else if use_dflash {
            runtime::DFLASH_SERVING_BLOCK
        } else {
            runtime::DEFAULT_SPECULATION_BLOCK
        }
    });
    let shaping_name = match shaping {
        turbospark_bench::real_model::ProtocolShaping::Sampled => "protocol-sampled",
        // Flagged in the artifact, not just in a doc: a greedy row's
        // joules-per-token is not comparable to any published row, all of
        // which are sampled.
        turbospark_bench::real_model::ProtocolShaping::Greedy => {
            "GREEDY (not the frozen protocol; not comparable to docs/POWER_BASELINE.md)"
        }
    };
    match resolved_block {
        None => println!("  speculative=off shaping={shaping_name}"),
        Some(block) => println!(
            "  speculative=on drafter={} block={block} shaping={shaping_name}",
            if use_dflash { "dflash2" } else { "mtp" },
        ),
    }
    println!(
        "  power_profile={} max_tok_s={} thermal_stepping={}",
        profile.as_str(),
        rate.max_tokens_per_sec
            .map_or_else(|| "-".to_string(), |r| format!("{r}")),
        rate.thermal_probe.is_some()
    );
    println!(
        "{:<18} {:>10} {:>10} {:>8} {:>9} {:>8} {:>9}",
        "case", "prompt_tok", "prefill_s", "new_tok", "decode_s", "tok_s", "peak_mib"
    );

    let mut sampler = AppMemorySampler::new();
    for case in cases {
        // Discarded warmup, then the measured run (frozen protocol).
        if let Err(e) = run_protocol_case_speculating(
            &mut runner,
            &tok,
            case,
            &mut sampler,
            rate,
            params.max_context,
            params.max_new,
            shaping,
            resolved_block,
        ) {
            eprintln!("{} warmup failed: {e}", case.id);
            return std::process::ExitCode::from(1);
        }
        // Wall-clock bounds of the MEASURED run, for `scripts/power.sh` to
        // window a `powermetrics` capture with. Everything outside them is
        // power this process burned but the protocol does not measure: the
        // 13 GB mmap and Metal pipeline compilation at open, and the
        // discarded warmup, which is itself a full 1024-token generation.
        // Inferring the window from process start or exit instead would
        // fold those in. The prefill/decode split INSIDE the window needs
        // no further markers: the footer below already carries both.
        eprintln!(
            "[power-window case={} phase=start unix_ms={}]",
            case.id,
            unix_millis()
        );
        let measured = run_protocol_case_speculating(
            &mut runner,
            &tok,
            case,
            &mut sampler,
            rate,
            params.max_context,
            params.max_new,
            shaping,
            resolved_block,
        );
        eprintln!(
            "[power-window case={} phase=end unix_ms={}]",
            case.id,
            unix_millis()
        );
        match measured {
            Ok(r) => {
                let peak_mib = r
                    .peak_footprint_bytes
                    .map_or(f64::NAN, |b| b as f64 / 1_048_576.0);
                println!(
                    "{:<18} {:>10} {:>10.2} {:>8} {:>9.2} {:>8.3} {:>9.1}",
                    r.case_id,
                    r.prompt_tokens,
                    r.prefill_seconds,
                    r.new_tokens,
                    r.decode_seconds,
                    r.tokens_per_second(),
                    peak_mib
                );
                eprintln!(
                    "{}",
                    swift_footer(
                        r.reason,
                        r.prompt_tokens,
                        r.prefill_seconds,
                        r.new_tokens,
                        r.decode_seconds
                    )
                );
            }
            Err(e) => {
                eprintln!("{} failed: {e}", case.id);
                return std::process::ExitCode::from(1);
            }
        }
    }
    if let Some(peak) = sampler.peak_bytes() {
        println!(
            "session peak phys_footprint: {:.1} MiB",
            peak as f64 / 1_048_576.0
        );
    }
    std::process::ExitCode::SUCCESS
}

#[cfg(not(target_os = "macos"))]
#[allow(clippy::too_many_arguments)]
fn run_model_mode(
    _install_dir: &str,
    _case_filter: Option<&str>,
    _slots: usize,
    _profile: PowerProfile,
    _rate: RateControl,
    _speculative: Option<usize>,
    _drafter: Option<bool>,
    _shaping: turbospark_bench::real_model::ProtocolShaping,
) -> std::process::ExitCode {
    eprintln!("--model requires macOS (Metal)");
    std::process::ExitCode::from(2)
}

fn one_hot(vocab_size: usize, index: usize) -> Vec<LogitValue> {
    let mut v = vec![LogitValue::from_f32(0.0); vocab_size];
    if index < vocab_size {
        v[index] = LogitValue::from_f32(1.0);
    }
    v
}
