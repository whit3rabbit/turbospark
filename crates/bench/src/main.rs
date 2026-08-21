//! `turbospark-bench`: throughput benchmark harness, replicating the frozen
//! community benchmark protocol's structure -- three fixed prompts, a fixed
//! seed, and a discarded warmup run per prompt before the measured run.
//!
//! No real model weights exist for this port to load (see
//! `turbospark-runtime`'s and this crate's own module docs), so every
//! generation here runs through a scripted producer that always emits the
//! same fixed token, at a fixed token count, rather than real inference.
//! The numbers this prints are therefore a measurement of this port's
//! prefill+decode *loop* overhead (tokenizer, sampler, detokenizer, stop
//! matcher) -- not a Rust-vs-Swift inference throughput comparison. That
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
//! through `RealForwardRunner` -- a real GPU forward pass per token
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
//! and the peak `phys_footprint` in MiB -- the exact counter and cadence
//! the published Swift baselines were measured with -- plus the
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

mod model_mode;
mod scripted;

use std::path::PathBuf;

use foundation::runtime_config::ALLOWED_CACHE_SLOTS;
use model_mode::run_model_mode;
use runtime::{rate_control_for, PowerProfile};
use scripted::{run_real_mode, run_scripted_mode};
use tokenizer::MfTokenizer;
use turbospark_bench::protocol::PROTOCOL_EXPERT_CACHE_SLOTS;

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

    run_scripted_mode(&tok)
}
