use runtime::{PowerProfile, RateControl};

/// Wall clock, for the `[power-window ...]` markers. `Instant` is
/// deliberately not used: the marker has to be comparable against a
/// timeline `powermetrics` builds in another process.
#[cfg(target_os = "macos")]
fn unix_millis() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

/// The real-install protocol run (see module docs). One shared footprint
/// sampler across warmups and measured runs: the number that matters is
/// the process peak under the whole workload, which is what the Swift
/// baselines report.
#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
pub fn run_model_mode(
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
pub fn run_model_mode(
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
