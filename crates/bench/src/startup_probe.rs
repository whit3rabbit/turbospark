//! Startup is measured without the frozen throughput protocol's discarded generation.

use std::fs::File;
use std::io::Read;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use runtime::{GenerationConfig, RawDecodeProgress, RealForwardRunner};
use selection::ShapingConfig;
use serde_json::{json, Value};
use tokenizer::{Message, MfTokenizer, Role};

use crate::memory::{chip_brand_string, task_fault_counters, AppMemorySampler};
use crate::protocol::{swift_reason_name, PROTOCOL_CASES};

const USAGE: &str = "Usage: turbospark-startup-probe --model DIR [options]
  --case short-explanation|medium-review|long-synthesis (default short-explanation)
  --context N              KV window (default 4096)
  --max-new N              Generation budget (default 64, truncated runs allowed)
  --slots N                Fixed expert slots (default 16)
  --residency streamed|mapped (default streamed)
  --shaping greedy|sampled (default greedy, frozen seed)
  --repeats N              Requests per runner (default 2, no prefix reuse)
  --reopens N              Runner opens per process (default 1)
  --expert-trace FILE      JSON array of [layer, expert] pairs for bounded prefetch
  --prefetch-bytes N       Whole-expert byte ceiling, requires --expert-trace
  --page-cache-label TEXT  Caller-supplied OS cache condition (default unknown)
  --label TEXT             Experiment arm label (default unlabeled)

JSONL reports separate load, first token, compiler work, cache hits, faults,
prefetch cost, footprint, and token digests. No warmup is discarded.
TURBOSPARK_METAL_PRECOMPILED=1 enables embedded Metal IR libraries.
TURBOSPARK_METAL_PIPELINE_CACHE=1 enables persistent GPU binaries.
TURBOSPARK_METAL_CACHE_DIR overrides their cache directory.
Page-cache labels are claims, not eviction or residency measurements.";

#[derive(Debug)]
struct Options {
    model: PathBuf,
    case_index: usize,
    context: u32,
    max_new: u32,
    slots: usize,
    mapped: bool,
    sampled: bool,
    repeats: usize,
    reopens: usize,
    expert_trace: Option<PathBuf>,
    prefetch_bytes: u64,
    page_cache_label: String,
    label: String,
}

impl Options {
    fn parse(args: impl IntoIterator<Item = String>) -> Result<Option<Self>, String> {
        let mut options = Self {
            model: PathBuf::new(),
            case_index: 0,
            context: 4096,
            max_new: 64,
            slots: 16,
            mapped: false,
            sampled: false,
            repeats: 2,
            reopens: 1,
            expert_trace: None,
            prefetch_bytes: 0,
            page_cache_label: "unknown".into(),
            label: "unlabeled".into(),
        };
        let mut args = args.into_iter();
        while let Some(flag) = args.next() {
            if flag == "--help" || flag == "-h" {
                return Ok(None);
            }
            let value = args.next().ok_or_else(|| format!("{flag} needs a value"))?;
            let number = || {
                value
                    .parse::<u64>()
                    .map_err(|_| format!("{flag} needs a nonnegative integer"))
            };
            let positive_u32 = || {
                u32::try_from(number()?)
                    .ok()
                    .filter(|n| *n > 0)
                    .ok_or_else(|| format!("{flag} needs a positive 32-bit integer"))
            };
            match flag.as_str() {
                "--model" => options.model = PathBuf::from(value),
                "--case" => {
                    options.case_index = PROTOCOL_CASES
                        .iter()
                        .position(|case| case.id == value)
                        .ok_or_else(|| format!("unknown case {value:?}"))?;
                }
                "--context" => options.context = positive_u32()?,
                "--max-new" => options.max_new = positive_u32()?,
                "--slots" => {
                    let n = positive_u32()?;
                    if !foundation::runtime_config::ALLOWED_CACHE_SLOTS.contains(&n) {
                        return Err(format!("unsupported expert slot count {n}"));
                    }
                    options.slots = n as usize;
                }
                "--repeats" | "--reopens" => {
                    let n = positive_u32()? as usize;
                    if n > 100 {
                        return Err(format!("{flag} is limited to 100"));
                    }
                    if flag == "--repeats" {
                        options.repeats = n;
                    } else {
                        options.reopens = n;
                    }
                }
                "--residency" => match value.as_str() {
                    "streamed" => options.mapped = false,
                    "mapped" => options.mapped = true,
                    _ => return Err("--residency must be streamed or mapped".into()),
                },
                "--shaping" => match value.as_str() {
                    "greedy" => options.sampled = false,
                    "sampled" => options.sampled = true,
                    _ => return Err("--shaping must be greedy or sampled".into()),
                },
                "--expert-trace" => options.expert_trace = Some(PathBuf::from(value)),
                "--prefetch-bytes" => options.prefetch_bytes = number()?,
                "--page-cache-label" => options.page_cache_label = value,
                "--label" => options.label = value,
                _ => return Err(format!("unknown option {flag}")),
            }
        }
        if options.model.as_os_str().is_empty() {
            return Err("--model is required".into());
        }
        if options.max_new >= options.context {
            return Err("--max-new must leave room for a prompt in --context".into());
        }
        if options.expert_trace.is_some() != (options.prefetch_bytes > 0) {
            return Err(
                "--expert-trace and positive --prefetch-bytes must be supplied together".into(),
            );
        }
        Ok(Some(options))
    }
}

fn read_trace(path: &PathBuf) -> Result<Vec<(usize, usize)>, String> {
    let mut bytes = Vec::new();
    File::open(path)
        .and_then(|file| file.take(1024 * 1024 + 1).read_to_end(&mut bytes))
        .map_err(|e| format!("expert trace {}: {e}", path.display()))?;
    if bytes.len() > 1024 * 1024 {
        return Err("expert trace exceeds 1 MiB".into());
    }
    serde_json::from_slice(&bytes)
        .map_err(|e| format!("expert trace needs [layer, expert] pairs: {e}"))
}

fn compilation_json(stats: runtime::MetalCompilationStats) -> Value {
    json!({
        "cache_enabled": stats.cache_enabled,
        "library_loads": stats.library_loads,
        "library_load_ms": stats.library_load_ms,
        "library_compiles": stats.library_compiles,
        "library_compile_ms": stats.library_compile_ms,
        "function_specializations": stats.function_specializations,
        "function_specialization_ms": stats.function_specialization_ms,
        "pipeline_creations": stats.pipeline_creations,
        "pipeline_create_ms": stats.pipeline_create_ms,
        "in_memory_pipeline_hits": stats.in_memory_pipeline_hits,
        "in_memory_function_hits": stats.in_memory_function_hits,
        "in_memory_library_hits": stats.in_memory_library_hits,
        "archive_hits": stats.archive_hits,
        "archive_misses": stats.archive_misses,
        "archive_errors": stats.archive_errors,
        "archive_load_ms": stats.archive_load_ms,
        "archive_collect_ms": stats.archive_collect_ms,
        "archive_writes": stats.archive_writes,
        "archive_write_ms": stats.archive_write_ms,
        "buffer_allocations": stats.buffer_allocations,
        "buffer_allocation_ms": stats.buffer_allocation_ms,
    })
}

fn fault_delta(before: Option<(u64, u64)>, after: Option<(u64, u64)>) -> Value {
    match (before, after) {
        (Some((a, b)), Some((c, d))) => json!({
            "process_faults": c.saturating_sub(a),
            "process_pageins": d.saturating_sub(b),
        }),
        _ => Value::Null,
    }
}

fn run(options: Options) -> Result<(), String> {
    let case = &PROTOCOL_CASES[options.case_index];
    let trace = options.expert_trace.as_ref().map(read_trace).transpose()?;
    let manifest_sha256 = model_io::hash_file(&options.model.join("manifest.json"), 1024 * 1024)
        .map_err(|e| e.to_string())?;
    for open_index in 0..options.reopens {
        let load_start = Instant::now();
        let load_faults = task_fault_counters();
        let metadata_start = Instant::now();
        let arch = repack::peek_manifest_arch(&options.model)?;
        let family = arch.family.as_str();
        let metadata_ms = metadata_start.elapsed().as_secs_f64() * 1000.0;
        let tokenizer_start = Instant::now();
        let tokenizer = MfTokenizer::load_from_dir(&options.model).map_err(|e| e.to_string())?;
        let tokenizer_ms = tokenizer_start.elapsed().as_secs_f64() * 1000.0;
        let mut runner = RealForwardRunner::open_with_residency(
            &options.model,
            arch,
            options.context as usize,
            runtime::ExpertCacheSlots::Fixed(options.slots),
            runtime::DraftPolicies::off(),
            runtime::SteeringPolicy::off(),
            1,
            runtime::KvQuant::Off,
            if options.mapped {
                runtime::ExpertResidency::Mapped
            } else {
                runtime::ExpertResidency::Streamed
            },
        )
        .map_err(|e| e.to_string())?;
        runner.set_prefix_reuse(false);
        let load_ms = load_start.elapsed().as_secs_f64() * 1000.0;
        let startup = runner.startup_stats();
        let compilation_at_open = compilation_json(runner.metal_compilation_stats());
        let open_faults = fault_delta(load_faults, task_fault_counters());
        let prefetch = trace
            .as_ref()
            .map(|selections| runner.prefetch_experts(selections, options.prefetch_bytes))
            .transpose()
            .map_err(|e| e.to_string())?;
        let rendered = tokenizer
            .apply_chat_template(&[Message::new(Role::User, case.content)])
            .map_err(|e| e.to_string())?;
        let prompt_ids = tokenizer.encode(&rendered, false);
        let shaping = if options.sampled {
            ShapingConfig::new(0.2, 64, Some(0.95), 1.0, Some(case.seed))
        } else {
            ShapingConfig::new(0.0, 1, None, 1.0, Some(case.seed))
        }
        .map_err(|e| e.to_string())?;
        let config = GenerationConfig {
            shaping,
            max_new_tokens: options.max_new,
            stop_strings: Vec::new(),
            extra_stop_tokens: Vec::new(),
            rate: runtime::RateControl::default(),
        };
        let vocab_size = runner.vocab_size();
        let mut memory = AppMemorySampler::new();
        memory.sample();
        for request_index in 0..options.repeats {
            let request_start = Instant::now();
            let request_faults = task_fault_counters();
            let mut first_token_ms = None;
            let mut load_to_first_token_ms = None;
            let mut token_ids = Vec::new();
            let result = runtime::run_raw_completion(
                &mut runner,
                &tokenizer,
                &prompt_ids,
                &config,
                options.context,
                vocab_size,
                |progress| {
                    if let RawDecodeProgress::Token { index, id, .. } = progress {
                        if index == 0 {
                            first_token_ms = Some(request_start.elapsed().as_secs_f64() * 1000.0);
                            // Only the first request includes loading and explicit expert preparation.
                            if request_index == 0 {
                                load_to_first_token_ms =
                                    Some(load_start.elapsed().as_secs_f64() * 1000.0);
                            }
                        }
                        token_ids.push(id);
                        if index % 8 == 0 {
                            memory.sample();
                        }
                    }
                },
            )
            .map_err(|e| e.to_string())?;
            let request_ms = request_start.elapsed().as_secs_f64() * 1000.0;
            let request_faults = fault_delta(request_faults, task_fault_counters());
            memory.sample();
            let compilation_after_request = compilation_json(runner.metal_compilation_stats());
            let flush_start = Instant::now();
            runner.flush_pipeline_cache();
            let flush_ms = flush_start.elapsed().as_secs_f64() * 1000.0;
            let ids_bytes: Vec<u8> = token_ids.iter().flat_map(|id| id.to_le_bytes()).collect();
            let phases = runner.phase_counters();
            let prefetch_json = prefetch.as_ref().map(|p| {
                json!({
                    "requested_experts": p.requested_experts,
                    "prepared_experts": p.prepared_experts,
                    "duplicate_experts": p.duplicate_experts,
                    "skipped_budget_experts": p.skipped_budget_experts,
                    "bytes_prepared": p.bytes_prepared,
                    "mapped_page_touches": p.mapped_page_touches,
                    "elapsed_ms": p.elapsed_ms,
                    "physical_residency_guaranteed": false,
                })
            });
            let mut record = json!({
                "schema_version": 1,
                "protocol": "moe-startup-v1",
                "label": options.label,
                "model": options.model,
                "manifest_sha256": manifest_sha256,
                "chip": chip_brand_string(),
                "page_cache_label": options.page_cache_label,
                "page_cache_label_is_verified": false,
                "open_index": open_index,
                "request_index": request_index,
                "family": family,
                "context": options.context,
                "max_new": options.max_new,
                "expert_cache_slots": runner.expert_cache_slots(),
                "residency": if options.mapped { "mapped" } else { "streamed" },
                "case": case.id,
                "shaping": if options.sampled { "sampled" } else { "greedy" },
                "seed": case.seed,
                "prefix_reuse": false,
                "discarded_warmups": 0,
                "precompiled_requested": std::env::var("TURBOSPARK_METAL_PRECOMPILED").as_deref() == Ok("1"),
                "kernel_warmup_requested": std::env::var("TURBOSPARK_METAL_KERNEL_WARMUP").as_deref() == Ok("1"),
                "pipeline_cache_requested": std::env::var("TURBOSPARK_METAL_PIPELINE_CACHE").as_deref() == Ok("1"),
                "shared_read_overlap_requested": std::env::var("TURBOSPARK_QWEN_SHARED_READ_OVERLAP").as_deref() == Ok("1"),
                "shared_read_overlap_effective": phases.qwen_shared_submissions > 0,
                "mapped_demand_prep_requested": std::env::var("TURBOSPARK_MAPPED_DEMAND_PREP").unwrap_or_else(|_| "off".into()),
                "mapped_demand_prep_effective": if phases.mapped_prepare_calls > 0 {
                    std::env::var("TURBOSPARK_MAPPED_DEMAND_PREP").unwrap_or_else(|_| "off".into())
                } else { "off".into() },
                "mapped_demand_budget_bytes": 16 * 1024 * 1024,
                "metadata_peek_ms": metadata_ms,
                "tokenizer_ms": tokenizer_ms,
                "load_ms": load_ms,
            });
            let metrics = json!({
                "startup": {
                    "total_open_ms": startup.total_open_ms,
                    "manifest_index_ms": startup.manifest_index_ms,
                    "resident_mapping_ms": startup.resident_mapping_ms,
                    "kv_scratch_ms": startup.kv_scratch_ms,
                    "expert_setup_ms": startup.expert_setup_ms,
                    "family_state_ms": startup.family_state_ms,
                    "session_state_ms": startup.session_state_ms,
                    "kernel_warmup_ms": startup.kernel_warmup.elapsed_ms,
                    "kernel_warmup_registrations": startup.kernel_warmup.registrations,
                    "kernel_warmup_unique_keys": startup.kernel_warmup.unique_keys,
                    "kernel_warmup_pipeline_creations": startup.kernel_warmup.pipeline_creations,
                },
                "pipeline_inventory": runner.metal_pipeline_inventory(),
                "compiler_at_open": compilation_at_open,
                "compiler_after_request": compilation_after_request,
                "compiler_after_flush": compilation_json(runner.metal_compilation_stats()),
                "cache_flush_ms": flush_ms,
                "open_faults": open_faults,
                "request_faults": request_faults,
                "expert_prefetch": prefetch_json,
                "prefetch_applied_to_this_request": request_index == 0 && prefetch.is_some(),
                "first_token_ms": first_token_ms,
                "load_to_first_token_ms": load_to_first_token_ms,
                "request_ms": request_ms,
                "prefill_ms": result.prefill_seconds * 1000.0,
                "decode_ms": result.decode_seconds * 1000.0,
                "prompt_tokens": result.prompt_tokens,
                "new_tokens": result.new_tokens,
                "stop": swift_reason_name(result.reason),
                "token_ids": token_ids,
                "token_sha256": model_io::hash_data(&ids_bytes),
                "peak_footprint_bytes": memory.peak_bytes(),
                "footprint_sampling_scope": "generation-after-open-and-prefetch",
                "expert_schedule": {
                    "shared_submissions_cumulative": phases.qwen_shared_submissions,
                    "mapped_prepare_calls_cumulative": phases.mapped_prepare_calls,
                    "mapped_prepared_bytes_cumulative": phases.mapped_prepared_bytes,
                    "mapped_page_touches_cumulative": phases.mapped_page_touches,
                    "mapped_advice_calls_cumulative": phases.mapped_advice_calls,
                    "mapped_advice_failures_cumulative": phases.mapped_advice_failures,
                    "mapped_prepare_ms_cumulative": phases.mapped_prepare_nanos as f64 / 1e6,
                    "expert_io_ms_cumulative": phases.expert_io_nanos as f64 / 1e6,
                    "gpu_wait_ms_cumulative": phases.gpu_wait_nanos as f64 / 1e6,
                },
                "expert_io": {
                    "requested_bytes_cumulative": phases.expert_io_bytes_requested,
                    "physical_bytes_cumulative": if phases.expert_io_samples > 0 { Some(phases.expert_io_bytes_physical) } else { None },
                    "samples_cumulative": phases.expert_io_samples,
                },
            });
            let Value::Object(metrics) = metrics else {
                unreachable!("metrics are an object");
            };
            record.as_object_mut().unwrap().extend(metrics);
            println!("{record}");
        }
    }
    Ok(())
}

/// CLI entry point, returning invalid arguments before opening a model.
pub fn main_entry(args: impl IntoIterator<Item = String>) -> ExitCode {
    match Options::parse(args) {
        Ok(None) => {
            println!("{USAGE}");
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("{e}\n{USAGE}");
            ExitCode::from(2)
        }
        Ok(Some(options)) => match run(options) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("startup probe: {e}");
                ExitCode::from(1)
            }
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(args: &[&str]) -> Result<Option<Options>, String> {
        Options::parse(args.iter().map(|s| (*s).to_owned()))
    }

    #[test]
    fn refuses_invalid_budgets_before_a_model_open() {
        for extra in [
            ["--slots", "0"],
            ["--slots", "17"],
            ["--context", "0"],
            ["--context", "4294967296"],
            ["--repeats", "0"],
            ["--max-new", "4096"],
            ["--prefetch-bytes", "10"],
            ["--expert-trace", "/missing.json"],
        ] {
            assert!(parse(&["--model", "/missing", extra[0], extra[1]]).is_err());
        }
    }

    #[test]
    fn repeated_requests_do_not_imply_a_discarded_warmup() {
        let options = parse(&["--model", "/missing"]).unwrap().unwrap();
        assert_eq!(options.repeats, 2);
        assert_eq!(options.reopens, 1);
        assert!(!options.sampled);
        assert!(!options.mapped);
        assert!(parse(&["--help"]).unwrap().is_none());
    }

    #[test]
    fn unavailable_fault_counters_remain_unknown() {
        assert_eq!(fault_delta(None, Some((10, 2))), Value::Null);
        assert_eq!(
            fault_delta(Some((7, 1)), Some((10, 2))),
            json!({"process_faults": 3, "process_pageins": 1})
        );
    }
}
