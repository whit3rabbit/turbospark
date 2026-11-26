//! `serve_bench` and `soak`: the path a client actually uses, measured over
//! loopback HTTP and SSE.
//!
//! ```text
//! tslab serve-bench (--server http://127.0.0.1:PORT [--pid P] |
//!                    --server-bin PATH --model M [--server-arg A]... [--env K=V]...)
//!     [--model-id ID] [--experiment serve] [--lab DIR] [--sizes short,4k,16k] [--seeds 3]
//!     [--max-tokens 512] [--burst 4] [--warmup 1] [--request-timeout-s 900] [--allow-battery]
//!     [--arrival closed|periodic|poisson [--rate R] [--arrival-seed 1] [--requests N]
//!      [--concurrency 4] [--rounds 1] [--slo-first-output-s S] [--slo-duration-s S]]
//! tslab soak (same target flags) [--experiment soak] [--duration 3600] [--concurrency 3]
//!     [--max-tokens 64] [--disconnect-pct 10] [--long-prompt-tokens 8192] [--stall-s 180]
//!     [--sample-s 5] [--bucket-s 60] [--max-failures 1] [--seed 1]
//! ```
//!
//! Latencies are the client's monotonic clock. Token counts and rates are
//! only what the server reports; a rate it does not report stays null. No
//! request is ever retried and load is never lowered to make one pass.
//! Exit 3 means a request failed or the soak failed.

pub mod http;
pub mod load;
mod soak;
pub mod target;

use std::path::PathBuf;
use std::time::Duration;

use serde_json::{json, Value};

use crate::args::Args;
use crate::schema::CmdOutput;
use http::{json_request, ChatRequest};
use load::{ArrivalMode, Batch, Sample, Size, Slo};
use target::Target;

pub use soak::soak_cmd;

/// Bumped when the workload, the protocol or a metric's meaning changes, so
/// server rows never merge across methods or with in-process rows.
pub const METHODOLOGY_SERVE_BENCH: &str = "serve_bench/1:http-sse-loopback";
pub const METHODOLOGY_SOAK: &str = "soak/1:http-sse-loopback";

/// A request count past this is a typo, and open loop starts a thread each.
const MAX_REQUESTS: usize = 4096;
/// The exit code for a measured failure, as `hash-diff` and a discard use.
const EXIT_FAILED: i32 = 3;

fn lab_dir(args: &mut Args) -> Result<PathBuf, String> {
    Ok(args
        .take("--lab")?
        .map_or_else(crate::paths::lab_dir, PathBuf::from))
}

/// The server's own text-generation series, by name without the prefix.
fn parse_series(text: &str) -> std::collections::BTreeMap<String, f64> {
    text.lines()
        .filter_map(|l| l.strip_prefix("turbospark_text_"))
        .filter_map(|l| {
            let (name, value) = l.split_once(' ')?;
            Some((name.to_string(), value.trim().parse().ok()?))
        })
        .collect()
}

/// A scrape, or why there is none (an older server, or one that wants a key).
fn scrape(server: &http::Server) -> Result<std::collections::BTreeMap<String, f64>, String> {
    let (status, text) = http::text_request(server, "/v1/metrics", Duration::from_secs(10))?;
    let series = parse_series(&text);
    if status != 200 || series.is_empty() {
        return Err(format!(
            "/v1/metrics answered {status} with no turbospark_text_ series"
        ));
    }
    Ok(series)
}

/// Counters as their growth over the run, gauges as their last value, plus a
/// warning when the server generated a different number of tokens from what
/// this client received (another client, or a retried generation).
fn series_delta(
    before: &std::collections::BTreeMap<String, f64>,
    after: &std::collections::BTreeMap<String, f64>,
    received_tokens: u64,
) -> (Value, Option<String>) {
    let delta: serde_json::Map<String, Value> = after
        .iter()
        .map(|(name, &v)| {
            let value = if name.ends_with("_total") {
                v - before.get(name).copied().unwrap_or(0.0)
            } else {
                v
            };
            (name.clone(), json!(value))
        })
        .collect();
    let warning = delta
        .get("predicted_tokens_total")
        .and_then(Value::as_f64)
        .filter(|&g| g != received_tokens as f64)
        .map(|g| {
            format!(
                "the server counted {g} generated tokens, this client received {received_tokens}: \
                 another client or a retried generation shares these series"
            )
        });
    (Value::Object(delta), warning)
}

/// Filler word counts that land near each target token count, using the
/// server's own counter when it has one.
fn calibrate(target: &Target, sizes: &[Size]) -> (Vec<Option<usize>>, Vec<String>) {
    let mut warnings = Vec::new();
    let words = sizes
        .iter()
        .map(|size| {
            let want = size.target_tokens? as usize;
            let mut words = (want * 3 / 4).max(1);
            for _ in 0..2 {
                let body = json!({
                    "model": target.model_id,
                    "messages": [{"role": "user", "content": load::prompt(1, Some(words))}],
                });
                let counted = json_request(
                    &target.server,
                    "POST",
                    "/v1/messages/count_tokens",
                    Some(&body),
                    Duration::from_secs(60),
                )
                .ok()
                .and_then(|(_, v)| v["input_tokens"].as_u64())
                .filter(|&n| n > 0);
                let Some(n) = counted else {
                    warnings.push(format!(
                        "{}: no count_tokens; prompt length is a word-count guess, \
                         read prompt_tokens",
                        size.name
                    ));
                    break;
                };
                words = (words * want / n as usize).max(1);
            }
            Some(words)
        })
        .collect();
    (words, warnings)
}

struct Plan {
    sizes: Vec<Size>,
    seeds: u64,
    max_tokens: u32,
    burst: usize,
    warmup: usize,
    timeout: Duration,
    arrival: Option<Arrival>,
}

struct Arrival {
    mode: ArrivalMode,
    rate: Option<f64>,
    seed: u64,
    requests: usize,
    concurrency: usize,
    rounds: usize,
    slo: Slo,
}

fn positive(args: &mut Args, flag: &str) -> Result<Option<f64>, String> {
    match args.take(flag)? {
        None => Ok(None),
        Some(v) => v
            .parse::<f64>()
            .ok()
            .filter(|x| x.is_finite() && *x > 0.0)
            .map(Some)
            .ok_or_else(|| format!("{flag}: {v:?} must be a positive number")),
    }
}

fn take_plan(args: &mut Args) -> Result<Plan, String> {
    let sizes = load::parse_sizes(
        &args
            .take("--sizes")?
            .unwrap_or_else(|| "short,4k,16k".into()),
    )?;
    let seeds: u64 = args.parsed("--seeds", 3)?;
    let max_tokens: u32 = args.parsed("--max-tokens", 512)?;
    if seeds == 0 || seeds > 1000 || max_tokens == 0 {
        return Err("--seeds is 1..1000 and --max-tokens at least 1".into());
    }
    let burst: usize = args.parsed("--burst", 4)?;
    let warmup: usize = args.parsed("--warmup", 1)?;
    let timeout = Duration::from_secs(args.parsed("--request-timeout-s", 900)?);
    let arrival = match args.take("--arrival")? {
        None => None,
        Some(mode) => {
            let mode = ArrivalMode::parse(&mode)?;
            let rate = positive(args, "--rate")?;
            if (mode == ArrivalMode::Closed) == rate.is_some() {
                return Err("--rate goes with periodic or poisson arrivals, and only them".into());
            }
            Some(Arrival {
                mode,
                rate,
                seed: args.parsed("--arrival-seed", 1)?,
                requests: args.parsed("--requests", sizes.len() * seeds as usize)?,
                concurrency: args.parsed("--concurrency", 4)?,
                rounds: args.parsed("--rounds", 1)?,
                slo: Slo {
                    first_output_s: positive(args, "--slo-first-output-s")?,
                    duration_s: positive(args, "--slo-duration-s")?,
                },
            })
        }
    };
    let too_many = |n: usize| n > MAX_REQUESTS;
    if too_many(burst) || too_many(warmup) {
        return Err(format!("--burst and --warmup are at most {MAX_REQUESTS}"));
    }
    if let Some(a) = &arrival {
        if a.requests == 0 || too_many(a.requests) || a.rounds == 0 || a.concurrency == 0 {
            return Err(format!(
                "--requests is 1..{MAX_REQUESTS}; --rounds and --concurrency at least 1"
            ));
        }
    }
    Ok(Plan {
        sizes,
        seeds,
        max_tokens,
        burst,
        warmup,
        timeout,
        arrival,
    })
}

impl Plan {
    fn request(&self, target: &Target, words: Option<usize>, seed: u64) -> ChatRequest {
        ChatRequest {
            model: target.model_id.clone(),
            prompt: load::prompt(seed, words),
            max_tokens: self.max_tokens,
            seed,
            timeout: self.timeout,
            cancel_after_deltas: None,
        }
    }

    fn protocol(&self) -> Value {
        json!({
            "sizes": self.sizes,
            "seeds": self.seeds,
            "max_tokens": self.max_tokens,
            "temperature": 0,
            "burst": self.burst,
            "warmup": self.warmup,
            "request_timeout_s": self.timeout.as_secs(),
            "min_prefill_prompt_tokens": load::MIN_PREFILL_PROMPT_TOKENS,
            "arrival": self.arrival.as_ref().map(|a| json!({
                "mode": a.mode, "rate": a.rate, "seed": a.seed, "requests": a.requests,
                "concurrency": a.concurrency, "rounds": a.rounds, "slo": a.slo,
            })),
        })
    }
}

fn show(s: &Sample) {
    let f = |v: Option<f64>| v.map_or("-".to_string(), |v| format!("{v:.3}"));
    crate::say!(
        "{:<6} seed {:<5} prompt {:<6} out {:<5} ttft {} answer {} total {:.3} {}",
        s.size,
        s.seed,
        s.outcome.prompt_tokens.map_or(-1, |n| n as i64),
        s.outcome.completion_tokens.map_or(-1, |n| n as i64),
        f(s.outcome.ttft_s),
        f(s.outcome.first_answer_s),
        s.outcome.total_s,
        s.outcome
            .error
            .as_deref()
            .or(s.outcome.finish_reason.as_deref())
            .unwrap_or("?"),
    );
}

/// Latency medians per prompt size. Rates are the server's, never derived.
fn per_size(sizes: &[Size], samples: &[Sample]) -> Vec<Value> {
    sizes
        .iter()
        .map(|size| {
            let mine: Vec<&Sample> = samples.iter().filter(|s| s.size == size.name).collect();
            let ok = || mine.iter().filter(|s| s.outcome.completed());
            json!({
                "size": size.name,
                "requests": mine.len(),
                "completed": ok().count(),
                "capped": mine.iter().filter(|s| s.capped).count(),
                "prompt_tokens": load::spread(ok().map(|s| s.outcome.prompt_tokens.map(|n| n as f64))),
                "ttft_s": load::spread(ok().map(|s| s.outcome.ttft_s)),
                "first_answer_s": load::spread(ok().map(|s| s.outcome.first_answer_s)),
                "server_decode_tok_s": load::spread(ok().map(|s| s.server_decode_tok_s)),
                "server_prefill_tok_s": load::spread(ok().map(|s| s.server_prefill_tok_s)),
            })
        })
        .collect()
}

pub fn serve_bench_cmd(mut args: Args) -> Result<CmdOutput, String> {
    let spec = target::TargetSpec::take(&mut args)?;
    let lab = lab_dir(&mut args)?;
    let experiment = args.take("--experiment")?.unwrap_or_else(|| "serve".into());
    let allow_battery = args.flag("--allow-battery");
    let plan = take_plan(&mut args)?;
    args.finish()?;
    let dir = crate::paths::experiment_dir(&lab, &experiment)?;
    let _lock =
        crate::lock::MachineLock::acquire(&lab, &format!("tslab serve-bench {experiment}"))?;
    let ledger = crate::ledger::Ledger::open(&dir)?;
    let target = spec.open(&dir)?;
    let watch = target::HostWatch::start(target.pid, allow_battery);
    let (words, mut warnings) = calibrate(&target, &plan.sizes);
    // Scraped around warmup and every measured request, so the deltas cover
    // exactly this run's traffic on a server nobody else is using.
    let before = scrape(&target.server);

    // Warmup is reported apart and counts toward nothing but failures.
    // Its seeds are outside every measured range.
    let warmup: Vec<(String, ChatRequest)> = (0..plan.warmup)
        .map(|i| {
            let req = plan.request(&target, None, 900_000 + i as u64);
            ("short".to_string(), req)
        })
        .collect();
    let (warmup, _) = load::run_arrivals(&target.server, &warmup, ArrivalMode::Closed, &[], 1);

    let mut samples = Vec::new();
    let mut burst: Option<Batch> = None;
    let mut rounds: Vec<Batch> = Vec::new();
    match &plan.arrival {
        None => {
            let matrix: Vec<(String, ChatRequest)> = (1..=plan.seeds)
                .flat_map(|seed| {
                    plan.sizes
                        .iter()
                        .zip(&words)
                        .map(move |(size, w)| (seed, size, *w))
                })
                .map(|(seed, size, w)| (size.name.clone(), plan.request(&target, w, seed)))
                .collect();
            for one in &matrix {
                let (mut got, _) = load::run_arrivals(
                    &target.server,
                    std::slice::from_ref(one),
                    ArrivalMode::Closed,
                    &[],
                    1,
                );
                got.iter().for_each(show);
                samples.append(&mut got);
            }
            if plan.burst > 0 {
                let requests: Vec<(String, ChatRequest)> = (0..plan.burst)
                    .map(|i| {
                        let req = plan.request(&target, None, 800_000 + i as u64);
                        ("short".to_string(), req)
                    })
                    .collect();
                let (got, wall) = load::run_arrivals(
                    &target.server,
                    &requests,
                    ArrivalMode::Closed,
                    &[],
                    plan.burst,
                );
                got.iter().for_each(show);
                burst = Some(load::batch(got, wall, &Slo::default()));
            }
        }
        Some(a) => {
            for round in 0..a.rounds {
                // Fresh seeds each round, so no round replays the last.
                let requests: Vec<(String, ChatRequest)> = (0..a.requests)
                    .map(|i| {
                        let k = i % plan.sizes.len();
                        let seed = (round * a.requests + i + 1) as u64;
                        (
                            plan.sizes[k].name.clone(),
                            plan.request(&target, words[k], seed),
                        )
                    })
                    .collect();
                let offsets = load::schedule(
                    a.mode,
                    a.rate.unwrap_or(1.0),
                    a.seed.wrapping_add(round as u64),
                    a.requests,
                );
                let (got, wall) =
                    load::run_arrivals(&target.server, &requests, a.mode, &offsets, a.concurrency);
                got.iter().for_each(show);
                rounds.push(load::batch(got, wall, &a.slo));
            }
        }
    }

    let after = scrape(&target.server);
    let memory = target.pid.and_then(target::proc_stats);
    let (mut eligible, mut reasons, host) = watch.finish();
    let all: Vec<&Sample> = warmup
        .iter()
        .chain(&samples)
        .chain(burst.iter().flat_map(|b| &b.samples))
        .chain(rounds.iter().flat_map(|b| &b.samples))
        .collect();
    let failed = all.iter().filter(|s| s.outcome.error.is_some()).count();
    let capped = all.iter().filter(|s| s.capped).count();
    let mismatched = all
        .iter()
        .filter(|s| s.accounting.status == "mismatch")
        .count();
    if failed > 0 {
        eligible = false;
        reasons.push(format!("{failed} request(s) failed"));
        warnings.push(format!("{failed} request(s) failed"));
    }
    if mismatched > 0 {
        eligible = false;
        reasons.push(format!("{mismatched} token accounting mismatch(es)"));
        warnings.push(format!("{mismatched} token accounting mismatch(es)"));
    }
    let received: u64 = all.iter().filter_map(|s| s.outcome.completion_tokens).sum();
    let server_metrics = match (&before, &after) {
        (Ok(b), Ok(a)) => {
            let (delta, warning) = series_delta(b, a, received);
            warnings.extend(warning);
            delta
        }
        (Err(e), _) | (_, Err(e)) => {
            warnings.push(format!("no server metrics: {e}"));
            Value::Null
        }
    };
    if all.iter().all(|s| s.server_decode_tok_s.is_none()) {
        warnings.push("the server sends no timings: server tok/s is null".into());
    }

    let mut record = json!({
        "experiment": experiment,
        "methodology": METHODOLOGY_SERVE_BENCH,
        "target": target.record,
        "protocol": plan.protocol(),
        "eligible": eligible,
        "ineligible": reasons,
        "host": host,
        "counts": {"requests": all.len(), "failed": failed, "capped": capped,
                   "accounting_mismatch": mismatched},
        "server_memory": memory,
        "server_metrics": server_metrics,
        "warmup": warmup,
        "provenance": target::provenance(),
    });
    if plan.arrival.is_none() {
        record["by_size"] = json!(per_size(&plan.sizes, &samples));
        record["requests"] = json!(samples);
        record["burst"] = json!(burst);
    } else {
        // Medians and spread across rounds; percentiles are never pooled.
        record["goodput_req_s"] = load::spread(rounds.iter().map(|b| b.goodput_req_s));
        record["aggregate_tok_s"] = load::spread(rounds.iter().map(|b| b.aggregate_tok_s));
        record["rounds"] = json!(rounds);
    }
    ledger.append("serve_bench", record.clone())?;
    crate::say!(
        "{} requests, {failed} failed, {capped} capped, eligible={eligible}",
        all.len()
    );
    let code = if failed > 0 { EXIT_FAILED } else { 0 };
    let mut out = CmdOutput::with_code(code, record);
    out.warnings = warnings;
    Ok(out)
}

#[cfg(test)]
mod metrics_tests {
    use super::*;

    const BEFORE: &str = "# HELP x y\n\
        turbospark_text_requests_total 2\n\
        turbospark_text_predicted_tokens_total 10\n\
        turbospark_text_queue_waiting 3\n\
        turbospark_audio_requests_total 9\n";

    #[test]
    fn counters_become_deltas_and_gauges_stay() {
        let before = parse_series(BEFORE);
        assert_eq!(before.len(), 3, "audio series are not text series");
        let after = parse_series(
            "turbospark_text_requests_total 5\n\
             turbospark_text_predicted_tokens_total 82\n\
             turbospark_text_queue_waiting 0\n",
        );
        let (d, warning) = series_delta(&before, &after, 72);
        assert_eq!(d["requests_total"], 3.0);
        assert_eq!(d["predicted_tokens_total"], 72.0);
        assert_eq!(d["queue_waiting"], 0.0);
        assert_eq!(warning, None);
        let (_, warning) = series_delta(&before, &after, 70);
        assert!(warning.unwrap().contains("received 70"));
    }
}
