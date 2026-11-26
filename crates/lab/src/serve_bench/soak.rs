//! `soak`: steady concurrent load for a long time, watching for the failures
//! a short bench cannot show. `/health` answering is not liveness, so
//! completions are watched directly. A regression signal, not a benchmark.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde::Serialize;
use serde_json::{json, Value};

use super::http::{chat, health_ok, ChatRequest, Outcome, Server};
use super::load::{self, Rng};
use super::target::{self, ProcStats, Target};
use super::{calibrate, lab_dir, EXIT_FAILED, METHODOLOGY_SOAK};
use crate::args::Args;
use crate::schema::CmdOutput;

const TICK: Duration = Duration::from_millis(50);
/// Output deltas a client reads before walking away.
const DISCONNECT_AFTER: usize = 3;
/// Error strings kept in the record; the count is always complete.
const KEPT_ERRORS: usize = 20;

#[derive(Debug, Clone, Copy)]
struct Config {
    duration: Duration,
    concurrency: usize,
    max_tokens: u32,
    disconnect_pct: f64,
    long_prompt_tokens: u32,
    stall: Duration,
    sample: Duration,
    bucket_s: u64,
    max_failures: usize,
    seed: u64,
    timeout: Duration,
}

#[derive(Debug, Clone, Default, Serialize, PartialEq)]
pub struct Bucket {
    pub minute: u64,
    pub completed: u64,
    pub failed: u64,
    pub disconnected: u64,
    pub completion_tokens: u64,
}

#[derive(Default)]
struct Shared {
    buckets: Vec<Bucket>,
    errors: Vec<String>,
    failed: usize,
    completed: usize,
    disconnected: usize,
    last_completion: Option<Instant>,
    long_prefill: Option<Value>,
}

fn record(shared: &Mutex<Shared>, start: Instant, bucket_s: u64, o: &Outcome) {
    let mut s = shared.lock().unwrap_or_else(|p| p.into_inner());
    let index = start.elapsed().as_secs() / bucket_s;
    while s.buckets.len() as u64 <= index {
        let minute = s.buckets.len() as u64;
        s.buckets.push(Bucket {
            minute,
            ..Default::default()
        });
    }
    let i = index as usize;
    if let Some(e) = &o.error {
        s.failed += 1;
        s.buckets[i].failed += 1;
        if s.errors.len() < KEPT_ERRORS {
            s.errors.push(e.clone());
        }
    } else if o.disconnected_by_client {
        s.disconnected += 1;
        s.buckets[i].disconnected += 1;
    } else {
        s.completed += 1;
        s.buckets[i].completed += 1;
        s.buckets[i].completion_tokens += o.completion_tokens.unwrap_or(0);
        s.last_completion = Some(Instant::now());
    }
}

/// Least-squares slope of footprint over time, bytes per minute.
pub fn slope_per_minute(points: &[(f64, f64)]) -> Option<f64> {
    let n = points.len() as f64;
    if points.len() < 2 {
        return None;
    }
    let mx = points.iter().map(|p| p.0).sum::<f64>() / n;
    let my = points.iter().map(|p| p.1).sum::<f64>() / n;
    let sxx: f64 = points.iter().map(|p| (p.0 - mx).powi(2)).sum();
    if sxx == 0.0 {
        return None;
    }
    let sxy: f64 = points.iter().map(|p| (p.0 - mx) * (p.1 - my)).sum();
    Some(60.0 * sxy / sxx)
}

fn process_summary(samples: &[(f64, ProcStats)]) -> Value {
    let footprint: Vec<(f64, f64)> = samples
        .iter()
        .map(|(t, s)| (*t, s.phys_footprint as f64))
        .collect();
    let counts = |pick: fn(&ProcStats) -> Option<u32>| {
        let v: Vec<u32> = samples.iter().filter_map(|(_, s)| pick(s)).collect();
        json!({"first": v.first(), "last": v.last(), "max": v.iter().max()})
    };
    json!({
        "samples": samples.len(),
        "footprint_first": footprint.first().map(|p| p.1),
        "footprint_last": footprint.last().map(|p| p.1),
        "footprint_max": footprint.iter().map(|p| p.1).fold(None, |m: Option<f64>, x| Some(m.map_or(x, |m| m.max(x)))),
        "footprint_slope_bytes_per_min": slope_per_minute(&footprint),
        // The first half holds the cache filling up; a leak shows here too.
        "footprint_slope_second_half": slope_per_minute(&footprint[footprint.len() / 2..]),
        "threads": counts(|s| s.threads),
        "fds": counts(|s| s.fds),
    })
}

struct Load<'a> {
    server: Server,
    model: &'a str,
    cfg: Config,
    long_words: Option<usize>,
    start: Instant,
    shared: &'a Mutex<Shared>,
    stop: &'a AtomicBool,
    abort: Arc<AtomicBool>,
    long_sent: AtomicBool,
}

fn client(load: &Load<'_>, index: usize) {
    let cfg = load.cfg;
    let mut rng = Rng(cfg.seed.wrapping_add(index as u64));
    let mut n = 0u64;
    while !load.stop.load(Ordering::Relaxed) && load.start.elapsed() < cfg.duration {
        n += 1;
        // One long prefill, halfway through, so short requests queue behind it.
        let long = load.long_words.is_some()
            && load.start.elapsed() >= cfg.duration / 2
            && !load.long_sent.swap(true, Ordering::Relaxed);
        let seed = (index as u64) << 32 | n;
        let cancel = !long && rng.unit() * 100.0 < cfg.disconnect_pct;
        let req = ChatRequest {
            model: load.model.to_string(),
            prompt: load::prompt(seed, load.long_words.filter(|_| long)),
            max_tokens: cfg.max_tokens,
            seed,
            timeout: cfg.timeout,
            cancel_after_deltas: cancel.then_some(DISCONNECT_AFTER),
        };
        let out = chat(&load.server, &req, Some(load.abort.clone()));
        // Requests cut off by the harness's own abort are not server failures.
        if load.abort.load(Ordering::Relaxed) && out.error.is_some() {
            return;
        }
        if long {
            load.shared
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .long_prefill = Some(json!(out));
        }
        record(load.shared, load.start, cfg.bucket_s, &out);
    }
}

/// Why the soak stopped early, if it should.
fn watch(
    load: &Load<'_>,
    target: &mut Target,
    memory: &mut Vec<(f64, ProcStats)>,
    done: &dyn Fn() -> bool,
) -> Option<String> {
    let cfg = load.cfg;
    let mut next_sample = Instant::now();
    while !done() {
        if Instant::now() >= next_sample {
            next_sample += cfg.sample;
            if let Some(status) = target.exited() {
                return Some(format!("server exited: {status}"));
            }
            if !health_ok(&load.server) {
                return Some("/health stopped answering".into());
            }
            if let Some(s) = target.pid.and_then(target::proc_stats) {
                memory.push((load.start.elapsed().as_secs_f64(), s));
            }
        }
        let (failed, last) = {
            let s = load.shared.lock().unwrap_or_else(|p| p.into_inner());
            (s.failed, s.last_completion)
        };
        if failed >= cfg.max_failures {
            return Some(format!("{failed} request(s) failed"));
        }
        let quiet = last.unwrap_or(load.start).elapsed();
        // Past the duration no new requests start, but the ones in flight
        // must still complete.
        if quiet >= cfg.stall {
            let health = if health_ok(&load.server) {
                "while /health is OK"
            } else {
                "and /health fails"
            };
            return Some(format!(
                "no completion for {:.0} s {health}",
                quiet.as_secs_f64()
            ));
        }
        std::thread::sleep(TICK);
    }
    None
}

fn take_config(args: &mut Args) -> Result<Config, String> {
    let secs = |args: &mut Args, flag: &str, default: f64| -> Result<Duration, String> {
        let v: f64 = args.parsed(flag, default)?;
        (v.is_finite() && v > 0.0 && v <= 7.0 * 86400.0)
            .then(|| Duration::from_secs_f64(v))
            .ok_or_else(|| format!("{flag} must be positive seconds"))
    };
    let cfg = Config {
        duration: secs(args, "--duration", 3600.0)?,
        concurrency: args.parsed("--concurrency", 3)?,
        max_tokens: args.parsed("--max-tokens", 64)?,
        disconnect_pct: args.parsed("--disconnect-pct", 10.0)?,
        long_prompt_tokens: args.parsed("--long-prompt-tokens", 8192)?,
        stall: secs(args, "--stall-s", 180.0)?,
        sample: secs(args, "--sample-s", 5.0)?,
        bucket_s: args.parsed("--bucket-s", 60)?,
        max_failures: args.parsed("--max-failures", 1)?,
        seed: args.parsed("--seed", 1)?,
        timeout: secs(args, "--request-timeout-s", 900.0)?,
    };
    if cfg.concurrency == 0
        || cfg.concurrency > 256
        || cfg.max_tokens == 0
        || cfg.bucket_s == 0
        || cfg.max_failures == 0
        || !(0.0..=100.0).contains(&cfg.disconnect_pct)
    {
        return Err(
            "soak: --concurrency 1..256, --disconnect-pct 0..100, and positive \
                    --max-tokens, --bucket-s, --max-failures"
                .into(),
        );
    }
    Ok(cfg)
}

pub fn soak_cmd(mut args: Args) -> Result<CmdOutput, String> {
    let spec = target::TargetSpec::take(&mut args)?;
    let lab = lab_dir(&mut args)?;
    let experiment = args.take("--experiment")?.unwrap_or_else(|| "soak".into());
    let allow_battery = args.flag("--allow-battery");
    let cfg = take_config(&mut args)?;
    args.finish()?;
    let dir = crate::paths::experiment_dir(&lab, &experiment)?;
    let _lock = crate::lock::MachineLock::acquire(&lab, &format!("tslab soak {experiment}"))?;
    let ledger = crate::ledger::Ledger::open(&dir)?;
    let mut target = spec.open(&dir)?;
    let mut warnings = Vec::new();
    let long_words = if cfg.long_prompt_tokens == 0 {
        None
    } else {
        let size = load::Size {
            name: "long".into(),
            target_tokens: Some(cfg.long_prompt_tokens),
        };
        let (words, w) = calibrate(&target, &[size]);
        warnings.extend(w);
        words[0]
    };
    if target.pid.and_then(target::proc_stats).is_none() {
        warnings.push("no readable server pid: footprint, threads and fds are not watched".into());
    }
    let host_watch = target::HostWatch::start(target.pid, allow_battery);
    let shared = Mutex::new(Shared::default());
    let stop = AtomicBool::new(false);
    let model = target.model_id.clone();
    let load = Load {
        server: target.server,
        model: &model,
        cfg,
        long_words,
        start: Instant::now(),
        shared: &shared,
        stop: &stop,
        abort: Arc::new(AtomicBool::new(false)),
        long_sent: AtomicBool::new(false),
    };
    let mut memory = Vec::new();
    let mut failure = std::thread::scope(|scope| {
        let clients: Vec<_> = (0..cfg.concurrency)
            .map(|i| {
                let load = &load;
                scope.spawn(move || client(load, i))
            })
            .collect();
        let failure = watch(&load, &mut target, &mut memory, &|| {
            clients.iter().all(|c| c.is_finished())
        });
        stop.store(true, Ordering::Relaxed);
        if failure.is_some() {
            load.abort.store(true, Ordering::Relaxed);
        }
        failure
    });
    let elapsed = load.start.elapsed().as_secs_f64();

    // The server exposes no text scheduler depth, so "back to idle" is read
    // by one request after the load: it must complete, promptly.
    let drain = failure.is_none().then(|| {
        chat(
            &load.server,
            &ChatRequest {
                model: model.clone(),
                prompt: load::prompt(0, None),
                max_tokens: cfg.max_tokens,
                seed: 0,
                timeout: cfg.stall,
                cancel_after_deltas: None,
            },
            None,
        )
    });
    if let Some(e) = drain.as_ref().and_then(|d| d.error.as_ref()) {
        failure = Some(format!("drain probe after the load failed: {e}"));
    }
    if let Some(s) = target.pid.and_then(target::proc_stats) {
        memory.push((load.start.elapsed().as_secs_f64(), s));
    }
    let (eligible, reasons, host) = host_watch.finish();
    let s = shared.into_inner().unwrap_or_else(|p| p.into_inner());
    // A failure that landed as the clients finished is still a failure.
    if failure.is_none() && s.failed > 0 {
        failure = Some(format!("{} request(s) failed", s.failed));
    }
    if failure.is_none() && s.completed == 0 {
        failure = Some("no request completed".into());
    }
    if long_words.is_some() && s.long_prefill.is_none() {
        warnings.push("the long prefill was never sent".into());
    }
    let passed = failure.is_none();
    let record = json!({
        "experiment": experiment,
        "methodology": METHODOLOGY_SOAK,
        "target": target.record,
        "protocol": {
            "duration_s": cfg.duration.as_secs_f64(), "concurrency": cfg.concurrency,
            "max_tokens": cfg.max_tokens, "disconnect_pct": cfg.disconnect_pct,
            "long_prompt_tokens": cfg.long_prompt_tokens, "stall_s": cfg.stall.as_secs_f64(),
            "sample_s": cfg.sample.as_secs_f64(), "bucket_s": cfg.bucket_s,
            "max_failures": cfg.max_failures, "seed": cfg.seed,
            "request_timeout_s": cfg.timeout.as_secs_f64(),
        },
        "passed": passed,
        "failure": failure,
        "elapsed_s": elapsed,
        "completed": s.completed,
        "failed": s.failed,
        "disconnected": s.disconnected,
        "errors": s.errors,
        "per_bucket": s.buckets,
        "long_prefill": s.long_prefill,
        "drain_probe": drain,
        "process": process_summary(&memory),
        "eligible": eligible,
        "ineligible": reasons,
        "host": host,
        "provenance": target::provenance(),
    });
    ledger.append("soak", record.clone())?;
    crate::say!(
        "soak {}: {} completed, {} failed, {} disconnected in {elapsed:.0} s{}",
        if passed { "passed" } else { "FAILED" },
        record["completed"],
        record["failed"],
        record["disconnected"],
        record["failure"]
            .as_str()
            .map(|f| format!(" ({f})"))
            .unwrap_or_default(),
    );
    let mut out = CmdOutput::with_code(if passed { 0 } else { EXIT_FAILED }, record);
    out.warnings = warnings;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slope_is_per_minute() {
        let p: Vec<(f64, f64)> = (0..10)
            .map(|i| (i as f64 * 6.0, 100.0 + i as f64 * 5.0))
            .collect();
        assert!((slope_per_minute(&p).unwrap() - 50.0).abs() < 1e-9);
        assert_eq!(slope_per_minute(&p[..1]), None);
        assert_eq!(slope_per_minute(&[(1.0, 2.0), (1.0, 3.0)]), None);
    }
}
