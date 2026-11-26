//! Prompts, arrival schedules and the statistics over request outcomes.
//! Pure but for `run_arrivals`, so the rules are testable without a server.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use serde::Serialize;
use serde_json::{json, Value};

use super::http::{chat, ChatRequest, Outcome, Server};

/// No prefill rate is reported below this many prompt tokens: the number is
/// dominated by request overhead, not prefill.
pub const MIN_PREFILL_PROMPT_TOKENS: u64 = 512;

/// SplitMix64: a seeded, dependency-free generator for prompts and arrivals.
pub struct Rng(pub u64);

impl Rng {
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in [0, 1).
    pub fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// A prompt size: `None` is the fixed short question.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Size {
    pub name: String,
    pub target_tokens: Option<u32>,
}

/// `short`, `N` or `Nk` (1024 tokens), comma separated.
pub fn parse_sizes(s: &str) -> Result<Vec<Size>, String> {
    s.split(',')
        .map(|part| {
            let name = part.trim().to_ascii_lowercase();
            let target_tokens = if name == "short" {
                None
            } else {
                let (digits, mult) = match name.strip_suffix('k') {
                    Some(d) => (d, 1024),
                    None => (name.as_str(), 1),
                };
                let n = digits
                    .parse::<u32>()
                    .ok()
                    .and_then(|n| n.checked_mul(mult))
                    .filter(|n| (1..=1 << 20).contains(n))
                    .ok_or_else(|| format!("size {part:?}: short, N or Nk"))?;
                Some(n)
            };
            Ok(Size {
                name,
                target_tokens,
            })
        })
        .collect()
}

const WORDS: &[&str] = &[
    "river", "station", "ledger", "harbor", "signal", "garden", "engine", "winter", "market",
    "copper", "letter", "bridge", "orchard", "meter", "valley", "record", "lantern", "survey",
    "timber", "window", "channel", "granite", "morning", "archive", "pilot", "cedar", "current",
    "index", "meadow", "relay", "summit", "quarry",
];

/// The prompt for one seed. The seed leads the text, so two seeds share no
/// prefix worth caching.
pub fn prompt(seed: u64, filler_words: Option<usize>) -> String {
    let Some(words) = filler_words else {
        return format!("Request {seed}: explain how a lighthouse works, in a few paragraphs.");
    };
    let mut rng = Rng(seed);
    let mut text = format!("Request {seed}. Field notes:\n");
    for i in 0..words {
        text.push_str(WORDS[(rng.next_u64() % WORDS.len() as u64) as usize]);
        text.push_str(if i % 12 == 11 { ".\n" } else { " " });
    }
    text.push_str("\n\nSummarize the field notes above in detail.");
    text
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ArrivalMode {
    Closed,
    Periodic,
    Poisson,
}

impl ArrivalMode {
    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "closed" => Ok(Self::Closed),
            "periodic" => Ok(Self::Periodic),
            "poisson" => Ok(Self::Poisson),
            _ => Err(format!("arrival {s:?}: closed, periodic or poisson")),
        }
    }
}

/// Open-loop send times as offsets from the batch start. They are fixed
/// before the first request goes out, so a slow server cannot slow its own
/// load. `Closed` has no schedule.
pub fn schedule(mode: ArrivalMode, rate: f64, seed: u64, n: usize) -> Vec<Duration> {
    let mut rng = Rng(seed);
    let mut at = 0.0f64;
    (0..n)
        .map(|i| {
            match mode {
                ArrivalMode::Closed => {}
                ArrivalMode::Periodic => at = i as f64 / rate,
                ArrivalMode::Poisson => at += -(1.0 - rng.unit()).ln() / rate,
            }
            Duration::from_secs_f64(at)
        })
        .collect()
}

#[derive(Debug, Clone, Serialize)]
pub struct Sample {
    pub size: String,
    pub seed: u64,
    /// Offset of the scheduled arrival from the batch start. In a closed
    /// loop, the moment a client slot became free.
    pub arrival_s: f64,
    /// How late the request left, so an overloaded client is visible.
    pub send_lag_s: f64,
    #[serde(flatten)]
    pub outcome: Outcome,
    /// Latencies measured from arrival rather than from send.
    pub arrival_first_output_s: Option<f64>,
    pub arrival_done_s: f64,
    pub capped: bool,
    pub accounting: Accounting,
    pub server_decode_tok_s: Option<f64>,
    pub server_prefill_tok_s: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Accounting {
    /// `ok`, `mismatch`, or `unavailable` (no usage to check).
    pub status: &'static str,
    /// Whether the server's rate times its seconds was checked too.
    pub rate_checked: bool,
    pub problems: Vec<String>,
}

/// Cross-check the server's own counts against each other.
pub fn account(o: &Outcome, max_tokens: u32) -> Accounting {
    let mut problems = Vec::new();
    let mut rate_checked = false;
    let (Some(prompt), Some(completion)) = (o.prompt_tokens, o.completion_tokens) else {
        return Accounting {
            status: "unavailable",
            rate_checked,
            problems,
        };
    };
    if completion > u64::from(max_tokens) {
        problems.push(format!(
            "completion {completion} over max_tokens {max_tokens}"
        ));
    }
    if o.total_tokens.is_some_and(|t| t != prompt + completion) {
        problems.push("total_tokens is not prompt + completion".into());
    }
    if completion == 0 && o.output_deltas > 0 {
        problems.push("output streamed but completion_tokens is 0".into());
    }
    if let Some(t) = &o.timings {
        if let (Some(n), Some(rate), Some(ms)) = (
            t["predicted_n"].as_f64(),
            t["predicted_per_second"].as_f64(),
            t["predicted_ms"].as_f64(),
        ) {
            rate_checked = true;
            let secs = ms / 1000.0;
            if (n - rate * secs).abs() > 0.05 * secs + 1.0 {
                problems.push(format!("{n} decoded is not {rate} tok/s x {secs} s"));
            }
            if (n - completion as f64).abs() > 1.0 {
                problems.push(format!("timings {n} against usage {completion}"));
            }
        }
    }
    Accounting {
        status: if problems.is_empty() {
            "ok"
        } else {
            "mismatch"
        },
        rate_checked,
        problems,
    }
}

pub fn sample(
    size: &str,
    req: &ChatRequest,
    arrival_s: f64,
    send_lag_s: f64,
    outcome: Outcome,
) -> Sample {
    let timing = |k: &str| outcome.timings.as_ref().and_then(|t| t[k].as_f64());
    // What the server says it PREFILLED when it says so: with prefix reuse a
    // long prompt can cost only a few tokens.
    let long_enough = timing("prompt_n")
        .map(|n| n as u64)
        .or(outcome.prompt_tokens)
        .is_some_and(|n| n >= MIN_PREFILL_PROMPT_TOKENS);
    Sample {
        size: size.to_string(),
        seed: req.seed,
        arrival_s,
        send_lag_s,
        arrival_first_output_s: outcome.ttft_s.map(|t| send_lag_s + t),
        arrival_done_s: send_lag_s + outcome.total_s,
        capped: outcome.finish_reason.as_deref() == Some("length"),
        accounting: account(&outcome, req.max_tokens),
        server_decode_tok_s: timing("predicted_per_second"),
        server_prefill_tok_s: long_enough.then(|| timing("prompt_per_second")).flatten(),
        outcome,
    }
}

#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct Slo {
    pub first_output_s: Option<f64>,
    pub duration_s: Option<f64>,
}

/// Judged on arrival-basis times: a request that waited in a queue waited.
pub fn meets(s: &Sample, slo: &Slo) -> bool {
    s.outcome.completed()
        && slo
            .first_output_s
            .is_none_or(|limit| s.arrival_first_output_s.is_some_and(|t| t <= limit))
        && slo.duration_s.is_none_or(|limit| s.arrival_done_s <= limit)
}

/// One concurrent batch. `aggregate_tok_s` is accepted tokens over the
/// batch's wall time; per-request rates are never averaged.
#[derive(Debug, Clone, Serialize)]
pub struct Batch {
    pub wall_s: f64,
    pub requests: usize,
    pub completed: usize,
    pub failed: usize,
    pub completion_tokens: u64,
    pub aggregate_tok_s: Option<f64>,
    pub met_slo: Option<usize>,
    /// Requests meeting the SLO per second of batch wall time.
    pub goodput_req_s: Option<f64>,
    pub samples: Vec<Sample>,
}

pub fn batch(samples: Vec<Sample>, wall_s: f64, slo: &Slo) -> Batch {
    let done = || samples.iter().filter(|s| s.outcome.completed());
    let tokens: u64 = done().filter_map(|s| s.outcome.completion_tokens).sum();
    let have_usage = done().all(|s| s.outcome.completion_tokens.is_some());
    let met = (slo.first_output_s.is_some() || slo.duration_s.is_some())
        .then(|| samples.iter().filter(|s| meets(s, slo)).count());
    Batch {
        wall_s,
        requests: samples.len(),
        completed: done().count(),
        failed: samples.iter().filter(|s| s.outcome.error.is_some()).count(),
        completion_tokens: tokens,
        // A completed request without usage would make the sum too small.
        aggregate_tok_s: (have_usage && wall_s > 0.0).then(|| tokens as f64 / wall_s),
        met_slo: met,
        goodput_req_s: met.filter(|_| wall_s > 0.0).map(|m| m as f64 / wall_s),
        samples,
    }
}

pub fn median(values: &[f64]) -> Option<f64> {
    let mut v = values.to_vec();
    v.sort_by(f64::total_cmp);
    let n = v.len();
    match n {
        0 => None,
        _ if n % 2 == 1 => Some(v[n / 2]),
        _ => Some((v[n / 2 - 1] + v[n / 2]) / 2.0),
    }
}

/// `{median, min, max, n}` of the values present, or null.
pub fn spread(values: impl Iterator<Item = Option<f64>>) -> Value {
    let v: Vec<f64> = values.flatten().collect();
    match median(&v) {
        None => Value::Null,
        Some(m) => json!({
            "median": m,
            "min": v.iter().copied().fold(f64::INFINITY, f64::min),
            "max": v.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            "n": v.len(),
        }),
    }
}

/// Send `requests` closed-loop (`concurrency` clients) or open-loop on
/// `offsets`. Open loop starts one thread per request, so nothing the
/// server does delays a later send. Returns samples in request order and
/// the batch wall time.
pub fn run_arrivals(
    server: &Server,
    requests: &[(String, ChatRequest)],
    mode: ArrivalMode,
    offsets: &[Duration],
    concurrency: usize,
) -> (Vec<Sample>, f64) {
    let start = Instant::now();
    let next = AtomicUsize::new(0);
    let mut samples: Vec<(usize, Sample)> = std::thread::scope(|scope| {
        let threads: Vec<_> = match mode {
            ArrivalMode::Closed => (0..concurrency.max(1).min(requests.len()))
                .map(|_| {
                    let next = &next;
                    scope.spawn(move || {
                        let mut mine = Vec::new();
                        loop {
                            let i = next.fetch_add(1, Ordering::Relaxed);
                            let Some((size, req)) = requests.get(i) else {
                                return mine;
                            };
                            let at = start.elapsed().as_secs_f64();
                            let out = chat(server, req, None);
                            mine.push((i, sample(size, req, at, 0.0, out)));
                        }
                    })
                })
                .collect(),
            _ => requests
                .iter()
                .zip(offsets)
                .enumerate()
                .map(|(i, ((size, req), &offset))| {
                    scope.spawn(move || {
                        std::thread::sleep(
                            (start + offset).saturating_duration_since(Instant::now()),
                        );
                        let lag = start.elapsed().saturating_sub(offset).as_secs_f64();
                        let out = chat(server, req, None);
                        vec![(i, sample(size, req, offset.as_secs_f64(), lag, out))]
                    })
                })
                .collect(),
        };
        threads
            .into_iter()
            .flat_map(|t| t.join().expect("request thread panicked"))
            .collect()
    });
    let wall = start.elapsed().as_secs_f64();
    samples.sort_by_key(|(i, _)| *i);
    (samples.into_iter().map(|(_, s)| s).collect(), wall)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn outcome(completion: u64, ttft: f64, total: f64) -> Outcome {
        Outcome {
            status: 200,
            ttft_s: Some(ttft),
            total_s: total,
            finish_reason: Some("stop".into()),
            prompt_tokens: Some(20),
            completion_tokens: Some(completion),
            total_tokens: Some(20 + completion),
            output_deltas: 1,
            ..Default::default()
        }
    }

    fn req() -> ChatRequest {
        ChatRequest {
            model: "m".into(),
            prompt: "p".into(),
            max_tokens: 100,
            seed: 1,
            timeout: Duration::from_secs(1),
            cancel_after_deltas: None,
        }
    }

    #[test]
    fn sizes_parse() {
        let s = parse_sizes("short,4k,16K,300").unwrap();
        let t: Vec<_> = s.iter().map(|s| s.target_tokens).collect();
        assert_eq!(t, [None, Some(4096), Some(16384), Some(300)]);
        assert!(parse_sizes("long").is_err());
        assert!(parse_sizes("0").is_err());
    }

    #[test]
    fn schedules_are_seeded_and_absolute() {
        let p = schedule(ArrivalMode::Periodic, 4.0, 1, 3);
        assert_eq!(p, [0, 250, 500].map(Duration::from_millis));
        let a = schedule(ArrivalMode::Poisson, 10.0, 7, 4000);
        assert_eq!(a, schedule(ArrivalMode::Poisson, 10.0, 7, 4000));
        assert_ne!(a, schedule(ArrivalMode::Poisson, 10.0, 8, 4000));
        assert!(a.windows(2).all(|w| w[0] <= w[1]), "arrivals not ordered");
        let mean = a.last().unwrap().as_secs_f64() / a.len() as f64;
        assert!((mean - 0.1).abs() < 0.01, "mean gap {mean} for rate 10");
    }

    #[test]
    fn slo_uses_arrival_basis_times() {
        let slo = Slo {
            first_output_s: Some(1.0),
            duration_s: None,
        };
        // On the wire 0.5 s, but it left 0.8 s late: 1.3 s from arrival.
        let late = sample("short", &req(), 0.0, 0.8, outcome(10, 0.5, 0.9));
        assert!(!meets(&late, &slo));
        let on_time = sample("short", &req(), 0.0, 0.0, outcome(10, 0.5, 0.9));
        assert!(meets(&on_time, &slo));
        let mut failed = on_time.clone();
        failed.outcome.error = Some("x".into());
        assert!(!meets(&failed, &slo));
        let b = batch(vec![late, on_time, failed], 2.0, &slo);
        assert_eq!((b.met_slo, b.goodput_req_s), (Some(1), Some(0.5)));
    }

    #[test]
    fn aggregate_is_tokens_over_wall_not_a_mean_of_rates() {
        // 90 tokens in 1 s and 10 in 10 s: the mean of rates would be 45.5.
        let samples = vec![
            sample("a", &req(), 0.0, 0.0, outcome(90, 0.1, 1.0)),
            sample("b", &req(), 0.0, 0.0, outcome(10, 0.1, 10.0)),
        ];
        let b = batch(samples, 10.0, &Slo::default());
        assert_eq!(b.aggregate_tok_s, Some(10.0));
        assert_eq!(b.goodput_req_s, None);
        let mut missing = outcome(5, 0.1, 1.0);
        missing.completion_tokens = None;
        let b = batch(
            vec![sample("a", &req(), 0.0, 0.0, missing)],
            1.0,
            &Slo::default(),
        );
        assert_eq!(b.aggregate_tok_s, None);
    }

    #[test]
    fn accounting_catches_each_inconsistency() {
        assert_eq!(account(&outcome(10, 0.1, 1.0), 100).status, "ok");
        assert_eq!(account(&outcome(101, 0.1, 1.0), 100).status, "mismatch");
        assert_eq!(account(&Outcome::default(), 100).status, "unavailable");
        let mut o = outcome(10, 0.1, 1.0);
        o.total_tokens = Some(99);
        assert_eq!(account(&o, 100).status, "mismatch");
        let mut o = outcome(100, 0.1, 1.0);
        o.timings =
            Some(json!({"predicted_n": 100, "predicted_per_second": 50.0, "predicted_ms": 2000.0}));
        let a = account(&o, 100);
        assert_eq!((a.status, a.rate_checked), ("ok", true));
        o.timings =
            Some(json!({"predicted_n": 100, "predicted_per_second": 40.0, "predicted_ms": 2000.0}));
        assert_eq!(account(&o, 100).status, "mismatch");
    }

    #[test]
    fn no_prefill_rate_for_tiny_prompts() {
        let mut o = outcome(10, 0.1, 1.0);
        o.timings = Some(json!({"prompt_per_second": 900.0}));
        assert_eq!(
            sample("a", &req(), 0.0, 0.0, o.clone()).server_prefill_tok_s,
            None
        );
        o.prompt_tokens = Some(MIN_PREFILL_PROMPT_TOKENS);
        assert_eq!(
            sample("a", &req(), 0.0, 0.0, o.clone()).server_prefill_tok_s,
            Some(900.0)
        );
        // Mostly reused: the long prompt was not a long prefill.
        o.timings = Some(json!({"prompt_per_second": 900.0, "prompt_n": 8}));
        assert_eq!(sample("a", &req(), 0.0, 0.0, o).server_prefill_tok_s, None);
    }
}
