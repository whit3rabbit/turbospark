//! Is the host quiet enough to time anything?
//!
//! A port of `scripts/moe_expert_schedule_pairs.py`'s `HostMonitor` and
//! `host_eligibility`, with thresholds re-set for how this machine actually
//! idles: the script flagged any process over 5% CPU, and WindowServer alone
//! sits near 30% with a display on, so every run would read dirty. The rules
//! below flag what has moved results here before (`docs/SPEEDUP_PLAYBOOK.md`,
//! "Measurement traps"): another model process, a compile, battery power, a
//! thermal speed limit, and sustained heavy foreign CPU.
//!
//! CPU is interval CPU between samples (from `ps time=` deltas), not `ps`'s
//! lifetime average, and the measuring process tree is excluded.

use std::collections::HashMap;
use std::process::Command;
use std::time::Instant;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForeignProcess {
    pub pid: i32,
    pub cpu_percent: f64,
    pub command: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HostSample {
    /// Seconds since the monitor started.
    pub t_s: f64,
    /// Seconds since the previous sample, when there was one.
    pub interval_s: Option<f64>,
    pub load1: f64,
    pub on_ac: Option<bool>,
    pub thermal_limited: Option<bool>,
    pub foreign_cpu_percent: f64,
    /// The busiest foreign processes, at most 8.
    pub foreign_top: Vec<ForeignProcess>,
    /// Commands of other model or measurement processes.
    pub foreign_models: Vec<String>,
    pub rustc_processes: usize,
    pub swap_used_mib: Option<f64>,
    /// `vm_stat`'s cumulative compressor counters (pages). A run's delta is
    /// the memory-pressure reading swap alone misses: a 96-slot qwen4 cache
    /// made about 5 million of each per run with swap barely moving.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compressions: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decompressions: Option<u64>,
    pub error: Option<String>,
    /// The orchestrator's own busy process beside this run (`~hog:N`),
    /// when one was started. Inside the owned tree, so never foreign CPU;
    /// recorded so the run can say it was loaded on purpose.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub contention: Option<HogProcess>,
}

/// A controlled CPU load the orchestrator runs beside the worker.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct HogProcess {
    pub pid: u32,
    pub threads: usize,
}

/// Process-name fragments that mean "another model or measurement is
/// running". Test binaries live under `target/*/deps/<name>-<hash>`, so the
/// gate, oracle and probe fragments catch them by name.
const MODEL_MARKERS: &[&str] = &[
    "mlx_lm",
    "mlx-lm",
    "llama-server",
    "llama-cli",
    "mference",
    "ollama",
    "kapitsa",
    "turbospark-server",
    "turbospark-bench",
    "turbospark-check",
    "turbospark-startup-probe",
    "tslab-worker",
    "memory_oracle",
    "quality_gate",
    "_probe-",
    "_gate-",
    "_oracle-",
    "_bench-",
];

pub fn is_model_process(command: &str) -> bool {
    let lower = command.to_ascii_lowercase();
    MODEL_MARKERS.iter().any(|m| lower.contains(m))
}

pub struct HostMonitor {
    started: Instant,
    previous: HashMap<i32, f64>,
    previous_at: Option<Instant>,
    hog: Option<HogProcess>,
}

impl Default for HostMonitor {
    fn default() -> Self {
        Self::new()
    }
}

struct PsRow {
    pid: i32,
    ppid: i32,
    cpu_percent: f64,
    cpu_seconds: f64,
    command: String,
}

/// `[[dd-]hh:]mm:ss.ss` from `ps time=`.
fn cpu_seconds(field: &str) -> Option<f64> {
    let (days, clock) = match field.split_once('-') {
        Some((d, c)) => (d.parse::<f64>().ok()?, c),
        None => (0.0, field),
    };
    let parts: Vec<f64> = clock
        .split(':')
        .map(|p| p.parse::<f64>())
        .collect::<Result<_, _>>()
        .ok()?;
    let secs = match parts.as_slice() {
        [m, s] => m * 60.0 + s,
        [h, m, s] => h * 3600.0 + m * 60.0 + s,
        _ => return None,
    };
    Some(days * 86400.0 + secs)
}

fn parse_ps(output: &str) -> Vec<PsRow> {
    output
        .lines()
        .filter_map(|line| {
            let mut it = line.split_whitespace();
            let pid = it.next()?.parse().ok()?;
            let ppid = it.next()?.parse().ok()?;
            let _pgid: i32 = it.next()?.parse().ok()?;
            let cpu_percent = it.next()?.parse().ok()?;
            let cpu_seconds = cpu_seconds(it.next()?)?;
            let command = it.collect::<Vec<_>>().join(" ");
            Some(PsRow {
                pid,
                ppid,
                cpu_percent,
                cpu_seconds,
                command,
            })
        })
        .collect()
}

fn run(cmd: &str, args: &[&str]) -> Result<String, String> {
    let out = Command::new(cmd)
        .args(args)
        .output()
        .map_err(|e| format!("{cmd}: {e}"))?;
    if !out.status.success() {
        return Err(format!("{cmd} exited {}", out.status));
    }
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}

/// `Some(true)` when `pmset -g batt` reports AC power.
pub fn on_ac_power() -> Option<bool> {
    run("pmset", &["-g", "batt"])
        .ok()
        .map(|s| s.contains("'AC Power'"))
}

/// `Some(true)` when `pmset -g therm` reports a CPU or GPU speed limit
/// below 100.
pub fn thermal_limited() -> Option<bool> {
    let text = run("pmset", &["-g", "therm"]).ok()?;
    Some(text.lines().any(|line| {
        let l = line.trim();
        (l.starts_with("CPU_Speed_Limit") || l.starts_with("GPU_Speed_Limit"))
            && l.split('=')
                .nth(1)
                .and_then(|v| v.trim().parse::<u32>().ok())
                .is_some_and(|v| v != 100)
    }))
}

pub fn load1() -> f64 {
    let mut loads = [0f64; 3];
    // SAFETY: getloadavg writes at most 3 doubles into the array we pass.
    let n = unsafe { libc::getloadavg(loads.as_mut_ptr(), 3) };
    if n >= 1 {
        loads[0]
    } else {
        f64::NAN
    }
}

/// `used = 1127.38M` from `sysctl vm.swapusage`, in MiB.
pub fn swap_used_mib() -> Option<f64> {
    let text = run("sysctl", &["-n", "vm.swapusage"]).ok()?;
    let used = text.split("used = ").nth(1)?.split_whitespace().next()?;
    let (num, unit) = used.split_at(used.len() - 1);
    let v: f64 = num.parse().ok()?;
    Some(match unit {
        "G" => v * 1024.0,
        "M" => v,
        "K" => v / 1024.0,
        _ => return None,
    })
}

/// `Compressions:` and `Decompressions:` from `vm_stat`, cumulative pages.
pub fn compressor_counters() -> (Option<u64>, Option<u64>) {
    match run("vm_stat", &[]) {
        Ok(text) => parse_compressor_counters(&text),
        Err(_) => (None, None),
    }
}

fn parse_compressor_counters(text: &str) -> (Option<u64>, Option<u64>) {
    let field = |name: &str| {
        text.lines().find_map(|line| {
            let rest = line.strip_prefix(name)?.strip_prefix(':')?;
            rest.trim().trim_end_matches('.').parse::<u64>().ok()
        })
    };
    (field("Compressions"), field("Decompressions"))
}

impl HostMonitor {
    pub fn new() -> Self {
        Self {
            started: Instant::now(),
            previous: HashMap::new(),
            previous_at: None,
            hog: None,
        }
    }

    /// Name the controlled-contention process every later sample records.
    /// It is a child of this process, so the ppid walk already excludes it
    /// from foreign CPU; this only makes the load visible in the record.
    pub fn set_hog(&mut self, pid: u32, threads: usize) {
        self.hog = Some(HogProcess { pid, threads });
    }

    /// Take one sample. `owned_root` is the pid of the measured child, whose
    /// whole tree is excluded along with this process.
    pub fn sample(&mut self, owned_root: Option<u32>) -> HostSample {
        let now = Instant::now();
        let mut sample = HostSample {
            t_s: now.duration_since(self.started).as_secs_f64(),
            interval_s: self
                .previous_at
                .map(|p| now.duration_since(p).as_secs_f64()),
            load1: load1(),
            on_ac: on_ac_power(),
            thermal_limited: thermal_limited(),
            foreign_cpu_percent: 0.0,
            foreign_top: Vec::new(),
            foreign_models: Vec::new(),
            rustc_processes: 0,
            swap_used_mib: swap_used_mib(),
            compressions: None,
            decompressions: None,
            error: None,
            contention: self.hog,
        };
        (sample.compressions, sample.decompressions) = compressor_counters();
        let ps = match run("ps", &["-Ao", "pid=,ppid=,pgid=,pcpu=,time=,comm="]) {
            Ok(s) => s,
            Err(e) => {
                sample.error = Some(e);
                return sample;
            }
        };
        let rows = parse_ps(&ps);
        if rows.is_empty() {
            sample.error = Some("ps returned no rows".into());
            return sample;
        }
        let mut owned: std::collections::HashSet<i32> = std::iter::once(std::process::id() as i32)
            .chain(owned_root.map(|p| p as i32))
            .collect();
        loop {
            let before = owned.len();
            for r in &rows {
                if owned.contains(&r.ppid) {
                    owned.insert(r.pid);
                }
            }
            if owned.len() == before {
                break;
            }
        }
        let interval = sample.interval_s;
        let mut foreign: Vec<ForeignProcess> = Vec::new();
        let mut seen = HashMap::new();
        for r in rows.iter().filter(|r| !owned.contains(&r.pid)) {
            let mut cpu = r.cpu_percent;
            if let (Some(prior), Some(dt)) = (self.previous.get(&r.pid), interval) {
                if dt > 0.0 && r.cpu_seconds >= *prior {
                    cpu = 100.0 * (r.cpu_seconds - prior) / dt;
                }
            }
            seen.insert(r.pid, r.cpu_seconds);
            if r.command.rsplit('/').next() == Some("rustc") {
                sample.rustc_processes += 1;
            }
            if is_model_process(&r.command) {
                sample.foreign_models.push(r.command.clone());
            }
            sample.foreign_cpu_percent += cpu;
            foreign.push(ForeignProcess {
                pid: r.pid,
                cpu_percent: cpu,
                command: r.command.clone(),
            });
        }
        self.previous = seen;
        self.previous_at = Some(now);
        foreign.sort_by(|a, b| b.cpu_percent.total_cmp(&a.cpu_percent));
        foreign.truncate(8);
        sample.foreign_top = foreign;
        sample
    }
}

/// Thresholds for a clean run. Defaults are set for this machine with a
/// display on; `tslab` exposes them so a quieter host can tighten them.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct HostRules {
    /// Total foreign interval CPU, percent of one core.
    pub max_total_cpu_percent: f64,
    /// Any one foreign process.
    pub max_single_cpu_percent: f64,
    /// How long CPU must stay over a threshold to count.
    pub sustained_s: f64,
    pub require_ac: bool,
    /// How long to wait for the host to go quiet before each pair, seconds.
    /// 0 measures regardless, which is the old behaviour. A pair started on a
    /// quiet host can still turn dirty mid-run; this only stops the lab from
    /// burning a whole pair on a host that was already busy at the start.
    #[serde(default = "default_quiet_wait_s")]
    pub quiet_wait_s: f64,
    /// Refuse to start with less free disk than this on the lab's volume,
    /// GiB. 0 disables. Release builds are about 5 GB per worktree.
    #[serde(default = "default_min_free_gb")]
    pub min_free_gb: f64,
}

fn default_quiet_wait_s() -> f64 {
    300.0
}

fn default_min_free_gb() -> f64 {
    10.0
}

impl Default for HostRules {
    fn default() -> Self {
        Self {
            max_total_cpu_percent: 150.0,
            max_single_cpu_percent: 90.0,
            sustained_s: 5.0,
            require_ac: true,
            quiet_wait_s: default_quiet_wait_s(),
            min_free_gb: default_min_free_gb(),
        }
    }
}

/// Whether a run sampled as `samples` is clean, and every reason it is not.
pub fn eligibility(samples: &[HostSample], rules: &HostRules) -> (bool, Vec<String>) {
    let mut reasons = std::collections::BTreeSet::new();
    if samples.is_empty() {
        reasons.insert("no host samples".to_string());
    }
    let mut total_since: Option<f64> = None;
    let mut single_since: HashMap<i32, f64> = HashMap::new();
    for s in samples {
        if let Some(e) = &s.error {
            reasons.insert(format!("host monitor failed: {e}"));
        }
        if !s.foreign_models.is_empty() {
            reasons.insert(format!(
                "another model or measurement process: {}",
                s.foreign_models.join(", ")
            ));
        }
        if s.rustc_processes > 0 {
            reasons.insert("a compile (rustc) was running".into());
        }
        if rules.require_ac && s.on_ac != Some(true) {
            reasons.insert("AC power was not observed".into());
        }
        if s.thermal_limited != Some(false) {
            reasons.insert("thermal speed limit observed or unreadable".into());
        }
        // Interval CPU describes the time before this sample, so an episode
        // starts at the previous sample's time.
        let start = s.t_s - s.interval_s.unwrap_or(0.0);
        if s.interval_s.is_some() && s.foreign_cpu_percent > rules.max_total_cpu_percent {
            let since = *total_since.get_or_insert(start);
            if s.t_s - since >= rules.sustained_s {
                reasons.insert(format!(
                    "foreign CPU over {:.0}% for {:.0}s or more",
                    rules.max_total_cpu_percent, rules.sustained_s
                ));
            }
        } else {
            total_since = None;
        }
        let busy: std::collections::HashSet<i32> = s
            .foreign_top
            .iter()
            .filter(|p| s.interval_s.is_some() && p.cpu_percent > rules.max_single_cpu_percent)
            .map(|p| p.pid)
            .collect();
        single_since.retain(|pid, _| busy.contains(pid));
        for p in s.foreign_top.iter().filter(|p| busy.contains(&p.pid)) {
            let since = *single_since.entry(p.pid).or_insert(start);
            if s.t_s - since >= rules.sustained_s {
                reasons.insert(format!(
                    "{} over {:.0}% for {:.0}s or more",
                    p.command, rules.max_single_cpu_percent, rules.sustained_s
                ));
            }
        }
    }
    (reasons.is_empty(), reasons.into_iter().collect())
}

/// What waiting for a quiet host found.
#[derive(Debug, Clone)]
pub struct QuietReport {
    pub quiet: bool,
    pub waited_s: f64,
    pub checks: usize,
    /// Why the last check was not clean. Empty when `quiet`.
    pub reasons: Vec<String>,
}

/// Polls `check` until it reports a clean host or `max_wait` has passed.
///
/// Generic over its three effects so the loop is testable without a clock:
/// `check` returns `(clean, reasons)`, `sleep` waits `poll`, and `elapsed`
/// is the time spent so far. It always checks once, so a `max_wait` of zero
/// still reports the current state.
pub fn wait_until_quiet_with(
    max_wait: std::time::Duration,
    poll: std::time::Duration,
    mut check: impl FnMut() -> (bool, Vec<String>),
    mut sleep: impl FnMut(std::time::Duration),
    mut elapsed: impl FnMut() -> std::time::Duration,
) -> QuietReport {
    let mut checks = 0;
    loop {
        let (clean, reasons) = check();
        checks += 1;
        let waited = elapsed();
        if clean {
            return QuietReport {
                quiet: true,
                waited_s: waited.as_secs_f64(),
                checks,
                reasons: Vec::new(),
            };
        }
        if waited + poll > max_wait {
            return QuietReport {
                quiet: false,
                waited_s: waited.as_secs_f64(),
                checks,
                reasons,
            };
        }
        sleep(poll);
    }
}

/// Waits (up to `rules.quiet_wait_s`, and not past `deadline`) for a window
/// in which the host meets `rules`: the same test a run is graded by, applied
/// BEFORE the run so a pair is not spent on a host that is already busy.
pub fn wait_until_quiet(rules: &HostRules, deadline: Option<Instant>) -> QuietReport {
    let started = Instant::now();
    let mut max_wait = std::time::Duration::from_secs_f64(rules.quiet_wait_s.max(0.0));
    if let Some(d) = deadline {
        max_wait = max_wait.min(d.saturating_duration_since(started));
    }
    // One window is long enough for a sustained-CPU episode to be observed:
    // `sustained_s` of intervals plus the priming sample.
    let samples_needed = rules.sustained_s.ceil().max(1.0) as usize + 2;
    wait_until_quiet_with(
        max_wait,
        std::time::Duration::from_secs(10),
        || {
            let mut monitor = HostMonitor::new();
            let mut samples = Vec::with_capacity(samples_needed);
            for i in 0..samples_needed {
                if i > 0 {
                    std::thread::sleep(std::time::Duration::from_secs(1));
                }
                samples.push(monitor.sample(None));
            }
            eligibility(&samples, rules)
        },
        std::thread::sleep,
        || started.elapsed(),
    )
}

/// Free space for an unprivileged user on the volume holding `path` (or its
/// nearest existing ancestor, since the lab directory may not exist yet).
pub fn free_disk_bytes(path: &std::path::Path) -> Option<u64> {
    use std::os::unix::ffi::OsStrExt;
    let mut probe = Some(path);
    while let Some(p) = probe {
        if p.exists() {
            let c = std::ffi::CString::new(p.as_os_str().as_bytes()).ok()?;
            // SAFETY: `c` is a valid NUL-terminated path and `stats` is a
            // zeroed out-parameter that `statvfs` fills on success.
            let mut stats: libc::statvfs = unsafe { std::mem::zeroed() };
            let rc = unsafe { libc::statvfs(c.as_ptr(), &mut stats) };
            if rc != 0 {
                return None;
            }
            return Some((stats.f_bavail as u64).saturating_mul(stats.f_frsize as u64));
        }
        probe = p.parent();
    }
    None
}

/// The refusal message when `free_bytes` is under `min_gb` GiB, or `None`.
/// An unreadable free-space figure is not a refusal: the check exists to
/// stop a build or a run from filling the disk, and cannot say it would.
pub fn disk_shortfall(free_bytes: Option<u64>, min_gb: f64) -> Option<String> {
    let free = free_bytes?;
    if min_gb <= 0.0 {
        return None;
    }
    let free_gb = free as f64 / (1u64 << 30) as f64;
    (free_gb < min_gb).then(|| {
        format!(
            "only {free_gb:.1} GiB free on the lab's volume; need at least {min_gb:.0} GiB \
             (release builds are about 5 GB per worktree). Free space, or pass --min-free-gb 0"
        )
    })
}

/// Refuses to start when the lab's volume is nearly full.
pub fn check_free_disk(path: &std::path::Path, min_gb: f64) -> Result<(), String> {
    match disk_shortfall(free_disk_bytes(path), min_gb) {
        Some(msg) => Err(msg),
        None => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;
    use std::time::Duration;

    /// Drives `wait_until_quiet_with` on a fake clock: `busy_checks` checks
    /// report a dirty host, every later one a clean host. Returns the report
    /// and how many times `check` ran.
    fn wait_on_fake_clock(
        busy_checks: usize,
        max_wait_s: u64,
        poll_s: u64,
    ) -> (QuietReport, usize) {
        let clock = Cell::new(Duration::ZERO);
        let calls = Cell::new(0usize);
        let report = wait_until_quiet_with(
            Duration::from_secs(max_wait_s),
            Duration::from_secs(poll_s),
            || {
                calls.set(calls.get() + 1);
                if calls.get() <= busy_checks {
                    (false, vec!["mediaanalysisd over 90% for 5s or more".into()])
                } else {
                    (true, vec![])
                }
            },
            |d| clock.set(clock.get() + d),
            || clock.get(),
        );
        (report, calls.get())
    }

    #[test]
    fn a_quiet_host_returns_on_the_first_check() {
        let (r, calls) = wait_on_fake_clock(0, 300, 10);
        assert!(r.quiet);
        assert_eq!((calls, r.checks), (1, 1));
        assert_eq!(r.waited_s, 0.0);
        assert!(r.reasons.is_empty());
    }

    #[test]
    fn it_waits_through_busy_checks_until_the_host_goes_quiet() {
        let (r, calls) = wait_on_fake_clock(3, 300, 10);
        assert!(r.quiet, "{r:?}");
        assert_eq!((calls, r.checks), (4, 4));
        assert_eq!(r.waited_s, 30.0);
    }

    /// The point of the wait: it must give up, and say WHY, rather than
    /// loop forever or silently measure on a host that never settled.
    #[test]
    fn it_gives_up_at_max_wait_and_reports_the_last_reasons() {
        let (r, calls) = wait_on_fake_clock(usize::MAX, 25, 10);
        assert!(!r.quiet);
        // t=0 and t=10 sleep; at t=20 another poll would pass 25, so it stops.
        assert_eq!((calls, r.checks), (3, 3));
        assert_eq!(r.waited_s, 20.0);
        assert_eq!(r.reasons, vec!["mediaanalysisd over 90% for 5s or more"]);
    }

    #[test]
    fn a_zero_wait_still_checks_once() {
        let (r, calls) = wait_on_fake_clock(usize::MAX, 0, 10);
        assert!(!r.quiet);
        assert_eq!((calls, r.checks), (1, 1));
        assert!(!r.reasons.is_empty());
    }

    #[test]
    fn disk_shortfall_refuses_only_when_it_can_tell() {
        let gib = 1u64 << 30;
        let msg = disk_shortfall(Some(5 * gib), 10.0).expect("5 GiB is under 10");
        assert!(
            msg.contains("5.0 GiB") && msg.contains("--min-free-gb"),
            "{msg}"
        );
        assert_eq!(disk_shortfall(Some(20 * gib), 10.0), None);
        assert_eq!(
            disk_shortfall(Some(10 * gib), 10.0),
            None,
            "exactly enough passes"
        );
        // An unreadable figure is not a refusal, and 0 disables the check.
        assert_eq!(disk_shortfall(None, 10.0), None);
        assert_eq!(disk_shortfall(Some(1), 0.0), None);
    }

    #[test]
    fn free_disk_is_readable_for_a_path_that_does_not_exist_yet() {
        let here = free_disk_bytes(&std::env::temp_dir()).expect("temp dir has a volume");
        assert!(here > 0);
        let missing = std::env::temp_dir()
            .join("tslab-no-such-dir")
            .join("deeper");
        assert!(free_disk_bytes(&missing).is_some());
        assert!(check_free_disk(&std::env::temp_dir(), 0.0).is_ok());
    }

    /// Ledger records written before these two rules existed must still load.
    #[test]
    fn older_host_rules_records_get_the_new_defaults() {
        let old = r#"{"max_total_cpu_percent":150.0,"max_single_cpu_percent":90.0,
                      "sustained_s":5.0,"require_ac":true}"#;
        let rules: HostRules = serde_json::from_str(old).expect("old record parses");
        assert_eq!(rules.quiet_wait_s, 300.0);
        assert_eq!(rules.min_free_gb, 10.0);
    }

    fn quiet(t: f64) -> HostSample {
        HostSample {
            t_s: t,
            interval_s: Some(1.0),
            load1: 1.0,
            on_ac: Some(true),
            thermal_limited: Some(false),
            foreign_cpu_percent: 40.0,
            foreign_top: vec![ForeignProcess {
                pid: 7,
                cpu_percent: 35.0,
                command: "/System/WindowServer".into(),
            }],
            foreign_models: vec![],
            rustc_processes: 0,
            swap_used_mib: Some(1000.0),
            compressions: None,
            decompressions: None,
            error: None,
            contention: None,
        }
    }

    #[test]
    fn parses_compressor_counters() {
        let text = "Mach Virtual Memory Statistics: (page size of 16384 bytes)\n\
                    Pages stored in compressor:                  1029795.\n\
                    Decompressions:                           2259287100.\n\
                    Compressions:                             2502180604.\n";
        assert_eq!(
            super::parse_compressor_counters(text),
            (Some(2_502_180_604), Some(2_259_287_100))
        );
        assert_eq!(super::parse_compressor_counters("junk"), (None, None));
    }

    #[test]
    fn parses_ps_time_forms() {
        assert_eq!(cpu_seconds("1:02.50"), Some(62.5));
        assert_eq!(cpu_seconds("1:00:00.00"), Some(3600.0));
        assert_eq!(cpu_seconds("2-00:00:01.00"), Some(172801.0));
        assert_eq!(cpu_seconds("junk"), None);
        let rows = parse_ps("  12   1  12  3.5   0:01.20 /usr/bin/some thing\n");
        assert_eq!(rows[0].command, "/usr/bin/some thing");
        assert_eq!(rows[0].cpu_seconds, 1.2);
    }

    #[test]
    fn quiet_host_is_eligible() {
        let samples: Vec<_> = (0..10).map(|i| quiet(i as f64)).collect();
        assert_eq!(eligibility(&samples, &HostRules::default()), (true, vec![]));
    }

    #[test]
    fn each_dirty_rule_fires() {
        let rules = HostRules::default();
        let mut s = quiet(1.0);
        s.on_ac = Some(false);
        assert!(!eligibility(&[s], &rules).0, "battery accepted");
        let mut s = quiet(1.0);
        s.foreign_models = vec!["turbospark-check".into()];
        assert!(!eligibility(&[s], &rules).0, "model process accepted");
        let mut s = quiet(1.0);
        s.rustc_processes = 2;
        assert!(!eligibility(&[s], &rules).0, "compile accepted");
        let mut s = quiet(1.0);
        s.thermal_limited = None;
        assert!(!eligibility(&[s], &rules).0, "unreadable thermal accepted");
    }

    #[test]
    fn cpu_must_be_sustained() {
        let rules = HostRules::default();
        // One busy second is a blip.
        let mut blip: Vec<_> = (0..10).map(|i| quiet(i as f64)).collect();
        blip[4].foreign_cpu_percent = 400.0;
        assert!(eligibility(&blip, &rules).0);
        // Six busy seconds in a row is not.
        let mut busy: Vec<_> = (0..10).map(|i| quiet(i as f64)).collect();
        for s in &mut busy[2..8] {
            s.foreign_cpu_percent = 400.0;
        }
        assert!(!eligibility(&busy, &rules).0);
    }

    #[test]
    fn model_markers_match_binaries_and_test_targets() {
        assert!(is_model_process("/x/target/release/turbospark-check"));
        assert!(is_model_process(
            "/x/target/release/deps/int4_mma8_drift_probe-761bbb49"
        ));
        assert!(is_model_process("/x/deps/qwen36_quality_gate-abc"));
        assert!(!is_model_process("/System/Library/WindowServer"));
    }
}
