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
    pub error: Option<String>,
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

impl HostMonitor {
    pub fn new() -> Self {
        Self {
            started: Instant::now(),
            previous: HashMap::new(),
            previous_at: None,
        }
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
            error: None,
        };
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
}

impl Default for HostRules {
    fn default() -> Self {
        Self {
            max_total_cpu_percent: 150.0,
            max_single_cpu_percent: 90.0,
            sustained_s: 5.0,
            require_ac: true,
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

#[cfg(test)]
mod tests {
    use super::*;

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
            error: None,
        }
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
