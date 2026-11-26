//! The server under test: an already running loopback URL or a spawned
//! binary, plus what can be read about its process and the host.

use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use serde::Serialize;
use serde_json::{json, Value};

use super::http::{health_ok, json_request, parse_loopback_url, Server};
use crate::args::Args;

/// The line `turbospark-server` prints once its socket is bound.
const LISTENING: &str = "listening on ";

pub struct TargetSpec {
    url: Option<String>,
    server_bin: Option<PathBuf>,
    model: Option<String>,
    server_args: Vec<String>,
    env: Vec<(String, String)>,
    model_id: Option<String>,
    pid: Option<u32>,
    start_timeout: Duration,
}

impl TargetSpec {
    /// `--server URL [--pid P]` or `--server-bin PATH --model M`.
    pub fn take(args: &mut Args) -> Result<Self, String> {
        // First, so a server flag passed through is never read as ours.
        let server_args = args.take_all("--server-arg")?;
        let env = args
            .take_all("--env")?
            .iter()
            .map(|e| crate::args::parse_env(e))
            .collect::<Result<Vec<_>, _>>()?;
        let spec = Self {
            url: args.take("--server")?,
            server_bin: args.take("--server-bin")?.map(PathBuf::from),
            model: args.take("--model")?,
            server_args,
            env,
            model_id: args.take("--model-id")?,
            pid: match args.take("--pid")? {
                None => None,
                Some(p) => Some(
                    p.parse()
                        .ok()
                        .filter(|&p: &u32| p > 0 && p <= i32::MAX as u32)
                        .ok_or_else(|| format!("--pid {p:?}"))?,
                ),
            },
            start_timeout: Duration::from_secs(args.parsed("--start-timeout-s", 900)?),
        };
        match (&spec.url, &spec.server_bin) {
            (Some(_), Some(_)) => Err("give --server or --server-bin, not both".into()),
            (None, None) => Err("--server URL or --server-bin PATH --model M is required".into()),
            (None, Some(_)) if spec.model.is_none() => Err("--server-bin needs --model".into()),
            (None, Some(_)) if spec.pid.is_some() => {
                Err("--pid belongs to --server; a spawned server is its own pid".into())
            }
            (Some(_), None)
                if spec.model.is_some() || !spec.server_args.is_empty() || !spec.env.is_empty() =>
            {
                Err("--model, --server-arg and --env only apply to --server-bin".into())
            }
            _ => Ok(spec),
        }
    }

    pub fn open(self, log_dir: &Path) -> Result<Target, String> {
        let (server, child, log) = match (&self.url, &self.server_bin) {
            (Some(url), _) => (parse_loopback_url(url)?, None, None),
            (None, Some(bin)) => {
                std::fs::create_dir_all(log_dir)
                    .map_err(|e| format!("{}: {e}", log_dir.display()))?;
                let log = log_dir.join(format!("server-{}.log", crate::ledger::new_id()));
                let (server, child) = self.spawn(bin, &log)?;
                (server, Some(child), Some(log))
            }
            (None, None) => return Err("no server".into()),
        };
        let pid = child.as_ref().map(|c| c.0.id()).or(self.pid);
        if !health_ok(&server) {
            return Err(format!("{}/health did not answer 200", server.url()));
        }
        let model_id = match self.model_id {
            Some(id) => id,
            None => {
                let (status, v) =
                    json_request(&server, "GET", "/v1/models", None, Duration::from_secs(30))?;
                v["data"][0]["id"]
                    .as_str()
                    .map(str::to_string)
                    .ok_or_else(|| {
                        format!("GET /v1/models ({status}) named no model; pass --model-id")
                    })?
            }
        };
        let record = json!({
            "url": server.url(),
            "spawned": child.is_some(),
            "model_id": model_id,
            "model": self.model,
            "server_bin": self.server_bin,
            "server_bin_sha256": self.server_bin.as_deref().and_then(sha256),
            "server_args": self.server_args,
            "server_env": self.env.into_iter().collect::<std::collections::BTreeMap<_, _>>(),
            "server_log": log,
            "pid": pid,
        });
        Ok(Target {
            server,
            model_id,
            pid,
            record,
            child,
        })
    }

    /// A fresh server on an ephemeral loopback port, with a cleared
    /// environment as every lab worker gets.
    fn spawn(&self, bin: &Path, log: &Path) -> Result<(Server, Owned), String> {
        let file = std::fs::File::create(log).map_err(|e| format!("{}: {e}", log.display()))?;
        let mut cmd = Command::new(bin);
        cmd.env_clear();
        for keep in ["HOME", "TMPDIR", "TURBOSPARK_HOME"] {
            if let Some(v) = std::env::var_os(keep) {
                cmd.env(keep, v);
            }
        }
        cmd.envs(self.env.iter().cloned())
            .args(["--model", self.model.as_deref().unwrap_or_default()])
            .args(["--port", "0"])
            .args(&self.server_args)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::from(file));
        let mut child = Owned(cmd.spawn().map_err(|e| format!("{}: {e}", bin.display()))?);
        let deadline = Instant::now() + self.start_timeout;
        loop {
            let text = std::fs::read_to_string(log).unwrap_or_default();
            // Only whole lines: a half-written port must not be parsed.
            let url = text
                .split_inclusive('\n')
                .filter(|l| l.ends_with('\n'))
                .find_map(|l| l.split_once(LISTENING).map(|(_, u)| u.trim().to_string()));
            if let Some(url) = url {
                return Ok((parse_loopback_url(&url)?, child));
            }
            if let Ok(Some(status)) = child.0.try_wait() {
                return Err(format!(
                    "server exited ({status}) before listening; see {}",
                    log.display()
                ));
            }
            if Instant::now() >= deadline {
                return Err(format!(
                    "server not listening after {:?}; see {}",
                    self.start_timeout,
                    log.display()
                ));
            }
            std::thread::sleep(Duration::from_millis(100));
        }
    }
}

/// Killed and reaped on drop, so no error path leaves a model resident.
pub struct Owned(Child);

impl Drop for Owned {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

pub struct Target {
    pub server: Server,
    pub model_id: String,
    pub pid: Option<u32>,
    pub record: Value,
    child: Option<Owned>,
}

impl Target {
    /// `Some` once a spawned server has exited.
    pub fn exited(&mut self) -> Option<String> {
        let child = self.child.as_mut()?;
        match child.0.try_wait() {
            Ok(Some(status)) => Some(status.to_string()),
            _ => None,
        }
    }
}

/// One reading of another process, same-user only.
#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct ProcStats {
    /// `phys_footprint`, the counter every published memory figure uses
    /// (`crates/bench/src/memory.rs` reads it for the current process).
    pub phys_footprint: u64,
    pub lifetime_max_footprint: u64,
    pub threads: Option<u32>,
    pub fds: Option<u32>,
}

#[cfg(target_os = "macos")]
pub fn proc_stats(pid: u32) -> Option<ProcStats> {
    let pid = i32::try_from(pid).ok()?;
    // SAFETY: both structs are plain C data for which zero is valid.
    let mut usage: libc::rusage_info_v4 = unsafe { std::mem::zeroed() };
    // SAFETY: the flavor matches the struct the pointer addresses.
    let rc = unsafe {
        libc::proc_pid_rusage(
            pid,
            libc::RUSAGE_INFO_V4,
            (&mut usage as *mut libc::rusage_info_v4).cast(),
        )
    };
    if rc != 0 {
        return None;
    }
    // SAFETY: as above.
    let mut task: libc::proc_taskinfo = unsafe { std::mem::zeroed() };
    let size = std::mem::size_of::<libc::proc_taskinfo>() as i32;
    // SAFETY: `size` is the size of the buffer passed.
    let n = unsafe {
        libc::proc_pidinfo(
            pid,
            libc::PROC_PIDTASKINFO,
            0,
            (&mut task as *mut libc::proc_taskinfo).cast(),
            size,
        )
    };
    let threads = (n == size).then_some(task.pti_threadnum as u32);
    // The size-only query over-reports, so list into a buffer and count.
    let each = std::mem::size_of::<libc::proc_fdinfo>();
    let mut fds = vec![
        libc::proc_fdinfo {
            proc_fd: 0,
            proc_fdtype: 0
        };
        16 * 1024
    ];
    // SAFETY: the byte length passed is the buffer's own.
    let n = unsafe {
        libc::proc_pidinfo(
            pid,
            libc::PROC_PIDLISTFDS,
            0,
            fds.as_mut_ptr().cast(),
            (fds.len() * each) as i32,
        )
    };
    Some(ProcStats {
        phys_footprint: usage.ri_phys_footprint,
        lifetime_max_footprint: usage.ri_lifetime_max_phys_footprint,
        threads,
        fds: (n > 0).then(|| (n as usize / each) as u32),
    })
}

#[cfg(not(target_os = "macos"))]
pub fn proc_stats(_pid: u32) -> Option<ProcStats> {
    None
}

#[cfg(target_os = "macos")]
fn sha256(path: &Path) -> Option<String> {
    crate::provenance::sha256_file(path)
}

#[cfg(not(target_os = "macos"))]
fn sha256(_: &Path) -> Option<String> {
    None
}

pub fn provenance() -> Value {
    #[cfg(target_os = "macos")]
    return json!(crate::provenance::collect());
    #[cfg(not(target_os = "macos"))]
    return Value::Null;
}

/// Host samples taken beside a run, judged by the lab's usual rules.
#[cfg(target_os = "macos")]
pub struct HostWatch {
    stop: std::sync::mpsc::Sender<()>,
    thread: std::thread::JoinHandle<Vec<crate::host::HostSample>>,
    allow_battery: bool,
}

#[cfg(target_os = "macos")]
impl HostWatch {
    pub fn start(server_pid: Option<u32>, allow_battery: bool) -> Self {
        let (stop, stopped) = std::sync::mpsc::channel();
        let thread = std::thread::spawn(move || {
            let mut monitor = crate::host::HostMonitor::new();
            let mut samples = vec![monitor.sample(server_pid)];
            loop {
                let wait = stopped.recv_timeout(Duration::from_secs(2));
                samples.push(monitor.sample(server_pid));
                if wait != Err(std::sync::mpsc::RecvTimeoutError::Timeout) {
                    return samples;
                }
            }
        });
        Self {
            stop,
            thread,
            allow_battery,
        }
    }

    /// `(eligible, reasons, summary)`.
    pub fn finish(self) -> (bool, Vec<String>, Value) {
        let _ = self.stop.send(());
        let samples = self.thread.join().unwrap_or_default();
        let rules = crate::host::HostRules {
            require_ac: !self.allow_battery,
            ..Default::default()
        };
        let (eligible, reasons) = crate::host::eligibility(&samples, &rules);
        let summary = json!({
            "samples": samples.len(),
            "first": samples.first(),
            "last": samples.last(),
        });
        (eligible, reasons, summary)
    }
}

#[cfg(not(target_os = "macos"))]
pub struct HostWatch;

#[cfg(not(target_os = "macos"))]
impl HostWatch {
    pub fn start(_: Option<u32>, _: bool) -> Self {
        Self
    }

    pub fn finish(self) -> (bool, Vec<String>, Value) {
        (
            false,
            vec!["host state is not observed off macOS".into()],
            Value::Null,
        )
    }
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::*;

    #[test]
    fn proc_stats_reads_this_process() {
        let held: Vec<_> = (0..8)
            .map(|_| std::fs::File::open("/dev/null").unwrap())
            .collect();
        let s = proc_stats(std::process::id()).expect("own pid is readable");
        assert!(s.phys_footprint > 0);
        assert!(s.lifetime_max_footprint >= s.phys_footprint / 2);
        assert!(s.threads.unwrap() >= 1);
        assert!(s.fds.unwrap() >= held.len() as u32);
        assert!(proc_stats(u32::MAX).is_none());
    }
}
