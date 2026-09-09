//! Managed background daemon inspection and control C ABI entry points.

use std::fs;
use std::os::raw::{c_char, c_int};
use std::path::PathBuf;
use std::thread;
use std::time::Duration;

use crate::abi::{self, guard_result};
use crate::strings;

fn root_dir() -> PathBuf {
    catalog::default_root().unwrap_or_else(|| PathBuf::from(".turbospark"))
}

fn run_dir() -> PathBuf {
    root_dir().join("run")
}

fn logs_dir() -> PathBuf {
    root_dir().join("logs")
}

fn pid_file() -> PathBuf {
    run_dir().join("server.pid")
}

fn meta_file() -> PathBuf {
    run_dir().join("server.meta")
}

fn log_file() -> PathBuf {
    logs_dir().join("server.log")
}

fn is_pid_alive(pid: i32) -> bool {
    if pid <= 0 {
        return false;
    }
    #[cfg(unix)]
    {
        extern "C" {
            fn kill(pid: i32, sig: i32) -> i32;
        }
        // A process owned by another user answers EPERM rather than 0, which
        // means "alive but not mine to signal", not "dead". Treating EPERM as
        // dead deletes the pid/meta files out from under a live daemon. EPERM
        // is 1 on every POSIX target this crate builds for (Linux and the BSD
        // family, macOS included), so this is read from the OS error rather
        // than pulled in from `libc` for one constant.
        const EPERM: i32 = 1;
        unsafe {
            kill(pid, 0) == 0 || std::io::Error::last_os_error().raw_os_error() == Some(EPERM)
        }
    }
    #[cfg(not(unix))]
    {
        let _ = pid;
        false
    }
}

fn get_running_pid() -> Option<i32> {
    let path = pid_file();
    let text = fs::read_to_string(&path).ok()?;
    let pid = text.trim().parse::<i32>().ok()?;
    if is_pid_alive(pid) {
        Some(pid)
    } else {
        let _ = fs::remove_file(&path);
        let _ = fs::remove_file(meta_file());
        None
    }
}

fn daemon_lock_path() -> PathBuf {
    run_dir().join("daemon.lock")
}

/// Acquires an exclusive OS-level lock over the daemon's run directory,
/// blocking until any other start/stop holding it releases. Held for the
/// whole check-then-spawn-then-write sequence, which closes the race where
/// two near-simultaneous starts both observe "not running" and both spawn
/// a server -- the second call blocks here, and once it gets the lock,
/// `get_running_pid()` sees the first call's now-running PID and does
/// nothing. Released automatically when the returned file is dropped
/// (`flock` is tied to the open file description, not the path).
#[cfg(unix)]
fn acquire_daemon_lock() -> Result<fs::File, String> {
    extern "C" {
        fn flock(fd: i32, operation: i32) -> i32;
    }
    const LOCK_EX: i32 = 2;

    let path = daemon_lock_path();
    let file = fs::OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(&path)
        .map_err(|e| format!("opening daemon lock {}: {e}", path.display()))?;
    let rc = unsafe {
        use std::os::unix::io::AsRawFd;
        flock(file.as_raw_fd(), LOCK_EX)
    };
    if rc != 0 {
        return Err(format!(
            "acquiring daemon lock {}: {}",
            path.display(),
            std::io::Error::last_os_error()
        ));
    }
    Ok(file)
}

#[cfg(not(unix))]
fn acquire_daemon_lock() -> Result<(), String> {
    Ok(())
}

/// Inspects whether a background turbospark server daemon is running.
/// Writes a JSON object to `*out` (free with `ts_string_free`):
///   `{"running": true, "pid": 1234, "port": 8080, "endpoint": "http://127.0.0.1:8080/v1", "logPath": "..."}`
/// or `{"running": false}`.
#[no_mangle]
pub unsafe extern "C" fn ts_daemon_status_json(out: *mut *mut c_char) -> c_int {
    guard_result(|| {
        if out.is_null() {
            return Err((
                abi::TS_ERR_INVALID_ARGUMENT,
                "output pointer must not be null".to_string(),
            ));
        }

        let json = if let Some(pid) = get_running_pid() {
            let port = if let Ok(text) = fs::read_to_string(meta_file()) {
                serde_json::from_str::<serde_json::Value>(&text)
                    .ok()
                    .and_then(|v| v.get("port").and_then(|p| p.as_u64()))
                    .map(|p| p as u16)
                    .unwrap_or(8080)
            } else {
                8080
            };
            let endpoint = format!("http://127.0.0.1:{port}/v1");
            let log_path = log_file().to_string_lossy().into_owned();
            serde_json::json!({
                "running": true,
                "pid": pid,
                "port": port,
                "endpoint": endpoint,
                "logPath": log_path,
            })
        } else {
            serde_json::json!({
                "running": false,
            })
        };

        let text = serde_json::to_string(&json).map_err(|e| (abi::TS_ERR_JSON, e.to_string()))?;
        strings::emit(&text, out).map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))
    })
}

fn find_server_binary() -> PathBuf {
    if let Ok(mut exe) = std::env::current_exe() {
        exe.pop();
        let candidate = exe.join("turbospark-server");
        if candidate.is_file() {
            return candidate;
        }
    }
    if let Ok(path_var) = std::env::var("PATH") {
        for dir in std::env::split_paths(&path_var) {
            let candidate = dir.join("turbospark-server");
            if candidate.is_file() {
                return candidate;
            }
        }
    }
    if let Some(home) = std::env::var_os("HOME") {
        let candidate = PathBuf::from(home).join(".cargo/bin/turbospark-server");
        if candidate.is_file() {
            return candidate;
        }
    }
    PathBuf::from("turbospark-server")
}

fn extract_port(args: &[String]) -> u16 {
    let mut i = 0;
    while i < args.len() {
        if args[i] == "--port" {
            if let Some(val) = args.get(i + 1) {
                if let Ok(p) = val.parse::<u16>() {
                    return p;
                }
            }
        }
        i += 1;
    }
    8080
}

/// Splits `--api-key <value>` out of `args`, returning the redacted args
/// and the last value the flag carried.
///
/// **THE KEY MUST NOT RIDE THE CHILD'S COMMAND LINE.** A `turbospark-server`
/// started with `--api-key KEY` puts that key in `ps` for the daemon's
/// whole life and, one block below, in the meta file this function writes
/// next to the pid -- the two surfaces `crates/server`'s own env fallback
/// exists to keep a key off. So the pair is lifted out here and reattached
/// as the child's `TURBOSPARK_API_KEY`, which the server's arg parser
/// resolves when the flag is absent (`crates/server/src/main.rs`). The
/// port extractor runs on the REDACTED args, so a key whose value is
/// literally `--port` cannot redirect it.
fn extract_api_key(args: &[String]) -> (Vec<String>, Option<String>) {
    let mut redacted = Vec::with_capacity(args.len());
    let mut api_key = None;
    let mut i = 0;
    while i < args.len() {
        if args[i] == "--api-key" {
            match args.get(i + 1) {
                Some(value) => {
                    api_key = Some(value.clone());
                    i += 2;
                    continue;
                }
                // A valueless flag cannot be honoured and cannot stay: left
                // in place it would make the child eat the next flag as the
                // key. Dropped, with no key set.
                None => {
                    i += 1;
                    continue;
                }
            }
        }
        redacted.push(args[i].clone());
        i += 1;
    }
    (redacted, api_key)
}

fn start_daemon_internal(args: &[String]) -> Result<(), String> {
    let run = run_dir();
    fs::create_dir_all(&run)
        .map_err(|e| format!("creating run directory {}: {e}", run.display()))?;
    // Held across the whole check-then-spawn-then-write sequence below; see
    // `acquire_daemon_lock`'s own doc for the race this closes.
    let _lock = acquire_daemon_lock()?;

    if get_running_pid().is_some() {
        return Ok(());
    }

    let logs = logs_dir();
    fs::create_dir_all(&logs)
        .map_err(|e| format!("creating logs directory {}: {e}", logs.display()))?;

    let log_path = log_file();
    let log_handle = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)
        .map_err(|e| format!("opening log file {}: {e}", log_path.display()))?;

    let server_bin = find_server_binary();
    let (args, api_key) = extract_api_key(args);
    let port = extract_port(&args);

    let mut cmd = std::process::Command::new(&server_bin);
    cmd.args(&args);
    if let Some(key) = api_key {
        cmd.env("TURBOSPARK_API_KEY", key);
    }
    cmd.stdout(log_handle.try_clone().map_err(|e| e.to_string())?);
    cmd.stderr(log_handle);

    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        cmd.process_group(0);
    }

    let child = cmd
        .spawn()
        .map_err(|e| format!("failed to launch {}: {e}", server_bin.display()))?;

    let pid = child.id() as i32;
    fs::write(pid_file(), format!("{pid}\n")).map_err(|e| format!("writing pid file: {e}"))?;

    let meta = serde_json::json!({
        "pid": pid,
        "port": port,
        "args": args,
    });
    let _ = fs::write(meta_file(), meta.to_string());

    thread::sleep(Duration::from_millis(400));
    if !is_pid_alive(pid) {
        let _ = fs::remove_file(pid_file());
        let _ = fs::remove_file(meta_file());
        let snippet = fs::read_to_string(&log_path)
            .unwrap_or_default()
            .lines()
            .rev()
            .take(10)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect::<Vec<_>>()
            .join("\n");
        return Err(format!(
            "server process exited immediately with error.\nLast log output:\n{snippet}"
        ));
    }
    Ok(())
}

fn stop_daemon_internal() -> Result<(), String> {
    let run = run_dir();
    fs::create_dir_all(&run)
        .map_err(|e| format!("creating run directory {}: {e}", run.display()))?;
    // Same lock `start_daemon_internal` holds, so a stop cannot land between
    // a concurrent start's own check and its pid-file write.
    let _lock = acquire_daemon_lock()?;

    let Some(pid) = get_running_pid() else {
        return Ok(());
    };

    #[cfg(unix)]
    {
        extern "C" {
            fn kill(pid: i32, sig: i32) -> i32;
        }
        unsafe {
            kill(pid, 15); // SIGTERM
        }
        for _ in 0..30 {
            thread::sleep(Duration::from_millis(100));
            if !is_pid_alive(pid) {
                break;
            }
        }
        if is_pid_alive(pid) {
            unsafe {
                kill(pid, 9); // SIGKILL
            }
        }
    }

    let _ = fs::remove_file(pid_file());
    let _ = fs::remove_file(meta_file());
    Ok(())
}

/// Stops the background turbospark server daemon if running.
#[no_mangle]
pub unsafe extern "C" fn ts_daemon_stop() -> c_int {
    guard_result(|| stop_daemon_internal().map_err(|e| (abi::TS_ERR_OPEN, e)))
}

/// Starts the turbospark background server daemon with optional arguments.
/// `args_json` is a JSON array of string arguments: `["--port", "8080"]`, or NULL/empty.
#[no_mangle]
pub unsafe extern "C" fn ts_daemon_start(args_json: *const c_char) -> c_int {
    guard_result(|| {
        let args: Vec<String> = if !args_json.is_null() {
            let s = strings::optional(args_json, "argsJson")
                .map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))?
                .unwrap_or_default();
            if s.trim().is_empty() {
                Vec::new()
            } else {
                serde_json::from_str(s).map_err(|e| (abi::TS_ERR_JSON, e.to_string()))?
            }
        } else {
            Vec::new()
        };
        start_daemon_internal(&args).map_err(|e| (abi::TS_ERR_OPEN, e))
    })
}

/// Restarts the background turbospark server daemon with optional arguments.
#[no_mangle]
pub unsafe extern "C" fn ts_daemon_restart(args_json: *const c_char) -> c_int {
    guard_result(|| {
        let _ = stop_daemon_internal();
        thread::sleep(Duration::from_millis(200));
        let args: Vec<String> = if !args_json.is_null() {
            let s = strings::optional(args_json, "argsJson")
                .map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))?
                .unwrap_or_default();
            if s.trim().is_empty() {
                Vec::new()
            } else {
                serde_json::from_str(s).map_err(|e| (abi::TS_ERR_JSON, e.to_string()))?
            }
        } else {
            Vec::new()
        };
        start_daemon_internal(&args).map_err(|e| (abi::TS_ERR_OPEN, e))
    })
}

#[cfg(test)]
mod api_key_redaction_tests {
    use super::extract_api_key;

    fn args(values: &[&str]) -> Vec<String> {
        values.iter().map(|v| (*v).to_string()).collect()
    }

    /// The pair leaves the argv the child is handed and the value is
    /// returned for the env attachment; everything around it is untouched.
    #[test]
    fn the_pair_is_lifted_out_whole() {
        let (redacted, key) = extract_api_key(&args(&[
            "--port",
            "8080",
            "--api-key",
            "sk-secret",
            "--system",
            "be brief",
        ]));
        assert_eq!(key.as_deref(), Some("sk-secret"));
        assert_eq!(redacted, args(&["--port", "8080", "--system", "be brief"]));
    }

    #[test]
    fn no_flag_leaves_args_alone_and_no_key() {
        let original = args(&["--port", "9000"]);
        let (redacted, key) = extract_api_key(&original);
        assert_eq!(redacted, original);
        assert_eq!(key, None);
    }

    /// Last occurrence wins, matching a flag parser, and BOTH pairs are
    /// redacted -- a first value left behind in argv would still be a key
    /// in `ps`, just a stale one.
    #[test]
    fn every_occurrence_is_redacted_and_the_last_value_wins() {
        let (redacted, key) = extract_api_key(&args(&[
            "--api-key",
            "sk-first",
            "--guardrails",
            "on",
            "--api-key",
            "sk-second",
        ]));
        assert_eq!(key.as_deref(), Some("sk-second"));
        assert_eq!(redacted, args(&["--guardrails", "on"]));
    }

    /// A valueless flag cannot be honoured and must not stay: left in place
    /// the child's parser would eat the next flag as the key.
    #[test]
    fn a_dangling_flag_is_dropped_without_setting_a_key() {
        let (redacted, key) = extract_api_key(&args(&["--port", "8080", "--api-key"]));
        assert_eq!(key, None);
        assert_eq!(redacted, args(&["--port", "8080"]));
    }

    /// `extract_port` runs on the REDACTED args, so a key whose value is
    /// the word `--port` cannot set the daemon's port.
    #[test]
    fn a_key_value_cannot_smuggle_a_port_flag() {
        let (redacted, key) = extract_api_key(&args(&["--api-key", "--port", "--port", "9001"]));
        assert_eq!(key.as_deref(), Some("--port"));
        assert_eq!(redacted, args(&["--port", "9001"]));
    }
}
