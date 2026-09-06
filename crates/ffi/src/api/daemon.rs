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
        unsafe { kill(pid, 0) == 0 }
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

fn start_daemon_internal(args: &[String]) -> Result<(), String> {
    if get_running_pid().is_some() {
        return Ok(());
    }

    let run = run_dir();
    let logs = logs_dir();
    fs::create_dir_all(&run)
        .map_err(|e| format!("creating run directory {}: {e}", run.display()))?;
    fs::create_dir_all(&logs)
        .map_err(|e| format!("creating logs directory {}: {e}", logs.display()))?;

    let log_path = log_file();
    let log_handle = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)
        .map_err(|e| format!("opening log file {}: {e}", log_path.display()))?;

    let server_bin = find_server_binary();
    let port = extract_port(args);

    let mut cmd = std::process::Command::new(&server_bin);
    cmd.args(args);
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
    guard_result(|| {
        stop_daemon_internal().map_err(|e| (abi::TS_ERR_OPEN, e))
    })
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
                serde_json::from_str(&s).map_err(|e| (abi::TS_ERR_JSON, e.to_string()))?
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
                serde_json::from_str(&s).map_err(|e| (abi::TS_ERR_JSON, e.to_string()))?
            }
        } else {
            Vec::new()
        };
        start_daemon_internal(&args).map_err(|e| (abi::TS_ERR_OPEN, e))
    })
}
