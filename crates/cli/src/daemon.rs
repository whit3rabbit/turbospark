//! Background server daemon management: start, stop, restart, and status.

use std::fs::{self, OpenOptions};
use std::path::PathBuf;
use std::process::Command;
use std::thread;
use std::time::Duration;

/// Locate the `turbospark-server` binary: adjacent to the current executable first,
/// then falling back to PATH.
pub fn find_server_binary() -> PathBuf {
    if let Ok(mut exe) = std::env::current_exe() {
        exe.pop();
        let candidate = exe.join("turbospark-server");
        if candidate.is_file() {
            return candidate;
        }
    }
    PathBuf::from("turbospark-server")
}

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

fn daemon_lock_path() -> PathBuf {
    run_dir().join("daemon.lock")
}

/// Acquires an exclusive OS-level lock over the daemon's run directory,
/// blocking until any other `start`/`stop` holding it releases. Held for
/// the whole check-then-spawn-then-write sequence, which closes the race
/// where two near-simultaneous `start` calls both observe "not running"
/// and both spawn a server: the second call blocks here, and once it gets
/// the lock, `get_running_pid()` sees the first call's now-running PID and
/// does nothing. Released automatically when the returned file is
/// dropped -- `flock` is tied to the open file description, not the path,
/// so closing the handle is enough.
///
/// # Safety
/// `libc::flock` is safe to call with a valid, open file descriptor.
#[cfg(unix)]
fn acquire_daemon_lock() -> Result<fs::File, String> {
    let path = daemon_lock_path();
    let file = OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(&path)
        .map_err(|e| format!("opening daemon lock {}: {e}", path.display()))?;
    let rc = unsafe {
        use std::os::unix::io::AsRawFd;
        libc::flock(file.as_raw_fd(), libc::LOCK_EX)
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

fn meta_file() -> PathBuf {
    run_dir().join("server.meta")
}

pub fn log_file() -> PathBuf {
    logs_dir().join("server.log")
}

/// Check if a process with the given PID is alive.
#[cfg(unix)]
pub fn is_pid_alive(pid: i32) -> bool {
    if pid <= 0 {
        return false;
    }
    unsafe { libc::kill(pid, 0) == 0 }
}

#[cfg(not(unix))]
pub fn is_pid_alive(_pid: i32) -> bool {
    false
}

/// Read the PID file, verifying whether the recorded process is alive.
pub fn get_running_pid() -> Option<i32> {
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

/// Start the server as a background daemon.
pub fn start(args: &[String]) -> Result<(), String> {
    let run = run_dir();
    fs::create_dir_all(&run)
        .map_err(|e| format!("creating run directory {}: {e}", run.display()))?;
    // Held across the whole check-then-spawn-then-write sequence below; see
    // `acquire_daemon_lock`'s own doc for the race this closes.
    let _lock = acquire_daemon_lock()?;

    if let Some(pid) = get_running_pid() {
        println!("turbospark server is already running (PID {pid}).");
        return Ok(());
    }

    let logs = logs_dir();
    fs::create_dir_all(&logs)
        .map_err(|e| format!("creating logs directory {}: {e}", logs.display()))?;

    let log_path = log_file();
    let log_handle = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)
        .map_err(|e| format!("opening log file {}: {e}", log_path.display()))?;

    let server_bin = find_server_binary();
    let port = extract_port(args);

    let mut cmd = Command::new(&server_bin);
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

    // Give process a moment to verify it didn't abort on startup.
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

    println!("turbospark server started in background (PID {pid})");
    println!("  Endpoint: http://127.0.0.1:{port}/v1");
    println!("  Logs:     {}", log_path.display());
    Ok(())
}

/// Stop the background daemon if running.
pub fn stop() -> Result<(), String> {
    let run = run_dir();
    fs::create_dir_all(&run)
        .map_err(|e| format!("creating run directory {}: {e}", run.display()))?;
    // Same lock `start` holds, so a stop cannot land between a concurrent
    // start's own check and its pid-file write.
    let _lock = acquire_daemon_lock()?;

    let pid = match get_running_pid() {
        Some(p) => p,
        None => {
            println!("turbospark server is not running.");
            return Ok(());
        }
    };

    #[cfg(unix)]
    {
        unsafe {
            libc::kill(pid, libc::SIGTERM);
        }
        for _ in 0..30 {
            thread::sleep(Duration::from_millis(100));
            if !is_pid_alive(pid) {
                break;
            }
        }
        if is_pid_alive(pid) {
            unsafe {
                libc::kill(pid, libc::SIGKILL);
            }
        }
    }

    let _ = fs::remove_file(pid_file());
    let _ = fs::remove_file(meta_file());
    println!("turbospark server stopped (PID {pid}).");
    Ok(())
}

/// Restart the background daemon.
pub fn restart(args: &[String]) -> Result<(), String> {
    stop()?;
    thread::sleep(Duration::from_millis(200));
    start(args)
}

/// Print the status of the server daemon.
pub fn status() -> Result<(), String> {
    let pid = match get_running_pid() {
        Some(p) => p,
        None => {
            println!("Status: STOPPED");
            println!("Use 'turbospark start' or 'turbospark serve' to launch.");
            return Ok(());
        }
    };

    let port = if let Ok(text) = fs::read_to_string(meta_file()) {
        serde_json::from_str::<serde_json::Value>(&text)
            .ok()
            .and_then(|v| v["port"].as_u64())
            .map(|p| p as u16)
            .unwrap_or(8080)
    } else {
        8080
    };

    println!("Status:   RUNNING");
    println!("PID:      {pid}");
    println!("Port:     {port}");
    println!("Endpoint: http://127.0.0.1:{port}/v1");
    println!("Log file: {}", log_file().display());
    Ok(())
}
