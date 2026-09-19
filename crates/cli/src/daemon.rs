//! Background server daemon management: start, stop, restart, and status.

use std::fs::{self, OpenOptions};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::thread;
use std::time::Duration;

const API_KEY_ENV: &str = "TURBOSPARK_API_KEY";

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
    let file = private_open_options()
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

/// The value following `name` in an argument list, e.g. the install after
/// `--model`. First occurrence wins.
fn flag_value(args: &[String], name: &str) -> Option<String> {
    let mut i = 0;
    while i < args.len() {
        if args[i] == name {
            if let Some(value) = args.get(i + 1) {
                return Some(value.clone());
            }
        }
        i += 1;
    }
    None
}

/// A running daemon's own record of itself, read from `run/server.meta`
/// with the pid verified live.
pub struct RunningServer {
    pub port: u16,
    /// The `--model` argument the daemon was started with, as passed (an
    /// alias or a path, not resolved). `None` for the scripted-tokenizer mode.
    pub model: Option<String>,
}

/// The running daemon, or `None` when no pid file survives liveness (a dead
/// pid cleans up its own pid and meta files, matching [`get_running_pid`]).
pub fn running_server() -> Option<RunningServer> {
    // The call itself is the gate: a dead pid cleans up its own pid and
    // meta files and returns None.
    get_running_pid()?;
    let text = fs::read_to_string(meta_file()).ok()?;
    let value = serde_json::from_str::<serde_json::Value>(&text).ok()?;
    let port = value["port"].as_u64()? as u16;
    let args = value["args"]
        .as_array()?
        .iter()
        .filter_map(|a| a.as_str().map(str::to_string))
        .collect::<Vec<String>>();
    let model = flag_value(&args, "--model");
    Some(RunningServer { port, model })
}

/// The last lines of the server log, for an error message that names what
/// went wrong rather than a file to go read.
pub fn log_tail() -> String {
    fs::read_to_string(log_file())
        .unwrap_or_default()
        .lines()
        .rev()
        .take(10)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect::<Vec<_>>()
        .join("\n")
}

fn server_args_and_api_key(args: &[String]) -> (Vec<String>, Option<String>) {
    let mut server_args = Vec::with_capacity(args.len());
    let mut api_key = None;
    let mut index = 0;
    while index < args.len() {
        if args[index] == "--api-key" && args.get(index + 1).is_some_and(|value| !value.is_empty())
        {
            api_key = Some(args[index + 1].clone());
            index += 2;
        } else {
            server_args.push(args[index].clone());
            index += 1;
        }
    }
    (server_args, api_key)
}

#[cfg(unix)]
fn private_open_options() -> OpenOptions {
    use std::os::unix::fs::OpenOptionsExt;

    let mut options = OpenOptions::new();
    options.create(true).truncate(true).write(true).mode(0o600);
    options
}

#[cfg(not(unix))]
fn private_open_options() -> OpenOptions {
    let mut options = OpenOptions::new();
    options.create(true).truncate(true).write(true);
    options
}

fn write_private_file(path: PathBuf, contents: &[u8]) -> Result<(), std::io::Error> {
    use std::io::Write;

    let mut file = private_open_options().open(path)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        file.set_permissions(fs::Permissions::from_mode(0o600))?;
    }
    file.write_all(contents)
}

fn create_private_run_dir(path: &Path) -> Result<(), std::io::Error> {
    fs::create_dir_all(path)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(path, fs::Permissions::from_mode(0o700))?;
    }
    Ok(())
}

/// Start the server as a background daemon.
pub fn start(args: &[String]) -> Result<(), String> {
    let run = run_dir();
    create_private_run_dir(&run)
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
    let (server_args, api_key) = server_args_and_api_key(args);

    let mut cmd = Command::new(&server_bin);
    cmd.args(&server_args);
    if let Some(api_key) = api_key {
        cmd.env(API_KEY_ENV, api_key);
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
    write_private_file(pid_file(), format!("{pid}\n").as_bytes())
        .map_err(|e| format!("writing pid file: {e}"))?;

    let meta = serde_json::json!({
        "pid": pid,
        "port": port,
        "args": server_args,
    });
    let _ = write_private_file(meta_file(), meta.to_string().as_bytes());

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
    create_private_run_dir(&run)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn api_keys_leave_server_arguments_and_last_value_moves_to_environment() {
        let args = [
            "--model",
            "gemma4",
            "--api-key",
            "first-secret",
            "--port",
            "9000",
            "--api-key",
            "final-secret",
        ]
        .map(String::from);

        let (server_args, api_key) = server_args_and_api_key(&args);

        assert_eq!(
            server_args,
            ["--model", "gemma4", "--port", "9000"].map(String::from)
        );
        assert_eq!(api_key.as_deref(), Some("final-secret"));
    }

    #[test]
    fn invalid_api_key_arguments_remain_for_server_validation() {
        for args in [
            vec![
                "--model".to_string(),
                "gemma4".to_string(),
                "--api-key".to_string(),
            ],
            vec![
                "--model".to_string(),
                "gemma4".to_string(),
                "--api-key".to_string(),
                String::new(),
            ],
        ] {
            let (server_args, api_key) = server_args_and_api_key(&args);
            assert_eq!(server_args, args);
            assert_eq!(api_key, None);
        }
    }

    #[cfg(unix)]
    #[test]
    fn private_state_files_ignore_a_permissive_umask() {
        use std::os::unix::fs::PermissionsExt;

        let path = std::env::temp_dir().join(format!(
            "turbospark-private-state-{}-{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        let _ = fs::remove_file(&path);
        write_private_file(path.clone(), b"secret").unwrap();

        assert_eq!(
            fs::metadata(&path).unwrap().permissions().mode() & 0o777,
            0o600
        );
        fs::remove_file(path).unwrap();
    }
}
