//! `ts_daemon_stop` and `ts_daemon_status_json` read a pid from
//! `<root>/run/server.pid`. A daemon that crashed, or a reboot, leaves that
//! file behind, and the OS later hands the pid to an unrelated process. A
//! liveness probe alone then reports that process as the daemon, and stop
//! sends it SIGTERM and SIGKILL. These tests point the pid file at real
//! child processes and check that only one whose executable is the server
//! binary is signalled.
//!
//! The store root is process-wide (`ts_store_root_set`), so every test holds
//! one lock for its whole run.
#![cfg(unix)]

use std::ffi::{CStr, CString};
use std::fs;
use std::path::PathBuf;
use std::process::{Child, Command};
use std::ptr;
use std::sync::{Mutex, MutexGuard};
use std::thread;
use std::time::{Duration, Instant};

use turbospark_ffi::{
    abi, ts_daemon_status_json, ts_daemon_stop, ts_store_root_set, ts_string_free,
};

static SERIAL: Mutex<()> = Mutex::new(());

struct Fixture {
    base: PathBuf,
    _serial: MutexGuard<'static, ()>,
}

impl Fixture {
    fn new(tag: &str) -> Self {
        let serial = SERIAL.lock().unwrap_or_else(|p| p.into_inner());
        let base = std::env::temp_dir().join(format!(
            "turbospark-ffi-daemon-{tag}-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&base);
        fs::create_dir_all(base.join("home/run")).unwrap();
        let home = CString::new(base.join("home").to_str().unwrap()).unwrap();
        assert_eq!(unsafe { ts_store_root_set(home.as_ptr()) }, abi::TS_OK);
        Self {
            base,
            _serial: serial,
        }
    }

    fn pid_file(&self) -> PathBuf {
        self.base.join("home/run/server.pid")
    }

    fn write_pid(&self, child: &Child) {
        fs::write(self.pid_file(), format!("{}\n", child.id())).unwrap();
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        unsafe { ts_store_root_set(ptr::null()) };
        let _ = fs::remove_dir_all(&self.base);
    }
}

/// Kills the child on drop so a failing assertion never leaks a sleeper.
struct Reaped(Child);

impl Drop for Reaped {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn spawn_sleep(program: &std::path::Path) -> Reaped {
    Reaped(Command::new(program).arg("30").spawn().unwrap())
}

/// Waits up to `limit` for the child to exit, returning whether it did.
fn exited_within(child: &mut Child, limit: Duration) -> bool {
    let start = Instant::now();
    while start.elapsed() < limit {
        if child.try_wait().unwrap().is_some() {
            return true;
        }
        thread::sleep(Duration::from_millis(50));
    }
    false
}

fn status_running() -> bool {
    let mut out: *mut std::os::raw::c_char = ptr::null_mut();
    assert_eq!(unsafe { ts_daemon_status_json(&mut out) }, abi::TS_OK);
    let text = unsafe { CStr::from_ptr(out) }.to_str().unwrap().to_string();
    unsafe { ts_string_free(out) };
    let json: serde_json::Value = serde_json::from_str(&text).unwrap();
    json["running"].as_bool().unwrap()
}

/// A stale pid file naming an unrelated live process: stop must not signal
/// it, and status must not report it as the daemon.
#[test]
fn a_recycled_pid_is_not_signalled_by_stop() {
    let fixture = Fixture::new("recycled");
    let mut other = spawn_sleep(std::path::Path::new("/bin/sleep"));
    fixture.write_pid(&other.0);

    assert_eq!(unsafe { ts_daemon_stop() }, abi::TS_OK);
    assert!(
        !exited_within(&mut other.0, Duration::from_millis(500)),
        "ts_daemon_stop signalled a process that is not the server"
    );
    assert!(!fixture.pid_file().exists(), "stale pid file left behind");

    // Stop cleared the stale file; write it again for status.
    fixture.write_pid(&other.0);
    assert!(
        !status_running(),
        "an unrelated process reported as the daemon"
    );
    assert!(!fixture.pid_file().exists(), "stale pid file left behind");
}

/// The positive half: a process whose executable is named like the server
/// binary is still stopped, so the identity check does not disable stop.
#[test]
fn a_process_running_the_server_binary_is_stopped() {
    let fixture = Fixture::new("server");
    let fake = fixture.base.join("turbospark-server");
    fs::copy("/bin/sleep", &fake).unwrap();
    let mut server = spawn_sleep(&fake);
    fixture.write_pid(&server.0);

    assert!(status_running());
    assert_eq!(unsafe { ts_daemon_stop() }, abi::TS_OK);
    assert!(
        exited_within(&mut server.0, Duration::from_secs(5)),
        "the server process was not stopped"
    );
    assert!(!fixture.pid_file().exists());
}
