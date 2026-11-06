//! `turbospark stop` with neither TURBOSPARK_HOME nor HOME set must refuse
//! rather than fall back to a CWD-relative `.turbospark`, where a planted
//! `run/server.pid` would choose which process gets signalled.

#[cfg(unix)]
#[test]
fn stop_without_home_refuses_and_ignores_cwd_pid_file() {
    use std::process::Command;

    // No tempfile dev-dependency here; use a per-process dir under the
    // cargo-provided test tmpdir.
    let cwd = std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("daemon-root-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&cwd);
    let mut victim = Command::new("sleep").arg("30").spawn().unwrap();
    let run = cwd.join(".turbospark").join("run");
    std::fs::create_dir_all(&run).unwrap();
    std::fs::write(run.join("server.pid"), format!("{}\n", victim.id())).unwrap();

    let out = Command::new(env!("CARGO_BIN_EXE_turbospark"))
        .arg("stop")
        .current_dir(&cwd)
        .env_clear()
        .output()
        .unwrap();

    let still_alive = victim.try_wait().unwrap().is_none();
    let _ = victim.kill();
    let _ = victim.wait();
    let _ = std::fs::remove_dir_all(&cwd);
    assert!(still_alive, "stop signalled the pid planted under CWD");
    assert!(!out.status.success(), "stop should fail without a home");
}
