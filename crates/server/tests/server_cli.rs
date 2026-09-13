use std::process::Command;

#[test]
fn tailnet_bind_without_api_key_exits_with_error() {
    let output = Command::new(env!("CARGO_BIN_EXE_turbospark-server"))
        .args(["--model", "/nonexistent", "--bind", "tailnet"])
        .env_remove("TURBOSPARK_API_KEY")
        .output()
        .expect("turbospark-server should execute");

    assert_eq!(output.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("--bind tailnet requires --api-key KEY or a non-empty TURBOSPARK_API_KEY"),
        "unexpected stderr: {stderr}"
    );
}

#[test]
fn tailnet_bind_with_empty_env_key_exits_with_error() {
    let output = Command::new(env!("CARGO_BIN_EXE_turbospark-server"))
        .args(["--model", "/nonexistent", "--bind", "tailnet"])
        .env("TURBOSPARK_API_KEY", "")
        .output()
        .expect("turbospark-server should execute");

    assert_eq!(output.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("--bind tailnet requires --api-key KEY or a non-empty TURBOSPARK_API_KEY"),
        "unexpected stderr: {stderr}"
    );
}
