#[cfg(target_os = "macos")]
fn main() -> std::process::ExitCode {
    turbospark_bench::startup_probe::main_entry(std::env::args().skip(1))
}

#[cfg(not(target_os = "macos"))]
fn main() -> std::process::ExitCode {
    eprintln!("turbospark-startup-probe requires macOS and a Metal-capable device");
    std::process::ExitCode::from(2)
}
