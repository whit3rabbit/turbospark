//! The tslab orchestrator. See `turbospark_lab::cli` for the commands.

#[cfg(target_os = "macos")]
fn main() {
    std::process::exit(turbospark_lab::cli::main());
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("tslab: macOS only");
    std::process::exit(2);
}
