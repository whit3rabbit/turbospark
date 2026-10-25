//! Measures one arm once and writes one record. See `turbospark_lab::worker`.

#[cfg(target_os = "macos")]
fn main() {
    std::process::exit(turbospark_lab::worker::main_entry());
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("tslab-worker: macOS only");
    std::process::exit(2);
}
