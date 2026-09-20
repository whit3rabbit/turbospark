//! Progress indicators for long-running network and streaming operations.
//!
//! All styles and character sets are strictly ASCII (no emojis, no em dashes)
//! to conform to project conventions.

use std::time::Duration;

use indicatif::{ProgressBar, ProgressStyle};

/// Create a spinner with standard ASCII ticks and a message.
pub fn spinner(msg: impl Into<String>) -> ProgressBar {
    let pb = ProgressBar::new_spinner();
    if let Ok(style) = ProgressStyle::with_template("[{elapsed_precise}] {spinner} {msg}") {
        pb.set_style(style.tick_chars("-\\|/"));
    }
    pb.set_message(msg.into());
    pb.enable_steady_tick(Duration::from_millis(100));
    pb
}

/// Create a byte-tracking progress bar, or a spinner if `total_bytes` is unknown (0).
pub fn byte_progress_bar(total_bytes: u64) -> ProgressBar {
    let pb = if total_bytes > 0 {
        let pb = ProgressBar::new(total_bytes);
        if let Ok(style) = ProgressStyle::with_template(
            "[{elapsed_precise}] [{bar:40.cyan/blue}] {bytes:>10}/{total_bytes:10} ({bytes_per_sec}, {eta}) {msg}",
        ) {
            pb.set_style(style.progress_chars("=>-"));
        }
        pb
    } else {
        let pb = ProgressBar::new_spinner();
        if let Ok(style) = ProgressStyle::with_template(
            "[{elapsed_precise}] {spinner} {bytes:>10} ({bytes_per_sec}) {msg}",
        ) {
            pb.set_style(style.tick_chars("-\\|/"));
        }
        pb
    };
    pb.enable_steady_tick(Duration::from_millis(100));
    pb
}

/// Create a count-tracking progress bar for iterating over N items.
pub fn count_progress_bar(total: u64, msg: impl Into<String>) -> ProgressBar {
    let pb = ProgressBar::new(total);
    if let Ok(style) =
        ProgressStyle::with_template("[{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} {msg}")
    {
        pb.set_style(style.progress_chars("=>-"));
    }
    pb.set_message(msg.into());
    pb.enable_steady_tick(Duration::from_millis(100));
    pb
}
