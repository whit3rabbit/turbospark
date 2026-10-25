//! Where human-readable output goes.
//!
//! Under `--json` stdout carries exactly one envelope (`schema.rs`), so an
//! agent can parse it without scraping. Every table and progress line then
//! goes to stderr instead. Commands print through [`say!`](crate::say) and
//! never through `println!`, so the switch is one flag rather than a branch
//! at every call site.

use std::sync::atomic::{AtomicBool, Ordering};

static JSON: AtomicBool = AtomicBool::new(false);

pub fn set_json(on: bool) {
    JSON.store(on, Ordering::Relaxed);
}

pub fn json_mode() -> bool {
    JSON.load(Ordering::Relaxed)
}

/// `println!` that moves to stderr under `--json`.
#[macro_export]
macro_rules! say {
    ($($t:tt)*) => {
        if $crate::out::json_mode() {
            eprintln!($($t)*)
        } else {
            println!($($t)*)
        }
    };
}

/// `print!` that moves to stderr under `--json`.
#[macro_export]
macro_rules! say_part {
    ($($t:tt)*) => {
        if $crate::out::json_mode() {
            eprint!($($t)*)
        } else {
            print!($($t)*)
        }
    };
}
