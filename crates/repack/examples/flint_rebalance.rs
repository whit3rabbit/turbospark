//! Migrates an EXISTING Flint install, in place, to the up/down rebalance that
//! keeps `silu(gate) * up` inside FP16 (`crates/repack/src/flint_rebalance.rs`
//! has the reasoning). An install built after that change already carries it.
//!
//!   cargo run --release -p turbospark-repack --example flint_rebalance -- <install dir>
//!
//! Refuses an install whose `flint_rebalance.json` already records a shift, so
//! it cannot be applied twice. `--undo` reverses it exactly.

use std::path::PathBuf;

fn main() {
    let mut args = std::env::args().skip(1);
    let dir = PathBuf::from(args.next().expect("usage: flint_rebalance <install dir> [--undo]"));
    let undo = args.next().as_deref() == Some("--undo");
    let have = turbospark_repack::recorded_shift(&dir).expect("reading the marker");
    let steps = if undo {
        if have == 0 {
            eprintln!("refused: the install records no shift, so there is nothing to undo");
            std::process::exit(2);
        }
        -have
    } else {
        if have != 0 {
            eprintln!("refused: the install already records a net shift of {have} steps");
            std::process::exit(2);
        }
        turbospark_repack::FLINT_SWIGLU_SHIFT
    };
    let stats = turbospark_repack::rebalance_install_in_place(&dir, steps)
        .expect("rebalancing in place");
    println!("shifted by {steps:+} steps: {stats:?}");
    if stats.left_alone != 0 {
        eprintln!("warning: {} non-zero values were left alone for range", stats.left_alone);
    }
}
