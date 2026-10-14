//! Prints the inputs, outputs and states of a Core ML model.
//!
//!     cargo run -p turbospark-coreml --example coreml_describe -- <model> [function] [cache_dir]
//!
//! A `.mlpackage` is compiled into `cache_dir` (default: a `.compiled`
//! directory next to the package). Loads on the CPU so a first Neural
//! Engine compile is never triggered just to read a description.

use std::path::PathBuf;

use turbospark_coreml::{ComputeUnits, CoreMlModel, LoadOptions};

fn main() {
    let mut args = std::env::args().skip(1);
    let Some(model) = args.next().map(PathBuf::from) else {
        eprintln!("usage: coreml_describe <model> [function] [cache_dir]");
        std::process::exit(2);
    };
    let function = args.next().filter(|f| !f.is_empty() && f != "-");
    let cache = args.next().map(PathBuf::from).unwrap_or_else(|| {
        model
            .parent()
            .map(|p| p.join(".compiled"))
            .unwrap_or_else(|| PathBuf::from(".compiled"))
    });
    let mut opts = LoadOptions::new(ComputeUnits::CpuOnly);
    opts.function = function;
    match CoreMlModel::open(&model, &cache, &opts) {
        Ok(m) => {
            for (label, specs) in [
                ("input", m.inputs()),
                ("output", m.outputs()),
                ("state", m.states()),
            ] {
                for s in specs {
                    println!(
                        "{label:6} {:24} {:?}{}",
                        s.name,
                        s.kind,
                        if s.optional { " (optional)" } else { "" }
                    );
                }
            }
        }
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    }
}
