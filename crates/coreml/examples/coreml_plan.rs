//! Prints where Core ML plans to run each operation of a model
//! (`MLComputePlan`, macOS 14.4+).
//!
//!     cargo run -p turbospark-coreml --example coreml_plan -- <model> \
//!         [--units cpu|gpu|ane|all] [--function name] [--cache dir] [--ops]
//!
//! The default is `--units all`, which is what an app load requests. The
//! summary has one row per function and preferred device, with the op count
//! and the summed cost estimate. `--ops` also lists every operation. A
//! `.mlpackage` is compiled into `--cache` (default: a `.compiled` directory
//! next to the package). Asking for the Neural Engine can trigger its first
//! compile, which takes minutes for large models.

use std::path::PathBuf;

use turbospark_coreml::{
    compile_if_needed, compute_plan, ComputeDevice, ComputeUnits, LoadOptions,
};

fn device(d: &Option<ComputeDevice>) -> &str {
    d.as_ref().map_or("none", ComputeDevice::as_str)
}

fn main() {
    let mut model: Option<PathBuf> = None;
    let mut units = ComputeUnits::All;
    let mut function: Option<String> = None;
    let mut cache: Option<PathBuf> = None;
    let mut ops = false;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--units" => {
                let v = args.next().unwrap_or_default();
                units = ComputeUnits::parse(&v).unwrap_or_else(|| {
                    eprintln!("unknown --units {v} (cpu, gpu, ane, all)");
                    std::process::exit(2);
                });
            }
            "--function" => function = args.next(),
            "--cache" => cache = args.next().map(PathBuf::from),
            "--ops" => ops = true,
            _ if model.is_none() => model = Some(PathBuf::from(a)),
            _ => {
                eprintln!("unexpected argument {a}");
                std::process::exit(2);
            }
        }
    }
    let Some(model) = model else {
        eprintln!("usage: coreml_plan <model> [--units cpu|gpu|ane|all] [--function name] [--cache dir] [--ops]");
        std::process::exit(2);
    };
    let cache = cache.unwrap_or_else(|| {
        model
            .parent()
            .map(|p| p.join(".compiled"))
            .unwrap_or_else(|| PathBuf::from(".compiled"))
    });
    let mut opts = LoadOptions::new(units);
    opts.function = function;
    let plan = compile_if_needed(&model, &cache).and_then(|c| compute_plan(&c, &opts));
    let plan = match plan {
        Ok(p) => p,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    };
    println!("units {}", units.as_str());
    println!(
        "{:24} {:6} {:>6} {:>8}",
        "function", "device", "ops", "cost"
    );
    for r in plan.summary() {
        println!(
            "{:24} {:6} {:>6} {:>8.4}",
            r.function,
            device(&r.device),
            r.ops,
            r.cost
        );
    }
    if ops {
        for op in &plan.ops {
            let supported: Vec<&str> = op.supported.iter().map(ComputeDevice::as_str).collect();
            println!(
                "{}\t{}\t{}\t{}\t[{}]\t{}",
                op.function,
                op.operator,
                op.output.as_deref().unwrap_or("-"),
                device(&op.preferred),
                supported.join(","),
                op.cost.map_or("-".to_string(), |c| format!("{c:.5}"))
            );
        }
    }
}
