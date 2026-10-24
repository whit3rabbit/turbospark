//! `tslab kernel`: microbenches at the install's own shapes.
//!
//! Each bench takes its shapes from the `ArchConfig` in the install's
//! manifest, so the same command answers "which arm wins at which M" for any
//! family without editing constants. Results go to the ledger as
//! `kernel_curve` records with per-point times, c(M), achieved TFLOPS and
//! GB/s, and affine fits with break-even M.
//!
//! Benches run in this process (no worker): they touch no model weights, and
//! one Metal context serves every arm. The machine lock and host guard still
//! apply, because a GPU microbench is as sensitive to a neighbour as a model
//! run is.

pub mod attn_decode;
pub mod fit;
pub mod int4_matmul;
pub mod support;

use serde_json::json;

use crate::args::Args;
use crate::host::{eligibility, HostMonitor};
use crate::ledger::Ledger;
use crate::lock::MachineLock;
use crate::paths::{experiment_dir, lab_dir};

pub(crate) fn parse_list<T: std::str::FromStr>(s: &str, what: &str) -> Result<Vec<T>, String> {
    s.split(',')
        .map(|v| v.trim().parse().map_err(|_| format!("{what} {v:?}")))
        .collect()
}

pub fn kernel_cmd(mut args: Args) -> Result<(), String> {
    let model = args.required("--model")?;
    let experiment = args
        .take("--experiment")?
        .unwrap_or_else(|| "kernel".into());
    let bench = args.take("--bench")?.unwrap_or_else(|| "all".into());
    let ms: Vec<usize> = parse_list(
        &args
            .take("--rows")?
            .unwrap_or_else(|| "1,2,4,8,16,32,64,128,256,512".into()),
        "rows",
    )?;
    let contexts: Vec<u32> = parse_list(
        &args
            .take("--contexts")?
            .unwrap_or_else(|| "512,2048,8192,16384,32768".into()),
        "context",
    )?;
    let reps: usize = args.parsed("--reps", 5)?;
    let only_shape = args.take("--shape")?;
    let rules = crate::cli::host_rules_from(&mut args)?;
    args.finish()?;
    if !matches!(bench.as_str(), "all" | "int4_matmul" | "attn_decode") {
        return Err(format!(
            "--bench {bench:?}: int4_matmul, attn_decode or all"
        ));
    }
    if ms.contains(&0) || reps == 0 {
        return Err("rows and reps must be positive".into());
    }

    let dir = catalog::resolve_model_arg(&model);
    let arch = repack::peek_manifest_arch(&dir)?;
    let lab = lab_dir();
    let _lock = MachineLock::acquire(&lab, &format!("tslab kernel {experiment}"))?;
    let ledger = Ledger::open(&experiment_dir(&lab, &experiment)?)?;
    let mut context = gpu::MetalContext::new().map_err(|e| e.to_string())?;
    let provenance = crate::provenance::collect();
    let mut monitor = HostMonitor::new();
    eprintln!(
        "tslab kernel: {} ({}), ledger {}",
        dir.display(),
        arch.family.as_str(),
        ledger.path().display()
    );

    let record = |bench_rec: serde_json::Value, samples: Vec<crate::host::HostSample>| {
        let (eligible, reasons) = eligibility(&samples, &rules);
        if !eligible {
            println!("  DIRTY: {}", reasons.join("; "));
        }
        ledger.append(
            "kernel_curve",
            json!({"model": dir.display().to_string(), "family": arch.family.as_str(),
                   "eligible": eligible, "reasons": reasons, "result": bench_rec,
                   "provenance": provenance}),
        )
    };

    if bench == "all" || bench == "int4_matmul" {
        for (rows, cols) in int4_matmul::shapes(&arch) {
            if only_shape
                .as_deref()
                .is_some_and(|s| s != format!("{rows}x{cols}"))
            {
                continue;
            }
            let before = monitor.sample(None);
            let rec = int4_matmul::run_shape(&mut context, rows, cols, &ms, reps)?;
            let after = monitor.sample(None);
            int4_matmul::print_shape(&rec);
            record(rec, vec![before, after])?;
        }
    }
    if bench == "all" || bench == "attn_decode" {
        let before = monitor.sample(None);
        match attn_decode::run(&mut context, &arch, &contexts, reps)? {
            Some(rec) => {
                let after = monitor.sample(None);
                attn_decode::print(&rec);
                record(rec, vec![before, after])?;
            }
            None => println!("attention decode: no plain full-attention layers to drive"),
        }
    }
    Ok(())
}
