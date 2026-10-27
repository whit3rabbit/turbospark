//! `tslab ane-plan`: where Core ML would actually run a program.
//!
//! Timing a Core ML program does not say where it ran, and "supported on the
//! Neural Engine" does not mean "placed there" (`docs/COREML.md`, Gotchas;
//! the ANE expert-program experiment found CPU placement at every group size
//! up to 64 rows). `MLComputePlan` answers the placement question directly.
//! This command reads it for one program or a directory of them (for
//! example an ANE prefill program set) and records the shares in the ledger.
//!
//! The ANE+Steel split itself is an ordinary A/B arm (`ane_prefill` in the
//! knob table), measured by `tslab ab` with a worker built with the
//! `ane-prefill` feature.

use std::path::{Path, PathBuf};

use serde_json::{json, Value};
use turbospark_coreml::{
    compile_if_needed, compute_plan, supports, Capability, ComputeDevice, ComputePlan,
    ComputeUnits, LoadOptions,
};

use crate::args::Args;
use crate::ledger::Ledger;
use crate::paths::{experiment_dir, lab_dir};

/// Programs under `path`: the path itself if it is a package, else every
/// `.mlpackage` and `.mlmodelc` directly inside it, sorted.
fn programs(path: &Path) -> Result<Vec<PathBuf>, String> {
    let is_program = |p: &Path| {
        matches!(
            p.extension().and_then(|e| e.to_str()),
            Some("mlpackage" | "mlmodelc")
        )
    };
    if is_program(path) {
        return Ok(vec![path.to_path_buf()]);
    }
    let mut out: Vec<PathBuf> = std::fs::read_dir(path)
        .map_err(|e| format!("{}: {e}", path.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| is_program(p))
        .collect();
    out.sort();
    if out.is_empty() {
        return Err(format!(
            "{} holds no .mlpackage or .mlmodelc",
            path.display()
        ));
    }
    Ok(out)
}

/// Per-function device shares plus what keeps ops off the Neural Engine.
pub fn summarize(plan: &ComputePlan) -> Value {
    let mut functions: Vec<String> = plan.ops.iter().map(|o| o.function.clone()).collect();
    functions.sort();
    functions.dedup();
    let rows: Vec<Value> = functions
        .iter()
        .map(|f| {
            let ops: Vec<_> = plan.ops.iter().filter(|o| &o.function == f).collect();
            let total_cost: f64 = ops.iter().filter_map(|o| o.cost).sum();
            let share = |d: &ComputeDevice| -> f64 {
                ops.iter()
                    .filter(|o| o.preferred.as_ref() == Some(d))
                    .filter_map(|o| o.cost)
                    .sum::<f64>()
                    / total_cost.max(f64::MIN_POSITIVE)
            };
            let count = |d: &ComputeDevice| ops.iter().filter(|o| o.preferred.as_ref() == Some(d)).count();
            let off_ane: Vec<_> = ops
                .iter()
                .filter(|o| o.preferred.as_ref() != Some(&ComputeDevice::NeuralEngine) && o.preferred.is_some())
                .collect();
            let mut blockers: Vec<(String, usize, bool)> = Vec::new();
            for o in &off_ane {
                let ane_ok = o.supported.contains(&ComputeDevice::NeuralEngine);
                match blockers.iter_mut().find(|b| b.0 == o.operator && b.2 == ane_ok) {
                    Some(b) => b.1 += 1,
                    None => blockers.push((o.operator.clone(), 1, ane_ok)),
                }
            }
            blockers.sort_by_key(|b| std::cmp::Reverse(b.1));
            json!({
                "function": f,
                "ops": ops.len(),
                "cost_share": {
                    "ane": share(&ComputeDevice::NeuralEngine),
                    "gpu": share(&ComputeDevice::Gpu),
                    "cpu": share(&ComputeDevice::Cpu),
                },
                "op_count": {
                    "ane": count(&ComputeDevice::NeuralEngine),
                    "gpu": count(&ComputeDevice::Gpu),
                    "cpu": count(&ComputeDevice::Cpu),
                },
                // Ops placed elsewhere, by operator: `ane_supported` true
                // means Core ML chose not to, false means it cannot.
                "off_ane": blockers.iter().take(8).map(|(op, n, ok)| json!({"operator": op, "count": n, "ane_supported": ok})).collect::<Vec<_>>(),
            })
        })
        .collect();
    json!(rows)
}

pub fn ane_plan_cmd(mut args: Args) -> Result<(), String> {
    let path = PathBuf::from(args.required("--path")?);
    let units_s = args.take("--units")?.unwrap_or_else(|| "ane".into());
    let experiment = args
        .take("--experiment")?
        .unwrap_or_else(|| "ane-plan".into());
    let cache = args.take("--cache")?.map(PathBuf::from);
    args.finish()?;
    if !supports(Capability::ComputePlan) {
        return Err(format!(
            "Core ML compute plans need macOS {}",
            Capability::ComputePlan.required_macos()
        ));
    }
    let units = ComputeUnits::parse(&units_s).ok_or_else(|| format!("--units {units_s:?}"))?;
    let ledger = Ledger::open(&experiment_dir(&lab_dir(), &experiment)?)?;
    for program in programs(&path)? {
        let cache = cache.clone().unwrap_or_else(|| {
            program
                .parent()
                .map(|p| p.join(".compiled"))
                .unwrap_or_else(|| PathBuf::from(".compiled"))
        });
        let compiled = compile_if_needed(&program, &cache).map_err(|e| e.to_string())?;
        let plan = compute_plan(&compiled, &LoadOptions::new(units)).map_err(|e| e.to_string())?;
        let summary = summarize(&plan);
        crate::say!("{} (units {})", program.display(), units.as_str());
        for f in summary.as_array().into_iter().flatten() {
            let s = &f["cost_share"];
            crate::say!(
                "  {:<24} {:>5} ops   cost on ane {:>5.1}%  gpu {:>5.1}%  cpu {:>5.1}%",
                f["function"].as_str().unwrap_or("?"),
                f["ops"],
                (s["ane"].as_f64().unwrap_or(0.0) * 100.0 + 0.0),
                (s["gpu"].as_f64().unwrap_or(0.0) * 100.0 + 0.0),
                (s["cpu"].as_f64().unwrap_or(0.0) * 100.0 + 0.0)
            );
            for b in f["off_ane"].as_array().into_iter().flatten() {
                crate::say!(
                    "    off the ANE: {} x {} ({})",
                    b["count"],
                    b["operator"].as_str().unwrap_or("?"),
                    if b["ane_supported"] == true {
                        "supported there, not chosen"
                    } else {
                        "not supported there"
                    }
                );
            }
        }
        ledger.append(
            "ane_plan",
            json!({"program": program.display().to_string(), "units": units.as_str(),
                   "functions": summary, "provenance": crate::provenance::collect()}),
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use turbospark_coreml::OpPlacement;

    fn op(
        function: &str,
        operator: &str,
        preferred: Option<ComputeDevice>,
        supported: &[ComputeDevice],
        cost: Option<f64>,
    ) -> OpPlacement {
        OpPlacement {
            function: function.into(),
            operator: operator.into(),
            output: None,
            preferred,
            supported: supported.to_vec(),
            cost,
        }
    }

    #[test]
    fn shares_are_per_function_and_name_what_keeps_ops_off_the_ane() {
        use ComputeDevice::*;
        let plan = ComputePlan {
            ops: vec![
                op(
                    "main",
                    "matmul",
                    Some(NeuralEngine),
                    &[NeuralEngine, Gpu],
                    Some(0.6),
                ),
                op(
                    "main",
                    "matmul",
                    Some(NeuralEngine),
                    &[NeuralEngine],
                    Some(0.2),
                ),
                // Supported there, not chosen.
                op(
                    "main",
                    "softmax",
                    Some(Gpu),
                    &[Gpu, NeuralEngine],
                    Some(0.15),
                ),
                // Not supported there.
                op("main", "gather", Some(Cpu), &[Cpu], Some(0.05)),
                op("main", "gather", Some(Cpu), &[Cpu], None),
                // A folded constant: no device, no share, not a blocker.
                op("main", "const", None, &[], None),
                op("other", "add", Some(Cpu), &[Cpu, NeuralEngine], Some(1.0)),
            ],
        };
        let v = summarize(&plan);
        let rows = v.as_array().unwrap();
        assert_eq!(rows.len(), 2);
        let main = &rows[0];
        assert_eq!(main["function"], "main");
        assert_eq!(main["ops"], 6);
        let close = |a: &Value, b: f64| (a.as_f64().unwrap() - b).abs() < 1e-9;
        assert!(close(&main["cost_share"]["ane"], 0.8));
        assert!(close(&main["cost_share"]["gpu"], 0.15));
        assert!(close(&main["cost_share"]["cpu"], 0.05));
        assert_eq!(main["op_count"]["ane"], 2);
        assert_eq!(main["op_count"]["cpu"], 2);
        let off = main["off_ane"].as_array().unwrap();
        // Most frequent first; `const` has no device and is not listed.
        assert_eq!(off.len(), 2);
        assert_eq!(off[0]["operator"], "gather");
        assert_eq!(off[0]["count"], 2);
        assert_eq!(off[0]["ane_supported"], false);
        assert_eq!(off[1]["operator"], "softmax");
        assert_eq!(off[1]["ane_supported"], true);
        assert_eq!(rows[1]["function"], "other");
        assert!(close(&rows[1]["cost_share"]["cpu"], 1.0));
    }

    #[test]
    fn a_plan_with_no_cost_estimates_has_zero_shares_not_nan() {
        let plan = ComputePlan {
            ops: vec![op("main", "relu", Some(ComputeDevice::Gpu), &[], None)],
        };
        let v = summarize(&plan);
        assert_eq!(v[0]["cost_share"]["gpu"].as_f64(), Some(0.0));
    }

    #[test]
    fn programs_lists_packages_sorted_and_refuses_an_empty_dir() {
        let dir = std::env::temp_dir().join(format!(
            "tslab-ane-programs-{}-{}",
            std::process::id(),
            crate::ledger::unix_ms()
        ));
        std::fs::create_dir_all(dir.join("b.mlmodelc")).unwrap();
        std::fs::create_dir_all(dir.join("a.mlpackage")).unwrap();
        std::fs::write(dir.join("manifest.json"), "{}").unwrap();
        let found = programs(&dir).unwrap();
        let names: Vec<_> = found
            .iter()
            .map(|p| p.file_name().unwrap().to_str().unwrap().to_string())
            .collect();
        assert_eq!(names, ["a.mlpackage", "b.mlmodelc"]);
        assert_eq!(programs(&dir.join("a.mlpackage")).unwrap().len(), 1);
        let empty = dir.join("empty");
        std::fs::create_dir_all(&empty).unwrap();
        assert!(programs(&empty).unwrap_err().contains("holds no"));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
