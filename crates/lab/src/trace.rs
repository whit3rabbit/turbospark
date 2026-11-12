//! `tslab trace`: a Metal System Trace of one arm's measured work
//! (evaluator item L5, `docs/MLX_SERVE_KERNELS.md`).
//!
//! The worker starts the way every tslab run does (fresh process, cleared
//! environment plus the arm's variables). When it prints its measuring
//! marker, after the model load and the warm pass, `xctrace` attaches to it
//! for a bounded window with the Metal System Trace template and the Metal
//! GPU Counters instrument. Attach, not launch: a process `xctrace`
//! launches here never reached the GPU, and a long recording was killed
//! while saving, so the window is short by default.
//!
//! The analyzer (`tools/trace_tables.py`, standard library only, run with
//! `python3 -I`) reports compute-encoder busy percent and idle gaps, and
//! per-kernel shares of the shader timeline. A traced run is never
//! quotable: the ledger record says so, and no tok/s from it is printed.

use std::path::Path;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use serde_json::{json, Value};

use crate::args::{parse_arm, Args};
use crate::cli::{host_rules_from, workload_from};
use crate::ledger::new_id;
use crate::lock::MachineLock;
use crate::orchestrate::{sibling_worker, PASSTHROUGH, PASSTHROUGH_TURBOSPARK};
use crate::paths::{experiment_dir, lab_dir};
use crate::schema::CmdOutput;
use crate::worker::{Mode, WorkerRequest, MEASURING_MARKER};
use crate::workload::OpenParams;

const ANALYZER: &str = include_str!("../tools/trace_tables.py");

pub fn trace_cmd(mut args: Args) -> Result<CmdOutput, String> {
    let model = args.required("--model")?;
    let experiment = args.take("--experiment")?.unwrap_or_else(|| "trace".into());
    let workload = workload_from(&mut args)?;
    let record_s: u64 = args.parsed("--record-s", 10)?;
    let wait_s: u64 = args.parsed("--marker-timeout-s", 600)?;
    let arm_spec = args.take("--arm")?.unwrap_or_else(|| "traced".into());
    let common: std::collections::BTreeMap<String, String> = args
        .take_all("--env")?
        .iter()
        .map(|kv| crate::args::parse_env(kv))
        .collect::<Result<_, _>>()?;
    // Host rules are taken for the usage line's sake; a trace is never
    // judged eligible, so they only label the record.
    let _ = host_rules_from(&mut args)?;
    args.finish()?;
    if !(1..=60).contains(&record_s) {
        return Err("--record-s 1..=60: longer recordings were killed while saving here".into());
    }
    let arm = parse_arm(&arm_spec, OpenParams::default(), &common)?;

    let lab = lab_dir();
    let _lock = MachineLock::acquire(&lab, &format!("tslab trace {experiment}"))?;
    let exp = experiment_dir(&lab, &experiment)?;
    let ledger = crate::ledger::Ledger::open(&exp)?;
    let worker = sibling_worker()?;
    let run_id = new_id();
    let dir = exp.join("traces").join(&run_id);
    std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let request_path = dir.join("request.json");
    let record_path = dir.join("record.json");
    let stderr_path = dir.join("worker.stderr");
    let trace_path = dir.join("metal.trace");
    let analyzer = dir.join("trace_tables.py");
    crate::atomic::write_new(&analyzer, ANALYZER.as_bytes())?;
    crate::atomic::write_new_json(
        &request_path,
        &serde_json::to_value(WorkerRequest {
            run_id: run_id.clone(),
            model: model.clone(),
            arm: arm.clone(),
            workload: workload.clone(),
            mode: Mode::Timing,
            logits_path: None,
        })
        .map_err(|e| e.to_string())?,
    )?;

    let stderr = std::fs::File::create(&stderr_path)
        .map_err(|e| format!("{}: {e}", stderr_path.display()))?;
    let mut cmd = Command::new(&worker);
    cmd.arg("--request")
        .arg(&request_path)
        .arg("--out")
        .arg(&record_path)
        .env_clear()
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::from(stderr));
    for key in PASSTHROUGH.iter().chain(PASSTHROUGH_TURBOSPARK) {
        if let Some(v) = std::env::var_os(key) {
            cmd.env(key, v);
        }
    }
    cmd.envs(crate::orchestrate::ORCHESTRATOR_ENV.iter().copied());
    cmd.envs(&arm.env);
    let mut child = cmd
        .spawn()
        .map_err(|e| format!("{}: {e}", worker.display()))?;
    let pid = child.id();
    eprintln!("tslab trace: worker {pid} started; waiting for it to finish warming up");

    let started = Instant::now();
    loop {
        if let Some(status) = child.try_wait().map_err(|e| e.to_string())? {
            return Err(format!(
                "the worker exited ({status}) before measuring; see {}",
                stderr_path.display()
            ));
        }
        let text = std::fs::read_to_string(&stderr_path).unwrap_or_default();
        if text.contains(MEASURING_MARKER) {
            break;
        }
        if started.elapsed() > Duration::from_secs(wait_s) {
            let _ = child.kill();
            return Err(format!("no measuring marker within {wait_s} s"));
        }
        std::thread::sleep(Duration::from_millis(100));
    }

    eprintln!("tslab trace: recording {record_s} s of process {pid}");
    let xctrace = Command::new("xcrun")
        .args(["xctrace", "record", "--template", "Metal System Trace"])
        .args(["--instrument", "Metal GPU Counters"])
        .arg("--time-limit")
        .arg(format!("{record_s}s"))
        .arg("--output")
        .arg(&trace_path)
        .arg("--attach")
        .arg(pid.to_string())
        .stdin(Stdio::null())
        .output()
        .map_err(|e| format!("xcrun xctrace: {e}"))?;
    let recorder_log = String::from_utf8_lossy(&xctrace.stdout).to_string()
        + &String::from_utf8_lossy(&xctrace.stderr);
    // Let the worker finish its workload; the trace is already cut.
    let worker_status = child.wait().map_err(|e| e.to_string())?;
    if !trace_path.exists() {
        return Err(format!(
            "xctrace wrote no trace (exit {:?}): {}",
            xctrace.status.code(),
            recorder_log.trim()
        ));
    }
    let analysis = analyze(&analyzer, &trace_path, pid)?;
    print_summary(&analysis);
    let rec = json!({
        "model": model, "arm": arm, "workload": workload, "record_s": record_s,
        "quotable": false, "trace": trace_path.display().to_string(),
        "worker_exit": worker_status.code(), "recorder_exit": xctrace.status.code(),
        "engaged": crate::knobs::engaged_lines(&std::fs::read_to_string(&stderr_path).unwrap_or_default()),
        "encoders": analysis["encoders"], "shaders": analysis["shaders"],
        "provenance": crate::provenance::collect(),
    });
    ledger.append("trace", rec.clone())?;
    Ok(CmdOutput::ok(rec)
        .warn("a traced run is never quotable: compare shares between traces, not times"))
}

fn analyze(script: &Path, trace: &Path, pid: u32) -> Result<Value, String> {
    let out = Command::new("python3")
        .arg("-I")
        .arg(script)
        .arg(trace)
        .arg(pid.to_string())
        .output()
        .map_err(|e| format!("python3: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "trace analyzer failed: {}",
            String::from_utf8_lossy(&out.stderr).trim()
        ));
    }
    serde_json::from_slice(&out.stdout).map_err(|e| format!("trace analyzer output: {e}"))
}

fn print_summary(a: &Value) {
    let e = &a["encoders"];
    crate::say!(
        "  compute encoders: {} over {:.1} ms, GPU busy {:.1}%, idle gap p50 {} us, p90 {} us",
        e["count"],
        e["window_ms"].as_f64().unwrap_or(0.0),
        e["busy_percent"].as_f64().unwrap_or(f64::NAN),
        e["idle_gap_us"]["p50"]
            .as_f64()
            .map_or("-".into(), |v| format!("{v:.1}")),
        e["idle_gap_us"]["p90"]
            .as_f64()
            .map_or("-".into(), |v| format!("{v:.1}")),
    );
    for s in a["shaders"].as_array().into_iter().flatten().take(12) {
        crate::say!(
            "  {:>5.1}%  {:>6} samples  {}",
            s["share"].as_f64().unwrap_or(0.0) * 100.0,
            s["samples"],
            s["name"].as_str().unwrap_or("?")
        );
    }
}
