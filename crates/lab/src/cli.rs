//! `tslab` subcommands.

use std::collections::BTreeMap;
use std::time::Duration;

use serde_json::{json, Value};

use crate::args::{parse_arm, parse_env, parse_prompt, parse_residency, parse_slots, Args};
use crate::host::HostRules;
use crate::lock::MachineLock;
use crate::orchestrate::{run_ab, Outcome, Session};
use crate::paths::lab_dir;
use crate::report;
use crate::stats;
use crate::workload::{Arm, Metric, OpenParams, PromptSource, Workload};

const USAGE: &str = "\
tslab: the evaluator for kernel and method experiments (macOS).

  tslab profile --model M [workload] [--slots auto,16,32] [--residency streamed,mapped]
                [--arm NAME[@SLOTS/RESIDENCY][:KEY=V,...]]... [--env KEY=V]... [--rounds 3] [--experiment NAME]
  tslab ab      --model M [workload] --baseline NAME[:KEY=V,...] --candidate NAME[:KEY=V,...]
                [--null] [--metric prefill_tok_s|decode_tok_s|prefill_s|peak_bytes]
                [--min-effect 0.02] [--pairs 3] [--max-pairs 6] [--experiment NAME]
  tslab gate    --spec SPEC.json
  tslab run-experiment --spec SPEC.json [--base REV]
  tslab confirm --spec SPEC.json
  tslab kernel  --model M [--bench gemv|prefill_gemm|all] [--experiment NAME]
  tslab ane-plan --path PROGRAM.mlpackage|MANIFEST_DIR [--units ane|all|gpu|cpu]
  tslab ledger  --experiment NAME [--kind run|verdict|...]

workload: [--prompt protocol:ID|messages:PATH|text:PATH]... [--decode N]
          [--chunk N|off] [--context N]
host:     [--max-foreign-cpu 150] [--max-single-cpu 90] [--allow-battery]
          [--timeout-min 60]

Results: $TURBOSPARK_LAB_DIR, else $TURBOSPARK_HOME/lab, else ~/.turbospark/lab.
KEY is a TURBOSPARK_* name or a knob short name (tslab knobs lists them).";

pub fn main() -> i32 {
    let mut argv: Vec<String> = std::env::args().skip(1).collect();
    if argv.is_empty() || argv[0] == "--help" || argv[0] == "-h" {
        println!("{USAGE}");
        return if argv.is_empty() { 2 } else { 0 };
    }
    let cmd = argv.remove(0);
    let args = Args::new(argv);
    let result = match cmd.as_str() {
        "profile" => profile(args),
        "ab" => ab(args),
        "ledger" => ledger(args),
        "knobs" => knobs_cmd(args),
        "gate" => crate::experiment::gate_cmd(args),
        "run-experiment" => crate::experiment::run_experiment_cmd(args),
        "confirm" => crate::experiment::confirm_cmd(args),
        "kernel" => crate::kernel::kernel_cmd(args),
        "ane-plan" => crate::ane::ane_plan_cmd(args),
        other => Err(format!("unknown command {other:?}\n\n{USAGE}")),
    };
    match result {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("tslab {cmd}: {e}");
            1
        }
    }
}

pub fn workload_from(args: &mut Args) -> Result<Workload, String> {
    let mut prompts: Vec<PromptSource> = args
        .take_all("--prompt")?
        .iter()
        .map(|p| parse_prompt(p))
        .collect::<Result<_, _>>()?;
    if prompts.is_empty() {
        prompts.push(PromptSource::Protocol("long-synthesis".into()));
    }
    let prefill_chunk = match args.take("--chunk")?.as_deref() {
        None => Some(128),
        Some("off") => None,
        Some(v) => Some(v.parse().map_err(|_| format!("--chunk {v:?}"))?),
    };
    Ok(Workload {
        prompts,
        decode_tokens: args.parsed("--decode", 0usize)?,
        prefill_chunk,
        max_context: args.parsed("--context", 4096u32)?,
    })
}

pub fn host_rules_from(args: &mut Args) -> Result<HostRules, String> {
    let d = HostRules::default();
    Ok(HostRules {
        max_total_cpu_percent: args.parsed("--max-foreign-cpu", d.max_total_cpu_percent)?,
        max_single_cpu_percent: args.parsed("--max-single-cpu", d.max_single_cpu_percent)?,
        sustained_s: d.sustained_s,
        require_ac: !args.flag("--allow-battery"),
    })
}

fn common_env(args: &mut Args) -> Result<BTreeMap<String, String>, String> {
    args.take_all("--env")?
        .iter()
        .map(|kv| parse_env(kv))
        .collect()
}

fn timeout_from(args: &mut Args) -> Result<Duration, String> {
    Ok(Duration::from_secs(
        60 * args.parsed("--timeout-min", 60u64)?,
    ))
}

fn parse_metric(s: &str) -> Result<Metric, String> {
    serde_json::from_value(json!(s)).map_err(|_| format!("metric {s:?}"))
}

fn profile(mut args: Args) -> Result<(), String> {
    let model = args.required("--model")?;
    let experiment = args
        .take("--experiment")?
        .unwrap_or_else(|| "profile".into());
    let workload = workload_from(&mut args)?;
    let rules = host_rules_from(&mut args)?;
    let timeout = timeout_from(&mut args)?;
    let rounds: usize = args.parsed("--rounds", 3)?;
    let common = common_env(&mut args)?;
    let slots: Vec<String> = args
        .take("--slots")?
        .unwrap_or_else(|| "auto".into())
        .split(',')
        .map(str::to_string)
        .collect();
    let residencies: Vec<String> = args
        .take("--residency")?
        .unwrap_or_else(|| "auto".into())
        .split(',')
        .map(str::to_string)
        .collect();
    let custom = args.take_all("--arm")?;
    args.finish()?;

    let mut arms = Vec::new();
    if custom.is_empty() {
        for s in &slots {
            for r in &residencies {
                let open = OpenParams {
                    slots: parse_slots(s)?,
                    residency: parse_residency(r)?,
                };
                arms.push(Arm {
                    name: format!("s{s}-{r}"),
                    env: common.clone(),
                    open,
                });
            }
        }
    } else {
        if slots.len() > 1 || residencies.len() > 1 {
            return Err("with --arm, give at most one --slots and one --residency".into());
        }
        let open = OpenParams {
            slots: parse_slots(&slots[0])?,
            residency: parse_residency(&residencies[0])?,
        };
        for a in &custom {
            arms.push(parse_arm(a, open, &common)?);
        }
    }

    let lab = lab_dir();
    let _lock = MachineLock::acquire(&lab, &format!("tslab profile {experiment}"))?;
    let session = Session::new(&lab, &experiment, &model, rules, timeout)?;
    eprintln!(
        "tslab profile: {} arm(s), {} round(s) after a warmup round; ledger {}",
        arms.len(),
        rounds,
        session.ledger.path().display()
    );
    let outcomes = session.run_rounds(&arms, &workload, rounds)?;
    let names: Vec<String> = arms.iter().map(|a| a.name.clone()).collect();
    let summary = report::print_profile(&names, &outcomes);
    session.ledger.append(
        "profile_summary",
        json!({"model": model, "workload": workload, "arms": arms, "summary": summary}),
    )?;
    Ok(())
}

/// The latest null calibration for the same workload and metric in this
/// experiment: (effect floor, peak-footprint noise in bytes).
pub fn null_floor(
    session: &Session,
    workload: &Workload,
    metric: Metric,
) -> Result<(f64, u64), String> {
    let w = serde_json::to_value(workload).map_err(|e| e.to_string())?;
    let m = serde_json::to_value(metric).map_err(|e| e.to_string())?;
    Ok(session
        .ledger
        .of_kind("null_calibration")?
        .iter()
        .rev()
        .find(|r| r["workload"] == w && r["metric"] == m)
        .map(|r| {
            (
                r["floor"].as_f64().unwrap_or(0.0),
                r["memory_noise"].as_f64().unwrap_or(0.0) as u64,
            )
        })
        .unwrap_or((0.0, 0)))
}

fn median_peak(outcomes: &[&Outcome]) -> Option<f64> {
    let v: Vec<f64> = outcomes
        .iter()
        .filter(|o| o.eligible)
        .filter_map(|o| o.record["peak_footprint_bytes"].as_f64())
        .collect();
    (!v.is_empty()).then(|| stats::median(&v))
}

/// Peak-footprint comparison between the arms' clean runs.
pub fn memory_check(pairs: &[(Outcome, Outcome)], requested: Option<u64>, noise: u64) -> Value {
    let a: Vec<&Outcome> = pairs.iter().map(|p| &p.0).collect();
    let b: Vec<&Outcome> = pairs.iter().map(|p| &p.1).collect();
    match (median_peak(&a), median_peak(&b)) {
        (Some(pa), Some(pb)) => {
            let allowance = crate::gates::memory_allowance(pa as u64, requested, noise) as f64;
            json!({
                "baseline_peak": pa,
                "candidate_peak": pb,
                "growth": pb - pa,
                "allowance": allowance,
                "passed": pb - pa <= allowance,
            })
        }
        _ => json!({"passed": false, "reason": "no clean peak readings"}),
    }
}

fn ab(mut args: Args) -> Result<(), String> {
    let model = args.required("--model")?;
    let experiment = args.take("--experiment")?.unwrap_or_else(|| "ab".into());
    let workload = workload_from(&mut args)?;
    let rules = host_rules_from(&mut args)?;
    let timeout = timeout_from(&mut args)?;
    let common = common_env(&mut args)?;
    let open = OpenParams {
        slots: parse_slots(&args.take("--slots")?.unwrap_or_else(|| "auto".into()))?,
        residency: parse_residency(&args.take("--residency")?.unwrap_or_else(|| "auto".into()))?,
    };
    let null = args.flag("--null");
    let baseline = parse_arm(
        &args
            .take("--baseline")?
            .unwrap_or_else(|| "baseline".into()),
        open,
        &common,
    )?;
    let candidate = if null {
        if args.take("--candidate")?.is_some() {
            return Err("--null runs the baseline against itself; drop --candidate".into());
        }
        Arm {
            name: format!("{}-again", baseline.name),
            ..baseline.clone()
        }
    } else {
        parse_arm(&args.required("--candidate")?, open, &common)?
    };
    let metric = parse_metric(
        &args
            .take("--metric")?
            .unwrap_or_else(|| "prefill_tok_s".into()),
    )?;
    let requested_effect: f64 = args.parsed("--min-effect", stats::MIN_EFFECT_FLOOR)?;
    let min_pairs: usize = args.parsed("--pairs", stats::MIN_PAIRS)?;
    let max_pairs: usize = args.parsed("--max-pairs", 6)?;
    args.finish()?;
    if min_pairs < stats::MIN_PAIRS || max_pairs < min_pairs {
        return Err(format!(
            "--pairs must be >= {} and --max-pairs >= --pairs",
            stats::MIN_PAIRS
        ));
    }

    let lab = lab_dir();
    let _lock = MachineLock::acquire(&lab, &format!("tslab ab {experiment}"))?;
    let session = Session::new(&lab, &experiment, &model, rules, timeout)?;
    let (floor, memory_noise) = if null {
        (0.0, 0)
    } else {
        null_floor(&session, &workload, metric)?
    };
    let min_effect = requested_effect.max(floor);
    eprintln!(
        "tslab ab: {} vs {}, metric {metric:?}, effect floor {:.2}% (null floor {:.2}%)",
        baseline.name,
        candidate.name,
        min_effect.max(stats::MIN_EFFECT_FLOOR) * 100.0,
        floor * 100.0
    );
    let (verdict, pairs) = run_ab(
        &session, &baseline, &candidate, &workload, metric, min_effect, min_pairs, max_pairs, None,
    )?;
    let memory = memory_check(&pairs, None, memory_noise);
    let run_ids: Vec<(String, String)> = pairs
        .iter()
        .map(|(a, b)| (a.run_id.clone(), b.run_id.clone()))
        .collect();
    report::print_verdict(
        &format!("{} vs {}", baseline.name, candidate.name),
        &verdict,
    );
    println!("  memory: {memory}");
    if null {
        let floor = stats::null_floor(&verdict.improvements);
        // Twice the identical arms' peak gap is what this host's peak
        // reading cannot resolve.
        let memory_noise = 2.0 * memory["growth"].as_f64().unwrap_or(0.0).abs();
        println!(
            "  null floor for this workload and metric: {:.2}% (2 x MAD); later ab runs in this experiment use it",
            floor * 100.0
        );
        session.ledger.append(
            "null_calibration",
            json!({"workload": workload, "metric": metric, "arm": baseline,
                   "improvements": verdict.improvements, "floor": floor,
                   "memory_noise": memory_noise, "pairs": run_ids}),
        )?;
    } else {
        session.ledger.append(
            "verdict",
            json!({"model": model, "workload": workload, "metric": metric,
                   "baseline": baseline, "candidate": candidate, "verdict": verdict,
                   "null_floor": floor, "memory": memory, "pairs": run_ids}),
        )?;
    }
    Ok(())
}

fn ledger(mut args: Args) -> Result<(), String> {
    let experiment = args.required("--experiment")?;
    let kind = args.take("--kind")?;
    args.finish()?;
    let dir = crate::paths::experiment_dir(&lab_dir(), &experiment)?;
    let ledger = crate::ledger::Ledger::open(&dir)?;
    for rec in ledger.read_all()? {
        if kind.as_deref().is_some_and(|k| rec["kind"] != k) {
            continue;
        }
        let brief = match rec["kind"].as_str() {
            Some("run") => format!(
                "{} {} round {} {} eligible={} prefill_tok_s={}",
                rec["arm"],
                if rec["warmup"] == true {
                    "warmup"
                } else {
                    "measured"
                },
                rec["round"],
                rec["status"],
                rec["eligible"],
                rec["totals"]["prefill_tok_s"]
            ),
            Some("verdict") => format!(
                "{} vs {}: {} median {}",
                rec["baseline"]["name"],
                rec["candidate"]["name"],
                rec["verdict"]["verdict"],
                rec["verdict"]["median"]
            ),
            _ => String::new(),
        };
        println!(
            "{} {:<16} {brief}",
            rec["id"].as_str().unwrap_or("?"),
            rec["kind"].as_str().unwrap_or("?")
        );
    }
    Ok(())
}

fn knobs_cmd(args: Args) -> Result<(), String> {
    args.finish()?;
    for k in crate::knobs::KNOBS {
        println!(
            "{:<22} {:<34} {:<26} {:?}/{:?}{}",
            k.name,
            k.env,
            k.values.join("|"),
            k.read_at,
            k.exactness,
            if k.quotable { "" } else { " (unquotable)" }
        );
    }
    Ok(())
}
