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
use crate::schema::{CmdOutput, Envelope, RecordRef};
use crate::stats;
use crate::workload::{Arm, Metric, OpenParams, PromptSource, Workload};

const USAGE: &str = "\
tslab: the evaluator for kernel and method experiments (macOS).

  tslab profile --model M [workload] [--slots auto,16,32] [--residency streamed,mapped]
                [--arm NAME[@SLOTS/RESIDENCY][:KEY=V,...]]... [--env KEY=V]... [--rounds 3] [--experiment NAME]
                [--dispatch-probe 0,4,8,16 | --standin attn,mlp,gdn]   (L1 / L4: their own arms, need --decode)
  tslab ab      --model M [workload] --baseline NAME[:KEY=V,...] --candidate NAME[:KEY=V,...]
                [--null] [--metric prefill_tok_s|decode_tok_s|prefill_s|peak_bytes|followup_ttft_s|audio_run_s|rtf]
                [--min-effect 0.02] [--pairs 3] [--max-pairs 6] [--experiment NAME]
  tslab gate    --spec SPEC.json [--base REV]      (loads the model: takes minutes)
  tslab run-experiment --spec SPEC.json [--base REV]
  tslab confirm --spec SPEC.json [--base REV]
  tslab kernel  --model M [--bench NAME[,NAME]|all] [--rows 1,2,4,...] [--contexts 512,...]
                [--case LABEL] [--reps 5] [--hash] [--verify-from EXP] [--experiment NAME]
  tslab kernel  --list                             (the registered benches)
  tslab hash-diff --experiment A --against B      (exit 3 when any hashed output changed)
  tslab project --knob NAME|--bench B --default ARM --mode ARM
                --curves EXP | --default-from A --mode-from B      (kernel_curve experiments)
                [--x 1] [--shapes LABEL,...] [--min-rows N] [--counts LABEL=N,...] [--scale S] [--dispatches N --us-per-dispatch X | --probe-from EXP]
                --base-ms X | --base-from EXP[:ARM] [--base-metric decode_tok_s|prefill_tok_s]
                | --round-from EXP --width W [--flags L] [--kv N] [--phase verify]
                [--floor-from EXP] [--share-from EXP --block gdn|mlp|attn] [--experiment NAME]
                (price a kernel mode from curves before any model run; no GPU work)
  tslab calibrate --projection EXP[:ID] (--verdict EXP[:ID] | --measured X --source TEXT)
                [--metric decode_tok_s] [--invert] [--experiment NAME]   |   --all
                (join a projection to the verdict that followed; --all lists every miss and the bias)
  tslab tree-replay --model M --drafter mtp|dflash [--prompt ...]... [--generate 512]
                [--depth 8|7] [--blocks 2,4,DEPTH] [--serve-block 2] [--top-k 4] [--slots 16] [--experiment NAME]
  tslab ane-plan --path PROGRAM.mlpackage|MANIFEST_DIR [--units ane|all|gpu|cpu]
  tslab chain-capture --model M [--prompt ...]... [--tokens 512] [--depth 8] [--experiment NAME]
  tslab depth-replay [--experiment NAME] [--costs flag_off,flag_on,ledger:EXP[@FLAGS][#KV]] [--tokens 512] [--top 10]
  tslab hybrid-replay [--experiment NAME] [--costs flag_on,ledger:EXP[@FLAGS][#KV]] [--tokens 512]
                [--ngram 4] [--lookup-draft 8] [--tuned 3,1,1,2] [--cap 8]
                (prompt lookup over the MTP chain, replayed on a chain capture that kept its ids)
  tslab ledger  --experiment NAME [--kind run|verdict|...]
  tslab trace   --model M [workload] [--arm NAME[:KEY=V,...]] [--record-s 10] [--experiment NAME]
                (Metal System Trace of the arm's measured work; never quotable)
  tslab workflow NAME|--plan PLAN.json [--model M] [--experiment E] [--var KEY=V]...
                [--step ID] [--from ID] [--resume RUN] [--dry-run]   (--list, --print NAME)
  tslab capabilities --model M                    (what tslab can measure on this install, and tags)
  tslab inventory --model M                       (the install's weight shapes by dtype)
  tslab pinned  [--repo kapitsa|omlx|mlx-serve] [--class C] [--id ID] [--dir PATH]
                (the verified kernel settings pinned from the compared engines)
  tslab knobs
  tslab export  --experiment A[,B]|--all [--kind K[,K]] [--format jsonl|json|csv] [--redact-paths]
                [--out PATH [--force]]            (ledgers as tslab/1 series)
  tslab report  --experiment A[,B]|--all --html OUT.html [--title T] [--kind K] [--redact-paths] [--force]
  tslab schema                                     (the JSON Schema of everything tslab writes)

workload: [--prompt protocol:ID|messages:PATH|text:PATH|corpus:N|chat:PATH]... [--decode N]
          [--chunk N|off] [--context N]
          or, for an audio install: --audio-job JOB.json (see workloads/audio/)
host:     [--max-foreign-cpu 150] [--max-single-cpu 90] [--allow-battery]
          [--timeout-min 60] [--quiet-wait-min 5] [--min-free-gb 10]
          (--quiet-wait-min 0 measures on a busy host; --min-free-gb 0 skips the disk check)

--base REV: the commit the experiment's diff is checked against (default: main).
  An experiment tree that merges in the lab itself needs --base <that merge commit>.
exit codes (run-experiment, confirm): 0 keep, 3 discard, 4 inconclusive (do not revert),
  1 error, 2 usage.
--json (any command, anywhere on the line): stdout is exactly one tslab/1 envelope
  {schema, command, ok, exit_code, decision?, records, result, warnings, error?};
  tables and progress go to stderr.

Results: $TURBOSPARK_LAB_DIR, else $TURBOSPARK_HOME/lab, else ~/.turbospark/lab.
KEY is a TURBOSPARK_* name or a knob short name (tslab knobs lists them).";

pub fn main() -> i32 {
    let mut argv: Vec<String> = std::env::args().skip(1).collect();
    // `--json` is global: it may sit anywhere on the line.
    let json_out = {
        let before = argv.len();
        argv.retain(|a| a != "--json");
        argv.len() != before
    };
    crate::out::set_json(json_out);
    if argv.is_empty() || argv[0] == "--help" || argv[0] == "-h" {
        crate::say!("{USAGE}");
        return if argv.is_empty() { 2 } else { 0 };
    }
    let cmd = argv.remove(0);
    let result = dispatch(&cmd, Args::new(argv));
    finish(&cmd, result, json_out)
}

/// Print the outcome and return the exit code. Under `--json` stdout gets
/// exactly one envelope; otherwise an error goes to stderr as before.
pub fn finish(cmd: &str, result: Result<CmdOutput, String>, json_out: bool) -> i32 {
    let appended = crate::ledger::take_appended();
    let result = result.map(|mut out| {
        if out.result.is_null() {
            out.result = summarize_appended(&appended);
        }
        out
    });
    let code = match &result {
        Ok(out) => out.code,
        Err(e) => {
            eprintln!("tslab {cmd}: {e}");
            1
        }
    };
    if json_out {
        let refs = appended.into_iter().map(|(r, _)| r).collect();
        let envelope = Envelope::new(cmd, &result, refs);
        match serde_json::to_string(&envelope) {
            Ok(line) => println!("{line}"),
            Err(e) => {
                eprintln!("tslab {cmd}: envelope: {e}");
                return 1;
            }
        }
    }
    code
}

/// Every command `dispatch` knows, which a workflow step may name.
pub const COMMANDS: &[&str] = &[
    "profile",
    "ab",
    "ledger",
    "knobs",
    "pinned",
    "schema",
    "hash-diff",
    "project",
    "calibrate",
    "export",
    "report",
    "inventory",
    "capabilities",
    "trace",
    "gate",
    "run-experiment",
    "confirm",
    "kernel",
    "ane-plan",
    "chain-capture",
    "depth-replay",
    "hybrid-replay",
    "tree-replay",
];

/// Run one command in this process. `tslab workflow` calls this for each
/// step, so a step and the same command typed alone are one code path.
pub fn dispatch(cmd: &str, args: Args) -> Result<CmdOutput, String> {
    let unit = |r: Result<(), String>| r.map(|()| CmdOutput::ok(Value::Null));
    match cmd {
        "profile" => unit(profile(args)),
        "ab" => unit(ab(args)),
        "ledger" => ledger(args),
        "knobs" => knobs_cmd(args),
        "pinned" => pinned_cmd(args),
        "schema" => schema_cmd(args),
        "hash-diff" => hash_diff_cmd(args),
        "project" => crate::projection::project_cmd(args),
        "calibrate" => crate::projection::calibrate_cmd(args),
        "export" => crate::export::export_cmd(args),
        "inventory" => inventory_cmd(args),
        "capabilities" => crate::families::capabilities_cmd(args),
        "trace" => crate::trace::trace_cmd(args),
        "workflow" => crate::workflow::workflow_cmd(args),
        "report" => crate::export::report_cmd(args),
        "gate" => crate::experiment::gate_cmd(args),
        "run-experiment" => crate::experiment::run_experiment_cmd(args),
        "confirm" => crate::experiment::confirm_cmd(args),
        "kernel" => unit(crate::kernel::kernel_cmd(args)),
        "ane-plan" => unit(crate::ane::ane_plan_cmd(args)),
        "chain-capture" => unit(crate::depth_replay::chain_capture_cmd(args)),
        "depth-replay" => unit(crate::depth_replay::depth_replay_cmd(args)),
        "hybrid-replay" => unit(crate::hybrid_replay::hybrid_replay_cmd(args)),
        "tree-replay" => unit(crate::tree_replay::tree_replay_cmd(args)),
        other => Err(format!("unknown command {other:?}\n\n{USAGE}")),
    }
}

/// The default envelope payload: what the command appended, by kind. A run
/// record carries the worker's whole record and host samples, so only its
/// headline fields are repeated here; the ledger has the rest by id.
fn summarize_appended(appended: &[(RecordRef, Value)]) -> Value {
    let mut by_kind: BTreeMap<String, Vec<Value>> = BTreeMap::new();
    for (r, rec) in appended {
        let v = if r.kind == "run" {
            json!({"id": rec["id"], "arm": rec["arm"], "round": rec["round"],
                   "warmup": rec["warmup"], "status": rec["status"],
                   "eligible": rec["eligible"], "reasons": rec["reasons"],
                   "totals": rec["totals"], "wall_s": rec["wall_s"]})
        } else {
            rec.clone()
        };
        by_kind.entry(r.kind.clone()).or_default().push(v);
    }
    json!({ "by_kind": by_kind })
}

pub fn workload_from(args: &mut Args) -> Result<Workload, String> {
    let mut prompts: Vec<PromptSource> = args
        .take_all("--prompt")?
        .iter()
        .map(|p| parse_prompt(p))
        .collect::<Result<_, _>>()?;
    // `--audio-job PATH`: an audio install's job as JSON (`AudioJob`, see
    // crates/lab/workloads/audio/). It replaces the text prompts.
    let audio = match args.take("--audio-job")? {
        Some(path) => {
            let text =
                std::fs::read_to_string(&path).map_err(|e| format!("--audio-job {path}: {e}"))?;
            Some(serde_json::from_str(&text).map_err(|e| format!("--audio-job {path}: {e}"))?)
        }
        None => None,
    };
    if audio.is_some() && !prompts.is_empty() {
        return Err("--audio-job replaces --prompt: give one or the other".into());
    }
    if prompts.is_empty() && audio.is_none() {
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
        speculation: None,
        audio,
    })
}

pub fn host_rules_from(args: &mut Args) -> Result<HostRules, String> {
    let d = HostRules::default();
    Ok(HostRules {
        max_total_cpu_percent: args.parsed("--max-foreign-cpu", d.max_total_cpu_percent)?,
        max_single_cpu_percent: args.parsed("--max-single-cpu", d.max_single_cpu_percent)?,
        sustained_s: d.sustained_s,
        require_ac: !args.flag("--allow-battery"),
        // Minutes on the command line, seconds in the rule.
        quiet_wait_s: 60.0 * args.parsed("--quiet-wait-min", d.quiet_wait_s / 60.0)?,
        min_free_gb: args.parsed("--min-free-gb", d.min_free_gb)?,
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
    let probe: Option<Vec<usize>> = args
        .take("--dispatch-probe")?
        .map(|v| crate::kernel::parse_list(&v, "--dispatch-probe"))
        .transpose()?;
    let standin: Option<Vec<String>> = args
        .take("--standin")?
        .map(|v| v.split(',').map(|b| b.trim().to_string()).collect());
    args.finish()?;

    let mut arms = Vec::new();
    let open_one = || -> Result<OpenParams, String> {
        Ok(OpenParams {
            slots: parse_slots(&slots[0])?,
            residency: parse_residency(&residencies[0])?,
        })
    };
    if probe.is_some() || standin.is_some() {
        if !custom.is_empty() || probe.is_some() == standin.is_some() {
            return Err(
                "--dispatch-probe and --standin each build their own arms: give one, without --arm"
                    .into(),
            );
        }
        if workload.decode_tokens < 8 {
            return Err("probe and stand-in arms price decode: give --decode 8 or more".into());
        }
        let open = open_one()?;
        let arm = |name: String, env: Option<(&str, String)>| {
            let mut e = common.clone();
            if let Some((k, v)) = env {
                e.insert(k.to_string(), v);
            }
            Arm {
                name,
                env: e,
                open,
                speculation: None,
                expect_engaged: Vec::new(),
            }
        };
        if let Some(ns) = &probe {
            for &n in ns {
                // N = 0 is the baseline: the variable unset, not set to 0.
                let env = (n > 0).then(|| ("TURBOSPARK_DISPATCH_PROBE", n.to_string()));
                arms.push(arm(format!("probe{n}"), env));
            }
            if !ns.contains(&0) || ns.len() < 2 {
                return Err(
                    "--dispatch-probe needs 0 and at least one other count, e.g. 0,4,8".into(),
                );
            }
        }
        if let Some(blocks) = &standin {
            arms.push(arm("base".into(), None));
            for b in blocks {
                if !matches!(b.as_str(), "attn" | "mlp" | "gdn") {
                    return Err(format!("--standin {b:?}: attn, mlp or gdn"));
                }
                arms.push(arm(
                    format!("standin_{b}"),
                    Some(("TURBOSPARK_STANDIN", b.clone())),
                ));
            }
        }
    } else if custom.is_empty() {
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
                    speculation: None,
                    expect_engaged: Vec::new(),
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
    // Probe and stand-in arms are unquotable by design, which keeps them out
    // of the profile's clean medians. The fits read the runs the host guard
    // passed instead: ok, measured, and no reason but that one.
    let decode_ms = |arm: &str| {
        let tok_s: Vec<f64> = outcomes
            .iter()
            .filter(|o| o.arm == arm && !o.warmup && host_clean(o))
            .filter_map(|o| o.record["totals"]["decode_tok_s"].as_f64())
            .filter(|t| *t > 0.0)
            .collect();
        (!tok_s.is_empty()).then(|| 1e3 / stats::median(&tok_s))
    };
    if let Some(ns) = &probe {
        let layers = crate::kernel::Subject::open(&model)?
            .arch()?
            .num_layers
            .max(1) as usize;
        let rec = dispatch_probe_fit(ns, layers, decode_ms)?;
        crate::say!(
            "  dispatch probe: {:.2} us per chained dispatch (fit over {} arms, {} layers; max residual {:.1}%)",
            rec["us_per_chained_dispatch"].as_f64().unwrap_or(f64::NAN),
            ns.len(),
            layers,
            rec["fit"]["max_rel_residual"].as_f64().unwrap_or(f64::NAN) * 100.0
        );
        session.ledger.append(
            "dispatch_probe",
            json!({"model": model, "workload": workload, "quotable": false, "result": rec,
                   "points": rec["points"], "us_per_chained_dispatch": rec["us_per_chained_dispatch"]}),
        )?;
    }
    if let Some(blocks) = &standin {
        let base = decode_ms("base").ok_or("the base arm has no clean decode reading")?;
        let mut out = serde_json::Map::new();
        for b in blocks {
            let Some(ms) = decode_ms(&format!("standin_{b}")) else {
                crate::say!("  stand-in {b}: no clean decode reading");
                continue;
            };
            let block = base - ms;
            crate::say!(
                "  stand-in {b}: {block:.3} ms per token in place ({:.1}% of {base:.3})",
                100.0 * block / base
            );
            out.insert(
                b.clone(),
                json!({"ms_per_token": block, "share": block / base, "standin_ms": ms}),
            );
        }
        session.ledger.append(
            "standin",
            json!({"model": model, "workload": workload, "quotable": false,
                   "base_ms_per_token": base, "blocks": out}),
        )?;
    }
    Ok(())
}

/// A run the host guard and the worker passed, whose only blemish is an
/// unquotable switch. Measurement hooks need these; verdicts never use them.
pub fn host_clean(o: &Outcome) -> bool {
    o.status == "ok"
        && o.reasons
            .iter()
            .all(|r| r.ends_with("perturbs timing; run is not quotable"))
}

/// Fit decode ms per token against the chained dispatches the probe adds
/// (N per dense layer per token). The slope is the price of one dispatch
/// on the dependency chain (evaluator item L1).
pub fn dispatch_probe_fit(
    ns: &[usize],
    layers: usize,
    decode_ms: impl Fn(&str) -> Option<f64>,
) -> Result<Value, String> {
    let mut points = Vec::new();
    for &n in ns {
        if let Some(ms) = decode_ms(&format!("probe{n}")) {
            points.push(json!({"n": n, "added_dispatches": n * layers, "ms_per_token": ms}));
        }
    }
    let xy: Vec<(f64, f64)> = points
        .iter()
        .map(|p| {
            (
                p["added_dispatches"].as_f64().unwrap(),
                p["ms_per_token"].as_f64().unwrap(),
            )
        })
        .collect();
    let fit = crate::fit::fit_affine(&xy)
        .ok_or("the probe needs clean decode readings at two or more counts")?;
    Ok(json!({"points": points, "layers": layers, "fit": fit,
              "us_per_chained_dispatch": fit.b * 1e3}))
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
    let crate::orchestrate::AbRun {
        verdict,
        pairs,
        stopped,
    } = run_ab(
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
    crate::say!("  memory: {memory}");
    if let Some(why) = &stopped {
        crate::say!("  stopped early: {why}");
    }
    if null {
        let floor = stats::null_floor(&verdict.improvements);
        // Twice the identical arms' peak gap is what this host's peak
        // reading cannot resolve.
        let memory_noise = 2.0 * memory["growth"].as_f64().unwrap_or(0.0).abs();
        crate::say!(
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

fn ledger(mut args: Args) -> Result<CmdOutput, String> {
    let experiment = args.required("--experiment")?;
    let kind = args.take("--kind")?;
    args.finish()?;
    let dir = crate::paths::experiment_dir(&lab_dir(), &experiment)?;
    let ledger = crate::ledger::Ledger::open(&dir)?;
    let mut out = Vec::new();
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
            Some("kernel_curve") => format!(
                "{} {} eligible={}",
                rec["result"]["bench"].as_str().unwrap_or("?"),
                rec["result"]["case"]
                    .as_str()
                    .map(str::to_string)
                    .unwrap_or_else(|| format!(
                        "{}x{}",
                        rec["result"]["rows"], rec["result"]["cols"]
                    )),
                rec["eligible"]
            ),
            Some("experiment" | "confirm") => format!(
                "{} {}",
                rec["decision"].as_str().unwrap_or("?"),
                rec["reason"].as_str().unwrap_or("")
            ),
            _ => String::new(),
        };
        crate::say!(
            "{} {:<16} {brief}",
            rec["id"].as_str().unwrap_or("?"),
            rec["kind"].as_str().unwrap_or("?")
        );
        out.push(rec);
    }
    Ok(CmdOutput::ok(
        json!({"experiment": experiment, "records": out}),
    ))
}

fn knobs_cmd(args: Args) -> Result<CmdOutput, String> {
    args.finish()?;
    for k in crate::knobs::KNOBS {
        crate::say!(
            "{:<22} {:<34} {:<26} {:?}/{:?}{}",
            k.name,
            k.env,
            k.values.join("|"),
            k.read_at,
            k.exactness,
            if k.quotable { "" } else { " (unquotable)" }
        );
    }
    crate::say!();
    crate::say!("kernel modes (docs/KERNEL_MODES.md): the bench that prices each, and how");
    for l in crate::knobs::LANES {
        let how = match l.compare {
            crate::knobs::Compare::Arms { baseline, lane } => {
                format!("{baseline} -> {lane}")
            }
            crate::knobs::Compare::Process { arm, baseline_env } => {
                format!("{arm} across processes, baseline with {baseline_env}")
            }
        };
        crate::say!("{:<22} {:<14} {how}", l.knob, l.bench);
    }
    Ok(CmdOutput::ok(
        json!({"knobs": crate::knobs::KNOBS, "lanes": crate::knobs::LANES}),
    ))
}

/// `tslab schema`: the JSON Schema of everything tslab writes.
fn schema_cmd(args: Args) -> Result<CmdOutput, String> {
    args.finish()?;
    let doc: Value = serde_json::from_str(crate::schema::SCHEMA_DOC).map_err(|e| e.to_string())?;
    if !crate::out::json_mode() {
        crate::say!("{}", crate::schema::SCHEMA_DOC.trim_end());
    }
    Ok(CmdOutput::ok(json!({
        "schema": crate::schema::SCHEMA,
        "kinds": crate::schema::KINDS.iter().map(|(k, d)| json!({"kind": k, "holds": d})).collect::<Vec<_>>(),
        "document": doc,
    })))
}

/// The bench results of one experiment's eligible-or-not `kernel_curve`
/// records that carry the version-1 shape (older ones have no hashes).
pub fn bench_results(experiment: &str) -> Result<Vec<crate::bench::BenchResult>, String> {
    let dir = crate::paths::experiment_dir(&lab_dir(), experiment)?;
    Ok(crate::ledger::Ledger::open(&dir)?
        .of_kind("kernel_curve")?
        .into_iter()
        .filter_map(|r| serde_json::from_value(r["result"].clone()).ok())
        .collect())
}

/// `tslab hash-diff`: did any kernel's output bits change between two
/// hashed `tslab kernel --hash` runs (two builds, or two arms' env)?
fn hash_diff_cmd(mut args: Args) -> Result<CmdOutput, String> {
    let before = args.required("--experiment")?;
    let after = args.required("--against")?;
    args.finish()?;
    let (a, b) = (bench_results(&before)?, bench_results(&after)?);
    // A case that ran more than once in one experiment: the latest wins.
    let latest = |v: Vec<crate::bench::BenchResult>| {
        let mut m: BTreeMap<(String, String), crate::bench::BenchResult> = BTreeMap::new();
        for r in v {
            m.insert((r.bench.clone(), r.case.clone()), r);
        }
        m.into_values().collect::<Vec<_>>()
    };
    let (a, b) = (latest(a), latest(b));
    let (changes, compared, unmatched) = crate::bench::hash_diff(&a, &b);
    if compared == 0 {
        return Err(format!(
            "no hashed points in common between {before} and {after}; run both with tslab kernel --hash"
        ));
    }
    for c in &changes {
        crate::say!(
            "CHANGED {} {} {} x={}: {} -> {}",
            c.bench,
            c.case,
            c.arm,
            c.x,
            c.before,
            c.after
        );
    }
    crate::say!(
        "{} of {compared} hashed points changed ({unmatched} only in {after})",
        changes.len()
    );
    let rec = json!({"experiment": before, "against": after, "compared": compared,
                     "unmatched": unmatched, "changes": changes,
                     "identical": changes.is_empty()});
    let lab = lab_dir();
    crate::ledger::Ledger::open(&crate::paths::experiment_dir(&lab, &after)?)?
        .append("hash_diff", rec.clone())?;
    // Exit 3, like a discard: a script gating a bit-exact change branches
    // on it without parsing.
    let code = if changes.is_empty() {
        0
    } else {
        crate::experiment::EXIT_DISCARD
    };
    Ok(CmdOutput::with_code(code, rec))
}

/// `tslab inventory`: every distinct weight shape the install binds, by
/// dtype, with how many tensors share it. The shapes the matmul benches and
/// the verify ladder use.
fn inventory_cmd(mut args: Args) -> Result<CmdOutput, String> {
    let model = args.required("--model")?;
    args.finish()?;
    let dir = catalog::resolve_model_arg(&model);
    let shapes = crate::inventory::load(&dir)?;
    for s in &shapes {
        crate::say!(
            "{:<12} {:>7} x {:<7} x{:<4} {}",
            s.dtype_name,
            s.rows,
            s.cols,
            s.count,
            s.roles.join(", ")
        );
    }
    Ok(CmdOutput::ok(
        json!({"model": dir.display().to_string(), "shapes": shapes}),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn outcome(status: &str, reasons: &[&str]) -> Outcome {
        Outcome {
            run_id: "r".into(),
            arm: "a".into(),
            round: 1,
            warmup: false,
            record: Value::Null,
            status: status.into(),
            eligible: reasons.is_empty(),
            reasons: reasons.iter().map(|r| r.to_string()).collect(),
        }
    }

    /// The bug this exists for: probe arms are unquotable, so the fit that
    /// read only eligible runs never had a reading to fit.
    #[test]
    fn unquotable_alone_is_host_clean_but_a_busy_host_is_not() {
        let unq = "TURBOSPARK_DISPATCH_PROBE perturbs timing; run is not quotable";
        assert!(host_clean(&outcome("ok", &[])));
        assert!(host_clean(&outcome("ok", &[unq])));
        assert!(!host_clean(&outcome(
            "ok",
            &[unq, "foreign CPU over 150% for 5s or more"]
        )));
        assert!(!host_clean(&outcome("refused", &[unq])));
    }

    #[test]
    fn the_probe_slope_is_microseconds_per_added_dispatch() {
        // 64 layers, 2 us per added dispatch: +4 per layer is +256
        // dispatches, +0.512 ms per token.
        let ms = |arm: &str| match arm {
            "probe0" => Some(40.0),
            "probe4" => Some(40.512),
            "probe8" => Some(41.024),
            _ => None,
        };
        let r = dispatch_probe_fit(&[0, 4, 8], 64, ms).unwrap();
        assert!(
            (r["us_per_chained_dispatch"].as_f64().unwrap() - 2.0).abs() < 1e-9,
            "{r}"
        );
        assert_eq!(r["points"][2]["added_dispatches"], 512);
        assert!(dispatch_probe_fit(&[0, 4], 64, |_| None).is_err());
    }
}

/// List or inspect the pinned kernel settings (`crates/lab/pinned/`).
/// The embedded set is the verified baseline. `--dir` loads a sweep
/// directory instead, and `--dir` plus `--id` shows one entry from it.
/// The matched entries are the envelope's `result` under the global `--json`.
fn pinned_cmd(mut args: Args) -> Result<CmdOutput, String> {
    let dir = args.take("--dir")?;
    let repo = args.take("--repo")?;
    let class = args.take("--class")?;
    let id = args.take("--id")?;
    args.finish()?;
    if let Some(r) = &repo {
        if !crate::pinned::REPOS.contains(&r.as_str()) {
            return Err(format!(
                "repo {r:?} is not one of {:?}",
                crate::pinned::REPOS
            ));
        }
    }
    let set = match &dir {
        Some(d) => crate::pinned::PinnedSet::load_dir(std::path::Path::new(d))?,
        None => crate::pinned::PinnedSet::embedded()?,
    };
    set.validate()?;
    let mut matches: Vec<_> = set
        .entries()
        .iter()
        .filter(|e| repo.as_deref().is_none_or(|r| e.repo == r))
        .filter(|e| class.as_deref().is_none_or(|c| e.kernel_class == c))
        .filter(|e| id.as_deref().is_none_or(|i| e.id == i))
        .collect();
    if let Some(want) = &id {
        if matches.is_empty() {
            return Err(format!("no pinned entry with id {want:?} in this set"));
        }
    }
    matches.sort_by(|a, b| a.id.cmp(&b.id));
    for entry in &matches {
        let model = entry.primary_model();
        crate::say!(
            "{:<28} {:<18} {:<10} {:>3} settings  {} {} [{}]",
            entry.id,
            entry.kernel_class,
            entry.source.license,
            entry.settings.len(),
            model.name,
            model.quant,
            model.machine,
        );
    }
    Ok(CmdOutput::ok(json!({"pinned": matches})))
}
