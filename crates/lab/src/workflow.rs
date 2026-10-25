//! `tslab workflow`: a sequence of tslab commands as one resumable run.
//!
//! A plan is JSON: a name, the steps, and per step a command, its
//! arguments, the steps it needs, and a `when` predicate over the install's
//! coverage tags (`tslab capabilities`). Each step calls `cli::dispatch`,
//! the same function its command typed alone calls, so a step and the
//! command are one code path, and the step's result lands in the ledger
//! like any other run of it.
//!
//! The predicates are how one plan serves every family: a step that needs a
//! drafter is skipped, with its reason ledgered, on an install without one.
//! A failed step stops the run unless it says `continue_on_error`; steps
//! that need a failed or skipped step are skipped too. `--resume RUN`
//! re-runs a plan and skips the steps that run already finished with the
//! same plan bytes.
//!
//! The whole run holds the machine lock (`MachineLock::hold`), so no other
//! session measures between two steps.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::args::Args;
use crate::ledger::{new_id, Ledger};
use crate::lock::MachineLock;
use crate::paths::{experiment_dir, lab_dir};
use crate::schema::CmdOutput;

/// Built-in plans, by name.
pub const BUILTIN: &[(&str, &str)] = &[
    ("new-family", include_str!("../workflows/new-family.json")),
    (
        "kernel-candidate",
        include_str!("../workflows/kernel-candidate.json"),
    ),
    (
        "mode-candidate",
        include_str!("../workflows/mode-candidate.json"),
    ),
    ("spec-depth", include_str!("../workflows/spec-depth.json")),
    (
        "audio-survey",
        include_str!("../workflows/audio-survey.json"),
    ),
    ("report", include_str!("../workflows/report.json")),
];

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Plan {
    pub name: String,
    #[serde(default)]
    pub description: String,
    /// Variables a caller must give with `--var KEY=V` (beyond the built-in
    /// `model`, `experiment`, `lab`, `lab_src` (the folder holding
    /// `workloads/`, see `assets_dir`) and `run`).
    #[serde(default)]
    pub requires: Vec<String>,
    /// Defaults for variables, overridden by `--var`.
    #[serde(default)]
    pub vars: BTreeMap<String, String>,
    pub steps: Vec<Step>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Step {
    pub id: String,
    pub command: String,
    /// `--key value` pairs. A `true` is a bare flag, an array repeats the
    /// flag, `false` or null omits it. Strings take `{var}` placeholders.
    #[serde(default)]
    pub args: BTreeMap<String, Value>,
    #[serde(default)]
    pub needs: Vec<String>,
    /// Coverage tags, all of which must hold: `moe`, `drafter`,
    /// `bench:verify_ladder`, `!audio`, `audio:music|audio:transcribe`.
    #[serde(default)]
    pub when: Vec<String>,
    #[serde(default)]
    pub continue_on_error: bool,
}

/// Where the lab's data files (`workloads/`) are: the folder above the
/// running binary when it has them (the packaged skill's `bin/` sits beside
/// its `workloads/`), else this crate's source directory. A binary copied out
/// of a build tree keeps working only in the first layout, so the packaged
/// skill ships the workloads with it.
pub fn assets_dir() -> std::path::PathBuf {
    let beside_exe = std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent()?.parent().map(|p| p.to_path_buf()))
        .filter(|root| root.join("workloads").is_dir());
    beside_exe.unwrap_or_else(|| std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")))
}

/// Does a `when` list hold for these tags? `!t` negates, `a|b` is any.
pub fn when_holds(when: &[String], tags: &BTreeSet<String>) -> Result<(), String> {
    for clause in when {
        let ok = clause.split('|').any(|alt| match alt.strip_prefix('!') {
            Some(t) => !tags.contains(t),
            None => tags.contains(alt),
        });
        if !ok {
            return Err(format!("needs {clause}"));
        }
    }
    Ok(())
}

/// Replace `{name}` placeholders. An unknown one is an error, not left in:
/// a literal `{against}` reaching a command would name a missing ledger.
pub fn substitute(text: &str, vars: &BTreeMap<String, String>) -> Result<String, String> {
    let mut out = String::new();
    let mut rest = text;
    while let Some(i) = rest.find('{') {
        out.push_str(&rest[..i]);
        let after = &rest[i + 1..];
        let j = after
            .find('}')
            .ok_or_else(|| format!("unclosed placeholder in {text:?}"))?;
        let key = &after[..j];
        out.push_str(
            vars.get(key)
                .ok_or_else(|| format!("{text:?} names {{{key}}}, which no --var gives"))?,
        );
        rest = &after[j + 1..];
    }
    out.push_str(rest);
    Ok(out)
}

/// A step's argv, after substitution.
pub fn argv(step: &Step, vars: &BTreeMap<String, String>) -> Result<Vec<String>, String> {
    let mut out = Vec::new();
    for (k, v) in &step.args {
        let flag = format!("--{k}");
        let mut push = |v: &Value| -> Result<(), String> {
            match v {
                Value::Bool(true) => out.push(flag.clone()),
                Value::Bool(false) | Value::Null => {}
                Value::String(s) => {
                    out.push(flag.clone());
                    out.push(substitute(s, vars)?);
                }
                Value::Number(n) => {
                    out.push(flag.clone());
                    out.push(n.to_string());
                }
                other => return Err(format!("step {} arg {k}: {other} is not a value", step.id)),
            }
            Ok(())
        };
        match v {
            Value::Array(items) => items.iter().try_for_each(&mut push)?,
            other => push(other)?,
        }
    }
    Ok(out)
}

pub fn load(name_or_path: &str) -> Result<(Plan, String), String> {
    let text = match BUILTIN.iter().find(|(n, _)| *n == name_or_path) {
        Some((_, t)) => t.to_string(),
        None => std::fs::read_to_string(name_or_path).map_err(|e| {
            format!("{name_or_path}: not a built-in workflow or a readable plan: {e}")
        })?,
    };
    let plan: Plan = serde_json::from_str(&text).map_err(|e| format!("{name_or_path}: {e}"))?;
    validate(&plan)?;
    Ok((plan, text))
}

pub fn validate(plan: &Plan) -> Result<(), String> {
    let mut seen = BTreeSet::new();
    for s in &plan.steps {
        if !seen.insert(s.id.as_str()) {
            return Err(format!("step id {:?} appears twice", s.id));
        }
        if s.command == "workflow" {
            return Err("a step may not run a workflow".into());
        }
        #[cfg(target_os = "macos")]
        if !crate::cli::COMMANDS.contains(&s.command.as_str()) {
            return Err(format!(
                "step {} runs unknown command {:?}",
                s.id, s.command
            ));
        }
        for n in &s.needs {
            if !seen.contains(n.as_str()) || n == &s.id {
                return Err(format!(
                    "step {} needs {n}, which is not an earlier step",
                    s.id
                ));
            }
        }
    }
    Ok(())
}

/// The run's exit code: an error over a discard over an inconclusive over 0.
pub fn worst(codes: &[i32]) -> i32 {
    for c in [
        1,
        crate::experiment::EXIT_DISCARD,
        crate::experiment::EXIT_INCONCLUSIVE,
    ] {
        if codes.contains(&c) {
            return c;
        }
    }
    codes.iter().copied().find(|c| *c != 0).unwrap_or(0)
}

pub fn workflow_cmd(mut args: Args) -> Result<CmdOutput, String> {
    let positional = args.positional();
    if args.flag("--list") {
        args.finish()?;
        let mut list = Vec::new();
        for (name, text) in BUILTIN {
            let plan: Plan = serde_json::from_str(text).map_err(|e| format!("{name}: {e}"))?;
            crate::say!("{name:<18} {}", plan.description);
            list.push(json!({"name": name, "description": plan.description,
                             "requires": plan.requires, "steps": plan.steps.len()}));
        }
        return Ok(CmdOutput::ok(json!({"workflows": list})));
    }
    if let Some(name) = args.take("--print")? {
        args.finish()?;
        let (plan, text) = load(&name)?;
        if !crate::out::json_mode() {
            crate::say!("{}", text.trim_end());
        }
        return Ok(CmdOutput::ok(
            serde_json::to_value(plan).map_err(|e| e.to_string())?,
        ));
    }
    let name = args
        .take("--plan")?
        .or(positional)
        .ok_or("tslab workflow NAME|--plan PLAN.json --model M (or --list, --print NAME)")?;
    let model = args.take("--model")?;
    let experiment = args.take("--experiment")?;
    let only = args.take("--step")?;
    let from = args.take("--from")?;
    let resume = args.take("--resume")?;
    let dry = args.flag("--dry-run");
    let user_vars = args.take_all("--var")?;
    args.finish()?;

    let (plan, text) = load(&name)?;
    let plan_sha = model_io::hash_data(text.as_bytes());
    let run = resume.clone().unwrap_or_else(new_id);
    let lab = lab_dir();
    let experiment = experiment.unwrap_or_else(|| plan.name.clone());
    let mut vars = plan.vars.clone();
    vars.insert("experiment".into(), experiment.clone());
    vars.insert("lab".into(), lab.display().to_string());
    vars.insert("lab_src".into(), assets_dir().display().to_string());
    vars.insert("run".into(), run.clone());
    if let Some(m) = &model {
        vars.insert("model".into(), m.clone());
    }
    for kv in &user_vars {
        let (k, v) = kv
            .split_once('=')
            .ok_or_else(|| format!("--var {kv:?}: expected KEY=VALUE"))?;
        vars.insert(k.to_string(), v.to_string());
    }
    for r in &plan.requires {
        if !vars.contains_key(r) {
            return Err(format!("workflow {} needs --var {r}=...", plan.name));
        }
    }
    // A variable's default may name another (`{lab_src}/workloads/...`).
    let snapshot = vars.clone();
    for v in vars.values_mut() {
        if v.contains('{') {
            *v = substitute(v, &snapshot)?;
        }
    }

    // Coverage decides `when`; a plan with no model (report) has only the
    // `nomodel` tag.
    let tags: BTreeSet<String> = match &model {
        Some(m) => {
            let subject = crate::kernel::Subject::open(m)?;
            crate::families::coverage(&subject)
                .tags
                .into_iter()
                .collect()
        }
        None => BTreeSet::from(["nomodel".to_string()]),
    };

    let ledger = Ledger::open(&experiment_dir(&lab, &experiment)?)?;
    let finished: BTreeSet<String> = match &resume {
        Some(r) => ledger
            .of_kind("workflow_step")?
            .into_iter()
            .filter(|s| s["run"] == r.as_str() && s["plan_sha256"] == plan_sha.as_str())
            .filter(|s| s["status"] == "ok")
            .filter_map(|s| s["step"].as_str().map(str::to_string))
            .collect(),
        None => BTreeSet::new(),
    };
    let _lock = if dry {
        None
    } else {
        Some(MachineLock::hold(
            &lab,
            &format!("tslab workflow {} {run}", plan.name),
        )?)
    };
    crate::say!(
        "workflow {} run {run}: {} steps, experiment {experiment}{}",
        plan.name,
        plan.steps.len(),
        if dry { " (dry run)" } else { "" }
    );

    let mut started = from.is_none();
    let mut status: BTreeMap<String, String> = BTreeMap::new();
    let mut codes = Vec::new();
    let mut steps_out = Vec::new();
    let mut stop = false;
    for step in &plan.steps {
        started |= from.as_deref() == Some(step.id.as_str());
        let decision = if only.as_deref().is_some_and(|o| o != step.id) || !started {
            Some(("not selected".to_string(), "skipped"))
        } else if stop {
            Some(("an earlier step failed".to_string(), "skipped"))
        } else if finished.contains(&step.id) {
            Some((format!("finished in run {run}"), "done"))
        } else if let Some(n) = step
            .needs
            .iter()
            .find(|n| status.get(*n).is_some_and(|s| s != "ok" && s != "done"))
        {
            Some((format!("needs {n}, which did not finish"), "skipped"))
        } else if let Err(why) = when_holds(&step.when, &tags) {
            Some((why, "skipped"))
        } else {
            None
        };
        if let Some((why, word)) = decision {
            crate::say!("  {:<20} {word}: {why}", step.id);
            status.insert(step.id.clone(), word.to_string());
            let rec = json!({"step": step.id, "status": word, "reason": why});
            if !dry && word == "skipped" {
                ledger.append(
                    "workflow_step",
                    json!({"workflow": plan.name, "run": run, "plan_sha256": plan_sha,
                           "step": step.id, "command": step.command, "status": "skipped",
                           "reason": rec["reason"]}),
                )?;
            }
            steps_out.push(rec);
            continue;
        }
        let argv = argv(step, &vars)?;
        crate::say!(
            "  {:<20} tslab {} {}",
            step.id,
            step.command,
            argv.join(" ")
        );
        if dry {
            status.insert(step.id.clone(), "ok".into());
            steps_out.push(json!({"step": step.id, "status": "planned",
                                  "command": step.command, "argv": argv}));
            continue;
        }
        // The step's own records are collected for its line, then put back
        // so the workflow's envelope lists every record the run appended.
        let before = crate::ledger::take_appended();
        let result = crate::cli::dispatch(&step.command, Args::new(argv.clone()));
        let mut appended = crate::ledger::take_appended();
        let ids: Vec<Value> = appended
            .iter()
            .map(|(r, _)| json!({"id": r.id, "kind": r.kind, "ledger": r.ledger}))
            .collect();
        let (code, word, detail) = match &result {
            Ok(out) => (
                out.code,
                if out.code == 0 { "ok" } else { "finished" },
                out.result.clone(),
            ),
            Err(e) => (1, "failed", json!({"error": e})),
        };
        if let Err(e) = &result {
            crate::say!("  {:<20} failed: {e}", step.id);
        }
        codes.push(code);
        status.insert(step.id.clone(), word.to_string());
        if code == 1 && !step.continue_on_error {
            stop = true;
        }
        let rec = json!({"workflow": plan.name, "run": run, "plan_sha256": plan_sha,
                         "step": step.id, "command": step.command, "argv": argv,
                         "status": word, "exit_code": code, "records": ids});
        ledger.append("workflow_step", rec.clone())?;
        let mut restored = before;
        restored.append(&mut appended);
        restored.extend(crate::ledger::take_appended());
        crate::ledger::put_back(restored);
        let mut line = rec;
        line["result"] = detail;
        steps_out.push(line);
    }
    let code = worst(&codes);
    let summary = json!({"workflow": plan.name, "run": run, "plan_sha256": plan_sha,
                         "experiment": experiment, "exit_code": code, "dry_run": dry,
                         "tags": tags, "steps": steps_out});
    if !dry {
        ledger.append(
            "workflow_run",
            json!({"workflow": plan.name, "run": run, "plan_sha256": plan_sha,
                   "model": model, "exit_code": code, "tags": tags,
                   "statuses": status}),
        )?;
    }
    Ok(CmdOutput::with_code(code, summary))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tags(t: &[&str]) -> BTreeSet<String> {
        t.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn predicates_and_placeholders() {
        let t = tags(&["text", "moe", "bench:int4_matmul"]);
        assert!(when_holds(&["moe".into(), "!audio".into()], &t).is_ok());
        assert!(when_holds(&["drafter".into()], &t).is_err());
        assert!(when_holds(&["mtp|moe".into()], &t).is_ok());
        let vars = BTreeMap::from([("model".to_string(), "m".to_string())]);
        assert_eq!(substitute("x/{model}/y", &vars).unwrap(), "x/m/y");
        assert!(substitute("{against}", &vars).is_err());
        assert!(substitute("{model", &vars).is_err());
    }

    #[test]
    fn argv_expands_flags_lists_and_numbers() {
        let s: Step = serde_json::from_value(json!({
            "id": "p", "command": "profile",
            "args": {"model": "{model}", "arm": ["a", "b"], "decode": 64, "null": true, "x": false}
        }))
        .unwrap();
        let vars = BTreeMap::from([("model".to_string(), "M".to_string())]);
        let a = argv(&s, &vars).unwrap();
        assert_eq!(
            a,
            ["--arm", "a", "--arm", "b", "--decode", "64", "--model", "M", "--null"]
        );
    }

    #[test]
    fn every_builtin_parses_and_validates() {
        for (name, _) in BUILTIN {
            let (plan, _) = load(name).unwrap();
            assert_eq!(&plan.name, name);
            for s in &plan.steps {
                assert!(
                    crate::cli::COMMANDS.contains(&s.command.as_str()),
                    "{name}: step {} runs unknown command {}",
                    s.id,
                    s.command
                );
            }
        }
        let bad: Plan = serde_json::from_value(json!({"name": "b", "steps": [
            {"id": "a", "command": "knobs", "needs": ["z"]}]}))
        .unwrap();
        assert!(validate(&bad).is_err());
    }

    #[test]
    fn the_worst_code_wins() {
        assert_eq!(worst(&[0, 4, 3]), 3);
        assert_eq!(worst(&[0, 4]), 4);
        assert_eq!(worst(&[3, 1]), 1);
        assert_eq!(worst(&[0, 0]), 0);
    }
}
