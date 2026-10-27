//! `tslab sweep`: several candidates against one baseline, each judged by
//! the paired A/B rule, in one interleaved session.
//!
//! A sweep comes from the install's family profile (`--sweep ID`, or `all`
//! for every sweep whose `when` holds), from a file holding one sweep
//! object (`--sweep-file`), or from the command line (`--baseline A
//! --candidate B --candidate C`). Each candidate gets its own `verdict`
//! record, the same shape `tslab ab` writes, and the sweep gets one `sweep`
//! record naming the best WIN, or none.

use std::collections::BTreeMap;
use std::collections::BTreeSet;

use serde_json::json;

use crate::args::{parse_arm, parse_env, parse_prompt, parse_residency, parse_slots, Args};
use crate::cli::{host_rules_from, memory_check, null_floor, timeout_from};
use crate::family_profile::{ProfileSet, Sweep};
use crate::lock::MachineLock;
use crate::orchestrate::{run_sweep, Session, SweepRun};
use crate::paths::lab_dir;
use crate::report;
use crate::schema::CmdOutput;
use crate::stats;
use crate::workload::{Arm, Metric, OpenParams, PromptSource, Workload};

/// The workload flags the user typed, each optional, laid over a sweep's
/// own workload when it has one.
struct WorkloadOverride {
    prompts: Vec<PromptSource>,
    audio: Option<crate::workload::AudioJob>,
    decode: Option<usize>,
    chunk: Option<Option<usize>>,
    context: Option<u32>,
}

impl WorkloadOverride {
    fn take(args: &mut Args) -> Result<Self, String> {
        let prompts = args
            .take_all("--prompt")?
            .iter()
            .map(|p| parse_prompt(p))
            .collect::<Result<_, _>>()?;
        let audio = match args.take("--audio-job")? {
            Some(path) => {
                let text = std::fs::read_to_string(&path)
                    .map_err(|e| format!("--audio-job {path}: {e}"))?;
                Some(serde_json::from_str(&text).map_err(|e| format!("--audio-job {path}: {e}"))?)
            }
            None => None,
        };
        let chunk = match args.take("--chunk")?.as_deref() {
            None => None,
            Some("off") => Some(None),
            Some(v) => Some(Some(v.parse().map_err(|_| format!("--chunk {v:?}"))?)),
        };
        Ok(Self {
            prompts,
            audio,
            decode: args
                .take("--decode")?
                .map(|v| v.parse().map_err(|_| format!("--decode {v:?}")))
                .transpose()?,
            chunk,
            context: args
                .take("--context")?
                .map(|v| v.parse().map_err(|_| format!("--context {v:?}")))
                .transpose()?,
        })
    }

    fn apply(&self, base: Option<&Workload>) -> Result<Workload, String> {
        let mut w = base.cloned().unwrap_or(Workload {
            prompts: Vec::new(),
            decode_tokens: 0,
            prefill_chunk: Some(128),
            max_context: 4096,
            speculation: None,
            audio: None,
        });
        if !self.prompts.is_empty() {
            w.prompts = self.prompts.clone();
            w.audio = None;
        }
        if let Some(a) = &self.audio {
            w.audio = Some(a.clone());
            w.prompts.clear();
        }
        if let Some(d) = self.decode {
            w.decode_tokens = d;
        }
        if let Some(c) = self.chunk {
            w.prefill_chunk = c;
        }
        if let Some(c) = self.context {
            w.max_context = c;
        }
        if w.prompts.is_empty() && w.audio.is_none() {
            w.prompts
                .push(PromptSource::Protocol("long-synthesis".into()));
        }
        Ok(w)
    }
}

/// One sweep as the command runs it: the arms parsed, the workload settled.
struct Resolved {
    id: String,
    sweep: Option<Sweep>,
    baseline: Arm,
    candidates: Vec<Arm>,
    workload: Workload,
    metric: Metric,
}

fn resolve_one(
    sweep: &Sweep,
    open: OpenParams,
    common: &BTreeMap<String, String>,
    over: &WorkloadOverride,
    metric: Option<Metric>,
) -> Result<Resolved, String> {
    let baseline = parse_arm(&sweep.baseline, open, common)?;
    let candidates = sweep
        .candidates
        .iter()
        .map(|c| parse_arm(c, open, common))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Resolved {
        id: sweep.id.clone(),
        sweep: Some(sweep.clone()),
        baseline,
        candidates,
        workload: over.apply(sweep.workload.as_ref())?,
        metric: metric.unwrap_or(sweep.metric),
    })
}

fn parse_metric(s: &str) -> Result<Metric, String> {
    serde_json::from_value(json!(s)).map_err(|_| format!("metric {s:?}"))
}

pub fn sweep_cmd(mut args: Args) -> Result<CmdOutput, String> {
    let model = args.required("--model")?;
    let experiment = args.take("--experiment")?.unwrap_or_else(|| "sweep".into());
    let over = WorkloadOverride::take(&mut args)?;
    let rules = host_rules_from(&mut args)?;
    let timeout = timeout_from(&mut args)?;
    let common: BTreeMap<String, String> = args
        .take_all("--env")?
        .iter()
        .map(|kv| parse_env(kv))
        .collect::<Result<_, _>>()?;
    let open = OpenParams {
        slots: parse_slots(&args.take("--slots")?.unwrap_or_else(|| "auto".into()))?,
        residency: parse_residency(&args.take("--residency")?.unwrap_or_else(|| "auto".into()))?,
    };
    let sweep_id = args.take("--sweep")?;
    let sweep_file = args.take("--sweep-file")?;
    let dir = args.take("--dir")?;
    let baseline = args.take("--baseline")?;
    let candidates = args.take_all("--candidate")?;
    let metric = args
        .take("--metric")?
        .map(|m| parse_metric(&m))
        .transpose()?;
    let requested_effect: f64 = args.parsed("--min-effect", stats::MIN_EFFECT_FLOOR)?;
    let min_pairs: usize = args.parsed("--pairs", stats::MIN_PAIRS)?;
    let max_pairs: usize = args.parsed("--max-pairs", 6)?;
    let fail_on_loss = args.flag("--fail-on-loss");
    args.finish()?;
    if min_pairs < stats::MIN_PAIRS || max_pairs < min_pairs {
        return Err(format!(
            "--pairs must be >= {} and --max-pairs >= --pairs",
            stats::MIN_PAIRS
        ));
    }

    // Which sweeps run. The install's coverage decides `when` for a profile
    // sweep; an explicit arm list or a file needs no profile.
    let subject = crate::kernel::Subject::open(&model)?;
    let coverage = crate::families::coverage(&subject);
    let tags: BTreeSet<String> = coverage.tags.iter().cloned().collect();
    let family = coverage.family.clone();
    let mut sweeps: Vec<Resolved> = Vec::new();
    match (sweep_id, sweep_file, baseline) {
        (Some(id), None, None) => {
            let set = ProfileSet::resolve(dir.as_deref())?;
            let profile = set
                .by_family(&family)
                .ok_or_else(|| format!("no family profile for {family:?}"))?;
            let (_, applicable) = profile.applicable(&tags);
            if id == "all" {
                if applicable.is_empty() {
                    return Err(format!(
                        "no sweep in the {family} profile applies to this install (tags: {})",
                        coverage.tags.join(" ")
                    ));
                }
                for s in applicable {
                    sweeps.push(resolve_one(s, open, &common, &over, metric)?);
                }
            } else {
                let s = profile.sweep(&id).ok_or_else(|| {
                    format!(
                        "no sweep {id:?} in the {family} profile; it has: {}",
                        profile
                            .sweeps
                            .iter()
                            .map(|s| s.id.as_str())
                            .collect::<Vec<_>>()
                            .join(", ")
                    )
                })?;
                if !applicable.iter().any(|a| a.id == id) {
                    return Err(format!(
                        "sweep {id:?} needs {:?}, which this install's tags ({}) do not hold",
                        s.when,
                        coverage.tags.join(" ")
                    ));
                }
                sweeps.push(resolve_one(s, open, &common, &over, metric)?);
            }
        }
        (None, Some(path), None) => {
            let text =
                std::fs::read_to_string(&path).map_err(|e| format!("--sweep-file {path}: {e}"))?;
            let s: Sweep =
                serde_json::from_str(&text).map_err(|e| format!("--sweep-file {path}: {e}"))?;
            sweeps.push(resolve_one(&s, open, &common, &over, metric)?);
        }
        (None, None, Some(b)) => {
            if candidates.is_empty() {
                return Err("--baseline needs at least one --candidate".into());
            }
            let baseline = parse_arm(&b, open, &common)?;
            let cands = candidates
                .iter()
                .map(|c| parse_arm(c, open, &common))
                .collect::<Result<Vec<_>, _>>()?;
            let mut names = BTreeSet::new();
            for a in std::iter::once(&baseline).chain(cands.iter()) {
                if !names.insert(a.name.clone()) {
                    return Err(format!("arm name {:?} repeats", a.name));
                }
            }
            sweeps.push(Resolved {
                id: "adhoc".into(),
                sweep: None,
                baseline,
                candidates: cands,
                workload: over.apply(None)?,
                metric: metric.unwrap_or(Metric::PrefillTokS),
            });
        }
        (None, None, None) => {
            return Err(
                "give --sweep ID|all, --sweep-file F, or --baseline A --candidate B...".into(),
            )
        }
        _ => return Err("--sweep, --sweep-file and --baseline are exclusive".into()),
    }
    if !candidates.is_empty() && sweeps.iter().any(|s| s.sweep.is_some()) {
        return Err("--candidate goes with --baseline, not with --sweep or --sweep-file".into());
    }

    let lab = lab_dir();
    let _lock = MachineLock::acquire(&lab, &format!("tslab sweep {experiment}"))?;
    let session = Session::new(&lab, &experiment, &model, rules, timeout)?;
    let mut results = Vec::new();
    let mut worst = 0;
    for r in &sweeps {
        let (floor, memory_noise) = null_floor(&session, &r.workload, r.metric)?;
        let min_effect = requested_effect.max(floor);
        eprintln!(
            "tslab sweep {}: {} vs {}, metric {:?}, effect floor {:.2}% (null floor {:.2}%)",
            r.id,
            r.baseline.name,
            r.candidates
                .iter()
                .map(|c| c.name.as_str())
                .collect::<Vec<_>>()
                .join(", "),
            r.metric,
            min_effect.max(stats::MIN_EFFECT_FLOOR) * 100.0,
            floor * 100.0
        );
        let SweepRun { arms, stopped } = run_sweep(
            &session,
            &r.baseline,
            &r.candidates,
            &r.workload,
            r.metric,
            min_effect,
            min_pairs,
            max_pairs,
            None,
        )?;
        let mut cands = Vec::new();
        let mut best: Option<(String, f64)> = None;
        let mut code = 0;
        for arm in &arms {
            let memory = memory_check(&arm.pairs, None, memory_noise);
            let run_ids: Vec<(String, String)> = arm
                .pairs
                .iter()
                .map(|(a, b)| (a.run_id.clone(), b.run_id.clone()))
                .collect();
            report::print_verdict(
                &format!("{} vs {}", r.baseline.name, arm.candidate.name),
                &arm.verdict,
            );
            if let Some(why) = &arm.refused {
                crate::say!("  refused: {why}");
            }
            let rec = json!({"model": model, "workload": r.workload, "metric": r.metric,
                "baseline": r.baseline, "candidate": arm.candidate, "verdict": arm.verdict,
                "null_floor": floor, "memory": memory, "pairs": run_ids,
                "sweep_id": r.id, "family": family});
            let id = session.ledger.append("verdict", rec)?;
            let v = arm.verdict.verdict;
            match v {
                stats::Verdict::Win => {
                    if best.as_ref().is_none_or(|(_, m)| arm.verdict.median > *m) {
                        best = Some((arm.candidate.name.clone(), arm.verdict.median));
                    }
                }
                stats::Verdict::Inconclusive => code = code.max(4),
                stats::Verdict::Loss if fail_on_loss => code = code.max(3),
                _ => {}
            }
            cands.push(json!({
                "arm": arm.candidate.name, "env": arm.candidate.env,
                "speculation": arm.candidate.speculation, "open": arm.candidate.open,
                "verdict": arm.verdict, "memory": memory, "refused": arm.refused,
                "clean_pairs": arm.verdict.pairs, "dirty_pairs": arm.pairs.len() - arm.verdict.pairs,
                "verdict_record": id,
            }));
        }
        if let Some(why) = &stopped {
            crate::say!("  stopped early: {why}");
        }
        match &best {
            Some((arm, m)) => crate::say!("  best: {arm} ({:+.2}% median)", m * 100.0),
            None => crate::say!("  best: none (no candidate won)"),
        }
        let record = json!({
            "sweep_id": r.id, "family": family, "model": model, "install": coverage.install,
            "baseline": r.baseline, "metric": r.metric, "floor": min_effect,
            "null_floor": floor, "exactness": r.sweep.as_ref().map(|s| s.exactness),
            "description": r.sweep.as_ref().map(|s| s.description.clone()),
            "workload": r.workload, "candidates": cands,
            "best": best.as_ref().map(|(arm, m)| json!({"arm": arm, "median": m})),
            "stopped": stopped,
        });
        let id = session.ledger.append("sweep", record.clone())?;
        worst = worst.max(code);
        results.push(json!({"id": id, "sweep": record}));
    }
    Ok(CmdOutput::with_code(
        worst,
        json!({"experiment": experiment, "sweeps": results}),
    ))
}
