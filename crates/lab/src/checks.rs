//! The one-model checks beside the A/B: golden answers (LAB-08), run-to-run
//! noise (LAB-03), the quality floor table (LAB-04), install-vs-install KL at
//! equal bytes (LAB-22), launch counts (LAB-26), toolchain digest anchors
//! (LAB-30), the cross-build release gate (LAB-29) and the pair rule's
//! false-verdict replay (LAB-05).
//!
//! Each runs through the same fresh-process worker as every other number.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::args::{parse_arm, parse_prompt, Args};
use crate::cli::{host_rules_from, timeout_from};
use crate::experiment::{
    compare_traces, decode_trace_workload, quality_gate_arms, traced_run, GateArms,
};
use crate::gates::{judge_quant, GateClass};
use crate::ledger::Ledger;
use crate::lock::MachineLock;
use crate::orchestrate::{Outcome, Session};
use crate::paths::lab_dir;
use crate::schema::CmdOutput;
use crate::stats;
use crate::worker::Mode;
use crate::workload::{Arm, Metric, OpenParams, PromptSource, Workload};

fn session(
    args: &mut Args,
    experiment: &str,
    model: &str,
) -> Result<(Session, MachineLock), String> {
    let rules = host_rules_from(args)?;
    let timeout: Duration = timeout_from(args)?;
    let lab = lab_dir();
    let lock = MachineLock::acquire(&lab, &format!("tslab {experiment}"))?;
    Ok((Session::new(&lab, experiment, model, rules, timeout)?, lock))
}

fn plain_arm(name: &str) -> Arm {
    Arm {
        name: name.into(),
        env: BTreeMap::new(),
        open: OpenParams::default(),
        speculation: None,
        expect_engaged: Vec::new(),
        contention: None,
        ballast_gib: None,
        cold: false,
    }
}

/// `--arm NAME:K=V,...`, or the plain default arm.
fn one_arm(args: &mut Args) -> Result<Arm, String> {
    match args.take("--arm")? {
        Some(a) => parse_arm(&a, OpenParams::default(), &BTreeMap::new()),
        None => Ok(plain_arm("default")),
    }
}

fn workload(prompts: Vec<PromptSource>, decode: usize, context: u32) -> Workload {
    Workload {
        prompts,
        decode_tokens: decode,
        prefill_chunk: Some(128),
        max_context: context,
        speculation: None,
        audio: None,
    }
}

fn ok(o: &Outcome) -> Result<(), String> {
    if o.status == "ok" {
        Ok(())
    } else {
        Err(format!(
            "arm {} was {}: {}",
            o.arm,
            o.status,
            o.record["refusal"].as_str().unwrap_or("(see its stderr)")
        ))
    }
}

fn protocol_prompts() -> Vec<PromptSource> {
    ["short-explanation", "medium-review", "long-synthesis"]
        .iter()
        .map(|c| PromptSource::Protocol((*c).into()))
        .collect()
}

/// The install's family and whether it routes experts.
fn family_of(model: &str) -> Result<(String, bool), String> {
    let arch = repack::peek_manifest_arch(&catalog::resolve_model_arg(model))?;
    Ok((arch.family.as_str().to_string(), arch.num_experts > 0))
}

// ---- sanity (LAB-08) ----

/// `tslab sanity --model M`: golden answers through the chat template, and
/// with `--tools` / `--needle` the functional grid. Exit 1 when an exact
/// answer is missing. With `--candidate`, both arms run and the candidate
/// fails if it loses any case the other arm passes (the functional gate).
pub fn sanity_cmd(mut args: Args) -> Result<CmdOutput, String> {
    use crate::functional::{self, Check, Expect};
    let model = args.required("--model")?;
    let experiment = args
        .take("--experiment")?
        .unwrap_or_else(|| "sanity".into());
    let text = match args.take("--file")? {
        Some(p) => std::fs::read_to_string(&p).map_err(|e| format!("{p}: {e}"))?,
        None => crate::golden::DEFAULT_SET.to_string(),
    };
    let set = crate::golden::parse_set(&text)?;
    let arm = one_arm(&mut args)?;
    let candidate = args
        .take("--candidate")?
        .map(|a| parse_arm(&a, OpenParams::default(), &BTreeMap::new()))
        .transpose()?;
    let with_tools = args.flag("--tools");
    // `--needle 4096,16384` or `--needle all`.
    let needle: Vec<usize> = match args.take("--needle")?.as_deref() {
        None => Vec::new(),
        Some("all") => functional::NEEDLE_SIZES.to_vec(),
        Some(v) => v
            .split(',')
            .map(|n| n.parse().map_err(|_| format!("--needle {v:?}")))
            .collect::<Result<_, _>>()?,
    };
    // Room for a reasoning model to finish thinking and answer.
    let decode: usize = args.parsed("--decode", 512)?;
    let longest = needle.iter().copied().max().unwrap_or(0);
    let context: u32 = args.parsed("--context", 4096.max((longest + decode + 512) as u32))?;
    let (session, _lock) = session(&mut args, &experiment, &model)?;
    args.finish()?;

    let tokenizer = tokenizer::MfTokenizer::load_from_dir(&catalog::resolve_model_arg(&model))
        .map_err(|e| format!("tokenizer: {e}"))?;
    let dir = session.exp_dir.join("golden").join(crate::ledger::new_id());
    std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let mut checks = Vec::new();
    for case in &set.cases {
        let path = dir.join(format!("{}.messages.json", case.id));
        crate::atomic::write_new_json(&path, &json!([{"role": "user", "content": case.prompt}]))?;
        checks.push(Check {
            id: case.id.clone(),
            source: PromptSource::MessagesFile(path),
            expect: Expect::Answer(case.clone()),
        });
    }
    let mut skipped = Vec::new();
    if with_tools {
        match functional::tool_checks(&tokenizer, &dir) {
            Ok(mut tools) => checks.append(&mut tools),
            Err(why) => skipped.push(why),
        }
    }
    if !needle.is_empty() {
        let filler =
            turbospark_bench::quality_corpus::corpus_text(crate::provenance::source_root())?;
        let (mut grid, why) = functional::needle_checks(
            &tokenizer,
            &filler,
            &dir,
            &needle,
            context as usize,
            decode,
        )?;
        checks.append(&mut grid);
        skipped.extend(why);
    }
    for why in &skipped {
        crate::say!("  skipped: {why}");
    }
    let w = workload(
        checks.iter().map(|c| c.source.clone()).collect(),
        decode,
        context,
    );

    let run = |arm: &Arm| -> Result<(Vec<Value>, Value, String), String> {
        let o = session.run_once(arm, &w, Mode::Timing, 0, false)?;
        ok(&o)?;
        let prompts = o.record["prompts"].as_array().cloned().unwrap_or_default();
        if prompts.len() != checks.len() {
            return Err(format!("{} of {} cases ran", prompts.len(), checks.len()));
        }
        let mut cases = Vec::new();
        for (check, p) in checks.iter().zip(&prompts) {
            let ids: Vec<i32> = serde_json::from_value(p["decode_ids"].clone()).unwrap_or_default();
            let text = p["decode_text"].as_str().unwrap_or("");
            let (pass, mut read) = functional::judge(check, &tokenizer, text, &ids);
            crate::say!(
                "  {:<10} {:<22} {}  {}",
                arm.name,
                check.id,
                if pass { "pass" } else { "FAIL" },
                read["answer"]
            );
            read["id"] = json!(check.id);
            read["passed"] = json!(pass);
            cases.push(read);
        }
        Ok((cases, o.record["family"].clone(), o.run_id.clone()))
    };
    let vector = |cases: &[Value]| -> Vec<(String, bool)> {
        cases
            .iter()
            .map(|c| {
                (
                    c["id"].as_str().unwrap_or("").to_string(),
                    c["passed"] == true,
                )
            })
            .collect()
    };

    let (cases, family, run_id) = run(&arm)?;
    let failed = vector(&cases).iter().filter(|c| !c.1).count();
    crate::say!("sanity: {} of {} pass", cases.len() - failed, cases.len());
    let (passed, other) = match &candidate {
        None => (failed == 0, Value::Null),
        Some(c) => {
            let (theirs, _, candidate_run) = run(c)?;
            let diff = functional::pass_vector_difference(&vector(&cases), &vector(&theirs));
            let same = diff["lost"].as_array().is_some_and(Vec::is_empty);
            crate::say!(
                "functional gate: {} {diff}",
                if same { "PASS" } else { "FAIL" }
            );
            (
                same,
                json!({"arm": c.name, "arm_env": c.env, "cases": theirs,
                       "run_id": candidate_run, "difference": diff}),
            )
        }
    };
    let rec = json!({"model": model, "family": family, "arm": arm.name, "arm_env": arm.env,
                     "passed": passed, "cases": cases, "run_id": run_id,
                     "candidate": other, "skipped": skipped});
    session.ledger.append("sanity", rec.clone())?;
    Ok(CmdOutput::with_code(i32::from(!passed), rec))
}

// ---- noise (LAB-03) ----

/// The first index where two id sequences differ, `None` when equal.
pub fn first_divergence(a: &[i64], b: &[i64]) -> Option<usize> {
    a.iter()
        .zip(b)
        .position(|(x, y)| x != y)
        .or_else(|| (a.len() != b.len()).then(|| a.len().min(b.len())))
}

/// `tslab noise --model M --runs K`: the same arm in K fresh processes; how
/// many distinct outputs per prompt, and where they first part.
pub fn noise_cmd(mut args: Args) -> Result<CmdOutput, String> {
    let model = args.required("--model")?;
    let experiment = args.take("--experiment")?.unwrap_or_else(|| "noise".into());
    let arm = one_arm(&mut args)?;
    let runs: usize = args.parsed("--runs", 5)?;
    if !(2..=20).contains(&runs) {
        return Err("--runs is 2..=20".into());
    }
    let mut prompts: Vec<PromptSource> = args
        .take_all("--prompt")?
        .iter()
        .map(|p| parse_prompt(p))
        .collect::<Result<_, _>>()?;
    if prompts.is_empty() {
        prompts = protocol_prompts();
    }
    let decode: usize = args.parsed("--decode", 128)?;
    let context: u32 = args.parsed("--context", 4096)?;
    let (session, _lock) = session(&mut args, &experiment, &model)?;
    args.finish()?;

    let w = workload(prompts, decode, context);
    let mut ids: Vec<Vec<Vec<i64>>> = Vec::new();
    for round in 0..runs {
        let o = session.run_once(&arm, &w, Mode::Timing, round, false)?;
        ok(&o)?;
        ids.push(
            o.record["prompts"]
                .as_array()
                .cloned()
                .unwrap_or_default()
                .iter()
                .map(|p| serde_json::from_value(p["decode_ids"].clone()).unwrap_or_default())
                .collect(),
        );
    }
    let mut prompts_out = Vec::new();
    for i in 0..w.prompts.len() {
        let seen: BTreeSet<&Vec<i64>> = ids.iter().filter_map(|r| r.get(i)).collect();
        let first = ids
            .iter()
            .skip(1)
            .filter_map(|r| first_divergence(&ids[0][i], &r[i]))
            .min();
        crate::say!(
            "  prompt {i}: {} distinct outputs over {runs} runs{}",
            seen.len(),
            first.map_or(String::new(), |f| format!(
                ", first divergence at token {f}"
            ))
        );
        prompts_out.push(json!({"digest_distinct_count": seen.len(),
                                "first_divergence": first,
                                "tokens": ids[0].get(i).map(Vec::len)}));
    }
    let rec = json!({"model": model, "arm": arm, "runs": runs, "workload": w,
                     "prompts": prompts_out});
    session.ledger.append("noise", rec.clone())?;
    Ok(CmdOutput::ok(rec))
}

/// The share of prompts on which the newest `tslab noise` record for `model`
/// saw more than one output. Zero with no record: nothing is excused without
/// a measurement.
pub fn noise_floor(lab: &Path, model: &str) -> f64 {
    let records = crate::paths::experiment_dir(lab, "noise")
        .and_then(|d| Ledger::open(&d))
        .and_then(|l| l.of_kind("noise"))
        .unwrap_or_default();
    records
        .iter()
        .rev()
        .find(|r| r["model"] == model)
        .and_then(|r| {
            let p = r["prompts"].as_array()?;
            let parted = p
                .iter()
                .filter(|p| p["digest_distinct_count"].as_u64().unwrap_or(1) > 1)
                .count();
            Some(parted as f64 / p.len().max(1) as f64)
        })
        .unwrap_or(0.0)
}

// ---- spec-rounds (LAB-23, LAB-25, LAB-13) ----

/// The per-position acceptance chance `p` whose expected committed length is
/// `len`: `p + p^2 + .. + p^block = len - 1`.
pub fn acceptance_for_length(len: f64, block: usize) -> Option<f64> {
    if !(1.0..=1.0 + block as f64).contains(&len) {
        return None;
    }
    let sum = |p: f64| (1..=block).map(|k| p.powi(k as i32)).sum::<f64>();
    let (mut lo, mut hi) = (0.0, 1.0);
    for _ in 0..60 {
        let mid = (lo + hi) / 2.0;
        if sum(mid) < len - 1.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    Some((lo + hi) / 2.0)
}

/// Median verify ms by (width, seconds into the run), the busy-age split.
pub fn rounds_by_age(rounds: &[Value]) -> Value {
    const AGES: [(&str, f64); 4] = [
        ("0-6s", 6.0),
        ("6-15s", 15.0),
        ("15-60s", 60.0),
        ("60s+", f64::INFINITY),
    ];
    let mut cells: BTreeMap<(u64, usize), Vec<f64>> = BTreeMap::new();
    for r in rounds {
        let (Some(width), Some(at), Some(ms)) = (
            r["width"].as_u64(),
            r["at_s"].as_f64(),
            r["verify_ms"].as_f64(),
        ) else {
            continue;
        };
        let age = AGES.iter().position(|a| at < a.1).unwrap_or(AGES.len() - 1);
        cells.entry((width, age)).or_default().push(ms);
    }
    Value::Array(
        cells
            .into_iter()
            .map(|((width, age), ms)| {
                json!({"width": width, "age": AGES[age].0, "rounds": ms.len(),
                       "verify_ms": stats::median(&ms)})
            })
            .collect(),
    )
}

/// `tslab spec-rounds --model M --arm NAME~mtp:4 [--synth-len 1.5,2,3]`: the
/// speculative loop under the runtime's probe. With `--synth-len`, acceptance
/// is forced to that mean committed length (output wrong by design).
pub fn spec_rounds_cmd(mut args: Args) -> Result<CmdOutput, String> {
    let model = args.required("--model")?;
    let experiment = args
        .take("--experiment")?
        .unwrap_or_else(|| "spec_rounds".into());
    let arm = parse_arm(
        &args.required("--arm")?,
        OpenParams::default(),
        &BTreeMap::new(),
    )?;
    let block = arm
        .speculation
        .as_ref()
        .map(|s| s.block)
        .ok_or("--arm needs a drafter, e.g. mtp~mtp:4")?;
    let lengths: Vec<f64> = match args.take("--synth-len")? {
        Some(v) => v
            .split(',')
            .map(|n| n.parse().map_err(|_| format!("--synth-len {v:?}")))
            .collect::<Result<_, _>>()?,
        None => Vec::new(),
    };
    let decode: usize = args.parsed("--decode", 256)?;
    let context: u32 = args.parsed("--context", 4096)?;
    let prompt = match args.take("--prompt")? {
        Some(p) => parse_prompt(&p)?,
        None => PromptSource::Protocol("long-synthesis".into()),
    };
    let (session, _lock) = session(&mut args, &experiment, &model)?;
    args.finish()?;
    let mut plan: Vec<(Option<f64>, Option<f64>)> = vec![(None, None)];
    for len in lengths {
        let p = acceptance_for_length(len, block)
            .ok_or_else(|| format!("--synth-len {len}: within 1..{}", block + 1))?;
        plan.push((Some(len), Some(p)));
    }
    let w = workload(vec![prompt], decode, context);
    let mut rows = Vec::new();
    for (target, chance) in plan {
        let o = session.run_probe(&arm, &w, chance)?;
        ok(&o)?;
        let probe = &o.record["prompts"][0]["spec_probe"];
        let rounds = probe["rounds"].as_array().cloned().unwrap_or_default();
        if rounds.is_empty() {
            return Err("the probe recorded no speculative round".into());
        }
        let total = |k: &str| rounds.iter().filter_map(|r| r[k].as_f64()).sum::<f64>();
        let n = rounds.len() as f64;
        let committed = 1.0 + total("accepted") / n;
        let verify: Vec<f64> = rounds
            .iter()
            .filter_map(|r| r["verify_ms"].as_f64())
            .collect();
        let accepted = total("accepted") + n;
        crate::say!(
            "  {}: {} rounds, {committed:.2} committed/round, verify median {:.1} ms{}",
            target.map_or("model acceptance".into(), |t| format!("synthetic L={t}")),
            rounds.len(),
            stats::median(&verify),
            if total("expert_requests") > 0.0 {
                format!(
                    ", {:.1} expert reads and {:.2} MiB per committed token",
                    total("expert_requests") / accepted,
                    total("expert_bytes") / accepted / (1 << 20) as f64
                )
            } else {
                String::new()
            }
        );
        rows.push(
            json!({"synthetic_length": target, "synthetic_chance": chance,
                         "rounds": rounds.len(), "committed_per_round": committed,
                         "verify_ms_median": stats::median(&verify),
                         "expert_requests": total("expert_requests"),
                         "expert_bytes": total("expert_bytes"),
                         "by_age": rounds_by_age(&rounds),
                         "decode_s": o.record["totals"]["decode_s"],
                         "run_id": o.run_id}),
        );
    }
    let rec = json!({"model": model, "arm": arm, "block": block, "rows": rows,
                     "quotable": false});
    session.ledger.append("spec_rounds", rec.clone())?;
    Ok(CmdOutput::ok(rec))
}

// ---- floor (LAB-04) ----

/// `tslab floor --model M`: the quality gate of an arm against ITSELF, keyed
/// by family. A floor only explains a bound or argues for a tighter one; no
/// command widens a bound from it.
pub fn floor_cmd(mut args: Args) -> Result<CmdOutput, String> {
    let model = args.required("--model")?;
    let arm = one_arm(&mut args)?;
    let context: u32 = args.parsed("--context", 4096)?;
    let (family, routes) = family_of(&model)?;
    let (session, _lock) = session(&mut args, "floor", &model)?;
    args.finish()?;
    let again = Arm {
        name: format!("{}-again", arm.name),
        ..arm.clone()
    };
    let class = GateClass::kl_class_for(routes);
    let q = quality_gate_arms(
        &session,
        &GateArms {
            baseline: &arm,
            candidate: &again,
            workload: &workload(
                vec![PromptSource::Protocol("short-explanation".into())],
                crate::gates::DECODE_TRACE_TOKENS,
                context,
            ),
            exactness: class,
            quality_bounds: None,
        },
    )?;
    let decode = q.decode.as_ref().map(|d| &d.overall);
    let bits = q.all_bits_equal && decode.is_none_or(|d| d.all_bits_equal);
    crate::say!(
        "floor {family}: bits identical {bits}, prefill mean KL {:.3e}, top-1 {:.2}",
        q.mean_kl,
        q.top1_fraction
    );
    if !bits {
        crate::say!("  this family's baseline is NOT run-to-run exact");
    }
    let rec = json!({"family": family, "model": model, "arm": arm, "class": class,
                     "bits_identical": bits, "mean_kl": q.mean_kl,
                     "top1_fraction": q.top1_fraction, "decode": decode});
    session.ledger.append("floor", rec.clone())?;
    Ok(CmdOutput::ok(rec))
}

/// The newest floor recorded for a family, if any.
pub fn latest_floor(lab: &Path, family: &str) -> Option<Value> {
    let ledger = Ledger::open(&crate::paths::experiment_dir(lab, "floor").ok()?).ok()?;
    ledger
        .of_kind("floor")
        .ok()?
        .into_iter()
        .rev()
        .find(|r| r["family"] == family)
}

// ---- quant-ab (LAB-22) ----

fn dir_bytes(dir: &Path) -> Result<u64, String> {
    let mut total = 0;
    for entry in std::fs::read_dir(dir).map_err(|e| format!("{}: {e}", dir.display()))? {
        let path = entry.map_err(|e| e.to_string())?.path();
        total += if path.is_dir() {
            dir_bytes(&path)?
        } else {
            std::fs::metadata(&path).map_err(|e| e.to_string())?.len()
        };
    }
    Ok(total)
}

/// `tslab quant-ab --ref M --base M --cand M [--cand M]`: every install
/// replays the reference's token ids, and a candidate must not predict the
/// reference worse than the base install at no more bytes.
pub fn quant_ab_cmd(mut args: Args) -> Result<CmdOutput, String> {
    let reference = args.required("--ref")?;
    let base = args.required("--base")?;
    let candidates = args.take_all("--cand")?;
    if candidates.is_empty() {
        return Err("--cand is required".into());
    }
    let experiment = args
        .take("--experiment")?
        .unwrap_or_else(|| "quant_ab".into());
    let context: u32 = args.parsed("--context", 4096)?;
    let rules = host_rules_from(&mut args)?;
    let timeout = timeout_from(&mut args)?;
    args.finish()?;
    let lab = lab_dir();
    let _lock = MachineLock::acquire(&lab, "tslab quant-ab")?;
    let session_for = |model: &str| Session::new(&lab, &experiment, model, rules, timeout);

    let w = decode_trace_workload(&workload(Vec::new(), 0, context));
    let ref_session = session_for(&reference)?;
    let r = traced_run(&ref_session, &plain_arm("ref"), &w, None)?;
    let compare = |model: &str, name: &str| -> Result<Vec<crate::kl_stats::Position>, String> {
        let s = session_for(model)?;
        let t = traced_run(&s, &plain_arm(name), &w, Some(r.ids.clone()))?;
        // Equal prompt lengths are the cheap check that both installs
        // tokenize the corpus alike; ids replayed across tokenizers would be
        // noise.
        if t.prompt_tokens != r.prompt_tokens {
            return Err(format!(
                "{model} tokenizes the corpus differently from the reference"
            ));
        }
        let compared = compare_traces(&r.trace, &t.trace, &r.prompt_tokens, &r.ids);
        let _ = std::fs::remove_file(&t.trace);
        Ok(compared?.into_iter().flat_map(|(_, p)| p).collect())
    };
    let result = (|| {
        let base_positions = compare(&base, "base")?;
        let base_summary = crate::kl_stats::summarize(&base_positions);
        let base_bytes = dir_bytes(&catalog::resolve_model_arg(&base))?;
        let mut out = Vec::new();
        for (i, model) in candidates.iter().enumerate() {
            let positions = compare(model, &format!("cand{i}"))?;
            let summary = crate::kl_stats::summarize(&positions);
            let bytes = dir_bytes(&catalog::resolve_model_arg(model))?;
            // Paired against the base on the same positions.
            let paired: Vec<crate::kl_stats::Position> = base_positions
                .iter()
                .zip(&positions)
                .map(|(b, c)| crate::kl_stats::Position {
                    ref_logp: b.arm_logp,
                    ..*c
                })
                .collect();
            let paired = crate::kl_stats::summarize(&paired);
            let verdict = judge_quant(base_summary.mean_kl, base_bytes, summary.mean_kl, bytes);
            crate::say!(
                "  {model}: KL(ref) {:.3e} against base {:.3e}, {} bytes against {}, NLL against base {:+.4} +/- {:.4}: {}",
                summary.mean_kl,
                base_summary.mean_kl,
                bytes,
                base_bytes,
                paired.nll_delta,
                paired.nll_delta_se,
                if verdict.passed { "PASS" } else { "FAIL" }
            );
            out.push(json!({"model": model, "summary": summary, "bytes": bytes,
                            "against_base": {"nll_delta": paired.nll_delta,
                                             "nll_delta_se": paired.nll_delta_se},
                            "verdict": verdict}));
        }
        Ok::<_, String>(json!({"reference": reference, "base": base,
                               "base_summary": base_summary, "base_bytes": base_bytes,
                               "positions": base_positions.len(), "candidates": out}))
    })();
    let _ = std::fs::remove_file(&r.trace);
    let rec = result?;
    ref_session.ledger.append("quant_ab", rec.clone())?;
    let passed = rec["candidates"]
        .as_array()
        .is_some_and(|c| c.iter().all(|c| c["verdict"]["passed"] == true));
    Ok(CmdOutput::with_code(i32::from(!passed), rec))
}

// ---- launch-count (LAB-26) ----

/// `tslab launch-count --model M [--arm ...]`: dispatches and command
/// buffers per decoded token, the pipelines used, and with
/// `--us-per-dispatch` (from `profile --dispatch-probe`) the most a
/// launch-reduction item could save.
pub fn launch_count_cmd(mut args: Args) -> Result<CmdOutput, String> {
    let model = args.required("--model")?;
    let experiment = args
        .take("--experiment")?
        .unwrap_or_else(|| "launch_count".into());
    let mut arms = args
        .take_all("--arm")?
        .iter()
        .map(|a| parse_arm(a, OpenParams::default(), &BTreeMap::new()))
        .collect::<Result<Vec<_>, _>>()?;
    if arms.is_empty() {
        arms.push(plain_arm("default"));
    }
    let prompt = match args.take("--prompt")? {
        Some(p) => parse_prompt(&p)?,
        None => PromptSource::Protocol("short-explanation".into()),
    };
    let decode: usize = args.parsed("--decode", 32)?;
    let context: u32 = args.parsed("--context", 4096)?;
    let us: Option<f64> = args
        .take("--us-per-dispatch")?
        .map(|v| v.parse().map_err(|_| format!("--us-per-dispatch {v:?}")))
        .transpose()?;
    let (session, _lock) = session(&mut args, &experiment, &model)?;
    args.finish()?;

    let w = workload(vec![prompt], decode, context);
    let mut rows: Vec<Value> = Vec::new();
    let mut default_routes: Option<BTreeSet<String>> = None;
    for mut arm in arms {
        arm.env.insert("TURBOSPARK_LAUNCH_COUNT".into(), "1".into());
        let o = session.run_once(&arm, &w, Mode::Timing, 0, false)?;
        ok(&o)?;
        let l = &o.record["prompts"][0]["launches"];
        if l.is_null() {
            return Err("the worker counted no launches; rebuild tslab-worker".into());
        }
        let per_token = l["decode_dispatches_per_token"].as_f64().unwrap_or(0.0);
        let routes: BTreeSet<String> =
            serde_json::from_value(l["decode"]["routes"].clone()).unwrap_or_default();
        let first = default_routes.get_or_insert_with(|| routes.clone());
        let added: Vec<&String> = routes.difference(first).collect();
        let removed: Vec<&String> = first.difference(&routes).collect();
        crate::say!(
            "  {:<12} {per_token:.1} dispatches and {:.1} command buffers per decoded token, {} pipelines{}",
            arm.name,
            l["decode_command_buffers_per_token"].as_f64().unwrap_or(0.0),
            routes.len(),
            us.map_or(String::new(), |us| format!(
                "; {:.2} ms/token if every dispatch were free",
                per_token * us / 1e3
            ))
        );
        for r in &added {
            crate::say!("    + {r}");
        }
        for r in &removed {
            crate::say!("    - {r}");
        }
        rows.push(json!({"arm": arm.name, "arm_env": arm.env, "launches": l,
                         "routes_added": added, "routes_removed": removed}));
    }
    let rec = json!({"model": model, "workload": w, "us_per_dispatch": us, "arms": rows,
                     "quotable": false});
    session.ledger.append("launch_count", rec.clone())?;
    Ok(CmdOutput::ok(rec))
}

// ---- idle-gap (LAB-24) ----

/// The first step after each gap against the first step after the shortest
/// one; the largest increase, as a fraction.
pub fn idle_penalty(rows: &Value) -> Option<f64> {
    let first: Vec<f64> = rows
        .as_array()?
        .iter()
        .filter_map(|r| r["first_ms"].as_f64())
        .collect();
    let base = *first.first()?;
    (base > 0.0 && first.len() > 1).then(|| {
        first[1..]
            .iter()
            .map(|v| v / base - 1.0)
            .fold(f64::MIN, f64::max)
    })
}

/// A gap that costs the next token less than this is not worth chasing.
pub const IDLE_PENALTY_FLOOR: f64 = 0.20;

/// `tslab idle-gap --model M`: sleep 0.5 to 8 s between short decodes.
pub fn idle_gap_cmd(mut args: Args) -> Result<CmdOutput, String> {
    let model = args.required("--model")?;
    let experiment = args
        .take("--experiment")?
        .unwrap_or_else(|| "idle_gap".into());
    let arm = one_arm(&mut args)?;
    let context: u32 = args.parsed("--context", 1024)?;
    let (session, _lock) = session(&mut args, &experiment, &model)?;
    args.finish()?;
    let w = workload(
        vec![PromptSource::Protocol("short-explanation".into())],
        0,
        context,
    );
    let o = session.run_once(&arm, &w, Mode::IdleGap, 0, false)?;
    ok(&o)?;
    let rows = o.record["idle_gap"].clone();
    for r in rows.as_array().into_iter().flatten() {
        crate::say!(
            "  gap {:>4} s: first step {:.1} ms (busy {:.1}), its first buffer waited {:.2} ms for the GPU (busy {:.2})",
            r["gap_s"],
            r["first_ms"].as_f64().unwrap_or(f64::NAN),
            r["busy_median_ms"].as_f64().unwrap_or(f64::NAN),
            r["first_submit_ms"].as_f64().unwrap_or(f64::NAN),
            r["busy_submit_median_ms"].as_f64().unwrap_or(f64::NAN)
        );
    }
    let penalty = idle_penalty(&rows);
    let worth = penalty.is_some_and(|p| p > IDLE_PENALTY_FLOOR);
    crate::say!(
        "  worst first-step increase over the 0.5 s gap: {}: {}",
        penalty.map_or("unmeasured".into(), |p| format!("{:+.1}%", p * 100.0)),
        if worth {
            "an idle gap costs something here"
        } else {
            "no gap worth chasing"
        }
    );
    let rec = json!({"model": model, "arm": arm, "rows": rows, "penalty": penalty,
                     "worth_chasing": worth, "eligible": o.eligible, "reasons": o.reasons});
    session.ledger.append("idle_gap", rec.clone())?;
    Ok(CmdOutput::ok(rec))
}

// ---- anchor (LAB-30) ----

/// One anchored reading.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Anchor {
    pub git_rev: String,
    pub decode_digests: Vec<String>,
    pub prefill_digests: Vec<String>,
}

/// `family | toolchain` to the digests recorded there.
pub type Anchors = BTreeMap<String, Anchor>;

/// The digests that changed against the anchor for this key, or against the
/// same family under another toolchain when the key is new.
pub fn compare_anchor(anchors: &Anchors, family: &str, toolchain: &str, now: &Anchor) -> Value {
    let key = format!("{family} | {toolchain}");
    let (basis, old) = match anchors.get(&key) {
        Some(a) => ("same_toolchain", Some(a)),
        None => (
            "other_toolchain",
            anchors
                .iter()
                .find(|(k, _)| k.split(" | ").next() == Some(family))
                .map(|(_, a)| a),
        ),
    };
    let Some(old) = old else {
        return json!({"basis": "none", "changed": []});
    };
    let mut changed = Vec::new();
    for (what, a, b) in [
        ("decode", &old.decode_digests, &now.decode_digests),
        ("prefill_logits", &old.prefill_digests, &now.prefill_digests),
    ] {
        for i in 0..a.len().max(b.len()) {
            if a.get(i) != b.get(i) {
                changed.push(format!("{what} prompt {i}"));
            }
        }
    }
    json!({"basis": basis, "anchored_at": old.git_rev, "changed": changed})
}

/// `tslab anchor --model M --file anchors.json [--update]`: greedy digests on
/// three fixed prompts, keyed by family and toolchain. Exit 3 lists changes
/// for a person to read: a compiler bump may legitimately move float order,
/// which the KL gate judges, not this.
pub fn anchor_cmd(mut args: Args) -> Result<CmdOutput, String> {
    let model = args.required("--model")?;
    let file = PathBuf::from(args.required("--file")?);
    let update = args.flag("--update");
    let (family, _) = family_of(&model)?;
    let (session, _lock) = session(&mut args, "anchor", &model)?;
    args.finish()?;
    let mut anchors: Anchors = match std::fs::read_to_string(&file) {
        Ok(t) => serde_json::from_str(&t).map_err(|e| format!("{}: {e}", file.display()))?,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Anchors::new(),
        Err(e) => return Err(format!("{}: {e}", file.display())),
    };
    let w = workload(protocol_prompts(), 64, 4096);
    let o = session.run_once(&plain_arm("anchor"), &w, Mode::Timing, 0, false)?;
    ok(&o)?;
    let toolchain = o.record["provenance"]["toolchain_id"]
        .as_str()
        .ok_or("the worker recorded no toolchain; rebuild tslab-worker")?
        .to_string();
    let digests = |field: &str| -> Vec<String> {
        o.record["prompts"]
            .as_array()
            .map(|p| {
                p.iter()
                    .map(|p| p[field].as_str().unwrap_or("").to_string())
                    .collect()
            })
            .unwrap_or_default()
    };
    let now = Anchor {
        git_rev: o.record["provenance"]["git_rev"]
            .as_str()
            .unwrap_or("unrecorded")
            .to_string(),
        decode_digests: digests("decode_digest"),
        prefill_digests: digests("prefill_logits_digest"),
    };
    let compared = compare_anchor(&anchors, &family, &toolchain, &now);
    crate::say!("anchor {family} under {toolchain}: {compared}");
    if update {
        anchors.insert(format!("{family} | {toolchain}"), now.clone());
        let text = serde_json::to_string_pretty(&anchors).map_err(|e| e.to_string())?;
        std::fs::write(&file, text + "\n").map_err(|e| format!("{}: {e}", file.display()))?;
        crate::say!("  wrote {}", file.display());
    }
    let rec = json!({"model": model, "family": family, "toolchain_id": toolchain,
                     "anchor": now, "compared": compared, "updated": update});
    session.ledger.append("anchor", rec.clone())?;
    let changed = compared["changed"]
        .as_array()
        .is_some_and(|c| !c.is_empty());
    Ok(CmdOutput::with_code(
        if changed && !update { 3 } else { 0 },
        rec,
    ))
}

// ---- release-gate (LAB-29) ----

/// A change the release declares: a family (`*` suffix matches a prefix) and
/// why its output or speed may differ.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Declared {
    pub family: String,
    pub reason: String,
}

pub fn declared_reason<'a>(declared: &'a [Declared], family: &str) -> Option<&'a str> {
    declared
        .iter()
        .find(|d| match d.family.strip_suffix('*') {
            Some(prefix) => family.starts_with(prefix),
            None => d.family == family,
        })
        .filter(|d| !d.reason.trim().is_empty())
        .map(|d| d.reason.as_str())
}

/// The row's outcome. A slower candidate fails unless declared; changed
/// tokens never block alone (float order moves them), they send the row to
/// the KL gate unless declared.
pub fn release_status(
    verdict: stats::Verdict,
    digests_equal: bool,
    declared: Option<&str>,
) -> &'static str {
    use stats::Verdict::*;
    match (verdict, digests_equal, declared) {
        (Inconclusive, _, _) => "INCONCLUSIVE",
        (Loss, _, None) => "FAIL",
        (_, true, _) if verdict != Loss => "PASS",
        (_, _, Some(_)) => "PASS-with-note",
        (_, false, None) => "NEEDS-KL-GATE",
        _ => "PASS",
    }
}

/// `tslab release-gate --base-worker PATH --model M [--model M] [--declared
/// FILE]`: the same plain arm on a released build and on this one,
/// interleaved, one install at a time.
pub fn release_gate_cmd(mut args: Args) -> Result<CmdOutput, String> {
    let base_worker = PathBuf::from(args.required("--base-worker")?);
    if !base_worker.is_file() {
        return Err(format!("--base-worker {} not found", base_worker.display()));
    }
    let models = args.take_all("--model")?;
    if models.is_empty() {
        return Err("--model is required".into());
    }
    let declared: Vec<Declared> = match args.take("--declared")? {
        Some(p) => {
            serde_json::from_str(&std::fs::read_to_string(&p).map_err(|e| format!("{p}: {e}"))?)
                .map_err(|e| format!("{p}: {e}"))?
        }
        None => Vec::new(),
    };
    let metric: Metric = serde_json::from_value(json!(args
        .take("--metric")?
        .unwrap_or_else(|| "decode_tok_s".into())))
    .map_err(|e| format!("--metric: {e}"))?;
    let pairs: usize = args.parsed("--pairs", stats::MIN_PAIRS)?;
    if pairs < stats::MIN_PAIRS {
        return Err(format!("--pairs must be >= {}", stats::MIN_PAIRS));
    }
    let rules = host_rules_from(&mut args)?;
    let timeout = timeout_from(&mut args)?;
    args.finish()?;
    let lab = lab_dir();
    let _lock = MachineLock::acquire(&lab, "tslab release-gate")?;
    let w = workload(protocol_prompts(), 64, 4096);

    let mut rows = Vec::new();
    for model in &models {
        let head = Session::new(&lab, "release_gate", model, rules, timeout)?;
        let base = Session::new(&lab, "release_gate", model, rules, timeout)?
            .with_worker(base_worker.clone());
        let arm = plain_arm("release");
        let row = release_row(&head, &base, &arm, &w, metric, pairs)?;
        let family = row["family"].as_str().unwrap_or("").to_string();
        let reason = declared_reason(&declared, &family);
        let status = match (row["skip"].as_str(), row["verdict"]["verdict"].clone()) {
            (Some(_), _) => "SKIP",
            (None, v) => release_status(
                serde_json::from_value(v).map_err(|e| e.to_string())?,
                row["digests_equal"] == true,
                reason,
            ),
        };
        crate::say!("  {model} ({family}): {status} {}", reason.unwrap_or(""));
        let mut row = row;
        row["status"] = json!(status);
        row["declared"] = json!(reason);
        rows.push(row);
    }
    let failed = rows.iter().any(|r| r["status"] == "FAIL");
    let rec = json!({"base_worker": base_worker.display().to_string(),
                     "base_worker_sha256": crate::provenance::sha256_file(&base_worker),
                     "metric": metric, "rows": rows, "passed": !failed});
    Ledger::open(&crate::paths::experiment_dir(&lab, "release_gate")?)?
        .append("release_gate", rec.clone())?;
    Ok(CmdOutput::with_code(i32::from(failed), rec))
}

fn release_row(
    head: &Session,
    base: &Session,
    arm: &Arm,
    w: &Workload,
    metric: Metric,
    pairs: usize,
) -> Result<Value, String> {
    // Discarded warmup of each build.
    for s in [base, head] {
        let o = s.run_once(arm, w, Mode::Timing, 0, true)?;
        if o.status != "ok" {
            return Ok(json!({"model": s.model, "skip": format!("{} ({})",
                o.status, o.record["refusal"].as_str().unwrap_or("see stderr"))}));
        }
    }
    let digests = |o: &Outcome| -> Vec<Value> {
        o.record["prompts"]
            .as_array()
            .map(|p| p.iter().map(|p| p["decode_digest"].clone()).collect())
            .unwrap_or_default()
    };
    let mut improvements = Vec::new();
    let (mut a_values, mut b_values) = (Vec::new(), Vec::new());
    let mut equal = true;
    let mut family = Value::Null;
    // Dirty pairs are replaced, never more than doubling the budget.
    for round in 1..=pairs * 2 {
        if improvements.len() >= pairs {
            break;
        }
        let (a, b) = if round % 2 == 1 {
            let a = base.run_once(arm, w, Mode::Timing, round, false)?;
            (a, head.run_once(arm, w, Mode::Timing, round, false)?)
        } else {
            let b = head.run_once(arm, w, Mode::Timing, round, false)?;
            (base.run_once(arm, w, Mode::Timing, round, false)?, b)
        };
        ok(&a)?;
        ok(&b)?;
        family = b.record["family"].clone();
        equal &= digests(&a) == digests(&b);
        if a.eligible && b.eligible {
            if let (Some(x), Some(y)) = (a.metric(metric), b.metric(metric)) {
                a_values.push(x);
                b_values.push(y);
                improvements.extend(stats::improvement(x, y, metric.direction()));
            }
        }
    }
    // The floor is 2% plus the wider arm's own relative spread.
    let spread = |v: &[f64]| {
        if v.is_empty() {
            return 0.0;
        }
        let lo = v.iter().copied().fold(f64::INFINITY, f64::min);
        let hi = v.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        (hi - lo) / stats::median(v)
    };
    let floor = stats::MIN_EFFECT_FLOOR + spread(&a_values).max(spread(&b_values));
    let verdict = stats::decide(&improvements, floor);
    Ok(
        json!({"model": head.model, "family": family, "verdict": verdict,
              "digests_equal": equal, "floor": floor}),
    )
}

// ---- type1 (LAB-05a) ----

/// `tslab type1`: replay every stored `null_calibration` through the pair
/// rule at several pair counts and report the false WIN and LOSS rates.
pub fn type1_cmd(mut args: Args) -> Result<CmdOutput, String> {
    let trials: usize = args.parsed("--trials", 20_000)?;
    args.finish()?;
    let lab = lab_dir();
    let mut pool = Vec::new();
    let mut calibrations = 0;
    for entry in std::fs::read_dir(&lab).map_err(|e| format!("{}: {e}", lab.display()))? {
        let dir = entry.map_err(|e| e.to_string())?.path();
        if !dir.join("ledger.jsonl").is_file() {
            continue;
        }
        // Reading only: a torn line in another experiment is not this
        // command's to refuse.
        let Ok(records) = Ledger::open(&dir)?.of_kind("null_calibration") else {
            eprintln!("  skipped {}: unreadable ledger", dir.display());
            continue;
        };
        for r in records {
            let v: Vec<f64> = serde_json::from_value(r["improvements"].clone()).unwrap_or_default();
            if !v.is_empty() {
                calibrations += 1;
                pool.extend(v);
            }
        }
    }
    if pool.is_empty() {
        return Err(format!(
            "no null_calibration records under {}",
            lab.display()
        ));
    }
    let abs: Vec<f64> = pool.iter().map(|v| v.abs()).collect();
    crate::say!(
        "{} null pair improvements from {calibrations} calibrations, median |x| {:.2}%",
        pool.len(),
        stats::median(&abs) * 100.0
    );
    let mut rows = Vec::new();
    for n in 3..=6 {
        let (win, loss) = stats::type1_rate(&pool, n, stats::MIN_EFFECT_FLOOR, trials);
        crate::say!(
            "  n={n}: false WIN {:.2}%, false LOSS {:.2}% (sign rule alone {:.2}%)",
            win * 100.0,
            loss * 100.0,
            stats::sign_rule_false_win(n) * 100.0
        );
        rows.push(json!({"pairs": n, "false_win": win, "false_loss": loss}));
    }
    let rec = json!({"calibrations": calibrations, "pool": pool.len(), "trials": trials,
                     "effect_floor": stats::MIN_EFFECT_FLOOR, "rows": rows});
    Ok(CmdOutput::ok(rec))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn divergence_is_the_first_differing_or_missing_token() {
        assert_eq!(first_divergence(&[1, 2, 3], &[1, 2, 3]), None);
        assert_eq!(first_divergence(&[1, 2, 3], &[1, 9, 3]), Some(1));
        assert_eq!(first_divergence(&[1, 2], &[1, 2, 3]), Some(2));
    }

    #[test]
    fn anchors_compare_within_a_toolchain_before_across_one() {
        let anchor = |d: &str| Anchor {
            git_rev: "r".into(),
            decode_digests: vec![d.into()],
            prefill_digests: vec!["p".into()],
        };
        let mut a = Anchors::new();
        assert_eq!(
            compare_anchor(&a, "qwen", "t1", &anchor("x"))["basis"],
            "none"
        );
        a.insert("qwen | t1".into(), anchor("x"));
        let same = compare_anchor(&a, "qwen", "t1", &anchor("x"));
        assert_eq!(same["basis"], "same_toolchain");
        assert_eq!(same["changed"].as_array().unwrap().len(), 0);
        let moved = compare_anchor(&a, "qwen", "t2", &anchor("y"));
        assert_eq!(moved["basis"], "other_toolchain");
        assert_eq!(moved["changed"][0], "decode prompt 0");
        // Another family's anchor is not a basis.
        assert_eq!(
            compare_anchor(&a, "qwen3", "t1", &anchor("y"))["basis"],
            "none"
        );
    }

    #[test]
    fn the_idle_penalty_is_read_against_the_shortest_gap() {
        let rows = json!([{"first_ms": 10.0}, {"first_ms": 11.0}, {"first_ms": 14.0}]);
        assert!((idle_penalty(&rows).unwrap() - 0.4).abs() < 1e-9);
        assert_eq!(idle_penalty(&json!([{"first_ms": 10.0}])), None);
        assert_eq!(idle_penalty(&json!(null)), None);
    }

    #[test]
    fn the_acceptance_chance_reproduces_the_asked_length() {
        let p = acceptance_for_length(2.5, 3).unwrap();
        assert!((1.0 + p + p * p + p * p * p - 2.5).abs() < 1e-6, "{p}");
        assert!(acceptance_for_length(1.0, 3).unwrap() < 1e-6);
        assert!(acceptance_for_length(4.0, 3).unwrap() > 0.999);
        assert_eq!(acceptance_for_length(4.5, 3), None);
        assert_eq!(acceptance_for_length(0.5, 3), None);
    }

    #[test]
    fn rounds_split_by_width_and_age() {
        let r = |width: u64, at: f64, ms: f64| json!({"width": width, "at_s": at, "verify_ms": ms});
        let t = rounds_by_age(&[
            r(2, 1.0, 10.0),
            r(2, 2.0, 12.0),
            r(2, 70.0, 20.0),
            r(4, 1.0, 15.0),
        ]);
        assert_eq!(
            t[0],
            json!({"width": 2, "age": "0-6s", "rounds": 2, "verify_ms": 11.0})
        );
        assert_eq!(t[1]["age"], "60s+");
        assert_eq!(t[2]["width"], 4);
    }

    #[test]
    fn release_status_blocks_an_undeclared_loss_only() {
        use stats::Verdict::*;
        assert_eq!(release_status(Noise, true, None), "PASS");
        assert_eq!(release_status(Win, true, None), "PASS");
        assert_eq!(release_status(Loss, true, None), "FAIL");
        assert_eq!(release_status(Loss, true, Some("known")), "PASS-with-note");
        assert_eq!(release_status(Noise, false, None), "NEEDS-KL-GATE");
        assert_eq!(
            release_status(Noise, false, Some("new kernel")),
            "PASS-with-note"
        );
        assert_eq!(
            release_status(Inconclusive, true, Some("x")),
            "INCONCLUSIVE"
        );
        let d = vec![
            Declared {
                family: "qwen*".into(),
                reason: "tile".into(),
            },
            Declared {
                family: "lfm3".into(),
                reason: " ".into(),
            },
        ];
        assert_eq!(declared_reason(&d, "qwen3_5"), Some("tile"));
        assert_eq!(declared_reason(&d, "gemma4"), None);
        // An empty reason declares nothing.
        assert_eq!(declared_reason(&d, "lfm3"), None);
    }
}
