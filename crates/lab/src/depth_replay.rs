//! Offline tuning for the adaptive speculation depth.
//!
//! Timing every controller setting on the GPU is slow and noisy. The two
//! things a setting's decode speed depends on are separable:
//!
//! - **How far the MTP head's chain matches the target**, per position. That
//!   is a property of the model and the text, not of the controller, so it is
//!   captured ONCE (`tslab chain-capture`): greedy decode with the head open,
//!   and after every committed token a depth-D chain is drafted, compared
//!   with the stream that follows, and the head is rewound.
//! - **What a round at depth d costs**, in milliseconds. That is a property of
//!   the kernels (the flat MMA8 verify changes it most), taken from
//!   measurements (`CostTable`).
//!
//! `tslab depth-replay` then walks the speculative loop's own round rules
//! over the capture for each setting: propose `min(depth, budget room)`,
//! accept `min(proposed, match length)`, commit accepted + 1, and feed the
//! controller only rounds the budget did not truncate. The controller is the
//! runtime's own `AdaptiveDepth` (`with_params`), so the replay cannot drift
//! from what ships. The best setting is then confirmed live with `tslab ab`
//! and `TURBOSPARK_ADAPTIVE_DEPTH`.
//!
//! One bias, as in `kflash_tree_replay_probe`: replay head rows are primed
//! from the trunk's real hidden state, so the replay drafter is slightly
//! better informed than the live loop's. The replay reports its fixed
//! block 2 committed/round so it can be read against a live
//! `[spec-stats]` figure.

use std::collections::BTreeMap;

use runtime::{AdaptiveDepth, AdaptiveParams};
use serde::{Deserialize, Serialize};

/// One captured case: the per-position match length of a depth-`depth`
/// chain drafted after committed token `i`, against the tokens after it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChainCapture {
    pub case: String,
    pub depth: usize,
    pub prompt_tokens: usize,
    pub match_len: Vec<u8>,
    /// SHA-256 of the greedy stream's ids.
    pub stream_digest: String,
    /// The prompt's token ids. Absent on captures written before the ids
    /// were kept; `tslab hybrid-replay` needs them and says so.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_ids: Option<Vec<i32>>,
    /// The greedy stream's token ids, one per `match_len` position, so a
    /// replay can match n-grams over prompt plus history. Same age rule.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream: Option<Vec<i32>>,
}

/// Round cost by proposed depth, milliseconds.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostTable {
    pub name: String,
    pub ms_per_round: BTreeMap<usize, f64>,
    /// Depths whose cost was measured; the rest are modeled.
    pub measured: Vec<usize>,
    pub source: String,
}

impl CostTable {
    pub fn cost(&self, depth: usize) -> Option<f64> {
        self.ms_per_round.get(&depth).copied()
    }

    pub fn max_depth(&self) -> usize {
        self.ms_per_round.keys().copied().max().unwrap_or(0)
    }
}

/// MTP rounds on `qwen38-27b-mtp`, M4 Max, without the MMA8 tile.
/// Depths 2 and 4: the mean of the short and medium rows of
/// `docs/KAPITSA_KERNELS.md` ("Measured: DFlash2 and the flat verify", MTP
/// 67/68 and 130/129 ms), which tslab reproduced on 2026-10-22 (63 and 129
/// ms from committed/round over decode tok/s). Other depths are a straight
/// line through those two, an estimate.
pub fn flag_off() -> CostTable {
    linear(
        "flag_off",
        67.5,
        129.5,
        "docs/KAPITSA_KERNELS.md MTP 2 and 4 (flag off); other depths linear (modeled)",
    )
}

/// The same with `TURBOSPARK_INT4_MMA8=1` (the flat verify). Depths 2 and
/// 4 from the same table (72/73 and 86/87 ms). Other depths linear, about 7
/// ms per extra depth, which matches the doc's per-step parts: an MTP draft
/// step of about 4.8 ms and about 1.3 ms per verify row (modeled).
pub fn flag_on() -> CostTable {
    linear(
        "flag_on",
        72.5,
        86.5,
        "docs/KAPITSA_KERNELS.md MTP 2 and 4 (flag on); other depths linear (modeled)",
    )
}

fn linear(name: &str, at2: f64, at4: f64, source: &str) -> CostTable {
    let slope = (at4 - at2) / 2.0;
    CostTable {
        name: name.to_string(),
        ms_per_round: (1..=8)
            .map(|d| (d, at2 + slope * (d as f64 - 2.0)))
            .filter(|(_, ms)| *ms > 0.0)
            .collect(),
        measured: vec![2, 4],
        source: source.to_string(),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum Policy {
    Fixed(usize),
    Adaptive {
        floor: usize,
        cap: usize,
        climb_low: usize,
        climb_mid: usize,
        climb_high: usize,
        drop: usize,
    },
}

impl Policy {
    pub fn adaptive(floor: usize, cap: usize, p: AdaptiveParams) -> Self {
        Policy::Adaptive {
            floor,
            cap,
            climb_low: p.climb_low,
            climb_mid: p.climb_mid,
            climb_high: p.climb_high,
            drop: p.drop,
        }
    }

    pub fn label(&self) -> String {
        match self {
            Policy::Fixed(b) => format!("fixed {b}"),
            Policy::Adaptive {
                floor,
                cap,
                climb_low,
                climb_mid,
                climb_high,
                drop,
            } => format!(
                "adaptive {floor}..{cap} climb {climb_low},{climb_mid},{climb_high} drop {drop}"
            ),
        }
    }

    /// The `TURBOSPARK_ADAPTIVE_DEPTH` value for an adaptive policy.
    pub fn env_value(&self) -> Option<String> {
        match self {
            Policy::Adaptive {
                climb_low,
                climb_mid,
                climb_high,
                drop,
                ..
            } => Some(format!("{climb_low},{climb_mid},{climb_high},{drop}")),
            Policy::Fixed(_) => None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimResult {
    pub rounds: usize,
    pub committed: usize,
    pub ms: f64,
    pub tok_s: f64,
    pub committed_per_round: f64,
    /// Rounds proposed at each depth.
    pub depth_hist: BTreeMap<usize, usize>,
}

/// Walk the speculative loop over one capture. `max_new` caps the decode,
/// as the loop's budget does.
pub fn simulate(
    cap: &ChainCapture,
    policy: Policy,
    costs: &CostTable,
    max_new: usize,
) -> Result<SimResult, String> {
    let n = cap.match_len.len().min(max_new);
    let mut ctrl = match policy {
        Policy::Fixed(_) => None,
        Policy::Adaptive {
            floor,
            cap: c,
            climb_low,
            climb_mid,
            climb_high,
            drop,
        } => Some(AdaptiveDepth::with_params(
            floor,
            c,
            AdaptiveParams {
                climb_low,
                climb_mid,
                climb_high,
                drop,
            },
        )),
    };
    let max_depth = match policy {
        Policy::Fixed(b) => b,
        Policy::Adaptive { cap: c, .. } => c,
    };
    if max_depth > cap.depth {
        return Err(format!(
            "policy reaches depth {max_depth}, the capture drafted only {}",
            cap.depth
        ));
    }
    // Index of the committed token the next round drafts from. The first
    // token after the prompt is committed by prefill, so generation starts
    // with one token already out.
    let mut i = 0usize;
    let mut generated = 1usize;
    let (mut rounds, mut ms) = (0usize, 0f64);
    let mut hist = BTreeMap::new();
    while generated < n && i < cap.match_len.len() {
        let wanted = match (&ctrl, policy) {
            (Some(c), _) => c.depth(),
            (None, Policy::Fixed(b)) => b,
            _ => unreachable!(),
        };
        let room = n - generated;
        let round = wanted.min(room);
        let truncated = round < wanted;
        let accepted = round.min(cap.match_len[i] as usize);
        ms += costs
            .cost(round)
            .ok_or_else(|| format!("{} has no cost for depth {round}", costs.name))?;
        *hist.entry(round).or_insert(0) += 1;
        rounds += 1;
        let commit = (accepted + 1).min(n - generated);
        generated += commit;
        i += accepted + 1;
        if !truncated {
            if let Some(c) = ctrl.as_mut() {
                c.observe(round, accepted);
            }
        }
    }
    let committed = generated - 1;
    Ok(SimResult {
        rounds,
        committed,
        ms,
        tok_s: committed as f64 * 1000.0 / ms.max(f64::MIN_POSITIVE),
        committed_per_round: committed as f64 / rounds.max(1) as f64,
        depth_hist: hist,
    })
}

/// The settings `depth-replay` scores: fixed blocks, the shipped tuning at
/// several caps, and a grid around it.
pub fn grid(max_depth: usize) -> Vec<Policy> {
    let mut out: Vec<Policy> = (2..=max_depth.min(8)).map(Policy::Fixed).collect();
    let caps: Vec<usize> = [3, 4, 5, 6, 7, 8]
        .into_iter()
        .filter(|&c| c <= max_depth)
        .collect();
    for &cap in &caps {
        out.push(Policy::adaptive(2, cap, AdaptiveParams::default()));
        for low in [1, 2, 3, 4, 6, 8] {
            for mid in [1, 2, 4, 6] {
                for high in [1, 2, 4] {
                    if mid > low || high > mid {
                        continue;
                    }
                    for drop in [1, 2, 3] {
                        let p = AdaptiveParams {
                            climb_low: low,
                            climb_mid: mid,
                            climb_high: high,
                            drop,
                        };
                        if p != AdaptiveParams::default() {
                            out.push(Policy::adaptive(2, cap, p));
                        }
                    }
                }
            }
        }
    }
    out
}

/// Capture one case on an open runner with an MTP head at least `depth`
/// deep: prefill `prompt` token by token while priming the head (the public
/// MTP-priming path; the chunked one is crate-private), then decode
/// `tokens` greedy tokens, drafting a depth-`depth` chain after each
/// committed token and rewinding the head, as `kflash_tree_replay_probe`
/// does. Returns the per-position match lengths and the stream.
pub fn capture_case(
    runner: &mut runtime::RealForwardRunner,
    prompt: &[i32],
    depth: usize,
    tokens: usize,
) -> Result<(Vec<u8>, Vec<i32>), String> {
    use foundation::LogitValue;
    use runtime::LogitProducer;
    let argmax = |l: &[LogitValue]| -> Result<i32, String> {
        let mut best = (0usize, f32::NEG_INFINITY);
        for (i, v) in l.iter().enumerate() {
            let f = v.to_f32();
            if f.is_nan() {
                return Err(format!("NaN logit at {i}"));
            }
            if f > best.1 {
                best = (i, f);
            }
        }
        Ok(best.0 as i32)
    };
    if runner.mtp_draft_depth() < depth {
        return Err(format!(
            "the open MTP head drafts {} deep, the capture needs {depth}",
            runner.mtp_draft_depth()
        ));
    }
    let vocab = runner.vocab_size();
    let mut logits = vec![LogitValue::from_f32(0.0); vocab];
    let mut draft = vec![LogitValue::from_f32(0.0); vocab];
    runner.reset();
    for (i, &t) in prompt.iter().enumerate() {
        runner.produce(t, i, &mut logits)?;
        if i + 1 < prompt.len() {
            runner
                .mtp_prime_step(prompt[i + 1], i)
                .map_err(|e| e.to_string())?;
        }
    }
    let mut stream: Vec<i32> = Vec::with_capacity(tokens);
    let mut chains: Vec<Vec<i32>> = Vec::with_capacity(tokens);
    let mut next = argmax(&logits)?;
    while stream.len() < tokens {
        stream.push(next);
        let base = prompt.len() + stream.len() - 1;
        let mut chain = Vec::with_capacity(depth);
        let mut chained = next;
        for d in 0..depth {
            runner
                .mtp_draft_step(chained, base - 1 + d, &mut draft)
                .map_err(|e| e.to_string())?;
            chained = argmax(&draft)?;
            chain.push(chained);
        }
        runner.mtp_rewind_to(base).map_err(|e| e.to_string())?;
        chains.push(chain);
        runner.produce(next, base, &mut logits)?;
        next = argmax(&logits)?;
    }
    // A chain drafted after stream[i] predicts stream[i+1..]. Its match
    // length is how many leading proposals equal those tokens; positions
    // near the end compare against what is left.
    let match_len = chains
        .iter()
        .enumerate()
        .map(|(i, chain)| {
            chain
                .iter()
                .zip(stream.iter().skip(i + 1))
                .take_while(|(a, b)| a == b)
                .count() as u8
        })
        .collect();
    Ok((match_len, stream))
}

/// `tslab chain-capture`.
pub fn chain_capture_cmd(mut args: crate::args::Args) -> Result<(), String> {
    use serde_json::json;
    let model = args.required("--model")?;
    let experiment = args
        .take("--experiment")?
        .unwrap_or_else(|| "depth-replay".into());
    let mut prompts: Vec<crate::workload::PromptSource> = args
        .take_all("--prompt")?
        .iter()
        .map(|p| crate::args::parse_prompt(p))
        .collect::<Result<_, _>>()?;
    if prompts.is_empty() {
        prompts = vec![
            crate::workload::PromptSource::Protocol("short-explanation".into()),
            crate::workload::PromptSource::Protocol("medium-review".into()),
        ];
    }
    let tokens: usize = args.parsed("--tokens", 512)?;
    let depth: usize = args.parsed("--depth", 8)?;
    let context: usize = args.parsed("--context", 4096)?;
    args.finish()?;
    if depth == 0 || depth > 15 || tokens == 0 {
        return Err("--depth must be 1..=15 and --tokens positive".into());
    }
    let lab = crate::paths::lab_dir();
    let _lock =
        crate::lock::MachineLock::acquire(&lab, &format!("tslab chain-capture {experiment}"))?;
    let ledger = crate::ledger::Ledger::open(&crate::paths::experiment_dir(&lab, &experiment)?)?;
    let dir = catalog::resolve_model_arg(&model);
    let arch = repack::peek_manifest_arch(&dir)?;
    let tokenizer = tokenizer::MfTokenizer::load_from_dir(&dir)
        .map_err(|e| format!("tokenizer from {}: {e}", dir.display()))?;
    let mut runner = runtime::RealForwardRunner::open_with_residency(
        &dir,
        arch,
        context,
        runtime::ExpertCacheSlots::Auto,
        runtime::DraftPolicies::mtp(runtime::MtpDraftPolicy::Fixed(16)),
        runtime::SteeringPolicy::off(),
        1,
        runtime::KvQuant::Off,
        runtime::ExpertResidency::Auto,
    )
    .map_err(|e| e.to_string())?;
    let ids = crate::worker::workload_ids(&tokenizer, &prompts)?;
    for (source, prompt) in prompts.iter().zip(&ids) {
        if prompt.len() + tokens + depth + 1 > context {
            return Err(format!(
                "{source:?} does not fit --context {context} with {tokens} tokens"
            ));
        }
        let started = std::time::Instant::now();
        let (match_len, stream) =
            gpu::autorelease_pool(|| capture_case(&mut runner, prompt, depth, tokens))?;
        let bytes: Vec<u8> = stream.iter().flat_map(|t| t.to_le_bytes()).collect();
        let cap = ChainCapture {
            case: match source {
                crate::workload::PromptSource::Protocol(id) => id.clone(),
                other => format!("{other:?}"),
            },
            depth,
            prompt_tokens: prompt.len(),
            match_len,
            stream_digest: model_io::hash_data(&bytes),
            prompt_ids: Some(prompt.clone()),
            stream: Some(stream),
        };
        let full = cap
            .match_len
            .iter()
            .filter(|&&m| m as usize >= depth)
            .count();
        eprintln!(
            "  {}: {} prompt tokens, {} positions, mean match {:.2}, full-depth matches {:.1}%, {:.0} s",
            cap.case,
            cap.prompt_tokens,
            cap.match_len.len(),
            cap.match_len.iter().map(|&m| m as f64).sum::<f64>() / cap.match_len.len() as f64,
            100.0 * full as f64 / cap.match_len.len() as f64,
            started.elapsed().as_secs_f64()
        );
        ledger.append(
            "chain_capture",
            json!({"model": dir.display().to_string(), "capture": cap,
                   "provenance": crate::provenance::collect()}),
        )?;
    }
    Ok(())
}

/// Round costs measured in an experiment's speculative runs (L7):
/// `ledger:EXPERIMENT`, optionally `@FLAGS` (a `round_cost` flags label such
/// as `TURBOSPARK_INT4_MMA8=1`, default `default`) and `#KV` (one KV
/// bucket, default all). Measured depths are exact; the rest sit on the
/// line through them and the source says which.
pub fn ledger_costs(spec: &str, max_depth: usize) -> Result<CostTable, String> {
    let rest = spec.trim_start_matches("ledger:");
    let (rest, kv) = match rest.split_once('#') {
        Some((r, k)) => (
            r,
            Some(k.parse::<u64>().map_err(|_| format!("KV bucket {k:?}"))?),
        ),
        None => (rest, None),
    };
    let (experiment, flags) = rest.split_once('@').unwrap_or((rest, "default"));
    let ledger = crate::ledger::Ledger::open(&crate::paths::experiment_dir(
        &crate::paths::lab_dir(),
        experiment,
    )?)?;
    let records = ledger.of_kind("round_cost")?;
    let measured = crate::round_cost::by_depth(&records, flags, kv);
    let ms_per_round = crate::round_cost::fill(&measured, max_depth.max(2))?;
    Ok(CostTable {
        name: spec.to_string(),
        ms_per_round,
        measured: measured.keys().copied().collect(),
        source: format!(
            "round_cost records in {experiment}, flags {flags}{}: depths {:?} measured ({} readings), others on their line (modeled)",
            kv.map_or(String::new(), |k| format!(", KV bucket {k}")),
            measured.keys().collect::<Vec<_>>(),
            measured.values().map(|m| m.1).sum::<usize>()
        ),
    })
}

/// One `--costs` entry: a built-in table or `ledger:EXP[@FLAGS][#KV]`.
pub fn cost_table(name: &str, depth: usize) -> Result<CostTable, String> {
    match name {
        "flag_off" => Ok(flag_off()),
        "flag_on" => Ok(flag_on()),
        other if other.starts_with("ledger:") => ledger_costs(other, depth),
        other => Err(format!(
            "--costs {other:?}: flag_off, flag_on or ledger:EXPERIMENT[@FLAGS][#KV_BUCKET]"
        )),
    }
}

/// `tslab depth-replay`.
pub fn depth_replay_cmd(mut args: crate::args::Args) -> Result<(), String> {
    use serde_json::json;
    let experiment = args
        .take("--experiment")?
        .unwrap_or_else(|| "depth-replay".into());
    let tokens: usize = args.parsed("--tokens", 512)?;
    let top: usize = args.parsed("--top", 10)?;
    let tables = args
        .take("--costs")?
        .unwrap_or_else(|| "flag_off,flag_on".into());
    args.finish()?;
    let ledger = crate::ledger::Ledger::open(&crate::paths::experiment_dir(
        &crate::paths::lab_dir(),
        &experiment,
    )?)?;
    let captures: Vec<ChainCapture> = ledger
        .of_kind("chain_capture")?
        .into_iter()
        .map(|r| serde_json::from_value(r["capture"].clone()).map_err(|e| e.to_string()))
        .collect::<Result<_, _>>()?;
    if captures.is_empty() {
        return Err(format!(
            "no chain_capture records in {experiment}; run tslab chain-capture"
        ));
    }
    let depth = captures.iter().map(|c| c.depth).min().unwrap_or(0);
    for name in tables.split(',') {
        let costs = cost_table(name, depth)?;
        let policies = grid(depth.min(costs.max_depth()));
        // Per policy: tok/s per case, then speedup over fixed 2 per case.
        let mut rows = Vec::new();
        let base: Vec<f64> = captures
            .iter()
            .map(|c| simulate(c, Policy::Fixed(2), &costs, tokens).map(|r| r.tok_s))
            .collect::<Result<_, _>>()?;
        for p in &policies {
            let sims: Vec<SimResult> = captures
                .iter()
                .map(|c| simulate(c, *p, &costs, tokens))
                .collect::<Result<_, _>>()?;
            let speedups: Vec<f64> = sims.iter().zip(&base).map(|(s, b)| s.tok_s / b).collect();
            let geo = (speedups.iter().map(|s| s.ln()).sum::<f64>() / speedups.len() as f64).exp();
            let worst = speedups.iter().copied().fold(f64::INFINITY, f64::min);
            rows.push((*p, geo, worst, sims, speedups));
        }
        crate::say!(
            "\ncosts {} ({}); {} cases, {tokens} tokens each",
            costs.name,
            costs.source,
            captures.len()
        );
        crate::say_part!("  {:<46} {:>7} {:>7}", "policy", "geo", "worst");
        for c in &captures {
            crate::say_part!(" {:>18}", c.case.chars().take(18).collect::<String>());
        }
        crate::say!();
        let print_row = |r: &(Policy, f64, f64, Vec<SimResult>, Vec<f64>)| {
            crate::say_part!("  {:<46} {:>6.3}x {:>6.3}x", r.0.label(), r.1, r.2);
            for (s, sp) in r.3.iter().zip(&r.4) {
                crate::say_part!(" {:>7.1} t/s {:>5.2}c", s.tok_s, s.committed_per_round);
                let _ = sp;
            }
            crate::say!();
        };
        for r in rows.iter().filter(|r| matches!(r.0, Policy::Fixed(_))) {
            print_row(r);
        }
        let shipped: Vec<_> = rows
            .iter()
            .filter(|r| match r.0 {
                Policy::Adaptive {
                    climb_low,
                    climb_mid,
                    climb_high,
                    drop,
                    ..
                } => {
                    AdaptiveParams {
                        climb_low,
                        climb_mid,
                        climb_high,
                        drop,
                    } == AdaptiveParams::default()
                }
                _ => false,
            })
            .collect();
        crate::say!("  -- shipped tuning:");
        for r in &shipped {
            print_row(r);
        }
        // Best adaptive settings that never lose more than 1% on any case.
        let mut safe: Vec<_> = rows
            .iter()
            .filter(|r| matches!(r.0, Policy::Adaptive { .. }) && r.2 >= 0.99)
            .collect();
        safe.sort_by(|a, b| b.1.total_cmp(&a.1));
        crate::say!("  -- best adaptive, worst case at least 0.99x of fixed 2:");
        for r in safe.iter().take(top) {
            print_row(r);
        }
        let record: Vec<serde_json::Value> = rows
            .iter()
            .map(|r| {
                json!({"policy": r.0, "label": r.0.label(), "env": r.0.env_value(),
                       "geo_speedup": r.1, "worst_speedup": r.2,
                       "per_case": r.3.iter().zip(&captures).map(|(s, c)| json!({"case": c.case, "result": s})).collect::<Vec<_>>()})
            })
            .collect();
        ledger.append(
            "depth_replay",
            json!({"costs": costs, "tokens": tokens, "cases": captures.iter().map(|c| &c.case).collect::<Vec<_>>(),
                   "results": record}),
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cap(lens: &[u8]) -> ChainCapture {
        ChainCapture {
            case: "t".into(),
            depth: 8,
            prompt_tokens: 10,
            match_len: lens.to_vec(),
            stream_digest: String::new(),
            prompt_ids: None,
            stream: None,
        }
    }

    #[test]
    fn a_capture_without_ids_still_reads_and_writes_none_without_the_fields() {
        // Records written before the ids were kept have neither field.
        let old = serde_json::json!({"case": "c", "depth": 8, "prompt_tokens": 3,
                                     "match_len": [1, 2], "stream_digest": "d"});
        let c: ChainCapture = serde_json::from_value(old).unwrap();
        assert_eq!((c.prompt_ids, c.stream), (None, None));
        let v = serde_json::to_value(cap(&[1])).unwrap();
        assert!(v.get("prompt_ids").is_none() && v.get("stream").is_none());
        let mut with = cap(&[1]);
        with.stream = Some(vec![7]);
        let v = serde_json::to_value(&with).unwrap();
        assert_eq!(v["stream"], serde_json::json!([7]));
    }

    fn flat(ms: f64) -> CostTable {
        CostTable {
            name: "flat".into(),
            ms_per_round: (1..=8).map(|d| (d, ms)).collect(),
            measured: vec![],
            source: "test".into(),
        }
    }

    #[test]
    fn fixed_block_rounds_follow_the_loop() {
        // Match lengths 8 everywhere: block 2 commits 3 per round.
        let c = cap(&[8; 100]);
        let r = simulate(&c, Policy::Fixed(2), &flat(10.0), 31).unwrap();
        // 30 tokens to generate after the first: 10 rounds of 3.
        assert_eq!(r.rounds, 10);
        assert_eq!(r.committed, 30);
        assert!((r.ms - 100.0).abs() < 1e-9);
        assert!((r.tok_s - 300.0).abs() < 1e-9);
    }

    #[test]
    fn accept_stops_at_the_match_length() {
        // Every position matches only 1: block 4 commits 2 per round.
        let c = cap(&[1; 100]);
        let r = simulate(&c, Policy::Fixed(4), &flat(10.0), 21).unwrap();
        assert_eq!(r.committed_per_round, 2.0);
    }

    #[test]
    fn the_last_round_is_truncated_by_the_budget() {
        let c = cap(&[8; 100]);
        // 4 tokens to generate with block 3: one round of 3 (commits 4 but
        // only 3 fit... the round proposes min(3, room=3)) then nothing.
        let r = simulate(&c, Policy::Fixed(3), &flat(10.0), 4).unwrap();
        assert_eq!(r.committed, 3);
        assert_eq!(r.rounds, 1);
    }

    #[test]
    fn shipped_tuning_never_leaves_the_floor_on_prose_like_matches() {
        // Every round accepts 1 of the 2 proposals, so no clean streak ever
        // forms and the controller stays at the floor, as the live prose
        // runs did (every round at depth 2). The round stride decides which
        // positions are read, so a periodic pattern would test the stride.
        let r = simulate(
            &cap(&[1; 600]),
            Policy::adaptive(2, 4, AdaptiveParams::default()),
            &flag_off(),
            512,
        )
        .unwrap();
        // The last round may be shorter (the budget truncates it, as the
        // loop does); no round may go above the floor.
        assert_eq!(r.depth_hist.keys().max(), Some(&2), "{:?}", r.depth_hist);
        // And with full matches it does climb, so the floor above is the
        // controller's choice and not a simulator that cannot move.
        let r = simulate(
            &cap(&[8; 600]),
            Policy::adaptive(2, 4, AdaptiveParams::default()),
            &flag_off(),
            512,
        )
        .unwrap();
        assert!(r.depth_hist.contains_key(&4), "{:?}", r.depth_hist);
    }

    #[test]
    fn a_perfect_drafter_climbs_and_pays_under_the_flat_verify() {
        let c = cap(&[8; 600]);
        let quick = Policy::adaptive(
            2,
            4,
            AdaptiveParams {
                climb_low: 1,
                climb_mid: 1,
                climb_high: 1,
                drop: 1,
            },
        );
        let fixed2 = simulate(&c, Policy::Fixed(2), &flag_on(), 512).unwrap();
        let adapt = simulate(&c, quick, &flag_on(), 512).unwrap();
        assert!(
            adapt.tok_s > fixed2.tok_s,
            "{} vs {}",
            adapt.tok_s,
            fixed2.tok_s
        );
        assert!(adapt.depth_hist.get(&4).copied().unwrap_or(0) > 100);
    }

    #[test]
    fn cost_tables_hold_their_measured_points() {
        assert_eq!(flag_off().cost(2), Some(67.5));
        assert_eq!(flag_off().cost(4), Some(129.5));
        assert_eq!(flag_on().cost(2), Some(72.5));
        assert_eq!(flag_on().cost(4), Some(86.5));
    }

    #[test]
    fn a_policy_deeper_than_the_capture_is_refused() {
        let mut c = cap(&[8; 10]);
        c.depth = 4;
        assert!(simulate(&c, Policy::Fixed(5), &flat(1.0), 8).is_err());
    }

    #[test]
    fn grid_holds_fixed_blocks_and_the_shipped_tuning() {
        let g = grid(8);
        assert!(g.contains(&Policy::Fixed(2)));
        assert!(g.contains(&Policy::adaptive(2, 4, AdaptiveParams::default())));
        assert_eq!(
            g.iter()
                .filter(|p| **p == Policy::adaptive(2, 4, AdaptiveParams::default()))
                .count(),
            1
        );
    }
}
