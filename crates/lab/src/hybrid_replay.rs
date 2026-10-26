//! A hybrid of prompt lookup and the MTP head, replayed offline.
//!
//! A standalone lookup drafter was declined by replay
//! (`crates/bench/tests/pld_replay_probe.rs`, `docs/KAPITSA_KERNELS.md`
//! "Prompt lookup"): on prose it loses to the head, and on copy-heavy output
//! it only matches what MTP block 2 already gives. The open question was a
//! hybrid: when a long n-gram of the committed text has occurred before,
//! propose the tokens that followed it instead of the head's chain, and
//! otherwise run the head as today.
//!
//! `tslab hybrid-replay` prices that over ONE greedy stream per case, the
//! chain capture `tslab chain-capture` writes once it keeps the token ids.
//! Per round it asks the matcher for the continuation of the last `ngram`
//! tokens of prompt plus history, and takes it when it is longer than the
//! depth the head would propose; acceptance is exact token equality against
//! the stream for both sources, as the live loop's greedy accept is. The
//! head's rounds walk `depth_replay::simulate`'s rules with the runtime's own
//! `AdaptiveDepth`. Two rules from the engine that ships the hybrid
//! (`docs/MLX_SERVE_KERNELS.md`, E6) are kept: a lookup round never feeds the
//! controller's acceptance statistics, and acceptance in a lookup round is
//! exact, never approximate.
//!
//! Cost: a lookup round of L proposals is priced at the MTP table's depth-L
//! round, which includes L head draft steps the lookup round does not run.
//! The tok/s column is therefore conservative for the hybrid; the
//! committed-per-round column does not depend on cost and is what the 5% bar
//! is read from.

use std::collections::BTreeMap;

use runtime::{AdaptiveDepth, AdaptiveParams};
use serde::{Deserialize, Serialize};

use crate::depth_replay::{simulate, ChainCapture, CostTable, Policy, SimResult};

/// What followed the most recent earlier occurrence of the last `n` tokens
/// of `history`, up to `want` tokens. The tail's own position is excluded,
/// so a run of one repeated token cannot match itself. The matcher of
/// `pld_replay_probe.rs`, unchanged.
pub fn propose(history: &[i32], n: usize, want: usize) -> Vec<i32> {
    if n == 0 || want == 0 || history.len() < n + 1 {
        return Vec::new();
    }
    let tail = &history[history.len() - n..];
    let last = history.len() - n;
    for start in (0..last).rev() {
        if &history[start..start + n] == tail {
            let from = start + n;
            let take = want.min(history.len() - from);
            if take > 0 {
                return history[from..from + take].to_vec();
            }
        }
    }
    Vec::new()
}

/// When a lookup proposal replaces the head's chain.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct HybridRule {
    /// Suffix length that must recur. Four and up is where the standalone
    /// replay stopped losing on prose.
    pub ngram: usize,
    /// Longest lookup proposal, also bounded by the cost table's depth.
    pub max_draft: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridResult {
    pub rounds: usize,
    pub committed: usize,
    pub ms: f64,
    pub tok_s: f64,
    pub committed_per_round: f64,
    /// Rounds that took the lookup proposal, and their proposals and
    /// accepted tokens.
    pub lookup_rounds: usize,
    pub lookup_proposed: usize,
    pub lookup_accepted: usize,
    /// Rounds that ran the head, by proposed depth.
    pub head_depth_hist: BTreeMap<usize, usize>,
    /// Lookup rounds by proposal length.
    pub lookup_len_hist: BTreeMap<usize, usize>,
}

/// Walk the speculative loop over one capture with the hybrid draft source.
pub fn simulate_hybrid(
    cap: &ChainCapture,
    policy: Policy,
    costs: &CostTable,
    max_new: usize,
    rule: &HybridRule,
) -> Result<HybridResult, String> {
    let (Some(prompt), Some(stream)) = (&cap.prompt_ids, &cap.stream) else {
        return Err(format!(
            "capture {} has no token ids; re-run tslab chain-capture with this build",
            cap.case
        ));
    };
    if stream.len() != cap.match_len.len() {
        return Err(format!(
            "capture {}: {} stream ids for {} positions",
            cap.case,
            stream.len(),
            cap.match_len.len()
        ));
    }
    if rule.ngram == 0 || rule.max_draft == 0 {
        return Err("the hybrid rule needs an n-gram and a draft length of at least 1".into());
    }
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
    let max_draft = rule.max_draft.min(costs.max_depth());
    // Prompt plus everything committed so far; the n-gram is matched over it.
    let mut history: Vec<i32> = prompt.clone();
    let mut out = HybridResult {
        rounds: 0,
        committed: 0,
        ms: 0.0,
        tok_s: 0.0,
        committed_per_round: 0.0,
        lookup_rounds: 0,
        lookup_proposed: 0,
        lookup_accepted: 0,
        head_depth_hist: BTreeMap::new(),
        lookup_len_hist: BTreeMap::new(),
    };
    // As in `simulate`: `i` is the committed token the round drafts from,
    // and the first token after the prompt is out before any round.
    let mut i = 0usize;
    let mut generated = 1usize;
    if !stream.is_empty() {
        history.push(stream[0]);
    }
    while generated < n && i < cap.match_len.len() {
        let wanted = match (&ctrl, policy) {
            (Some(c), _) => c.depth(),
            (None, Policy::Fixed(b)) => b,
            _ => unreachable!(),
        };
        let room = n - generated;
        let round = wanted.min(room);
        let truncated = round < wanted;
        let lookup = propose(&history, rule.ngram, max_draft.min(room));
        let accepted = if lookup.len() > round {
            // Exact acceptance against the stream, never the head's match.
            let accepted = lookup
                .iter()
                .zip(&stream[i + 1..])
                .take_while(|(a, b)| a == b)
                .count();
            out.ms += costs
                .cost(lookup.len())
                .ok_or_else(|| format!("{} has no cost for depth {}", costs.name, lookup.len()))?;
            out.lookup_rounds += 1;
            out.lookup_proposed += lookup.len();
            out.lookup_accepted += accepted;
            *out.lookup_len_hist.entry(lookup.len()).or_insert(0) += 1;
            // The controller is not fed: a copied span says nothing about
            // how far the head's chain would have matched.
            accepted
        } else {
            let accepted = round.min(cap.match_len[i] as usize);
            out.ms += costs
                .cost(round)
                .ok_or_else(|| format!("{} has no cost for depth {round}", costs.name))?;
            *out.head_depth_hist.entry(round).or_insert(0) += 1;
            if !truncated {
                if let Some(c) = ctrl.as_mut() {
                    c.observe(round, accepted);
                }
            }
            accepted
        };
        out.rounds += 1;
        let commit = (accepted + 1).min(n - generated);
        generated += commit;
        // The accepted proposals and the bonus that opens the next round.
        let next = i + accepted + 1;
        history.extend_from_slice(&stream[i + 1..next.min(stream.len() - 1) + 1]);
        i = next;
    }
    out.committed = generated - 1;
    out.tok_s = out.committed as f64 * 1000.0 / out.ms.max(f64::MIN_POSITIVE);
    out.committed_per_round = out.committed as f64 / out.rounds.max(1) as f64;
    Ok(out)
}

/// The three policies the hybrid is read against: fixed 2, the shipped
/// adaptive controller (2..4, default tuning) and a tuned controller.
pub fn policies(tuned: AdaptiveParams, cap: usize) -> Vec<Policy> {
    vec![
        Policy::Fixed(2),
        Policy::adaptive(2, 4, AdaptiveParams::default()),
        Policy::adaptive(2, cap, tuned),
    ]
}

/// Short case label for a table: a protocol id as is, a messages file by its
/// stem.
pub fn short_case(case: &str) -> String {
    match case
        .strip_prefix("MessagesFile(\"")
        .and_then(|r| r.strip_suffix("\")"))
    {
        Some(path) => std::path::Path::new(path)
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| case.to_string()),
        None => case.to_string(),
    }
}

/// `tslab hybrid-replay`.
pub fn hybrid_replay_cmd(mut args: crate::args::Args) -> Result<(), String> {
    use serde_json::json;
    let experiment = args
        .take("--experiment")?
        .unwrap_or_else(|| "depth-replay".into());
    let tokens: usize = args.parsed("--tokens", 512)?;
    let ngram: usize = args.parsed("--ngram", 4)?;
    let max_draft: usize = args.parsed("--lookup-draft", 8)?;
    let cap: usize = args.parsed("--cap", 8)?;
    let tuned = args.take("--tuned")?.unwrap_or_else(|| "3,1,1,2".into());
    let tuned = AdaptiveParams::parse(&tuned)
        .ok_or_else(|| format!("--tuned {tuned:?}: LOW,MID,HIGH,DROP, each at least 1"))?;
    let tables = args.take("--costs")?.unwrap_or_else(|| "flag_on".into());
    args.finish()?;
    let rule = HybridRule { ngram, max_draft };
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
        let costs = crate::depth_replay::cost_table(name, depth)?;
        let policies = policies(tuned, cap.min(depth).min(costs.max_depth()));
        crate::say!(
            "\ncosts {} ({}); {} cases, {tokens} tokens each; lookup n-gram {ngram}, draft up to {max_draft}",
            costs.name,
            costs.source,
            captures.len()
        );
        crate::say!(
            "  {:<44} {:<18} {:>8} {:>8} {:>8} {:>8} {:>7} {:>7} {:>9}",
            "policy",
            "case",
            "mtp c/r",
            "hyb c/r",
            "mtp t/s",
            "hyb t/s",
            "d c/r",
            "d t/s",
            "lookup"
        );
        let mut rows = Vec::new();
        for p in &policies {
            for c in &captures {
                let head: SimResult = simulate(c, *p, &costs, tokens)?;
                let hyb = simulate_hybrid(c, *p, &costs, tokens, &rule)?;
                let d_cpr = hyb.committed_per_round / head.committed_per_round - 1.0;
                let d_tok = hyb.tok_s / head.tok_s - 1.0;
                crate::say!(
                    "  {:<44} {:<18} {:>8.2} {:>8.2} {:>8.1} {:>8.1} {:>+6.1}% {:>+6.1}% {:>4} ({:.0}%)",
                    p.label(),
                    short_case(&c.case).chars().take(18).collect::<String>(),
                    head.committed_per_round,
                    hyb.committed_per_round,
                    head.tok_s,
                    hyb.tok_s,
                    100.0 * d_cpr,
                    100.0 * d_tok,
                    hyb.lookup_rounds,
                    100.0 * hyb.lookup_rounds as f64 / hyb.rounds.max(1) as f64
                );
                rows.push(json!({
                    "policy": p, "label": p.label(), "env": p.env_value(), "case": c.case,
                    "head": head, "hybrid": hyb,
                    "delta_committed_per_round": d_cpr, "delta_tok_s": d_tok,
                }));
            }
        }
        ledger.append(
            "hybrid_replay",
            json!({"costs": costs, "tokens": tokens, "rule": rule,
                   "cases": captures.iter().map(|c| &c.case).collect::<Vec<_>>(),
                   "results": rows}),
        )?;
    }
    crate::say!(
        "\n'd c/r' is the hybrid's committed per round over the head-only policy on the same\n\
         capture, which no cost table enters. 'd t/s' prices lookup rounds at the MTP table's\n\
         round of the same width, draft steps included, so it is conservative for the hybrid."
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat(ms: f64) -> CostTable {
        CostTable {
            name: "flat".into(),
            ms_per_round: (1..=8).map(|d| (d, ms)).collect(),
            measured: vec![],
            source: "test".into(),
        }
    }

    fn cap(prompt: Vec<i32>, stream: Vec<i32>, match_len: Vec<u8>) -> ChainCapture {
        assert_eq!(stream.len(), match_len.len());
        ChainCapture {
            case: "t".into(),
            depth: 8,
            prompt_tokens: prompt.len(),
            match_len,
            stream_digest: String::new(),
            prompt_ids: Some(prompt),
            stream: Some(stream),
        }
    }

    const RULE: HybridRule = HybridRule {
        ngram: 4,
        max_draft: 8,
    };

    #[test]
    fn propose_prefers_the_most_recent_match_and_never_the_tail() {
        let h = [5, 6, 7, 1, 5, 6, 9, 2, 5, 6];
        assert_eq!(propose(&h, 2, 3), vec![9, 2, 5]);
        assert!(propose(&[3, 3, 3], 3, 4).is_empty());
        assert!(propose(&[1, 2, 3], 2, 4).is_empty());
        assert_eq!(propose(&[1, 2, 3, 1, 2], 2, 8), vec![3, 1, 2]);
    }

    #[test]
    fn a_capture_without_ids_is_refused() {
        let mut c = cap(vec![1, 2, 3, 4, 5], vec![1; 8], vec![1; 8]);
        c.stream = None;
        let e = simulate_hybrid(&c, Policy::Fixed(2), &flat(1.0), 8, &RULE).unwrap_err();
        assert!(e.contains("no token ids"), "{e}");
    }

    #[test]
    fn a_copied_span_commits_more_than_the_head_and_never_feeds_the_controller() {
        // The prompt is a period-5 cycle; the stream continues it for 40
        // tokens (lookup matches, 8 at a time), then turns novel. The head's
        // chain matches 1 everywhere, so head rounds at the floor accept 1.
        let prompt: Vec<i32> = (0..40).map(|i| i % 5).collect();
        let mut stream: Vec<i32> = (40..80).map(|i| i % 5).collect();
        stream.extend(1000..1040);
        let c = cap(prompt, stream, vec![1; 80]);
        let quick = Policy::adaptive(
            2,
            8,
            AdaptiveParams {
                climb_low: 1,
                climb_mid: 1,
                climb_high: 1,
                drop: 1,
            },
        );
        let head = simulate(&c, quick, &flat(10.0), 80).unwrap();
        let hyb = simulate_hybrid(&c, quick, &flat(10.0), 80, &RULE).unwrap();
        assert_eq!(
            hyb.committed, head.committed,
            "both decode the whole stream"
        );
        assert!(
            hyb.rounds < head.rounds,
            "{} vs {}",
            hyb.rounds,
            head.rounds
        );
        assert!(hyb.lookup_rounds >= 4, "{hyb:?}");
        // Inside the repeat every proposal lands; only the round that runs
        // off the end of the copied span loses part of one draft.
        assert!(
            hyb.lookup_accepted > 0 && hyb.lookup_proposed - hyb.lookup_accepted <= 8,
            "{hyb:?}"
        );
        // Had the lookup rounds fed the controller (eight for eight, climb
        // after one clean round), the first head round would propose deep.
        // Every head round is at the floor.
        assert_eq!(
            hyb.head_depth_hist.keys().max(),
            Some(&2),
            "{:?}",
            hyb.head_depth_hist
        );
        assert!(hyb.head_depth_hist[&2] > 10);
    }

    #[test]
    fn lookup_is_taken_only_when_longer_than_the_heads_chain() {
        let prompt: Vec<i32> = (0..40).map(|i| i % 5).collect();
        let stream: Vec<i32> = (40..80).map(|i| i % 5).collect();
        let c = cap(prompt, stream, vec![8; 40]);
        // Fixed 8 proposes 8; a lookup draft capped at 8 is never longer.
        let r = simulate_hybrid(&c, Policy::Fixed(8), &flat(1.0), 40, &RULE).unwrap();
        assert_eq!(r.lookup_rounds, 0);
        let r = simulate_hybrid(&c, Policy::Fixed(2), &flat(1.0), 40, &RULE).unwrap();
        assert!(r.lookup_rounds > 0);
        // And the draft is bounded by the cost table's depth.
        let short = CostTable {
            name: "short".into(),
            ms_per_round: (1..=3).map(|d| (d, 1.0)).collect(),
            measured: vec![],
            source: "test".into(),
        };
        let r = simulate_hybrid(&c, Policy::Fixed(2), &short, 40, &RULE).unwrap();
        assert_eq!(
            r.lookup_len_hist.keys().max(),
            Some(&3),
            "{:?}",
            r.lookup_len_hist
        );
    }

    #[test]
    fn a_wrong_lookup_match_is_rejected_exactly_and_charged() {
        // History repeats `1 2 3 4 5`, so after `1 2 3 4` lookup proposes
        // `5 1 2 ...`; the stream says 9. Nothing is accepted and the round
        // still pays a wide verify.
        let prompt = vec![1, 2, 3, 4, 5, 1, 2, 3];
        let stream = vec![4, 9, 9, 9, 9, 9];
        let c = cap(prompt, stream, vec![0; 6]);
        let r = simulate_hybrid(&c, Policy::Fixed(2), &flat(10.0), 6, &RULE).unwrap();
        assert_eq!(r.lookup_rounds, 1);
        assert_eq!(r.lookup_accepted, 0);
        assert_eq!(r.committed, 5);
        let head = simulate(&c, Policy::Fixed(2), &flat(10.0), 6).unwrap();
        assert_eq!(
            r.rounds, head.rounds,
            "a rejected lookup round commits one, like a failed head round"
        );
    }

    #[test]
    fn novel_text_never_fires_and_matches_the_head_only_walk() {
        let prompt: Vec<i32> = (0..30).collect();
        let stream: Vec<i32> = (1000..1064).collect();
        let lens: Vec<u8> = (0..64).map(|i| (i % 3) as u8).collect();
        let c = cap(prompt, stream, lens);
        for p in policies(AdaptiveParams::parse("3,1,1,2").unwrap(), 8) {
            let head = simulate(&c, p, &flat(10.0), 64).unwrap();
            let hyb = simulate_hybrid(&c, p, &flat(10.0), 64, &RULE).unwrap();
            assert_eq!(hyb.lookup_rounds, 0);
            assert_eq!(hyb.rounds, head.rounds);
            assert_eq!(hyb.committed, head.committed);
            assert_eq!(hyb.head_depth_hist, head.depth_hist);
            assert!((hyb.ms - head.ms).abs() < 1e-9);
        }
    }

    #[test]
    fn short_case_strips_a_messages_file_to_its_stem() {
        assert_eq!(short_case("medium-review"), "medium-review");
        assert_eq!(
            short_case("MessagesFile(\"/a/b/copy-heavy.messages.json\")"),
            "copy-heavy.messages"
        );
    }
}
