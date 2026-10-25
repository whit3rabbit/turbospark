//! `tslab tree-replay`: would a KFlash-style tree verify beat the chain at
//! the SAME verify row budget, on this install's drafter?
//!
//! The check `docs/KAPITSA_KERNELS.md` (KFlash section) ran once by hand for
//! the qwen38 MTP head, made repeatable for any install with an MTP head or
//! a DFlash2 drafter. A greedy sequential decode drafts a fresh lattice after
//! every committed token (`turbospark_bench::tree_replay_record`), then
//! `turbospark_bench::tree_replay` walks the speculative loop for the chain,
//! a best-first tree and a per-round oracle at each block.
//!
//! What it is and is not:
//!
//! - Exact on the token count: an accepted path is always the reference
//!   prefix, so no tree kernel is needed to count it.
//! - A count, not a speed. It models neither a tree's own overhead (masked
//!   attention, per-branch recurrent state, KV compaction) nor the verify
//!   cost c(M), so it never produces a tok/s claim. A gain here is the
//!   precondition for building tree verify, not evidence that it pays.
//! - Not a timing run, so host noise cannot change its numbers. The host
//!   guard and eligibility are still recorded, because the run holds a whole
//!   model resident and a contended host is how this machine has choked.
//!
//! Runs in this process, like `tslab kernel`: one model, no arms to isolate.

use serde_json::json;

use crate::args::{parse_prompt, Args};
use crate::host::{eligibility, HostMonitor};
use crate::kernel::parse_list;
use crate::ledger::Ledger;
use crate::lock::MachineLock;
use crate::paths::{experiment_dir, lab_dir};
use crate::workload::PromptSource;
use turbospark_bench::tree_replay::{replay, ReplayLattice, ReplayStats, Shape};
use turbospark_bench::tree_replay_record::{plain, record_dflash, record_mtp};

/// The tree must beat the chain by at least this share of committed/round,
/// averaged over the prompts, AT THE SERVED BLOCK before tree verify is worth
/// costing (the decision rule in `docs/KAPITSA_KERNELS.md`). A gain at a
/// block the verify cost already rules out is not a reason to build one.
const MIN_TREE_GAIN: f64 = 0.05;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Drafter {
    Mtp,
    Dflash,
}

fn shape_stats<L: ReplayLattice>(
    lattices: &[Option<L>],
    stream: &[i32],
    block: usize,
) -> [ReplayStats; 3] {
    [Shape::Chain, Shape::BestFirst, Shape::Oracle].map(|s| replay(lattices, stream, block, s))
}

pub fn tree_replay_cmd(mut args: Args) -> Result<(), String> {
    let model = args.required("--model")?;
    let experiment = args
        .take("--experiment")?
        .unwrap_or_else(|| "tree-replay".into());
    let drafter = match args.required("--drafter")?.as_str() {
        "mtp" => Drafter::Mtp,
        "dflash" => Drafter::Dflash,
        other => return Err(format!("--drafter {other:?}: mtp or dflash")),
    };
    let mut prompts: Vec<PromptSource> = args
        .take_all("--prompt")?
        .iter()
        .map(|p| parse_prompt(p))
        .collect::<Result<_, _>>()?;
    if prompts.is_empty() {
        prompts = ["short-explanation", "medium-review"]
            .map(|id| PromptSource::Protocol(id.into()))
            .to_vec();
    }
    let generate: usize = args.parsed("--generate", 512)?;
    // DFlash2's trained width is 7 proposals (`dflash2_accept_length_probe`).
    let depth: usize = args.parsed(
        "--depth",
        match drafter {
            Drafter::Mtp => 8,
            Drafter::Dflash => 7,
        },
    )?;
    // DFlash2 keeps its selector's own 16 candidates per depth.
    let top_k_arg: Option<usize> = match args.take("--top-k")? {
        Some(v) => Some(v.parse().map_err(|_| format!("--top-k {v:?}"))?),
        None => None,
    };
    if drafter == Drafter::Dflash && top_k_arg.is_some() {
        return Err("--top-k is MTP only; DFlash2 keeps its selector's 16".into());
    }
    let top_k = top_k_arg.unwrap_or(4);
    let blocks: Vec<usize> = match args.take("--blocks")? {
        Some(s) => parse_list(&s, "block")?,
        None => {
            let mut b: Vec<usize> = [2, 4].into_iter().filter(|&b| b < depth).collect();
            b.push(depth);
            b
        }
    };
    // The block the install serves: the one whose gain decides. MTP's is the
    // only block that pays on this verify (`docs/MTP.md` break-even column);
    // DFlash2's is the runtime's serving default.
    let serve_block: usize = args.parsed(
        "--serve-block",
        match drafter {
            Drafter::Mtp => 2,
            Drafter::Dflash => runtime::DFLASH_SERVING_BLOCK,
        },
    )?;
    let mut blocks = blocks;
    if !blocks.contains(&serve_block) {
        blocks.insert(0, serve_block);
    }
    let slots: usize = args.parsed("--slots", 16)?;
    let rules = crate::cli::host_rules_from(&mut args)?;
    args.finish()?;
    if generate == 0 || depth == 0 || top_k == 0 {
        return Err("--generate, --depth and --top-k must be positive".into());
    }
    if blocks.iter().any(|&b| b == 0 || b > depth) || serve_block > depth {
        return Err(format!(
            "every block must be in 1..={depth} (the draft depth)"
        ));
    }

    let dir = catalog::resolve_model_arg(&model);
    let arch = repack::peek_manifest_arch(&dir)?;
    let lab = lab_dir();
    let _lock = MachineLock::acquire(&lab, &format!("tslab tree-replay {experiment}"))?;
    let ledger = Ledger::open(&experiment_dir(&lab, &experiment)?)?;
    let provenance = crate::provenance::collect();
    let mut monitor = HostMonitor::new();
    let before = monitor.sample(None);

    let policies = match drafter {
        Drafter::Mtp => runtime::DraftPolicies::mtp(runtime::MtpDraftPolicy::Fixed(depth)),
        Drafter::Dflash => runtime::DraftPolicies {
            mtp: runtime::MtpDraftPolicy::Off,
            dflash: runtime::DflashDraftPolicy::Fixed(depth),
        },
    };
    let (mut runner, tokenizer) =
        turbospark_bench::real_model::open_model_runner_speculative(&dir, slots, policies)?;
    let ids = crate::worker::workload_ids(&tokenizer, &prompts)?;
    eprintln!(
        "tslab tree-replay: {} ({}), {drafter:?} depth {depth}, {generate} tokens x {} prompt(s); ledger {}",
        dir.display(),
        arch.family.as_str(),
        ids.len(),
        ledger.path().display()
    );

    println!("\nprompt              block  chain c/rd  tree c/rd  oracle c/rd  tree/chain");
    let mut per_prompt = Vec::new();
    let mut serve_gains = Vec::new();
    for (source, prompt) in prompts.iter().zip(&ids) {
        let name = match source {
            PromptSource::Protocol(id) => id.clone(),
            PromptSource::MessagesFile(p) | PromptSource::TextFile(p) => p
                .file_name()
                .map_or_else(|| p.display().to_string(), |f| f.to_string_lossy().into()),
            PromptSource::Corpus(n) => format!("corpus:{n}"),
        };
        let reference = plain(&mut runner, prompt, generate)?;
        let rows: Vec<(usize, [ReplayStats; 3])> = match drafter {
            Drafter::Mtp => {
                let (stream, lattices) = record_mtp(&mut runner, prompt, generate, depth, top_k)?;
                check_stream(&name, &reference, &stream)?;
                blocks
                    .iter()
                    .map(|&b| (b, shape_stats(&lattices, &stream, b)))
                    .collect()
            }
            Drafter::Dflash => {
                let (stream, lattices) = record_dflash(&mut runner, prompt, generate)?;
                check_stream(&name, &reference, &stream)?;
                blocks
                    .iter()
                    .map(|&b| (b, shape_stats(&lattices, &stream, b)))
                    .collect()
            }
        };
        let mut block_recs = Vec::new();
        for (block, [chain, tree, oracle]) in rows {
            let (c, t, o) = (
                chain.committed_per_round(),
                tree.committed_per_round(),
                oracle.committed_per_round(),
            );
            let gain = t / c - 1.0;
            if block == serve_block {
                serve_gains.push(gain);
            }
            println!(
                "{name:<18}  {block:>5}  {c:>10.3}  {t:>9.3}  {o:>11.3}  {:>9.3}x",
                t / c
            );
            block_recs.push(json!({
                "block": block,
                "chain": {"rounds": chain.rounds, "committed": chain.committed, "per_round": c},
                "tree": {"rounds": tree.rounds, "committed": tree.committed, "per_round": t},
                "oracle": {"rounds": oracle.rounds, "committed": oracle.committed, "per_round": o},
                "tree_over_chain": t / c,
            }));
        }
        per_prompt
            .push(json!({"prompt": name, "prompt_tokens": prompt.len(), "blocks": block_recs}));
    }
    drop(runner);

    let after = monitor.sample(None);
    let (eligible, reasons) = eligibility(&[before, after], &rules);
    if !eligible {
        println!("  DIRTY: {}", reasons.join("; "));
    }
    let serve_gain = serve_gains.iter().sum::<f64>() / serve_gains.len().max(1) as f64;
    let worth = serve_gain >= MIN_TREE_GAIN;
    println!(
        "\n'c/rd' is committed tokens per verify round (accepted + bonus), at equal verify rows.\n\
         Mean tree gain at the served block {serve_block}: {:.1}% against a {:.0}% bar: {}.\n\
         Deeper blocks only count if a cheaper verify makes them pay (--serve-block).\n\
         A count, not a speed: neither tree overhead nor c(M) is modelled.",
        serve_gain * 100.0,
        MIN_TREE_GAIN * 100.0,
        if worth {
            "clears it; the next step is a cost model at this block"
        } else {
            "does not clear it; tree verify is not worth building for this drafter"
        }
    );
    ledger.append(
        "tree_replay",
        json!({"model": dir.display().to_string(), "family": arch.family.as_str(),
               "drafter": format!("{drafter:?}").to_lowercase(), "depth": depth,
               "top_k": if drafter == Drafter::Mtp { top_k } else { 16 },
               "generate": generate, "results": per_prompt,
               "serve_block": serve_block, "serve_tree_gain": serve_gain,
               "min_tree_gain": MIN_TREE_GAIN,
               "clears_bar": worth, "eligible": eligible, "reasons": reasons,
               "provenance": provenance}),
    )?;
    Ok(())
}

/// The lattices were drafted against the recorded stream; if drafting moved
/// the trunk off the plain greedy stream, they describe some other decode.
fn check_stream(name: &str, reference: &[i32], stream: &[i32]) -> Result<(), String> {
    match reference.iter().zip(stream).position(|(a, b)| a != b) {
        None => Ok(()),
        Some(at) => Err(format!(
            "{name}: drafting changed the greedy stream at token {at}; the lattices are void"
        )),
    }
}
