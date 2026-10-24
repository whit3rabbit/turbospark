//! Lattice recorders for [`crate::tree_replay`]: a greedy sequential decode
//! on a real runner that drafts a fresh lattice after every committed token.
//!
//! Shared by `tests/kflash_tree_replay_probe.rs` and `tslab tree-replay`, so
//! both measure the same thing on every family that has a drafter.
//!
//! One bias, and its direction: every drafter row here is primed from the
//! trunk's real hidden state, while the live loop's rows for accepted
//! proposals come from the drafter's own residual stream (MTP) or from
//! draft-written KV until the next context write (DFlash2). So the replay
//! drafter is a little better informed than the live one, and absolute
//! committed/round reads HIGH. The decision number is the tree/chain RATIO,
//! which both arms share the bias of.

use foundation::LogitValue;
use runtime::{LogitProducer, RealForwardRunner};

use crate::tree_replay::{DepthCandidates, Lattice};

/// Greedy argmax, first maximum on ties.
pub fn argmax(logits: &[LogitValue]) -> i32 {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, v) in logits.iter().enumerate() {
        let f = v.to_f32();
        if f > best_v {
            best_v = f;
            best = i;
        }
    }
    best as i32
}

/// Top-k by logit with full-vocabulary softmax probabilities.
pub fn top_k(logits: &[LogitValue], k: usize) -> Result<DepthCandidates, String> {
    let max = logits
        .iter()
        .map(|v| v.to_f32())
        .fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        return Err("non-finite draft logits".into());
    }
    let mut sum = 0.0f64;
    let mut best: Vec<(f32, i32)> = Vec::with_capacity(k + 1);
    for (i, v) in logits.iter().enumerate() {
        let f = v.to_f32();
        sum += ((f - max) as f64).exp();
        if best.len() < k || f > best[best.len() - 1].0 {
            let at = best.partition_point(|&(b, _)| b >= f);
            best.insert(at, (f, i as i32));
            best.truncate(k);
        }
    }
    Ok(DepthCandidates {
        tokens: best.iter().map(|&(_, t)| t).collect(),
        probs: best
            .iter()
            .map(|&(f, _)| (((f - max) as f64).exp() / sum) as f32)
            .collect(),
    })
}

fn err<E: std::fmt::Display>(what: &str) -> impl Fn(E) -> String + '_ {
    move |e| format!("{what}: {e}")
}

/// Plain greedy, no drafter: the reference stream.
pub fn plain(
    runner: &mut RealForwardRunner,
    prompt: &[i32],
    generate: usize,
) -> Result<Vec<i32>, String> {
    let mut logits = vec![LogitValue::from_f32(0.0); runner.vocab_size()];
    runner.reset();
    for (i, &t) in prompt.iter().enumerate() {
        runner.produce(t, i, &mut logits).map_err(err("produce"))?;
    }
    let mut out = Vec::with_capacity(generate);
    let mut next = argmax(&logits);
    while out.len() < generate {
        out.push(next);
        runner
            .produce(next, prompt.len() + out.len() - 1, &mut logits)
            .map_err(err("produce"))?;
        next = argmax(&logits);
    }
    Ok(out)
}

/// The greedy decode with an MTP head drafting a depth-`depth` chain after
/// every committed token, keeping the top `k` per depth. `lattices[i]` is
/// drafted once `stream[i]` is known and predicts `stream[i + 1 ..]`.
pub fn record_mtp(
    runner: &mut RealForwardRunner,
    prompt: &[i32],
    generate: usize,
    depth: usize,
    k: usize,
) -> Result<(Vec<i32>, Vec<Option<Lattice>>), String> {
    if runner.mtp_draft_depth() < depth {
        return Err(format!(
            "the MTP head drafts {} deep, {depth} asked",
            runner.mtp_draft_depth()
        ));
    }
    let vocab = runner.vocab_size();
    let mut logits = vec![LogitValue::from_f32(0.0); vocab];
    let mut draft = vec![LogitValue::from_f32(0.0); vocab];
    runner.reset();
    for (i, &t) in prompt.iter().enumerate() {
        runner.produce(t, i, &mut logits).map_err(err("produce"))?;
        if i + 1 < prompt.len() {
            runner
                .mtp_prime_step(prompt[i + 1], i)
                .map_err(err("prime"))?;
        }
    }
    let mut stream = Vec::with_capacity(generate);
    let mut lattices = Vec::with_capacity(generate);
    let mut next = argmax(&logits);
    while stream.len() < generate {
        stream.push(next);
        // `next` sits at `base`; the head's row `base - 1` pairs h_{base-1}
        // with it, which is exactly the priming step, so drafting depth 0
        // leaves that row correct and the rewind below keeps only it.
        let base = prompt.len() + stream.len() - 1;
        let mut depths = Vec::with_capacity(depth);
        let mut chained = next;
        for d in 0..depth {
            runner
                .mtp_draft_step(chained, base - 1 + d, &mut draft)
                .map_err(err("draft step"))?;
            let c = top_k(&draft, k)?;
            chained = c.tokens[0];
            depths.push(c);
        }
        runner.mtp_rewind_to(base).map_err(err("rewind the head"))?;
        lattices.push(Some(Lattice { depths }));
        runner
            .produce(next, base, &mut logits)
            .map_err(err("produce"))?;
        next = argmax(&logits);
    }
    Ok((stream, lattices))
}

/// The greedy decode with a DFlash2 drafter (opened at the block it drafts)
/// reading back its whole candidate lattice after every committed token.
///
/// A sequential `produce` leaves exactly one captured row at `base - 1`, so
/// each round's context write covers one row and the drafter's cursor sits
/// at `base` when it drafts; its block rows above the cursor are rewritten
/// before anything reads them again (`dflash_rewind_to`'s contract).
pub fn record_dflash(
    runner: &mut RealForwardRunner,
    prompt: &[i32],
    generate: usize,
) -> Result<(Vec<i32>, Vec<Option<runtime::DflashLattice>>), String> {
    let mut logits = vec![LogitValue::from_f32(0.0); runner.vocab_size()];
    runner.reset();
    for (i, &t) in prompt.iter().enumerate() {
        runner.produce(t, i, &mut logits).map_err(err("produce"))?;
        if i + 1 < prompt.len() {
            runner
                .dflash_prime_from_capture(i)
                .map_err(err("prime the drafter"))?;
        }
    }
    let mut stream = Vec::with_capacity(generate);
    let mut lattices = Vec::with_capacity(generate);
    let mut proposals = Vec::new();
    let mut next = argmax(&logits);
    while stream.len() < generate {
        stream.push(next);
        let base = prompt.len() + stream.len() - 1;
        runner
            .dflash_draft_block(next, base, &mut proposals)
            .map_err(err("draft block"))?;
        let lattice = runner
            .dflash_lattice(next)
            .map_err(err("read the lattice"))?;
        // The selector's own proposals must be the lattice's greedy path, or
        // the lattice is not what the selector walked.
        let greedy: Vec<i32> = lattice
            .greedy_path()
            .iter()
            .enumerate()
            .map(|(d, &c)| lattice.candidates[d][c])
            .collect();
        if greedy != proposals {
            return Err(format!(
                "lattice greedy path {greedy:?} differs from the selector's {proposals:?}"
            ));
        }
        lattices.push(Some(lattice));
        runner
            .dflash_rewind_to(base)
            .map_err(err("rewind the drafter"))?;
        runner
            .produce(next, base, &mut logits)
            .map_err(err("produce"))?;
        next = argmax(&logits);
    }
    Ok((stream, lattices))
}
