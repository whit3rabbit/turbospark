//! The speculative decode loop: draft a block, verify it in one batched
//! pass, commit the accepted prefix plus the free bonus token.
//!
//! This is the DELIVERY half of `docs/MTP.md`. The measurement half landed
//! first and lives in `crates/bench/tests/mtp_accept_length_probe.rs`, which
//! measured 1.44x at block 2 on the real `qwen3_5` MTP install and then
//! reached no user, because the primitives it drives
//! (`RealForwardRunner::{mtp_draft_step, mtp_prime_step, produce_batched,
//! checkpoint, rollback, mtp_rewind_to}`) were called from that probe and
//! from nothing else. [`run_raw_completion_speculative`] is the entry point
//! that puts them behind the same generation contract
//! [`crate::run_raw_completion`] offers.
//!
//! **The loop commits tokens through [`crate::raw_completion::TokenSink`],
//! the same path a sequential decode uses.** Everything a caller observes --
//! the stop-token ladder, the streamed deltas, the stop-string matcher's
//! withheld tail, `MaxTokens`, cancellation and their precedence -- is
//! therefore shared code rather than a second implementation that agrees
//! with the first until it does not. The losslessness gate cannot see any of
//! it: that gate compares the tokens the two paths COMMIT, and every one of
//! those behaviours is downstream of the commit.

use std::time::Instant;

use foundation::{LogitValue, LogitsView, TokenId};
use selection::derive::{derive_step_value, to_unit_interval};
use selection::{residual, select, shaped_distribution, ShapedDistribution};
use tokenizer::MfTokenizer;

use crate::config::GenerationConfig;
use crate::error::RuntimeError;
use crate::producer::SpeculativeProducer;
use crate::raw_completion::{
    CancelFlag, RawDecodeProgress, RawDecodeResult, StopReason, TokenSink, NEVER,
};
use crate::speculative_adaptive::AdaptiveDepth;

/// How many tokens a round proposes. Block 2 is the measured optimum on the
/// one real drafter (`docs/MTP.md`: 1.44x, against 1.03x at 4 and 0.71x at
/// 8), and the shape is not a tuning accident -- verify cost scales close to
/// linearly in the block, the accept chain decays, and the probability that a
/// round has to roll back rises from 10% at block 2 to 98% at 15. A caller
/// with a different drafter should re-measure rather than assume this
/// transfers.
pub const DEFAULT_SPECULATION_BLOCK: usize = 2;

/// Runs a speculative generation. `block` is how many tokens each round
/// proposes; [`DEFAULT_SPECULATION_BLOCK`] is the measured optimum.
///
/// Refuses rather than falling back when it cannot serve the request
/// losslessly. See [`RuntimeError::SpeculationUnavailable`] for why silence
/// would be the worse answer.
#[allow(clippy::too_many_arguments)]
pub fn run_raw_completion_speculative<P: SpeculativeProducer>(
    producer: &mut P,
    tokenizer: &MfTokenizer,
    prompt_ids: &[TokenId],
    config: &GenerationConfig,
    max_context: u32,
    vocab_size: usize,
    block: usize,
    on_progress: impl FnMut(RawDecodeProgress),
) -> Result<RawDecodeResult, RuntimeError> {
    run_raw_completion_speculative_cancellable(
        producer,
        tokenizer,
        prompt_ids,
        config,
        max_context,
        vocab_size,
        block,
        NEVER,
        on_progress,
    )
}

/// [`run_raw_completion_speculative`], polling `cancel` once per prefill
/// token and once per COMMITTED token.
///
/// The cancellation granularity is the committed token and not the round: a
/// round that commits three tokens polls three times, through the same
/// [`TokenSink::commit`] a sequential decode uses. What a caller does not get
/// is a poll DURING a verify pass, which is one batched forward and is not
/// interruptible.
#[allow(clippy::too_many_arguments)]
pub fn run_raw_completion_speculative_cancellable<P: SpeculativeProducer>(
    producer: &mut P,
    tokenizer: &MfTokenizer,
    prompt_ids: &[TokenId],
    config: &GenerationConfig,
    max_context: u32,
    vocab_size: usize,
    block: usize,
    cancel: CancelFlag<'_>,
    on_progress: impl FnMut(RawDecodeProgress),
) -> Result<RawDecodeResult, RuntimeError> {
    run_speculative(
        producer,
        tokenizer,
        prompt_ids,
        config,
        max_context,
        vocab_size,
        block,
        None,
        cancel,
        on_progress,
    )
}

/// [`run_raw_completion_speculative`] with an ADAPTIVE round size: `cap` is
/// the most a round may propose (the drafter must have been opened at least
/// that deep) and `floor` the least, with [`AdaptiveDepth`] moving between
/// them from the acceptance each round saw. Opt-in; the fixed entry points
/// above are unchanged. See `speculative_adaptive` for why every constant
/// behind this is unmeasured until the paired gate has run.
#[allow(clippy::too_many_arguments)]
pub fn run_raw_completion_speculative_adaptive<P: SpeculativeProducer>(
    producer: &mut P,
    tokenizer: &MfTokenizer,
    prompt_ids: &[TokenId],
    config: &GenerationConfig,
    max_context: u32,
    vocab_size: usize,
    cap: usize,
    floor: usize,
    on_progress: impl FnMut(RawDecodeProgress),
) -> Result<RawDecodeResult, RuntimeError> {
    run_raw_completion_speculative_adaptive_cancellable(
        producer,
        tokenizer,
        prompt_ids,
        config,
        max_context,
        vocab_size,
        cap,
        floor,
        NEVER,
        on_progress,
    )
}

/// [`run_raw_completion_speculative_adaptive`], polling `cancel` the way
/// [`run_raw_completion_speculative_cancellable`] does.
#[allow(clippy::too_many_arguments)]
pub fn run_raw_completion_speculative_adaptive_cancellable<P: SpeculativeProducer>(
    producer: &mut P,
    tokenizer: &MfTokenizer,
    prompt_ids: &[TokenId],
    config: &GenerationConfig,
    max_context: u32,
    vocab_size: usize,
    cap: usize,
    floor: usize,
    cancel: CancelFlag<'_>,
    on_progress: impl FnMut(RawDecodeProgress),
) -> Result<RawDecodeResult, RuntimeError> {
    run_speculative(
        producer,
        tokenizer,
        prompt_ids,
        config,
        max_context,
        vocab_size,
        cap,
        Some(AdaptiveDepth::new(
            floor,
            AdaptiveDepth::tile_cap(cap, verify_tile_rows()),
        )),
        cancel,
        on_progress,
    )
}

/// The rows one pass of the INT4 MMA8 tile verifies, when the tile is in
/// the verify path (`TURBOSPARK_INT4_MMA8` set, either mode), so the
/// adaptive cap never asks for a width that pays a second pass
/// ([`AdaptiveDepth::tile_cap`]). `None` off macOS and with the tile off.
#[cfg(target_os = "macos")]
fn verify_tile_rows() -> Option<usize> {
    use crate::real_forward_dispatch::{int4_mma8_mode, Int4Mma8Mode};
    (int4_mma8_mode() != Int4Mma8Mode::Off).then_some(gpu::INT4_MMA8_MAX_ROWS)
}

#[cfg(not(target_os = "macos"))]
fn verify_tile_rows() -> Option<usize> {
    None
}

/// The one loop. `block` is the CAP: with `adaptive` None every round
/// proposes `block`, which is the loop exactly as it was before the
/// controller existed.
#[allow(clippy::too_many_arguments)]
fn run_speculative<P: SpeculativeProducer>(
    producer: &mut P,
    tokenizer: &MfTokenizer,
    prompt_ids: &[TokenId],
    config: &GenerationConfig,
    max_context: u32,
    vocab_size: usize,
    block: usize,
    mut adaptive: Option<AdaptiveDepth>,
    cancel: CancelFlag<'_>,
    mut on_progress: impl FnMut(RawDecodeProgress),
) -> Result<RawDecodeResult, RuntimeError> {
    crate::raw_completion::check_admission(prompt_ids, config, max_context)?;

    if block == 0 {
        return Err(RuntimeError::SpeculationUnavailable(
            "block size 0 proposes nothing; call run_raw_completion instead".to_string(),
        ));
    }

    // THE SAMPLING GATE, and it now has exactly one refusal left.
    //
    // Acceptance below is `target == proposal` at temperature 0, which is
    // exact speculative decoding; at any other temperature the exact
    // algorithm is rejection sampling with residual correction (Leviathan
    // et al., arXiv 2211.17192; Chen et al., arXiv 2302.01318), which
    // needs the drafter's and the target's full SHAPED distributions.
    // The MTP step drafter has both on the host already, so a sampled run
    // takes the rejection path below and is exact in DISTRIBUTION: the
    // committed stream matches what the sequential sampler would have
    // produced in law, not token for token. The DFlash2 BLOCK drafter
    // cannot take that path at all -- its selector is a greedy structured
    // search (`unary + bilinear dot`), not a distribution, so there is no
    // q(x) to ratio against and no residual to sample -- and is refused
    // here rather than approximated, for the same reason it always was:
    // a quiet narrowing of the sampled distribution while reporting
    // success would change what the model writes.
    if !config.shaping.is_deterministic() && producer.drafts_block_passes() {
        return Err(RuntimeError::SpeculationUnavailable(
            "sampled speculation cannot be served by the block drafter: DFlash2's selection is \
             a greedy structured search, not a distribution, so exact rejection sampling has \
             no q(x) to ratio against; use the mtp drafter or temperature 0"
                .to_string(),
        ));
    }

    // THE REUSE CONTRACT, the sequential loop's own: ask how much of this
    // prompt the producer's state already covers, and reset only when the
    // answer is none. On the real producer a drafter install reuses only
    // the PURE-CONTINUATION case (`try_reuse_prefix`'s drafter guard) --
    // exactly the multi-turn case a chat client generates, and the case
    // this loop used to re-prefill in full every turn.
    let reused = producer
        .try_reuse_prefix(prompt_ids)
        .min(prompt_ids.len() - 1);
    if reused == 0 {
        producer.reset();
    }
    let mut logits = vec![LogitValue::from_f32(0.0); vocab_size];
    let mut draft_logits = vec![LogitValue::from_f32(0.0); vocab_size];
    // The confirmed token plus every proposal.
    let mut batch_logits = vec![LogitValue::from_f32(0.0); (block + 1) * vocab_size];
    let mut history: Vec<TokenId> =
        Vec::with_capacity(prompt_ids.len() + config.max_new_tokens as usize);

    // -- Prefill, priming the drafter as it goes.
    //
    // Every prompt token but the LAST goes through `produce_prefill`, which
    // skips the output head -- a full-vocab GEMV per prompt token that
    // nothing here reads. The sequential loop has always done exactly this
    // (`run_raw_completion`'s own prefill); the speculative loop used to pay
    // the head on every token because the drafter reads the trunk's hidden
    // state, and whether that input survived `skip_head` was unmeasured.
    // It is measured now, and the answer is PER DRAFTER
    // (`SpeculativeProducer::supports_headless_prefill`): the DFlash2 taps
    // are raw layer outputs, captured per layer ahead of the skip early
    // return, so they are bit-identical either way -- the fixture pins it.
    // The MTP head reads the POST-FINAL-NORM hidden, which the skip also
    // skips, so it keeps the headful walk. The LAST token always pays the
    // head: the first `next` is sampled from its logits.
    //
    // The reused positions are already in the producer's KV -- and, by the
    // drafter guard that allowed the reuse, in the drafter's context KV too
    // -- so both cursors start past them and `history` is seeded with the
    // ids that built them.
    let headless = producer.supports_headless_prefill();
    let prefill_start = Instant::now();
    let mut position = reused;
    history.extend_from_slice(&prompt_ids[..reused]);
    // A producer that can prime its drafter from a chunked prefill takes the
    // prompt in the chunked driver's spans instead: the same drafter pairs,
    // primed after each micro-batch, at the batched trunk's prefill speed
    // rather than one full forward pass (head included) per prompt token.
    let chunked = producer.supports_chunked_drafter_prefill();
    // Where the producer keeps its state for the next turn, if anywhere
    // (`LogitProducer::turn_snapshot_position`): both walks stop there for
    // one call. The drafter is primed through the cut by then, because the
    // pass that ends at it primes its last row with the known next token.
    let cut = producer.turn_snapshot_position(prompt_ids, reused);
    if chunked {
        let spans = crate::raw_completion_chunked::prefill_spans_with_cut(
            prompt_ids.len() - reused,
            reused,
            producer.drafter_prefill_chunk(),
            cut,
        );
        for span in &spans {
            if cut == Some(span.start_position) {
                producer.take_turn_snapshot(span.start_position);
            }
            let begin = reused + span.token_offset;
            let end = begin + span.token_count;
            let chunk = &prompt_ids[begin..end];
            producer
                .prefill_chunk_priming_drafter(
                    chunk,
                    span.start_position,
                    &mut logits,
                    prompt_ids.get(end).copied(),
                )
                .map_err(RuntimeError::Producer)?;
            history.extend_from_slice(chunk);
            position = reused + span.completed_count;
            on_progress(RawDecodeProgress::Prefill {
                done: position,
                total: prompt_ids.len(),
            });
            if cancel() {
                return Ok(crate::raw_completion::cancelled_during_prefill(
                    history,
                    position,
                    prompt_ids.len(),
                    prefill_start,
                    reused,
                    producer.session_slot_evicted(),
                ));
            }
        }
    }
    for (i, &token) in
        prompt_ids
            .iter()
            .enumerate()
            .skip(if chunked { prompt_ids.len() } else { reused })
    {
        if cut == Some(i) {
            producer.take_turn_snapshot(i);
        }
        let last = i + 1 == prompt_ids.len();
        if last || !headless {
            producer
                .produce(token, position, &mut logits)
                .map_err(RuntimeError::Producer)?;
        } else {
            producer
                .produce_prefill(token, position, &mut logits)
                .map_err(RuntimeError::Producer)?;
        }
        if i + 1 < prompt_ids.len() {
            producer
                .prime_drafter(prompt_ids[i + 1], i)
                .map_err(RuntimeError::Producer)?;
        }
        position += 1;
        history.push(token);
        on_progress(RawDecodeProgress::Prefill {
            done: position,
            total: prompt_ids.len(),
        });
        if cancel() {
            return Ok(crate::raw_completion::cancelled_during_prefill(
                history,
                position,
                prompt_ids.len(),
                prefill_start,
                reused,
                producer.session_slot_evicted(),
            ));
        }
    }
    let prefill_seconds = prefill_start.elapsed().as_secs_f64();

    // -- Decode.
    let decode_start = Instant::now();
    let mut sink = TokenSink::new(tokenizer, config, history);
    let mut proposals: Vec<TokenId> = Vec::with_capacity(block);
    // The drafter's shaped distribution per proposal, kept for the
    // rejection step. Only the step-wise drafter fills it; the block
    // drafter is refused above before any of this runs sampled.
    let mut draft_qs: Vec<ShapedDistribution> = Vec::with_capacity(block);
    let mut feed: Vec<TokenId> = Vec::with_capacity(block + 1);
    let reason;
    // TREE ROUNDS (`TURBOSPARK_DFLASH_TREE`): greedy and fixed-depth only,
    // and only at the producer's own node count, so a tree round verifies
    // the rows a chain round would. Budget-shortened rounds and adaptive
    // runs keep the chain.
    let tree_nodes = producer
        .tree_nodes()
        .filter(|&n| n == block && config.shaping.is_deterministic());
    let mut tree: Vec<crate::DflashTreeNode> = Vec::with_capacity(block);
    let mut tree_tokens: Vec<TokenId> = Vec::with_capacity(block);

    // Every uniform THIS loop draws comes from one monotonic counter, so a
    // seeded run is reproducible and no two draws share a value. Sharing
    // one would not be a performance question but a correctness one: the
    // acceptance uniform must be independent of the uniform that drew the
    // proposal it judges, or the accepted stream is biased by construction.
    let sampled = !config.shaping.is_deterministic();
    let mut draw_step: u64 = 0;

    let mut next = select(
        LogitsView::new(&logits),
        &config.shaping,
        &sink.history,
        sink.generated as u64,
    )?;

    // THE ACCEPTANCE COUNTER, off by default and env-gated like every other
    // diagnostic here (`TURBOSPARK_PHASES`, `TURBOSPARK_ROUTER_HIST`). It exists
    // because nothing else can see this loop's acceptance: both probes
    // hand-roll their own round, so a drafter that measures 7.09 of 8 in
    // `dflash2_accept_length_probe` and 0 here reads as a THROUGHPUT
    // mystery rather than as the acceptance gap it is.
    let stats = std::env::var("TURBOSPARK_SPEC_STATS").as_deref() == Ok("1");
    let mut stat_rounds = 0usize;
    let mut stat_accepted = 0usize;
    let mut stat_offered = vec![0usize; block];
    let mut stat_matched = vec![0usize; block];
    let mut stat_rollbacks = 0usize;
    // Wall time per phase of a round, stats runs only: draft, verify,
    // accept, rollback, commit, rewind, bonus. Every phase that reads back
    // from the GPU is a sync point, so these are wall times, not GPU times.
    let mut ph = [std::time::Duration::ZERO; 7];
    // Rounds per proposed depth, so an adaptive run shows where it sat.
    let mut stat_depth = vec![0usize; block + 1];

    'rounds: loop {
        // `next` was sampled but not yet fed. Committing it first is what
        // makes the invariant below hold: after every round,
        // `position == sink.history.len()`.
        if let Some(stop) = sink.commit(next, cancel, &mut on_progress) {
            reason = stop;
            break;
        }
        // The index `next` occupies. At least 1, since the prompt is
        // non-empty and `next` sits after it.
        let base = sink.history.len() - 1;
        position = sink.history.len();

        // Never propose past the generation budget or the context window.
        // Without this the last round of a run overshoots both: it would
        // feed tokens the budget forbids (paying a guaranteed rollback to
        // give them back) and could write KV rows beyond `max_context`,
        // which `check_admission` sized for the COMMITTED stream.
        let budget_room = (config.max_new_tokens as usize).saturating_sub(sink.generated);
        let context_room = (max_context as usize).saturating_sub(position);
        let wanted = adaptive.as_ref().map_or(block, |a| a.depth().min(block));
        let round_block = wanted.min(budget_room).min(context_room);
        // A round shortened by the budget or the window says nothing about
        // acceptance, so it must not feed the controller.
        let truncated = round_block < wanted;
        if round_block == 0 {
            return Err(RuntimeError::Producer(
                "round_block is 0: no room after admission; invariant violation".to_string(),
            ));
        }

        // -- A TREE round, when the producer verifies trees. The walk is the
        //    chain's acceptance generalized: at the current node's row, the
        //    target's pick is accepted when a child of that node carries it.
        //    The accepted path is then compacted into rows `0..keep`, after
        //    which retaining rollback, the fold, the record and the drafter
        //    rewind run exactly as for a chain of that length.
        if let Some(nodes) = tree_nodes.filter(|_| round_block == block && adaptive.is_none()) {
            let t_draft = std::time::Instant::now();
            producer
                .draft_tree(next, base, nodes, &mut tree, &mut tree_tokens)
                .map_err(RuntimeError::Producer)?;
            ph[0] += t_draft.elapsed();
            let parents = crate::dflash_tree_slot_parents(&tree);
            feed.clear();
            feed.push(next);
            feed.extend_from_slice(&tree_tokens);
            let t_verify = std::time::Instant::now();
            let point = producer.checkpoint();
            producer
                .verify_tree(
                    &feed,
                    &parents,
                    base,
                    &mut batch_logits[..feed.len() * vocab_size],
                )
                .map_err(RuntimeError::Producer)?;
            ph[1] += t_verify.elapsed();
            stat_rounds += 1;
            if stats {
                stat_depth[round_block] += 1;
            }
            // (committed, bonus, stop, accepted path of slots).
            type TreeRound = (usize, Option<TokenId>, Option<StopReason>, Vec<u32>);
            let mut tree_round = || -> Result<TreeRound, RuntimeError> {
                let t_accept = std::time::Instant::now();
                let mut path: Vec<u32> = vec![0];
                let mut at: Option<usize> = None;
                let mut stopped = None;
                let mut bonus = None;
                loop {
                    let slot = at.map_or(0, |id| id + 1);
                    let row = &batch_logits[slot * vocab_size..(slot + 1) * vocab_size];
                    let target = select(
                        LogitsView::new(row),
                        &config.shaping,
                        &sink.history,
                        sink.generated as u64,
                    )?;
                    let depth = path.len() - 1;
                    let offered = tree.iter().any(|n| n.parent == at);
                    if stats && offered && depth < block {
                        stat_offered[depth] += 1;
                    }
                    let hit = (0..tree.len())
                        .find(|&id| tree[id].parent == at && tree_tokens[id] == target);
                    let Some(id) = hit else {
                        bonus = Some(target);
                        break;
                    };
                    if stats {
                        stat_matched[depth] += 1;
                    }
                    path.push(id as u32 + 1);
                    at = Some(id);
                    if let Some(stop) = sink.commit(target, cancel, &mut on_progress) {
                        stopped = Some(stop);
                        break;
                    }
                }
                stat_accepted += path.len() - 1;
                let committed = sink.history.len() - (base + 1);
                ph[2] += t_accept.elapsed();
                let t_roll = std::time::Instant::now();
                producer
                    .compact_tree(base, &path[..committed + 1])
                    .map_err(RuntimeError::Producer)?;
                if committed + 1 < feed.len() {
                    stat_rollbacks += 1;
                    producer
                        .rollback_retaining(&point, committed + 1)
                        .map_err(RuntimeError::Producer)?;
                }
                ph[3] += t_roll.elapsed();
                let t_commit = std::time::Instant::now();
                producer
                    .commit_pending(committed + 1)
                    .map_err(RuntimeError::Producer)?;
                ph[4] += t_commit.elapsed();
                Ok((committed, bonus, stopped, path))
            };
            let (committed, bonus, stopped, path) = match tree_round() {
                Ok(r) => r,
                Err(e) => {
                    producer.discard_pending();
                    return Err(e);
                }
            };
            position = sink.history.len();
            let t_rewind = std::time::Instant::now();
            let path_feed: Vec<TokenId> = path[..committed + 1]
                .iter()
                .map(|&s| feed[s as usize])
                .collect();
            producer.record_committed(&path_feed, base);
            producer
                .rewind_drafter(base + committed)
                .map_err(RuntimeError::Producer)?;
            ph[5] += t_rewind.elapsed();
            if let Some(stop) = stopped {
                reason = stop;
                break 'rounds;
            }
            next = bonus.expect("a walk that did not stop ends on a bonus pick");
            continue 'rounds;
        }

        // -- Draft. Two shapes, and the producer says which it is:
        //
        //    A STEP-WISE drafter (the MTP head) takes `round_block + 1`
        //    steps for `round_block` proposals: the last one is taken for
        //    its cache ROW alone, because a round where every proposal is
        //    accepted needs a drafter row at `base + round_block` that the
        //    proposal-producing steps do not write. Without it the
        //    fully-accepted case is the one that desyncs, which is the case
        //    a good drafter hits most often.
        //
        //    A BLOCK drafter (DFlash2) proposes the whole block in ONE
        //    pass, its bonus row carrying the anchor's embedding and its
        //    mask rows carrying the proposals. Its own state management --
        //    the context-KV write for the accepted prefix, its cache cursor
        //    -- happens inside the call, from the capture the previous
        //    verify filled.
        let t_draft = std::time::Instant::now();
        proposals.clear();
        draft_qs.clear();
        if producer.drafts_block_passes() {
            producer
                .draft_block(next, base, round_block, &mut proposals)
                .map_err(RuntimeError::Producer)?;
            assert_eq!(
                proposals.len(),
                round_block,
                "draft_block must populate exactly round_block proposals ({round_block} expected, {} produced)",
                proposals.len()
            );
        } else {
            let mut chained = next;
            for d in 0..=round_block {
                if d == round_block {
                    // The row-only step: its proposal is never offered, so
                    // it goes through `prime_drafter`, which writes the same
                    // drafter row without the full-vocab head, the logits
                    // readback or a selection nobody reads. On Qwen3.8 that
                    // head is ~0.7 GB of the ~0.95 GB a draft step reads.
                    // A sampled run still advances `draw_step` once here, as
                    // the step that drew and discarded a token did, so the
                    // sampled stream stays byte-identical to that path.
                    producer
                        .prime_drafter(chained, base - 1 + d)
                        .map_err(RuntimeError::Producer)?;
                    if sampled {
                        draw_step += 1;
                    }
                    break;
                }
                producer
                    .draft_step(chained, base - 1 + d, &mut draft_logits)
                    .map_err(RuntimeError::Producer)?;
                if sampled {
                    // The proposal is DRAWN from the drafter's own shaped
                    // distribution -- the q of the rejection ratio -- rather
                    // than argmaxed, and the distribution is kept for the
                    // acceptance step. Same shaping, same history state, so
                    // this is the q a sequential decode through the drafter
                    // would have sampled.
                    let q = shaped_distribution(
                        LogitsView::new(&draft_logits),
                        &config.shaping,
                        &sink.history,
                        sink.generated as u64,
                    )?;
                    chained = q.draw(config.shaping.seed(), draw_step) as TokenId;
                    draw_step += 1;
                    if d < round_block {
                        proposals.push(chained);
                        draft_qs.push(q);
                    }
                } else {
                    chained = select(
                        LogitsView::new(&draft_logits),
                        &config.shaping,
                        &sink.history,
                        sink.generated as u64,
                    )?;
                    if d < round_block {
                        proposals.push(chained);
                    }
                }
            }
        }

        // -- Verify. One batched pass over the confirmed token plus every
        //    proposal. Row `i` predicts the token after `feed[i]`, so row 0
        //    is checked against `proposals[0]`, and the row after the last
        //    accepted proposal carries the free bonus token.
        ph[0] += t_draft.elapsed();
        feed.clear();
        feed.push(next);
        feed.extend_from_slice(&proposals);
        let t_verify = std::time::Instant::now();
        let point = producer.checkpoint();
        producer
            .verify(&feed, base, &mut batch_logits[..feed.len() * vocab_size])
            .map_err(RuntimeError::Producer)?;
        ph[1] += t_verify.elapsed();

        // -- Accept. Each accepted proposal is committed through the sink
        // IMMEDIATELY, which is what keeps the shaped distributions exact:
        // the next row is evaluated with the history and step counter a
        // sequential decode would have had at that position, so a
        // repetition penalty or a step-dependent shaping rule sees the same
        // inputs on both paths.
        //
        // At temperature 0 acceptance is argmax equality, as it always
        // was. Sampled, it is the Leviathan/Chen rejection step: proposal
        // x (drawn from q above) is accepted when r * q(x) <= p(x) for a
        // fresh uniform r, and on rejection the corrected token is drawn
        // from the normalized residual max(p - q, 0) -- the draw that makes
        // the composite match the sequential sampler in distribution.
        let mut accepted = 0usize;
        let mut stopped: Option<StopReason> = None;
        let mut corrected: Option<TokenId> = None;
        stat_rounds += 1;
        let mut evaluate_round = || -> Result<usize, RuntimeError> {
            let t_accept = std::time::Instant::now();
            for (i, &proposal) in proposals.iter().enumerate() {
                if stats {
                    stat_offered[i] += 1;
                }
                let row = &batch_logits[i * vocab_size..(i + 1) * vocab_size];
                if sampled {
                    let p = shaped_distribution(
                        LogitsView::new(row),
                        &config.shaping,
                        &sink.history,
                        sink.generated as u64,
                    )?;
                    let q = &draft_qs[i];
                    let r = to_unit_interval(derive_step_value(config.shaping.seed(), draw_step));
                    draw_step += 1;
                    if r * q.probability(proposal as u32) <= p.probability(proposal as u32) {
                        if stats {
                            stat_matched[i] += 1;
                        }
                        accepted += 1;
                        if let Some(stop) = sink.commit(proposal, cancel, &mut on_progress) {
                            stopped = Some(stop);
                            break;
                        }
                    } else {
                        // The correction comes from the residual over THIS
                        // row's target distribution and the drafter's q for
                        // this step. The p == q corner underflows the residual
                        // and falls back to a p draw: exact in the limit, and
                        // unreachable while q is the drafter's own distribution
                        // unless the two really are identical (where every
                        // proposal is accepted and no correction is drawn).
                        corrected = Some(match residual(&p, q) {
                            Some(residual_p) => residual_p.draw(config.shaping.seed(), draw_step),
                            None => p.draw(config.shaping.seed(), draw_step),
                        } as TokenId);
                        draw_step += 1;
                        break;
                    }
                } else {
                    let target = select(
                        LogitsView::new(row),
                        &config.shaping,
                        &sink.history,
                        sink.generated as u64,
                    )?;
                    if target != proposal {
                        break;
                    }
                    if stats {
                        stat_matched[i] += 1;
                    }
                    accepted += 1;
                    if let Some(stop) = sink.commit(proposal, cancel, &mut on_progress) {
                        stopped = Some(stop);
                        break;
                    }
                }
            }
            stat_accepted += accepted;
            if stats {
                stat_depth[round_block] += 1;
            }
            // A round a stop cut short never saw the rest of its proposals, so
            // its `accepted` undercounts and would read as a miss.
            if let Some(a) = adaptive.as_mut() {
                if !truncated && stopped.is_none() {
                    a.observe(proposals.len(), accepted);
                }
            }
            // What the engine has absorbed beyond the committed stream. A
            // committed proposal advanced `history`; a rejected or unreached one
            // did not, and neither did a proposal that stopped the run.
            let committed = sink.history.len() - (base + 1);

            // -- Rewind, when the verify overshot what was committed.
            //
            //    A SEQUENTIAL verify stops at the first rejection and so never
            //    overshoots; only a batched pass can, which is the asymmetry that
            //    makes block 2 pay and block 8 lose (`docs/MTP.md`).
            //
            //    Two restore shapes. With a TAPE (the real producer on the GDN
            //    flow), the kept rows' KV stays in place and their recurrent
            //    state is replayed over the inputs the verify recorded -- a few
            //    small dispatches, and the verify's own logits for the bonus row
            //    remain valid, so no forward pass runs. Without one, the whole
            //    state goes back to the block start -- on the real drafter's
            //    family that is a copy of the entire gated-DeltaNet snapshot,
            //    because a recurrent layer cannot be rewound incrementally the
            //    way a KV cursor can -- and a shortened re-verify replays the
            //    accepted prefix through the model. The re-verify is the term
            //    that made throughput track the rollback rate
            //    (`docs/DFLASH2.md`); the tape removes it.
            ph[2] += t_accept.elapsed();
            let t_roll = std::time::Instant::now();
            if committed < proposals.len() {
                stat_rollbacks += 1;
                if producer.supports_retaining_rollback() {
                    producer
                        .rollback_retaining(&point, committed + 1)
                        .map_err(RuntimeError::Producer)?;
                } else {
                    producer.rollback(&point);
                    let keep = committed + 1;
                    producer
                        .verify(&feed[..keep], base, &mut batch_logits[..keep * vocab_size])
                        .map_err(RuntimeError::Producer)?;
                }
            }
            // THE LAZY COMMIT: with TURBOSPARK_GDN_LAZY_TAPE on (the default),
            // the verify left the recurrent state at the block start and
            // recorded rank-1 updates, so every round -- full acceptance
            // included, which is the case the eager path served for free --
            // ends by folding exactly the rows it committed. A no-op on the
            // eager path, and after the partial round's retaining rollback,
            // which folded its kept rows and cleared the tape.
            ph[3] += t_roll.elapsed();
            let t_commit = std::time::Instant::now();
            producer
                .commit_pending(committed + 1)
                .map_err(RuntimeError::Producer)?;
            ph[4] += t_commit.elapsed();
            Ok(committed)
        };
        let committed = match evaluate_round() {
            Ok(c) => c,
            Err(e) => {
                producer.discard_pending();
                return Err(e);
            }
        };
        position = sink.history.len();

        // THE RECORD, the half that lets the NEXT turn reuse: the verify fed
        // these rows and the round committed them, so the KV rows are real.
        // The batched half of what `produce` records for the plain-step
        // fallback above. `committed + 1` rows: the confirmed token plus
        // every committed proposal, positions `base .. base + committed +
        // 1` -- the bonus is not fed yet and belongs to the next round.
        let t_rewind = std::time::Instant::now();
        producer.record_committed(&feed[..committed + 1], base);

        // The drafter goes to where the accepted prefix ENDED and continues;
        // the target went back to where the block STARTED and replayed. Two
        // targets, which is why one call cannot do both.
        producer
            .rewind_drafter(base + committed)
            .map_err(RuntimeError::Producer)?;
        ph[5] += t_rewind.elapsed();

        if let Some(stop) = stopped {
            reason = stop;
            break 'rounds;
        }

        // The bonus: the row after the last accepted proposal. Free, in
        // the sense that the verify pass computed it whether or not the
        // block was accepted, and it is why a round commits `accepted + 1`
        // tokens. Sampled, the same row serves two DIFFERENT draws: on
        // full acceptance it is a fresh draw from the target's shaped
        // distribution (the token a sequential decode would sample next);
        // on a rejection it is the row the correction was already drawn
        // from, and the corrected token IS `next` -- a second draw from it
        // would double-sample one position.
        let t_bonus = std::time::Instant::now();
        if let Some(token) = corrected {
            next = token;
        } else if sampled {
            let p = shaped_distribution(
                LogitsView::new(&batch_logits[accepted * vocab_size..(accepted + 1) * vocab_size]),
                &config.shaping,
                &sink.history,
                sink.generated as u64,
            )?;
            next = p.draw(config.shaping.seed(), draw_step) as TokenId;
            draw_step += 1;
        } else {
            next = select(
                LogitsView::new(&batch_logits[accepted * vocab_size..(accepted + 1) * vocab_size]),
                &config.shaping,
                &sink.history,
                sink.generated as u64,
            )?;
        }
        ph[6] += t_bonus.elapsed();
    }

    if stats {
        let per_round = stat_accepted as f64 / stat_rounds.max(1) as f64;
        eprintln!(
            "[spec-stats] rounds={stat_rounds} accepted/round={per_round:.2} \
             committed/round={:.2} rollbacks={stat_rollbacks}",
            per_round + 1.0
        );
        let curve: Vec<String> = stat_offered
            .iter()
            .zip(&stat_matched)
            .map(|(o, m)| {
                if *o == 0 {
                    "-".to_string()
                } else {
                    format!("{:.2}", *m as f64 / *o as f64)
                }
            })
            .collect();
        eprintln!("[spec-stats] per-position acceptance: {}", curve.join(" "));
        let depths: Vec<String> = stat_depth
            .iter()
            .enumerate()
            .filter(|(_, n)| **n > 0)
            .map(|(d, n)| format!("{d}:{n}"))
            .collect();
        eprintln!("[spec-stats] rounds per depth: {}", depths.join(" "));
        let per = |d: std::time::Duration| d.as_secs_f64() * 1000.0 / stat_rounds.max(1) as f64;
        eprintln!(
            "[spec-stats] ms/round: draft={:.2} verify={:.2} accept={:.2} rollback={:.2} \
             commit={:.2} rewind={:.2} bonus={:.2}",
            per(ph[0]),
            per(ph[1]),
            per(ph[2]),
            per(ph[3]),
            per(ph[4]),
            per(ph[5]),
            per(ph[6])
        );
    }

    Ok(RawDecodeResult {
        reused_prefix_tokens: reused,
        session_slot_evicted: producer.session_slot_evicted(),
        prompt_tokens: prompt_ids.len(),
        new_tokens: sink.generated,
        prefill_seconds,
        decode_seconds: decode_start.elapsed().as_secs_f64(),
        reason,
        kv_position: position,
        kv_backed_token_ids: sink.history,
        // The speculative loop does not pace and polls no pressure
        // watcher, on either side of the temperature gate (sampled runs
        // reach here since rejection sampling landed). Reporting `Normal`
        // is the absence of a reading, consistent with every other
        // unpolled path.
        peak_memory_pressure: crate::power::MemoryPressure::Normal,
    })
}
