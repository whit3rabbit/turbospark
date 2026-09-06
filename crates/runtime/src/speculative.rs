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
use selection::select;
use tokenizer::MfTokenizer;

use crate::config::GenerationConfig;
use crate::error::RuntimeError;
use crate::producer::SpeculativeProducer;
use crate::raw_completion::{
    CancelFlag, RawDecodeProgress, RawDecodeResult, StopReason, TokenSink, NEVER,
};

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
    mut on_progress: impl FnMut(RawDecodeProgress),
) -> Result<RawDecodeResult, RuntimeError> {
    crate::raw_completion::check_admission(prompt_ids, config, max_context)?;

    if block == 0 {
        return Err(RuntimeError::SpeculationUnavailable(
            "block size 0 proposes nothing; call run_raw_completion instead".to_string(),
        ));
    }

    // THE SAMPLING GATE, and it is the honest half of this function.
    //
    // Acceptance below is `target == proposal`, which is exact speculative
    // decoding at temperature 0 and WRONG at any other temperature: a
    // sampled run that only ever accepts the target's argmax is biased
    // toward the mode, so it would silently narrow the distribution the
    // caller asked for. The correct algorithm for the sampled case is
    // rejection sampling with residual correction (Leviathan et al., arXiv
    // 2211.17192; Chen et al., arXiv 2302.01318), which needs the drafter's
    // and the target's full distributions rather than their argmaxes and is
    // NOT implemented here.
    //
    // So the sampled case is refused, not approximated. A quiet fallback to
    // greedy would change what the model writes while reporting success, and
    // a quiet fallback to the sequential loop would report a speculative run
    // that never speculated.
    if !config.shaping.is_deterministic() {
        return Err(RuntimeError::SpeculationUnavailable(format!(
            "acceptance is exact only at temperature 0 (this run has temperature {}); \
             sampled speculation needs rejection sampling with residual correction, \
             which is not implemented",
            config.shaping.temperature()
        )));
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
    // Every prompt token goes through `produce` rather than
    // `produce_prefill`, which costs a full-vocab GEMV per prompt token that
    // the sequential path skips. That is deliberate: the drafter reads the
    // trunk's hidden state for the position just run, and `produce_prefill`
    // is licensed to skip the output head and everything that feeds it. This
    // is also the configuration every number in `docs/MTP.md` was measured
    // in. Cheapening it is a real opportunity and needs the drafter's input
    // to be shown to survive `skip_head`, which nobody has measured.
    //
    // The reused positions are already in the producer's KV -- and, by the
    // drafter guard that allowed the reuse, in the drafter's context KV too
    // -- so both cursors start past them and `history` is seeded with the
    // ids that built them.
    let prefill_start = Instant::now();
    let mut position = reused;
    history.extend_from_slice(&prompt_ids[..reused]);
    for (i, &token) in prompt_ids.iter().enumerate().skip(reused) {
        producer
            .produce(token, position, &mut logits)
            .map_err(RuntimeError::Producer)?;
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
    let mut feed: Vec<TokenId> = Vec::with_capacity(block + 1);
    let reason;

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
        let round_block = block.min(budget_room).min(context_room);
        if round_block == 0 {
            // Nothing left worth proposing; fall back to one plain step.
            producer
                .produce(next, position, &mut logits)
                .map_err(RuntimeError::Producer)?;
            position += 1;
            next = select(
                LogitsView::new(&logits),
                &config.shaping,
                &sink.history,
                sink.generated as u64,
            )?;
            continue;
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
        proposals.clear();
        if producer.drafts_block_passes() {
            producer
                .draft_block(next, base, round_block, &mut proposals)
                .map_err(RuntimeError::Producer)?;
        } else {
            let mut chained = next;
            for d in 0..=round_block {
                producer
                    .draft_step(chained, base - 1 + d, &mut draft_logits)
                    .map_err(RuntimeError::Producer)?;
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

        // -- Verify. One batched pass over the confirmed token plus every
        //    proposal. Row `i` predicts the token after `feed[i]`, so row 0
        //    is checked against `proposals[0]`, and the row after the last
        //    accepted proposal carries the free bonus token.
        feed.clear();
        feed.push(next);
        feed.extend_from_slice(&proposals);
        let point = producer.checkpoint();
        producer
            .verify(&feed, base, &mut batch_logits[..feed.len() * vocab_size])
            .map_err(RuntimeError::Producer)?;

        // -- Accept. Each accepted proposal is committed through the sink
        //    IMMEDIATELY, which is what keeps `select` exact: the next row is
        //    sampled with the history and step counter a sequential decode
        //    would have had at that position, so a repetition penalty or a
        //    step-dependent shaping rule sees the same inputs on both paths.
        let mut accepted = 0usize;
        let mut stopped: Option<StopReason> = None;
        stat_rounds += 1;
        for (i, &proposal) in proposals.iter().enumerate() {
            let row = &batch_logits[i * vocab_size..(i + 1) * vocab_size];
            let target = select(
                LogitsView::new(row),
                &config.shaping,
                &sink.history,
                sink.generated as u64,
            )?;
            if stats {
                stat_offered[i] += 1;
            }
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
        stat_accepted += accepted;
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
        position = sink.history.len();

        // THE RECORD, the half that lets the NEXT turn reuse: the verify fed
        // these rows and the round committed them, so the KV rows are real.
        // The batched half of what `produce` records for the plain-step
        // fallback above. `committed + 1` rows: the confirmed token plus
        // every committed proposal, positions `base .. base + committed +
        // 1` -- the bonus is not fed yet and belongs to the next round.
        producer.record_committed(&feed[..committed + 1], base);

        // The drafter goes to where the accepted prefix ENDED and continues;
        // the target went back to where the block STARTED and replayed. Two
        // targets, which is why one call cannot do both.
        producer
            .rewind_drafter(base + committed)
            .map_err(RuntimeError::Producer)?;

        if let Some(stop) = stopped {
            reason = stop;
            break 'rounds;
        }

        // The bonus: the row after the last accepted proposal. Free, in the
        // sense that the verify pass computed it whether or not the block was
        // accepted, and it is why a round commits `accepted + 1` tokens.
        next = select(
            LogitsView::new(&batch_logits[accepted * vocab_size..(accepted + 1) * vocab_size]),
            &config.shaping,
            &sink.history,
            sink.generated as u64,
        )?;
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
        // The speculative loop does not pace: acceptance is exact only at
        // temperature 0, so a run that reaches here is deterministic and
        // uncapped by construction. Reporting `Normal` is the absence of a
        // reading, consistent with every other unpolled path.
        peak_memory_pressure: crate::power::MemoryPressure::Normal,
    })
}
