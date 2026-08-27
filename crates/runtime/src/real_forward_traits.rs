use foundation::LogitValue;

use super::real_forward::{RealForwardRunner, RollbackPoint};
use crate::producer::{ChunkedPrefillRunner, LogitProducer, SpeculativeProducer};

impl LogitProducer for RealForwardRunner {
    fn reset(&mut self) {
        self.kv.reset();
        if let Some(qwen) = self.real_qwen.as_mut() {
            qwen.reset();
        }
        // The head keeps its OWN KV, so the trunk's reset leaves it holding
        // the previous generation's context -- the same shape of leak
        // Gotcha 4 records for the GDN recurrent state, one cache over.
        if let Some(mtp) = self.real_mtp.as_mut() {
            mtp.reset();
        }
        // And the drafter its own, one cache over again.
        if let Some(dflash) = self.real_dflash.as_mut() {
            dflash.reset();
        }
        // Re-arm the residual capture for the next generation, so a caller
        // that opens once and walks a corpus gets one snapshot per prompt
        // rather than one per process.
        if let Some(capture) = self.resid_capture.as_mut() {
            capture.note_generation_start();
        }
    }

    fn produce(
        &mut self,
        token: i32,
        position: usize,
        logits: &mut [LogitValue],
    ) -> Result<(), String> {
        gpu::autorelease_pool(|| self.produce_inner(token, position, logits))
            .map_err(|e| e.to_string())
    }

    fn produce_prefill(
        &mut self,
        token: i32,
        position: usize,
        scratch: &mut [LogitValue],
    ) -> Result<(), String> {
        self.skip_head = true;
        let result = self.produce(token, position, scratch);
        self.skip_head = false;
        result
    }
}

/// The MTP head as a drafter and the M-row pass as the verify
/// (`docs/MTP.md`). Every method here is a thin forward to an inherent one
/// that already existed and was reachable only from
/// `crates/bench/tests/mtp_accept_length_probe.rs`; the trait is what lets
/// `run_raw_completion_speculative` drive them.
///
/// The three refusals the pieces carry are NOT re-stated here, deliberately.
/// `mtp_draft_step` errors on an install with no head or a step off the
/// drafter's cursor, and `produce_batched` refuses a non-dense, non-INT4 or
/// KV-wrapping block by name. Restating them would be a second copy of a
/// condition that has to agree with the first, and the failure mode of
/// disagreeing is a refusal message that names the wrong cause.
impl SpeculativeProducer for RealForwardRunner {
    type Checkpoint = RollbackPoint;

    // WHICH DRAFTER answers is a property of the open state, and at most
    // one is ever open in practice: the CLI refuses a request that names
    // both. If both somehow are, the BLOCK drafter wins here, which keeps
    // `drafts_block_passes` and the priming/rewind pair in one branch.
    fn prime_drafter(&mut self, next: i32, position: usize) -> Result<(), String> {
        if self.real_dflash.is_some() {
            // The DFlash2 context write needs only the captured states for
            // `position`; `next` is an MTP-head input it has no use for.
            self.dflash_prime_from_capture(position)
                .map_err(|e| e.to_string())
        } else {
            self.mtp_prime_step(next, position)
                .map_err(|e| e.to_string())
        }
    }

    fn draft_step(
        &mut self,
        token: i32,
        position: usize,
        logits: &mut [LogitValue],
    ) -> Result<(), String> {
        self.mtp_draft_step(token, position, logits)
            .map_err(|e| e.to_string())
    }

    fn drafts_block_passes(&self) -> bool {
        self.real_dflash.is_some()
    }

    fn draft_block(
        &mut self,
        anchor: i32,
        base: usize,
        block: usize,
        proposals: &mut Vec<i32>,
    ) -> Result<(), String> {
        let want = self.real_dflash.as_ref().map_or(0, |d| d.block);
        // A SHORTER round is the normal end of a run, not a caller error:
        // `run_raw_completion_speculative` shrinks `round_block` against
        // the generation budget and the context window, so every capped
        // run passes through 1..want-1 on its way to 0. The drafter always
        // runs its own `want + 1` rows; the loop reads the first `block`
        // proposals and the rows above them are rewritten before anything
        // reads them again (`families/qwen/dflash.rs`'s cursor rules). A
        // LONGER one really is a caller error -- there is no forward to
        // take those proposals from.
        if block == 0 || block > want {
            return Err(format!(
                "the open DFlash2 drafter proposes at most {want} tokens, and the loop asked for \
                 {block}; open it at or above the block the loop is running"
            ));
        }
        self.dflash_draft_block(anchor, base, proposals)
            .map_err(|e| e.to_string())?;
        proposals.truncate(block);
        Ok(())
    }

    fn rewind_drafter(&mut self, position: usize) -> Result<(), String> {
        if self.real_dflash.is_some() {
            self.dflash_rewind_to(position).map_err(|e| e.to_string())
        } else {
            self.mtp_rewind_to(position).map_err(|e| e.to_string())
        }
    }

    fn checkpoint(&mut self) -> RollbackPoint {
        RealForwardRunner::checkpoint(self)
    }

    fn rollback(&mut self, point: &RollbackPoint) {
        RealForwardRunner::rollback(self, point)
    }

    fn verify(
        &mut self,
        feed: &[i32],
        base: usize,
        logits: &mut [LogitValue],
    ) -> Result<(), String> {
        self.produce_batched(feed, base, logits)
            .map_err(|e| e.to_string())
    }
}

impl ChunkedPrefillRunner for RealForwardRunner {
    /// Gemma 4, the DENSE half of `llama` (Mistral, Llama 2/3.x) and
    /// `muse_glimmer` today, and the refusal is BY NAME rather than a
    /// silent fallback to the sequential path. A caller that asked for
    /// chunked prefill and quietly got the token-at-a-time loop would
    /// measure the old engine and report it as the new one, which is the
    /// failure mode this whole phase exists to avoid.
    /// [`Self::supports_chunked_prefill`] is the SAME predicate this
    /// refusal uses, so a caller deciding whether to route here at all and
    /// this method's own hard refusal can never disagree.
    fn prefill_chunk(
        &mut self,
        tokens: &[i32],
        start_position: usize,
        logits: &mut [LogitValue],
    ) -> Result<(), String> {
        if self.real.is_some() {
            return gpu::autorelease_pool(|| {
                self.prefill_chunk_real_gemma4(tokens, start_position, logits)
            })
            .map_err(|e| e.to_string());
        }
        if self.real_llama.as_ref().is_some_and(|s| s.dense) {
            return gpu::autorelease_pool(|| {
                self.prefill_chunk_real_llama_dense(tokens, start_position, logits)
            })
            .map_err(|e| e.to_string());
        }
        if self.real_muse.is_some() {
            return gpu::autorelease_pool(|| {
                self.prefill_chunk_real_muse(tokens, start_position, logits)
            })
            .map_err(|e| e.to_string());
        }
        Err(format!(
            "chunked prefill is wired for the real Gemma 4 flow, the dense llama flow \
             (Mistral, Llama 2/3.x) and muse_glimmer only; this install is {:?}",
            self.arch.family
        ))
    }
}
