//! Chunked prefill for Inkling 2: the prompt through the multi-row verify
//! forward ([`super::verify`]), up to `PANEL_MAX_ROWS` rows per command
//! buffer, instead of one `produce` per token.
//!
//! Correctness rides an existing contract: every multi-row kernel computes
//! exactly the arithmetic of its decode twin, and
//! `tests/real_forward_inkling2_verify.rs` holds verify rows bit-identical to
//! sequential decode. So a chunked prefill leaves the same KV rows, conv
//! tails and last-row logits as the token-at-a-time loop, bit for bit
//! (`tests/real_forward_inkling2_prefill.rs`).
//!
//! Resident residency and native (NVFP4 / MXFP8) experts only, the verify
//! path's own limits; [`RealForwardRunner::inkling2_chunked_refusal`] answers
//! both `supports_chunked_prefill` and this driver's backstop, so they cannot
//! disagree. `TURBOSPARK_INKLING2_CHUNKED_PREFILL=0` restores the per-token
//! loop on an install that supports chunking (the A/B arm).

use foundation::LogitValue;

use crate::real_forward::{RealForwardError, RealForwardRunner};

/// Rows per prefill pass: `TURBOSPARK_INKLING2_PANEL_ROWS` (1 to
/// `PANEL_MAX_ROWS`), default `PANEL_MAX_ROWS`. A throughput choice only:
/// both prefill arms give a result independent of the panel size (the exact
/// arm bit for bit against sequential decode, the matrix-unit arm by row
/// independence).
fn panel_rows() -> usize {
    static ROWS: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *ROWS.get_or_init(|| {
        let max = super::verify::PANEL_MAX_ROWS;
        std::env::var("TURBOSPARK_INKLING2_PANEL_ROWS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .map_or(max, |r| r.clamp(1, max))
    })
}

fn chunked_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("TURBOSPARK_INKLING2_CHUNKED_PREFILL").as_deref() != Ok("0"))
}

/// The open-time GPU touch (`RealForwardRunner::inkling2_prewarm`): on unless
/// `TURBOSPARK_INKLING2_PREWARM=0`.
fn prewarm_enabled() -> bool {
    std::env::var("TURBOSPARK_INKLING2_PREWARM").as_deref() != Ok("0")
}

impl RealForwardRunner {
    /// The first command buffer that references the resident experts (about
    /// 15.7 GB) and the dense weights waits about 1.2 to 2.8 s before its GPU
    /// work starts, while the driver maps them; measured 2026-11-23, a first
    /// decode token at 1.5 to 2.8 s wall for 30 ms of GPU. This submits one
    /// tiny dispatch per buffer at the end of open WITHOUT waiting, so that
    /// mapping overlaps whatever comes next (the rest of the caller's setup,
    /// a user typing the first message). The first real pass queues behind
    /// it on the same queue, so it is never slower than without. Resident
    /// installs only (a streamed open has no expert buffers to map). Writes
    /// one half into a scratch buffer; the model's buffers are only read.
    pub(crate) fn inkling2_prewarm(&mut self) -> Result<(), RealForwardError> {
        if !prewarm_enabled() {
            return Ok(());
        }
        let Some(st) = self.real_inkling2.as_ref() else {
            return Ok(());
        };
        let Some(res) = st.resident.as_ref() else {
            return Ok(());
        };
        foundation::engaged!("inkling2_prewarm");
        let scratch = self.context.new_output_buffer(16);
        let pass = self.context.begin_pass_labeled("inkling2 prewarm");
        let mut touch = |b: &gpu::MetalBuffer| {
            gpu::encode_inkling2_copy_half(&mut self.context, &pass, (b, 0), (&scratch, 0), 1)
                .map_err(RealForwardError::Gpu)
        };
        for b in &res.layers {
            touch(b)?;
        }
        touch(self.weights.buffer())?;
        // Not waited on: the command buffer retains what it references.
        let _ = pass.commit_checked().map_err(RealForwardError::Gpu)?;
        Ok(())
    }

    /// Why this install cannot take the chunked driver, or `None` when it
    /// can. Install-level only.
    pub(crate) fn inkling2_chunked_refusal(&self) -> Option<String> {
        let st = self.real_inkling2.as_ref()?;
        if st.resident.is_none() {
            return Some(
                "Inkling 2 chunked prefill needs resident expert residency; the streamed \
                 path routes on the host and has no multi-row form"
                    .to_string(),
            );
        }
        st.multi_row_refusal()
    }

    pub(crate) fn prefill_chunk_real_inkling2(
        &mut self,
        tokens: &[i32],
        start_position: usize,
        logits: &mut [LogitValue],
    ) -> Result<(), RealForwardError> {
        if tokens.is_empty() {
            return Err(RealForwardError::Unsupported(
                "prefill_chunk called with an empty chunk".to_string(),
            ));
        }
        if let Some(why) = self.inkling2_chunked_refusal() {
            return Err(RealForwardError::Unsupported(why));
        }
        if !chunked_enabled() {
            return self.prefill_chunk_inkling2_sequential(tokens, start_position, logits);
        }
        foundation::engaged!("inkling2_chunked_prefill");
        let rows = panel_rows();
        let mut offset = 0usize;
        while offset < tokens.len() {
            let take = (tokens.len() - offset).min(rows);
            let last = offset + take == tokens.len();
            // Only the final row's logits are returned, so only it pays for
            // the 201k-row head; a non-final panel runs no head at all.
            let (out, head): (&mut [LogitValue], _) = if last {
                (&mut *logits, super::verify::HeadRows::Last)
            } else {
                (&mut [], super::verify::HeadRows::None)
            };
            self.rows_real_inkling2(
                &tokens[offset..offset + take],
                start_position + offset,
                out,
                head,
            )?;
            offset += take;
        }
        // A prefill is not a verify: nothing may roll it back.
        if let Some(st) = self.real_inkling2.as_mut() {
            st.verify = None;
        }
        Ok(())
    }

    /// One `produce` per token, the head only on the last.
    fn prefill_chunk_inkling2_sequential(
        &mut self,
        tokens: &[i32],
        start_position: usize,
        logits: &mut [LogitValue],
    ) -> Result<(), RealForwardError> {
        let last = tokens.len() - 1;
        for (i, &token) in tokens.iter().enumerate() {
            let position = start_position + i;
            if i == last {
                self.produce_real_inkling2(token, position, logits)?;
            } else {
                self.skip_head = true;
                let result = self.produce_real_inkling2(token, position, logits);
                self.skip_head = false;
                result?;
            }
        }
        Ok(())
    }
}
