//! Chunked prefill for Inkling 2: the prompt through the multi-row verify
//! forward ([`super::verify`]), up to `VERIFY_MAX_ROWS` rows per command
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

fn chunked_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("TURBOSPARK_INKLING2_CHUNKED_PREFILL").as_deref() != Ok("0"))
}

impl RealForwardRunner {
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
        st.native_expert_types().err()
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
        let vocab = self.arch.vocab_size as usize;
        let rows = super::verify::VERIFY_MAX_ROWS;
        let mut scratch = vec![LogitValue::ZERO; rows * vocab];
        let mut offset = 0usize;
        while offset < tokens.len() {
            let take = (tokens.len() - offset).min(rows);
            let out = &mut scratch[..take * vocab];
            self.verify_real_inkling2(
                &tokens[offset..offset + take],
                start_position + offset,
                out,
            )?;
            offset += take;
            if offset == tokens.len() {
                logits.copy_from_slice(&out[(take - 1) * vocab..take * vocab]);
            }
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
