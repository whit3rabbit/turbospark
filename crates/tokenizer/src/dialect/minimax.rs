//! MiniMax uses checkpoint framing, with EOS also serving as unused padding.
use super::resolve::{required_id, Resolved};
use super::NO_SUCH_TOKEN_ID;
use crate::error::TokenizerError;
use tokenizers::Tokenizer;

pub(super) fn resolve(tokenizer: &Tokenizer) -> Result<Resolved, TokenizerError> {
    let bos = required_id(tokenizer, "]~!b[")?;
    let eos = required_id(tokenizer, "[e~[")?;
    let think_start = required_id(tokenizer, "<think>")?;
    let think_end = required_id(tokenizer, "</think>")?;
    Ok(Resolved {
        bos_id: bos,
        // The Jinja template writes BOS itself, so encoding must not add one.
        bos_prefix_id: None,
        eos_id: eos,
        pad_id: eos,
        end_of_turn_id: eos,
        tool_call_start_id: NO_SUCH_TOKEN_ID,
        tool_call_end_id: NO_SUCH_TOKEN_ID,
        tool_response_id: NO_SUCH_TOKEN_ID,
        tool_response_end_id: NO_SUCH_TOKEN_ID,
        tool_call_stop_id: NO_SUCH_TOKEN_ID,
        channel_start_id: think_start,
        channel_end_id: think_end,
        message_start_id: NO_SUCH_TOKEN_ID,
        message_end_id: NO_SUCH_TOKEN_ID,
        think_start_id: Some(think_start),
        think_end_id: Some(think_end),
        stop_token_ids: [eos].into_iter().collect(),
        vocab_size: tokenizer.get_vocab_size(true),
    })
}
