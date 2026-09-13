use std::path::Path;
use tokenizer::{
    render_generic_chat_template, Message, MfTokenizer, ReasoningEffort, Role, TokenizerError,
};

pub const PAD_TOKEN_ID: i64 = 151643; // <|endoftext|>
pub const MAX_SEQUENCE_LENGTH: usize = 512;

/// Frame a prompt using the Qwen chat template with generation prompt and thinking enabled.
pub fn frame_prompt(prompt: &str, tokenizer: &MfTokenizer) -> Result<String, TokenizerError> {
    let msg = Message::new(Role::User, prompt);
    // ReasoningEffort::Low renders enable_thinking=true without setting unsupported effort keys
    render_generic_chat_template(tokenizer, &[msg], &[], true, ReasoningEffort::Low)
}

/// Tokenize framed prompt, applying truncation and right-padding to max_length (512).
///
/// Returns (token_ids, attention_mask) as int64 vectors matching the frozen PyTorch contract.
pub fn tokenize_prompt(
    framed_prompt: &str,
    tokenizer: &MfTokenizer,
    max_length: usize,
) -> (Vec<i64>, Vec<i64>) {
    let raw_ids = tokenizer.encode(framed_prompt, false);
    let mut token_ids: Vec<i64> = raw_ids.into_iter().map(|id| id as i64).collect();

    // Truncate if over max_length
    if token_ids.len() > max_length {
        token_ids.truncate(max_length);
    }

    let real_len = token_ids.len();
    let mut attention_mask = vec![1i64; real_len];

    // Pad if under max_length
    while token_ids.len() < max_length {
        token_ids.push(PAD_TOKEN_ID);
        attention_mask.push(0i64);
    }

    (token_ids, attention_mask)
}

/// Load tokenizer from a model directory containing tokenizer.json and tokenizer_config.json.
pub fn load_tokenizer(model_tokenizer_dir: &Path) -> Result<MfTokenizer, TokenizerError> {
    MfTokenizer::load_from_dir(model_tokenizer_dir)
}
