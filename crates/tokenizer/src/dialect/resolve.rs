//! Special token detection and per-dialect token ID resolution.

use std::collections::BTreeSet;
use tokenizers::Tokenizer;

use super::config::TokenizerConfig;
use super::ChatDialect;
use crate::error::TokenizerError;

pub(crate) const DEEPSEEK_USER_MARK: &str = "<\u{FF5C}User\u{FF5C}>";
pub(crate) const DEEPSEEK_ASSISTANT_MARK: &str = "<\u{FF5C}Assistant\u{FF5C}>";
pub(crate) const DEEPSEEK_BOS_MARK: &str = "<\u{FF5C}begin\u{2581}of\u{2581}sentence\u{FF5C}>";
pub(crate) const DEEPSEEK_EOS_MARK: &str = "<\u{FF5C}end\u{2581}of\u{2581}sentence\u{FF5C}>";
pub(crate) const IM_END_MARK: &str = "<|im_end|>";
/// Gemma's end-of-turn marker, used as the POSITIVE test for that dialect
/// now that it is no longer the fallback for everything unrecognized.
pub(crate) const GEMMA_TURN_MARK: &str = "<turn|>";
/// Mistral / Mixtral frame turns as PLAIN TEXT (`[INST] ... [/INST]`), so
/// there is no instruction marker in the special-token table to key on and
/// the only reliable witness is the sentence pair.
pub(crate) const MISTRAL_BOS_MARK: &str = "<s>";
pub(crate) const MISTRAL_EOS_MARK: &str = "</s>";
/// The one marker Mistral's template teaches for tool calls. OPTIONAL in the
/// resolver: the first Mistral tables (Mixtral 8x7B-Instruct v0.1) carry only
/// `<unk>` / `<s>` / `</s>`, and a required id would refuse every early
/// checkpoint that cannot emit the marker at all. Read off the real
/// `mistral7b-dense.gturbo` install's table, where it is id 5.
pub(crate) const MISTRAL_TOOL_CALLS_MARK: &str = "[TOOL_CALLS]";
pub(crate) const IM_START_MARK: &str = "<|im_start|>";
/// Harmony's turn frame (ROADMAP M5). `<|start|>` is the witness because it
/// opens every turn and appears in no other family's table; `<|return|>` is
/// the end-of-turn marker and `<|call|>` the tool-call one, and BOTH end a
/// generation. All three names were read off `openai/gpt-oss-20b`'s
/// `tokenizer_config.json` rather than recalled -- their ids there are
/// 200006 / 200002 / 200012, and they are looked up BY NAME here for the
/// reason crate Gotcha 2 gives.
pub(crate) const HARMONY_START_MARK: &str = "<|start|>";
pub(crate) const HARMONY_MESSAGE_MARK: &str = "<|message|>";
pub(crate) const HARMONY_END_MARK: &str = "<|end|>";
pub(crate) const HARMONY_RETURN_MARK: &str = "<|return|>";
pub(crate) const HARMONY_CALL_MARK: &str = "<|call|>";
pub(crate) const HARMONY_CHANNEL_MARK: &str = "<|channel|>";
pub(crate) const HARMONY_BOS_MARK: &str = "<|startoftext|>";
pub(crate) const HARMONY_PAD_MARK: &str = "<|endoftext|>";
/// `muse_glimmer`'s turn frame. It SHARES `<|start|>` and `<|message|>` with
/// Harmony and shares nothing else, which is why `detect_dialect` orders the
/// two and why Harmony's probe grew a third marker. Read off
/// `mlx-community/Muse-Glimmer-30B-4bit`'s `tokenizer.json` rather than
/// recalled -- ids 200000 / 200008 / 200018 / 200022 / 200023 there, looked
/// up BY NAME here for the reason crate Gotcha 3 gives.
pub(crate) const MUSE_BOS_MARK: &str = "<|begin_of_text|>";
pub(crate) const MUSE_EOS_MARK: &str = "<|end_of_text|>";
pub(crate) const MUSE_EOT_MARK: &str = "<|eot|>";
pub(crate) const MUSE_EOM_MARK: &str = "<|eom|>";
pub(crate) const MUSE_PAD_MARK: &str = "<|finetune_right_pad|>";
pub(crate) const MUSE_START_MARK: &str = "<|start|>";
pub(crate) const MUSE_MESSAGE_MARK: &str = "<|message|>";
/// Meta's Llama-3 family (base and Instruct). SHARES `<|begin_of_text|>` and
/// `<|end_of_text|>` with `muse_glimmer` and nothing else -- crate Gotcha 6's
/// lesson arriving on a third pair. Neither is the witness here:
/// `<|start_header_id|>` and `<|eot_id|>` are, because this family's frame is
/// `<|start_header_id|>role<|end_header_id|>\n\ncontent<|eot_id|>` and neither
/// marker collides with Muse Glimmer's `<|start|>` / `<|message|>` / `<|eot|>`
/// (no `_header_id` / `_id` suffix on any of those three). Read off
/// `meta-llama/Meta-Llama-3-8B-Instruct`'s `tokenizer_config.json` rather than
/// recalled -- ids 128000 / 128001 / 128006 / 128007 / 128009 there, looked up
/// BY NAME here for the reason crate Gotcha 2 gives.
pub(crate) const LLAMA3_BOS_MARK: &str = "<|begin_of_text|>";
pub(crate) const LLAMA3_EOS_MARK: &str = "<|end_of_text|>";
pub(crate) const LLAMA3_EOT_MARK: &str = "<|eot_id|>";
pub(crate) const LLAMA3_START_HEADER_MARK: &str = "<|start_header_id|>";
pub(crate) const LLAMA3_END_HEADER_MARK: &str = "<|end_header_id|>";
/// Spark-X2.5. BOS/EOS use the DEEPSEEK fullwidth-bar spelling but with
/// `start` where DeepSeek writes `begin`, so the DeepSeek constants above do
/// not match and must not be reused. The TURN markers are plain ASCII and
/// `<|Bot|>` is the witness: no other table this port resolves carries it,
/// and `resolve_spark` requires it, so probe and resolver agree (Gotcha
/// 52's rule). Ids on `XHToken/Spark-X2.5-4B`: bos 0, eos 1, `<think>` 3,
/// `</think>` 4, `<|System|>` 130972, `<|User|>` 130973, `<|Tool|>` 130974,
/// `<|Bot|>` 130976 -- read off the tokenizer, never hardcoded.
pub(crate) const SPARK_BOS_MARK: &str = "<\u{FF5C}start\u{2581}of\u{2581}sentence\u{FF5C}>";
pub(crate) const SPARK_EOS_MARK: &str = "<\u{FF5C}end\u{2581}of\u{2581}sentence\u{FF5C}>";
pub(crate) const SPARK_USER_MARK: &str = "<|User|>";
pub(crate) const SPARK_BOT_MARK: &str = "<|Bot|>";

pub(crate) struct Resolved {
    pub(crate) bos_id: i32,
    pub(crate) bos_prefix_id: Option<i32>,
    pub(crate) eos_id: i32,
    pub(crate) pad_id: i32,
    pub(crate) end_of_turn_id: i32,
    pub(crate) tool_call_start_id: i32,
    pub(crate) tool_call_end_id: i32,
    pub(crate) tool_response_id: i32,
    pub(crate) tool_response_end_id: i32,
    pub(crate) tool_call_stop_id: i32,
    pub(crate) channel_start_id: i32,
    pub(crate) channel_end_id: i32,
    pub(crate) message_start_id: i32,
    pub(crate) message_end_id: i32,
    pub(crate) think_start_id: Option<i32>,
    pub(crate) think_end_id: Option<i32>,
    pub(crate) stop_token_ids: BTreeSet<i32>,
    pub(crate) vocab_size: usize,
}

/// Resolves a token string to its ID, rejecting the unk-token fallback some
/// tokenizers substitute for out-of-vocabulary strings.
pub(crate) fn special_token_id(tokenizer: &Tokenizer, token: &str) -> Option<i32> {
    let id = tokenizer.token_to_id(token)?;
    if tokenizer.id_to_token(id).as_deref() == Some(token) {
        Some(id as i32)
    } else {
        None
    }
}

pub(crate) fn required_id(tokenizer: &Tokenizer, token: &str) -> Result<i32, TokenizerError> {
    special_token_id(tokenizer, token)
        .ok_or_else(|| TokenizerError::MissingSpecialToken(token.to_string()))
}

pub(crate) fn detect_dialect(tokenizer: &Tokenizer) -> ChatDialect {
    // Gemma is still the FALLBACK, deliberately: every fixture and
    // install predating ROADMAP Phase M2 lands there, and moving the
    // default would change what an unrecognized tokenizer does. Mistral
    // is therefore tested for POSITIVELY, and only after Gemma's own
    // marker has been ruled out -- `</s>` is far too common a token to
    // decide a dialect on its own.
    if special_token_id(tokenizer, "]~!b[").is_some()
        && special_token_id(tokenizer, "]~b]").is_some()
        && special_token_id(tokenizer, "[e~[").is_some()
    {
        ChatDialect::MiniMax
    } else if special_token_id(tokenizer, DEEPSEEK_USER_MARK).is_some() {
        ChatDialect::Deepseek
    } else if special_token_id(tokenizer, SPARK_BOT_MARK).is_some() {
        // Spark-X2.5, probed BEFORE ChatML on a marker no other table
        // carries. Order relative to the DeepSeek arm is not load-bearing --
        // this table has no fullwidth `<｜User｜>`, so the arm above cannot
        // fire here -- but it IS load-bearing relative to nothing at all:
        // `<|Bot|>` in a Gemma-fallback vocab would silently misframe, so it
        // is tested positively rather than left to the fallback.
        ChatDialect::Spark
    } else if special_token_id(tokenizer, HARMONY_START_MARK).is_some()
        && special_token_id(tokenizer, HARMONY_MESSAGE_MARK).is_some()
        && special_token_id(tokenizer, HARMONY_CHANNEL_MARK).is_some()
    {
        // TESTED BEFORE ChatML, and the order is load-bearing rather than
        // arbitrary: Harmony's table carries neither `<|im_end|>` nor
        // `<|im_start|>` today, so the two probes are disjoint on the real
        // checkpoints.
        //
        // **THE THIRD MARKER WAS ADDED AFTER THIS PROBE MIS-FIRED ON A REAL
        // CHECKPOINT, which is Gotcha 41's lesson arriving exactly as this
        // comment predicted it would.** `<|start|>` and `<|message|>` looked
        // like a specific pair and are not: `mlx-community/Muse-Glimmer-30B-4bit`
        // carries BOTH and is not Harmony -- it has no `<|channel|>`, no
        // `<|return|>`, no `<|call|>`, no `<|end|>` and no `<|startoftext|>`,
        // and its tool DSL is `<atem:function_calls>` rather than a channel
        // recipient. It resolved here and then failed to LOAD at all, on
        // `<|startoftext|>` missing, which is the good failure mode and is
        // still the wrong answer.
        // `<|channel|>` is the witness because channels ARE the format's
        // defining feature and because `resolve_harmony` already requires it
        // -- a probe that can pass where the resolver will fail is the actual
        // bug, and matching them is what fixes it rather than a third
        // arbitrary token.
        ChatDialect::Harmony
    } else if special_token_id(tokenizer, MUSE_START_MARK).is_some()
        && special_token_id(tokenizer, MUSE_EOT_MARK).is_some()
    {
        // `muse_glimmer`. Probed AFTER Harmony, and on the two tokens
        // Harmony does NOT have: this family frames turns as
        // `<|start|>role<|message|>content<|eot|>` where Harmony closes with
        // `<|end|>`, so `<|eot|>` beside `<|begin_of_text|>` is what tells
        // the two apart. Both are checked because `<|eot|>` alone is a
        // Llama-3-family spelling that says nothing about the frame.
        ChatDialect::MuseGlimmer
    } else if special_token_id(tokenizer, LLAMA3_START_HEADER_MARK).is_some()
        && special_token_id(tokenizer, LLAMA3_EOT_MARK).is_some()
    {
        // Meta's Llama-3 family. Probed on its OWN frame markers rather than
        // on `<|begin_of_text|>` / `<|end_of_text|>`, which it shares with
        // `muse_glimmer` (crate Gotcha 6's lesson on a third pair): neither
        // string collides with that family's `<|start|>` / `<|eot|>`, so the
        // order relative to the MuseGlimmer arm above does not matter, but it
        // still has to come before Gemma's fallback and before Mistral, whose
        // `<s>`/`</s>` probe this table does not carry at all.
        ChatDialect::Llama3
    } else if special_token_id(tokenizer, IM_END_MARK).is_some() {
        ChatDialect::ChatMl
    } else if special_token_id(tokenizer, GEMMA_TURN_MARK).is_none()
        && special_token_id(tokenizer, MISTRAL_BOS_MARK).is_some()
        && special_token_id(tokenizer, MISTRAL_EOS_MARK).is_some()
    {
        ChatDialect::Mistral
    } else {
        ChatDialect::Gemma
    }
}

use super::resolvers::{
    resolve_chatml, resolve_deepseek, resolve_gemma, resolve_harmony, resolve_llama3,
    resolve_mistral, resolve_muse_glimmer, resolve_spark,
};

pub(crate) fn resolve_dialect(
    dialect: ChatDialect,
    tokenizer: &Tokenizer,
    config: &TokenizerConfig,
) -> Result<Resolved, TokenizerError> {
    match dialect {
        ChatDialect::Gemma => resolve_gemma(tokenizer, config),
        ChatDialect::ChatMl => resolve_chatml(tokenizer),
        ChatDialect::Deepseek => resolve_deepseek(tokenizer),
        ChatDialect::Mistral => resolve_mistral(tokenizer),
        ChatDialect::Harmony => resolve_harmony(tokenizer),
        ChatDialect::MuseGlimmer => resolve_muse_glimmer(tokenizer),
        ChatDialect::Llama3 => resolve_llama3(tokenizer),
        ChatDialect::Spark => resolve_spark(tokenizer),
        ChatDialect::MiniMax => super::minimax::resolve(tokenizer),
    }
}
