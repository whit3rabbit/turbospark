//! Tokenizer wrapper and chat-dialect resolution. Ported from the loading
//! and special-token-resolution parts of `Tokenization/Tokenizer.swift`.
//!
//! The dialect resolved here decides SPECIAL TOKEN IDS and the stop set. It
//! does NOT decide chat framing: that is a property of the checkpoint, which
//! ships its own Jinja template (loaded below, in either of HF's two
//! conventions) and is preferred by `MfTokenizer::apply_chat_template`. See
//! `chat_template` for why the two came apart.

mod config;
mod minimax;
mod resolve;
mod resolvers;

use std::collections::BTreeSet;
use std::io::Read;
use std::path::Path;

use tokenizers::Tokenizer;

use self::config::{GenerationConfig, TokenizerConfig};
pub(crate) use self::resolve::{
    DEEPSEEK_BOS_MARK, DEEPSEEK_EOS_MARK, GLM_SOP_MARK, HARMONY_END_MARK, HARMONY_MESSAGE_MARK,
    HARMONY_START_MARK, KIMI_IM_ASSISTANT_MARK, KIMI_IM_END_MARK, KIMI_IM_MIDDLE_MARK,
    KIMI_IM_USER_MARK, MUSE_EOT_MARK, MUSE_MESSAGE_MARK, MUSE_START_MARK, SPARK_BOS_MARK,
    SPARK_BOT_MARK, SPARK_EOS_MARK, SPARK_USER_MARK,
};
use crate::error::TokenizerError;

const MAX_GENERATION_CONFIG_BYTES: u64 = 1 << 20;

fn read_generation_config(path: &Path) -> Option<GenerationConfig> {
    let file = std::fs::File::open(path).ok()?;
    if file.metadata().ok()?.len() > MAX_GENERATION_CONFIG_BYTES {
        return None;
    }
    let mut bytes = Vec::new();
    file.take(MAX_GENERATION_CONFIG_BYTES + 1)
        .read_to_end(&mut bytes)
        .ok()?;
    if bytes.len() as u64 > MAX_GENERATION_CONFIG_BYTES {
        return None;
    }
    serde_json::from_slice(&bytes).ok()
}

/// Chat framing dialect, resolved from the loaded tokenizer's special
/// tokens. `Deepseek` is detected by the presence of the `<|User|>` special
/// token (DeepSeek-V4), `ChatMl` by `<|im_end|>` (Qwen-style ChatML);
/// everything else uses the Gemma 4 contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatDialect {
    Gemma,
    ChatMl,
    Deepseek,
    /// Mistral / Mixtral: `[INST] user [/INST] assistant</s>`, framed in
    /// PLAIN TEXT rather than special tokens (ROADMAP Phase M2). The only
    /// special tokens involved are `<s>` and `</s>`.
    Mistral,
    /// `gpt-oss`'s Harmony format (ROADMAP M5):
    /// `<|start|>role<|message|>content<|end|>`, with an explicit
    /// `<|channel|>` for the analysis / final split.
    ///
    /// **THE RENDERER IS THE CHECKPOINT'S OWN 17 KB TEMPLATE, NOT A CASE IN
    /// `chat_template`.** Harmony carries a system preamble, a reasoning
    /// effort knob and a tool namespace written in TypeScript syntax; a
    /// hand-rolled renderer for it would be a large second implementation of
    /// something the checkpoint already ships, and Gotcha 1 makes the
    /// checkpoint's version win anyway. This variant therefore exists for
    /// what a dialect IS evidence for -- the ids and the STOP SET.
    Harmony,
    /// `muse_glimmer`'s frame:
    /// `<|start|>role<|message|>content<|eot|>`.
    ///
    /// **SHARES `<|start|>` AND `<|message|>` WITH HARMONY AND NOTHING
    /// ELSE**, which is why `detect_dialect` tests Harmony first and why
    /// Harmony's probe requires `<|channel|>`. This family has no channels,
    /// no `<|return|>` and no `<|call|>`; it closes a turn with `<|eot|>` and
    /// a handoff with `<|eom|>`, and frames tool calls as
    /// `<atem:function_calls>` PLAIN TEXT rather than as special tokens.
    ///
    /// Like [`ChatDialect::Harmony`] it has NO fallback renderer: the
    /// checkpoint's own template carries an image/video content macro and
    /// the ATEM tool DSL, so this variant exists for what a dialect IS
    /// evidence for -- the ids and the STOP SET.
    MuseGlimmer,
    /// Meta's Llama-3 family (base and Instruct):
    /// `<|start_header_id|>role<|end_header_id|>\n\ncontent<|eot_id|>`,
    /// opened by a literal `<|begin_of_text|>`.
    ///
    /// **SHARES `<|begin_of_text|>` AND `<|end_of_text|>` WITH
    /// [`ChatDialect::MuseGlimmer`] AND NOTHING ELSE**, which is why
    /// `detect_dialect` keys on the header markers instead. No tool-calling
    /// or thinking markup: the base 8B-Instruct table this was built against
    /// carries none, so every such id resolves to [`NO_SUCH_TOKEN_ID`]. A
    /// 3.1-family checkpoint's `<|eom_id|>` / `<|python_tag|>` tool-calling
    /// pair is out of scope here and untested.
    Llama3,
    /// Spark-X2.5 (`XHToken/Spark-X2.5-4B`):
    /// `<｜start▁of▁sentence｜><|User|>...<｜end▁of▁sentence｜>` per turn,
    /// assistant turns opened by `<|Bot|>` with a FORCED-OPEN think block
    /// (the generation prompt ends `<think>` when thinking is on and
    /// `</think>` when it is off), so a turn always begins inside a frame
    /// the template already wrote.
    ///
    /// **THE FULLWIDTH TOKENS ARE DEEPSEEK'S SPELLING, THE HALFWIDTH TURN
    /// MARKERS ARE NOT.** BOS/EOS/pad use the fullwidth-bar DeepSeek forms
    /// (with `start` where DeepSeek writes `begin`), but the turn markers
    /// are plain ASCII `<|User|>` / `<|Bot|>` / `<|Tool|>` / `<|System|>`,
    /// and the fullwidth `<｜User｜>` / `<｜Assistant｜>` the DeepSeek
    /// probe keys on are ABSENT from this table -- which is what makes the
    /// two dialects separable and a separate variant rather than a DeepSeek
    /// reskin (`resolve.rs`'s probe comment).
    ///
    /// The checkpoint ships its own template and it is the renderer (same
    /// doctrine as Harmony and Muse Glimmer); this variant exists for the
    /// ids and the STOP SET. Tool calls are framed `<tool_call>` /
    /// `<arg_key>` / `<arg_value>` and are deliberately UNPARSED for now:
    /// every such id resolves to [`NO_SUCH_TOKEN_ID`] and the markup flows
    /// as ordinary content, where a rescue layer can reach it.
    Spark,
    /// MiniMax-M2 checkpoint framing and EOS; native tool parsing is deferred.
    MiniMax,
    /// GLM-4.7-Flash and siblings (`Glm4MoeLiteForCausalLM`, an MLA MoE that
    /// reports the same `deepseek2` GGUF architecture string as Kimi K2):
    /// `[gMASK]<sop>` opens the prompt, turns frame as the special tokens
    /// `<|system|>` / `<|user|>` / `<|assistant|>` / `<|observation|>`, and a
    /// generation prompt ends `<|assistant|>` plus a FORCED think frame
    /// (`<think>` with thinking on, `</think>` with it off), the Spark/Kimi
    /// arrangement.
    ///
    /// Every id here was read off `zai-org/GLM-4.7-Flash`'s `tokenizer.json`
    /// (2026-09-19), not recalled: `<|endoftext|>` 154820, `<|user|>` 154827,
    /// `<|assistant|>` 154828, `<|observation|>` 154829, the think pair 154841
    /// / 154842. `generation_config.json` declares THREE end-of-sequence ids
    /// -- `<|endoftext|>`, `<|user|>` and `<|observation|>` -- because a GLM
    /// turn ends by handing off to the caller, the observation request, or
    /// the next user turn, and all three must stop generation.
    ///
    /// **THE TOOL MARKUP IS ADDED BUT NOT SPECIAL** (`<tool_call>` 154843,
    /// `<arg_key>` 154847, `<arg_value>` 154849 and friends, `special:
    /// false`), so unlike Gemma's or ChatML's brackets it SURVIVES
    /// detokenization as literal text (the same finding the MiniMax entry
    /// records), and the decoder arm is a TEXT-MARKER arm like DeepSeek's
    /// DSML scan, not an id-bracket arm. Tool calls are
    /// `<tool_call>NAME<arg_key>K</arg_key><arg_value>V</arg_value>...</tool_call>`
    /// where a string argument's value is RAW text and every other value is
    /// JSON-encoded -- exactly what the checkpoint's own template renders and
    /// parses with `tojson`.
    Glm,
    /// Moonshot's Kimi K2 line (`KimiK25ForConditionalGeneration` / the
    /// DeepSeek-V3-line MLA MoE behind it): turns frame as the special tokens
    /// `<|im_user|>` / `<|im_assistant|>` / `<|im_system|>` with a Kimi-owned
    /// `<|im_middle|>` ending each role header, and a generation prompt ends
    /// `<|im_assistant|>assistant<|im_middle|>` plus a FORCED think frame,
    /// the Spark/GLM arrangement.
    ///
    /// Every id here was read off `moonshotai/Kimi-K2.5`'s
    /// `tokenizer_config.json` (2026-09-19), not recalled: `[EOS]` 163585,
    /// `<|im_end|>` 163586, `<|im_assistant|>` 163588, `<|im_middle|>`
    /// 163601, the think pair 163606 / 163607. `<|im_end|>` is SPECIAL and
    /// ends every rendered turn, which is why this variant's probe must be
    /// tested BEFORE the ChatML arm -- a Kimi table carries `<|im_end|>` and
    /// would otherwise be probed as ChatML and fail to load.
    ///
    /// **THE ENTIRE K2 LINE IS TIKTOKEN-ONLY**: every Moonshot K2 checkpoint
    /// ships `tiktoken.model` and no `tokenizer.json` at all, so the table
    /// this dialect resolves from is the added-token list of
    /// `tokenizer_config.json`, and no K2 install is loadable by this engine
    /// today (the fixture that tests it carries the real added tokens and a
    /// disclosed synthetic BPE body). The tool markup
    /// (`<|tool_calls_section_begin|>` 163595 through `<|tool_call_end|>`
    /// 163599) is ADDED BUT NOT SPECIAL like GLM's, so the decoder arm is a
    /// TEXT-MARKER arm: a call is
    /// `<|tool_call_begin|>functions.NAME:IDX<|tool_call_argument_begin|>{json}<|tool_call_end|>`
    /// inside a `<|tool_calls_section_begin|>` / `<|tool_calls_section_end|>`
    /// wrapper, and the NAME lives inside the id string, Moonshot's own
    /// `functions.NAME:IDX` convention.
    Kimi,
    /// DeepSeek V2-line checkpoints whose tables predate the reasoning
    /// channel (`DeepSeek-V2-Lite` / `-Chat` and siblings): the same
    /// fullwidth `<｜User｜>` / `<｜Assistant｜>` marks and BOS/EOS the
    /// [`ChatDialect::Deepseek`] probe keys on, but NO `<think>` / `</think>`
    /// pair and a smaller table. That absence is the probe's split: a table
    /// WITH the think pair resolves Deepseek (V3/V4 semantics), one WITHOUT
    /// resolves this, so neither can pass where its own resolver would fail
    /// (AGENTS.md Gotcha 52).
    ///
    /// No renderer, on the Harmony/Muse/Spark doctrine: the checkpoint
    /// ships its template (`User: ... \n\nAssistant: ...` plain text) and
    /// `apply_chat_template` prefers it. This variant exists for the ids
    /// and the STOP SET (EOS only; no tool or reasoning markup, so every
    /// structured id is [`NO_SUCH_TOKEN_ID`] and the decoder passes plain
    /// content through).
    DeepseekV2,
}

/// Whether a dialect's own markup carries tool calls that this engine PARSES.
///
/// See [`ChatDialect::tool_call_support`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolCallSupport {
    /// [`crate::StructuredAssistantDecoder`] can emit
    /// `StructuredAssistantEvent::ToolCall` on this dialect.
    Native,
    /// It cannot. The model may still be PROMPTED into emitting a call as
    /// ordinary prose, which a caller's own parser or a guardrail rescue can
    /// recover -- but nothing in this engine's framing will hand one over.
    Prompted,
}

/// Sentinel for token roles a dialect frames as plain text rather than a
/// single special token (DeepSeek's tool markers). Never a valid token ID.
pub const NO_SUCH_TOKEN_ID: i32 = -1;

impl ChatDialect {
    /// Does this dialect's own markup carry tool calls this engine parses?
    ///
    /// **THE INVARIANT: this answers `Native` for exactly the dialects whose
    /// arm of [`crate::StructuredAssistantDecoder::consume`] can construct a
    /// `StructuredAssistantEvent::ToolCall`, and `Prompted` for the rest.**
    /// It is a MATCH with no default arm, so a new dialect cannot inherit a
    /// neighbour's answer (root Gotcha 37's shape: a table keyed by X holding
    /// a property of Y stays right for exactly as long as the mapping happens
    /// to be injective).
    ///
    /// The two are separate matches because they cannot be merged -- Muse
    /// Glimmer needs its decoder arm for the reasoning split while answering
    /// `Prompted` here -- so the tie is a `debug_assert!` at every site that
    /// builds a `ToolCall`, driven by the decoder's own test suite. Adding a
    /// dialect to one and not the other panics in debug rather than drifting
    /// quietly.
    ///
    /// **`Prompted` IS NOT "TOOLS DO NOT WORK"**, and a caller that reads it
    /// that way will hide the wrong control. `docs/FORGE_GUARDRAILS.md`'s
    /// whole first failure mode is a small model emitting a call in a syntax
    /// its own template never taught it, which is what a `Prompted` dialect
    /// does BY DEFAULT -- so this is the case a rescue helps most, not the
    /// case to refuse.
    pub fn tool_call_support(self) -> ToolCallSupport {
        match self {
            // `tool_call_start_id` / `tool_call_end_id` bracket a call, parsed
            // by the Gemma arm in `structured_decoder/mod.rs`.
            ChatDialect::Gemma => ToolCallSupport::Native,
            // Qwen's `<tool_call>` pair, parsed by `QwenToolCallParser`.
            ChatDialect::ChatMl => ToolCallSupport::Native,
            // `<|DSML|tool_calls>` in plain text, parsed by
            // `DeepseekToolCallParser`.
            ChatDialect::Deepseek => ToolCallSupport::Native,
            // Harmony's channel recipient plus `<|call|>`, parsed by the
            // harmony arm.
            ChatDialect::Harmony => ToolCallSupport::Native,
            // `[TOOL_CALLS]` plus a JSON-array body that runs to end of turn,
            // parsed by `MistralToolCallParser` out of the decoder's `finish`.
            // Resolved OPTIONALLY: the earliest Mistral tables carry no such
            // token, and a checkpoint without it keeps this dialect's
            // passthrough behavior -- a call it cannot emit is also a call it
            // cannot parse.
            ChatDialect::Mistral => ToolCallSupport::Native,
            // GLM's `<tool_call>` / `<arg_key>` / `<arg_value>` XML, parsed
            // by `GlmToolCallParser` out of the TEXT stream -- the markup is
            // added-but-not-special, so the arm is a text scan and the
            // bracketing ids stay the sentinel (they exist in the table but
            // nothing reads them; carrying them would claim an id-bracket
            // arm this dialect does not have).
            ChatDialect::Glm => ToolCallSupport::Native,
            // Kimi K2's section/call markers around a
            // `functions.NAME:IDX` + JSON body, parsed by
            // `KimiToolCallParser` out of the TEXT stream for the same
            // added-but-not-special reason.
            ChatDialect::Kimi => ToolCallSupport::Native,
            // No tool markup in either table at all: every tool id resolves to
            // `NO_SUCH_TOKEN_ID` and the decoder's arm is a content-only
            // passthrough.
            ChatDialect::Llama3 => ToolCallSupport::Prompted,
            // **NOT AN OVERSIGHT, AND NOT THE SAME AS THE TWO ABOVE.** This dialect DOES frame tool calls -- `<atem:function_calls>` on a
            // `to=<tool>` message -- and this engine has NO PARSER for that
            // block (`structured_decoder/muse.rs`'s `parse_header`). The
            // decoder routes it to the REASONING stream on purpose, since
            // unparseable markup is better hidden there than emitted as the
            // reply. So a caller never sees a `ToolCall` here, and a rescue
            // over the reply text will not see the markup either.
            ChatDialect::MuseGlimmer => ToolCallSupport::Prompted,
            // The markup EXISTS (`<tool_call>` / `<arg_key>` / `<arg_value>`,
            // all special tokens in the table) and this engine has NO PARSER
            // for it yet -- a deliberate descope recorded in DEVIATIONS.md.
            // The arm routes the markup as ordinary CONTENT, which is where
            // a rescue layer can parse it, rather than muse's reasoning
            // stream: unlike muse's header/body DSL, Spark's call syntax is
            // self-contained plain text a caller's parser can read.
            ChatDialect::Spark | ChatDialect::MiniMax => ToolCallSupport::Prompted,
            // No tool markup at all: every call id is NO_SUCH_TOKEN_ID and the
            // decoder passes plain content through, where a rescue layer can
            // reach a call the model attempts in prose.
            ChatDialect::DeepseekV2 => ToolCallSupport::Prompted,
        }
    }

    /// Why [`ChatDialect::tool_call_support`] is not `Native`, or `None` when
    /// it is. One wording, so a GUI and a log cannot describe it differently.
    pub fn tool_call_unsupported_reason(self) -> Option<String> {
        match self.tool_call_support() {
            ToolCallSupport::Native => None,
            ToolCallSupport::Prompted => Some(match self {
                ChatDialect::MuseGlimmer => "this checkpoint frames tool calls as an \
                     <atem:function_calls> block, which this engine has no parser for, so \
                     calls are reported as reasoning rather than handed over. A tool call \
                     has to be prompted for and parsed by the caller."
                    .to_string(),
                ChatDialect::MiniMax => "MiniMax native tool-call parsing is not implemented; markup remains reply text for caller-side recovery.".to_string(),
                ChatDialect::Spark => "this checkpoint frames tool calls as <tool_call> / \
                     <arg_key> / <arg_value> markup, which this engine does not parse yet, \
                     so the markup is reported as ordinary content. A tool call has to be \
                     prompted for and recovered from the reply text."
                    .to_string(),
                _ => format!(
                    "the {self:?} dialect defines no tool-call markup, so nothing in this \
                     checkpoint's framing hands a call over. A tool call has to be prompted \
                     for and recovered from the reply text."
                ),
            }),
        }
    }
}

/// Tokenizer wrapper combining Hugging Face's `tokenizers` backend with
/// chat dialect detection, special token ID mapping, and Jinja chat templates.
pub struct MfTokenizer {
    pub dialect: ChatDialect,
    pub bos_id: i32,
    pub eos_id: i32,
    pub pad_id: i32,
    pub end_of_turn_id: i32,
    pub tool_call_start_id: i32,
    pub tool_call_end_id: i32,
    pub tool_response_id: i32,
    pub tool_response_end_id: i32,
    /// The one member of this dialect's STOP SET that means "the model is
    /// invoking a tool" rather than "the turn is over", or
    /// [`NO_SUCH_TOKEN_ID`] where the dialect has none.
    ///
    /// Read by `run_raw_completion`'s stop ladder, and by nothing else: it
    /// answers a question about how a generation ENDED, which is separate from
    /// the markup ids the structured decoder reads. Two dialects have one and
    /// they are not the same shape of token -- Gemma's is the tool-RESPONSE
    /// marker it hands over with, Harmony's is `<|call|>` -- which is exactly
    /// why the ladder cannot derive it from `tool_response_id` alone.
    pub tool_call_stop_id: i32,
    pub channel_start_id: i32,
    pub channel_end_id: i32,
    /// The token that ends a channel HEADER and opens its body, and the one
    /// that closes that body. Harmony's frame is a header/body pair rather
    /// than the bracketing token pair `channel_start_id`/`channel_end_id`
    /// describes, so those two cannot express it: `<|channel|>` opens a
    /// header and `<|end|>` closes a whole message, with `<|message|>`
    /// between them. Both are [`NO_SUCH_TOKEN_ID`] for every other dialect
    /// here, whose channels either bracket (Gemma) or are plain text
    /// (DeepSeek). Read by [`crate::structured_decoder`]"'"s Harmony arm.
    pub message_start_id: i32,
    pub message_end_id: i32,
    pub think_start_id: Option<i32>,
    pub think_end_id: Option<i32>,
    pub stop_token_ids: BTreeSet<i32>,
    pub vocab_size: usize,
    /// BOS actually prepended by `encode(_, add_bos: true)`; `None` for
    /// dialects that never use a BOS prefix (ChatML).
    bos_prefix_id: Option<i32>,
    tokenizer: Tokenizer,
    /// The checkpoint's own chat-template source, from whichever of HF's
    /// two conventions the directory shipped it in. Rendered by
    /// [`crate::jinja_chat_template`] for BOTH tool chat and plain text
    /// chat; `None` means fall back to the per-dialect renderer.
    pub(crate) chat_template_source: Option<String>,
}

impl MfTokenizer {
    /// Load a `tokenizer.json` (and, if present alongside it,
    /// `tokenizer_config.json` for the Gemma dialect's BOS/EOS token names,
    /// plus `generation_config.json` for the checkpoint's full EOS set)
    /// from a directory.
    pub fn load_from_dir(dir: &Path) -> Result<Self, TokenizerError> {
        let tokenizer_path = dir.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| TokenizerError::InvalidChatTemplate(e.to_string()))?;
        let config_path = dir.join("tokenizer_config.json");
        let config: TokenizerConfig = std::fs::read_to_string(&config_path)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default();
        // Two conventions, and which one a checkpoint uses is a matter of
        // its converter's vintage rather than of its family: HF moved the
        // template out of `tokenizer_config.json` into a standalone
        // `chat_template.jinja` partway through, and llama.cpp's GGUF
        // converter still writes the older embedded key. Reading only the
        // file made every pre-move checkpoint look template-less and fall
        // through to the dialect renderer, which is how TinyLlama-1.1B-Chat
        // (Zephyr framing) came to be fed Mistral's `[INST]`.
        let chat_template_source = std::fs::read_to_string(dir.join("chat_template.jinja"))
            .ok()
            .or_else(|| config.chat_template_source());
        // `generation_config.json` is the authority for the checkpoint's
        // FULL end-of-sequence set: `tokenizer_config.json` only ever
        // carries one `eos_token` string, so a multi-stop checkpoint
        // looks single-stop without this. The real Gemma 4 26B-A4B
        // declares `eos_token_id: [1, 106, 50]` while its
        // tokenizer_config says only `<eos>`.
        let extra_eos = read_generation_config(&dir.join("generation_config.json"))
            .map(|g| g.eos_ids())
            .unwrap_or_default();
        Self::new(tokenizer, &config, chat_template_source, &extra_eos)
    }

    fn new(
        tokenizer: Tokenizer,
        config: &TokenizerConfig,
        chat_template_source: Option<String>,
        extra_eos: &[i32],
    ) -> Result<Self, TokenizerError> {
        let dialect = resolve::detect_dialect(&tokenizer);
        let mut resolved = resolve::resolve_dialect(dialect, &tokenizer, config)?;
        resolved.stop_token_ids.extend(
            extra_eos
                .iter()
                .copied()
                .filter(|&id| id >= 0 && (id as usize) < resolved.vocab_size),
        );
        Ok(Self {
            dialect,
            bos_id: resolved.bos_id,
            bos_prefix_id: resolved.bos_prefix_id,
            eos_id: resolved.eos_id,
            pad_id: resolved.pad_id,
            end_of_turn_id: resolved.end_of_turn_id,
            tool_call_start_id: resolved.tool_call_start_id,
            tool_call_end_id: resolved.tool_call_end_id,
            tool_response_id: resolved.tool_response_id,
            tool_response_end_id: resolved.tool_response_end_id,
            tool_call_stop_id: resolved.tool_call_stop_id,
            channel_start_id: resolved.channel_start_id,
            channel_end_id: resolved.channel_end_id,
            message_start_id: resolved.message_start_id,
            message_end_id: resolved.message_end_id,
            think_start_id: resolved.think_start_id,
            think_end_id: resolved.think_end_id,
            stop_token_ids: resolved.stop_token_ids,
            vocab_size: resolved.vocab_size,
            tokenizer,
            chat_template_source,
        })
    }

    /// Encode UTF-8 text to token IDs. `add_bos = true` prepends the
    /// dialect's BOS token, when it has one.
    pub fn encode(&self, text: &str, add_bos: bool) -> Vec<i32> {
        let ids = self
            .tokenizer
            .encode(text, false)
            .map(|e| e.get_ids().iter().map(|&id| id as i32).collect::<Vec<_>>())
            .unwrap_or_default();
        match (add_bos, self.bos_prefix_id) {
            (true, Some(bos)) => {
                let mut out = Vec::with_capacity(ids.len() + 1);
                out.push(bos);
                out.extend(ids);
                out
            }
            _ => ids,
        }
    }

    /// Decode token IDs to text. `skip_special_tokens` strips BOS/EOS/turn
    /// markers from the output.
    pub fn decode(&self, ids: &[i32], skip_special_tokens: bool) -> String {
        let ids_u32: Vec<u32> = ids.iter().map(|&id| id as u32).collect();
        self.tokenizer
            .decode(&ids_u32, skip_special_tokens)
            .unwrap_or_default()
    }

    /// Look up the token ID for a string token, or `None` if not in vocabulary.
    pub fn token_to_id(&self, token: &str) -> Option<i32> {
        self.tokenizer.token_to_id(token).map(|id| id as i32)
    }

    /// Look up the string token representation for a token ID, or `None` if invalid.
    pub fn id_to_token(&self, id: i32) -> Option<String> {
        self.tokenizer.id_to_token(id as u32)
    }
}
