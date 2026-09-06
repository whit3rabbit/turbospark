//! The structured turn stream: one home for turning a raw completion's
//! token events into visible content, reasoning, and parsed tool calls.
//!
//! This wiring used to live in three copies that had already started to
//! drift in wording if not yet in behavior: `crates/cli/src/generate/
//! format.rs::ChannelSplit`, `crates/ffi/src/generate/channel.rs::
//! ChannelSplit` (whose own doc said "Ported from" the CLI one), and the
//! inline `needs_decoder` block in `crates/server/src/handler/exec.rs`
//! (the fullest copy, with a tool allowlist). All three keyed the same
//! decision -- build a [`StructuredAssistantDecoder`] or pass text through
//! -- off the same dialect-and-request conditions, and all three had to
//! respect the same two traps: feed EVERY token to the decoder including
//! the empty deltas (AGENTS.md Gotcha 44), and call `finish` after the run
//! or lose a stop-token-terminated call (Gotcha 49).
//!
//! The loop stays with the caller. [`TurnSplitter`] wraps whichever
//! `run_raw_completion*` variant the consumer picked, the same way the
//! server's `ChatModel::run_completion`, the CLI's speculation/chunked
//! match, and the FFI's session-level match each pick their own.

use std::collections::HashSet;

use tokenizer::{
    ChatDialect, MfTokenizer, ParsedToolCall, ReasoningEffort, StructuredAssistantDecoder,
    StructuredAssistantEvent, ToolCallParserError, NO_SUCH_TOKEN_ID,
};

use crate::raw_completion::RawDecodeProgress;

/// One decoded unit of assistant output, split the way a chat consumer
/// needs it: what to show, what was thinking, and what the model invoked.
///
/// There are no token ids here on purpose. Ids are the RAW stream's
/// information ([`RawDecodeProgress::Token`] carries them); once text has
/// gone through the dialect decoder it belongs to no single token, and no
/// consumer of the structured stream uses one.
#[derive(Debug, Clone, PartialEq)]
pub enum TurnEvent {
    /// Prefill progress showing processed vs total prompt tokens.
    Prefill {
        /// Number of prompt tokens processed so far.
        done: usize,
        /// Total number of prompt tokens to prefill.
        total: usize,
    },
    /// Visible answer text. Never empty: empty deltas are decoder INPUT,
    /// not output (Gotcha 44's rule, enforced on the output side where it
    /// belongs).
    Content(String),
    /// Reasoning the model produced on its way to the answer. Never empty.
    Reasoning(String),
    /// A parsed tool call invocation.
    ToolCall(ParsedToolCall),
}

/// Splits one turn's raw completion events into [`TurnEvent`]s.
///
/// Whether a decoder is built at all is a property of the DIALECT and the
/// REQUEST together, and the three conditions stay independent because the
/// reasons are:
///
/// - Tools need the decoder because the tool-chat generation prompt opens a
///   thought channel and the calls arrive as markup that has to be parsed.
/// - Harmony and Muse Glimmer ALWAYS need one: both templates put the
///   model's scratch work in a reasoning frame before its answer whatever
///   the request asked for, so without decoding the reasoning and the frame
///   markup reach the caller as the reply.
/// - A ChatML or Gemma request that ASKED for reasoning needs one because
///   the thought channel it just enabled would otherwise arrive as the
///   reply: the frame tokens render to the empty string, so the client gets
///   the model's scratch work run together with its answer and no way to
///   tell them apart. Keyed on the REQUEST rather than the dialect, unlike
///   Harmony's: with no level asked for, the template renders a pre-closed
///   `<think></think>` (ChatML) or no thought channel at all (Gemma), so
///   every existing request keeps the pass-through path it always had.
///
/// Skipping a needed decoder is not a cosmetic loss. Measured on the real
/// Gemma 4 install the first time a level was asked for, the reply began
/// with a bare `thought` (the channel LABEL, as prose), then the model's
/// scratch work, then its answer, all as one run of content.
pub struct TurnSplitter<'a> {
    decoder: Option<StructuredAssistantDecoder<'a>>,
    /// Set on the first decoder error, after which every event passes
    /// through raw. `StructuredAssistantDecoder::consume` fails permanently
    /// by itself -- it sets its own `failed` flag and answers `Err` for
    /// every later token -- so this flag only short-circuits the calls; the
    /// observable stream is identical either way.
    degraded: bool,
}

impl<'a> TurnSplitter<'a> {
    /// Creates a splitter over the turn whose generation prompt is
    /// `prompt_ids`.
    ///
    /// **THE PROMPT IS AN ARGUMENT BECAUSE A CHATML PROMPT CAN OPEN THE
    /// THOUGHT FRAME ITSELF** when thinking is on, so the model never emits
    /// the opening `<think>` and a decoder starting in the visible channel
    /// reports the whole scratchpad as the reply
    /// (`StructuredAssistantDecoder::new`). Pass the rendered generation
    /// prompt, exactly what the loop is about to prefill.
    ///
    /// `id_generator` names parsed tool calls; it is only reached when the
    /// allowlist is non-empty, so a caller with no tools can pass anything
    /// (`String::new` is the convention).
    pub fn new(
        tokenizer: &'a MfTokenizer,
        tools: &HashSet<String>,
        effort: ReasoningEffort,
        id_generator: impl FnMut() -> String + 'a,
        prompt_ids: &[i32],
    ) -> Self {
        let wanted = !tools.is_empty()
            || matches!(
                tokenizer.dialect,
                ChatDialect::Harmony | ChatDialect::MuseGlimmer
            )
            || (effort != ReasoningEffort::Off
                && matches!(tokenizer.dialect, ChatDialect::ChatMl | ChatDialect::Gemma));
        Self {
            decoder: wanted.then(|| {
                StructuredAssistantDecoder::new(tokenizer, tools.clone(), id_generator, prompt_ids)
            }),
            degraded: false,
        }
    }

    /// Whether the run degraded to raw pass-through. Diagnostic only; the
    /// stream itself already says so, by carrying markup as content.
    pub fn degraded(&self) -> bool {
        self.degraded
    }

    /// Feeds one raw completion event, forwarding anything it decodes to
    /// `out`.
    ///
    /// NO EARLY RETURN ON EMPTY TEXT, and none here: a special token
    /// decodes to the empty string, so every Harmony frame token arrives as
    /// `(id, "")` and skipping those means the state machine never sees a
    /// single `<|channel|>` -- the whole turn reads as one run of content.
    /// The emptiness check belongs on DECODER OUTPUT, which is why
    /// [`TurnEvent::Content`] and [`TurnEvent::Reasoning`] are never empty
    /// by construction: the decoder emits no empty deltas, and the
    /// pass-through branch filters here.
    pub fn feed(&mut self, event: RawDecodeProgress, out: &mut impl FnMut(TurnEvent)) {
        // The decoder keys on TOKEN IDS, never on text (a special token's
        // delta is the empty string, so markup is unrecognizable by text);
        // the structured events the decoder emits carry no id, which is why
        // `TurnEvent` does not either.
        let (id, text) = match event {
            RawDecodeProgress::Prefill { done, total } => {
                out(TurnEvent::Prefill { done, total });
                return;
            }
            RawDecodeProgress::Token { id, delta, .. } => (id, delta),
            // A withheld tail has no token id behind it, which is what the
            // tokenizer's "no such token" sentinel means.
            RawDecodeProgress::Tail(tail) => (NO_SUCH_TOKEN_ID, tail),
        };
        let Some(decoder) = self.decoder.as_mut().filter(|_| !self.degraded) else {
            if !text.is_empty() {
                out(TurnEvent::Content(text));
            }
            return;
        };
        match decoder.consume(id, &text) {
            Ok(events) => out_structured(events, out),
            // A model writing prose that merely looks like a tool call must
            // not fail the turn. On a parser error the decoder is abandoned
            // and the rest of the run is emitted as raw text; what the
            // decoder had already buffered when it gave up is lost with it.
            Err(_) => {
                self.degraded = true;
                if !text.is_empty() {
                    out(TurnEvent::Content(text));
                }
            }
        }
    }

    /// Flushes whatever the decoder was still holding, after the loop
    /// returned.
    ///
    /// NOT OPTIONAL, AND NOT SYMMETRIC WITH `feed`, for the caller that
    /// calls it: Harmony ends a tool call with `<|call|>`, which is a stop
    /// token, so the loop breaks before the decoder sees the token that
    /// terminates the call it is parsing and `finish` is where that call is
    /// emitted (AGENTS.md Gotcha 49). Its other job is releasing the tail
    /// DeepSeek's arm withholds as a possible tool-marker prefix.
    ///
    /// Nothing is emitted when the run degraded or no decoder was built.
    /// The error is returned rather than swallowed: the run itself
    /// succeeded, and what a decoder abandons is the markup it could not
    /// parse, which is a degraded path rather than a failed turn.
    pub fn finish(&mut self, out: &mut impl FnMut(TurnEvent)) -> Result<(), ToolCallParserError> {
        let Some(decoder) = self.decoder.as_mut().filter(|_| !self.degraded) else {
            return Ok(());
        };
        match decoder.finish() {
            Ok(events) => {
                out_structured(events, out);
                Ok(())
            }
            Err(e) => Err(e),
        }
    }

    /// Test visibility into the build-decoder decision, which is the one
    /// piece of [`TurnSplitter`] no downstream event can observe directly.
    #[cfg(test)]
    fn decodes(&self) -> bool {
        self.decoder.is_some()
    }
}

/// Maps decoder events onto `out`.
fn out_structured(events: Vec<StructuredAssistantEvent>, out: &mut impl FnMut(TurnEvent)) {
    for event in events {
        out(match event {
            StructuredAssistantEvent::Content(c) => TurnEvent::Content(c),
            StructuredAssistantEvent::Reasoning(r) => TurnEvent::Reasoning(r),
            StructuredAssistantEvent::ToolCall(c) => TurnEvent::ToolCall(c),
        });
    }
}

#[cfg(test)]
mod tests {
    //! Driven directly with hand-built [`RawDecodeProgress`] streams against
    //! the bundled dialect fixtures, no model and no producer. The fixture
    //! ids are resolved from the loaded tokenizer, never read off the JSON
    //! (AGENTS.md Gotcha 11).

    use std::collections::HashSet;
    use std::path::PathBuf;

    use super::{TurnEvent, TurnSplitter};
    use crate::raw_completion::RawDecodeProgress;
    use tokenizer::{MfTokenizer, ReasoningEffort};

    fn fixture(name: &str) -> MfTokenizer {
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../tokenizer/tests/fixtures")
            .join(name);
        MfTokenizer::load_from_dir(&dir).expect("fixture tokenizer should load")
    }

    fn splitter<'a>(
        tok: &'a MfTokenizer,
        tools: &[&str],
        effort: ReasoningEffort,
    ) -> TurnSplitter<'a> {
        let tools: HashSet<String> = tools.iter().map(|s| s.to_string()).collect();
        TurnSplitter::new(tok, &tools, effort, String::new, &[])
    }

    fn feed_all(
        splitter: &mut TurnSplitter,
        events: &mut Vec<TurnEvent>,
        inputs: &[RawDecodeProgress],
    ) {
        for input in inputs {
            splitter.feed(input.clone(), &mut |e| events.push(e));
        }
    }

    fn delta(id: i32, text: &str) -> RawDecodeProgress {
        RawDecodeProgress::Token {
            index: 0,
            id,
            delta: text.to_string(),
        }
    }

    #[test]
    fn the_decoder_decision_is_a_table_over_dialect_request_and_tools() {
        let harmony = fixture("HarmonyTokenizer");
        let muse = fixture("MuseGlimmerTokenizer");
        let chatml = fixture("ChatMLTokenizer");
        let gemma = fixture("GemmaTokenizer");
        let deepseek = fixture("DeepseekTokenizer");

        // Harmony and Muse Glimmer reason on every turn, whatever the
        // request asked for.
        assert!(splitter(&harmony, &[], ReasoningEffort::Off).decodes());
        assert!(splitter(&muse, &[], ReasoningEffort::Off).decodes());
        // ChatML and Gemma need one only when a reasoning LEVEL was asked
        // for; their thought channels are unreachable otherwise.
        assert!(!splitter(&chatml, &[], ReasoningEffort::Off).decodes());
        assert!(splitter(&chatml, &[], ReasoningEffort::Low).decodes());
        assert!(!splitter(&gemma, &[], ReasoningEffort::Off).decodes());
        assert!(splitter(&gemma, &[], ReasoningEffort::Low).decodes());
        // No other dialect builds one today, whatever the request carries.
        assert!(!splitter(&deepseek, &[], ReasoningEffort::Low).decodes());
        // Tools change the answer for every dialect, because calls arrive
        // as markup that has to be parsed.
        assert!(splitter(&chatml, &["f"], ReasoningEffort::Off).decodes());
        assert!(splitter(&gemma, &["f"], ReasoningEffort::Off).decodes());
        assert!(splitter(&deepseek, &["f"], ReasoningEffort::Off).decodes());
    }

    #[test]
    fn empty_deltas_reach_the_decoder_and_frame_it() {
        // THE GOTCHA 44 SHAPE: Harmony's frame tokens decode to the empty
        // string. Feeding them is what walks the state machine through the
        // `<|channel|>analysis<|message|>` frame, so the body that follows
        // comes back as Reasoning and not as Content. The label token is
        // part of the frame too: it is accumulated, never emitted.
        let tok = fixture("HarmonyTokenizer");
        let mut s = splitter(&tok, &[], ReasoningEffort::Off);
        let mut events = Vec::new();
        feed_all(
            &mut s,
            &mut events,
            &[
                delta(tok.channel_start_id, ""),
                delta(-6, "analysis"),
                delta(tok.message_start_id, ""),
                delta(-7, "thinking"),
            ],
        );
        assert_eq!(events, vec![TurnEvent::Reasoning("thinking".to_string())]);
    }

    #[test]
    fn without_a_decoder_text_passes_through_and_empty_deltas_stay_dropped() {
        let tok = fixture("ChatMLTokenizer");
        let mut s = splitter(&tok, &[], ReasoningEffort::Off);
        assert!(!s.decodes());
        let mut events = Vec::new();
        feed_all(
            &mut s,
            &mut events,
            &[delta(11, ""), delta(12, "hel"), delta(13, "lo")],
        );
        // A flushed tail has no token id behind it; the structured stream
        // carries no ids, so it is just text.
        s.feed(RawDecodeProgress::Tail(" tail".to_string()), &mut |e| {
            events.push(e)
        });
        assert_eq!(
            events,
            vec![
                TurnEvent::Content("hel".to_string()),
                TurnEvent::Content("lo".to_string()),
                TurnEvent::Content(" tail".to_string()),
            ]
        );
    }

    #[test]
    fn prefill_passes_through_whether_or_not_a_decoder_was_built() {
        let tok = fixture("ChatMLTokenizer");
        for effort in [ReasoningEffort::Off, ReasoningEffort::High] {
            let mut s = splitter(&tok, &[], effort);
            let mut events = Vec::new();
            s.feed(RawDecodeProgress::Prefill { done: 3, total: 9 }, &mut |e| {
                events.push(e)
            });
            assert_eq!(events, vec![TurnEvent::Prefill { done: 3, total: 9 }]);
        }
    }

    #[test]
    fn a_decoder_error_degrades_the_rest_of_the_turn_to_raw_text() {
        // Gemma with a level asked for builds a decoder; a repeated
        // tool-call start token is a malformed span, which is the one
        // failure this fixture can reach without parsing JSON.
        let tok = fixture("GemmaTokenizer");
        let start = tok.tool_call_start_id;
        let mut s = splitter(&tok, &[], ReasoningEffort::Low);
        assert!(s.decodes());
        let mut events = Vec::new();
        feed_all(
            &mut s,
            &mut events,
            &[
                delta(start, ""),
                delta(start, ""),
                delta(21, "raw "),
                delta(22, "text"),
            ],
        );
        // The erroring event's text was empty, so nothing is emitted for
        // it; everything AFTER the error is markup-free pass-through.
        assert_eq!(
            events,
            vec![
                TurnEvent::Content("raw ".to_string()),
                TurnEvent::Content("text".to_string()),
            ]
        );
        assert!(s.degraded());
    }

    #[test]
    fn a_degraded_finish_emits_nothing() {
        let tok = fixture("GemmaTokenizer");
        let start = tok.tool_call_start_id;
        let mut s = splitter(&tok, &[], ReasoningEffort::Low);
        s.feed(delta(start, ""), &mut |_| {});
        s.feed(delta(start, ""), &mut |_| {});
        assert!(s.degraded());
        let mut sink = Vec::new();
        assert!(s.finish(&mut |e| sink.push(e)).is_ok());
        assert!(sink.is_empty());
    }

    #[test]
    fn a_decoder_free_splitter_finish_is_a_no_op() {
        let tok = fixture("ChatMLTokenizer");
        let mut s = splitter(&tok, &[], ReasoningEffort::Off);
        let mut sink = Vec::new();
        assert!(s.finish(&mut |e| sink.push(e)).is_ok());
        assert!(sink.is_empty());
    }
}
