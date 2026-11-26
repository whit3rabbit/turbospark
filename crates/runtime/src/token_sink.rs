use foundation::TokenId;
use tokenizer::{MfDetokenizer, MfTokenizer, StreamingStopMatcher};

use crate::config::GenerationConfig;
use crate::raw_completion::{CancelFlag, RawDecodeProgress, StopReason};

/// Everything a decoded token has to pass through between the sampler and
/// the caller: the stop-token ladder, the detokenizer, the stop-string
/// matcher, the progress callback, and the budget and cancellation checks.
///
/// Extracted so a SPECULATIVE round, which commits several tokens at once,
/// runs each of them through the same statement sequence a sequential decode
/// does rather than reimplementing it. Reimplementing it is how a
/// speculative path acquires a subtly different stop rule -- one that emits a
/// stop token as text, or drops the stop matcher's withheld tail, or reports
/// `MaxTokens` a token late -- none of which the losslessness gate can see,
/// because that gate compares the tokens the two paths COMMIT and every one
/// of these is downstream of the commit.
pub(crate) struct TokenSink<'a> {
    tokenizer: &'a MfTokenizer,
    config: &'a GenerationConfig,
    detok: MfDetokenizer<'a>,
    stop_matcher: StreamingStopMatcher,
    /// True while the stream is inside a think frame (`<think>` .. `</think>`
    /// style, the dialects that resolve `think_start_id`/`think_end_id`).
    /// User stop strings are matched on the ANSWER only: a stop the model
    /// merely writes while reasoning (a counted "10", an agent's closing
    /// tag named in its plan) would otherwise end the request with empty
    /// visible content. Always false for a dialect without those markers,
    /// which keeps raw matching exactly as before.
    in_thinking: bool,
    /// Tokens handed to the caller, including the one that stops the run.
    pub(crate) generated: usize,
    /// Tokens handed to the PRODUCER. Excludes the stopping token, which is
    /// never fed, so this and `position` describe the same cache.
    pub(crate) history: Vec<TokenId>,
}

impl<'a> TokenSink<'a> {
    pub(crate) fn new(
        tokenizer: &'a MfTokenizer,
        config: &'a GenerationConfig,
        history: Vec<TokenId>,
    ) -> Self {
        // Seeded from the rendered prompt (`history` is the prompt here), the
        // same backward scan the structured decoder uses: a thinking-on
        // template opens the frame itself so the model never emits the
        // opening marker, and the last marker wins over balanced pairs a
        // tool preamble quotes in its instructions.
        let in_thinking = history
            .iter()
            .rev()
            .find_map(|&id| {
                if tokenizer.think_start_id == Some(id) {
                    Some(true)
                } else if tokenizer.think_end_id == Some(id) {
                    Some(false)
                } else {
                    None
                }
            })
            .unwrap_or(false);
        Self {
            tokenizer,
            config,
            detok: MfDetokenizer::new(tokenizer),
            stop_matcher: StreamingStopMatcher::new(config.stop_strings.clone()),
            in_thinking,
            generated: 0,
            history,
        }
    }

    /// Flushes whatever the stop matcher was withholding. Every exit from a
    /// generation goes through this; skipping it truncates the reply.
    pub(crate) fn flush_tail(&mut self, on_progress: &mut dyn FnMut(RawDecodeProgress)) {
        let residue = self.detok.flush();
        let mut tail = if self.in_thinking {
            String::new()
        } else {
            self.stop_matcher.push(&residue)
        };
        tail += &self.stop_matcher.finish();
        if self.in_thinking {
            tail += &residue;
        }
        if !tail.is_empty() {
            on_progress(RawDecodeProgress::Tail(tail));
        }
    }

    /// Runs `delta` through the stop matcher unless it is reasoning text.
    ///
    /// The marker tokens flip the scope. Text the matcher was withholding as
    /// a possible stop prefix is released ahead of a reasoning span, so the
    /// output order is preserved and a stop prefix cannot be completed by
    /// reasoning text that follows it.
    fn match_stops(&mut self, token_id: TokenId, delta: String) -> String {
        if self.tokenizer.think_start_id == Some(token_id) {
            self.in_thinking = true;
            let mut out = self.stop_matcher.finish();
            out += &delta;
            return out;
        }
        if self.tokenizer.think_end_id == Some(token_id) {
            self.in_thinking = false;
            return delta;
        }
        if self.in_thinking {
            delta
        } else {
            self.stop_matcher.push(&delta)
        }
    }

    /// Commits one sampled token. `Some(reason)` means the run stops and the
    /// tail has already been flushed; `None` means the token was appended to
    /// `history` and decoding continues.
    ///
    /// A stopping token is deliberately NOT pushed to `history`: it was never
    /// fed to the producer, so pushing it would leave `kv_backed_token_ids`
    /// describing a cache row that does not exist.
    pub(crate) fn commit(
        &mut self,
        token_id: TokenId,
        cancel: CancelFlag<'_>,
        on_progress: &mut dyn FnMut(RawDecodeProgress),
    ) -> Option<StopReason> {
        self.generated += 1;

        let is_stop_token = self.tokenizer.stop_token_ids.contains(&token_id)
            || self.config.extra_stop_tokens.contains(&token_id);
        if is_stop_token {
            // `tool_call_stop_id` rather than `tool_response_id`: the two are
            // the same token on Gemma and are NOT on Harmony, whose
            // `tool_response_id` is `NO_SUCH_TOKEN_ID` and whose tool stop is
            // `<|call|>`. Reading the response marker here sent every gpt-oss
            // tool call to a client as `finish_reason: "stop"`.
            let reason = if token_id == self.tokenizer.end_of_turn_id {
                StopReason::EndOfTurn
            } else if token_id == self.tokenizer.tool_call_stop_id {
                StopReason::ToolCalls
            } else {
                StopReason::Eos
            };
            self.flush_tail(on_progress);
            return Some(reason);
        }

        let delta = self.detok.push(token_id);
        let visible = self.match_stops(token_id, delta);
        on_progress(RawDecodeProgress::Token {
            index: self.generated - 1,
            id: token_id,
            delta: visible,
        });

        let hit_stop_string = self.stop_matcher.is_stopped();
        let hit_max = self.generated as u32 >= self.config.max_new_tokens;
        // Polled here rather than at the top of the loop so a cancelled run
        // takes the SAME exit path the other two do, flushing the stop
        // matcher's withheld tail. Breaking early instead would silently drop
        // whatever the matcher was holding back, which is a truncated reply
        // rather than a cancelled one.
        let hit_cancel = cancel();
        if hit_stop_string || hit_max || hit_cancel {
            self.flush_tail(on_progress);
            // Cancellation is LAST in precedence: a run that would have
            // stopped on its own terms this token reports why it really
            // stopped, so a Stop button pressed as the model finishes does
            // not relabel a complete turn as a truncated one.
            return Some(if hit_stop_string {
                StopReason::StopString
            } else if hit_max {
                StopReason::MaxTokens
            } else {
                StopReason::Cancelled
            });
        }

        self.history.push(token_id);
        None
    }
}
