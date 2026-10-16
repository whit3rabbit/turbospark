---
description: "Turn generated token ids into text, stop on text strings, and split reasoning, content, and tool calls while a model streams."
icon: stream
---
# Streaming output

Use this page when your code receives generated token ids one at a time. Three types cover the loop. `MfDetokenizer` turns ids into text that is safe to show. `StructuredAssistantDecoder` splits each token into content, reasoning, or a parsed tool call. `StreamingStopMatcher` stops the stream when the output contains a text stop string.

The detokenizer is needed for almost any streaming caller, because a single token can hold half a UTF-8 character. The decoder is needed when the dialect has reasoning or tool markup you want to separate. The stop matcher is needed only for text stop strings, such as a stop sequence from an API request.

## The loop

The runtime calls the types in this order. Each token goes through the detokenizer and then the decoder. After the last token, flush the detokenizer, feed the tail to the decoder, and then call `finish` on the decoder.

```rust
use std::collections::HashSet;
use std::path::Path;

use turbospark_tokenizer::{
    Message, MfDetokenizer, MfTokenizer, Role, StreamingStopMatcher, StructuredAssistantDecoder,
    StructuredAssistantEvent, ToolCallParserError,
};

fn main() -> Result<(), ToolCallParserError> {
    let tokenizer = MfTokenizer::load_from_dir(Path::new("path/to/checkpoint")).expect("load tokenizer");
    let prompt = tokenizer
        .apply_chat_template(&[Message::new(Role::User, "hi")])
        .expect("render prompt");
    let prompt_ids = tokenizer.encode(&prompt, false);

    // Stand-in for the model's sampled token ids.
    let generated = tokenizer.encode("hello world STOP never reached", false);

    let mut next_call = 0u32;
    let allowed: HashSet<String> = HashSet::new();
    let mut decoder = StructuredAssistantDecoder::new(
        &tokenizer,
        allowed,
        || {
            next_call += 1;
            format!("call_{next_call}")
        },
        &prompt_ids,
    );
    let mut detokenizer = MfDetokenizer::new(&tokenizer);
    let mut stops = StreamingStopMatcher::new(vec!["STOP".to_string()]);
    let mut content = String::new();

    for &id in &generated {
        let delta = detokenizer.push(id);
        for event in decoder.consume(id, &delta)? {
            if let StructuredAssistantEvent::Content(text) = event {
                content += &stops.push(&text);
            }
        }
        if stops.is_stopped() {
            break;
        }
    }

    // The detokenizer holds bytes back, so flush it into the decoder before
    // finishing. The decoder keys on token ids, and a flushed tail has none.
    let tail = detokenizer.flush();
    for event in decoder.consume_flushed_text(&tail)? {
        if let StructuredAssistantEvent::Content(text) = event {
            content += &stops.push(&text);
        }
    }
    for event in decoder.finish()? {
        match event {
            StructuredAssistantEvent::Content(text) => content += &stops.push(&text),
            StructuredAssistantEvent::Reasoning(text) => println!("reasoning: {text}"),
            StructuredAssistantEvent::ToolCall(call) => println!("tool call: {}", call.name),
        }
    }
    content += &stops.finish();

    println!("content: {content:?}");
    println!("stopped: {}", stops.is_stopped());
    Ok(())
}
```

Expected output on the ChatML test fixture, not run during this audit: `content: "hello world "` and `stopped: true`. The stop string is removed from the output, and the text after it does not reach the caller.

`consume_flushed_text` is the same as `consume` with the "no such token" id. Use it for the detokenizer tail, which has no token behind it.

## Stop on token ids first

`MfDetokenizer` decodes with special tokens skipped, so a special token such as the ChatML end-of-turn token contributes no text. A text stop string will not match it. Check generated ids against `tokenizer.stop_token_ids` before you push them into the detokenizer. The runtime's token sink does this check, and a text matcher cannot replace it for special tokens.

`stop_token_ids` is a `BTreeSet<i32>`. It holds the dialect's stop set plus any extra EOS ids from `generation_config.json`.

## The detokenizer

`MfDetokenizer::push` takes one id and returns the text that is now safe to emit. It holds back two kinds of partial output:

- Byte-fallback tokens, such as `<0x..>` pieces, that have not yet completed a UTF-8 character.
- A trailing run of U+FFFD, which is the decoder's placeholder for bytes still in flight.

`flush` returns whatever it still holds, and you call it once at the end of the stream. The detokenizer also repairs a case where a later decode rewrites a space that an earlier decode already emitted. It emits the changed tail instead of dropping it.

The detokenizer borrows the tokenizer, so it cannot outlive it.

## The decoder

`StructuredAssistantDecoder::new` takes four arguments: the tokenizer, the set of tool names the caller offered, an id generator, and the prompt ids. The prompt ids are the rendered generation prompt, encoded. The decoder reads them to see whether the prompt already opened a thought block. ChatML with thinking on does this, and without that check the reasoning would appear as content.

`consume(token_id, delta)` returns a `Vec<StructuredAssistantEvent>`. An event is one of three values:

- `Content(String)` is visible answer text.
- `Reasoning(String)` is text inside a thought frame. It appears for dialects whose frame the decoder splits, such as ChatML, Harmony, Gemma, GLM, and Kimi. The DeepSeek arm does not emit reasoning events.
- `ToolCall(ParsedToolCall)` is a parsed call. See [Tool call parsing](tool-calls.md).

The id generator supplies ids for calls whose format carries no id of its own. Kimi reads the id from the model's text, so the generator is not called for Kimi calls.

Most dialects emit a tool call when its closing token arrives, so a call can come out in the middle of the stream. Harmony and Mistral are different. Harmony closes a call with `<|call|>`, which is a stop token, so the loop breaks before the decoder sees it. Mistral's call runs to the end of the turn. For both, the call is emitted from `finish`, and a loop that skips `finish` drops it.

## Errors and recovery

`consume` and `finish` return `ToolCallParserError`. Once a call fails to parse, the decoder refuses every later token with `Malformed`, so the stream is over for the decoder.

On that error, call `take_failed_span_text`. It returns the text the decoder had buffered for the failed span, or `None` when nothing was buffered. The runtime uses it to fall back to raw text, so a model that wrote a call-shaped span in prose keeps its text. `TurnSplitter` in `turbospark-runtime` (`src/turn_stream.rs`) shows this path.

## The stop matcher

`StreamingStopMatcher::new` takes a `Vec<String>` of stop strings. Empty strings are dropped. `push` returns the text that is safe to emit now. It holds back the longest suffix that could still become a stop string, so a stop split across two chunks still matches.

`is_stopped` becomes true once a stop string matches. After that, `push` returns an empty string. `finish` releases any text still held, unless the matcher has already stopped.

## Where to go next

- [Tool call parsing](tool-calls.md) covers `ToolCall` events and the standalone parsers.
- [Chat templates and dialects](chat-templates.md) covers the prompt that `prompt_ids` comes from.
