---
description: "Parse tool calls from model output, either as ToolCall events from the streaming decoder or from text you already have."
icon: wrench
---
# Tool call parsing

Use this page when your model emits tool calls and your code needs them as structured values. There are two routes. For a dialect the crate marks `Native`, the streaming decoder emits `ToolCall` events while the stream runs. Any dialect can also take the second route. You can call a parser directly on text you already hold, such as a reply the decoder released as raw text.

Every parser checks the call against the set of tools you offered. A call to a tool outside that set is an error. It is never passed through. That check is the main reason to use a parser.

Reading the markup yourself skips the check. A tool name you never offered then reaches your dispatcher unchallenged.

## Which dialects hand over calls

`ChatDialect::tool_call_support` answers `Native` for dialects whose markup the decoder parses. It answers `Prompted` for the rest. A `Prompted` dialect can still produce a call as prose, and the caller must recover it.

| Dialect | Support | Parser | Markup the parser reads |
| --- | --- | --- | --- |
| `Gemma` | Native | `GemmaToolCallParser` | `call:NAME{key:value,...}` |
| `ChatMl` | Native | `QwenToolCallParser` | `<function=NAME>` with `<parameter=KEY>` blocks |
| `Deepseek` | Native | `DeepseekToolCallParser` | One or more DSML `invoke` blocks |
| `Mistral` | Native | `MistralToolCallParser` | A JSON array of `{"name", "arguments"}` objects |
| `Glm` | Native | `GlmToolCallParser` | `NAME` followed by `<arg_key>` and `<arg_value>` pairs |
| `Kimi` | Native | `KimiToolCallParser` | `functions.NAME:IDX` and a JSON argument body |
| `Harmony` | Native | Decoder arm | Channel recipient, closed by `<\|call\|>` |
| `Llama3`, `MuseGlimmer`, `Spark`, `MiniMax`, `DeepseekV2` | Prompted | None in the decoder | Call text reaches the caller as content |

`Spark` uses the same tag names as GLM. The crate does not route Spark output through `GlmToolCallParser`. Test that path against real output before you rely on it.

## Parser signatures

Each parser is a unit struct with `new()` and `parse`. The signatures differ. Each format carries its call id in a different place, so each `parse` asks for the id in its own way.

| Parser | `parse` arguments | Returns |
| --- | --- | --- |
| `GemmaToolCallParser` | `text: &str`, `allowed_tools: &HashSet<String>`, `id: &str` | `ParsedToolCall` |
| `QwenToolCallParser` | `text: &str`, `allowed_tools: &HashSet<String>`, `id: &str` | `ParsedToolCall` |
| `DeepseekToolCallParser` | `text: &str`, `allowed_tools: &HashSet<String>`, `id_generator: impl FnMut() -> String` | `Vec<ParsedToolCall>` |
| `MistralToolCallParser` | `text: &str`, `allowed_tools: &HashSet<String>`, `id_generator: impl FnMut() -> String` | `(Vec<ParsedToolCall>, String)` |
| `GlmToolCallParser` | `body: &str`, `allowed_tools: &HashSet<String>`, `id: impl FnOnce() -> String` | `ParsedToolCall` |
| `KimiToolCallParser` | `body: &str`, `allowed_tools: &HashSet<String>` | `ParsedToolCall` |

The `body` for GLM and Kimi is the text between the block markers, with the markers removed. Kimi takes no id argument. The model's id is already in the text. Mistral returns the text that follows the JSON array. A model that keeps talking after its call still gives you the prose.

A `ParsedToolCall` has four public fields: `id`, `name`, `arguments` as a `JsonValue`, and `arguments_json` as a compact JSON string.

## Parse text you already have

The Qwen parser takes the body between the `<tool_call>` markers. The example parses one call. It then shows the refusal for a tool you did not offer.

```rust
use std::collections::HashSet;

use turbospark_tokenizer::{QwenToolCallParser, ToolCallParserError};

fn main() -> Result<(), ToolCallParserError> {
    let allowed: HashSet<String> = ["get_weather".to_string()].into_iter().collect();

    let body = "<function=get_weather>\n<parameter=city>\nOslo\n</parameter>\n</function>";
    let call = QwenToolCallParser::new().parse(body, &allowed, "call_1")?;
    println!("{} {} {}", call.id, call.name, call.arguments_json);

    let refused = QwenToolCallParser::new().parse("<function=delete_all>\n</function>", &allowed, "call_2");
    println!("{refused:?}");
    Ok(())
}
```

Expected output, not run during this audit: `call_1 get_weather {"city":"Oslo"}`, then `Err(UnknownTool("delete_all"))`.

## Receive calls while streaming

For a native dialect, pass the tool names to the decoder and read `ToolCall` events from `consume`. The example below uses the ChatML fixture's `<tool_call>` markup as the generated text.

```rust
use std::collections::HashSet;
use std::path::Path;

use turbospark_tokenizer::{
    Message, MfDetokenizer, MfTokenizer, Role, StructuredAssistantDecoder, ToolCallParserError,
};

fn main() -> Result<(), ToolCallParserError> {
    let tokenizer = MfTokenizer::load_from_dir(Path::new("path/to/checkpoint")).expect("load tokenizer");
    let prompt = tokenizer
        .apply_chat_template(&[Message::new(Role::User, "weather in Oslo?")])
        .expect("render prompt");
    let prompt_ids = tokenizer.encode(&prompt, false);

    // Stand-in for the model's sampled token ids.
    let generated = tokenizer.encode(
        "<tool_call>\n<function=get_weather>\n<parameter=city>\nOslo\n</parameter>\n</function>\n</tool_call>",
        false,
    );

    let mut next_call = 0u32;
    let allowed: HashSet<String> = ["get_weather".to_string()].into_iter().collect();
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
    for &id in &generated {
        let delta = detokenizer.push(id);
        for event in decoder.consume(id, &delta)? {
            println!("{event:?}");
        }
    }
    let tail = detokenizer.flush();
    for event in decoder.consume_flushed_text(&tail)? {
        println!("{event:?}");
    }
    for event in decoder.finish()? {
        println!("{event:?}");
    }
    println!("has_tool_calls: {}", decoder.has_tool_calls());
    Ok(())
}
```

The `ToolCall` event arrives on the token that closes `</tool_call>`, so it comes back from `consume` in the loop. Read the events that `consume` returns, not only the ones from `finish`. Most dialects behave this way. Harmony and Mistral are the exceptions, and [the streaming page](streaming.md) explains why.

## Errors

`ToolCallParserError` has three variants:

- `Malformed` means the body does not match the format. It also covers a function name that is not 1 to 64 bytes of ASCII letters, digits, `_`, or `-`.
- `UnknownTool(String)` means the name is not in the allowed set. The name is in the variant, so you can log it.
- `Oversized` means the body is over 256 KiB, the shared `MAXIMUM_BYTES` limit.

In the streaming decoder, any of these errors is permanent for that decoder. Later `consume` calls return `Malformed`. Call `take_failed_span_text` to recover the buffered text, as described in [Streaming output](streaming.md#errors-and-recovery).

## Where to go next

- [Streaming output](streaming.md) for the full per-token loop and error handling.
- [Chat templates and dialects](chat-templates.md) for offering tools to the checkpoint's template with `encode_generic_tool_chat`.
