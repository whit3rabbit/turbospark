---
description: >-
  How turbospark-tokenizer loads a checkpoint, detects its chat dialect, and
  renders prompts, tool definitions, and reasoning levels.
icon: comments
---

# Chat templates and dialects

Use this page when you build the prompt for a model. Call `apply_chat_template` on a checkpoint that ships its own template, and check the dialect table before you assume a built-in renderer exists for your family. The crate prefers the checkpoint's Jinja template because the template decides the framing. The dialect decides the special token ids and the stop set.

That split matters when a checkpoint and a renderer disagree. A built-in renderer that is close but not exact produces fluent output that is not an answer, and no error reports it. The crate therefore refuses to render for dialects where it has no trustworthy renderer.

## Load a checkpoint

`MfTokenizer::load_from_dir` reads a directory with these files:

* `tokenizer.json` is required. It holds the vocabulary.
* `tokenizer_config.json` is optional. It supplies the BOS and EOS names and, for older checkpoints, a `chat_template` key.
* `chat_template.jinja` is optional. When present, it is the checkpoint's template and takes priority over the `chat_template` key.
* `generation_config.json` is optional. Its `eos_token_id` values join the stop set, filtered to ids inside the vocabulary.

Dialect resolution reads the special tokens in `tokenizer.json`. It does not read the directory name or the model family name.

## Dialects

The loader resolves one of twelve `ChatDialect` values. Gemma is the fallback, so a tokenizer that matches no other marker resolves to Gemma.

The "Chat text" column says what happens when the checkpoint ships no template. "Fallback" means the crate has a built-in renderer. "Required" means the call returns `TokenizerError::UnsupportedForDialect` or `InvalidChatTemplate` instead of rendering.

| Dialect       | Detected by                                                                  | Chat text without a template | Tool calls                                               |
| ------------- | ---------------------------------------------------------------------------- | ---------------------------- | -------------------------------------------------------- |
| `Gemma`       | Fallback when no other marker matches                                        | Fallback                     | Native, Gemma `call:NAME{...}` syntax                    |
| `ChatMl`      | `<\|im_end\|>`                                                               | Fallback                     | Native, Qwen `<function=...>` body inside `<tool_call>`  |
| `Deepseek`    | Fullwidth `User` marker and `<think>`                                        | Fallback                     | Native, DSML `invoke` blocks                             |
| `DeepseekV2`  | Fullwidth `User` marker without `<think>`, or the fullwidth BOS and EOS pair | Required                     | Prompted                                                 |
| `Mistral`     | `<s>` and `</s>`, with no Gemma turn marker                                  | Fallback                     | Native, `[TOOL_CALLS]` JSON array                        |
| `Harmony`     | `<\|start\|>`, `<\|message\|>`, and `<\|channel\|>`                          | Required                     | Native, channel recipient closed by `<\|call\|>`         |
| `MuseGlimmer` | `<\|start\|>` and `<\|eot\|>`                                                | Required                     | Prompted, `<atem:function_calls>` has no parser          |
| `Llama3`      | `<\|start_header_id\|>` and `<\|eot_id\|>`                                   | Fallback                     | Prompted                                                 |
| `Spark`       | `<\|Bot\|>`                                                                  | Required                     | Prompted, `<tool_call>` markup has no parser             |
| `MiniMax`     | `]~!b[`, `]~b]`, and `[e~[`                                                  | Required                     | Prompted                                                 |
| `Glm`         | `<\|observation\|>` and `<\|assistant\|>`                                    | Required                     | Native, `<tool_call>` text markers                       |
| `Kimi`        | `<\|im_assistant\|>` and `<\|im_middle\|>`                                   | Required                     | Native, section markers around a `functions.NAME:IDX` id |

A "Prompted" dialect can still produce a tool call. The model may write one as prose, and the caller has to recover it. `ChatDialect::tool_call_unsupported_reason` returns `Some` explanation for a Prompted dialect and `None` for a Native one.

## Render a prompt

The full example below loads a checkpoint, renders a prompt, checks reasoning support, and renders a prompt with a tool definition. The path is a placeholder.

```rust
use std::path::Path;

use turbospark_tokenizer::{
    FunctionDefinition, JsonValue, Message, MfTokenizer, ReasoningEffort, Role, TokenizerError,
};

fn main() -> Result<(), TokenizerError> {
    let tokenizer = MfTokenizer::load_from_dir(Path::new("path/to/checkpoint"))?;

    let messages = [
        Message::new(Role::System, "Answer briefly."),
        Message::new(Role::User, "What is the capital of France?"),
    ];
    let prompt = tokenizer.apply_chat_template(&messages)?;
    let prompt_ids = tokenizer.encode(&prompt, false);
    println!("{:?}: {} prompt tokens", tokenizer.dialect, prompt_ids.len());

    // Reasoning is a template input. Ask the checkpoint what it accepts.
    println!("reasoning support: {:?}", tokenizer.reasoning_support());
    println!("levels: {:?}", tokenizer.accepted_reasoning_levels());
    let thinking = tokenizer.apply_chat_template_with_reasoning(&messages, ReasoningEffort::Low)?;
    println!("thinking prompt: {} bytes", thinking.len());

    // Tool definitions render through the checkpoint template.
    let tools = [FunctionDefinition {
        name: "get_weather".to_string(),
        description: "Current weather for a city.".to_string(),
        parameters: JsonValue::parse(
            r#"{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}"#,
        )
        .expect("valid schema JSON"),
    }];
    let tool_ids = tokenizer.encode_generic_tool_chat(&messages, &tools, ReasoningEffort::Off)?;
    println!("tool prompt: {} tokens", tool_ids.len());
    Ok(())
}
```

Use `apply_chat_template` for plain chat. It renders the checkpoint template when one exists and falls back to the built-in renderer otherwise. `apply_chat_template_with_reasoning` takes a `ReasoningEffort` and is the same call with a level.

`apply_dialect_chat_template` skips the checkpoint template and runs the built-in renderer only. It exists so tests can compare the two. Do not use it for production prompts.

The built-in renderers trim message content, while a template trims only when it says to. For a byte-exact prompt, the checkpoint template is the reference.

## Tool definitions

`encode_generic_tool_chat` renders the checkpoint template with a `tools` list and returns token ids. It encodes the rendered text with `add_bos` off, the same as `apply_chat_template`. A checkpoint with no template returns an error here, because tools have no built-in renderer.

DeepSeek is the exception. `encode_deepseek_tool_chat` builds the DeepSeek-V4 tool chat in code, with a `## Tools` section, `<tool_result>` blocks, and DSML tool calls. It does not read the checkpoint template.

For the parsing side, see [Tool call parsing](tool-calls.md).

## Reasoning levels

`ReasoningEffort` has five values: `Off`, `Low`, `Medium`, `High`, and `XHigh`. `Off` is the default and renders the same bytes as before the type existed. Any other level turns thinking on, so a caller does not need to set `enable_thinking` separately.

Each checkpoint accepts a different set of levels. `accepted_reasoning_levels` renders a short conversation at each of the five values, drops the ones that produce identical prompts, and returns what remains. It reads the template at call time, so call it once when a model opens, not on every turn.

`reasoning_support` reports the template's shape as `ReasoningSupport`:

* `None` means the template reads no reasoning key, so a level changes nothing.
* `ToggleOnly` means the template reads `enable_thinking` but no effort key. Thinking turns on or off, and the level is ignored.
* `Level` means the template reads an effort key, so each level renders its own prompt.

On the crate's vendored test fixtures, the ChatML fixture reports `ToggleOnly` with levels `[Off, Low]`. The reasoning fixture reports `Level` with levels `[Off, Low, Medium, XHigh]`. For a `ToggleOnly` checkpoint, label the on-state as "On" in your UI. The `Low` value is a position in the list, not a label the model reads.

A level passed to a checkpoint with no template returns `UnsupportedForDialect`. Dropping the level silently would produce a prompt that did not think, with nothing to say so.

## Images

A `Message` can carry an ordered list of parts through `Message::with_parts`. A `ContentPart::Image` is a placeholder. It renders as a marker run, and the vision pipeline supplies the pixels later. The crate holds no image bytes.

Image messages need the checkpoint template. The built-in renderers refuse them. Call `verify_image_markers(vision_start, image_pad)` once when the vision config loads. It confirms that the two marker ids resolve and that a one-image turn renders exactly one of each.

## Errors on this path

* `UnsupportedForDialect` means the built-in renderer does not exist for the dialect, or it refuses a reasoning level or an image.
* `InvalidChatTemplate` means the template failed to render. That covers a `raise_exception` call from the template, an overrun of one of the two render limits from the overview, and a missing template for `encode_generic_tool_chat`.
* `InvalidVisionMarkers` means the marker ids do not hold up for this tokenizer.

Each message has a `role`, `content`, `content_parts`, `tool_calls`, `tool_call_id`, and `name`. The `Message::new` constructor sets only the role and the text. Set the other fields on the struct directly when a history turn needs them.

## Where to go next

* [Streaming output](streaming.md) takes the generated ids back to text.
* [Tool call parsing](tool-calls.md) covers the tool-call formats.
* The [Rust library overview](../) lists the crates that consume these prompts.
