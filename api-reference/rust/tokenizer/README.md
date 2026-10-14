---
description: "Overview of turbospark-tokenizer, the crate that loads checkpoint tokenizers, renders chat prompts, and parses model output."
icon: box
---
# turbospark-tokenizer

Use this crate when your code turns chat messages into prompt token ids, turns generated ids back into text, or parses a model's tool calls. It loads a Hugging Face `tokenizer.json`, works out the chat dialect from the checkpoint's special tokens, and renders the checkpoint's own chat template when one ships. It has no Metal or GPU dependency. Its one native build step is covered under [sharp edges](#sharp-edges).

It does not load weights or run a model. The `turbospark-runtime` crate uses these same types during generation, so reach for this crate when you need prompt and output handling without the model.

## Add the package

The Cargo package is `turbospark-tokenizer`. Rust code imports it as `turbospark_tokenizer`.

Add the package from crates.io once the version you need is published, or use a path or git dependency from a checkout:

```toml
[dependencies]
turbospark-tokenizer = { path = "../turbospark/crates/tokenizer" }
```

```toml
[dependencies]
turbospark-tokenizer = { git = "https://github.com/whit3rabbit/turbospark", package = "turbospark-tokenizer" }
```

The crate has no Cargo features. It needs Rust 1.82 or newer, which is the workspace `rust-version`.

Inside the workspace, the core primitives crate is imported as `foundation`. That alias is internal to the tokenizer. Use `turbospark_tokenizer::TokenId`, which is a re-export of `i32`.

## Render and encode a prompt

The smallest useful flow loads a checkpoint directory, renders messages, and encodes the result:

```rust
use std::path::Path;

use turbospark_tokenizer::{Message, MfTokenizer, Role, TokenizerError};

fn main() -> Result<(), TokenizerError> {
    let tokenizer = MfTokenizer::load_from_dir(Path::new("path/to/checkpoint"))?;

    let messages = [
        Message::new(Role::System, "Answer briefly."),
        Message::new(Role::User, "What is the capital of France?"),
    ];
    let prompt = tokenizer.apply_chat_template(&messages)?;

    // The rendered template writes its own BOS text when the checkpoint needs
    // one, so encode the rendered chat text without adding a second BOS.
    let prompt_ids = tokenizer.encode(&prompt, false);
    println!("{:?}: {} prompt tokens", tokenizer.dialect, prompt_ids.len());
    Ok(())
}
```

`load_from_dir` reads `tokenizer.json` from the directory and requires it. It also reads `tokenizer_config.json`, `chat_template.jinja`, and `generation_config.json` when they exist. The [chat page](chat-templates.md) covers how those files change the result.

## Pick a page

| Task | Page | Main types |
| --- | --- | --- |
| Load a checkpoint, detect its dialect, and render chat prompts | [Chat templates and dialects](chat-templates.md) | [`MfTokenizer`](https://docs.rs/turbospark-tokenizer/latest/turbospark_tokenizer/struct.MfTokenizer.html), [`ChatDialect`](https://docs.rs/turbospark-tokenizer/latest/turbospark_tokenizer/enum.ChatDialect.html), [`Message`](https://docs.rs/turbospark-tokenizer/latest/turbospark_tokenizer/struct.Message.html), [`ReasoningEffort`](https://docs.rs/turbospark-tokenizer/latest/turbospark_tokenizer/enum.ReasoningEffort.html) |
| Turn generated ids into text, match stop strings, split reasoning from content | [Streaming output](streaming.md) | [`MfDetokenizer`](https://docs.rs/turbospark-tokenizer/latest/turbospark_tokenizer/struct.MfDetokenizer.html), [`StreamingStopMatcher`](https://docs.rs/turbospark-tokenizer/latest/turbospark_tokenizer/struct.StreamingStopMatcher.html), [`StructuredAssistantDecoder`](https://docs.rs/turbospark-tokenizer/latest/turbospark_tokenizer/struct.StructuredAssistantDecoder.html) |
| Parse tool-call markup from text | [Tool call parsing](tool-calls.md) | [`ParsedToolCall`](https://docs.rs/turbospark-tokenizer/latest/turbospark_tokenizer/struct.ParsedToolCall.html), the `*ToolCallParser` types |

The dialect list in the [chat page](chat-templates.md#dialects) names every chat family the crate resolves. Read it before you assume a checkpoint renders correctly, because some families need the checkpoint's own template and have no built-in fallback.

## Core types

- [`MfTokenizer`](https://docs.rs/turbospark-tokenizer/latest/turbospark_tokenizer/struct.MfTokenizer.html) is the entry point. It holds the resolved special token ids, the stop set, and the encode, decode, and render methods.
- [`ChatDialect`](https://docs.rs/turbospark-tokenizer/latest/turbospark_tokenizer/enum.ChatDialect.html) is the resolved family. `tool_call_support` reports whether the dialect's markup yields native tool calls.
- [`Message`](https://docs.rs/turbospark-tokenizer/latest/turbospark_tokenizer/struct.Message.html), [`Role`](https://docs.rs/turbospark-tokenizer/latest/turbospark_tokenizer/enum.Role.html), and [`FunctionDefinition`](https://docs.rs/turbospark-tokenizer/latest/turbospark_tokenizer/struct.FunctionDefinition.html) describe the conversation and its tools.
- [`TokenizerError`](https://docs.rs/turbospark-tokenizer/latest/turbospark_tokenizer/enum.TokenizerError.html) and [`ToolCallParserError`](https://docs.rs/turbospark-tokenizer/latest/turbospark_tokenizer/enum.ToolCallParserError.html) are the two error types.

Token ids are `i32` throughout. The constant [`NO_SUCH_TOKEN_ID`](https://docs.rs/turbospark-tokenizer/latest/turbospark_tokenizer/constant.NO_SUCH_TOKEN_ID.html) (`-1`) marks a role that the dialect frames as plain text rather than a single special token. Compare against it, never against a literal id from a fixture file.

## Errors

`TokenizerError` covers loading and rendering. It has four variants: `MissingSpecialToken`, `InvalidChatTemplate`, `UnsupportedForDialect`, and `InvalidVisionMarkers`. Match on the variant only when you want to act on it. Most callers can propagate it with `?`.

`ToolCallParserError` covers the parsers and the streaming decoder. It has three variants: `Malformed`, `UnknownTool(String)`, and `Oversized`.

## Sharp edges

- `encode` and `decode` do not return errors. When the tokenizers backend fails, they return an empty vector or an empty string. If the input is not empty and the output is empty, treat that as a failure.
- A missing or unreadable `tokenizer.json` comes back as `TokenizerError::InvalidChatTemplate`, not as a load-specific variant. Treat any error from `load_from_dir` as a failed load.
- Template rendering runs with a fuel limit of 1,000,000 and a rendered-output cap of 1 MiB. A template that exceeds either returns `InvalidChatTemplate`. Templates come from model artifacts, so the limits are deliberate.
- Tool-call parsers reject a body over 256 KiB. The shared JSON parser rejects nesting deeper than 64 levels.
- Several dialects have no built-in renderer. For those, a checkpoint without its own template returns an error rather than a guessed prompt. A wrong guess would produce fluent output that is not an answer.
- The `tokenizers` dependency is built with the `onig` feature. That pulls in `onig_sys`, which compiles Oniguruma from C, so the build machine needs a C compiler.
- The crate is `#![forbid(unsafe_code)]`.

## Where to go next

- [Chat templates and dialects](chat-templates.md) for loading, rendering, and reasoning levels.
- [Streaming output](streaming.md) for the per-token loop.
- [Tool call parsing](tool-calls.md) for the parsers and the native tool-call path.
- [Runtime](../runtime/README.md) for the generation loop that calls these types.
- [Adding a model family](https://github.com/whit3rabbit/turbospark/blob/main/docs/NEW_MODEL.md) before you add a dialect or checkpoint path.
- [The Rust library overview](../../rust.md) for the other crates.
