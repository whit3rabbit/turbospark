---
uuid: "a8a695d0-3b88-4c92-baff-736945ac0e94"
title: "turbospark-tokenizer"
summary: "Tokenizer wrapper, chat dialects, and Jinja templates. The checkpoint's own template always beats the resolved dialect when the install ships one"
tags: ["crate", "tokenizer"]
depends_on: ["96040d84-192f-406a-8cd1-ab44c3a559ab"]
source: "crates/tokenizer/CLAUDE.md"
created: "2026-09-04"
updated: "2026-09-04"
---

## What does turbospark-tokenizer do?

Wraps HF `tokenizers` as `MfTokenizer`, resolves chat dialects (Gemma 4,
ChatML/Qwen, DeepSeek-V4, Mistral, Harmony/gpt-oss, Llama-3, Muse Glimmer),
renders chat templates (a per-dialect fallback in `chat_template/`, and the
real path through `jinja_chat_template.rs`'s minijinja wrapper),
incrementally detokenizes a token stream, matches stop sequences, and
parses per-dialect tool-call DSLs into a streaming structured decoder.
`#![forbid(unsafe_code)]`.

Key modules: `dialect/` (special-token probing and `ChatDialect`
resolution), `chat_template/` (fallback renderers, used only when an
install ships no template), `jinja_chat_template.rs` (the real path,
render the checkpoint's own template), `detokenizer.rs`,
`stop_matcher.rs`, `structured_decoder/`, `tool_call/`.

## Don't

- Don't treat `ChatDialect` as evidence of chat framing. It resolves IDS
  and the stop set only. Two checkpoints can present an identical
  special-token table, resolve to the same dialect, and be trained on
  completely different framing. `apply_chat_template` always renders the
  checkpoint's own Jinja template when one ships and falls back to
  `chat_template.rs`'s per-dialect renderer only for a malformed install.
- Don't hardcode a token id read out of fixture JSON. The vendored test
  fixtures embed placeholder added-token ids that the `tokenizers` crate
  renumbers at load time. Resolve ids from a loaded `MfTokenizer`
  (`token_to_id`, `end_of_turn_id`).
- Don't assume adding a `ChatDialect` variant is fully compiler-checked.
  Four matches are exhaustive and a missing arm is a build error
  (`resolve_dialect`, `apply_dialect_chat_template`,
  `encode_text_continuation`, `structured_decoder::consume`). Three more
  call sites key on `matches!(dialect, A | B)` and compile fine even when
  the new dialect belongs in neither arm: `server::handler::exec::needs_decoder`,
  `cli::generate::format`, `ffi::generate`. Decide those by hand.
- Don't render a multimodal message as a bare string with the image marker
  spelled into the text. It has to go through the ordered `ContentPart`
  list, or the id sequence never gets the special `<|image_pad|>` token and
  the model answers about a picture it never saw.
- Don't build a new minijinja compatibility workaround before checking
  `parenthesize_conditional_kwargs` in `jinja_chat_template.rs`. It exists
  because minijinja 2.x rejects a conditional expression as a keyword
  argument value, a real Jinja2 template shape. It's meant to be deleted
  once that lands upstream.

See [[crate-tokenizer-reasoning]] for the sharper per-dialect stop-set and
reasoning-channel gotchas.
