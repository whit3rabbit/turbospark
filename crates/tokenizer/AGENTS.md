# turbospark-tokenizer

Tokenizer loading, chat templates, dialect detection, and special-token
contracts.

## Read first

- [Detailed module guide](../../.claude/docs/modules/tokenizer.md)
- [CLI and model docs](../../docs/CLI.md)
- [Testing rules](../../docs/TESTING.md)

## Rules

- Keep `#![forbid(unsafe_code)]`.
- Resolve added-token IDs from a loaded tokenizer. Fixture JSON IDs may be
  renumbered by the tokenizer library.
- A dialect probe must test the same defining markers that its resolver
  requires. Do not infer a dialect from a coincidental token pair. Probe
  order is load-bearing when dialects share markers: the dialect that
  carries another dialect's markers is probed first (Flint shares GLM's
  role markers and is probed before GLM), because a wrong dialect selection
  misframes turns and can produce fluent but wrong output.
- Preserve template whitespace, reasoning controls, tool delimiters, stop
  IDs, and assistant prefixes. Prompt rendering is part of model behavior.
- Keep parser shims structurally scoped and delete them when the upstream
  parser supports the syntax directly.
- A dialect that thinks needs the reasoning split in four places: the
  resolver's `think_start_id`/`think_end_id`, an id-keyed
  `StructuredAssistantDecoder::consume` arm, and the dialect unions in
  `runtime::turn_stream::TurnSplitter::new` and
  `server::responses::map::may_produce_reasoning`. Only the decoder arm is
  compiler-enforced. Miss any other and the think block reaches the reply as
  text. An added-but-not-special think pair
  (GLM, Flint) still splits: the decoder keys on ids, so resolve the ids
  rather than leaving the sentinel. A template with no thinking-off switch
  (Flint) belongs in the always-split arm of both unions.
- Checkpoint `tokenizer.json` regexes run on Oniguruma (the `tokenizers`
  `onig` feature), so a crafted checkpoint can make encode panic (retry cap
  hit) or stall with no cancellation (quadratic patterns under the cap).
  This is an accepted risk for now: it needs a model the user chose to
  install, and the FFI guard and server poison recovery contain the panic.
  Do not swap the regex engine without a token-equality corpus across
  families plus the quality digests. `fancy-regex` closes the stall but moved
  token ids on qwen3vl-4b-vision and unlimited-ocr-2-4bit (long whitespace
  runs), anchors `$` differently from HF, and silently stops splitting on a
  limit hit. The ignored tests in `tests/regex_backend_hostile_pattern.rs`
  are the acceptance bar for that later research
  (https://github.com/whit3rabbit/turbospark/issues/250).
- Message text that spells a control marker (`<|im_end|>`, `<|im_start|>`,
  think or tool markers) encodes to the real marker id, so a tool result or
  pasted page can close its turn and open a new one. This is accepted
  behavior and matches HF transformers, vLLM and llama.cpp: every path
  renders one string, then encodes it with added-token extraction on.
  `tests/content_marker_spelling.rs` pins it. Do not "fix" it with
  escaping or `encode_special_tokens` on the whole string: templates parse
  content themselves (Qwen3 splits assistant text on `</think>`), the
  Unlimited-OCR-2 flatten path puts `<image>` in content on purpose, and
  benign prompts must keep their exact ids for the frozen digests. A real fix
  is span-protected encoding for untrusted roles, applied at every
  render-then-encode site (server `handler/plan.rs`, FFI
  `generate/prompt.rs`, CLI, lab, bench), with per-family template gates.

## Checks

```sh
cargo test -p turbospark-tokenizer
cargo fmt --check
```
