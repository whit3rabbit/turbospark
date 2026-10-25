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

## Checks

```sh
cargo test -p turbospark-tokenizer
cargo fmt --check
```
