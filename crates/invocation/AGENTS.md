# turbospark-invocation

Pure command-line and server invocation parsing.

## Read first

- [Detailed module guide](../../.claude/docs/modules/invocation.md)
- [CLI reference](../../docs/CLI.md)
- [Server reference](../../docs/TOOL_CALLING.md)
- [Testing rules](../../docs/TESTING.md)

## Rules

- Keep parsing pure and independent of model opening or process setup.
- Adding a flag is a cross-surface change. Update the value type, parser,
  defaults, help, serialization or wire shape, and tests together.
- Preserve refusal messages and allowed-value validation. Unknown values must
  fail explicitly rather than silently selecting a default.
- Keep CLI and server meanings aligned unless their request boundary
  necessarily differs.

## Checks

```sh
cargo test -p turbospark-invocation
cargo fmt --check
```
