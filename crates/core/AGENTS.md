# turbospark-core

Shared configuration, identifiers, and runtime-independent types.

## Read first

- [Detailed module guide](../../.claude/docs/modules/core.md)
- [Workspace rules](../../AGENTS.md)
- [Testing rules](../../docs/TESTING.md)

## Rules

- Keep `#![forbid(unsafe_code)]`.
- Token IDs crossing crate boundaries are signed 32-bit
  (`turbospark_core::TokenId`).
- Runtime knobs use one allowed-value set and one default. Update the parser,
  validation, serialization, and tests together.
- Keep shared types free of platform-specific runtime dependencies.

## Checks

```sh
cargo test -p turbospark-core
cargo check --target x86_64-unknown-linux-gnu -p turbospark-core
```
