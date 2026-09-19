# turbospark-cli

User-facing model, generation, and server command entry points.

## Read first

- [Detailed module guide](../../.claude/docs/modules/cli.md)
- [CLI reference](../../docs/CLI.md)
- [Model catalog](../../docs/MODELS.md)
- [Real-model verification](../../.claude/docs/verification.md)

## Rules

- `turbospark-check` owns process setup over
  `turbospark-invocation`'s pure parser. Keep parse, resolution, and runtime
  behavior separate.
- `--model` accepts a path or catalog alias. Preserve directory-first
  resolution and the startup diagnostics that explain the resolved model.
- Keep `--help` and `--version` usable without a model.
- If adding a flag, update parsing, help, invocation, runtime plumbing, and
  tests together. The detailed checklist is in the archived guide.
- Real smoke requires both greedy and sampled generation. Greedy alone does not
  exercise sampler distribution.

## Checks

```sh
cargo test -p turbospark-cli
cargo run -p turbospark-cli --bin turbospark-check -- --help
cargo run -p turbospark-cli --bin turbospark-check -- --version
```

Use the real-install commands in the verification reference for model changes.
