# turbospark-cli

User-facing model, generation, image, music, voice cloning, and server command entry points:
- `turbospark-check`: Text generation and interactive chat.
- `turbospark-model`: Model discovery, probing, and local installation.
- `turbospark-image`: Diffusion image generation.
- `turbospark-music`: Music generation via MiniMax Music 3.
- `turbospark-clone`: Zero-shot voice cloning and speech synthesis via MegaTTS3.
- `turbospark`: Unified front end dispatcher.

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
- Exit status: 0 success, 1 a valid command line whose generation was refused
  or failed (`generate::Outcome::Failed`), 2 an invalid command line (owned
  by `turbospark-invocation`). Any new early return in a `run_*` path must
  return `Failed`, not fall through as success.
- `--chat` is text-only: `--image` is refused there, not ignored.
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
