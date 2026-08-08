# mrefrust-cli

Command-line entry point binary (`mference-check`). Parses command-line arguments using `mrefrust-invocation`, applies exit status and output stream routing decisions, and drives GPU generation via `RealForwardRunner` on macOS.

## Binary Execution

```sh
# Run CLI against a model with a prompt string
cargo run -p mrefrust-cli --bin mference-check -- --model /path/to/model.gturbo --prompt "Hello"

# Run non-interactive generation using a JSON conversation messages file
cargo run -p mrefrust-cli --bin mference-check -- --model /path/to/model.gturbo --messages-file /path/to/messages.json

# Launch interactive chat REPL mode
cargo run -p mrefrust-cli --bin mference-check -- --model /path/to/model.gturbo --chat
```

## Key Modules

- `main.rs`: Process entry point reading `argv`, triggering argument parsing, printing resolved requests, and routing execution.
- `generate.rs`: Non-interactive text and chat template generation driver.
- `chat.rs`: Interactive REPL session runner using `fit_conversation_window` for context management.

## Development & Test Commands

```sh
# Run integration tests for mrefrust-cli
cargo test -p mrefrust-cli
```

## Real-Model Smoke Verification

When making changes to decode, KV cache, output head, or Metal encode logic, run both greedy and sampled smoke tests:

```sh
cargo build --release -p mrefrust-cli
printf '[{"role":"user","content":"Explain how coastal wetlands reduce flood damage."}]' > /tmp/p.json

# 1. Greedy generation (verifies numerical correctness)
./target/release/mference-check --model ~/models/gemma4.gturbo \
  --messages-file /tmp/p.json --max-new 400 --seed 1 --temperature 0.0001 --top-k 1

# 2. Sampled generation (verifies distribution coherence)
./target/release/mference-check --model ~/models/gemma4.gturbo \
  --messages-file /tmp/p.json --max-new 400 --seed 20260721
```

## Crate Gotchas

1. **Greedy is Incomplete**: `argmax` is invariant under monotone probability transforms. A broken distribution can pass greedy generation while failing completely under sampling. Always test sampled output coherence.
2. **Chat Template Requirement**: Passing raw prompt strings to instruction-tuned models yields babble. Use `--messages-file` or `--chat` to apply the model's native chat template (`<|turn>`).
