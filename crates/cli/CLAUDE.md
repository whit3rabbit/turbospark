# mrefrust-cli

Process entry point binary (`mference-check`). Parses `argv` using `mrefrust-invocation`, applies exit status and output stream routing, and drives GPU token generation (`RealForwardRunner`) on macOS.

## Directory & File Structure

```
crates/cli/
+-- Cargo.toml              # Crate manifest
+-- src/
|   +-- main.rs             # CLI binary process entry point
|   +-- generate.rs         # Non-interactive text & chat template generation driver
|   \-- chat.rs             # Interactive REPL session runner using window-fit
\-- tests/
    +-- mference_check.rs   # CLI flag parse & exit status integration tests
    \-- real_generation.rs  # End-to-end real generation integration tests
```

## Key Modules

- `main.rs`: Reads command-line arguments, delegates parsing to `mrefrust-invocation`, prints resolved requests, and routes execution to generation routines.
- `generate.rs`: Coordinates tokenizer loading, chat template rendering, prefill chunking, and GPU decode generation loops.
- `chat.rs`: Interactive REPL loop maintaining user/assistant turn history and applying `fit_conversation_window` to manage context window bounds.

## Development & Test Commands

```sh
# Run tests for mrefrust-cli
cargo test -p mrefrust-cli

# Run CLI against a model with prompt string
cargo run -p mrefrust-cli --bin mference-check -- --model /path/to/model --prompt "Hello"

# Interactive chat mode
cargo run -p mrefrust-cli --bin mference-check -- --model /path/to/model --chat
```

## Real-Model Smoke Tests (Run Before Handoff)

Always run BOTH greedy and sampled smoke commands whenever altering decode, KV cache, output head, or Metal encode logic:

```sh
cargo build --release -p mrefrust-cli
printf '[{"role":"user","content":"Explain how coastal wetlands reduce flood damage."}]' > /tmp/p.json

# 1. Greedy generation (catches math bugs)
./target/release/mference-check --model ~/models/gemma4.gturbo \
  --messages-file /tmp/p.json --max-new 400 --seed 1 --temperature 0.0001 --top-k 1

# 2. Sampled generation (catches distribution bugs greedy cannot see)
./target/release/mference-check --model ~/models/gemma4.gturbo \
  --messages-file /tmp/p.json --max-new 400 --seed 20260721
```

## Crate Gotchas

1. **Greedy is Not a Full Smoke Test**: `argmax` is invariant under monotone probability transformations. A broken logit distribution will often output identical greedy tokens while failing catastrophically under sampling. Always verify sampled output coherence.
2. **Chat Template Necessity**: Running `--prompt` on an instruction-tuned model yields babble because special chat markup (`<|turn>`) is absent. Use `--messages-file` or `--chat` to ensure chat templates are rendered correctly.
