# turbospark-cli

Process entry point binary (`turbospark-check`). Parses `argv` using `turbospark-invocation`, applies exit status and output stream routing, and drives GPU token generation (`RealForwardRunner`) on macOS.

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

- `main.rs`: Reads command-line arguments, delegates parsing to `turbospark-invocation`, prints resolved requests, and routes execution to generation routines.
- `generate.rs`: Coordinates tokenizer loading, chat template rendering, prefill chunking, and GPU decode generation loops.
- `chat.rs`: Interactive REPL loop maintaining user/assistant turn history and applying `fit_conversation_window` to manage context window bounds.

## Development & Test Commands

```sh
# Run tests for turbospark-cli
cargo test -p turbospark-cli

# Run CLI against a model with prompt string
cargo run -p turbospark-cli --bin turbospark-check -- --model /path/to/model --prompt "Hello"

# Interactive chat mode
cargo run -p turbospark-cli --bin turbospark-check -- --model /path/to/model --chat
```

## Real-Model Smoke Tests (Run Before Handoff)

Always run BOTH greedy and sampled smoke commands whenever altering decode, KV cache, output head, or Metal encode logic:

```sh
cargo build --release -p turbospark-cli
printf '[{"role":"user","content":"Explain how coastal wetlands reduce flood damage."}]' > /tmp/p.json

# 1. Greedy generation (catches math bugs)
./target/release/turbospark-check --model ~/models/gemma4.gturbo \
  --messages-file /tmp/p.json --max-new 400 --seed 1 --temperature 0.0001 --top-k 1

# 2. Sampled generation (catches distribution bugs greedy cannot see)
./target/release/turbospark-check --model ~/models/gemma4.gturbo \
  --messages-file /tmp/p.json --max-new 400 --seed 20260721
```

## Crate Gotchas

1. **Greedy is Not a Full Smoke Test**: `argmax` is invariant under monotone probability transformations. A broken logit distribution will often output identical greedy tokens while failing catastrophically under sampling. Always verify sampled output coherence.
2. **The power profile is resolved ONCE, in `open_session`.** That call is the only place this process asks the OS about Low Power Mode, and it happens before the first turn so an interactive `--chat` session cannot change pace mid-conversation because the machine was plugged in. `Session.rate` then feeds both `GenerationConfig` literals (`run_prompt` and `stream_turn`). `--max-tokens-per-sec` overrides whatever cap the profile carries, in both directions, so `--power-profile performance --max-tokens-per-sec 8` paces at 8 without enabling thermal stepping. `invocation::PowerProfile` and `runtime::PowerProfile` are separate enums (the parser crate is pure and depends only on `foundation`); `map_power_profile` is the single place they meet.
3. **Chat Template Necessity**: Running `--prompt` on an instruction-tuned model yields babble because special chat markup (`<|turn>`) is absent. Use `--messages-file` or `--chat` to ensure chat templates are rendered correctly.
