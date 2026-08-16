# turbospark-cli

Command-line entry point binary (`turbospark-check`). Parses command-line arguments using `turbospark-invocation`, applies exit status and output stream routing decisions, and drives GPU generation via `RealForwardRunner` on macOS.

## Binary Execution

```sh
# Run CLI against a model with a prompt string
cargo run -p turbospark-cli --bin turbospark-check -- --model /path/to/model.gturbo --prompt "Hello"

# Run non-interactive generation using a JSON conversation messages file
cargo run -p turbospark-cli --bin turbospark-check -- --model /path/to/model.gturbo --messages-file /path/to/messages.json

# Launch interactive chat REPL mode
cargo run -p turbospark-cli --bin turbospark-check -- --model /path/to/model.gturbo --chat
```

## Memory vs speed: `--expert-cache-slots`

Routed experts stream from disk, and a decoded token blocks on the ones that
miss the per-layer cache. That makes the slot count the one knob that trades
RAM for throughput directly, measured on the real Gemma 4 install
(`docs/DECODE_BUDGET.md`):

| slots | peak RAM | decode |
|---|---|---|
| 16 | ~2.1 GB | ~44 tok/s |
| 32 | ~3.7 GB | ~51 tok/s |

GPU busy time is identical in both rows; the difference is entirely how long
the GPU spends idle waiting on the read.

**The default is `auto`**, which picks the largest count fitting a quarter of
the memory left after the install's own weights and a 4 GiB reserve. It never
resolves below 16, so a machine without headroom behaves exactly as it did
before the flag learned to size itself. The resolved count is printed at
startup, and is worth recording beside any timing:

```sh
turbospark-check --model ~/models/gemma4.gturbo --prompt "hi"
# expert cache: 32 slots per layer (auto)

turbospark-check --model ~/models/gemma4.gturbo --prompt "hi" --expert-cache-slots 16
# expert cache: 16 slots per layer
```

Pin it to `16` to reproduce a published benchmark, or to hold the memory
ceiling on a shared machine. Allowed values are `auto`, 8, 16, 24 and 32; a
count above the model's own expert count is capped rather than allocated.
Changing it cannot change what the model writes -- routed slots dispatch in
the router's ranking, so output is byte-identical at every slot count.

## Key Modules

- `main.rs`: Process entry point reading `argv`, triggering argument parsing, printing resolved requests, and routing execution.
- `generate.rs`: Non-interactive text and chat template generation driver.
- `chat.rs`: Interactive REPL session runner using `fit_conversation_window` for context management.

## Development & Test Commands

```sh
# Run integration tests for turbospark-cli
cargo test -p turbospark-cli
```

## Real-Model Smoke Verification

When making changes to decode, KV cache, output head, or Metal encode logic, run both greedy and sampled smoke tests:

```sh
cargo build --release -p turbospark-cli
printf '[{"role":"user","content":"Explain how coastal wetlands reduce flood damage."}]' > /tmp/p.json

# 1. Greedy generation (verifies numerical correctness)
./target/release/turbospark-check --model ~/models/gemma4.gturbo \
  --messages-file /tmp/p.json --max-new 400 --seed 1 --temperature 0.0001 --top-k 1

# 2. Sampled generation (verifies distribution coherence)
./target/release/turbospark-check --model ~/models/gemma4.gturbo \
  --messages-file /tmp/p.json --max-new 400 --seed 20260721
```

## Crate Gotchas

1. **Greedy is Incomplete**: `argmax` is invariant under monotone probability transforms. A broken distribution can pass greedy generation while failing completely under sampling. Always test sampled output coherence.
2. **Chat Template Requirement**: Passing raw prompt strings to instruction-tuned models yields babble. Use `--messages-file` or `--chat` to apply the model's native chat template (`<|turn>`).
3. **`--expert-cache-slots` defaults to `auto`, so two machines will print different slot counts and different tok/s for the same command.** That is the flag working, not drift. Pin `16` before comparing a timing against a published one, and read the startup line rather than assuming a count.
