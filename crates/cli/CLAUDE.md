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
4. **On `gpt-oss`, STDOUT is the answer and STDERR is the reasoning.** Harmony puts the model's reasoning in an `analysis` channel before its answer, so `ChannelSplit` runs that one dialect's output through `StructuredAssistantDecoder` and routes the two streams apart; redirecting stdout therefore captures the answer alone. For the other five families no decoder is built at all and the printing path is byte-identical to what it was. **Only the ANSWER accumulates into the returned reply**, which is what `chat.rs` appends to history: that is a correctness point rather than cosmetics, because Harmony's own convention drops the analysis channel from prior turns and feeding it back sends the model framing it was never trained to read. Note the consequence for any test asserting stderr is empty (`tests/real_generation.rs`): true for every install on this machine except a gpt-oss one, where reasoning is expected there.

   **DO NOT SKIP AN EMPTY DELTA BEFORE THE SPLIT.** `stream_turn` used to return early on empty text, which is harmless for five families and total for this one: the detokenizer skips special tokens, so EVERY Harmony frame token (`<|channel|>`, `<|message|>`, `<|end|>`, `<|start|>`) arrives as `(id, "")`. Skipping those means the state machine never sees a single transition and the whole turn prints as one run of content, markup words and all, with no error anywhere. Measured on the real install: the first end-to-end run after wiring the splitter printed `analysisThe user asks...assistantfinalThe sky appears blue...` to stdout, which looks exactly like a decoder that was never built. Every transition arrives as an empty delta; the emptiness check belongs AFTER `ChannelSplit::push`, on its output.
