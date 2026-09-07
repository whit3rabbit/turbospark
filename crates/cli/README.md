# turbospark-cli

[![crates.io](https://img.shields.io/crates/v/turbospark-cli.svg)](https://crates.io/crates/turbospark-cli)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/whit3rabbit/turbospark/blob/main/LICENSE)

The command-line binaries for [turbospark](https://github.com/whit3rabbit/turbospark), a Rust LLM inference engine for Apple Silicon. `turbospark-model` finds, inspects, and installs models. `turbospark-check` runs them. `turbospark` is a unified front end over both, plus the server and the benchmark harness.

Generation is macOS-only and needs a Metal device. On other platforms the binaries build and parse arguments, and `turbospark-check` stops after printing the resolved request.

## Install

```sh
cargo install turbospark-cli
```

Homebrew gets you the same binaries plus the server:

```sh
brew install --cask whit3rabbit/tap/turbospark
```

## Quickstart

Pull a small model and talk to it. Nothing here needs a path:

```sh
turbospark-model pull tinyllama
turbospark-check --model tinyllama --chat
```

`--model` takes a catalog alias or a directory. An existing directory always wins, so a bare name can never quietly serve a different model than the one you typed.

## turbospark-model

The catalog and download surface. `pull` streams a checkpoint a layer at a time, so a 27 GB source never lands on disk whole.

```sh
turbospark-model list                    # the curated catalog
turbospark-model info gemma4             # one row in detail
turbospark-model probe Qwen/Qwen3-30B-A3B-GGUF   # will this repo run here?
turbospark-model pull gemma4             # install it
turbospark-model path gemma4             # where it went
turbospark-model rm gemma4               # remove it
```

`probe` reads headers rather than weights, so it answers in seconds and a few KB: which architecture the file claims, whether every block type has a kernel, and whether the expert cache arithmetic leaves it able to fit. It exits 0 only if the model would actually run.

## turbospark-check

Three ways in. Use `--messages-file` or `--chat` on any instruction-tuned model, because a raw `--prompt` skips the chat template and the output babbles:

```sh
# A raw prompt, no template applied
turbospark-check --model gemma4 --prompt "Hello"

# A JSON conversation, rendered through the model's own chat template
turbospark-check --model gemma4 --messages-file ./messages.json

# Interactive REPL, trimming old turns to fit the context window
turbospark-check --model gemma4 --chat
```

## turbospark

A unified front end that execs whichever peer binary a command needs (find it beside the executable, or on `PATH`):

```sh
turbospark run gemma4                    # turbospark-check --model gemma4 --chat
turbospark run gemma4 "hello"            # turbospark-check --model gemma4 --prompt "hello"
turbospark serve                         # turbospark-server
turbospark start                         # launch a background server daemon
turbospark stop / restart / status       # manage that daemon
turbospark start claude                  # point Claude Code at the local server
turbospark list / pull / info / rm / probe / recommend / path / auth
                                          # turbospark-model
turbospark bench                         # turbospark-bench
```

`start <agent>` connects an external coding agent (`claude`, `codex`, `opencode`, `hermes`, `openclaw`, `dsh`) to the local server through `ANTHROPIC_BASE_URL`/`OPENAI_BASE_URL`. `start` with anything else launches the background daemon that `stop`, `restart`, and `status` manage.

## Memory versus speed: `--expert-cache-slots`

Routed experts stream from disk, and a decoded token blocks on the ones that miss the per-layer cache. That makes the slot count the one knob trading RAM for throughput directly. Measured on the real Gemma 4 install (`docs/DECODE_BUDGET.md`):

| slots | peak RAM | decode |
|---|---|---|
| 16 | ~2.1 GB | ~44 tok/s |
| 32 | ~3.7 GB | ~51 tok/s |

GPU busy time is identical in both rows. The whole difference is how long the GPU sits idle waiting on the read.

Read those numbers narrowly. They are one model on one machine, an M4 Max with 36 GB, and the ratio transfers to other hardware far better than the absolute figures do.

**The default is `auto`.** It picks the largest count fitting a quarter of the memory left after the install's own weights and a 4 GiB reserve. It never resolves below 16, so a machine without headroom behaves exactly as it did before the flag learned to size itself.

The resolved count is printed at startup, and is worth recording beside any timing:

```sh
turbospark-check --model gemma4 --prompt "hi"
# expert cache: 32 slots per layer (auto)

turbospark-check --model gemma4 --prompt "hi" --expert-cache-slots 16
# expert cache: 16 slots per layer
```

Pin `16` to reproduce a published benchmark, or to hold the memory ceiling on a shared machine. Allowed values are `auto`, 8, 16, 24, 32, 48, 64, 96, and 128. A count above the model's own expert count is capped rather than allocated.

Changing it cannot change what the model writes. Routed slots dispatch in the router's ranking, so output is byte-identical at every slot count.

## Key modules

- `main.rs`: process entry point. Reads `argv`, triggers parsing, prints the resolved request, and routes execution.
- `generate/`: non-interactive text and chat-template generation driver.
- `chat.rs`: the REPL, using `fit_conversation_window` to keep history inside the context bound.
- `bin/model.rs` and `bin/model_cmd/`: `turbospark-model`'s argument parse and its nine subcommands. Nothing there decides anything, `turbospark-catalog` does.
- `bin/turbospark.rs`, `agent.rs`, `daemon.rs`: the unified `turbospark` front end, its coding-agent connectors, and its background server daemon.

## Development

```sh
cargo test -p turbospark-cli
```

Changes to decode, the KV cache, the output head, or a Metal encode loop need both real-model smokes. Greedy alone is not enough, see the first gotcha below:

```sh
cargo build --release -p turbospark-cli
printf '[{"role":"user","content":"Explain how coastal wetlands reduce flood damage."}]' > /tmp/p.json

# 1. Greedy, catches broken math
./target/release/turbospark-check --model ~/models/gemma4.gturbo \
  --messages-file /tmp/p.json --max-new 400 --seed 1 --temperature 0.0001 --top-k 1

# 2. Sampled at the CLI defaults, catches distribution bugs greedy cannot see
./target/release/turbospark-check --model ~/models/gemma4.gturbo \
  --messages-file /tmp/p.json --max-new 400 --seed 20260721
```

## Gotchas

1. **Greedy is not a weaker test, it is a different one.** `argmax` is invariant under every monotone transform of the distribution, so a broken distribution can stay byte-identical to correct under greedy and fail completely under sampling. Always check sampled coherence too.
2. **A raw `--prompt` on an instruction-tuned model babbles.** That is the chat template missing, not a decode bug. `--messages-file` and `--chat` apply it.
3. **`--expert-cache-slots` defaults to `auto`, so two machines print different slot counts and different tok/s for the same command.** That is the flag working, not drift. Pin `16` before comparing against a published timing, and read the startup line rather than assuming a count.

## License

MIT. See [LICENSE](https://github.com/whit3rabbit/turbospark/blob/main/LICENSE).
