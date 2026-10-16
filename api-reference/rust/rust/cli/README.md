---
description: "The turbospark-cli package: six command-line binaries for model management, text generation and chat, images, music, voice cloning, and the server."
icon: terminal
---

# turbospark-cli

`turbospark-cli` ships the command-line binaries for TurboSpark. Each binary reads its arguments, asks a library crate to make the decision, and prints the result. The crate does not contain the inference engine. It is the layer you call from a shell or from a script.

Generation needs macOS on Apple Silicon with a Metal-capable device. On other platforms the binaries still parse arguments and validate configuration. `turbospark-check` prints the resolved request and stops there.

## The binaries

| Binary | Use it to | Source |
| --- | --- | --- |
| `turbospark-check` | Generate text from a prompt or a conversation file, run an interactive chat, or rewrite a draft with the humanizer model. | `src/main.rs` |
| `turbospark-model` | List, inspect, probe, install, and remove models. Store a Hugging Face token. | `src/bin/model.rs` |
| `turbospark-image` | Generate one PNG, or pack a Diffusers export into an install. | `src/bin/image.rs` |
| `turbospark-music` | Generate a WAV file with MiniMax Music 3. | `src/bin/music.rs` |
| `turbospark-clone` | Clone a voice from a short reference recording and speak English text with MegaTTS3. | `src/bin/clone.rs` |
| `turbospark` | Front end. Routes subcommands to the binaries above, manages the background server, and connects coding agents. | `src/bin/turbospark.rs` |

The package declares these six binaries in `Cargo.toml`. The `turbospark` front end also calls two binaries this package does not build: `turbospark-server` (for `serve`, `start`, and `restart`) and `turbospark-bench` (for `bench`). See [Commands overview](commands-overview.md#turbospark-front-end) for how it finds them.

## How the pieces fit

- **Parsing.** `turbospark-check` hands its argument list to [`turbospark-invocation`](../invocation/README.md). That crate is a pure function from tokens to one outcome: a validated request, a help request, a version request, or one of six typed failures.
- **Exit status and stream routing.** The binary applies the exit status and stream routing that `invocation` computes. It does not decide them itself. The option table is in [`options.md`](../invocation/options.md).
- **Other binaries.** `turbospark-model`, `turbospark-image`, `turbospark-music`, and `turbospark-clone` parse their own arguments. They do not use `turbospark-invocation`.
- **Library crates.** The binaries call the library crates for the real work. Model catalog, probing, and installs come from [`turbospark-catalog`](../catalog/README.md). Image generation comes from [`turbospark-image`](../image/README.md). Music and voice cloning come from [`turbospark-audio`](../audio/README.md). Text generation uses the runtime, tokenizer, and selection crates.

## Exit status and output

Success exits `0`. The binaries differ on failures. `turbospark-model`, `turbospark-music`, and `turbospark-clone` exit `2` for a malformed invocation and `1` for a run that failed. `turbospark-image` exits `1` for both. `turbospark-check` exits `0` or `2` only, so a generation failure does not change its exit status.

Read [Exit status and output](exit-status-and-output.md) before you script against any of these binaries.

## Start with the job

| You want to | Start with | Read |
| --- | --- | --- |
| Generate text from one prompt | `turbospark-check --model MODEL --prompt TEXT` | [Commands overview](commands-overview.md#turbospark-check) |
| Chat in the terminal | `turbospark run MODEL` or `turbospark-check --model MODEL --chat` | [Examples](examples.md#chat-in-the-terminal) |
| Run a conversation with roles or images | `turbospark-check --messages-file FILE` | [Examples](examples.md#generate-from-a-conversation-file) |
| See which models are installed and where they are | `turbospark-model list`, `info`, `path` | [Commands overview](commands-overview.md#turbospark-model) |
| Check whether a Hugging Face repo will run here | `turbospark-model probe REPO` | [Examples](examples.md#check-a-repo-before-installing-it) |
| Pick a model that fits this machine | `turbospark-model recommend` | [Examples](examples.md#pick-a-model-for-this-machine) |
| Install or delete a model | `turbospark-model pull`, `rm` | [Examples](examples.md#install-and-remove-models) |
| Generate an image | `turbospark-image generate` | [Commands overview](commands-overview.md#turbospark-image) |
| Generate music | `turbospark-music generate` | [Commands overview](commands-overview.md#turbospark-music) |
| Clone a voice | `turbospark-clone speak` | [Commands overview](commands-overview.md#turbospark-clone) |
| Run a local server and point a coding agent at it | `turbospark start AGENT` | [Examples](examples.md#run-the-server-and-connect-an-agent) |
| Use these tools from a script | Exit codes and stream routing | [Exit status and output](exit-status-and-output.md) |

## Pages

- [Commands overview](commands-overview.md): every binary, subcommand, and flag, checked against the parser.
- [Exit status and output](exit-status-and-output.md): exit codes, what goes to stdout and stderr, and the cases where the source does not match what a script would expect.
- [Examples](examples.md): runnable command lines grouped by job.

## Limits

- Generation and image or audio synthesis need macOS with Metal. The parsing and validation work everywhere.
- `serve`, `start`, `restart`, and `bench` depend on binaries this package does not build. Their flags are owned by those binaries and are not documented here.
- The flag set for `turbospark-check` comes from `turbospark-invocation`. A flag only appears in these pages if the parser accepts it.
