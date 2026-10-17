---
description: "The turbospark-cli crate: the front-end binary, the generation and model-management command entry points, and the process surface every other crate is driven through."
icon: terminal
---

# cli

`crates/cli` (package `turbospark-cli`) owns the workspace's process entry
points: the unified `turbospark` front end and the per-surface binaries it
dispatches to. The crate is intentionally thin at every layer. Each binary
reads `argv`, hands the decisions to a library that owns them
(`invocation` for the generation command line, `catalog` for model rows and
installs), and applies the result: print, exit code, stream routing. End
users and scripts are the consumers; there is no library target to embed.

## What it does

The `Cargo.toml` declares eight binaries:

| Binary | Responsibility | Source |
| --- | --- | --- |
| `turbospark` | Unified front end; routes subcommands to the peer binaries | `src/bin/turbospark.rs` |
| `turbospark-check` | Text generation, chat REPL, humanization; parses with `invocation` | `src/main.rs` |
| `turbospark-model` | Catalog listing, probing, recommendation, and installs | `src/bin/model.rs` |
| `turbospark-image` | Diffusion image generation and packing | `src/bin/image.rs` |
| `turbospark-music` | Music generation | `src/bin/music.rs` |
| `turbospark-clone` | Zero-shot voice cloning (MegaTTS3) | `src/bin/clone.rs` |
| `turbospark-speak` | Speech synthesis (Fish Audio Drama 3) | `src/bin/speak.rs` |
| `turbospark-ocr` | Page images to text over Core ML | `src/bin/ocr.rs` |

The `turbospark` front end groups subcommands and mostly delegates:

| Group | Subcommands | Goes to |
| --- | --- | --- |
| Run and chat | `run`, `humanize` | `turbospark-check` (a bare positional prompt becomes `--prompt`, no prompt becomes `--chat`) |
| Server lifecycle | `serve`, `start`, `stop`, `restart`, `status` | `turbospark-server` for `serve`; `src/daemon.rs` for the rest |
| Agent connectors | `start claude`, `turbospark gemma4 claude`, ... | `src/agent.rs` (claude, codex, opencode, grok, gemini, hermes, openclaw, dsh) |
| Model management | `list`, `pull`, `info`, `rm`, `probe`, `recommend`, `path`, `auth` | `turbospark-model` |
| Benchmark | `bench` | `turbospark-bench` (from `crates/bench`) |
| Image | `image generate`, `image pack`, `ocr` | `turbospark-image` and `turbospark-ocr` |

## Where it sits

`turbospark-cli` depends on `invocation` (parsing), `catalog` (rows,
installs, alias resolution), `runtime`, `tokenizer`, `selection`, `repack`,
`model-io`, `core`, `image`, `audio`, `window-fit`, and `vision-io`. It has
**no lib target**, so nothing links against it; the `turbospark` dispatcher
finds peer binaries (`turbospark-server`, `turbospark-bench`, and its own
crate's siblings) beside its own executable or on `PATH` and execs them.
Users reach the crate through the shells documented in
[The CLI](../command-line-and-local-api/cli.md).

## Public API tour

There is no public Rust API. The surfaces are command grammars and exit
codes, and each binary keeps its decisions in a library:

- **`turbospark-check`** (`src/main.rs`): calls `invocation::parse`, applies
  `invocation::stream_routing` (which stream gets which text), prints the
  resolved request, then, on macOS, attempts real generation through
  `src/generate/` (`mod.rs` loop coordination, `session.rs` lifecycle and
  enum mapping, `format.rs` rendering, `vision.rs` image intake) and the
  `src/chat.rs` REPL. The generation half resolves `--model` through
  `catalog::resolve_model_arg`, where an existing directory always wins over
  a catalog alias.
- **`turbospark-model`** (`src/bin/model.rs`): a shell that parses a
  subcommand grammar (`list`, `list-speech`, `list-audio`, `info`, `probe`,
  `recommend`, `pull`, `pull-image`, `pull-audio`, `pull-drama3`,
  `pull-vision`, `path`, `rm`, `auth`) and calls `turbospark-catalog` for
  every verdict; `src/bin/model_cmd/` only renders. Its flat `Options` struct
  rejects any flag the chosen command does not read, so a misplaced `--force`
  errors instead of doing nothing.
- **`src/daemon.rs`**: the background server daemon behind `start`, `stop`,
  `restart`, and `status`. An `flock`-held lock over
  `<TURBOSPARK_HOME>/run/daemon.lock` closes the race between two
  near-simultaneous starts; pid and metadata live in `run/server.pid` and
  `run/server.meta`.
- **`src/agent.rs`**: launches coding agents against the local server,
  ensuring the daemon serves the requested model first and waiting for
  `GET /health` before exec. Credentials travel in the child environment,
  never in a process argument.

Exit codes are a contract across binaries: 0 for success, 1 for a command
line that parsed but whose generation was refused or failed, 2 for an
invalid command line (owned by `invocation` on the check side and by
`turbospark-model`'s `Error::Usage` on the model side).

## Examples

### Generate text with `turbospark-check`

Development commands from the crate's module guide. `--model` takes a
`.gturbo` directory or a catalog alias:

```sh
cargo run -p turbospark-cli --bin turbospark-check -- --model /path/to/model --prompt "Hello"
cargo run -p turbospark-cli --bin turbospark-check -- --model gemma4 --messages-file /tmp/p.json
cargo run -p turbospark-cli --bin turbospark-check -- --model gemma4 --chat
```

`--prompt` encodes verbatim with no chat template; `--messages-file` and
`--chat` render the checkpoint's own template.

### Drive the unified front end

From the dispatcher's own usage text:

```sh
turbospark run gemma4 "explain quantum computing"
turbospark run gemma4 --temperature 0.7 --max-new 256
turbospark pull gemma4
turbospark serve --model gemma4 --port 8080
turbospark start claude --model gemma4
```

Each of these resolves to a peer binary exec: `run` rebuilds a
`turbospark-check` argv, `pull` routes to `turbospark-model`, `serve` execs
`turbospark-server` (see [server](server.md)), and `start claude` goes
through `agent.rs`.

### Manage models with `turbospark-model` directly

Scripts usually call the dedicated binary so the exit codes stay
distinguishable:

```sh
turbospark-model list --filter gemma
turbospark-model probe owner/name
turbospark-model --out /tmp/installs pull tinyllama
```

The store root is `$TURBOSPARK_HOME` (default `~/.turbospark`); `HF_TOKEN`
covers gated repositories. See the full command reference in
[The CLI](../command-line-and-local-api/cli.md).

## Gotchas

{% hint style="warning" %}
Exit statuses are load-bearing for scripts. 0 is success, 1 is a valid
command line whose generation was refused or failed, and 2 is an invalid
command line. Any new early return in a `run_*` path must return the failed
status, not fall through as success, per the crate's `AGENTS.md`.
{% endhint %}

{% hint style="warning" %}
The crate has no lib target by design, so integration tests
(`tests/mference_check.rs`, `tests/model_cli.rs`, `tests/turbospark_cli.rs`)
can only drive the built binaries as processes. Logic worth unit-testing
lives in a `#[cfg(test)]` module inside its own source file; adding a lib
target would put every private helper on a public surface.
{% endhint %}

## Related pages

- [invocation](invocation.md)
- [catalog](catalog.md)
- [server](server.md)
- [The CLI](../command-line-and-local-api/cli.md)
- [Quickstart](../getting-started/quickstart.md)
