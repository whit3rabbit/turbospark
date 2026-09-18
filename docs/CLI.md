# CLI reference

The primary command-line interface is the unified `turbospark` binary, alongside specialized standalone binaries and a benchmark harness:

- `turbospark` -- unified entry point providing intuitive subcommands for chat/generation (`run`, `image`), server management (`serve`, `start`, `stop`, `restart`, `status`), agent connectors (`start claude`, `start codex`), and model operations (`list`, `pull`, `info`, `rm`, `probe`, `recommend`, `auth`).
- `turbospark-check` -- run generation once against an install: a raw prompt, a rendered chat conversation, or an interactive REPL. See [`crates/cli/CLAUDE.md`](../crates/cli/CLAUDE.md).
- `turbospark-model` -- find, inspect, and install text models and image installs into the `~/.turbospark` store. See [`docs/MODELS.md`](MODELS.md).
- `turbospark-image` -- pack a local Diffusers image export or generate one PNG. See [`docs/IMAGE_GENERATION.md`](IMAGE_GENERATION.md).
- `turbospark-server` -- an OpenAI- and Anthropic-compatible HTTP server. See [`crates/server/CLAUDE.md`](../crates/server/CLAUDE.md).
- `turbospark-bench` -- throughput and memory benchmark harness. See [`docs/BENCHMARKING.md`](BENCHMARKING.md).

`--help` and `--version` are available globally across all binaries.

This page documents every flag and command.

## `turbospark` (unified CLI)

```sh
turbospark <command> [flags...]
```

Inspired by oMLX and Unsloth workflows, `turbospark` provides a single unified tool:

### Commands

| Category | Command | Description |
| --- | --- | --- |
| **Run & Chat** | `turbospark run <model> [prompt]` | Run interactive chat REPL (if prompt omitted) or single-turn prompt completion. Accepts flags like `--temperature`, `--max-new`, etc. |
| **Server (Foreground)** | `turbospark serve [options]` | Run OpenAI & Anthropic HTTP server in foreground (supports `--model-dir`, `--memory-guard`, etc.) |
| **Server (Daemon)** | `turbospark start [options]` | Start managed background server daemon (`~/.turbospark/run/server.pid`, logs to `~/.turbospark/logs/server.log`) |
| | `turbospark stop` | Stop running background server daemon |
| | `turbospark restart [options]` | Restart background server daemon |
| | `turbospark status` | Inspect background daemon status, port, endpoint, and health check |
| **Agents** | `turbospark start <agent>` | Connect coding agent (`claude`, `codex`, `opencode`, `hermes`, `openclaw`, `dsh`) to local server |
| **Models** | `turbospark list` | List catalog models, marking installed ones (`--filter TEXT`) |
| | `turbospark pull <alias\|--repo>` | Install model into store |
| | `turbospark info <alias>` | Inspect model details and gate targets |
| | `turbospark rm <alias>` | Remove installed model |
| | `turbospark probe <repo>` | Inspect remote Hugging Face headers without downloading |
| | `turbospark recommend` | Rank catalog models by fit for current hardware |
| | `turbospark auth` | Inspect, set, or clear Hugging Face credentials |
| | `turbospark path <alias>` | Print install directory |
| **Image** | `turbospark image generate [options]` | Generate one 1024x1024 PNG, using native Metal on macOS by default |
| | `turbospark image pack [options]` | Pack a local Diffusers image export into the separate image install format |
| **Bench** | `turbospark bench [options]` | Run benchmark harness |

### Image generation

```sh
turbospark image generate \
  --model z-image-turbo \
  --prompt "A red rabbit under a moonlit sky" \
  --seed 42 \
  --width 1024 --height 1024 \
  --steps 9 --backend native \
  --output image.png

turbospark image pack \
  --source /path/to/Z-Image-Turbo \
  --output /path/to/z-image-turbo.image.gturbo \
  --model-id Tongyi-MAI/Z-Image-Turbo \
  --model-revision f332072aa78be7aecdf3ee76d5c247082da564a6
```

The image command supports the frozen 1024-by-1024, batch-one, nine-step
Z-Image-Turbo envelope. `--seed` is optional and is randomized when omitted.
`--width`, `--height`, and `--steps` are validated against the selected
install before model I/O; there is no separate validation switch. Native Metal
is the default on macOS. The CPU backend is available only as an explicit
`--backend reference` diagnostic path. A completed output is never overwritten,
and failed or cancelled output is not published. Image installs use their own
manifest and `.image.gturbo` suffix; they are not text model rows and are not
opened by `turbospark-check`.

### oMLX compatibility

TurboSpark provides argument and lifecycle compatibility with oMLX:

```sh
# Managed background server
turbospark start
turbospark stop
turbospark restart
turbospark status

# Serve models with default settings or auto-discovery from a directory
turbospark serve --model-dir ~/models

# Memory guard tier at startup (safe maps to balanced tier)
turbospark serve --model-dir ~/models --memory-guard safe

# Custom memory guard ceiling in GB
turbospark serve --model-dir ~/models --memory-guard-gb 48

# Concurrency control (maps to session slots)
turbospark serve --model-dir ~/models --max-concurrent-requests 16

# Hugging Face mirror endpoint (e.g. for restricted regions)
turbospark serve --model-dir ~/models --hf-endpoint https://hf-mirror.com

# API key authentication
turbospark serve --model-dir ~/models --api-key your-secret-key

# Extended cache and MCP configuration options
turbospark serve --model-dir ~/models --paged-ssd-cache-dir ~/.omlx/cache --hot-cache-max-size 20% --mcp-config mcp.json
```

### Unsloth compatibility

```sh
# Quick interactive chat
turbospark run gemma4

# Quick one-shot prompt
turbospark run gemma4 "Explain quantum physics in three sentences"

# Connect coding agents to local server (Unsloth Start style)
turbospark start claude
turbospark start codex
turbospark start opencode
```

## `turbospark-check`

```sh
turbospark-check --model <path-or-alias> (--prompt TEXT | --messages-file PATH | --chat) [flags...]
```

`--model` takes either an install directory or a `turbospark-model` alias
(an existing directory always wins if both would resolve). Exactly one of
`--prompt`, `--messages-file`, `--chat` selects the mode; giving none, or more
than one, is a parse error.

### Mode selection

| Flag | Takes | Meaning |
| --- | --- | --- |
| `--model` | path or alias | required |
| `--prompt` | text | single-turn raw completion, no chat template applied |
| `--messages-file` | path | a JSON array of `{"role": ..., "content": ...}` turns, rendered through the checkpoint's own chat template |
| `--chat` | (none) | interactive REPL, template applied per turn |
| `--system` | text, repeatable | leading system message; `--chat` only, default none |

### Generation

| Flag | Takes | Default | Meaning |
| --- | --- | --- | --- |
| `--max-new` | positive integer | `1024` | generated-token limit |
| `--max-context` | positive integer, or `auto` | `auto` | context window; `auto` resolves to the checkpoint's own trained context, capped by what memory holds, and `4096` when the install declares none |
| `--load-guard` | `off\|relaxed\|balanced\|strict`, or a size in bytes | `relaxed` | how much of the machine a session may commit. A size is an absolute ceiling on what the engine ALLOCATES (expert cache + KV), never on the install's size. `relaxed` is what shipped before this flag and what every published memory figure was measured under; see `docs/LOAD_GUARD.md` |
| `--min-auto-context` | non-negative integer | `0` | refuse to open when `--max-context auto` resolves below this. Says nothing about an explicit `--max-context`: a caller naming a number has decided how to spend their own machine |
| `--temperature` | float | `0.2` | sampling temperature; `0.0` is greedy |
| `--top-k` | integer, `0`-`256` | `64` | rank-based candidate count, `0` disables |
| `--top-p` | float, `(0, 1]` | `0.95` | cumulative-probability threshold (sub-1.0 threshold requires `--top-k > 0`) |
| `--repetition-penalty` | float `> 0` | `1.0` | repetition penalty factor |
| `--seed` | non-negative integer | unset | determinism seed |
| `--stop` | text, repeatable | none | extra stop string, in addition to the tokenizer's own stop set |
| `--reasoning` | `off\|low\|medium\|high\|xhigh` | `off` | asks the checkpoint's own chat template to think first; the accepted set is the checkpoint's, not this CLI's -- an unsupported level is refused by name |

### Performance and memory

| Flag | Takes | Default | Meaning |
| --- | --- | --- | --- |
| `--image` | path, repeatable | none | image to include; all of them land in ONE turn unless `--image-batch` |
| `--image-batch` | flag | off | run the prompt once per `--image` instead of once with all of them |
| `--vision-sidecar` | path, or `auto` | none | attach a standalone vision-tower sidecar install to a text-only trunk. `auto` resolves one from the installed store by the trunk's own family and hidden size after the trunk opens: the trunk's own tower wins, zero installed towers runs text-only with the reason on stderr, and two candidate towers REFUSE the open because the revision pin is load-bearing; see [`docs/VISION.md`](VISION.md) |
| `--rdadvise` | `off\|normal\|aggressive` | `off` | read-ahead hint mode for streamed expert reads (macOS) |
| `--expert-cache-slots` | `8\|16\|24\|32`, or `auto` | `auto` | routed-expert slot cache size; `auto` never resolves below `16` |
| `--expert-residency` | `auto\|streamed\|mapped` | `auto` | routed-expert storage; `auto` selects mapped only when the minimum streamed cache cannot fit the measured headroom |
| `--prefill-chunk` | `32\|64\|128\|256\|512\|1024\|2048\|4096`, or `auto` | `128` | prompt-processing chunk size; drives chunked prefill for supported families (Gemma 4, dense Llama/Mistral) and falls back to sequential prefill for others; `TURBOSPARK_PREFILL_CHUNK` environment variable overrides when set |
| `--power-profile` | `performance\|balanced\|efficiency` | `performance` (or `efficiency` under Low Power Mode) | decode rate governance |
| `--max-tokens-per-sec` | float `> 0` | uncapped (or the efficiency profile's reading speed) | hard decode rate cap |
| `--kv-bits` | `off\|2\|3\|3.5\|4` | `off` | TurboQuant KV-cache quantization width; `3.5` splits K3/V4. An unsupported `head_dim` or an install this decode flow does not carry the codec for REFUSES at open by name rather than silently running FP16; see [`docs/TRUBOQUANT.md`](TRUBOQUANT.md) |

### Speculative decoding

| Flag | Takes | Default | Meaning |
| --- | --- | --- | --- |
| `--speculative` | `off\|auto`, or a block size `1`-`15` | `auto` | a named block size FAILS at open if the install cannot serve it; acceptance is exact only at `--temperature 0` |
| `--speculative-drafter` | `auto\|mtp\|dflash` | `auto` | which drafter `--speculative` drives; `auto` enables an MTP head but only REPORTS a DFlash one (DFlash measures 0.88x on prose -- name it explicitly to actually run it; see [`docs/MTP.md`](MTP.md) and [`docs/DFLASH2.md`](DFLASH2.md)) |

### Steering (obliteration)

Six flags, all requiring `--steering` itself; see the dedicated
[Steering (obliteration)](#steering-obliteration) section below for the full
picture. `--steering` and the three per-vector knobs among these are
REPEATABLE, paired positionally: the i-th `--steering-mode`/`--steering-scale`/
`--steering-layers` configures the i-th `--steering` vector. A knob list
shorter than the vector list extends by its last value, so one scale steers
every vector; more knob values than vectors is a parse error.

| Flag | Takes | Default |
| --- | --- | --- |
| `--steering` | path to a `.gguf` control vector; repeatable | none (off) |
| `--steering-mode` | `ablate\|add\|clamp\|renorm`; repeatable | `ablate`, or whatever the file declares |
| `--steering-scale` | float; repeatable | `1.0` |
| `--steering-layers` | `START:END`; repeatable | every layer the vector covers |
| `--steering-target` | float | `0.0` |
| `--steering-gate` | float | `0.0` |

### Misc

| Flag | Meaning |
| --- | --- |
| `--quiet` | suppress incidental output |
| `--help` | print usage and exit |
| `--version` | print the version and exit |

### Examples

```sh
# Raw prompt. Babbles on an instruction-tuned model -- there is no chat
# template applied here, which is the point of --messages-file below.
turbospark-check --model ~/models/gemma4.gturbo --prompt "hi"
```

```sh
# A rendered conversation. printf writes the file inline; a real caller
# would generate this JSON programmatically.
printf '[{"role":"user","content":"Explain how coastal wetlands reduce flood damage."}]' > /tmp/p.json
turbospark-check --model ~/models/gemma4.gturbo --messages-file /tmp/p.json \
  --max-new 400 --seed 1 --temperature 0.0001 --top-k 1
```

```sh
# Interactive chat.
turbospark-check --model ~/models/gemma4.gturbo --chat
```

### Images

`--image` is repeatable and every path lands in ONE turn, prepended to the
last user message -- which is where the reference processor puts them, so the
rendered prompt matches what mlx-vlm builds for the same request.

```sh
printf '[{"role":"user","content":"Transcribe the text in this image."}]' > /tmp/p.json
turbospark-check --model ~/models/qwen38-27b-vision.gturbo --messages-file /tmp/p.json \
  --image page.png --max-new 200 --temperature 0.0001 --top-k 1
```

`--image-batch` runs the prompt once PER image instead, over ONE open runner:
the bulk-OCR shape. The tower's scratch is allocated and dropped per page, so
peak memory is flat in the page count rather than growing with it.

```sh
turbospark-check --model ~/models/qwen38-27b-vision.gturbo --messages-file /tmp/p.json \
  --image p1.png --image p2.png --image p3.png --image-batch --max-new 200
```

A `--messages-file` can place images itself, which is what a multi-turn
conversation needs. The shape is HF's content-part list plus a `path`, since
nothing else in this JSON could name a file:

```json
[{"role": "user", "content": [
  {"type": "image", "path": "page.png"},
  {"type": "text",  "text": "Transcribe the text in this image."}
]}]
```

Four things this refuses rather than guessing at:

- **`--prompt` with `--image`.** That mode encodes verbatim with no template,
  so there is no marker run for the image to land in. Use `--messages-file`.
- **`--image` alongside a file that already carries image parts.** The file
  says where each image goes and the flag does not, so the pairing between a
  path and a marker run would be ambiguous.
- **`--image-batch` with a file that carries image parts**, for the same
  reason: which of the file's own parts would each page keep?
- **An install with no vision tower.** Named at the point the images are
  attached rather than ignored.

The install must carry its `preprocessor_config.json`: the pixel budget is
read from the checkpoint and has no safe default (`crates/vision-io` Gotcha
6). An install missing it is refused by name.

A text-only trunk can still take images through a standalone tower:
`--vision-sidecar` attaches an `<alias>.gturbo-vision` install beside it
(the format and its gates are [`docs/VISION.md`](VISION.md)'s sidecar
section), and `--vision-sidecar auto` resolves one from the installed store
by family and hidden size when you would rather not name the path.

## `turbospark-model`

```sh
turbospark-model <command> [flags...]
```

| Command | Args | Flags | What it does |
| --- | --- | --- | --- |
| `list` | none | `--filter TEXT` | prints every curated catalog row, marks installed ones |
| `info` | `<alias>` | none | prints one catalog row in full, including its gate targets |
| `probe` | `<repo>[@rev]` | `--file NAME.gguf`, `--sidecar-repo REPO[@rev]` | reads a Hugging Face repo's headers only, no download; reports whether this engine would run it |
| `recommend` | none | `--context N`, `--budget BYTES`, `--load-guard TIER`, `--probe`, `--discover [N]` | ranks models by whether they fit this machine and how much is known about them; `--load-guard` MUST match what the session will open with, since the ranking and the loader's refusal share one memory budget; `--budget` accepts bare bytes or suffixes (`36G`, `36GB`, `36GiB`); `--context` defaults to `4096`; `--probe` reads every curated row's header for exact numbers, `--discover` also ranks the N most-downloaded GGUF repos on Hugging Face (default `20`) through the same probe |
| `pull` | `<alias>`, or `--repo REPO[@rev] --alias NAME` | `--out DIR`, `--file NAME.gguf`, `--sidecar-repo REPO[@rev]`, `--force` | installs a curated model, or any repository the probe accepts; `--out` overrides install destination; `--force` installs past a probe refusal |
| `path` | `<alias>` | none | prints the install directory (fails loudly if not installed) |
| `rm` | `<alias>` | `--yes` / `-y` | deletes an install; without `--yes`, prompts for the alias name to confirm |
| `auth` | none | `--clear`, `--status`, `--set` | inspects, saves, or clears Hugging Face credentials in `~/.turbospark/hf_token`; `--set` reads from a non-echoing prompt when interactive, or from stdin when piped |

Global options: `--help` / `-h` / `help`, `--version` / `-V` / `version`. For gated repositories, set `HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN`; credentials are never accepted in command-line arguments.

Flags not accepted by the given command are rejected rather than silently ignored.

See [`docs/MODELS.md`](MODELS.md) for the catalog itself, and what it takes
to add a row.

## `turbospark-server`

```sh
turbospark-server --model <path-or-alias>... [flags...]
turbospark-server <tokenizer-dir> [port]   # legacy scripted mode, see below
```

Serves OpenAI `/v1/chat/completions`, OpenAI's legacy raw-prompt
`/v1/completions`, OpenAI's `/v1/responses`, Anthropic `/v1/messages` and
`/v1/messages/count_tokens` (no generation, just the prefill token count),
`/v1/models`, and `GET /health` from one open install, one request at a
time (one runner per process). `/health` takes no lock and answers even
while a generation is in flight. `/v1/responses` is stateless: it refuses
`previous_response_id` rather than faking continuity across requests.

For Claude Code gateway discovery, each generative backend appears twice in
`GET /v1/models`: its canonical install id and the deterministic
`claude-turbospark-<canonical-id>` alias. The alias selects that exact backend
on a multi-model server, while the canonical id remains the one shown by the
FFI server-info surface. A `HEAD /api/hello` from Claude Code is only a
best-effort connection-warming probe; this server intentionally leaves it
unimplemented and uses `GET /health` for liveness.

| Flag | Takes | Default | Meaning |
| --- | --- | --- | --- |
| `--model` | path or alias, repeatable | optional | attach one or more generation installs; exact ids and aliases route requests to the matching install. A single model keeps the legacy unknown-name fallback |
| `--model-dir` | path | none | directory containing `.gturbo` models; enables auto-discovery (omlx compatibility) |
| `--port` | u16 | `8080` | listen port |
| `--max-context` | integer, or `auto` | `auto` | same semantics as `turbospark-check`'s |
| `--load-guard` | `off\|relaxed\|balanced\|strict`, or a size | `relaxed` | same as `turbospark-check`'s, resolved once at startup |
| `--memory-guard` | `safe\|balanced\|strict\|relaxed\|off` | unset | alias for `--load-guard`; `safe` maps to `balanced` (omlx compatibility) |
| `--memory-guard-gb` | integer | unset | set custom memory guard ceiling in gigabytes (omlx compatibility) |
| `--min-auto-context` | non-negative integer | `0` | same as `turbospark-check`'s |
| `--expert-cache-slots` | `8\|16\|24\|32`, or `auto` | `auto` | same semantics as `turbospark-check`'s |
| `--pool-size` | positive integer | `1` | how many independent runners to open of `--model`'s ONE install; N > 1 serves N concurrent generations behind one public id, routed to the least-busy member. Each member pays the load guard on its own, so one that does not fit refuses at startup |
| `--session-slots` | positive integer | `1` | concurrent conversation KV states maintained in pool |
| `--max-concurrent-requests` | positive integer | `1` | alias for `--session-slots` (omlx compatibility) |
| `--hf-endpoint` | url | unset, `$HF_ENDPOINT` | Hugging Face mirror endpoint (e.g. `https://hf-mirror.com`) |
| `--paged-ssd-cache-dir` | path | none | tiered KV SSD cache directory hint (omlx compatibility) |
| `--hot-cache-max-size` | string | none | in-memory hot cache size hint, e.g. `20%` (omlx compatibility) |
| `--mcp-config` | path | none | MCP tools configuration file path (omlx compatibility) |
| `--bind` | `loopback\|tailnet` | `loopback` | `tailnet` binds this machine's Tailscale IPv4 address and requires `--api-key` or `$TURBOSPARK_API_KEY`; there is no TLS |
| `--api-key` | string | unset, `$TURBOSPARK_API_KEY` | require this key, as `x-api-key: <key>` or `Authorization: Bearer <key>`, on every route except `GET /health`; see [Bearer/x-api-key auth](#bearerx-api-key-auth) |
| `--power-profile` | `performance\|balanced\|efficiency` | unset | same as `turbospark-check`'s |
| `--max-tokens-per-sec` | float `> 0` | uncapped | same as `turbospark-check`'s |
| `--speculative` | `off\|auto`, or `1`-`15` | `auto` | resolved ONCE at process open, not per request; acceptance is exact only at temperature 0, so a request sampled above that falls back to sequential decode silently |
| `--speculative-drafter` | `auto\|mtp\|dflash` | `auto` | same as `turbospark-check`'s |
| `--kv-bits` | `off\|2\|3\|3.5\|4` | `off` | same as `turbospark-check`'s, resolved once at open for every request the process serves |
| `--guardrails` | `on\|off` | `on` | tool-call rescue, argument validation, one retry (see [`docs/FORGE_GUARDRAILS.md`](FORGE_GUARDRAILS.md)); a request carrying `tools` is BUFFERED rather than streamed while this is on, since a verdict needs the whole turn -- a request without tools streams exactly as it always did |
| `--prefix-reuse` | `on\|off` | `off` | reuse matching KV state from the preceding request; opt in only when every request belongs to one trusted client, because the process-wide cache is not partitioned by API key or client and prefix matches affect response timing |
| `--reasoning` | `off\|low\|medium\|high\|xhigh` | `off` | default reasoning effort when a request omits `reasoning_effort` |
| `--system` | text, repeatable | none | default system prompt for requests that carry no `system` or `developer` message of their own. Repeats join with a newline. THE CALLER WINS: a request that sends its own is left exactly as it arrived, so the two can never produce two system turns (which three of the five fallback renderers refuse outright). Applied at the one point every wire format funnels through, so it reaches `/v1/chat/completions`, `/v1/messages`, `/v1/responses` and `/api/chat` alike, and is counted by `/v1/messages/count_tokens`. `/v1/completions` renders no chat template and is unaffected |
| `--system-file` | path | none | the same prompt read from a file, for one too long to sit on a command line. Mutually exclusive with `--system`; an empty file is refused rather than read as "no default" |
| `--steering` | path to `.gguf` | none | see [Steering (obliteration)](#steering-obliteration) |
| `--steering-mode` | `ablate\|add\|clamp\|renorm` | `ablate`, or the file's own | as above |
| `--steering-scale` | float | `1.0` | as above |
| `--steering-layers` | `START:END` | every layer the vector covers | as above |
| `--steering-target` | float | `0.0` | as above |
| `--steering-gate` | float | `0.0` | as above |
| `--help` / `-h`, `--version` / `-V` | -- | -- | -- |

**Not present on the server:** `--prefill-chunk`, `--rdadvise`, and `--quiet`
(chunked prefill, streaming read-ahead hint, and quiet mode have no server-side
flags). Per-turn generation parameters (`--temperature`, `--top-k`, `--top-p`,
`--max-new`, `--stop`, `--seed`) are not server CLI flags --
they are received per-request on the OpenAI and Anthropic wire protocols rather
than pinned for the process, unlike `--steering`, `--speculative` and `--reasoning` which are
resolved once at open and configure defaults/capabilities for requests the process serves.

The legacy positional form (`turbospark-server <tokenizer-dir> [port]`) runs
the portable scripted backend against canned completions rather than a real
install, and always binds loopback regardless of `--bind`, unauthenticated
regardless of `--api-key` -- neither flag exists in that mode's `ModelArgs`.

```sh
turbospark-server --model ~/models/gemma4.gturbo
turbospark-server --model gemma4 --bind tailnet --port 8080
turbospark-server --model gemma4 --guardrails off
```

### Bearer/x-api-key auth

`--api-key KEY` (or `$TURBOSPARK_API_KEY`, checked when the flag is absent so
the key never appears in `ps`) requires that key on every route except
`GET /health`, as either `x-api-key: KEY` (what Anthropic-native clients,
including Claude Code, send) or `Authorization: Bearer KEY`. A missing or
wrong key gets a 401 with an OpenAI-shaped error body. `--model` mode only;
the legacy scripted mode has no `--api-key`.

`--bind tailnet` refuses to start unless `--api-key KEY` or a non-empty
`$TURBOSPARK_API_KEY` supplies an application credential. The credential is
required in addition to Tailnet ACL reachability. Loopback mode keeps
authentication optional for compatibility with local-only deployments.

```sh
TURBOSPARK_API_KEY=sk-... turbospark-server --model gemma4 --bind tailnet
curl -H 'x-api-key: sk-...' http://100.x.y.z:8080/v1/models
```

## `turbospark-bench`

```sh
turbospark-bench <tokenizer-dir> [--real]
turbospark-bench --model <install-dir> [flags...]
```

Harness for measuring throughput tok/s, split prefill/decode latencies, and
peak `phys_footprint` memory usage under the frozen community benchmark protocol.
See [`docs/BENCHMARKING.md`](BENCHMARKING.md) for background and baseline numbers.

### Modes

- **Scripted mode**: `turbospark-bench <tokenizer-dir>` (portable, runs on Linux;
  measures loop and tokenizer overhead with canned logit replay).
- **Synthetic real mode**: `turbospark-bench <tokenizer-dir> --real` (macOS;
  builds a tiny temporary synthetic model to exercise real GPU forward dispatch).
- **Real model mode**: `turbospark-bench --model <install-dir> [flags...]` (macOS;
  runs the full benchmark protocol against a `.gturbo` install).

### Real model flags

| Flag | Takes | Default | Meaning |
| --- | --- | --- | --- |
| `--model` | path | required | path to `.gturbo` install directory |
| `--case` | case ID | all 3 cases | restrict run to one protocol case (`short-explanation`, `medium-review`, `long-synthesis`) |
| `--expert-cache-slots` | `8\|16\|24\|32` | `16` | routed-expert slot cache size per layer |
| `--power-profile` | `performance\|balanced\|efficiency` | `performance` | power governance mode (explicitly defaults to `performance` rather than LPM state) |
| `--max-tokens-per-sec` | float `> 0` | uncapped | decode rate cap |
| `--speculative` | `off\|auto`, or block size `> 0` | `off` | speculative decoding (defaults off to preserve sampled protocol numbers) |
| `--speculative-drafter` | `auto\|mtp\|dflash` | `auto` | drafter to drive under `--speculative` |
| `--shaping` | `protocol\|greedy` | `protocol` | `protocol` uses the protocol's fixed temperature/top-k/top-p; `greedy` samples argmax |
| `--kv-bits` | `off\|2\|3\|3.5\|4` | `off` | TurboQuant KV-cache quantization width; OFF BY DEFAULT so every frozen memory-oracle and quality-gate row in this crate stays valid without the flag (see `crates/bench/CLAUDE.md` and AGENTS.md Gotcha 35) |

```sh
# Run full protocol benchmark against Gemma 4
cargo run --release -p turbospark-bench --bin turbospark-bench -- --model ~/models/gemma4.gturbo

# Run single protocol case
cargo run --release -p turbospark-bench --bin turbospark-bench -- --model ~/models/gemma4.gturbo --case short-explanation
```

## Steering (obliteration)

`--steering` applies a runtime, reversible edit to the residual stream --
no weight byte in the install is ever touched, and the edit can be switched
on and off between generations in the same process. Both `turbospark-check`
and `turbospark-server` take the identical six flags. This section covers
what you need to actually drive it from the CLI; for the full measurement
record, every refuted hypothesis, and the reasoning behind each default, see
[`docs/OBLITERATION.md`](OBLITERATION.md) -- that page is a working research
log, this one is the quick reference.

### The four modes

Writing `c = d . x` for the raw dot product of the direction with the
residual stream and `c_hat = c / ||d||` for the coefficient along the unit
direction:

| `--steering-mode` | operation | what it's for |
| --- | --- | --- |
| `ablate` (default) | `x -= alpha * c_hat * d_hat` | remove a behavior; equals the classic weight-orthogonalization edit at `alpha=1` |
| `add` | `x += alpha * d` | push toward (positive scale) or away from (negative scale) a concept; the llama.cpp control-vector / ActAdd form |
| `clamp` | `x += (target - c_hat) * d_hat` | pin the coefficient to a fixed value (feature clamping) |
| `renorm` | `ablate`, then rescale the row back to its original norm | `ablate` without the norm collapse that causes full-strength ablation to destroy fluency |

### The six flags

| Flag | Takes | Default | Notes |
| --- | --- | --- | --- |
| `--steering` | path to a `.gguf` control vector (llama.cpp layout); repeatable | none (off) | every flag below is a parse error without this one |
| `--steering-mode` | `ablate\|add\|clamp\|renorm`; repeatable | `ablate`, or whatever the file declares | per vector |
| `--steering-scale` | float; repeatable | `1.0` | `0.0` is the exact identity in every mode; large values on `add`/`clamp` can overflow the FP16 residual stream; per vector |
| `--steering-layers` | `START:END`, inclusive, 0-based; repeatable | every layer the vector covers | per vector |
| `--steering-target` | float | `0.0` | the coefficient `clamp` pins to; shared by every vector |
| `--steering-gate` | float | `0.0` (always fires) | only steer where the direction's own coefficient reaches this magnitude |

### Several vectors at once

`--steering` is repeatable, and each vector carries its own mode, scale and
layer band, paired by position:

```sh
turbospark-check --model ~/models/gptoss-20b.gturbo \
  --messages-file /tmp/p.json \
  --steering /tmp/steer/ocean.gguf  --steering-scale 0.4 --steering-layers 20:30 \
  --steering /tmp/steer/register.gguf --steering-scale 0.8 --steering-layers 40:50
```

Rules, in one place (`invocation::steering_knob` is the definition, and the
server resolves the same command line through the same rule):

- Vectors apply IN the order given. At each steered layer, vector 0's
  dispatch edits the row, then vector 1's dispatch reads that edited row --
  composition is sequential by construction, and each vector's reported
  coefficient is measured against the stream the previous vector left.
- Per-vector usable strengths differ (that is the page's central measured
  fact), which is why alpha and mode are per-vector while `--steering-target`
  and `--steering-gate` stay shared.
- A knob list shorter than the vector list extends by its LAST value
  (`--steering a --steering b --steering-scale 0.4` steers both at 0.4 --
  also exactly what every pre-multi-vector invocation resolves to).
- More knob values than vectors is refused: the extras can never reach a
  vector, so they are a typo, not a configuration.
- A zero-alpha vector is the exact identity, which makes it the plumbing
  probe: N dispatches where one is a no-op must be bit-identical to N-1.

### Choosing how to steer

Three independent choices:

**Which mode.** Use `ablate` to remove a behavior the direction captures.
Use `add` to push generation toward or away from a concept -- flip the sign
of `--steering-scale` to reverse it, without touching the vector file. Use
`clamp` to pin an activation to a specific value rather than push it.
Reach for `renorm` when `ablate` at a workable strength still reads as
degraded rather than merely different -- it buys graceful degradation
through the same edit at no extra cost.

**Which scale and layer band.** Start with a layer band, not every layer:
ablating at `alpha=1` across an entire 64-layer model has been measured to
collapse generation to an immediate end-of-turn token, with no text at all.
There is no formula that predicts a safe alpha from the vector alone -- a
"derived ceiling" computed from activation norms was tried and refuted (it
does not transfer between directions, and the relationship can invert). The
only trustworthy method is the empirical alpha sweep described below, run
once per direction.

**Gate threshold.** Leave `--steering-gate` at its default of `0.0` (always
fires) unless you specifically want the edit to skip tokens where the
direction's own coefficient is small. This is an advanced knob most direction
files never need.

### Getting a direction vector

There is no catalog or download command for these in this repo -- unlike
models, nothing here curates or fetches steering vectors for you. Two paths:

**Extract your own.** Build two directories of contrastive prompts: a
"positive" set representing the concept present, a "negative" set
representing it absent (or its opposite). For each prompt, capture the
residual stream with the engine itself:

```sh
TURBOSPARK_RESID_CAPTURE=/tmp/steer/pos/p1.json \
  turbospark-check --model ~/models/qwen38-27b.gturbo \
  --messages-file /tmp/prompt-pos-1.json --max-new 1 --temperature 0.0001 --top-k 1
```

`--max-new 1` is enough -- the capture keys on the prefill-to-decode
transition, not on how much is generated. Repeat per prompt into `pos/` and
`neg/` directories, then extract:

```sh
uv run --python 3.12 --with numpy scripts/extract_direction.py \
  --positive /tmp/steer/pos --negative /tmp/steer/neg --out /tmp/steer/d.gguf
```

The script prints four diagnostic columns per layer: `norm` (raw magnitude,
not comparable across layers), `sep` (effect size -- **use this one** to pick
which layers to steer), `share` (`||d||/||x||`, a cost estimate) and
`removed` (the fraction `ablate` actually subtracts, which differs from
`share` by up to 180x at the earliest layers). It also prints a derived
alpha ceiling; treat it as a rough starting point only; it does not reliably
predict a safe operating alpha (see above).

**Use a published vector.** Sets like `jukofyork`'s or `repeng`'s ship as the
same llama.cpp-layout control-vector GGUF this engine reads natively:

```sh
hf download jukofyork/creative-writing-control-vectors-v3.0 \
  "Mistral-7B-Instruct-v0.3/mistral-0.3:7b-honesty_vs_machiavellianism__machiavellianism.gguf" \
  --local-dir ~/models/steering-vectors
```

`~/models/steering-vectors/` is a convention, not a path this repo enforces
or manages -- pick any local directory. The one hard requirement: **the
vector must be built for the checkpoint you point it at, not merely a
checkpoint with the same `hidden_size`.** Loading only checks tensor shape,
never the base model or the concept, so a shape-compatible vector for the
wrong model loads and silently steers something. Sanity-check any file
offline before spending a generation on it:

```sh
TURBOSPARK_FOREIGN_CONTROL_VECTOR=~/models/steering-vectors/your-file.gguf \
  cargo test -p turbospark-repack --test control_vector_file -- --ignored --nocapture
```

This reports the covered/spanned layer count and metadata with no model
load and no GPU.

### Tuning: the alpha sweep

Never guess a working alpha; sweep it. This runs entirely against the
UNsteered model to score a steered generation, so it needs no second
checkpoint:

```sh
TURBOSPARK_PROBE_INSTALL_DIR=~/models/qwen38-27b.gturbo \
TURBOSPARK_STEERING_VECTOR=/tmp/steer/d.gguf \
  cargo test -p turbospark-bench --test steering_sweep --release -- --ignored --nocapture
```

`TURBOSPARK_STEERING_ALPHAS` overrides the strengths tried (must start at
`0`); `TURBOSPARK_STEERING_BANDS` sweeps the layer band alongside it (`all`
or `START:END`, same spelling as `--steering-layers`); `TURBOSPARK_STEERING_MODE`
switches the edit under test without rewriting the vector file.

Read the curve, never one row: the score rises both when steering is working
and when it is doing damage, and a collapsed, near-empty output can score as
a *larger* divergence than a working edit -- perplexity and KL alone cannot
tell those apart. Read the `distinct`-token-count column beside the verdict;
a collapse warning fires when the continuation degenerates to one or two
repeated tokens.

### Family coverage

Requesting `--steering` against a family with no wired dispatch is refused at
open, by name, rather than silently accepted and ignored.

| Coverage | Families |
| --- | --- |
| Wired | the `qwen` flow (dense and MoE), `families/llama/` (Mixtral, `qwen3moe`, dense Mistral/Llama), `families/gemma4/`, `gpt-oss`, `museGlimmer` |
| Refused | `DeepSeek-V4-Flash` |

### Swift bindings and C ABI

Steering is available in the C ABI (`ts_session_open` / `ts_session_info_json`)
and the Swift package (`OpenOptions.steering`, `SessionInfo.steering`). An app
embedding the engine through `swift/TurboSpark` can configure directional
steering at session open.

Image catalog/install bindings are separate from text catalog rows. The C ABI
exposes `ts_image_catalog_json` and `ts_image_install` for the pinned image
rows and staged install progress. Swift wraps these as
`TurboSparkCatalog.imageAvailable()` and `installImage(_:)`; image generation
itself remains the `TurboSparkImageSession` surface. See
[`docs/IMAGE_GENERATION.md`](IMAGE_GENERATION.md) for the runtime and real-model
gate boundaries.

For everything measured about these edits -- throughput cost, the collapse
mechanism, cross-family and cross-direction replication, and what is still
open -- see [`docs/OBLITERATION.md`](OBLITERATION.md).

## Environment variables

For the comprehensive reference of all environment variables across CLI, server, Swift app,
and test harnesses, see [`docs/ENV.md`](ENV.md).

All runtime environment variables use the `TURBOSPARK_*` prefix.

| Variable | Affected binaries | Purpose | Default |
| --- | --- | --- | --- |
| `TURBOSPARK_HOME` | `turbospark-check`, `turbospark-model`, `turbospark-server` | Base directory for the local model store and catalog | `~/.turbospark` |
| `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` | `turbospark-model` | Authentication token for gated Hugging Face repositories | none |
| `TURBOSPARK_PHASES` | `turbospark-check` | Set to `1` to print forward-pass phase timing breakdowns on stderr | unset |
| `TURBOSPARK_DISPATCH_PROFILE` | `turbospark-check`, `turbospark-server`, `turbospark-bench` | Set to `1` to collect and print per-dispatch GPU kernel timing and ranking profile (see [`docs/DECODE_BUDGET.md`](DECODE_BUDGET.md)) | unset |
| `TURBOSPARK_PREFIX_REUSE` | `turbospark-check --chat` | Set to `quiet` to silence the per-turn `[prefix-reuse] N/M` line on stderr. It reports how many prompt tokens continued from the previous turn's KV instead of being re-prefilled; the match rate is a property of the checkpoint's template and tokenizer, and a reuse that never fires differs from a working one only in wall-clock. `--quiet` silences it too | unset |
| `TURBOSPARK_PILOT_PROBE` | `turbospark-check` | `1` records the one-layer-ahead router prediction beside the actual selection in the `TURBOSPARK_ROUTER_HIST` capture (Gemma only; analysed by `scripts/pilot_ceiling.py`). `self` aims the probe at its own layer instead, where recall MUST read 100% -- the guard that says the instrument works before any finding is believed | unset |
| `TURBOSPARK_CHAT_DATE` | all (chat template rendering) | Override current date/time (`YYYY-MM-DD` or `YYYY-MM-DDTHH:MM:SS`) in chat templates (e.g. gpt-oss Harmony preamble) | current system UTC time |
| `TURBOSPARK_PREFILL_CHUNK` | `turbospark-check` | Override prompt-processing chunk size (e.g. `128`, `256`, `512`, `1024`) | unset |
| `TURBOSPARK_RESID_CAPTURE` | `turbospark-check` | File path to dump residual stream activations (JSON) at prefill-to-decode transition | unset |
| `TURBOSPARK_SPEC_STATS` | `turbospark-check`, `turbospark-server`, `turbospark-bench` | Set to `1` to log speculative decoding acceptance rate per block position and rollback counts to stderr | unset |
| `TURBOSPARK_READ_QOS` | all (streaming reads) | Set to `utility` to drop background routed-expert streaming read QoS on macOS from user-initiated to utility | unset |
| `TURBOSPARK_SHARED_CB` | all (forward pass) | Set to `0` to disable command buffer overlap | `1` (enabled) |
| `TURBOSPARK_ROUTED_PIPELINE` | all (MoE dispatch) | Set to `0` to disable routed expert pipeline execution | `1` (enabled) |
| `TURBOSPARK_ROUTED_BATCH` | all (MoE prefill) | Set to `1` to enable experimental routed batch prefill | `0` (off) |
| `TURBOSPARK_BATCHED_GEMV` | all (MoE prefill) | Set to `1` to enable experimental batched GEMV prefill | `0` (off) |
| `TURBOSPARK_ROUTER_HIST` | all (MoE runtime) | File path to dump expert routing frequency histogram (JSON) at exit | unset |
| `TURBOSPARK_ROUTER_TRACE` | all (MoE runtime) | Set to collect per-layer routed expert activation trace | unset |
| `TURBOSPARK_FFN_HIST` | all (dense runtime) | File path to dump dense FFN neuron activation frequency histogram (JSON) at exit | unset |
| `TURBOSPARK_MTP_DUMP` | all (MTP drafter) | Directory path to dump MTP intermediate hidden states | unset |
| `TURBOSPARK_MTP_DRAFT` | `turbospark-check`, `turbospark-server`, `turbospark-bench` | Draft block depth (positive integer) or policy (`0` to disable, unset for `auto`) | unset (`auto`) |
| `TURBOSPARK_DFLASH_DRAFT` | `turbospark-check`, `turbospark-server`, `turbospark-bench` | DFlash2 draft block depth (positive integer) or policy (`0` to disable, unset for `auto`) | unset (`auto`) |

### Test oracle and benchmark environment variables

The integration tests and benchmark oracle suites (`turbospark-bench`, `turbospark-repack`) recognize dedicated environment variables to point at local `.gturbo` model directories or control vectors:

- Model install directories: `TURBOSPARK_GEMMA4_INSTALL_DIR`, `TURBOSPARK_QWEN36_INSTALL_DIR`, `TURBOSPARK_QWEN3MOE_INSTALL_DIR`, `TURBOSPARK_MISTRAL_INSTALL_DIR`, `TURBOSPARK_GPTOSS_INSTALL_DIR`, `TURBOSPARK_MUSEGLIMMER_INSTALL_DIR`, `TURBOSPARK_QWEN38_DFLASH2_INSTALL_DIR`, `TURBOSPARK_MTP_INSTALL_DIR`, `TURBOSPARK_ORNITH35B_INSTALL_DIR`, `TURBOSPARK_ORNITH9B_INSTALL_DIR`, `TURBOSPARK_QWEN35_INSTALL_DIR`, `TURBOSPARK_TERNARY_INSTALL_DIR`, `TURBOSPARK_IQ3_INSTALL_DIR`.
- Logit dump and KLD comparison: `TURBOSPARK_LOGIT_DUMP_DIR`, `TURBOSPARK_LOGIT_DUMP_COLD`.
- Steering sweep suite: `TURBOSPARK_PROBE_INSTALL_DIR`, `TURBOSPARK_STEERING_VECTOR`, `TURBOSPARK_STEERING_ALPHAS`, `TURBOSPARK_STEERING_BANDS`, `TURBOSPARK_STEERING_MODE`, `TURBOSPARK_CONTROL_VECTOR`, `TURBOSPARK_FOREIGN_CONTROL_VECTOR`.
