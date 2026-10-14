---
description: "Every turbospark-cli binary, subcommand, and flag, with defaults, mode rules, and the places where a flag is refused."
icon: list
---

# Commands overview

This page lists each binary's commands and flags in the order a user meets them. Defaults and allowed values come from the source. Where a flag's wording in `--help` and the parser disagree, the parser wins and the page says so.

Run `--help` on any binary for its own usage text. `turbospark-check --help` and the `turbospark` front end print text on stdout and exit `0`. The usage text is not a stable contract, so scripts should not parse it.

## turbospark-check

Text generation, chat, and humanizer rewrites. Parsing is done by [`turbospark-invocation`](../invocation/README.md). The full option table, with its defaults, is in [`options.md`](../invocation/options.md).

```text
turbospark-check --model MODEL <MODE> [OPTIONS]
```

### Mode: pick exactly one

Every run needs exactly one of these. A run with no mode flag is refused. A run with two mode flags is refused, for every pair.

| Flag | Input | Behavior |
| --- | --- | --- |
| `--prompt TEXT` | A string | Sends the text as a single raw completion. No chat template is applied. Instruction-tuned models often produce unrelated text in this mode. |
| `--messages-file PATH` | A JSON array | Renders a conversation through the model's own chat template. See [Messages file](#messages-file) below. |
| `--chat` | Stdin, one line per turn | Starts an interactive REPL. Slash commands are listed under [Chat](#chat). |
| `--humanize-file PATH` | A UTF-8 file, or `-` for stdin | Rewrites the draft with a humanizer model. See [Humanize](#humanize). |

With no mode flag, the run is refused with a missing-mode error. A mode flag cannot be combined with `--prompt`, `--messages-file`, `--chat`, or `--humanize-file` except as the pairs listed in the parser.

### Required

| Flag | Meaning |
| --- | --- |
| `--model PATH or ALIAS` | A `.gturbo` install directory, or a catalog alias such as `gemma4` or `tinyllama`. An existing path always wins over an alias with the same name. The directory must contain a tokenizer. |

### Generation

| Flag | Default | Allowed values |
| --- | --- | --- |
| `--max-new N` | `1024` | Positive integer. |
| `--max-context N or auto` | `auto` | Positive integer, or `auto`. `auto` uses the checkpoint's trained context, capped by what memory holds. |
| `--min-auto-context N` | `0` | Refuses to open when `auto` resolves below `N`. `0` sets no floor. |
| `--temperature X` | `0.2` | Zero or more, finite. |
| `--top-k N` | `64` | `0` to `256`. `0` disables the cutoff. |
| `--top-p X` | `0.95` | Greater than `0` and at most `1`. |
| `--repetition-penalty X` | `1.0` | Greater than `0`. |
| `--seed N` | unset | Non-negative integer. |
| `--stop TEXT` | none | Repeatable. Stop strings accumulate in the order given. |
| `--reasoning LEVEL` | `off` | `off`, `low`, `medium`, `high`, `xhigh`. The checkpoint decides which levels it accepts. |
| `--system TEXT` | none | Repeatable. Joined with newlines. Chat mode only. Any other mode refuses it. |
| `--quiet` | off | Suppresses the statistics footer. |

### Memory, cache, and speed

| Flag | Default | Allowed values |
| --- | --- | --- |
| `--load-guard TIER or BYTES` | `relaxed` | `off`, `relaxed`, `balanced`, `strict`, or a byte ceiling. Ranking in `turbospark-model recommend` must use the same tier the session opens with. |
| `--rdadvise MODE` | `off` | `off`, `normal`, `aggressive`. |
| `--expert-cache-slots N or auto` | `auto` | `8`, `16`, `24`, `32`, `48`, `64`, `96`, `128`, or `auto`. |
| `--expert-residency MODE` | `auto` | `auto`, `streamed`, `mapped`. |
| `--prefill-chunk N or auto` | `128` | Chunk sizes the runtime accepts, or `auto`. |
| `--power-profile PROFILE` | `performance`, or `efficiency` under Low Power Mode | `performance`, `balanced`, `efficiency`. |
| `--max-tokens-per-sec X` | uncapped | Greater than `0`. |
| `--kv-bits BITS` | `off` | `off`, `2`, `3`, `3.5`, `4`. An unsupported head size or family refuses the flag at open. It does not fall back to FP16. |
| `--speculative MODE` | `auto` | `auto`, `off`, or a block size from `1` to `15`. A named block size fails if the model cannot serve it. |
| `--speculative-drafter NAME` | `auto` | `auto`, `mtp`, `dflash`. `auto` enables an MTP head when present. It only reports a DFlash head, so name `dflash` to run it. |

### Steering

Steering edits the model's activations with a control vector. Each `--steering` path is one vector. The other steering flags pair with paths by position: the first `--steering-mode` applies to the first `--steering` path, and so on.

| Flag | Default | Notes |
| --- | --- | --- |
| `--steering PATH` | none | Path to a `.gguf` control vector in llama.cpp layout. Repeatable. |
| `--steering-mode MODE` | per vector file | `ablate`, `add`, `clamp`, `renorm`. Repeatable. |
| `--steering-scale X` | `1.0` | Finite. `0.0` is the identity. Large values on `add` or `clamp` can overflow FP16. Repeatable. |
| `--steering-layers START:END` | every covered layer | Inclusive, 0-based. An inverted range is refused. Repeatable. |
| `--steering-target X` | `0.0` | Used by `clamp` only. |
| `--steering-gate X` | `0.0` | Steer only where the coefficient reaches this magnitude. `0.0` means always. |

The parser refuses these combinations:

- A steering parameter without any `--steering` path. The run would decode unsteered.
- More `--steering-mode`, `--steering-scale`, or `--steering-layers` values than `--steering` paths. The extras can never apply.

A shorter list is allowed. The last value extends to the remaining paths.

### Vision

| Flag | Notes |
| --- | --- |
| `--image PATH` | Repeatable. All images go into one turn. Use it with `--messages-file`, and only when that file has no image parts. `--prompt` refuses images at run time. In `--chat`, the source accepts the flag but the chat loop never passes images to the model. Do not rely on `--image` with `--chat`. |
| `--image-batch` | Runs the conversation once per `--image` instead of once with all of them. |
| `--vision-sidecar PATH or auto` | Attaches a standalone vision tower to a text-only trunk. `auto` looks up a matching install in the store. |

### Chat

`--chat` prints a prompt on stderr and reads one line at a time from stdin. Replies go to stdout. Slash commands:

- `/clear` clears history.
- `/history` prints the history on stderr.
- `/quit` or `/exit` leaves the loop. Ctrl-D (end of input) also leaves it.

Any other `/` input prints an unknown-command note on stderr.

### Messages file

`--messages-file` reads a JSON array. Each element needs a `role` and a `content`:

```json
[
  {"role": "system", "content": "Answer in one sentence."},
  {"role": "user", "content": "Why is the sky blue?"}
]
```

Accepted roles are `system`, `developer`, `user`, `assistant`, and `tool`. An unknown role is an error.

`content` is either a string or a list of parts. The parts are `{"type": "text", "text": "..."}` and `{"type": "image", "path": "..."}`. An unknown part type is an error, not a skip. Image paths are matched to the image markers in order.

### Humanize

`--humanize-file` rewrites a draft with a humanizer model. The model's own sampling recipe is forced, so `--temperature`, `--top-k`, `--top-p`, and `--repetition-penalty` do not reach this mode. `--max-new` and `--seed` do. The rewrite goes to stdout. The audit notes go to stderr.

Use an alias from the catalog, such as `humanizer-12b`, or a path.

### Output and limits

- Stdout receives the resolved request on every validated run, then the generated text. See [Exit status and output](exit-status-and-output.md).
- Stats go to stderr and are hidden by `--quiet`.
- Setting `TURBOSPARK_PHASES=1` or `TURBOSPARK_DISPATCH_PROFILE=1` adds phase or dispatch detail to stderr. These are environment variables read directly by the binary, not flags.
- Generation runs only on macOS. Elsewhere the run stops after printing the resolved request.

## turbospark-model

Model catalog, Hugging Face probes, installs, and the Hugging Face token. The binary does not use `turbospark-invocation`. Its own parser rejects any option that a command does not read, so `--force` on `list` is an error, not a no-op.

```text
turbospark-model <COMMAND> [ARGS] [OPTIONS]
```

### Commands

| Command | Arguments | What it does |
| --- | --- | --- |
| `list` | `--filter TEXT` | Lists curated rows. Installed rows are marked with `*`. |
| `list-speech` | none | Lists the bundled speech (whisper) entries. |
| `list-audio` | none | Lists the pinned MiniMax Music 3 profiles. This is the only music-listing command. |
| `info` | `ALIAS` | Prints one row in full. |
| `probe` | `REPO[@REV]` | Reads the repo headers and reports whether this engine would run it. Does not download weights. |
| `recommend` | none | Ranks catalog rows against this machine's memory. |
| `pull` | `ALIAS`, or `--repo REPO[@REV] --alias NAME` | Installs a curated row, or any repo the probe accepts. |
| `pull-image` | see below | Installs an image model. |
| `pull-audio` | `ALIAS` | Installs one pinned MiniMax Music 3 profile. |
| `pull-vision` | `ALIAS`, or `--repo REPO[@REV] --alias NAME` | Installs a vision-tower sidecar. |
| `path` | `ALIAS` | Prints the install directory. Fails if the alias is not installed. |
| `rm` | `ALIAS` | Deletes an install after confirmation. |
| `auth` | `--status`, `--set`, or `--clear` | Checks, saves, or clears the Hugging Face token. |

`pull-image` has three forms:

- `--source ROOT --model-id ID --model-revision REV --alias NAME` packs a local Diffusers export.
- `--repo REPO@REV --alias NAME` downloads and installs a pinned Hugging Face export.
- `--alias z-image-turbo` installs the pinned catalog source.

`--model-revision` must be an immutable 40-character revision.

### Options

| Option | Used by | Meaning |
| --- | --- | --- |
| `--out DIR` | `pull`, `pull-image`, `pull-vision` | Install here instead of the default store. |
| `--file NAME.gguf` | `probe`, `pull`, `pull-vision` | Pick one file when a repo offers several. |
| `--sidecar-repo REPO[@REV]` | `probe`, `pull` | Take tokenizer files from another repo. A GGUF carries llama.cpp's tokenizer, so GGUF pulls usually need this. |
| `--reuse-trunk-from ALIAS` | `pull` | For a row with an MTP source, reuse an installed trunk instead of downloading it again. Refused unless that install came from the same repo and revision. |
| `--alias NAME` | `pull`, `pull-image`, `pull-vision` | Local name for a `--repo` install. Letters, digits, `.`, `_`, and `-` only for `pull-image`. |
| `--context N` | `recommend` | Context window to fit against. Default `4096`. |
| `--budget SIZE` | `recommend` | Overrides the memory probe. A bare number is bytes. Suffixes `k`, `m`, `g` (and their `b` and `ib` forms) are accepted. |
| `--load-guard TIER or SIZE` | `recommend` | Ranks under the same tier the session will open with. |
| `--discover [N]` | `recommend` | Also ranks the N most-downloaded GGUF repos. Default `20`. |
| `--probe` | `recommend` | Reads every curated row's header. Slower. |
| `--force` | `pull`, `pull-vision` | Installs past a probe refusal. |
| `--yes` or `-y` | `rm` | Skips the typed confirmation. |
| `--set` | `auth` | Reads a token from stdin and saves it after validation. |
| `--clear` | `auth` | Removes the saved token. |
| `--status` | `auth` | Verifies the token and shows status. |

`--help` and `-h` print the usage text on stdout and exit `0`. `--version` and `-V` print the version.

### Environment

- `TURBOSPARK_HOME` sets the store root. The default is `~/.turbospark`.
- `HF_TOKEN` supplies the token for gated repositories.
- `TURBOSPARK_DISABLE_DOWNLOAD_CACHE=1` reduces peak disk use for pinned pulls. An interrupted download then fetches byte ranges again.

### Limits

- An install streams the whole checkpoint and cannot resume. A failed pull starts again from the beginning.
- `list` shows aliases only. Use `probe` for anything not in the table.
- Some older examples use `list-music` and `minimax-music-0.3`. Neither name exists in the source. The commands are `list-audio` and `pull-audio`, and the pinned aliases include `minimax-music3-4bit` and `minimax-music3-mxfp8`. Run `list-audio` for the current set.
- `pull` prints a sample `turbospark-check` command that reads `/tmp/p.json`. The source does not create that file.

## turbospark-image

Diffusion image generation (Z-Image-Turbo) and packing of Diffusers exports. This binary's errors all exit `1`, including bad flags. See [Exit status and output](exit-status-and-output.md).

### generate

```text
turbospark-image generate --model PATH_OR_ALIAS --prompt TEXT --output PNG [OPTIONS]
```

| Flag | Default | Notes |
| --- | --- | --- |
| `--model` | required | An install directory, or a local-store alias such as `z-image-turbo`. |
| `--prompt` | required | The prompt text. |
| `--output` | required | A new PNG path. Existing files are refused, never overwritten. |
| `--seed N` | random | Explicit seed. The chosen seed is not printed. Pass one to repeat a result. |
| `--width PIXELS` | `1024` | `512` to `1024`, multiple of `16`. |
| `--height PIXELS` | `1024` | `512` to `1024`, multiple of `16`. |
| `--steps 9` | `9` | Only `9` is supported. |
| `--backend native or reference` | `native` | `native` uses Metal. `reference` is a CPU path for diagnostics and supports only `1024x1024`. |

On success the output path is printed on stdout.

### pack

```text
turbospark-image pack --source ROOT --output INSTALL --model-id ID --model-revision REV
```

`ROOT` must contain `tokenizer/`, `text_encoder/`, `transformer/`, `scheduler/`, and `vae/`. `INSTALL` is a new directory and is never overwritten. The output is an install directory. The install format is covered in [Install format and components](../image/install-and-components.md).

## turbospark-music

Music generation with MiniMax Music 3. Errors exit `2` for usage and `1` for failed runs.

```text
turbospark-music generate --model ALIAS_OR_DIR --caption TEXT \
    (--lyrics TEXT | --lyrics-file PATH) --output WAV \
    [--duration SECONDS] [--steps N] [--seed N] \
    [--precision checkpoint|float32] [--wav-format pcm16|float32] [--timings] \
    [--flow-uncond-interval N] [--flow-uncond-warmup N]
```

| Flag | Default | Notes |
| --- | --- | --- |
| `--model` | required | A pinned profile installed with `turbospark-model pull-audio`, or a local converted model directory. |
| `--caption` | required | Style description. |
| `--lyrics TEXT` or `--lyrics-file PATH` | one is required | Give exactly one. Use `[instrumental]` for instrumental output. |
| `--output` | required | WAV path. An existing file is overwritten without a prompt. |
| `--duration SECONDS` | library default | Decimal number. The default is set in `turbospark-audio`, not in this binary. |
| `--steps N` | library default | Diffusion steps. The default is set in `turbospark-audio`. |
| `--seed N` | unset | Integer. |
| `--precision` | `checkpoint` | `checkpoint` or `float32`. |
| `--wav-format` | `pcm16` | `pcm16` rounds to 16 bits. `float32` writes samples unrounded. |
| `--timings` | off | Prints per-stage timing and the real-time factor on stderr. Give it at most once. |
| `--flow-uncond-interval N` | `1` | `1` is the reference trajectory. Larger values change the output. |
| `--flow-uncond-warmup N` | `2` | Steps before the interval applies. |

Repeating a value flag is an error. On success the binary prints the sample count, sample rate, and output path on stdout.

## turbospark-clone

Zero-shot voice cloning and English speech synthesis with MegaTTS3. Usage errors exit `2`. Failed runs exit `1`.

```text
turbospark-clone speak --model MODEL --reference WAV (--text TEXT | --text-file PATH) \
    --output WAV --consent [--quality fast|balanced|best | --steps N] \
    [--p-w X] [--t-w X] [--seed N] [--wav-format pcm16|float32] [--metal | --cpu]
```

`speak` is the only subcommand.

| Flag | Default | Notes |
| --- | --- | --- |
| `--model` | required | An installed audio alias, or a converted MegaTTS3 directory with `aligner_lm/`, `diffusion_transformer/`, `duration_lm/`, `wavvae/`, and `g2p/`. |
| `--reference WAV` | required | Clean recording of the target voice. At most the first 28 seconds are used. |
| `--text TEXT` or `--text-file PATH` | one is required | English only. Text files are capped at 32768 bytes. |
| `--output WAV` | required | Overwritten without a prompt. |
| `--consent` | required | Confirms you have the right to clone the voice. The run is refused without it. |
| `--quality` or `--steps` | `balanced` (16 steps) | `fast` is 8, `balanced` is 16, `best` is 32. `--steps` takes `1` to `100`. The two flags cannot be combined. |
| `--p-w X` | `1.4` | `0.1` to `5`. Higher favors intelligibility. |
| `--t-w X` | `3.0` | `0.1` to `10`. Higher favors similarity to the reference. |
| `--seed N` | `0` | Makes a run repeatable. |
| `--wav-format` | `pcm16` | `pcm16` or `float32`. |
| `--metal` or `--cpu` | `--metal` on macOS, `--cpu` elsewhere | Choose at most one. The CPU path is much slower. |

The source notes that the WavVAE encoder used for cloning has no verified provenance or license. Only clone voices you are allowed to use.

## turbospark front end

The `turbospark` binary dispatches. It parses only the first argument, then passes the rest to the target binary unchanged. Exit codes from the child process are returned as is.

| Form | Runs |
| --- | --- |
| `turbospark run MODEL [PROMPT] [FLAGS]` | `turbospark-check`. A positional prompt becomes `--prompt`. With no prompt and no mode flag, it adds `--chat`. |
| `turbospark humanize MODEL DRAFT [FLAGS]` | `turbospark-check --humanize-file DRAFT`. |
| `turbospark image ...` | `turbospark-image generate ...`. Adds `generate` unless the first argument already is `generate`. This also applies to `pack`, so `turbospark image pack` becomes `generate pack` and fails with an unknown-option error, although the front-end usage text lists it. Call `turbospark-image pack` directly. |
| `turbospark list`, `pull`, `info`, `rm`, `probe`, `recommend`, `path`, `auth` | `turbospark-model` with the same arguments. |
| `turbospark serve [OPTIONS]` | `turbospark-server` in the foreground. Options belong to that binary. |
| `turbospark bench [OPTIONS]` | `turbospark-bench`. |
| `turbospark start [OPTIONS]` | Starts the background server. Arguments pass through to `turbospark-server`, except `--api-key`, which becomes an environment variable. Prints `already running` and exits `0` if one is up. |
| `turbospark start AGENT [FLAGS] [-- AGENT_ARGS]` | Connects a coding agent to the local server. |
| `turbospark MODEL AGENT [AGENT_ARGS]` | Same as `start AGENT --model MODEL`. |
| `turbospark stop` | Stops the background server. |
| `turbospark restart [OPTIONS]` | Restarts the background server. |
| `turbospark status` | Prints a `Status:` line reading `RUNNING` (followed by PID, port, endpoint, and log path) or `STOPPED`. The `RUNNING` line is padded with extra spaces, so match the word, not the full string `Status: RUNNING`. Exits `0` either way. |
| `turbospark --help`, `-h`, `help` | Usage on stdout, exit `0`. |
| `turbospark --version`, `-V`, `version` | `turbospark VERSION`, exit `0`. |

### Locating the child binaries

The front end first looks for the target next to its own executable. If it finds nothing there, it runs the bare name and lets the shell search `PATH`. `serve` and `restart` need `turbospark-server`, and `bench` needs `turbospark-bench`. This package does not build either one.

### Agent connections

Known agents: `claude`, `codex`, `opencode`, `grok`, `gemini`, `hermes`, `openclaw`, `dsh`.

| Flag | Meaning |
| --- | --- |
| `--model INSTALL-OR-ALIAS` | Required when no server is running. Restarts a running server that serves a different model. |
| `--port N` | Port for a newly started server. Default `8080`. |
| `--dry-run` | Prints the launch plan and changes nothing. |
| `-- AGENT_ARGS` | Everything after `--` goes to the agent unchanged. |

When the front end starts the server, `--api-key` is removed from the server's arguments and passed through the `TURBOSPARK_API_KEY` environment variable, so it does not appear in the server's argument list. `status` reports the endpoint as `127.0.0.1` on the chosen port.
