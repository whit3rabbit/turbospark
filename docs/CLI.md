# CLI reference

Three binaries, three jobs:

- `turbospark-check` -- run generation once against an install: a raw prompt, a
  rendered chat conversation, or an interactive REPL. See
  [`crates/cli/CLAUDE.md`](../crates/cli/CLAUDE.md) for how it is wired.
- `turbospark-model` -- find, inspect, and install models into the
  `~/.turbospark` store. See [`docs/MODELS.md`](MODELS.md) for the catalog
  concept this drives.
- `turbospark-server` -- an OpenAI- and Anthropic-compatible HTTP server over
  one open install. See [`crates/server/CLAUDE.md`](../crates/server/CLAUDE.md).

`--help` on any of the three prints its own flag list and exits; `--version`
prints the workspace version. Both short-circuit before `--model` is read, so
neither needs a real install on disk.

This page documents every flag. It does not repeat the *why* behind a
default or a caveat where a dedicated page already carries that -- follow the
links inline rather than expecting the full story here.

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
| `--max-context` | positive integer, or `auto` | `auto` | context window; `auto` resolves to the checkpoint's own trained context, capped by what memory holds, and `4096` when the install declares neither |
| `--temperature` | float | `0.2` | sampling temperature; `0.0` is greedy |
| `--top-k` | integer, `0`-`256` | `64` | rank-based candidate count, `0` disables |
| `--top-p` | float, `(0, 1]` | `0.95` | cumulative-probability threshold |
| `--repetition-penalty` | float `> 0` | `1.0` | repetition penalty factor |
| `--seed` | non-negative integer | unset | determinism seed |
| `--stop` | text, repeatable | none | extra stop string, in addition to the tokenizer's own stop set |
| `--reasoning` | `off\|low\|medium\|high\|xhigh` | `off` | asks the checkpoint's own chat template to think first; the accepted set is the checkpoint's, not this CLI's -- an unsupported level is refused by name |

### Performance and memory

| Flag | Takes | Default | Meaning |
| --- | --- | --- | --- |
| `--rdadvise` | `off\|normal\|aggressive` | `off` | read-ahead hint mode for streamed expert reads (macOS) |
| `--expert-cache-slots` | `8\|16\|24\|32`, or `auto` | `auto` | routed-expert slot cache size; `auto` never resolves below `16` |
| `--prefill-chunk` | integer, or `auto` | `128` | **parsed and printed, but not wired** -- this flag currently changes nothing; chunked prefill is controlled by the `MFERENCE_PREFILL_CHUNK` environment variable instead |
| `--power-profile` | `performance\|balanced\|efficiency` | `performance` (or `efficiency` under Low Power Mode) | decode rate governance |
| `--max-tokens-per-sec` | float `> 0` | uncapped (or the efficiency profile's reading speed) | hard decode rate cap |

### Speculative decoding

| Flag | Takes | Default | Meaning |
| --- | --- | --- | --- |
| `--speculative` | `off\|auto`, or a block size `1`-`15` | `auto` | a named block size FAILS at open if the install cannot serve it; acceptance is exact only at `--temperature 0` |
| `--speculative-drafter` | `auto\|mtp\|dflash` | `auto` | which drafter `--speculative` drives; `auto` enables an MTP head but only REPORTS a DFlash one (DFlash measures 0.88x on prose -- name it explicitly to actually run it) |

### Steering (obliteration)

Six flags, all requiring `--steering` itself; see the dedicated
[Steering (obliteration)](#steering-obliteration) section below for the full
picture.

| Flag | Takes | Default |
| --- | --- | --- |
| `--steering` | path to a `.gguf` control vector | none (off) |
| `--steering-mode` | `ablate\|add\|clamp\|renorm` | `ablate`, or whatever the file declares |
| `--steering-scale` | float | `1.0` |
| `--steering-layers` | `START:END` | every layer the vector covers |
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

## `turbospark-model`

```sh
turbospark-model <command> [flags...]
```

| Command | Args | Flags | What it does |
| --- | --- | --- | --- |
| `list` | none | `--filter TEXT` | prints every curated catalog row, marks installed ones |
| `info` | `<alias>` | none | prints one catalog row in full, including its gate targets |
| `probe` | `<repo>[@rev]` | `--file NAME.gguf`, `--sidecar-repo REPO[@rev]` | reads a Hugging Face repo's headers only, no download; reports whether this engine would run it |
| `recommend` | none | `--context N`, `--budget BYTES`, `--probe`, `--discover [N]` | ranks models by whether they fit this machine and how much is known about them; `--probe` reads every curated row's header for exact numbers, `--discover` also ranks the N most-downloaded GGUF repos on Hugging Face (default `20`) through the same probe |
| `pull` | `<alias>`, or `--repo REPO[@rev] --alias NAME` | `--out DIR`, `--file NAME.gguf`, `--sidecar-repo REPO[@rev]`, `--force` | installs a curated model, or any repository the probe accepts |
| `path` | `<alias>` | none | prints the install directory (fails loudly if not installed) |
| `rm` | `<alias>` | `--yes` / `-y` | deletes an install; without `--yes`, prompts for the alias name to confirm |

Environment: `TURBOSPARK_HOME` overrides the store root (default
`~/.turbospark`); `HF_TOKEN` is read for gated Hugging Face repositories.
`--help`/`-h` and `--version`/`-V` are global. Flags not accepted by the
given command are rejected rather than silently ignored.

See [`docs/MODELS.md`](MODELS.md) for the catalog itself, and what it takes
to add a row.

## `turbospark-server`

```sh
turbospark-server --model <path-or-alias> [flags...]
turbospark-server <tokenizer-dir> [port]   # legacy scripted mode, see below
```

Serves OpenAI `/v1/chat/completions`, Anthropic `/v1/messages`, and
`/v1/models` from one open install, one request at a time (one runner per
process).

| Flag | Takes | Default | Meaning |
| --- | --- | --- | --- |
| `--model` | path or alias | required | same resolution as `turbospark-check`'s |
| `--port` | u16 | `8080` | listen port |
| `--max-context` | integer, or `auto` | `auto` | same semantics as `turbospark-check`'s |
| `--expert-cache-slots` | `8\|16\|24\|32`, or `auto` | `auto` | same semantics as `turbospark-check`'s |
| `--bind` | `loopback\|tailnet` | `loopback` | `tailnet` binds this machine's Tailscale IPv4 address; there is no authentication and no TLS either way -- the Tailnet ACL is the only access control under `tailnet` |
| `--power-profile` | `performance\|balanced\|efficiency` | unset | same as `turbospark-check`'s |
| `--max-tokens-per-sec` | float `> 0` | uncapped | same as `turbospark-check`'s |
| `--speculative` | `off\|auto`, or `1`-`15` | `auto` | resolved ONCE at process open, not per request; acceptance is exact only at temperature 0, so a request sampled above that falls back to sequential decode silently |
| `--speculative-drafter` | `auto\|mtp\|dflash` | `auto` | same as `turbospark-check`'s |
| `--guardrails` | `on\|off` | `on` | tool-call rescue, argument validation, one retry (see [`docs/FORGE_GUARDRAILS.md`](FORGE_GUARDRAILS.md)); a request carrying `tools` is BUFFERED rather than streamed while this is on, since a verdict needs the whole turn -- a request without tools streams exactly as it always did |
| `--steering` | path to `.gguf` | none | see [Steering (obliteration)](#steering-obliteration) |
| `--steering-mode` | `ablate\|add\|clamp\|renorm` | `ablate`, or the file's own | as above |
| `--steering-scale` | float | `1.0` | as above |
| `--steering-layers` | `START:END` | every layer the vector covers | as above |
| `--steering-target` | float | `0.0` | as above |
| `--steering-gate` | float | `0.0` | as above |
| `--help` / `-h`, `--version` / `-V` | -- | -- | -- |

**Not present on the server:** `--prefill-chunk` (chunked prefill has no
server-side flag at all). `--reasoning` is not a server flag either -- a
reasoning effort is a per-request field on the OpenAI/Anthropic wire schema,
so it is chosen per request rather than pinned for the process, unlike
`--steering` and `--speculative` which are resolved once at open and apply to
every request the process serves for its whole life.

The legacy positional form (`turbospark-server <tokenizer-dir> [port]`) runs
the portable scripted backend against canned completions rather than a real
install, and always binds loopback regardless of `--bind`.

```sh
turbospark-server --model ~/models/gemma4.gturbo
turbospark-server --model gemma4 --bind tailnet --port 8080
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
| `--steering` | path to a `.gguf` control vector (llama.cpp layout) | none (off) | every flag below is a parse error without this one |
| `--steering-mode` | `ablate\|add\|clamp\|renorm` | `ablate`, or whatever the file declares | |
| `--steering-scale` | float | `1.0` | `0.0` is the exact identity in every mode; large values on `add`/`clamp` can overflow the FP16 residual stream |
| `--steering-layers` | `START:END`, inclusive, 0-based | every layer the vector covers | |
| `--steering-target` | float | `0.0` | the coefficient `clamp` pins to; ignored by `ablate`/`add` |
| `--steering-gate` | float | `0.0` (always fires) | only steer where the direction's own coefficient reaches this magnitude |

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
MFERENCE_RESID_CAPTURE=/tmp/steer/pos/p1.json \
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

### Not available today

There is no C ABI or Swift binding surface for steering -- it is
`turbospark-check`/`turbospark-server` only. An app embedding this engine
through `crates/ffi` cannot drive it yet.

For everything measured about these edits -- throughput cost, the collapse
mechanism, cross-family and cross-direction replication, and what is still
open -- see [`docs/OBLITERATION.md`](OBLITERATION.md).
