---
title: turbospark-check
description: Every flag of the turbospark-check binary, captured verbatim from --help.
---

<!-- generated: cli-help lane, signal: crates/cli/src/main.rs -->

# turbospark-check

`turbospark-check` runs generation once against a model install: a raw prompt
(`--prompt`), a rendered chat conversation (`--messages-file`), or an
interactive REPL (`--chat`). `--model` is required; the three mode flags are
mutually mode-selecting.

Captured verbatim from `turbospark-check --help` (exit 0, no model needed):

```text
usage:
  --model  path to the model (required)
  --prompt  single-turn completion prompt (mode-selecting)
  --messages-file  conversation-file path (mode-selecting)
  --chat  interactive chat mode (mode-selecting)
  --system  leading message, repeatable, chat mode only (default: none)
  --max-new  generated-token limit, positive integer (default 1024)
  --max-context  context-size limit, positive integer, or auto (default auto: the checkpoint's trained context, capped by what memory holds, and 4096 when the install declares none)
  --load-guard  how much memory may be committed: off, relaxed, balanced, strict, or a byte ceiling on what the engine allocates (default relaxed, which is what shipped before this flag existed)
  --min-auto-context  refuse to open when `--max-context auto` resolves below this many tokens; 0 imposes no floor and does not constrain an explicit --max-context (default 0)
  --temperature  sampling temperature, zero or more (default 0.2)
  --top-k  rank-based candidate count, 0 disables (default 64, allowed 0-256)
  --top-p  cumulative-probability threshold, (0, 1] (default 0.95)
  --repetition-penalty  repetition penalty factor, greater than 0 (default 1.0)
  --seed  determinism seed, non-negative integer (default: unset)
  --stop  stop string, repeatable, accumulates in order (default: none)
  --image  image path, repeatable, all images land in ONE turn (default: none)
  --image-batch  run the prompt once PER --image instead of once with all of them
  --rdadvise  read-ahead mode: off, normal, aggressive (default off)
  --expert-cache-slots  routed-cache slot count, allowed 8/16/24/32/48/64/96/128, or auto (default auto, which never resolves below 16)
  --speculative  speculative decoding: auto, off, or a block size 1-15 (default auto; a named block FAILS if the model cannot serve it)
  --speculative-drafter  drafter --speculative drives: auto, mtp or dflash (default auto, which enables an mtp head but only REPORTS a dflash one -- dflash is 0.88x on prose, so name it to run it)
  --prefill-chunk  prompt-processing chunk size, or auto (default 128)
  --power-profile  power profile: performance, balanced, efficiency (default performance, or efficiency under Low Power Mode)
  --steering  path to a control vector (.gguf, llama.cpp layout) to steer with (default none; see docs/OBLITERATION.md)
  --steering-mode  steering edit: ablate, add, clamp, renorm (default ablate, or whatever the vector file declares)
  --steering-scale  steering strength (default 1.0; 0.0 is the exact identity, and large values on add/clamp can overflow the FP16 residual stream)
  --steering-layers  layer range to steer, START:END inclusive, 0-based (default every layer the vector covers)
  --steering-target  coefficient --steering-mode clamp pins the stream to (default 0.0; ignored by ablate and add)
  --steering-gate  only steer where the direction's coefficient reaches this magnitude (default 0.0, meaning always)
  --reasoning  reasoning effort: off, low, medium, high, xhigh (default off; the accepted set is the checkpoint's, not this one)
  --max-tokens-per-sec  decode rate cap, greater than 0 (default: uncapped, or the efficiency profile's reading speed)
  --quiet  suppress incidental output, plain toggle (default off)
  --help  print usage text and exit
  --version  print the version and exit
```

## Defaults at a glance

Default values exactly as the usage text states them.

| Flag | Default |
|---|---|
| `--system` | none |
| `--max-new` | 1024 |
| `--max-context` | auto (the checkpoint's trained context, capped by what memory holds, and 4096 when the install declares none) |
| `--load-guard` | relaxed |
| `--min-auto-context` | 0 |
| `--temperature` | 0.2 |
| `--top-k` | 64 (allowed 0-256) |
| `--top-p` | 0.95 |
| `--repetition-penalty` | 1.0 |
| `--seed` | unset |
| `--stop` | none |
| `--image` | none |
| `--rdadvise` | off |
| `--expert-cache-slots` | auto (never resolves below 16) |
| `--speculative` | auto |
| `--speculative-drafter` | auto |
| `--prefill-chunk` | 128 |
| `--power-profile` | performance, or efficiency under Low Power Mode |
| `--steering` | none |
| `--steering-mode` | ablate, or whatever the vector file declares |
| `--steering-scale` | 1.0 |
| `--steering-layers` | every layer the vector covers |
| `--steering-target` | 0.0 |
| `--steering-gate` | 0.0 (meaning always) |
| `--reasoning` | off |
| `--max-tokens-per-sec` | uncapped, or the efficiency profile's reading speed |
| `--quiet` | off |

## Provenance

- Captured from the prebuilt release binary at `target/release/turbospark-check`
  on 2026-09-05; the binary was newer than `crates/cli/src/main.rs` and every
  file under `crates/invocation/src/` at capture time.
- `--version` prints `turbospark 0.1.0`.
- The narrative reference for this binary, with the steering walkthrough and
  per-family context, is `docs/CLI.md` in the repository root.
