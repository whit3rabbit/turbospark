---
title: turbospark-server
description: Every flag of the turbospark-server binary, captured verbatim from --help.
---

<!-- generated: cli-help lane, signal: crates/server/src/main.rs -->

# turbospark-server

`turbospark-server` serves an HTTP API from a `.gturbo` install directory or a
`turbospark-model` catalog alias. It takes flags only; there are no
sub-commands. The usage line names the alternative scripted form
(`<tokenizer-dir> [port]`) on its own line.

Captured verbatim from `turbospark-server --help` (exit 0, no model needed):

```text
usage: turbospark-server --model <install-dir|alias> [--port N] [--max-context N|auto] [--load-guard TIER|BYTES] [--min-auto-context N] [--expert-cache-slots auto|N] [--bind loopback|tailnet] [--power-profile performance|balanced|efficiency] [--max-tokens-per-sec R] [--speculative off|auto|N] [--speculative-drafter auto|mtp|dflash] [--guardrails on|off] [--prefix-reuse on|off] [--session-slots N] [--reasoning off|low|medium|high|xhigh] [--system TEXT] [--system-file PATH] [--api-key KEY] [--steering PATH] [--steering-mode ablate|add|clamp|renorm] [--steering-scale F] [--steering-layers S:E] [--steering-target F] [--steering-gate F]
       turbospark-server <tokenizer-dir> [port]
       turbospark-server --help | --version

options:
  --model              a .gturbo directory or a turbospark-model alias (`turbospark-model list`)
  --port               listen port (default 8080)
  --max-context        context window in tokens, or auto (default auto: the
                       checkpoint's trained context, capped by what memory
                       holds, and 4096 when the install declares none)
  --load-guard         how much of the machine a session may commit: off,
                       relaxed (default), balanced, strict, or a byte ceiling on
                       what the engine ALLOCATES. relaxed is what shipped before
                       this flag and what every published memory figure was
                       measured under; see docs/LOAD_GUARD.md
  --min-auto-context   refuse to open when --max-context auto resolves below this
                       many tokens (default 0, no floor). Says nothing about an
                       explicit --max-context
  --expert-cache-slots routed-cache slots per layer: auto or 8/16/24/32/48/64/96/128 (default auto)
  --bind               loopback or tailnet (default loopback; tailnet requires --api-key or $TURBOSPARK_API_KEY)
  --power-profile      performance, balanced or efficiency
  --max-tokens-per-sec decode rate cap, greater than 0
  --speculative        off, auto, or a block size 1-15 (default auto). Speculation
                       applies to temperature-0 requests only; others decode
                       sequentially
  --speculative-drafter auto, mtp or dflash (default auto; auto reports a DFlash2
                       drafter but does not enable it -- see docs/DFLASH2.md)
  --guardrails         on or off (default on). Rescues a tool call the decoder
                       could not parse, checks arguments against the request's
                       own schema, and re-asks once. A request carrying TOOLS is
                       buffered rather than streamed while this is on, because a
                       verdict needs the whole turn; requests without tools are
                       unaffected
  --prefix-reuse       on or off (default off). When explicitly enabled, a request
                       continues from the previous request's KV cache wherever
                       the prompts agree, instead of re-prefilling the whole
                       transcript. Enable only when every request belongs to one
                       trusted client: the shared cache is not partitioned by API
                       key or client, so reuse can expose prefix matches through
                       response timing. It also raises the
                       idle-memory floor between requests, not the peak, since
                       pages that would normally be released stay resident.
                       See crates/runtime/CLAUDE.md Gotcha 30
  --session-slots      how many DISTINCT conversations this runner may keep
                       reusable KV/recurrent state for at once (default 1, i.e.
                       no pool). Real committed memory per extra slot, unlike
                       --prefix-reuse's floor-only cost; needs --prefix-reuse on,
                       since a parked session is never reused
                       without it. See crates/server/CLAUDE.md's --session-slots
                       Gotcha
  --reasoning          default reasoning effort for requests that do not specify
                       reasoning_effort: off, low, medium, high or xhigh
                       (default off)
  --system             default system prompt for requests that carry no system
                       or developer message of their own. Repeatable; repeats
                       join with a newline. A request that sends its own system
                       message is left exactly as it arrived
  --system-file        read the same default system prompt from a file, for a
                       prompt too long to sit on a command line. Mutually
                       exclusive with --system
  --api-key            require this key on every request except GET /health,
                       as `Authorization: Bearer <key>` or `x-api-key: <key>`.
                       Falls back to $TURBOSPARK_API_KEY when absent (keeps
                       the key out of `ps`); with neither, the server has no
                       auth at all, same as before this flag existed
  --steering           path to a control vector (.gguf, llama.cpp layout). Applies a
                       directional edit to the residual stream of EVERY request this
                       process serves; no weight byte is modified. See
                       docs/OBLITERATION.md
  --steering-mode      ablate, add, clamp or renorm (default: the vector file's declared mode,
                       or ablate)
  --steering-scale     strength (default 1.0 when --steering is given; 0.0 is the exact
                       identity)
  --steering-layers    START:END, inclusive and 0-based (default every layer the
                       vector covers)
  --steering-target    coefficient --steering-mode clamp pins the stream to (default 0)
  --steering-gate      only steer where the coefficient reaches this magnitude
                       (default 0, meaning always)
  --help               print this text and exit
  --version            print the version and exit
```

## Defaults at a glance

Default values exactly as the options text states them.

| Flag | Default |
|---|---|
| `--port` | 8080 |
| `--max-context` | auto (the checkpoint's trained context, capped by what memory holds, and 4096 when the install declares none) |
| `--load-guard` | relaxed |
| `--min-auto-context` | 0 (no floor) |
| `--expert-cache-slots` | auto |
| `--bind` | loopback (tailnet requires --api-key or $TURBOSPARK_API_KEY) |
| `--speculative` | auto |
| `--speculative-drafter` | auto |
| `--guardrails` | on |
| `--prefix-reuse` | off |
| `--session-slots` | 1 (no pool) |
| `--reasoning` | off |
| `--system` | none (repeatable; repeats join with a newline) |
| `--api-key` | none; falls back to `$TURBOSPARK_API_KEY` when absent |
| `--steering` | none |
| `--steering-mode` | the vector file's declared mode, or ablate |
| `--steering-scale` | 1.0 when `--steering` is given |
| `--steering-layers` | every layer the vector covers |
| `--steering-target` | 0 |
| `--steering-gate` | 0 (meaning always) |

## Provenance

- Captured from the prebuilt release binary at
  `target/release/turbospark-server` on 2026-09-05; the binary was newer than
  `crates/server/src/main.rs` at capture time.
- `--version` prints `turbospark 0.1.0`.
- The routes this server exposes (OpenAI `/v1/chat/completions`, Anthropic
  `/v1/messages`, `/v1/models`) and the tool-call guardrail behavior are
  documented in `crates/server/CLAUDE.md` and `docs/CLI.md`. `--session-slots`
  and `--prefix-reuse` currently have no flag rows in `docs/CLI.md` (drift
  finding, see the [CLI Reference overview](./cli)).
