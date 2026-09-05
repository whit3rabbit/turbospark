---
title: turbospark-bench
description: The turbospark-bench benchmark harness usage lines, flags and validation errors, captured live.
---

<!-- generated: cli-help lane, signal: crates/bench/src/main.rs -->

# turbospark-bench

`turbospark-bench` is the throughput benchmark harness. It has no `--help`
flag: with no arguments it prints one usage line to stderr and exits 2. Both
of its usage lines below were captured live from the release binary.

```text
usage: turbospark-bench <tokenizer-dir> [--real] | --model <install-dir> [--case <id>]
```

```text
usage: turbospark-bench --model <install-dir> [--case <id>] [--expert-cache-slots N] [--power-profile performance|balanced|efficiency] [--max-tokens-per-sec R]
```

The first line is the bare-invocation usage; the second is printed when
`--model` is given without an install directory, and again after any
unrecognized argument (prefixed `unexpected argument "<arg>"; `).

## Modes

From the binary's own module documentation (`crates/bench/src/main.rs`):

- `<tokenizer-dir>`: the scripted producer mode. Runs the frozen three-prompt
  protocol through a producer that always emits the same fixed token, so the
  numbers measure the port's prefill+decode loop overhead (tokenizer, sampler,
  detokenizer, stop matcher), not inference.
- `<tokenizer-dir> --real`: macOS only. Builds a small synthetic dense
  `.gturbo` install (deterministic INT4 weights) and drives the same prompts
  through `RealForwardRunner`, measuring the real dispatch path.
- `--model <install-dir>`: macOS only. Opens a real `.gturbo` install and runs
  the frozen community-protocol cases, printing split prefill/decode seconds,
  tok/s and peak `phys_footprint` in MiB, one discarded warmup then one
  measured run per case.

## Flags in `--model` mode

Every flag below was exercised against the binary; the validation messages are
verbatim, each exiting 2. The first four also appear in the binary's own
second usage line; `--speculative`, `--shaping` and `--speculative-drafter`
are accepted but not listed in that line.

| Flag | Values | On bad input (verbatim) |
|---|---|---|
| `--case <id>` | one protocol case id | `--case needs a case id` |
| `--expert-cache-slots N` | one of 8, 16, 24, 32, 48, 64, 96, 128 | `--expert-cache-slots needs one of [8, 16, 24, 32, 48, 64, 96, 128]` |
| `--power-profile P` | performance, balanced or efficiency | `--power-profile needs performance, balanced or efficiency` |
| `--max-tokens-per-sec R` | finite number greater than 0 | `--max-tokens-per-sec needs a number greater than 0` |
| `--speculative V` | off, auto, or a block above 0 | `--speculative needs off, auto or a block above 0` |
| `--shaping S` | protocol or greedy | `--shaping needs protocol or greedy` |
| `--speculative-drafter D` | auto, mtp or dflash | `--speculative-drafter needs auto, mtp or dflash` |

Any other flag: `unexpected argument "<arg>"; usage: turbospark-bench --model <install-dir> [--case <id>] [--expert-cache-slots N] [--power-profile performance|balanced|efficiency] [--max-tokens-per-sec R]`

Defaults, from `crates/bench/src/main.rs` (read, not printed by any usage
line): the slot count is `PROTOCOL_EXPERT_CACHE_SLOTS` (16, the pinned
protocol value, not `auto`); the power profile is `performance` unconditionally,
deliberately not defaulted from Low Power Mode the way `turbospark-check` and
`turbospark-server` are, because a measurement tool has to be explicit;
speculation and the drafter default off.

## No --help and no --version

Verified live: `--version` is taken as the tokenizer directory argument and
fails with `failed to load tokenizer: invalid chat messages: No such file or
directory (os error 2)`, exit 1. The bare invocation (no arguments) prints the
first usage line above and exits 2.

## Provenance

- Captured from the prebuilt release binary at
  `target/release/turbospark-bench` on 2026-09-05; the binary was newer than
  `crates/bench/src/main.rs` at capture time.
- The flag semantics beyond the usage line were read from
  `crates/bench/src/main.rs` and each one confirmed against the binary's own
  validation error (all exit 2).
- The three benchmark modes, the frozen protocol and the memory oracles are
  documented in `docs/BENCHMARKING.md` and `crates/bench/CLAUDE.md`.
