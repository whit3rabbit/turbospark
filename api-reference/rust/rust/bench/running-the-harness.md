---
description: "Run turbospark-bench in its three modes, with every flag checked against crates/bench/src/main.rs."
icon: terminal
---

# Running the harness

This page covers the three modes of `turbospark-bench`, the flags each one accepts, and the refusals that stop a run before it starts. Read the [overview](README.md) first for the protocol conditions. A run that breaks those conditions can still print numbers, so check them before you record anything.

All examples assume a release build:

```sh
cargo build --release -p turbospark-bench --bin turbospark-bench
BIN=./target/release/turbospark-bench
```

Replace `~/models/<install>.gturbo` with a real install directory. The harness does not download models.

## Pick a mode

| Mode | Command shape | Use it for | Comparable to published rows |
| --- | --- | --- | --- |
| Scripted (default) | `turbospark-bench <tokenizer-dir>` | Timing the tokenizer, sampler, and detokenizer loop | No |
| Synthetic real | `turbospark-bench <tokenizer-dir> --real` | Timing the GPU dispatch path on a tiny generated install | No |
| Real install | `turbospark-bench --model <install-dir> [options]` | Throughput and peak footprint of a real checkpoint | Yes, under the protocol only |

## Scripted mode

The scripted producer always emits the same token. It measures this port's loop overhead, not inference. Each of three fixed prompts runs once as a discarded warmup, then once measured.

```sh
$BIN ~/models/<install>.gturbo
```

The tokenizer is loaded from `<tokenizer-dir>`, so the directory must contain a tokenizer. The output is a table with one row per prompt, then an `aggregate:` line. Do not compare these tokens per second with a real model.

## Synthetic real mode (macOS)

```sh
$BIN ~/models/<install>.gturbo --real
```

`--real` builds a small synthetic dense install in a temporary directory under the system temp directory, then drives the same three prompts through the real forward runner. It measures the dispatch path. The synthetic model is too small to say anything about a real checkpoint. The source comment in `crates/bench/src/main.rs` says the binary is macOS only.

## Real install mode (macOS)

This is the only mode that produces the Swift-comparison numbers. It runs the frozen `real-generation-v1` cases with one discarded warmup and one measured run per case. On other platforms it prints `--model requires macOS (Metal)` and exits with status 2.

```sh
$BIN --model ~/models/<install>.gturbo
```

Without `--case`, all three cases run in one process, and one peak sampler spans the whole session. The memory oracle needs that. With `--case`, one case runs per process. This is the protocol's fresh-process leg. A cross-engine comparison has to match it, because the Swift CLI launches once per case.

```sh
$BIN --model ~/models/<install>.gturbo --case short-explanation
```

Valid case ids are `short-explanation`, `medium-review`, and `long-synthesis`. An unknown id prints the valid list and exits with status 2.

### Flags in `--model` mode

Every flag below is checked against the argument parser in `crates/bench/src/main.rs`. Anything unrecognized prints usage and exits with status 2.

| Flag | Values | Default | Notes |
| --- | --- | --- | --- |
| `--case` | a case id | all three | One case per process when set. |
| `--expert-cache-slots` | a value in the allowed set | `16` (protocol value) | Must match the arm you compare against. The allowed set comes from `foundation::runtime_config::ALLOWED_CACHE_SLOTS`. |
| `--prefill-chunk` | `off`, `auto`, or an allowed chunk size | `off` | Off is the sequential path. Every frozen row was measured sequentially, so do not turn this on for a protocol row. |
| `--power-profile` | `performance`, `balanced`, `efficiency` | `performance` | Deliberately not read from Low Power Mode. A power A/B has to name its arm explicitly. |
| `--max-tokens-per-sec` | a number above 0 | none | Overrides the profile's rate cap. |
| `--speculative` | `off`, `auto`, or a block size above 0 | `off` | `auto` picks the drafter's default block. |
| `--speculative-drafter` | `auto`, `mtp`, `dflash` | `auto` | `auto` reads the install's resident index. An explicit value is treated as a promise. The Qwen DFlash2 state refuses an install with no DFlash2 drafter (`crates/runtime/src/families/qwen/dflash_state.rs`). This page did not trace the other families, so check the header line for `drafter=` before you record a row. |
| `--shaping` | `protocol`, `greedy` | `protocol` | `protocol` is the sampled setting the frozen rows use. `greedy` is a different arm. |
| `--kv-bits` | `off`, `2`, `3`, `3.5`, `4` | `off` | Off keeps FP16 KV, which every frozen row used. |

The context window and generation budget are not flags. The harness resolves them per family from the install's own manifest. The header prints them.

### Refusals

The harness exits with status 2 before it opens the model in these cases:

- `--speculative` and `--prefill-chunk` together. The speculative loop has no chunked variant, so a run with both would measure one path under the other's name.
- `--speculative` with `--shaping protocol`. Speculation is exact only at temperature 0. The frozen protocol samples, so the run must pass `--shaping greedy`.
- `TURBOSPARK_PREFILL_CHUNK` set in the environment. The `--model` mode does not read it. Pass `--prefill-chunk` instead, so no arm is inherited silently.

One refusal happens after the open, because the family is known only from the loaded runner. `--prefill-chunk` on an install whose family has no chunked driver exits with status 2. It does not fall back to the sequential loop.

## Example: one protocol case with stderr split out

Long runs should send stdout and stderr to separate files. The stop footer is on stderr.

```sh
$BIN --model ~/models/<install>.gturbo --case medium-review \
  > run-medium.out 2> run-medium.err
grep -h '^\[stop=' run-medium.err
```

The `[stop=...]` footer has the same shape as the Swift CLI footer, so the same grep works for both engines. Reading the table and the header is covered in [Reading the output](reading-the-output.md).

## Example: an A/B arm

Name every arm on the command line. The header prints the resolved values, so a reader can check the arm after the fact.

```sh
$BIN --model ~/models/<install>.gturbo --expert-cache-slots 16 --power-profile performance
$BIN --model ~/models/<install>.gturbo --expert-cache-slots 16 --power-profile efficiency
```

Run the arms interleaved, not one after the other, and discard the warmup each time. `scripts/power.sh` automates the power arms and validates them before it asks for `sudo`. Read its header before you use it. The procedure is in `.claude/docs/power-measurement.md`.

## Environment seams

The runtime reads some `TURBOSPARK_*` variables inside the forward pass. The `--model` header echoes the values of `TURBOSPARK_ROUTED_BATCH` and `TURBOSPARK_BATCHED_GEMV` on every run. Two of them only take effect on the chunked prefill path, so the header prints them as inert on the sequential arm. Check the header before you trust an arm label.

The `TURBOSPARK_METAL_*` variables affect startup and compilation. The startup probe echoes them, and [Startup probe](startup-probe.md) lists them. Expert prefetch is a probe flag, not an environment variable.

## Limits

- Only real-install `--model` rows can be compared with published throughput rows, and only under the protocol. Scripted and synthetic output is a diagnostic.
- A greedy row is not comparable with a sampled row. The header says so in its shaping line.
- A speculative row is greedy and is not comparable with the non-speculative rows. The header says so too.
- The peak column in the table is the process peak so far, not a per-case value. The sampler is shared across cases in one process. See [Reading the output](reading-the-output.md).
- Most integration tests under `crates/bench/tests/` are ignored by default and need model-specific environment variables. `mference_bench.rs` is not ignored: it runs the scripted mode offline. The other names and gates are in `crates/bench/README.md`. This page does not restate them.
