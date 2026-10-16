---
description: "turbospark-startup-probe: load time, first-token latency, compiler work, and expert prefetch, one JSON line per request, with no discarded warmup."
icon: rocket
---

# Startup probe

`turbospark-startup-probe` measures what a user waits for before a model answers. It times the open, the first request, and any Metal compilation or expert preparation those steps trigger. It writes one JSON object per request to stdout. Use it for startup experiments, such as the opt-in compilation caches and bounded expert prefetch described in `docs/MOE_STARTUP.md`.

It does not follow the throughput protocol. It discards no warmup. The first request after an open is the thing being measured. Do not compare its first-request numbers with the frozen throughput rows in `docs/BENCHMARKS.md`. Read the [overview](README.md) for the rules that apply to every number from this crate.

The binary runs on macOS only. On any other platform it prints an error and exits with status 2.

## Run it

```sh
cargo build --release -p turbospark-bench --bin turbospark-startup-probe
PROBE=./target/release/turbospark-startup-probe

$PROBE --model ~/models/<install>.gturbo --label baseline > probe-baseline.jsonl
```

`--help` or `-h` prints the usage text and exits with status 0. A bad flag prints the error and the usage text to stderr and exits with status 2. A failure during the open or a request prints `startup probe: <error>` and exits with status 1.

## Flags

Every flag below is checked against `crates/bench/src/startup_probe.rs`. Each flag takes a value, except `--help` and `-h`.

| Flag | Default | Notes |
| --- | --- | --- |
| `--model DIR` | required | The install directory. The tokenizer is read from the same directory. |
| `--case ID` | `short-explanation` | One of `short-explanation`, `medium-review`, `long-synthesis`. |
| `--context N` | `4096` | KV window. Must be a positive 32-bit integer. |
| `--max-new N` | `64` | Generation budget. Must be a positive 32-bit integer and less than `--context`. |
| `--slots N` | `16` | Fixed expert slots. Must be in the allowed set from `foundation::runtime_config::ALLOWED_CACHE_SLOTS`. |
| `--residency streamed\|mapped` | `streamed` | Expert residency. Any other value is refused. |
| `--shaping greedy\|sampled` | `greedy` | `greedy` is the default here, unlike the throughput harness. `sampled` uses the case's frozen seed. |
| `--repeats N` | `2` | Requests per runner, from 1 to 100. Requests do not reuse a prefix. |
| `--reopens N` | `1` | Runner opens per process, from 1 to 100. Each open reloads the model. |
| `--expert-trace FILE` | none | A JSON array of `[layer, expert]` pairs. Capped at 1 MiB. |
| `--prefetch-bytes N` | `0` | A whole-expert byte ceiling. Requires `--expert-trace`, and the trace requires it. |
| `--page-cache-label TEXT` | `unknown` | A caller-supplied label for the OS cache state. The output marks it unverified. |
| `--label TEXT` | `unlabeled` | The experiment arm name, copied into every record. |

The parser refuses `--max-new` values that are not less than `--context`. It does not check that the prompt fits. Read the `prompt_tokens` field in the output to confirm the prompt fit.

## Example: two arms, run interleaved

```sh
$PROBE --model ~/models/<install>.gturbo --label arm-a > arm-a-1.jsonl
$PROBE --model ~/models/<install>.gturbo --label arm-b > arm-b-1.jsonl
$PROBE --model ~/models/<install>.gturbo --label arm-a > arm-a-2.jsonl
$PROBE --model ~/models/<install>.gturbo --label arm-b > arm-b-2.jsonl
```

Each invocation is a fresh process. That matches the fresh-process leg of the protocol. Repeat the arms, alternate their order, and keep the raw files. Run on AC power with a quiet host, as the overview describes.

## Example: bounded expert prefetch

```sh
$PROBE --model ~/models/<install>.gturbo \
  --expert-trace ~/traces/<trace>.json --prefetch-bytes 268435456 \
  --label prefetch-256m > prefetch.jsonl
```

The byte value above is only an example. Pick a ceiling for your experiment. The trace is a JSON array of pairs such as `[[0, 3], [0, 7]]`. The probe reads it and passes it to the runner's prefetch call. Prefetch runs once per open, and the record marks it as applied only on the first request.

Each record reports the cost in `expert_prefetch`. The record sets `physical_residency_guaranteed` to `false`. A prefetch that succeeded is therefore not proof that the pages are resident.

## Output

Stdout gets one JSON object per request, one line each. There is no table. The number of lines is `--reopens` times `--repeats`.

The fields fall into four groups. The names below come from the record builder in `startup_probe.rs`.

**Identity and arm.** `schema_version` (always `1`), `protocol` (`moe-startup-v1`), `label`, `model`, `manifest_sha256`, `chip`, `family`, `case`, `seed`, `shaping`, `context`, `max_new`, `expert_cache_slots`, `residency`, `open_index`, `request_index`, `prefix_reuse` (always `false`), `discarded_warmups` (always `0`).

**Requested switches.** `precompiled_requested`, `kernel_warmup_requested`, `pipeline_cache_requested`, `shared_read_overlap_requested`, `shared_read_overlap_effective`, `mapped_demand_prep_requested`, `mapped_demand_prep_effective`, `mapped_demand_budget_bytes`. The `*_requested` values are read from the environment. They say the variable was set, not that the code path ran. The `*_effective` fields report whether the path ran.

**Timing.** `metadata_peek_ms`, `tokenizer_ms`, `load_ms`, `startup` (the runner's open breakdown, including `kernel_warmup_ms`), `cache_flush_ms`, `first_token_ms`, `load_to_first_token_ms`, `request_ms`, `prefill_ms`, `decode_ms`, `prompt_tokens`, `new_tokens`, `stop`.

`first_token_ms` is measured from the start of the request. `load_to_first_token_ms` is measured from the start of the load, and is present only on the first request after each open. `stop` uses the same Swift reason spellings as the throughput harness.

**Counters.** `compiler_at_open`, `compiler_after_request`, `compiler_after_flush`, `pipeline_inventory`, `open_faults`, `request_faults` (process fault and pagein deltas, or `null` if the counters are unavailable), `expert_prefetch`, `prefetch_applied_to_this_request`, `expert_schedule`, `expert_io`, `token_ids`, `token_sha256`, `peak_footprint_bytes`, `footprint_sampling_scope` (`generation-after-open-and-prefetch`).

`token_sha256` hashes the generated token ids. Compare it only across arms that share the model, case, and shaping. A different hash is a finding to investigate, not a measurement error by itself.

`peak_footprint_bytes` covers generation after the open and prefetch. It does not cover the open, so it is not the open's footprint.

## Page cache labels

`--page-cache-label` records what you say the OS cache looked like. The probe does not check it. The output sets `page_cache_label_is_verified` to `false`, and the usage text says the label is a claim, not an eviction or residency measurement. Purge or warm the cache yourself, write the state in the label, and keep the run notes with the raw file.

## Environment switches

These variables change the startup path. The probe copies the relevant ones into each record, so check the record before you trust an arm.

| Variable | Effect |
| --- | --- |
| `TURBOSPARK_METAL_PRECOMPILED=1` | Load the Metal IR libraries embedded in the binary. |
| `TURBOSPARK_METAL_PIPELINE_CACHE=1` | Read and collect GPU pipeline archives. `docs/MOE_STARTUP.md` says this needs macOS 15 or later. |
| `TURBOSPARK_METAL_CACHE_DIR` | Override the cache directory. Use an absolute path. |
| `TURBOSPARK_METAL_KERNEL_WARMUP=1` | Prepare the experimental Qwen MoE pipeline plan at open. |
| `TURBOSPARK_QWEN_SHARED_READ_OVERLAP=1` | Recorded as requested. The record's `shared_read_overlap_effective` field shows whether the path ran. Semantics are in the runtime, not this crate. |
| `TURBOSPARK_MAPPED_DEMAND_PREP` | Recorded as requested, with the effective value alongside. Semantics are in the runtime, not this crate. |

The usage text names the first three. The last two are read only by the record builder and the runtime. This page does not define their behavior, so check the runtime source before you rely on a particular value.

`docs/MOE_STARTUP.md` says that `TURBOSPARK_METAL_KERNEL_WARMUP` stays opt-in and that the defaults are unchanged. Read that page before changing any default.

## Limits

- The first request is not a steady-state throughput number. It includes whatever the open and first request did.
- Compiler and cache counters depend on the state of the disk cache. Record whether the cache was empty before the run.
- Two runs of one arm can differ because of OS cache state. Use the page-cache label and repeat the arms.
- A `token_sha256` match does not prove output quality. Quality gates are in `crates/bench/tests/`.
- Expert prefetch with `physical_residency_guaranteed: false` is a cost measurement, not a residency guarantee.
