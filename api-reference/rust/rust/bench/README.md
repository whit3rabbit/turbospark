---
description: "Development harness for measuring TurboSpark throughput, peak memory, and cold startup. For contributors, not an API."
icon: gauge
---

# turbospark-bench

`turbospark-bench` is the measurement harness the TurboSpark team uses to time the engine. It ships two binaries: `turbospark-bench` for throughput and peak memory, and `turbospark-startup-probe` for load time, first-token latency, and Metal compiler work. Reach for it when you change decode, sampling, KV, Metal, or model loading and need to know whether the number moved.

This is a development harness, not a public API. The crate is marked `publish = false`. It reads local model installs and expects a Metal-capable Mac. Nothing here is stable across versions, and the flags may change with the protocol.

## Who this is for

Contributors who measure the engine. You need a local `.gturbo` install, an Apple Silicon Mac with Metal, and a working checkout. If you only want to run a model, use the CLI or the server instead.

## Read this before you trust a number

A number from this harness is meaningful only under the measurement protocol. A row without its conditions is a timing sample, not a result. The conditions are defined in the repo, not here:

- Run formal measurements on AC power, on a quiet host, with the CPU load checked first.
- Discard the first GPU warmup. Apple Silicon clocks ramp (DVFS), so a cold pass runs slow.
- Interleave A/B arms instead of running one arm to completion first.
- Record the model family, artifact, context window, generation budget, chip, power source, and warmup conditions beside every number.
- A peak `phys_footprint` is not an on-disk model size. Do not read it as one.
- CPU, synthetic, projected, or host-contaminated runs do not establish a real Metal, quality, or memory claim.

The canonical sources are `docs/BENCHMARKING.md`, `docs/BENCHMARKS.md`, `docs/POWER_BASELINE.md`, `docs/MOE_STARTUP.md`, and `crates/bench/AGENTS.md`. The longer procedures are in `.claude/docs/benchmarks.md` and `.claude/docs/power-measurement.md`. These pages describe how to run the binaries. They do not restate the protocol or any frozen result.

## What it measures

- Decode and prefill throughput, split into prefill seconds and decode seconds, for the three frozen community-protocol cases (`real-generation-v1`).
- Peak `phys_footprint` for the process. This is the Mach counter the Swift baselines use. It includes mapped weights, KV cache, expert slots, and framework heap.
- Per-request load, first-token, and compiler counters through the startup probe.
- Scripted loop overhead (tokenizer, sampler, detokenizer, stop matcher) when no model is present.

## What it deliberately does not measure

- Output quality. Quality gates and perplexity live in the integration tests under `crates/bench/tests/`, not in these binaries.
- Cross-engine parity by itself. `scripts/parity.sh` drives this harness and the Swift engine together, and `scripts/phasediff.sh` compares decode phase buckets.
- Power draw. Power runs are driven by `scripts/power.sh` and the power-measurement reference.
- Anything about a model family the install does not contain.
- Product latency for a user. The harness drives the raw completion loop with fixed prompts and fixed seeds.

## Start with the job

| Job | Start here |
| --- | --- |
| Time the tokenizer and sampler loop with no model | [Running the harness](running-the-harness.md), scripted mode |
| Measure throughput and peak footprint of a real install | [Running the harness](running-the-harness.md), `--model` mode |
| Run one protocol case in its own process | [Running the harness](running-the-harness.md), `--case` |
| Understand a result line or the `[stop=...]` footer | [Reading the output](reading-the-output.md) |
| Measure load, first token, and compiler work without warmup | [Startup probe](startup-probe.md) |
| Decide whether a number is quotable | The protocol in `docs/BENCHMARKING.md` (repo path, not a link here) |

## Build

The binaries live in the workspace. Use release builds for any measurement.

```sh
cargo build --release -p turbospark-bench --bin turbospark-bench
cargo build --release -p turbospark-bench --bin turbospark-startup-probe
```

Offline unit tests run without a model:

```sh
cargo test -p turbospark-bench
```

Real-install integration tests are `#[ignore]`d and need a model-specific install directory from the environment. Run only the test that covers the behavior you changed. The test names and the per-family gates are listed in `crates/bench/README.md` and `.claude/docs/model-gates.md`.
