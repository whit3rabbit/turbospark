---
description: "Measurement harnesses for the frozen real-generation-v1 protocol, real-install memory oracles, quality gates, and the phys_footprint sampler behind the repo's published measurements."
icon: gauge
---

# bench

`crates/bench` (package `turbospark-bench`) owns the workspace's measurement
harnesses: the `turbospark-bench` and `turbospark-startup-probe` binaries, the
frozen community benchmark protocol, the real-install run flow shared by the
binary and the oracle tests, and the Mach `phys_footprint` sampler that every
published memory number in this repo is measured with. It is a dev harness
(`publish = false`) that reads local `.gturbo` installs; it is not part of the
shipped engine surface. Protocols and procedures live here and in the canonical
docs (`docs/BENCHMARKING.md`, `docs/BENCHMARKS.md`); frozen results never live
in this page.

## What it does

| Responsibility | Where |
| --- | --- |
| Frozen protocol cases, seeds, sampling settings, and the `[stop=...]` footer format | `src/protocol.rs` |
| Binary entry point, mode dispatch, and `--model` flag parsing | `src/main.rs` |
| Real-install protocol runs: warmup, measured run, per-case results | `src/real_model.rs` |
| Real-install openers, including the protocol-resolved open | `src/real_model_open.rs` |
| Per-family context window and generation budget | `src/real_model_params.rs` |
| `phys_footprint` sampler, chip brand, page-fault counters (macOS) | `src/memory.rs` |
| Scripted and synthetic-model diagnostic modes | `src/scripted.rs` (binary) |
| Real-install binary mode with a resolved-parameter header and power markers | `src/model_mode.rs` (binary) |
| Unwarmed startup and first-request probing, JSONL output | `src/startup_probe.rs` |
| Memory oracles, quality gates, and architectural probes | `tests/` |

### The three binary modes

| Mode | Invocation | Platform | What it measures |
| --- | --- | --- | --- |
| Scripted (default) | `turbospark-bench <tokenizer-dir>` | any | This port's prefill and decode loop overhead (tokenizer, sampler, detokenizer, stop matcher) through a scripted logit producer; no model weights |
| Synthetic real | `turbospark-bench <tokenizer-dir> --real` | macOS | The real GPU dispatch path, driven over a tiny deterministic synthetic `.gturbo` install |
| Real install | `turbospark-bench --model <install-dir> [--case <id>]` | macOS | The frozen real-generation-v1 protocol on a real install; the only mode whose rows are comparable to published rows and to the Swift reference |

The first two are diagnostic instruments for timing the loop and dispatch path
without a multi-gigabyte checkout.

### The frozen protocol

`src/protocol.rs` vendors the community protocol `real-generation-v1`: three
cases (`short-explanation`, `medium-review`, `long-synthesis`), each with its
own frozen seed and the byte-exact prompt text vendored under
`prompts/real-generation-v1/` (the module docs record the source JSON SHA-256
digests for provenance). The sampling settings
(`PROTOCOL_TEMPERATURE`, `PROTOCOL_TOP_K`, `PROTOCOL_TOP_P`), the shared
window and generation budget (`PROTOCOL_MAX_CONTEXT`, `PROTOCOL_MAX_NEW`), and
the routed-expert slot count (`PROTOCOL_EXPERT_CACHE_SLOTS`) are all named
constants. Each case runs one discarded warmup pass, then one measured run.
Results print the Swift-format `[stop=...]` footer so
`grep -h '^\[stop='` works against either engine.

The window and budget are per-family. `protocol_parameters(family)` in
`src/real_model_params.rs` resolves them from the opened install's own
manifest, in an exhaustive match with no wildcard arm, so adding a family
fails to compile until someone has answered for its parameters. The resolved
pair is printed in the run header beside the family, slot count, and KV
quantization, because a row measured at one window says nothing about a row
measured at another.

### Oracles and gates

`tests/` holds the integration targets: per-family memory oracles, per-family
quality gates, and architectural probes (steering, speculative acceptance, KV
quantization, mapped residency, logit dumps). The shared bodies live in
`tests/oracle_common/mod.rs` and `tests/quality_common/mod.rs`. A memory
oracle asserts, against a real install:

- peak `phys_footprint` at or under a per-chip ceiling,
- an `endOfTurn` validity gate on every measured case,
- a steady-state replay that must stop growing the footprint (the leak gate),
- a decode floor where the chip has a baseline row.

`oracle_common::assert_agrees_with_catalog` cross-checks self-measured
`ChipBaseline` rows against the `measured` block in the catalog's
`models.json`. It runs offline in `cargo test --workspace`, so an edit that
makes the two sides contradict fails on the edit rather than on the next
oracle run with an install on disk.

## Where it sits

Depends on (verified against `Cargo.toml`):

- All targets: `turbospark-core` (aliased `foundation`),
  `turbospark-runtime`, `turbospark-selection`, `turbospark-tokenizer`.
- macOS targets: `turbospark-repack` and `turbospark-model-io` (manifest
  peeking and the per-family parameter match).
- macOS dev-dependencies: `turbospark-catalog` (the oracle-catalog agreement
  test), `turbospark-gpu` (the mapped expert probe), `turbospark-vision-io`
  (the vision logit dump), plus `libc`, `half`, and `serde_json`.

Used by:

- `crates/ffi` (macOS-gated dependency): its telemetry sampler wraps
  `bench::memory::AppMemorySampler`, so the app reports the same
  `phys_footprint` counter the oracles assert.
- Developers and the repo-root helper scripts (`scripts/parity.sh`,
  `scripts/power.sh`, `scripts/phasediff.sh`), which drive the binary for
  cross-engine parity and power A/Bs.

## Public API tour

Library root `crates/bench/src/lib.rs`. `protocol` builds on every platform;
the other modules are macOS-only and cfg-gated.

| Item | Path | Meaning |
| --- | --- | --- |
| `ProtocolCase`, `PROTOCOL_CASES` | `src/protocol.rs` | The three frozen cases: id, seed, byte-exact prompt text |
| `PROTOCOL_TEMPERATURE`, `PROTOCOL_TOP_K`, `PROTOCOL_TOP_P` | `src/protocol.rs` | Frozen sampling settings |
| `PROTOCOL_MAX_CONTEXT`, `PROTOCOL_MAX_NEW`, `PROTOCOL_EXPERT_CACHE_SLOTS` | `src/protocol.rs` | Shared window, budget, and expert-slot defaults |
| `swift_footer(...)`, `swift_reason_name(...)` | `src/protocol.rs` | The `[stop=...]` run-summary footer, shaped for grep parity with the Swift reference |
| `AppMemorySampler` (`sample`, `peak_bytes`, `reset_peak`) | `src/memory.rs` | `phys_footprint` sampler with peak tracking; port of the Swift `AppMemorySampler` |
| `chip_brand_string()` | `src/memory.rs` | `sysctl machdep.cpu.brand_string`, used to pick oracle baseline rows |
| `task_fault_counters()` | `src/memory.rs` | Monotonic page-fault and page-in counters for the mapped-residency eviction probe |
| `open_model_runner_for_protocol(dir, slots)` | `src/real_model_open.rs` | Opens a real install and returns the runner, tokenizer, and family-resolved `ProtocolParameters` together, so open and run cannot disagree |
| `open_model_runner_with_context`, `open_model_runner_speculative`, `open_model_runner_steered`, `open_model_runner_with_context_and_vision_sidecar` | `src/real_model_open.rs` | One opener per measurement axis; a measuring caller cannot acquire a drafter, steering policy, or KV quantization by inheriting a default |
| `ProtocolParameters`, `protocol_parameters(family)` | `src/real_model_params.rs` | The per-family KV window and generation budget |
| `ProtocolShaping` (`Sampled`, `Greedy`) | `src/real_model.rs` | The shaping axis, deliberately separate from the speculation axis |
| `run_protocol_case_with_budget(...)` (plus `_with_context`, `_speculating`, and the plain form) | `src/real_model.rs` | One warmup-or-measured run of one protocol case |
| `CaseResult` | `src/real_model.rs` | Per-case split prefill and decode seconds, the prefill chunk actually used, token counts, peak footprint, and stop reason |
| `startup_probe::main_entry(args)` | `src/startup_probe.rs` | Entry point behind the `turbospark-startup-probe` binary |

## Examples

### Run a measurement from the shell

```sh
# Scripted loop-overhead mode (any platform, no model needed).
cargo run -p turbospark-bench --bin turbospark-bench -- <tokenizer-dir>

# Real install, one frozen protocol case in a fresh process (the
# protocol's fresh-process leg, and how scripts/parity.sh drives it).
cargo run --release -p turbospark-bench --bin turbospark-bench -- \
  --model ~/models/gemma4.gturbo --case short-explanation

# Unwarmed startup probe: JSONL load and first-request measurements,
# no discarded warmup by design.
cargo run --release -p turbospark-bench --bin turbospark-startup-probe -- \
  --model ~/models/gemma4.gturbo --label my-arm
```

`--model` mode takes explicit arms only: `--power-profile`,
`--prefill-chunk`, `--speculative` (with `--shaping greedy`),
`--speculative-drafter`, `--kv-bits`, `--expert-cache-slots`, and
`--max-tokens-per-sec`. Every resolved value is printed in the run header so
each artifact labels itself.

### Drive the protocol from Rust

Adapted from `tests/oracle_common/mod.rs`; simplified. The shared sampler is
deliberate: the oracle asserts a whole-session peak across cases.

```rust
use std::path::Path;

use turbospark_bench::memory::AppMemorySampler;
use turbospark_bench::protocol::{PROTOCOL_CASES, PROTOCOL_EXPERT_CACHE_SLOTS};
use turbospark_bench::real_model::{
    open_model_runner_for_protocol, run_protocol_case_with_budget,
};

fn measure_install(dir: &Path) -> Result<(), String> {
    // Window and budget come back resolved from the install's own manifest,
    // so the open and every run agree on them.
    let (mut runner, tokenizer, params) =
        open_model_runner_for_protocol(dir, PROTOCOL_EXPERT_CACHE_SLOTS)?;
    let mut sampler = AppMemorySampler::new();

    for case in &PROTOCOL_CASES {
        // The frozen protocol: one discarded warmup, then the measured run.
        for _ in 0..2 {
            let result = run_protocol_case_with_budget(
                &mut runner,
                &tokenizer,
                case,
                &mut sampler,
                Default::default(),
                params.max_context,
                params.max_new,
            )?;
            eprintln!(
                "{} peak so far {} MiB",
                result.case_id,
                sampler.peak_bytes().unwrap_or(0) / 1_048_576,
            );
        }
    }
    Ok(())
}
```

### Run a memory oracle

```sh
# Offline fixtures and guards, no install needed (includes the
# oracle-catalog agreement test).
cargo test -p turbospark-bench

# The real-install gate for one family, with that family's env var.
TURBOSPARK_GEMMA4_INSTALL_DIR=~/models/gemma4.gturbo \
  cargo test -p turbospark-bench --test memory_oracle --release -- --ignored --nocapture
```

The ignored real-install targets need the model-specific environment variable
named in each test's header and the release profile. Keep one model per
process: each family oracle is its own test target precisely because the
assertion is a whole-session peak, and a second `#[test]` in one file would
open a second model against the first one's high-water mark.

## Gotchas

{% hint style="warning" %}
A peak footprint is not an on-disk model size. `phys_footprint` counts
everything committed to the process: resident weights, zero-copy `MTLBuffer`
mappings over mapped weights, the KV cache, expert slot buffers, and framework
heap. Every ceiling is a ceiling at one window and budget, so read a row
together with the resolved-parameter header (family, context, max_new, slots,
kv_bits) the harness prints beside it.
{% endhint %}

{% hint style="warning" %}
The harness refuses ambiguous arms rather than inheriting them: every
measurement axis defaults off explicitly, `--speculative` requires
`--shaping greedy` and is not composable with `--prefill-chunk`, and the
`TURBOSPARK_PREFILL_CHUNK` environment variable is rejected in `--model` mode
so a run cannot inherit an arm nobody typed. Warmup passes are discarded
because a cold GPU's DVFS makes the first pass unrepresentative. Run formal
measurements on AC power with a quiet host, and interleave A/B arms instead of
running one arm after the other.
{% endhint %}

## Related pages

- [Performance benchmarks](../benchmarks/performance.md)
- [Testing and verification](../developer/testing-and-verification.md)
- [runtime](runtime.md)
- [catalog](catalog.md)
- [ffi](ffi.md)
