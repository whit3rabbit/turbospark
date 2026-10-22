# ANE/GPU experiment status

Baseline: `3483525722c97decbc0aa4298b887299b0964958`.

The experiments use independent `codex/ane-moe`, `codex/ane-async`, and
`codex/ane-events` branches. Their work is uncommitted. The release checkout
is unchanged. No new inference mode is enabled by default.

## Routing capture and replay

`codex/ane-moe` adds bounded batch detail alongside the existing router
histogram. The existing `counts`, ordered `trace`, and pilot fields keep
their formats. Enable detail with:

```sh
TURBOSPARK_ROUTER_HIST=/tmp/routes.json \
TURBOSPARK_ROUTER_BATCHES=1 \
TURBOSPARK_ROUTER_TRACE=1 \
  target/release/turbospark-check --model MODEL.gturbo --prompt 'PROMPT'
python3 scripts/ane_moe_route_replay.py /tmp/routes.json
python3 scripts/ane_moe_route_replay.py /tmp/routes.json --window 512
```

The additive `batches` object has version 1, a `truncated` flag, and rows
containing reset epoch (`request`), batch identity, phase, layer, token
range, offsets into that layer's legacy trace, ordered routes, completion,
and observed streaming-cache counters. `bytes_requested` counts logical
read requests, not physical disk traffic. `io_samples == 0` means I/O was
not observed for that row. Mapped experts do not masquerade as measured
streamed-cache hits.

`TURBOSPARK_ROUTER_BATCH_LIMIT` bounds stored selections, default 1,000,000.
Legacy `TURBOSPARK_ROUTER_TRACE` remains a separate, unbounded opt-in.
Batch detail covers the Qwen MoE chunked driver and single-token trait
calls. Single-token `produce_prefill` calls are labeled prefill;
`produce` calls are labeled decode. A caller that uses `produce` for its
last prompt token must not interpret that label as an external request
boundary. Request identity is a reset epoch, not an app conversation ID.

The analyzer refuses legacy-only or truncated captures by default. With
`--allow-truncated`, it analyzes complete rows only. Wider windows stay
within contiguous prefill ranges of one reset epoch and layer, are labeled
hypothetical, and carry no invented cache measurements. Candidate counts
use inclusive thresholds, so the suggested condition of more than 16 uses
is represented by 17. Optional weight-byte estimates are labeled modeled.

Tests exercise actual Metal inference through the synthetic repack path:
64 tokens plus a one-token tail, decode, reset, hits/misses, and output-bit
identity with observation disabled. Replay tests cover malformed captures,
thresholds, trace agreement, conservation, and window isolation. Mutation
checks verify that off-by-one eligibility and false executable-window
claims fail their regressions.

```sh
cargo test -p turbospark-runtime --lib router_hist
cargo test -p turbospark-runtime --test router_batch_capture \
  --test real_forward_qwen_moe_chunked -- --test-threads=1
python3 -m unittest discover -s scripts -p 'test_ane_moe_route_replay.py'
python3 scripts/check_ane_route_mutations.py
```

Record explicit settings, cache state, and artifact hashes with
`scripts/ane_experiment_manifest.py`. Hashing model artifacts itself warms
file-cache pages; capture cold-cache state after establishing it separately.

The ignored real-model capture probe requires both paths and refuses an
existing output. It executes prompt lengths 64, 65, 129, 256, 512, 1024,
then repeats the first prompt after reset. Each request has one decode
position. It fixes streamed residency at 16 slots, disables drafting,
checks finite logits and reset/reuse bit equality, then validates every
captured layer boundary and I/O sample count. Its corpus comes from three
named repository documents; the source manifest identifies their revision.
The probe tokenizes raw document slices without a chat template, as the
existing distribution probes do. Its route distribution is evidence for
that corpus, not a representative sample of all application requests.
This instrumented corpus run is routing evidence, not a throughput gate.
The first request is also compared bit for bit with a fresh unobserved
runner, which is dropped before the capture runner opens.

```sh
TURBOSPARK_QWEN36_INSTALL_DIR=/Users/whit3rabbit/models/qwen36-ane-experiment.gturbo \
TURBOSPARK_ROUTER_HIST=/tmp/qwen36-real-routes.json \
  cargo test -p turbospark-bench --test ane_moe_route_capture --release \
    -- --ignored --nocapture
python3 scripts/ane_moe_route_replay.py /tmp/qwen36-real-routes.json \
  --expert-bytes 1769472
```

The real probe passed using a newly streamed text-only
`mlx-community/Qwen3.6-35B-A3B-4bit` install, pinned to source commit
`38740b847e4cb78f352aba30aa41c76e08e6eb46`. The network install test
executed and passed. Its published payload is 19,550,072,750 bytes; manifest
SHA-256 is
`6dec8a58a44e6d831f195dc745cfd8eb22cc57e1fac00e2ac256f9478c1a67a6`.
No existing model was deleted. The install is at
`/Users/whit3rabbit/models/qwen36-ane-experiment.gturbo`.

The accepted serial capture contains 1,680 complete layer batches: 1,400
prefill batches (including 80 one-row tails) and 280 decode positions over
seven requests. Observation-off and reset/reuse output bits matched. There
were 676,480 prefill expert selections. Full 64-row layer batches had expert
unions from 57 to 207, median 95, which exceeds the 16 streamed cache slots
and requires bounded rounds in any future grouped driver.

| Replay width | Executable | Selection coverage at >=17 uses | Largest expert group |
| --- | --- | --- | --- |
| Actual 64-row batches and tails | Yes | 28.75% | 63 |
| 128-row windows | No, hypothetical | 45.22% | 120 |
| 256-row windows | No, hypothetical | 63.09% | 230 |

Coverage counts expert selections, not tokens. All three replays conserve
the same 676,480 selections. At the actual width, the >=16 threshold covers
30.77%; 17 is an experimental eligibility floor, not an ANE threshold.
Across both execution phases, actual cache counters recorded 351,461 hits,
327,259 misses, and 579,075,637,248 logical requested bytes. These are
not physical disk bytes. The optional replay estimate uses the packed
1,769,472-byte expert stride; it excludes F16 expansion and other ANE
preparation costs.

Raw evidence on the MoE branch is under `target/ane-experiments/`:
`real-routes-serial.json`, `real-route-manifest.json`, `real-replay-64.json`,
`real-replay-128.json`, `real-replay-256.json`, `route-summary.json`,
`model-inventory.json`, and `moe-install.log`. The aborted overlapping
attempt is retained separately as `route-overlap-aborted.log`.

The existing Qwen MoE family quality gate passed on this install:
reference perplexity 6.2536, both frozen greedy/sampled digests, repeated
warm generation, and constrained-slot determinism. It exercises the
existing per-token path. The capture parity checks above exercise the
existing batched path; neither test qualifies grouped or hybrid arithmetic.
The raw family log is `target/ane-experiments/moe-family-quality.log`.

## Expert program placement

`crates/coreml/tools/gen_ane_expert_programs.py` generates independent
shape-driven programs and an `expert-probe-v1` manifest. It shares the
projection builder with the dense generator, not its presets or manifest.
The sweep uses Qwen MoE's hidden size 2048 and intermediate size 512, with
token counts 8, 16, 17, 24, 32, 64, 128, and 256. Each count has separate
gate/up, down, and fused gate/up/SiLU/down programs, all with F16 inputs.
This is the math after external INT4-affine dequantization. It does not
include disk reads, dequantization, gather/scatter, or router weighting.

All 24 programs executed the Metal producer/Core ML/Metal consumer probe
and passed exact synchronous/asynchronous output comparison. The campaign
checks that both executors actually ran both recorded rounds and retains
the complete compute plan. A separate CPU oracle checks non-square
64/128-dimensional projections and fused SiLU at 17 tokens against NumPy
FP64 contractions, within the existing isolated Steel test's 0.005
row-relative bound. A valid-shape mutation using `sigmoid(up)` instead of
`sigmoid(gate)` compiled and failed the fused numeric assertion. On the
local M4 Max, with CPU-and-ANE compute
units, planned preferred placement was:

| Token rows per expert | Gate/up matmuls | Down matmul | Fused expert operations |
| --- | --- | --- | --- |
| 8, 16, 17, 24, 32, 64 | CPU | CPU | CPU |
| 128 | Neural Engine | CPU | Neural Engine |
| 256 | Neural Engine | Neural Engine | Neural Engine |

The programs list ANE support even when CPU is preferred. Supported does
not establish execution on ANE. This finding prevents qualification of
these programs for hybrid offload at the current 64-row prefill width.
It is not a universal ANE threshold: padding, another export, OS, or chip
could change placement and would require its own measured cost and quality
experiment. Groups of 128/256 are hypothetical at the current runtime
width. Full expert costs and GPU-grouped execution remain unqualified.
The host was on battery during this check, so raw timing is not used for
a performance decision.

```sh
uv run --python 3.12 --with coremltools==9.0 --with numpy==2.2.6 \
  python -I crates/coreml/tools/gen_ane_expert_programs.py PROGRAMS \
  --hidden 2048 --intermediate 512
# PROBE is the ane_async_probe executable built on codex/ane-async.
python3 scripts/ane_expert_placement.py PROBE PROGRAMS RESULTS
uv run --python 3.12 --with coremltools==9.0 --with numpy==2.2.6 \
  python -I crates/coreml/tools/test_ane_expert_programs.py
```

Final raw output and `complete: true` summary are in
`target/ane-experiments/expert-shape-placements-pinned/`. The campaign
explicitly pins CPU-and-ANE compute units in each child. Generated packages
are in `expert-shape-programs/`; both are on the MoE branch. These are
isolated program checks, not full-model hybrid quality or a memory oracle.

## Public asynchronous Core ML

`codex/ane-async` adds a macOS 14-gated `predict_shared_async` API under
`ane-prefill`. Its ticket owns input/output planes, retains Objective-C
resources, and holds the same model mutex as synchronous predictions.
`wait` returns the result and planes. Drop waits for completion. A lost
completion is not cancellation and cannot release buffers for reuse.
Models with MLState are refused before stateless submission.
Shared planes remain Send and not Sync. Cold Core ML compilation and cache
publication are serialized within the process after concurrent loading exposed
a shared temporary-directory race. An eight-caller cold-load regression
checks completed model publication and prediction. Removing the compile lock
survived that timing-dependent test, so a separate controlled-publication
regression was added. Removing the lock now fails that regression. Cold
compiles of different packages also serialize in this implementation, which
can increase concurrent cold-load latency. Predictions remain per-model.

The standalone `ane_async_probe` uses a checked Metal producer notification,
Core ML prediction, and a checked dependent Metal consumer. It compares
output bits across sync/async execution and prints timings. Compute plans
must be inspected: the tiny fixture is placed on CPU. The existing dense
Qwen gate/up program is planned on Neural Engine. Output equality passed
for both. These are isolated hardware checks, not whole-model quality or
throughput gates.

```sh
cargo test -p turbospark-coreml --features ane-prefill
python3 scripts/check_ane_async_mutations.py
cargo build -p turbospark-runtime --features ane-prefill --release \
  --example ane_async_probe
python3 scripts/ane_async_campaign.py \
  target/release/examples/ane_async_probe PROGRAM.mlpackage OUTPUT_DIR
```

The campaign alternates six fresh-process pairs by default, discards a
warmup within each process, preserves stdout/stderr, and reports a paired
bootstrap interval for prediction-time deltas. It does not report request
p99, energy, or real-model prefill speed. Its total wall interval includes
output validation and the consumer, so prediction and wall columns answer
different questions.

Tests include output-backing identity, repeated plane reuse, independent
sessions, model serialization, drop/drain behavior, and a subprocess
watchdog for a simulated lost notification. Both the backing guard and
missing-completion regression detect verified source mutations.

An exploratory AC-powered run on M4 Max used six fresh-process pairs of
Qwen's existing 1,024-token gate/up program (K=5,120, two 7,168-row ANE
outputs). Synthetic inputs varied by row and warmup/measurement pair.
The two six-pair campaigns gave these prediction-time results:

| Campaign | Sync mean ms | Async mean ms | Async minus sync ms, bootstrap 95% interval |
| --- | ---: | ---: | --- |
| Initial | 17.174 | 17.002 | -0.173, [-0.420, 0.073] |
| Final source | 17.265 | 17.543 | +0.278, [0.112, 0.401] |

The initial interval crossed zero. The final-source run measured slower
async prediction. Neither supports promoting public async into inference.
Thermal state was not recorded, and neither host contention nor a full
model was measured. Raw campaigns are under
`target/ane-experiments/dense-fresh-pairs/` and
`target/ane-experiments/dense-fresh-pairs-final/` on the async branch.


The runtime scheduler is unchanged. There is no
`TURBOSPARK_ANE_PREFILL_EXECUTOR` mode yet. Integration requires a supported
performance or handoff-tail result and the full dense quality/failure
campaign from the original plan. Reduced CPU blocking alone is insufficient.

The existing real dense quality probe was also executed with the
`qwen38-27b-mtp.gturbo` install. The benchmark opener disables drafting,
so the MTP head did not force the narrow prefill path. The weight SHA-256
matched its manifest:
`4b53a13572fe928653ae7a9cffac4fd06d9cf520aecfbc237a49a051afbd3644`.
On this M4 Max (36 GiB, macOS 26.6.2), the 64/200/500/1000/2000/3000-token
distribution probe passed the unchanged `1e-3` mean KL bound:

| Arm | Mean KL against per-token reference | Top-1 agreement |
| --- | --- | --- |
| Default Steel | 1.893e-5 | 6/6 |
| Break, width 2048 | 1.547e-5 | 6/6 |
| Synchronous ANE split | 1.512e-5 | 6/6 |

The split loaded all ten programs and differed from break by at most
0.023438 in logits. This is synchronous full-model quality evidence.
It does not qualify asynchronous scheduling, sampled generation, or memory.
The host was on battery with a concurrent network install; elapsed times
are excluded from performance evidence. The raw log and artifact manifest
are `target/ane-experiments/dense-real-quality.log` and
`dense-real-quality-manifest.json` on the async branch.

The existing dense family quality gate also passed with this same install:
reference perplexity 4.9444, the frozen greedy and sampled digests, repeated
warm generation, and constrained-slot determinism. It exercised the
existing per-token path, not a new asynchronous scheduler. The raw log is
`target/ane-experiments/dense-family-quality.log` on the async branch.

The existing dense memory oracle passed its three protocol cases and
reset/reuse assertion with a peak of 661.7 MiB against the unchanged
750 MiB process-footprint ceiling, at context 4096 and maximum generation
1024. This measures the existing default path with drafting disabled.
Its raw log is `target/ane-experiments/dense-memory-oracle.log` on the async
branch. It does not qualify a new scheduler or remove the host-condition
limits below.

Host swap rose from about 2.9 GiB to 6.5 GiB across the combined install
and model jobs. The first route probe and dense memory oracle overlapped.
The route host was paused, then that attempt was terminated to release its
retained GPU allocations. The accepted route capture uses a fresh serial
attempt after the memory oracle. The overlapped records are contaminated
and cannot qualify throughput, handoff tails, or the no-sustained-swap
requirement.

## Private device events

`codex/ane-events` contains a standalone Objective-C probe and process
watchdog under `research/ane-events`. It dynamically loads the private
framework. Nothing links this directory into Cargo, FFI, or the app.

The probe passed six iterations on this M4 Max. It independently withheld
producer readiness and an additional ANE wait, then checked changing
sentinels through an identity matmul and dependent Metal readback. Removing
either private wait failed the matching stall test. The graph uses private
ANE wait/signal requests over Metal's IOSurface shared events; no host
completion callback signals the dependent Metal event. Host signals only
release deliberately withheld test gates.

```sh
python3 research/ane-events/run.py
python3 research/ane-events/test_mutations.py
```

Interfaces were inspected in [phhusson's pinned ANE backend](https://github.com/phhusson/llama.cpp/blob/abcc7f8fa8920f50bc321ee7e7dcfa31b133a6a0/ggml/src/ggml-ane/ggml-ane.m).
Availability on another chip or OS, long-running reliability, integration
with public Core ML, and performance remain unqualified. The private probe
uses a different model submission interface from the public Core ML runner.
A successful fence experiment does not establish that MLModel can accept it.

## Remaining gates

The missing-model prerequisite has been resolved and actual 64-row route
distributions are recorded above. The tested public Core ML exports prefer
CPU at every executable expert group size. There is no qualified ANE
placement or positive measured cost model for dispatch at that width.
Padding, complete expert-cost comparisons (including grouped Steel and
concurrent workloads), and memory admission remain open experiments.
GPU-grouped prefill, hybrid dispatch, and their experimental environment
controls have not been added. The current export experiment is a negative
finding; it does not establish that every possible export or padded shape
would fail.

No new family quality tolerances, memory ceilings, defaults, ABI values,
or application controls changed. Full-model quality, memory, cold-cache,
controlled-contention, and release qualification remain open. The acceptance
threshold remains at least 5% reproducible end-to-end prefill improvement,
or independently demonstrated public-async handoff-tail improvement.

## Validation record

The routing branch passed `cargo build --workspace`, `cargo test --workspace`,
`cargo fmt --check`, and `cargo clippy --workspace --tests` before the final
real-model harness and standalone expert tools were added. Those additions
passed their focused build/lint, real capture, numerical oracle, and
hardware placement checks. Clippy retains existing warnings outside the
changed code. Focused capture tests and both replay mutations passed.

The async branch passed Core ML tests (including actual shared predictions),
the GPU suite with `ane-prefill`, runtime feature unit tests, and dense/MoE
chunked-prefill integration tests. The private probe and both event-wait
mutations passed on the local M4 Max.

The documented eleven-package Linux check could not compile `onig_sys`
without `x86_64-linux-gnu-gcc`. The ten-package check excluding image passed.
Core ML and GPU also passed with `ane-prefill` on that target. The additional
bench test cross-check, with default and experimental features, was blocked
by the same missing C cross-compiler in `onig_sys`. This does not
establish Linux runtime execution.

Raw logs, captures, replay output, source/binary/program hashes, and hardware
manifests are retained under each worktree's ignored
`target/ane-experiments/` directory. No application bundle, DMG, or release
gate was run. The dense distribution, family quality, and memory-oracle
checks above executed and passed. Performance acceptance, no sustained
swap, and new scheduler/hybrid qualification remain open.

## Evidence index and decisions

These links point to local, ignored artifacts in the experiment worktree.
Small records from the other branches are copied here so one report can
open their evidence. Binaries and model weights stay in their original
locations; the manifests record their hashes.
The [source snapshot index](../target/ane-experiments/source-snapshots.json)
records SHA-256 hashes for archives containing each branch's tracked patch
and complete changed/untracked source files, so the uncommitted state can
be reconstructed from the pinned baseline.

| Experiment | Provenance and results |
| --- | --- |
| Real routing | [Manifest](../target/ane-experiments/real-route-manifest.json), [capture](../target/ane-experiments/real-routes-serial.json), [driver log](../target/ane-experiments/real-route-serial.log), [replay summary](../target/ane-experiments/route-summary.json) |
| MoE model | [Install log](../target/ane-experiments/moe-install.log), [inventory](../target/ane-experiments/model-inventory.json), [family quality](../target/ane-experiments/moe-family-quality.log) |
| Expert exports | [Manifest](../target/ane-experiments/expert-placement-manifest.json), [placements](../target/ane-experiments/expert-shape-placements-pinned/summary.json), [numeric oracle](../target/ane-experiments/ane-expert-generator-oracle.log), [detected mutation](../target/ane-experiments/ane-expert-generator-mutation.log) |
| Public async | [Manifest](../target/ane-experiments/async/manifest.json), [final paired campaign](../target/ane-experiments/async/dense-fresh-pairs-final/summary.json), [lifetime/serialization tests](../target/ane-experiments/async/coreml-tests.log) |
| Dense real model | [Manifest](../target/ane-experiments/async/dense-real-quality-manifest.json), [distribution gate](../target/ane-experiments/async/dense-real-quality.log), [family gate](../target/ane-experiments/async/dense-family-quality.log), [memory oracle](../target/ane-experiments/async/dense-memory-oracle.log) |
| Private events | [Manifest](../target/ane-experiments/events/manifest.json), [raw results](../target/ane-experiments/events/private-events.jsonl), [mutations](../target/ane-experiments/events/private-events-mutations.log) |
| Limits and failed attempts | [Host conditions](../target/ane-experiments/host-conditions.json), [aborted overlapping route run](../target/ane-experiments/route-overlap-aborted.log), [default cross-check](../target/ane-experiments/ane-bench-cross-default.log), [feature cross-check](../target/ane-experiments/ane-bench-cross-feature.log) |

| Integration | Decision |
| --- | --- |
| Public async API | Keep the feature-gated API and standalone probe. Hold dense scheduling changes: the paired handoff campaign did not establish improvement. |
| Expert offload | Keep captures, replay, and shape probes. Hold grouped/hybrid runtime dispatch: the current exports lack qualified ANE placement at executable group sizes, and full costs remain unmeasured. |
| Private events | Keep the isolated research executable. No private runtime, FFI, or app dependency; shipping integration needs a separate evidence-backed decision. |
