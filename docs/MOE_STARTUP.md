# MoE startup experiments

TurboSpark has opt-in compilation caches and a startup probe that includes the
first request. The normal loading and generation defaults remain unchanged.
Enable these paths only for measurements until paired startup, output, quality,
memory, and packaging checks support a default change.

## Compilation caches

| Control | Behavior |
| --- | --- |
| `TURBOSPARK_METAL_PRECOMPILED=1` | Load matching Metal IR libraries embedded in the Rust binary or FFI static library. |
| `TURBOSPARK_METAL_PIPELINE_CACHE=1` | Read and collect specialized GPU pipeline archives on macOS 15 or later. |
| `TURBOSPARK_METAL_CACHE_DIR=/absolute/path` | Override `~/Library/Caches/TurboSpark/metal` for an isolated experiment. |
| `TURBOSPARK_METAL_PRECOMPILE_STRICT=1` | At build time, fail if any production shader composition cannot be bundled. |
| `TURBOSPARK_METAL_KERNEL_WARMUP=1` | Prepare the experimental Qwen affine-MoE text pipeline plan during runner open. |

The build discovers the existing production `SOURCE` compositions, preserving
concatenation order, generated IQ tables, and separators. It compiles Metal 3.1
for macOS 14 with fast math. The libraries are embedded, so packaging needs no
runtime shader file lookup. Builds without the Metal compiler keep source
compilation available; use strict mode when validating a shipping artifact.

Precompiled IR skips source-to-IR work. GPU-specific pipeline creation still
occurs on an archive miss. `TURBOSPARK_METAL_PRECISE_MATH`, when present, keeps
the existing precise source-compilation path. Unknown compositions or rejected
libraries also fall back to source compilation.

Disk identities include shader content, function name, function-constant bytes,
compiler options, GPU name and registry ID, host architecture, OS version and
build, and cache schema. Packaged libraries also include the build compiler,
SDK, and actual library fingerprint. Pointer identities remain process-local.
Each specialization has its own archive; writes use an exclusive temporary file
and atomic rename. Missing, corrupt, incompatible, and unwritable entries remain
nonfatal. `archive_hits` counts only pipelines accepted with Metal's
`FailOnBinaryArchiveMiss` option.

New archives are collected during compilation and published by
`RealForwardRunner::flush_pipeline_cache()` or context destruction. The probe
flushes after each request and reports the cost separately. Ordinary runner
reopening can reuse archives after the previous runner has been dropped.
Idle app shutdown closes the Rust session and reaches that drop path. Terminating
during generation or killing the process can lose pending archives, producing
ordinary misses on the next open. Production requests do not flush automatically;
the explicit flush API is available to callers that need an earlier boundary.
macOS 14 keeps source or embedded IR loading and normal pipeline creation,
without specialized archive serialization.

## Compile-only kernel preparation

The warmup experiment registers compile-time metadata from the loaded model,
deduplicates matching keys, then prepares Metal pipelines with the production
function-constant builders. It creates no command buffers or tensor buffers,
runs no forward pass, and reads no expert blobs. Shared shader sources have
stable static addresses so preparation and dispatch hit the same process cache.
The build inventory includes both static and const source compositions.

The current plan covers unfolded Qwen GDN MoE text with affine routed experts,
top-8 routing, an INT8 router, gated shared experts, and linear FP16 attention
KV. It includes resident projections, embedding/readout, GDN, full attention,
router, routed experts, normalization, and elementwise kernels. Attention keys
cover every reachable chunk bucket through the requested context capacity.
Speculation, steering, batched prefill, quantized KV, and other families are
refused when this experiment is explicitly enabled. Without the flag, opening
retains lazy compilation. This is a bounded experiment, not full family support.

`kernel_warmup_ms` includes registration and preparation and is charged to
`total_open_ms` and `load_to_first_token_ms`. Request counters must show no new
libraries, specializations, or pipelines after warmup. The synthetic regression
compares exact logits through all attention buckets and a reset, checks that
warmup creates no GPU buffers or forward calls, and checks repeat preparation.
Removing the largest attention bucket makes that regression fail.

Build the probe as described below, then run at least three alternating fresh-process
pairs per workload, on AC power with a quiet host:

```sh
python3 scripts/moe_kernel_warmup_pairs.py \
  --binary target/release/turbospark-startup-probe \
  --model "$HOME/.turbospark/models/text/qwen36.gturbo" \
  --out target/moe-warmup/source-greedy \
  --cache source --shaping greedy --pairs 3
```

Repeat with `--cache ir-archive` and `--shaping sampled` in new output
directories. Each process makes two requests with no discarded generation.
The script checks matching output digests, compiler coverage, archive hits,
binary and manifest identity, process activity, power, and thermal snapshots.
It records kernel-reported whole-process peak footprint using `/usr/bin/time -l`
and retains partial evidence when a bounded process times out. OS page-cache
and Apple compiler-cache coldness remain unknown. Its promotion flag remains
false because startup pairs alone cannot establish quality or packaging gates.

## Run the startup probe

Build with complete bundles:

```sh
TURBOSPARK_METAL_PRECOMPILE_STRICT=1 \
  cargo build -p turbospark-bench --bin turbospark-startup-probe --release
```

Compare an open, a repeated request, and a second open in one process:

```sh
TURBOSPARK_METAL_PRECOMPILED=1 \
TURBOSPARK_METAL_PIPELINE_CACHE=1 \
TURBOSPARK_METAL_CACHE_DIR="$TMPDIR/turbospark-moe-cache" \
  target/release/turbospark-startup-probe \
  --model "$HOME/.turbospark/models/text/qwen36.gturbo" \
  --repeats 2 --reopens 2 --max-new 64 --page-cache-label unknown
```

The probe emits JSONL with no discarded generation. `load_to_first_token_ms`
on request zero covers metadata, tokenizer, runner open, optional expert
prefetch, prompt preparation, and first inference. `first_token_ms` covers the
request alone. A truncated 64-token result is useful for startup comparisons
but does not establish normal end-of-turn behavior.
Process launch/loader time, the pre-timer manifest hash, and expert-trace file
reading are excluded. The script's process wall time includes all requests and
runner destruction, so it is also a different boundary from first-token latency.

Runner phases separate manifest/index, resident mapping, KV/scratch, expert
setup, family state, and session state. Compilation counters separate library
load/compile, function specialization, pipeline creation, archive access, and
buffer allocation. These counters overlap the runner phases, so do not sum the
two axes. Counters are cumulative within a runner, including lazy compilation.
Repeated requests reset generation state and disable prefix reuse.
`resident_mapping_ms` includes Metal device, queue, and cache initialization.
Buffer allocation counters cover `MetalContext` helpers, excluding direct
device allocations such as KV and resident wrappers. `library_compiles` counts
source-compilation API calls, not verified misses in Apple's compiler cache.

Process faults and page-ins are reported where available. Physical expert I/O
is unknown unless the existing disk-I/O diagnostic collected samples. Use
`--io-diagnostics` in the pairing script for that investigation; instrumented
runs are separate from timing evidence. `peak_footprint_bytes` samples
generation after open and prefetch, so it does not establish startup peak
memory. The existing memory oracle also starts sampling after runner open; it
gates session memory. A transient startup peak needs separate periodic sampling
during open and prefetch.

## Compare fresh processes

Use AC power, a quiet host, the same binary and installed checkpoint, and at
least three alternating pairs:

```sh
python3 scripts/moe_startup_pairs.py \
  --binary target/release/turbospark-startup-probe \
  --model "$HOME/.turbospark/models/text/qwen36.gturbo" \
  --out "$TMPDIR/turbospark-moe-greedy" --pairs 3 --shaping greedy

python3 scripts/moe_startup_pairs.py \
  --binary target/release/turbospark-startup-probe \
  --model "$HOME/.turbospark/models/text/qwen36.gturbo" \
  --out "$TMPDIR/turbospark-moe-sampled" --pairs 3 --shaping sampled
```

The output directory must be new. Four arms compare source compilation,
embedded IR, populated source pipeline archives, and populated IR pipeline
archives. Archive population runs are retained separately. Every measured
invocation is a fresh process; requests within it remain visible. Population
uses the same case, shapes, residency, and generation budget as the paired runs.
On macOS 14, select `--arms source,ir`; archive arms refuse unavailable caches.
Each process has a 300-second ceiling (`--timeout-seconds`); timeouts retain
partial records and stop the comparison without qualifying an optimization.

The script records binary/manifest hashes, Git state, platform, power and
thermal snapshots, process activity at launch and exit, raw records, and paired
deltas. It rejects changed token digests, counts or stop reasons, partial
precompiled fallback, unavailable archives, and populated archive misses.
`--allow-contended` permits correctness smoke runs and prevents quiet-host
qualification. Activity snapshots do not prove the host was quiet throughout.
The summary always leaves default promotion unsupported; inspect quality,
memory, packaging, and page-cache conditions before making that decision.

These controls isolate application caches. They neither clear nor verify OS
checkpoint pages or Apple's own compiler caches. `--page-cache-label` is a
caller-supplied claim and remains marked unverified. Distinguish empty
application caches from a truly cold machine. Binary and manifest hashing
avoid walking the large weight files before the startup timer.

## Bounded expert prefetch

`RealForwardRunner::prefetch_experts` accepts caller-selected `(layer, expert)`
pairs and a whole-expert byte budget. The probe accepts the same selections as
a JSON array, for example `[[0, 12], [1, 7]]`:

```sh
target/release/turbospark-startup-probe \
  --model "$HOME/.turbospark/models/text/qwen36.gturbo" \
  --residency streamed --expert-trace /absolute/path/experts.json \
  --prefetch-bytes 67108864 --repeats 2
```

Selections are validated before I/O, deduplicated in caller order, and skipped
when a whole expert does not fit. Streamed experts are read through bounded
64 KiB scratch. Mapped experts are touched on CPU by page. This does not change
router order, slot-cache contents, or model state. Streamed `F_NOCACHE` rejects
prefetch because its reads cannot serve this warming experiment.

The budget bounds logical selected ranges, not physical OS readahead or retained
RAM. No usage history is recorded automatically and no default open calls
prefetch. Use workload-derived selections and compare both residency modes
against no prefetch, charging its elapsed time to load-to-first-token. Random
selections establish mechanics, not a useful expert hit rate.

## Required gates

Focused checks cover actual GPU output identity, every bundled library's kernel
exports, archive reuse and invalidation, corruption/failure fallback, concurrent
writes, prefetch bounds and invalid selections, and unchanged synthetic logits.
Run the normal workspace checks plus the installed Qwen3.6 gates:

```sh
TURBOSPARK_METAL_PRECOMPILED=1 TURBOSPARK_METAL_PIPELINE_CACHE=1 \
TURBOSPARK_QWEN36_INSTALL_DIR="$HOME/.turbospark/models/text/qwen36.gturbo" \
  cargo test -p turbospark-bench --test qwen36_quality_gate --release \
  -- --ignored --nocapture

TURBOSPARK_METAL_PRECOMPILED=1 TURBOSPARK_METAL_PIPELINE_CACHE=1 \
TURBOSPARK_QWEN36_INSTALL_DIR="$HOME/.turbospark/models/text/qwen36.gturbo" \
  cargo test -p turbospark-bench --test qwen36_memory_oracle --release \
  -- --ignored --nocapture
```

Use strict precompile mode for FFI and app packaging checks. A static library
build establishes embedded data delivery; it does not establish a mounted DMG,
macOS 14 execution, or release qualification. Follow [verification](../.claude/docs/verification.md)
and [release procedures](RELEASE.md) for those gates.

## Implementation and validation record

Implementation is in `crates/gpu/build.rs`, `crates/gpu/src/precompiled.rs`,
and `crates/gpu/src/context/{device,pipeline_cache}.rs`; runtime instrumentation
and prefetch live in `crates/runtime/src/startup.rs`, `real_forward_open.rs`,
and the mapped/pread streaming helpers. The probe is
`crates/bench/src/startup_probe.rs`, with fresh-process comparisons in
`scripts/moe_startup_pairs.py`. Cargo manifests, module exports, cache/prefetch
tests, and the benchmarking index were updated alongside them.

The 2026-10-02 witness used Apple M4 Max, 36 GB, AC power, macOS 26.6.2
(25G83), release builds, and the installed
`mlx-community/Qwen3.6-35B-A3B-4bit` artifact. It has 40 layers, 256 experts
per layer, and top-8 routing. Manifest SHA-256:
`045f48590e25560950ad05b1f4d39cf0cafe346d86ab9a8b44ab23b7b721b52d`.
The manifest has no source snapshot hash; this run did not rehash all weights.
Probe binary SHA-256:
`f94af2995ca1fd587b6c4b0eb20d5dcc059afefa82f689f3020a50cdb697fa84`.
The checkout included pre-existing changes in FFI, server, Swift, and an
attention shader. Validation applies to that checkout, not an isolated patch.

One fresh-process pair per shaping mode compared all four arms at 4,096
context, 16 expert slots, streamed residency, short-explanation, a 64-token
budget, and two requests per runner. No generation was discarded. Archive
population was separate. OS page and Apple compiler caches were unverified;
earlier GPU tests could warm Apple's caches. Another benchmark was active.

| Arm | Source library calls | Embedded libraries loaded | Verified archive hits | Archive misses |
| --- | --- | --- | --- | --- |
| Source | 10 | 0 | 0 | 0 |
| IR | 0 | 10 | 0 | 0 |
| Populated source archive | 10 | 0 | 29 | 0 |
| Populated IR archive | 0 | 10 | 29 | 0 |

Greedy and seeded sampled 64-token outputs matched across arms and repeats.
Two runner opens also returned identical 16-token outputs and reused all 29
pipelines. These are cache-use and output checks, not a qualified speedup.
Source library calls took only about 6.6 ms in the greedy source arm, showing
that this warmed-driver run cannot establish a large first-ever compilation
benefit. Archive file loading itself adds cost and needs paired evaluation.

Using a same-prompt router histogram, streamed prefetch prepared 37 experts
(65,470,464 logical bytes) within 64 MiB, skipped 283 over-budget selections,
and preserved the 16-token output. Its approximately 101 ms preparation was
included in load-to-first-token. This single ordered comparison does not
establish a benefit. The mapped baseline was stopped after several minutes
without a result on the memory-contended host; real mapped performance
and prefetch benefit remain unqualified. Synthetic mapped parity passed.

| Check | Result |
| --- | --- |
| Workspace build, formatting, standard Clippy, diff checks | Passed; existing FFI and attention-test warnings remain. |
| Workspace tests | 2,943 passed, 217 ignored, zero failures. |
| GPU cache/bundle tests | Passed actual GPU parity, all bundled kernel export sets, invalidation, verified-hit diagnostics, fallback and concurrent writes. Key and fail-on-miss mutations were detected. |
| Runtime/streaming prefetch tests | Passed bounds, duplicates, invalid ranges, no slot changes, streamed/mapped logits, and missing-offset mutations. |
| Pairing helper | Evidence guards passed, partial-fallback mutation rejected, real subprocess timeout retained partial evidence without a summary. |
| Qwen3.6 quality gate, both caches enabled | Passed frozen perplexity 6.2536, greedy/sampled digests, and the 8-slot check. |
| Qwen3.6 memory/throughput oracle, both caches enabled | Memory ceiling and replay guards passed: session peak 1,612 MiB under 1,700 MiB, zero replay growth. Combined target failed the 25 token/s floor (short case 19.631). |
| Source-only full-budget control | Two sampled short requests reached endOfTurn with 549 tokens, at 21.139 and 22.505 token/s, also below that floor. This control overlapped the oracle and remained contended; it cannot establish cache overhead or causality. |
| Strict Swift static library and binding tests | All 36 bundles (2,142,503 bytes) verified inside the staged archive; 90 binding tests, 18 skipped, zero failures. |
| Swift app build | Passed; smoke-run process stopped afterward. |
| App bundle / DMG | App bundle blocked by missing published `OPENKIND_RELEASE_REVISION`; no mounted-DMG or release gate run. |
| Linux GPU and streaming check | Passed. Runtime/bench and broader checks blocked by missing `x86_64-linux-gnu-gcc` for existing `onig_sys`. |
| macOS 14 hardware | Not run; source fallback and version guards were checked. |

Raw local records and captured gate logs are under the ignored
`target/moe-startup/` directory, including `validation/summary.json`. No
performance default was promoted and no frozen tolerance was changed.

## Follow-up startup measurements

A second 2026-10-02 run used the same M4 Max, installed Qwen3.6 artifact,
release probe hash, context 4,096, 16 slots, streamed residency, 64 new tokens,
and two requests per process. Builds completed before timing. Fresh processes
compared all four arms; archive population remained separate. No model format,
generation default, or tolerance changed.

External process activity was sampled every five seconds in addition to the
pairing helper's launch/exit snapshots. Whole pairs were excluded when either
snapshot reported contention or a periodic external process reached 100% CPU
within a measured process interval, with a five-second margin. These checks
cannot prove continuous GPU isolation. Application caches were isolated; OS
pages and Apple's compiler cache remained unverified and previously exercised.
This measures later launches, not first-ever cold-machine compilation.

Three retained pairs per shaping mode passed both activity filters, token
parity, and cache-use guards. Sampled retained pairs were 0/1/4 from a five-pair
campaign. Greedy used pairs 0/1 from a three-pair campaign plus one replacement
pair, preserving forward/reverse/forward arm order. Earlier campaigns and
rejected pairs remain in the raw records. All comparisons used the same binary;
greedy retained pairs span two separately populated application-cache sets.

Differences below are medians of matched arm-minus-source differences. Positive
values mean slower. They are not differences between unpaired arm medians.

| Arm | Greedy paired difference | Sampled paired difference |
| --- | --- | --- |
| Embedded IR | +6.99 ms (+0.39%) | +392.87 ms (+19.66%) |
| Source archive | +7.03 ms (+0.38%) | -145.64 ms (-7.29%) |
| IR + archive | -16.42 ms (-0.90%) | +6.73 ms (+0.34%) |

IR loaded all 10 required libraries without source fallback. Populated archives
verified all 29 pipeline hits with zero misses or errors. Greedy and seeded
sampled tokens matched across arms and repeated requests. Recorded compiler
work was only about 3-7 ms in this warmed-driver setting; most latency variation
occurred during first inference. The combined-cache sampled differences were
+20.04, +6.73, and -480.56 ms. These direction changes and the small retained
greedy differences do not establish a consistent, material startup benefit.
The original per-campaign summaries are preserved; the local consolidated
summary records the stricter pair selection without rewriting their flags.

The fresh quality gate passed with both caches enabled: perplexity 6.2536,
frozen greedy and sampled digests, and the constrained 8-slot check. Both
source-only and cached memory/throughput oracles passed their unchanged
end-of-turn, memory, replay-growth, and 25 token/s gates:

| Configuration | Short token/s | Medium token/s | Long token/s | Session peak | Replay growth |
| --- | --- | --- | --- | --- | --- |
| Source-only | 27.334 | 37.407 | 36.545 | 1,611.3 MiB | +0.17 MiB |
| IR + archive | 43.020 | 40.487 | 36.000 | 1,610.6 MiB | +0.06 MiB |

Each oracle used context 4,096, a 1,024-token budget, sampled frozen seeds, one
discarded warmup per case, and streamed residency. All three measured cases
reached endOfTurn at 549, 724, and 621 new tokens. The oracles ran sequentially,
not as alternating throughput pairs, and their monitors caught external Swift
tests/builds and renderer activity. Their rate differences therefore do not
establish cache speedup. Their gate results replace the earlier failed
throughput qualification without widening a limit. The quality run had no
periodic contention samples. Kernel-reported whole-process peaks from
`/usr/bin/time -l` also remained about 1,611 MiB; the oracle's own sampler still
covers session memory rather than continuous startup memory.

Three alternating streamed prefetch-off/on pairs had no detected launch/exit
or periodic contention. The same historical short-prompt trace prepared 37 of
320 experts, 65,470,464 logical bytes within 64 MiB, with median preparation
3.83 ms. Total load-to-first-token differences were -133.20, -3.35, and
+248.87 ms; the median was -3.35 ms (-0.19%). Median startup was approximately
1.80 seconds in both arms and output tokens matched. This does not establish
useful prefetch benefit on these exercised OS pages. The trace is not a
held-out workload; its SHA-256 is
`64dadadf5dd12e36fc4f4fe63551e8ac7fefb0d2b692b696e9bdcbcedbf513aa`.

Mapped no-prefetch and 64 MiB prefetch attempts both exceeded their 120-second
ceilings without producing a complete JSONL record. Periodic monitoring found
no external process at 100% CPU during either attempt. A separate diagnostic
stack sample in the prefetch attempt showed the main thread in
`MTLCommandBuffer waitUntilCompleted` during first prefill. This locates the
stall after open; it does not establish its underlying cause. Mapped
performance and any warming benefit remain unqualified. All owned probes and
the bounded sample process were stopped after evidence collection.

Raw commands, stdout/stderr, launch metadata, five-second activity records,
archive-population runs, trace, stack, and analysis are retained locally under
`target/moe-startup/quiet-20261002-215115/`. `qualified-summary.json` contains
the selected pairs and evidence guards; `analysis/` contains breakdowns.
Compilation caches and expert prefetch remain opt-in. First-ever cold-machine
startup, macOS 14 hardware, and packaged-app/DMG qualification remain outstanding;
this run did not resolve the published OpenKind release-pin requirement.

## Compile-only warmup results (2026-10-03)

The experiment used the same Qwen3.6-35B-A3B INT4 install on an M4 Max with
36 GB, macOS 26.6.2, context 4,096, 16 streamed expert slots, a 64-token budget,
and two requests per fresh process. Timing includes preparation during open.
OS pages and Apple's compiler cache had already been exercised. This does
not measure first-ever cold-machine startup.

Warmup deduplicated 689 registrations into 33 pipelines, including the later
attention buckets. Both requests created no additional pipelines, specialized
functions, or libraries. Source compilation used eight libraries; the cached
arm loaded all eight embedded libraries with no source fallback. Populated
archive runs verified all 33 warmup pipeline hits without misses or errors.
Greedy and seeded sampled output digests matched across modes, caches, and
request resets.

An initial campaign was contended by OpenKind tests and the Codex renderer
and contributed no retained pairs. Later alternating campaigns and confirmations
retained only pairs on AC, with normal thermal snapshots and no external process
at 100% CPU or above at launch, exit, or one-second samples. GPU isolation was
not continuously verified. Positive differences below mean slower startup:

| Compilation mode | Shaping | Retained pairs | Median warmup minus lazy startup | Pair range |
| --- | --- | --- | --- | --- |
| Source | Greedy | 5 | +34.36 ms (+1.96%) | -10.27 to +64.02 ms |
| Source | Sampled | 3 | +35.18 ms (+2.01%) | +10.05 to +141.52 ms |
| IR + archive | Greedy | 3 | -29.41 ms (-1.61%) | -66.14 to +53.11 ms |
| IR + archive | Sampled | 4 | -3.34 ms (-0.19%) | -66.50 to +37.95 ms |

Median preparation was about 1.95-2.01 ms with source compilation and 4.28 ms
with IR plus archives. Source greedy confirmations changed direction between
campaigns. Cached pairs also changed direction, and the sampled gain was small.
These readings do not establish a repeatable, material startup benefit. Kernel
reported median whole-process peaks were approximately 1,590-1,593 MiB across
both arms. Keep `TURBOSPARK_METAL_KERNEL_WARMUP` opt-in; loading defaults remain
unchanged.

The fresh quality gate with source compilation and warmup passed at perplexity 6.2536 with the
frozen greedy and sampled digests and the unchanged constrained 8-slot check.
The memory oracle also passed its unchanged end-of-turn, 1,700 MiB ceiling,
replay-growth, and 25 token/s gates. Its short, medium, and long cases generated
549, 724, and 621 tokens at 44.384, 35.966, and 34.217 token/s. Session and
kernel-reported whole-process peaks were 1,607.5 MiB; replay growth rounded to
0.00 MiB. Both real-model gate monitors observed external CPU contention, so
their rates do not establish a throughput benefit.
The synthetic regression passed exact logits through 257 decode steps and a
reset, repeat preparation, and unchanged GPU-buffer/forward-call counters.
Deleting coverage for the 16-chunk attention bucket made it fail; the mutation
was restored before the final checks.

Workspace build, tests, formatting, Clippy, and the GPU's Linux empty-shell
check passed. A full Linux workspace attempt could not build `onig_sys` because
`x86_64-linux-gnu-gcc` is unavailable. Packaged-app/DMG and macOS 14 hardware
checks were not run for this experiment.

Raw paired commands, records, compiler inventories, whole-process peaks, and
host snapshots are retained locally in
`target/moe-warmup/quiet-20261003-083906/`. Its `retained-summary.json` pools
pairs by the recorded host guards without changing original campaign flags.
The rejected initial campaign remains in
`target/moe-warmup/pairs-20261003-082750/`. Checks, real-model gate logs, the
mutation result, and a hash snapshot of 1,046 Rust/Metal/build input files are
under `target/moe-warmup/`. The paired probe binary SHA-256 is
`8c6bec8893fe1fd0bbd6fb462ac79aa10cdc79debca4e7875a4681ba5ee95b50`;
the manifest SHA-256 is
`045f48590e25560950ad05b1f4d39cf0cafe346d86ab9a8b44ab23b7b721b52d`.

## Expert scheduling experiments

Two default-off experiments borrow scheduling ideas from
[Strata](https://github.com/Niko1221/Strata/tree/99f3dbd0b21d1401b3769e0c0d963913607f380b):

- `TURBOSPARK_QWEN_SHARED_READ_OVERLAP=1` submits the existing shared expert
  before blocking streamed expert reads. Its operations and routed ranking are
  unchanged. The same Metal queue orders shared output before routed reduction;
  completion handles retain error-path lifetime protection through the final wait.
- `TURBOSPARK_MAPPED_DEMAND_PREP=advice` or `touch` prepares only actual selected
  experts before mapped GPU access. All IDs, explicit offsets, and ranges are
  validated before any access. A 16 MiB per-layer budget counts unique mapped
  page spans using the host page size. It does not bound physical disk I/O or
  guarantee residency. Advice errors are reported and tolerated.

Both settings are captured at runner open. The existing default-on
`TURBOSPARK_SHARED_CB` setting belongs to Gemma 4 and does not enable this Qwen
experiment. Streamed mode ignores mapped preparation; mapped mode keeps the
shared branch inline. No prediction, routing, model-format, or generation-API
change is involved. Batched MoE prompt prefill and cache admission changes remain
separate experiments.

The startup probe reports requested/effective settings and cumulative scheduling
counters. Router accounting excludes shared encoding and mapped preparation.
Preparation time is included in first-token and total request latency. Counters
remain cumulative across request resets; subtract consecutive records when
comparing later requests.

```sh
python3 scripts/moe_expert_schedule_pairs.py \
  --binary target/release/turbospark-startup-probe \
  --model "$HOME/.turbospark/models/text/qwen36.gturbo" \
  --out "$TMPDIR/turbospark-qwen-overlap" \
  --experiment overlap --pairs 3 --max-attempts 9 --shaping greedy

python3 scripts/moe_expert_schedule_pairs.py \
  --binary target/release/turbospark-startup-probe \
  --model "$HOME/.turbospark/models/text/qwen36.gturbo" \
  --out "$TMPDIR/turbospark-qwen-mapped" \
  --experiment mapped --pairs 3 --max-attempts 3 --timeout-seconds 120
```

The harness fixes compiler caches and warmup off, alternates arm order, uses
fresh processes, checks token digests/counts/stop reasons, records faults and
requested/effective behavior, and samples unrelated CPU activity, power, and
thermal state throughout. Timed-out and contended attempts remain in the output
directory and cannot qualify an improvement. Page-cache coldness is unknown;
these runs cannot establish first-ever cold-machine loading.

## Design sources

Apple distinguishes precompiled IR from GPU binaries in its
[Metal binary archive guidance](https://developer.apple.com/documentation/metal/metal-binary-archives)
and documents specialized serialization availability in
[device-built pipeline archives](https://developer.apple.com/documentation/metal/creating-binary-archives-from-device-built-pipeline-state-objects).
[vLLM compile-only warmup](https://docs.vllm.ai/en/v0.30.0/contributing/jit_kernel_warmup/)
motivates covering unique kernel specializations.
[Colibri's usage-history warmup](https://github.com/JustVugg/colibri/blob/ce370e87d7b623d7759b52ec2007d75fc5b0e87e/c/warmup.ps1)
motivates a separate expert-prefetch experiment. TurboSpark retains its existing
packed weights and resident/streamed loading design.
