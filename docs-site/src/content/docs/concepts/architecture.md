---
title: "How the Engine Is Layered"
description: "Why the turbospark workspace is split into 17 crates, what each tier owns, and which limits (macOS-only GPU, sequential prefill, stock release profile) are deliberate."
diataxisType: "explanation"
---

This is the why-it-looks-like-this page for the workspace. It explains the
layering as a set of decisions: what each tier is responsible for, what it is
deliberately not responsible for, and what a token's path through the stack
looks like as a result. It is not a tutorial and not an API listing; per-crate
reference lives in the Rust API pages.

## Context: a port that proves itself against its original

The workspace describes itself in `Cargo.toml` as a "Behavior-compatible Rust
implementation of an approved local inference engine and its command-line
surface." The original is a Swift inference engine, and the port does not
assume compatibility with it: it measures it. Two facts about the codebase
follow from that posture.

First, correctness has an independent arbiter at every layer where one is
cheap. `crates/compute` exists almost entirely to be that arbiter for the
GPU: its README states its CPU kernels "serve as the numerical ground truth
against which `crates/gpu` Metal kernels are validated," with a relative-error
tolerance table (`tolerance.rs`, `RelError`) used by the parity tests in
`crates/gpu/tests`. At the whole-engine level, benchmark and parity scripts
compare generated output and logit divergence against the Swift engine and
against mlx-lm and llama.cpp.

Second, the port is built around one memory strategy rather than a general
one: **routed experts stream, and nothing else does**. A dense or MoE model's
resident core (embeddings, attention projections, norms, the output head) is
memory-mapped whole out of the install (`model-io`'s `ResidentBuffer`, wrapped
zero-copy into Metal buffers by `gpu`'s `ResidentGpuWeights`). Only the routed
expert tables, which are the bulk of an MoE checkpoint's bytes, are packed
into per-layer blobs and read on demand through the `pread` streamer in
`crates/streaming`. The install format (`.gturbo`), the manifest, the packed
layout metadata, and the slot cache all exist to serve that split.

## The layer map

Seventeen crates, in six tiers. Dependencies point strictly downward.

### Tier 1: leaf primitives (`core`)

`turbospark-core` is the workspace's foundation, imported everywhere under the
alias `foundation`. It defines the interchange types every other crate uses:
`pub type TokenId = i32` (the signed 32-bit token id width that crosses every
crate boundary), `LogitValue`, and `LogitsView` (`primitives.rs`). It also owns
`RuntimeConfig` with the allowed numeric sets (`ALLOWED_CACHE_SLOTS = [8, 16,
24, 32]`, `ALLOWED_CHUNK_SIZES = [128, 256, 512, 1024, 2048, 4096]`), the
three-state automatic chunk-size resolution (`chunk_sizing.rs`), prefill
chunking primitives (`prefill.rs`), and the central `CoreError` enum. Its
README is explicit that it contains no `unsafe` code. Nothing below this tier
may depend on anything except this tier.

### Tier 2: CPU references and pure logic (`compute`, `selection`, `tokenizer`, `invocation`, `window-fit`)

This tier is platform-free and mostly `#![forbid(unsafe_code)]`. `compute`
holds the CPU reference kernels: causal attention, RoPE, RMSNorm, INT4/INT8
affine quantization and GEMV, the MoE FFN bridge, the gated-DeltaNet reference
chain (`gdn.rs`, `GdnReference`), and the Walsh-Hadamard transform. Its README
notes the design rule for this tier: kernels here "prioritize mathematical
reference exactness over maximum CPU vectorization," because their job is to
define what the Metal kernels must reproduce, not to be fast. Beside it sit
the sampler (`selection`), the tokenizer wrapper and chat-template machinery
(`tokenizer`), the pure CLI argument parser (`invocation`), and the
conversation window fitter (`window-fit`). One contract in this tier shapes everything above it:
`LogitProducer::produce`
returns raw unnormalized logits and `selection::select` performs the softmax
itself, so normalization lives in exactly one place.

### Tier 3: the model data path (`model-io`, `repack`, `streaming`)

Three crates move checkpoint bytes into memory on the engine's terms.

`model-io` owns the install layout once it exists on disk: `manifest.json`
decoding and architecture validation (`ArchConfig`), the per-family baseline
specifications (Gemma 4, Qwen 3.6, DeepSeek-V4-Flash per its README), the
packed expert layout metadata (`PackedExpertsLayout`) for streamed MoE
layouts, the resident tensor index (`ResidentIndex`), the zero-copy `mmap`
wrapper (`ResidentBuffer`), SHA-256 verification, and install receipts. Its
only `unsafe` code is the mmap itself, "restricted specifically to
memory-mapping (`mmap`) operations inside `resident_buffer.rs`."

`repack` is intake: safetensors and GGUF header parsing, ranged HTTP
downloads (`RangeSource`), INT4/INT8 quantization repack, and the `.gturbo`
directory writer that assembles the manifest, the resident
`model_weights.bin` blob, and the per-layer packed expert files. It is
`#![forbid(unsafe_code)]` and streams a checkpoint a layer at a time; the
checkpoint is never written to disk whole. It also builds the synthetic
install generators the test suite runs against.

`streaming` is the demand side: `PreadExpertStreamer` reads expert weight
blobs on demand from `packed_experts/blobs.bin` via `pread`, `ExpertCache`
implements the pure LFU/LRU slot eviction policy, a process-wide pool of
parked reader threads (`read_pool.rs`) parallelizes chunk reads, and
`rdadvice.rs` wraps the macOS `F_RDADVISE` kernel hint (a documented no-op
elsewhere). The README's design note explains the split: cache eviction logic
carries no file I/O, so eviction behavior is unit-testable without
multi-gigabyte checkpoint files.

### Tier 4: the GPU backend and its portable neighbor (`gpu`, `vision-io`)

`turbospark-gpu` is the Metal backend, and it is **macOS only**: every source
file is gated with `#[cfg(target_os = "macos")]`, and on other platforms the
crate "compiles to an empty module." It owns the `MetalContext` device and
command queue with its address-keyed pipeline cache, the `PassEncoder` /
`CommittedPass` command-buffer discipline, `KvCacheManager` for per-layer KV
buffers, and `ResidentGpuWeights` for zero-copy Metal buffers over mapped
slices. The per-kernel dispatches cover split-KV decode attention (up to 16
splits), the MoE router GEMV / phase-1 / host wait / phase-2 down-reduction
chain, and the dequantization GEMV family. Its `shaders/` directory holds
the MSL sources, which its README counts at 98 compute kernels across 12
quantization formats.

`turbospark-vision-io` is the deliberate exception beside it: vision
preprocessing (decode, resize, position-table arithmetic) split out
specifically so a non-macOS build can reach it. It is portable by design,
and a cfg or dependency change that drops it from the cross-platform check
list is treated as a bug in the crate.

### Tier 5: the engine (`runtime`)

`turbospark-runtime` wires everything below it into generation. Three names
carry the architecture:

- `LogitProducer` (in `producer.rs`): the trait the generation loops drive,
  with `ScriptedLogitProducer` as the replayed-logit mock that unit tests and
  the scripted server backend use on any platform.
- `RealForwardRunner` (in `real_forward*.rs`): the real GPU-forward-pass
  engine, macOS only, which vets the architecture at open, sets up the
  expert streamers, and dispatches Metal kernels.
- `families/`: the per-family decode flows. The runtime README is explicit
  that flow selection "keys on `ArchConfig.family`, not on tensor naming,"
  because families such as Gemma 4 and Qwen 3.6 share standard weight names
  (`language_model.model.embed_tokens.weight` on both), so a naming probe
  cannot tell architectures apart; picking a neighbour's flow yields fluent
  wrong output rather than an error. `gemma4/`, `qwen/`, and `synthetic/` are
  the family directories its README documents, each with its own attention,
  MoE, and state modules.

The generation loops themselves (`run_raw_completion`,
`run_raw_completion_chunked`) live above the family flows, integrating the
producer, the streaming detokenizer, the stop matcher, and the sampler.

### Tier 6: the frontends (`cli`, `server`, `catalog`, `bench`, `ffi`)

Five crates face users, none of them in the decode path. `cli` provides the
`turbospark-check` and `turbospark-model` binaries; `server` the OpenAI- and
Anthropic-shaped HTTP endpoints (with a scripted backend for non-macOS
platforms); `catalog` the curated model table, the header-only Hugging Face
probe, and the install driver behind `pull`; `bench` the throughput harness
and the memory oracle; and `ffi` the C ABI (`turbospark.h`) that the Swift
application packages bind to.

## How a token moves through the stack

One prompt token's journey, end to end:

1. **Intake** (before any token exists). `repack` parses the checkpoint
   header (safetensors or GGUF), plans which tensors are resident and which
   are routed, streams the weights over ranged HTTP, and writes the `.gturbo`
   directory: `manifest.json`, `model_weights.bin` with its binary index,
   and `packed_experts/layer_NN.bin` blobs. `catalog` drives this for
   catalog installs; the engine never reads a raw checkpoint.
2. **Open.** `RealForwardRunner::open` loads and validates the manifest
   through `model-io` (architecture checked field by field against the
   family baseline, quantization slots checked against the block types that
   have kernels), mmaps the resident region, wraps it zero-copy in Metal
   buffers, allocates the KV cache, and opens the expert streamers over the
   packed layout.
3. **Prefill.** The generation loop calls the producer once per prompt token
   (or in chunks through the chunked driver). Each pass walks the family
   flow: embedding lookup, per-layer norms and projections, RoPE, attention
   or the linear-attention recurrence, and the FFN.
4. **Decode.** On an MoE layer the router GEMV runs on the GPU, the host
   waits for the routing readback and takes the top-k, the expert cache is
   consulted for slot residency, misses are filled by parallel `pread` from
   the packed blobs through the read pool, and the phase-1 GEMVs and phase-2
   down-reduction dispatch over the bound slots. Slots dispatch in the
   router's ranking, because slot order is summation order.
5. **Sample.** `produce` returns the raw logit row; `selection::select`
   applies the softmax, temperature, top-k and top-p; the chosen `TokenId`
   feeds back into the next decode step, and the streaming detokenizer turns
   ids into text for the stop matcher.

## Tradeoffs and deliberate limits

The layering encodes decisions that are easy to read as oversights. They are
not.

**The GPU backend is macOS-only, and portability is decided in dependency
tables.** The `#[cfg(target_os = "macos")]` gates in `crates/gpu/src` would
mean little if the dependency graph quietly made everything macOS-only
anyway, so the decision is recorded where it binds: `crates/runtime/Cargo.toml`
declares `gpu`, `model_io`, `compute` and `streaming` under
`[target.'cfg(target_os = "macos")'.dependencies]`, which is what actually
makes that crate macOS-only however portable its source reads.
`turbospark-vision-io` sits in the same target block but is portable code,
with a comment in that file explaining why: its consumer (`src/vision/`) is
macOS-gated, so a non-macOS build has nothing to call it from, and a
cross-target `cargo check` keeps the portable set honest.

**The base generation loop is sequential per token.** `run_raw_completion`
walks one `produce` call per token; batched execution exists as
family-scoped chunked-prefill and speculative-verify drivers rather than as a
property of the engine, and unsupported families are refused by name rather
than silently falling back, so a caller never measures the sequential engine
under a batched label.

**There is no `[profile.release]`, and that is a measured decision** recorded
in the root `Cargo.toml`. `lto = "thin"` plus `codegen-units = 1` measured at
**+0.27%** decode (44.545 to 44.667 tok/s) while taking the CLI's release
build from 6.9 s to 30.6 s; since every gate in the repo is a release build,
that is 4.5x on the edit-measure loop for a gain inside run-to-run spread.
The same comment records why `panic = "abort"` must stay off regardless: two
`Drop` impls (`PassEncoder`'s `endEncoding` and `read_pool`'s `Claim`) are
load-bearing on the unwind path.

**Numerics parity with any upstream implementation is out of scope** except
where a real CPU-vs-GPU parity test exists; the structural and configuration
contracts are what the tests pin, plus the measured cross-engine comparisons.

## Where the code lives

| Tier | Crates | Paths |
|---|---|---|
| Leaf primitives | `core` | `crates/core` |
| CPU references and pure logic | `compute`, `selection`, `tokenizer`, `invocation`, `window-fit` | `crates/compute`, `crates/selection`, `crates/tokenizer`, `crates/invocation`, `crates/window-fit` |
| Model data path | `model-io`, `repack`, `streaming` | `crates/model-io`, `crates/repack`, `crates/streaming` |
| GPU backend and vision preprocessing | `gpu`, `vision-io` | `crates/gpu`, `crates/vision-io` |
| Engine | `runtime` | `crates/runtime` (`producer.rs`, `raw_completion.rs`, `real_forward*.rs`, `families/`) |
| Frontends | `cli`, `server`, `catalog`, `bench`, `ffi` | `crates/cli`, `crates/server`, `crates/catalog`, `crates/bench`, `crates/ffi` |

The member list in the root `Cargo.toml` is the authoritative inventory: 17
crates, edition 2021, MSRV 1.82. Each crate directory also carries its own
`README.md` and `CLAUDE.md` with module-level detail and gotchas; this page
is the map, those are the territory.
