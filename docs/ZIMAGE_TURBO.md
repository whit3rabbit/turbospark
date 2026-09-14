# Z-Image-Turbo: image-model bring-up record

Status: IG0 and IG1 are closed for the pinned 1024-by-1024 reference case.
IG2 implementation is in progress. The checked image format, local packer,
macOS Metal backend, and CLI path exist, but no complete pinned packed install
has passed the real native parity, quality, cancellation, and resource gates.
This page is both the summary of what was learned from Z-Image-Turbo and the
reusable process for bringing up another image-generation model in this
repository.

This page is the case-study index. The detailed design lives in
[IMAGE_GENERATION.md](IMAGE_GENERATION.md), the evidence ledger lives in
[IMAGE_GENERATION_PHASE0.md](IMAGE_GENERATION_PHASE0.md), and the frozen
resource and install contract lives in
[z-image-ig0-resource-contract.json](verification/z-image-ig0-resource-contract.json).

## What this work established

Z-Image-Turbo is a dense text-to-image pipeline. It is not an autoregressive
text model, not the vision pipeline, and not an MoE workload. The v1 pipeline
has five logical components:

| Component | Role | Pinned reference fact |
| --- | --- | --- |
| Tokenizer | Prompt framing and padded token IDs | Qwen2 tokenizer assets, maximum 512 tokens |
| Text encoder | Qwen3 hidden-state conditioning | 36 blocks, hidden size 2560, output `hidden_states[-2]` |
| Transformer | Dense diffusion denoiser | 2 noise-refiner blocks, 2 context-refiner blocks, 30 main blocks |
| Scheduler | FlowMatch Euler latent updates | Shift 3.0, 9 scheduler steps, 9 actual transformer forwards |
| VAE decoder | Final latent to RGB pixels | Decoder-only execution for v1, output `[3,1024,1024]` |

The first supported envelope is deliberately narrow:

- 1024 by 1024 output
- batch size 1
- 9 scheduler steps and 9 transformer forwards
- guidance scale 0
- one heavyweight stage owning managed GPU residency at a time
- an explicit unsigned 64-bit seed
- prompt input capped at 512 tokens after the model's framing and padding rules

The resource ceilings are reference-process observations, not a packed INT4
runtime measurement:

| Stage | Qualified peak process footprint | Retained driver allocation | Inclusive non-parameter budget |
| --- | ---: | ---: | ---: |
| Text encoder | 8,969,979,152 bytes | 8,322,236,416 bytes | 925,042,960 bytes |
| Transformer | 17,512,599,008 bytes | 14,018,134,016 bytes | 5,202,781,536 bytes |
| VAE decoder | 11,025,630,840 bytes | 10,060,791,808 bytes | 10,690,352,108 bytes |

The non-parameter budget is inclusive. It covers activations, temporary
tensors, staging, process overhead, allocator retention, and unobservable MPS
scratch. Exact operator scratch is not available from the current MPS
instrumentation. Driver allocation minus live allocation is not scratch, and
the three stage peaks must not be added together. The measurements make no
minimum whole-machine RAM claim.

The complete contract, including physical reads, swap observations, ownership,
manifest fields, hashes, and qualification rules, is the
[IG0 resource contract](verification/z-image-ig0-resource-contract.json).

## Artifact and runtime boundary

The production target is not an arbitrary Hugging Face MLX directory. The
canonical source is the pinned Diffusers-style Z-Image-Turbo export. The image
packer converts that source into the repository's separate `.image.gturbo`
format, and `turbospark-model pull-image` currently accepts the source as a
local directory rather than downloading by repository ID.

The first runtime profile is affine INT4 linear weights with group size 64.
This does not quantize every tensor: embeddings, norms, modulation, positional
data, and other protected tensors remain at higher precision, as do the VAE
and other image-sensitive operations. The already-quantized
[`andrevp/Z-Image-Turbo-MLX-4bit`](https://huggingface.co/andrevp/Z-Image-Turbo-MLX-4bit)
and [`uqer1244/MLX-z-image`](https://huggingface.co/uqer1244/MLX-z-image) exports
are candidate inputs for a future adapter, not supported drop-in installs.
Their MLX tensor layout must be translated and checked against the pinned
reference before admission. The 2-bit, 8-bit, and full-precision variants are
not current IG2 profiles.

`crates/image` is an intentional new Rust crate. It owns the image graph,
install schema, packed storage, scheduler, VAE, and native Metal backend. It
shares only matching context, pass, and resident-buffer contracts with the
general GPU crate. This lets the CLI and the later C ABI/Swift package use the
same image runtime without putting diffusion state into the autoregressive
text runner.

Memory work starts with ownership, not a disk-size headline: keep only one
heavyweight stage resident, map packed payloads without expanding every matrix,
dequantize INT4 linear weights at use, and release stage state before loading
the next component. The current Metal wrappers are correctness-first and
still create many operation-level command buffers and temporary buffers. The
native minimum-memory claim therefore remains open until pooled scratch,
activation reuse, safe BF16/FP16 storage, and cold/warm resource measurements
are complete.

## Pinned inputs

| Input | Revision | Purpose |
| --- | --- | --- |
| `Tongyi-MAI/Z-Image-Turbo` | `f332072aa78be7aecdf3ee76d5c247082da564a6` | Canonical checkpoint and configs |
| `huggingface/diffusers` | `a71e62e0d226c284b86abf518791a5ffbba064bf` | Scheduler and pipeline reference |
| `mflux-community/mflux` | `051ba9ff25c9a8a8703356053a012a0dfad3fe39` | Independent Apple Silicon operator reference |

The model is Apache-2.0. Diffusers is Apache-2.0. MFLUX is MIT. The isolated
reference environment and all source revisions are recorded in
[z-image-ig0-inputs.json](verification/z-image-ig0-inputs.json). The probe
records 1,163 tensors, their shapes, dtypes, spans, component ownership, and
source file hashes. All seven downloaded weight shards pass local SHA-256
verification, recorded in
[z-image-ig0-download.json](verification/z-image-ig0-download.json).

The canonical pipeline requires the text encoder, transformer, scheduler,
tokenizer, pipeline index, and VAE. There is no CLIP or image encoder. The
VAE contains encoder tensors in the source checkpoint, but text-to-image v1
needs only the decoder subset. Text embedding and output weights are tied, and
the vocabulary head is not part of the image-conditioning execution.

## Facts that changed implementation decisions

### Conditioning

The image pipeline uses Qwen3 as a hidden-state encoder, not as a text
generator. The exact contract is:

1. Apply the checkpoint chat template to one user message.
2. Use `add_generation_prompt=True` and `enable_thinking=True`.
3. Pad and truncate to 512 tokens.
4. Use the causal attention mask and the checkpoint Q/K RMSNorm epsilon of
   `1e-6`.
5. Extract `hidden_states[-2]`, which is the output after block 35 of 36 and
   before the final block and final normalization.
6. Remove padding rows before passing conditioning to the transformer.

The seven prompt fixtures include empty, Unicode, overlong, and four visual
review prompts. Empty input still produces eight framing tokens. The overlong
case contains 2,409 input tokens and truncates to 512. Token IDs and masks are
captured fixtures, not hand-written examples.

The native FP32 CPU encoder reaches relative L2 error from `8.69e-3` to
`8.86e-3` against the captured BF16 MPS conditioning. That tolerance reflects
accumulation and reduction differences across 35 layers. It is not a reason to
substitute the final text-generation state or to reuse a text runner's KV
cache.

### Scheduler and evaluation count

The official example and the pinned implementation do not describe the same
evaluation count. The model-card comment says eight transformer forwards, but
the pinned Diffusers invocation produces nine. The capture contains these nine
transitions:

```text
initial_noise -> latent_00 -> latent_01 -> latent_02 -> latent_03
              -> latent_04 -> latent_05 -> latent_06 -> latent_07
              -> latent_08 == final_latents
```

The actual observed forward count, not the prose comment, is the contract. The
native scheduler matches the captured timesteps and sigmas and passes all nine
Euler updates within the local floating-point gate. The accumulated FP32 CPU
versus BF16 MPS state-drift envelope is frozen at `0.196`; it is not a local
timestep error budget and must not be used to excuse a bad packed kernel.

Integer seeds do not imply equal initial noise across implementations. The
request and output metadata must record the resolved seed and the noise
provenance used by the runtime.

### Transformer layout

The transformer is a dense scan, not expert routing. The native reference
proved the following before any Metal implementation was considered:

- learned padding tokens remain attendable;
- the image and caption sequence construction is explicit;
- patch ordering and unpatching preserve the `[1,16,128,128]` latent geometry;
- positional encoding uses the required three-axis RoPE;
- timestep modulation and gated residual ordering are preserved;
- the two noise-refiner, two context-refiner, and thirty main blocks are all
  executed;
- reductions and linear bias application use the reference operation order.

The full-width 64-token checkpoint block passes the frozen cross-engine gate.
The complete nine-step native rollout passes its captured-input gate. These
are correctness references, not proof of the production packed representation.

### VAE

The VAE path is a 2D decoder with convolution, group normalization, and
upsampling. It is not interchangeable with a text convolution or the causal
convolution used by another model family. The real 1024-by-1024 decode gate
produces `[3,1024,1024]` and the native path converts the result to verified
interleaved RGB PNG bytes.

The optional raw-pixel arrays were absent from this checkout. The real decode
and geometry gates therefore passed, while the conditional raw-pixel
assertions did not run. A future model must distinguish an absent optional
comparison from a passed comparison.

### Quantization

The evidence selected four-bit linear weights with group size 64 as the
production candidate. The group-32 policy reduces error in sampled projections
but does not match the existing group-64 kernel contract. The quantization
study is emulation evidence and uses BF16 allocations; it is not a packed
Metal memory measurement.

The production manifest must explicitly list quantized and protected tensors,
group size, nibble order, scale and bias convention, source dtype, packed file
span, and file hash. A tensor must not become INT4 merely because its name
looks like a linear projection. Protected norms, embeddings, modulation,
positional data, and any other exception must be named by the installed
manifest.

## Evidence outcome

The completed reference work includes:

- 18 locked Python quantization-policy tests;
- 16 mutation-checked native IG1 assertions;
- eight 1024-by-1024 review captures covering composition, typography, detail,
  and lighting in BF16 and quantization-emulation forms;
- exact tokenizer framing, token IDs, masks, scheduler schedule, and nine
  captured latent transitions;
- a full-width checkpoint DiT block gate, a complete nine-step native rollout,
  and a real VAE decode and PNG geometry gate;
- quiet-AC resource records for encoder, transformer, and VAE cold/reuse arms;
- a frozen resource and manifest contract that records the limitations instead
  of promoting them to unsupported guarantees.

The optional raw-pixel VAE arrays were not present, so only the real decode and
geometry contract was exercised for that conditional comparison. That is an
open evidence limitation, not a reason to claim a failed VAE implementation.

## The phase model

The phases are intentionally ordered. A later phase cannot repair a missing
contract from an earlier one.

### IG0: resource evidence and manifest contract

IG0 answers: what is the model, what exactly is executed, what does the
reference consume, and what does an install need to declare?

Exit requirements:

- immutable model and reference revisions, licenses, and dependency lock;
- complete component and tensor inventory with file and payload hashes;
- tokenizer, scheduler, dimensions, latent geometry, and actual forward-count
  probes;
- reference captures with deterministic noise, prompts, intermediate arrays,
  final latents, decoded output, and recursive validation;
- operator reuse and gap table;
- quantization candidate and protected-tensor policy;
- quiet-AC cold and reuse measurements for every stage;
- retained memory, physical reads, swap observation, and qualification status;
- an inclusive resource envelope that does not pretend exact MPS scratch is
  known;
- a manifest contract with component order, file hashes, tensor spans,
  capability identity, supported dimensions, and resource policy.

IG0 does not implement the production runtime, CLI, C ABI, or Swift app.

### IG1: native correctness reference

IG1 builds independently testable CPU or portable reference components. It
proves conditioning, scheduler, transformer math, VAE geometry, and PNG
conversion against pinned fixtures. It also mutation-checks the assertions.

IG1 is complete only when a discrepancy is explained at its first divergent
intermediate. A final-image comparison alone is not enough.

### IG2: production quantized Metal runtime and CLI

IG2 consumes the frozen manifest contract. It implements installation,
quantized packed tensors, staged text-encoder, transformer, and VAE ownership,
then exposes one CLI generation path.

The CLI gate requires:

- a complete verified image install;
- valid PNG output;
- prompt, dimensions, seed, and settings metadata;
- progress by owned stage;
- overwrite protection;
- cancellation that stops scheduling and preserves ownership safety;
- packed-runtime numerical parity against the IG1 fixtures;
- stage measurements against the IG0 envelope.

No app work starts here. The CLI is the first production consumer because it
isolates runtime ownership and cancellation from Swift persistence and chat
state.

Current implementation state (2026-09-13): the repository has the checked
image manifest, affine INT4 group-64 packed component format, atomic local
packer, verified-install receipt, staged lifecycle, PNG metadata, progress and
cancellation seams, a macOS-only native Metal backend, image-specific MSL
operators, and an explicit `turbospark-model pull-image` route that records
image installs separately from text rows. `turbospark image generate` selects
native Metal by default on macOS; `--backend reference` is explicit. The
packed comparison, metadata, quality, and resource gates are present as
opt-in tests. A pinned local export now packs successfully to 11 files and
6,906,461,695 bytes. Its native conditioning error is 0.081541 against the
frozen 0.084 INT4 quality envelope. The grouped Metal attention kernel passes
a focused GQA and causal-mask parity fixture, but the nine-step native denoise
still did not complete within roughly six minutes after that optimization, so
no complete packed install has passed the real gates. IG2 therefore remains
open. Keep the CPU backend as the diagnostic oracle, not as an unrecorded
fallback. See
[IMAGE_GENERATION.md](IMAGE_GENERATION.md#ig2-handoff-checklist) for the
file-level checklist and stop conditions.

### IG3: memory and lifetime proof

IG3 proves resident versus streamed behavior, bounded buffer lifetimes,
refusal before execution, and repeated-job memory stability. It accounts for
weights, conditioning, latents, activations, scratch, staging, in-flight GPU
work, and allocator retention.

Cancellation must stop future scheduling and wait for outstanding I/O and GPU
consumers before freeing or reusing buffers. VAE tiling is a conditional
optimization. Add it only if measured VAE resource use requires it, then
re-run geometry, seam, quality, and memory gates.

### IG4: app and ABI integration

IG4 exposes the already-proven runtime through the C ABI and Swift wrapper. It
adds image chat mode, preview, save, regenerate, profile-scoped persistence,
and serialized heavyweight image jobs. The app must use the same conditioning,
scheduler, quantization, cancellation, and output implementation as the CLI.

Image chat does not begin before IG2 and IG3 establish stable native ownership,
memory, and cancellation contracts.

### IG5: measured optimization

IG5 changes only measured bottlenecks. Candidates include bounded block
streaming, staging reduction, allocator reuse, or VAE tiling when the evidence
requires them. Every optimization re-runs correctness, memory, and cancellation
gates. It does not widen the supported envelope by assertion.

## Process used for Z-Image-Turbo

This is the reusable sequence. Keep the order, but replace model-specific
constants and component roles.

1. **Write the scope before code.** Define the first output size, batch policy,
   steps, guidance, seed semantics, prompt policy, deferred features, and the
   release boundary. Do not start with the app.
2. **Pin every external input.** Record the model commit, reference commits,
   licenses, package lock, Python version, and source file hashes. Use an
   immutable revision, not a moving branch or a model-card URL alone.
3. **Probe without downloading weights.** Read config, pipeline index,
   tokenizer metadata, scheduler fields, shard indexes, and bounded safetensors
   headers. Reject malformed spans, missing components, duplicate tensors, and
   unbounded range responses.
4. **Download and verify separately.** Fetch only after the inventory is
   valid. Verify every required shard locally and keep published hashes
   distinct from locally computed hashes.
5. **Capture a reference pipeline.** Record prompt framing, token IDs, masks,
   initial noise, every scheduler state, every transformer output boundary,
   final latents, decoded arrays, PNG bytes, timings, and provenance. Generate
   and validate every fixture before using it in a benchmark.
6. **Build the operator gap table.** For each component, record reuse,
   adaptation, or new implementation. Do not assume that a vision kernel, text
   kernel, or causal convolution has the same layout or lifetime contract.
7. **Establish native correctness.** Implement portable references first,
   compare at intermediate boundaries, freeze tolerances from measured error,
   and mutation-check each assertion. Keep CPU correctness separate from
   production Metal ownership.
8. **Measure quantization as a policy.** Compare group sizes and protected
   tensors on representative activations. Treat this as numerical evidence,
   not as packed-runtime resource evidence.
9. **Run quiet-AC resource measurements.** Use fresh processes, cold-copy and
   reuse arms, physical-read classification, 100 ms resource sampling, kernel
   peak footprint, retained driver bytes, swap observations, exact-output
   checks, and a strict competing-process gate.
10. **Freeze a conservative contract.** Bound what is observable. If exact
    accelerator scratch is unavailable, use an inclusive process budget and
    say so. Do not convert one machine's result into a minimum RAM claim.
11. **Build the staged runtime and CLI.** Validate the packed manifest, load
    one heavyweight stage at a time, and measure the actual production path.
12. **Prove lifetimes before the app.** Test repeated jobs, streaming,
    cancellation, refusal, and memory stability before adding chat persistence
    or Swift events.

## What counted as evidence

The following distinctions prevented false closure:

| Observation | What it proves | What it does not prove |
| --- | --- | --- |
| Exact fixture output | The requested reference path reproduced the expected arrays | Production Metal ownership or packed quantization |
| BF16/INT4 emulation comparison | Candidate numerical behavior and protected-tensor policy | Packed memory footprint or kernel throughput |
| Busy-AC capture | Correctness and artifact provenance | Benchmark latency or quiet resource use |
| Cold-copy physical reads | The observed file-cache regime | A system-wide SSD cache purge |
| MPS live bytes | Currently live allocations exposed by the monitor | Total process peak or exact scratch |
| Retained driver bytes | Allocator or driver capacity remaining after execution | Scratch bytes |
| `/usr/bin/time -l` peak footprint | Kernel-reported whole-process peak | A hard whole-machine RAM cap |
| Qualified resident phase | Stable execution under the quiet-window policy | Whole-process warm-load eligibility when load is mixed |
| One successful job | A path works once | Repeated-job memory stability or cancellation safety |

The denoiser reuse load is the important example. Its load and first-execution
windows were physically mixed and were not whole-process eligible, but its
resident phases were separately qualified. The contract records that nuance
instead of flattening it into a false cached-load claim.

## Resource measurement protocol

Use a new output directory for every measurement request. The suite pauses
only identified user-owned display and photo-analysis work, leaves system
services running, records the paused PIDs, and has a restoration watchdog.

The reference commands are:

```sh
target/ig0/venv/bin/python scripts/z_image_benchmark_suite.py \
  --stages encode denoise decode \
  --out target/ig0/benchmarks/quiet-NN \
  --pause-display-and-photo-work

target/ig0/venv/bin/python scripts/z_image_benchmark_summary.py \
  target/ig0/benchmarks/quiet-NN \
  --out docs/verification/z-image-ig0-benchmarks-quiet-NN.json
```

The suite is not a substitute for fixture generation. Run capture and
validation first, and stop if a stage fixture is missing. A missing VAE decode
fixture once caused the suite to skip the intended VAE evidence; the correct
response was to generate and validate the fixture, then run a new suite.

For another model, change only the model-specific capture and benchmark
adapters. Preserve these rules:

- AC power is required for resource evidence.
- Cold copies use `F_NOCACHE` and are verified after measurement, not before.
- Disk-read, cached, and mixed regimes are recorded from physical-read
  counters.
- Any failed quiet interval excludes the affected whole-process arm.
- Exact output checks and the model's actual forward count are mandatory.
- Swap is recorded, but pre-existing swap is not attributed to the workload.
- Fresh-process peaks are stage observations and are not summed.

The measurement work was iterative, and the failures were useful evidence:

| Run | Result | Lesson retained |
| --- | --- | --- |
| `quiet-03` | Only warm text encoding qualified; cold arms were rejected by the quietness gate, denoiser warm execution was refused after host processes became active, and VAE did not run | A correctness capture is not a qualified benchmark |
| `quiet-04` | Cold text encoding qualified; the warm arm was rejected because the host `ChatGPT` process became active | The host must remain quiet for the complete suite |
| `quiet-05` | Both encoder arms and cold denoiser qualified; denoiser reuse was mixed; VAE was blocked by a missing decode fixture | Validate every stage fixture before starting resource measurement |
| `quiet-06` | Both VAE arms qualified after the missing fixture was generated and validated | Correct the fixture issue and use a new suite directory |
| `quiet-07` | Denoiser cold load and first resident phases qualified; final cold phase failed one interval; all three warm resident phases qualified, while reuse load and first execution stayed mixed | Preserve phase-level qualification instead of claiming a cached whole-process load |

The summaries are [quiet-03](verification/z-image-ig0-benchmarks-quiet-03.json),
[quiet-04](verification/z-image-ig0-benchmarks-quiet-04.json),
[quiet-05](verification/z-image-ig0-benchmarks-quiet-05.json),
[quiet-06](verification/z-image-ig0-benchmarks-quiet-06.json), and
[quiet-07](verification/z-image-ig0-benchmarks-quiet-07.json).

## Manifest contract to copy

An image install must be self-describing and independently verifiable. The
manifest must declare:

- a distinct image-generation capability and schema version;
- ordered components and stage owners;
- source revisions, licenses, and reference inventory identity;
- supported dimensions, batch, steps, forwards, guidance, and prompt limits;
- tokenizer framing, thinking flag, truncation, padding, and asset hashes;
- text-encoder extraction point, mask, epsilon, tensor inventory, and protected
  tensors;
- transformer patch order, sequence padding, timestep modulation, RoPE axes,
  block layout, packed group/nibble/scale/bias conventions, and protected
  tensors;
- scheduler equations, shift, sigma endpoints, evaluation count, guidance,
  and noise provenance;
- VAE decoder subset, latent layout, normalization, scaling, output range, and
  pixel conversion;
- every installed file's owner, size, SHA-256, storage dtype, quantization
  description, and tensor-inventory binding;
- resource envelope, qualification status, and exact limitations;
- references to the correctness and resource evidence used to validate it.

Text-only consumers must reject the image capability clearly. Installation is
not complete until all declared component files and hashes validate. The
installed packed manifest is an IG2 artifact; this contract defines the fields
and safety rules it must satisfy.

## Evidence index

- [Design and release boundaries](IMAGE_GENERATION.md)
- [Phase 0 evidence ledger](IMAGE_GENERATION_PHASE0.md)
- [Frozen resource and manifest contract](verification/z-image-ig0-resource-contract.json)
- [Pinned inputs and tensor inventory](verification/z-image-ig0-inputs.json)
- [Download and local hash receipt](verification/z-image-ig0-download.json)
- [Reference capture validation](verification/z-image-ig0-validation.json)
- [Real capture and intermediate evidence](verification/z-image-ig0-real.json)
- [Quantization policy evidence](verification/z-image-ig0-quant.json)
- [Image review record](verification/z-image-ig0-image-review.json)
- [Quiet text and denoiser evidence](verification/z-image-ig0-benchmarks-quiet-05.json)
- [Quiet VAE evidence](verification/z-image-ig0-benchmarks-quiet-06.json)
- [Quiet denoiser phase evidence](verification/z-image-ig0-benchmarks-quiet-07.json)
- [Native IG1 mutation record](verification/z-image-ig1-mutations.json)

## Starting another image model

Use this page as the process reference, not as a source of Z-Image constants.
For a new model:

1. Create `docs/<MODEL>_PHASE0.md` with the pinned inputs, component facts,
   fixtures, and evidence commands.
2. Create `docs/<MODEL>.md` only if the model needs a design page distinct
   from the general image-generation architecture.
3. Add model-specific probe, capture, validation, and benchmark scripts under
   `scripts/` with immutable revisions and new verification filenames.
4. Do not copy the Z-Image manifest values, tensor counts, forward count,
   dimensions, or memory ceilings.
5. Repeat IG0 and IG1 independently, then create a new resource/manifest
   contract before starting that model's IG2 runtime.
6. Add the model to `ROADMAP.md` only after its evidence page states the
   supported envelope and the unresolved limitations.

The reusable rule is simple: pin the model, observe the actual pipeline,
prove native intermediates, measure each stage under quiet conditions, freeze
only what the evidence supports, and delay app integration until ownership and
cancellation are proven in the production runtime.
