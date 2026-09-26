# Z-Image-Turbo: current implementation and evidence

Status: IG0, IG1, IG2, IG3, and IG4 are closed for the pinned 1024-by-1024
case. The checked image format, local and remote packers, macOS Metal backend,
CLI path, native quality/resource evidence, real cancellation, no-device
execution evidence, bounded memory, lifetime proof, Swift integration, and
macOS app image mode are complete. IG5 is intentionally open for measured
optimization work only.
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

## Recommended model and benchmark record

The default image source is the MLX export
[`andrevp/Z-Image-Turbo-MLX-4bit`](https://huggingface.co/andrevp/Z-Image-Turbo-MLX-4bit).
It is the default because it preserves the image-quality-oriented protected
tensors while reducing the published download to 6.48 GB. All four MLX
variants below are first-class inputs to the same image install and runtime
path; the selected variant is recorded in the install manifest.

| Variant | Install alias | Size | Quantization | Source |
| --- | --- | ---: | --- | --- |
| Full precision (fp16) | `z-image-turbo-mlx-fp16` | 20.54 GB | None | [`andrevp/Z-Image-Turbo-MLX`](https://huggingface.co/andrevp/Z-Image-Turbo-MLX) |
| 8-bit | `z-image-turbo-mlx-8bit` | 11.37 GB | 8-bit, group size 64 | [`andrevp/Z-Image-Turbo-MLX-8bit`](https://huggingface.co/andrevp/Z-Image-Turbo-MLX-8bit) |
| 4-bit (default) | `z-image-turbo-mlx-4bit` | 6.48 GB | 4-bit, group size 64 | [`andrevp/Z-Image-Turbo-MLX-4bit`](https://huggingface.co/andrevp/Z-Image-Turbo-MLX-4bit) |
| 2-bit | `z-image-turbo-mlx-2bit` | 4.04 GB | 2-bit, group size 64 | [`andrevp/Z-Image-Turbo-MLX-2bit`](https://huggingface.co/andrevp/Z-Image-Turbo-MLX-2bit) |

The header-only source gate pins the four published revisions at `2bit`
(`32b4e9ceb3a813485027b1ea942f199608fb8200`), `4bit`
(`9adc576198c9126874792d35569b53cf2f45a03c`), `8bit`
(`c9f70995562299b1eda9b9145a94dd7a5a1ae0d6`), and `fp16`
(`e186d7d65d66883270671fcee05324178928ea03`). It reads the transformer
safetensors headers and quantization metadata without downloading payloads,
and proves the published 2/4/8-bit U32 plus scale/bias layout and the F16
full-precision layout. This is source-shape evidence, not install or quality
parity evidence.

Run that gate with:

```sh
cargo test -p turbospark-repack --test zimage_mlx_source_network \
  --release -- --ignored --nocapture
```

A companion payload-range gate reads one small real transformer tensor from
each pinned variant, packs it through the production image adapter, and
decodes it without staging a complete multi-gigabyte source tree:

```sh
cargo test -p turbospark-image --test zimage_mlx_payload_network \
  --release -- --ignored --nocapture
```

Together these are source and payload compatibility checks. They do not
replace a complete install, image-quality, resource, or real-install Swift
image-generation gate. The complete 2-bit, 4-bit, and 8-bit install gates
below now cover the published quantized MLX variants; the FP16 install remains
open.

When a machine has enough free space for both the source staging tree and the
packed output, the opt-in installer gate exercises the real `pull-image` path
for one selected published variant and verifies its complete manifest:

```sh
TURBOSPARK_ZIMAGE_MLX_VARIANT=4bit \
TURBOSPARK_ZIMAGE_MLX_INSTALL_DIR=~/models/z-image-turbo-mlx-4bit.image.gturbo \
  cargo test -p turbospark-cli --test zimage_mlx_install_network --release -- --ignored --nocapture
```

The `2bit`, `4bit`, and `8bit` gates passed on 2026-09-18: each pinned source
was staged, packed, file-verified, and reported its expected observed-width
label in the installed manifest. The remaining `fp16` gate is intentionally
tracked separately because it is an unquantized source path, not an affine
bit-width claim; it remains disk-bound in the current checkout because its
source and packed output must coexist.

The synthetic complete-install and decode tests pass all supported affine
widths, 2, 3, 4, 5, 6, and 8 bits. The ignored native Metal packed-row parity
test also passes the complete width set. The published repositories currently
provide real full-install sources for 2, 4, and 8 bits only, so 3, 5, and
6-bit support is covered structurally until matching upstream artifacts exist.

The fresh full-install benchmark record is below. Source bytes are the exact
staged download reported by `pull-image`; install bytes are the recursive size
of the published `.image.gturbo` directory, including its manifest and
receipt; peak RSS is the macOS `/usr/bin/time -l` maximum for the installer
process. Each install was deleted immediately after its gate passed, so these
are not retained model artifacts.

| Variant | Source bytes | Packed install bytes | Install time | Installer peak RSS |
| --- | ---: | ---: | ---: | ---: |
| 2-bit | 4,040,703,000 (4.04 GB) | 5,009,974,980 (5.01 GB) | 338.942 s | 2,709,520,384 (2.71 GB) |
| 4-bit | 6,484,850,717 (6.48 GB) | 7,454,121,834 (7.45 GB) | 551.911 s | 3,617,767,424 (3.62 GB) |
| 8-bit | 11,373,145,154 (11.37 GB) | 12,342,415,237 (12.34 GB) | 603.442 s | 6,585,450,496 (6.59 GB) |

The complete-install source-plus-output working-set floors are 9.05 GB,
13.94 GB, and 23.72 GB for 2-, 4-, and 8-bit. Allowing 20% for filesystem
overhead, build activity, and temporary files, reserve at least 11 GB, 17 GB,
and 29 GB of free storage respectively. These are installation requirements,
not image-generation runtime requirements. The fp16 source was not rerun as a
full install because it is unquantized and needs more temporary storage than
the current free-space budget supports.

For a practical provisional machine recommendation, use 4-bit as the default
with 17 GB free storage and 32 GB unified memory. The 32 GB memory figure is
conservative, not an MLX runtime qualification: the existing pinned native
image resource record reached a 20,725,728,336-byte process peak, while the
MLX numbers above measure installation only. A variant-specific generation
memory floor remains open until the real MLX image resource gate is run.

The benchmark record below remains the pinned native Rust/Metal gate record;
the upstream sizes in this table are download sizes, not runtime memory
claims. The original `Tongyi-MAI/Z-Image-Turbo` Diffusers export remains the
reference source for parity and quality evidence at revision
`f332072aa78be7aecdf3ee76d5c247082da564a6`.

The repository does have image-creation benchmarks. They are kept beside the
image bring-up record rather than in the text-model table in
[`docs/BENCHMARKS.md`](BENCHMARKS.md), because the image pipeline has staged
diffusion work instead of token throughput. The pinned 1024-by-1024 record is:

| Path | Measured result | Scope |
| --- | --- | --- |
| Text encoder reference | 0.395-0.400 s resident execution; 8,969,979,152-byte peak | BF16 reference stage, quiet AC, [`quiet-05`](verification/z-image-ig0-benchmarks-quiet-05.json) |
| Nine-forward denoiser reference | 51.773-56.223 s resident execution; 17,512,599,008-byte peak | BF16 reference stage, quiet AC, [`quiet-05`](verification/z-image-ig0-benchmarks-quiet-05.json) |
| VAE reference decode | 0.933-0.938 s resident execution; 11,025,630,840-byte peak | FP32 reference stage, quiet AC, [`quiet-06`](verification/z-image-ig0-benchmarks-quiet-06.json) |
| Packed native, resident | 4,911.114 s; 7,037,387,832-byte peak | Current-source 1024-by-1024 generation; exact PNG agreement with the streamed path |
| Packed native, two-slot streamed | 4,840.856 s; 7,328,138,608-byte peak | 0.985694 streamed/resident latency ratio, 400 payload reads, 393 fenced slot reuses |
| Repeated packed jobs | 2 complete jobs; peak growth 113,557,576 bytes | Exact repeated PNGs, zero page-ins, zero swap delta, [`IG3`](IMAGE_GENERATION.md#ig3-bound-memory-and-add-dense-streaming-where-necessary) |
| Swift app image gate | 4,665.912 s for the pinned image; cancellation 21.827 s | Real-install 1024-by-1024 app path, [`IG4`](IMAGE_GENERATION.md#ig4-expose-the-runtime-to-swift-and-the-images-destination) |
| Packed native, SIMD-linear experiment | 3,608.010 s; one run | Release metadata-gate generation on Apple M4 Max with the local 8-bit MLX-affine install; 1024-by-1024, nine steps, guidance 0, seed 42 |

The SIMD-linear experiment replaced the tiled linear kernel's serial inner
product and repeated threadgroup barriers with contiguous-K SIMD lanes and a
single reduction. Its one native-run latency is 26.5% below the historical
resident native row and 22.7% below the historical Swift app result. These are
not controlled paired runs: the app and native records use different
harnesses, and this is one optimized run. The metadata gate confirms
generation and PNG completion; it does not check image quality. Keep this as
an experiment rather than a replacement frozen result. The focused Metal
parity test passes F32, local INT4, and supported MLX-affine row formats.

The first three rows are reference-stage observations and must not be read as
packed-runtime memory claims. The packed rows are full image-generation
measurements on the pinned install, and the long wall times are why IG5 is
limited to measured optimization proposals. The full evidence ledger remains
in the [IG0 resource records](verification/z-image-ig0-benchmarks-quiet-05.json),
the [IG3 runtime record](IMAGE_GENERATION.md#ig3-bound-memory-and-add-dense-streaming-where-necessary),
and the [IG4 app closure](IMAGE_GENERATION.md#ig4-expose-the-runtime-to-swift-and-the-images-destination).

## Artifact and runtime boundary

The production source is a Hugging Face MLX safetensors directory. The image
packer normalizes that source into the repository's separate `.image.gturbo`
format, preserving the variant and protected-tensor policy in the manifest.
`turbospark-model pull-image --repo OWNER/NAME@REV` streams the selected MLX
source into temporary staging before packing; `--source` remains the offline
local-directory form. The pinned Diffusers export is retained as the
independent parity reference, not as the default user-facing source.

The pinned IG2 quality record uses the repository's legacy affine INT4 linear
profile at group size 64. The default published MLX 4-bit source instead uses
MLX affine U32 rows with F16 or BF16 companions; it is not requantized into
the legacy local row format. Neither profile quantizes every tensor:
embeddings, norms, modulation, positional data, and other protected tensors
remain at higher precision, as do the VAE and other image-sensitive
operations. All four published MLX variants use the same tensor-layout adapter
and install contract; they are not separate model families. MLX is a supported
source format and does not require an MLX runtime dependency.

The adapter's affine-width contract is now explicit: MLX U32 weight planes
with F16 or BF16 `.scales` and `.biases` companions are accepted at 2, 3, 4,
5, 6, or 8 bits, with group size 64. The packed store preserves the logical
matrix shape and the native Metal path carries the bit width, companion dtype,
and group size into its decoder. The width decoder has a focused round-trip
test for every supported width and both companion dtypes. The image manifest
records the supported width set separately from the widths observed in the
packed component, and the CLI, FFI, PNG metadata, and Swift image listing
carry the selected observed-width label instead of hardcoding INT4. This is the
complete upstream `mx.quantize` width set;
upstream MLX refuses 1-bit quantization, so 1-bit is deliberately outside this
contract. The real full-install gates pass for the published 2-, 4-, and
8-bit variants; the FP16 install remains open. Quality, memory, and real-
install Swift image-generation gates remain open for the non-INT4 variants;
the closed IG2 claim still applies only to the pinned INT4 profile. The Swift
catalog/install binding surface is implemented separately from the real
image-generation gate.

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

### Current packed first-step trace

On 2026-09-15, a fresh pinned Diffusers download was verified and packed into
an 11-file `.image.gturbo` install. The matched-noise diagnostic
`packed_native_first_step_trace_localizes_divergent_boundary` passed after
576.69 seconds and kept the frozen conditioning envelope intact:

| Boundary | Relative L2 |
| --- | ---: |
| Conditioning | 0.0815408 |
| Patchification | 0.0016627 |
| Noise refiner | 0.0509079 |
| Main transformer, block 0 | 0.0688876 |
| Main transformer, block 15 | 0.0762979 |
| Main transformer, block 16 | 0.0878091 |
| Main transformer, block 20 | 0.1795987 |
| Main transformer, block 24 | 0.4174181 |
| Main transformer, block 28 | 0.9539717 |
| Main transformer, block 29 | 1.0846142 |
| Velocity | 0.5992641 |
| Scheduler latent | 0.0392788 |

This rules out patchification and the noise-refiner sequence as the first major
source of drift. The error is small through block 16, then accumulates across
the later main-transformer blocks and crosses the frozen `0.923` rollout
envelope by block 28. It does not yet distinguish packed INT4 error from
BF16-versus-F32 accumulation or a repeated layout/dispatch error. The next
probe must compare the packed and F32/BF16 paths inside the block recurrence;
the quality envelope remains frozen.

### IG2 recurrence controls and reproducible setup

The next controls used the same pinned revision and the same captured lighting
noise. A Diffusers reference denoise with `--quantize` applied the repository's
group-64 affine INT4 policy while retaining BF16 execution. Against that
control, the native packed trace measured conditioning `0.0042043`,
noise-refiner `0.0521734`, main transformer `1.0849981`, block 0 `0.0636551`,
block 16 `0.0836585`, block 24 `0.4202046`, block 28 `0.9507074`, and block 29
`1.0849981`. The ordinary BF16 reference comparison was effectively unchanged.
This rules out a simple INT4 source-policy or conditioning provenance cause.

The packer now stores unquantized F32 source tensors as BF16, matching the
reference model's loaded transformer dtype while leaving the 238 eligible
projection matrices as `INT4_AFFINE`; the tested transformer component had
283 BF16 tensors and 238 INT4 tensors. A fresh install with that storage
alignment reproduced the original curve, including block 29 `1.0846142`. A
diagnostic GPU round-to-BF16 pass at each completed block changed the final
value only to `1.0834699`, so it was removed. The remaining candidate is
intra-block native activation or accumulator precision, not a block-boundary
conversion. The frozen `0.923` envelope was not widened.

Two additional seeded controls narrowed that candidate. Forcing a CPU wait after
every image primitive with `TURBOSPARK_IMAGE_FORCE_SYNC_DISPATCH=1` reproduced
the baseline checkpoints exactly, including block 28 `0.953971744` and block 29
`1.08461416`; repeated command-buffer completion is therefore not the cause.
Disabling Metal fast math with `TURBOSPARK_METAL_PRECISE_MATH=1` changed only
block 28 to `0.953971624` and block 29 to `1.08461368`, which is sub-ppm. The
native image tensors and shader outputs are FP32, while the captured block
inputs and outputs carry `torch.bfloat16` source dtype. The next experiment must
therefore emulate BF16 storage or otherwise match the reference's intra-block
activation and reduction contract. The `0.923` envelope remains frozen.

The first bounded intra-block experiment is now available through
`MetalImageBackend::intra_block_trace`. It accepts the captured input and AdaLN
embedding for one 1024x1024 main-transformer block, then reports labeled
snapshots for modulation, Q/K/V projection, RoPE, attention, output projection
and residual, and every FFN operation. With `round_bf16=true`, a Metal
round-to-nearest-even BF16 storage pass is inserted after each snapshot before
the next operation. On block 28 of the pinned packed install, the isolated
block-output comparison moved from `0.36354998` to `0.36287159` relative L2
against `block_28_output.npy`. That small improvement is evidence that blanket
per-operation BF16 storage is not sufficient by itself; it does not advance the
quality gate or justify changing the `0.923` envelope. The ignored test is
`packed_native_intra_block_bf16_trace_reports_operation_deltas` in
`crates/image/tests/metal_parity.rs`.

The matching Diffusers operation trace is generated by
`scripts/z_image_intra_block.py` from the same pinned block-28 capture and
quantization policy. It exposed a concrete adjacent-RoPE layout error: the odd
lane used `even*cos + odd*sin` instead of the required `even*sin + odd*cos`.
After the one-line Metal fix, native versus reference operation error fell from
about `0.9` at Q/K RoPE to `0.004`, and the isolated block-28 output error fell
to `0.08484` (BF16 storage control `0.08495`). The seeded full first-step trace
then measured block 28 at `0.43851` and block 29 at `0.43361`, versus the prior
`0.95397` and `1.08461`. The complete matched-noise quality/VAE gate and the
dedicated PNG metadata gate now pass on the pinned install.

The frozen-latent VAE comparison fixture was subsequently regenerated from
the pinned Diffusers VAE. The first 180-second observation was too short for
the production-shape decoder and was stopped before a result. A later
stage-isolated run completed every decoder boundary in 395.13 seconds, and the
non-instrumented pinned arm passed in 370.37 seconds with max absolute error
`4.7907233e-6` and relative L2 `3.1853588e-7`, inside the frozen limits. The
native VAE now shares attention score work across eight queries, reduces
GroupNorm statistics cooperatively, and reuses convolution weights across
eight spatial outputs. The quiet resource oracle's cold arm completed in
6,515.892 seconds with PNG relative L2 `0.4284977`, zero page-ins, and a
`20,725,728,336`-byte process peak dominated by the VAE. Its corrected
warm-only arm also passed in 4,572.05 seconds with PNG relative L2
`0.4284977`, zero page-ins, unchanged swap, and a
`19,874,055,248`-byte peak. Real CLI cancellation and sparse real-install
missing-component refusal pass. The no-device integration test also executes
on Linux ARM64 and passes the
unsupported-platform refusal before model I/O.

The pinned setup used for this work is reproducible with:

```sh
mkdir -p target/ig0
python3 -m venv target/ig0/venv
target/ig0/venv/bin/python -m pip install -r scripts/z_image_reference_requirements.txt
target/ig0/venv/bin/python scripts/z_image_download.py --model "$PWD/target/ig0/model"
mkdir -p ig2-work
cp -al target/ig0/model ig2-work/model
cargo run --release -p turbospark-cli --bin turbospark-image -- pack \
  --source "$PWD/ig2-work/model" \
  --output "$PWD/ig2-work/z-image-turbo-bf16-control.image.gturbo" \
  --model-id Tongyi-MAI/Z-Image-Turbo \
  --model-revision f332072aa78be7aecdf3ee76d5c247082da564a6
```

The downloader verifies the text encoder, transformer, and VAE payload hashes
before packing. Keeping the verified source outside `target` avoids losing a
31 GB source tree when the Cargo build cache is reclaimed.

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
a focused GQA and causal-mask parity fixture, and the isolated
production-shape VAE parity arm now passes its frozen pixel limits. The
complete matched-noise nine-step gate, PNG metadata gate, cancellation path,
quiet resource oracle, remote install path, and Linux ARM64 no-device
integration test all pass, so IG2 is closed. Keep the CPU backend as the
diagnostic oracle, not as an unrecorded fallback. See
[IMAGE_GENERATION.md](IMAGE_GENERATION.md#ig2-handoff-checklist) for the
file-level checklist and stop conditions.

### IG3: memory and lifetime proof

IG3 proves resident versus streamed behavior, bounded buffer lifetimes,
refusal before execution, and repeated-job memory stability. It accounts for
weights, conditioning, latents, activations, scratch, staging, in-flight GPU
work, and allocator retention.

The runtime admission and lifetime seams are now present and unit-tested:
`ImageMemoryPlan` plus `generate_with_memory_budget` refuse an infeasible
largest-block/workspace lower bound before backend execution,
`ImageWorkTracker` provides cancellation-safe consumer accounting, and
`SequentialImageSlots` proves no early reuse with a two-slot live-byte bound.
These are implementation contracts backed by the measured IG3 closure below.
The public generation wrapper also invokes the backend idle hook after cancellation, so
registered synchronous I/O and staging consumers are drained before the
cancelled request returns. The pinned Metal oracle below qualified repeated
jobs, repeated denoise cycles, a real resident-versus-streamed comparison, and
the latency tradeoff.

The resource oracle now repeats complete jobs and matched-noise denoise cycles.
It records the separate memory-plan categories plus peak and idle
`phys_footprint`, managed allocations, physical reads, and swap deltas.
Repeated PNGs and final latents must agree exactly, and repeated idle footprint
growth defaults to a 256 MiB bound.

The pre-execution plan gate passed on 2026-09-17 for the pinned artifact, with
6,651,207,110 resident payload bytes, 214,918,144 two-slot bytes, and a
107,459,072-byte largest block. The current-source resident-versus-streamed
gate then passed with exact PNG agreement. Resident latency was 4,911,114 ms
and streamed latency was 4,840,856 ms, a streamed/resident ratio of 0.985694.
Resident peak `phys_footprint` was 7,037,387,832 bytes; streamed peak was
7,328,138,608 bytes. The streamed path measured 214,918,144 slot bytes, 400
payload reads totaling 33,298,002,054 bytes, and 393 fenced slot reuses. The
earlier fixture-missing, Metal out-of-memory, and first-step stall runs remain
retained as failed qualification attempts and do not establish a
machine-memory requirement.

The repeated warm-only oracle also passed on 2026-09-17. Complete jobs 1 and 2
grew peak `phys_footprint` by 113,557,576 bytes and idle footprint by
83,476,552 bytes, with 7,784 managed allocations for each job, zero page-ins,
zero swap growth, and identical `0.4284977` PNG relative L2. Matched-noise
denoise cycles 1 and 2 each completed nine forwards over 262,144 finite latent
values and asserted exact final-latent equality. Peak footprint grew by
15,663,176 bytes, while idle footprint fell by 52,035,680 bytes; both cycles
reported 7,138 managed allocations, zero page-ins, and zero swap delta. The
complete oracle passed in 18,839.90 seconds.

With the repeated-job and denoise evidence, the lifetime-safe cancellation
seams, pre-execution budget refusal, and resident/streamed gate all pass for
the pinned artifact. IG3 is closed. VAE tiling, prefetch, and allocator reuse
remain separate IG5 work. Swift image integration is closed under IG4 and is
documented below.

Cancellation must stop future scheduling and wait for outstanding I/O and GPU
consumers before freeing or reusing buffers. VAE tiling is a conditional
optimization. Add it only if measured VAE resource use requires it, then
re-run geometry, seam, quality, and memory gates.

Handoff after IG3: keep the `0.923` quality envelope fixed, separate
`phys_footprint` from the managed allocation ledger and driver-retained
capacity, and leave VAE tiling, prefetch, and allocator reuse for IG5. IG4 app
and Swift integration is closed: the canonical top-level `Images` destination
provides `Create` and `Gallery` tabs, profile-scoped PNG artifacts, thumbnails,
and a previous/next carousel. The supported envelope remains one 1024-by-1024
image, batch one, nine scheduler steps, guidance zero, and native Metal
execution.

### IG4: app and ABI integration

IG4 exposes the already-proven runtime through the C ABI and Swift wrapper. It
adds the top-level `Images` destination, prompt-first `Create` and `Gallery`
tabs, preview, save, regenerate, profile-scoped persistence, thumbnails, a
previous/next carousel, and serialized heavyweight image jobs. The app uses
the same conditioning, scheduler, quantization, cancellation, and output
implementation as the CLI. Image jobs remain transient until saved, while
saved request metadata makes regeneration reuse the original seed and options.

The Swift catalog now exposes the pinned image rows through
`TurboSparkCatalog.imageAvailable()` and installs them through
`TurboSparkCatalog.installImage(_:)`. The app's image model picker uses those
bindings, reports staged download progress, and routes a completed
`.image.gturbo` install into the existing native `TurboSparkImageSession`.
This proves the catalog and ABI surface without retaining a multi-gigabyte
fixture; the real image-generation Swift gate remains opt-in through
`TURBOSPARK_TEST_IMAGE_MODEL`.

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
