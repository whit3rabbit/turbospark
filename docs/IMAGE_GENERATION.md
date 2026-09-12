# Native image generation: Z-Image-Turbo

Status: IG0 in progress, 2026-09-10. [Phase 0 evidence](IMAGE_GENERATION_PHASE0.md)
records pinned inputs, real-image captures, and component comparisons.

Current IG1 evidence status:

- Conditioning, scheduler, and capture manifests are now validated by locked tests,
  including schema, shape, and checksum checks for fixture provenance.
- Quantization candidate policy evidence and scheduler-step parity remain the
  remaining IG1 gates before runtime assembly.

No image-generation runtime, CLI command, catalog alias, or app mode is
implemented by this document. The open work is tracked in
[ROADMAP](../ROADMAP.md). This page owns the design, gates, and rationale;
the roadmap owns the remaining task checklist.

## Direction and first release

Build Z-Image-Turbo text-to-image inference in Rust with the existing Metal
backend. Deliver a quantized CLI pipeline first, then expose that same
runtime through the C ABI and Swift package to an explicit image mode in
chat. External engines are correctness references and benchmark tools,
not production subprocesses or a new MLX dependency.

The first release generates one PNG per request from an explicit prompt,
with a resolved seed, dimensions, progress, cancellation, and reproducibility
metadata. The initial reference case is 1024 by 1024. Phase 0 must establish
supported dimension constraints and a measured memory/latency envelope
before promising that resolution on a particular Mac.

Image understanding remains the separate [vision pipeline](VISION.md).
Z-Image needs a text encoder, diffusion transformer, scheduler, and VAE;
its repeated dense block scans are not MoE routing. Do not retrofit it into
the autoregressive text runner or treat a working vision tower as a working
diffusion model.

Deferred: image editing, image-to-image, negative-prompt controls, LoRA,
batch generation, agent tool invocation, HTTP image endpoints, PISA,
approximate timestep reuse, and concurrent heavyweight text/image execution.

## What this engine already provides

| Existing foundation | Reuse and boundary |
| --- | --- |
| [PreadExpertStreamer](../crates/streaming/src/pread_streamer.rs), aligned slots, read pool, packed layout | Reuse bounded loading and ownership. Dense sequential scheduling has different reuse behavior from an expert cache. |
| [Vision block streaming](../crates/runtime/src/vision/mod.rs) | Existing precedent for serving sequential blocks through the expert store. Its two slots and per-block completion waits do not establish overlapped diffusion execution. |
| [Expert residency](EXPERT_RESIDENCY.md) | Keep the distinction between pinned slots and OS-managed mapped pages. A lower counted footprint is not a whole-machine RAM guarantee. |
| [Load guards](LOAD_GUARD.md) | Extend admission/accounting for image allocations; do not introduce a second competing budget policy. |
| [Model installation](MODELS.md) and [install format](GTURBO.md) | Reuse discovery, validation, receipts, and bounded payload handling. Add an image-pipeline identity and complete component inventory. |
| [Swift bindings](SWIFT_BINDINGS.md) | Keep inference in process and share the runtime with the CLI. Add an image session contract rather than overloading token events. |

The production Rust GPU backend is Metal. MLX array ownership experiments
are not prerequisites for this direction. Existing quantized kernels are
reuse candidates, not proof that a community checkpoint's packing or shapes
are compatible. Likewise, a text-generation Qwen runner is not automatically
the hidden-state encoder the image pipeline requires.

MoE demand loading already exists, and [KV quantization](TRUBOQUANT.md)
already ships as an opt-in text feature. Neither needs rebuilding to unlock
image generation. Read [expert routing](EXPERT_ROUTING.md) before proposing
prefetch and [batched prefill](BATCHED_PREFILL.md) before carrying text-side
I/O assumptions into a new workload.

## Implementation sequence and decision gates

### IG0: Pin the model contract and reference evidence

Follow the evidence-first approach in [new model bring-up](NEW_MODEL.md),
adapted to diffusion rather than forcing image configuration into text-only
architecture fields.

- Pin the canonical checkpoint, tokenizer, component configs, and reference
  implementation revisions. Record licenses and every required component.
- Inventory tensor roles, dimensions, dtypes, normalization, positional
  encoding, attention masks, modulation, patch packing, and VAE scaling.
  Capture prompt framing, hidden-state selection, padding, and truncation.
- Specify noise initialization and scheduler timesteps, shifts, update
  equations, and actual transformer evaluation count. Equal integer seeds
  across backends do not imply equal initial noise.
- Compare the required operators against existing CPU/Metal implementations.
  Record reuse, adaptation, or missing implementation for each operator.
- Select the production quantized representation from verified tensor layout
  and kernel compatibility. Target four-bit supported linear weights; record
  exceptions for sensitive or unsupported tensors. Preserve compatible
  packed payloads and avoid repeated dequantization/requantization.
- Record hardware, RAM, macOS, storage, tensor bytes, reference revisions,
  target dimensions, and candidate memory budgets. Establish explicit
  numerical tolerances and image-quality criteria from reference evidence.

Gate: a reproducible component contract, reference fixtures, operator gap
table, chosen quantization layout, and target memory/latency envelope. Do not
freeze a disk format or advertise a memory minimum before this evidence.

The [official Turbo example](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo)
requests nine scheduler steps, guidance zero, and 1024-by-1024 output; its
comment says eight transformer forwards. The pinned IG0 Diffusers probe
observes NINE forwards for that invocation. See [the discrepancy and exact
schedules](IMAGE_GENERATION_PHASE0.md#noise-scheduler-and-the-model-card-discrepancy).
Record both steps and forwards; do not inherit the comment as a contract.

### IG1: Validate native components

Implement the text encoder, diffusion block flow, scheduler, and VAE in
independently testable pieces. Reuse kernels only when their mathematical
contracts match. Use portable CPU references for new operators and real
Metal parity tests for their implementations.

Capture conditioning, identical initial latents, block outputs, scheduler
updates, final latents, and decoded pixels from a pinned reference. Validate
components before assembling the full pipeline. Higher-precision reference
work can use bounded fixtures or one component at a time; it need not keep
the complete unquantized pipeline resident on the development Mac.

For v1, IG1 must produce at least one deterministic CPU reference fixture set
for each component:

- Conditioning fixture: padded caption rows, attention masks, hidden state slice,
  and three-axis position IDs from fixed seeds.
- Transformer fixture: nine-forward latent evolution for a fixed prompt with
  exact timesteps and scheduler settings.
- Scheduler fixture: sigma sequence, sigma shifts, transformer scaling, and
  Euler update outputs.
- VAE fixture: decoded pixel tensor and final postprocessing outputs for the same
  final latent.

Each fixture must include mutation checks for boundary, numeric, and
truncation failure modes, with explicit tolerances scoped to the tested scope.

Current evidence gates for IG1 include:

- token and framing parity fixtures in `scripts/test_z_image_tokenizer_parity.py`
- capture-manifest contracts and provenance checks in
  `scripts/test_z_image_capture_contracts.py`
- component capture and mutation checks in `scripts/test_z_image_evidence.py`

Gate: component agreement within IG0's stated tolerances, with discrepancies
explained at the first divergent intermediate rather than judged only from
the final picture.

### IG2: Ship the staged quantized CLI path

Add an image-specific runtime session and extend model intake to validate
the full pipeline before marking installation complete. Text-only consumers
must reject image installs with a clear capability error. Keep existing
text manifests and commands compatible; a new image manifest schema is an
IG0 deliverable, not an invented extension of every text-family field.

Execute with explicit stage ownership:

```text
load text encoder -> materialize conditioning -> complete GPU work -> release encoder
load transformer -> denoise, retaining conditioning and latents -> release transformer
load VAE -> decode final latent -> write image -> release stage resources
```

Start with the quantized transformer resident if it fits. Avoid a persistent
conditioning cache in v1; retaining one request's conditioning is sufficient.
Serialization alone does not save memory if an idle text session still owns
weights: the app integration must release incompatible resident sessions
before admitting the image job.

Gate: an installed quantized model produces a valid PNG through the unified
CLI, with progress, cancellation, settings metadata, and measured stage peaks.
If the transformer cannot fit the target budget, IG3 becomes a release
prerequisite rather than reporting that budget as supported.

### IG3: Bound memory and add dense streaming where necessary

Account in bytes for component weights, conditioning, latents, activations,
scratch, staging, in-flight buffers, and retained allocator capacity. Reserve
headroom for macOS and allocations outside the managed ledger. Shared views
of one allocation count once; genuine staging copies count separately.

Reuse the existing streamer for bounded sequential block reads. Compare
resident execution, two-slot streaming, and a deliberately retained subset.
A small generic LRU cache can thrash across repeated full denoising scans.
Preserve dense attention and all required blocks. Add prefetch only after
the synchronous lifetime contract and memory accounting are proven.

Cancellation requests stop further scheduling, then wait for outstanding
read/GPU consumers before freeing or reusing their buffers. The minimum
feasible budget includes the largest required block and its workspace;
refuse before execution if that lower bound cannot fit. Consider VAE tiling
only when its measured stage peak requires it, with seam/quality validation.

Gate: resident-versus-streamed agreement, bounded live storage over repeated
jobs and denoising steps, no early slot reuse, and a useful measured latency
tradeoff at each advertised budget. A managed budget does not claim a hard
cap on whole-process or system physical memory.

### IG4: Expose the runtime to Swift and image mode in chat

Extend the C ABI and Swift wrapper with an image session, request, progress,
result, cancellation, and explicit buffer ownership. Keep the existing text
session ABI behavior intact. The CLI and app must use the same conditioning,
scheduler, quantization, and output-generation implementation.

Add a chat-scoped image mode with compatible installed-model selection,
prompt, dimensions, seed, stage progress, Stop, result preview, Save, and
Regenerate. Use the submitted prompt directly in v1; do not silently run a
text-model prompt enhancer or include the entire conversation as conditioning.
Regenerate reuses the recorded request and seed; changing the seed is explicit.

Capture the originating chat ID and job identity when submitting. Switching
chats cannot redirect events or results. Serialize heavyweight jobs across
chat, background text work, and the in-process server. Show why a job is
waiting, and release idle text weights/KV before image admission when needed;
preserve conversation history for later text-session reconstruction.

Persist generated files under [AppStorageRoot](../swift/docs/SWIFT_PROFILES.md)
with profile-scoped chat ownership; downloaded weights and install receipts
remain shared. Store relative artifact references and request metadata in
history, not embedded image bytes. Old chats decode with text behavior by
default. Handle missing files visibly and never restore an interrupted job as
completed. Profile switching and shutdown must respect active image work.

Gate: the same request reaches the same native pipeline from CLI and app;
progress and cancellation remain responsive; history, exports, profile
isolation, and interrupted-job recovery work after relaunch. Validate the
real app bundle, localization, and accessibility before release.

### IG5: Optimize only measured bottlenecks

Investigate bounded next-block prefetch, retained blocks, or VAE tiling only
after IG3 measurements identify the limiting stage. PISA and approximate
timestep reuse remain separate proposals requiring quality and end-to-end
benefit on the quantized pipeline. Faster attention is not automatically
useful when storage dominates. No MoE cache-policy rewrite is required.

## Proposed public surfaces

This command and alias are design targets, not available commands:

```sh
turbospark image generate \
  --model z-image-turbo \
  --prompt "A lighthouse in winter" \
  --seed 42 \
  --width 1024 --height 1024 \
  --output lighthouse.png
```

Use the existing model-management infrastructure to install and discover
image-capable models. Add shipped syntax to [CLI](CLI.md) only when the
parser and implementation exist. Output is PNG, progress goes to stderr,
and successful completion reports the resulting path. Resolve and record a
seed when omitted. Require an output path and refuse accidental overwrite.
Publish a result only after encoding succeeds; remove incomplete output on
failure or cancellation. Record prompt, seed, dimensions, model/component
revisions, quantization, scheduler settings, actual evaluations, and engine
revision in PNG metadata so exported images retain their generation context.

The shared runtime interface needs a model/session handle, prompt and image
options, resolved settings, phase progress, cancellation, owned pixel output,
and metadata. Encoding/export can be shared without making the core runner
own app storage paths. Specify C allocation/free and callback-thread rules
before exposing these types through the ABI; do not reuse text token event
numbers for image progress. Rust, CLI, and Swift validation must agree.

## Verification and measurement

Keep three comparisons independent: higher precision versus quantized for
quantization quality, reference versus native for implementation correctness,
and resident versus streamed quantized for residency correctness.

- Unit/component gates: checked shapes and byte arithmetic, malformed and
  incomplete component inventories, quantization metadata, prompt encoding,
  scheduler edge cases, and CPU/Metal parity. Mutation-check new tests under
  [the repository testing rules](TESTING.md).
- Real-model gates: a fixed prompt suite covering composition, typography,
  fine detail, and challenging lighting; intermediate numerical comparisons
  plus visual review. Pixel similarity alone is not a quality verdict.
- Lifecycle gates: cancellation during loading, encoding, denoising, VAE,
  and export; repeated jobs; failed reads; budget refusal; no buffer reuse
  before GPU completion; no cumulative retained-weight growth.
- CLI gates: help, invalid options and dimensions, missing model/components,
  resolved seed, valid PNG and metadata, output failure, overwrite refusal,
  and cancellation exit behavior.
- App gates: mode/model compatibility, chat switching mid-job, Stop and
  retry, saved settings, relaunch, missing artifacts, profile isolation,
  legacy chat decoding, and serialization with other inference consumers.
  Follow [localization](../swift/docs/SWIFT_LOCALIZATION.md) and
  [accessibility](../swift/docs/KEYBOARD_SHORTCUTS.md) conventions.

For implementation handoff run the required Rust workspace gates. FFI changes
also need `make swift-lib` and `make swift-test`; app changes need Swift build
and tests plus inspection of a built app bundle. Shared text decode/kernel
changes additionally run all applicable per-family real-model smoke,
memory-oracle, and quality gates. New image gates supplement those tests.

Measure one pipeline at a time on a recorded machine. Report per-stage
latency, total seconds/image, actual transformer evaluations, peak process
footprint, managed allocations, retained buffers, physical reads, and swap
growth. Separate cold storage from warm filesystem cache and record competing
processes. Freeze a memory-versus-latency curve, not an isolated RAM headline.

## References and evidence boundary

- [Canonical Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo):
  model/configuration source and official generation example.
- [MFLUX Z-Image implementation](https://github.com/mflux-community/mflux/tree/main/src/mflux/models/z_image):
  Apple Silicon reference for operator and intermediate comparisons.
- [Diffusers Z-Image pipeline](https://github.com/huggingface/diffusers/tree/main/src/diffusers/pipelines/z_image):
  scheduler and pipeline reference, pinned in the Phase 0 evidence.
- [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp):
  optional independent GGUF/Metal baseline. Verify current component,
  segmentation, and disk-residency support before selecting benchmark flags.

Upstream links above are discovery entry points, not pinned implementation
dependencies. [Phase 0 evidence](IMAGE_GENERATION_PHASE0.md) supplies exact
comparison revisions, the limited visual review, and resource observations.
Qualified performance limits and a minimum RAM figure remain open.
