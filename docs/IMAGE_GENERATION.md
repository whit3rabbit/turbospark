# Native image generation: Z-Image-Turbo

Status: IG0 resource evidence and the IG0 resource/manifest contract are
closed for the reference envelope. IG1 native component parity is closed for
the available fixtures: the full-width checkpoint block, complete
nine-step DiT rollout, and real 1024-by-1024 VAE decode gate pass their frozen
contracts. The optional raw-pixel arrays were not present in this checkout, so
the VAE test's conditional pixel comparisons were not exercised.
[Phase 0 evidence](IMAGE_GENERATION_PHASE0.md) records pinned inputs,
real-image captures, and component comparisons.
The reusable bring-up process and the lessons from this model are summarized
in [ZIMAGE_TURBO.md](ZIMAGE_TURBO.md).

Quiet-AC resource evidence now exists for all three reference stages. The
encoder cold and reuse arms, both VAE arms, and the denoiser cold load and
resident phases qualify. The denoiser reuse load and first-execution windows
were physically mixed and not whole-process eligible, but all three resident
reuse phases passed exact nine-forward and quiet-window checks. These are
reference-stage observations, not a packed INT4 runtime measurement. The
resource records are [quiet-05](verification/z-image-ig0-benchmarks-quiet-05.json),
[quiet-06](verification/z-image-ig0-benchmarks-quiet-06.json), and
[quiet-07](verification/z-image-ig0-benchmarks-quiet-07.json).
The frozen resource and install-manifest contract is
[z-image-ig0-resource-contract.json](verification/z-image-ig0-resource-contract.json).
It limits the first production envelope to 1024-by-1024, batch one, nine
steps, nine transformer forwards, guidance zero, and one heavyweight stage
resident at a time. Its stage budgets are inclusive process ceilings from the
BF16 reference runs, not a packed INT4 runtime result.

Current IG1 evidence status:

- Conditioning, scheduler, and capture manifests are validated by locked tests,
  including schema, shape, and checksum checks for fixture provenance.
- Quantization candidate policy evidence is closed: 18 locked Python tests pass
  under the reference venv.
- Native scheduler-step parity is implemented and verified in crates/image:
  exact schedule bit-parity for (1,1), (8,8), (9,9) against contracts JSON,
  exact timesteps/sigmas capture agreement, and Euler step parity across
  all nine latent steps to within 1e-6 float roundoff.
- Native conditioning is implemented and verified in crates/image: exact prompt
  framing string parity across all seven test prompts, exact tokenization IDs
  and attention mask parity against captured arrays, and verified untruncated/retained
  token bounds.
- Native text encoder forward pass is implemented in FP32 on CPU: extracting
  layer 34 output (block 35 of 36 pre-final-norm, hidden_states[-2]) achieves
  8.69e-3 to 8.86e-3 relative L2 agreement against captured BF16 MPS conditioning
  across lighting, empty, and unicode captures.
- The portable crate now has a checkpoint-backed FP32 DiT reference that streams
  one decoded transformer block at a time through two noise-refiner, two
  context-refiner, and thirty main blocks. It preserves learned-pad-token
  attention, exact 3-axis RoPE positions, timestep modulation, the final
  projection, and the output sign. Batched matrix projections and independent
  token work use bounded CPU parallelism while each dot product remains FP32.
- The full-width 64-token checkpoint block now passes the frozen `3e-5`
  absolute and `1e-6` relative-L2 gates against both pinned references. The
  fix uses a tree-shaped FP32 RMSNorm reduction and applies linear bias after
  the dot product, matching the reference operation boundaries.
- The VAE reference now validates convolution and normalization inputs and can
  convert decoded `[3,H,W]` floats into verified interleaved RGB PNG bytes.
- Sixteen native assertions are mutation-checked in
  `z-image-ig1-mutations.json`, including RoPE pairing, patch layout, AdaLN
  gating, FP32 reduction order, affine bias order, batched projection scatter,
  the captured-input checkpoint ceiling, cumulative BF16 envelope, and RGB
  conversion. The real VAE gate's captured latent geometry mutation also
  fails in isolation.
- All nine captured-input scheduler updates pass the unchanged `0.02` local
  ceiling. The maximum local scheduler relative L2 is `3.13403807e-3`; the
  transformer-output values are recorded as diagnostics because the captured
  BF16 reference already reaches `3.08770984e-2` on update 1.
- The complete rollout measures accumulated relative L2 values from
  `2.02384288e-3` at update 1 through `1.95015728e-1` at update 9. The maximum
  measured error is `1.95015728e-1`; applying the preselected upward-to-0.001
  rule freezes the cumulative BF16 envelope at `0.196`. The widening is
  evidence-derived and applies only to accumulated FP32 CPU versus BF16 MPS
  state drift, not to local timestep math.
- The real 1024-by-1024 VAE gate passes in 4695.11 seconds and produces the
  required `[3,1024,1024]` decoded tensor. The optional `decoded_pixels.npy`
  and `mlx_decoded_pixels.npy` files are absent, so this run exercises the
  real decode and geometry contract but not the conditional pixel assertions.
  The independent pinned MFLUX comparison remains recorded in
  `verification/z-image-ig0-vae.json` with max absolute error
  `3.764033317565918e-05` and relative L2 `1.3302375354987454e-06`, below the
  frozen `6e-5` and `3e-6` limits.

The current IG2 implementation includes a checked packed component format,
image-manifest admission validation, a staged runtime seam, a macOS-only
Metal backend, image-specific MSL operators, and an explicit CPU reference
CLI path. The local `turbospark-image pack` command assembles the tokenizer,
text encoder, transformer, scheduler, and VAE source tree into an atomic
install and emits the existing `verified-install.json` receipt. The model CLI
also has an explicit `pull-image` route that records image installs separately
from text catalog rows. Native packed parity and resource tests are present as
opt-in gates. This does not close IG2: no complete packed install has passed
the real-model quality, memory, latency, and cancellation evidence gates yet.
The remaining work is tracked in
[ROADMAP](../ROADMAP.md). This page owns the design, gates, and rationale;
the roadmap owns the remaining task checklist.

### Artifact support boundary

The current runtime does not load arbitrary Hugging Face MLX image exports
directly. The intended source is a pinned Diffusers-style Z-Image-Turbo
export, which `turbospark image pack` converts into the repository's separate
`.image.gturbo` install format. `turbospark-model pull-image` currently accepts
that source as a local directory; it does not yet download an image model by
Hugging Face repository ID.

The first IG2 production profile is this repository's affine INT4 linear
format at group size 64. It is not an all-tensor INT4 claim: embeddings,
normalization, modulation, positional, and other protected tensors remain at
higher precision, as do image-sensitive operations such as the VAE. The
already-quantized [`andrevp/Z-Image-Turbo-MLX-4bit`](https://huggingface.co/andrevp/Z-Image-Turbo-MLX-4bit)
and [`uqer1244/MLX-z-image`](https://huggingface.co/uqer1244/MLX-z-image) exports
are close candidates by model and bit width, but their MLX tensor layout needs
an explicit adapter and packed parity evidence. The 2-bit, 8-bit, and
full-precision variants are not current IG2 runtime profiles. Do not register
any of these as ordinary text `Mlx` catalog rows.

The image crate is intentional. `turbospark-image` owns the image graph,
image-specific install schema, packed storage, scheduler, VAE, and the native
Metal backend. It shares only matching context, pass, and resident-buffer
contracts with `turbospark-gpu`. The same image crate is the future shared
runtime for the CLI and the C ABI/Swift package; no second image GPU crate is
needed at this stage.

The memory strategy is stage ownership first: text encoder, transformer, and
VAE do not remain resident together. Packed linear weights stay in the mapped
component payload and are decoded by Metal at use rather than expanded into a
full-precision copy. The current wrappers are still correctness-first and
create many operation-level command buffers and temporary buffers, so the
native packed memory envelope is not frozen. Pooled scratch, activation reuse,
fewer command-buffer boundaries, safe BF16/FP16 storage for non-INT4 tensors,
and repeated cold/warm measurements are the next memory steps.

IG0 is now closed by the measured reference contract. Exact MPS operator
scratch is not observable, so the contract uses an inclusive non-parameter
process budget and explicitly does not call driver-retained bytes scratch or
claim a minimum whole-machine RAM size. Repeated full-pipeline stability,
resident-versus-streamed ownership, and cancellation lifetime proof remain IG3
work. IG2 implementation has begun; app work remains IG4 after IG3 establishes
bounded lifetimes.

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
table, chosen quantization candidate, and resource/manifest contract. This gate
is closed by [the frozen IG0 contract](verification/z-image-ig0-resource-contract.json).
The packed representation and its measured Metal behavior remain IG2 gates;
no minimum whole-machine RAM claim is made.

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
- the opt-in full-width Rust checkpoint block in
  `crates/image/tests/transformer_math_parity.rs`
- the opt-in nine-step Rust DiT evolution gate in
  `crates/image/tests/pipeline_parity.rs`
- the opt-in full Rust VAE decode gate in `crates/image/tests/vae_parity.rs`

The completed DiT gate uses the pinned ignored artifacts under `target/ig0`.
Run the captured-input diagnostic with:

```sh
cargo test --release -p turbospark-image --test pipeline_parity -- \
  --ignored --nocapture test_z_image_all_steps_from_captured_input_parity
```

This records transformer-output and captured-input scheduler-output relative
L2 for every update. The complete rollout gate then records accumulated error
for every update and asserts the frozen `0.196` cumulative BF16 envelope:

```sh
cargo test --release -p turbospark-image --test pipeline_parity -- \
  --ignored --nocapture test_z_image_full_nine_step_checkpoint_parity
```

The capture contains nine actual forwards. The nine update fixture pairs are
`initial_noise.npy -> latent_00.npy`, `latent_00.npy -> latent_01.npy`,
`latent_01.npy -> latent_02.npy`, `latent_02.npy -> latent_03.npy`,
`latent_03.npy -> latent_04.npy`, `latent_04.npy -> latent_05.npy`,
`latent_05.npy -> latent_06.npy`, `latent_06.npy -> latent_07.npy`, and
`latent_07.npy -> latent_08.npy`. `latent_08.npy` is byte-identical to
`final_latents.npy`; the latter remains checked as the final-latent alias.

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

The offline packer accepts this source tree shape and refuses to overwrite its
destination:

```text
source/
  tokenizer/                         tokenizer.json and config/assets
  text_encoder/model.safetensors.index.json and shards
  transformer/diffusion_pytorch_model.safetensors.index.json and shards
  scheduler/scheduler_config.json
  vae/diffusion_pytorch_model.safetensors.index.json and shards
```

The packer also accepts the internal `scheduler/config.json` and
`vae_decoder/` names used by synthetic fixtures, but standard Diffusers names
are preferred for a pinned export.

```sh
turbospark image pack \
  --source /path/to/Z-Image-Turbo \
  --output /path/to/z-image-turbo.image.gturbo \
  --model-id Tongyi-MAI/Z-Image-Turbo \
  --model-revision f332072aa78be7aecdf3ee76d5c247082da564a6
```

The packer writes component `index.json` and `tensors.bin` files, derives
the manifest inventory from those indexes, writes `verified-install.json`,
and verifies the final hashes against the future published path before the
directory rename.

#### IG2 handoff checklist

The following records implementation status and the remaining evidence work:

1. **Native backend implementation.** The macOS-only `MetalImageBackend`,
   image-specific shaders, and host wrappers now cover text conditioning, the
   nine-step DiT transformer, scheduler state, and VAE decode. Existing GPU
   primitives were reused only for context, pass, and resident-buffer
   contracts; image linear and attention layouts use dedicated wrappers.
2. **Packed storage on the device.** The backend loads each packed component
   from its checked `index.json`, preserves the affine INT4 group-64 nibble,
   scale, and bias layout, and maps the component payload without copying a
   whole tensor for a row read. Real-install stage ownership remains to be
   measured.
3. **Packed parity gate.** The opt-in test now compares conditioning, all nine
   latent updates, final latents, and VAE decoded output against the IG1
   fixtures. It has not passed on a complete pinned packed install in this
   checkout.
4. **Explicit image install route.** `turbospark-model pull-image` packages a
   pinned local Diffusers export and records it as an image install, not as a
   text `Mlx` row. A network-backed catalog source plan and rot-guard remain
   open if remote installation is required.
5. **CLI production selection.** On macOS, `turbospark image generate`
   selects native Metal by default; `--backend reference` is explicit. The
   offline gates cover help, invalid envelope values, overwrite refusal, and
   unsupported-platform refusal. Real missing-component, cancellation,
   deterministic-output, and no-device paths remain to be exercised.
6. **Real gates.** The packed PNG metadata, quality, and resource-oracle
   tests are present as ignored tests. The resource report records stage and
   total latency, nine forwards, peak `phys_footprint`, Metal buffer
   allocations, idle retained buffers, process page-ins, and swap deltas.
   Quiet cold and warm runs against a complete install remain required.

The original implementation requirements are preserved below as the
acceptance contract:

1. **Native backend contract.** Use the existing
   `turbospark-gpu` `MetalContext` and `PassEncoder` APIs, and reuse an
   existing GPU primitive only after checking its tensor layout, precision,
   dispatch shape, and buffer lifetime.
2. **Packed storage contract.** Load each packed component from
   its checked `index.json`, preserve the affine INT4 group-64 nibble, scale,
   and bias layout, and bind resident or bounded staging buffers without
   copying an entire tensor for a row read. Keep stage ownership explicit:
   text encoder, transformer, and VAE must not all remain resident together.
3. **Packed parity contract.** Compare the packed native backend with the
   existing CPU/reference fixtures at conditioning, each transformer update,
   final latents, and decoded output. Use the fixed 1024-by-1024, batch-one,
   nine-step, guidance-zero, fixed-seed contract. Explain the first divergent
   intermediate and mutation-check every assertion.
4. **Real install contract.** Extend `crates/catalog` and
   `turbospark-model` with an explicit image install plan for the pinned
   Diffusers export. Do not encode this as an ordinary text `Mlx` model row.
   Fetch and verify all five components before the atomic publish, preserve
   the source revision in the manifest and receipt, and add catalog rot-guard
   coverage before adding an alias.
5. **CLI contract.** On macOS, `turbospark image generate`
   should select the native backend. The CPU backend remains an explicit
   reference path for fixtures and diagnostics. Non-macOS and no-device
   failures must be clear. Add CLI coverage for invalid options, missing
   components, deterministic metadata, valid PNG output, overwrite refusal,
   export failure, and cancellation in every stage.
6. **Evidence contract.** Add a packed image quality gate and image memory
   oracle. Run cold and warm measurements on a quiet machine and record stage
   peaks, total seconds/image, actual transformer evaluations, peak
   `phys_footprint`, managed allocations, retained buffers, physical reads,
   and swap. Compare against the inclusive IG0 reference ceilings, but do not
   relabel retained driver capacity as exact scratch or claim a whole-machine
   minimum RAM figure.

Prerequisites for this work are the pinned Z-Image-Turbo export at the
revision recorded in the manifest example, a Metal-capable macOS machine with
the Xcode Metal toolchain, the existing `target/ig0` fixtures, and enough
free disk for the source tree plus a staged packed install. Keep the first
release envelope fixed at 1024-by-1024, batch one, nine steps, nine forwards,
guidance zero, one image per prompt, and no app/Swift integration. IG2 is not
closed until the native packed path passes all six items on one real install.

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
The reference latency observations are frozen in the IG0 contract, but packed
runtime performance limits and a minimum RAM figure remain intentionally open
until IG2 and IG3 measure them.
