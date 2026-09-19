# Native image generation: Z-Image-Turbo

Status: IG0, IG1, IG2, IG3, and IG4 are closed for the pinned 1024-by-1024
case. The full-width checkpoint block, complete nine-step DiT rollout,
production-shape VAE parity, PNG metadata, cancellation, remote intake,
packed resource evidence, bounded streaming, repeated-job stability, Swift
integration, and app image mode pass their recorded contracts. IG5 remains
open only for measured optimization work.
[Phase 0 evidence](IMAGE_GENERATION_PHASE0.md) records pinned inputs,
real-image captures, and component comparisons.
The reusable bring-up process and the lessons from this model are summarized
in [ZIMAGE_TURBO.md](ZIMAGE_TURBO.md).

Image installs use the shared `~/.turbospark/models/image` namespace and are
not rows in the text model registry. Existing flat `.image.gturbo` installs
remain readable for compatibility. The image benchmark summary is maintained in
[ZIMAGE_TURBO.md](ZIMAGE_TURBO.md#recommended-model-and-benchmark-record).

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
opt-in gates. IG2 is now closed: the pinned real packed install passed the
quality, VAE, PNG metadata, cancellation, resource, intake, and no-device
execution evidence gates. The remaining work is tracked in
[ROADMAP](../ROADMAP.md). This page owns the design, gates, and rationale;
the roadmap owns the remaining task checklist.

### Default MLX source and artifact support

MLX safetensors is the default source format for Z-Image-Turbo image installs.
The default source is
[`andrevp/Z-Image-Turbo-MLX-4bit`](https://huggingface.co/andrevp/Z-Image-Turbo-MLX-4bit),
and the installer accepts all four upstream variants below through the same
source adapter and image install format.

| Variant | Install alias | Size | Quantization | Link |
| --- | --- | ---: | --- | --- |
| Full precision (fp16) | `z-image-turbo-mlx-fp16` | 20.54 GB | None | [`andrevp/Z-Image-Turbo-MLX`](https://huggingface.co/andrevp/Z-Image-Turbo-MLX) |
| 8-bit | `z-image-turbo-mlx-8bit` | 11.37 GB | 8-bit, group size 64 | [`andrevp/Z-Image-Turbo-MLX-8bit`](https://huggingface.co/andrevp/Z-Image-Turbo-MLX-8bit) |
| 4-bit (default) | `z-image-turbo-mlx-4bit` | 6.48 GB | 4-bit, group size 64 | [`andrevp/Z-Image-Turbo-MLX-4bit`](https://huggingface.co/andrevp/Z-Image-Turbo-MLX-4bit) |
| 2-bit | `z-image-turbo-mlx-2bit` | 4.04 GB | 2-bit, group size 64 | [`andrevp/Z-Image-Turbo-MLX-2bit`](https://huggingface.co/andrevp/Z-Image-Turbo-MLX-2bit) |

The pinned header-only intake gate covers these four revisions and checks the
published transformer layout before any payload download. It confirms U32
affine planes with matching F16/BF16 scale and bias companions for the 2, 4,
and 8-bit exports, and F16 tensors for the full-precision export. The gate does
not claim real-install or image-quality parity for those variants. The
payload-range gate reads one selected real transformer tensor from each pinned
variant, runs it through the production image packer and decoder, and still
avoids staging the complete source tree.

When the destination volume can hold both the source staging tree and the
packed output, run the full installer gate for one variant at a time:

```sh
TURBOSPARK_ZIMAGE_MLX_VARIANT=4bit \
TURBOSPARK_ZIMAGE_MLX_INSTALL_DIR=~/.turbospark/models/image/z-image-turbo-mlx-4bit.gturbo \
  cargo test -p turbospark-cli --test zimage_mlx_install_network --release \
  -- --ignored --nocapture
```

The `2bit`, `4bit`, and `8bit` gates passed on 2026-09-18: `pull-image`
completed each pinned source staging and pack, and every installed manifest
verified with its expected observed-width label. The `fp16` gate remains open
as a separate unquantized source path. This installer gate does not close
image-quality, resource, or real-install Swift image-generation evidence for
any non-INT4 variant. The Swift catalog/ABI/app integration itself is
implemented and covered by the binding and app build gates below.

These are published download sizes, not runtime memory guarantees. The
original `Tongyi-MAI/Z-Image-Turbo` Diffusers export remains the independent
parity and quality reference. `turbospark image pack` normalizes either source
shape into the repository's image install format, while
`turbospark-model pull-image --repo OWNER/NAME@REV` handles remote MLX source
intake and `--source` handles a local MLX directory.

The pinned IG2 production profile is this repository's legacy affine INT4
linear format at group size 64. It is not an all-tensor INT4 claim: embeddings,
normalization, modulation, positional, and other protected tensors remain
unquantized, as do image-sensitive operations such as the VAE. The default
published MLX 4-bit source uses MLX affine U32 rows with F16 or BF16
scale/bias companions and remains in that representation through the image
adapter. F32 source weights that remain unquantized are stored as BF16 to match
the reference transformer's loaded dtype; this is a storage alignment, not a
claim that the native activation path is BF16-exact. All four published
variants use the same adapter, manifest, and runtime family. Do not register
any image variant as an ordinary text `Mlx` catalog row.

The source adapter and native row decoder accept MLX affine U32 weights at 2,
3, 4, 5, 6, and 8 bits, with group size 64 and F16 or BF16 scale/bias
companions. The packed index records the logical matrix shape rather than the
source U32 packing shape. Focused tests cover every supported width and both
companion dtypes. The CLI and generated image request, PNG metadata, and
Swift's installed-image listing carry the observed MLX width label. This is
the complete upstream `mx.quantize` width set;
upstream rejects 1-bit quantization. Full installer gates pass for the
published 2-, 4-, and 8-bit variants; the FP16 install remains open. Quality,
resource, and real-install Swift image-generation gates are still required
before any non-INT4 variant is considered fully verified. The Swift
catalog/install binding surface is covered separately by `make swift-test` and
the app build.

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
native packed memory envelope is now measured for the pinned install. Pooled
scratch, activation reuse, fewer command-buffer boundaries, and safer BF16/FP16
storage for non-INT4 tensors remain later optimization work.

IG0 is now closed by the measured reference contract. Exact MPS operator
scratch is not observable, so the contract uses an inclusive non-parameter
process budget and explicitly does not call driver-retained bytes scratch or
claim a minimum whole-machine RAM size. IG3 closed repeated full-pipeline
stability, resident-versus-streamed ownership, and cancellation lifetime proof.
IG2 and IG4 are closed for the pinned install. MLX variant installation is
closed for the published 2-, 4-, and 8-bit rows, while FP16 installation,
quality, resource, and real-install Swift image-generation gates remain open.
The Swift catalog/install surface is covered separately by the binding and app
build gates.

## IG4 app seam

The app-facing image path is now a separate session rather than an overload of
the text token stream. `TsImageSession` opens a verified image install, emits
stage progress through `TsImageEventCallback`, supports cancellation from
another thread, returns explicit PNG ownership, and returns the same
camelCase metadata that the runtime embeds in the PNG. `TurboSparkImageSession`
copies the PNG before releasing the C buffer.

The macOS app's top-level `Images` destination is the canonical image workflow.
Its `Create` tab accepts a direct prompt and lists valid installed image
artifacts through the separate `ts_image_installed_json` catalog surface. A
folder chooser remains available for a side-loaded install, but image
generation never falls back to the selected text model. The `Gallery` tab
shows profile-owned PNG thumbnails and opens a previous/next carousel. A
process-wide FIFO coordinator serializes heavyweight image jobs across chats in
the app. The result stays in transient
job state until the user saves it, which provides preview, regeneration, and
safe interruption without turning an incomplete job into durable history.
Saved PNGs live below `AppStorageRoot.subdirectory("image-artifacts")`, and a
saved result registers an `.imageGeneration` artifact plus a chat message, so
profile switching and chat persistence keep the image in the originating
profile. Regeneration reuses the recorded request and seed, so the result is
reproducible without overwriting the prior saved artifact. The native app path
required a real pinned install and Metal execution evidence before IG4 could be
checked closed; that evidence now exists, while the independent roadmap
measurement and validation gates remain outstanding.

The real Swift seam is gated separately from text and vision installs:

```sh
make swift-test-real IMAGE_MODEL=~/.turbospark/models/image/z-image-turbo.gturbo
```

The image-generation tests open the verified install, assert PNG and metadata
return through Swift, and cancel during a stage.

The recorded native pinned-install gate passed both Swift image-session arms:
the full 1024-by-1024 generation returned a valid PNG and decoded runtime
metadata in 4,665.912 seconds, including the public `modelID` spelling used by
Swift; the cancellation arm returned `cancelled` without publishing a PNG in
21.827 seconds. The app bundle was built and strict deep-signature verification
passed. The focused app image-job suite passes 9 tests, including FIFO
serialization, preview/save/regenerate state, profile isolation, chat-delete
protection, and persisted relative artifact paths.

The current no-model `make swift-test` run passes 79 tests with 17 expected
real-model skips. The image-generation tests remain opt-in because they need a
retained verified `.image.gturbo` install. Run the real gate above with
`IMAGE_MODEL` set to close Swift runtime evidence for a specific MLX variant;
the catalog/install binding surface is already covered by the regular Swift
suite. The roadmap still owns the independent measurement and validation
backlog that must be closed before the milestone checkbox is marked.

## Direction and first release

Build Z-Image-Turbo text-to-image inference in Rust with the existing Metal
backend. Deliver a quantized CLI pipeline first, then expose that same
runtime through the C ABI and Swift package to an explicit image mode in
chat. MLX safetensors are a supported source format; external engines remain
correctness references and benchmark tools, not production subprocesses or
runtime dependencies.

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
are not prerequisites for this direction because the source adapter normalizes
MLX safetensors before native execution. Existing quantized kernels are reuse
candidates, not proof that a community checkpoint's packing or shapes are
compatible. Likewise, a text-generation Qwen runner is not automatically
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
The packed representation and its measured Metal behavior were IG2 gates and
are recorded in the closure evidence; no minimum whole-machine RAM claim is
made.

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
  --output ~/.turbospark/models/image/z-image-turbo.gturbo \
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
   contracts; image linear and grouped attention layouts use dedicated
   wrappers. The attention kernel is validated independently on a small GQA
   and causal-mask fixture before the full image gate.
2. **Packed storage on the device.** The backend loads each packed component
   from its checked `index.json`, preserves the affine INT4 group-64 nibble,
   scale, and bias layout, and maps the component payload without copying a
   whole tensor for a row read. Real-install stage ownership remains to be
   measured.
3. **Packed quality and parity gate.** The opt-in test compares packed
   conditioning and the nine latent updates against the frozen INT4 emulation
   envelopes, then isolates VAE implementation parity by decoding the frozen
   higher-precision final latent. The rollout seeds from the captured
   `initial_noise` fixture because the envelopes are matched-noise bounds
   against the reference capture. The packed end-to-end decode is also checked
   for finite output; the complete matched-noise gate now passes on a pinned
   packed install, including isolated frozen-latent VAE parity.
4. **Explicit image install route.** `turbospark-model pull-image` packages a
   pinned local MLX export and records it as an image install, not as a text
   `Mlx` row. `pull-image --repo OWNER/NAME@REV` streams the selected MLX
   variant into temporary staging before packing. The four upstream variants
   share the same adapter and artifact path; the separate image catalog and
   its live source-file rot guard pass.
5. **CLI production selection.** On macOS, `turbospark image generate`
   selects native Metal by default; `--backend reference` is explicit. The
   offline gates cover help, invalid envelope values, overwrite refusal, and
   unsupported-platform refusal. The sparse real-install missing-component
   refusal and real native cancellation path pass. The deterministic-output
   assertion passes in the corrected cold/warm resource oracle. The no-device
   path executes on Linux ARM64 and passes the unsupported-platform refusal
   before model I/O.
6. **Real gates.** The packed PNG metadata, quality, and resource-oracle
   tests are present as ignored tests. The resource report records stage and
   total latency, nine forwards, peak `phys_footprint`, Metal buffer
   allocations, idle retained buffers, process page-ins, and swap deltas.
   The cold arm completed at 6,515.892 seconds with PNG relative L2
   `0.4284977`, zero page-ins, and a `20,725,728,336`-byte process peak
   dominated by the VAE. The corrected warm-only arm also passed in 4,572.05
   seconds with PNG relative L2 `0.4284977`, zero page-ins, unchanged swap,
   and a `19,874,055,248`-byte process peak.
   The original pinned local export packs to 11 files and 6,906,461,695
   bytes. Its packed conditioning error is 0.081541 against the 0.084 IG0
   INT4 envelope. A second pack using the corrected IG0 projection policy
   reported 11 files and 6,944,955,456 bytes; conditioning also completed,
   but it did not improve the denoise result. The grouped Metal attention
   kernel passes the focused GQA and causal-mask parity test. The two real
   native nine-step runs both fail at the first rollout check, after the long
   device run: the original artifact ran 4,618.35 seconds and measured
   relative L2 1.4135604 against the 0.923 envelope; the corrected-policy
   artifact ran 4,711.53 seconds and measured 1.4136423. These are quality
   failures, not successful parity runs. Cancellation, PNG quality, and the
   resource oracle remain unclosed.

   The next diagnostic boundary is now known. A precise layer-29 mini-component
   probe found RMSNorm, projections, attention, modulation, SiLU, and FFN
   operations internally consistent at roughly 1e-6 relative L2, while the
   full block remained about 0.600 against the BF16 MPS fixture. The probe is
   not a production gate because it uses F32-expanded weights and was removed
   after use. Continue from the end-to-end packed rollout and compare the
   first divergent denoise intermediate against the IG0 BF16 capture; do not
   widen the 0.923 envelope without new reference evidence.

   The bounded first-step diagnostic is now available as
   `packed_native_first_step_trace_localizes_divergent_boundary`. It reads only
   the first native step and reports conditioning, patchification,
   noise-refiner output, selected main-transformer block states, velocity, and
   scheduler latent errors. Run it with:

   ```sh
   TURBOSPARK_IMAGE_INSTALL_DIR=/path/to/pinned.gturbo \
   TURBOSPARK_IMAGE_TRACE_DIR=/path/to/fresh/lighting-trace \
     cargo test -p turbospark-image --test metal_parity \
     packed_native_first_step_trace_localizes_divergent_boundary -- \
     --ignored --nocapture
   ```

   To add matching reference arrays, run the `encode` and `denoise` stages of
   `scripts/z_image_capture.py` into that fresh run directory. The denoise
   manifest then declares the bounded trace arrays and validates their shapes
   against the conditioning predecessor. The patchification fixture is
   optional until that capture is refreshed; the existing block and latent
   fixtures still localize the first packed boundary.

   The current matched-noise run passes the conditioning boundary at `0.0815`
   and reports main-transformer block errors of `0.0689` at block 0, `0.0763`
   at block 15, `0.0878` at block 16, `0.1796` at block 20, `0.4174` at block
   24, `0.9540` at block 28, and `1.0846` at block 29. This is progressive
   accumulation in the later main-transformer recurrence, not an isolated
   block-0 failure. The packed path is therefore still unresolved between
   INT4 execution, BF16-versus-F32 behavior, and a repeated layout or dispatch
   error. Keep the `0.923` envelope unchanged while narrowing that boundary.

   **The 1.414 rollout failure is a noise-realization mismatch, not packed
   drift.** The frozen fixtures were captured from the reference pipeline's
   torch CPU `randn(seed 42)`, while `MetalImageBackend` rolls out from its
   own `seeded_noise` xorshift and Box-Muller generator. The two fields are
   independent draws: measured directly, the native noise reads relative L2
   1.4132 against the captured `initial_noise` array with correlation
   -0.0014, and two uncorrelated unit-scale fields sit at sqrt(2) = 1.4142.
   Both complete nine-step runs failed at 1.4135604 and 1.4136423, within
   0.03 percent of that floor, and the noise-independent conditioning
   boundary passed at 0.0815. Any implementation, however correct, reads
   about 1.414 against these fixtures. The packed parity and trace gates now
   seed the denoiser from the captured `initial_noise` fixture through
   `denoise_steps_from_noise` and the trace's `initial_noise` parameter, so
   the frozen 0.923 envelope measures what it was frozen to measure. The
   envelope is unchanged. Production generation keeps the native noise; any
   unit-scale Gaussian is a valid draw, and the PNG metadata records the
   generator provenance. The first matched-noise trace on the pinned install
   (`docs/verification/z-image-ig2-noise-trace.json`) then reads conditioning
   0.0815, noise-refiner 0.0644, main-transformer intermediate 1.0856,
   velocity 0.6029, and first scheduler latent 0.0395, so the packed first
   step sits well inside the 0.923 envelope; the two intermediate readings
   have no frozen analogue because IG0 froze only conditioning and final
   latents. The first complete matched-noise gate run (4,778.46 seconds)
   passed every denoise check and reached the VAE for the first time, where
   it exposed a load-time defect: the packed VAE records the mid-block
   attention projections as 2-D Diffusers `nn.Linear` weights while the
   Metal path expected `[512, 512, 1, 1]`. The 1x1 conv kernel indexes a
   weight as `oc * in_channels + ic`, byte-identical to the Linear layout
   the CPU reference applies, so the Metal shape expectations now match the
   packed 2-D form, and a cross-check of the whole VAE index confirms those
   four are the only 2-D decoder tensors. The focused
   `packed_native_vae_decodes_the_frozen_latent` gate decodes the frozen
   latent alone so VAE issues no longer require a nine-step denoise in
   front of them.

   A matched quantized-reference control then ran the same seeded first-step
   trace. Native errors were conditioning `0.0042043`, noise-refiner
   `0.0521734`, main transformer `1.0849981`, block 0 `0.0636551`, block 16
   `0.0836585`, block 24 `0.4202046`, block 28 `0.9507074`, and block 29
   `1.0849981`. The ordinary BF16 reference comparison was effectively the
   same, so the late recurrence is not explained by the INT4 projection policy
   or conditioning provenance. A fresh packed install that stored the
   remaining F32 source tensors as BF16 also reproduced the original curve,
   including block 29 `1.0846142`. A diagnostic GPU round-to-BF16 pass at
   block boundaries changed that value only to `1.0834699` and was removed.
   The next diagnostic must inspect intra-block activation and accumulator
   precision or a repeated native dispatch contract. The `0.923` envelope is
   unchanged, and the complete quality, VAE, PNG, cancellation, and quiet
   resource gates remain open.

   Two seeded controls then closed the repeated-dispatch and compiler-math
   branches without changing the production path. Setting
   `TURBOSPARK_IMAGE_FORCE_SYNC_DISPATCH=1` forced a CPU completion after every
   image primitive; the selected block errors were identical to the baseline,
   including block 28 `0.953971744` and block 29 `1.08461416`. Setting
   `TURBOSPARK_METAL_PRECISE_MATH=1` disabled Metal fast math; block 28 moved
   only to `0.953971624` and block 29 to `1.08461368`. These are sub-ppm
   changes, so same-queue hazard handling and relaxed compiler math are not the
   cause. The remaining boundary is intra-block activation or reduction
   precision: the captured block arrays have `torch.bfloat16` source dtype,
   while the native image tensors and shader outputs remain FP32. Keep the
   quality envelope frozen until an intra-operation BF16/accumulator control or
   a concrete layout mismatch explains the late recurrence.

   During the fresh rebuild, the release test exposed a source-level Metal
   ABI mismatch: `metal_ops` requested `image_rope_orthogonal` while the shader
   exported `image_rope`. The shader export was corrected and the release
   parity test rebuilt from current sources. Do not trust a stale compiled
   artifact to validate this path.

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
3. **Packed comparison contract.** Compare packed native conditioning and
   transformer updates with the frozen INT4 quality envelope, and compare the
   protected VAE on the frozen higher-precision latent with the IG1 decoded
   fixture. Check the actual packed end-to-end decode separately through the
   PNG quality oracle. Use the fixed 1024-by-1024, batch-one, nine-step,
   guidance-zero, fixed-seed contract. Explain the first divergent
   intermediate and mutation-check every assertion.
4. **Real install contract.** Extend `crates/catalog` and
   `turbospark-model` with an explicit image install plan for the default
   pinned MLX export and its four supported variants. Do not encode this as an
   ordinary text `Mlx` model row. Fetch and verify all required components
   before the atomic publish, preserve the source revision and selected
   variant in the manifest and receipt, and add catalog rot-guard coverage
   before adding an alias.
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

#### 2026-09-14 implementation checkpoint

- The tiled linear Metal launch now uses the shader's 32-column threadgroup
  stride. The focused packed linear parity test passes on real Metal; the
  host-only grid change was rejected by the packed conditioning gate before
  the shader stride was corrected.
- Image operation wrappers now defer command-buffer waits until a CPU read
  seam, retain the producing pass for the output lifetime, and keep the
  submission inside the existing autorelease pool. The caption-refiner result
  is also reused between denoise steps.
- The full packed native parity gate was rerun against the pinned local
  install after these changes. The original artifact ran 4,618.35 seconds and
  failed at rollout step 1 with relative L2 1.4135604, above the frozen 0.923
  envelope. The corrected-policy artifact ran 4,711.53 seconds and failed at
  the same check with 1.4136423. This is an end-to-end quality failure, not a
  cancellation or memory pass. The next target is the first divergent denoise
  intermediate, followed by the cancellation, PNG, and quiet-machine resource
  gates after quality is explained and fixed.

Prerequisites for this work are the pinned Z-Image-Turbo export at the
revision recorded in the manifest example, a Metal-capable macOS machine with
the Xcode Metal toolchain, the existing `target/ig0` fixtures, and enough
free disk for the source tree plus a staged packed install. Keep the first
release envelope fixed at 1024-by-1024, batch one, nine steps, nine forwards,
guidance zero, one image per prompt, and no app/Swift integration. IG2 is
closed only after the native packed path passes all six items on one real
install. Those six items are now evidenced in the records above.

### IG3: Bound memory and add dense streaming where necessary

Account in bytes for component weights, conditioning, latents, activations,
scratch, staging, in-flight buffers, and retained allocator capacity. Reserve
headroom for macOS and allocations outside the managed ledger. Shared views
of one allocation count once; genuine staging copies count separately.

The runtime now exposes this admission contract through `ImageMemoryPlan`,
`generate_with_memory_budget`, and `ImageWorkTracker`. Resident component
teardown still waits the latest ordered Metal fence, while the tracker is the
explicit seam for I/O or staging consumers. `SequentialImageSlots` rejects a
third acquisition while both slots are live and records peak live bytes. These
portable contracts have focused unit coverage. The pinned hardware oracle
below also populated the category ledger and compared resident execution with
a real two-slot path. `generate` invokes the backend idle hook before returning
a cancelled request, so synchronous I/O and staging leases are drained at the
public cancellation boundary.

The resource oracle now repeats complete jobs and matched-noise denoise cycles.
Each cycle records peak and idle `phys_footprint`, managed Metal allocations,
physical reads, and swap deltas, while repeated PNGs and final latents must
agree exactly. Repeated idle footprint growth is limited to 256 MiB by default,
with an explicit environment override for a machine-specific qualification.

On 2026-09-17, the pinned install's pre-execution plan gate passed. It reported
6,651,207,110 resident payload bytes, 214,918,144 two-slot bytes, a
107,459,072-byte largest block, and a 1,136,072,192-byte streamed lower bound.
The current-source resident-versus-streamed gate then passed with exact PNG
agreement: resident latency was 4,911,114 ms, streamed latency was 4,840,856
ms, and the streamed/resident ratio was 0.985694. Resident peak
`phys_footprint` was 7,037,387,832 bytes and streamed peak was 7,328,138,608
bytes. The streamed path held 214,918,144 slot bytes, performed 400 payload
reads totaling 33,298,002,054 bytes, and recorded 393 fenced slot reuses.
The earlier fixture-missing, Metal out-of-memory, and first-step stall runs
remain retained as failed qualification attempts, not IG3 evidence.

The repeated warm-only oracle then passed on the same pinned install. Complete
jobs 1 and 2 reported peak `phys_footprint` values of 7,038,518,400 and
7,152,075,976 bytes, an increase of 113,557,576 bytes; idle values were
623,903,872 and 707,380,424 bytes, an increase of 83,476,552 bytes. Both
reported 7,784 managed allocations, zero page-ins, zero swap delta, and the
same PNG relative L2 of `0.4284977`. Matched-noise denoise cycles 1 and 2
completed nine forwards over 262,144 finite latent values. Their peak
footprints were 2,497,283,440 and 2,512,946,616 bytes, a 15,663,176-byte
increase; idle footprints were 789,415,208 and 737,379,528 bytes. Each cycle
reported 7,138 managed allocations, zero page-ins, and zero swap delta, and
the test asserted exact final-latent equality. The full oracle passed in
18,839.90 seconds.

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
tradeoff at each advertised budget. These gates now pass for the pinned
install. A managed budget does not claim a hard cap on whole-process or system
physical memory.

### IG4: Expose the runtime to Swift and the Images destination

IG4 is implemented for the pinned 1024-by-1024 envelope. The C ABI and Swift
wrapper expose image sessions, requests, progress, results, cancellation, and
explicit buffer ownership. The app Images destination includes curated image
catalog download, progress, cancellation, model selection, generation, and
gallery behavior. The remaining opt-in gate is real runtime verification
against a retained verified image install.

The `Images` destination has `Create` and `Organize` tabs. Both show saved
outputs with prompt reuse, export, regeneration, and Move to Trash actions.
Create keeps its prompt bar at the bottom; Settings expands below it and moves
the composer up. Model and quantization are separate controls, with Z-Image
Turbo exports grouped together. Organize adds prompt search and bulk selection.
Thumbnails open a previous/next carousel; missing files retain a removable row.

New requests default to one image at the selected install's default size.
Z-Image Turbo currently permits only 1024-by-1024, so the size control explains
that limit instead of offering unsupported dimensions. Selecting two to four
images runs single-image requests serially under one coordinator admission,
with successive seeds and unchanged native dimensions. Each result saves before
the next request starts; cancellation or save failure stops the remainder.
An unsaved result stays visible with Save and Remove actions and blocks another
request until resolved. Move to Trash only touches this profile's generated
image directory and removes the matching artifact and transcript image paths
only after a successful move (or when the file is already missing).

The submitted prompt is used directly, without a text-model prompt enhancer or
conversation history. Regenerate reuses the recorded request and seed with the
currently selected image install; changing the seed is explicit.

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

## Current public surfaces

The unified image command is available through the `turbospark` wrapper:

```sh
turbospark image generate \
  --model z-image-turbo \
  --prompt "A lighthouse in winter" \
  --seed 42 \
  --width 1024 --height 1024 \
  --steps 9 --backend native \
  --output lighthouse.png
```

`--seed`, `--width`, `--height`, `--steps`, and `--backend` are explicit
options. The native backend is the default on macOS; `reference` is an
explicit CPU diagnostic backend. `ImageRequest::validate()` enforces the
installed manifest's dimensions and scheduler envelope. Output is PNG,
progress goes to stderr, a missing seed is resolved randomly, and an existing
output path is refused. Publication happens only after encoding succeeds, and
incomplete output is removed on failure or cancellation.

Image installs are separate from text installs. A local source can be packed
and a pinned remote source can be streamed into the verified store:

```sh
turbospark-model pull-image \
  --source /path/to/Z-Image-Turbo \
  --alias z-image-turbo \
  --model-id Tongyi-MAI/Z-Image-Turbo \
  --model-revision f332072aa78be7aecdf3ee76d5c247082da564a6

turbospark-model pull-image \
  --repo Tongyi-MAI/Z-Image-Turbo@f332072aa78be7aecdf3ee76d5c247082da564a6 \
  --alias z-image-turbo

# The pinned MLX variants are also available through the image catalog.
turbospark-model pull-image --alias z-image-turbo-mlx-4bit
```

Swift hosts can list and install the same pinned image rows without duplicating
catalog data. `TurboSparkCatalog.imageAvailable()` returns aliases, model IDs,
revisions, and quantization labels. `TurboSparkCatalog.installImage(_:)`
streams staged byte progress and returns a verified `ImageInstalledModel`; the
macOS Images destination exposes these rows with progress and cancellation.

The C ABI uses `TsImageSession`, `TsImageEventCallback`, explicit PNG buffer
ownership, and cancellation. The Swift package wraps it with
`TurboSparkImageSession`, `ImageGenerateOptions`, `ImageGenerationEvent`, and
`ImageGenerationResult`. The options JSON contract remains camelCase:
`prompt`, `seed`, `width`, `height`, and `steps`.

The shared runtime interface provides a model/session handle, prompt and image
options, resolved settings, phase progress, cancellation, owned pixel output,
and metadata. Encoding/export stays separate from app storage paths. C
allocation/free and callback-thread rules are part of the ABI contract; image
progress does not reuse text token event numbers. Rust, CLI, and Swift
validation must agree.

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
The reference latency observations are frozen in the IG0 contract. IG3 records
the pinned resident/streamed latency tradeoff, while broader performance
curves and any minimum whole-machine RAM figure remain intentionally open.
