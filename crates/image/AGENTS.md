# turbospark-image

Native Z-Image-Turbo image generation: prompt conditioning, the Qwen3 text
encoder, the dense diffusion transformer, the FlowMatch Euler scheduler, the
VAE decoder, the packed component install format, the macOS Metal backend,
and the pure-Rust reference backend. `docs/ZIMAGE_TURBO.md` is the HOME for
the bring-up sequence, the resource ceilings, and every measured number;
`docs/IMAGE_GENERATION.md` is the design record. This file carries none of
those numbers.

This crate is NOT a text-model family and must never become one: image
installs carry their own manifest, receipt, and fixed envelope in
`install.rs`, and image installs deliberately do not reuse the text-model
manifest (`install.rs`'s module doc says why -- accepting one would make an
invalid install look runnable).

## Directory & File Structure

```
crates/image/
+-- Cargo.toml
+-- src/
|   +-- lib.rs               # module list, re-exports
|   +-- install.rs           # image manifest, receipt, envelope constants, validation
|   +-- builder.rs           # install builder entry point
|   |   +-- builder_files.rs     # payload/file writing
|   |   +-- builder_manifest.rs  # manifest + receipt assembly
|   |   \-- builder_tests.rs     # builder unit tests
|   +-- packed.rs            # packed component store: index.json + tensors.bin,
|   |                        # local INT4 and MLX affine rows over FP32 stragglers
|   +-- conditioning.rs      # prompt framing, tokenizer loading, padded token ids
|   +-- text_encoder.rs      # Qwen3 encoder forward, sharded safetensors reader
|   +-- transformer.rs       # ZImageTransformer blocks (CPU math)
|   +-- pipeline.rs          # timestep embedding, transformer driver, constants
|   +-- scheduler.rs         # FlowMatchEulerScheduler
|   +-- vae.rs               # VAE decoder math, latents -> RGB8
|   +-- rope.rs              # RopeEmbedder (text + image axes)
|   +-- patchify.rs          # image patchify/unpatchify, coordinate grid, sequence build
|   +-- runtime.rs           # ImageBackend trait, ImageRequest/Result, stage lifecycle,
|   |                        # cancellation, the generate() driver both backends share
|   +-- reference.rs         # CpuReferenceBackend: layout/cancel/metadata oracle
|   +-- fixtures.rs          # NPY/NPZ readers for reference fixtures
|   +-- metal.rs             # MetalImageBackend: stage driving over metal_ops kernels
|   +-- metal_ops.rs         # image-specific Metal kernel wrappers + FunctionConstantValues
|   \-- shaders/
|       \-- image.metal      # the one shader source, include_str!'d by metal_ops.rs
\-- tests/
    +-- conditioning_parity.rs
    +-- scheduler_parity.rs
    +-- text_encoder_parity.rs
    +-- transformer_math_parity.rs
    +-- pipeline_parity.rs
    +-- vae_parity.rs
    +-- metal_parity.rs      # opt-in packed Metal gates; needs a pinned install
    \-- zimage_mlx_payload_network.rs # selected real MLX payload gate
```

The CPU modules (`transformer.rs`, `vae.rs`, `pipeline.rs`) are the portable
arithmetic the parity tests check the Metal path against; `reference.rs` is
the full-pipeline CPU `ImageBackend` for install-layout, cancellation, and
metadata coverage on any platform.

## Development & Test Commands

```sh
# Offline: units and math parity against checked-in fixtures. No install.
cargo test -p turbospark-image

# The off-macOS gate this crate owes. The portable core must compile with
# the Metal modules compiled out; a cfg or dependency that breaks it is a
# bug in this crate (see AGENTS.md, cross-target check, this crate is in
# the list BY DESIGN).
cargo check --target x86_64-unknown-linux-gnu -p turbospark-image
```

The `metal_parity.rs` gates are `#[ignore]`d and opt-in: point
`TURBOSPARK_IMAGE_INSTALL_DIR` at a complete packed install produced by the
image packer and run with `--ignored --nocapture`. The install stays outside
the repository on purpose; nothing in `tests/` builds one.

```sh
cargo test -p turbospark-image --test metal_parity -- --ignored --nocapture
```

## Gotchas

1. **THE DEPENDENCY TABLE IS WHERE THE PLATFORM SPLIT LIVES.** `gpu` and
   `metal` sit under `[target.'cfg(target_os = "macos")'.dependencies]`, and
   `metal.rs` / `metal_ops.rs` are `#[cfg(target_os = "macos")]` behind them
   -- the same shape as `crates/runtime` (root AGENTS.md Gotcha 8). Adding an
   unconditional call into either module, or moving a portable type into one
   of them, breaks the cross-target check above rather than the macOS build,
   so the failure shows up in a command nobody runs by default.

2. **THE ENCODE PATH OPENS NO AUTORELEASE POOL.** Every
   `MetalContext::begin_pass_labeled` creates an autoreleased command buffer
   (root AGENTS.md Gotcha 17: ~6 KiB held per buffer until a pool drains),
   and one generation encodes hundreds of passes -- every text-encoder
   block, every transformer block across all nine forwards, and the whole
   VAE -- with no `gpu::autorelease_pool` anywhere in this crate. Nothing
   fails and no output changes; the pool-retained buffers just accumulate
   for the length of `generate()`. If the IG2 footprint gates drift, the
   first suspect is a new encode loop without its own pool boundary: wrap
   each scheduler step or stage in `gpu::autorelease_pool` here, and note
   that the resource ceilings this crate is gated against were measured WITH
   this accumulation present, so adding pools moves those numbers.

3. **ONE HEAVYWEIGHT COMPONENT AT A TIME.** Each stage opens its packed
   payload inside the stage and DROPS it before the next stage starts
   (`metal_ops.rs`'s `Component`, opened per stage by `metal.rs` and dropped
   before the next component loads; `reference.rs` follows the same
   discipline and says so in its module doc). Hoisting the three
   `Component`s into backend scope so they live for the whole request
   defeats the residency contract the resource gates measure, and it fails
   no test -- it just makes the footprint three times the frozen ceiling.

4. **PACKED AFFINE WEIGHTS ARE AN IMAGE COMPONENT FORMAT, NOT `.gturbo`.**
   One payload file per component (`tensors.bin`) has a checked byte span per
   tensor in `index.json`. The legacy local profile uses affine INT4 group-64
   rows. MLX source weights use affine U32 planes at 2, 3, 4, 5, 6, or 8 bits
   with F16/BF16 scale-bias companions, normalized into the same image
   component store. Embeddings, norms, modulation tensors, and anything
   non-matrix stay at their protected storage precision. `metal_ops.rs`
   deliberately does NOT call the text runtime kernels: image activations and
   accumulators are FP32 and the packed row layouts are image-specific. The
   only shared GPU contracts are `MetalContext`, `PassEncoder`, and
   `ResidentGpuWeights`.

5. **A TOLERANCE AGAINST A HIGHER-PRECISION FIXTURE MEASURES THE
   QUANTIZATION, NOT THE IMPLEMENTATION.** The packed conditioning and
   rollout rel-L2 limits are large by design: the pinned quality gate bounds
   local INT4-emulation drift against FP32/BF16 reference fixtures, while MLX
   width-specific parity still needs its own real-source evidence. A "tighter"
   limit is not a stricter test, it is a different question. The unquantized VAE limits,
   by contrast, ARE parity tolerances and are correspondingly tight. Name
   each constant for what it bounds, never reuse one limit name across the
   two classes, and assert with the MEASURED error in the message, not the
   limit alone -- `A <= LIMIT` failing with "exceeds {LIMIT}" names nothing,
   and the fix that interpolated the measured drift first (`c00d6b8`) is
   the shape to keep.

6. **THE RECEIPT IS THE ADMISSION BOUNDARY; SHAPE CHECKS ARE THE SECOND.**
   `ImageManifest::load` -> `validate` -> `verify_files` all run in
   `open()` before anything is mapped, and the receipt records the verified
   file set. `Component::weight(name, expected_shape)` then re-checks every
   tensor's shape at dispatch time. Skipping the second layer is how a
   hand-edited index passes `open()` and breaks Metal indexing mid-request;
   a new kernel that takes a new tensor must add its expected-shape check
   in the same commit.

7. **CANCELLATION AND PUBLICATION ARE LIFECYCLE PROPERTIES OF
   `runtime.rs`, NOT OF THE BACKENDS.** `generate()` owns stage sequencing,
   the `CancellationToken` checks between stages, and the rule that the PNG
   is published only after the VAE decode completes. A backend implements
   the stages and nothing else; backend-local output paths or early
   publication bypass the contract the CLI and the reference backend both
   rely on.

8. **THE CLI SURFACE LIVES IN `crates/cli`, AND ITS SECRET RULES APPLY
   HERE.** `turbospark-image` is `crates/cli/src/bin/image.rs`; flag
   conventions and the no-secrets-in-argv rules are recorded in
   `crates/cli/CLAUDE.md` Gotchas 16 and 17. Image-specific flags follow
   the same five-place rule as every other CLI flag.

9. **COMPONENT MMAP DROPS MUST DRAIN PENDING WORK BEFORE UNMAPPING.**
   When a stage finishes and drops its `Component`, any queued Metal command
   buffers that reference resident memory from the component must be
   drained (`Component::drop` waits on `latest_pass`). Intermediate
   activation-only passes (such as attention or elementwise ops) do not
   borrow from `Component` and commit via `commit_deferred`, while
   weight-backed passes register their pass via `commit_component_deferred`.

10. **IMAGE METAL SHADER ABI CONTRACT.**
    `image_linear_tiled` expects a six-field `LinearParams` (rows, in_dim,
    out_dim, row_stride, storage, bias_storage) and binds the resident weight
    buffer at index 0 and activations at index 1. Because linear operations
    access resident component memory, passes must be committed using
    `commit_component_deferred` so component mmap drops wait for the kernel to
    complete.
