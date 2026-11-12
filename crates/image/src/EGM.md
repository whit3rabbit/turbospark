# turbospark-image src

## Overview
Native text-to-image generation pipeline (Z-Image-Turbo family and related
installs). It provides a portable pure-Rust CPU reference path, a Metal
backend (macOS only), a packed quantized tensor store, install manifest
validation, an install builder, and a staged `generate` lifecycle
(conditioning, denoise, VAE decode, PNG encode).

## Interface
- Lifecycle: [runtime.rs:generate], [runtime.rs:generate_with_memory_budget], [runtime.rs:ImageBackend] (trait with `encode_conditioning`, `denoise`, `decode`, `wait_for_idle`), [runtime.rs:ImageRequest], [runtime.rs:ImageResult], [runtime.rs:ImageProgress].
- Cancellation and admission: [runtime.rs:CancellationToken], [runtime.rs:ImageWorkTracker], [runtime.rs:ImageWorkLease], [runtime.rs:admit_image_budget], [runtime.rs:ImageMemoryPlan], [runtime.rs:ImageMemoryBudget], [runtime.rs:SequentialImageSlots].
- Backends: [reference.rs:CpuReferenceBackend], [metal.rs:MetalImageBackend] (cfg macos), [metal.rs:ImageResidency], [metal.rs:ImageStreamMetrics].
- Model parts: [pipeline.rs:ZImageTransformer], [transformer.rs:ZImageTransformerBlock], [transformer.rs:FinalLayer], [text_encoder.rs:encode_tokens], [text_encoder.rs:ShardedSafetensors], [vae.rs:VaeDecoder], [vae.rs:encode_rgb8_png], [rope.rs:RopeEmbedder], [scheduler.rs:FlowMatchEulerScheduler], [patchify.rs:patchify_image], [patchify.rs:unpatchify].
- Prompting: [conditioning.rs:frame_prompt], [conditioning.rs:tokenize_prompt], [conditioning.rs:load_tokenizer].
- Install: [install.rs:ImageManifest] (`load`, `validate`, `verify_files`), [install.rs:ImageFamily] (`license`; ZImage, QwenImage21, QwenImage21Turbo, Krea2Turbo), [install.rs:image_family_for_model_id] (unknown ids fall back to the Z-Image envelope, so an unpinned source cannot loosen validation), [builder.rs:build_image_install], [builder.rs:build_image_install_with_progress], [packed.rs:PackedTensorStore], [packed.rs:pack_component].
- Reporting: [runtime.rs:image_quantization_label] derives the quantization label from an install manifest (legacy INT4, unquantized, or MLX affine with observed bit widths); the PNG carries an iTXt chunk keyed `turbospark` with the request metadata JSON [runtime.rs:encode_png_with_metadata].
- Fixtures: [fixtures.rs:read_npz], [fixtures.rs:read_npy_f32] and siblings.
- Subdirectory: [./shaders/EGM.md](./shaders/EGM.md) (Metal kernels).

## Invariants
- `#![forbid(unsafe_code)]` applies crate-wide [lib.rs:forbid(unsafe_code)].
- Per-family envelopes in [install.rs:ImageFamily::supported] are enforced by `ImageManifest::validate`; the request check below has no family gate and compares against the Z-Image constants, so a change to them changes which families can generate.
- Supported dimensions are 512 to 1024 per side, multiples of 16, at most 1024x1024 pixels [runtime.rs:is_supported_image_dimensions].
- `ImageRequest::validate` rejects empty model id, revision, prompt, noise provenance, unsupported quantization label ([runtime.rs:is_supported_quantization_label]), and any batch, step count, or guidance other than the fixed envelope (batch 1, 9 steps, guidance 0.0) [runtime.rs:ImageRequest::validate, install.rs:IMAGE_STEPS].
- `generate` checks cancellation between stages [runtime.rs:generate].
- Packed stores use group size 64 and the magic `turbospark.image.packed.v1` [packed.rs:PACKED_GROUP_SIZE, packed.rs:PACKED_MAGIC].
- Install manifests are capped at 4 MiB and carry magic `GTURBO` schema 1 [install.rs:IMAGE_MANIFEST_MAX_BYTES, install.rs:IMAGE_MANIFEST_MAGIC].
- Installs are published by staging then rename so a partial tree is never exposed [builder.rs:build_image_install_with_progress].
- Metal module is macOS only [lib.rs:cfg(target_os = "macos")].

## Side Effects
- Disk: reads safetensors shards and packed `tensors.bin`; `pack_component` refuses an existing output directory, then creates it [packed.rs:pack_component]; the builder creates a staging dir, copies scheduler config, writes `manifest.json` and the install receipt [builder_files.rs:write_receipt], verifies files via `verify_files_for_publication`, then renames and removes staging on failure [builder.rs:build_image_install_with_progress].
- GPU: Metal command buffers are committed inside `gpu::autorelease_pool` around `pass.commit()` [metal_ops.rs:459].
- Environment: `TURBOSPARK_IMAGE_STREAM_PROGRESS` and `TURBOSPARK_IMAGE_FORCE_SYNC_DISPATCH` are read by the Metal ops [metal_ops.rs:TURBOSPARK_IMAGE_STREAM_PROGRESS]. `TURBOSPARK_IMAGE_RESIDENCY` is read only under `cfg(test)` [metal_ops.rs:streamed_requested].

## Verification
- Run: `cargo test -p turbospark-image`
