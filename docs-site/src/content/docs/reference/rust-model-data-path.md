---
title: "Rust API: Model Data Path"
description: "Public API of the install-and-execute data path crates (turbospark-model-io, turbospark-repack, turbospark-streaming, turbospark-gpu, turbospark-vision-io), generated from cargo doc and source doc comments."
diataxisType: "reference"
---

<!-- generated: rust lane, signal: Cargo.toml -->

This page covers the crates that take a checkpoint from the network to the
GPU: manifest validation and the mmap resident index, the `.gturbo`
repack/intake walk, the routed-expert `pread` streamer, the Metal backend,
and portable vision preprocessing. Signatures are read from source; doc
text is quoted or condensed from `///` comments. The full rustdoc
inventory is under `target/doc/` after `cargo doc --no-deps`.

## turbospark-model-io (`crates/model-io`)

Model install layout: manifest/arch validation, packed-expert layout,
resident tensor index, SHA-256 verification, and the trusted install
receipt. Ported from `Infrastructure/ModelIO` minus the Metal-specific
buffer wrapping, which belongs to the `gpu` crate. This crate carries a
narrow amount of `unsafe`, confined to the `mmap` call in
`resident_buffer`. 143/149 public items documented.

The crate root re-exports the module surfaces: `arch_baselines`,
`arch_config`, `context_policy`, `error`, `expert_cache_policy`,
`install_receipt`, `load_guard`, `manifest`, `ngram_hash`, `ngram_table`,
`packed_experts_layout`, `resident_buffer`, `resident_index`, `sha256`,
`steering_set`.

| Symbol | Kind | Signature / doc |
|---|---|---|
| `ArchConfig` | struct | The per-architecture shape table. `full_attention_layer_mask` values: 0 = sliding-window attention, 1 = full attention, 2 = gated-DeltaNet linear attention, 3 = compressed sparse attention (CSA), 4 = heavily compressed attention (HCA). |
| `known_architecture` | fn | `pub fn known_architecture(family: ModelFamily) -> ArchConfig`. Registry keyed by `manifest.arch.family` for auto-detection at load. |
| `Manifest` | struct | Top-level model manifest deserialized from `manifest.json`. |
| `ResidentIndex` | struct | Complete resident index including header and named tensor entries map. |
| `ResidentBuffer` | struct | `mmap`'d window covering `[file_offset, file_offset + resident_size)` inside a file, exposed as a byte slice starting at the resident bytes. |
| `gemma4_26b_a4b` | fn | Canonical Gemma 4 26B-A4B baseline; `intermediate_size = 2112` is the shared-expert FFN width (3x moe). |
| `gpt_oss_20b` | fn | Canonical `gpt-oss-20b` baseline: 24 layers, hidden 2880, 64 query heads over 8 KV heads at head_dim 64, 32 routed experts at top-4. |
| `mixtral_8x7b` | fn | Canonical Mixtral-8x7B-Instruct baseline: 32 dense full-attention layers with GQA, 8 routed experts. |
| `qwen_gdn_dense_27b` | fn | Canonical `qwen3_5` baseline: a 64-layer hybrid of 48 gated-DeltaNet linear-attention layers and 16 full-attention layers (every 4th), a dense SwiGLU FFN. |
| `muse_glimmer_30b` | fn | Canonical `Muse-Glimmer-30B` baseline: 52 dense layers, GQA at 32 query heads over 2 KV heads, an alternating three-sliding/one-full window. |
| `muse_glimmer_layer_mask` | fn | The `[0, 0, 0, 1]` window pattern, repeated to `num_layers`. |
| `deepseek_v4_flash_284b_a13b` | fn | Canonical DeepSeek-V4-Flash 284B-A13B baseline: 43 all-MoE layers, shared-KV MQA attention, sliding window 128 on every layer. |

## turbospark-repack (`crates/repack`)

`#![forbid(unsafe_code)]`. `.gturbo` repack support: safetensors header
parsing, ranged-download planning, per-row int4/int8 quantization repack,
and full-SHA256 install verification. Ported from
`Infrastructure/ModelIO/VerifiedInstallReceipt.swift` and the intent of the
HF streaming installer described in the ROADMAP. 324/356 public items
documented; this crate also carries the synthetic model builders the
fixtures use (`synthetic_gguf`, `synthetic_llama`, `synthetic_model`,
`synthetic_muse`, `synthetic_qwen`, `synthetic_real`).

The crate root re-exports: `arch_registry`, `gemma4_checkpoint`,
`gguf_checkpoint`, `gguf_config`, `gguf_header`, `gguf_names`,
`gturbo_writer`, `hf_checkpoint`, `install_verifier`, `manifest_peek`,
`museglimmer_config`, `qwen36_config`, `ranged_download`, `repack`,
`resident_reader`, `resident_writer`, `safetensors_header`, and the
synthetic builders above.

| Symbol | Kind | Signature / doc |
|---|---|---|
| `PlannedArch` | struct | A recognized architecture that has no decode flow here. |
| `ArchSupport` | enum | What this port can do with an architecture string. |
| `gguf_arch_support` | fn | Resolve a GGUF `general.architecture` string. `None` means the string is not recognized at all, which is a different message from `Planned`. |
| `hf_family_for_model_type` | fn | Resolve an HF `config.json -> model_type` string; deliberately carries only the supported rows. |
| `config_json_family` | fn | The family an HF `config.json` claims, or `None` when it claims nothing this port recognizes; reads the root `model_type` and then `text_config.model_type`. |
| `refuse_foreign_config` | fn | Guards a family-specific `config.json` parser against being handed another family's file; the failure it prevents is silent. |
| `describe_gguf_architecture` | fn | One sentence explaining what this port makes of an architecture string, for an error message. |
| `read_tensor` | fn | `pub fn read_tensor(header: &GgufHeader, source: &dyn RangeSource, name: &str) -> Result<Vec<u8>, GgufRepackError>` (`gguf_checkpoint/types.rs`). Reads one tensor's bytes through the range source. |
| `planned_gguf_architectures` | fn | Every planned GGUF row, for the network probe that re-reads each witness. |

## turbospark-streaming (`crates/streaming`)

Routed-expert streaming: `pread`-based cache streamer, LFU/LRU eviction
policy, and readahead advice. Ported from `Infrastructure/Streaming`.
Carries a narrow amount of `unsafe` and platform `cfg` (the macOS
`F_RDADVISE` `fcntl` in `rdadvice`, and `disk_io`'s `proc_pid_rusage` and
`F_NOCACHE`). 57/60 public items documented.

The crate root re-exports: `aligned_slot`, `disk_io`, `error`,
`expert_cache`, `mapped_experts`, `pread_streamer`, `rdadvice`,
`stream_layout`.

| Symbol | Kind | Signature / doc |
|---|---|---|
| `AlignedSlot` | struct | One expert slot's backing memory: page-aligned, page-rounded, allocated once. Exposes its base pointer so a GPU backend can wrap it no-copy. |
| `ExpertIoStats` | struct | Cumulative byte accounting for one streamer's cache-miss reads. `requested` is free and always collected; `physical` and `samples` are zero unless sampled. |
| `amplification` | fn | Physical bytes per requested byte, or `None` when nothing was sampled or nothing was requested. Above 1.0 is read amplification. |
| `ExpertCachePolicy` | enum | Cache eviction policy for routed experts. |
| `ExpertCache` | struct | Fixed per-layer slot cache: which expert each slot currently holds, plus the bookkeeping (last-use clock, per-expert hit counts, in-flight speculative reads) the eviction policy reads. |

## turbospark-gpu (`crates/gpu`)

Metal GPU backend: device/pipeline-cache context and per-kernel dispatch,
validated against `turbospark-compute`'s CPU reference kernels. macOS-only
by design; on other platforms this crate exposes nothing so
`cargo build --workspace` still succeeds on Linux CI. 338/346 public items
documented. The kernel inventory spans the `dequant_*_gemv` family, the
`moe_*` prefill/decode batch kernels, `gdn`, `ple`, `qsa_indexer`,
`hyper_connection`, `attention_decode`, `attention_indexed`, `rope`,
`rms_norm`, `utility`, and `vision`.

| Symbol | Kind | Signature / doc |
|---|---|---|
| `MetalContext` | struct | Owns the Metal device and command queue, and caches one `ComputePipelineState` per (library source, function name) pair so repeated dispatches of the same kernel skip recompilation. |
| `KvCacheManager` | struct | KV cache manager orchestrating per-layer Metal buffers for token generation. |
| `attention_decode` | fn | `Q: [num_q_heads, head_dim]`, `K`/`V: [seq_len, num_kv_heads, head_dim]` (same layout `turbospark_compute::causal_attention` uses); returns `[num_q_heads, head_dim]`. |
| `encode_attention_decode` | fn | Encoder-level variant: Q at a `(buffer, byte offset)` view, K/V read in place from persistent cache buffers, output written in place. |
| `attention_decode_buffers` | fn | Same two-pass decode attention, but K and V are read straight out of caller-owned persistent buffers (`KvCacheManager`'s per-layer K/V). |
| `encode_attention_decode_indexed` | fn | Q at a `(buffer, byte offset)` view; K/V read in place from linear `[stored_tokens, num_kv_heads, head_dim]` buffers over an explicit position list. |
| `attention_decode_indexed` | fn | Slice-taking entry for parity tests, mirroring `attention_decode`: uploads Q/K/V and the position list, runs both passes, reads the result back. |
| `AttentionScratch` | struct | Caller-owned scratch for the two-pass decode attention: the partial max/denominator/output accumulators the combine pass folds. |

## turbospark-vision-io (`crates/vision-io`)

Portable vision preprocessing for the qwen3_5 vision tower: image decode,
PIL-bicubic smart resize, normalize, patchify, and the three position
tables the tower and the trunk need. Nothing here touches Metal, macOS, or
a model install; plain arithmetic on `Vec<f32>`, so it builds and tests
for a non-macOS target. 37/51 public items documented.

Public modules: `decode`, `error`, `mrope`, `normalize`, `params`,
`patchify`, `pos_embed`, `preprocess`, `resize`, `rope`, `rounding`,
`smart_resize`.

| Symbol | Kind | Signature / doc |
|---|---|---|
| `Rgb8Image` | struct | Decoded pixels, 8 bits per sample, three interleaved channels, row-major; interleaved rather than planar because that is what every decoder hands back. |
| `decode_image_bytes` | fn | Decode an encoded image, converting to RGB8; grayscale is broadcast to three channels and alpha is dropped, matching the reference. |
| `decode_image_file` | fn | Read and decode an image file. |
| `PreprocessedImage` | struct | One preprocessed image: the patch matrix, its grid, and the token cost. |
| `preprocess` | fn | `pub fn preprocess(image: &Rgb8Image, params: &PreprocessParams) -> Result<PreprocessedImage, VisionIoError>`. Run the pipeline over decoded pixels; the order is fixed and each step depends on the previous one's exact output. |
| `ImageSpan` | struct | Where one image's placeholder tokens sit in the id sequence. |
| `VisionSpecialIds` | struct | The special ids the mrope walk keys on. |
| `MropePositions` | struct | The result of the mrope walk. |

## Coverage for this tier

Per-crate public-item doc coverage (excluding `pub use` re-exports, parsed
from `crates/*/src`): model-io 143/149, repack 324/356, streaming 57/60,
gpu 338/346, vision-io 37/51. Repack and vision-io carry the largest
undocumented shares in the workspace; most of the gaps are synthetic
fixture builders (repack) and low-level arithmetic helpers (vision-io).
