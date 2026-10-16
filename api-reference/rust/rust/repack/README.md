---
description: "Overview of the turbospark-repack crate, which reads a model checkpoint and writes a .gturbo install directory."
icon: box
---
# turbospark-repack

`turbospark-repack` reads a model checkpoint and writes a `.gturbo` install directory, the layout the TurboSpark runtime loads. Use it when you are building a converter or an intake step for a model. It does not run a model. A successful write shows that the files are well formed, not that the model produces correct output.

The package is `turbospark-repack`. Rust code imports it as `turbospark_repack`.

Inside the TurboSpark workspace, the crate's `Cargo.toml` renames some dependencies. For example, `turbospark-core` appears as `foundation`, and `turbospark-model-io` appears as `model_io`. Those short names are internal to the workspace. Outside it, depend on the package names and import them by their Rust paths, such as `turbospark_model_io`.

## Add it to a project

Add the package from crates.io once the version you need is published. Until then, use a path dependency or a git dependency from a checkout:

```toml
[dependencies]
turbospark-repack = { path = "../turbospark/crates/repack" }
```

Most converter code also names types from two other packages. `ArchConfig` and the other model types come from `turbospark-model-io` (`turbospark_model_io`), and `SteeringMode` comes from `turbospark-core` (`turbospark_core`). Add those only when your code names them.

## What a checkpoint must look like

The crate reads two source formats. Each has its own entry point and its own limits.

- **Hugging Face safetensors:** A `config.json` and one or more `.safetensors` files. Tensor names must follow the family's convention, and the config must parse under that family's parser. Gemma 4 checkpoints can span shards described by `model.safetensors.index.json`, which `Gemma4Shards` reads.
- **GGUF:** One `.gguf` file whose `general.architecture` string appears in the supported list on the [checkpoint intake](checkpoint-intake.md#gguf-installs) page. For split GGUF status, see [docs/MODELS.md](https://github.com/whit3rabbit/turbospark/blob/main/docs/MODELS.md).

The repo's design probes the header, config, and tensor names before any weights are streamed. Check the intake page for the exact calls each path makes.

## What you get

The output is a directory with `manifest.json`, `model_weights.bin`, and `packed_experts/` containing `layout.json` and one `layer_NN.bin` per routed MoE layer. A vision component adds `packed_vision/`, and a Hadamard-rotated install adds `hadamard.bin`. The byte layout is in the [`.gturbo` format spec](https://github.com/whit3rabbit/turbospark/blob/main/docs/GTURBO.md), and this crate does not repeat it.

## Public surface

| Task | Main items | Page |
|---|---|---|
| Read a safetensors header | `fetch_safetensors_header`, `parse_header`, `SafetensorsHeader`, `TensorInfo` | [Checkpoint intake](checkpoint-intake.md) |
| Read a GGUF header | `fetch_gguf_header`, `parse_gguf_header`, `GgufHeader` | [Checkpoint intake](checkpoint-intake.md) |
| Read byte ranges | `RangeSource`, `HttpRangeSource`, `MemoryRangeSource`, `CancelFlag` | [Checkpoint intake](checkpoint-intake.md) |
| Quantize a matrix | `quantize_matrix_int4`, `quantize_matrix_int8`, `RepackError` | [Checkpoint intake](checkpoint-intake.md) |
| Convert a Llama-style safetensors checkpoint | `orchestrate_llama_checkpoint`, `LlamaCheckpointDims` | [Checkpoint intake](checkpoint-intake.md) |
| Parse a config | `parse_gemma4_config`, `parse_qwen2_config`, `parse_qwen3_vl_config` | [Checkpoint intake](checkpoint-intake.md) |
| Write a GGUF install | `write_gguf_install_streamed`, `write_gguf_install_streamed_resumable` | [Checkpoint intake](checkpoint-intake.md) |
| Write an install from tensors | `write_gturbo_install_with_resident_index`, `build_resident_weights_bin`, `ResidentTensorSpec` | [Write a .gturbo install](write-gturbo-install.md) |
| Stream expert layers to disk | `StreamingGturboWriter`, `LayerBlobs`, `ExpertBlob`, `SubTensor` | [Write a .gturbo install](write-gturbo-install.md) |
| Verify an install | `verify_install_full_sha256` | [Write a .gturbo install](write-gturbo-install.md) |
| Steering vectors | `control_vector::parse_control_vector`, `control_vector::load_control_vector`, `control_vector::write_control_vector` | This page, below |
| Test fixtures | `build_synthetic_*`, `tiny_*_arch` | This page, below |

Most items are re-exported at the crate root, so `use turbospark_repack::...` reaches them directly. The full reference is on [docs.rs](https://docs.rs/turbospark-repack/latest/turbospark_repack/).

### Steering vectors

A control vector holds one steering direction per block. `parse_control_vector` reads one from bytes, and `load_control_vector` reads one from a path. `write_control_vector` takes a `BTreeMap<usize, Vec<f32>>` keyed by 0-based block index, an architecture string, and an optional `SteeringMode`.

`write_control_vector` refuses a direction for block 0 with `ControlVectorError::LayerZeroNotExpressible`. The file format numbers blocks from 1, so block 0 cannot be applied, and the writer refuses it rather than dropping the edit silently.

### Test fixtures

The `build_synthetic_*` functions and the `tiny_*_arch` helpers build small installs for the crate's own tests. Their output is a fixture. Do not treat it as a model, and do not benchmark it.

## Errors

Each error type this crate returns implements `std::error::Error`, so `?` converts it into `Box<dyn Error>`. The types you will see most are `DownloadError` for reads and `RepackError` for quantizer input. `WriterError` covers install files, `GgufRepackError` covers the GGUF walk, `OrchestrateError` covers the Llama converter, and `Gemma4Error` covers config parsing. Check each function's return type for the one it uses.

## Limits to know before you build on this

- **Intake is not model quality:** A header parse, a synthetic fixture, or a clean write shows structure. It does not show that the runtime generates correct tokens. The gates for that are in [NEW_MODEL.md](https://github.com/whit3rabbit/turbospark/blob/main/docs/NEW_MODEL.md) and [MODELS.md](https://github.com/whit3rabbit/turbospark/blob/main/docs/MODELS.md).
- **Verification checks consistency, not origin:** `verify_install_full_sha256` re-hashes every file the manifest lists. Anyone who can replace both the files and the manifest passes it.
- **This crate does not decide what runs:** The architecture tables in `arch_registry` only choose the error message a refused checkpoint gets. Whether a family runs is a runtime and catalog decision. For example, the `llama` GGUF architecture covers dense and MoE models, and the dense half is refused by the runtime, not at intake.
- **The crate forbids `unsafe` code:** Its root sets `#![forbid(unsafe_code)]`.
- **Network reads use a blocking client:** `HttpRangeSource` uses `reqwest` with rustls, and it speaks HTTP/1.1 only. Connections time out after 30 seconds, and requests time out after 600 seconds. It attaches a Hugging Face token only when the request URL is an https address on `huggingface.co` or `hf.co`.
- **Platform coverage is narrow:** I built and ran the examples on macOS (Darwin 25.6.0) with rustc 1.98.1. The workspace minimum Rust version is 1.82 per the root `AGENTS.md`, and I did not test that version. I did not build on other operating systems.

## Where to go next

- [Checkpoint intake](checkpoint-intake.md) covers reading headers, quantizing matrices, and the two converter paths.
- [Write a .gturbo install](write-gturbo-install.md) covers the install writers and verification.
- [The .gturbo format](https://github.com/whit3rabbit/turbospark/blob/main/docs/GTURBO.md) defines the files on disk.
- [Wiring a new model family](https://github.com/whit3rabbit/turbospark/blob/main/docs/NEW_MODEL.md) is the path for adding a family that this crate does not cover.
- The sibling [model-io](../model-io/README.md) page covers the `ArchConfig` and manifest types that every install needs.
