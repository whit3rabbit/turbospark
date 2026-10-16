---
description: "How to read safetensors and GGUF headers, fetch byte ranges, quantize matrices, and convert a checkpoint with turbospark-repack."
icon: box
---
# Checkpoint intake

Start with the header. Both formats keep their tensor index at the start of the file, so you can read the layout without downloading the weights. Read the header, pick the converter for your model family, and then write the install. This page covers the readers, the quantizers, and the two converter paths. The install writers are on [Write a .gturbo install](write-gturbo-install.md).

## Read a safetensors header

A safetensors file starts with an 8-byte little-endian length, then that many bytes of JSON describing each tensor. `fetch_safetensors_header` reads the length, then reads exactly the header bytes. `absolute_range` turns a tensor name into a byte range in the file. This example builds a one-tensor file in memory and reads it back. It compiles and runs as written.

```rust
use turbospark_repack::{fetch_safetensors_header, MemoryRangeSource, RangeSource};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // A one-tensor safetensors file: an 8-byte length prefix, the JSON header,
    // then the raw tensor bytes.
    let header_json = br#"{"w":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]}}"#;
    let mut file = (header_json.len() as u64).to_le_bytes().to_vec();
    file.extend_from_slice(header_json);
    for v in [1.0f32, 2.0, 3.0, 4.0] {
        file.extend_from_slice(&v.to_le_bytes());
    }

    let source = MemoryRangeSource::new(&file);
    let header = fetch_safetensors_header(&source)?;
    let (start, end) = header.absolute_range("w").ok_or("tensor w is missing")?;
    let bytes = source.read_range(start, end)?;
    let values: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    println!("w = {values:?}");
    Ok(())
}
```

If you already hold the leading bytes of a file, call `parse_header(leading_bytes, max_bytes)` instead. The crate's default cap is `DEFAULT_MAX_HEADER_BYTES`, which is 64 MiB. A larger declared header fails with `SafetensorsHeaderError::HeaderTooLarge`.

## Read a remote checkpoint

`HttpRangeSource` implements `RangeSource` over an HTTPS URL, so the same header call works on a remote file. The first request fetches the 8-byte prefix, and the second fetches exactly the header. The example below uses a placeholder URL. It compiles, but I did not run it because it needs network access and a real file.

```rust
use turbospark_repack::{fetch_safetensors_header, HttpRangeSource};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Placeholder URL. This binary is compiled, not run.
    let url = "https://huggingface.co/example-org/example-model/resolve/main/model.safetensors";
    let source = HttpRangeSource::new(url);
    let header = fetch_safetensors_header(&source)?;
    println!("{} tensors", header.tensors.len());
    Ok(())
}
```

The builder methods on `HttpRangeSource` add the options you are likely to need:

- `with_token` and `with_optional_token` attach a bearer token. The source sends it only when the request URL passes `hf_token_allowed_for_url`, which accepts https addresses on `huggingface.co` or `hf.co` and their subdomains. A mirror or a non-Hugging-Face host never receives the token.
- `HttpRangeSource::with_progress(url, callback)` is a constructor, not a chained method. It sets a byte-progress callback.
- `with_cancel` takes a `CancelFlag`. Once the flag fires, reads return `DownloadError::Cancelled`.
- `with_cache_dir` keeps completed ranges on disk for later runs.
- `with_pinned_identity` binds cached ranges to a known source size and optional ETag.

To copy a whole remote file, `download_to(destination, total_bytes)` writes through a `.partial` file and renames it into place. It refuses to overwrite an existing destination.

## Read a GGUF header

GGUF has no length prefix, so `fetch_gguf_header` cannot know the header size in advance. It reads 1 MiB first and grows the read until the parser has the whole header. A file that ends before its own header returns `DownloadError::TruncatedGguf`. The result is a `GgufHeader`, which the GGUF converter takes directly.

## Quantize a matrix

`quantize_matrix_int4` and `quantize_matrix_int8` take a row-major `f32` matrix and return one affine row per matrix row. The INT4 form returns the compute crate's `Int4AffineRow`, and the INT8 form returns `Int8AffineRow`. Each row has public `packed`, `scales`, and `biases` fields. The scales and biases are BF16 bit patterns stored as `u16`.

The inputs must satisfy two rules:

- `data.len()` must equal `rows * cols`. Otherwise you get `RepackError::ShapeMismatch`.
- `cols` must be a multiple of 64, the crate's group size. A zero `cols` also fails. Otherwise you get `RepackError::RowNotGroupAligned`.

The crate has an internal helper that concatenates rows into a `ResidentTensorSpec`, but it is not re-exported at the root. The install example on the [write page](write-gturbo-install.md) shows the concatenation in full.

## Convert a Llama-style safetensors checkpoint

`orchestrate_llama_checkpoint` walks a checkpoint with the standard Llama tensor names and returns one quantized `ResidentTensorSpec` per resident weight. Its arguments are the parsed header, a `RangeSource`, and a `LlamaCheckpointDims` value. The dims are `hidden_size`, `intermediate_size`, `num_heads`, `num_kv_heads`, `head_dim`, `num_layers`, `vocab_size`, and `tie_word_embeddings`. Read them from the checkpoint's own `config.json`.

The walk expects names such as `model.embed_tokens.weight` and `model.layers.N.self_attn.q_proj.weight`. It reads `BF16` and `F32` tensors, and an `F16` tensor returns `OrchestrateError::UnsupportedDtype`. A missing tensor returns `OrchestrateError::MissingTensor`.

Every tensor the walk returns is INT4 quantized, norms included. The source comments say that choice serves a valid install, not a production numerical treatment of norms. The `llama` key is also absent from the Hugging Face table in `arch_registry`, so this function does not pick a model family. Whether the result runs is a runtime and catalog decision.

## Parse a config

Three parsers turn a `config.json` string into an `ArchConfig`:

- `parse_gemma4_config(json: &str) -> Result<ArchConfig, Gemma4Error>`
- `parse_qwen2_config(json: &str) -> Result<ArchConfig, Gemma4Error>`
- `parse_qwen3_vl_config(json: &str) -> Result<ArchConfig, Gemma4Error>`

Each parser refuses a config that belongs to another family. To check whether the crate recognizes a Hugging Face `model_type` at all, call `hf_family_for_model_type(model_type: &str) -> Option<ModelFamily>`. A `None` result means the crate has no family for that type, and you should stop there.

## GGUF installs

`write_gguf_install_streamed` takes the header, a source, a model ID, and a progress callback. It writes the resident tensors first, then one expert layer at a time, and returns the `ArchConfig` it wrote. The example below uses a synthetic Qwen2 GGUF held in memory. It compiles and runs, and it writes a tiny install with no routed experts.

```rust
use std::path::Path;

use turbospark_repack::{
    build_synthetic_qwen2_gguf, fetch_gguf_header, verify_install_full_sha256,
    write_gguf_install_streamed, MemoryRangeSource,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = Path::new("out/synthetic-qwen2");
    // Stand-in for a real GGUF: a synthetic file held in memory.
    let (bytes, _ranges) = build_synthetic_qwen2_gguf();
    let source = MemoryRangeSource::new(&bytes);
    let header = fetch_gguf_header(&source)?;
    let arch = write_gguf_install_streamed(
        dir,
        &header,
        &source,
        "example/synthetic-qwen2",
        |line| println!("{line}"),
    )?;
    verify_install_full_sha256(dir, &arch)?;
    println!("wrote and verified {}", dir.display());
    Ok(())
}
```

For a real download, swap `MemoryRangeSource` for an `HttpRangeSource`, and change the directory to your own. The resumable variant, `write_gguf_install_streamed_resumable`, takes an optional `ResumeProvenance`. It adopts leftover expert layers from an earlier run only when the provenance matches the record that run wrote. Otherwise it deletes them.

The GGUF path maps `general.architecture` strings through a table in `arch_registry`. These are the strings the registry marks as supported, with their families:

| `general.architecture` | Family |
|---|---|
| `gemma4` | `Gemma4` |
| `llama` | `Llama` |
| `qwen35moe` | `QwenGdnMoe` |
| `qwen35` | `QwenGdnDense` |
| `qwen3moe` | `Qwen3Moe` |
| `qwen4exp` | `Qwen4Exp` |
| `gpt-oss` | `GptOss` |
| `spark2_5` | `Spark25` |
| `qwen3` | `Qwen3Dense` |
| `qwen2` | `Qwen2Dense` |
| `minimax-m2` | `MiniMaxM2` |
| `deepseek2` | `Deepseek2` |

Being in this table means the registry has a baseline, a name mapping, and a decode flow for the string. It does not mean every checkpoint with that string is in the catalog. The `llama` string covers dense Llama and Mixtral. Intake accepts both halves, and whether the runtime runs a given install is decided when it opens the model. The GGUF walk also refuses the `DeepseekV4Flash` family with `GgufRepackError::UnsupportedFamily`.

## Errors on this path

The errors you will match on most are:

- `DownloadError::Header` for a malformed safetensors header, and `DownloadError::TruncatedGguf` for a GGUF file that ends inside its header.
- `OrchestrateError::MissingTensor` and `OrchestrateError::UnsupportedDtype` for the Llama walk.
- `RepackError::ShapeMismatch` and `RepackError::RowNotGroupAligned` for the quantizers.
- `GgufRepackError::UnsupportedFamily` for an architecture the GGUF walk does not handle.

Each one implements `std::error::Error`, so `?` in `main` returns it as a boxed error. Match on the variant only when your code needs to react to it differently.
