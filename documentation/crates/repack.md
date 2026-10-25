---
description: "Checkpoint intake for the .gturbo format: safetensors and GGUF header parsing, ranged downloads, INT4/INT8 repack, streamed install writing, and full-SHA256 verification."
icon: boxes
---

# repack

`crates/repack` (package `turbospark-repack`, library `turbospark_repack`)
turns raw Hugging Face safetensors and GGUF checkpoints into `.gturbo`
install directories. It plans byte-range downloads so multi-gigabyte
shards are never staged whole, quantizes float matrices into the affine
INT4/INT8 rows the runtime's GEMV kernels expect, separates resident trunk
weights from streamable MoE expert blobs, and writes the manifests,
layouts, and checksums the `model-io` loaders then read back. It is the
writer half of the format contract; `model-io` is the reader half.

## What it does

| Area | Modules | Contents |
| --- | --- | --- |
| Header parsing | `src/safetensors_header.rs`, `src/gguf_header/` | Parse safetensors and GGUF headers from leading bytes only, with hard byte caps on untrusted metadata |
| Ranged downloads | `src/ranged_download/` | The `RangeSource` trait, `HttpRangeSource` (cancel, pause and resume, HF token policy, local cache), and header-only fetches |
| Quantization | `src/repack.rs` | Per-row affine INT4/INT8 matrix repack, reusing `turbospark-compute`'s quantizer |
| Resident blobs | `src/resident_writer.rs`, `src/resident_reader.rs` | Assemble and read back `model_weights.bin` (index plus tensor region) |
| Install writing | `src/gturbo_writer/` | `write_gturbo_install` and `StreamingGturboWriter`: byte-exact assembly of `packed_experts/`, `layout.json`, `model_weights.bin`, and `manifest.json` |
| Checkpoint pipelines | `src/hf_checkpoint.rs`, `src/gemma4_checkpoint/`, `src/gguf_checkpoint/`, `src/fish_drama3.rs` | Per-family orchestration from `config.json` plus shards to a streamed install (Gemma 4, Qwen lines, GGUF architectures, Fish Drama 3) |
| Verification | `src/install_verifier.rs` | `verify_install_full_sha256` over every file the manifest lists |
| Registry and probes | `src/arch_registry.rs`, `src/manifest_peek.rs`, `src/trained_context.rs` | Family resolution from HF `model_type`, `config.json`, or GGUF architecture; manifest peeking; trained-context metadata |
| Synthetic installs | `src/synthetic_*.rs` | Deterministic tiny installs per family, used by the offline test suite |

## Where it sits

`turbospark-repack` depends on `turbospark-core` (aliased `foundation`),
`turbospark-compute` (the quantizers it repacks with), and
`turbospark-model-io` (the format contract it writes for). It adds
`reqwest` (blocking client, rustls TLS) for ranged downloads. Downstream,
`catalog` and `cli` drive intake, while `gpu`, `runtime`, `image`,
`server`, `ffi`, and `bench` consume the installs it produces. The crate is
`#![forbid(unsafe_code)]` throughout and has no Cargo feature flags; the
`recursion_limit = "512"` at the top of `src/lib.rs` exists because the
manifest writer's single `serde_json::json!` literal writes every arch
field.

## Public API tour

The crate root in `crates/repack/src/lib.rs` re-exports the surface.
The entry points you will reach for most often:

| Item | Path | Meaning |
| --- | --- | --- |
| `parse_header(bytes, max_bytes)`, `required_prefix_len`, `SafetensorsHeader::absolute_range(name)` | `src/safetensors_header.rs` | Safetensors header parse and absolute byte-range lookup |
| `parse_gguf_header`, `GgufHeader`, `GgufValue`, `ggml_type_name`, `ggml_type_block` | `src/gguf_header/` | GGUF header parse and ggml type layout facts |
| `HttpRangeSource::new(url)` with `.with_token`, `.with_cancel`, `.with_cache_dir`, `CancelFlag` | `src/ranged_download/http.rs` | HTTP range source with cancellation, pause, and resume |
| `fetch_safetensors_header(&dyn RangeSource)`, `fetch_gguf_header` | `src/ranged_download/mod.rs` | Read only the header bytes of a remote file |
| `quantize_matrix_int4(data, rows, cols)`, `quantize_matrix_int8` | `src/repack.rs` | Row-major FP32 matrix to quantized rows |
| `build_resident_weights_bin`, `build_resident_weights_bin_mixed`, `read_resident_entries` | `src/resident_writer.rs`, `src/resident_reader.rs` | `model_weights.bin` assembly and read-back |
| `write_gturbo_install`, `write_gturbo_install_with_resident_index_and_experts` | `src/gturbo_writer/layers.rs` | Whole-install assembly from finished blobs |
| `StreamingGturboWriter` (`write_layer`, `adopt_layer`, `finish_streaming`) | `src/gturbo_writer/streaming.rs` | Feed one layer at a time; each layer file hits disk immediately |
| `LayerBlobs`, `ExpertBlob`, `SubTensor`, `WriterError` | `src/gturbo_writer/types.rs` | The writer's input types (roles, bytes, dtype strings, shapes) |
| `verify_install_full_sha256(dir, &arch)` | `src/install_verifier.rs` | Hash every manifest-listed file and compare |
| `orchestrate_llama_checkpoint(&header, &source, dims)` | `src/hf_checkpoint.rs` | Dense Llama/Qwen safetensors walk producing resident tensor specs |
| `write_qwen_gdn_moe_install_streamed`, `write_gguf_install_streamed`, `orchestrate_gemma4_checkpoint`, `write_fish_drama3_install_streamed` | `src/gemma4_checkpoint/`, `src/gguf_checkpoint/`, `src/fish_drama3.rs` | Family intake pipelines |
| `hf_family_for_model_type`, `gguf_arch_support`, `refuse_foreign_config` | `src/arch_registry.rs` | Resolve or refuse a checkpoint's family before streaming weights |
| `peek_manifest_arch(dir)` | `src/manifest_peek.rs` | Read an install's architecture without full validation |
| `GgufSet::new`, `gguf_shard_names` | `src/gguf_set.rs` | Split multi-file GGUF sets |
| `build_synthetic_gemma4_install`, `build_synthetic_qwen_gdn_moe_install`, `GgufBuilder`, ... | `src/synthetic_*.rs` | Deterministic test installs and fixtures |

## Examples

### Quantize matrices, write an install, and verify it

Adapted from `crates/repack/tests/gturbo_writer.rs` and `tests/repack.rs`
(simplified: the test builds a small toy `ArchConfig` literal; here a real
baseline stands in). The writer takes finished bytes; producing them from
a downloaded checkpoint is the caller's or a pipeline's job.

```rust
use turbospark_model_io::{known_architecture, ModelFamily};
use turbospark_repack::{
    quantize_matrix_int4, verify_install_full_sha256, write_gturbo_install,
    ExpertBlob, LayerBlobs, SubTensor,
};

let arch = known_architecture(ModelFamily::Gemma4);

// One quantized row per matrix row; columns must be group-aligned.
let data: Vec<f32> = (0..64).map(|i| i as f32 * 0.01).collect();
let quantized = quantize_matrix_int4(&data, 1, 64).unwrap();
assert_eq!(quantized.len(), 1);

// Blob bytes, roles, and dtype strings are the writer's contract with
// `model_io`'s packed-experts layout reader.
let layers = vec![LayerBlobs {
    layer: 0,
    experts: (0..2)
        .map(|expert| ExpertBlob {
            expert,
            sub_tensors: vec![
                SubTensor {
                    role: "gate".to_string(),
                    bytes: vec![0u8; 64],
                    dtype: "int4".to_string(),
                    shape: vec![64, 64],
                },
                SubTensor {
                    role: "gate_scales".to_string(),
                    bytes: vec![0xAB, 0xCD],
                    dtype: "bf16".to_string(),
                    shape: vec![1],
                },
            ],
        })
        .collect(),
}];

let dir = std::path::Path::new("/tmp/example.gturbo");
write_gturbo_install(dir, &arch, "toy-model", 4096, 2, &layers, &[1, 2, 3, 4])
    .unwrap();
verify_install_full_sha256(dir, &arch).unwrap();
```

### Stream a checkpoint straight from remote shards

From the real example `crates/repack/examples/fish_drama3_intake.rs`.
Shards are read by range and never staged whole; only headers are fetched
up front and canonicalized before the streamed write.

```rust
use turbospark_repack::{
    canonicalize_drama3_header, fetch_safetensors_header,
    write_fish_drama3_install_streamed, Gemma4Shards, HttpRangeSource,
    RangeSource,
};

// One source per shard; the prefix names an immutable revision and ends
// with '/', e.g. "https://huggingface.co/<repo>/resolve/<rev>/".
let sources: Vec<HttpRangeSource> = (1..=count)
    .map(|i| HttpRangeSource::new(format!("{prefix}model-{i:05}.safetensors")))
    .collect();

let mut headers = Vec::with_capacity(count);
for source in &sources {
    let mut header = fetch_safetensors_header(source)?;
    canonicalize_drama3_header(&mut header)?;
    headers.push(header);
}

let shards = Gemma4Shards::new(
    headers
        .iter()
        .zip(&sources)
        .map(|(h, s)| (h, s as &dyn RangeSource))
        .collect(),
)?;

let arch = write_fish_drama3_install_streamed(
    &out_dir,
    &config_text,
    &model_id,
    &shards,
    |stage| eprintln!("{stage}"),
)?;
```

### Build a synthetic install for tests

From `crates/repack/tests/synthetic_model.rs`. The synthetic builders write
a structurally valid install and return the `ArchConfig` to open it with,
so tests round-trip through the real `model-io` loaders without any
download.

```rust
use turbospark_model_io::{load_manifest, DEFAULT_MAX_BYTES};
use turbospark_repack::build_synthetic_gemma4_install;

let dir = std::env::temp_dir().join("turbospark-synthetic-example");
let arch = build_synthetic_gemma4_install(&dir, 300, 2, "tiny-test").unwrap();

let manifest = load_manifest(&dir, &arch, DEFAULT_MAX_BYTES).unwrap();
assert_eq!(manifest.arch.num_layers, 2);
```

## Gotchas

{% hint style="warning" %}
MLX mxfp4 and GGUF MXFP4 hold the same values in different nibble orders:
MLX packs adjacent elements per byte, while GGUF pairs element `j` with
`j + 16` and folds the exponent into a 17-byte block. Re-block with
`mlx_mxfp4_reblock_to_gguf`; reading one as the other yields a correctly
scaled permutation that correlates well and is wrong. An exponent byte of
255 is NaN in MLX but finite in ggml, so it is refused rather than copied.
{% endhint %}

{% hint style="warning" %}
Copy tokenizer sidecars (`tokenizer.json`, `tokenizer_config.json`,
`chat_template.jinja`, `preprocessor_config.json`) into the install intact.
Stripping or renaming them prevents dialect resolution at runtime.
{% endhint %}

Two more rules from the crate docs shape intake work. First, a header
parse or synthetic fixture proves intake structure only; independent
decoder or real-install checks are required before publishing a new intake
path, and the network test targets are opt-in (`cargo test -p
turbospark-repack --test <target> -- --ignored`) because they download real
weights. Second, when a multimodal or vision component is present, both
writers must be updated: a headless install can compile and still be
invalid for the advertised model.

## Related pages

- [model-io](model-io.md)
- [The .gturbo Format](../models/gturbo-format.md)
- [Wiring New Models](../developer/wiring-new-models.md)
- [Extreme Quantization](../advanced/extreme-quantization.md)
