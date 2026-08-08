# The .gturbo Model Installation Specification

This document provides a comprehensive specification of the `.gturbo` model installation directory format, its binary layout, streaming mechanics, and compatibility with the upstream [turbo-fieldfare](https://github.com/drumih/turbo-fieldfare) (Mference) inference engine.

---

## 1. Overview & Architectural Motivation

Traditional LLM file formats (such as monolithic `.gguf` files or `.safetensors` weight shards) pack all model parameters into single large binary files. To run inference, frameworks typically mmap or load the entire 20 GB to 35 GB parameter set into unified host/GPU RAM. On Apple Silicon systems with limited memory (8 GB, 16 GB, 24 GB, or 36 GB), this leads to excessive memory pressure or out-of-memory (OOM) failures.

`mrefrust` adopts the **`.gturbo`** installation format specified in upstream [`SYSTEM_DESIGN.md`](https://github.com/drumih/turbo-fieldfare/blob/main/docs/SYSTEM_DESIGN.md).

### Key Design Principle: Decoupled Working Set
The `.gturbo` format decouples a Mixture-of-Experts (MoE) model into two distinct layers:
1. **Resident Core**: Non-expert weight tensors that must be evaluated on every token (layer norms, linear attention states, embeddings, output head, and expert router projections). These stay mapped in host RAM/VRAM via zero-copy Metal buffers (`MTLBuffer`).
2. **Packed Experts**: Routed expert weight matrices stored on disk in contiguous, fixed-stride layer blobs (`packed_experts/layer_NN.bin`). During token generation, only the active expert slots selected by the layer router are streamed from high-speed SSD storage into host memory using OS `pread` calls and an LFU/LRU cache.

This decoupling allows 26B-35B parameter MoE models to run within a physical RAM footprint (`phys_footprint`) of only **~1.6 GiB to 2.2 GiB**.

---

## 2. Directory Layout & Upstream Compatibility

A `.gturbo` model installation is a directory structured as follows:

```
<model-name>.gturbo/
├── manifest.json            # Architecture parameters, quant metadata, file table & SHA-256 hashes
├── model_weights.bin        # Resident Index header + binary payload for non-expert resident tensors
├── packed_experts/
│   ├── layout.json          # Expert stride size, layer counts, and sub-tensor offset mapping
│   ├── layer_00.bin         # Expert weight blobs for layer 0 (fixed stride per expert)
│   ├── layer_01.bin         # Expert weight blobs for layer 1
│   └── ...
├── tokenizer.json           # Hugging Face tokenizer specification
└── chat_template.jinja      # Jinja2 chat template for conversation formatting
```

### Upstream Parity & Interoperability
- **Magic Signature**: `manifest.json` specifies `"magic": "GTURBO"` and `"versionMajor": 1`.
- **100% Binary Compatible**: Both `mrefrust` (Rust) and `turbo-fieldfare` (Swift) load, validate, and execute the exact same `.gturbo` directory format.
- **Verification**: In cross-engine benchmarks, both `mrefrust-cli` and Swift's `MferenceCLI` run against identical `.gturbo` model directories under strict `.fullSha256` integrity verification.

---

## 3. Binary & Structural Specification

### 3.1 `manifest.json`
The root `manifest.json` contains metadata for model architecture validation, quantization parameters, and file checksums.

```json
{
  "magic": "GTURBO",
  "versionMajor": 1,
  "versionMinor": 0,
  "flags": {},
  "modelID": "ggml-org/gemma-4-26B-A4B-it-GGUF",
  "arch": {
    "family": "gemma4",
    "hiddenSize": 2560,
    "ffnIntermediate": 10240,
    "moeIntermediateSize": 2048,
    "numHeads": 16,
    "numKVHeads": 8,
    "numFullKVHeads": 8,
    "headDim": 256,
    "fullHeadDim": 256,
    "vocabSize": 262144,
    "slidingWindow": 1024,
    "numLayers": 30,
    "numExperts": 128,
    "topKExperts": 8
  },
  "quant": {
    "groupSize": 64
  },
  "files": {
    "model_weights.bin": {
      "size": 1845491200,
      "sha256": "d4eb56075092403..."
    },
    "packed_experts/layout.json": {
      "size": 5242880,
      "sha256": "8a31e40c721..."
    },
    "packed_experts/layer_00.bin": {
      "size": 4294967296,
      "sha256": "1f2e3d4c..."
    }
  },
  "expertsPerLayer": 128,
  "numLayers": 30,
  "expertStride": 33554432
}
```

---

### 3.2 `model_weights.bin` (Resident Index & Tensor Payload)
`model_weights.bin` packages all resident (non-expert) tensors into a single binary file.

#### File Binary Layout
```
+-------------------------------------------------------------+
| Header (24 bytes)                                           |
| - indexSize (u64 LE): Total size of header + index table    |
| - residentSize (u64 LE): Size of resident tensor binary data|
| - entryCount (u64 LE): Number of entry descriptors          |
+-------------------------------------------------------------+
| Entry Descriptors (72 bytes per entry x entryCount)         |
| - nameOffset (u32), nameLen (u32)                           |
| - dtype (u8), padding (7 bytes)                             |
| - fileOffset (u64 LE): Absolute file offset (>= indexSize)  |
| - sizeBytes (u64 LE): Data byte size                        |
| - shape (4 x u32 LE)                                        |
| - scaleOffset (u64 LE), scaleSize (u64 LE)                  |
| - biasOffset (u64 LE), biasSize (u64 LE)                    |
+-------------------------------------------------------------+
| String Table & Page Alignment Padding                        |
+-------------------------------------------------------------+
| Raw Resident Tensor Data Payload (starting at indexSize)    |
+-------------------------------------------------------------+
```

During startup, `mrefrust-model-io` reads the leading index region (`indexSize` bytes), constructs the `ResidentIndex`, and mmaps the resident tensor data payload starting at offset `indexSize`.

---

### 3.3 `packed_experts/layout.json`
`layout.json` maps each logical expert `(layer, expert_id)` to its exact byte offset and sub-tensor offsets inside `packed_experts/layer_NN.bin`.

```json
{
  "expertStride": 33554432,
  "numLayers": 30,
  "expertsPerLayer": 128,
  "layers": [
    {
      "layer": 0,
      "file": "layer_00.bin",
      "experts": [
        {
          "expert": 0,
          "offset": 0,
          "size": 33554432,
          "tensors": {
            "gate": { "offset": 0, "size": 1048576, "dtype": "int4", "shape": [2048, 2560] },
            "gate_scales": { "offset": 1048576, "size": 32768, "dtype": "fp16", "shape": [2048, 40] },
            "up": { "offset": 1081344, "size": 1048576, "dtype": "int4", "shape": [2048, 2560] },
            "up_scales": { "offset": 2130000, "size": 32768, "dtype": "fp16", "shape": [2048, 40] },
            "down": { "offset": 2162768, "size": 1048576, "dtype": "int4", "shape": [2560, 2048] },
            "down_scales": { "offset": 3211344, "size": 32768, "dtype": "fp16", "shape": [2560, 32] }
          }
        }
      ]
    }
  ]
}
```

---

### 3.4 `packed_experts/layer_NN.bin`
Each layer file contains all experts for layer `NN` packed sequentially:

```
layer_00.bin:
+-------------------------------+-------------------------------+-- ...
| Expert 0 Blob                 | Expert 1 Blob                 |
| (size = expertStride bytes)   | (size = expertStride bytes)   |
+-------------------------------+-------------------------------+-- ...
```

- **Fixed Expert Stride (`expertStride`)**: Every expert blob within a layer occupies exactly `expertStride` bytes (e.g. 32 MiB). If the sum of an expert's sub-tensors is less than `expertStride`, it is zero-padded to the stride boundary.
- **O(1) Direct Seeking**: Because the stride is fixed, the byte offset for `expert_id` inside `layer_NN.bin` is simply `expert_id * expert_stride`. The streamer can immediately execute an OS `pread` at that exact file offset without scanning.

---

## 4. Ingestion & Streaming Pipeline

### 4.1 Repack Pipeline (`mrefrust-repack`)
The `repack` module converts upstream Safetensors or published GGUF checkpoints into `.gturbo` format:

1. **Header Parsing & Manifest Peek**: Reads model headers (HF `config.json` or GGUF metadata header) to derive `ArchConfig`.
2. **Streaming Range Fetching**: When converting from Hugging Face or remote GGUFs, `HttpRangeSource` fetches specific byte ranges over HTTP. The 20-27 GB raw model payload is **never written to disk as a whole monolith or held in RAM**.
3. **Resident Core Transcoding**:
   - Layer Norms -> Transcoded to BF16 / FP16.
   - Router Projections -> Transcoded to INT8 affine quantization.
   - Linear Attention State & Embeddings -> Formatted into `model_weights.bin`.
4. **Expert Layer Repacking**: Expert weights are sliced by layer, quantized (or kept in native GGUF block types Q8_0, Q4_K, Q6_K), formatted into fixed-stride blobs, and written incrementally into `packed_experts/layer_NN.bin`.

---

### 4.2 Runtime Streaming & Execution (`mrefrust-streaming`)

During model execution:
1. **Resident Core Load**: `mrefrust-model-io` maps `model_weights.bin` using zero-copy `MTLBuffer` (`newBufferWithBytesNoCopy`).
2. **Router Evaluation**: For each token and layer, GPU router GEMV kernels evaluate top-$K$ expert selections (e.g. top-8 experts out of 128).
3. **Expert Streaming & Cache**:
   - `PreadExpertStreamer` checks the in-memory LFU/LRU expert cache.
   - Cache Hits -> Reused directly.
   - Cache Misses -> Asynchronous OS `pread` reads `expertStride` bytes from `layer_NN.bin` directly into pinned Metal buffers.
4. **Execution Ceiling**: Expert cache size is controlled by `--expert-cache-slots` (default 16 slots). This keeps overall physical RAM footprint capped strictly at **~1.6 GiB (Qwen 3.6)** and **~2.1 GiB (Gemma 4)**.

---

## 5. Verification & Integrity Checking

Every `.gturbo` installation enforces strict integrity checking:
- **Checksum Manifest**: Every file inside `.gturbo` (including each `layer_NN.bin`) has its SHA-256 hash recorded in `manifest.json`.
- **Startup Integrity**: `mrefrust-model-io::load_manifest` verifies file sizes and checksums against `manifest.json` before execution.
- **Quality Verification**: Perplexity, greedy digests, and cross-engine KL divergence tests confirm that GGUF and Safetensors repacks produce byte-identical or numerically equivalent outputs compared to reference baselines.

---

## 6. Document References

- Upstream System Design: [`turbo-fieldfare SYSTEM_DESIGN.md`](https://github.com/drumih/turbo-fieldfare/blob/main/docs/SYSTEM_DESIGN.md)
- Benchmark Parity & Measurements: [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md)
- Repack Crate: [`crates/repack/README.md`](crates/repack/README.md)
- Model-IO Crate: [`crates/model-io/README.md`](crates/model-io/README.md)
