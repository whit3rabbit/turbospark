# turbospark-model-io

Model installation layout, `manifest.json` parsing and architecture validation (`ArchConfig`), packed expert layout metadata (`PackedExpertsLayout`), resident tensor index reader (`ResidentIndex`), memory-mapped resident weight buffer (`ResidentBuffer`), SHA-256 verification (`sha256.rs`), and install receipt validation (`InstallReceipt`).

## Safety

- Contains `unsafe` code specifically restricted to memory-mapping (`mmap`) inside `resident_buffer.rs`.

## Directory & File Structure

```
crates/model-io/
+-- Cargo.toml                  # Crate manifest
+-- src/
|   +-- lib.rs                  # Library root re-exporting model-io API
|   +-- manifest.rs             # Decodes and validates manifest.json
|   +-- arch_config.rs          # ArchConfig struct and field resolution
|   +-- arch_baselines.rs       # Canonical baselines (Gemma 4, Qwen 3.6, DeepSeek-V4)
|   +-- arch_validation.rs      # Structural validation rules for architecture configs
|   +-- packed_experts_layout.rs# Decodes packed_experts/layout.json for streamed MoE
|   +-- resident_index.rs       # Reads tensor index entries from model_weights.bin
|   +-- resident_buffer.rs      # Zero-copy mmap wrapper (ResidentBuffer)
|   +-- sha256.rs               # Streaming SHA-256 checksum verifier
|   +-- install_receipt.rs      # Parses and validates .gturbo install receipts
|   \-- error.rs                # ModelIoError enum definition
\-- tests/
    +-- arch_config.rs          # Architecture config resolution unit tests
    +-- install_receipt.rs      # Install receipt parsing unit tests
    +-- manifest.rs             # Manifest JSON decoding & baseline validation tests
    +-- packed_experts_layout.rs# Layout decoding unit tests
    +-- resident_index.rs       # Binary resident index parsing tests
    \-- sha256.rs               # SHA-256 verification unit tests
```

## Key Modules

- `manifest.rs`: Decodes and validates `manifest.json`. Its `validate_quant` accepts THREE shapes: the INT4/INT8 affine one at group 64 with BF16 companions, the 1-bit affine one at group 128 with FP16 companions (ROADMAP's 1-bit entry), and (ROADMAP Phase G) a `scheme: "gguf"` slot whose declared block types are all in `EXECUTABLE_GGUF_TYPES` (`q8_0`, `q4_k`, `q6_k`, Phase S's `iq3_xxs`, `iq4_nl`, `iq4_xs`, Phase M2's `q5_k`, and M5's `mxfp4`) -- the block types this port has kernels for, which is deliberately narrower than the set the repack walk can WRITE. Widening it means landing kernels; `crates/runtime` applies the same rule again to the resident index's dtype tags. A slot declares its types with `ggmlType` (the dominant one) plus an optional `ggmlTypes` ARRAY when it carries more than one, which a mixed sub-4-bit install does; every member is checked, because checking only the dominant one would pass an install on the strength of its majority and fail at a dispatch thirty layers in. Several types are in the list on weaker grounds than a full kernel set: Q5_K has a resident GEMV and nothing else (Mixtral puts it on `attn_output` alone), Q6_K has a resident GEMV, an embedding lookup and -- since Phase M2 -- a routed PHASE 2 but no phase 1, IQ3_XXS and IQ4_XS have a routed phase 1 and no phase 2, IQ4_NL the reverse. Each covers what its real file asks for, and an install that used one elsewhere passes here and fails at the dispatch, by name. **MXFP4 is narrow in the OPPOSITE direction and is the first type this list and `crates/runtime`'s disagree about**: both routed phases and no resident GEMV at all, because `gpt-oss` puts it only in `ffn_*_exps`. The two lists are TWINS AND NOT COPIES -- this one reads the manifest's per-SLOT `ggmlType` and asks whether the type has the kernels its slot needs, while `EXECUTABLE_GGUF_DTYPES` reads a RESIDENT tensor's dtype tag and asks whether that tensor can be dispatched. So `"mxfp4"` is here and tag 14 is deliberately not there, and an install with MXFP4 attention passes this gate and is stopped by that backstop. AGENTS.md Gotcha 29 states the rule; `mxfp4_is_refused_as_a_resident_tensor_though_its_experts_run` asserts both halves on one install. **The 1-bit shape is checked as ONE CONJUNCTION and that is the point of its being a separate predicate rather than three widenings of the affine one.** Adding `1` to the bit lists, `128` to the group sizes and `fp16` to the companion types accepts six combinations no kernel implements; the two real shapes are `(4|8, bf16, 64)` and `(1, fp16, 128)`, because a checkpoint's bit width, companion dtype and group size travel together. The FP16-versus-BF16 axis is the dangerous one: the planes are the same width, so a wrong reading passes every length check and decodes the 1-bit checkpoint's 0.0271 scales as 1.7e-16, which is why the affine error message now names the observed triple instead of saying "unsupported". 1-bit is accepted on ALL FIVE slots including `routedExpert`, which has no 1-bit kernel: the published checkpoint is dense, so three slots describe components it does not have and fall back to the type the rest of the model uses, and refusing them is exactly how M4's dense `llama` failed to open (`crates/repack` Gotcha 8). `tests/manifest.rs` moves ONE axis per case; one case moves two, and its comment says why (it is the only shape the bit-width conjunct alone refuses, and without it that conjunct can be deleted with the file still green).
- `arch_config.rs`: Architecture configuration structs and field resolution. `RopeScalingConfig` (ROADMAP M5) carries YaRN's four scalars as one grouped field with a `NONE`, following `LinearAttentionConfig`; `factor: 0.0` is the inactive sentinel because 1.0 would read as "declared, and the identity".
- `arch_baselines.rs`: Baseline specifications, one per `ModelFamily`: Gemma 4, Qwen 3.6, DeepSeek-V4-Flash, `llama` (Mixtral's, covering the dense half too), `qwen3moe`, and `gptOss`. A family gets a baseline when it gets a decode flow OR a name table, never as a placeholder -- `arch_validation` compares one field by field, so an invented baseline validates installs against fiction.
- `arch_validation.rs`: Structural validation of architecture configs.
- `packed_experts_layout.rs`: Decodes `packed_experts/layout.json` for streamed MoE layouts. `expert_stride` exists at TWO levels and they mean different things: `PackedExpertsLayout::expert_stride` is the model-wide maximum (what `manifest.json` declares and what a slot is sized from), while `LayerLayout::expert_stride` is what that layer's file is actually padded to. Address or size a layer with the second, never the first -- see Gotcha 2.
- `resident_index.rs`: Reads and parses tensor metadata entries from `model_weights.bin`.
- `resident_buffer.rs`: Zero-copy `mmap` wrapper (`ResidentBuffer`) for mapped model weights.
- `sha256.rs`: Streaming SHA-256 checksum calculator for installation integrity.
- `install_receipt.rs`: Parses and verifies `.gturbo` install receipts.

## Development & Test Commands

```sh
# Run tests for turbospark-model-io
cargo test -p turbospark-model-io
```

## Crate Gotchas

1. **Resident Memory Pinning**: While clean file-backed `mmap` pages are normally unpinned in host OS memory, wrapping `ResidentBuffer` into Metal buffers via `newBufferWithBytesNoCopy` pins the mapped virtual memory range. Resident weights count directly against process physical memory footprint (`phys_footprint`).
2. **The expert stride is PER LAYER, and it was model-wide until ROADMAP Phase S.** Every install written before then is uniform across layers, which makes a uniform-stride assumption invisible: the per-layer field simply falls back to the top-level one and nothing changes. It stops being invisible on a mixed sub-4-bit install. The Phase S candidate puts IQ3_XXS + IQ4_NL experts on 29 layers and IQ4_XS + Q8_0 on the thirtieth, whose blob is 1.6x the others, so padding every layer to the maximum writes 16.2 GB where 10.3 is needed and over-reads 29 of 30 layers by that factor on every cache miss. That inverts the phase's whole -24.2% into a +35% regression, which is why this is a prerequisite rather than a tuning step. The loader refuses a layer stride ABOVE the top-level value, because a slot is allocated from the latter.
