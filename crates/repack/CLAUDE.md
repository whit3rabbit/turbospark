# mrefrust-repack

Safetensors and GGUF header parsing, ranged HTTP/in-memory downloads (`RangeSource`), INT4/INT8 quantization repack, `.gturbo` directory installation assembly (`gturbo_writer.rs`), synthetic install generation (`synthetic_model.rs`, `synthetic_real.rs`, `synthetic_gguf.rs`), Hugging Face Llama repacker (`hf_checkpoint.rs`), Gemma 4 checkpoint repacker (`gemma4_checkpoint.rs`), and the GGUF intake (`gguf_*.rs`, ROADMAP Phase G: Stage 1 landed, Stage 2 in progress).

## Safety

- `#![forbid(unsafe_code)]` is enforced in this crate.

## Directory & File Structure

```
crates/repack/
+-- Cargo.toml                      # Crate manifest
+-- src/
|   +-- lib.rs                      # Library root
|   +-- safetensors_header.rs       # Pure safetensors JSON header parser
|   +-- ranged_download.rs          # RangeSource trait and ranged HTTP downloader
|   +-- repack.rs                   # Quantization repack algorithms (FP32/BF16 to INT4/INT8)
|   +-- gturbo_writer.rs            # Writes .gturbo directory tree and manifest/layout JSON
|   +-- resident_writer.rs          # Writes model_weights.bin resident tensor blob and index
|   +-- synthetic_model.rs          # Synthetic model generator (build_synthetic_gemma4_install)
|   +-- synthetic_real.rs           # Real-named synthetic generator (build_synthetic_gemma4_real_install)
|   +-- synthetic_qwen.rs           # Real-named synthetic Qwen 3.6 generator (build_synthetic_qwen36_real_install)
|   +-- gemma4_checkpoint.rs        # Gemma 4 mlx-community checkpoint converter & streamer
|   +-- gguf_header.rs              # Pure GGUF v3 header/metadata/tensor-table parser
|   +-- gguf_names.rs               # GGUF tensor names -> canonical HF-style names
|   +-- gguf_config.rs              # GGUF metadata -> ArchConfig (arch_from_gguf)
|   +-- gguf_checkpoint.rs          # GGUF repack walk (verbatim bytes, no quantize step)
|   +-- synthetic_gguf.rs           # In-memory GGUF writer for fixtures (GgufBuilder)
|   +-- qwen36_config.rs            # Qwen 3.6 config.json -> ArchConfig (parse_qwen36_config)
|   +-- hf_checkpoint.rs            # Hugging Face Llama checkpoint converter
|   +-- install_verifier.rs         # Validates repacked install directory structure & receipt
|   \-- manifest_peek.rs            # Pre-fetches remote checkpoint manifests without full download
\-- tests/
    +-- gemma4_checkpoint.rs        # Gemma 4 repack pipeline unit tests
    +-- gguf_header.rs              # GGUF parser round trip + rejection cases
    +-- gguf_names.rs               # GGUF name mapping, both families
    +-- gguf_config.rs              # GGUF metadata -> ArchConfig
    +-- gguf_checkpoint.rs          # GGUF walk: byte identity of every expert slice, plus the F32 transcode
    +-- gguf_checkpoint_network.rs  # Real GGUF header fetch + cross-checks (ignored)
    +-- gguf_fused_gate_network.rs  # Settles FUSED_GATE_FIRST by correlation (ignored)
    +-- gguf_f32_transcode_network.rs # Evidence for the transcode decision (ignored)
    +-- gguf_q4_k_network.rs        # Q4_K dequant vs the real Qwen Q4_K_M, by correlation (ignored)
    +-- gemma4_checkpoint_network.rs# Real Gemma 4 checkpoint download integration test (ignored)
    +-- qwen36_config.rs            # parse_qwen36_config vs the pinned Qwen 3.6 baseline
    +-- qwen36_checkpoint_network.rs# Real Qwen 3.6 checkpoint download integration test (ignored)
    +-- gturbo_writer.rs            # .gturbo layout writer unit tests
    +-- hf_checkpoint.rs            # HF Llama converter unit tests
    +-- hf_checkpoint_network.rs    # Real HF checkpoint download integration test (ignored)
    +-- install_verifier.rs         # Install verifier unit tests
    +-- repack.rs                   # Quantization repack unit tests
    +-- safetensors_header.rs       # Safetensors header parsing unit tests
    \-- synthetic_model.rs          # Synthetic install builder unit tests
```

## Key Modules

- `safetensors_header.rs`: Pure safetensors header parser.
- `ranged_download.rs`: Ranged HTTP download engine (`RangeSource`).
- `repack.rs`: Quantization repack pipelines converting FP32/BF16 weights to INT4/INT8.
- `gturbo_writer.rs`: `.gturbo` directory layout and binary file writer.
- `resident_writer.rs`: `model_weights.bin` resident tensor index writer.
- `synthetic_model.rs`: Synthetic dense and MoE model generators (`build_synthetic_gemma4_install`).
- `synthetic_real.rs`: Synthetic Gemma 4 real-named model generator (`build_synthetic_gemma4_real_install`); its tensor helpers are `pub(crate)` and shared with `synthetic_qwen.rs`.
- `synthetic_qwen.rs`: Synthetic Qwen 3.6 real-named model generator (`build_synthetic_qwen36_real_install`, `tiny_qwen36_arch`).
- `gemma4_checkpoint.rs`: Gemma 4 checkpoint repacker and streamed expert pipeline builder. Family-parameterized: `classify_for_family` and `manifest_quant` take a `ModelFamily`, so `write_qwen36_install` / `write_qwen36_install_streamed` are the same walk with a different routed-expert marker (`.mlp.switch_mlp.`) and different quant probe names, plus a guard that `arch.family` really says Qwen.
- `qwen36_config.rs`: `parse_qwen36_config`, the ONE family-specific piece of the Qwen repack path. Shares only the `text_config` wrapper with `parse_gemma4_config`; every other key name differs (see its module header for the mapping table). `attention_scale` has no config key and is `head_dim ** -0.5` from the reference implementation.
- `hf_checkpoint.rs`: Hugging Face checkpoint layout converter.
- `install_verifier.rs`: Verifies checksums and directory layout of repacked installs.
- `manifest_peek.rs`: Inspects remote checkpoint manifests without full downloads.
- `gguf_header.rs`: Pure GGUF v3 parser. Unlike safetensors there is no length prefix, so `TooShort` carries the offset the walk wanted and `ranged_download::fetch_gguf_header` grows geometrically toward it (following `needed` literally would be one HTTP round trip per metadata field). `ggml_type_block` is DELIBERATELY PARTIAL: only types whose block size was read off the ggml spec are listed, and everything else is named in the error rather than guessed at.
- `gguf_names.rs`: GGUF-to-canonical name tables, one per family, every row read off a real published file and cross-checked against the corresponding real install's resident index.
- `gguf_config.rs`: `arch_from_gguf`. Starts from `known_architecture(family)` and overrides ONLY the fields GGUF actually determines, because behavioral fields are hardcoded in llama.cpp's graph builder and absent from the metadata.
- `gguf_checkpoint.rs`: the GGUF repack walk. Expert bytes are sliced per expert and copied through with no quantization step. Splits Gemma's fused `ffn_gate_up_exps` at `FUSED_GATE_FIRST`, which is now MEASURED against the real file rather than assumed (gate is the first half; `tests/gguf_fused_gate_network.rs`). The one thing it does NOT carry verbatim is the resident F32 core: `transcode_f32` narrows norms to BF16 and quantizes the router to INT8 affine, because no F32 kernel exists here (Gotcha 6).
- `synthetic_gguf.rs`: `GgufBuilder` plus `build_synthetic_gemma4_gguf`, a tiny file carrying every name and metadata key the real Gemma 4 GGUF has. Its Q8_0 quants deliberately span only `[-8, 7]` at a scale of 0.0625: at full byte range the weights reach +/-32 and a fixture install overflows FP16 before it reaches the head, which reads as a kernel bug and is not one. The default shape's `moe_intermediate` is 16, which is NOT a whole Q8_0 block, so a test that wants to RUN a fixture install has to widen it (`crates/runtime/tests/gguf_install_refused.rs` does).

## Development & Test Commands

```sh
# Run fast unit tests for mrefrust-repack
cargo test -p mrefrust-repack

# Run network checkpoint integration tests (ignored by default, downloads large files)
cargo test -p mrefrust-repack --test gemma4_checkpoint_network --release -- --ignored --nocapture
cargo test -p mrefrust-repack --test hf_checkpoint_network --release -- --ignored --nocapture

# Qwen 3.6: ~20.4 GB across four shards. The ONLY test that covers the
# multi-shard walk on this family (the synthetic fixture is one shard).
MREFRUST_QWEN36_INSTALL_DIR=~/models/qwen36.gturbo \
  cargo test -p mrefrust-repack --test qwen36_checkpoint_network --release -- --ignored --nocapture

# GGUF intake: reads only the HEADER of the real published GGUFs (a few MB
# off a 20-27 GB file, ~4 s each). With the install vars set it also asserts
# every tensor name maps and that the ArchConfig derived from GGUF metadata
# equals the one the .gturbo install declares.
MREFRUST_GEMMA4_INSTALL_DIR=~/models/gemma4.gturbo \
MREFRUST_QWEN36_INSTALL_DIR=~/models/qwen36.gturbo \
  cargo test -p mrefrust-repack --test gguf_checkpoint_network --release -- --ignored --nocapture

# Which half of Gemma's fused ffn_gate_up_exps is the gate, by correlating a
# dequantized layer 0 expert 0 against the MLX install. Few KB, ~5 s.
MREFRUST_GEMMA4_INSTALL_DIR=~/models/gemma4.gturbo \
  cargo test -p mrefrust-repack --test gguf_fused_gate_network --release -- --ignored --nocapture

# The evidence behind Gotcha 6's transcode decision. Needs no install.
cargo test -p mrefrust-repack --test gguf_f32_transcode_network --release -- --ignored --nocapture

# The Q4_K reference against real published bytes: a dequantized layer 0
# expert 0 gate row of the real Qwen 3.6 Q4_K_M against the same row in the
# install. Few KB, ~5 s. AGENTS.md Gotcha 30 before reading a low number.
MREFRUST_QWEN36_INSTALL_DIR=~/models/qwen36.gturbo \
  cargo test -p mrefrust-repack --test gguf_q4_k_network --release -- --ignored --nocapture
```

## Crate Gotchas

1. **Synthetic Model Weight Meaning**: Synthetic models built by `build_synthetic_gemma4_install` use deterministic pseudo-random numbers rather than trained weights. Generated text on synthetic installs is structurally valid but semantically gibberish (and short generations may yield empty strings).
2. **`build_manifest_json` writes every family-extension field unconditionally.** `arch_validation` resolves omitted ones against the GEMMA baseline whatever family the manifest claims, so a Qwen install that leaves them out can never load. Gemma installs are unaffected (those are its own fallbacks). Do not make any of them conditional. Float fields additionally have to be binary fractions to survive serde_json's ~1-ULP default parser -- see AGENTS.md Gotcha 24.
3. **Synthetic Model Uniform Routing**: Synthetic MoE routers feature near-uniform routing logits. Expert slot permutation bugs cannot be caught by testing synthetic models alone; routing assignments must be validated structurally.
4. **The walk writes an install for every block type it can parse; whether that install RUNS is decided elsewhere, per block type.** As of Phase G Stage 2 a Q8_0 install opens and decodes (resident GEMV, embedding lookup, and the routed-expert decode pair all exist and are parity-tested); Q4_K, Q6_K and Q4_0 install and are refused, because routed experts are where a GGUF's bytes are and those kernels do not exist. The two gates are `model_io::validate_quant` reading the manifest's `ggmlType` against `model_io::EXECUTABLE_GGUF_TYPES`, and `RealForwardRunner::open` reading the resident index's dtype tags against its own copy of that set. Do not widen either without landing kernels: both directions are asserted in `crates/runtime/tests/gguf_install_refused.rs`, which also decodes a Q8_0 install. AGENTS.md Gotchas 29 and 30 list the traps in the format and in checking it against real files.
5. **All four GGUF `*_network` tests cost KB or MB, not GB, and none downloads a checkpoint.** They read ranges off a 20-27 GB remote file and finish in seconds. `gguf_checkpoint_network.rs` reads the header and is the only place a name-mapping hole or a converter disagreement can surface, because a synthetic fixture only ever contains names its author already knew: run it after touching `gguf_names.rs` or `gguf_config.rs`. `gguf_fused_gate_network.rs` reads two output rows (a Q8_0 row of `hidden` elements is `hidden / 32 * 34` CONTIGUOUS bytes) and `gguf_f32_transcode_network.rs` reads whole norm and router tensors, which are vectors and a small matrix, and `gguf_q4_k_network.rs` reads 16 consecutive Q4_K rows (`hidden / 256 * 144` contiguous bytes each) to hold the Q4_K reference against real bytes. Before budgeting a download for the next GGUF question, check whether the answer is a contiguous byte range; the block layout makes more of them so than the planar affine layout would.

6. **GGUF's F32 norms and F32 router are TRANSCODED at repack time, and nothing F32 reaches the install.** `gguf_checkpoint.rs::transcode_f32` narrows F32 to BF16 by default and INT8-affine-quantizes the router. The decision rests on measurement, not preference (`tests/gguf_f32_transcode_network.rs`): llama.cpp UPCAST norms that are BF16 in the original checkpoint, so narrowing them back is bit-exact and costs nothing, and the router transcode applies the same INT8 affine the MLX path already applies to the same tensor without moving the routing decision. Do not re-open this as a tradeoff; it was measured to a conclusion. See AGENTS.md Gotcha 29. Three things to know about the implementation. The INT8 set is keyed by CANONICAL NAME per family (`int8_transcode_targets`), not by rank: Qwen's `linear_attn.conv1d.weight` is a rank-2 F32 tensor the runtime reads as BF16 while `mlp.gate.weight` is rank-2 and must be dtype 5, so rank does not decide. BF16 is the safe default because a mis-targeted tensor fails LOUDLY at `open()` (`norm_view`'s byte-size check, or `encode_gemv_any`'s "no dispatched GEMV kernel"). And the manifest's `router` slot consequently says `affine`/8/group-64, which is not a relaxation of the Stage 1 refusal but a true statement about the bytes: the other four slots still say `"gguf"` and are what `load_manifest` refuses on. A converter that did NOT upcast is not rejected (there is nowhere else to put the values), but every lossy value is counted into `GgufRepackOutput::lossy_narrowing` and reported through the streamed writer's `progress` callback.
