# turbospark-repack

Safetensors and GGUF header parsing, ranged HTTP/in-memory downloads (`RangeSource`), INT4/INT8 quantization repack, `.gturbo` directory installation assembly (`gturbo_writer/`), synthetic install generation (`synthetic_model/`, `synthetic_real.rs`, `synthetic_gguf/`), Hugging Face Llama repacker (`hf_checkpoint.rs`), Gemma 4 checkpoint repacker (`gemma4_checkpoint/`), and the GGUF intake (`gguf_*`, `gguf_checkpoint/`, ROADMAP Phase G, landed in full 2026-08-08; six real families install from published bytes).

## Safety

- `#![forbid(unsafe_code)]` is enforced in this crate.

## Directory & File Structure

```
crates/repack/
+-- Cargo.toml                      # Crate manifest
+-- src/
|   +-- lib.rs                      # Library root
|   +-- safetensors_header.rs       # Pure safetensors JSON header parser
|   +-- ranged_download/            # RangeSource trait and ranged HTTP downloader
|   |   +-- mod.rs                  # RangeSource trait and range fetchers
|   |   +-- chunks.rs               # Chunk arithmetic and concurrent fill_chunks
|   |   \-- http.rs                 # HttpRangeSource implementation
|   +-- repack.rs                   # Quantization repack algorithms (FP32/BF16 to INT4/INT8)
|   +-- gturbo_writer/              # Writes .gturbo directory tree and manifest/layout JSON
|   |   +-- mod.rs                  # Entry points and re-exports
|   |   +-- layers.rs               # packed_experts/layer_NN.bin writing
|   |   +-- manifest.rs             # build_manifest_json (see Gotcha on unconditional fields)
|   |   +-- streaming.rs            # The streamed, layer-at-a-time walk
|   |   \-- types.rs                # Writer inputs and layout records
|   +-- resident_writer.rs          # Writes model_weights.bin resident tensor blob and index
|   +-- synthetic_model/            # Synthetic model generator (build_synthetic_gemma4_install)
|   |   +-- mod.rs                  # Entry points and re-exports
|   |   +-- arch.rs                 # The ArchConfig the fixtures declare
|   |   +-- dense.rs                # Dense install builder
|   |   \-- moe.rs                  # MoE and streamed-expert variants
|   +-- synthetic_real.rs           # Real-named synthetic generator (build_synthetic_gemma4_real_install)
|   +-- synthetic_llama.rs          # Real-named synthetic Mixtral / Qwen3-MoE / DENSE llama generators
|   +-- synthetic_qwen/             # Real-named synthetic Qwen generators (MoE and dense)
|   |   +-- mod.rs                  # Re-exports synthetic Qwen builders
|   |   +-- dense.rs                # Dense sub-4-bit Qwen generator (build_synthetic_qwen_gdn_dense_install)
|   |   \-- moe.rs                  # MoE Qwen 3.6 generator (build_synthetic_qwen_gdn_moe_install)
|   +-- gemma4_checkpoint/          # Gemma 4 / Qwen 3.6 mlx-community safetensors converter & streamer
|   |   +-- mod.rs                  # Module root and install writer entrypoints
|   |   +-- classify.rs             # Tensor classification (resident vs routed)
|   |   +-- config.rs               # config.json & quantization spec parsing
|   |   +-- expert_blobs.rs         # Packed expert blob packing & layout calculation
|   |   +-- manifest_quant.rs       # Manifest quantization spec generation
|   |   +-- mtp.rs                  # The MTP head's ingest: the walk's one QUANTIZING arm
|   |   +-- narrow.rs               # F16/F32 to BF16 lossy narrowing
|   |   +-- orchestrate.rs          # Repack orchestration & layer planning
|   |   \-- shards.rs               # Multi-shard reader & classification
|   +-- arch_registry.rs            # Architecture strings: what runs, what is only recognized
|   +-- gguf_header/                # Pure GGUF v3 header/metadata/tensor-table parser
|   |   +-- mod.rs                  # Entry points and re-exports
|   |   +-- parser.rs               # The walk itself
|   |   +-- types.rs                # Metadata value and tensor-entry types
|   |   \-- ggml.rs                 # ggml_type_block: block size and byte size per type
|   +-- gguf_names/                 # GGUF tensor names -> canonical HF-style names
|   |   +-- mod.rs                  # Name mapping router and types
|   |   +-- gemma4.rs               # Gemma 4 GGUF tensor name mappings
|   |   +-- gpt_oss.rs              # gpt-oss GGUF tensor name mappings
|   |   +-- llama.rs                # Llama/Mixtral GGUF tensor name mappings
|   |   \-- qwen.rs                 # Qwen GGUF tensor name mappings
|   +-- gguf_config/                # GGUF metadata -> ArchConfig (arch_from_gguf)
|   |   +-- mod.rs                  # arch_from_gguf entry point
|   |   +-- attention.rs            # Attention & head dimension resolvers
|   |   +-- masks.rs                # Attention mask & layer type decoders
|   |   \-- meta.rs                 # GGUF metadata helper accessors
|   +-- gguf_checkpoint/            # GGUF repack walk (verbatim bytes, F32 transcode)
|   |   +-- mod.rs                  # Module root and streamed install writer
|   |   +-- types.rs                # Error types & ggml dtype mappings
|   |   +-- plan.rs                 # Tensor classification & layer planning
|   |   +-- transcode.rs            # Resident F32 transcode & V-head conventions
|   |   \-- manifest.rs             # GGUF manifest quantization spec generator
|   +-- synthetic_gguf/             # In-memory GGUF writer for fixtures
|   |   +-- mod.rs                  # Module root and re-exports
|   |   +-- builder/                # GgufBuilder and GGUF value serialization
|   |   +-- gemma4.rs               # SyntheticGgufShape, QuantMix & build_synthetic_gemma4_gguf
|   |   \-- gptoss.rs               # SyntheticGptOssShape: the M5 layer's SHAPE, not just its types
|   +-- qwen36_config.rs            # Qwen 3.6 config.json -> ArchConfig (parse_qwen_gdn_moe_config)
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
    +-- gguf_iq_network.rs          # IQ3_XXS/IQ4_NL/IQ4_XS vs the Phase S candidate, by correlation (ignored)
    +-- gguf_install_network.rs     # Streams the real Q8_0 GGUF into a full install (ignored)
    +-- gguf_qwen_install_network.rs# Same for the real Qwen Q4_K_M, the mixed-block-type case (ignored)
    +-- gguf_mixtral_install_network.rs # Same for the real Mixtral Q4_K_M, plus the two DENSE llama installs (ignored)
    +-- gguf_qwen3moe_install_network.rs # Same for the real Qwen3-30B-A3B Q4_K_M, the FINE-GRAINED MoE (ignored)
    +-- gguf_gptoss_install_network.rs # Same for the real gpt-oss-20b MXFP4 (ignored)
    +-- gguf_llama_rope_patch.rs    # The rotary pair convention: in-place diagnostic + the walk's inverse (ignored)
    +-- gguf_qwen_core_probe.rs     # A GGUF install's resident core vs the MLX one, tensor by tensor (ignored)
    +-- gguf_qwen_quant_probe.rs    # The same question for the QUANTIZED V-head tensors, by correlation (ignored)
    +-- gguf_qwen_convention_patch.rs # The V-head convention on every layer, and the in-place patch loop (ignored)
    +-- gguf_norm_convention_probe.rs # GGUF install's resident BF16 core vs the MLX install's (ignored)
    +-- gemma4_checkpoint_network.rs# Real Gemma 4 checkpoint download integration test (ignored)
    +-- qwen36_config.rs            # parse_qwen_gdn_moe_config vs the pinned Qwen 3.6 baseline
    +-- qwen35_config.rs            # parse_qwen_gdn_dense_config vs the pinned qwen3_5 baseline, BOTH checkpoints
    +-- synthetic_qwen35.rs         # The dense 1-bit install, end to end through the walk
    +-- qwen35_checkpoint_network.rs# The REAL Bonsai-27B 1-bit checkpoint, streamed (ignored)
    +-- qwen38_checkpoint_network.rs# The REAL Qwen3.8-27B INT4 checkpoint, streamed (ignored)
    +-- mtp_head_network.rs         # The MTP head's inventory off the official BF16 header (ignored)
    +-- mtp_quantize_network.rs     # Its INT4 round trip against real bytes, by correlation (ignored)
    +-- ternary_checkpoint_network.rs# The REAL Ternary-Bonsai-27B 2-bit checkpoint, streamed (ignored)
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
- `ranged_download/`: Ranged HTTP download engine (`RangeSource`). A call is split into `MAX_RANGE_BYTES` chunks, `RANGE_CONCURRENCY` of them in flight, each under the `RANGE_ATTEMPTS` retry ladder. **The concurrency is not the optimization; `http1_only()` on the client is** (AGENTS.md Gotcha 46). Every one of these URLs redirects to the Xet LFS bridge, which speaks HTTP/2, and reqwest will multiplex all the concurrent chunk GETs onto ONE connection -- one CloudFront edge, one per-edge rate cap, and therefore the old wall clock with no error and nothing in a log to say so. Two more things about the shape. `fill_chunks` hands each worker a DISJOINT `&mut [u8]` carved out of the caller's buffer rather than concatenating returned `Vec`s: that is what keeps this `#![forbid(unsafe_code)]` (unlike `crates/streaming/src/read_pool.rs`, which is the same idea in a crate that permits unsafe), makes ordering structural, and keeps peak memory where it was, since eight 64 MiB buffers in flight would add half a gigabyte to an allocation the caller already made. And a SINGLE-chunk range is the common case, not a degenerate one -- `fetch_gguf_header` reads a megabyte per growth step and every norm is well under the cap -- so it runs on the calling thread and is asserted to. **The concurrency therefore helps MoE walks and not dense ones**: `read_tensor` issues one `read_range` per TENSOR, so a routed tensor (the whole expert table for a layer, ~410 MiB on Gemma) chunks and parallelizes while a dense model's largest tensor sits under the cap and does not. Measured 3.4x on a 512 MiB span; TinyLlama's re-stream was unchanged at 3:28, correctly.
- `repack.rs`: Quantization repack pipelines converting FP32/BF16 weights to INT4/INT8.
- `gturbo_writer/`: `.gturbo` directory layout and binary file writer (`layers.rs` for the packed experts, `manifest.rs` for `manifest.json`, `streaming.rs` for the layer-at-a-time walk).
- `resident_writer.rs`: `model_weights.bin` resident tensor index writer.
- `synthetic_model/`: Synthetic dense (`dense.rs`) and MoE (`moe.rs`) model generators (`build_synthetic_gemma4_install`).
- `synthetic_real.rs`: Synthetic Gemma 4 real-named model generator (`build_synthetic_gemma4_real_install`); its tensor helpers are `pub(crate)` and shared with `synthetic_qwen/`.
- `synthetic_qwen/`: Synthetic Qwen generators for MoE (`moe.rs`: `build_synthetic_qwen_gdn_moe_install`) and dense sub-4-bit (`dense.rs`: `build_synthetic_qwen_gdn_dense_install`, `build_synthetic_qwen_gdn_dense_install_at_bits`). For the dense sibling, **the WIDTH is a parameter, not a fork**: the two real checkpoints are one architecture at two quantizations, so one fixture covers both and neither gets a second copy of the dense-path assertions. What varies with it is the quantizer and the packed word count; what does not is anything about being dense. Two things differ from the Qwen 3.6 fixture and both come off the real checkpoints: it is dense (one `mlp.{gate,up,down}_proj` per layer, no router or shared or routed experts, so ZERO packed-expert files), and it is 1- or 2-bit at group 128 with FP16 companions -- all three axes together, because that is how a checkpoint carries them. **It earned its place on its first run**, catching both of the dense-path bugs Gotcha 8 records the GGUF walk having had, which this walk had kept because its dense path wrote no quant block at all. Note the shape constraint that decided its constants: only COLUMN counts must be a whole number of groups (`pass_through_packed` checks `cols % group` and leaves rows free), so it is the Qwen 3.6 fixture with `HIDDEN` doubled from 64 to 128 and nothing else moved.
- `gemma4_checkpoint/`: Gemma 4 / Qwen 3.6 safetensors checkpoint repacker and streamed expert pipeline builder (`config.rs`, `shards.rs`, `orchestrate.rs`, `classify.rs`, `expert_blobs.rs`, `manifest_quant.rs`, `narrow.rs`). **THREE THINGS IN IT STOPPED BEING CONSTANTS FOR ROADMAP's 1-BIT ENTRY, and each was a true statement about every install that existed.** `is_supported_affine_shape` in `config.rs` now validates the effective `(bits, group_size)` pair for the default AND every per-tensor override, as ONE conjunction: `(4|8, 64)`, `(1, 128)` and -- since the ternary entry -- `(2, 128)` are the shapes with kernels, and the cross-products are combinations no real file has. `AFFINE_1BIT_GROUP_SIZE` and `AFFINE_2BIT_GROUP_SIZE` are two constants of equal value on purpose: they agree because one publisher chose 128 twice, not because sub-4-bit implies 128. `pass_through_packed` reads the group size off the checkpoint rather than assuming 64, and requires the companion dtype PER WIDTH -- F16 at one and two bits, BF16 at four and eight. That reads as a threshold and is a TABLE of what each published file carries; a future 2-bit checkpoint in BF16 would be a third row rather than a moved boundary. That last one is the axis that fails silently: the two planes are the same width and share no exponent field, so accepting either produces an install of exactly the right SIZE whose scales are wrong by orders of magnitude (0.0271 read as 1.7e-16). And `manifest_quant` writes the companion dtype and group size it actually wrote, where it used to emit `bf16` and `64` literals; `model_io::validate_quant` reads those three fields together, so a literal is an install that cannot open. Keying the manifest's dtype on the DEFAULT bits is sound only because the shape check has already refused every mixture that would straddle the two groups. Family-parameterized: `classify_for_family` and `manifest_quant` take a `ModelFamily`, so `write_qwen_gdn_moe_install` / `write_qwen_gdn_moe_install_streamed` are the same walk with a different routed-expert marker (`.mlp.switch_mlp.`) and different quant probe names, plus a guard that `arch.family` really says Qwen.
- `gemma4_checkpoint/mtp.rs`: the multi-token-prediction head's ingest (`docs/MTP_SPECULATIVE.md`, step 1), and **the walk's only QUANTIZING arm**. Every other resident tensor is either passed through already-packed (`pass_through_packed`) or narrowed to BF16 (`narrow_raw_to_bf16`); the head takes neither, because it comes from a DIFFERENT REPOSITORY than the trunk around it. `mlx-community/Qwen3.8-27B-4bit` drops `mtp.*` in conversion, so the head can only be read from the official `Qwen/Qwen3.8-27B`, where it is BF16 at full width. Narrowing it would leave 849 MB of BF16 matrices in an install whose engine dispatches no unquantized GEMV, failing at the first draft dispatch four layers from the cause. **Quantizing a drafter is not a numerics compromise** -- its output is verified by the target and either accepted or discarded, so its quality is a throughput axis and can never change what the engine emits. Three implementation notes. The MATRIX/NORM split is by RANK (2 quantizes, 1 narrows, anything else is refused by name), not by a name list that a differently-shaped head would invalidate. **No new `RangeSource` parameter was needed**: `Gemma4Shards::new` already takes a `(header, source)` pair PER SHARD, so a shard from another repository is expressible today -- the cross-repo case fell out of the multi-shard registry that multi-shard checkpoints had already paid for. And the head is kept out of `resident_bases` rather than merged into it, because `lm_order_key` sorts on `layer_index`, which finds `.layers.` inside `mtp.layers.0.*` and would interleave the head's block with TRUNK LAYER 0's tensors. Whether an install has a drafter is answered by whether `mtp.fc.weight` is in the resident index -- there is no manifest field and no flag, so nothing can disagree with the bytes. **THE INGEST HAS TO EXIST IN BOTH WRITERS AND FOR ONE RELEASE IT DID NOT.** It landed in `orchestrate_gemma4_checkpoint_sharded` alone, which is the NON-streamed path that every fixture takes; every REAL install goes through `write_gemma4_install_streamed`, which classified `mtp.*` correctly into `plan.mtp_bases` and then never read it. So the first real stream that asked for a head wrote a byte-identical HEADLESS install -- the same 851 resident tensors and the same 15,132,916,736-byte region -- with no error and nothing in the progress log to say so, and it cost a 15-minute stream to find. Gotcha 8 says to build the fixture before the download; the clause this adds is that **the fixture has to exercise the WRITER the download will use**, which is why `build_synthetic_qwen_gdn_dense_install_with_mtp_streamed` exists beside its sibling and why `both_writers_carry_the_mtp_head` asserts the two agree tensor for tensor. That test is the only one of fourteen that reddens when the streamed arm is removed.
- `qwen36_config.rs`: `parse_qwen_gdn_moe_config` AND `parse_qwen_gdn_dense_config`, the family-specific piece of the Qwen repack path. **One parameterized body serves both, which is the opposite of how this module relates to the Gemma parser and for the opposite reason**: against Gemma almost every key name differs, so that one is a separate function; against each other almost none does, because `qwen3_5` and `qwen3_5_moe` share a behavioural profile entirely. Exactly four fields fork, all about the FFN -- `intermediate_size` (the DENSE width) against `shared_expert_intermediate_size`, `moe_intermediate_size` / `num_experts` / `num_experts_per_tok` against zeros, and `shared_expert_gated`. A dense config that ALSO declares `num_experts > 0` is REFUSED rather than half-read: half-reading yields an `ArchConfig` whose FFN width and expert count disagree, which validates structurally and then dispatches the wrong branch. `refuse_foreign_config` keeps the two from parsing each other's files, which matters more here than anywhere else in the registry -- the two `model_type` strings are one suffix apart. Shares only the `text_config` wrapper with `parse_gemma4_config`; every other key name differs (see its module header for the mapping table). `attention_scale` has no config key and is `head_dim ** -0.5` from the reference implementation.
- `hf_checkpoint.rs`: Hugging Face checkpoint layout converter.
- `install_verifier.rs`: Verifies checksums and directory layout of repacked installs.
- `manifest_peek.rs`: Inspects remote checkpoint manifests without full downloads.
- `gguf_header/`: Pure GGUF v3 parser (`parser.rs`, with the per-type block table in `ggml.rs`). Unlike safetensors there is no length prefix, so `TooShort` carries the offset the walk wanted and `ranged_download::fetch_gguf_header` grows geometrically toward it (following `needed` literally would be one HTTP round trip per metadata field). `ggml_type_block` is DELIBERATELY PARTIAL: only types whose block size was read off the ggml spec are listed, and everything else is named in the error rather than guessed at. It is WIDER than the executable set on purpose (Q2_K, Q3_K, Q5_0 and Q5_1 are sizeable and have no kernel; the rest of the parse-only rows have since grown one, MXFP4 most recently in ROADMAP M5 -- which is the point of the split, since M5's Phase 0 survey could not have sized gpt-oss's experts at all without a parse-only row for a type nothing could yet decode), so a candidate checkpoint can be header-probed without a download; adding a row buys parsing only, and reddens `names_an_unsupported_but_real_ggml_type`, whose exemplar has to be re-picked.
- `arch_registry.rs`: the architecture STRING tables (ROADMAP Phase M1), GGUF `general.architecture` and HF `model_type`, split into `Supported` (baseline + name mapping + metadata mapping) and `Planned` (recognized, refused, carries a one-clause `needs` and the witness URL its key was read off). Same admission rule as `gguf_names.rs` -- every key comes off a real published file, and `tests/arch_registry_network.rs` re-reads each witness. Planned rows get NO `ModelFamily` variant on purpose: `known_architecture` is exhaustive and `arch_validation` compares its baseline field by field, so a placeholder would validate installs against invented numbers. `family_for_architecture` therefore still returns `None` for them. **ROADMAP's 1-bit entry added the closest pair in the HF table and it is worth knowing why the lookup is exact equality**: Bonsai-27B reports `model_type: qwen3_5` while Qwen 3.6 reports `qwen3_5_moe`, and a prefix match would resolve every Bonsai checkpoint to the MoE family -- a baseline with 256 experts and a decode flow with a router in it, i.e. fluent wrong output rather than an error. `tests/arch_registry.rs::the_two_qwen_model_types_do_not_collapse_into_one_family` pins both directions. **`Supported` does NOT imply a decode flow, and two rows prove it**: `llama`'s dense half was refused at `open()` for a whole phase after promotion, and ROADMAP M5 promoted `gpt-oss` with its kernels landed and its flow still pending. The registry answers "which family is this string"; `RealForwardRunner::open` answers "can it run", and they have been separate gates since M2. Promoting ahead of the flow is what lets `arch_from_gguf` derive an `ArchConfig` from the real header and be checked against the baseline BEFORE a multi-GB stream -- the check that saved M3 a second download and that M4 skipped, at two re-streams.
- `gguf_names/`: GGUF-to-canonical name tables (`gemma4.rs`, `gpt_oss.rs`, `llama.rs`, `qwen.rs`), one per family, every row read off a real published file and cross-checked against the corresponding real install's resident index. ROADMAP M5's `gpt-oss` table differs from the `llama` one next door in three ways that would each surface as an unmapped name mid-walk: the post-attention norm is spelled `post_attention_norm` rather than `ffn_norm`, EVERY projection and the router carry a `.bias`, and `attn_sinks.weight` has no counterpart in any other family. Its PER-EXPERT biases map to the ROUTED roles `gate_biases` / `up_biases` / `down_biases`, which the INT4-affine layout already defines and every GGUF install so far has left empty -- so they land in `MoeExpertOffsets`' three unused bias fields with no new plumbing, and the affine-vs-GGUF discriminator still keys on `gate_scales`. Mapping them `Resident` instead would have been silent rather than fatal (the offsets would simply stay 0), which is why that row has its own test.
- `gguf_config/`: `arch_from_gguf` (`attention.rs`, `masks.rs`, `meta.rs`). Starts from `known_architecture(family)` and overrides ONLY the fields GGUF actually determines, because behavioral fields are hardcoded in llama.cpp's graph builder and absent from the metadata.
- `gguf_checkpoint/`: the GGUF repack walk (`types.rs`, `plan.rs`, `transcode.rs`, `manifest.rs`). Expert bytes are sliced per expert and copied through with no quantization step. Splits Gemma's fused `ffn_gate_up_exps` at `FUSED_GATE_FIRST`, which is now MEASURED against the real file rather than assumed (gate is the first half; `tests/gguf_fused_gate_network.rs`). The one thing it does NOT carry verbatim is the resident F32 core: `transcode_f32` narrows norms to BF16 and quantizes the router to INT8 affine, because no F32 kernel exists here (Gotcha 6). It also un-does Qwen's V-head ordering (`v_head_axis`, Gotcha 7) and, for the `llama` family, llama.cpp's ROTARY PAIR permutation of `attn_q`/`attn_k` (`unpermute_rotary_rows`, ROADMAP Phase M2) -- both move bytes without changing any.
- `synthetic_gguf/`: `GgufBuilder` (`builder/`) plus `build_synthetic_gemma4_gguf` (`gemma4.rs`), carrying every name and metadata key the real Gemma 4 GGUF has. `SyntheticGgufShape::mix` picks the block-type mixture (`QuantMix`). `iq_mixed()` is ROADMAP Phase S's, and it is the only fixture mixed along TWO axes: IQ3_XXS gate/up over an IQ4_NL down (the phases of one expert differ) with the LAST layer at IQ4_XS over Q8_0 (the layers differ). Its odd layer is last, not first, because a resolve-once bug would take layer 0's answer and a first-layer difference would hide it. Its IQ tensors are random VALID CODE POINTS rather than quantized weights, since this port has no IQ encoder and will not grow one. `mxfp4()` is ROADMAP M5's: MXFP4 routed experts over a Q8_0 everything-else, the first fixture whose expert type has NO resident GEMV, which is what makes it exercise the split between the manifest gate and the resident dtype backstop. It carries none of gpt-oss's FLOW differences -- it is a Gemma-shaped install with gpt-oss's block types, exactly as `iq_mixed()` is one with the Phase S candidate's -- so it proves the MXFP4 pair dispatches inside a whole forward pass and nothing about the gpt-oss layer. Its `moe_intermediate` is 32 rather than 256 on purpose: MXFP4's block is 32, and a fixture whose every dimension is the largest block in the file cannot catch a kernel striding by the wrong one. `k_quant()` switches it to the mixture a real `Q4_K_M` carries (Q4_K experts and embedding, Q6_K attention, the rest Q8_0), which is the only in-repo way to exercise a MIXED install; its dimensions are all 256 because a K-quant row cannot be a partial superblock, and its weights go through the CPU quantizers at [-0.5, 0.5) for the dynamic-range reason below. Its Q8_0 quants deliberately span only `[-8, 7]` at a scale of 0.0625: at full byte range the weights reach +/-32 and a fixture install overflows FP16 before it reaches the head, which reads as a kernel bug and is not one. The default shape's `moe_intermediate` is 16, which is NOT a whole Q8_0 block, so a test that wants to RUN a fixture install has to widen it (`crates/runtime/tests/gguf_install_refused.rs` does). **`gptoss.rs` is a different KIND of fixture from all of those and that is the point** (ROADMAP M5): the three `QuantMix` variants are Gemma-shaped installs carrying another model's block TYPES, which proves a kernel dispatches and nothing about the layer. `SyntheticGptOssShape` reproduces gpt-oss's tensor INVENTORY instead -- unfused gate/up, a bias beside every projection, a sink vector per block, RANK-2 per-expert biases in the routed blob, an untied head, an alternating window with no published pattern -- so it can be built before the 12.1 GB stream rather than after it. It found two walk holes in milliseconds each that would each have cost a 25-minute re-stream (AGENTS.md Gotcha 42). Its gate/up and down bias widths are deliberately UNEQUAL (32 against 64) where the real 20b has both at 2880: the published file cannot distinguish the bias of a projection that outputs `n_ff` from one that outputs `n_embd`, and a fixture that copied its proportions could not either.

## Development & Test Commands

```sh
# Run fast unit tests for turbospark-repack
cargo test -p turbospark-repack

# Run network checkpoint integration tests (ignored by default, downloads large files)
cargo test -p turbospark-repack --test gemma4_checkpoint_network --release -- --ignored --nocapture
cargo test -p turbospark-repack --test hf_checkpoint_network --release -- --ignored --nocapture

# Qwen 3.6: ~20.4 GB across four shards. The ONLY test that covers the
# multi-shard walk on this family (the synthetic fixture is one shard).
TURBOSPARK_QWEN36_INSTALL_DIR=~/models/qwen36.gturbo \
  cargo test -p turbospark-repack --test qwen36_checkpoint_network --release -- --ignored --nocapture

# GGUF intake: reads only the HEADER of the real published GGUFs (a few MB
# off a 20-27 GB file, ~4 s each). With the install vars set it also asserts
# every tensor name maps and that the ArchConfig derived from GGUF metadata
# equals the one the .gturbo install declares.
TURBOSPARK_GEMMA4_INSTALL_DIR=~/models/gemma4.gturbo \
TURBOSPARK_QWEN36_INSTALL_DIR=~/models/qwen36.gturbo \
  cargo test -p turbospark-repack --test gguf_checkpoint_network --release -- --ignored --nocapture

# Which half of Gemma's fused ffn_gate_up_exps is the gate, by correlating a
# dequantized layer 0 expert 0 against the MLX install. Few KB, ~5 s.
TURBOSPARK_GEMMA4_INSTALL_DIR=~/models/gemma4.gturbo \
  cargo test -p turbospark-repack --test gguf_fused_gate_network --release -- --ignored --nocapture

# The evidence behind Gotcha 6's transcode decision. Needs no install.
cargo test -p turbospark-repack --test gguf_f32_transcode_network --release -- --ignored --nocapture

# The Q4_K reference against real published bytes: a dequantized layer 0
# expert 0 gate row of the real Qwen 3.6 Q4_K_M against the same row in the
# install. Few KB, ~5 s. AGENTS.md Gotcha 30 before reading a low number.
TURBOSPARK_QWEN36_INSTALL_DIR=~/models/qwen36.gturbo \
  cargo test -p turbospark-repack --test gguf_q4_k_network --release -- --ignored --nocapture

# The same, for Phase S's three IQ types against the candidate checkpoint.
# Covers all three in one run because the file puts a different one in each
# half of a routed expert, and a THIRD on layer 29 alone. Few KB, ~5 s.
TURBOSPARK_GEMMA4_INSTALL_DIR=~/models/gemma4.gturbo \
  cargo test -p turbospark-repack --test gguf_iq_network --release -- --ignored --nocapture
```

## The MTP head's fidelity check

`tests/mtp_install_fidelity_network.rs` correlates the INSTALLED head against
the published BF16 shard, dequantizing exactly as `dequant_int4_gemv_simd`
does (0.992-0.996 on 2026-08-18). It exists because
`tests/mtp_quantize_network.rs` answers a narrower question than its name
suggests: it quantizes a freshly-fetched row and dequantizes it with its OWN
helper, so it validates the quantizer against itself and passes whenever the
writer and the reader share a mistake (AGENTS.md Gotcha 48). Only a check
against an independent source can see a walk that wrote well-formed,
distinct, non-zero bytes that are not the RIGHT bytes -- which is the shape of
the failure this head has already had once.

## Crate Gotchas

1. **Synthetic Model Weight Meaning**: Synthetic models built by `build_synthetic_gemma4_install` use deterministic pseudo-random numbers rather than trained weights. Generated text on synthetic installs is structurally valid but semantically gibberish (and short generations may yield empty strings).
2. **`build_manifest_json` writes every family-extension field unconditionally.** `arch_validation` resolves omitted ones against the GEMMA baseline whatever family the manifest claims, so a Qwen install that leaves them out can never load. Gemma installs are unaffected (those are its own fallbacks). Do not make any of them conditional. Float fields additionally have to be binary fractions to survive serde_json's ~1-ULP default parser -- see AGENTS.md Gotcha 24.
3. **Synthetic Model Uniform Routing**: Synthetic MoE routers feature near-uniform routing logits. Expert slot permutation bugs cannot be caught by testing synthetic models alone; routing assignments must be validated structurally.
4. **The walk writes an install for every block type it can parse; whether that install RUNS is decided elsewhere, per block type.** As of Phase G Stage 2 a Q8_0 or Q4_K install opens and decodes (each has a resident GEMV, an embedding lookup and a routed-expert decode pair, all parity-tested), and so does a Q6_K one; ROADMAP Phase S adds IQ3_XXS, IQ4_XS and IQ4_NL, each with a resident GEMV plus the HALF of the routed pair its real file asks for (phase 1 for the first two, phase 2 for the third). Q4_0 installs and is refused. The two gates are `model_io::validate_quant` reading the manifest's `ggmlType` against `model_io::EXECUTABLE_GGUF_TYPES`, and `RealForwardRunner::open` reading the resident index's dtype tags against its own copy of that set. Do not widen either without landing kernels: both directions are asserted in `crates/runtime/tests/gguf_install_refused.rs`, which also decodes a Q8_0 install. AGENTS.md Gotchas 29 and 30 list the traps in the format and in checking it against real files.
5. **The ranged GGUF `*_network` tests cost KB or MB, not GB, and none downloads a checkpoint. The two INSTALL tests are the exception and stream instead.** `gguf_install_network.rs` (Gemma Q8_0) and `gguf_qwen_install_network.rs` (Qwen Q4_K_M) each read the published file over HTTP a layer at a time and write only the install, 25 GB and 19 GB respectively, in about 24 minutes. What they buy over the ranged tests is everything the fixture cannot show: every hole the real files exposed was in the SHAPE of the model rather than the block format (a rank-1 INT8 transcode target, a manifest slot probe anchored to `blk.0.` on a hybrid model whose layer 0 has no `attn_q`, an INT4-only fused kernel, and Qwen's gated-DeltaNet convention gap -- ROADMAP item 10). `gguf_qwen_core_probe.rs` is the diagnostic that followed, comparing a GGUF install's resident core against the MLX one tensor by tensor. The rest read ranges off a 20-27 GB remote file and finish in seconds. `gguf_checkpoint_network.rs` reads the header and is the only place a name-mapping hole or a converter disagreement can surface, because a synthetic fixture only ever contains names its author already knew: run it after touching `gguf_names.rs` or `gguf_config.rs`. Its three `scopes_phase_s_*` cases survey candidate sub-4-bit checkpoints for ROADMAP Phase S and print a ggml type histogram in BYTES; read the UNSIZED rows before the percentages, because a type with no `ggml_type_block` row is one this port cannot ingest and on a mixed file that is usually the routed experts (printing it as 0 bytes once made an imatrix file look 76% Q8_0, and the sized total not matching the published file size is what catches it). `gguf_fused_gate_network.rs` reads two output rows (a Q8_0 row of `hidden` elements is `hidden / 32 * 34` CONTIGUOUS bytes) and `gguf_f32_transcode_network.rs` reads whole norm and router tensors, which are vectors and a small matrix, and `gguf_q4_k_network.rs` reads 16 consecutive Q4_K rows (`hidden / 256 * 144` contiguous bytes each) to hold the Q4_K reference against real bytes. Before budgeting a download for the next GGUF question, check whether the answer is a contiguous byte range; the block layout makes more of them so than the planar affine layout would. `scopes_the_dense_llama_candidates` (ROADMAP M4 Phase 0) is the same trick applied to CHOOSING a checkpoint rather than decoding one: three headers decide which dense `llama` file the dense half should target, by reading whether it carries `rope_freqs.weight` and whether every block type in it already has a kernel. It uses `try_fetch` rather than `fetch`, because a candidate this parser REFUSES is itself a result and must not kill the survey -- TheBloke's 2023 Llama 2 is GGUF v2 and reports as a row. Note the gate it must NOT apply: F32/F16/BF16 are transcoded at repack (Gotcha 6) and never reach a dispatch, so checking them against `EXECUTABLE_GGUF_TYPES` (a set of BLOCK types) marks every candidate blocked, which is what its first run did.

   **Pin a new network test from HEADERS, no download.** `curl -sI` on `.../resolve/main/<any file>` returns `x-repo-commit`, which is the revision to pin instead of `main`; `curl -sI` on a shard returns `x-linked-size` and `x-linked-etag`, the published byte size and the file's SHA-256. Seconds, no bytes, and it is how `qwen38_checkpoint_network.rs`'s MTP arm pinned `Qwen/Qwen3.8-27B`. Note a content GET needs `-L`: a `resolve/` URL without it returns a ~300-byte redirect page that fails as invalid JSON rather than as an HTTP error.

   **A streamed install writes its resident region ONCE, at the end.** `read_resident_entries` accumulates the whole region in memory before `build_resident_weights_bin_mixed` writes it, so the target directory stays 0 bytes for the entire walk (~15 min on qwen38). `du` on it is not a progress signal; watch the process RSS and `nettop -P -l 1 -n -x` instead.

6. **GGUF's F32 norms and F32 router are TRANSCODED at repack time, and nothing F32 reaches the install.** `gguf_checkpoint/transcode.rs::transcode_f32` narrows F32 to BF16 by default and INT8-affine-quantizes the router. The decision rests on measurement, not preference (`tests/gguf_f32_transcode_network.rs`): llama.cpp UPCAST norms that are BF16 in the original checkpoint, so narrowing them back is bit-exact and costs nothing, and the router transcode applies the same INT8 affine the MLX path already applies to the same tensor without moving the routing decision. Do not re-open this as a tradeoff; it was measured to a conclusion. See AGENTS.md Gotcha 29. Three things to know about the implementation. The INT8 set is keyed by CANONICAL NAME per family (`int8_transcode_targets`), not by rank: Qwen's `linear_attn.conv1d.weight` is a rank-2 F32 tensor the runtime reads as BF16 while `mlp.gate.weight` is rank-2 and must be dtype 5, so rank does not decide. BF16 is the safe default because a mis-targeted tensor fails LOUDLY at `open()` (`norm_view`'s byte-size check, or `encode_gemv_any`'s "no dispatched GEMV kernel"). And the manifest's `router` slot consequently says `affine`/8/group-64, which is not a relaxation of the Stage 1 refusal but a true statement about the bytes: the other four slots still say `"gguf"` and are what `load_manifest` refuses on. A converter that did NOT upcast is not rejected (there is nowhere else to put the values), but every lossy value is counted into `GgufRepackOutput::lossy_narrowing` and reported through the streamed writer's `progress` callback.

7. **A Qwen GGUF's V-HEAD AXIS is ordered llama.cpp's way, and the convention belongs to the AXIS rather than to a list of tensors.** llama.cpp interleaves V heads where MLX keeps them contiguous: GGUF head `h` is MLX head `2h` for the first half and `2(h - heads/2) + 1` for the second, which is Qwen's two-V-heads-per-K-head grouping written one way by each side. `v_head_axis` in `gguf_checkpoint/transcode.rs` owns the table and `apply_source_convention{,_bytes}` apply it; `ssm_a` additionally holds `-exp(A_log)` where the install carries `A_log`, so that one tensor takes a value transform as well. All measured, never read off the converter (`tests/gguf_qwen_core_probe.rs`, `tests/gguf_qwen_quant_probe.rs`). THE COSTLY MISTAKE WAS SCOPING IT AS THREE TENSORS: the three GGUF ships as F32 were characterized first only because a BF16 probe can compare them directly, and fixing just those left the model generating gibberish, because five more tensors index the same axis and are QUANTIZED (`in_proj_qkv`, `in_proj_z`, `in_proj_a`, `in_proj_b`, `out_proj`). Ask whether a new tensor has a V-head dimension, never whether it is F32. Two implementation notes. The SPAN differs per tensor (one element of `A_log`, `value_head_dim` rows of `in_proj_z`, `value_head_dim` COLUMNS of `out_proj`), so one shared helper call with one stride is wrong in a way that still correlates high. And the quantized ones are permuted AS BYTES, never dequantized: every block layout here tiles along the fastest-varying dim, so a row is a contiguous run and a V head is a whole number of blocks -- checked rather than assumed, since a Q4_K superblock (256 elements) is wider than a 128-element head and is refused by name rather than shuffled in halves.

   The measurements, against the real `Qwen3.6-35B-A3B-Q4_K_M` install and the MLX one, both local and both deterministic (no error bars; treat movement as a real change). Mean |Pearson| per tensor over all 30 gated-DeltaNet layers, from `tests/gguf_qwen_convention_patch.rs`:

   | tensor | as-is | de-interleaved |
   |---|---|---|
   | `A_log`, `dt_bias`, `conv1d.weight` | exact after their transform | -- |
   | `in_proj_a.weight` | 0.23927 | 0.99372 |
   | `in_proj_b.weight` | 0.22200 | 0.99359 |
   | `in_proj_qkv.weight` | 0.08339 | 0.99514 |
   | `in_proj_z.weight` | 0.09297 | 0.99525 |
   | `out_proj.weight` (COLUMNS) | 0.12832 | 0.99542 |

   Correlation and not equality because the two installs hold different quantizations of the same trained weights; the three BF16 rows ARE exact, at worst relative error 0.000000 for the pure permutations and worst ABSOLUTE 0.003783 for `A_log` (against a BF16 double-rounding bound of `(1 + max|A_log|) / 512 = 0.0111`). Relative error is the wrong metric for `A_log` because it passes through zero, which reads 0.196 and means nothing -- the same trap the router transcode hit at 166. Coverage: norms bit-identical on all 40 layers (110 tensors), the de-interleave exact on all 60 permutation-only tensors. Where a tensor has one row per head the permutation is RECOVERABLE outright by argmax over a 32x32 row correlation matrix, which is stronger than confirming a guess; `in_proj_b` recovers `[0, 2, ..., 30, 1, 3, ..., 31]` exactly while `in_proj_a` scatters four entries because those rows are near-constant and `pearson` returns 0.0 by contract (Gotcha 30 again).

   Repacking the real file reports 130 lossy-narrowing warnings: 101 norms (40 `attn_norm`, 40 `post_attention_norm`, 10 `attn_q_norm`, 10 `attn_k_norm`, 1 `output_norm`) plus 29 of the 30 `ssm_a`. That count moved from 131 when this transform landed and is NOT a regression: `ssm_a` is now narrowed as `ln(-ssm_a)`, so what is measured is the logarithm's BF16 alignment rather than the source's. Layer 0's 32 values happen to land on the grid; the rest lose 3 to 14 each.

   VERIFY THIS CLASS OF FIX BY PATCHING THE INSTALL, NOT BY REPACKING. The tensors sit at fixed `file_offset`/`size_bytes`, every transform preserves length, and `RealForwardRunner::open` runs no receipt or SHA-256 check (`model_io` has a verifier; the open path never calls it), so a whole-model coherence test costs seconds against a streamed repack's ~21 minutes. Run the repack ONCE at the end as the proof that the walk writes what the patch wrote. `tests/gguf_qwen_convention_patch.rs` is idempotent (it patches only a tensor whose transform agrees with the MLX install better than the bytes already on disk), so a second run is a no-op rather than a double permutation.

8. **A DENSE checkpoint takes the same walk, and every place the walk assumed routed experts turned out to be a DEFAULT stated wrong rather than a missing branch** (ROADMAP M4). `write_gguf_install_streamed` already had a `plan.routed.is_empty()` arm, so the shape was there; what was not there was what the empty case should SAY.
   - **`manifest.quant` has five fixed slots and no architecture fills all five.** A probe that finds nothing answers `absent`, `validate_quant` refuses `absent` because it is not a block type with a kernel, and a perfectly runnable install fails to open with a message about a component it never had. Three slots did this on a dense model (router, routed expert, shared expert), each found by one five-minute re-stream of the real Mistral. Every inapplicable slot now falls back to the ATTENTION type, as one rule (`or_attention`) rather than three patches: it is executable exactly when the install is. These are DEFAULTED statements, not measured ones, so anything resolving a DISPATCH must read the resident index or `packed_experts/layout.json` (Gotcha 10 in `crates/runtime` already requires that).
   - **The dense branch goes through `StreamingGturboWriter` at ZERO layers**, not through `write_gturbo_install_with_resident_index`, purely because the latter cannot carry a quant block and writes `"quant": null`. The two produce a byte-identical `layout.json`. It matters as soon as the model's `(num_layers, hidden_size)` collides with a shipped baseline, which `is_production_arch` keys on: Mistral 7B's `(32, 4096)` is Mixtral 8x7B's exactly.
   - **`head_dim` fell back to the family BASELINE when `attention.key_length` was absent**, where llama.cpp's own default is `embedding_length / head_count`. See AGENTS.md Gotcha 39; the general rule is that an optional key's fallback belongs to the FORMAT, never to a neighbouring model.
   - **`rope_freqs.weight` is REFUSED, not ignored.** It was an `Ignored` row until this phase made dense installs runnable and so made a Llama 3.1 checkpoint reachable. Dropping learned RoPE scaling gives a model wrong only past the training length.
   `tests/gguf_checkpoint.rs::a_dense_llama_gguf_installs_and_its_manifest_loads` is the fixture that catches all of it in milliseconds, and its absence is why the real file found the slots one per round trip: every other GGUF fixture is MoE, so nothing had ever asked what the walk writes when `plan.routed` is empty.

   **THE SAFETENSORS WALK HAD THE SAME TWO BUGS AND KEPT THEM UNTIL ROADMAP's 1-BIT ENTRY**, because its dense path wrote no quant block at all so nothing could read one. `write_gemma4_install` and `write_gemma4_install_streamed` both routed an empty-layer install through `write_gturbo_install_with_resident_index`, which writes `"quant": null`; both now go through `StreamingGturboWriter` at zero layers with `set_quant`, exactly as the GGUF side does. That immediately exposed the second half: `manifest_quant`'s router probe answers the model's DEFAULT width for a model with no router, and `validate_quant`'s router row accepts 8 only, so the dense `llama` fixture stopped opening the moment it started declaring its quantization. `manifest_quant_for(quant, family, has_experts)` now mirrors the ATTENTION slot into the three MoE slots when there are no experts -- `or_attention` on this side -- and `validate_quant` accepts a slot byte-identical to the attention one as the DEFAULTED statement it is. Found by a fixture in milliseconds; it would have been a 20-minute stream.

9. **The walk NARROWS every unquantized tensor to BF16 and records what that cost, because BF16 is the only unquantized width this port can dispatch.** `narrow_raw_to_bf16` replaced a `raw_dtype_tag` that mapped `BF16`/`F16`/`F32` onto three raw tags, of which `crates/runtime` reads exactly one: every consumer of an unquantized tensor (`norm_view`, `read_bf16_host`, every kernel binding a `device const bfloat*`) identifies it by BYTE SIZE and decodes it as BF16. F16 is the same width, so a verbatim F16 norm is misread rather than refused -- no error, values wrong by up to 2^112. Nothing found it for four checkpoints because all four are BF16 throughout; `prism-ml/Bonsai-27B-mlx-1bit` writes F16 for every unquantized tensor. **This is the GGUF side's `transcode_f32` decision made on the OPPOSITE measurement**, which is why both are worth reading together: that one narrows F32 that llama.cpp had upcast from BF16 and is exactly lossless (Gotcha 6), this one narrows real F16 and is not. Measured off the real header before any code: the five RMS-norm families lose 19.5% of their values at a worst relative error of 0.003891 (2^-8, BF16's quantum) and the gated-DeltaNet tensors lose nothing, because that QAT checkpoint stores them on a grid coarse enough to be exact in both. The real repack reproduces it: 161 tensors narrowed lossily, 546,190 values, 192 GDN tensors clean, reported through the streamed writer's `progress` callback rather than counted in silence. AGENTS.md Gotcha 45 states the rule and the judgement; `tests/synthetic_qwen35.rs`'s two narrowing cases pin it, and the fixture had to be changed to write F16 for them to mean anything -- it was forked from the Qwen 3.6 one and had inherited BF16 norms beside correctly-F16 companions.
