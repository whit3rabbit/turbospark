---
title: "The .gturbo Install Format"
description: "On-disk reference for .gturbo model installs: directory inventory, manifest.json fields, the resident core, packed-expert stride arithmetic, and the validation rules the reader enforces."
diataxisType: "reference"
---

A `.gturbo` install is a directory holding one quantized model: a JSON
manifest, a binary resident-weight file, and one binary blob file per layer
of routed experts. It is the only install format this engine reads.

- **Writer**: `crates/repack/src/gturbo_writer/` assembles the directory
  (`write_gturbo_install*` for in-memory assembly,
  `StreamingGturboWriter` for the layer-at-a-time streamed walk used by
  every real install). Building tensors from a downloaded checkpoint is the
  caller's job; this module is the on-disk assembly step after it.
- **Reader**: `crates/model-io` decodes it back
  (`manifest::load`, `packed_experts_layout::load`,
  the resident index reader for `model_weights.bin`), and
  `turbospark-streaming`'s `PreadExpertStreamer` reads the expert blobs at
  decode time.
- **Swift parity**: the decoders are ports of the upstream Swift engine's
  readers (`manifest/mod.rs` and `packed_experts_layout.rs` module headers
  name `Infrastructure/ModelIO/ManifestReader.swift` and
  `Infrastructure/ModelIO/PackedExpertsLayout.swift`), so a `.gturbo`
  directory written by either engine reads in the other.

## Directory layout

```
<install>.gturbo/
  manifest.json                  # model manifest: arch, quant, file hashes (see below)
  model_weights.bin              # resident core: ResidentIndex header + raw tensor region
  packed_experts/
    layout.json                  # per-layer, per-expert byte offsets and sub-tensor roles
    layer_00.bin                 # one file per MoE layer; experts at fixed stride inside
    layer_01.bin
    ...
  packed_vision/                 # present only when the arch declares a tower
    layout.json                  # same schema as packed_experts/layout.json
    blobs.bin                    # ONE blob file: the tower is one "layer" of blocks
```

File facts:

| File | Written by | Notes |
|---|---|---|
| `manifest.json` | written LAST, after every other file | hashes every file it lists, so it must exist after them (`gturbo_writer/manifest.rs`) |
| `model_weights.bin` | `write_gturbo_install*` or `finish()` | see [Resident core](#resident-core-model_weightsbin) |
| `packed_experts/layer_NN.bin` | `build_layer_file` via `write_layer` / the batch writers | two-digit zero-padded name, `layer_{l:02}.bin` (`layers.rs`) |
| `packed_experts/layout.json` | same | four top-level keys; up to ~22 MB on a 40x256-expert model (`packed_experts_layout.rs`) |
| `packed_vision/{layout.json,blobs.bin}` | `write_packed_vision` | listed in `manifest.files` only when `arch.vision.is_active()`; the writer keys on the arch declaration, never on probing the filesystem |

A dense install with every weight resident is the same shape with an empty
expert set: `packed_experts/layout.json` carries `expertStride: 0`,
`numLayers: 0`, `expertsPerLayer: 0`, `layers: []`
(`write_gturbo_install_with_resident_index`, `layers.rs`).

## manifest.json

Top-level fields (writer: `build_manifest_json`; reader: `Manifest` in
`crates/model-io/src/manifest/types.rs`):

| Field | Type | Meaning |
|---|---|---|
| `magic` | string | Always `"GTURBO"`. Anything else fails with `NotAGTurboDirectory`. |
| `versionMajor` | i64 | `1`. Any other major fails with `UnsupportedVersion`. |
| `versionMinor` | i64 | `0`. |
| `flags` | map string -> bool | Only `streamingPresent`, `turboQuantKV`, `aneSharedExpert` are recognized; any other key is an error. `turboQuantKV: true` is refused (removed runtime support). |
| `modelID` | string | Model identifier. Note the exact spelling: `modelID`, not `modelId`. |
| `sourceSnapshotHash` | string or null | Source HF repository snapshot hash when recorded. The writer emits `null`. |
| `arch` | object | Architecture table; see the two tables below. |
| `quant` | object or null | Quantization slots; see [quant slots](#quant-slots). The batch writer emits `null`; `StreamingGturboWriter::set_quant` sets the real value. |
| `files` | map path -> `{size, sha256}` | Inventory and integrity for every file listed above. |
| `expertsPerLayer` | i64 | Routed experts per layer. |
| `numLayers` | i64 | Layer count; drives the per-layer file check on read. |
| `expertStride` | u64 | Model-wide MAXIMUM per-expert stride in bytes. Must be a multiple of 4096. Do not address a specific layer with it; use the layer's own stride. |

### arch: shape fields (required)

Decoded into `ManifestArch` (serde `camelCase`; `numKVHeads` and
`numFullKVHeads` are explicit renames that differ from mechanical
camelCase).

| Field | Type | Meaning |
|---|---|---|
| `hiddenSize` | i64 | Hidden dimension. |
| `ffnIntermediate` | i64 | Shared-expert FFN intermediate size. |
| `moeIntermediateSize` | i64 | Routed-expert intermediate size. |
| `numHeads` | i64 | Query attention heads. |
| `numKVHeads` | i64 | KV heads for sliding-window layers. |
| `numFullKVHeads` | i64 | KV heads for full-attention layers. |
| `headDim` | i64 | Head dimension, SWA layers. |
| `fullHeadDim` | i64 | Head dimension, full-attention layers. |
| `vocabSize` | i64 | Vocabulary size. |
| `slidingWindow` | i64 | Sliding-window context limit. |
| `finalLogitSoftcap` | f64 | Final logit soft-capping value. |
| `ropeTheta` | f64 | RoPE base, SWA layers. |
| `fullRopeTheta` | f64 | RoPE base, full-attention layers. |
| `partialRotaryFactor` | f64 | Partial rotary dimension factor. |
| `numLayers` | i64 | Total layer count. |
| `numExperts` | i64 | Routed experts per MoE layer. |
| `topKExperts` | i64 | Experts selected per token. |
| `tieWordEmbeddings` | bool | Embeddings tied to `lm_head`. |
| `attentionKEqV` | bool | K and V projections share memory structures. |
| `hiddenActivation` | string | MLP activation name. |
| `fullAttentionLayerMask` | [i64] | Per-layer attention type mask. |

### arch: family and extension fields (optional on the wire, written unconditionally)

Every field below is `#[serde(default)]` on the reader. The writer emits
ALL of them unconditionally anyway (`gturbo_writer/manifest.rs`):
`arch_validation` resolves an omitted field against the GEMMA 4 baseline
whatever family the manifest claims, so a non-Gemma install that leaves one
out can never validate. Gemma installs are unaffected; these are exactly
its fallbacks.

| Field | Type | Meaning; absent means |
|---|---|---|
| `family` | string | Model family identifier (`arch.family.as_str()`); absent means Gemma 4 (`peek_family`). |
| `attnOutputGate` | bool | Attention output gated; Gemma's value. |
| `attentionScale` | f64 | Attention scale factor; Gemma's value. |
| `embeddingScaledBySqrtHidden` | bool | Embedding scaled by sqrt(hidden); Gemma's value. |
| `routerScaled` | bool | Router weights scaled; Gemma's value. |
| `ffnSandwichNorms` | bool | Sandwich RMSNorms in the FFN; Gemma's value. |
| `sharedExpertGated` | bool | Shared-expert output scalar-gated; Gemma's value. |
| `ropeNeoxSubdim` | bool | RoPE subdim uses NeoX ordering; Gemma's value. |
| `linearNumKHeads` / `linearNumVHeads` | i64 | Linear-attention (gated DeltaNet) head counts. |
| `linearKeyHeadDim` / `linearValueHeadDim` | i64 | Linear-attention head dims. |
| `linearConvKernelSize` | i64 | Depthwise conv kernel size. |
| `linearOutputGateSigmoid` | bool | Gated-DeltaNet output-norm activation; absent means SILU (every family before `qwen4_exp`). |
| `swigluLimit` | f64 | SwiGLU clamp limit; absent means 0.0. |
| `ropeScalingFactor` / `ropeScalingOriginalContext` / `ropeScalingBetaFast` / `ropeScalingBetaSlow` | f64 / i64 / f64 / f64 | YaRN scalars; absent means no scaling (`RopeScalingConfig::NONE`). |
| `visionDepth` ... `visionVideoTokenId` (15 fields) | i64 | Vision tower shape, token ids, and `visionMropeSection` (fixed `[i64; 3]`); absent means no tower (`VisionConfig::NONE`). |
| `caIndexNHeads`, `caIndexKvHeads`, `caIndexHeadDim`, `caIndexTopK`, `caIndexBudget` | i64 | Compressed-attention indexer shape (`qwen4_exp`). |
| `caCSACompressRate`, `caQLoraRank`, `caOLoraRank`, `caOGroups`, `caRopeHeadDim`, `caHCACompressRate`, `caCompressRopeTheta`, `caRopeScalingFactor`, `caRopeScalingOriginalMax`, `caRopeScalingBetaFast`, `caRopeScalingBetaSlow` | i64 / f64 | Compressed attention. Exact spellings `caCSACompressRate` and `caHCACompressRate` are serde renames, not the camelCase the rest derives; a mismatch deserializes to none and then silently validates against 0. |
| `hcMult`, `hcLowrank`, `hcSinkhornIters`, `hcEps` | i64 / f64 | Hyper-connections (residual stream count, bottleneck rank, Sinkhorn iterations, epsilon). |
| `pleNgramSize` ... `pleEosTokenId` (10 fields) | i64 / [i64] | The hashed n-gram PLE table (`qwen4_exp`); absent means no table. `pleLayerIds` is one-based, as the checkpoint spells them. |
| `numHashRoutedLayers`, `routerScoringFunc`, `routedScalingFactor` | i64 / string / f64 | Router variants; decoded when present, not written by this writer. |

:::caution
Float fields are compared with `!=` on read against a ~1-ULP serde_json
parse, so a value that is not a binary fraction may not round-trip. The
real families' values (1.0, 0.0625, 2^-4.5) do.
:::

### trainedContext

`arch.trainedContext` (u32) is install metadata, deliberately NOT an
`ArchConfig` field: validation compares arch fields one by one against a
per-FAMILY baseline while a trained context is per-CHECKPOINT. It is
annotated into `manifest.json` in place, AFTER the walk, by
`repack::trained_context::record` (idempotent; the manifest hashes every
file except itself, so rewriting it invalidates nothing) and read back by
`trained_context::peek`. Absent means the install predates the field and
resolves to the default context window.

### quant slots

`quant` holds five fixed component slots
(`ManifestQuant`); no architecture fills all five, and absent slots on
dense models mirror the attention slot's statement.

| Slot | Component |
|---|---|
| `embedding` | Embedding layer. |
| `attention` | Attention projections. |
| `router` | Router projection. |
| `sharedExpert` | Shared expert. |
| `routedExpert` | Routed experts. |

Each slot (`ManifestQuantSlot`):

| Field | Type | Meaning |
|---|---|---|
| `weightBits` | i64 | Bit width (4, 8, 1, 2). |
| `scheme` | string | `"affine"` or `"gguf"`. |
| `scaleType` | string | Scale dtype, e.g. `"BF16"` / `"FP16"`. |
| `biasType` | string | Bias dtype. |
| `groupSize` | i64 | Elements per quantization group (64 or 128). |
| `ggmlType` | string, optional | The ggml block type; present only on `scheme: "gguf"` slots. The DOMINANT type when the slot is mixed. |
| `ggmlTypes` | [string], optional | Every ggml block type the slot carries when it carries more than one; absent means "exactly `ggmlType`". This is the list the loader gate checks, member by member. |

Production-shape manifests (any install whose `(numLayers, hiddenSize)`
matches a shipped baseline, `is_production_arch`) are rejected by the
loader without a `quant` block.

## Resident core: model_weights.bin

Layout (writer: `build_empty_resident_index` and
`resident_writer::build_resident_weights_bin`; reader: the resident index
reader in `crates/model-io`):

```
offset 0   u64 LE  index_size     # bytes of the entry table that follows
offset 8   u64 LE  resident_size  # bytes of the raw tensor region
offset 16  u64 LE  entry_count     # named tensor entries
offset 24  [entry_count entries]  # named resident tensor index
then       [resident_size bytes]  # raw tensor data region, mmap'ed at open
```

The minimal form the batch writer produces has `index_size = 24` and zero
entries, wrapping just the raw tensor bytes. Real installs carry a full
named index built by `resident_writer`; a caller-supplied complete
`model_weights.bin` goes through
`write_gturbo_install_with_resident_index_and_experts`. Whether an install
carries a drafter (MTP head) is answered by the resident index contents
(`mtp.fc.weight` present or not); there is no manifest flag that could
disagree with the bytes.

## packed_experts/ layout

### Stride arithmetic

Per layer (`build_layer_file`, `layers.rs`):

```
widest        = max over the layer's experts of sum(sub_tensor bytes)
expert_stride = (ceil(widest / 16384) * 16384) min(model_wide_ceiling)
```

- The per-layer stride is the layer's OWN widest blob rounded up to
  `GTURBO_PAGE_BYTES` (16,384), clamped to the manifest's model-wide
  ceiling. On a mixed sub-4-bit checkpoint, model-wide padding writes
  16.2 GB where 10.3 suffice (ROADMAP Phase S).
- The round-up keeps every expert offset page-aligned for the streamer.
- The reader refuses a layer stride ABOVE the top-level `expertStride`,
  because a decode-time slot is allocated from the latter.
- `PackedExpertsLayout::expert_stride` (and `manifest.expertStride`) is
  the model-wide MAXIMUM. Address or size a layer with
  `LayerLayout::expert_stride`, never the top-level value.

### Addressing

Inside one `layer_NN.bin`:

- Expert `i`'s blob starts at absolute byte `i * expert_stride` and is
  exactly `expert_stride` bytes; experts are sequential, each zero-padded
  past its sub-tensors.
- Sub-tensors are written back to back from the blob start. Each entry
  records `offset` RELATIVE to the blob's start, plus `size`, `dtype`,
  and `shape`.
- Roles are free-form keys (`gate`, `gate_scales`, `gate_biases`, ...).
- The reader lowercases `dtype` at decode. Names the writer uses include
  `"int4"`, `"bf16"`, `"q8_0"`, `"iq3_xxs"`. A mixed install needs the
  dtype PER SUB-TENSOR; the manifest's single `ggmlType` cannot say which
  kernel a given dispatch wants.
- A decode-time reader `pread`s exactly one expert's
  `[offset, offset + size)` window; nothing else in the file is touched.

### layout.json schema

Top level (all required on read): `expertStride` (u64, model-wide max),
`numLayers`, `expertsPerLayer`, `layers` (array). Each layer entry:
`layer`, `file` (basename, e.g. `"layer_00.bin"`), `expertStride`
(optional per layer; falls back to the top-level value, which is what
every pre-Phase-S install relies on), `experts` (array of
`{expert, offset, size, tensors}`). `PackedExpertsLayout::expert(layer,
expert)` resolves an entry O(1). Read cap: 64 MiB
(`packed_experts_layout::DEFAULT_MAX_BYTES`).

### packed_vision/

`write_packed_vision` packs the vision tower through the same
`build_layer_file`: the tower is ONE layer of `depth` "experts" (blocks)
in a single `blobs.bin`, and `packed_vision/layout.json` carries the same
four top-level keys (`numLayers: 1`, `expertsPerLayer:` block count).
`model_io::load_packed_layout_from` takes the subdirectory as a parameter,
so the same decoder reads both without a second parser.

## Writer entry points

| Symbol (from `turbospark_repack::gturbo_writer`) | Purpose |
|---|---|
| `write_gturbo_install` | Batch write: manifest, layout.json, layer files, minimal `model_weights.bin` wrapping the given resident bytes. |
| `write_gturbo_install_with_resident_index` | Real named resident index, NO packed experts (dense synthetic installs). |
| `write_gturbo_install_with_resident_index_and_experts` | Caller-supplied complete `model_weights.bin` plus packed-expert layer files; the streamed-MoE shape. |
| `write_packed_vision` | The tower's two files; must be called BEFORE the manifest, which hashes them. |
| `StreamingGturboWriter::new(dir, expert_stride, experts_per_layer)` | Start a streamed install. |
| `StreamingGturboWriter::write_layer(&LayerBlobs)` | One layer to disk immediately; its bytes can then be dropped. |
| `StreamingGturboWriter::adopt_layer(&LayerBlobs)` | Record a layout entry for a layer file ALREADY on disk (resume). Refuses a truncated or differently-strided leftover by size check. |
| `StreamingGturboWriter::set_quant(value)` | The manifest `quant` block; production-shape installs need it. |
| `StreamingGturboWriter::finish(arch, model_id, resident_weights_bin)` | Writes layout.json, `model_weights.bin`, then the manifest. |

Writer inputs (`types.rs`): `SubTensor { role, bytes, dtype, shape }`,
`ExpertBlob { expert, sub_tensors }`, `LayerBlobs { layer, experts }`.
Errors: `WriterError::{Io, ExpertOversized, WrongExpertCount}`; an expert
wider than the declared stride is reported against the widest expert,
because the caller's ceiling is what is wrong.

## Validation rules on read

`manifest::load` / `validate` (`crates/model-io/src/manifest/mod.rs`), in
order:

1. `manifest.json` exists, else `PartialInstall`; size within the read cap
   (default `DEFAULT_MAX_BYTES` = 4 MiB), else `IndexCorrupt`.
2. `magic == "GTURBO"`, else `NotAGTurboDirectory`.
3. `versionMajor == 1`, else `UnsupportedVersion`.
4. Every `flags` key in `{streamingPresent, turboQuantKV, aneSharedExpert}`;
   `turboQuantKV: true` refused.
5. `arch` validated field by field against the resolved `ArchConfig`
   (`arch_validation`). Omitted extension fields resolve against the Gemma
   4 baseline; the writer therefore never omits them.
6. `quant` validated by `validate_quant` when present; REQUIRED when
   `(numLayers, hiddenSize)` matches any shipped production baseline.
7. `expertStride % 4096 == 0` (hardcoded page size; both target platforms
   use 4 KiB pages and the format carries no per-install page-size field),
   else `ExpertStrideNotPageAligned`.
8. `files` contains `model_weights.bin` and `packed_experts/layout.json`.
9. `files` contains a `layer_NN.bin` entry for every layer in
   `0..numLayers` (zero-padded or plain name both accepted).
10. When the resolved arch has an active vision tower, `files` also
    contains `packed_vision/layout.json` and `packed_vision/blobs.bin`.

`packed_experts_layout::load_from` additionally requires the four
top-level keys, well-formed layer/expert/tensor entries, expert ids (and
optional `physicalRank`) within `expertsPerLayer`, and every expert slot
filled.

### Twin gates on ggml block types

Whether an install RUNS is decided per block type, by two independent
checks:

- The manifest gate: `validate_quant` reads each slot's
  `ggmlType`/`ggmlTypes` against `model_io::EXECUTABLE_GGUF_TYPES`.
- The resident-index backstop: `RealForwardRunner::open` reads the
  resident index's dtype tags against its own copy of the executable set.
  This catches a hand-edited manifest.

They are twins, not copies. MXFP4 is the split case: executable as a
routed-expert block type, refused as a resident tensor tag. Widen neither
gate without landing kernels.

## Provenance

- `files` maps every listed path to its byte `size` and `sha256`
  (`model_io::hash_data`). `build_manifest_json` computes them by READING
  the finished files from disk, never the bytes in hand. A walk that
  forgot to write a file fails here rather than validating without it.
- The manifest is written last for the same reason.
- `sourceSnapshotHash` records the source HF snapshot when available.
- `trainedContext` is annotated post-walk by `trained_context::record`
  (idempotent; see above).
- `crates/repack`'s `install_verifier` re-verifies every file in
  `manifest.files` rather than trusting a prior receipt. The decode open
  path does not re-hash on every open; verification is the installer's
  job.

## Cross-links

- [Rust API: Model Data Path](/reference/rust-model-data-path/):
  reader-side crate APIs and the repack walk that drives this writer.
- [CLI: turbospark-model](/reference/cli-turbospark-model/): `pull`, the
  streamed installer that produces these directories.
- [CLI: turbospark-check](/reference/cli-turbospark-check/): `--model`,
  which accepts a `.gturbo path or catalog alias.

Source files this page is grounded in: `crates/repack/src/gturbo_writer/`
(`mod.rs`, `manifest.rs`, `layers.rs`, `streaming.rs`, `types.rs`),
`crates/model-io/src/manifest/` (`mod.rs`, `types.rs`),
`crates/model-io/src/packed_experts_layout.rs`, with the page-size
constant from `crates/repack` (`GTURBO_PAGE_BYTES = 16_384`) and the
post-walk `trainedContext` writer `crates/repack/src/trained_context.rs`.
