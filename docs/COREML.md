# Core ML models

TurboSpark runs three FluidInference Core ML exports. None of them is a
generic format: each is a set of fixed-shape Core ML programs plus host files,
and each has its own pipeline in `crates/runtime/src/coreml/`. The generic
runner (load, compile, marshal, state) is `crates/coreml`; read its
[AGENTS.md](../crates/coreml/AGENTS.md) before changing it.

All three need **macOS 15 or newer** (iOS 18 programs, multifunction packages,
or `MLState`). On macOS 14 every entry point refuses with a reason, and the
audio catalog reports `can_run: false`.

## Models

| Export | Pinned revision | Pipeline | License |
|---|---|---|---|
| [FluidInference/parakeet-redux-coreml](https://huggingface.co/FluidInference/parakeet-redux-coreml/tree/8c5ef97a29cd120dc76b354b3f22b7fec3b486f9) | `8c5ef97a29cd120dc76b354b3f22b7fec3b486f9` | `coreml::parakeet` | CC-BY-4.0 |
| [FluidInference/embeddinggemma-2-coreml](https://huggingface.co/FluidInference/embeddinggemma-2-coreml/tree/3f978ddf92cac9aa07eb2dcf6eb36cf450f3dba4) | `3f978ddf92cac9aa07eb2dcf6eb36cf450f3dba4` | `coreml::embedding_gemma2`, `embedding_gemma2_media` | Apache-2.0 |
| [FluidInference/unlimited-ocr-mlx-coreml](https://huggingface.co/FluidInference/unlimited-ocr-mlx-coreml/tree/4568571eacd864e0a253c609b3c9d0e00450cebc) (Core ML parts only) | `4568571eacd864e0a253c609b3c9d0e00450cebc` | `coreml::unlimited_ocr` | MIT |

Attribution: the Parakeet export is a Core ML conversion by Fluid Inference of
`moondream/parakeet-redux`, itself derived from NVIDIA
`parakeet-tdt-0.6b-v3`, under CC-BY-4.0. EmbeddingGemma 2 is a Google model;
the export's card says Apache-2.0, and whether the Gemma terms also apply has
not been checked. Unlimited OCR is baidu/Unlimited-OCR under MIT.

Host-side logic follows FluidInference/FluidUse at
`d278e17df85e39e8c21bb64a467da81c39576fcd` (Apache-2.0) where the model cards
leave details out: the OCR mask, ring and state hand-off, and the
EmbeddingGemma 2 audio windowing. Image preprocessing follows transformers
`image_processing_pil_gemma4.py` and `Gemma4VisionPooler` at
`28034545c35acac05793695bd5c878f66e6b544e`.

## What each one needs from the host

**Parakeet.** 15 s windows on the 80 ms encoder grid, about 2 s of overlap.
The greedy TDT loop follows the Redux branch of the CPU decoder: a blank
advances at least one frame, and steps are capped at `10 * frames`. Windows are
merged with the nemo contiguous-run merge, with the subsequence merge as the
fallback. Segments are sentences with word timings (`timing: "token"`).
Detection is by layout (the four `.mlmodelc` directories plus
`parakeet_vocab.json`), before any `config.json` read.

**EmbeddingGemma 2.** Text is a task prefix from `config.json`, then `<bos>`,
the tokens and `<eos>`, gathered from the bf16 table times sqrt(512). It runs
through the smallest `embed_S` that fits, with `<pad>` rows masked out. Special
ids are resolved from the tokenizer and checked against `config.json`.
Matryoshka sizes keep the leading values and re-normalize. Audio is 10 s
windows with a 160-sample left pad. Images use the Gemma 4 resize (sides a
multiple of 48 px), 16 px patches and 3 x 3 pooling. Both produce soft tokens
that go through the text model between their boundary tokens. `pack_256` is not
used: it needs its own parity gate first.

**Unlimited OCR.** Pages are aspect-fit into 1024 x 1024 with a gray-127
border (PIL `ImageOps.pad`, PIL bicubic) and normalized to [-1, 1]. The prefix
is `<bos>`, 273 page embeddings per page, then the prompt ids. Prefill runs in
128-row chunks into the `prefill_P` state, and the cache is copied into the
`decode_P` state (`CoreMlState::copy_from`). Generated token t writes ring slot
`P + t % 128`. The two vocabulary halves' top-16 lists are merged, and the
35-gram ban over the last 1024 tokens filters candidates. The bucket limit is
8 pages.

## Experiment: eg2demo word CTC

Branch `experiment/embeddinggemma2-ctc` also runs the three compiled programs
of Haku's Embedding Gemma Transcribe (eg2demo v5): the audio encoder, the
reader, and the block scorer. `coreml::embeddinggemma2_ctc` implements the
portable `audio::stt::embeddinggemma2` backend traits over them. Everything
else (frontend, windows, rerank, CTC) is portable Rust. The programs are
`.mlmodelc` directories, so nothing is compiled into `.compiled/`.

- The assets are not a Hugging Face export and are not cleared for
  redistribution. Download the v5 manifest yourself (curl, sha256 per file)
  and pass the directory as `TURBOSPARK_EG2_ASSET_DIR`.
- At open, each program's inputs and outputs are checked against the v5
  shapes, and a mismatch is refused.
- How it works, what was verified, and what was measured:
  [EMBEDDINGGEMMA2_WORD_CTC.md](EMBEDDINGGEMMA2_WORD_CTC.md).

## Placement and first loads

| Model | Default compute units | Override |
|---|---|---|
| Parakeet | preprocessor CPU, encoder GPU, decoder and joint CPU | `TURBOSPARK_PARAKEET_COREML_ENCODER_UNITS` |
| EmbeddingGemma 2 | text Neural Engine, audio and vision GPU | `TURBOSPARK_EMBEDDINGGEMMA2_COREML_UNITS` (text) |
| Unlimited OCR | page encoder GPU (fixed), decoder GPU | none; Neural Engine refused |
| eg2demo word CTC (experiment) | encoder, reader, scorer `all` | `TURBOSPARK_EG2_CTC_{ENCODER,READER,SCORER}_UNITS` |

Override values are `cpu`, `cpu_gpu`, `cpu_ane` and `all`.

- **Parakeet:** the Parakeet encoder defaults to the GPU because its first
  Neural Engine compile takes minutes (the card says about 7) and cannot be
  cancelled.
- **EmbeddingGemma 2:** text compiles each function for the Neural Engine on
  first use. Measured here: 15 s for `embed_32` at open, 37 s for `embed_128` on
  its first request.
- **OCR:** the card reports wrong page embeddings on the Neural Engine.

To see where Core ML will actually run a model, read its compute plan
(`MLComputePlan`, macOS 14.4+) instead of timing placements:

```sh
cargo run -p turbospark-coreml --example coreml_plan -- <model> --units ane [--function embed_64] [--ops]
```

It prints op counts and Core ML's cost estimate per function and preferred
device, and with `--ops` every operation with its supported devices. Only
the function the configuration binds gets placements, and constants get
none. Asking for the Neural Engine runs its compile, so the first call on a
large model takes as long as a first load.

## Disk

- `.mlpackage` files compile once into `<install>/.compiled/`, keyed by
  content and OS build. Pruning removes older entries for the same package.
  Measured sizes: 1.1 GB for EmbeddingGemma 2 (all three packages), 3.5 GB for
  OCR.
- Core ML also keeps a per-executable GPU cache under
  `~/Library/Caches/<executable name>/`. It was 2.7 GB for one OCR test
  binary. Every binary that loads a model gets its own copy.

## Gates

Real-model tests are `#[ignore]`d and skip when the model is absent.

```sh
cargo test --release -p turbospark-runtime --test coreml_parakeet -- --ignored --nocapture
cargo test --release -p turbospark-runtime --test coreml_embeddinggemma2 -- --ignored --nocapture
cargo test --release -p turbospark-runtime --test coreml_ocr -- --ignored --nocapture
cargo test --release -p turbospark-ffi --test embedding_request --test ocr_surface -- --include-ignored
TURBOSPARK_EG2_ASSET_DIR=~/models/eg2demo \
  cargo test --release -p turbospark-runtime --lib embeddinggemma2_ctc -- --include-ignored
```

Default model directories are under `~/models/coreml/`. The
`TURBOSPARK_*_COREML_MODEL` variables override them; see each test's header.

Measured on an M4 Max, macOS 26.6 (2026-10-13):

**Parakeet**
- Against the CPU Redux decoder on `moondream/parakeet-redux`, three `say`
  clips (15 s or less) gave identical token ids and frames.
- On a 37 s clip in three windows: WER 0.010 against the script and 0.000
  against the CPU single pass.
- Before the merge fix, the same clip measured WER 0.232 (tokens past the
  overlap were dropped).

**EmbeddingGemma 2**
- Against a coremltools CPU reference
  (`crates/runtime/tools/ref_embeddinggemma2_coreml.py`) with independently
  written preprocessing (the transformers image processor): text, image and
  audio cosine 0.999999 to 1.000000, with identical token ids.
- Neural Engine text against that reference: worst cosine 0.999582.

**Unlimited OCR**
- Two rendered pages (`crates/runtime/tools/gen_ocr_pages.py`): CER 0.000 with
  the single-page prompt and with the two-page prompt.
- The first run of a freshly built test binary failed once without a captured
  reason. Four later runs passed, including one with that binary's Core ML
  cache cleared. Disk had about 13 GiB free at the time, so a disk-space cause
  is possible but unconfirmed.

## Benchmarks (M4 Max, macOS 26.6, AC power, 2026-10-14)

Two scripts produce these, both warm and interleaved: each sample is a fresh
process (open, one discarded warmup, one measured pass), arm order rotates
per run, and peak `phys_footprint` comes from `/usr/bin/time -l`. Every
sample records its 1-minute load, and `quiet_host` is false when any sample
started at a load of 4 or more. The raw reports are in
`docs/verification/coreml-bench-2026-10-14-*.json`. Medians of 3 runs, range
in brackets. These are readings, not frozen rows in `docs/BENCHMARKS.md`.

```sh
cargo build --release -p turbospark-runtime --example coreml_backend_probe --example embeddinggemma3_probe
python3 crates/runtime/scripts/benchmark_coreml.py {parakeet|embed|ocr} ...
uv run --python 3.12 --with mlx-embeddings==0.1.0 --with mlx-lm==0.32.0 \
    python crates/runtime/scripts/benchmark_embeddinggemma3.py --mlx-dir ... --coreml-dir ...
```

**EmbeddingGemma 2, text, by placement.** 40 texts (32 short, 8 of about 100
tokens), document prompt, one text per call. `quiet_host` true (load 3.2 to
3.6).

| Arm | Texts/s | Open (warm) | Peak |
|---|---|---|---|
| Neural Engine | 288 [279-293] | 0.96 s | 317 MiB |
| GPU | 115 [113-120] | 0.99 s | 1060 MiB |
| CPU | 102 [101-103] | 0.92 s | 315 MiB |

The first Neural Engine open in this run took 12.9 s (function compiles);
the later ones hit the OS cache.

**EmbeddingGemma 300m (Gemma 3): MLX vs Core ML vs the CPU port.** The same
40 texts (18 to 99 tokens). Every arm embeds identical token ids, tokenized
once by the CPU port. Weights:

- CPU and MLX: `mlx-community/embeddinggemma-300m-bf16` at
  `2f420efb007317a1ef2aa1608f87ef24226e6807`.
- Core ML, three community exports. We did not run their converters.
  - `mlboydaisuke/embeddinggemma-300m-coreml` at
    `c2db8b8a68b710c4197685051c6936f0bfa99923`: compiled `encoder.mlmodelc`,
    fixed 1 x 128 tokens, int8 weights, iOS 26 opset (macOS 26 only). Recipe:
    john-rocky/CoreML-LLM `conversion/build_embeddinggemma_bundle.py`. Its
    `weight.bin` SHA-256 is
    `f81f60eb283f2c1479100beb27bbadfa7efc4576c6511ca3d13751e7680e950e`, the
    digest `visible-cx/embeddinggemma-300m-CoreML` reproduced from that recipe.
  - `erjigit17/embeddinggemma-300m-ane-coreml` at
    `886ec86e59d963569747301f25c668c99944dfca`: `.mlpackage`, fixed 1 x 128,
    8-bit linear weights, macOS 15. Its recipe is `convert.py` in the repo.
  - `shirochenkov90/embeddinggemma-300m-coreml` at
    `db295fedd5984b8c6d3cb83e0db1f209be8cb3d4`: fp16, enumerated 64 or 256
    tokens.

`quiet_host` false: two run-1 samples started at a load of 4.6 and 4.8, the
other 25 at 1.4 to 3.9. Report:
`docs/verification/coreml-bench-2026-10-14-eg3-fixed-shape.json`.

| Arm | Texts/s | Open | Peak | Cosine to CPU (min / mean) |
|---|---|---|---|---|
| Core ML mlboydaisuke, `all` (the default request) | 223 [223-224] | 31 s | 1209 MiB | 0.99914 / 0.99942 |
| Core ML mlboydaisuke, Neural Engine | 224 [222-226] | 31 s | 1207 MiB | 0.99914 / 0.99942 |
| Core ML mlboydaisuke, GPU requested | 55.5 [55.4-55.6] | 12 s | 831 MiB | 0.99942 / 0.99965 |
| Core ML mlboydaisuke, CPU | 55.2 [55.2-55.5] | 12 s | 830 MiB | 0.99942 / 0.99965 |
| Core ML erjigit17, Neural Engine | 316 [313-316] | 23 s | 1207 MiB | 0.99321 / 0.99699 |
| Core ML erjigit17, `all` | 90 [90-90] | 18 s | 1751 MiB | NaN for 40 of 40 texts |
| Core ML shirochenkov90, Neural Engine requested | 9.2 [9.2-9.2] | 3.7 s | 1069 MiB | 0.99996 / 0.99997 |
| MLX (GPU, bf16), mlx-embeddings 0.1.0, mlx 0.32.3 | 179 [175-180] | 0.85 s | 1489 MiB | 0.99413 / 0.99757 |
| CPU port (`crates/embeddings`, f32) | 9.7 [9.6-9.7] | 0.59 s | 1495 MiB | reference |

- **Placement** (`coreml_plan`):
  - mlboydaisuke under `ane` and `all`: 1950 ops on the Neural Engine and 7
    on the CPU, the 262k-row token gather among them. With the GPU
    requested, every op is on the CPU, which is why its GPU and CPU rows
    match.
  - erjigit17: 2001 ops on the Neural Engine and 24 on the CPU. Under
    `all`, all 2025 go to the GPU, where it returns NaN.
  - shirochenkov90: every op on the CPU under every request (see Gotchas).
    The earlier reading of this export, at 9.0 texts/s, is
    `coreml-bench-2026-10-14-eg3.json`.
- **Opens:** every Core ML open in this rotation recompiled. The same model
  and placement opened twice in a row took 64 to 91 ms the second time.
  Opening another model on the Neural Engine in between (erjigit17, then
  mlboydaisuke) made the next open recompile again, 22 to 31 s. The cache is
  Core ML's per-executable one. Why a second model evicts the first was not
  established, but an app that runs two Neural Engine models may hit it.
- **Peak** includes that in-process compile. A cached open of mlboydaisuke
  peaked at 29 to 30 MiB `phys_footprint`. What that figure counts for
  Neural Engine buffers was not checked.
- **Pick:** mlboydaisuke is the only export that is fast and finite under
  the default request: 23 times the CPU port and 1.25 times MLX, with closer
  agreement to the CPU port than MLX. erjigit17 is faster on the Neural
  Engine but agrees less, and returns NaN wherever Core ML places it on the
  GPU or the CPU.
- **Limits:** mlboydaisuke needs macOS 26. The
  `mihijito/embeddinggemma-300m-coreml-ios18` card says it is the same
  recipe rebuilt for macOS 15 (not measured here). Texts over 128 tokens
  need a longer export or truncation. visible-cx publishes buckets from 64
  to 2048 tokens, each a separate model with its own copy of the weights.

EmbeddingGemma 2's fixed-shape functions, above, also reach the
Neural Engine. The two tables are different models (300m Gemma 3 vs the Gemma 4
based EmbeddingGemma 2), so compare placements within a table, not across
them.

**Parakeet, one 37.3 s clip (three windows).** `quiet_host` false: load
went from 3.2 to 17.3 while another session started building. The CPU Redux
and Core ML transcripts were identical in every sample.

| Arm | Wall time | Real-time factor | Open | Peak |
|---|---|---|---|---|
| Core ML, GPU encoder (default) | 249 ms [248-600] | 150x | 0.48 s | 485 MiB |
| Core ML, Neural Engine encoder | 303 ms [290-303] | 123x | 13.3 s, first 417 s | 2328 MiB |
| CPU Redux (`audio::stt::parakeet`) | 14.1 s [12.7-16.5] | 2.6x | 2.9 s | 5081 MiB |

The first Neural Engine open compiled for 417 s (the card says about 7
minutes), and even cached it took 13 s. That is why the GPU stays the
default. The 600 ms GPU outlier ran at a load of 15.

**Unlimited OCR, one page.** `quiet_host` false (load 11.6 to 15.1). One
read of `testdata/coreml_ocr/page1.png`, single-page prompt, 72 output
tokens:

| Total read | Encode | Prefill | Decode | Peak |
|---|---|---|---|---|
| 1.54 s [1.52-1.59] | 152 ms | 309 ms | 1.09 s (66 tok/s) | 7.9 GB |

## Surfaces

| | Parakeet | EmbeddingGemma 2 | Unlimited OCR |
|---|---|---|---|
| Runtime | `ParakeetCoreMl` | `EmbeddingGemma2CoreMl` | `UnlimitedOcrCoreMl` |
| Server | `--stt-model`, `/v1/audio/transcriptions` | `--embedding-model`, `/v1/embeddings` (text) | none |
| C ABI | `ts_stt_*`, `ts_audio_*` sessions | `ts_embedding_encode_json`, `ts_embedding_encode_request_json`, `ts_embedding_catalog_json`, `ts_embedding_install` | `ts_ocr_*` |
| CLI | none | none | `turbospark ocr` |
| Swift | `TurboSparkSTTModel`, `AudioSession` | `TurboSparkEmbedding.encode(request:)`, `EmbeddingCatalog` | `TurboSparkOcr` |
| Catalog pin | `parakeet-redux-coreml` in `audio_assets.json` | `embeddinggemma-2-coreml` in `embedding_models.json` | none (explicit path) |
| App | Audio workspace, Transcribe and Record pages (listed, downloaded and run like any audio profile) | Settings, Memory and Server > Advanced: a Recommended row under the embedding field offers Download, then Use | none |

### Pinned installs

Both pins list every file by path, size and SHA-256 at an immutable
revision, and the transfer checks each one before the install is published
with one rename.

- **Parakeet** installs as an audio profile to
  `models/audio/parakeet-redux-coreml.gturbo`. Opening it from the app
  re-verifies the files, then checks that the loaded family is
  `parakeet_tdt`.
- **EmbeddingGemma 2** installs to `models/embeddings/embeddinggemma-2-coreml`
  with an `embedding-receipt.json`. It has no registry row, so it never
  shows up in the chat model pickers. `resolve_model_arg` falls back to this
  namespace, so the bare alias works in every embedding field, in
  `--embedding-model` and in the C ABI.

The network gate installs both into a scratch store (`TURBOSPARK_HOME`) and
runs them:

```sh
cargo test --release -p turbospark-ffi --test pinned_coreml_installs_network -- --ignored --nocapture
```

It passed on 2026-10-14 after downloading about 1.7 GB. The embedding alias
encoded at 128 dimensions, and Parakeet transcribed the clip exactly.

## Gotchas

These cost time while building this. Each one is guarded in code or a test
now; the note is here so the guard is not removed.

- **No NSException catching, ever.** objc2's `exception` feature compiles an
  Objective-C `@try` helper, which adds `__objc_personality_v0` to the link.
  The app already links three unwind personality routines (two Rust static
  libraries plus C++), and compact unwind encodes at most three. The app link
  then fails with "Too many personality routines for compact unwind to
  encode". `cargo build`, `make swift-lib` and `make swift-test` all pass;
  only `make swift-app` shows it. Calls are made safe up front instead:
  inputs are checked against the model description, and newer APIs are gated.
- **macOS 15 APIs fail at run time, not link time.** objc2 looks classes and
  selectors up when called, so `setFunctionName:` or `newState` on macOS 14
  raises instead of failing to build. Everything newer than the macOS 13
  package floor goes through `supports(Capability::...)`.
- **A fixed shape is often not "unspecified".** ML programs describe a fixed
  input as a one-entry enumeration or as ranges with equal bounds. Shape
  checks that only read the unspecified form treat every input as flexible
  and skip the check. `sys::fixed_shape` reads all three.
- **Flexible input shapes keep a model off the accelerators.** The
  shirochenkov90 EmbeddingGemma 300m export (enumerated 64 or 256 tokens)
  ran at the same speed on every placement, and its compute plan puts all
  2521 non-constant ops on the CPU whether the Neural Engine, the GPU or
  everything is requested. The graph computes shapes at run time: `shape`
  (121 ops), `tile` and `reshape` ops that only the CPU supports sit in
  every layer. Fixed-shape exports of the same model put the encoder on the
  Neural Engine. Prefer one fixed-shape function or model per length, and
  check with `coreml_plan`, not by timing.
- **One placement can return NaN while another is fine.** The erjigit17
  EmbeddingGemma 300m export (fp16, RMSNorm without rescaling) gives finite
  vectors on the Neural Engine and NaN for every text on the CPU and the
  GPU. Its plan under `all` is all GPU, so the default request returns NaN.
  Check every placement an export can get for non-finite output, not only
  the intended one.
- **Padded strides.** Neural Engine outputs can have padded rows. Every read
  and write goes through the reported strides, bounds-checked first.
- **These exports have no `model_type`.** STT and embedding dispatch used to
  read `config.json` first and refuse. `coreml::detect_layout` runs before
  any config read.
- **The speech probe refused every `.bin`,** to keep GGML out. Compiled Core
  ML programs store weights in `weights/weight.bin` and `coremldata.bin`, so
  the `coreml` format exempts `.bin` only inside a `*.mlmodelc/` directory.
- **The nemo merge dropped audio.** `merge_longest_contiguous` and
  `merge_longest_common_subsequence` kept only the overlap part of the
  second window, so a 37 s clip lost about 13 s in the middle (WER 0.232).
  Fixed to append the rest of the window, as upstream does. A regression
  test with a window longer than its overlap covers it.
- **Pad with `<pad>` rows, not zeros.** EmbeddingGemma 2 padding rows hold
  the `<pad>` embedding, as the reference does, and the mask hides them.
  Special ids are resolved from the tokenizer and must match `config.json`,
  or the model is refused.
- **Two caches, two places.**
  - Packages compile into `<install>/.compiled/`. Install verification
    ignores extra files for this reason.
  - Core ML keeps its own GPU and Neural Engine cache per executable name
    under `~/Library/Caches/<executable>/`. That cache was 2.7 GB for one OCR
    test binary, and every binary that loads a model builds its own copy.
- **First loads are slow and cannot be cancelled.**
  - The Neural Engine compiles each EmbeddingGemma 2 function on first use:
    15 to 37 s each here.
  - The card puts the Parakeet encoder at about 7 minutes on the Neural
    Engine, which is why it defaults to the GPU.
- **OCR prefill and decode are separate functions with separate states.**
  The prefill cache is copied into the decode state
  (`CoreMlState::copy_from`). It is not shared.
- **Audio admission undercounts Core ML.** `native_audio::estimate_open_bytes`
  counts only `*.safetensors`, so a Core ML model gets the 512 MiB floor.
  Parakeet's measured 488 MB peak fits under it. A larger Core ML audio model
  would not, and needs this fixed first.
- **`hf download --include` with several patterns** in this shell was read
  as a list of filenames and silently skipped a package. List files
  explicitly, or check sizes after the download.

## Not done

- **App:** there is no OCR in the app. `DocumentTextExtractor` is
  synchronous with a 30 s limit, and the model loads for several seconds and
  peaks near 8 GB, so wiring it in is a UX and memory decision.
- **Removing an embedding install:** the app has no delete button for one.
  Remove `models/embeddings/<alias>` by hand.
- **Catalog:** OCR has no catalog pin.
- **Server:** there is no OCR endpoint and no image or audio embeddings.
- **Memory admission:** the OCR footprint is not admitted through the load
  guard.
- **`pack_256`:** batching for EmbeddingGemma 2 is unused until it has its own
  parity gate.
- **EmbeddingGemma 300m on Core ML** is benchmarked only. There is no
  runtime pipeline, catalog pin or install for it. The candidate is the
  mlboydaisuke export (see Benchmarks), which needs macOS 26 and holds 128
  tokens at most.
