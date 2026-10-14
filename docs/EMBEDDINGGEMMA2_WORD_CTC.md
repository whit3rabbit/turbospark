# Speech recognition from an embedding space

How Embedding Gemma Transcribe turns EmbeddingGemma 2 into a speech-to-text
engine, what its distributed files actually contain, and what a portable
reimplementation in TurboSpark measured.

- Subject: Haku's "Embedding Gemma Transcribe" (`ai.haku.eg2demo`), app 1.6
  (build 8), model distribution eg2demo v5.
- Prepared by: whit3rabbit, TurboSpark project, branch
  `experiment/embeddinggemma2-ctc`.
- Period: 12 to 14 October 2026. Hardware for every measurement: Apple M4 Max
  (16 CPU, 40 GPU cores, 36 GB), macOS 26.6.2.
- Status: experiment. Evidence levels are labeled throughout (Section 1.2).

## Summary

Embedding Gemma Transcribe recognizes speech without a text decoder. A
trained reader maps each 40 ms frame of EmbeddingGemma 2 audio features to a
unit vector in the model's text-embedding space plus a blank score. The
nearest of 229,457 word embeddings wins the frame when its scaled cosine
beats the blank score, and greedy CTC collapse turns frames into words. The
idea works: TurboSpark's reimplementation transcribes real audio with it.

What this investigation established from the distributed files themselves:

- **The pipeline.** Five compiled or tabular stages: a Gemma4 log-mel
  frontend, a 12-block audio encoder, an 86,448,897-parameter transformer
  reader, a two-stage vocabulary search, and greedy word CTC. A separate
  4-layer model adds punctuation and casing. Every shape, constant, and
  decision rule in Section 4 was read from the programs, tables, or the app
  binary, not inferred from marketing text.
- **The decision rule.** A frame emits word `w` when
  `39.8567 * cos(z, e_w) - 0.4626 > blank`, strictly greater, so a NaN reads
  as blank. This was confirmed in the app's machine code.
- **The vocabulary.** The 229,457 rows are the stock EmbeddingGemma 2 text
  embeddings of each word string, with no task prompt (median cosine
  0.99999). 442 rows were changed, and the last entry, `<rep>`, is a trained
  marker for an immediately repeated word.
- **The search.** A 768-to-128 projection with orthonormal columns feeds a
  block top-16 over 4-word blocks in plain word-id order. It is not semantic
  clustering and not Matryoshka truncation. On the clips measured here the
  shortlist never lost a word that mattered.
- **The hardware story.** Core ML's compute plan puts most of the encoder and
  reader on the Neural Engine, but 28% of the encoder's estimated cost and
  73% of the scorer's sit on the GPU. The app also carries a GPU path. "Runs
  on the Neural Engine" is true in the main and false in the strict sense.

What the TurboSpark port measured, with every caveat in Section 8:

- The checked-in reference clip transcribes exactly. Three synthetic clips
  totaling 60.6 s gave 2 wrong words out of 195.
- The Core ML (FP16) and portable CPU (f32) shortlists made identical
  decisions on 1,527 frames, and both kept every emitted word's block.
- Word spikes trail the spoken word by 0.04 to 0.32 s. Without trailing
  audio, the final word of a clip can be lost. The port pads 0.5 s.
- One warm run took 0.38 to 0.46 s for 47 s of audio. That is an indicative
  reading, not a benchmark.

What remains open: the training recipe, whether the audio encoder was
fine-tuned (the app's NOTICE and its website disagree), WER on real
recordings, and redistribution terms for the assets.

## 1. Scope, sources, and evidence

### 1.1 Scope

In scope: the eg2demo v5 inference pipeline end to end, the app binary where
it decides frames and windows, and a reimplementation that runs the shipped
programs from Rust.

Out of scope, and stated now so nobody discovers it later:

- Training. The developer has not released code, data, or loss. Nothing here
  reproduces the reader's training.
- Punctuation and casing. Described from the program, not ported.
- WER on real recordings. Only synthetic and checked-in clips were measured.
- Licensing. The assets are modified derivatives of an Apache-2.0 model with
  a prohibited-use policy attached. Redistribution was not cleared, so no
  asset is committed and no catalog entry exists.

### 1.2 Evidence levels

| Label | Meaning |
|---|---|
| **Verified** | Read directly from a distributed file (MIL program, metadata, table, manifest) or from the app's arm64 machine code, with the location given |
| **Measured** | Produced by running code in this repository on the hardware above |
| **Reported** | Stated by the developer on the project site or on Hacker News, quoted at source on 14 October 2026, not reproduced |
| **Inferred** | Consistent with the evidence, not proven by it. Always worded as such |
| **Unknown** | Not determinable from what was available |

### 1.3 Sources

| Source | Pin |
|---|---|
| Asset manifest | `https://haku-models.tslfiles.org/eg2demo/v5/manifest.json`, version 5, created 12 October 2026, 26 files, 714,975,724 bytes, every file sha256-checked |
| App binary | `/Applications/Embedding Gemma Transcribe.app/Contents/MacOS/EG2Transcribe`, arm64, 2,297,376 bytes, sha256 `7bcdc0085df93f5e5538e54e943d82c9701172361c56ec6a4241e325fa182348` |
| Project site | `https://eg2.haku.sh/`, fetched 14 October 2026 |
| Developer comments | Hacker News item 50069971, "We turned EmbeddingGemma 2 into a speech-to-text model", 13 October 2026, user `Shershah1024` (introduces himself as Shahir) |
| Frontend reference | transformers `Gemma4AudioFeatureExtractor` at `28034545c35acac05793695bd5c878f66e6b544e` |
| Text model for the vocabulary probe | `google/embeddinggemma-2` at `914f7f89142e33e77833254d9c9b90c3cef7303b` |

## 2. Background

Most open recognizers decode autoregressively. An audio encoder builds a
representation, and a text decoder emits one token at a time, each step
waiting on the last. That loop is sequential and memory-bandwidth bound.

CTC models skip the loop. They label every frame at once, with a blank label
for "nothing new here", then collapse repeated labels and drop blanks. The
cost is a closed label set and no language model in the decision itself.

EmbeddingGemma 2 is a multimodal embedding model: text, images, audio, and
video map into one vector space. The project's premise, in the site's words,
is that a spoken word "lands close to the written word". Embedding Gemma
Transcribe uses the text embeddings of 229,457 English words as its CTC label
set. The only trained piece is the reader that moves audio frames into that
space.

The developer describes it the same way: "the reader is a CTC head on the
audio encoder, but its output words are EmbeddingGemma's own text embeddings
for ~229k words, instead of a separately learned vocabulary" (Hacker News,
13 October 2026).

## 3. Method

1. **Assets.** The v5 manifest was downloaded with curl and every file
   checked against its sha256. The host refuses Python's default urllib user
   agent with HTTP 403. The installed app had not downloaded its models, so
   the manifest was the only source.
2. **Programs.** Each `.mlmodelc` ships `model.mil` (the ML program as text),
   `metadata.json`, and `weights/weight.bin`. Shapes, operators, constants,
   and parameter counts were read from the MIL. Weight blobs were parsed at
   the offsets the MIL names.
3. **Placement.** Core ML's `MLComputePlan` was read through
   `crates/coreml/examples/coreml_plan.rs` with `--units all`.
4. **Binary.** `otool -tV` disassembly of the arm64 binary, demangled Swift
   symbols, and specific address ranges, cited where used.
5. **Vocabulary provenance.** Every word string re-embedded with the stock
   text model in float32 and compared row by row
   (`crates/audio/scripts/probe_embeddinggemma2_word_table.py`).
6. **Reimplementation.** A portable Rust pipeline with fixture parity tests,
   driven by a Core ML backend over the shipped programs (Section 7).
7. **Audio.** The checked-in forced-aligner clip (2.79 s, known transcript)
   and three macOS `say -v Samantha` clips rendered at 16 kHz float32.

## 4. The system, stage by stage

```
16 kHz PCM
  -> log-mel frontend (CPU)                       [2000, 128] per 20 s window
  -> eg2_audio_l11 (Core ML)                      [500, 1024]  40 ms frames
  -> reader_P6 (Core ML)                          vec [500, 768] unit norm, blank [500, 1]
  -> scorer_blk_d128_k16_b4_t500 (Core ML)        16 block ids per frame
  -> exact rerank of 64 candidates (CPU)          best word id, cosine
  -> 39.8567 * cos - 0.4626 > blank ?             word id or blank
  -> collapse adjacent ids, drop blanks, <rep>    words with frame spans
  -> punct_headonly + punct_case.json             punctuation and casing (not ported)
```

### 4.1 The distribution

| File or directory | Bytes | From | Role |
|---|---:|---|---|
| `eg2_audio_l11.mlmodelc` | 227,833,295 | v4 | Audio encoder |
| `reader_P6.mlmodelc` | 65,826,578 | v4 | Reader |
| `reader_P6.json` | 294 | v1 | Calibration sidecar |
| `scorer_blk_d128_k16_b4_t500.mlmodelc` | 58,942,693 | v1 | Block shortlist program |
| `word_unit_f16.bin` | 352,445,952 | v1 | 229,457 x 768 FP16 word rows |
| `reader_words.json` | 2,688,088 | v1 | Word strings |
| `hann_320.f32`, `mel_filters_257x128.f32` | 132,864 | v1 | Frontend tables |
| `punct_headonly.mlmodelc` | 6,829,228 | v3 | Punctuation and casing |
| `punct_case.json` | 275,841 | v3 | Casing overrides |
| `NOTICE` | 891 | v5 | Attribution |

The manifest's "split" layout pulls files from five earlier version
directories. One consequence is worth noting: the v1 calibration sidecar
rides with a v4 reader program. The sidecar's parity numbers may describe an
earlier export (**Inferred**). The calibration works with the v4 reader in
every test here (**Measured**).

All four programs were converted with coremltools 9.0 from TorchScript
(torch 2.7.0+cpu) between 9 and 11 October 2026 (**Verified**, metadata).
The encoder, reader, and scorer are iOS 18 programs that declare macOS 15.0.
The site lists macOS 14+.

### 4.2 Frontend

**Verified** (tables and fixture): the frontend is transformers'
`Gemma4AudioFeatureExtractor` with the `google/embeddinggemma-2`
configuration.

- 16 kHz, 320-sample periodic Hann window, 160-sample hop, 512-point real
  FFT, 128 HTK mel bands over 0 to 8000 Hz with no area normalization.
- Magnitude spectrum, not power. `log(mel + 0.001)`, an additive floor rather
  than a clamp. No preemphasis or dither.
- Semicausal padding: 160 zeros prepended, nothing appended. A frame is valid
  only when the last sample of its 321-sample unfold window is real audio.
- A full 20 s window yields 1,999 valid frames, not 2,000. The encoder's
  2,000th input frame is masked padding.

The shipped `hann_320.f32` matches a periodic Hann within 2.9e-8. The shipped
filterbank is the f64 HTK bank in `[bin][mel]` layout within 3.0e-8. Reading
it as `[mel][bin]` is off by 0.99, which makes the layout unambiguous. The
binary has an `EG2Transcriber.LogMel` type and calls Accelerate's `vDSP_DFT`
routines.

### 4.3 Windows and stitching

**Verified** (binary). Windows are 320,000 samples (20 s) with a 288,000
sample step (18 s), so neighbors overlap by 2 s. Each window yields 500
frames, and interior windows keep local frames `[25, 475)`. Kept ranges tile
the timeline exactly: window `w` contributes global frames
`[450 w + 25, 450 w + 475)`.

The edges were an open question, since dropping 25 frames would discard the
first and last second. The app answers it in two places,
`EG2Transcriber.transcribeShared` (0x100076464 to 0x1000764b4) and
`Stream.advance(final:)` (0x100077670 to 0x100077698):

```
keep_start = first window ? 0 : min(n, 25)
keep_end   = last window  ? n : min(n, 475)
```

`n` is the app's frame count for that window. Whether `n` counts only frames
with real audio was not traced (**Unknown**). TurboSpark arrived at the same
rule independently, keeping through the last valid frame.

### 4.4 Audio encoder (`eg2_audio_l11`)

**Verified** (MIL). Inputs are `input_features [1, 2000, 128]` and
`feature_mask [1, 2000]` (f32), and the output is `l11 [1, 500, 1024]`
(f16). The graph:

- Multiplies the features by the mask, then two 3 x 3 stride-2 convolutions
  (1 to 128 to 32 channels) and a linear projection to 1024. That is the 4x
  time reduction from 2,000 to 500 frames.
- Runs 12 blocks, indexed 0 to 11. Each holds two feed-forward modules, q/k/v
  attention with an output projection, and a light convolution module
  (pointwise, depthwise kernel 5, pointwise). Norms are RMS norms, and the
  graph clips activations at 312 sites.
- Has one `tanh` per block. That is consistent with an attention logit
  soft-cap (**Inferred**).
- Ends at `layers_11_norm_out`, the last block's normalized output.

So "layer 11" is the zero-indexed final block, not a cut before the last
layer. `config.json` lists 12 audio layers, and all 12 are in the program.
What the full model does after the conformer stack is absent. The encoder
holds about 290.7 M weights, palettized to 227 MB ("Mixed (Float16,
Palettized (12 bits), Palettized (13 bits), Palettized (14 bits), Palettized
(7 bits), UInt6)").

Whether these weights were fine-tuned is contested. The app's NOTICE says
"EmbeddingGemma 2's audio encoder was fine-tuned". The site says the encoder
"stays exactly as it is". Comparing the dequantized program weights with the
stock tower would settle it (Section 12).

### 4.5 Reader (`reader_P6`)

**Verified** (MIL). Inputs are `l11 [1, 500, 1024]` and `mask [1, 500]`
(f32), and the outputs are `vec [1, 500, 768]` and `blank [1, 500, 1]`
(f16). Summing the MIL's weight shapes gives 86,448,897 parameters, which
matches the sidecar's `params_M: 86.45` exactly.

| Stage | Operation |
|---|---|
| Input | LayerNorm over 1024, linear 1024 to 768 |
| Position | `x + DepthwiseConv1d(x)`, kernel 15, padding 7, 768 groups. No other position signal in the graph |
| Body | 12 pre-LayerNorm transformer layers. Fused QKV (768 to 2304), 12 heads of 64, scores scaled by 0.125, FFN 768 to 3072 to 768 with exact GELU, residuals around both |
| Attention mask | `(1 - mask) * -10000` added to every head's scores. Full bidirectional attention over the 500-frame window |
| Word head | Final LayerNorm, linear 768 to 768, then `x / sqrt(sum(x^2) + 2^-24)`. The output is unit norm inside the graph |
| Blank head | Linear 768 to 1 from the same final LayerNorm output |

LayerNorm epsilon is 1e-5 (FP16 `0x1.5p-17`). Linear and convolution
weights are palettized with 6-bit indices.

It is a transformer with convolutional position encoding, not a Conformer.
There is no convolution module inside the layers. Because attention spans the
whole window, every frame sees about 20 s of context. The developer's "about
20 words around it" is consistent with that.

The sidecar `reader_P6.json` holds scale `39.85666275024414`, bias
`-0.4626207649707794`, `params_M`, parity maxima (`vec` 1.67e-7, `blank`
7.63e-6, against an unnamed reference), and `arm: {layers: ["L11"], d: 768,
layers_n: 12, heads: 12, lr: 0.0003}`, with `epoch: null`. It names no
optimizer and no loss. Claims that the reader was trained with AdamW or a
cosine-margin CTC loss are not supported by this file.

### 4.6 Vocabulary

**Verified** (tables). `reader_words.json` lists 229,457 strings, and
`word_unit_f16.bin` is exactly `229457 * 768 * 2` bytes of headerless
little-endian FP16. Sampled rows have unit norm within 2e-3.

The list has structure:

- Ids 0 to 319 form a head block of common and programming words, plain and
  apostrophe-prefixed.
- Ids 320 to 229,425 are an alphabetically sorted main list, ending at
  `zzzz`.
- Ids 229,426 to 229,455 hold 30 appended custom words, such as `haku`,
  `coreml`, `mlx`, `parakeet`, `vllm`, and `kenlm`. The last entry, id
  229,456, is `<rep>`.
- 229,304 strings are distinct. 153 appear twice under different ids, mostly
  once in the head block and once in the main list.

**Measured** (provenance probe, Linux, CPU, float32). Re-embedding every
string with the stock `google/embeddinggemma-2`, with no task prefix, mean
pooling, and L2 normalization, reproduces the table:

- Median cosine to its row 0.99999. 99% of rows at 0.99996 or better, and
  229,014 of 229,456 at 0.999 or better.
- On a 64-word sample, every task prefix scores worse. The best was
  `task: search result | query: `, at a mean of 0.90.
- The 30 custom words match at 0.99997 or better, so Haku added words by
  embedding the raw string.

442 rows are below 0.999 and look trained. 142 sit in the head block (cosine
0.88 to 0.97), usually the first id of a duplicate pair whose second id is
the stock row. 300 are frequent function words in the main list, such as
`and`, `the`, and `can` (0.980 to 0.995).

`<rep>` is not an embedding of its string (cosine 0.02). How these rows were
changed is **Unknown**.

### 4.7 Two-stage search

**Verified** (MIL and weights). The scorer program is short:

```
vec [1,500,768] fp16
  matmul with P [768,128]              -> [1,500,128]
  matmul with V [128,229460]           -> [1,500,229460]
  reshape                              -> [1,500,57365,4]
  reduce_max over the last axis        -> [1,500,57365]
  topk k=16, descending, sorted, uint16 indices, cast to int32
                                       -> bidx [1,500,16]
```

Facts about `P` and `V` that the shapes alone do not give:

- **Blocks are word-id order.** Column `c` of `V` equals `P^T` times word
  row `c`, for the first and last 2,000 words (cosine 0.9999994 or better).
  So block `b` holds word ids `4b` to `4b + 3`, neighbors in the alphabetical
  list. The blocks are not semantic clusters.
- **`P` has orthonormal columns.** `P^T P` off-diagonal max 4.4e-5, diagonal
  1 within 6e-5. That is consistent with a PCA basis but does not prove one.
- **`P` is not Matryoshka truncation.** Truncation would put all 128 units of
  energy in the first 128 input dimensions. They hold 32.2, and every one of
  the 768 rows carries between 0.07 and 0.35.
- **Padding is not zero.** Columns 229,457 to 229,459 are three copies of one
  vector (norm 0.436). Reranking must skip ids past the table, or a padding
  id can win.

Stage two runs on the CPU. `Rerank.run(UnsafePointer<Float16>, Int,
MLMultiArray) -> [(Int, Float)]` takes the reader vectors and `bidx`. It
returns one `(word id, similarity)` per frame from the 64 candidates
(**Verified**, symbol). Because `vec` is unit norm, a plain dot product with
the unit rows is the cosine up to FP16 rounding.

### 4.8 Frame decision, collapse, and timestamps

**Verified** (binary). In `EG2Transcriber.back([Float16], [Float], Int)`,
right after the call to `Rerank.run` at 0x100075770:

```
0x1000757d0  fmul  s2, s0, s2         ; scale * cosine
0x1000757d4  fadd  s2, s1, s2         ; + bias
0x1000757d8  ldr   s3, [x19, x9, lsl #2]  ; blank logit for this frame
0x1000757dc  fcmp  s2, s3
0x1000757e0  csinv x11, x11, xzr, hi  ; keep the word id, else -1
0x1000757e4  str   x11, [x24, x9, lsl #3]
```

As a rule:

```
word_logit(t) = 39.85666 * cos(z_t, e_w*) - 0.46262
label(t)      = w*     if word_logit(t) > blank(t)
                blank  otherwise (including any NaN comparison)
```

The scale works as an inverse temperature of about 0.0251. The rule compares
only the best word against blank, so it is greedy and frame-local. A
sequence-level search could find a better path, and nothing here tries one.

**Measured** (Section 8). Collapse merges adjacent equal ids and drops blanks,
so `w, blank, w` yields two words. `<rep>` covers the case CTC cannot
express: the same word twice with no blank between. The reader emits it for
exactly that case, and rendering it as the preceding word is correct.

Each frame is 40 ms, so a word spans `[first * 0.04, (last + 1) * 0.04)`
seconds. Those are spike times, not word boundaries. Spikes land after the
word.

### 4.9 Punctuation and casing (not ported)

**Verified** (MIL). `punct_headonly` takes `vec [1, 96, 768]` and
`mask [1, 96]` (f16). It outputs `punct [1, 96, 4]` (none, comma, period,
question mark) and `case [1, 96, 3]` (lower, title, upper). Its description
reads "Head-only punctuation + case on the EG2 reader's word-table vectors".
So it formats word embeddings and never sees audio.

The network: LayerNorm over 768, a linear to 256 with GELU, four transformer
layers (4 heads of 64, FFN 1024), a final norm, and two linear heads. It holds
about 3.4 M weights. `punct_case.json` maps 23,868 words to title (1) or
upper (2) case. The binary's `TextFormatter` carries `capitaliseAfterStop`
and `iForms` routines.

### 4.10 App integration

**Verified** (symbols). The binary has `PushToTalkKey`, `MicRecorder`, and
an `EG2Transcriber.Stream` with `feed`, `finish`, and `advance(final:)`. It
calls `CGEventPost`, consistent with typing the result into the focused app.
The site describes holding Right Option to dictate.

Compute units come from the callers. `Front(folder:units:readerUnits:...)`
takes `units` and a separate `readerUnits`.
`EG2Transcriber(folder:prefer:units:...)` hardcodes `cpuAndGPU` at one call
site (0x100074648), and `transcribeShared(_:gpu:...)` takes a GPU front. The
app therefore has a GPU path. Which units it picks at run time was not
traced (**Unknown**).

## 5. Compute and memory

Per 20 s window, computed from the program shapes (**Inferred** estimates,
not timings):

| Stage | Multiply-accumulates | Notes |
|---|---:|---|
| Encoder | about 145 G | Linear layers only. Excludes attention scores and the subsampling convolutions |
| Reader | 47.8 G | Includes attention scores over 500 x 500 |
| Scorer, compressed search | 14.7 G | 128-d over 229,460 columns |
| Exhaustive 768-d search | 88.1 G | What the scorer replaces |
| Exact rerank, 64 candidates | 24.6 M | CPU |

The scorer is the cheap stage. The encoder and reader carry most of the
arithmetic, so a faster vocabulary search cannot by itself change the
end-to-end cost much.

| Buffer | Size |
|---|---:|
| Word table, FP16 | 352,445,952 bytes |
| Projected vocabulary in the scorer, FP16 | 58,741,760 bytes |
| Scorer score tensor, 500 x 229,460 FP16 | 229.46 MB (218.8 MiB) logical |
| CPU f32 copy of the scorer columns (TurboSpark reference only) | about 117 MB |

The score tensor is logical. Whether Core ML materializes it across a device
boundary is not observable from the plan, but the plan's cost estimates
point that way (Section 6).

## 6. Device placement

**Measured** (`MLComputePlan`, `coreml_plan --units all`, M4 Max, macOS
26.6.2). A compute plan is Core ML's intended placement before execution, not
a trace of what ran.

| Program | Neural Engine | GPU | No device |
|---|---|---|---|
| Encoder | 994 ops, 61.0% of estimated cost | 701 ops, 28.0% | 1,931 ops |
| Reader | 292 ops, 87.1% | 4 ops, 0.2% | 429 ops |
| Scorer | 2 ops, 26.5% | 4 ops, 73.5% | 16 ops |

Scorer operations one by one:

| Operation | Preferred | Supported | Cost share |
|---|---|---|---:|
| `matmul_0` (768 to 128) | Neural Engine | CPU, GPU, Neural Engine | 0.1% |
| `matmul_1` (128 x 229,460) | Neural Engine | CPU, GPU, Neural Engine | 26.4% |
| `reshape_0` | GPU | CPU, GPU | 42.0% |
| `reduce_max_0` | GPU | CPU, GPU, Neural Engine | 26.2% |
| `topk` | GPU | CPU, GPU | 5.2% |
| `cast` | GPU | CPU, GPU | 0.0% |

Three conclusions follow from the plan:

- The Neural Engine does the heavy matmuls in the reader and the scorer, and
  most of the encoder.
- `reshape` and `topk` list no Neural Engine support at these shapes, so the
  scorer's second half leaves it. The reshape of the 114.7 M-score tensor is
  the largest single cost estimate in the scorer.
- `reduce_max` is supported on the Neural Engine here, but the plan still
  prefers the GPU. An earlier analysis said it was unsupported. That does not
  hold on this host.

The plan cannot show synchronization cost or the actual dispatch path. An
Instruments trace of one inference would. Private framework hooks are
weaker evidence, because a missing hit does not prove a missing path.

## 7. The TurboSpark reimplementation

### 7.1 Split

The portable half lives in `crates/audio/src/stt/embeddinggemma2/`. It is
plain f32 Rust with no Core ML, Metal, or macOS dependency, and it builds off
macOS. The compiled programs run behind two traits:

- `AcousticBackend`: `encode(features, mask)` and `read(hidden, mask)`.
- `BlockShortlister`: `shortlist(vectors, frames)` returning block ids.

The macOS backend is `crates/runtime/src/coreml/embeddinggemma2_ctc.rs`, over
`turbospark-coreml`. It needs macOS 15. At open it checks every program's
input and output names, dtypes, and shapes against the v5 contract, and it
refuses a mismatch rather than decode it. Compute units are set per program
through `TURBOSPARK_EG2_CTC_{ENCODER,READER,SCORER}_UNITS` and default to
`all`.

| Module | Contents |
|---|---|
| `frontend.rs` | Gemma4 log-mel contract, shipped-table loader |
| `windows.rs` | 20 s / 18 s schedule, `[25, 475)` keep range, edge rule |
| `vocabulary.rs` | Word list and FP16 table, kept as raw bits |
| `scorer.rs` | CPU reference of the scorer program, MIL blob parser, rerank, exhaustive search |
| `ctc.rs` | Calibration, decision, collapse, `<rep>`, timestamps |
| `pipeline.rs` | `Transcriber`, tail padding |
| `runtime: coreml/embeddinggemma2_ctc.rs` | Core ML backend and scorer |
| `runtime: examples/eg2_ctc_transcribe.rs` | WAV to transcript, `--cpu-scorer`, `--recall` |

### 7.2 Choices the port makes

| Question | Choice | Basis |
|---|---|---|
| Window edges | First keeps from 0, last keeps through its last valid frame | Matches the binary (Section 4.3) |
| Encoder frames with real audio | `ceil(valid_mel / 4)` | Gemma audio convention. The transcripts are consistent with it. The encoder was not inspected for it |
| Trailing audio | 0.5 s of zeros appended (`tail_padding`, 0 disables) | Spike lag (Section 8.4). The reference's end handling is **Unknown** |
| Collapse key | Word id, not string | The reference stores ids. No duplicate-id alternation seen |
| `<rep>` | Repeat of the preceding word, dropped at the start | Section 8.3 |
| Rerank tie | First candidate in shortlist order | Deterministic |

### 7.3 Tests

- Frontend parity against a fixture from the pinned transformers extractor:
  frame count and mask exact, mel energy within 3e-6 absolute plus 1e-6
  relative, log-mel within 5e-4.
- Asset tests (ignored, need the files): shipped tables against the
  reference contract, word table against list and scorer, calibration.
- Synthetic pipeline tests for window seams and tail padding. The tail test
  fails when the default padding is set to 0 (mutation-checked).
- An ignored runtime test transcribes the reference clip exactly with both
  scorers and checks their frame paths are identical.

## 8. Results

All on the M4 Max above, release builds, every program on `all`.

### 8.1 Transcripts

| Clip | Length | Windows | Errors | Notes |
|---|---:|---:|---|---|
| `qwen3_forced_aligner_reference.wav` | 2.79 s | 1 | 0 of 9 | Exact |
| `say` weather sentence | 6.0 s | 1 | 0 of 24 | |
| `say` paragraph | 47.0 s | 3 | 2 of 133 | `and` read as `an`, `counts` as `count`. No word lost or doubled at a seam |
| `say` repeats | 7.5 s | 1 | 0 of 29 | With tail padding. 1 deletion without it |

That is 2 errors in 195 words. One synthetic voice on clean audio is not a
WER claim, and nothing here should be compared with the developer's 5%.

### 8.2 Shortlist recall

Over 1,527 kept frames of three clips, the 64 candidates held the exhaustive
best word in 1,442 frames (94.4%). On the 189 frames whose exhaustive winner
beats blank, they held it every time. The misses fall on blank frames, where
the word choice does not matter.

The Core ML scorer (FP16) and the CPU reference (f32) made the same decision
on every frame. No decision differed from the exhaustive search. The 6 s and
47 s clips were measured before tail padding existed, and the repeats clip
after.

### 8.3 Repeated words and duplicate ids

On the repeats clip the reader emitted `<rep>` (id 229,456) exactly where a
word repeats with nothing between: "no `<rep>` `<rep>`", "very `<rep>`", and
"that `<rep>`". The last two pairs sat on adjacent frames with no blank, so
plain collapse would have merged them. "walked and walked" used the plain id
both times.

`<rep>` repeats whatever word precedes it. In one run where trailing context
shifted an earlier frame, a lost "that" turned "that `<rep>`" into
"think think".

Across the four transcripts, no head-block id (0 to 319) was emitted.
Duplicated strings (`a`, `with`, `for`, and others) always came from their
main-list id, and no adjacent pair alternated between two ids of one string.

### 8.4 Spike timing and the end of the audio

Every word is a one- or two-frame spike. Against the forced-aligner spans of
the reference clip, spikes land 0.04 to 0.32 s after the word ends.

At the end of a clip, a spike can fall past the last frame of real audio. The
repeats clip lost "night" this way. Its spike lands at 7.52 s, and the audio
ends at 7.498 s.

| Tail padding | Repeats clip | Weather clip | Paragraph |
|---|---|---|---|
| none | 1 of 29 wrong | 0 of 24 | 2 of 133 |
| 0.25 s | 0 | 0 | 2 |
| 0.5 s | 0 | 0 | 2 |
| 1.0 s | 2 of 29 wrong | 0 | 2 |

One second of padding flipped an earlier frame, because the reader attends
over the whole window and trailing context moves earlier outputs. The port
uses 0.5 s. Three clips is a small basis, so treat the value as a starting
point.

### 8.5 Speed, indicative only

One warm run per clip, not a benchmark protocol:

- 47.0 s of audio in 0.38 to 0.46 s, RTFx 103 to 123. The first run after
  load took 0.70 s, because Core ML specializes on first prediction.
- 6.0 s in 0.10 s, and 2.79 s in 0.11 s. Every window costs the same,
  however much of it is real audio.
- First open of the encoder and reader: 11.7 s. Scorer, word table, and list:
  1.9 s.

The developer reports 288x on an 8 GB M3 MacBook Air. These numbers come
from different hardware and audio, so they neither confirm nor contradict
that figure.

## 9. Corrections to the earlier analysis

| Earlier claim | Finding | Evidence |
|---|---|---|
| The encoder is truncated or pruned at layer 11 | All 12 blocks are present. `l11` is the zero-indexed last block | MIL, Section 4.4 |
| The encoder is frozen | The NOTICE says fine-tuned and the site says unchanged. Unresolved | Section 4.4 |
| The reader is a Conformer | Transformer with a depthwise convolution for position. No in-layer convolution module | MIL, Section 4.5 |
| Trained with AdamW and a cosine-margin CTC loss | The sidecar names a learning rate only. The loss is **Unknown** | `reader_P6.json` |
| Blocks are semantic clusters, the projection is Matryoshka or PCA | Blocks are word-id order. The projection is orthonormal, not truncation. PCA remains possible | Section 4.7 |
| Repeated words need a blank between them | `<rep>` handles back-to-back repeats | Section 8.3 |
| The first and last second of audio may be dropped | First window keeps from 0, last keeps through the end | Binary, Section 4.3 |
| `reduce_max` is not Neural Engine supported | Supported here, but the GPU is preferred | Section 6 |
| Everything runs on the Neural Engine | Scorer is 73.5% GPU by estimated cost, encoder 28.0%. The app has a GPU path | Sections 4.10 and 6 |
| The word table is unmodified text embeddings | 442 rows changed, `<rep>` trained | Section 4.6 |

## 10. Developer-reported claims

Each was quoted at its source on 14 October 2026. None was reproduced here.

| Claim | Source | Status |
|---|---|---|
| 288x real time, 1 hour in 12.6 s, base M3 MacBook Air 8 GB | Site | Not reproduced. Different hardware here |
| Word error rate "about 5% right now, against 2-3% for leading speech models" | Site | Not reproduced. No test set named |
| Only the reader is trained, "under two hours on 8xH100, on openly licensed English speech" | Site | Not reproducible without code or data |
| The audio encoder "stays exactly as it is" | Site | Contradicted by the app's own NOTICE |
| Letter-level CTC: 7.0 vs 8.2 WER on clean audiobooks, 16.5 vs 10.8 on a lecture | Hacker News | An earlier model iteration. Two informal points |
| About 20 words of context per word | Hacker News | Consistent with full attention over 20 s |
| New words are added by embedding them, "When a new word is very close to a known one, it tends to fall to it" | Hacker News | Consistent with the 30 custom rows. The compiled scorer has a fixed width, so a new word also needs a shortlist path |
| English only, multilingual reader planned | Hacker News | |
| As a decision model, it "never generalized. Out of the trained domains, it was just above chance" | Hacker News | No evaluation published |
| Source release planned "soon" | Hacker News | Not released as of 14 October 2026 |

## 11. Discussion

**Where the speed comes from.** No autoregressive loop, one fixed-shape pass
per 20 s window, a palettized encoder and reader on the Neural Engine, and a
compressed vocabulary search. The scorer is the smallest compute stage, so
the encoder and reader decide throughput. The system is not small. It ships
715 MB, and the reader alone runs about 48 G multiply-accumulates per window.

**What the embedding vocabulary buys.** The developer's two informal
comparisons suggest a lexical prior that helps on hard audio and costs a
little on clean audio. That is plausible, but it is not established. The
reader is a capable 12-layer transformer, and a learned classifier on the
same features might do as well. Only a controlled comparison can separate the
two (Section 12).

**The closed vocabulary.** A word with no row cannot be emitted. Adding one
means embedding the raw lowercase string with the stock text model and no
prompt, as Haku did. The compiled scorer's `[128, 229460]` shape is fixed,
though, so a new word also needs a second shortlist path. A small CPU search
over custom rows, merged into the 64 candidates before rerank, is the
obvious design.

TurboSpark already runs the FluidInference EmbeddingGemma 2
Core ML text model. Its default task prefix would have to be dropped, and its
FP16 drift measured against these rows, before it could mint vocabulary.

**Semantic versus phonetic neighbors.** The output space was trained for
meaning, not sound. Near-homophones (`there`, `their`, `they're`) and short
contrasts (`can`, `can't`) depend on the reader's context, not on the
geometry. The 300 adjusted function-word rows hint that someone hit this and
moved rows by hand or by training. How they were moved is unknown.

**Greedy decisions.** Each frame compares one candidate with blank. There is
no beam, no lexicon constraint, and no language model. A sequence-level
search over the top few candidates per frame is a cheap experiment that keeps
the non-autoregressive shape. Two of the 30 custom words are `pyctcdecode`
and `kenlm`, a CTC beam decoder and an n-gram language-model toolkit. That
possibly reflects the developer's tooling, and nothing more can be read from
it.

**Throughput is not latency.** With full attention over a 20 s window, a
frame's label can depend on audio that comes after it. The
app's `Stream` feeds samples as they arrive, but how often it re-runs a
window and when a word becomes stable were not traced. Interactive latency
needs its own measurement.

**Timestamps.** Spikes trail words by up to a third of a second here.
Subtitle or editing use needs real alignment. Transcription accuracy and
alignment accuracy are separate properties and should be reported
separately.

## 12. Open questions and next experiments

In rough priority order:

1. **WER on real recordings.** A pinned LibriSpeech test-clean subset first,
   then harder audio, with an existing TurboSpark recognizer as baseline.
   Report the normalization used.
2. **Was the encoder fine-tuned?** Dequantize the palettized encoder weights
   and compare them with the stock `google/embeddinggemma-2` audio tower.
   This settles the NOTICE-versus-site contradiction.
3. **Controlled output-head comparison.** Same layer-11 features, reader,
   data, and budget. Compare the frozen 229K embedding classifier, a learned
   word classifier, and a subword CTC head on WER, out-of-vocabulary
   behavior, robustness, and training cost. Report parameter counts.
   Yang et al.'s Pruned CTC is directly relevant to training against a
   vocabulary this large.
4. **How were `P` and the 442 changed rows made?** Compare `P` with a PCA of
   the word table. Compare the changed rows with their stock embeddings.
5. **Dynamic vocabulary.** The two-path shortlist from Section 11, with
   calibration so custom words do not crowd out common ones.
6. **Scorer without the full score tensor.** A fused block-max and top-k (in
   Metal, or chunked) that never materializes 114.7 M scores. Measure it
   against the Core ML partition.
7. **Streaming latency.** Measure time to a stable word with shorter windows
   and limited lookahead.
8. **Layer choice.** Reader quality from other encoder depths.
9. **Placement trace.** An Instruments capture of one inference, to confirm
   or correct Section 6.

## 13. Limitations

- One machine, one OS build, and one Core ML compiler. Placement and timing
  can differ on other chips and OS versions.
- Four short clips, three of them one synthetic voice. No noise, accents,
  overlap, numbers, or names were tested.
- The training procedure is unobservable from inference artifacts. Every
  statement about training is reported or unknown.
- The app binary was read statically at selected addresses. Run-time
  behavior, such as compute-unit choices and streaming cadence, was not
  traced.
- Compute figures in Section 5 are estimates from shapes, not measurements.

## Appendix A. Reproduction

Assets (downloaded separately, never committed):

```sh
# Fetch every file in the v5 manifest with curl (the host refuses Python's
# default user agent), verify each sha256, and lay them out at their
# manifest paths, for example under ~/models/eg2demo.
```

Tests and the transcriber:

```sh
cargo test -p turbospark-audio --lib embeddinggemma2
TURBOSPARK_EG2_ASSET_DIR=$HOME/models/eg2demo \
  cargo test --release -p turbospark-audio --lib embeddinggemma2 -- --include-ignored
cargo test -p turbospark-runtime --lib embeddinggemma2_ctc
TURBOSPARK_EG2_ASSET_DIR=$HOME/models/eg2demo \
  cargo test --release -p turbospark-runtime --lib embeddinggemma2_ctc -- --include-ignored
cargo run --release -p turbospark-runtime --example eg2_ctc_transcribe -- \
  ~/models/eg2demo clip.wav [--cpu-scorer] [--recall]
cargo run --release -p turbospark-coreml --example coreml_plan -- \
  ~/models/eg2demo/scorer_blk_d128_k16_b4_t500.mlmodelc --units all --ops
```

Synthetic clips:

```sh
say -v Samantha -o clip.wav --data-format=LEF32@16000 -f clip.txt
```

The family README,
`crates/audio/src/stt/embeddinggemma2/README.md`, carries the per-file
hashes and the gate table.

## Appendix B. Key hashes

| File | sha256 |
|---|---|
| `word_unit_f16.bin` | `36edb3cae02101c49ad3f3289ce5c526b1698916eb6b9faf116f4df227a25c9b` |
| `reader_words.json` | `d651e2ec1ebd55de2d31c70f64803894580eb6f5e85239edcd0bbec99c918b53` |
| `reader_P6.json` | `7a43e6557e6a202b59761ea0720fad9b807796e24776b2b331b0f03d67a6f700` |
| `scorer_blk_d128_k16_b4_t500.mlmodelc/weights/weight.bin` | `823e7cace9dfc1d8c2c6575dc802814f15879c2b124b9da6693e5cf1f6f7034a` |
| `eg2_audio_l11.mlmodelc/weights/weight.bin` | `1781b3491deef798302ff0594cf89af84512dca0baaee093f71dada0985f0aab` |
| `reader_P6.mlmodelc/weights/weight.bin` | `b0697e6d084048918dc2ddde822da156e1b2400ff585bc63d4929d8e9721adbf` |
| `EG2Transcribe` (app 1.6) | `7bcdc0085df93f5e5538e54e943d82c9701172361c56ec6a4241e325fa182348` |

## References

1. Haku, "Embedding Gemma Transcribe," https://eg2.haku.sh/, accessed
   14 October 2026.
2. Haku, eg2demo model manifest, version 5,
   https://haku-models.tslfiles.org/eg2demo/v5/manifest.json, created
   12 October 2026.
3. "We turned EmbeddingGemma 2 into a speech-to-text model," Hacker News,
   item 50069971, 13 October 2026. Developer comments by `Shershah1024`,
   items 50071209, 50071451, 50071483, 50071579, 50075470, 50075535, and
   50075704.
4. Google DeepMind, `google/embeddinggemma-2`, Hugging Face, revision
   `914f7f89142e33e77833254d9c9b90c3cef7303b`.
5. Hugging Face transformers, `Gemma4AudioFeatureExtractor`, commit
   `28034545c35acac05793695bd5c878f66e6b544e`.
6. Y. Yang et al., "Pruned CTC for Memory-Efficient Large-Vocabulary ASR
   Training," arXiv:2609.33645, September 2026.
