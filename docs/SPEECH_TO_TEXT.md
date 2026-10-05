# Speech to text

Whisper-class transcription runs natively in Rust: PCM in, ordered segments
with second offsets out. No external runtime, no network after the model
downloads.

## Status

- Family: `SpeechFamily::Whisper` (wire string `whisper`), separate from the
  decoder `ModelFamily`.
- Execution: Metal by default on macOS (f32 kernels end to end: mel
  frontend on CPU, conv front end, Pre-LN encoder stack, cross caches,
  and the greedy decoder with the SOT grammar all on the GPU), with the
  CPU reference path as the fallback when no Metal device initializes or
  `TURBOSPARK_WHISPER_DEVICE=cpu` is set. The CPU kernels in
  `compute::whisper` remain the parity partner and the portable path.
- Witness: `openai/whisper-tiny.en` (English-only, 80 mel bands,
  vocab 51864). Multilingual models are supported through the same code
  paths; the prompt gains the language and task tokens and language
  auto-detection scores the first window.

## Other STT model work

- Moonshine Tiny has a portable CPU reference in `turbospark-audio` and an
  opt-in Metal runner in `turbospark-runtime`. The Metal path uses request
  sized scratch, precomputed cross-attention K/V, packed self-attention QKV,
  GEMV for one-token projections, and one command pass per token. The local
  pinned-checkpoint test compares encoder activations and a transcript with
  the CPU path. See the [Moonshine profile](../crates/audio/src/stt/moonshine/README.md).
- Qwen3-ASR now prefills its CPU decoder once and appends per-layer K/V for
  each generated token. A grouped-query synthetic test compares it with the
  full-prefix reference. Packed 8-bit Metal weights and real-checkpoint gates
  remain open.
- The shared FastConformer attention used by Parakeet and Sortformer now
  computes one score row at a time instead of retaining quadratic attention
  matrices. Nemotron's limited-context attention computes only its allowed
  chunk window. Both changes have focused reference tests. Their Metal
  encoder and predictor/joint paths remain open.
- Granite Speech 5 CTC now processes midpoint probabilities and final
  logits in bounded row batches. Midpoint self-conditioning still computes
  the same full-vocabulary softmax and projection for each row. The GPU
  encoder and real-checkpoint gates remain open.

These families are not connected to `ts_stt_*`, install receipts, the speech
catalog, or Swift selection yet. Moonshine Metal remains opt-in because its
observed peak footprint is above the CPU reference on the available clip;
the broader quality and paired performance gates are also open. A model
without alignment must emit one clip-level segment when it reaches the
session API, with no word-level timing claim.

## Evidence recorded from real runs

These numbers come from real runs on this checkout, not projections:

- End-to-end transcription of a 5.91 second `say`-generated clip
  (16 kHz mono WAV) matches the reference openai-whisper Python package's
  text output on the same audio; differences are greedy-decode
  punctuation/casing variants.
- Encoder hidden states match the reference encoder element-wise on the
  same mel input: max absolute difference 7e-5, mean 3e-6 (f32 noise).
- The mel frontend matches a numpy reference built from an independent
  FFT and filterbank implementation; the golden fixture is pinned in
  `crates/audio/src/whisper.rs`.
- The full FFI path (`ts_stt_open` -> stream open/append/finish) produces
  byte-identical segment JSON across two runs of the same audio
  (`crates/ffi/tests/c_surface.rs`, gated on `TS_STT_TEST_MODEL`).
- CPU decode throughput is roughly real-time-per-window on the witness
  machine (a 5.9 second clip decodes in about 6.5 seconds). This is a
  single-machine observation, NOT a frozen benchmark row; a quality or
  speed claim still requires the benchmark-doc gates.
- On the Metal path the same witness clip class runs far below real
  time; the observed numbers remain single-machine observations under
  the same gate, and the WER/speed benchmark rows of spec task 13.3 are
  still owed before any published claim.

## Usage

- CLI inspection: `turbospark-model list-speech` shows the bundled pinned
  entries (tiny.en, base.en, small.en, multilingual base).
- FFI: `ts_stt_open` opens a speech install directory;
  `ts_stt_stream_open` / `ts_stt_stream_append` / `ts_stt_stream_finish`
  carry bounded PCM chunks (at most two seconds of base64 f32 LE 16 kHz
  mono per append; the stream buffers prior audio, callers never resend);
  `ts_stt_stream_cancel` discards; `ts_stt_close` closes the resident
  model.
- Runtime API: `turbospark_runtime::WhisperRunner::open(dir)` then
  `transcribe(&pcm, language)`, where `language` is `None`/`"auto"` or a
  code such as `"en"`.

The Slaney frontend review corrected area normalization from mel units to
Hz, and now pins OpenAI Whisper v20250625's exported filterbank as the
independent oracle. It also drops the final centered frame before the
global peak clamp, matching the reference. The historical real-model
transcript and encoder witnesses above, and the MLX witnesses below, need
rerunning against this corrected frontend. They do not qualify this
checkout's transcription quality.

## MLX conversions (mlx-community)

The `mlx-community/whisper-*` conversions are a first-class format, not a
converter afterthought:

- **Config**: openai's original field names (`n_audio_state`, `n_vocab`,
  `n_mels`, ...) resolve through the same parser as the HF names; the FFN
  widths openai omits derive as 4x the stream width.
- **Weights**: linear tensors pack as U32 words with per-group F16 scales
  and biases (group 64, 8-bit); the dequant formula
  `w = q * scale + bias` over little-endian-unpacked bytes was verified
  element-wise against the fp32 checkpoint (mean abs error 1.5e-4) before
  implementation. Conv weights ship F16 `[out, 3, in]` and transpose at
  load; the token embedding is quantized like any other linear.
- **Positional table**: MLX conversions omit the encoder's sinusoidal
  table (mlx-examples computes it at runtime); the runner generates it
  with the reference formula, verified element-wise against the stored
  checkpoint table (max abs diff 2.4e-4). The generator's scaling was
  itself a caught bug: max_timescale folds into the log increment alone,
  and multiplying it twice decoded as fluent garbage.
- **Tokenizer**: MLX repos ship no `tokenizer.json`; registry entries
  pair one from the corresponding `openai/whisper-*` repo
  (`tokenizer_source`) and an install fetches it alongside the weights.

4-bit conversions decode with the same affine formula, low nibble of each
byte preceding the high nibble, group 64 -- verified element-wise against
the fp32 checkpoint (mean abs error 2.4e-3, which is 4-bit quantization
noise; the alternative orders/formulas measured 0.018+) before
implementation.

Verified on real runs through the Rust path (same `say`-generated clip as
the fp32 witness): `whisper-tiny.en-8bit`, `whisper-base.en-8bit`,
`whisper-tiny.en-4bit`, and `whisper-base.en-4bit` all transcribe the
reference text (4-bit differs only in casing variants -- quantization
noise, not pipeline error). fp16 `*-mlx` conversions ship `weights.npz`,
which this runtime does not read.

## Registry

`turbospark-model list-speech` prints the supported table: nine pinned
entries, four `hf-safetensors` (tiny.en, base.en, small.en, multilingual
base), three `mlx-8bit` (tiny.en, base.en, multilingual tiny) and two
`mlx-4bit` (tiny.en, base.en), each with a 40-hex revision, an exact file
list, and -- for MLX -- the tokenizer pairing. The catalog tests pin every row's shape so a drifted
entry fails a test, not an install.

## Metal path

`crates/runtime/src/whisper/metal.rs` executes the whole stack on Metal
through `crates/gpu`'s f32 kernels (`shaders/whisper_encoder.metal` plus
the front-end conv): a 32x32-tiled GEMM for the encoder's dense
projections, a per-head score GEMM with the `head_dim^-0.5` scale riding
the reduction, a warp-per-row GEMV with fused bias/residual for every
`m == 1` projection in the decode loop (qkv packed as one `[3d, d]`
weight), row softmax, exact-erf GELU, LayerNorm, and a single-query
attention step over the K/V caches. Every kernel is parity-tested on
real hardware against the CPU reference (`tests/whisper_encoder_parity.rs`,
`tests/whisper_conv_parity.rs`), and the runtime tests pin cross-device
agreement: the encoder hidden states to f32 noise (6.4e-4 max on
tiny.en), the top-8 decoded logits token-for-token, and the synthetic
decoder stream per step. The GEMV path reduces each dot as a 32-lane
simd tree, a different f32 summation order from the reference's
sequential loop; the measured noise (1.3e-2 on logits near 10) is
recorded in the synthetic test's bound.

Two dispatch rules this path paid for, both structurally prevented now:
directly bound buffers are visible to later dispatches in the same pass,
without an explicit barrier (the conv front end's zero output came from
incorrect argument slots, not missing barriers), and buffer bindings must
name their argument slot. `F32View::binding(index)` takes the index as a
required parameter after an every-view-on-slot-0 bug presented first as
an all-zero front end and then as a 33-minute GPU stall in the GEMV.
Indirect buffer access, such as the MoE argument-blob path, has a separate
barrier requirement.

`TURBOSPARK_WHISPER_PROFILE=1` prints per-window phase timings (mel,
encode, cross caches, decode with its token count) and per-step pass
timings to stderr; `TURBOSPARK_DISPATCH_PROFILE=1` adds the gpu crate's
per-kernel ranking (the runtime prints it under the whisper profile).
Attribution only; these are not benchmark rows. The performance story,
profiling method, and the porting checklist this path produced live in
[WHISPER](WHISPER.md).

Single-machine observation from bringing this up (M4 Max, 20.1 s
`say`-generated clip, whisper-tiny.en HF and mlx-8bit): end-to-end
transcription runs at roughly 160-165 ms warm against Homebrew
whisper.cpp's ~180 ms on the same clip and machine, and both produce the
same text; the CPU reference path on the same checkout takes minutes.
These observations drove the implementation and are NOT frozen benchmark
rows; the WER/speed hardware gates of spec task 13.3 are still owed
before any published claim. Two measured inefficiencies that remain,
recorded so nobody re-derives them as surprises: the mel frontend costs
~30 ms of it (the reference's 400-point frames ride Bluestein's chirp-z
because 400 and 200 are not powers of two; whisper.cpp's smaller figure
comes from a different transform strategy), and the encoder's naive
tiled f32 GEMM reads ~250 GFLOPS where a simdgroup-tiled kernel would
do several times better.

## Install layout

A speech install is a directory containing exactly:

- `config.json` - the whisper config; validated against the shapes the
  kernels implement (n_mels 80 or 128, exact-erf gelu, divisibility),
  refused otherwise with the reason.
- `model.safetensors` - the weights; every tensor the kernels consume is
  name-checked and shape-checked at open.
- `tokenizer.json` - the tokenizer; the special tokens (SOT grammar) are
  read from it, never hardcoded.
- `speech-receipt.json` - written at install time: per-file size and
  SHA-256, the speech family, the mel width, and the config-derived
  estimated allocations for the CPU runner (f32 weights, caches, scratch,
  and loading margin). This estimate is not a measured process-memory ceiling.

The catalog refuses GGML/GGUF speech distributions and foreign layouts by
name before anything is parsed.

## Storage locality and egress

- Audio capture, buffering, transcription, and transcript storage never
  contact the network. The only egress in the speech path is the model
  download itself, which rides the shared install pipeline.
- The FFI streams buffer PCM inside the process; nothing is written to
  disk by the transcription path.
- The Swift app's storage surface (recordings, transcripts) is the
  SQLCipher profile database per the speech-to-text design; the Rust side
  owns none of it.

## Not in this pass

- Streaming partials with the revised tail window: `append` reports the
  buffered extent; decode happens at `finish`. The stable-prefix
  machinery is where the Metal path plugs in.
- The Swift speech subsystem (capture, gallery, dictation) - the approved
  spec's remaining tasks.
- WER and speed benchmark rows - require the real-hardware model gates
  before any claim.
