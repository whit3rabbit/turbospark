# Kokoro TTS (82M)

Kokoro uses PLBert, bidirectional LSTMs, duration alignment, a prosody predictor,
and an iSTFTNet vocoder to return 24000 Hz mono f32 PCM. The checked frontend
supports en-US text and af_heart. Frontend provenance and corpus gates are in
[frontend/resources/provenance.json](frontend/resources/provenance.json).

The numerical reference is mlx-audio 0.5.7 at
[`e1b19b9054bf163f5d812221a54fcc346f1890e9`](https://github.com/Blaizzy/mlx-audio/tree/e1b19b9054bf163f5d812221a54fcc346f1890e9/mlx_audio/tts/models/kokoro),
with MLX 0.31.2. The installed checkpoint gate uses
[mlx-community/Kokoro-82M-bf16 at a71e4d38](https://huggingface.co/mlx-community/Kokoro-82M-bf16/tree/a71e4d38b236d968966a2002c4c895dbd12b1c3c).
Its actual 548 tensors are F32. The Metal loader checks descriptors and refuses
other dtypes; the repository alias does not select precision.

## Execution contract

`Kokoro::open` and `KokoroSynthesizer::open` retain portable numerical execution.
`open_with_backend` uses the shared safe audio backend for resident embeddings,
linear projections, attention, ordinary/grouped/transpose convolutions,
normalization, erf-GELU, and LSTM recurrence. The runtime `KokoroRunner` supplies real Metal operators and
remains on the existing native audio worker, with Rc state owned there.

BERT materializes QK scores before scaling and softmax. Its three LayerNorm
sites use the device path. Weight normalization reduces in sanitized
`[channel,kernel,input]` order before division and gain multiplication, then
retains the checkpoint layout. The erf-GELU path follows pinned compiled MLX
arithmetic, including its seven-significant-digit F32 constant printing.
Matched-input precision fixtures require bitwise agreement on these real inputs.

Kokoro F32 GEMM uses the pinned shape-dependent split-K selection, with the
2048-tile threshold on Max/Ultra, 16-element partition boundaries, and the final
partition carrying any remainder. Style projections use contiguous four-product
GEMV lane accumulation. LayerNorm compiles through the precise Kokoro source.
AdaLayerNorm explicitly uses row reductions. Predictor AdaIN preserves the
column-reduced mean before squared differences materialize for row variance;
wide column reductions use separate partial and final passes. Norm output divides
by sqrt, and the residual divides by sqrt(2). Music3 keeps its compiler and
operator selection.

Recurrent gates use pinned GEMV order, stable sigmoid, precise tanh, and separate
cell products. The pinned ordinary convolution channel counts use 16-channel
chunks with taps visited inside each chunk. Depthwise transpose visits ascending
taps through full F32 MMA. Source sine, tanh, and normal transformation use the
precise Metal source without changing random draws or key advancement. Input
lengths, finite values, dtype, grouping, recurrence width, and reduction indexing
are checked before encoding.

Exact convolution arithmetic is established for the pinned checkpoint's channel
counts and taps. Optional 17-channel ordinary and 11-tap depthwise probes did not
establish exact MLX agreement; MLX routes these through different explicit-GEMM
paths. Those routes have no exact-parity claim or readiness promotion. The
existing broader synthetic convolution bounds remain separate from the strict
pinned arithmetic fixtures.

Requests reset the MLX key sequence once and retain its stream across segments.
SineGen draws initial uniform phases, a full sample/harmonic normal tensor, and
an unused noise-branch draw. Duration rounding uses ties to even. The generator
noise convolution floors `(stride + 1) / 2`, and the pinned helper named
`ReflectionPad1d` actually pads with zeros. These conventions apply to both
portable and device execution.

The Metal source scan follows the physically contiguous four-read scan in
[MLX 0.31.2](https://github.com/ml-explore/mlx/blob/v0.31.2/mlx/backend/metal/kernels/scan.h).
The FFT20/hop5 operator preserves paired-real FFT arithmetic and signed zeros
before phase extraction. Generic portable STFT and Whisper retain their existing
DSP contracts. A diagnostic vocoder replay separates identical reference F0
inputs from identical reference harmonic-source inputs.

Cancellation is checked before every segment. The worker serializes commands,
bounds PCM/progress delivery, and retains its heavy-work permit until inference
stops. Memory admission includes the unloaded device weights and a conservative
activation reserve derived from loaded geometry, longest checked segment, and
speed. These are estimates, not measured peak-memory qualification.

## Reproduce the component gates

Tests never download checkpoints or regenerate fixtures. With the pinned
mlx-audio clone on PYTHONPATH and a Python environment containing MLX 0.31.2:

```sh
python crates/gpu/tests/reference/generate_kokoro.py \
  crates/gpu/tests/fixtures/kokoro_mlx_0_31_2.json \
  crates/audio/tests/fixtures/kokoro_rng_mlx_0_31_2.json
python crates/gpu/tests/reference/generate_kokoro_precision.py \
  "$KOKORO_CHECKPOINT" \
  crates/gpu/tests/fixtures/kokoro_precision_mlx_0_31_2.json
python crates/gpu/tests/reference/generate_kokoro_contracts.py \
  crates/gpu/tests/fixtures/kokoro_contracts_mlx_0_31_2.json
python crates/runtime/tests/reference/generate_kokoro.py \
  "$KOKORO_CHECKPOINT" "$KOKORO_REFERENCE_OUTPUT"
cargo test -p turbospark-audio --lib tts::kokoro -- --nocapture
cargo test -p turbospark-gpu --test kokoro_parity -- --nocapture
cargo test -p turbospark-gpu --test kokoro_contracts -- --nocapture
TURBOSPARK_KOKORO_DIR="$KOKORO_CHECKPOINT" \
TURBOSPARK_KOKORO_REFERENCE="$KOKORO_REFERENCE_OUTPUT/reference.json" \
  cargo test -p turbospark-runtime --test kokoro_metal \
  -- --ignored --test-threads=1 --nocapture
```

Real Metal operator parity, exact checkpoint tensor/waveform parity, worker
control, application smoke, output quality, performance, and release readiness
are separate gates. Fixture or checkpoint tests do not qualify the catalog,
ABI, Swift application, or all supported request shapes.
