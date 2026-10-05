# Whisper on Metal: performance story and porting notes

Whisper-class transcription runs natively in Rust with the whole stack on
Metal by default. This page records how the Metal path got fast, how it
was profiled, what the whisper.cpp and MLX checkpoint comparisons
actually were, and what generalizes to the next speech or audio model
port. The functional contract, formats, and install layout live in
[speech-to-text](SPEECH_TO_TEXT.md); the audio DSP layer in
[audio](AUDIO.md). Nothing here is a frozen benchmark row; speed and
quality claims still owe the hardware gates in the benchmark docs.

## Measured standing

Single machine (Apple M4 Max, 32 GPU cores), 20.1 second `say`-generated
16 kHz mono clip, whisper-tiny.en, quiet desktop, warm runs interleaved
A/B against whisper.cpp on the same clip:

| Path | Total | Notes |
| --- | --- | --- |
| CPU reference (pre-Metal baseline) | 245 s | `compute::whisper` kernels, 0.08x realtime |
| TurboSpark Metal, cold first run | 231 ms | includes per-process pipeline compile |
| TurboSpark Metal, warm | 160-162 ms | mel 32 + encode 46 + cross 4 + decode 113 ms |
| mlx-community whisper-tiny.en-8bit, warm | 164 ms | same device path, dequantized at load |
| whisper.cpp (Homebrew, Metal, `-t 4`), cold | 275 ms | its own first run reads 224 ms |
| whisper.cpp, warm | 178-183 ms | |

All four engines produce the same text on this clip. Segment structure
differs by design: whisper.cpp emits timestamped sub-segments; this port
emits one segment per 30-second window with a one-second carry.

Warm phase split for the 160 ms: mel frontend 32 ms (CPU), encoder 46 ms
(100 dispatches), cross caches 4 ms, decode loop 113 ms (59 tokens,
1.6 ms per fused step pass including the logits readback, plus 0.23 ms
CPU argmax per token).

Reproduce:

```sh
cargo run --release -p turbospark-runtime --example whisper_transcribe \
    -- <model_dir> <wav>
/opt/homebrew/bin/whisper-cli -m ggml-tiny.en.bin -f clip.wav -t 4
```

## How it got fast

The first working Metal port read 1.65 s end to end. Three findings
account for most of the distance to 160 ms, in the order they were
found.

**1. Release builds, then one fused pass per token.** The first profile
was taken in a debug build and read mel 1164 ms; the same code in
release reads 32 ms. Debug numbers are 30x off on the FFT-heavy
frontend and tell you nothing. With that corrected, the decode loop
showed 5.3 ms per token across two command passes per token (one for
the layer stack, one for the final norm and logits projection). Merging
them into one pass that ends in the logits projection, and reading the
logits back once, cut a full CPU-GPU round trip per token.

**2. Decode loops want GEMVs and packed weights, not GEMM tiles.** The
tiled 32x32 GEMM is the right encoder kernel and the wrong decode
kernel: at m = 1 its chunk barriers and staging dominate, and three
separate q/k/v projections triple the dispatch count. The decode path
now uses a warp-per-row GEMV (one simdgroup per output row, lanes
striding the reduction, no staging, no threadgroup barriers) with the
bias and the residual add fused into the epilogue, and the three
self-attention projections packed into one `[3d, d]` weight so one GEMV
serves q, k, and v. The decoder step runs 8 dispatches per layer; the
layer norm feeding each one is a separate dispatch.

**3. Memory layout beat kernel cleverness.** Per-kernel ranking put
77% of the decode step in the cross-attention step. The cause was the
weighted value sum: with V stored `[key, dim]`, every read strides
`d_model` floats. Transposing the cross-V cache to `[dim, key]` once
per window (one transpose dispatch at cross-cache build) makes each
lane's inner loop contiguous and cut the step from 2.9 to 1.6 ms. The
same kernel then still wasted three quarters of its lanes in the value
phase (head_dim 64 vs a 256-thread group), so lanes now also split the
key range into blocks and reduce partials in shared memory. Decode
landed at 113 ms.

Two things that did not matter, recorded so nobody re-derives them:

- Memory barriers between dispatches in one pass. The first conv wrote
  zeros and the fix that worked also added barriers, so the barrier got
  the credit; the actual bug was every buffer bound to argument slot 0
  (below). The `begin_pass` contract guarantees serial dispatch
  visibility for directly-bound buffers, and removing all within-pass
  barriers changed nothing on the clock while making logits noise drop
  from 1.3e-2 to 3.4e-5 (dispatch order is deterministic). Explicit
  `memory_barrier_with_buffers` is for indirect access (the MoE
  argument-blob path), not ordinary buffer chains.
- The logits projection looks like the biggest single kernel (80 MB of
  embedding read per token, ~0.26 ms) but skipping it in a controlled
  experiment moved the step pass only 2.5 to 2.2 ms. Per-token memory
  across all decode kernels is ~88 MB; nothing in the step is compute
  bound.

Known remaining inefficiencies, measured not guessed:

- The mel frontend costs ~30 ms. Whisper's 400-point frames are not a
  power of two, so they ride Bluestein's chirp-z: three length-512
  transforms per frame. Do not "fix" this by zero-padding frames to
  512; that changes the bin frequencies the mel filterbank is defined
  on.
- The encoder's tiled f32 GEMM runs ~250 GFLOPS where a
  simdgroup-tiled kernel would do several times better. Its scores and
  value-mix projections are 12 of the 100 encoder dispatches per
  window.
- The conv front end is a naive one-thread-per-output kernel at
  ~10 ms.

## How it was profiled

Three instruments, in increasing order of resolution:

- `TURBOSPARK_WHISPER_PROFILE=1` prints per-window phase timings (mel,
  encode, cross, decode with token count), per-step pass wall and
  GPU-busy time, and the per-token decode split. Adding
  `TURBOSPARK_DISPATCH_PROFILE=1` also prints the gpu crate's
  per-kernel ranking at the end of the run. That ranking is what found
  the 77% cross-attention number. Read the profiler's own warning: it
  encodes one compute encoder per dispatch, so absolute times are
  inflated and only the RANKING is meaningful.
- `commit_and_wait_with_gpu_time` splits wall from GPU-busy per pass.
  Wall near GPU-busy means the GPU timeline is the cost; wall far above
  it means sync or CPU overhead. A related split times the CPU encode
  loop alone (the step pass encodes in ~0.12 ms; it was never the
  bottleneck).
- Attribution experiments. Skip one kernel behind an env flag and
  re-measure; that is how the logits GEMV was sized at ~0.26 ms. Or
  time N repetitions of one tiny kernel in a single pass to get the
  per-dispatch floor: ~5.4 us GPU / ~7 us wall at 100-200 dispatches
  per pass on this machine. That floor is how a 2.4 ms step pass over
  ~50 dispatches was proven to be kernel cost, not dispatch overhead.

Contamination discipline, learned the hard way. A concurrent
`cargo test --workspace` from another session on the same GPU inflated
encode 50 to 600 ms and steps by 40%. Every number quoted here was
taken after checking `ps aux | grep cargo` for other GPU users. The
first run after a build is a cold GPU (DVFS) and is never a baseline:
discard it and interleave A/B pairs. Debug builds are 30x off on the
mel frontend and are never quoted.

## Conversion notes: whisper.cpp, openai, and MLX

Nothing was ported FROM whisper.cpp's code; it has no Rust surface and
ggml is a different execution model. It served as two oracles: a
behavioral one (same text on the same clip, which is the acceptance
bar for greedy decoding) and a speed one (its `whisper_print_timings`
phase split calibrates what "fast" means; its warm decode is ~0.26 ms
per token, which is what the fused single-pass step was built toward).
The model semantics came from openai's reference and the mlx-examples
whisper implementation: Pre-LN encoder, the SOT grammar prompt
(language, transcribe, no-timestamps), greedy argmax with the
special-token suppression mask, and 30-second windows zero-padded so
the encoder sequence length is a constant (all scratch allocated at
open, no per-window allocation).

The mlx-community conversions are a first-class input format, not an
afterthought. Linear tensors pack as U32 words with per-group F16
scales and biases (8-bit and 4-bit, group 64; each dequant formula was
verified element-wise against the fp32 checkpoint before
implementation). Conv weights ship `[out, 3, in]` and transpose at
load. The encoder sinusoidal table is generated at runtime with the
reference formula, and tokenizer.json is paired from the corresponding
openai repo. Details and the verified error bounds are in
[speech-to-text](SPEECH_TO_TEXT.md). The device path itself is
dtype-uniform f32: quantization is a loader concern, dequant happens
once at open, and the GPU never sees a packed tensor. whisper.cpp runs
fp16 weights; this path runs f32 and still matches its warm clock on
this class of model, so dtype was not the lever (dispatch count and
memory layout were).

Numerics between the CPU reference and Metal are f32 end to end on both
sides, which is what keeps parity a tolerance question instead of a
numerics redesign: encoder hidden states agree to 6.4e-4 max on the
real model, top-8 logits are token-identical, and the GEMV's simd-tree
reduction (a different f32 summation order from the reference's
sequential loop) is bounded and documented in the synthetic test rather
than papered over.

## Porting checklist for the next speech/audio model

Moonshine Tiny is the first application of this checklist. Its opt-in Metal
runner reuses the pass, GEMV, cache, and scratch patterns, and its real
checkpoint encoder and transcript checks pass for one clip. Its current
peak footprint exceeds the CPU reference, so it is not a default backend.
Qwen3-ASR, Parakeet/Nemotron, and Granite CPU memory work and remaining
Metal gates are tracked in [Speech to text](SPEECH_TO_TEXT.md).

The reusable sequence, in the order that catches bug classes cheapest:

1. CPU reference kernels first, with independent fixtures. They are the
   parity partner, the portable fallback, and the thing that makes GPU
   bugs diagnosable by bisection instead of by staring.
2. Per-kernel Metal parity tests on real hardware against those
   references, at real shapes and at small hand-check shapes. Every
   structural bug in this port (wrong slot, wrong stride, missing
   scale) measured orders of magnitude above the f32 noise floor.
3. A synthetic composition test: deterministic pseudo-random weights,
   the full stack on both devices, compared per step. This is what
   caught the decoder residual-base bug and the missing transpose, both
   invisible to per-kernel tests.
4. Real-model checks gated on an installed witness: encoder hidden
   state bound, top-k logits token-identical, then end-to-end transcript
   equality. Greedy tie-flips from f32 noise are legitimate; anything
   else is a bug.
5. Only then benchmark, warm, interleaved, on a quiet machine.

Mechanical rules the hard way:

- Buffer bindings must name their argument slot. `F32View::binding()`
  originally reported slot 0 unconditionally; a multi-argument dispatch
  with every view on slot 0 presented first as an all-zero output and
  then, on the GEMV, as an out-of-bounds read that stalled the GPU for
  33 minutes. The API now requires the index at each call site.
- Dispatches within one pass see each other's writes; do not add
  barriers between directly-bound buffer chains, and do not take a
  barrier's presence as proof a visibility bug existed.
- Decode loops: count dispatches before optimizing kernels. Fuse the
  pass (one commit per token), pack small projections, use the GEMV not
  the GEMM at m = 1, and make strided reads contiguous by transposing
  once at setup instead of cleverly indexing in the hot loop.
- Keep the engine's context in a `RefCell` inside a `Mutex` held by the
  runner: the FFI contract hands out `&self` through an `Arc`, per-
  dispatch borrows stay short, and one transcription runs at a time.
- Gate every Metal encode loop in `gpu::autorelease_pool` (workspace
  invariant; unbounded autoreleased command objects otherwise).
- Env knobs: `TURBOSPARK_WHISPER_DEVICE=cpu` forces the reference path;
  `TURBOSPARK_WHISPER_PROFILE=1` and `TURBOSPARK_DISPATCH_PROFILE=1`
  are the profiling pair (see [ENV](ENV.md)).
