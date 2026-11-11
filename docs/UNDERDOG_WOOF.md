# Underdog Woof (text-to-speech): port plan and first slice

Status: first slice only. The checkpoint reader, the dequantized speech-stage
weights and the 60-tensor hash gate exist and pass against the real file. There
is no forward pass, decoder, text normalizer, RNG, catalog entry, FFI,
server route, CLI command or Swift surface yet. Nothing here is a speed or
quality claim about a TurboSpark implementation.

Underdog Woof is a small English TTS model with two fixed voices (Juno, Mira)
by Conway. Upstream: <https://huggingface.co/ConwayResearch/Underdog-Woof>,
revision `be702960539777fc3f698e54335276a8391b00f9`, a single 40,595,426 byte
`woof_int8.npz` (SHA-256
`d895fbbe84b8f4f461e3e94ab48b1164353b1988ce8d365f5aa8263d0de1203e`) plus a
NumPy-only reference runtime. The measurements below come from a review of the
`cowWhySo/Underdog-Woof-MLX` repo (an MLX and Core ML qualification suite built
on that revision), done 2026-11-11 on an M4 Max with other work on the host.
They are indicative, not frozen `docs/BENCHMARKS.md` rows.

## License and consent: read before shipping anything

The Underdog Non-Commercial License 1.0 covers the weights, the code, the voices
and the documentation, and every converted version.

- **Non-commercial use only.** Personal projects, research, teaching and
  non-profits. Commercial use (including use inside a company, a paid or
  ad-supported product, or a hosted API) needs a written license from Underdog,
  license@conway.tech.
- **Credit.** Show "Voice by Underdog Woof", linked to underdog.ai, in any app
  that uses it. Put "Underdog Woof" or "underdog.ai" in the title or description
  of anything published with its audio. Model cards and listings say the work is
  derived from Underdog Woof and link the license.
- **No training.** Do not use the model or its audio to train, fine-tune or
  distill another speech or audio model. TurboSpark must not offer Woof audio as
  a corpus for that, and Meeting Notes or Audio Workspace features must not feed
  it back into training data.
- **Responsible use.** No impersonation, no deceiving people about hearing a
  machine, no voice cloning without consent. Woof has no cloning path and the
  surface must not grow one.
- **Derivatives stay under this license.** Section 1 defines a Derivative as any
  modified version of the Model, including converted versions. A Rust
  reimplementation that loads these weights is plausibly a Derivative. That sits
  awkwardly with a project that has its own license, and it is not something to
  settle by reading the text alone. **Decision for the maintainer: ask Conway
  (license@conway.tech) before this ships, not after.** Until then keep the work
  in the tree, off by default, and out of the release catalog.

Rules for the surface, following the MegaTTS3 consent precedent
(`crates/audio/AGENTS.md`, `crates/server/src/audio/speech.rs`):

1. **Fetch from upstream, never mirror.** The catalog entry points at
   `ConwayResearch/Underdog-Woof` pinned to the revision above and verifies the
   file SHA-256. TurboSpark does not host or republish the weights.
2. **A non-commercial acknowledgement is required up front**, as a request
   field (the shape of MegaTTS3's `consent: true`) and as an explicit step in
   the app's install flow. A request without it is refused with a message that
   names the license. The exact field name is open.
3. **Attribution is built in.** The app shows "Voice by Underdog Woof" wherever
   Woof output is played or offered. The server returns it in a response header
   or the speech metadata so API clients can surface it. `LICENSE` and `NOTICE`
   are installed with the model unchanged.
4. **Fixed-voice only.** The surface exposes `voice` of `juno` or `mira` and an
   optional `seed`. It rejects reference audio, cloning, speed overrides and
   other languages.

## What the checkpoint holds

Everything is in one NumPy `savez` archive (176 stored members, no compression).
A 0-d string member `__tree__` holds the JSON structure, and arrays hang off it
as `{"@": key}`. Element types are `<f4`, `<f2` and `|i1`. The checkpoint
declares `int8_method = "rtn"`, and the loader refuses anything else.

| Part | Facts |
| --- | --- |
| Speech stage | 8 layers, width 448, 7 heads (head dim 64), feed-forward 1216 with gate and up packed into 2432 rows, RoPE base 10000. 45 matrix-vector products over 22,747,584 weights per frame, int8 with a per-row f32 scale. A one-step conditional flow head: width 320, depth 4, 32 latent channels. Text embedding 95 x 448 (character vocabulary), a BOS vector. |
| Voices | `voices.juno`, `voices.mira`, prefix embeddings prepended to the text. |
| Latents | 25 frames per second, normalized with stored `mean` and `std`. Up to 750 frames. |
| Decoder | Adapted from Kyutai's Mimi (CC-BY 4.0). A 2-layer causal transformer with a 250-frame sliding window (width 512, 8 heads of 64), then a causal SEANet: conv k7 512 to 768, four transposed convs with strides 8, 6, 5, 4 and kernel 2 x stride, a residual block (conv k3, conv k1) after each, ELU, a final conv k3 48 to 1. About 15.3M parameters, grouped int8 (group 32) with fp16 for the small tensors. |
| Sampling | Temperature 0.5. Stop when the per-frame EOS logit passes log(0.9 / 0.1). One `randn(32)` per frame, drawn after the EOS check, from a torch-compatible mt19937 with Box-Muller in groups of 16. Seed reproducibility depends on matching this stream exactly. |
| Takes | A piece counts as bad when it holds a run of one second or more under -50 dBFS, or when it passes 0.095 s of audio per character, or when it hits the 750-frame cap. A bad piece is redrawn up to 3 more times and the take with the lowest (bad, silence, seconds per character) score is kept. The RNG is seeded once per call, not per piece, so every redraw and every later piece continues the same stream. Long text is split into sentences, then at commas. |
| Frontend | A rule-based English normalizer (47 KB of Python plus a 590 KB respelling lexicon) that rewrites numbers, dates, times, money and codes the way they are spoken. It has to be ported exactly, and it is the largest piece after the decoder. |

## Crate layout

Follows the audio TTS convention in `crates/audio/AGENTS.md`: one directory per
family under `crates/audio/src/tts/`, a family `README.md`, tests that read
pinned fixtures and skip when an installed witness is absent.

```
crates/audio/src/tts/underdog_woof/
  mod.rs        module docs, constants, re-exports
  npz.rs        stored-only .npz reader, .npy parsing, exact f16 -> f32      (done)
  weights.rs    lin_w / deq mirrors, SpeechWeights, tensor hashes             (done)
  tests.rs      reader, dequantization and the hash gate                      (done)
  README.md     family record
  rng.rs        torch-compatible mt19937 randn                                (open)
  speech.rs     prefill and per-frame step, flow head, EOS                    (open)
  decoder.rs    windowed transformer and SEANet, CPU reference                (open)
  normalize.rs  rule-based text normalizer and respell lexicon                (open)
crates/audio/testdata/tts_underdog_woof/
  tensor_hash_reference.json   frozen hashes of the 60 dequantized tensors    (done)
```

The reader is local to the family on purpose. `crates/image` has an `.npz`
reader inside a fixtures module, but audio should not depend on the image
crate, and promoting a shared reader is a later cleanup, not part of this slice.
SHA-256 comes from `turbospark_model_io::hash_data`, so the slice adds no
dependency. The archive is stored, so no deflate crate is needed either.

## What exists now

- `Npz`: indexes the ZIP central directory, refuses compressed members, ZIP64,
  unknown dtypes, Fortran order and payloads whose length disagrees with their
  shape. It does not check per-member CRC-32 (that would need a crate). The
  whole-file SHA-256 and the tensor hashes cover integrity instead.
- `lin_w` and `deq`: the dequantization of `runtime.py`, one f32 multiply per
  element, bit for bit.
- `SpeechWeights::load`: checks every shape against the checkpoint's own `cfg`
  and refuses a mismatch.
- The gate: file SHA-256, then all 60 dequantized tensors (SHA-256 over
  contiguous little-endian f32 bytes, the way `mac_validate.py` records them),
  then the digest of the whole set. It passes on the real file. It has teeth: a
  one-ulp error in the scale multiply fails it for exactly the 42 int8 tensors,
  and a single corrupted int8 weight fails it for exactly one tensor.
- Run it with the checkpoint located by `TURBOSPARK_UNDERDOG_WOOF_NPZ`
  (fallbacks `~/models/underdog-woof/woof_int8.npz`, `~/models/woof_int8.npz`).
  Without the file the gate skips with a message.

```sh
TURBOSPARK_UNDERDOG_WOOF_NPZ=/path/to/woof_int8.npz \
    cargo test -p turbospark-audio --lib underdog_woof -- --nocapture
```

## Gates for the rest of the port

Record each separately (`crates/audio/AGENTS.md`). Compilation and parity alone
do not establish quality or speed.

1. **Integrity (done).** File hash, 60 tensor hashes, set digest.
2. **Geometry (done).** Shapes against `cfg`, refusal on mismatch.
3. **RNG.** Bit-exact `randn` against the torch-compatible stream for several
   seeds and lengths. Wrong here and every seed diverges at frame 0.
4. **Decoder on identical latents.** The only well-posed waveform gate. Feed the
   same latents to a fp32 NumPy decoder and to ours. Require max absolute error
   under one PCM16 step (3.05e-5) and the same sample count. The MLX decoder
   measured 2.2e-6 and 24 of 24 full renditions within 1 PCM16 step.
5. **Teacher-forced speech stage.** Feed the reference's own latents back as
   feedback and compare the predicted latents, with layer-by-layer prefix and
   KV-cache parity first. The MLX speech port holds 2e-5 at 8, 32, 96, 192 and
   256 frames, which is what the port should be held to.
6. **Normalizer.** Exact string equality against the Python normalizer on a
   corpus, including the respelling lexicon.
7. **Free-running output is behavioral, never sample-exact.** Do not require a
   free-running waveform to match the reference sample for sample. The
   autoregressive trajectory is chaotic: a 1e-6 perturbation of one latent in
   pure NumPy reproduces the same mid-sentence divergence, the EOS logit is a
   noisy spike train, and upstream's own six sample clips are not reproduced by
   its own runtime (correlation 0.546 to 1.000). A third-party ONNX port
   reports the same. Gate on a sample-count band, an ASR transcription match,
   stall and silence caps, and a listening check.
8. **Hardware.** Warm interleaved wall time, peak RSS and first-audio delay on
   real Apple Silicon, on AC power, with the host load recorded.

## Performance facts and plan

Measured on the NumPy reference and its variants (M4 Max, interleaved, warm):

- The reference runs about 16x real time on the standard sentences (464 ms for
  7.5 s of audio). The speech stage is 52 to 53 percent of wall time and the
  decoder 47 to 48 percent.
- **The decoder is a batch workload and moves to the GPU well.** An MLX decoder
  is about 7x faster than the NumPy one alone (228 to 32 ms) and 1.52 to 1.64x
  end to end, with output within 1 PCM16 step. fp16 and bf16 gave no speedup and
  33 and 274 PCM16 steps of error, so the decoder stays fp32.
- **The speech stage is launch-bound on a GPU and does not benefit.** One frame
  is a chain of many small dependent operations. An MLX speech stage ran at 0.63x
  of the reference. A single-pass Metal step would cost roughly one launch per
  op, and nothing measured suggests it beats the CPU.
- **On the CPU the step is 1.17 ms per frame.** NumPy over Accelerate does the 45
  matrix-vector products in about 0.72 ms. The other 0.45 ms is Python overhead
  (derived by subtraction). A C probe over the real shapes found that a
  hand-written NEON kernel with exact fp32 activations (fp32 or int8 weights)
  takes 1.13 ms and does not beat Accelerate. An int8-weight, int8-activation
  `sdot` kernel takes 0.29 to 0.36 ms but quantizes the activations, which
  changes the numerics. A third-party ONNX port found dynamic INT8 compounds
  over the autoregressive frames (up to 0.60 max sample error on a long
  sentence), and its fp32 ONNX is only 1.08x NumPy.

Plan, in order:

1. Decoder first, on Metal, as a batch of convolutions and elementwise ops, held
   to gate 4. It is the larger and safer win.
2. Speech stage on the CPU as a fused, exact-numerics step through
   Accelerate's sgemv, which removes the per-op overhead (a projection of about
   0.75 ms per frame, 1.5x on the speech stage, unbuilt). Keep fp32 activations.
3. int8 activations only behind gate 5 and a listening check. They are out by
   default.
4. Batching independent streams helps server throughput, not single-stream
   latency. Defer it.

## Cross-layer checklist (listed, not started)

Family support is cross-layer (root `AGENTS.md`). Kokoro is the template for a
fixed-voice TTS that reached the app. Trace the family enum and the persisted
string through each of these.

- [ ] **Audio crate.** RNG, speech stage, decoder, normalizer, retry logic,
      WAV output. Add a row to `crates/audio/MODELS.md` and `docs/audio/tts.md`,
      and a fixture generator under `crates/audio/tools/` (regeneration stays
      separate from tests).
- [ ] **Install format.** Decide whether the install is the upstream `.npz` as
      is or a repacked layout (`crates/model-io`, `crates/repack`). The `.npz`
      reader here is enough to load the upstream file directly.
- [ ] **Catalog.** `crates/catalog/src/audio_families.json`,
      `audio_assets.json`, `audio_catalog.rs`, `library/registry.rs`, and the
      catalog tests, with the pinned revision, file hash, license acknowledgement
      and attribution fields.
- [ ] **Runtime.** A `woof.rs` beside `crates/runtime/src/kokoro.rs`, wired
      through `native_audio.rs` and `lib.rs`.
- [ ] **FFI.** The audio speech capability in `crates/ffi`, the C header, and
      `c_surface` tests, with the acknowledgement field crossing the ABI.
- [ ] **Server.** `crates/server/src/audio/speech.rs`, `audio_real.rs`, the
      `--tts-model` help text in `args.rs`, `docs/openapi/audio.openapi.yaml`,
      `crates/server/tests/audio_api.rs`.
- [ ] **CLI.** The TTS commands and `crates/cli/README.md`.
- [ ] **Swift.** The Audio workspace controller and picker, the discover catalog,
      the install flow with the acknowledgement step, attribution on playback,
      the ten localization parity gates for every new string, and the audio
      tests. Read `swift/AGENTS.md` and `DESIGN.md` first.
- [ ] **Docs.** `docs/AUDIO.md`, `docs/AUDIO_API.md`, `docs/AUDIO_WORKSPACE.md`,
      `docs/SWIFT_BINDINGS.md`, `CHANGELOG.md` when it ships.
- [ ] **Release.** Nothing ships before the license question above is answered.

## Open questions

1. Will Conway agree that a Rust port is acceptable under the license, and on
   what terms? (Blocks shipping, not development.)
2. Name and shape of the acknowledgement field on the API and in the app.
3. Install as the upstream `.npz`, or repack?
4. Whether the shared `.npz` reader should move out of the family module once a
   second family needs it.
