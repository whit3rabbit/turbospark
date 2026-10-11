# MegaTTS3 (Zero-Shot Voice-Cloning TTS)

MegaTTS3 (ByteDance) clones a voice from a short prompt: a sparse-alignment
latent diffusion transformer generates 25 Hz WavVAE latents conditioned on
phonemes, predicted durations, and the prompt latent. The whole pipeline is
ported and gated against the reference on real weights, and it is wired through
the runtime (family `megatts3`), the FFI, the HTTP speech route and the Swift
app; every cloning request needs explicit consent. On macOS it runs on Metal by
default, at 16 diffusion steps unless asked otherwise (see
[Steps and quality](#steps-and-quality)). The app installs it from the catalog
as `megatts3-f32`, the converter output pinned at
[cowWhySo/MegaTTS3-TurboSpark](https://huggingface.co/cowWhySo/MegaTTS3-TurboSpark).
See [Pipeline](#pipeline), [Metal](#metal) and [Not done](#not-done).

## Upstream References and URLs

- **Reference implementation**: [bytedance/MegaTTS3](https://github.com/bytedance/MegaTTS3)
  (Apache-2.0). There is no mlx-audio port, so parity is against the
  unmodified PyTorch classes through `tools/gen_megatts3_dit_fixtures.py`.
- **Paper**: [MegaTTS 3](https://arxiv.org/abs/2502.18924).
- **Weights**: `tools/convert_megatts3.py` turns the pinned `.ckpt` files into
  MLX-layout safetensors (conv weights `[out, K, in]`, weight norm unfolded),
  plus a root `config.json` (`model_type: megatts3`, `voice_clone`), `LICENSE`,
  `NOTICE.md` (sources, hashes, encoder provenance) and a model card, so the
  output folder can be published as is. Output bytes are deterministic.

## Stage status

| Stage | Piece | Status |
|---|---|---|
| 0 | Weight converter | Done (`tools/convert_megatts3.py`) |
| 1 | WavVAE codec | Done ([`codec/wavvae`](../../codec/wavvae/README.md)) |
| 2 | DiT + sampler + relative-position encoder | Done (this directory) |
| 3 | Duration LM and chunk post-processing | Done (`duration.rs`, reuses [`rel_transformer`](rel_transformer.rs)) |
| 4 | Aligner (Whisper-style) and prompt token parser | Done (`aligner.rs`) |
| 5 | G2P (Qwen2) and English text frontend | Done (`g2p.rs`, `text.rs`); the English normalizer is a documented subset |
| 6 | `MegaTts3` facade, post-processing, CLI | Done (`facade.rs`, `postprocess.rs`, `rng.rs`, `turbospark-clone`) |
| 7 | Runtime, FFI, server, Swift | Done (`runtime::megatts3`, `ts_audio_install_with_options`, `POST /v1/audio/speech`, the Voice Clone page, catalog row `megatts3-f32`) |
| 8 | Metal | Default on macOS (`execution.rs`, `runtime::megatts3::Device::Metal`); see [Metal](#metal) |

## DiT (`dit.rs`, `sampler.rs`)

```text
phones + tones --RelTransformer (6 layers, d 1024, 4 heads, window 4)-->
  expand by mel2ph (100 fps) --two stride-2 convs--> x_ling   (25 Hz)
prompt latent + ctx_mask --local_cond_project--> prenet --+
noisy latent (zeroed inside the prompt) --x_prenet---------+--> h
h --24 x [adaLN-zero, rotary MHA (16 heads), adaLN-zero, SiLU MLP]--> adaLN out
  --out_proj--> postnet --> velocity (32 ch)
```

Three CFG rows share `x` and the timestep: spk+txt (real phones, prompt),
txt (real phones, no prompt), uncond (mask tokens `vocab - 1`, no prompt),
combined as `uncond + p_w (txt - uncond) + t_w (spk_txt - txt)`. Everything
independent of `x` and the timestep is computed once in `Dit::prepare`. The
three rows run as one stacked `[3 * frames, dim]` pass (attention over
`3 * heads` planes, the timestep-only adaLN projections computed once); this is
bit-identical to running each row alone because every op works per position
and each row's heads only see that row's keys.

The sampler is Euler flow matching with F5-TTS sway sampling
(`t -> 1 - cos(pi t / 2)`) and the AMO overshoot step. It draws fresh Gaussian
noise on every step, so it is stochastic: noise comes from a caller-supplied
filler, taken in the reference's order (initial latent, then one tensor per
step including the last, whose weight is zero).

```rust
let dit = Dit::open(dir)?;                                   // dir/model.safetensors
let cond = dit.prepare(&phone, &tone, &mel2ph, &prompt_latent)?;
let latent = sample(&dit, &cond, SamplerParams::default(), &mut fill_normals)?;
// latent: [frames, 32]; overwrite the first prompt_frames rows with the
// prompt latent, then decode with WavVae.
```

### Checkpoint tensors the reference does not use

`diffusion_transformer/model_only_last.ckpt` also holds `bn_eos_bos`,
`seg_embed.weight`, and `time_embedding.*`. The reference loads with
`strict=False` and never touches them, so the loader ignores them too. A real
checkpoint load reports exactly these as unexpected and nothing as missing.

### Deviations from the reference

- Zero (pad) phone ids are rejected instead of being masked out; inference
  never pads a batch.
- The AMO noise weight clamps its radicand at 0; torch would produce NaN when
  rounding pushes it a hair negative.
- f32 everywhere. The reference runs fp16 autocast on GPU; the parity targets
  are its f32 CPU outputs.

## Pipeline

```rust
use audio::tts::megatts3::{MegaTts3, SeededRng, SynthParams};

let mut tts = MegaTts3::open(dir)?;                 // aligner_lm/ diffusion_transformer/ ...
let mut rng = SeededRng::new(7);                    // or a ReplayRng of recorded draws
let prompt = tts.prepare_prompt(&reference_24k, &mut rng)?;   // `preprocess`
let audio = tts.synthesize(&prompt, "Hello world.", &SynthParams::default(), &mut rng)?;
// audio: mono f32 at 24 kHz
```

`synthesize_controlled` is the same pipeline with a `progress` callback
(before each chunk and after every diffusion step; an error from it cancels)
and an `emit` callback that streams finished audio: each chunk's samples are
released as soon as it is decoded, minus the 0.16 s tail the next chunk may
crossfade into (`Crossfader`; the concatenation equals `combine_segments` bit
for bit). Helpers for callers that validate before queueing: `plan_chunks`
(the language and empty-text checks), `resample_reference` (any rate to 24 kHz
with the long sinc-Hann filter) and `activation_reserve_bytes` (a measured,
conservative peak above the weights). `MegaTts3::open` checks the root
`config.json` when present: `model_type` must be `megatts3` and `voice_clone`
must match whether the WavVAE has its encoder.

`prepare_prompt` pads the prompt, measures its BS.1770 loudness, aligns it, encodes
it to a latent with the community WavVAE encoder, and primes the duration LM.
`synthesize` checks the language, normalizes and chunks the text (130-byte
budget), then per chunk runs G2P, the duration LM, the DiT sampler and the
WavVAE decoder, trims the prompt region, matches the prompt's loudness (with the
0.95 peak guard), and joins the chunks with a 0.16 s Hann crossfade.

Randomness is injected as `Randomness`, in the reference's order: one normal
draw for the prompt encoder, then per chunk two uniform draws (duration jitter)
and `steps + 1` normal draws (the sampler). `SeededRng` is a self-contained
seedable generator; `ReplayRng` replays recorded draws and is how the tests
reproduce a PyTorch run. The official decoder-only WavVAE has no encoder, so
`can_clone()` is false and `prepare_prompt` returns `Unsupported`.

Deviations from the reference: the aligner's 24 to 16 kHz resampling is the
crate's sinc-Hann filter, not librosa's soxr (see Aligner); the English
normalizer is a subset and non-Latin text is rejected (see G2P and text
frontend); the prompt is cut to 28 s like the Space; output is f32 (the
reference writes int16); a chunk shorter than the 0.16 s crossfade is an error
where the reference's broadcast would fail.

## Metal

With a device backend (the runtime's `gpu::Music3Device`), the DiT's per-block
projections (q, k and v stacked into one call) and attention, and the WavVAE
decoder's convolutions run on the GPU in F32. Norms, modulation, RoPE,
activations, the small projections, the aligner, the duration LM, G2P and the
WavVAE encoder stay on the CPU, and so do the two decoder transposed convs with
output padding (odd upsampling ratios), which the device contract cannot
express. The device path computes RoPE and attention position-major, the layout
the projections produce; with the host kernels behind the device interface it
reproduces the host path bit for bit (`device_path_with_host_kernels_matches_*`).
The runtime opens MegaTTS3 on Metal by default. The CPU path runs only as the
portable opt-in's fallback when Metal cannot open, and `turbospark-clone --cpu`
selects it directly.

Metal became the default after three checks (see [Verification](#verification)).
It reproduces the CPU path across 8 speakers and two multi-chunk passages at 32
steps, with an RMS difference of at most 2.3e-5, so it inherits the CPU path's
parity with the reference. The largest single-sample difference was 1.8e-3, on
a 13 s passage. Runtime admission bounded every measured peak. And it is about five
times faster than the CPU on a short request. It does hold 1.3 to 2.9 GiB more
than the CPU path, because the uploaded weights keep their host copies (see
[Not done](#not-done)).

## Steps and quality

Steps default to 16 (Balanced). Fast is 8, Best is 32 (the upstream sampler's
count), and `steps` takes any count from 1 to 100. The app's Quality picker,
`turbospark-clone --quality` and the HTTP `steps` field use these numbers, which
`Quality` in `facade.rs` defines once.

The step study found no intelligibility or speaker-similarity loss from 8 steps
up. It cloned 8 LibriSpeech test-clean speakers (5 to 9 s references), each
speaking two 10 to 22 word sentences from another speaker's transcripts, on
Metal at every step count with one seed per sentence. The same requests at 32
steps with a second seed are the noise floor. WER is distil-whisper large-v3,
with Whisper's English normalizer on both sides. Similarity is the cosine of
WavLM-base-plus-sv embeddings against the reference: a second real recording of
the same speaker scores 0.963 on average, and two different speakers 0.774.

| Steps | WER | Similarity | Change vs 32 (mean, sd) | Worst change | Sentences with worse WER than 32 |
|---|---|---|---|---|---|
| 4 | 8.5% | 0.968 | -0.006, 0.016 | -0.038 | 2 of 16 |
| 8 | 6.4% | 0.971 | -0.003, 0.011 | -0.040 | 0 of 16 |
| 12 | 6.1% | 0.970 | -0.004, 0.008 | -0.016 | 0 of 16 |
| 16 | 6.8% | 0.976 | +0.002, 0.008 | -0.010 | 1 of 16 |
| 24 | 6.7% | 0.976 | +0.002, 0.005 | -0.008 | 1 of 16 |
| 32 | 6.4% | 0.974 | | | |
| 32, second seed | 8.2% | 0.973 | -0.001, 0.010 | -0.020 | 2 of 16 |

From 8 steps up, no difference is larger than changing the seed. 4 steps is the
only count that trends worse on both measures, and it garbled one sentence ("a
germ, an insignificant seed" came back as "a ger man in significance"), so no
preset goes below 8. Most of the WER is the texts' rare names and archaic words,
which every step count and seed misread alike.

The study cannot see everything. It is 16 sentences, one ASR model and a
base-size speaker model, with no listening test or MOS, so a loss of
naturalness that keeps the words and the voice would not register.

Time grows with steps on top of a fixed part: reference preparation, G2P,
durations and the WavVAE decode. A 3 s sentence on Metal took 12.7 s at 8 steps,
15.5 s at 16 and 24.6 s at 32 (indicative, see Verification).

## Not done

- **CLI install.** `turbospark-model pull-audio` installs Music 3 profiles
  only. Install `megatts3-f32` from the app, or download the repository and
  pass the folder (`turbospark-clone --model DIR`,
  `turbospark-server --tts-model DIR`). Once installed, both take the alias.
- **Chinese.** The Chinese normalizer, chunker and `langdetect` routing.
- **16 GB hardware.** Not run on a 16 GB Mac. From the memory gate below, a
  16 GB machine's admission budget (10.5 GiB) admits a 3 s reference with one
  sentence on Metal (10.05 GiB checked) and refuses a 7 s reference on both
  devices. The refusal comes from the activation reserve (10 MiB per latent
  frame), which set aside 5.2 GiB for a request that grew by 1.3 GiB on Metal.
  Tightening it needs measurements over more request shapes.
- **Duplicate weights on Metal.** The uploaded DiT projections and WavVAE
  decoder weights keep their host copies, so the Metal path peaks 1.3 to 2.9
  GiB above the CPU path. Freeing them means keying the uploads by layer
  instead of by host address.
- **Metal follow-ups.** An F32 MMA attention kernel (attention is about half a
  step at 1000 frames), keeping activations on the device across a block, and
  the encoder, aligner and G2P on the GPU. Timing has not been measured on a
  quiet host on AC power.
- **Quality.** The step study measured WER and speaker similarity on 16
  sentences. No listening test or MOS has been run.

## G2P and text frontend (`g2p.rs`, `text.rs`)

**G2P.** A Qwen2-0.5B causal LM (the shared backbone in `src/lm/`) writes
phonemes. The prompt is `<BOT>` + text + `<BOS>` plus one speech token
(`145 + speech_start`, the `sil` phone), where `speech_start` is the id of the
added token `<Reserved_TTS_0>`. Decoding is greedy (the reference's
`do_sample=True, top_k=1`; the checkpoint's generation config adds no
processors), at most 256 new tokens, stopping at `800 + 1 + speech_start`.
Like the reference, the phone sequence starts with the prompt's `sil` token and
always drops the final generated token. Ids that no embedding table could hold
are errors rather than silent garbage.

**Text.** `ensure_english` stands in for `langdetect`: Latin-script text is
English and any other script is a clear error (the Chinese path needs a Chinese
normalizer this port does not have). `chunk_english` is an exact port of
`chunk_text_english`. `normalize_english` is **not** the reference normalizer:
WeTextProcessing's English grammar is NeMo's, compiled from large pynini FAR
files, and pynini ships no macOS wheel, so its output could not be generated.
This is a conservative subset (whitespace, integers with comma grouping,
decimals, percent, ordinals, dollar amounts, negatives; NeMo-style wording
with no "and"; everything else, including dates, times, phone numbers, codes
and 13+ digit numbers, passes through unchanged). It is unverified against the
FST; callers who need exact reference behavior should normalize first.

## Aligner (`aligner.rs`)

A Whisper-architecture model (6 + 6 layers, width 512, 8 heads, vocab 6800,
learned positional tables) reads the prompt's 80-band log-mel and greedily
emits `<BOS 798> phone t phone t ... <EOS 799>`, where `t = 800 + frame` at
100 frames per second. `align_tokens` drops the first and last token, splits
phones from timestamps (Chinese phones fold phone and tone into ids
`200..=788`; every other id keeps tone 3), fits the total to the mel frame
count (pad the last phone, or take one frame at a time off each phone, which
can drive short phones negative exactly as upstream does), and regulates to a
`mel2ph` a multiple of 8 frames long. The reference recomputes the whole
decoder at every step; this port caches keys and values, which is equivalent
for causal attention with absolute positions.

Front end: `pad_prompt_audio` (the 24 kHz zero padding of `preprocess`),
`resample_for_aligner` (24 kHz to 16 kHz), then `prompt_mel` (the crate's
Whisper log-mel, no 30 s padding, cut to a multiple of 8 frames).

Resampler: the reference uses librosa's `soxr_hq`, which this crate does not
ship. A scan against soxr on a real clip found width 64 at rolloff 0.99 the
closest sinc-Hann setting (waveform gap 4e-4, against 1.8e-2 at the torchaudio
default of width 6). The alignment is then not bit-identical to the soxr path:
on the test clip the phone sequence matches and timestamps move by at most one
10 ms frame.

## Duration LM (`duration.rs`)

An 8-layer pre-LN rotary decoder (width 512, 8 heads, GELU FFN, no attention
bias) regressing one duration per phone (`ar_mse`: Softplus on a 1-wide head,
rounded). Its text side is the shared `RelTransformer` plus a tone embedding
and sinusoid positions, projected by `enc_proj`. Inference is incremental:

1. `DurationLm::prompt` teacher-forces the prompt's own duration codes
   (`prompt_duration_tokens` from the aligner's `mel2ph`) and keeps only the
   key/value state; the model outputs are discarded, as upstream does.
2. `DurationLm::predict` clones that state per text chunk, starts from the
   prompt's last duration code, and decodes one code per phone (the first may
   be 0, later ones are at least 1).
3. `finalize_durations` applies `dur_pred`'s post-processing: shift to frame
   counts, the +32 crossfade margin or the 64..=128 clamp on the last phone,
   random jitter (two `rand_like` draws, injected), `torch.round` ties to
   even, 64/32-frame floors on sentence-end and comma phones, the first phone
   pinned to 8, and padding to a multiple of 8. `length_regulate` expands to
   `mel2ph`.

Keys are cached after rotation at their cache index, which is what the
reference recomputes each step; queries rotate at an explicit position that
equals the cache length. A predicted code outside the embedding (above
`code_size + 1`) is an error, where the reference raises `IndexError`.
`char_encoder.*` and `char_empty_embed` are in the checkpoint but never used
at inference and are ignored.

## Verification

| Gate | Evidence |
|---|---|
| Tiny fixture (committed, `testdata/megatts3_dit/`) | Two cases (12 and 3 phones; the second takes the short-sequence relative-attention branch) with replayed noise: phone encoder 7e-7, `x_ling` 4e-6, first CFG velocity 4e-6, full multi-step latent 3e-6 against PyTorch; test tolerance 2e-5. |
| Real checkpoint (env-gated) | 24-layer DiT, 10 phones, 16 frames, 6 prompt frames, 3 steps: phone encoder 1.2e-6, `x_ling` 1.5e-5, velocity 2.1e-5, final latent 5.7e-5 (peak 3.5); default tolerance 2e-4. Set `TURBOSPARK_DIT_MODEL` and `TURBOSPARK_DIT_GOLDEN`; goldens come from `gen_megatts3_dit_fixtures.py --real-src`. |
| Post-processing (`testdata/megatts3_post/`) | pyloudnorm 0.1.1 `integrated_loudness` on 9 signals (silence, gaps, a 30 dB level step, odd lengths, exactly one block): agreement to 1e-9 LU (observed 3e-14), silence and sub-gate noise give `-inf` like the reference; loudness matching with the peak guard identical; the crossfade join identical (3 shapes). |
| Whole pipeline, real checkpoints (env-gated) | The reference's own `align`, `make_dur_prompt`, `g2p`, `dur_pred`, `prepare_inputs_for_dit`, `Diffusion.inference`, WavVAE, pyloudnorm and `combine_audio_segments` composed on the real weights, with every random draw recorded and replayed: prompt phones, tones and `mel2ph` identical, prompt latent within 1.8e-5, prompt loudness identical, and the synthesized audio of two cases (a single chunk of 3.04 s, and two chunks with a crossfade) has correlation 1.000000 with the reference and a max sample difference of 2.3e-4 and 1.5e-4. The aligner input is the reference's own 16 kHz audio (the resampler differs). Set `TURBOSPARK_MEGATTS3_DIR` and `TURBOSPARK_MEGATTS3_E2E_GOLDEN`; goldens come from `tools/gen_megatts3_e2e_fixtures.py`. The production path (crate resampler, seeded RNG) finds the same 34 prompt phones and produces finite audio of a similar length. |
| Intelligibility (ASR round trip, one-off) | The reference pipeline's audio for both cases was transcribed with the crate's wav2vec2-base-960h: the prompt clip reads "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG" and both syntheses (one chunk, and two chunks with the crossfade) read exactly "HELLO WORLD THIS IS A SHORT TEST" (the target text). The Rust output matches that audio to 2.3e-4, so that is evidence for the port by transitivity. The WAV written by `turbospark-clone` itself (production path: crate resampler, seeded RNG, 8 steps, 3.04 s, 0.80 peak) reads exactly the same sentence. One short sentence is not a quality evaluation. |
| Chunker (`testdata/megatts3_text/chunks.json`) | 76 cases (19 texts x 4 budgets) through the reference `chunk_text_english`, including full-width stops, trailing whitespace, multi-byte last characters, and budget boundaries: all identical. |
| G2P, tiny fixture | A seeded tiny Qwen2 saved by transformers plus a small BPE tokenizer carrying the real added tokens: token ids and 12 greedy `generate` outputs (including EOS stops) identical. |
| G2P, real checkpoint (env-gated) | Six English sentences (5 to 28 text tokens, 11 to 83 phones) through the reference's own `g2p()` with the real tokenizer and fp32 weights: token ids, phones and tones identical. Set `TURBOSPARK_G2P_DIR` and `TURBOSPARK_G2P_GOLDEN`; goldens come from `gen_megatts3_text_fixtures.py --real-g2p`. |
| English normalizer | No reference fixture (see above); its tests pin this port's own stated behavior. |
| Aligner, tiny fixture (`testdata/megatts3_aligner/`) | Encoder 1e-6, teacher-forced logits over every position 1e-6 (the cached decode equals the reference's full recompute), greedy decode exact, tolerance 5e-6. |
| Whisper log-mel | Within 9e-6 of OpenAI's real `log_mel_spectrogram` on a synthetic waveform (tolerance 5e-5). |
| Token parser | Four scripted streams run through the reference's real `align()` with a mock model: short total, long total with a negative duration and overlapping frames, exact total, and zero-length phones; phones, tones and `mel2ph` exact. |
| Aligner, real checkpoint on real speech (env-gated) | All keys load strictly. From the reference's own 16 kHz audio: mel within 1.5e-5, all 70 tokens identical, 34 phones and the 328-frame `mel2ph` identical. From the 24 kHz prompt through the crate resampler: same length and phones, token deltas at most 1 frame, mel gap 3e-2. Set `TURBOSPARK_ALIGNER_MODEL` and `TURBOSPARK_ALIGNER_GOLDEN`; goldens come from `gen_megatts3_aligner_fixtures.py --real-src --clip`. |
| Duration LM, tiny fixture (`testdata/megatts3_duration/`) | Built from the reference's own `make_dur_prompt` and `dur_pred`: three chunks (first/non-final, non-final/final, first/final with `alpha` 1.2) with recorded `rand_like` draws. Unrounded regression 7e-7, raw codes exact, final `mel2ph` exact (jitter, pause floors, padding); decoding chunks in a different order or twice reproduces the same results; test tolerance 1e-5. |
| Duration LM, real checkpoint (env-gated) | All keys load strictly into the reference; 14 prompt phones, one 10-phone chunk: codes exact, `mel2ph` exact (320 frames), regression 5.6e-5 (values up to 11); default tolerance 3e-4. Set `TURBOSPARK_DUR_MODEL` and `TURBOSPARK_DUR_GOLDEN`; goldens come from `gen_megatts3_duration_fixtures.py --real-src`. |
| CFG batching | `stacked_cfg_rows_match_rows_run_alone_bit_for_bit` on the tiny DiT; a mutant that lets attention cross rows fails it. On real weights the `turbospark-clone` WAVs (one chunk, 4 steps; two chunks with the crossfade, 2 steps) are byte-identical before and after the change. |
| Streaming join | `Crossfader` against the original one-shot join on six shapes, including middle segments shorter than two fades: bitwise equal, and audio is released before the end whenever it is longer than one fade. A mutant holding back half a fade fails it and the reference `combine_audio_segments` fixtures. |
| Regenerated goldens | The e2e goldens were regenerated with torch 2.14.1 and transformers 4.57.6 (the originals' versions were not recorded): case 0 max diff 5.4e-5, case 1 1.07e-3, correlation 1.000000 for both. Case 1 exceeds the 1e-3 default tolerance; the code before and after the batching change gives exactly the same diffs, so the shift is in the reference build, not the port. The tolerance was not widened. |
| Memory (one reading each) | Peak process footprint, which runtime admission reads: 6.6 GB at about 120 latent frames (3.3 s reference), 11.1 GB at about 730 (27.5 s reference, short text), 13.8 GB at about 880 (27.5 s reference, 7.6 s of speech); the Metal path peaked at 13.7 GB on the last case. About 5.8 GB is resident weights (the bf16 G2P widens to 2 GB); the rest is at most about 9.5 MiB per frame, mostly the WavVAE. `activation_reserve_bytes` uses 10 MiB per frame. Max RSS reads lower (6.3 to 8.6 GB) and is not what admission measures. |
| Catalog install (network, ignored test) | `real_megatts3_install_needs_its_terms_then_opens_verified` in `crates/ffi/tests/audio_surface.rs`, with `TURBOSPARK_AUDIO_REAL_INSTALL=1` and a scratch `TURBOSPARK_HOME`: the plain install is refused naming `megatts3_community_encoder`; with it accepted, all 10 pinned files (4.38 GB) download, verify and publish, and the session opens as family `megatts3`. `turbospark-clone --model megatts3-f32` from that store wrote a WAV byte-identical to the local converter output's (same seed, 4 steps), and `turbospark-server --tts-model megatts3-f32` answered 400 without `consent` and 200 with it. |
| Swift binding and app (env-gated) | Set `TURBOSPARK_TEST_MEGATTS3` (a model folder) and `TURBOSPARK_TEST_VOICE_REFERENCE` (a mono WAV). `testRealMegaTTS3CloneThroughTheBinding` (`swift/TurboSpark`) is refused without consent, reports diffusion progress, and, with `TURBOSPARK_TEST_MEGATTS3_EXPECTED` set to the float32 WAV `turbospark-clone speak --steps 4 --seed 7 --wav-format float32` wrote for the same reference and text, matches the CLI sample for sample. `testRealVoiceClonePageClonesWithMegaTTS3` (`swift/TurboSparkApp`) drives the Voice Clone page: catalog profile, local model folder, imported reference, consent gate, the CPU session, and the saved encrypted take. Both passed on the 2.8 s test reference; run them with `swift test --build-system native` (the default Swift Build backend failed to link the `syntext` C target in a fresh build). |
| Metal parity | Against the CPU path: tiny DiT velocity 2.2e-6 at peak 3.6 (bound 1e-5 relative); tiny WavVAE decode 5e-7; real DiT velocity 1.1e-4 at peak 7.6 (250 frames) and 5.8e-4 at peak 9.1 (1000 frames); whole pipeline with the same seed, 8 steps: max sample difference 8e-5, cosine 1.000000 (3 s output) and 0.999999998 (7.6 s output). The Metal WAV transcribes through the server's Qwen3-ASR as the target sentence. Mutants: RoPE skipped on keys (device plumbing), and the device conv dispatch counts. |
| Metal across speakers (32 steps) | Same seed on both devices, 8 LibriSpeech test-clean speakers (5 to 9 s references, one 10 to 22 word sentence each): identical lengths, max sample difference 2.9e-5 to 9.5e-4 (RMS at most 2.3e-5), cosine at least 0.99999976. Two multi-chunk passages (48 and 53 words, 13.0 and 17.0 s of speech): identical lengths, max sample difference 1.8e-3 and 4.7e-4, RMS 1.9e-5 and 5.7e-6. Rounding accumulates over long outputs: those maxima exceed the 1e-3 the CPU port holds against PyTorch on short output, while the RMS stays below 2.5e-5. |
| Metal by default, real checkpoint (env-gated) | `real_megatts3_worker_clones_streams_and_cancels` opens with neither opt-in, streams, refuses missing consent and cancels. `real_metal_pipeline_matches_the_cpu_pipeline` (8 steps): identical length, max sample difference 7.9e-5. The Swift binding's clone equals `turbospark-clone`'s Metal output sample for sample, so the session ran on Metal, and Metal output is identical across processes. The app's Voice Clone page clones on the catalog's `metal` backend without the portable opt-in. |
| Memory gate (`benchmark_megatts3.py`, one sample per case) | Peak footprint against what admission checked (footprint before the request plus its activation reserve). 3 s reference, one sentence: Metal 8.81 GiB against 10.05, CPU 6.56 against 8.05. 7 s reference, 7.4 s of speech: Metal 9.79 against 13.74, CPU 8.44 against 10.96. 27.5 s reference, 13.3 s of speech: Metal 14.31 against 23.44, CPU 11.39 against 21.67. Every peak stayed within admission. |
| Speed, short request (`benchmark_megatts3.py`, three interleaved runs, on battery with other sessions active) | 2.8 s reference, 3.04 s of speech, medians: Metal 12.7 s at 8 steps, 15.5 s at 16 and 24.6 s at 32 (spreads 15 to 27%), and the CPU 128.5 s at 32 (spread 6%). Each scenario wrote byte-identical WAVs on every run. Indicative only: not AC, not quiet. |
| Metal speed (indicative, one reading each, M4 Max, host busy with other work) | One DiT step at 250 frames 0.51 s (CPU 3.2 to 4.6 s), at 1000 frames 2.7 s (CPU 13 to 17 s), where attention is about half. A 3 s sentence at 8 steps: 10.4 s warm (CPU 35.2 s); a 27.5 s reference with 7.6 s of speech: 80 s (CPU about 270 s). Not benchmark rows: see `docs/BENCHMARKING.md` before quoting. |
| Mutation checks (DiT) | Swapped `p_w`/`t_w`, neox rotary, swapped final adaLN scale/shift, uncond row using real phones, unmasked prompt region, zeroed ctx-mask embedding, dropped relative-value term, off-by-one key window, zero AMO noise, and a skipped per-step draw each fail the tiny-fixture test. |
| Mutation checks (pipeline) | A trim off by one latent frame, a prompt latent not restored over the sampler output, and swapped first/final chunk flags each fail the whole-pipeline test: the first on output length, the second on the audio comparison, the third on a replay draw-size mismatch (the changed flags change the draw order). |
| Mutation checks (G2P and chunker) | EOS never recognised, BOS marker dropped from the prompt, final token kept, `sil` token dropped, wrong prompt speech token, chunk budget off by one, joining space after wide characters, splitting on any ASCII stop, and a dropped last chunk each fail a test. The argmax tie rule is equivalent in the tests (no ties occur). |
| Mutation checks (aligner) | Skipped encoder position, decoder position off by one, dropped cross attention, ReLU for GELU, wrong attention scale, undropped last token, Chinese phone offset, tone 1 mapped wrong, no padding of a short alignment, a one-frame removal loop, a mel not cut to a multiple of 8, and overlapping frames assigned instead of summed each fail a test. Two mutants (padding at remainder 1199, `cum.max(prev)`) are equivalent. |
| Mutation checks (duration) | Off-by-one prompt input, wrong first chunk input, query/key rotary positions off by one, ReLU for Softplus or GELU, skipped final LayerNorm, dropped tone embedding, a changed crossfade margin, first-phone pin, padding at an aligned total, swapped draw order, and round-half-away each fail a test. One mutant (query rotated at the cache index) is equivalent because the two positions are always equal. |

Not gated: perceptual quality, a full-length (32-step, 10 s) sampler run
against PyTorch, and bf16/fp16 profiles.

Speed (indicative, one reading, release build, Apple M4 Max CPU, host not
verified quiet): one CFG step at 250 latent frames (10 s of audio including the
prompt) took about 4.4 s, so 32 steps is roughly 2.4 minutes on the CPU. See the
Metal rows above for the device path.
