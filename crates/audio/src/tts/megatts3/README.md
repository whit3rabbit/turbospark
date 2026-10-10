# MegaTTS3 (Zero-Shot Voice-Cloning TTS)

MegaTTS3 (ByteDance) clones a voice from a short prompt: a sparse-alignment
latent diffusion transformer generates 25 Hz WavVAE latents conditioned on
phonemes, predicted durations, and the prompt latent. The whole pipeline is
ported and gated against the reference on real weights; the product surfaces
(catalog, runtime, FFI, server, Swift) are not wired yet. See
[Pipeline](#pipeline) and [Not done](#not-done).

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
| 6 | `MegaTts3` facade, post-processing, CLI | Done (`facade.rs`, `postprocess.rs`, `rng.rs`, `turbospark-clone`); catalog, runtime, FFI, server, Swift are not done |

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

## Not done

- **Catalog, runtime, FFI, server, Swift.** No catalog alias exists because the
  converted weights are not hosted anywhere yet: a catalog row needs the
  repository and revision you upload them to, and a published sha256 for every
  file (`manifest.json` from the converter lists them). The runtime also has no
  reference-audio request shape: its TTS validation rejects input PCM and is
  Kokoro-specific, so cloning needs a request change across runtime, FFI, the
  server's speech route and the Swift workspace. `turbospark-clone` is the
  supported entry point until then.
- **Chinese.** The Chinese normalizer, chunker and `langdetect` routing.
- **Speed.** The CPU reference is slow (one DiT step at 250 latent frames took
  about 4.4 s). A Metal path is a separate piece of work.
- **Quality.** Parity with the reference is gated, and one short sentence
  passed an ASR round trip. No listening evaluation, speaker-similarity
  measure, or broader intelligibility (WER) study has been done.

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
| Memory (CPU path, one reading) | Peak RSS 6.3 GB for a 3.3 s reference and 3 s output, 8.6 GB for a 27.5 s reference and 7.6 s output: about 5.8 GB resident (the bf16 G2P widens to 2 GB) plus about 3.2 MiB per latent frame. `activation_reserve_bytes` uses 3.5 MiB per frame. |
| Mutation checks (DiT) | Swapped `p_w`/`t_w`, neox rotary, swapped final adaLN scale/shift, uncond row using real phones, unmasked prompt region, zeroed ctx-mask embedding, dropped relative-value term, off-by-one key window, zero AMO noise, and a skipped per-step draw each fail the tiny-fixture test. |
| Mutation checks (pipeline) | A trim off by one latent frame, a prompt latent not restored over the sampler output, and swapped first/final chunk flags each fail the whole-pipeline test: the first on output length, the second on the audio comparison, the third on a replay draw-size mismatch (the changed flags change the draw order). |
| Mutation checks (G2P and chunker) | EOS never recognised, BOS marker dropped from the prompt, final token kept, `sil` token dropped, wrong prompt speech token, chunk budget off by one, joining space after wide characters, splitting on any ASCII stop, and a dropped last chunk each fail a test. The argmax tie rule is equivalent in the tests (no ties occur). |
| Mutation checks (aligner) | Skipped encoder position, decoder position off by one, dropped cross attention, ReLU for GELU, wrong attention scale, undropped last token, Chinese phone offset, tone 1 mapped wrong, no padding of a short alignment, a one-frame removal loop, a mel not cut to a multiple of 8, and overlapping frames assigned instead of summed each fail a test. Two mutants (padding at remainder 1199, `cum.max(prev)`) are equivalent. |
| Mutation checks (duration) | Off-by-one prompt input, wrong first chunk input, query/key rotary positions off by one, ReLU for Softplus or GELU, skipped final LayerNorm, dropped tone embedding, a changed crossfade margin, first-phone pin, padding at an aligned total, swapped draw order, and round-half-away each fail a test. One mutant (query rotated at the cache index) is equivalent because the two positions are always equal. |

Not gated: perceptual quality, a full-length (32-step, 10 s) sampler run
against PyTorch, and bf16/fp16 profiles.

Speed (indicative, one reading, release build, Apple M4 Max CPU, host not
verified quiet): one CFG step at 250 latent frames (10 s of audio including the
prompt) took about 4.4 s, so 32 steps is roughly 2.4 minutes. This is a CPU
reference; it is not a Metal or throughput claim.
