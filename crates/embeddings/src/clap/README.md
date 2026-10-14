# CLAP HTSAT-unfused

Contrastive language-audio dual tower. An HTSAT Swin encoder (audio) and a
RoBERTa encoder (text) each map their input to a unit-norm 512-dim vector
in one shared space. `cos(audio, text)` scaled by `exp(logit_scale_a)` and
softmaxed over labels is zero-shot audio classification.

## Checkpoint and references

| What | Value |
|---|---|
| Checkpoint | [`laion/clap-htsat-unfused`](https://huggingface.co/laion/clap-htsat-unfused/tree/79b58ed25fc00386262a2bea4b19fd21dc4310a0) |
| Pinned revision | `79b58ed25fc00386262a2bea4b19fd21dc4310a0` (`refs/pr/3`, the safetensors conversion) |
| Not used | upstream `main` (`8fa0f1c6...`) ships only `pytorch_model.bin`, a pickle, and it is never loaded |
| `model.safetensors` sha256 | `abba8294f0c63d00eb3f8d998dacd16177cc5860cad3a11618e7f361a75587fe` |
| `config.json` sha256 | `9efb9557bc804f2ca6e394486af2e45dfed0b18554909735a99c6220b84e4288` |
| `preprocessor_config.json` sha256 | `9739f58296aa6f9ac18008fd0150fb2649bc554985fbde86d0a4041c882ac753` |
| `tokenizer.json` sha256 | `77ef92283d67f0d97e1454909a964afcbfa2019f0fb9f18f8e88d5c25c3ba729` |
| Numeric reference | transformers 4.57.1 `ClapModel`, `ClapFeatureExtractor`, fast RoBERTa tokenizer, torch 2.14.1 CPU fp32, numpy 2.5.4 |
| Paper | Wu et al., "Large-scale Contrastive Language-Audio Pretraining with Feature Fusion and Keyword-to-Caption Augmentation", arXiv:2211.06687 |
| Original code | [LAION-AI/CLAP](https://github.com/LAION-AI/CLAP) |
| Port guide | mlxrs `src/embeddings/clap` (structure only: its end-to-end gate was int8 ONNX goldens, so numbers come from HF fp32) |
| MLX conversion | none published on Hugging Face, so nothing to pin. Quantized layers are refused by `Weights::linear` with no scheme |

## API

```rust
use turbospark_embeddings::clap::{Clap, LongAudioCrop, PIPELINE_TEMPLATE};

let clap = Clap::load(dir)?;                       // config, preprocessor, tokenizer, safetensors
let a = clap.embed_audio(&pcm_48k_mono)?;          // >10 s keeps the first 10 s
let a = clap.embed_audio_with(&pcm, LongAudioCrop::Center)?;
let a = clap.embed_audio_file(path)?;              // WAV -> mono -> 48 kHz (see divergences)
let t = clap.embed_text("a dog barking")?;
let ranked = clap.classify(&a, &["a dog barking", "rain"])?; // Vec<(label, prob)>, descending
```

Lower level: `ClapModel` (`audio_embedding`, `text_embedding`,
`logits_per_audio`, `*_with` variants exposing named intermediates),
`ClapFeatureExtractor::extract(pcm, crop) -> MelFeatures`, `HtsatEncoder`,
`RobertaEncoder`, `ClapConfig` / `FeatureConfig`.

## Architecture (real checkpoint)

Feature extractor (`truncation="rand_trunc"`, `padding="repeatpad"`):

- 48 kHz mono, `max_length_s` 10 (480000 samples). Longer: crop a 480000
  window (`is_longer = true`). Shorter: tile `floor(480000 / len)` times,
  zero-pad to 480000.
- STFT: `n_fft` 1024, hop 480, periodic Hann (`np.hanning(1025)[:-1]`),
  centered with numpy `reflect` padding (no edge repeat): 1001 frames.
- FFT in f64, stored as complex64, `|X|^2` in f64.
- 64 mel bins, fmin 50, fmax 14000, Slaney scale with Slaney area
  normalization `2 / (f[m+2] - f[m])` (`mel_filters_slaney`). The HTK bank
  is only used by the fusion path. Mel floor 1e-10.
- `power_to_db`: `10 * log10(max(x, 1e-10))`, reference 1.0, no `top_db`.
  Output `[1001, 64]` f32.

HTSAT audio encoder:

- `BatchNorm2d(64)` over mel bins (eval stats, PyTorch's
  `x * alpha + beta` form, eps 1e-5).
- `reshape_mel2img`: `freq_ratio = spec_size / num_mel_bins = 4`. Time is
  stretched 1001 -> 1024 by `interpolate(mode="bicubic",
  align_corners=True)` (A = -0.75, edge-clamped taps, f32 source index).
  The 1024 frames fold into 4 chunks of 256 stacked along frequency:
  `img[c * 64 + m][t] = x[c * 256 + t][m]`, a 256 x 256 image.
- Patch embed: conv 4x4 stride 4 (1 -> 96 channels), 64 x 64 tokens,
  LayerNorm (eps 1e-5).
- Swin stages: depths [2, 2, 6, 2], dims 96, 192, 384, and 768, heads
  4, 8, 16, and 32 (head dim 24), window 8, resolutions 64, 32, 16, and 8. Odd blocks
  roll by -4 and add the region mask (-100, not -inf) except in the last
  stage, whose resolution equals the window
  (`set_shift_and_window_size` zeroes the shift). Relative position bias
  from a `[(2*8-1)^2, heads]` table. Scores are `q.k / sqrt(24) + bias
  (+ mask)`. Patch merging concatenates (even,even), (odd,even),
  (even,odd), (odd,odd), LayerNorm(4C, eps 1e-5), Linear(4C -> 2C, no
  bias). Block LayerNorms use `layer_norm_eps` 1e-5. GELU is erf.
- Final LayerNorm(768), then the mean over the 64 tokens. The reference
  regroups tokens (tscam layout) before `AdaptiveAvgPool1d(1)`. That only
  permutes them, so `pooler_output` is the token mean.

RoBERTa text encoder (12 layers, 768, 12 heads, 3072 FFN, vocab 50265):

- Byte-level BPE from `tokenizer.json`, `<s> ... </s>`, no truncation.
- Position ids: `cumsum(id != pad) * (id != pad) + pad`, so real tokens
  start at `padding_idx + 1 = 2`, and a pad id keeps position 1 without
  advancing the counter. Embeddings `(word + token_type[0]) + position`,
  LayerNorm eps 1e-12. Post-LN layers, erf GELU, scores scaled by
  `head_dim^-0.5`.
- `get_text_features` projects `pooler_output = tanh(dense(h[<s>]))`, not
  the raw CLS hidden state.

Heads and logits:

- `audio_projection` / `text_projection`: `Linear(768 -> 512) -> ReLU ->
  Linear(512 -> 512)`, then `F.normalize` (eps 1e-12).
- `logits_per_audio = exp(logit_scale_a) * a . t` (`exp` = 18.661 in this
  checkpoint). `logits_per_text` uses `exp(logit_scale_t)` (14.286). The
  HF zero-shot pipeline softmaxes `logits_per_audio` over labels, and
  `Clap::classify` does the same. (mlxrs ranks by plain cosine: the
  ranking is identical, the probabilities are not.)

## Verified stages

All numbers from this checkout on Apple M4 Max, against fixtures written
by `tools/gen_clap_fixtures.py` (manifest in `testdata/clap/manifest.json`).

| Stage | Test | Result |
|---|---|---|
| Synthetic clips replay the generator | `synthesis_matches_generator` | head samples equal, sums within 1e-6 |
| Feature extractor, real preprocessor config, 4 clips (3 s repeatpad, 10 s exact, 0.7 s repeatpad x14, 12 s rand_trunc at the reference's seeded offset 92975) | `feature_extractor_matches_reference` | max abs dB diff 0.0 on every clip (bit-exact), bound 1e-5 |
| Shared f32 frontend (`turbospark_audio` STFT + Slaney bank), measurement only | `f32_frontend_measurement` | max abs dB diff 2.76e-2 (tone), 7.3e-3 (chirp), 8.6e-5 (burst), 1.4e-4 (long), filterbank weights 6.3e-8 |
| Tiny random HTSAT (depths [2,2,2], window 4, 16 mels, spec 64), 200 frames (bicubic) and 256 frames | `tiny_audio_tower_matches_reference` | every intermediate (batch_norm, mel2img, patch_embed, stage blocks/out, norm, pooled, embed) at most 1.001e-6 scaled max diff, bound 1e-5 |
| Tiny RoBERTa (2 layers, hidden 32, vocab 100), 3 sequences incl. a mid-sequence pad id | `tiny_text_tower_matches_reference` | embeddings, layers, pooled, embed within 6.0e-7 scaled, bound 1e-5 |
| Tiny logits (scale 10) | `tiny_logits_match_reference` | max abs diff 3.4e-6 |
| Config refusals | `config_refuses_unsupported_values` | fusion, non-gelu, hidden_size mismatch, window padding, window above resolution, non-absolute positions, non-relu projection, fusion/pad/top_db frontends refused |
| Real checkpoint audio embeddings (4 clips) | `real_checkpoint_parity` (ignored) | max abs diff 2.2e-7, min cosine 1.000000000 |
| Real checkpoint text embeddings (5 labels) | same | max abs diff 5.0e-7, min cosine 0.99999994 |
| Real `logits_per_audio` (4 x 5) | same | max abs diff 1.9e-5, classify ranking identical on all 4 clips, max prob diff 3.6e-6 |
| Real tokenizer ids (10 probes incl. accents, `<mask>`, edge spaces) | `real_tokenizer_ids` (ignored) | equal |

Mutation checks (applied, tests run, restored byte-identical):

- shifted-window mask removed: `tiny_audio_tower_matches_reference` and
  `tiny_logits_match_reference` fail.
- cyclic roll removed (mask kept): the same two tests fail.
- RoBERTa positions starting at `padding_idx` instead of
  `padding_idx + 1`: `tiny_text_tower_matches_reference` and
  `positions_start_after_padding_idx` fail.

Timing (release, M4 Max, CPU f32, `real_timing`, three runs per session,
no warmup discarded, shared host with uncontrolled load): one 10 s clip
end to end took 0.18 to 0.21 s in one session and 0.57 to 0.71 s in an
earlier one (features 0.02 to 0.06 s). One short text took 0.02 to 0.03 s
and 0.05 to 0.14 s. Load took 0.18 s and 0.50 s. Not a frozen benchmark.

## Divergences from the reference

- **Long-clip crop is deterministic:** rand_trunc draws the offset with
  `np.random.randint`. Here the caller picks `LongAudioCrop::{Head,
  Center, Offset(n)}`, and `Clap::embed_audio` uses `Head`. Given the same
  offset the features are bit-exact (verified at offset 92975).
- **`is_longer`** is reported in `MelFeatures` but unused: it only gates
  the fusion branch, which the unfused model does not have.
- **Frontend runs in f64**, like numpy. The shared `turbospark_audio` f32
  STFT and filterbank were measured (table above) and not used: their
  error lands in low-energy bins that the dB scale magnifies (up to
  0.028 dB on a pure tone).
- **Text is embedded unpadded, one sequence at a time.** The pipeline
  pads a batch under an attention mask. The reference itself shows 3.2e-7
  between padded and unpadded embeddings.
- **Over-length text is refused** (more than 512 tokens). The reference
  does not truncate either. It fails indexing the position table.
- **WAV input** is mixed to mono (f64 average) and resampled with the
  torchaudio-compatible sinc-Hann filter at its defaults. The HF pipeline
  decodes files through ffmpeg, so features of resampled audio match the
  reference only as far as the resamplers agree. 48 kHz input skips
  resampling. Not measured.
- **GELU** uses the shared A&S erf approximation (abs error < 1.5e-7).
- **Refused, not implemented:** `enable_fusion=true` checkpoints,
  `truncation="fusion"`, padding other than `repeatpad`, `top_db`,
  non-square or overlapping patches, stage resolutions that need window
  padding or fall below the window, non-absolute position embeddings,
  projection activations other than ReLU.

## Regeneration

```sh
# venv with transformers 4.57.1, torch 2.14.1, numpy, safetensors, tokenizers
<venv>/bin/python -I crates/embeddings/tools/gen_clap_fixtures.py \
    --model-dir ~/models/embeddings/clap-htsat-unfused

CARGO_TARGET_DIR=target/emb-clap cargo test -p turbospark-embeddings \
    --no-default-features --features clap
CARGO_TARGET_DIR=target/emb-clap cargo test --release -p turbospark-embeddings \
    --no-default-features --features clap clap::tests::real -- --ignored --nocapture
```

The generator is deterministic (numpy-seeded tiny weights, xorshift
noise, no network). Two runs produced an identical manifest. Real-model
tests read `$TURBOSPARK_CLAP_DIR` or `~/models/embeddings/clap-htsat-unfused`
and skip when neither exists.
