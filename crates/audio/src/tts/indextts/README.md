# IndexTTS family

Port of `mlx_audio/tts/models/indextts/` (indextts.py, gpt2.py,
conformer.py, perceiver.py, attention.py, mel.py, bigvgan.py, and the
local `ecapa_tdnn/` speaker encoder) at mlx-audio
[`e1b19b9054bf163f5d812221a54fcc346f1890e9`](https://github.com/Blaizzy/mlx-audio/tree/e1b19b9054bf163f5d812221a54fcc346f1890e9/mlx_audio/tts/models/indextts)
(mlx 0.32.3), pinned by `crates/audio/testdata/tts_indextts/` (generated
by `tools/gen_indextts_fixtures.py`).

## Architecture

- A gpt2 backbone over the shared `crate::lm` port with BOTH embedding
  tables patched to identity (the reference sets `gpt.wte` and
  `gpt.wpe` to `nn.Identity`), so every position arrives as a
  precomputed embedding row: the text stream adds
  `text_embedding(tokens) + text_pos_embedding(arange)`, and each
  generated mel code adds
  `mel_embedding(code) + mel_pos_embedding(prefill_len + step)`.
- A conditioning stack over the reference log-mel (100 HTK bands, n_fft
  1024, hop 256, centered): the Conformer (conv2d2 subsampling,
  relative-position attention with per-head `pos_bias_u`/`pos_bias_v`
  score bias, GLU depthwise conv, SiLU feed-forwards) compresses the
  mel to `output_size`, and the `PerceiverResampler` (8 learned
  latents, bias-free attention over [context | latents] REBUILT each
  layer, gated exact-GELU feed-forwards) yields the conditioning rows.
- The generate loop: prefill the [conditioning | text] embedding block
  through the gpt2 backbone, apply `final_norm`, and per step draw one
  mel code from `mel_head` with `make_sampler(temp=0.8, top_k=30)`
  (the greedy temperature-0 path is exact), stopping at
  `stop_mel_token` or the token budget. EVERY step's post-norm last
  row (including the stop-token step) becomes a vocoder input.
- The conditioned BigVGAN vocoder: the ECPA-TDNN speaker encoder (the
  indextts-local variant: reflect-padded TDNNs, Res2Net with the
  segment-2+ chain, masked-mean attentive statistics pooling with
  global context and its own BatchNorm) embeds the reference mel; the
  latent rows run through the shared BigVGAN core (weight-norm convs,
  anti-aliased snake-beta AMPBlocks with checkpoint-stored kaiser
  filters, multi-receptive-field fusion) with the speaker vector added
  through a 1x1 condition conv before the resblock stack and, when
  configured, through per-stage 1x1 convs after each upsample.

Note the local `ecapa_tdnn` is a DIFFERENT network from the
`mlx_audio/codec/models/ecapa_tdnn` backbone the codec family ports
(`../../codec/ecapa_tdnn`): the TDNN blocks reflect-pad before their
grouped convs, the Res2Net chain feeds each branch only from the
second segment on, every SeRes2Net block carries a 1x1 shortcut when
its widths change, and the pooling keeps the global-context mean/std.

## Verification

Fixture protocol identical to the other LM-backed families:
`tools/gen_indextts_fixtures.py` builds the true indextts `Model` at
tiny seeded dims (gpt2 2 layers over dim 32 / 4 heads, conformer 2
blocks at output 32, perceiver 8 latents, 16 mel codes, 24 text
tokens, two-stage conditioned BigVGAN; F16-stored weights reloaded
upcast before the traces run, computed `pos_enc._pe` kept f32 because
the Rust recomputes the table, and BatchNorm running variances seeded
positive), runs the true `prepare_input_embedding` and generate loop,
and records into `testdata/tts_indextts/`:

- `generate/`: the reference waveform, its log-mel rows, the
  conditioning rows, the prompt embedding, and two greedy scenarios
  (a stop-token stop with the mel head's stop row amplified 1.8x, and
  a max-tokens stop), each with the sampled ids, the per-step latent
  states, and the vocoder waveform.
- `sampler/`: greedy argmax rows and the make_sampler top-k mask
  (kth-value, ties kept) on fixed logit rows.
- `manifest.json`: sha256 digests, backbone seed 30, stop
  amplification, sample rate.

Measured parity on the frozen fixtures (gates in `tests.rs`): mel
2.1e-5 (gate 1e-4; the crate FFT versus MLX), conditioning and prompt
embedding 1.7e-7 (gate 2e-5), per-step latents 6.0e-8 (gate 2e-5),
waveforms 3.0e-8 (gate 2e-4); sampler ids exact. Config defaults,
`use_mel_codes_as_input`/`condition_type` refusals, and the invalid
`input_layer: "conv2d"` refusal are pinned by a dedicated test.

Fixture gotchas found on the way: the reference SeRes2Net mutates its
input in place (`x += self.se_block(...)` with an Identity shortcut),
so downstream Python snapshots of earlier stages move - the port uses
explicit values, and the final concat/pooling is unaffected; and the
seeded weight scale must stay small enough that the pooling softmax
does not saturate into one-hot weights, where an f32 ordering
difference flips the argmax and moves the pooled context by tens
(the generator pins 0.08 and documents it).

## Deviations and scope

- Batch size 1, non-streaming: the reference pads prompts for batches;
  the single-prompt path needs no padding.
- Caller-owned text tokenization: the reference runs
  `normalize.normalize` + `tokenize_by_CJK_char` + sentencepiece; the
  Rust API takes text ids. The reference audio load/resample path is a
  caller concern; the reference log-mel is computed by
  [`Model::log_mel_spectrogram`].
- Sampled draws use the crate-local threefry RNG
  (`sampling::sample_categorical` over the filtered logits, the same
  chain order as `make_sampler`: top-p first - on RAW logits, a
  reference quirk this port reproduces when enabled - then top-k, then
  the temperature-scaled categorical). Temperature 0 never draws and
  is fully deterministic.
- `use_mel_codes_as_input=false` and `condition_type !=
  "conformer_perceiver"` are refused like the reference
  NotImplementedError; `input_layer: "conv2d"` (the dataclass default,
  a KeyError in the reference CONV_LAYERS table) is refused too, and
  `gpt.condition_module` is required.
- The reference `text_head` is loaded for checkpoint parity and never
  run (the reference generate path never uses it). The gpt2 tied-head
  forward is refused when the tables are absent
  (`Gpt2Model::from_config_without_tables_at_root`); the family only
  runs `forward_hidden`.
- Only `pos_enc_layer_type: "rel_pos"` is ported (the reference's
  fallback attention flavor is refused with a clear error); the
  streaming Conformer cache path is refused (the conditioning pass is
  always cacheless); `macaron_style` is refused.

## Open items

- Real-checkpoint gate not run: the fixtures are seeded tiny
  instances. The deployed IndexTTS-1.5/2 checkpoint layout (bigvgan
  tensor names, the gpt2 conv1d transposes the reference sanitize
  applies, presence of the `gpt.wte`/`gpt.wpe` dummy tables) needs one
  real download to confirm the loader probes.
- Sentencepiece text normalization (`normalize.py`) and the
  Hub tokenizer download stay Python-side by design.
