# Confucius4

Port of the mlx-audio Confucius4-TTS family at
[`e1b19b9054bf163f5d812221a54fcc346f1890e9`](https://github.com/Blaizzy/mlx-audio/tree/e1b19b9054bf163f5d812221a54fcc346f1890e9/mlx_audio/tts/models/confucius4):
multilingual zero-shot voice cloning over a w2v-BERT semantic encoder
(the condition vector), an ECAPA-TDNN speaker encoder and GPT-2 text
projector prefix, a GPT-2 text-to-semantic decoder (BOS/EOS 8192/8193),
a 13-block DiT + WaveNet flow-matching S2A mel estimator with RoPE
tables loaded from the checkpoint, and the shared BigVGAN vocoder.
22.05 kHz output; the reference audio intake is 16 kHz.

## Layout

- `mod.rs`: `ModelConfig` (the reference checkpoint `config.json`
  sections), the five-stage pipeline (`compute_condition`,
  `prefix_embeddings`, `t2s_generate` / `t2s_step_logits`, `s2a_solve`,
  `vocode`), and the reference language-token subset
  (`LANGUAGE_TOKEN`). Text tokenization and the wrapper sentence are
  caller-owned intake.
- `dsp.rs`: the three frontends - the SeamlessM4T-style 160-d fbank
  (checkpoint-provided `fbank_filters.npz` mel + povey window, per-bin
  CMVN with ddof=1, stride-2 frame stacking), the CAMPPlus kaldi fbank
  (the shared `crate::mel` HTK filterbank, f_min 20), and the
  reference-audio 80-band slaney mel (librosa `htk=false,
  norm="slaney"`, the shared `crate::mel` bank over the
  `stt::vibevoice_asr::resample_poly` 16k-to-22.05k resampler).
- `w2vbert.rs`: the two-pass hidden17 encoder (feature projection,
  relative-attention transformer layers, convolution module).
- `campplus.rs`: the CAMPPlus speaker embedding
  (`chatterbox/s3gen/xvector.py`): FCM ResNet head with H-only strides
  (the 1x1 shortcut convs carry no padding), strided TDNN, three dense
  CAM blocks with zero-padded segment pooling, transit convs, statistics
  pooling, and the affine-free dense head (`batchnorm_`, no trailing
  relu). Every channel count is config-driven; the reference's
  model-level `m_channels` kwarg is swallowed unused, so the FCM runs
  at `fcm_channels`.
- `prefix.rs`: the T2S prefix (`prefix.py`) - the text projector
  (`silu(fc1(embed))` -> fc2 -> learned positions) and the ECAPA speaker
  encoder over the w2v condition vector: block TDNN, three SE-Res2Net
  blocks (chunk 0 passes through, chunk 1 runs its conv WITHOUT the
  running add, chunks 2+ add the previous output; the block TDNNs end in
  relu), frame-wise MFA concat (`axis=2`, not whole-tensor blocks),
  attentive statistics pooling (relu before the tanh), and the fc head.
- `t2s.rs`: the GPT-2 decoder (`t2s.py`): semantic embedding + learned
  positions over GPT-2 blocks with fused Conv1D `[in, out]` weights,
  causal prefill + KV-cache steps (row t of a multi-row pass attends
  only through itself), `gelu_new` with the `sqrt(2/pi)` coefficient,
  per-row bias adds, final norm + semantic head, and the reference
  `_sample` filter chain (sign-aware repetition penalty over the
  generated set, temperature, kth-value top-k, descending-cumulative
  top-p returning vocabulary-order probabilities). The categorical draw
  uses the crate-local threefry RNG; greedy argmax is the
  fixture-verified path. `generate` re-passes `[prefix, scodes]` for the
  LM latent (`hful[:, 1 + Tt:-2]`).
- `s2a.rs`: the S2A flow-matching estimator (`s2a.py`): code embedding +
  k1 out-projection, latent concat projection, the nearest-neighbor
  length regulator (int(1.72 * T)) with four conv/GroupNorm/mish
  rounds, `build_mu` (checkpoint prompt broadcast to T_ref), the DiT
  estimator with interleaved-pair RoPE (frequencies from the checkpoint
  table; positions past the table extrapolate with the theta recovered
  from row 1), U-Net skip connections, row-wise qkv splits, a gated
  WaveNet tail, the `[x_res, x]` skip concat (`hid + mels` wide), the
  final adaLN modulated by `silu(t1)`, and the Euler CFG solver that
  re-zeros the prompt region every step.

## Checkpoints

One checkpoint directory: `w2vbert_mlx.safetensors`,
`t2s_model.safetensors` (prefix + T2S), `s2a_mlx.safetensors`,
`bigvgan_mlx.safetensors`, `campplus.safetensors`, and the stats/filter
tables. The reference reads `w2v_stats.npz` and `fbank_filters.npz`
with `np.load`; the port opens the same-key safetensors copies the
conversion writes (`w2v_stats.safetensors`, `fbank_filters.safetensors`).
The vocoder is the shared
[`codec::bigvgan`](../../codec/bigvgan) port.

## Reference quirks kept

- the CAMPPlus `m_channels` constructor kwarg is accepted but never
  forwarded to the FCM (which keeps its 32 default);
- kaldi fbank framing is snip-edges (only fully contained frames);
- the ECAPA MFA concat is per-frame channel concat; the attentive
  pooling applies relu after the asp TDNN before its tanh;
- the res2net chain skips the running add at chunk 1;
- the T2S Conv1D layout stores `[in, out]` and every block bias applies
  to all rows (the natural zip truncates to row 0);
- the T2S latent re-pass positions the codes from 0 while the decode
  feed starts at position 1, matching the reference;
- S2A `build_mu` interpolates with `int(T * 1.72)` nearest indices;
- the euler solver re-zeros the prompt mel of `x` after every step;
- BigVGAN weight-norm folds with `weight = g * v / normalize(v)`, so
  fixture `weight_g` carries the per-channel Frobenius norms.

## Verification

Fixtures at `crates/audio/testdata/tts_confucius4/`, regenerated by
`tools/gen_confucius4_fixtures.py` (run from the mlx-audio checkout
with the project venv; deterministic seeds, no network). The generator
runs the true reference modules on tiny seeded instances: w2v-bert at
hidden 32 / 2 layers, T2S at d_model 32 / 2 layers / 4 heads, S2A at
hidden 32 / 5 blocks / 4 WaveNet layers, CAMPPlus at embedding 16 /
init 16 / growth 8, BigVGAN at two ups stages (rates [4, 4], kernels
[8, 8], dilations [[1, 1, 1]]). The reference `_sample` is overridden
by an argmax stand-in so the end-to-end trajectory is deterministic;
the sampled distribution math is pinned separately by recording the
full post-filter probability vector of one real `_sample` call. The
`mx.random.normal` CFG noise is recorded and replayed by the Rust.

Recorded stage traces: the w2v fbank, hidden17, condition vector,
CAMPPlus fbank and style embedding, the reference mel, prefix
embeddings, per-step T2S logits, codes and LM latent, mu, the solved
mel, and the vocoded waveform.

Measured parity on the frozen fixtures (gates in `tests.rs`): w2v
fbank 1.4e-4 and CAMPPlus fbank 1.2e-4 against the 2e-4 gate; hidden17
5.7e-4 and condition vector 7.1e-4 against 1e-3 (conv/attention
reduction order); CAMPPlus style 1.2e-7 and mu 1.1e-6 against 2e-5;
reference mel 2.4e-4 against 4e-4 (the reference chain is numpy float64
end to end, the port f32); prefix embeddings 7.2e-7, text embeddings
1.2e-7, LM latent 2.4e-6, and per-step T2S logits 1.7e-6 against 1e-5;
solved mel 1.7e-5 against 1e-4; waveform 6.1e-7 relative against a
1e-5 relative gate; the sampled distribution mass 7.5e-9 against 2e-6.

Mutation notes from bring-up (each was a real bug the gates caught):
dropping the causal prefill mask, the `zip`-based bias add (row 0
only), the `sqrt(2/pi)` gelu coefficient, the silu-before-fc1 text
projector order, the res2net chunk-1 add, the MFA block-major concat,
the SE gate relu, the CAMPPlus strided-shortcut padding, the CAM
dilated-conv tap order, the segment-pool divisor, and the fixture
`weight_g = 1` fold all move their gates far past the frozen values.

Rust gates: `cargo test -p turbospark-audio --lib confucius4` (6 tests).

## Open items

- Real-checkpoint gate not run (fixtures are seeded tiny instances);
  the multilingual Confucius4 checkpoint path is the recorded open
  item.
- Sampling is exercised only through the greedy argmax trajectory and
  the pinned single-call `_sample` distribution; a seeded non-greedy
  end-to-end run is not pinned.
- Streaming and batching are not ported (single utterance, single
  batch).
