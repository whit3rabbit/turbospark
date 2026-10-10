# WavVAE (MegaTTS3 Latent Audio Codec)

WavVAE is the continuous latent codec behind MegaTTS3: a SEANet encoder
down to 25 Hz, a diagonal Gaussian over a 32-channel latent, and a
HiFiGAN-style generator back to 24 kHz. The MegaTTS3 diffusion transformer
generates in this latent space, and the prompt latent of a cloned voice is
this encoder's output.

## Upstream References and URLs

- **Reference implementation**: `tts/modules/wavvae/{decoder,encoder}`
  (`WavVAE_V3`) in [bytedance/MegaTTS3](https://github.com/bytedance/MegaTTS3)
  (Apache-2.0). There is no mlx-audio port, so parity is against the PyTorch
  classes, unmodified, through `tools/gen_wavvae_fixtures.py`.
- **Paper**: [MegaTTS 3](https://arxiv.org/abs/2502.18924); the codec follows
  [WavTokenizer](https://arxiv.org/abs/2408.16532).

## Model Overview

```text
mono PCM @ 24 kHz
   |
   v
SEANet encoder (weight-norm convs, ELU, reflect padding, extra right
padding so the last frame is complete):
  conv k7 (1 -> 32)
  -> 5 x [residual block (k3, k1, 1x1 shortcut), ELU, strided conv (k = 2r)]
     with reversed ratios [2,4,4,5,6]; width doubles each stage (32 -> 1024)
  -> 2-layer LSTM (1024) with a skip around the stack
  -> ELU -> conv k7 (1024 -> 512)                  25 Hz
   |
   v
proj_to_z (512 -> 64) -> mean = first 32, logvar = last 32 (clamped to [-30, 20])
   |   latent = mean + exp(0.5 * logvar) * eps       (time-major [frames, 32])
   v
proj_to_decoder (32 -> 320) -> Upsample x4 (320 -> 160, 100 Hz)
   |
   v
Generator: reflect pad, conv k7 (160 -> 2048),
  4 x [Upsample r in 5,4,4,3 (halves width) + ResBlockMRFV2Inter (k 3/7/11, averaged)],
  leaky ReLU, reflect pad, conv k7 (128 -> 1), tanh        24 kHz
```

`Upsample(r)` applies `x + sin(x)` first, then sums a nearest-upsample branch
(leaky ReLU, reflect pad 3, conv k7) and a transposed-conv branch
(`kernel = 2 * max(r, 5)`, `padding = max(r, 5) - r / 2`,
`output_padding = r % 2`).

## Checkpoint profiles

Weights come from `tools/convert_megatts3.py` (MLX layout, weight norm kept
unfolded; the loader folds it, as `dacvae` does).

| Profile | Source | Encoder | Notes |
|---|---|---|---|
| `official` | `ByteDance/MegaTTS3` `wavvae/decoder.ckpt` | No | Decode only. ByteDance withheld the encoder "for security issues"; cloning needs pre-extracted latents. |
| `community` | `mrfakename/MegaTTS3-VoiceCloning` `wavvae/model_only_last.ckpt` | Yes | Third-party encoder, credited to ModelScope `ACoderPassBy/MegaTTS-SFT`. Origin and license are unverified (the ModelScope page could not be fetched). Opt-in only. |

`WavVae::has_encoder` reports which one was loaded; `encode` returns
`SpeechError::Unsupported` on a decoder-only file.

## Rust API

```rust
let vae = WavVae::open(dir)?;                 // dir/model.safetensors
let post = vae.encode(&samples_24k)?;         // Posterior { mean, logvar, frames, z_dim }
let latent = post.sample(&eps)?;              // eps: caller's N(0,1) draw, [frames, 32]
let audio = vae.decode(&latent)?;             // frames * 960 samples
```

Upstream encoding is stochastic (`posterior.sample()`), so the draw belongs to
the caller: tests inject a recorded `eps`, and the MegaTTS3 facade will supply
a seeded generator. `post.mean` is the deterministic latent.

## Verification

| Gate | Evidence |
|---|---|
| Tiny fixture (committed, `testdata/wavvae/`) | Two clips (100 and 37 samples: extra-padding and short-reflect paths). Posterior, sampled latent, decoder input trace, and decoded audio (sampled and mean latent) agree with PyTorch to about 5e-7; the test tolerance is 2e-5. |
| Real checkpoint, community encoder (env-gated) | 1.0 s synthetic clip through the converted weights: encoder mean 3.3e-6, logvar 1.9e-5, decoder audio 4.6e-6 against PyTorch (CPU, f32). Set `TURBOSPARK_WAVVAE_MODEL` and `TURBOSPARK_WAVVAE_GOLDEN`; goldens come from `gen_wavvae_fixtures.py --real-src`. |
| Mutation checks | Dropping `bias_hh`, leaving `output_padding` at the DAC default, and removing the Encodec extra padding each fail the tiny-fixture test. |

Not gated: perceptual quality of reconstruction, real-speech clips, f16/bf16
profiles, and speed. The CPU path is a portable reference; the 1024-channel
generator stages dominate cost.

## Limits

- Fixed 24 kHz mono and the shipped geometry; `WavVaeConfig` is validated and
  bounded, but only the shipped geometry and the tiny test geometry have
  checkpoints.
- Batch encode/decode only; no streaming state.
- The unused `Audio2Mel`, `Downsample`, and discriminator code in the upstream
  files is not ported.
