# Voxtral TTS 4B

Port of `mlx_audio/tts/models/voxtral_tts/` at mlx-audio
[`e1b19b9054bf163f5d812221a54fcc346f1890e9`](https://github.com/Blaizzy/mlx-audio/tree/e1b19b9054bf163f5d812221a54fcc346f1890e9/mlx_audio/tts/models/voxtral_tts)
(MLX 0.32.3). Spec task 43 (`voxtral_tts`, repo
`mlx-community/Voxtral-4B-TTS-2603`).

## Architecture

A Mistral-geometry LM backbone over the shared [lm engine](../../lm/README.md)
(llama math, dim 3072, 26 layers, head_dim 128, 32/8 heads, rope_theta 1e6,
TRADITIONAL/interleaved rope, tied embeddings, no biases) conditioned by
voice presets: the speech-request prompt carries `[AUDIO]` token positions
whose text-table embeddings are REPLACED by per-voice embedding rows
(`_build_input_embeddings`; voice tables live in
`voice_embedding/<name>.safetensors`). The prompt ends with
`[BEGIN_AUDIO]`, so its last hidden state already predicts the first
frame.

Every step the last hidden state produces one 80 ms frame (37 codes at
24 kHz):

- the semantic code by masked argmax over the padded
  `(8192 // 128 + 1) * 128 = 8320`-way head; index 0 (empty) and the
  padded tail take a -1e9 mask, index 1 (end) stays live and ends the
  run;
- the 36 acoustic FSQ codes by an 8-step Euler flow matching head with
  classifier-free guidance (alpha 1.2, zero conditional fallback): the
  acoustic, sinusoidal time (cos,sin halves), and LM embeddings stack
  into a 3-token bidirectional transformer (no positional encoding);
  the reference batches the conditional and unconditional rows into one
  forward pass, the port runs them as two single-row passes (rows are
  independent);
- the frame's 37 codes (each with the +2 special-token offset) sum
  their per-codebook rows of ONE shared embedding table (padded
  semantic 8320 + padded acoustic 768 = 9088 rows at the real sizes)
  and feed the next LM step through the KV cache.

The code grid decodes to 24 kHz audio through the codec decoder
(`audio_tokenizer.py`, decoder-only): an EMA semantic codebook
(`embedding_sum / max(cluster_usage, 1e-8)`) concatenated with the
finite-scalar acoustic quantizer (`2i/(size-1) - 1`), then four
alternating conv/transformer stages - weight-normed causal convolutions
(folded `g * v / ||v||`, replicate `edge` pads, causal transposed
convolutions for strides > 1 trimmed to `T * stride` samples) and ALiBi
sliding-window transformer layers (full-width q/k RMSNorm, layer scale,
windows [2, 4, 8, 16] decoder-reversed) - and the reflect-padded output
projection to the 240-sample patch (8x upsample total, 1920 samples per
frame).

## Scope and deviations

- Single batch, non-streaming: the reference streaming overlap-add
  chunking (`stream=True`, 16-frame context decode) is not ported; the
  full code grid decodes in one pass, which the reference identifies as
  the identical non-streaming path.
- Caller-owned text tokenization: `_encode_text` needs mistral-common's
  tekken speech tokenizer (`SpeechRequest`), so `generate` takes the
  prompt token ids; the fixture pins the reference layout (text ids,
  the voice's `[AUDIO]` run, the trailing `[BEGIN_AUDIO]`).
- Unknown voice names are REFUSED; the reference silently falls back to
  no voice conditioning when the file is missing, which would silently
  degrade output.
- The reference accepts `temperature`, `top_k`, and `top_p` but never
  uses them: generation is deterministic given the per-frame flow draw.
  No sampler is involved (semantic argmax plus quantized flow), so the
  shared `tts/sampling.rs` filters are not exercised by this family.
- Flow noise comes from a [`FlowNoise`] provider: recorded-draw replay
  for the fixtures, the crate-local threefry RNG (which matches the
  reference draw stream under 1e-6, like the voxcpm family) for
  standalone runs.
- `forward_batch` (frame-parallel decode_one_frame) is not ported; the
  generate loop is per-frame.
- The checkpoints load under the raw consolidated Mistral names
  (`tok_embeddings`, `layers.N.attention.wq`, `feed_forward.w1`,
  `attention_norm`, `mm_audio_embeddings.audio_codebook_embeddings.*`)
  through an additive `Weights::with_name_map` remap in the shared lm
  module; sanitized HF-style names (`embed_tokens`,
  `self_attn.q_proj`, `mlp.gate_proj`, `lm_head`) load unchanged.

## Verification

Fixture protocol identical to the codec families:
`tools/gen_voxtral_tts_fixtures.py` builds seeded tiny instances of the
real reference modules (a 2-layer LM at hidden 32 with 4 query / 2 kv
heads and head_dim 8, a 2-layer acoustic head at dim 32, a 4-stage codec
decoder at dim 32 with a 12-sample patch, 96 samples per frame) and
records, per case in `testdata/tts_voxtral_tts/<case>/` (config.json +
model.safetensors + traces.json, digests in manifest.json):

- `generate/`: voice-conditioned run, the semantic end token fires at
  step 3 (3 frames).
- `length/`: voice-conditioned run to max_tokens (5 frames), the end
  token never wins.
- `codec/`: standalone codec decode of a random code grid (8 frames)
  with every per-stage hidden state.

The checkpoint saves under the raw consolidated Mistral names, so the
Rust name remap is fixture-gated. The stop row is tuned into a two-way
race (all non-nominal rows shrunk, the nominal row pointed along the
running hidden trajectory), and the end-row direction is chosen offline
against the recorded hidden states so the stop step is exact with a
0.02 dot-margin. Draw seeds are searched until every frame's final
Euler state clears an FSQ quantization boundary by >= 5e-3 (measured
minimum margin 1.3e-2), so the Rust flow's f32 noise (about 1e-6 after
the 7 Euler steps) cannot flip a code; codes are then EXACT and the
exactness gate means something. The generator verifies the recording
replay against a fresh `model.generate` and a second replay on the
Python side (identical waveforms and draw streams).

The Rust tests reload each case through `Model::open` and replay the
draws. Measured parity (gates 2e-5 unless noted): prompt embeddings
3.0e-8, per-step LM hidden states 2.0e-6, flow velocities 1.3e-6 and
Euler states 4.8e-7, codes EXACT, codec stages at or below 1.4e-6
(quantizer exact), waveforms 1.4e-7. The threefry draw stream matches
the recorded MLX draws under 1e-6.

Bugs the fixtures caught during bring-up: the codec transposed
convolutions store the kernel with the CONVOLUTION INPUT on the first
axis (`(C_in, C_out, K)` for the checkpoint parametrization), so the
folded weight feeds the ops kernel directly with swapped axis roles
(the reference's `transpose(1, 2, 0)` targets MLX's `(C_out, K, C_in)`
convention); the reflect pad is numpy-style (edge sample not repeated)
while the decoder convs use `edge` replicate, and both pad per channel;
the layer-scale tensors are bare arrays (no `.weight` suffix); and
`mx.round` rounds ties to even, which the FSQ quantizer needs.

Open item: the real-checkpoint gate. The fixtures are seeded tiny
instances; `mlx-community/Voxtral-4B-TTS-2603` is a 4B download and has
not been run through this port. The loader expects the checkpoint
layouts documented above (raw consolidated or sanitized HF names) plus
`tekken.json`-derived prompt ids from the caller.
