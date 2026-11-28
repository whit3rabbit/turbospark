# Deviations from the reference

Each entry is a place where the port does not reproduce the reference
bit for bit, and why that is accepted.

## VAE

- Weight-norm fold: the port folds `g * v / ||v||` once at load
  (`WnFold::Div`, as the DAC family does). PyTorch computes
  `v * (g / ||v||)` on every call. The two differ by rounding only;
  measured parity is in `README.md`.
- Snake: `exp(alpha)` and `1 / (exp(beta) + 1e-9)` are precomputed at
  load instead of per call. Same f32 operations, same order.
- Posterior sampling takes caller-supplied standard-normal noise.
  Upstream draws it with torch `randn` inside `latent_dist.sample()`.
  Tests replay recorded noise; the runtime will own its RNG.
- Decode tiling follows upstream's overlap-discard rule; the window is a
  caller choice (upstream picks it per device). At 512 / 64 it is
  bit-identical to untiled on the real checkpoint. Encode is untiled
  (upstream chunks 15 or 30 s with 2 s overlap).
- Device path: transposed convs run as `stride` kernel-2 phase convs.
  Same products as the direct form, different summation order.

## Condition encoder and DiT

- One request, no padding. The reference sorts packed sequences by mask
  and builds padding masks; with every mask set that is a plain concat
  `[lyric, timbre, text]`, which is what the port does. Padded or
  batched inputs are not supported.
- One reference clip, so one timbre token. Upstream's multi-reference
  unpack (one-hot matmul) is not ported.
- RoPE tables follow HF default (`1 / theta^(2i/d)` in f32), not the
  crate's `ops::rope_tables` (`theta^(-2i/d)`); the two differ by
  rounding only, and HF's form is the reference.
- Window masks use `f32::MIN`, as the reference's `finfo.min`; a window
  that covers the whole sequence builds no mask (identical result).
- `proj_in` / `proj_out` (kernel = stride = patch) run as linears over
  reordered weights. Same products, different summation order.
- The condition embedder runs once per request instead of per step
  (same values; upstream also caches cross-attention K/V per step).
- `time_embed_r` gets the literal `0.0`: the sampler always passes
  `timestep_r = t`, and `t - t` is `+0.0` for finite `t`.

## Text encoder and prompts

- `TextEncoder` refuses over-length ids; `PromptTokenizer` truncates as
  upstream does (and reports it), so the pipeline never hits the refusal.
- `extract_caption` hand-ports upstream's caption regex (no regex crate).

## Pipeline

- Initial noise: SplitMix64 plus Box-Muller, not torch's generator.
  Parity gates replay the reference draw.
- DCW ports the Haar basis only (upstream's default); other PyWavelets
  bases are not supported. The transform is written per pair instead of
  as a filter bank (same products, rounding-level differences).
- One request, no batching, no LM planner, no reference audio yet.

## Tokenizer and cover

- Single-quantizer FSQ only (`fsq_input_num_quantizers = 1`, the shipped
  config); other counts are refused at load. The residual loop's divide
  by `L^0 = 1.0` is skipped (exact).
- The 5-frame pad comes from the head of the silence latent, as the
  reference does; its attention mask is computed upstream but never
  reaches the tokenizer, so it is not ported.
- Cover sampling switches condition at `int(steps * strength)`; with
  strength 1.0 the fallback is never built, matching the reference.
