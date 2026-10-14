# Shared LM backbones

Port of the `mlx_audio/lm` subset the mlx-audio TTS families import
(vendored from mlx-lm upstream), pinned to mlx-audio 0.5.7 at
[`e1b19b9054bf163f5d812221a54fcc346f1890e9`](https://github.com/Blaizzy/mlx-audio/tree/e1b19b9054bf163f5d812221a54fcc346f1890e9/mlx_audio/lm)
with MLX 0.32.3. 34 of the 40 unported TTS families import this library, so
the backbones live once here and the families consume them; everything is
portable single-batch f32 with no runtime or Metal dependency.

## Contents

| Module | Reference file | Notes |
|---|---|---|
| `rope.rs` | `lm/models/rope_utils.py` | default/linear/llama3/yarn/longrope/proportional/mrope; custom-freqs types use the denominator convention (`angle = position / freqs`), verified empirically against `mx.fast.rope` |
| `cache.rs` | `lm/models/cache.py` | `KVCache` and `RotatingKVCache` (ring buffer with the reference trim/rotate semantics) behind the `LayerKv` trait plus the `PromptCache` mix; `BatchKVCache`/`ArraysCache`/`CacheList` wait for a family that needs them |
| `engine.rs` | shared math of `llama.py`/`qwen2.py`/`qwen3.py` | RMSNorm blocks, GQA attention (causal or sliding-window masks, optional rope), SwiGLU MLP, tied/untied head, `forward_hidden` for audio adapters; the `Attention`/`Mlp` parts are shared with the sibling backbones; additive `load_with_embed_table` + `embed_table` for families that rename or reuse the text table |
| `llama.rs` | `lm/models/llama.py` | llama defaults (no biases, kv heads default to query heads, optional head_dim, layer_types) |
| `qwen2.rs` | `lm/models/qwen2.py` | forced q/k/v biases, `head_dim = hidden / heads`, no q/k norms |
| `qwen3.rs` | `lm/models/qwen3.py` | per-head q/k RMSNorm, all dataclass fields required |
| `gpt2.rs` | `lm/models/gpt2.py` | fused biased `c_attn` QKV, tanh-gelu MLP, learned token/position tables, LayerNorm pre-norm, tied head; no rope; additive constructors for the family patches (`from_config_without_tables`, `from_config_without_tables_at_root`) |
| `cohere2.rs` | `lm/models/cohere2.py` | sliding-window layers (traditional rope, rotating ring, window mask) alternating with global layers (no rope, dense cache), parallel attn+MLP on one shared norm, `logit_scale` head |
| `bailing_moe.rs` | `lm/models/bailing_moe.py` | fused `query_key_value`, optional q/k RMSNorm, dense-first layers then softmax/sigmoid-routed SwitchGLU experts with shared expert and `expert_bias`, `routed_scaling_factor`, norm/tied head options |
| `switch_layers.rs` | `lm/models/switch_layers.py` | `SwitchGLU` over `[expert, out, in]` stacks; `SwitchMLP` and the quantized switch linears wait for a family |
| `falcon_h1.rs` | `lm/models/falcon_h1.py` | stage 2, for arktts: every layer carries Mamba2 + GQA attention + MLP on shared pre-norms (`h + mamba(ln(h)) + attn(ln(h))`, then the FF block), the reference `Model.sanitize` port (shape-gated muP multiplier fold + conv layout, so raw torch-layout and pre-converted saves load through one path), tied head scaled by `lm_head_multiplier / embedding_multiplier`, and `forward_stack`/`load_without_embed_table` as the arktts composite-embedding seam |
| `ssm.rs` | `lm/models/ssm.py` | the Mamba2 selective scan as one recurrence (`dt = clip(softplus(dt + dt_bias))`, `S = exp(A dt) S + dt x B`, `y = <S, C> + D x`); the reference's Metal initial/update kernels and the chunked `ssm_attn` form all reduce to it; masked/length-truncated variants stay unported (single-batch falcon builds neither) |

Not yet ported (stage 1 remainder, spec task 80): the sampler/logits
processor subset of `sample_utils.py` (top_p/min_p/XTC filtering, repetition
penalty, logits processors) and the batched cache variants
(`BatchKVCache`, `ArraysCache`, `CacheList`) when a family needs them.

## Verification

Fixture protocol identical to the codec families: `tools/gen_lm_fixtures.py`
builds seeded tiny instances of the Python backbones (2-4 layers, hidden 32,
4 query heads, 2 kv heads, vocab 96) and records prefill logits, three
cached decode-step logits, a cacheless pass, and the post-norm hidden state
into `testdata/lm/<case>/` (config.json + model.safetensors + traces.json,
digests in manifest.json). The Rust tests reload each case through the same
`Weights::open` path a real checkpoint takes and compare against a 2e-4
absolute gate; all sixteen cases pass.

Per-backbone coverage:

- `llama_*` (7 cases): every rope construction rides a llama case.
- `qwen2_default`, `qwen3_default`: their flavor paths.
- `gpt2_default`: LayerNorm pre-norm, biased fused QKV, tanh gelu, learned
  positions, tied head, kv-heads defaulting to query heads (the reference
  never reads the field).
- `gpt2_indextts_embeddings`: the indextts consumption shape (`wte`/`wpe`
  bypassed, float embeddings in, tied head out), exercising the
  `Input::Embeddings` path and `without_position_embeddings`.
- `cohere2_default`: window 4 over a 7-token prompt (window mask exercised
  in prefill), three sliding layers plus one global layer, then decode
  steps that rotate the ring.
- `cohere2_streaming`: a second two-token chunk arrives after the window
  is full, exercising the `_update_concat` trim path and the
  clamped-offset window mask, with decode steps continuing on the rotated
  ring. This case caught a real growth bug: the ring's per-head zero-row
  append recomputed the buffer length inside the loop, so growing a
  multi-head buffer past a retained trim walked the insert point past the
  end.
- `bailing_moe_default`: one dense layer then two MoE layers, 4 experts
  top-2, shared expert, q/k norms, untied head, nonzero `expert_bias`,
  `norm_topk_prob`, and `routed_scaling_factor = 2.0` (the reference
  multiplies it unconditionally).
- `falcon_h1_default`, `falcon_h1_untied` (stage 2): two hybrid layers
  with the published non-unity muP multipliers, so the sanitize fold is
  load bearing: the saved checkpoint is the RAW one (torch
  `[out, 1, kernel]` conv layout, multipliers unfolded) and the Python
  traces are computed through the reference `Model.sanitize`, which the
  Rust loader must reproduce. The Mamba conv and selective-scan states
  are pinned after the prefill and after every decode step (`mamba_states`
  in the traces). One tied and one untied head case.

Measured parity on the frozen fixtures: every diff (prefill, chunk,
decode, cacheless, hidden, falcon conv/ssm states) is at or below 4.2e-6,
47x inside the 2e-4 gate; the cohere2 streaming case sits near 1e-7 and
the falcon cases measure 1.43e-6 worst (prefill, decode, cacheless,
hidden, and the step-pinned Mamba states).

Mutation checks run against these fixtures: disabling the sliding-window
mask moves cohere2 prefill by 9.7e-2, dropping the tanh gelu moves gpt2 by
3.4, and dropping the router scaling multiply moves bailing_moe by 1.7, so
the fixtures pin all three. For falcon_h1, skipping the sanitize fold (or
double-folding a converted save) moves the prefill logits by
O(embedding_multiplier) - roughly 1.3 on this fixture - and perturbing the
selective-scan decay moves the pinned states well past the gate. Router
slot ORDER is not pinned by the decoder (top-2 sums are exact under
permutation); it is pinned by dedicated `MoeGate` unit tests, and the ring
buffer order is pinned by the `RotatingKvCache` unit tests (attention
output is invariant to permuting whole (k, v) pairs, so decoder fixtures
alone cannot catch order bugs).

Regeneration (deterministic, no network). The checkout's `mlx_audio/lm`
differs from the venv's installed mlx-audio release in exactly these files,
so the generator must import from the checkout; run it from the checkout
root with PYTHONPATH set:

```sh
cd ../mlx-audio
PYTHONPATH="$PWD" ../.venv-mlxaudio/bin/python \
    ../turbospark/crates/audio/tools/gen_lm_fixtures.py
```

## Gotchas found during verification

- MLX `mx.fast.rope` treats a passed `freqs` vector as the per-pair angle
  DENOMINATOR: `Llama3RoPE`'s stored vector reproduces
  `cos(position / freqs[0])`, and an `inf` denominator leaves the tail
  untouched (which is exactly the partial-rotary behavior).
- Appending a `[heads, new, dim]` block flat onto a
  `[heads, total, dim]` cache corrupts the layout: the new rows land inside
  the last head's region. Inserts are per head. Attention output is
  invariant to permuting whole (k, v) pairs, so the decoder fixtures alone
  cannot catch pure order permutations; `cache.rs` pins the canonical
  MLX order with dedicated tests.
- Python `Attention` applies q/k norms before rope and before the cache
  update; the keys stored in the cache are the roped ones.
- `RotatingKVCache` never trims on the FIRST update (`_update_concat`
  direct-assigns when empty), so a prompt longer than the window leaves
  more rows than `max_size` in the buffer until the next update; the
  window mask accounts for this by clamping the causal offset to
  `min(window - 1, cache offset)`. Once a single-token update runs, the
  buffer is physically rotated and holds exactly the last
  `min(offset, window)` positions, which is why the reference passes no
  decode mask at all when `max_size == window` (cohere2's only mix).
- `mx.argpartition`'s top-k order is unspecified; the bailing aggregation
  sum over the slot axis is exact under permutation for top-2 and within
  tolerance above. The Rust router emits slots in descending score order
  (ties to the lower expert index) per the workspace reduction-order
  invariant.
- `SwitchLinear` weights are `[expert, out, in]` stacks, not HF `[out, in]`
  pairs; the down projection is applied through an explicit transpose
  (`gather_mm` semantics), which is where a flat re-reading of the stacked
  tensor goes wrong.
- gpt2 checkpoints exist in two layouts: HF `conv1d` stores `c_attn`/
  `c_fc`/`c_proj` transposed and the reference `Model.sanitize` fixes them,
  while MLX-side saves (and these fixtures) are already `[out, in]`. The
  Rust loader expects the sanitized layout; family loaders own the
  transpose. The reference sanitize also forces the `model.` prefix onto
  every tensor, so the loader uses those names explicitly.
- The pinned `mlx_audio/lm` checkout differs from the venv-installed
  mlx-audio 0.5.7 release in `cache.py`, `base.py`, `cohere2.py`,
  `bailing_moe.py`, `switch_layers.py`, `activations.py`, and
  `rope_utils.py`; fixtures must be generated against the checkout (see
  the regeneration command above), not site-packages.
