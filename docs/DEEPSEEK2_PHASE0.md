# `deepseek2` Phase 0: pinned facts for the MLA bring-up

The fact-finding record for bringing up the `deepseek2` architecture
(DeepSeek V2 / V3 line: Multi-head Latent Attention plus fine-grained MoE).
Written before implementation, per `docs/NEW_MODEL.md` Phase 0, and updated
only with measured corrections. Every number here was read off one of the
sources below, not derived from a formula or remembered.

## Sources

1. **The witness GGUF's own header**, range-read (first 4 MiB) from
   `mradermacher/DeepSeek-V2-Lite-Chat-GGUF` at revision `9ee9f1f8f39f`,
   file `DeepSeek-V2-Lite-Chat.Q8_0.gguf`, on 2026-09-16. 377 tensors, 45
   metadata keys, `general.architecture = "deepseek2"`, 9.7 GiB for the
   Q4_K_M variant of the same repo (Q8_0 is ~17 GiB).
2. **The official config**, `deepseek-ai/DeepSeek-V2-Lite-Chat/config.json`
   (`model_type: "deepseek_v2"`, `DeepseekV2ForCausalLM`). The GGUF header
   and this config agree on every shared key.
3. **The reference layer graph**, llama.cpp
   `src/models/deepseek2.cpp` at `e5a8d43` (the commit the parity matrix
   cites for `qwen3`), which implements BOTH the absorbed (MLA) and the
   expanded (MHA) forms of this architecture.
4. **The HF reference**, `modeling_deepseek.py` `DeepseekV2Attention`
   (`deepseek-ai/DeepSeek-V2-Lite` repo), for the YaRN mscale semantics that
   llama.cpp's `[TAG_DEEPSEEK2_YARN_LOG_MUL_FIX]` comment also names.

## Why this checkpoint

The registry's `deepseek2` row was recognition-only with the witness pinned
to `unsloth/DeepSeek-V3-GGUF` (671B, unusable here). The bring-up needs a
checkpoint whose expert arithmetic this engine can stream (AGENTS.md Gotcha
36) and whose block types all have kernels. DeepSeek-V2-Lite-Chat (16B total,
2.4B active) is the smallest real `deepseek2` checkpoint with published GGUFs:

- one routed expert is `3 x 1408 x 2048 = 8,650,752` elements; at Q8_0
  (34 bytes per 32-element block) that is **9.19 MB per expert**. The slot
  cache at the default 16 slots over 26 MoE layers is `16 x 26 x 9.19 MB =
  3.56 GiB` -- streams, in the same neighbourhood as `qwen4_exp` (3.95 GiB).
  Fine-grained enough; Mixtral's dead end is avoided.
- `top_k_experts = 6`, which fits the GGUF MoE kernels' fixed phase-2 reduce
  width of 8 (`gpu::PHASE2_FIXED_SLOTS`), and leaves the pipelined prefill
  path legal at 12 or more cache slots (`2 * top_k`).
- The Q4_K_M variant of this repo is REFUSED as a witness despite its size:
  mradermacher's per-tensor calibration gives four layers Q5_K routed experts
  (`blk.3.ffn_down_exps`, `blk.4.ffn_down_exps`, and two shared-expert
  tensors), and the routed-pair kernel matrix has no Q5_K arm. The **Q8_0
  variant is uniform**: 269 tensors Q8_0, 108 tensors F32 (norms and
  routers), nothing else. That uniformity is why it is the witness.
- No tokenizer sidecars ship in the GGUF repo; the sidecar repo is
  `deepseek-ai/DeepSeek-V2-Lite-Chat` (`tokenizer.json`,
  `tokenizer_config.json`, `generation_config.json`).

## Config, as the file states it

| key | GGUF metadata key (prefix `deepseek2.`) | value | config.json key |
|---|---|---|---|
| hidden | `embedding_length` | 2048 | `hidden_size` |
| layers | `block_count` | 27 | `num_hidden_layers` |
| dense lead | `leading_dense_block_count` | 1 | `first_k_dense_replace` |
| q heads | `attention.head_count` | 16 | `num_attention_heads` |
| kv heads | `attention.head_count_kv` | 16 | `num_key_value_heads` |
| q head dim | `attention.key_length` | **192** | `qk_nope_head_dim 128 + qk_rope_head_dim 64` |
| v head dim | `attention.value_length` | **128** | `v_head_dim` |
| latent rank | `attention.kv_lora_rank` | **512** | `kv_lora_rank` |
| q lora rank | (absent) | 0 (lite: no q low-rank) | `q_lora_rank: null` |
| rope dim | `rope.dimension_count` | 64 | `qk_rope_head_dim` |
| rope theta | `rope.freq_base` | 10000 | `rope_theta` |
| rms eps | `attention.layer_norm_rms_epsilon` | 1e-6 (f32 9.99999997e-07) | `rms_norm_eps` |
| context | `context_length` | 163840 | `max_position_embeddings` |
| vocab | `vocab_size` | 102400 | `vocab_size` |
| experts | `expert_count` | 64 | `n_routed_experts` |
| top-k | `expert_used_count` | 6 | `num_experts_per_tok` |
| shared | `expert_shared_count` | **2** | `n_shared_experts` |
| expert ffn | `expert_feed_forward_length` | 1408 | `moe_intermediate_size` |
| dense/shared ffn | `feed_forward_length` | 10944 (dense); shared runs 2x1408 = **2816** | `intermediate_size` / shexp tensor shape |
| routed scale | `expert_weights_scale` | 1.0 | `routed_scaling_factor` |
| weights norm | (absent) | false | `norm_topk_prob: false` |
| gating | (absent) | softmax over ALL 64, top-6, NO renorm | `scoring_func: "softmax"`, `topk_method: "greedy"` |
| yarn | `rope.scaling.{type,factor,original_context_length,yarn_log_multiplier}` | yarn, 40, 4096, 0.0707 | `rope_scaling` |
| yarn betas | (absent) | llama.cpp format defaults 32 / 1 | `beta_fast 32, beta_slow 1` |
| head | `output.weight` present | untied | `tie_word_embeddings: false` |

The two derived floats, from the HF reference (`yarn_get_mscale(scale,
mscale) = 0.1 * mscale * ln(scale) + 1`, here with `mscale = 0.707`, read
off `yarn_log_multiplier * 10`):

- **rope magnitude (`mscale`, multiplies cos and sin)**:
  `1 + 0.0707 * ln(40) = 1.2608037774058554`
- **attention scale** (`1/sqrt(key_length)` times the mscale SQUARED, per
  `modeling_deepseek.py`'s `softmax_scale * yarn_get_mscale(factor,
  mscale_all_dim) ** 2`, and identical in llama.cpp's
  `kq_scale = mscale * mscale / sqrt(n_embd_head_k)`):
  `1.2608037774058554^2 / sqrt(192) = 0.11472138679292612`

Gotcha 24 risk flagged now: this is not a binary fraction, and the manifest
round-trips it through serde_json. A round-trip unit test lands with the
baseline; if the parsed value differs the fallback is a comparison fix in
`arch_validation`, never a different constant.

## Tensor inventory (per layer, GGUF names, dims fastest-varying first)

MoE layers 1..26 (26 of them):

| GGUF name | dims [in, out, ...] | notes |
|---|---|---|
| `attn_norm.weight` | [2048] | F32, RMS |
| `attn_q.weight` | [2048, 3072] | 16 heads x 192 (nope 128 + pe 64) |
| `attn_kv_a_mqa.weight` | [2048, 576] | down-proj: 512 latent + 64 pe |
| `attn_kv_a_norm.weight` | [512] | F32, RMS on the latent half |
| `attn_kv_b.weight` | [512, 4096] | up-proj: 16 x (k_nope 128 + v 128) |
| `attn_output.weight` | [2048, 2048] | 16 x 128 -> hidden |
| `ffn_norm.weight` | [2048] | F32, RMS |
| `ffn_gate_inp.weight` | [2048, 64] | F32 router (transcodes to INT8, qwen3moe precedent) |
| `ffn_gate_exps.weight` | [2048, 1408, 64] | routed, separate gate/up (qwen3moe shape) |
| `ffn_up_exps.weight` | [2048, 1408, 64] | routed |
| `ffn_down_exps.weight` | [1408, 2048, 64] | routed |
| `ffn_gate_shexp.weight` | [2048, 2816] | shared expert, fused 2 experts, RESIDENT |
| `ffn_up_shexp.weight` | [2048, 2816] | shared |
| `ffn_down_shexp.weight` | [2816, 2048] | shared |

Dense lead layer 0: same attention rows, plus plain `ffn_gate.weight` /
`ffn_up.weight` [2048, 10944] and `ffn_down.weight` [10944, 2048] (down is
Q8_0, gate/up Q4_K in the Q4_K_M file; all Q8_0 in the witness). Global:
`token_embd.weight` [2048, 102400] Q8_0, `output_norm.weight` [2048] F32,
`output.weight` [2048, 102400] Q8_0.

No q/k per-head norms, no attention biases, no attention output gate, no
softcap, no rope_freqs tensor, no MTP/nextn block in this checkpoint.

## The MLA design this port implements: absorbed, compressed cache

llama.cpp runs this file through its EXPANDED path (`is_mla` is false when
`attention.key_length_mla` is absent, which it is in every 2024-era GGUF):
it expands `kv_b` per token and caches full K (192/head) and V (128/head)
for all 16 heads -- 5120 halves per token per layer. That path reuses a
standard attention kernel but gives up MLA's memory result, which is the
reason the roadmap calls this family the high-leverage unlock (Kimi K2 and
GLM-4.7-Flash share the 512 latent rank; a compressed cache is what makes
them fit).

This port implements the ABSORBED form, which llama.cpp's own code carries
for newer files and which its comment states directly: "MLA with the
absorption optimization converts into MQA". Per layer:

- `q = attn_q @ x` -> 16 heads x 192; per head `q_nope = [0..128]`,
  `q_pe = [128..192]`.
- `kv_a = attn_kv_a_mqa @ x` -> 576; `c = rms(kv_a[0..512], attn_kv_a_norm)`,
  `k_pe = kv_a[512..576]`.
- **cache ONE row per token: `K = [c (512) ; k_pe (64)]` = 576 halves.**
  `V` is read as the first 512 halves of the same row. No V buffer is
  written or read. 1152 bytes per token per layer, 27 layers -> **243 MiB
  of KV at 8192 context** against 2160 MiB for the expanded form.
- rope (yarn, freq table from `compute::yarn_frequencies`, `mscale`
  above) applies to `q_pe` and `k_pe` ONLY: 32 pairs of the trailing 64
  dims. `q_pe`'s pairs sit at per-head offset 128 inside a 192-wide row,
  which no existing rope kernel expresses (they all rotate from element 0);
  a port-local kernel is listed below.
- absorb: per head `q' = W_k_h @ q_nope_h` where `W_k_h` is kv_b's rows
  `[h*256, h*256+128)`, giving `[512]`; the score against a cached token is
  then a single 576-dot: `q'.c + q_pe.k_pe`.
- attention: MQA over the shared row, online softmax, per head
  `attn_h = sum_t p_t * c_t` -> `[512]`.
- v-combine: per head `out_h = W_v_h @ attn_h` where `W_v_h` is kv_b's rows
  `[h*256+128, h*256+256)`, giving `[128]`; concat heads -> `[2048]`, then
  the ordinary `attn_output` GEMV.

Scale `0.11472138679292612` is applied to the 576-dot (the kernel applies
it to the reduced score, like `attention.metal` does). Decode kernel
inventory, all port-local with CPU references in `crates/compute`
(`mla.metal` + `mla.rs`; no Swift original exists -- Swift has no MLA):

1. `mla_kv_norm`: RMS-norm the first 512 halves of a 576 row in place.
2. `mla_rope_q_pe`: rope the trailing 64 of each 192-wide head (32 pairs,
   stride 192, window offset 128), from the yarn freq table + mscale.
   `k_pe` itself is a plain contiguous 64-wide row, so it uses the existing
   `rope_neox_freqs` kernel with a byte offset of 1024.
3. `mla_absorb_q` (q8_0 weights): per head a `[512 x 128]` GEMV over kv_b
   rows; 16 threadgroups. NOT 16 dispatches of the resident GEMV kernel:
   that would add ~860 encodes per token across 27 layers.
4. `mla_attention_decode`: MQA, score dim 576, v read from the K row's
   first 512. Online softmax, fp32 accumulation, scale on the reduced dot.
   Existing `encode_attention_decode` is NOT reusable: one head_dim drives
   Q, K and V (`576 > kAttnMaxHeadDim 512`), and out would be 576-wide.
5. `mla_v_combine` (q8_0): per head a `[128 x 512]` GEMV; 16 threadgroups.
6. K-row write: GEMV into scratch, norm in place, rope pe in place, then a
   576-half copy into the cache row (`scalar_mul_fp16` at scale 1.0 as the
   copy, or a dedicated kernel if the signature does not fit).

Prefill: the chunked driver runs the SAME per-token dispatches (the llama
MoE prefill pattern: one commit for all M tokens' attention, then the
pipelined routed path), because a chunked T-row absorbed-attention kernel
is a follow-up optimization, not a bring-up requirement. Routed experts in
prefill ride the per-token streamer path (no GGUF batched pair exists).

## One decoder layer, as pseudocode

```
h = x
a = rms(h, attn_norm)                      # eps 1e-6
q   = a @ Wq.T                             # [16*192]
kva = a @ Wkv_a_mqa.T                      # [576]
c   = rms(kva[0:512], attn_kv_a_norm)
kpe = yarn_rope(kva[512:576], pos)         # 32 pairs, mscale
q_nope, q_pe = split(q)                    # per head [128], [64]
q_pe = yarn_rope(q_pe, pos)
q_abs_h = W_k_h @ q_nope_h                 # per head -> [512]
cache.write_k(pos, [c ; kpe])              # 576 halves, V = c
s_h(t) = (q_abs_h . c_t + q_pe_h . kpe_t) * 0.114721...
p = softmax(s_h over t <= pos)
attn_h = sum_t p_t * c_t                   # [512]
out_h  = W_v_h @ attn_h                    # [128]
h = h + Wo @ concat(out_h)                 # attn_output, [2048]
f = rms(h, ffn_norm)
if layer == 0:
    h = h + silu(f@Wg.T)*(f@Wu.T) @ Wd.T   # dense SwiGLU, width 10944
else:
    r = softmax(f @ Winp.T)                # over ALL 64, fp32 logits
    top6 = top_k(r, 6); w = r[top6]        # NO renorm; scale 1.0
    moe = sum_i w_i * expert_i(f)          # silu(gate)*up -> down
    sh  = (silu(f@Wgs.T)*(f@Wus.T)) @ Wds.T  # fused shared expert, width 2816
    h = h + moe + sh
```

Final: `rms(h, output_norm)`, then logits = `h @ output.T` (untied Q8_0,
vocab 102400). `produce` writes logits; the sampler softmaxes once
(Gotcha 16).

## Tokenizer, dialect, and stop facts

- The GGUF carries the published template inline: BOS + `User: {content}
  \n\n` / `Assistant: {content}<eos>` turns, system messages appended bare
  (no prefix), generation prompt `Assistant:` (no trailing space).
- Special tokens: bos `<|begin_of_sentence|>` id 100000 (fullwidth bars in
  the real table), eos `<|end_of_sentence|>` id 100001, pad = eos.
  GPT2 BPE, pre `deepseek-llm`, 102400 rows.
- **The existing `ChatDialect::Deepseek` does NOT fit**: its probe fires on
  the fullwidth `<|User|>` mark, but `resolve_deepseek` then REQUIRES
  `<think>`/`</think>` (a V3-era requirement; V2-Lite's table has neither)
  and hardcodes `vocab_size: 129_280` against this table's 102400. The
  bring-up adds a V2 dialect (probe + resolver + plain renderer, no think
  pair, vocab from the table) rather than weakening the V3 resolver, whose
  fixture tests pin the forced `</think>` suffix.
- No tool-call markup, no reasoning channel: passthrough decode. The
  structured decoder gets no arm for this dialect.

## Memory accounting, computed from shapes (Phase 5 baseline)

- KV: 1152 bytes/token/layer x 27 x context -> 243.0 MiB at 8192.
- Expert slot cache: 16 slots x 26 layers x 9.19 MB = 3.56 GiB.
- Resident: embed 212.5 MiB + head 212.5 MiB + per-layer attention 13.9 MiB
  x 27 = 376.5 MiB + shared experts 17.5 MiB x 26 = 455.8 MiB + dense lead
  FFN 68.1 MiB + norms (negligible). Total resident ~1.33 GiB (the whole
  install minus the streamed expert table; install on disk ~17.5 GB).
- Process baseline plus scratch: target ceiling for the oracle row is the
  sum rounded up with ~5% headroom, set after the first measured run.

## Kernel / capacity cross-checks recorded at Phase 0

- `top_k = 6 <= PHASE2_FIXED_SLOTS = 8`: the GGUF phase-2 reduce is fine
  unmodified. Phase-1 dispatch is runtime-sized.
- `MAX_STREAMED_EXPERTS` (16, the vendored INT4-affine pair) is not the
  binding ceiling here: GGUF-sourced blobs dispatch the moe_gguf kernels.
- Slot floor for the pipelined prefill path: `2 * top_k = 12`; the default
  16 slots qualify for 2 banks.
- Q8_0 executable everywhere it appears: resident GEMV, embed lookup,
  routed pair, head. F32 appears only on norms and routers (norms transcode
  to BF16, the router transcodes to INT8).
- YaRN: `compute::yarn_frequencies(64, 10000, 40, 4096, 32, 1)` gives the
  position-independent table; the `mscale` scalar is passed to the rope
  kernels separately, so the 0.707 parameter needs its own path (a new
  `RopeScalingConfig` field; zero-fallback preserves gpt-oss exactly).

## The numerics gap: FOUND AND CLOSED (2026-09-17), and the instruments that found it

The flow's per-position logits now reproduce llama.cpp's on the IDENTICAL
Q8_0 bytes: layer-1 cache rows match at corr 0.99998+ at EVERY position 0-14,
the per-layer residual grid matches to worst-layer corr 0.9998, full-vocab
logits match at corr 0.984-0.9999 with argmax identical on 14 of 15 prompt
positions (the two soft spots are near-ties under this port's f16 GPU path
against llama's f32 CPU), and greedy AND sampled generation are coherent with
EndOfTurn reached. Instruments, now permanent: `logit_dump.rs`'s
`TURBOSPARK_DSV2_INSTALL_DIR` arm, `scripts/llamacpp_logits.c` with
rope-override argv, and the `~/dsv2ref/` harnesses (a llama.cpp eval-callback
per-layer dump plus the numpy confrontations that consumed it). What was
verified CORRECT piece by piece against independent implementations: the
embed lookup, both input norms, the q and kv_a GEMVs, the install weights
against the GGUF source bytes, the latent-norm, the absorb algebra (exact
algebraic identity with llama.cpp's expanded form), the fused score scale
(0.11472138679292612 -- reproducing llama's own attention output at corr
1.0000 from exact inputs), and the whole pos-0 stack end to end.

Three REAL bugs were found and fixed along the way; the third closed the gap:

1. **The absorb was not transposed** (fixed): q'_h must read
   `sum_i W_uk[i][j]·q_nope[i]` -- reduction over the nope ROWS -- not the
   per-row GEMV every other kernel here uses. A self-consistent fixture
   (kernel vs CPU mirror of the same wrong indexing) passed while both were
   wrong; only the real install's garbage output exposed it.
2. **The FFN scratch overflowed** (fixed): the dense lead's 10944-wide and
   the shared expert's 2816-wide gate/up runs were written into hidden-sized
   (2048) or kv-sized (8192) buffers, clobbering whatever the allocator
   placed next -- including the yarn frequency table. Position 0 was immune
   (any table gives the identity rope at angle 0), which made the corruption
   look like a position-dependent rope bug. Also the source of run-to-run
   NONDETERMINISM that initially read as a race.
3. **The MLA rope paired the WRONG ELEMENTS** (fixed, the root cause): the
   q_pe and cache-tail rotations paired `(i, i + dim/2)` -- the half-split
   layout the port's other rope kernels use -- where ggml's
   `ggml_rope_cache_init` pairs CONSECUTIVE elements `(2i, 2i + 1)`.
   Invisible at position 0 (all angles zero, any pairing is the identity),
   small-but-present at position 1, and degrading smoothly with position
   (pair 0 is an EXTRAPOLATED YaRN pair at 1.0 rad/position, so the two
   conventions differ by a full radian there): exactly the observed
   row-by-row decay of corr 1.0000 (row 0) through 0.9692 (row 1) to ~0.89
   (row 14). `compute::mla_rope_window` + `mla_rope_q_pe` now pair
   consecutively, the cache tail goes through the same kernel (one "head"
   whose window starts at kv_lora), and
   `rope_pairs_consecutive_elements_not_split_halves` pins the convention
   with hand-derived expected values that the split form CANNOT satisfy.

Two false leads are recorded so nobody re-walks them:

- **The yarn ramp direction is NOT the bug, and "fixing" it IS one.** ggml
  interpolates the SLOW (long-wavelength) pairs and extrapolates the fast
  ones -- `ramp` is 1 at the fast end, mixed toward `theta_extrap`. A draft
  read the YaRN paper's prose the other way, flipped `yarn_ramp`, and saw an
  apparent improvement -- measured through the scratch-overflow corruption
  and the mis-paired rope, i.e. through two broken components. Reverted to
  the ggml-identical form; `yarn_ramp`'s doc comment now carries the trap.
- **llama.cpp's deepseek2 K cache is NOT the compressed `[c ; k_pe]` row.**
  On CPU (no flash attn) it de-absorbs: 16 kv heads x 192
  (`[k_nope ; k_pe]` each), V 16 x 128. The compressed row appears only in
  views. Comparing a 576-wide window against this port's compressed row
  produced a bogus "latent corr ~0" that sent the bisect through a wasted
  lstsq recovery of `c` from the wrong tensor. The eval-callback's node-name
  log (`cache_k_l1 (view)` ne=[3072,256] plus `(permuted)` views) is what
  settled the layout.

## Frozen release gates (2026-09-17)

The family's three owed rows, frozen on the real Q8_0 install
(`~/.turbospark/models/dsv2lite-16b.gturbo`), Apple M4 Max, AC, release,
16 expert-cache slots, protocol at the family's own 8,192/1,024 window
(`real_model_params`' `Deepseek2` arm, confirmed by the oracle run: all
three cases stop endOfTurn, `long-synthesis` prefills 3,153 tokens):

- **Quality** (`crates/bench/tests/dsv2_quality_gate.rs`): reference-answer
  perplexity **14.3688**, greedy digest
  `8a0e247593e9487004eeb152723edea46f1e645626d4b496945eef598fab44e9`,
  sampled digest
  `d212a8365d648b285f8de6fd19225b802eb0e23d6ab9c0f1789e97b72db99d56`.
  Two fresh processes agreed on every digit and hex character. NO assistant
  prefix: V2's answer slot is plain prose after `Assistant:`. The 8-slot
  constrained digest equals the 16-slot one; the constrained arm's
  throughput ratio read 0.73x in one process and 1.51x in the other (the
  16-slot decode itself moved 13.1 to 24.4 tok/s on identical work), which
  is machine-state spread on a shared machine, recorded in the gate's row
  rather than explained.
- **Memory** (`crates/bench/tests/dsv2_memory_oracle.rs`): session peak
  **4,103 MiB** at the 8,192 window, ceiling frozen at 4,300 (measured
  +5%), replay growth +0.05 MiB. The peak is slot cache (3,828 MiB) + KV
  (243 MiB) + ~33 MiB: the ~1.33 GiB resident core is NOT counted, the
  same phenomenon AGENTS.md Gotcha 40 measured on a dense install, now
  observed on a streamed MoE one. The counted figure is a leak sentinel,
  not a capacity number.
- **Throughput**: decode **17.533 / 11.285 / 5.722 tok/s** on
  short-explanation / medium-review / long-synthesis, oracle floor frozen
  at 4.0 tok/s (0.70x the slow case). The slow case is slow for two
  stacked reasons: it decodes at 3,754-token context, and it gets there
  through a 375.57 s sequential per-token prefill (8.4 tok/s prefill) --
  the descope list's unbuilt T-row absorbed prefill kernel is the single
  biggest known throughput lever this family has.

## Deliberately not done at bring-up (the descope list)

Named in the ROADMAP entry and detailed once in `DEVIATIONS.md`'s
`deepseek2` section, which stays the list's only home: split `wk_b`/`wv_b`
GGUF files, `q_lora_rank > 0`, a batched (T-row) absorbed prefill kernel,
pure-576 KV (the never-written V buffer), and the V3/GLM/Kimi checkpoints
themselves.
