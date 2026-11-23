# Inkling 2 Mini (`inkling2_loop`) Phase 0

Target: `thinkingmachines/Inkling-2-Mini`, revision
`ed40e7da625f594fc9c4990f5f66be24873c22d7`. Apache 2.0, ~26B physical / 3B
active per pass, 1M context, text + image in, text out.

STATUS: **supported on three verified catalog installs (2026-11-22).** The family enum, the HF and MLX intakes
and a Metal decode flow have landed. The flow is greedy-matched to the CPU
reference and llama.cpp on real installs (Stage A below). Three catalog rows
are `verified` (2026-11-22): `inkling2-mini` (the original at `ed40e7da`),
`inkling2-mini-mlx4` and `inkling2-mini-lamq54`, each pulled by alias here
and chatted. `inkling2-mini-mlx4` is featured at rank 9 (after the
existing eight); at the app's default 4,096-token fit context it sits under
"Not checked on this Mac yet", since its peak was measured at 8,192. The memory oracle is frozen for all three
(streamed at 16 slots, 8,192 context: 2,662, 2,663 and 1,587 MiB; see
"Memory oracle" below). The Phase 6 quality gates are frozen too
(2026-11-22; see "Quality gate" below).
GGUF installs through its own intake (GGUF sections below). LookAtMyQuant
packed releases install and decode with their codes native (`docs/LAMQ.md`,
5.4 GB release: 19/19 and 183/232 top-1 against the BF16 install; the loss
is in the release's weights, not the runtime). Every install type is reachable from the CLI
(`turbospark model pull-local`, `pull --repo`) and the app (Discover >
Install from Files), and chats through `turbospark run` (2026-11-21, after
the padded vocabulary rows stopped being -inf: see step 4 below).

Sources (formulas are transcribed, not derived): Hugging Face
`transformers/models/inkling2/modeling_inkling2.py` (layer math; it has NO loop
code, so it describes the non-looped Inkling 2) and llama.cpp
`src/models/inkling2.cpp`, `inkling2-loop.cpp`, `conversion/inkling2.py` (loop,
controllers, checkpoint conventions).

## Config (`config.json -> text_config`)

hidden 2304, 24 layers, vocab 201024 (unpadded 200058), 24 q heads / 8 kv
heads, head_dim 128 (SWA identical), `d_rel` 16, `rel_extent` 1024, sliding
window 1024, `local_layer_ids` = every layer except 5, 11, 17, 23 (those four
are full attention), sconv kernel 4, final logit softcap 30, 128 routed
experts, top 8, no shared expert, expert width 1152, `route_scale` 8,
`mlp_gate_softcap` 4, `mlp_up_softcap` 25, `qk_l2_clip_threshold` 22.627417
(= 16 sqrt 2), `rel_l2_clip_threshold` 8, `log_scaling_n_floor` 8192,
`log_scaling_alpha` 0.1, attention-residual block size 6,
`loop_config`: core [6, 18), 3 passes, gate rank 64, fixed loop.
`eos_token_id` 200006. Embed and unembed are separate (never tied).

## Checked against the checkpoint header (1,244 tensors, 822 `model.llm.*`)

Read from the safetensors header at the pinned revision. `config.json` matches
the section above field for field. Facts the transcription missed:

Confirmed against llama.cpp `src/models/inkling2.cpp`, `inkling2-loop.cpp` and
`conversion/inkling2.py` at `29b52d398`:

- `model.llm.embed_norm.weight` (BF16 `[2304]`) is a plain-`w` RMS norm on
  the token embedding, applied to TEXT tokens only (never to image
  embeddings). It is the first op of the flow.
- No RoPE at all: the relative bias carries position. Distance is
  `qpos - kpos`, biased only for `0 <= d < extent` (else 0), under the usual
  causal and sliding-window masks.
- Static conv taps: the SWA `k_sconv.weight` / `v_sconv.weight`
  (`[1024, 1, 4]`) and both post convs (`[2304, 1, 4]`) are `[ch, taps]` with
  tap 3 on the current row, which is the kernel's layout. Full-attention
  `static_w` is `[4, 1024]` indexed by LAG (0 = current), so the repack writes
  `w[ch, j] = static_w[3 - j, ch]` (the converter's `flip(0).T`).
- `weight_proj` `[32, 2304]` produces dynamic taps indexed `head * 4 + lag`
  (lag 0 = current), which is the kernel's `dyn[head * taps + lag]`.
- Conv state per cache slot: K at channel offset 0, V at 1024, the attention
  out-projection at 2048, the FFN output at 4352 (6656 channels x 3 rows).
- Controllers `controllers.{attn,k,v,mlp}` are Q, K, V and FFN, in that order.
  When a message is present, a SWA layer's Q, K and V inputs are each
  controlled and RMS-normed SEPARATELY, so they differ. Without a message,
  one norm feeds all three.
- Message queries for core layer `6 + i` are
  `[q[2i], kv[2i], kv[2i+1], q[2i+1]]` (Q, K, V, FFN). The bank for pass
  `p > 0` is the previous pass's blocks AFTER the first core block, plus its
  output.
- `w13` rows: even rows are gate, odd rows are up.
- Expert weights are normalized with eps 1e-20 and scaled by `route_scale`.
- F32, not BF16: every `attn_res.*` query (`attn_queries.{4,10,16,22}` are
  `[3, 2304]`, feeding full layers 5, 11, 17, 23), `output_query`,
  `middle_cycle_cross_pass_{queries,kv_queries}`, the router bias
  `mlp.gate.bias` `[128]`, and the expert `amax` and denominators.
  `encode_inkling2_depth_mix` takes BF16 queries, so either the repack narrows
  them (and the parity test must cover that) or the kernel learns F32.
- `q_l2_clip.scale` is `[24, 128]` (one per q head: stride `dim`),
  `rel_l2_clip.scale` `[24, 16]`. `gate_attn` is `[3072, 2304]`: a gate per q
  channel, not per head. `wr_du` is `[384, 2304]` (24 heads x `d_rel` 16).

Reference implementation: llama.cpp master since
[ggml-org/llama.cpp#32694](https://github.com/ggml-org/llama.cpp/pull/32694)
(commit `29b52d398`). Apply
[#32704](https://github.com/ggml-org/llama.cpp/pull/32704) on top. Without it,
llama.cpp copies the core layers into their loop slots before attaching the
NVFP4 per-expert scales, so passes 2 and 3 run unscaled experts (the PR reports
perplexity 130.43 against 7.6531). Our expert blobs carry the outer scales, so
every loop slot of a physical layer reads the same scales. A loop-pass test
must still assert that.

## CPU reference against llama.cpp (2026-11-20, M4 Max)

`turbospark_compute::inkling2::forward` (f32, weights decoded from the HF
checkpoint) against llama.cpp `29b52d398` plus #32704, running CPU-only on a
BF16 GGUF from its own converter. On Metal, 52 GB of weights ran out of GPU
memory. The prompt was 19 tokens of plain prose, "The capital of France is Paris. The
capital of Japan is Tokyo. The capital of Italy is". The ids are identical
from `tokenizer.json` and from llama.cpp's tokenizer.

| measure | result |
|---|---|
| top-1 agreement | 19 / 19 (the last position predicts " Rome" on both) |
| top-5 overlap | 5 / 5 at every position |
| KL(llama.cpp, ours), mean / max | 4.2e-4 / 1.9e-3 nats |
| max abs logit difference | 1.13 (position 8), most positions 0.03 to 0.5 |

Not bit-exact, by construction. The GGUF rounds every dequantized expert
weight to BF16 (the F32 outer scale makes the product inexact), and ggml's CPU
BF16 matmul also rounds activations to BF16. Both compound over 48 layer
applications. The low-entropy positions agree to about 1e-6. The larger
values sit at high-entropy positions (after a period or newline), which looks
like rounding, not a wiring error. That is a judgement, not a proof.

NOT exercised by 19 tokens: `tau` (exactly 1 below position 8191), distances
past the 1024 bias extent, and the sliding-window cut. The 12,550-token
Metal gate under step 3 below covers all three.

Reproduce: `crates/repack/examples/inkling2_reference_logits.rs` for ours,
and a 60-line `llama_get_logits_ith` dumper against the llama.cpp build.
Our reference runs 3.5 to 10 s per token. llama.cpp took 55 s for the batch
on CPU with a 19.4 GB peak resident set.

## Layer flow (one physical layer `il`, position index `ip`)

```
inpSA = x
a = (il > 0) ? mix(blocks, x, attn_res_query[il]) : x
      SWA: 1 query -> one input;  full: 3 queries -> separate q, k, v inputs
loop pass > 0: each input_j = control(input_j, msg[j], ctrl_j)
norms: SWA attn_norm; full q/k/v_input_norm (all plain-w RMS, eps 1e-6)
q = wq a_q ; k = sconv(wk a_k) ; v = sconv(wv a_v) ; r = wr a_q
  SWA sconv: static depthwise taps, scale 1/sqrt(K+1)
  full sconv: static + per-token per-kv-head taps from weight_proj(a_k|a_v),
              scale 1/sqrt(2K+1)
q = l2clip(q * q_scale, 16 sqrt2) ; k = l2clip(k, 16 sqrt2)
v = rms_norm_no_gamma(v) ; r = l2clip(r * r_scale, 8)
full only: q *= tau(pos), r *= tau(pos)
bias[h, qpos, kpos] = (r_h @ rel_logits_proj)[qpos - kpos], 0 outside [0, extent)
sink logit per q head (sink_proj a_q), added to the softmax denominator with
  a zero value; softmax scale 1/sqrt(head_dim)
o = attn * sigmoid(gate_attn a_q) ; o = sconv_post(wo o)      (scale 1/sqrt(K+1))
if ip % 6 == 0: blocks.push(inpSA)      else: x = inpSA + o
m = mix(blocks, x, mlp_res_query[il]) ; loop pass > 0: m = control(m, msg_ffn)
m = mlp_norm(m)
router: sigmoid(W m) ; select top 8 of sigmoid + e_score_correction_bias,
        weights = unbiased scores / sum * 8, dispatched in RANK order
expert: gate,up from interleaved w13 rows ; softcap(gate,4)*sigmoid(gate)*softcap(up,25)
y = sconv_mlp(sum) ; x = x_after_attn + y
```

`mix(blocks, cur, queries)`: stack blocks plus `cur`, RMS-norm each (no gamma),
softmax of `query . state / sqrt(h)`, weighted sum of the NORMED states.
Output: `x = mix(blocks, x, output_query)`, final norm, unembed, softcap 30,
slice to the unpadded vocab.

## Loop

Layers 0..5 run once (prelude), core 6..17 runs three times, 18..23 once.
Every pass restarts from the prelude output and `blocks = prelude blocks`.
Pass `p > 0` gets messages from the previous pass: a bank of its blocks after
the first core block plus its output, mixed by 4 per-layer message queries
(order q, k, v, ffn). Control:
`gate = down @ [rms(cur); rms(msg)]`, `cur' = cur - tanh(forget@gate) * cur +
tanh(gain@gate) * msg`. Each pass has its OWN KV and conv state: 24 + 2 x 12 =
48 cache slots, matching the MTP `target_layer_ids` 5, 23, 41, 48.
Checkpoint `middle_cycle_cross_pass_queries` and `..._kv_queries` are
`[24, 2304]`; layer `6+i` takes `[q[2i], kv[2i], kv[2i+1], q[2i+1]]`.
`attn_res.attn_queries.{i}` feeds layer `i+1`; `mlp_queries.{i}` feeds layer
`i`.

## Checkpoint quantization

Only routed experts are quantized. Everything else is BF16 (attention, router,
controllers, 2 x 926 MB embed/unembed).

| Layers | Weights | Scales |
|---|---|---|
| 0-19 (20) | NVFP4 `U8` packed: w13 `[128, 2304, 1152]`, w2 `[128, 2304, 576]` | `E4M3` per 16 (`_weight_scale` `[128, 2304, 144]` / `[.., 72]`), plus F32 `amax[128]` / `outer_scale_denominator[1]` |
| 20-23 (4) | MXFP8 `F8_E4M3`: w13 `[128, 2304, 2304]`, w2 `[128, 2304, 1152]` | `U8` E8M0 per 32 along the row (`[128, 2304, 72]` / `[.., 36]`; 255 = NaN) |

`hf_quant_config.json` gives MXFP8 a `weight_block_shape` of `[32, 32]`, but
the stored scales are one per 32 elements of each row, so the row transcoder
reads them as stored. The unembed is its own BF16 group with an F32 forward
output. An earlier draft of this table put MXFP8 on the full-attention layers
5, 11, 17 and 23. The checkpoint header and the release post (the last 4
layers in MXFP8) both contradict that.

Neither is accepted by `validate_quant` (affine INT4/8, group 64).

**Decided: keep both native** (not a requant to affine INT4, which would cost
the MXFP8 layers 4 bits for an unmeasured quality loss). Only routed experts
need kernels, since nothing resident is quantized.

The repack transcodes each expert row losslessly
(`turbospark_compute::inkling2::{nvfp4_row_to_blocks, mxfp8_row_to_blocks}`):

- NVFP4 into ggml `block_nvfp4` (36 B per 64), so routed rows decode through
  the GGUF NVFP4 `nvfp4_elem`. Only the nibble order moves. Scale bytes with
  bit 7 set or `0x7F` are refused, because ggml's decoder ignores bit 7 and
  reads `0x7F` as zero.
- MXFP8 into 33 B per 32 (the E8M0 byte, then 32 E4M3 bytes). NaN exponents and
  codes are refused.

Expert blob contract (`crates/gpu/src/shaders/moe_inkling2.metal`): a 16-byte
record of f32 outer scales first (`gate_s`, `up_s`, `down_s` offsets 0, 4, 8;
w13's `amax / denominator` for gate and up, w2's for down; 1.0 for MXFP8), then
gate, up and down rows. The outer scale stays separate because the gate and up
softcaps come after the dot product, so it cannot be folded into the routing
weight, and folding it into the E4M3 scales would re-round them.

The decode pair is `encode_inkling2_moe_phase1/2` (`crates/gpu/src/moe_inkling2.rs`).
Phase 1 is the softcapped SwiGLU with the caps as arguments. Phase 2 reduces
`top_k <= 8` slots in rank order and adds NO residual, because `sconv_mlp`
runs on the expert sum first. `tests/moe_inkling2_parity.rs` builds experts in
the HF layout, decodes them with the reference, and runs the transcoded bytes
on Metal, at Mini dims with top 8. Mutants that drop the down outer scale, put
the sigmoid on the capped gate, or swap the caps each fail it. Not yet built:
batched prefill for these types (decode only).

## Memory accounting (from shapes)

Expert, scales included: NVFP4 (w13 2304 x 1152 B + 2304 x 144 B, w2 2304 x
576 B + 2304 x 72 B) = 4.3 MiB; MXFP8 (2304 x 2304 + 2304 x 72, 2304 x 1152 +
2304 x 36) = 7.8 MiB. Whole table: 20 x 128 x 4.3 + 4 x 128 x 7.8 = ~15 GiB.
Resident BF16 (attention and conv ~54 MB per layer x 24, embed and unembed
2 x 926 MB, router, controllers, queries) = ~3.5 GB. Those two add up to the
19.5 GB shard. An earlier draft of this section said 6.8 + 4.1 GB of experts
and 9 GB resident, both wrong.

Expert residency (the existing streamer, `docs/EXPERT_RESIDENCY.md`): one
cache slot across all 24 layers is ~117 MiB, so 16 slots pin ~1.9 GiB and 32
pin ~3.7 GiB. Streamed, the Mini should fit in ~6-8 GB plus KV. Mapped
residency needs resident + table + the 10 GiB reserve (~28.5 GiB) within
Metal's working set (~27 GiB on a 36 GB Mac), so expect `auto` to refuse it
here.

Decode bytes per token (estimate): BF16 attention 54 MB x 48 applications =
2.6 GB, experts 8 x 4.3 MiB x 48 = ~1.6 GB, unembed 0.93 GB. That is ~5 GB, so
the BF16 attention weights, not the experts, set the decode ceiling.

Conv state per slot: (2 x 1024 + 2 x 2304) x 3 rows. KV:
8 heads x 128 x 2 x 2 B = 4 KiB per token per slot; 8 full-attention slots
(layers 5, 23 once, 11 and 17 three times) = 32 KiB per token, 32 GiB at 1M
context; 40 SWA slots ring at 1024 rows (~160 MB fixed). The shared
`KvCacheManager` still allocates its own 24-layer cache that this family
never reads; dropping it is open work.

## Speed and memory by install (2026-11-22)

tslab `profile` development readings, not frozen rows. Apple M4 Max 36 GB,
AC power, worker `568bbaed` built at `e2929518`, experts resident (8 slots
reported), two interleaved passes of three rounds per install and workload,
about 11 GB of swap in use throughout. Medians of the clean runs; experiment
`inkling2-install-bench2`.

| Install | Decode, `short-explanation` (64-token prompt, 128 decode) | `long-synthesis` (3,252-token prompt, 64 decode): prefill, decode | Peak footprint | Clean runs (decode, long) |
|---|---|---|---|---|
| `inkling2-mini-gturbo` (18 GB) | 34.2 tok/s (32.8 to 36.3) | 32.5, 27.7 tok/s | 17.7 GiB | 6/6, 5/6 |
| `inkling2-mini-mlx4-gturbo` (16 GB) | 45.6 tok/s (45.6 to 45.7) | 37.9, 33.5 tok/s | 17.7 GiB | 3/6, 4/6 |
| `inkling2-mini-lamq54-gturbo` (6.9 GB) | 30.5 tok/s (30.5 to 30.7) | 27.7, 24.3 tok/s | 8.4 GiB | 6/6, 3/6 |

Summary records, pass 1 then pass 2: original `1795373901693-51813-8`,
`1795375394542-9293-8` (decode) and `1795374341570-58473-8`,
`1795375844051-16326-8` (long); MLX `1795373472063-9085-8`,
`1795374931307-61609-8` and `1795373849753-15090-8`,
`1795375337020-68125-8`; LAMQ `1795372934446-55330-8`,
`1795374381290-1940-8` and `1795373428052-60787-8`,
`1795374880655-7446-8`. Dirty runs came from Spotlight and other foreign
CPU, not from these processes. Prefill is sequential on every install
(`--chunk off`; a chunked profile is refused), so prefill tok/s tracks
decode. The MLX build shares the original's packed NVFP4 and MXFP8
experts, so its peak matches; the gain is its 4-bit dense weights. Energy
per decode token on the short workload: 1.30 J (original), 0.80 J (MLX),
1.02 J (LAMQ). An earlier experiment, `inkling2-install-bench`, ran a stale
worker and a refused long workload; do not read it.

## Memory oracle (2026-11-22)

`crates/bench/tests/inkling2_memory_oracle.rs`,
`inkling2_mlx4_memory_oracle.rs` and `inkling2_lamq54_memory_oracle.rs`,
one target per install. Streamed at the protocol's 16 slots (the opener
pins `Streamed`; `auto` would make a 36 GB Mac resident), 8,192 context,
4,096 budget, two sessions each on AC, then a third asserting the frozen
rows.

| Install | Peak (two sessions) | Decode, slowest to fastest case | Ceiling / floor |
|---|---|---|---|
| `inkling2-mini` | 2,651 and 2,662 MiB | 14.51 to 17.98 tok/s | 2,800 MiB / 10.0 |
| `inkling2-mini-mlx4` | 2,663 and 2,661 MiB | 17.94 to 20.52 tok/s | 2,800 MiB / 12.5 |
| `inkling2-mini-lamq54` | 1,573 and 1,587 MiB | 16.51 to 18.95 tok/s | 1,680 MiB / 11.5 |

Every case stopped at end of turn with the same token counts in both
sessions, and a warm replay grew +0.00 MiB. The peak is the slot cache
(16 x 24 layers x the expert stride: 4.49 MB for NVFP4 and MXFP8, 1.67 MB
for LAMQ) plus KV (about 0.27 GB on the 8 full slots, 0.16 GB of SWA ring)
plus about 0.6 GB that both kinds of install show alike; the resident core
is mapped, not counted. LAMQ drives short-explanation and medium-review
only: on long-synthesis it reasoned to the 4,096 budget (the original
stops at 2,070 tokens, MLX at 1,943), and a larger budget does not fit the
window over its 3,252-token prompt. That reads as the release's quality
loss, not a footprint question. With experts resident (`auto` here) the
peak is about 17.7 GiB and decode about twice as fast; that is a tslab
reading, not a frozen row ("Speed and memory by install").

The third session, asserting the frozen rows, passed for all three. One MLX
attempt failed its floor first: medium-review read 11.5 tok/s against 19.6
and 19.5 in calibration, the same 2,391 tokens in 207 s instead of 122,
while `mediaanalysisd` ran at about 300% CPU; the other two cases and the
peak were normal. Rerun once the daemon was idle, it read 19.0 tok/s and
passed. The floor was not loosened: a contended host reads as a floor
failure here, so check `ps` before reading one as a regression.

## Quality gate (2026-11-22)

`crates/bench/tests/inkling2_quality_gate.rs`,
`inkling2_mlx4_quality_gate.rs` and `inkling2_lamq54_quality_gate.rs` on
`quality_common`. The reference answer is scored in the text segment after
`<|message_model|><|content_text|>` (the template writes no generation
prompt; the model opens its own turn), and the gate asserts that prefix
encodes as those two special tokens. Streamed, 16 slots, 8,192 context.

| Install | Reference perplexity | 8-slot digest = 16-slot | 8-slot decode |
|---|---|---|---|
| `inkling2-mini` | 4.7024 | yes | 0.89x |
| `inkling2-mini-mlx4` | 4.8026 (+2.1%) | yes | 0.87x, 0.89x |
| `inkling2-mini-lamq54` | 5.7391 (+22%; 5.7382 before the 2026-11-23 GEMV move) | yes | 0.95x, 0.96x |

Two fresh processes per install agreed on the perplexity's last digit and
on both digests' every hex character; a third asserted the frozen rows. The
perplexity orders the installs as the KL measurements do (MLX's 4-bit dense
weights cost a little, the LookAtMyQuant release clearly more). It is no
ranking against other families: each template places the answer elsewhere.

## Remaining work, in dependency order

Stage A: reachable and correct (decode only)
1. DONE: `ModelFamily::Inkling2Loop` (persisted `inkling2_loop`) and the
   `inkling2_mini()` baseline. Every exhaustive match answers it: GGUF intake
   refuses it, runtime open refuses it ("no decode flow yet"), steering and
   mapped residency are off, and lab and bench have provisional rows.
   `num_layers` is 24 (physical): the expert streamer opens one layout entry
   per checkpoint layer, and the flow maps each of the 48 applications back
   to its physical layer. DECIDED: the Inkling-only values (loop core and
   passes, gate rank, d_rel and extent, clips, log scale, MLP softcaps, res
   block, unpadded vocab) go in an `inkling2.json` sidecar beside the
   manifest, not in `ArchConfig`, which has about 50 constructor sites. The
   generic fields (softcap, window, mask, sconv taps, route scale, router
   function) are in `ArchConfig`. Swift's per-family UI lists name the
   family (2026-11-22): `ModelFamilyVisuals` gives it its own tile (the
   family letter on a brown category hue; no vendor mark is bundled) and
   `ModelFeatureDescriptor.knownMoEFamilies` marks the catalog rows as MoE
   before they are installed.
2. DONE: repack (`crates/repack/src/inkling2_checkpoint.rs`,
   `examples/inkling2_local_install.rs`). `parse_inkling2_config` turns the real
   `config.json` (fixture) into exactly `inkling2_mini()` plus
   `Inkling2Config::mini()` (`model_io::inkling2_config`, the `inkling2.json`
   sidecar). It refuses biases, a shared expert, a V norm with a gamma, a
   variable loop and SWA heads that differ. Experts go to
   `packed_experts/layer_NN.bin` in the blob contract above, with sub-tensors
   `outer` (F32 [4]), `gate`, `up` and `down` (dtype `inkling2_nvfp4` or
   `inkling2_mxfp8`). The resident region holds every other `model.llm.*`
   tensor: the router is widened to F32, `static_w` is rewritten to
   `{k,v}_sconv.weight` [ch, 1, taps], and vision and MTP are left out.
   `manifest.quant.routedExpert.scheme` is `inkling2_nvfp4_mxfp8`
   (`model_io::INKLING2_ROUTED_SCHEME`, admitted by `validate_quant`).

   Found on the way: the manifest writer never wrote `routedScalingFactor`,
   though validation reads it (the `swigluLimit` gotcha again), so an install
   whose route scale was not 1.0 could never validate. It is written now. And
   `1/sqrt(128)` is 2 ULP from the `128^-0.5` literal the other baselines use,
   so the parser uses `powf(-0.5)`.

   Real intake (2026-11-20, M4 Max, local snapshot): 97 s, 9.3 GB peak RSS,
   18 GB install (3.23 GB resident; 575 MB per NVFP4 layer file, 1.05 GB per
   MXFP8 one). The example decodes 60 gate, up and down rows over 5 layers
   (NVFP4 and MXFP8) from the blobs, and each equals the HF reference decode
   bit for bit. The outer records match `amax / denominator`. Unit tests cover
   a synthetic checkpoint read back through the loaders, plus refusals.
   Mutants (gate on the odd rows, no `static_w` flip, w13's outer scale on
   down) each fail. Not yet: a synthetic install that opens through
   `RealForwardRunner::open` (open refuses the family until step 3).
3. DONE (decode, greedy-checked): `crates/runtime/src/families/inkling2/`.
   `RealInkling2State` owns 48 cache slots of KV and conv tails (the shared
   `KvCacheManager` is per physical layer and is used only for the position),
   the per-layer expert offsets rebuilt from `packed_experts/layout.json`
   (`gate_s` / `up_s` / `down_s` point into the outer record), and its own
   `RoutedBlobsBuffer` reflected from `moe_inkling2`. Open skips the generic
   routed-layout setup for this family and admits its F32 resident tensors by
   family, each checked by dtype and size at build. Per application: depth
   mix, controllers, norms, projections, convs straight into the slot's KV
   row, L2 clips with tau, rel-bias plus sink attention, output gate, o-proj
   conv, block-residual step, MoE input mix, then the F32 router GEMV
   (`minimax_router`). The host selects with `compute::inkling2::route_sigmoid`
   (the reference's own selection) and streams the experts for the PHYSICAL
   layer through the existing `PreadExpertStreamer`. Prefill is the decode
   path per token; chunked prefill is refused by name.

   Gate (2026-11-20, M4 Max, the 18 GB install, the 19-token prompt above;
   `crates/runtime/examples/inkling2_logits.rs`):

   | comparison | top-1 | mean KL | max KL |
   |---|---|---|---|
   | Metal vs CPU reference | 19/19 | 1.2e-4 | 5.1e-4 |
   | Metal vs llama.cpp | 19/19 | 4.9e-4 | 1.9e-3 |

   The run reproduces its logits bit for bit. Rough per-token times on this
   run were 5.4 s for the first token (pipeline compiles, cold expert reads),
   then 120 to 730 ms as the slot cache and page cache warmed, about 7 tok/s
   at the end. That is an observation from one dirty run (other sessions
   active), not a protocol measurement. NOT yet: a synthetic-install test
   that opens through `RealForwardRunner::open` in CI (the gate above needs
   the real install), sampled generation, and the chat template.

   SWA ring: the 40 sliding-window slots hold `min(max_ctx, 1024)` rows and
   position p sits at row `p % rows`; the 8 full slots hold `max_ctx` rows.
   `inkling2_attention_decode` takes the ring size (buffer 12). The 19-token
   logits are bit-identical before and after the ring. Kernel parity covers a
   wrapped window (2,601 positions, ring 1024, poisoned unused rows) and a
   full slot at 1,500 positions past the extent.

   Long-context gate (2026-11-21, M4 Max, 12,550 tokens of prose, every 64th
   position plus the last 16 kept, 211 rows): Metal vs llama.cpp CPU-only.
   This covers tau (positions 8,192 and up), distances past the 1024 extent,
   the SWA cut, and the ring after 12 wraps.

   | comparison | top-1 | mean KL | max KL |
   |---|---|---|---|
   | Metal vs llama.cpp, 265 tokens | 265/265 | 5.7e-4 | 1.8e-2 |
   | Metal vs llama.cpp, 12,550 tokens | 209/211 | 2.2e-3 | 2.3e-2 |

   Both top-1 misses are top-2 swaps where llama.cpp's own margin is 0.12 and
   0.085 logits. Mean KL by quarter of the context is 2.0e-3, 2.3e-3, 2.2e-3,
   2.3e-3: no growth with length, so no drift from the ring or tau. There is
   no step at the window edge either: rows below position 1024 average
   1.2e-3 and rows from 1024 to 2,600 also 1.2e-3. The gap to the 265-token
   mean is a different prompt, not a different code path; that reading is a
   judgement.

   llama.cpp trap: with `n_gpu_layers 0`, op offload still sends large-batch
   `MUL_MAT_ID` to Metal, and every row came back NaN (first NaN at
   `blk.9.ffn_gate_up_exps`, the second loop pass). Set
   `llama_context_params.op_offload = false` for a CPU-only reference.
   Reproduce with `inkling2_logits <install> <ids> <out> <max ctx> 64` and a
   sparse dumper that writes the same rows.
4. PARTLY DONE: tokenizer dialect (`ChatDialect::Inkling2`, 2026-11-21:
   no BOS, stop `<|content_model_end_sampling|>`, thought span
   `<|content_thinking|>` to the next `<|end_message|>`; tslab now opens the
   install). DONE (2026-11-21): CLI reachability. `turbospark run` failed
   on every Inkling install with "score vector must not contain a non-finite
   entry": the flow wrote -inf into the padded vocabulary rows (200058 to
   201023) and `selection::select` refuses non-finite scores. The logits
   harness drops those rows, so no gate saw it. They are now the most
   negative finite f16 (`mask_padding`, unit test mutation-checked). After
   it, BF16, LAMQ 5.4 GB, MLX 4-bit and a local IQ2_M GGUF each answer a
   one-sentence factual prompt correctly and stop at end of turn. Open:
   the CLI prints the thought span run-on with the answer, and a cold BF16
   prefill took 88 s for 27 tokens (others 2 to 5 s; not investigated).
   DONE (2026-11-22): the probe and pull for the original checkpoint and MLX
   conversions. They take the same whole-file route as LAMQ and GGUF
   (`crates/catalog/src/inkling2_repo.rs`, kinds `Hf` and `Mlx`): the
   repository is downloaded whole at a pinned commit (the DSpark drafter
   `mtp.safetensors` is copied into `<install>/dspark/`, and an MLX row
   fetches it through its `mtp` source), then converted by
   `write_inkling2_install(_from)`. A LAMQ release also ships an HF
   `config.json` declaring `inkling2_loop` and a `steering.safetensors`, so
   its manifest is read first; checking the checkpoint first took the
   release for a checkpoint (the catalog network guard caught it, and the
   fixture now ships a config so `tests/inkling2_repo.rs` does too).
5. DONE: long-context check (above). Catalog rows landed at `runs`
   (2026-11-22), then the memory oracle and the quality gates (below), and
   the rows moved to `verified`.

Stage B: speed and memory (after A's gates, each with a kill switch)
- DONE (2026-11-21): GGUF name table (`crates/repack/src/inkling2_gguf_names.rs`),
  from llama.cpp's `conversion/inkling2.py` and `gguf-py` tables. Both
  conventions map: llama.cpp master (`ggml-org/Inkling-2-Mini-GGUF`, arch
  `inkling2_loop`, `loop_*_ctrl_*`, per-layer `loop_msg_query`) and the older
  converter (`unsloth/Inkling-2-Mini-GGUF`, arch `inkling2`, `loop_{attn,k,v,mlp}_*`,
  global `loop_res_query`). Each tensor carries the converter's transform:
  transposed `rel_logits_proj`, squeezed conv weights (the full-attention
  `static_w` arrives already flipped, matching the install's rewrite), gate
  rows before up rows in `ffn_gate_up_exps`, per-expert NVFP4 `.scale`.
  `examples/inkling2_gguf_names.rs` checks a header's tensor list against an
  install: both real headers map all 646 text tensors with nothing missing or
  extra. Not done: a GGUF intake. Experts there are ggml NVFP4 re-encoded by
  the converter (not the source bytes) or re-quantized types (Q8_0 in the
  ggml-org Q8_0 file, Q5_K/Q8_0 on the MXFP8 layers of unsloth's), which the
  Inkling MoE kernels do not read.
- DONE (2026-11-21): GGUF expert kernels. Header inventory of the published
  files (ggml-org Q8_0 and IQ4_NL, `inkling2_loop`, 704 tensors; eight
  unsloth UD mixes, the older `inkling2`, 734 tensors with per-expert F32
  `.scale` on 20 layers): gate-up (the fused `ffn_gate_up_exps`) ships Q8_0,
  NVFP4, IQ4_NL, IQ4_XS, IQ3_XXS, IQ3_S, IQ2_S, IQ2_XS, IQ2_XXS, IQ1_S, Q3_K,
  Q4_K or Q5_K; down ships Q8_0, NVFP4, IQ4_NL, Q2_0, MXFP4 or Q5_0 (the
  expert width 1152 is not a multiple of 256, so no 256-block type can be a
  down); attention, embedding and head use Q4_K to Q8_0, IQ3_XXS, IQ3_S and
  IQ4_XS, which `encode_gemv_any` already runs. The GGUF routed pairs
  (`moe_gguf.metal`, `moe_gguf_ext.metal`) gained an Inkling mode behind
  function constants 112-114 (`moe.metal`: outer scales from a record at
  `gate_s`/`up_s`/`down_s`, the softcapped SwiGLU, an outer-scaled down
  value, no residual), reached through `gpu::encode_inkling2_gguf_phase{1,2}`
  for Q8_0, Q4_K and every `MoeExtType`. Unset, the helpers are the exact
  previous expressions. Parity (`moe_gguf_ext_parity.rs`): all 27 types in
  the mode, eight slots with poisoned unused acts, the six published mixes.
  Four Inkling-branch mutants fail only the Inkling cases (the 27
  `moe_gguf_parity` cases still pass), and two default-branch mutants fail
  both suites. MXFP4 is refused in the mode (its pair binds a bias plane and
  bakes `top_k`); one layer of one UD file uses it.
- DONE (2026-11-21): GGUF intake and flow. `write_inkling2_gguf_install`
  (`crates/repack/src/inkling2_gguf.rs`, `examples/inkling2_gguf_install.rs`):
  names and converter transforms from the name table, block-quantized
  matrices kept under their GGUF tags, float tensors F32 where the flow reads
  F32 and BF16 elsewhere (lossy values counted), experts as an outer-scale
  record (`.scale` or 1.0) plus gate, up and down rows (the fused gate-up is
  gate rows first). Manifest `routedExpert` scheme `inkling2_gguf`. The flow
  takes GGUF resident matrices wherever it takes a matrix (including the loop
  controller planes, which these files ship as Q8_0), and runs GGUF expert
  layers through the Inkling mode of the GGUF pairs. GGUF experts have no
  resident (indexed) kernels, so `auto` streams such an install and
  `resident` is refused by name (`resident_blobs_supported`).
  Gate (M4 Max, `Inkling-2-Mini-Uncensored-AD-IQ2_M.gguf`, a community
  fine-tune: Q8_0 attention and controllers, Q4_K embedding, Q6_K head,
  IQ2_XXS gate-up, IQ4_NL and Q2_0 down): intake 26 s, 8.1 GB, all 646
  resident names and shapes equal to the HF install's, 0 lossy floats.
  Logits against llama.cpp on the same file and ids:

  | prompt | llama.cpp | top-1 | mean KL | max KL |
  |---|---|---|---|---|
  | 19 tokens | CPU | 19/19 | 1.2e-3 | 5.0e-3 |
  | 232 tokens | Metal (`ngl 99`) | 228/232 | 1.3e-3 | 1.0e-2 |

  ORACLE TRAP: llama.cpp's CPU path (`dump_logits ... 0`) returns all-zero
  logits for this file at 32 tokens or more while reporting success (32, 33,
  64, 128 and 232 tokens all zero; 19 fine). Use `ngl 99` for it, and check a
  reference for zero rows before reading a KL. The BF16 and LAMQ installs'
  gates reproduce bit for bit after this change. Not run: the unsloth UD
  files (not downloaded; their `.scale` path is unit-shaped only), sampled
  generation, a speed profile (decode 65 ms per token, unpaired).
- MEASURED (2026-11-21): DSpark drafter acceptance, offline. The drafter is
  the bundled `mtp.safetensors` of `mlx-community/Inkling-2-Mini-oQ4e-mtp`
  (BF16, 3 layers, block 8; the same weights unsloth ships as a GGUF). The
  flow records each position's context taps under `TURBOSPARK_INKLING2_TAPS=1`
  (the stream entering applications 5, 23, 41 and the final-norm output;
  logits unchanged), and `examples/inkling2_dspark_accept.rs` runs a CPU
  reference of the drafter (ollama `draft.go` / oMLX `dspark.py`) at every
  generated position against the target's own greedy continuation. On the
  265-token prose prompt, 96 greedy tokens, 88 rounds: mean accepted 4.30 of
  7 (5.30 tokens per verify), per-position 0.84, 0.77, 0.67, 0.61, 0.53, 0.48,
  0.39; 34 rounds accepted all 7. One prompt, so not a rate to quote broadly.
  What a real speedup still needs: a batched multi-row target forward (every
  Inkling kernel in a multi-row form), rollback of the per-slot conv tails,
  an SWA ring with block slack (a rejected row must not overwrite a row the
  window still needs), the Metal drafter, and the `SpeculativeProducer` arms.
- DONE (2026-11-21): MLX intake. `Inkling2MlxSource` (`crates/repack/src/inkling2_mlx.rs`)
  reads an mlx-lm PR 2086 conversion (`mlx-community/Inkling-2-Mini-4bit`)
  under the HF names by inverting its `sanitize`: renames, the attention-input
  index shift, the merged cross-pass queries, widened router and gate planes
  (narrowed back, refused unless exact), and NVFP4 `global_scale` read as the
  outer scale times 2688. Experts keep their source bytes; dense projections,
  the embedding and the head stay MLX affine 4-bit (resident tags 4/5), and the
  flow runs them through `encode_gemv_any` / `encode_embed_any`. Checked against
  the HF install: the same 646 resident names and dtypes, expert rows byte-
  identical, outer scales equal, and dequantized affine matrices at cosine
  0.994 to 0.996 with their BF16 counterparts. Logits vs llama.cpp BF16:
  18/19 and 262/265 top-1 (every miss is the reference's second choice, margins
  0.02 to 0.43), mean KL 7.7e-3 and 6.9e-3, the size 4-bit dense weights cost.
  tslab profile (3 of 3 clean, unpaired, swap 7 GB): 46.4 tok/s against 36.3
  for the BF16 install, both resident. Not compared against mlx-lm itself
  (that needs the unmerged PR installed).
- DONE (2026-11-21): vectorized expert rows. `TURBOSPARK_DISPATCH_PROFILE=1`
  on `inkling2_logits` (48 warm tokens, 32 slots) ranked the NVFP4 pair at
  42% of GPU time, about 50 GB/s: the row loop decoded one element per lane
  with a byte load and a scale decode each. NVFP4 now takes a 16-element
  sub-block per lane (two word loads, one scale); MXFP8 a 33-byte block per
  lane with a branchless E4M3 decode. Profile rows: NVFP4 phase 1 18.6 to
  6.5 ms, phase 2 9.2 to 3.7 ms per token (profile mode inflates absolutes;
  rank only). The trace's GPU time went 51 to 43 ms per token on a host with
  18 GB of swap in use, so that is not a protocol number. Same dequantized
  values, different summation order: 19/19 and 265/265 top-1 against
  llama.cpp, mean KL 4.9e-4 and 5.7e-4, unchanged. No kill switch: it is
  the same math, and the old kernel is in git history. NVFP4 rows must be
  4-byte aligned, and open refuses an install where they are not.
  Next GPU costs by the same profile: 634 BF16 GEMVs per token (about half
  of GPU time, many tiny: sinks, gates, conv projections) and about 11 ms
  per token of wait above GPU time from the 48 commit round trips.
- DONE (2026-11-21): family-local BF16 GEMV. The shared `bf16_gemv_rows`
  runs a 256-thread threadgroup per output row with scalar loads and a
  barrier. `inkling2_bf16_gemv_multi` runs a simdgroup per row with 8-byte
  loads and takes up to 8 matrices per dispatch; every Inkling GEMV uses it,
  and the 6 to 8 attention-input projections share one dispatch. Other
  families keep the shared kernel. Profile: GEMV rows 29.1 to 13.6 ms per
  token, 634 to 383 dispatches. Trace GPU time 42.6 to 25.9 ms per token
  (wall 104 to 45 ms, but swap fell from 18.8 to 7.5 GB between those runs,
  so the expert-read share of that drop is the host, not the kernel). Logits:
  19/19 and 265/265 top-1 vs llama.cpp; mean KL 3.3e-4 and 7.5e-4 (was 4.9e-4
  and 5.7e-4). Old and new Metal differ from each other by about as much as
  each differs from llama.cpp, and the new run is closer at 131 of 265
  positions, which reads as reduction-order noise over 48 applications; a
  judgement, not a proof.
- TRIED, not kept (2026-11-21): spin-waiting each per-application sync on
  the command buffer status instead of `waitUntilCompleted`. Per token the
  waits exceed GPU time by only 3.4 to 4.4 ms (about 80 us per sync), and an
  interleaved A/B (2 pairs, 48 warm tokens, a busy host: load 10 to 13 and
  another session's Swift build) did not shrink that gap. Logits were
  identical. The round trips cost more through what they cause than through
  the wake-up: the GPU idles during expert reads and host encoding (wall
  minus GPU was about 19 ms of 45), and the routing notes
  (`docs/EXPERT_ROUTING.md`, E2) found that idle lowers the GPU clock. Both
  remaining levers trade something: a keep-alive arm costs power, and fewer
  reads cost memory (more slots) or need mapped residency, which this 36 GB
  machine refuses.
- TRIED, not kept (2026-11-21): a GPU keep-alive, the arm the routing notes
  projected but never ran. A capped ALU spin (1 or 8 threadgroups of 32,
  ~1 ms cap, plus a stop flag) on a second command queue, started after each
  per-application wait and stopped at the next commit. Interleaved A/B, 2
  rounds of off / 1 / 8, 48 warm tokens, 32 slots: GPU time 29.6 to 29.9 ms
  per token in every arm, logits identical, wall time following the expert
  reads only. So at this family's gaps (~0.4 ms per sync) either the clock
  is not falling or a light spin does not hold it; GPU clock was not read
  directly (powermetrics needs root). Side finding: the GPU saw the host's
  stop-flag write about 17 ms late, so a flag alone cannot end such a spin
  promptly; a cap is required.
- OPT-IN (2026-11-21): resident experts, `TURBOSPARK_INKLING2_RESIDENT_EXPERTS=1`.
  tslab found the per-application idle gaps (trace: GPU busy 46.8%, idle gap
  p50 405 us) are not expert copies: 128 cache slots remove every read and
  moved decode only +3.46% (ledger `ink2-speed` 1795273309694-65257-31,
  against a null that read a 4% order effect). The gaps are submit and wake
  latency at the 48 host syncs. With every expert resident (one GPU buffer
  per layer, read from the packed layer file at open, expert `e` at
  `e * stride`), routing runs on the GPU (`inkling2_router_select_k8`) and
  the indexed MoE pair reads the selected experts in place, so a token is
  one command buffer. Per-arm run records (no paired verdict: tslab excludes
  pairs whose page-cache classes differ, and resident reads nothing): 48
  streamed slots 21.6 tok/s median of 7, resident 36.8 of 7, peak 6.6 to
  19.0 GB, swap 8.6 GB used (record 1795274057444-99249-28). The indexed pair
  is bit-identical to the blob pair on the same experts (parity test, slot
  mutant fails). The mode is NOT bit-exact with the host-routed path: GPU and
  host `exp` round differently, so 19/19 tokens are identical but 262/265
  top-1 against llama.cpp on the 265-token prompt (the 3 misses are top-2
  swaps at llama.cpp margins 0.04 to 0.19), mean KL 5.6e-4 against 7.5e-4.
  Since 2026-11-21 it is the `resident` expert residency mode
  (`--expert-residency resident`, the app's Engine settings): `auto` takes it
  where the resident weights, the table and the 10 GiB reserve fit Metal's
  working set (yes on a 36 GB M4 Max, no below about 30 GB of working set),
  so `auto` opens of this family are resident there and no longer bit-exact
  with the host-routed path. A quality gate at its exactness is still owed.
- DONE (2026-11-22): fused decode kernels, oMLX 8dddad9a / a93fd7b ported
  as techniques (this port's own Metal, Apache-2.0 credit in the shader
  headers). `inkling2_qkv_post` (K/V convs, three L2 clips, V unit RMS: six
  dispatches to one), `inkling2_depth_mix_norm` (the mix plus its one or
  three BF16 norms, reading `cur` in place so the copy goes too) and
  `inkling2_short_conv_residual` (the o-proj and MLP-output convs with the
  residual add or the block-start copies). On by default;
  `TURBOSPARK_INKLING2_FUSED=0` restores the separate kernels. Exactness:
  `gpu/tests/inkling2_fused_parity.rs` holds each fused kernel bit-identical
  to its chain over 24 seeds, zero and full conv tails and per-head
  magnitudes 0.05x to 30x (a tau, eps and K-scale mutant each fail; a V
  reduction-tree mutant survives, the two trees round alike on that data);
  the BF16 install's logits are bit-identical over 1,100 positions across
  the SWA ring wrap (stride 64 dump, `inkling2_logits`), and
  `real_forward_inkling2_verify` stays 56/56 and 1,025/1,025 bit-identical.
  The fast-math reassociation trap this hit is in `crates/gpu/AGENTS.md`.
  Speed (M4 Max, resident, BF16 install, one binary, 4 interleaved rounds of
  64 tokens, host load about 2.5; trace medians, a development reading):
  1,897 to 1,554 dispatches per token, GPU 26.3 to 25.75 ms per token,
  36.7 to 37.8 tok/s. That is about 2 us per removed dispatch, so this
  engine is not launch-bound the way oMLX's MLX graph is (their 2,070 to
  1,150 launches bought 8 to 21%): fusing every remaining small op would buy
  at most about 10%. The census (`TURBOSPARK_DISPATCH_PROFILE=1`, ranking
  only) puts the BF16 GEMVs at 40% and the expert pair at 25% of GPU time,
  and the F32 router GEMV at 3.9% for about 1.2 MB per visit, so those, and
  batched prefill, are the larger levers. Not run: the speculative gate (no
  DSpark drafter on this machine), the quality gates, the mlx4 and LAMQ
  installs (not on this machine; the fused kernels do not touch their
  GEMVs).
- The router round trip: 48 GPU-to-CPU syncs per token. Every family pays one
  per MoE layer today. The routed pipeline overlaps the next layer; measure
  before designing past it.
- PARTLY DONE (2026-11-22): chunked prefill, step 1
  (`families/inkling2/prefill.rs`). The prompt runs through the multi-row
  verify forward in 8-row micro-batches instead of one `produce` per token,
  resident residency with native experts only (`inkling2_chunked_refusal`
  answers `supports_chunked_prefill` and the driver's backstop);
  `TURBOSPARK_INKLING2_CHUNKED_PREFILL=0` restores the per-token loop. It
  inherits the verify gate's exactness: `tests/real_forward_inkling2_prefill.rs`
  prefills 1,100 tokens in uneven chunks (13, 512, 300, 275) across the ring
  wrap and gets the last prompt row and 32 decode rows after it bit-identical
  to the sequential reference. Development readings (M4 Max, BF16 install,
  busy host): that test 30.7 to 94.6 tok/s; the CLI on a 3,088-token prompt
  93.7 s to 44.8 s (33 to 69 tok/s), same output text. Dispatch profile of one
  8-row pass (ranking only): the expert pair 55%, BF16 row GEMVs 24%,
  attention 7.5%. The `_idx_rows` pair reads each (row, expert) pair's
  weight rows independently, so step 2 is an expert-grouped pair (each
  distinct expert read once per chunk, partials reduced in slot order to
  keep the bits), then a row cap above 8 and a BF16 GEMM for the dense
  projections, then a flash banded attention for long chunks. The grouped
  pair also serves speculative verify (mtplx's distinct-expert dedupe).
- PARTLY DONE (2026-11-23): chunked prefill, step 2. Panels of up to 32 rows
  (`PANEL_MAX_ROWS`; speculative verify keeps `VERIFY_MAX_ROWS` 8, the DSpark
  block). The SWA ring slack follows the panel (window + 31 = 1,055 rows):
  every K/V row of a pass is written before its attention, so a smaller
  slack lets a panel's last rows overwrite history its first rows read; the
  verify gate gained a block across the new wrap (64/64 bit-identical). The
  dense projections take `inkling2_bf16_gemv_rows32`, a separate kernel so
  verify's 8-row GEMV keeps its exact code (bit-identical to single-row at 1,
  8, 9, 23, 32 rows, widths 256 and 2,304). The routed experts take an
  expert-grouped pair (`encode_inkling2_moe_grouped_rows`: list the pairs by
  expert, decode each expert row once per lane chunk for up to 8 pairs,
  per-slot F32 down terms reduced in rank order), bit-identical to the
  `_idx_rows` pair over 48 seeded cases in both formats; a reversed slot
  reduction mutant fails. Its dot helpers pin association (the
  `crates/gpu/AGENTS.md` fast-math rule); pinning its activation product
  too broke parity, unpinned it matches. `TURBOSPARK_INKLING2_GROUPED_MOE=0`
  restores the per-pair kernels for panels. Prefill gate (1,100 tokens,
  bit-identical): 32-row panels alone 97.9 tok/s, with the grouped pair
  116.7 to 119.3 tok/s, against 32.2 sequential; the CLI on the 3,088-token
  prompt 93.7 s sequential, 44.8 s at step 1, 29.6 s now, same text. Dispatch profile of a
  32-row panel (ranking only): BF16 GEMV 39%, grouped experts 46%, attention
  7%. TRIED, not kept: four weight rows per simdgroup in the 32-row GEMV
  (117 against 98 ms per panel). The GEMV runs at a few percent of FP32 peak
  as scalar per-lane dots; the remaining lever is a matrix-unit BF16 GEMM,
  which changes summation order, so it would make prefill KL-bounded rather
  than bit-exact and needs its own gate.
- PARTLY DONE (2026-11-23): chunked prefill, step 3.
  - Head trimming (bit-exact, default): a prefill panel runs the 201k-row
    unembed on its last row only when it is the final panel, and not at all
    otherwise (`HeadRows`). Prefill gate 125.8 tok/s, bit-identical.
  - Matrix-unit BF16 GEMM, DEFAULT since 2026-11-23 (was opt-in the same
    day); `TURBOSPARK_INKLING2_PREFILL_MMA=0` restores the bit-exact GEMV
    (`inkling2_bf16_gemm_rows_mma`, tslab knob `inkling2_prefill_mma`,
    KL-bounded). FP32 `simdgroup_matrix` on register operands (the
    `docs/INT4_SMALL_M_MMA.md` lane layout), each 64-wide K block of the
    inputs staged in threadgroup memory, 16 weights per lane per block in two
    16-byte loads (k permuted inside the block, the same way for every row).
    Keyed on the pass kind, never the row count: a first version keyed on
    "more than 8 rows" put short tail panels on the GEMV and made the result
    depend on chunking, which the gate caught. Kernel tests
    (`gpu/tests/inkling2_mma_gemm.rs`): mean error against f64 equal to the
    GEMV's (3.79e-4), a row's bits the same alone, in a 9-row and in a 17-row
    panel, deterministic.
  - KL gate (`runtime/tests/real_forward_inkling2_prefill_mma.rs`, BF16
    install, 1,100-token prompt, last prompt row plus 64 teacher-forced
    decode rows). No llama.cpp dumps exist on this machine, so the bound is
    relative to a deviation the project already ships, measured in the same
    run: the streamed host-routed path against the resident one (mean KL
    1.93e-3, max 3.12e-2, top-1 62/65). Bounds: 2x the floor's mean and max,
    top-1 at most one row below it. The kernel as shipped: mean 2.43e-3
    (1.26x), max 1.99e-2 (0.64x), top-1 62/65, last prompt row 1.3e-6.
    Equally accurate kernel variants on the way read 1.29x and 1.63x the
    floor's mean on the same prompt, so a one-prompt KL moves this much
    between roundings and the 2x headroom is not generous. Prefill under it
    is bitwise independent of chunking (asserted with three chunkings).
  - Tuning (2026-11-23, dispatch profile of a 32-row panel, inflated,
    ranking only; the bit-exact GEMV reads 79 ms): first version 86 ms;
    inputs staged per 64-wide K block and 16-weight vector loads 58 ms;
    4-wide staging 53.6 ms; 4 simdgroups per threadgroup (more threadgroups:
    the o-proj alone is 36 threadgroups at 64 rows) 50.8 ms; k permuted as
    8 kk + s so a lane's inputs of a block are 8 contiguous halfs per row
    (two 16-byte threadgroup loads per tile instead of 16 scalar ones)
    29.3 ms. Slower and not kept: 2 or 4 output fragments per simdgroup
    sharing each input load (68.6 and 82.7 ms before the last change, 38.9
    after it), 2 simdgroups per threadgroup (57.3 ms).
  - Speed (development readings, host load about 3): CLI on the 3,088-token
    prompt 27.1 s without, 21.7 s with (about 20%, 114 to 142 tok/s), same
    generated text. The panel is now led by the grouped expert pair.
  - Default (2026-11-23, the owner's decision on the speed result). What
    stays true: a BF16 install's prefill is no longer bit-identical to
    sequential decode by default, only KL-bounded; speculative verify and
    decode are untouched and stay bit-exact against each other. The gates
    now split by arm: `real_forward_inkling2_prefill.rs` sets `=0` and holds
    the exact path to the bit, `real_forward_inkling2_prefill_mma.rs` holds
    the default to the KL bound and to chunking independence. Not covered:
    the quality gates open streamed, which prefills sequentially, so they do
    not exercise either arm; the KL bound rests on one 1,100-token prompt.
    The mlx4 install's dense GEMVs are 4-bit affine (the switch does not
    touch them) and its chunked prefill stays bit-exact; LAMQ installs refused
    chunked prefill by name (until the 2026-11-23 LAMQ entry below). Correction to the 2026-11-22 entries: the mlx4
    and LAMQ installs are on this machine, under `~/.turbospark/models/text/`.
- DONE (2026-11-23): chunked prefill, step 4: 128-row panels and matrix-unit
  experts.
  - Panels of up to 128 rows (`PANEL_MAX_ROWS`; `TURBOSPARK_INKLING2_PANEL_ROWS`
    picks 1 to 128 for tuning, a throughput choice only: both arms'
    results are independent of panel size). The BF16 GEMV and matrix-unit
    GEMM run per 32-row block. Prefill passes write no conv-tail snapshots
    (nothing rolls them back) and the logits and snapshot buffers are sized
    for the 8-row verify, not the panel. The SWA ring slack follows the
    panel (window + 127 = 1,151 rows); the prefill gate grew to 1,400 tokens
    and the verify gate gained a block across the new wrap.
  - The routed experts of a prefill pass take a matrix-unit form of the
    grouped pair (`encode_inkling2_moe_grouped_rows_mma`, under the same
    `TURBOSPARK_INKLING2_PREFILL_MMA` switch): each lane's slice of a 64-wide
    block is one NVFP4 sub-block (element i in byte i % 8, low nibble first)
    or 16 MXFP8 bytes, dequantized exactly into FP32 fragments, multiplied
    with up to 32 pairs per sweep. Kernel tests
    (`gpu/tests/moe_inkling2_grouped_mma.rs`): within about 3e-3 of a row's
    RMS of the scalar grouped pair in both formats, a row's bits the same
    alone, in 9 and in 32 rows, deterministic.
  - Why: the trace (`TURBOSPARK_INKLING2_TRACE=1` now also prints each
    multi-row pass's encode, GPU and wait time) put a 128-row panel at about
    620 ms of GPU, experts about 59%, at about 85 GB/s: dequant and scalar
    dots, not bytes. With the matrix units the panel reads 367 ms.
  - KL gate, now two prompts (prose, Rust source) on both installs, each
    against its own floor; mean KL as a multiple of the floor's: BF16 prose
    1.70x, code 0.67x; mlx4 prose 1.22x, code 0.51x; top-1 equal to or above
    the floor in all four. A one-prompt KL is noisy; the code prompt sits
    below the shipped deviation on both installs.
  - Speed (CLI, 3,088-token prompt, development readings): BF16 default
    21.7 s at 32-row panels, 19.4 s at 128 rows, 14.0 s with matrix-unit
    experts (exact arm 25.4 s); mlx4 default 15.4 s (exact arm 22.3 s).
    The mlx4 install's default prefill is now KL-bounded too (its experts
    take the matrix units); its exact arm stays bit-identical (prefill gate
    155.9 tok/s at 1,400 tokens, verify 72/72).
  - The first pass of a process waits 1 to 3 s before its GPU work starts
    (first decode token 1.5 to 2.8 s wall for 30 ms of GPU): the driver
    mapping the freshly filled resident buffers on first use. Touching the
    24 expert buffers at open moved about 1.2 s there. DONE: open now
    submits that touch (every expert buffer and the dense weights) without
    waiting (`inkling2_prewarm`, `TURBOSPARK_INKLING2_PREWARM=0` turns it
    off), so an idle gap after open absorbs it:
    `tests/real_forward_inkling2_prewarm.rs` reads the first token after a
    5 s gap at 262 ms with it and 1,518 ms without. A one-shot CLI run gains
    nothing (its first pass queues straight behind the touch).
- DONE (2026-11-23): chunked prefill, step 5: tiled attention.
  - `inkling2_attention_prefill` (`encode_inkling2_attention_prefill`), the
    prefill passes' attention under the same matrix-unit switch: a
    threadgroup takes 32 query rows of one head and walks 32-key tiles of K
    and V staged once in threadgroup memory (the rows kernel reads a row's
    whole window per row and head). QK^T and PV on FP32 simdgroup_matrix
    with register operands, online softmax started from the sink (m = sink,
    l = 1), bias added in band, window and causality masked. After oMLX
    8d2c9c6's flash banded attention (Apache-2.0); this port's own code.
    `TURBOSPARK_INKLING2_PREFILL_ATTN=0` keeps the rows kernel for
    attention while the GEMMs and experts stay on the matrix units.
  - Row independence: key tiles align to ABSOLUTE positions, and a tile
    fully masked for a row is an exact no-op for it, so a row's arithmetic
    does not depend on its block. `gpu/tests/inkling2_attn_prefill.rs`: max
    1.2e-4 to 2.4e-4 from the rows kernel (about one FP16 ulp), a row's bits
    the same alone, in the panel and in a block starting at row 5, over an
    SWA slot past two wraps, a full slot and the first rows of a sequence;
    starting tiles at the block's own first key instead fails it.
  - Found by the encoder's range check: aligned tiles reach up to 31 keys
    before a block's window, so the SWA ring slack grew by 32 (window + 159
    = 1,183 rows); the verify gate's wrap block moved with it.
  - Speed (CLI, 3,088-token prompt, BF16 default): 14.0 s to 10.8 to
    11.4 s; a 128-row panel's GPU time no longer grows with context (341 ms
    at position 128, 359 ms at 2,944).
  - Gates: exact arms bit-identical on both installs (BF16 130.5 tok/s,
    mlx4 138.0 tok/s at 1,400 tokens); verify 72/72 and 1,154/1,154 on
    both. `exp` in the tiled attention is `precise::exp`, as in the rows
    kernel.
- DONE (2026-11-23): the prefill KL gate redesigned
  (`runtime/tests/real_forward_inkling2_prefill_mma.rs`). The first design
  (one prompt, mean and max KL over the last prompt row plus 64
  teacher-forced decode rows, 2x a floor measured in the same run) was not
  a stable instrument: kernel variants with the same error against f64 read
  1.26x to 3.45x the floor on one prompt, swapping fast `exp` for
  `precise::exp` moved one from 2.07x to 3.45x, and the matrix-unit default
  committed in 3a986ff7 read 13x on a low-entropy code prompt whose mean
  came from one spike row (max 3.1e-2, p95 7.9e-4). A prompt the work itself
  edited (`verify.rs`) also moved the floor between runs. The redesign:
  four prompts frozen as fixtures (`runtime/tests/fixtures/inkling2_kl/`,
  two prose, two code, cut from HEAD at 4b66ed46); per-row KL pooled over
  all 260 rows and bounded at the MEDIAN and p90, each at most 3x the
  floor's pooled value; pooled top-1 at most 2 rows below the floor's; each
  prompt's last prompt row at most 3x the floor's pooled p90; chunking
  independence bit for bit. The bounds were set before the first run. One
  was changed after it: the last-row bound was an absolute 1e-3 nats, and
  on `prose_lamq` (a high-entropy last position) the shipped floor itself
  read 1.48e-3, so it is now relative to the floor's pooled p90 like the
  others. Results (default prefill, tiled attention included):
  BF16 pooled median 0.87x, p90 0.74x the floor, top-1 258 against 256,
  worst last row 1.09e-3; mlx4 pooled median 1.61x, p90 1.34x, top-1 255
  against 256, worst last row 3.4e-3 (the floor's own 6.0e-3). About 7
  minutes per install; it opens a resident runner (about 17.7 GB) and then a
  streamed one, so do not run it beside another resident-install test on a
  36 GB machine (one such overlap probably took this machine down on
  2026-11-23).
- DONE (2026-11-23): prefill expert tuning, no output change.
  - 256-row panels measured and not kept: GPU per 3,088-token prompt 8,592
    against 8,600 ms at 128 rows. With matrix-unit experts the cost follows
    the work per pair, not expert reuse, so a bigger panel buys nothing; the
    cap stays at 128 (smaller ring and scratch).
  - NVFP4 grouped experts take HALF matrix operands with FP32 accumulation:
    an NVFP4 weight (E2M1 x an E4M3 scale, at most 5 significant bits,
    within half range) and the FP16 inputs are exact in half. Output digests
    of `gpu/tests/moe_inkling2_grouped_mma.rs` are identical to the FP32
    operand build in all three fixtures, so the KL gates stand unchanged.
    MXFP8 keeps FP32 operands (its block exponents do not bound a weight to
    half range). Phase 1 81 to 68 ms per 128-row panel (profile, ranking).
  - `moe_inkling2_group` runs on one threadgroup (atomic counts, prefix,
    atomic scatter) instead of one thread. The order of pairs within an
    expert's list is no longer fixed, which no output depends on (every
    consumer computes each pair on its own); the grouped parity suites
    passed three repeats.
  - Together: GPU per 3,088-token prompt 8,600 to 8,210 ms (-4.5%).
- DONE (2026-11-23): decode BF16 GEMV headroom measured and one step taken.
  `gpu/tests/inkling2_gemv_bandwidth.rs` (ignored bench, ~192 MB cycled from
  DRAM): the one-row `inkling2_bf16_gemv_multi` reads 275 to 307 GB/s at q
  (3072 x 2304), 279 to 284 at k, 337 to 339 at the o-proj and 392 at the
  unembed, which is near this machine's practical read ceiling. A two-row
  form (`inkling2_bf16_gemv_multi2`, one input load feeding two weight
  streams, each row's accumulation unchanged, bit-identical by test) lifts q
  to about 335 and leaves the rest within noise. Decode, BF16 install, 3
  interleaved rounds: GPU 25.2 to 24.9 ms per token, 38.5 to 38.9 tok/s;
  verify gate still 72/72 (decode now runs the two-row form). Default on,
  `TURBOSPARK_INKLING2_GEMV2=0` restores the one-row form. Not done: the F32
  router GEMV (`minimax_router`, scalar loads, shared with MiniMax); a faster
  one changes its summation order, so the decode router, the verify router
  and the host-routed path would have to move together and be re-gated
  against llama.cpp, for an estimated 2% of decode.
- DONE (2026-11-23): speculative decoding now beats plain greedy on code
  and long prompts (MLX 4-bit install, its DSpark drafter in `dspark/`).
  - Speculative verify takes the expert-grouped pair (bit-identical to the
    per-pair pair; `TURBOSPARK_INKLING2_GROUPED_MIN_ROWS`, default 2, 9
    restores the per-pair pair below 9 rows): 8-row verify 83 to 74 ms per
    round on short prompts, 95 to 93 ms on the 3,299-token one, the same
    committed tokens. Verify gate 72/72 and 1,154/1,154 on both installs.
  - Draft depth 3 instead of 7 (mtplx's finding holds here): verify 46 ms per
    round (55 on the long prompt); per-position acceptance falls off fast
    enough that 7 rows do not pay for themselves. Lossless on all three
    prompts (`real_forward_inkling2_speculative.rs`, `INKLING2_SPEC_DEPTH`).

    | prompt | plain | depth 7 (grouped) | depth 3 (grouped) |
    |---|---|---|---|
    | prose, 47 tokens | 47.8 | 32.6 | 44.5 |
    | code, 50 tokens | 47.9 | 42.6 | 54.2 |
    | long, 3,299 tokens | 34.7 | 34.4 | 42.8 |

    tok/s, greedy, 256 new tokens, development readings. Recommended:
    `--speculate 3` on an install with its drafter. Speculation stays off
    under `auto` for this family (it loses about 7% on short prose and the
    drafter costs 636 MB), so that remains the caller's choice.
- DONE (2026-11-23): a quality gate on the default prefill
  (`bench/tests/inkling2_prefill_quality.rs`). The Phase 6 gates open
  streamed, which prefills one token at a time, so they never ran the
  chunked matrix-unit prefill. This scores the Phase 6 reference answer
  sequentially on a resident runner, through the default chunked prefill,
  and sequentially streamed (the floor; it reproduces the frozen Phase 6
  rows, 4.7024 and 4.8026, to the last digit). Bound, set before the first
  run: |chunked - sequential| at most max(3 x the floor's distance, 0.1%).
  BF16: sequential 4.6890, chunked 4.6906 (+0.0017), floor +0.0134, bound
  0.0403. MLX 4-bit: 4.8104, 4.8043 (-0.0061), floor -0.0078, bound 0.0233.
  The quality prompt is 66 tokens (one panel): it covers the kernels, not
  long-context prefill, which the KL gate covers.
- DONE (2026-11-23): batched affine projections for MLX-sourced installs in
  multi-row passes (`Ctx::gemv_rows`, `TURBOSPARK_INKLING2_BATCHED_AFFINE=0`
  restores one GEMV per row; tslab knob `inkling2_batched_affine`). The mlx4
  install's 4-bit dense projections went through one `dequant_int4_gemv` per
  row (about 1,700 dispatches per 32-row panel) because the q/k/v inputs
  share one buffer at three rows per token, which no token-major kernel
  takes. They are now packed token-major (`affine_pack`, an exact copy) and
  run through `encode_gemm_any`, the batched INT4/INT8 GEMMs that are
  bit-exact against the GEMV at every width, in calls of up to 16 rows.
  Gates on mlx4, all bit-exact: prefill 123.5 to 135.7 tok/s (32/32 decode
  rows); verify 64/64 and 1,019/1,019; speculative lossless on all three
  prompts (255 committed tokens identical), verify round 86 to about 80 ms.
  CLI on the 3,088-token prompt: 28.3 s to 24.7 s, same text. The DSpark
  drafter IS on this machine, in the mlx4 install's `dspark/`: the
  speculative gate ran there for the first time since 2026-11-21. Speculative
  is still slower than plain greedy on every prompt (prose 29.9 against 46.3
  tok/s, code 40.5 against 46.3, long 33.4 against 34.0); acceptance 1.8 to
  2.7 per round of 7.
- DONE (2026-11-23): multi-row forward on LookAtMyQuant installs (chunked
  prefill, speculative verify, and the DSpark drafter's target embed and
  head). Bit-exact by construction: every coded tensor (embed buckets,
  dense projections, experts, head) runs decode's own `LamqRuntime` call
  once per row on that row's slice; the BF16 tensors take the multi-row
  kernels. `multi_row_refusal()` replaces the by-name LAMQ refusal (a LAMQ
  install without coded experts still refuses). LAMQ does NOT take the
  matrix-unit default (`Ctx::prefill_mma`): with it, the reference
  perplexity moved 0.0087 against a bound of 0.0057 (the LAMQ floor is
  0.0005, so the 0.1% term set the bound); the bound was not widened.
  Gates on `inkling2-mini-lamq54` (M4 Max): prefill 1,400 tokens 28.7 to
  53.0 tok/s, 32/32 decode rows bit-identical; verify 72/72 and 1,154/1,154;
  reference perplexity 5.7377 sequential and chunked (diff 0); speculative
  depth 3 lossless on all three prompts but slower than plain except long
  context (prose 19.0 against 31.3 tok/s, code 25.2 against 31.2, long 27.2
  against 25.1): the per-row coded GEMVs make a 4-row verify cost about
  four tokens. DSpark stays opt-in, so no default changes.
- DONE (2026-11-23): a multi-row coded GEMV, `lamq_gemv_rows`
  (`gpu::encode_lamq_gemv_rows`), shared by decode and the multi-row passes.
  Each 16-column chunk is decoded once into registers for a block of 8 token
  rows. GPU time at 32 rows (M4 Max, `lamq_parity.rs` `gemv_rows_speed`):
  2.3x to 3.2x faster than 32 one-row dispatches for the trellis codes at the
  Mini's attention shapes, equal to `lamq_gemv` at one row; SLOWER for the
  level codes (lm6 0.63x to 0.73x, 2.6x slower at one row), which keep
  `lamq_gemv` everywhere. Its rows are within about 1e-6 of a row's RMS of
  `lamq_gemv` but not its bits (fast-math associates the chunk sum
  differently; two rewrites, one with reassociation pinned off, still
  differed), and a row's bits do not depend on the other rows in the
  dispatch.
  - First wired into chunked prefill alone, as a KL-bounded path, it FAILED
    the LAMQ perplexity bound (0.0069 with only `wo_ud`, 0.0367 with every
    trellis attention matrix, bound 0.0057). Bisected: the same wiring with
    the old GEMV over the rotated rows scores diff 0.0000, so the shift is the
    summation order alone; this install is that sensitive.
  - So decode moved onto it instead (`LamqRuntime::gemv_one`: trellis codes
    through `lamq_gemv_rows` at one row), and verify and chunked prefill run
    it over all rows (`LamqRuntime::gemv_rows`, a counted rotation per
    matrix, bit-identical per vector), so every row gets decode's bits with
    no switch. The head and level codes are unchanged.
  - Gates on `inkling2-mini-lamq54`: prefill 1,400 tokens 32/32 decode rows
    bit-identical, chunked 48.9 to 68.0 tok/s (+39%, read on a host with an
    intake and a build running); verify 72/72 and 1,154/1,154.
  - The quality gate was RE-FROZEN (`inkling2_lamq54_quality_gate.rs`): two
    fresh processes agreed on perplexity 5.7391 (was 5.7382, +0.016%) and
    both new digests, the 8-slot digest equals the 16-slot one, and a third
    process asserted the row.
- DONE (2026-11-23): resident GGUF experts and the multi-row forward on GGUF
  installs. No per-type kernels: `gpu::encode_inkling2_gguf_bind_rows` writes
  one `RoutedBlobs` table per row from the GPU router's ids
  (`layer + id * stride`), and the streamed GGUF Inkling pair runs unchanged
  over it (`_table` encoders), so every GGUF type with an Inkling pair gets
  resident decode, chunked prefill, verify and DSpark with decode's
  arithmetic. On Apple tier-2 argument buffers a table entry is the blob's
  GPU address; `gpu/tests/inkling2_gguf_tables.rs` checks the GPU tables
  against `bind_for`'s bytes and the pair's outputs bit for bit (a mutant
  ignoring the row fails). `resident_blobs_supported` now accepts GGUF, so
  `auto` opens a GGUF install resident when it fits, as for native installs.
  Multi-row dense GGUF matrices batch only through dtypes `encode_gemm_any`
  holds bit-exact (`GEMM_ANY_BATCHED`); others run per row. Gates on
  `Inkling-2-Mini-Uncensored-AD-IQ2_M.gguf` (AtomicChat @ `40886ab6`, sha256
  `fca99d48...`, re-downloaded; IQ2_XXS gate-up, Q2_0 and IQ4_NL down, Q8_0
  dense, Q6_K head), M4 Max:
  - chunked prefill 1,400 tokens 14.2 to 33.2 tok/s, 32/32 decode rows
    bit-identical; verify 72/72 and 1,154/1,154;
  - reference perplexity: resident sequential 5.3048, streamed 5.3225;
  - the matrix-unit default FAILED the KL gate (pooled median 0.95x and p90
    1.70x of the floor, but pooled top-1 256 against the floor's 259, the
    bound is 2 rows), so GGUF, like LAMQ, keeps the bit-exact prefill
    (`Ctx::prefill_mma` is native experts only); its default then scores
    5.3048, diff 0;
  - plain decode, `real_forward_inkling2_residency_speed.rs`, 128 tokens,
    interleaved: resident 20.8 and 21.2 tok/s, streamed 17.5 and 16.8;
  - speculative depth 3 with the mlx4 install's DSpark drafter: lossless on
    all four prompts; prose 12.3 against 17.8 tok/s, code 13.2 against 17.1,
    long 20.3 against 18.1, edit 29.2 against 19.7 (copy rounds accepted 211).
    GGUF experts run one row at a time in the verify, so short-answer rounds
    lose; DSpark stays opt-in.
  Second file, unsloth `Inkling-2-Mini-UD-IQ1_S.gguf` @ `8078d149` (sha256
  `17a7ff34...`; IQ1_S, IQ2_XXS, IQ2_XS and IQ2_S gate-up, Q2_0 and GGUF
  NVFP4 down): intake OK (646 names); chunked prefill 1,400 tokens 22.8 to
  39.2 tok/s, 32/32 rows bit-identical; verify 72/72 and 1,154/1,154.
  Logits against llama.cpp master `c212d25` (2026-11-23, has #32694, #32704
  and #32799; Homebrew's 1.0.0 bottle predates `inkling2_loop` and its HEAD
  formula fails on a missing `tools/ui`, so built from source; Metal, one
  batch, no zero rows), same file and ids, a 60-line `llama_get_logits_ith`
  dumper:

  | prompt | ours | top-1 | mean KL | max KL |
  |---|---|---|---|---|
  | 12 tokens | resident | 12/12 | 2.5e-5 | 2.3e-4 |
  | 711 tokens, every 4th row and the last 16 | resident | 182/189 | 4.9e-3 | 7.1e-2 |
  | same | streamed | 183/189 | 5.4e-3 | 1.3e-1 |

  Resident sits where streamed does; a second resident run reproduced the
  first byte for byte. `examples/inkling2_logits.rs` takes
  `INKLING2_RESIDENCY=resident|streamed` for this.
  Memory oracle (`bench/tests/inkling2_gguf_memory_oracle.rs`, the IQ2_M
  install, streamed at 16 slots, 8,192 context, all three protocol cases):
  frozen from two runs (peaks 1,669 and 1,668 MiB; 12.9 to 14.6 tok/s; every
  case at end of turn with the same token counts; replay flat), ceiling
  1,770 MiB and floor 9.0 tok/s, asserted by a third (peak 1,667 MiB, 13.3 to
  14.6 tok/s). Not run: the ggml-org files.
- DONE (2026-11-23): prompt-copy drafting, mtplx item (3)
  (`runtime/src/prompt_copy.rs`, wired in `speculative.rs`; on with the
  DSpark drafter, `TURBOSPARK_SPEC_PROMPT_COPY=0` turns it off). When the
  last 10 down to 6 committed tokens occurred earlier in the history, the
  round proposes what followed the most recent occurrence instead of
  drafting. Probation: a copy round starts at the drafter's own depth and
  doubles after a fully accepted round, up to 24 proposals (the Inkling
  verify now takes 25 rows, `VERIFY_COPY_MAX_ROWS`; snapshots and the
  logits buffer grew with it); a rejection drops it back. A round accepting
  fewer than 2 pauses copying for 2, 4, ... 16 rounds. Greedy only. Producers
  opt in through `copy_draft_max`; `skip_drafter_round` lets DSpark write its
  context rows from the previous verify's taps before the copy round's
  verify overwrites them. A first try at a fixed 24-wide copy lost (code
  50.9 against 54.4 tok/s on mlx4): a verify row costs about 10 ms there, and
  early-rejected copies paid for 25 rows. Gates, M4 Max, depth 3, 256
  greedy tokens, every run lossless (255 committed tokens identical); the
  new `edit` prompt (repo code, then a rewrite request) is the copy-heavy
  case:

  | install | prompt | copy off | copy on | copy rounds, accepted |
  |---|---|---|---|---|
  | mlx4 | edit | 51.9, 51.8 | 54.9, 54.8 | 10, 57 |
  | mlx4 | prose | 44.3, 44.3 | 45.1 | 3, 20 |
  | mlx4 | code | 54.4, 54.1 | 53.4 | 4, 11 |
  | mlx4 | long | 42.6, 42.7 | 41.8 | 10, 17 |
  | BF16 | edit | 42.0, 42.0 | 42.0, 42.1 | 3, 7 |
  | BF16 | code | 52.2, 52.4 | 52.0, 51.8 | 3, 10 |
  | BF16 | long | 47.6, 47.5 | 48.7, 48.8 | 11, 35 |

  Speculative tok/s, in-test wall time, not frozen rows. Copy wins where the
  output re-quotes its context and is within about 2% elsewhere.
- MEASURED, NOT BUILT (2026-11-23): GPU-side acceptance, mtplx item (6).
  `TURBOSPARK_INKLING2_TRACE=1` on mlx4, code prompt, depth 3, 83 rounds:
  a round is 57.3 ms wall (draft 9.85, verify 45.85, accept 0.95, rollback
  0.26, bonus 0.35). The verify is 44.2 ms of GPU time; its host share is
  encode 1.18, commit-to-return 0.23 and the logits readback 0.04 ms. So
  one command buffer per round with greedy acceptance on the GPU can win
  at most about 3 ms a round (5%), plus the draft's unmeasured host share.
  Only the 0.95 ms host accept goes away without the rest; overlapping the
  encode with the GPU needs every position, KV cursor and rollback count
  GPU-resident, a redesign of the decode flow for at most that 5%. Declined
  for now; the readback is not the cost.
- RESEARCH (2026-11-22): mtplx (github.com/youssofal/mtplx, Apache-2.0, MLX)
  runs Inkling 2 Mini with the same bundled DSpark drafter. Our port already
  has its Markov head (`dspark_markov_*`), so the draft logic matches.
  Portable runtime ideas, untested here: (1) fixed draft depth 3 by default
  (their v2.13.2; a load-time cost table under-priced deep rounds, k=4 cost
  +33.8% served against +14% predicted), and any adaptive depth may read
  only committed history; (2) the distinct-expert dedupe above; (3)
  prompt-copy n-gram drafting (n 6 to 10, up to 24 tokens, verify widths 9 to
  25, probation and backoff), which needs a verify above 8 rows; (4) commit
  kept rows from deltas instead of write-then-rollback; (5) a draft
  vocabulary subset (top 32,768 tokens) for the drafter head; (6) the round
  as one command buffer with GPU-side acceptance. Their published Inkling
  numbers are M5 Max only.
- Loop-aware stats and prefetch: `RouterHistogram` merges the three passes,
  and the pilot probe and Qwen prefetcher assume layer `L + 1` follows `L`
  (false at the 17-to-6 seam). Key them by application, or refuse them.
- Batched prefill: multi-row forms of the `inkling2_ops` kernels and a BF16
  GEMM for the dense projections. Then the ANE split
  (`docs/SPEEDUP_PLAYBOOK.md`, "ANE+Steel prefill split"). Every Inkling dense
  projection is BF16, which is the case the weightless Core ML programs serve.
- Decode attention: the GQA-shared kernel's design (one K/V read per group of
  3 q heads) for the rel-bias plus sink kernel.
- Quantizing the BF16 attention weights would cut the dominant decode bytes,
  at a quality cost to measure. Not a default without its own gate.

- PARTLY DONE (2026-11-21): DSpark speculative decoding, greedy only.
  - Stage 1, batched verify (`families/inkling2/verify.rs`, kernels in
    `gpu/src/shaders/inkling2_batch.metal` and the `_idx_rows_` MoE pair):
    up to 8 rows in one command buffer, resident residency only (streamed is
    refused by name). Kernel gate `crates/gpu/tests/inkling2_batch_parity.rs`:
    10 tests, every multi-row kernel bit-identical to 8 single-row calls
    (mutants on the conv snapshot and attention position fail). Model gate
    `crates/runtime/tests/real_forward_inkling2_verify.rs` (BF16 install, M4
    Max): 56 of 56 verify rows bit-identical to sequential decode, blocks at
    16, 1020, 1028 (across the 1,031-row ring wrap) and after rollbacks.
  - Stage 2, rollback: per-slot conv tails keep 9 snapshots per verify
    (block start, then after each row); SWA rings hold window + 7 rows;
    cursor via `KvCacheManager::rewind_cursor`. Same gate: 1,025 of 1,025
    later sequential rows bit-identical after rollbacks keeping 1, 2, 3, 5
    rows. Mutant (restore snapshot k+1): 8 of 56 exact, yet 55 of 56 top-1,
    so this class of bug reads as fluent output.
  - Fixed on the way: `reset()` never zeroed the conv tails (a second
    generation inherited the first's conv state), and `try_reuse_prefix`
    could rewind the cursor under in-place tails (now pure continuation only).
  - Stage 3, Metal drafter (`families/inkling2/dspark.rs`,
    `shaders/inkling2_dspark.metal`): one command buffer per draft, 9.3 to
    9.5 ms. Layer 1 sees the full context and layers 0 and 2 the last 1024
    rows (ollama `draft.go`; the CPU reference now does the same). Parity vs
    the CPU reference in `examples/inkling2_dspark_accept.rs`: 615 of 616
    and 1,063 of 1,064 proposals equal (raw prose, chat story).
  - Acceptance (CPU and Metal agree): raw story continuation 0.86 of 7
    (1.86 tokens per verify), chat-templated story 1.88 of 7 (2.88). Far
    below the 4.30 recorded earlier on the 265-token prompt; why is open.
  - Stage 4 wired: `DflashDraftPolicy::Fixed(n)` at open builds the drafter
    from `<install>/dspark/` or `TURBOSPARK_INKLING2_DSPARK`; `Auto` never
    does. `dflash_speculation_blocker` has an Inkling arm (no MoE refusal).
    The loop's existing block-drafter sampling refusal applies.
  - Fixed: the flow padded logits past `unpadded_vocab` with -inf, which the
    selector refuses, so `run_raw_completion` (and so every front end and
    tslab arm) could not commit a single Inkling token. Padding is now the
    most negative finite FP16 (`PAD_LOGIT`); a softcapped logit is within
    +-30, so a padding row never wins or carries probability.
  - Stage 5a, losslessness (`tests/real_forward_inkling2_speculative.rs`,
    2026-11-21, M4 Max, 256 greedy tokens per prompt, block 7): the
    speculative loop commits exactly the plain greedy stream on every prompt
    and both installs. Acceptance per round (TURBOSPARK_SPEC_STATS):

    | install | prose (47-token chat) | code (50-token chat) | long (3,343 tokens) |
    |---|---|---|---|
    | BF16 | 1.36 of 7 | 3.06 of 7 | 3.40 of 7 |
    | MLX 4-bit | 1.80 of 7 | 2.71 of 7 | 1.61 of 7 |

    Per round: draft 12 to 22 ms, verify 94 to 118 ms (8 rows). In-test wall
    tok/s came from a host that had just rebooted and had other sessions'
    jobs at 400%+ CPU, so they are not quoted here; they only say the verify
    costs about 3 to 4 plain tokens, so prose (about 2.4 tokens per round) is
    expected to lose and code and long context may gain on BF16.
  - MLX 4-bit install: the verify gate also passes there (56 of 56 verify rows
    and 1,025 of 1,025 later rows bit-identical). Its affine projections run
    once per row in the verify (`Ctx::gemv_rows`), so they are not
    amortized; a batched affine GEMV that matches the per-row bits is the
    missing piece for a speedup on that install.
  - Stage 5b, speed, FIRST READING (2026-11-22, M4 Max, ON BATTERY, so not
    quotable; experiments `ink2-dspark-mlx-battery`,
    `ink2-dspark-hf-battery`, `ink2-dspark-stats-battery`, worker at
    `36c6dd58`): code workload, 256 greedy tokens, `--chunk off`, experts
    resident, block 7. The drafter is the original's `mtp.safetensors` at
    `ed40e7da` in each install's `dspark/`.

    | install | plain decode | DSpark-7 decode | accepted / committed per round | draft | verify (8 rows) |
    |---|---|---|---|---|---|
    | `inkling2-mini` | 34.8 to 36.0 tok/s | 44.0 tok/s (clean); 42.5 to 44.7 dirty | 2.86 / 3.86 | 8.4 ms | 77 ms |
    | `inkling2-mini-mlx4` | 45.8 to 47.2 tok/s | 38.8 to 39.7 tok/s | 2.94 / 3.94 | 10.7 ms | 90 ms |

    The original gains about 24% (88 ms a round for 3.86 tokens against
    28 ms a plain token); its A/B had no clean pair (foreign CPU), but the
    stats run was clean and agrees. The MLX build loses 14.4% (3 clean
    pairs, CI -15.3% to -14.1%): the same acceptance, but its plain token
    is already 22 ms and its verify costs about four of them, because the
    affine projections run once per row (`Ctx::gemv_rows`). A batched
    affine GEMV matching the per-row bits is what would turn it into a win.
    The drafter's state adds about 1.4 GB to the resident peak. Re-run on AC
    before quoting any of this. Earlier attempts: two tslab attempts (experiment
    `ink2-dspark`, code workload, `--chunk off`, decode_tok_s, 30-minute
    quiet wait) got zero clean pairs: another session's Python job and
    mediaanalysisd held the host over 90% CPU. The candidate arm was also
    refused there because that worker predated the padding fix. Rerun on a
    quiet host:
    `tslab ab --model M --prompt messages:crates/lab/workloads/code.messages.json --decode 256 --chunk off --null --metric decode_tok_s --baseline plain@auto/resident`,
    then the same with `--candidate d7@auto/resident~dflash:7`; the drafter
    must sit at `<install>/dspark/`.
  - Still not checked: plain-decode bits against the build before this work
    (the only plain-path changes are the ring slack and the padding value;
    the attention reads the same rows either way, but no run compared them).

Stage C: drafter and comparisons
- `mtp.safetensors` is a DSpark block drafter, not a next-token head: block 8,
  `mask_embed`, context K/V projections (`k_proj_ctx`, `v_proj_ctx`) fed from
  trunk taps at `target_layer_ids` [5, 23, 41, 48] (cache-slot ids), a
  `[4, 4]` `fuse_weight`, and an RNN `markov_head`. The DFlash2 machinery
  (`docs/DFLASH2.md`) supplies verify, accept and rollback. The drafter
  forward and selector are new; the reference is llama.cpp PR #32218
  (open). Verify needs snapshot and rollback of every slot's conv tails.
- MLX baseline: mlx-lm PR #2086 (open) and the `mlx-community` /
  `nativ-community` MLX checkpoints, for a speed and quality comparison on
  the same Mac.
- Vision tower: out of scope until the text path ships.
