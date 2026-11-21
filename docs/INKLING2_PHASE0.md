# Inkling 2 Mini (`inkling2_loop`) Phase 0

Target: `thinkingmachines/Inkling-2-Mini`, revision
`ed40e7da625f594fc9c4990f5f66be24873c22d7`. Apache 2.0, ~26B physical / 3B
active per pass, 1M context, text + image in, text out.

STATUS: **not runnable.** The portable CPU reference
(`crates/compute/src/inkling2.rs`) and the routed-expert decode pair (Metal,
parity-tested on an M4 Max) exist. No family enum, repack, decode flow or
catalog row has landed. Nothing here has been compared with a real
checkpoint. Do not add a catalog row or call it supported until the Phase 3
to 6 gates in `docs/NEW_MODEL.md` have run on a Mac.

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
past the 1024 bias extent, and the sliding-window cut. A long-context
comparison must cover all three before they count as verified.

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

Expert: NVFP4 3 x 1152 x 2304 / 2 = 3.8 MiB plus scales; MXFP8 7.6 MiB.
Whole experts: ~6.8 GB NVFP4 + ~4.1 GB MXFP8. Resident BF16 attention
+ embeddings ~ 9 GB. Conv state per slot: (2 x 1024 + 2 x 2304) x 3 rows. KV:
8 heads x 128 x 2 x 2 B = 4 KiB per token per slot; 8 full-attention slots
(layers 5, 23 once, 11 and 17 three times) = 32 KiB per token, 32 GiB at 1M
context; 40 SWA slots ring at 1024.

## Remaining work, in dependency order

1. `ModelFamily` variant, `ArchConfig` block (loop, sconv, rel, attn-res), baseline.
2. Repack: config parser, tensor-name map (HF `wq_du` etc. to roles), the
   expert blob writer (transcoders and blob contract above, w13 deinterleave),
   and a `validate_quant` shape for the two native expert schemes.
3. Kernels. DONE, each parity-tested on Metal against this file's CPU
   reference and mutation-checked:
   - NVFP4/MXFP8 routed decode pair (above).
   - `crates/gpu/src/inkling2_ops.rs` and `tests/inkling2_ops_parity.rs`:
     - sigmoid router selection with the correction bias (top 8, rank order,
       through `moe.metal`'s shared SIMD argmax);
     - L2 clip with an optional per-channel scale and a fused post-multiplier
       for `tau(pos)`;
     - short conv decode (static, and the dynamic full-attention K/V taps),
       with an FP16 tail updated in place;
     - depth mix.

   OPEN: batched prefill for all of these, the router and `weight_proj` GEMVs
   (BF16, likely existing kernels), rel-bias + sink attention, loop control
   (two rank-64 GEMVs plus an elementwise kernel), the softcapped output head.
4. Decode flow with 48 cache slots and per-slot conv state; MTP/DSpark drafter
   and the vision tower are out of scope for a first pass.
5. Tokenizer dialect (`<|message_user|>`, `<|message_model|>`, `<|end_message|>`,
   reasoning-effort system line, EOS 200006) and `probe/safetensors.rs` arm.
6. Real-checkpoint greedy and sampled gates, memory oracle, then the catalog row.
