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
   function) are in `ArchConfig`. Swift's per-family UI lists
   (`ModelFeatureDescriptor`, `ModelFamilyVisuals`) wait for the catalog row.
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
4. Tokenizer dialect and `probe/safetensors.rs` arm; CLI reachability.
5. DONE: long-context check (above). Remaining: memory oracle, quality
   gates, then a catalog row.

Stage B: speed and memory (after A's gates, each with a kill switch)
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
- The router round trip: 48 GPU-to-CPU syncs per token. Every family pays one
  per MoE layer today. The routed pipeline overlaps the next layer; measure
  before designing past it.
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
