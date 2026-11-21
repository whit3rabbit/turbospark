# Inkling 2 Mini (`inkling2_loop`) Phase 0

Target: `thinkingmachines/Inkling-2-Mini`, revision
`ed40e7da625f594fc9c4990f5f66be24873c22d7`. Apache 2.0, ~26B physical / 3B
active per pass, 1M context, text + image in, text out.

STATUS: **not runnable.** Only the portable CPU reference exists
(`crates/compute/src/inkling2.rs`, 17 unit tests). No family enum, repack,
Metal kernel, decode flow or catalog row has landed. Nothing here has been
compared with a real checkpoint, and this was written on a Linux host with no
Metal device. Do not add a catalog row or call it supported until the Phase 3
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
| 0-4, 6-10, 12-16, 18-22 (20) | NVFP4 `U8` packed, `[128, rows, cols/2]` | `E4M3` per 16, plus `amax[128]` / `outer_scale_denominator[1]` |
| 5, 11, 17, 23 (4) | MXFP8 `E4M3`, `[128, rows, cols]` | `U8` E8M0 per 32 (255 = NaN) |

Neither is supported by `validate_quant` (affine INT4/8, group 64) or the GGUF
block set. Options: (a) new NVFP4 and MXFP8 expert and GEMV kernels (exact,
more MSL), or (b) requantize at install to affine INT4 (no new kernels, extra
loss, and 4-bit on the MXFP8 layers is a real quality regression to measure).
This has not been decided.

## Memory accounting (from shapes)

Expert: NVFP4 3 x 1152 x 2304 / 2 = 3.8 MiB plus scales; MXFP8 7.6 MiB.
Whole experts: ~6.8 GB NVFP4 + ~4.1 GB MXFP8. Resident BF16 attention
+ embeddings ~ 9 GB. Conv state per slot: (2 x 1024 + 2 x 2304) x 3 rows. KV:
8 heads x 128 x 2 x 2 B = 4 KiB per token per slot; 8 full-attention slots
(layers 5, 23 once, 11 and 17 three times) = 32 KiB per token, 32 GiB at 1M
context; 40 SWA slots ring at 1024.

## Remaining work, in dependency order

1. `ModelFamily` variant, `ArchConfig` block (loop, sconv, rel, attn-res), baseline.
2. Repack: config parser, tensor-name map (HF `wq_du` etc. to roles), quant decision above.
3. Kernels: NVFP4/MXFP8 (or requant), depth mix, sconv (static and dynamic),
   rel-bias + sink attention, L2 clip, loop control, sigmoid router with bias.
4. Decode flow with 48 cache slots and per-slot conv state; MTP/DSpark drafter
   and the vision tower are out of scope for a first pass.
5. Tokenizer dialect (`<|message_user|>`, `<|message_model|>`, `<|end_message|>`,
   reasoning-effort system line, EOS 200006) and `probe/safetensors.rs` arm.
6. Real-checkpoint greedy and sampled gates, memory oracle, then the catalog row.
