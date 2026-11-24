# GLM-4.5-Air (`glm4moe`) Phase 0

Status: RECOGNIZED, NOT PORTED, and not gateable on the development machine.
Everything below was read off one header (a ranged fetch of the first 40 MB of
`unsloth/GLM-4.5-Air-GGUF`'s `GLM-4.5-Air-UD-TQ1_0.gguf`, 2026-11-20), and the
arithmetic is done from that header. No weight was downloaded and no flow was
written, because the gate this project requires cannot be run (see the last
section).

## What the file says

- `general.architecture = glm4moe`, 47 blocks, hidden 4096, 96 q heads over 8 kv
  heads at head_dim 128 (q is 12288 wide), plain attention (no MLA), rope base
  1e6 over `rope.dimension_count` 64 of 128 (partial rotary, factor 0.5),
  RMS epsilon 1e-5, pre-tokenizer `glm4`, context 131072.
- Attention carries BIASES on q, k and v (`attn_q.bias` [12288], `attn_k.bias`
  [1024], `attn_v.bias` [1024]) and NO q/k norm tensors.
- Block 0 is a DENSE layer (`leading_dense_block_count` 1): `ffn_gate`, `ffn_up`
  [4096, 10944] and `ffn_down`. Blocks 1 to 45 are MoE: 128 routed experts,
  `expert_used_count` 8, expert ffn 1408, one shared expert (`ffn_*_shexp`,
  width 1408), a router `ffn_gate_inp` [4096, 128] and a router CORRECTION BIAS
  `exp_probs_b.bias` [128].
- Routing: `expert_gating_func` 2 (sigmoid), `expert_weights_norm` true,
  `expert_weights_scale` 1.0, `expert_group_count` 1. This is the `noaux_tc`
  scheme: the bias steers SELECTION only and the sigmoid score (not the biased
  one) is what is normalized into weights.
- Block 46 is the NEXT-TOKEN-PREDICTION layer (`nextn_predict_layers` 1): a full
  attention and MoE layer plus `nextn.eh_proj` [8192, 4096], `nextn.enorm`,
  `nextn.hnorm`, `nextn.embed_tokens`, `nextn.shared_head_head`,
  `nextn.shared_head_norm`. Plain decode must walk 46 layers (0 to 45) and ignore
  block 46; counting it as a layer would run a head that is not part of the model.
- Tied head? No: `output.weight` is present beside `token_embd.weight`.

## Fit against machinery that already exists

| Need | Where it already exists | Gap |
|---|---|---|
| Plain GQA attention, full rotary over a sub-dimension | Llama flow (`rope_neox_subdim`, MiniMax) | none |
| q/k/v biases | Llama flow (`qkv_bias`, Qwen2) | none |
| Sigmoid router with a selection-only bias | Llama flow, MiniMax (`router_scoring_func`, `exp_probs_b`-style correction) | verify weight normalization matches |
| Leading dense layer | `num_dense_leading_layers` (DeepSeek 2, Gemma 4 dense lead) | the Llama flow's per-layer dense/MoE switch is per-INSTALL today |
| Shared expert beside the routed ones | Qwen, DeepSeek 2 flows | NOT in the Llama flow, which states "no shared expert" |
| Dropping the NextN block | none | a name-table row and a layer-count rule |
| `glm4` pre-tokenizer | HF `tokenizer.json` sidecar | none (GGUF-built tokenizer would need the `glm4` regex) |

So the port is the Llama flow plus a shared expert and a per-layer dense/MoE
switch, with the MiniMax router. That touches the Llama flow's MoE tail and its
batched prefill, shared by every Llama-flow MoE family, which is exactly the
kind of change that needs the cross-engine gate to land safely.

## Memory arithmetic (Gotcha 36)

Per-layer routed expert (gate + up + down, header types) is 5.24 to 7.39 MiB
across the 45 MoE layers, so the streamed slot cache is 16 slots x 45 layers x
7.39 MiB = 5.2 GiB (10.4 GiB at 32 slots). That fits and streams like the other
fine-grained families. Total payload is 35.6 GiB (1.4 GiB of it block 46).

## Why it was not ported

The project's bar for a family is a real-file cross-engine gate
(`scripts/kld_llamacpp.py`: this port against llama.cpp on the identical bytes,
on Metal, with a shape floor and a backend floor). That cannot run here:

1. llama.cpp cannot stream, so the reference needs the whole 35.6 GiB file
   resident. This machine has 36 GiB of unified memory, so Metal cannot hold it
   beside the OS, and a CPU run reads 41x further from the Metal floor
   (Gotcha 34), so it would not be a valid reference.
2. The install walk writes another 35.6 GiB, and the reference GGUF is a second
   copy, on a disk with about 11 GiB free.
3. The smallest published `glm4moe` quant is this 35.6 GiB one.

An unverified flow for a family whose failure mode is fluent wrong output is
worse than a refusal, so the registry row stays recognition-only with this
page as its `needs`. A larger-memory machine (64 GiB or more) and about 80 GiB of
free disk would unblock it; the work list is the table above.
