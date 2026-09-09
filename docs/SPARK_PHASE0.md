# Spark-X2.5 Phase 0: facts

Phase 0 fact-finding for `spark2_5` (Spark-X2.5-4B), following
`docs/NEW_MODEL.md`. Every fact here was read off a real artifact, and the
two independent references agree: the HF repo's own `config.json` +
`modeling_spark.py` (`XHToken/Spark-X2.5-4B`), and the official GGUF
conversion's header (`XHToken/Spark-X2.5-4B-GGUF`, file
`Spark-X2.5-4B-Q4_K_M.gguf`), which is byte-written by llama.cpp PR 27868
(merged upstream 2026-09-06, commit `3ad1ba7`). Where the two name the same
thing differently, both spellings are recorded.

## Checkpoint identity

- Repo: `XHToken/Spark-X2.5-4B` (instruct), `XHToken/Spark-X2.5-4B-Base`
  (base), Apache-2.0. A 1.7B sibling exists and is OUT OF SCOPE for the
  first bring-up (different shapes; a family baseline is per-shape).
- `model_type: "spark2_5"`, `architectures: ["Spark2_5ForCausalLM"]`,
  remote code (`configuration_spark.py`, `modeling_spark.py`).
- BF16 safetensors, ~8.2 GB, 4,112,079,360 params.
- Official GGUF repo: BF16, Q8_0, Q4_K_M (2.6 GB).
- Trained context 1,048,576 (`max_position_embeddings` /
  `spark2_5.context_length`).

## Config (both references agree)

| field | value | GGUF kv |
|---|---|---|
| num_hidden_layers | 36 | `spark2_5.block_count` |
| hidden_size | 2560 | `spark2_5.embedding_length` |
| intermediate_size | 10240 | `spark2_5.feed_forward_length` |
| num_attention_heads | 16 | `spark2_5.attention.head_count` |
| num_key_value_heads | 4 | `spark2_5.attention.head_count_kv` |
| head_dim | 256 (both classes) | `attention.key_length` = `attention.value_length` = 256 |
| vocab_size | 131072 | `spark2_5.vocab_size` |
| sliding_window | 512 | `spark2_5.attention.sliding_window` |
| rms_norm_eps | 1e-6 | `attention.layer_norm_rms_epsilon` |
| tie_word_embeddings | true | (no `output.weight` in the GGUF) |
| rope theta FULL | 5,000,000 | `spark2_5.rope.freq_base` |
| rope theta SWA | 10,000 | `spark2_5.rope.freq_base_swa` |
| partial rotary FULL | 0.25 | `spark2_5.rope.dimension_count` = 64 |
| partial rotary SWA | 1.0 | `spark2_5.rope.dimension_count_swa` = 256 |
| hidden_act | "gelu" (exact erf; the code REFUSES anything else) | |
| gate_attn_act_mode | "sigmoid" | |
| headwise_attn_output_gate | true | |
| attention_bias / mlp_bias | false / false | |
| eos / bos / pad token id | 1 / 0 / 2 | |

NOTE THE GGUF NAMING INVERSION: llama.cpp stores the FULL-attention theta
in plain `rope.freq_base` and the SWA theta in `rope.freq_base_swa`, even
though most layers are SWA. Do not map these by "which class is the
default"; map by class.

## Layer mask

`config.layer_types` is 36 entries of `[sliding, sliding, sliding, full]`
repeating: full attention exactly at `i % 4 == 3`, 27 SWA + 9 full. GGUF
`spark2_5.attention.sliding_window_pattern` is `bool[36]` with 1 = SLIDING
(the inverse of this port's `full_attention_layer_mask`, where 1 = full):

```text
GGUF   1 1 1 0 | 1 1 1 0 | ...   (1 = sliding)
port   0 0 0 1 | 0 0 0 1 | ...   (1 = full)  == muse_glimmer_layer_mask
```

This mask is byte-identical to `muse_glimmer_layer_mask`
(`crates/model-io/src/arch_baselines/muse_glimmer.rs`). The single shared
`sliding_window` scalar covers all SWA layers; both classes use head_dim
256 and 4 KV heads (no gemma-style per-class head_dim / kv split).

## RoPE semantics (from `modeling_spark.py`)

- One RoPE cache PER LAYER CLASS (`for lt in set(layer_types)`), each with
  its own theta and its own partial factor. This is the gemma4 shape
  (per-class theta), not the muse shape (NoPE on full layers).
- Partial rotary: `rope_head_dim = head_dim * partial_rotary_factor`
  (64 on full layers, 256 on SWA layers). `apply_rotary_pos_emb` rotates
  ONLY the leading `rope_head_dim` dims as a standard neox half-split
  WITHIN that sub-dim (pairs `(i, rope_head_dim/2 + i)`), frequencies
  computed over `rope_head_dim` (divisor = 64, not 256). That is exactly
  `encode_rope_neox_subdim`'s convention (Qwen-side), NOT
  `rope_proportional_neox`'s (Gemma: full-head pairing, divisor =
  head_dim).
- Mapping onto existing fields: `rope_theta = 10000` (SWA),
  `full_rope_theta = 5000000` (full), `partial_rotary_factor = 0.25`
  applied by the flow to FULL layers only, with the SWA arm at factor 1.0
  hard-coded exactly as `families/gemma4/attn.rs` already does. No new
  ArchConfig field needed. RoPE math in fp32, cast back to bf16.

## Attention

- No QK-norm. q/k go straight into RoPE.
- Fused QKV: HF tensor `self_attn.q_k_v_proj.weight`, GGUF
  `blk.N.attn_qkv.weight`, `[2560, 6144]` laid out q (4096) then k (1024)
  then v (1024) rows. The engine's fused-QKV GEMV path applies.
- Attention scale: `kq_scale = 1 / sqrt(head_dim)` = 1/16 = 0.0625
  (exact binary fraction; llama.cpp computes the same `1/sqrtf(n_embd_head)`).
- Headwise output gate: `self_attn.g_proj.weight` (GGUF
  `blk.N.attn_gate.weight`), `[2560, 16]` = one scalar per head, sigmoid
  (fp32), broadcast over head_dim, multiplied into the attention output
  BEFORE `o_proj`. The gate input is the POST-norm attention input (same
  tensor qkv reads). This gate SHAPE matches nothing in the port today:
  qwen/qwen4 pack `[q; gate]` into one projection with a full-width
  per-head gate; muse has a full-width separate `gate_proj`. The 16-scalar
  variant needs one small kernel (or N tiny `sigmoid_scalar_mul` encodes,
  rejected: 576 extra encodes per token).

## FFN and norms

- Gated MLP `down(gelu(gate(x)) * up(x))`, GELU is HF `ACT2FN["gelu"]` =
  EXACT ERF (the implementation raises on anything else). The trunk paths
  only have silu and tanh-GELU today; erf exists only in the vision
  shader. One small `gelu_erf_mul` kernel needed (decode + prefill arms).
- Pre-norm RMSNorm only: `input_layernorm`, `post_attention_layernorm`,
  final `model.norm`. No sandwich norms, no embedding scaling, no logit
  softcap.

## Tokenizer (from `tokenizer.json` / `tokenizer_config.json`)

- HF fast BPE, vocab 131072, 113 added tokens. GGUF `tokenizer.ggml.pre =
  "spark2_5"` (irrelevant to this port: we consume tokenizer.json
  sidecars, not llama.cpp pre-tokenizer regexes).
- BOS `<｜start▁of▁sentence｜>` id 0, EOS `<｜end▁of▁sentence｜>` id 1,
  pad `<｜▁pad▁｜>` id 2, unk `<unk>` id 5. FIM tokens 11-13.
- `<think>` id 3, `</think>` id 4.
- Turn markers are HALFWIDTH and live in the vocab: `<|System|>` 130972,
  `<|User|>` 130973, `<|Tool|>` 130974, `<|Developer|>` 130975, `<|Bot|>`
  130976. There is NO fullwidth `<｜User｜>` / `<｜Assistant｜>`, so the
  existing `ChatDialect::Deepseek` resolver cannot match. New dialect
  variant required; per Gotcha 52 the detection probe must key on a token
  the resolver requires (use `<|Bot|>`, the defining marker).
- Tool DSL tokens exist (`<tool_call>` 130977, `<arg_key>` 130980,
  `<arg_value>` 130982, `<tool_response>` 130978) but tool calling is
  DESCOPED for the first bring-up.

## Chat template frame

Ships as `chat_template.jinja` and inside `tokenizer_config.json` (and in
the GGUF kv). minijinja-supported constructs throughout (`namespace`,
`tojson`, `loop.previtem`/`loop.nextitem`, type tests). Frame:

```text
system    <｜start▁of▁sentence｜><|System|>\n + "you are a helpful assistant."
          (+ tools block) (+ \n\n + first system message) <｜end▁of▁sentence｜>
user      <｜start▁of▁sentence｜><|User|> + content <｜end▁of▁sentence｜>
assistant <｜start▁of▁sentence｜><|Bot|> + (<think>..</think> | bare </think>)
          + content + tool calls <｜end▁of▁sentence｜>
tool      <｜start▁of▁sentence｜><|Tool|> + <tool_response>..</tool_response>...
          <｜end▁of▁sentence｜>
gen       ...<｜start▁of▁sentence｜><|Bot|> + <think>   (enable_thinking=true)
                                       + </think> (enable_thinking=false)
```

`enable_thinking` defaults to true and the generation prompt FORCES the
think block open (llama.cpp calls this "tagged arguments with forced-open
thinking", reasoning format DeepSeek). generation_config: temp 1.0,
top_p 0.95, top_k -1, single EOS id 1.

## GGUF tensor inventory (Q4_K_M, 290 tensors)

```text
token_embd.weight        [2560, 131072] Q6_K
output_norm.weight       [2560]         F32     (no output.weight: tied head)
per layer (x36):
  attn_norm.weight       [2560]         F32
  attn_qkv.weight        [2560, 6144]   Q4_K or Q6_K (imatrix-driven mix)
  attn_gate.weight       [2560, 16]     Q4_K
  attn_output.weight     [4096, 2560]   Q4_K
  ffn_gate.weight        [2560, 10240]  Q4_K
  ffn_up.weight          [2560, 10240]  Q4_K
  ffn_down.weight        [10240, 2560]  Q4_K or Q6_K (mixed)
  ffn_norm.weight        [2560]         F32
census: 180 Q4_K + 37 Q6_K + 73 F32 = 290
```

Per-tensor Q4_K/Q6_K mixing is the converter's imatrix choice; the port's
per-slot `ggmlType` manifest gate already models per-tensor types and both
block types are in the kernel matrix. Quantization version 2.

## Derived ArchConfig literal (spark_x25_4b)

```rust
hidden_size: 2560,
intermediate_size: 10240,
num_heads: 16,
num_kv_heads: 4,
head_dim: 256,
vocab_size: 131072,
sliding_window: 512,
final_logit_softcap: 0.0,
rope_theta: 10000.0,          // SWA class
full_rope_theta: 5000000.0,   // full class (NOT NoPE: nonzero is legal here,
                              // unlike museglimmer whose flow refuses nonzero)
partial_rotary_factor: 0.25,  // full layers; SWA arm is factor 1.0 (gemma4 pattern)
num_layers: 36,
num_experts: 0,
top_k_experts: 0,
tie_word_embeddings: true,
attention_k_eq_v: false,
full_attention_layer_mask: [0,0,0,1] x9,
hidden_activation: "gelu",    // exact erf; new dispatch, never the tanh fallback
family: ModelFamily::Spark25,
attn_output_gate: false,      // muse precedent: separate g_proj tensor, the
                              // flow applies its gate unconditionally
attention_scale: 0.0625,
embedding_scaled_by_sqrt_hidden: false,
ffn_sandwich_norms: false,
```

`rms eps 1e-6` rides the family baseline; `attentionScale` 0.0625 is 2^-4
and round-trips f64 exactly.

## Decoder layer pseudocode

```text
x     = rms(x, attn_norm)                       # post-norm input, saved as gate input
qkv   = x @ Wqkv.T                              # fused [4096 | 1024 | 1024]
split q, k, v into heads (16/4/4 x 256)
rope(q, k; theta, rotary_dim by class)          # full: 5e6/64, swa: 1e4/256
att   = softmax(mask(q k / 16) v)               # SWA window 512 + ring; full: causal
att   = att * sigmoid(x @ Wg.T)                 # [T,16] broadcast over head_dim
x     = att @ Wo.T + residual
x     = rms(x, ffn_norm)
x     = x + down(gelu_erf(x @ Wg.T) * (x @ Wu.T))
```

Output head: `logits = rms(x, output_norm) @ token_embd.T` (tied). No
softcap, no scaling.

## Open items carried into Phase 1+

- The GGUF pattern kv is 1 = SLIDING; invert when building the mask.
- `tokenizer.ggml` arrays are unused by this port; sidecars come from
  `XHToken/Spark-X2.5-4B` (tokenizer.json, tokenizer_config.json,
  chat_template.jinja, generation_config.json).
- llama.cpp maps the 1.7B (28 layers) and leaves the 4B (36 layers) as
  LLM_TYPE_UNKNOWN; this port keys nothing off layer count.
- Q8_0 and BF16 GGUF variants exist and share the kv/tensor layout; only
  per-slot dtypes differ.
- Upstream verified BF16 end to end; the Q4_K_M mix leans on the port's
  existing Q4_K/Q6_K kernels.
