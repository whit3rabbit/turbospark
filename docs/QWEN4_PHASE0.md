# qwen4 Phase 0 findings (Qwen3.8-Flash-Next, `qwen4_exp`)

Fact-finding for the qwen4 bring-up. Everything below is read off the real
published `config.json`, the real `model.safetensors.index.json`, and the
reference implementation in `transformers` at
`src/transformers/models/qwen4_exp/` (Apache-2.0, READ not copied -- see the
"finding a reference" note in AGENTS.md). No weights were downloaded and no
forward pass was run.

Checkpoint: `Qwen/Qwen3.8-Flash-Next`, 131 shards, 360 GB, 1,658 tensors,
BF16. `transformers_version: 5.8.0.dev0`. The repo ships no custom modeling
code.

**TWO independent references, and using both is what makes this document
worth trusting.** `transformers` covers the trunk and refuses the MTP head;
mlx-vlm (`Blaizzy/mlx-vlm`) covers both, at `mlx_vlm/models/qwen4_exp/` and
`mlx_vlm/speculative/drafters/qwen4_exp_mtp/`. They agree on every point
checked against both: the PLE hash and geometry, the centered norm
convention, the below-budget QSA fallback, and the packed `q_proj`. Where
only one covers something it is said so. mlx-vlm is also the project this
port already read for the `qwen3_5` MTP head, so the precedent and the
directory are both familiar.

Sources are cited per item so a post-merge re-verify is cheap. The
llama.cpp GGUF PR (ggml-org/llama.cpp#27742) is still OPEN and is NOT a
source for anything here; every pin below comes from the published
checkpoint or from a runtime reference, neither of which the PR can move.

### Reproducing this (nothing here is vendored)

The references are Apache-2.0 and were READ, not copied, per this repo's
standing licensing decision. Nothing was added to the tree, so re-fetch is
the way back to them. Exact revisions as read on 2026-08-27:

| what | where | revision |
|---|---|---|
| checkpoint | `Qwen/Qwen3.8-Flash-Next` | `de4b8e4d43b917e7706784d8bb445c9af86a3540` |
| trunk reference | `huggingface/transformers` `src/transformers/models/qwen4_exp/` | `fc5c5bde8e656dad91cbf34e61940d984b1c7b91` |
| MTP reference | `Blaizzy/mlx-vlm` `mlx_vlm/speculative/drafters/qwen4_exp_mtp/` | `507f6b0c313b967eeb5e4a0772428f750de914e8` |
| flow reference | `Blaizzy/mlx-vlm` `mlx_vlm/models/qwen4_exp/` | `dda2e04660eca41cc168670eee971c5bb1a15e5c` |

**The mlx-vlm side is moving fast** -- both of its paths were last touched
the same day this was written, one of them three hours before. Re-read rather
than assume, and treat a disagreement with the table above as news about the
reference rather than about this document. `transformers` has been still
since the model landed.

Read `modular_qwen4_exp.py` (1,186 lines, the authored source) rather than
`modeling_qwen4_exp.py` (2,707, generated), except where the modular file
inherits and you need the expansion. The two local mlx-vlm checkouts on this
machine (`~/Documents/GitHub/mlx-v/mlx-vlm`, `~/Documents/GitHub/mlxcel/mlx-vlm`)
both predate qwen4 and have none of this.

## 0. What this overturns in the approved plan

Read this section first. Five plan items were wrong, and three of them are
the fluent-wrong-text kind.

1. **PLE runs at ONE layer, not every layer, and the id is ONE-INDEXED.**
   `ple_layer_ids: [2]` with
   `ple_layer_index = ple_layer_ids.index(layer_idx + 1) if layer_idx + 1 in ple_layer_ids`
   resolves to `layer_idx == 1`. Key decision 6's "each row carrying ALL
   layers' chunks so one pread serves the stack" solves a problem that does
   not exist. Item 4 has the replacement design.
2. **The n-gram table is 320M rows of 160, not 20M rows of 2560.** Same 51.2B
   parameters, different geometry, and the difference decides the entire I/O
   profile: 16 lookups per token of 160 values each, not one lookup of 2560.
3. **Every trunk norm is CENTERED (`x * (1 + w)`), and the GDN gated norm is
   PLAIN.** This is Gotcha 50 across a whole family, with the assignment
   INVERTED relative to `muse_glimmer`. The port's existing `qwen3_5` trunk
   reads plain norms only because mlx-vlm's converter bakes the `+1` in; a
   repack from these BF16 safetensors gets no such favour.
4. **The MTP head DOES carry hyper-connections, and its input fusion is an
   ADDITION rather than the `qwen3_5` concatenation.** The plan expected
   neither. It does not carry PLE. `transformers` discards the head
   (`_keys_to_ignore_on_load_unexpected = [r"^mtp.*"]`); the reference is
   mlx-vlm's, and item 8 has it pinned.
5. **`ple_embed_dim / ngram_heads = 160`, which is not divisible by 64.** The
   port's affine INT4 is group 64. The PLE table cannot use it as-is.

Two plan items are confirmed rather than overturned, and both were the
load-bearing ones: QSA really is exactly dense attention below its budget
(item 5 proves it from source), and text-only mRoPE really does reduce to
`rope_neox_subdim` (item 7), so the Q1 staging and the zero-new-rope-kernels
assumption both hold.

## 1. Tensor inventory and the name table

Prefix `model.language_model.` for the trunk, `model.visual.` for the vision
tower (deferred, dropped at repack as `qwen3_5` does), bare `mtp.` for the
draft head. 1,658 tensors total.

Top level (12): `lm_head.weight`, `model.language_model.embed_tokens.weight`,
`model.language_model.hyper_connection_mixer.{hc_norm,input_mix_weight_down,input_mix_weight_up}.weight`,
and the six `mtp.` non-layer tensors in item 8. `tie_word_embeddings` is
**false**, so `lm_head` is its own tensor. The plan guessed tied.

A QSA layer (24 tensors, e.g. `layers.3.`):

```
attn_hyper_connection.{hc_norm, input_mix_weight_down, input_mix_weight_up,
                       block_inject_weight}.weight
mlp_hyper_connection. {same four}
self_attn.{q_proj, k_proj, v_proj, o_proj}.weight
self_attn.{q_norm, k_norm}.weight
self_attn.indexer.{index_qk_proj, q_layernorm, k_layernorm}.weight
mlp.gate.weight
mlp.experts.{gate_up_proj, down_proj}          <- routed, no `.weight`
mlp.shared_expert.{gate_proj, up_proj, down_proj}.weight
mlp.shared_expert_gate.weight
```

A GDN layer is the same minus `self_attn.*`, plus
`linear_attn.{in_proj_qkv, in_proj_z, in_proj_a, in_proj_b, out_proj, conv1d, norm}.weight`
and the two suffix-less `linear_attn.A_log` / `linear_attn.dt_bias`. Those two
are the same no-`.weight` trap AGENTS.md Gotcha 26 records for `qwen36`, and
`mlp.experts.gate_up_proj` / `mlp.experts.down_proj` are a THIRD spelling of
it -- routed experts here carry no `.weight` either, where Gemma's and
Mixtral's do.

**THERE ARE NO `input_layernorm` OR `post_attention_layernorm` TENSORS.** Every
other family in this port has them. Here the hyper-connection's own `hc_norm`
is the pre-norm for its block, so a name table written by analogy will look
for two tensors per layer that do not exist, and a flow written by analogy
will normalize twice.

The PLE layer (`layers.1.`, 161 tensors) adds:

```
ple.{key_proj, value_proj, conv1d}.weight
ple.{norm_key, norm_query, norm_conv}.weight
ple.ple_embedding.layer_multipliers                    <- i64[3], a BUFFER
ple.ple_embedding.ngram_embedding.shard_{0..127}.weight
```

128 shards, matching this checkpoint's `split_ngram_parts: 128` (the config
CLASS defaults to 512; read the file, not the class). The shards concatenate
on dim 0 into one logical table.

## 2. The decoder layer, as pseudocode

The Phase 0 gate. `H = 2560`, `C = hc_count = 4`, so the residual carried
between layers is `C*H = 10240` wide.

```
# entry, once:  hidden = embed_tokens(ids).repeat(1, 1, C)      # replicated, not padded

def layer(hidden):                       # hidden: [.., 10240]
    if layer_idx == 1:                   # PLE layer, and only this one
        hidden = hidden + ple(hidden, input_ids)                 # item 4

    mixed, raw, inject_w = attn_hc(hidden)                       # item 3
    out = gdn(mixed) if linear else qsa(mixed)                   # items 5, 6
    hidden = raw + (out[..,None,:] * inject_w[..,:,None]).flatten()

    mixed, raw, inject_w = mlp_hc(hidden)
    out = moe(mixed)                                             # item 7
    hidden = raw + (out[..,None,:] * inject_w[..,:,None]).flatten()
    return hidden

# exit, once:   hidden = hyper_connection_mixer(hidden)          # collapse 10240 -> 2560
#               logits = lm_head(hidden)                          # no softcap
```

The trap is on the inject lines: the injection adds to `raw`, the
**un-normalized** hyper input, while the mix that produced `out` read the
**normalized** one. Both are returned by the same call for exactly that
reason. Swapping them decodes fluently and is a different model.

## 3. Hyper-connections (`Qwen4ExpTextGatedResidual`)

```
raw          = hyper_input                                  # [.., C*H]
normed       = hc_norm(raw)                                 # grouped RMS, group=H, CENTERED
w = silu(input_mix_weight_down(normed) / C)                 # C*H -> hc_lowrank (320)
w = sigmoid(input_mix_weight_up(w))                         # 320 -> C*H
mixed        = (w.view(C,H) * normed.view(C,H)).mean(dim=C) # [.., H]   MEAN, not sum
inject_w     = 2 * sigmoid(block_inject_weight(normed) / C) # C*H -> C
return mixed, raw, inject_w
```

Five things that are each individually invisible if wrong: the two `/ C`
divisions, the `2 *` on the inject gate, `.mean` rather than `.sum` on the
mix, the `silu` then `sigmoid` order across the low-rank pair, and the
raw-versus-normed split above.

`hyper_connection_mixer` is the same module with `use_combine=False`: no
`block_inject_weight`, returns `mixed` alone. It runs once after layer 47 and
is what collapses 10240 back to 2560 for the head. Its tensor list confirms
this -- it has three tensors where a layer's has four.

## 4. PLE, re-derived (key decision 6, replaced)

### Geometry

`ngram_heads = (ngram_size - 1) * heads_per_ngram = 2 * 8 = 16`.
`head_dim_per_ngram = ple_embed_dim / ngram_heads = 2560 / 16 = 160`.

Each head gets its own PRIME modulus: head `i` uses the `(i+1)`-th prime
after `ngram_vocab_size_base - 1 = 19_999_999`, and `head_offsets` is the
running sum. Computed here (deterministic, no weights needed):

```
head_vocab_sizes = [20000003, 20000023, 20000033, 20000047, 20000059,
                    20000063, 20000069, 20000077, 20000081, 20000093,
                    20000107, 20000147, 20000153, 20000159, 20000161,
                    20000171]
head_offsets     = [0, 20000003, 40000026, 60000059, 80000106, 100000165,
                    120000228, 140000297, 160000374, 180000455, 200000548,
                    220000655, 240000802, 260000955, 280001114, 300001275]
total_vocab_size = 320001446
padded            = 320001536   (ceil to make_ngram_vocab_size_divisible_by=128)
```

So the table is **320,001,536 rows x 160**, which at BF16 is 102.40 GB =
**95.37 GiB**. That matches the reference's own in-source comment ("this
embedding is so big (~95 GiB)") to the gibibyte, which is what says the
geometry is read right rather than plausible.

Heads 0-7 are BIGRAMS and heads 8-15 are TRIGRAMS. Unigrams are not in PLE at
all; `embed_tokens` covers them.

### The hash

Two different integer widths in one function, which is the thing to get
right.

DERIVATION of the multipliers is **u64 wrapping** (`splitmix64`):

```
multiplier_max = (2^63 - 1) // vocab_size          # 37143089710272
half_bound     = multiplier_max // 2
base_seed      = seed + 10007 * ple_layer_index    # seed=1234, ple_layer_index=0
mult[i]        = 2 * (splitmix64(base_seed + 0x9E3779B97F4A7C15 * (i+1)) % half_bound) + 1
```

Frozen for this checkpoint (`ple_layer_index = 0`, `vocab_size = 248320`):

```
layer_multipliers = [23703573157769, 20109073645365, 8052911324071]
```

These also ship in the checkpoint as `ple.ple_embedding.layer_multipliers`,
so the port can READ them and use the derivation only as a cross-check. Do
both: a checkpoint that ever ships them wrong and a port that only derives
them are the same silent failure from opposite directions.

APPLICATION is **signed i64 and cannot overflow**, by construction:
`multiplier_max` is chosen so `(vocab_size - 1) * max(mult) < 2^63`
(verified: true). So the port uses i64 here, not u64, and the plan's worry
about "sign traps" is real for the derivation and absent for the application.

**CORRECTED 2026-09-02, against the actual reference (this section's
original `ctx = [prev_2, prev_1, cur]` labeling was WRONG and was never run
against source).** `modular_qwen4_exp.py` (`fc5c5bde8`, lines 682-699)
builds `shifted_tokens[shift]` for `shift in 0..ngram_size`, where
`shifted_tokens[0]` is the CURRENT token (shift 0, i.e. unshifted) and
`shifted_tokens[s]` for `s >= 1` is the token `s` positions back. The
multiplier index pairs with the SHIFT, not with a fixed "oldest-to-newest"
position:

```
ctx      = [cur, prev_1, prev_2]            # ctx[s] = shifted_tokens[s], mult[s] pairs with shift s
mixed_2  = ctx[0]*mult[0] ^ ctx[1]*mult[1]              # bigram  -> heads 0..7
mixed_3  = ctx[0]*mult[0] ^ ctx[1]*mult[1] ^ ctx[2]*mult[2]   # trigram -> heads 8..15
row[h]   = (mixed_{2 or 3} % head_vocab_sizes[h]) + head_offsets[h]
ple_embed = concat over h of table[row[h]]              # 16 * 160 = 2560
```

The original draft had `ctx = [prev_2, prev_1, cur]`, which pairs `mult[0]`
with `prev_2` and OMITS `cur` from the bigram entirely (`prev_2*mult[0] ^
prev_1*mult[1]`, a two-tokens-back bigram, not a bigram at all) -- a
plausible-looking formula that decodes fluently and hashes every n-gram to
the wrong row. Caught only by fetching and reading the actual source rather
than trusting this document's own summary; see AGENTS.md's "check the finding
against your own tree" and Gotcha 62 for why a note about an artifact is not
the artifact, even when the note is this one.

One hash value per n-gram order, taken modulo eight DIFFERENT primes to give
eight decorrelated slots. Cheap, and easy to mis-read as eight hashes.

`_shift_right_ignore_eos` resets the n-gram context at EOS boundaries and
pads the start of a sequence with EOS. The EOS used is
`text_config.eos_token_id = 248044` (scalar), NOT the two-entry
`generation_config.json` list. `validate_architecture` refuses PLE with no
EOS set, which is what says this is load-bearing rather than incidental.

**A DECODE-TIME SHIFT REGISTER, DERIVED FROM THE MASKING RULE RATHER THAN
GUESSED.** The reference computes `shifted_tokens[s]` over a whole sequence
with `torch.cummax` over EOS positions; a one-token-at-a-time decode loop
needs the equivalent update rule for its 2-slot state (recurrent state 2,
"PLE n-gram TOKEN IDS", `ngram_size - 1 = 2` long). Working through
`_shift_right_ignore_eos`'s validity condition (`source_position >
previous_eos`) at a single new position `t` collapses to this, carrying
`(prev_1, prev_2)` as the values to use for THIS step and updating them for
the next:

```
# state entering step t: (prev_1, prev_2), both eos_token_id at sequence start
use (cur = x, prev_1, prev_2) to compute ctx for this step's hash
# then update for step t + 1:
new_prev_2 = eos_token_id if x == eos_token_id else prev_1
new_prev_1 = x                                              # always, unconditionally
```

Two things make this rule exact rather than approximate, both provable from
the masking condition rather than assumed. `new_prev_1 = x` unconditionally
because `shifted_tokens[1]` at the next position is masked to `eos_token_id`
*exactly* when `x` itself is `eos_token_id` -- so reading the raw fed token
already agrees with the masked value in every case, eos or not. And
`new_prev_2` collapses to "`eos_token_id` if `x` was eos, else carry `prev_1`
forward" because on a non-eos step it is definitionally what
`shifted_tokens[1]` was at THIS step (the same masked value this step already
computed as its own `prev_1` input), and on an eos step the masking rule
forces it to `eos_token_id` regardless of what the two-back token actually
was -- the one case where "just read the raw FIFO" would silently disagree
with the reference for exactly one token after every EOS.

### The module

```
emb    = ngram_table_lookup(ids)                      # [.., 2560]
key    = norm_key(key_proj(emb)).view(C, H)           # 2560 -> 10240, grouped norm
value  = value_proj(emb)                              # 2560 -> 2560
query  = norm_query(hidden).view(C, H)                # hidden is the 10240 residual
gate   = (key * query).sum(-1) / sqrt(H)              # [.., C, 1]
gate   = sign(gate) * sqrt(max(|gate|, 1e-6))         # SIGNED sqrt
gv     = sigmoid(gate) * value                        # broadcast to [.., C, H]
out    = gv.flatten() + silu(dilated_depthwise_conv(norm_conv(gv.flatten())))
```

The signed square root is the sort of line that reads as a typo and is not.
The conv is depthwise over all 10240 channels, `kernel_size=4`,
**`dilation = ngram_size = 3`**, so its state length is
`(4 - 1) * 3 = 9`. Note the residual: the conv is fed the NORMED gated value
and its output is added to the UN-normed one.

### Three recurrent states, not two

`number_of_conv_states = 3 if ple_layer_ids else 1`:

| idx | contents | length |
|---|---|---|
| 0 | GDN causal conv | `linear_conv_kernel_dim` = 4 |
| 1 | PLE dilated conv | 9 |
| 2 | PLE n-gram TOKEN IDS | `ngram_size - 1` = 2 |

State 2 holds raw token ids rather than activations, reusing the conv-state
machinery as a shift register. The plan's instinct to keep the PLE conv off
the GDN conv was right and understated: they are three separate buffers, only
one of which holds floats, and only layer 1 has states 1 and 2 at all.

### What this means for the streamer

Per decoded token: **16 random row reads**, 160 values each. At BF16 that is
320 bytes per row; the rows are scattered across a 102 GB file, so each is a
separate page. The plan's "all layers per row" layout is gone; what replaces
it is a flat `[padded_vocab, 160]` table plus the 16 `(vocab_size, offset)`
pairs in `layout.json`.

Quantization has a constraint the plan did not have: **160 is not divisible
by 64**, so the port's affine INT4 group 64 does not fit a row. Group 32 does
(5 groups), giving 100 bytes per row and a 32.00 GB table. That is the number
to plan the install around, and `is_supported_affine_shape` has to admit it.

The open risk is unchanged in kind and larger in degree: 16 scattered reads
per token, mitigated by an LRU row cache over a Zipfian access pattern. The
hit rate is a measurement, not an estimate, and Phase 5 owes it.

## 5. QSA, and the proof that Q1 is exact

`layer_types` is rewritten in `__post_init__`: every `"full_attention"` entry
in the published config becomes `"qwen_sparse_attention"`. There are no dense
attention layers in this model.

The indexer, per QSA layer:

```
qk        = index_qk_proj(hidden)                    # 2560 -> (4+1)*128 = 640
q, k_tok  = split(qk, [4*128, 1*128])
q         = rope(q_layernorm(q), current_positions)  # per-head norm, dim 128
k_tok     -> cached RAW, per layer, un-normed and un-roped
```

Then per query, over the visible keys:

```
blocks    = consecutive runs of compress_ratio=4 visible tokens
pooled    = mean(block keys)  in FP32
pooled    = rope(k_layernorm(pooled), position of the block's FIRST token)
scores    = relu(q @ pooled^T).sum(over the 4 heads) / sqrt(128)
chosen    = topk(scores, min(block_topk, num_complete_blocks))   # block_topk = 2048/4 = 512
selected  = the 4 tokens of each chosen block, PLUS the ragged tail
```

The tail (the `visible % 4` trailing tokens that form no complete block) is
ALWAYS selected. The result is a boolean mask ANDed onto the causal mask.

**Below-budget exactness.** All blocks are chosen exactly when
`floor(visible / 4) <= 512`, i.e. `visible <= 2051`. At or under that, every
visible token is selected, the mask is a no-op, and QSA is bit-for-bit dense
attention. The plan's Q1 staging (hard-cap `max_context` at 2048) is
therefore exact, with three tokens of margin, and this is read off the source
rather than taken from the PR's measurement.

mlx-vlm's independent implementation makes the same claim structurally rather
than as a consequence: it computes `use_sparse = complete_counts > block_topk`
and returns the plain causal mask when that is false. Two references, two
codepaths, one answer.

The indexer cache is 1 head x 128 dims per token per QSA layer: at 8,192
context that is 2 MiB per layer, 25 MiB over 12 layers. Cheap, and it must
share a position counter with the KV cache -- the plan's `QsaCacheManager`
decision stands, and the upstream restore-misalignment bug it was written
against is a real hazard because these are two caches advanced by one loop.

`validate_architecture` asserts `rotary_dim (64) <= indexer_head_dim (128)`
and `indexer_kv_heads == 1`.

## 6. Attention proper, and the packed q_proj

`attention_bias: false`, so no biases anywhere. 24 q heads, 2 kv heads,
`head_dim` 256. `scaling = head_dim ** -0.5 = 0.0625`, a binary fraction and
therefore safe for Gotcha 24's `!=` float comparison.

**`q_proj` is PACKED `[query; gate]` and the interleave is PER HEAD.** It is
`hidden -> num_heads * head_dim * 2` (2560 -> 12288), viewed as
`(.., num_heads, 2 * head_dim)` and chunked in two along the LAST axis. So
the layout is `head0_query[256], head0_gate[256], head1_query[256], ...`, not
`all_queries[6144], all_gates[6144]`. Splitting it down the middle gets a
correctly-shaped, fluently-decoding wrong model.

**This is NOT a new convention and needs no new code.** mlx-vlm derives
`Qwen4ExpAttention` from `Qwen3_5Attention` unchanged, and the port already
has the kernel: `gpu::encode_split_q_gate` splits "a `[heads, 2 * dim]`
packed projection into contiguous `[heads, dim]` q and gate buffers",
documented against Qwen 3.6's `q_proj` and dispatched today from
`families/qwen/attn.rs`. The qwen4 flow reuses it and
`encode_sigmoid_gate_mul` beside it. Recorded here because the SHAPE is a
trap for a repack written from the tensor list alone, not because the flow
has anything to solve.

The gate is applied AFTER attention and BEFORE `o_proj`:
`attn_out = attn_out * sigmoid(gate)`. That is the port's existing
`attnOutputGate`.

`q_norm` and `k_norm` are per-head over `head_dim` 256, and CENTERED (item
9).

## 7. MoE

Every layer has one. 512 experts, top-10, `moe_intermediate_size` 640,
`shared_expert_intermediate_size` 640.

```
shared    = shared_expert(x)                              # gate/up/down, silu
routed_w  = softmax(gate(x), dim=-1, dtype=fp32)          # over all 512
top_v, top_i = topk(routed_w, 10)
top_v    /= top_v.sum(-1, keepdim=True)                   # norm_topk_prob = True
out       = experts(x, top_i, top_v) + sigmoid(shared_expert_gate(x)) * shared
```

Softmax comes BEFORE the top-k here, unlike `gpt-oss` where the selection is
made on biased raw logits. Selection is unaffected (softmax is monotone), but
the WEIGHTS are renormalized top-10 probabilities, and the renormalization is
not optional. The router is a plain `[512, 2560]` matrix with no bias term.

The shared expert is gated by a [1, 2560] linear through a sigmoid and
ADDED after the routed sum. SlotStream's Qwen4 reference casts the routed
sum to the model dtype before adding the gated shared output; preserve that
FP16 boundary in the runtime. This differs from Qwen3's shared-seeded
phase-2 reduction, so do not reuse that family convention here.

One expert at INT4 is ~2.5 MiB (`2 * 640 * 2560` gate_up plus
`640 * 2560` down, at 4 bits plus scales), so the slot cache at 16 slots over
48 layers is ~1.9 GiB. Gotcha 36's multiplication is favourable here, as the
plan said.

## 8. The MTP head

31 tensors under `mtp.`, and the structure is a **complete qwen4 decoder
layer**:

```
mtp.pre_fc_norm_embedding.weight        mtp.fc_embedding.weight
mtp.pre_fc_norm_hidden.weight           mtp.fc_hidden.weight
mtp.hyper_connection_mixer.{hc_norm, input_mix_weight_down, input_mix_weight_up}.weight
mtp.layers.0.{attn_hyper_connection, mlp_hyper_connection}.{four each}
mtp.layers.0.self_attn.{q,k,v,o}_proj.weight, {q,k}_norm.weight
mtp.layers.0.self_attn.indexer.{index_qk_proj, q_layernorm, k_layernorm}.weight
mtp.layers.0.mlp.{gate, experts.gate_up_proj, experts.down_proj,
                  shared_expert.*, shared_expert_gate}.weight
```

So, against the plan's expectations: the head **DOES** carry hyper-connections
(both of them, plus its own final mixer), and does **NOT** carry PLE. It is a
QSA layer with a full 512-expert MoE, which makes it far from free -- this is
not the ~1.5%-of-a-pass drafter `docs/MTP.md` records for `qwen3_5`. There is
no `mtp.embed_tokens` and no `mtp.lm_head`, consistent with
`mtp_use_dedicated_embeddings: false`: it shares the trunk's embedding and
head.

`transformers` does not run the head:
`_keys_to_ignore_on_load_unexpected = [r"^mtp.*"]` loads the checkpoint and
throws it away. **The reference is mlx-vlm's**, at
`mlx_vlm/speculative/drafters/qwen4_exp_mtp/`, which is the same project and
the same directory the port already read for the `qwen3_5` head. Everything
below is from it. A second implementation exists in the sglang lineage
(`python/sglang/srt/models/qwen4_exp_mtp.py`) if a tiebreak is ever wanted.

### The fusion, and why `docs/MTP.md` is the wrong prior for it

```
proj_embed  = fc_embedding(pre_fc_norm_embedding(token_embed))   # [.., H] -> [.., H]
streams     = pre_fc_norm_hidden(hidden).reshape(.., C, H)       # GLOBAL norm over C*H
proj_hidden = fc_hidden(streams)                                 # [H, H], applied per stream
fused       = (proj_embed[.., None, :] + proj_hidden).reshape(.., C*H)
```

**It is an ADDITION broadcast over the four streams, not a concatenation.**
Both `fc_embedding` and `fc_hidden` are `[2560, 2560]`, where `qwen3_5`'s
single `fc` is `[H, 2H]` over a concatenated pair. So the concat-order
question that `docs/MTP.md` records being got wrong does not exist here:
there is no concat. What replaces it is a broadcast axis, and getting THAT
wrong (summing over streams rather than broadcasting across them) is the
same class of fluent-wrong failure.

`pre_fc_norm_hidden` is a **plain full-width** norm over all 10240, not the
grouped-by-2560 norm the trunk uses. mlx-vlm says so in a comment ("one
global RMS normalization across all hyper-connection streams before
projecting them independently") and its tensor is `[10240]`. Item 9 has the
full norm taxonomy; this is the shape that only appears here.

### Lifecycle

The head runs as a complete qwen4 layer with `layer_types =
["qwen_sparse_attention"]` and `ple_layer_ids = []`, followed by its own
`hyper_connection_mixer(use_combine=False)`, then the TRUNK's `lm_head`. It
declares `supports_greedy_draft_argmax`, `prefer_requested_block_size` and
`requires_uniform_batch_acceptance`, and its `quant_predicate` refuses to
quantize `mlp.gate` (the router stays full precision, as this port already
does with its INT8 router).

On depth, mlx-vlm's README is explicit and matches the plan's instinct
without endorsing it: "The native checkpoint contains one MTP layer, so the
default draft block is one speculative token. A larger `--draft-block-size`
chains the same head autoregressively; whether that is faster depends on
prompt and hardware." So depth > 1 is CHAINING one head, not running trained
extra steps, and it is a measurement rather than a property. Per-position
acceptance is the metric, per Gotcha 50's closing note about saturated
scalars.

### One repack fact that falls out of it

`sanitize` splits the routed `mlp.experts.gate_up_proj` with
`mx.split(gate_up, 2, axis=-2)` into gate then up. **Gate is the FIRST
half**, which is the same question Gotcha 29 had to settle by correlation for
Gemma's fused GGUF tensor, answered here by reading a converter instead.

## 9. Norm inventory and checkpoint conventions

The Hugging Face model stores most `Qwen4ExpTextRMSNorm` parameters as offsets
from unity, so its source equation is `x * (1 + w) / sqrt(mean(x^2) + eps)`.
That is not the representation consumed by this runtime. The llama.cpp GGUF
converter folds `1 + w` into each published GGUF scale; `GGUF -> GTurbo`
repacking preserves those scale values. Applying a centered kernel after
repacking adds unity a second time. Use direct-scale kernels for the GGUF
weights in TurboSpark.

SlotStream's `qwen4_exp.py` confirms the source-level conversion by listing
these tensors in `CENTERED_NORMS` and shifting them during sanitization. The
matching llama.cpp graph comments that the converter folded each scale to
`(1 + w)` and multiplies the GGUF `hc_norm` values directly. The real IQ2_XS
comparison below caught the double shift in TurboSpark's prior implementation.

`linear_attn.norm.weight` is the separate GDN gated norm. It is a direct scale
and applies its gate activation after normalization:

```
h = h * rsqrt(mean(h^2) + eps); h = weight * h; h = h * sigmoid(gate)
```

`rms_norm_eps` is 1e-06 throughout. The runtime norm inventory is:

| shape | width / grouping | where |
|---|---|---|
| grouped direct scale | weight 10240, stat over each 2560 | `hc_norm` x2 per layer, final mixer, PLE `norm_key` / `norm_query` / `norm_conv` |
| per-head direct scale | weight = head_dim, stat per head | `q_norm`, `k_norm` (256), indexer `q_layernorm` / `k_layernorm` (128) |
| full-width direct scale | weight = full width, one stat | MTP `pre_fc_norm_embedding` (2560), `pre_fc_norm_hidden` (10240), if that path is added |
| direct gated scale | weight = head_v_dim, one stat | GDN `linear_attn.norm`, times `sigmoid(z)` |

The grouped direct-scale kernel uses one statistic per hidden-width residual
stream and a full-width scale vector. It must remain distinct from both the
grouped centered kernel and the full-width direct-scale kernel: the reduction
scope and scale convention are separate choices.

## 10. RoPE

`rope_theta` 1e7, `partial_rotary_factor` 0.25, `head_dim` 256, so
`rotary_dim = 64`. `rope_type` is `"default"` and `attention_scaling` is
1.0 -- no YaRN, no scaling function.

`apply_rotary_pos_emb` splits `q[..., :64]` from `q[..., 64:]`, rotates the
first 64 with `rotate_half` (NeoX half-split within the sub-dimension), and
concatenates. **That is exactly `rope_neox_subdim`, which the port already
has.**

mRoPE: `position_ids` is 3-channel `(t, h, w)` with
`mrope_section = [11, 11, 10]` summing to 32 = `rotary_dim / 2`, and
`mrope_interleaved: true` reorders the frequency layout from
`[TTT..HHH..WWW]` to `[THWTHW..]`. For TEXT-ONLY input all three channels
carry the same position, so the interleaved selection is a no-op and the
whole thing collapses to standard RoPE over the first 64 dims. **Zero new
rope kernels for the text path**, as the plan's decision 7 hoped. The
interleave matters only once images are in play, which is deferred.

## 11. Chat, EOS, and the token budget

`generation_config.json`: `eos_token_id: [248046, 248044]`,
`bos/pad: 248044`, `do_sample: true`, `temperature 1.0`, `top_k 20`,
`top_p 0.95`. The stop set is the two-entry list; PLE's internal EOS is the
scalar 248044 from `text_config` and the two must not be conflated.

The dialect is **ChatML** (`<|im_start|>` / `<|im_end|>`, `<think>`), so
`detect_dialect` should resolve it against the existing arm. Gotcha 52's
check still applies: confirm the probe cannot pass where `resolve_chatml`
would fail.

### The generation prompt opens a `<think>` frame it does not close

This is Gotcha 56's ChatML trap, in the form the port has already been bitten
by once:

```jinja
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
    {%- if enable_thinking is defined and enable_thinking is false %}
        {{- '<think>\n\n</think>\n\n' }}     {# closed, empty #}
    {%- else %}
        {{- '<think>\n' }}                   {# OPEN #}
    {%- endif %}
{%- endif %}
```

So with thinking on -- which is the DEFAULT, since the test is
`enable_thinking is defined and enable_thinking is false` -- the prompt ends
mid-frame and the model's first generated token is already scratchpad.
`think_start_id` never arrives, so a decoder that starts in the visible
channel stays there for the whole turn and returns the reasoning as the
answer. The port's fix for `qwen38` (give `StructuredAssistantDecoder::new`
the prompt ids to scan) is the same fix, and Phase 4 owes a check that it
actually engages for this family rather than assuming it generalizes.

### Reasoning levels, and a level with no text

Accepted: `xhigh` (default), `medium`, `low`. **`high` RAISES**, same as
`Qwen/Qwen3.8-27B`. The union-of-levels approach in Gotcha 56 already handles
that: the checkpoint's own error is the allowlist.

The trap is `medium`: the template has branches for `xhigh` and `low` only,
so `medium` renders an EMPTY `reasoning_instructions` and contributes no
system text at all. It is still distinguishable from `off` (open frame versus
pre-closed frame), so it is not a silent no-op and the `--reasoning` guard's
"either MOVES the render or is refused" invariant holds -- but it holds for a
reason that is one `elif` away from not holding, and the guard should assert
the frame state rather than just that the bytes differ.

`preserve_thinking` defaults to TRUE, so every historical assistant turn is
re-rendered with its `<think>...</think>` block. That inflates prompts across
a multi-turn conversation and interacts with window fitting.

### Tool calls are an XML DSL, not JSON

```
<tool_call>
<function=NAME>
<parameter=KEY>
value
</parameter>
</function>
</tool_call>
```

This is NOT the `<tool_call>{...json...}</tool_call>` that the port's ChatML
arm parses. A qwen4 tool call would open a recognized `<tool_call>` and then
fail to parse its body. Phase 7 owes either a new DSL arm or an explicit
refusal; silently returning prose is the outcome to avoid.

### minijinja risks (Gotcha 53)

Not yet run through the renderer. Four constructs to check before assuming
the template parses, in rough order of suspicion: `loop.previtem` and
`loop.nextitem` (the `tool` role branch depends on both), the reversed slice
`messages[::-1]`, the keyword argument
`namespace(multi_step_tool=true, last_query_index=messages|length - 1)` --
which is `k=EXPR` with arithmetic, adjacent to what
`parenthesize_conditional_kwargs` already shims but not the same expression
class -- and `raise_exception`, which the template calls in six places.

### Token budget

Thinking on at `xhigh` by default. The shared 1,024-token budget is very
unlikely to hold; measure it, as `gpt-oss` (3,072) and `muse_glimmer` (2,048)
both had to.

## 12. Memory model, revised

At 4,096 context, 16 expert slots, INT4:

| term | size | note |
|---|---|---|
| resident core | ~2.3-2.8 GiB | unchanged from plan |
| expert slot cache | ~1.9 GiB | 16 x 48 x ~2.5 MiB |
| KV | 96 MiB | 12 QSA layers x 2 kv heads x 256 |
| GDN state | 108 MiB | 36 layers |
| indexer cache | ~12 MiB | 12 layers, 1 head x 128 |
| PLE row cache | 0.5-1 GiB | policy, not forced |
| PLE conv + ngram state | negligible | one layer |
| scratch, baseline | ~1 GiB | includes the 10240-wide ping-pong |

Total ~6-7 GiB, unchanged from the plan's figure. **Disk is still the binding
constraint**, and the PLE half of it is now firm rather than estimated: 32.00
GB at INT4 group 32, against ~72 GB of trunk, so ~104 GB installed and 360 GB
streamed through.

## 13. Open items

1. **MTP cost.** Pinned and closed as a DATAFLOW (item 8), open as an
   ECONOMIC question. The head is a full QSA + 512-expert MoE layer, so
   `docs/MTP.md`'s "~1.5% of a pass" does not transfer and the ceiling has to
   be re-derived before any block-size table is trusted. Note also that a
   recurrent trunk makes a rejected batched round expensive, which is the
   term AGENTS.md records no composite having had.
2. **minijinja against the real template.** Four suspect constructs listed in
   item 11. Cheap to settle: render the protocol prompt and see.
3. **The tool-call DSL.** New XML form, no existing arm parses it.
4. **GDN parity against the port's existing baseline.** Shapes match
   `qwen_gdn_dense_27b` field for field on paper
   (`{16, 48, 128, 128, 4}`); the V-head interleave convention and the
   `A_log` / `dt_bias` transforms are inherited from `Qwen3_5GatedDeltaNet`
   and have not been re-verified against this checkpoint's actual bytes.
4. **The grouped-RMS kernel.** Whether
   `rmsnorm_bf16w_perhead_centered` generalizes or a new kernel is needed.
5. **Vision.** Deferred. Worth noting the tower is architecturally identical
   to the `qwen3_5` one being ingested in M-V3 (depth 27, hidden 1152,
   intermediate 4304, 2304 position rows) except `out_hidden_size` is 2560
   here against 5120 there, following the trunk width.

## 14. Lessons from doing this

Recorded here rather than in AGENTS.md because none of it is earned from
running code yet. Promote whatever survives Phase 3 to a numbered gotcha then.

**A plan is not evidence, and this one cost five wrong decisions.** The
approved plan was built from the model card and an open llama.cpp PR. One
`config.json` fetch and one reference read -- minutes, no download -- moved
five of its items, and one of those (key decision 6's PLE row layout) was a
DESIGN built on a premise nobody had checked. The generalisable form is
Gotcha 62 one level up: a plan's *decisions* inherit the confidence of its
weakest *fact*, and nothing in the document's shape tells you which fact that
is. Do Phase 0 before the decisions, not to ratify them.

**The blog's parameter count hid the geometry.** "51B n-gram embeddings"
times a 2560 hidden width says 20M rows, which is what the plan wrote. The
real table is 320M rows of 160, same 51.2B parameters, and the shape is what
decides the whole I/O design (16 scattered small reads per token, not one
large one). A total is invariant under exactly the reshape you care about.
Same species as Gotcha 48's level histogram.

**One-indexed layer ids exist.** `ple_layer_ids: [2]` means `layer_idx == 1`.
Nothing in the config says so; the resolution is in the model code
(`layer_idx + 1 in config.ple_layer_ids`) and the docstring. An off-by-one
here injects a 51B-parameter embedding into the wrong layer, which decodes
fluently. Whenever a config names a layer, find the line that RESOLVES it.

**`_keys_to_ignore_on_load_unexpected` is a one-grep answer to "is there a
reference for this component".** `transformers` loads this checkpoint and
throws the MTP head away. Finding that immediately is what redirected the
search to mlx-vlm instead of producing a half-derived head from tensor names.
Before deriving a component's dataflow, check whether your reference actually
RUNS it -- a reference that silently drops a component looks exactly like a
reference that supports it.

**Two references beat one, and the value was in the agreement.** Every
cross-checkable item matched (PLE hash and geometry, centered norms,
below-budget QSA, packed `q_proj`), which is what makes the single-sourced
items safe to build on. It also corrected me twice in the same pass: the
packed `q_proj` turned out to be an existing solved convention rather than a
new trap, and the centered-norm finding got STRONGER, because mlx-vlm bakes
the `+1` for `qwen3_5` and does not for qwen4. A single reference would have
left both wrong in opposite directions.

**Check the finding against your own tree before writing it down.** The
`q_proj` warning survived one draft as a Gotcha 33 hazard. `encode_split_q_gate`
has handled exactly that layout since Qwen 3.6, documented in its own
docstring. The cost of the check was one grep.

**The worktree was empty and the subject was not.** Gotcha 13, again: this
worktree opened at HEAD while the M-V3 vision work sat unstaged in the main
checkout, touching five of the files Phase 1 must edit. That grew from 60 to
76 modified files during this session alone. It is worth running
`git status` in the main checkout before accepting a worktree, every time.

## 15. Where this stands, and what unblocks it

**Done:** Phase 0, this document. No code written, no dependency added,
nothing vendored.

**Blocked on M-V3 landing**, and precisely: `arch_config/sub_configs.rs`,
`arch_config/config.rs`, `arch_validation.rs`, `gturbo_writer/manifest.rs`
and `gemma4_checkpoint/classify.rs` are all dirty in the main checkout with
vision work, and all six `arch_baselines/*.rs` are gaining a
`VisionConfig::NONE` field that a qwen4 baseline written today would lack.
Phase 1 edits every one of those files. Starting it now buys a merge
conflict in five files and a baseline that does not compile.

**Owed and deliberately not done here: the AGENTS.md layout entry for this
file.** It belongs in the `docs/` tree between `POWER_BASELINE.md` and
`RELEASE.md`. AGENTS.md went dirty in the main checkout partway through this
session (the M-V3 work is rewriting it), so editing it from this branch would
have contested a file another session holds. Add the line during the rebase
below, when there is one copy of it again.

**When vision lands**, in order:

1. Rebase this branch onto the commit that carries M-V3, and confirm
   `VisionConfig` is in `ArchConfig` before writing the qwen4 baseline --
   the new baseline literal needs a value for it.
2. Phase 1 as the approved plan has it, with these amendments from above:
   `PleConfig` carries the 16 `(vocab_size, offset)` pairs and a single
   `ple_layer_id`, not a per-layer table; `IndexerConfig` is the five QSA
   fields; the manifest keys go in unconditionally (Gotcha 24). Note qwen4's
   own `vision_config` is the M-V3 tower at a different `out_hidden_size`,
   so the field is genuinely populated here rather than `NONE`, even though
   the tower is dropped at repack.
3. Before any download, the header-only network test: assert the 1,658-name
   inventory of item 1 still maps, and that the PLE shards and MTP tensors
   are where item 1 says. Seconds, and it is the gate that keeps a wrong
   name table from costing a 360 GB re-stream.

**Cheap work available NOW that touches none of the contested files**, if
someone wants to move before vision lands:

- Render the real `chat_template.jinja` through minijinja and settle item
  11's four suspect constructs. `crates/tokenizer` only.
- The CPU reference for hyper-connections
  (`crates/compute/src/hyper_connections.rs`) and for the PLE hash and step
  (`crates/compute/src/ple.rs`), both written from items 3 and 4 and both
  new files in a crate the vision work does not touch. The plan's Phase 2
  wants these first anyway, and the frozen constants in item 4 make the PLE
  hash directly testable with no model at all.
- `gdn_gated_norm_sigmoid` plus its `GateActivation` CPU parameter and
  discrimination assertion (decision 4), also self-contained.
