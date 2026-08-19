# The DFlash2 drafter: architecture off real bytes, and where this port is

What `incoai/Qwen3.8-27B-DFlash2` is, read off the published checkpoint
header (8,928 B JSON, ranged reads, no download) and off the two reference
implementations (vLLM `qwen3_dflash.py` + `qwen3_dflash2.py` + their
speculators, and llama.cpp's `draft-dflash`). Sections 1-5 hold FACTS about
the drafter. Section 6 holds THIS PORT'S STATE, which as of 2026-08-19 is
"built and blocked". Section 7 holds the comparison against the two
references, including the divergences the comparison found. Section 8 is the
work list. Nothing projected appears in sections 1-5.

DFlash2 is a block-diffusion DRAFTER for speculative decoding, aimed at the
same checkpoint this engine runs as the dense `qwen3_5` family
(`Qwen/Qwen3.8-27B`). It proposes a whole block of tokens in ONE forward
pass over `block + 1` rows; the target verifies them in one batched pass.
Published end points: acceptance length 5.34 at 7 draft tokens and 3.51x at
concurrency 1 on an H200 through vLLM (PR 52816), and 1.77-1.85x at
acceptance 4.92-5.08 on an Apple M5 Pro through llama.cpp (PR 27342). Those
are OTHER ENGINES' numbers; this page does not carry them into any decision.

## 1. The checkpoint, off real bytes

One `model.safetensors`, 81 tensors, 3,848,808,960 B, BF16 throughout,
Apache-2.0. No `embed_tokens`, no `lm_head`, no `mask_embedding` sidecar:
the mask row is embedded through the TARGET's table at `mask_token_id`
248070, and logits come from the TARGET's `lm_head`. That is the same
sharing doctrine as the MTP head (`docs/MTP.md`).

| group | tensors | shape | notes |
| --- | ---: | --- | --- |
| `fc.weight` | 1 | [5120, 25600] | 5 aux states (5 x 5120) fused to 5120 |
| `hidden_norm.weight`, `norm.weight` | 2 | [5120] | encoder norm, final norm |
| `layers.{0..4}.self_attn.{q,k,v,o}_proj` | 4 x 5 | [4096/1024/1024/5120, 5120] etc. | 32 q heads, 8 kv heads, head_dim 128, PLAIN GQA (no trunk gate packing) |
| `layers.{0..4}.self_attn.{q,k}_norm` | 2 x 5 | [128] | per-head, over head_dim |
| `layers.{0..4}.{input,post_attention}_layernorm` | 2 x 5 | [5120] | |
| `layers.{0..4}.mlp.{gate,up,down}_proj` | 3 x 5 | [17408, 5120] etc. | silu MLP, inter 17408 |
| `layers.{0..4}.{attention,mlp}_conv.base_kernel` | 2 x 5 | [2, 2, 5120] | [side, tap, hidden] |
| `layers.{0..4}.{attention,mlp}_conv.kernel_projection` | 2 x 5 | [1280, 5120] | 2 sides x 2 taps x 320 groups |
| `candidate_selector.{predecessor,successor}_codebook` | 2 | [248320, 256] | 127 MB each |
| `candidate_selector.hidden_projection.weight` | 1 | [256, 5120] | |

Config beside the weights: 5 layers, all `sliding_attention`, window 2048;
hidden 5120; RoPE NeoX theta 1e7 at head_dim 128; rms eps 1e-6;
`block_size 8`; `conv_kernel_size 2`, `conv_group_size 16`;
`selector_rank 256`, `selector_top_k 16`; `target_layer_ids [5, 19, 33,
47, 61]` of 64; vocab 248320 (equal to the target's, so no id mapping);
`input_embedding_scale`, `output_multiplier` and `final_logit_softcapping`
absent, i.e. 1.0 / 1.0 / none.

**Every norm in the drafter is PLAIN** (`x * w`), unlike the MTP head's
seven CENTERED ones. Read off the bytes, not assumed (Gotcha 50's rule):
whole-vector means 1.71-2.08 (hidden_norm 1.8309, final norm 2.0812,
layer 0 input/post 1.7450/1.7052), per-head q/k norms 1.9432/1.9279 at
layer 0. Centered weights would read ~0.78.

**`block_size 8` BOUNDS THE ROW COUNT, NOT THE PROPOSAL COUNT.** llama.cpp
clamps its selector at `block_size = min(tokens_per_block,
hparams.dflash_block_size)` and emits proposals for rows `1 .. block_size`,
so a `block_size` of 8 yields at most SEVEN proposals out of an eight-row
block (anchor + 7 masks). vLLM's headline is likewise quoted "at 7 draft
tokens". See section 7 for what that means for this port, which currently
runs nine rows.

## 2. The aux states, and which boundary "layer 5" means

The drafter conditions on the TARGET's hidden states captured at trunk
layers `[5, 19, 33, 47, 61]`. Both references resolve this id to the same
tensor by ADDING ONE:

- vLLM: `eagle3_utils.get_eagle3_aux_layers_from_config` maps DFlash's
  `target_layer_ids` through `[i + 1 for i in ...]`, and the target model
  appends its aux entry with index `layer_idx + 1` AFTER layer `layer_idx`
  runs, so the captured tensor is the OUTPUT of layer `t`.
- llama.cpp: the GGUF converter (`conversion/qwen.py`) writes
  `extract_layer_ids = [i + 1 for i in target_layer_ids]`, the runtime arms
  the taps with `llama_set_embeddings_layer_inp(ctx_tgt, target_layer_ids[k],
  true)` against those already-incremented ids, and reads them back with
  `llama_get_embeddings_layer_inp`. Same tensor.

So the tap is the RESIDUAL STREAM AT THE OUTPUT of trunk layers
5/19/33/47/61: raw layer outputs, no final norm. That is the opposite
convention from the MTP head, whose hidden input is the trunk's
POST-FINAL-NORM state (`docs/MTP.md`). Two drafters, two conventions; the
fc weight was trained against the raw one here.

## 3. The round, end to end

Notation: the target has just run a pass over rows at positions
`P0 .. P0+M` (a verify batch or a prompt token); `k` of the verify rows
were accepted (the anchor row plus `k - 1` accepted proposals; for a
prompt walk, every row is accepted). The last committed token (the bonus)
sits at position `P0 + k`. All formulas below are the reference's, read
from vLLM and cross-checked against llama.cpp where it implements the same
step.

1. **Context KV write.** For every accepted row at position `p`:
   `h(p) = fc(concat(aux_5(p), ..., aux_61(p)))`, then
   `n = rms_norm(h, hidden_norm)`, then per drafter layer `l` the k/v rows
   are `n @ [k_l; v_l]` (a fused GEMM in the reference; the same weights
   as the layer's own `k_proj`/`v_proj` rows), `k_l` then per-head
   `k_norm` (plain), then RoPE (NeoX, theta 1e7) at `p`, and K/V are
   written into the drafter's cache at `p`. **The drafter never runs its
   layers over the context**; its cache holds target-derived K/V for every
   committed position, rewritten each round from the states the target
   just produced. Rejected rows write nothing. llama.cpp implements this as
   a second GRAPH MODE on the same model, keyed on batch type: an `embd`
   batch runs the encoder and injects K/V, a `token` batch drafts.
2. **One draft forward** over the block's query rows at positions
   `P0 + k + 1 ..`. Row 0 (the "bonus row") takes the REAL embedding of the
   bonus token; the remaining rows take the embedding of `mask_token_id`.
   The rows' own K/V are written by the forward itself at their positions.
   **ATTENTION INSIDE THE BLOCK IS NON-CAUSAL**: llama.cpp calls
   `llama_set_causal_attn(ctx_dft, false)` with the comment "DFlash needs
   non-causal attention", so every row of the block sees every other row of
   the block as well as the committed context (window 2048). That is the
   block-diffusion semantics: the masks are denoised jointly, not left to
   right. Each of the 5 layers runs: `input_layernorm` (plain, fused
   residual add), `attention_conv.prepare`, the attention block,
   `attention_conv.finish`, `post_attention_layernorm`, `mlp_conv.prepare`,
   the MLP, `mlp_conv.finish`. Then the final `norm` (plain). Only the MASK
   rows' hiddens feed the selector; the bonus row exists to put a
   real-token K/V row at `P0 + k + 1` until the target overwrites it next
   round.
3. **The selector** (greedy at T=0). For each mask row `l`:
   `unary = TRUNK_lm_head(hidden_l)`, keep `top_k = 16` ids and values;
   `hproj_l = hidden_projection @ hidden_l` (to rank 256). Edge scores over
   candidate pairs:
   `score[l, p, c] = unary[l, c] + dot(pred[pred_id_p, :] * hproj_l,
   succ[cand_id_c, :])`, where step 1's predecessor is the BONUS TOKEN and
   step l's predecessors are step l-1's 16 candidates. The greedy walk
   starts at `prev = argmax_c score[1, bonus, c]` and takes
   `proposal_l = cand_l[prev]; prev = argmax_c score[l, prev, c]`.
   The unary scores are produced by the TARGET's own lm_head applied to
   the DRAFTER's hidden, which is why vLLM guards on an unquantized head;
   this port's head is the trunk's INT4 one, the same head that verifies,
   so the guard does not apply. **The walk carries a SLOT INDEX, not a
   token id**: llama.cpp's driver keeps `predecessor` as an index into the
   previous row's top-k and reads `scores = row + top_k + predecessor *
   top_k` out of a packed `[ids | K x K scores]` lattice. That is the same
   function as carrying the id and re-gathering its codebook row, which is
   what this port does.

Proposals then go through the SAME verify/accept/rollback machinery the
MTP head uses (`run_raw_completion_speculative`); only the draft half of
the round differs.

## 4. The grouped conv, exactly

`out[i, c] = sum_t coeff[i, t, g(c)] * x[i - t, c]` for taps
`t < 2`, zero when `i < t`, where `g(c) = c / 16` is the channel's group
(320 groups of 16), `i` is the row's index within the block (0 = bonus
row), and the dynamic coefficients are
`coeff[i, t, g] = base[side, t, g*16 .. g*16+16] (broadcast over the
group's channels) + delta[i, side, t, g]`. `delta` comes from
`kernel_projection(x_in)` `[5120 -> 1280]`, reshaped
`[rows, 2 sides, 2 taps, 320 groups]`; side 0's delta is consumed by
`prepare` (which convolves the NORMED input that the sublayer will read)
and side 1's by `finish` (which convolves the sublayer's OUTPUT). The
projection input is the pre-sublayer normed stream in both cases. Static
base kernels are `[2, 2, 5120]` per conv.

llama.cpp's `build_dflash2_conv` implements exactly this and its shift
semantics agree with the formula: for `tap > 0` it prepends
`min(tap, block_size)` zero rows and takes the block's first
`block_size - tap` rows, which is the `i < t` zero clause written as a
concat.

## 5. Sources

- Checkpoint: `incoai/Qwen3.8-27B-DFlash2` (Apache-2.0), header read
  2026-08-19.
- vLLM PR 52816 (+ 52883): `vllm/model_executor/models/qwen3_dflash.py`,
  `qwen3_dflash2.py`, `v1/worker/gpu/spec_decode/dflash{,2}/speculator.py`,
  `eagle3_utils.py`, `qwen3_moe.py` aux capture.
- llama.cpp PR 27342 "spec : add DFlash2 support (local convolution +
  candidate selector)", +676 -83 across 20 files, read 2026-08-19:
  `src/models/dflash.cpp` (+232, the graph and `build_post_sampling`),
  `common/speculative.cpp` (+132 -75, the driver and the selector walk),
  `common/sampling.cpp` (+100), plus GGUF plumbing
  (`conversion/qwen.py`, `gguf-py/`, `src/llama-arch.*`, `llama-hparams.h`,
  `llama-model.h`).
- Blog: `inco.ai/blog/dflash2/` (the method, the two additions, the
  published tables).
- The engine-side decision context: `docs/MTP_SPECULATIVE.md` (the
  composite this drafter must beat, and the rollback term that gates long
  blocks), `docs/SPECULATIVE_DECODING.md` (the MoE-family verdict).
- Reference code was READ, not ported: those trees are MIT (llama.cpp) and
  Apache-2.0 (vLLM) and this workspace is MIT, and the same line
  `docs/MTP_SPECULATIVE.md` drew for MTPLX applies. Nothing below is a
  translation of either; what is taken is FACTS about the model.

## 6. Where this port is, 2026-08-19

**BUILT, UNCOMMITTED, AND BLOCKED.** Everything below sits in the working
tree; `ROADMAP.md` section 3 carries the same state as a work item.

Built:

- *Ingest*: `crates/repack/src/gemma4_checkpoint/dflash.rs`, wired into
  BOTH writers (`write_gemma4_install_streamed` and
  `orchestrate_gemma4_checkpoint_sharded`) rather than repeating the MTP
  head's one-writer bug. Rank-2 non-codebook tensors quantize to INT4
  affine; norms, the rank-3 `base_kernel` and the two selector codebooks
  narrow to BF16 raw, because nothing multiplies a codebook on the GPU (the
  selector GATHERS its rows on the host). Names arrive `dflash.`-prefixed:
  the published repository spells them bare and the CALLER renames the
  shard header before the walk sees them.
- *Kernels*: `crates/gpu/src/shaders/dflash_conv.metal` --
  `dflash_grouped_conv_fp16` (one kernel for all four call sites, `side`
  and `delta_col_base` as arguments) and `copy_strided_rows_fp16` (the aux
  capture). Parity-tested in `crates/gpu/tests/dflash_conv_parity.rs`.
- *Runtime*: `crates/runtime/src/families/qwen/dflash.rs` (state, shape
  derivation, policy, blocker) and `dflash_draft.rs` (context-KV write,
  block forward, host selector). Aux capture hooks in BOTH trunk paths:
  per-token in `families/qwen/mod.rs`, M-row in `batched.rs`.
- *Loop*: `SpeculativeProducer::drafts_block_passes` / `draft_block` plus
  the branch in `speculative.rs`, so a block drafter never sees
  `draft_step`.
- *Surface*: `--speculative-drafter mtp|dflash`, `auto` resolving to each
  drafter's own default block.
- *Artifact*: `~/models/qwen38-27b-dflash2.gturbo` (mlx trunk from
  `mlx-community/Qwen3.8-27B-4bit`, drafter from
  `incoai/Qwen3.8-27B-DFlash2` at a pinned revision), written by
  `crates/repack/tests/dflash2_checkpoint_network.rs`.

**MEASURED ON THE REAL INSTALL, 2026-08-19.** The acceptance figures are
deterministic (greedy, fixed prompt, fixed weights) and reproduced exactly
across two builds; the seconds are NOT baselines and the power source was
not recorded (AGENTS.md Gotcha 22).

| instrument | result |
| --- | --- |
| `dflash2_accept_length_probe`, block 8 | **0 of 256** first-proposal accepts, per-position acceptance 0.00 x8, 256 rollbacks of 256 rounds, **0.16x** against a 22.33 tok/s reference |
| what the drafter proposes | `[0, 0, 0, 0, 0, 0, 0, 0]` every round, from round 1 (base 32, anchor 71093) |
| `dflash2_bisect_probe`, 48 teacher-forced steps | true token at **median rank 0, top-1 48/48** on all three alignments, and the walk's first proposal equals the row-1 argmax **48/48** |

The probe's own functional-drafter guard (AGENTS.md Gotcha 57) is what
catches this; it fires rather than publishing a tidy `loses` table.

**WHAT THE PAIR ELIMINATES.** Under the bisect's driving the walk proposes
the true token 48 times out of 48, so the backbone, the aux capture, `fc`,
the norms, the RoPE, the conv and the SELECTOR are all right. The fault is
in how the LOOP drives the drafter. It is also not the M-row capture hook
in `batched.rs` on its own: round 1 already proposes zeros, and round 1's
capture comes from the prefill's per-token `produce`. Eight identical zero
proposals for 256 consecutive rounds is the signature of a CONSTANT logits
row (`top_k` then returns candidates 0..15 with equal unary scores and
`score > best_score` keeps the first), which is a far narrower claim than
"the drafter is weak".

## 7. Against the two references

### Can this be "added for Metal"?

**In llama.cpp it already is, and it cost ZERO backend kernels.** PR 27342
changes 20 files and NOT ONE of them is under `ggml/src/ggml-metal/`,
`ggml-cuda/`, `ggml-vulkan/` or `ggml.c`. Both new modules are composed
from existing ggml ops:

- the conv, from `ggml_cont_2d` / `ggml_reshape_{2,3,4}d` /
  `ggml_view_{1,2,3}d` / `ggml_fill` / `ggml_concat` / `ggml_repeat` /
  `ggml_mul` / `ggml_add`;
- the selector, from `ggml_top_k` / `ggml_get_rows` / `ggml_mul_mat` /
  `ggml_concat` / `ggml_pad` / `ggml_cast`, emitted in-graph by
  `build_post_sampling` and read back as a packed `[ids | K x K scores]`
  lattice.

So every llama.cpp backend, Metal included, gets DFlash2 for free, and the
published Apple numbers are that path: Qwen3.8-27B Q4_K_M on an Apple M5
Pro 64 GB, first 8 GSM8K problems, temperature 1.0 / top-p 0.95 / top-k 20,
xhigh reasoning, 2,048 max new tokens, block 8 -- autoregressive 10.42
decode TPS against 18.89 for DFlash2 (1.81x, acceptance 5.03), with BF16
and Q8_0 drafters at 1.85x/4.92 and 1.77x/5.08. A 10.42 TPS baseline for a
27B Q4_K_M is a GPU figure on that machine, so read the table as Metal even
though the PR body does not name the backend.

**This port cannot take that route and does not need to.** There is no
graph compiler here: every dispatch is written by hand, so "compose it from
existing ops" is not an option and the conv is a bespoke Metal kernel
(`dflash_conv.metal`) instead. That is one kernel, parity-tested, and it is
the RIGHT trade for this engine -- llama.cpp's version materializes a
`repeat` of the per-group coefficient up to full hidden width per tap, plus
a `concat` per tap, which a fused kernel avoids. The comparison is
therefore not "should we do what llama.cpp did" but "what do their two
implementations agree the SEMANTICS are", and on that the answer is worth
three corrections below.

### Divergences the comparison found

| axis | both references | this port | status |
| --- | --- | --- | --- |
| attention inside the draft block | NON-causal (`llama_set_causal_attn(ctx_dft, false)`, "DFlash needs non-causal attention"); the masks are denoised jointly | strictly causal: `dflash_forward_block` runs `encode_attention_decode` per row at `seq_len = position + 1`, so row `r` cannot see rows `> r` | **REAL GAP.** Bounds acceptance past position 1 by construction: rows 2..N are conditioned on mask embeddings alone where the reference conditions them on the whole block. Does not explain the zero proposals (the bisect shows row 1, the one row causality cannot affect, is correct). |
| block width | `block_size = 8` bounds ROWS; llama.cpp emits proposals for rows `1 .. min(rows, block_size)`, so at most 7; vLLM quotes "7 draft tokens" | `DFLASH_BLOCK = 8` means 8 PROPOSALS, so the forward runs 9 rows | **CHECK.** Row 8 is one past what the config bounds and is off-distribution for the trained drafter. Cheap to test: open at `Fixed(7)`. |
| selector location | in-graph on the GPU (llama.cpp `build_post_sampling`), or a Triton program per request (vLLM) | on the HOST, top-16 linear scan plus 17 codebook row gathers per step | **BY DESIGN.** Both references serve many concurrent requests; this engine serves one, and the selector is microseconds beside a pass that reads a gigabyte. Keep it. |
| selector walk state | a SLOT INDEX into the previous row's top-k, against a precomputed `K x K` score block | the previous step's TOKEN ID, re-gathering its codebook row | **EQUIVALENT.** Same function, different memoization. |
| context KV write | a second graph MODE on the same model, keyed on batch type (`embd` batch injects, `token` batch drafts) | `dflash_context_write` and `dflash_forward_block`, two functions | **EQUIVALENT.** |
| unquantized lm_head | vLLM GUARDS on it; PR 52883 relaxes the guard | the trunk's INT4 head, which is also the head that verifies | **NOT APPLICABLE**, and stated in section 3. |

Two smaller things worth knowing, neither a gap here: llama.cpp reads
`embedding_scale`, `logit_scale` and `final_logit_softcapping` from GGUF
and applies them if present, which for this checkpoint are 1.0 / 1.0 / none
(section 1); and its loader refuses a drafter whose
`n_embd < top_k * (top_k + 1)`, a constraint of packing the lattice into an
`n_embd`-wide row that this port's host selector does not have.

## 8. How to work on this next

In cost order. The first three are minutes each and need no download.

1. **Isolate the driving.** The bisect probe's loop-alignment arm calls
   `produce(anchor, pos)` BEFORE drafting; the real loop does not (the
   trunk absorbs the anchor in `verify`, afterwards). Add a third arm to
   `crates/bench/tests/dflash2_bisect_probe.rs` that drafts WITHOUT that
   preceding produce, and print row-1 rank plus `proposals[0]`. If it
   proposes 0, the cause is the capture or context state at draft time; if
   it still proposes the true token, the cause is
   `run_raw_completion_speculative`'s own bookkeeping (base, anchor, rewind
   order) rather than anything under `dflash*.rs`.
2. **Confirm the constant-row signature.** Call `dflash_probe_logits(1, ..)`
   inside a failing accept-probe round and check whether the row is
   constant. That separates "the block forward wrote nothing" from "the
   selector picked wrong", and the two lead opposite ways.
3. **Cover the untested path.** Nothing exercises the M-row capture hook in
   `families/qwen/batched.rs` (the bisect only ever uses per-token
   `produce`). It cannot explain round 1, but it is the one dispatch in
   this feature with no test at all, and rounds 2 onward read what it
   writes.
4. **Close the causality gap** (section 7). The block's rows must attend
   each other, not just their prefix. In this engine that means the draft
   forward stops calling the per-row decode-attention kernel and runs a
   block-wide attention over `[context window] + [all block rows]`, or a
   per-row call whose span is the whole block rather than `position + 1`.
   Expect this to be what moves acceptance from "position 1 only" to the
   references' 5.0.
5. **Try the trained width.** Open at `DflashDraftPolicy::Fixed(7)` and see
   whether the ninth row was costing anything.
6. **Then the gates**, before this can be called landed: the three
   real-model gates for `qwen3_5` (greedy plus sampled smoke, memory
   oracle) and `qwen38_quality_gate`, none of which this work has run; a
   frozen accept-length and speedup row in `docs/BENCHMARKS.md` with the
   measured numbers written back into section 6 here; a `scripts/power.sh`
   capture on a quiet machine (Gotcha 43); server wiring, which DFlash2
   inherits from the MTP follow-up list; and a decision on the two review
   findings deliberately left unfixed (the per-prompt-token
   `commit_and_wait` in `dflash_prime_from_capture`, and the unconditional
   batched-prefill scratch in `RealGemmaState`).

**A NOTE ON WHAT A COMPARISON CAN AND CANNOT SETTLE.** Sections 1-4 are
facts about the model and both references agree on them, so a disagreement
between this port and both of them is this port's bug. Section 7's
divergences were found by READING, which is the cheap instrument
(AGENTS.md's "READ THE PRIOR ART YOUR OWN DOCS NAME" -- the MTP head's norm
convention sat one grep away and was rediscovered by two hours of bisection
instead). But reading cannot say which divergence causes the measured zero,
and the bisect already rules out the two most inviting suspects. Run
experiment 1 before fixing anything in section 7.
