# The DFlash2 drafter: architecture off real bytes, and where this port is

What `incoai/Qwen3.8-27B-DFlash2` is, read off the published checkpoint
header (8,928 B JSON, ranged reads, no download) and off the two reference
implementations (vLLM `qwen3_dflash.py` + `qwen3_dflash2.py` + their
speculators, and llama.cpp's `draft-dflash`). Sections 1-5 hold FACTS about
the drafter. Section 6 holds THIS PORT'S STATE, which as of 2026-08-21 is
SHIPPED AND COMPLETE -- opt-in by default, wired through all four front ends,
and measured on three workloads across four blocks on a quiet machine. (It
read "built and blocked" for two days after it was neither; the blocker was
an FP16 overflow, fixed in `220635b`.) Section 7 holds the comparison against
the two references, including the divergences the comparison found -- all six
resolved. Section 8 is the work list and every item on it is closed. Nothing
projected appears in sections 1-5.

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

**BUILT AND WORKING.** Everything below sits in the working tree;
`ROADMAP.md` section 3 carries the same state as a work item.

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
  drafter's own default block. **`auto` DETECTS this drafter and does not
  ENABLE it** (section 8): it resolves to `mtp` even on a DFlash2-only
  install and reports a note naming the flag, so DFlash2 needs
  `--speculative-drafter dflash`. `turbospark-bench`'s own `--speculative`
  is the exception and drives whichever drafter the index carries, because
  it is the harness that has to be able to turn it on.
- *Artifact*: `~/models/qwen38-27b-dflash2.gturbo` (mlx trunk from
  `mlx-community/Qwen3.8-27B-4bit`, drafter from
  `incoai/Qwen3.8-27B-DFlash2` at a pinned revision), written by
  `crates/repack/tests/dflash2_checkpoint_network.rs`.

**MEASURED ON THE REAL INSTALL, 2026-08-19.** The acceptance figures are
deterministic (greedy, fixed prompt, fixed weights); the seconds are NOT
baselines -- the machine was running an interactive session throughout
(AGENTS.md Gotcha 43) and the power source was not recorded (Gotcha 22).
Every arm is asserted BYTE-IDENTICAL to the same generation with
speculation off, so the drafter is lossless in practice and not only by
construction.

`dflash2_accept_length_probe`, 600 greedy tokens per arm, 16 slots. **THREE
WORKLOADS SINCE 2026-08-21**, each swept across every block; the accepted and
rollback columns are deterministic and reproduced to the last digit across two
runs on different machine load.

| workload | block | accepted/round | committed/round | rollbacks | rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| code | 8 | 4.56 | 5.56 | 73 of 109 | 67% |
| code | 7 | 4.37 | 5.37 | 66 of 112 | 59% |
| code | 4 | 2.98 | 3.98 | 58 of 151 | 38% |
| code | 2 | 1.74 | 2.74 | 38 of 219 | 17% |
| math | 8 | 5.67 | 6.67 | 41 of 91 | 45% |
| math | 7 | 5.59 | 6.59 | 34 of 91 | 37% |
| math | 4 | 3.49 | 4.49 | 27 of 134 | 20% |
| math | 2 | 1.89 | 2.89 | 16 of 208 | 8% |
| prose | 8 | 1.84 | 2.84 | 204 of 213 | 96% |
| prose | 7 | 1.85 | 2.85 | 198 of 211 | 94% |
| prose | 4 | 1.60 | 2.60 | 193 of 231 | 84% |
| prose | 2 | 1.23 | 2.23 | 140 of 270 | 52% |

Per-position acceptance: **prose 0.53-0.81, code 0.84-0.94, math 0.86-0.98.**

**THE THIRD WORKLOAD WAS CHOSEN AS THE HARD CASE AND MEASURED AS THE EASY
ONE.** Multi-step arithmetic was added on the reasoning that a drafter
predicts a NUMBER far worse than the next word of an explanation, and that a
block ordering set on easy text would come apart there. It is the most
templated thing this checkpoint writes -- restated totals, `120 x $3.25 =
$390.00`, section headings -- and it accepts HIGHER than code. Prose was and
remains the hard case. Recorded rather than quietly swapped for something
harder: a workload that confirms is evidence, and the prediction being wrong
is the more useful half.

**WHAT THE THIRD WORKLOAD SETTLED, and it is not what it was added to
settle.** The two-workload version of this table could be read as "a large
block is fine where acceptance is high and catastrophic where it is not".
`math` refutes that: it accepts higher than `code` at every position and
block 8 still loses to block 2 on it. Acceptance decides whether speculation
pays AT ALL -- prose loses at every block but 2 -- while the BLOCK is decided
by compounding and rollback cost almost independently of it. A block of B
needs all B proposals to land, so 0.93 per position is a 45% rollback rate at
8.

**THESE ARE AT 600 GENERATED TOKENS AND AN EARLIER DRAFT OF THIS PAGE
PUBLISHED THE 256-TOKEN VERSION, which flattered every row.** At 256 the
same sweep read 7.09 accepted per round at block 8 and 1.43x, against 4.56
and 0.83x here. The first ~250 tokens of this prompt's answer are the CODE
BLOCK, which is far more predictable than the complexity discussion that
follows, so a short generation measures the easy part and reports it as the
whole. ACCEPT LENGTH AND SPEEDUP ARE BOTH FUNCTIONS OF GENERATION LENGTH,
not just of the prompt, and a probe is only as honest as its `GENERATE`.

Wall-clock on that run read 1.47x / 1.42x / 1.42x / 1.54x, but the arms sit
inside each other's noise on a busy machine and the ORDERING should not be
read off them. The acceptance column is the deterministic half and the one
to quote.

**5.56 committed per round at block 8 now sits BESIDE the published numbers**
(vLLM's 5.34 at 7 draft tokens, llama.cpp's 4.92-5.08) rather than above
them, which is the more believable place for it to be. The 256-token version
of this table read 8.09 and beat both, and that gap was the measurement
rather than the port.

### Energy, and the losslessness caveat that outranks it

`scripts/power.sh`, AC, `COOLING=max` (fans pinned), `ARMS=nospec,spec` so
both arms run `--shaping greedy` and differ ONLY in `--speculative`, 2 pairs
interleaved, `short-explanation`. Reproducibility 0.2% on tok/s and 0.13% on
decode seconds:

| arm | tok/s | J/token | watts | cpu_W | gpu_W |
| --- | ---: | ---: | ---: | ---: | ---: |
| nospec | 22.31 | 1.7248 | 39.36 | 5.78 | 33.58 |
| spec (block 2) | 19.71 | **2.0251** | 40.73 | 6.16 | 34.57 |

**0.88x throughput and +17.4% energy per token on this prose case.** Watts
barely move (+3.5%), so the energy penalty is almost entirely the TIME
penalty: the 5-layer draft forward runs every round whether or not its
proposals survive, and on a workload where they mostly do not it is pure
overhead. A pinned-fan row is an upper-headroom operating point (Gotcha 28),
so publish it beside a `COOLING=auto` row rather than instead of one.

**THE ARMS DID NOT GENERATE THE SAME TEXT, and that is the finding here.**
nospec committed 565 tokens and spec 533, deterministically, both stopping on
endOfTurn. Reproduced through the CLI on the same prompt, and now by the probe's own
prose arm at 154 tokens: the two greedy streams agree and then diverge --

```
off: ...a gentle slope rather than a sharp drop-off, wetlands allow water to spread
on : ...a gentle slope between the ocean and inland areas, wetlands allow water to
```

-- into two equally fluent continuations. Neither is degraded; both are the
model's own greedy output.

**SO "LOSSLESS" HOLDS FOR THE ACCEPTANCE LOGIC AND NOT FOR THE STREAM.** A
proposal is kept only when it equals the target's argmax, exactly. But the
committed token comes from a BATCHED verify row, and a batched row need not
equal a one-row pass in the last bit -- the reduce order differs, which is
AGENTS.md Gotcha 27's axis arriving on a new one. At a near-tie the argmax
falls the other way and the streams part for good.

**WHY EVERY EXISTING GATE MISSED IT, AND WHAT NOW CATCHES IT.**
`dflash2_accept_length_probe` asserted byte-identity at `GENERATE = 256`, and
the ad-hoc CLI check that corroborated it ran 200 tokens and stopped one
token short. But LENGTH WAS ONLY HALF OF IT: the probe's own prompt is a
code-shaped answer whose greedy stream tracks the sequential one for all 600
tokens, so on that prompt the check can never fail however long it runs. It
took the PROTOCOL's prose case to reach a near-tie, at 154 tokens.

The probe now varies both. It generates 600, it runs the protocol's
`short-explanation` as a second arm on the serving block, and it REPORTS the
common-prefix length rather than asserting an identity that does not hold.
Two assertions replaced the old one:

- a COMMON-PREFIX FLOOR of 64 tokens, which catches the failure that matters
  (corrupted drafter or rollback state, which diverges in the first handful
  of tokens) without failing on a near-tie that legitimately lands early.
  The floor is deliberately well under the 154 measured, because the
  divergence point is data-dependent and has no principled lower bound;
- EVERY BLOCK SIZE MUST GENERATE IDENTICAL TEXT, which is exact, has no
  length below which it stops looking, and is the assertion that ruled the
  batch WIDTH out as the cause. (It does not identify a kernel pair, which
  this line claimed for a day: what differs is `produce_batched` against
  `produce` as functions, and in nats that difference is a shape floor.)

The honest claim is "identical for the first hundred-odd tokens, then
divergent at the first near-tie". A caller who needs token-for-token
reproducibility against a sequential decode should leave speculation off.

**THE MECHANISM IS THE KERNEL, NOT THE BATCH WIDTH**, and the discriminator
that says so also refuted the first hypothesis. If the batch WIDTH caused
the divergence, different M would reassociate differently and the divergence
point would move. Measured at blocks 2, 4 and 8 on the same prompt:

| block | output chars | first divergence from sequential |
| ---: | ---: | ---: |
| 2 | 2825 | char 825 |
| 4 | 2825 | char 825 |
| 8 | 2825 | char 825 |

All three are byte-identical TO EACH OTHER and part from the sequential
stream at exactly the same character. So M is not the variable. What IS the
variable is that every speculative arm computes its tokens through
`produce_batched` and the sequential arm computes its through `produce`; the
streams part at the first near-tie whatever the block, because every
speculative token comes from one function and every sequential token from the
other.

That also explains why the block-size sweep above changes throughput and
NOTHING about the text: all three speculative arms are computing the same
thing, just in differently-sized batches of it.

**WHICH DIFFERENCE BETWEEN THOSE TWO FUNCTIONS IS RESPONSIBLE IS OPEN, and
the answer this section gave until 2026-08-21 was wrong.** It said the cause
was that a verify runs `dequant_int4_gemm_simd` where a decode step runs
`dequant_int4_gemv_simd` -- "two kernels, two accumulation orders". That was
read off the two shaders rather than measured, and the measurement refutes it:

- The pair agrees BIT-FOR-BIT at B = 1, 2, 8 and 16 on a fixture built to
  round, with ragged BF16 companions and full-mantissa FP16 activations
  spanning ~1e-3 to ~1e2 with alternating signs
  (`the_gemm_and_the_gemv_agree_on_data_that_can_see_reassociation`).
- That green is load-bearing only because the same fixture carries a POSITIVE
  CONTROL: `dequant_int4_gemm_mma`, which hands its reduction to
  `simdgroup_multiply_accumulate` and reduces in an order Apple does not
  document, differs from the GEMV on ~39% of the same outputs. A fixture that
  can see a reassociation sees none between the GEMM and the GEMV.
- Reading was never going to settle it in either direction. Metal's fast-math
  reassociates freely, so rewriting one kernel's sum in the opposite order
  compiles to identical code and that mutation survives every case in the
  parity file. SOURCE order does not determine COMPILED order here.
- The `gdn.metal` multi-row kernels were the next suspect and are exact too:
  0 of 896 outputs differ bitwise in `gdn_parity.rs`'s prefill-vs-decode case,
  so its 5e-2 state tolerance is conservative rather than descriptive.

**AND IT IS NOT A DEFECT AT ALL. Measured in NATS the same day, it is this
port's own SHAPE FLOOR.** `produce_batched` and `produce` differ by 6.2e-8 to
1.5e-5 nats with the argmax agreeing on every row, against a dense
batched-vs-cached shape floor of **7.4e-6** measured on MLX for this same
architecture, and against **1.57e-5**, this repo's own cross-engine result for
the family, published as "no detectable kernel gap" (`crates/bench/CLAUDE.md`
Gotcha 8). Every engine's batched and cached passes disagree by about this
much -- that disagreement is precisely what `scripts/kld.py` measures, by
running the REFERENCE twice.

**THE COUNT AND THE MAX DELTA ARE THE WRONG UNITS, and reading them as the
magnitude is the whole reason this looked like a bug.** "88% of the vocabulary
differs, worst 2e-2" sounds enormous and describes a distributional difference
of 1e-5 nats. The tell was available from the first measurement and went
unused: the argmax never moved. A greedy stream tracking for ~154 tokens and
then parting at the first near-tie is exactly what a floor this size predicts.

So there is nothing to make bit-identical that every engine does not also
have, and the advice to a caller needing a token-for-token reproducible stream
is unchanged: leave speculation off. What the eliminations bound is the
floor's CAUSE -- present at M=1 so not the batch width, absent at one and two
keys and present from three (and a two-key softmax exposes the score in full,
so q, k and V are exact there), and not the split-KV combine, which does not
run until span 32. That leaves the attention reduction, the only thing a key
count reaches. Note the block-size table above does NOT narrow this, though it
reads as though it does: identical text across blocks is consistent with ANY
difference between the two functions.

### Through the REAL generation loop, and the block that ships

The table above is `dflash2_accept_length_probe`, which hand-rolls its own
round. `run_raw_completion_speculative` is what a user actually runs, and it
was measured separately (`MFERENCE_SPEC_STATS=1`, 200 greedy tokens, against
a ~22.1 tok/s non-speculative arm on the same install). **The loop agrees
with the probe on the probe's own prompt, so the loop is not the variable --
the WORKLOAD is.**

| prompt | block | acceptance | rollbacks | tok/s | vs off |
| --- | ---: | --- | ---: | ---: | ---: |
| code | 2 | 0.93 0.98 | 6 / 70 (9%) | 32.476 | **1.47x** |
| code | 8 | 0.96 avg | 7 / 26 (27%) | 29.708 | 1.34x |
| prose | 2 | 0.80 0.66 | 41 / 86 (48%) | 21.413 | 0.97x |
| prose | 4 | 0.83 0.67 | 48 / 66 (73%) | 16.080 | 0.73x |
| prose | 8 | 0.78 0.64 | 59 / 60 (98%) | 10.612 | **0.48x** |

`code` is the probe's own prompt ("Write a Python function that merges two
sorted lists..."); `prose` is the wetlands question the repo's standing smoke
uses. Both greedy, both byte-identical to the non-speculative stream.

**THROUGHPUT TRACKS THE ROLLBACK RATE AND NOTHING ELSE**, and the mechanism
is the term `docs/MTP_SPECULATIVE.md` already names: a rejected batched round
on this recurrent family cannot stop early, so it restores a whole
gated-DeltaNet snapshot and replays the accepted prefix. The odds of paying
that rise 9% -> 27% -> 98% across those rows, and the tok/s column falls with
them. Acceptance itself barely moves with the block; what moves is how often
a round has to be undone.

**SO THE TRAINED BLOCK IS THE WRONG SERVING BLOCK.** `DFLASH_SERVING_BLOCK`
is 2, not the trained 8: block 2 wins on both workloads (1.47x against
1.34x) and its downside is BOUNDED where block 8's is not (0.97x against
0.48x). That is the same 2 the MTP head already defaults to, arrived at
independently on a different drafter and a different architecture -- which
is what makes it a property of this ENGINE rather than of either drafter.
`--speculative <n>` still names any block up to the trained 8 for a caller
who has measured their own workload.

**WHAT THIS RETIRES.** The probe's single prompt is not a workload model.
Section 6's own "a second workload is owed" was the right caveat and the
second workload changed the answer: on prose, speculation at the trained
block is a 2x LOSS. Do not quote an accept length without the prompt it came
from.

**THE ROW COUNT: THE NINTH ROW IS NOT THE PROBLEM AND NOT THE POINT.**
`block_size: 8` bounds ROWS, so the trained shape is 7 proposals plus the
bonus row (section 7), and the ninth row this port runs is off-distribution
twice over -- the drafter never saw it, and the reference's conv masks tap 1
there where this port applies it. It costs nothing measurable: block 8
commits 5.56 per round against block 7's 5.37, and its last position is
accepted 0.88 of the time. `DFLASH_BLOCK` stays 8 as the widest buildable
block. But BOTH lose to block 2 on throughput, which is the decision that
actually matters and is why `DFLASH_SERVING_BLOCK` is 2.

### What was wrong, and why every instrument said it was fine

The drafter proposed token id 0 at every position of every round (0 of 256
accepts, 0.16x). The cause was NOT the loop, the context write, the aux
capture, `fc`, the RoPE, the conv or the selector -- all of which the first
day of work cleared, correctly.

**THE DRAFT PASS OVERFLOWED FP16.** This drafter's residual stream peaks at
113,920 on the real install, against FP16's largest finite value of 65,504.
Measured by capping the drafter at N layers and scanning its buffers: the
embedding is +/-0.08, layer 1 takes the residual to 51,808, layer 2 to
65,152 -- where the first `inf` appears -- and by layer 3 every one of the
46,080 elements of `x` is NaN. Both references are BF16 throughout, whose
range is 3e38, and neither ever approaches a ceiling.

The fix is `DFLASH_RESIDUAL_SCALE`: the embedding rows and every `finish`
conv output are divided by 8, so the stream is held in a scaled
representation, and the norms that read it take a correspondingly divided
eps. It is a change of STORAGE and not of arithmetic -- `x` is read only by
RMS norms and by the residual add whose addend is scaled at the conv that
produces it, and a power of two shifts the exponent while leaving the
mantissa alone.

**THE EPS IS THE HALF THAT IS EASY TO GET WRONG, and this session got it
wrong first.** RMS norm is scale-invariant only where `eps` is negligible:
`(x/S) / sqrt(mean(x^2)/S^2 + eps)` equals `x / sqrt(mean(x^2) + eps*S^2)`,
so an unscaled eps acts as if it were `S^2` larger. On the EMBEDDING row,
whose mean square is ~4e-4, that is not a rounding difference. With S = 256
and the eps left alone the pass was finite and WRONG -- the true next token
sat at rank 13,202. With the eps scaled it sits at rank 0.
`crates/gpu/tests/rms_norm_parity.rs` pins the identity, and pins that its
own fixture can tell the two spellings apart.

**WHY NOTHING CAUGHT IT, which is the part worth carrying.** NaN is not a
loud value in this code path; it is a value every instrument scored as
PERFECT.

- `top_k` admits on `v <= val[k-1]`, and every comparison against NaN is
  false, so a NaN row is admitted at all 16 candidates; then `score >
  best_score` is false at each, the walk keeps `cand[0]`, and `cand` was
  initialized to zeros. Hence eight identical proposals of token id 0, with
  no error anywhere.
- The bisect probe's `rank_of` counts `v.to_f32() > target`, which NaN also
  fails, so an all-NaN row ranks the true token FIRST. Its reported "median
  rank 0, top-1 48/48" was not evidence that the backbone was right; it was
  the instrument reading its best possible value on garbage. That reading is
  withdrawn.

Both are the shape AGENTS.md Gotcha 30 records for `pearson` on a constant
input and Gotcha 57 for a degenerate accept-length table, with one turn of
the screw: those instruments return a NEUTRAL value on degenerate input,
where these return the BEST one, and nobody investigates a perfect score.
`dflash_select` now REFUSES a non-finite row by name.

**A SECOND BUG SAT BEHIND THE FIRST and could not be reached until it was
fixed.** `dflash_rewind_to` refused a target ahead of its cursor, and the
loop rewinds to `base + accepted`. While acceptance was zero that was
exactly the cursor and the arm never ran; the first round that accepted
anything failed with "rewind target 40 is ahead of the cache cursor 32".
The refusal was wrong for this drafter: unlike the step-wise MTP head, its
cursor is advanced only by the context write at the START of the next round,
so between the two it legitimately lags the accepted end. It is a no-op
forward now.

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
| attention inside the draft block | NON-causal (`llama_set_causal_attn(ctx_dft, false)`, "DFlash needs non-causal attention"); the masks are denoised jointly | NON-causal since 2026-08-19: every row runs at `seq_len = base + rows`, and the per-row k-norm and RoPE were hoisted above the attention loop so row 0 reads an already-rotated row 8 | **CLOSED.** Worth +0.24 accepted per round at block 8 (6.85 -> 7.09) and a wash at block 2, which is what the mechanism predicts: causality only ever bound rows 2 and up. This engine has no mask to flip -- causality here IS the span a row is given. |
| block width | `block_size = 8` bounds ROWS; llama.cpp emits proposals for rows `1 .. min(rows, block_size)`, so at most 7; vLLM quotes "7 draft tokens" and sizes its conv at `1 + num_speculative_tokens` | `DFLASH_BLOCK = 8` means 8 PROPOSALS, so the forward runs 9 rows | **MEASURED, KEPT.** Row 8 is off-distribution and the reference's conv would mask its tap 1 (that mask is `position % block_size >= tap`, which wraps at row 8). It is accepted 0.96 of the time anyway, and block 8 commits 8.09 per round against block 7's 7.22. Section 6. |
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

Items 1-5 of this list are DONE (section 6). What remains:

1. ~~**A quiet-machine row for the BLOCK ORDERING specifically.**~~ ANSWERED
   2026-08-21, first by a CONTROL and then by an actually quiet machine, which
   agreed. Four runs:

   | run | block order | ref tok/s | code (2/4/7/8) | math | prose |
   | --- | --- | ---: | --- | --- | --- |
   | 1 | 7,8,4,2 | 20.16 | 1.43 / 1.14 / 0.97 / 0.90 | 1.50 / 1.31 / 1.17 / 1.10 | 0.96 at 2 |
   | 2 | 7,8,4,2 | 15.95 | 1.86 / 1.20 / 0.98 / 0.93 | 2.07 / 1.47 / 1.28 / 1.21 | 1.26 / 0.68 / 0.51 / 0.47 |
   | 3 | **2,4,8,7** | 22.47 | 1.32 / 1.05 / 0.89 / 0.81 | 1.46 / 1.28 / 1.16 / 1.06 | 0.90 / 0.60 / 0.46 / 0.42 |
   | 4 | 7,8,4,2 | 21.93 | 1.33 / 1.06 / 0.90 / 0.82 | 1.47 / 1.28 / 1.18 / 1.07 | 0.90 / 0.60 / 0.46 / 0.42 |

   **The ordering is monotone decreasing in the block on every workload of
   every run: 2, then 4, then 7, then 8.**

   **RUN 4 IS THE QUIET-MACHINE ROW THIS ITEM ORIGINALLY ASKED FOR, and it
   sharpens the position finding rather than repeating it.** It sweeps in the
   ORIGINAL order -- block 2 last, the position that inflated runs 1 and 2 --
   on an idle machine, and it lands on run 3's REVERSED-order numbers: 1.33
   against 1.32 on code, 1.47 against 1.46 on math, and prose identical to two
   decimals at 0.90 / 0.60 / 0.46 / 0.42. Its own internal check is that the
   three no-drafter reference arms read 27.36 / 27.40 / 27.48 s on
   equal-cost work, a 0.4% spread.
   So the "position effect" measured above is a property of a machine whose
   load DRIFTS during a capture, not of sweeping in that order: when nothing
   else is running, first and last position agree to ~1%. Runs 1 and 2 are
   what a contaminated capture looks like, and run 2's 1.86x on code is the
   number to distrust rather than run 3's 1.32x. That also means the reversal
   remains the right control to REPEAT, since a future capture cannot know in
   advance whether it was quiet.

   THE CONFOUND WAS REAL AND IS NOW MEASURED RATHER THAN ARGUED AWAY. `BLOCKS`
   is swept in a fixed order, so block 2 always ran LAST in runs 1 and 2 --
   and run 2 got quieter as it went, which aliases a monotone load trend
   exactly with the block ordering. Run 3 reverses the sweep so block 2 runs
   FIRST. Two things fall out:

   - **The position effect is real**: block 2 on code reads 1.86x from last
     position in run 2 and 1.32x from first position in run 3. Anyone quoting
     an absolute ratio off this probe is quoting the position as much as the
     block.
   - **The ordering is not the position**: it survives the reversal intact on
     all three workloads. Had the ordering been an artifact of when each arm
     ran, reversing the sweep would have reversed or scrambled it.

   Two further legs, independent of any clock: the rollback RATES are
   deterministic and predict this ordering with no timing input (52-96% at
   blocks 4-8 on prose against 17-52% at 2), and prose block 8 reads 0.42-0.47x
   across runs, which is not a number a load story produces from a 1.0x
   baseline. The accepted-per-round and rollback columns reproduced to the
   last digit in all THREE runs.

   So the ratios are still not baselines and the ordering now is. The cheap
   control to repeat if this is ever re-opened is the reversal, not a quieter
   machine: interleaving the blocks properly is expensive here, because the
   DFlash2 block is fixed at OPEN and an interleaved arm costs a ~20 s model
   open every time it changes.

2. ~~**`scripts/power.sh`.**~~ DONE, and its table is in section 6 above:
   0.88x throughput at +17.4% J/token on prose, watts moving only 3.5%, so
   the energy penalty is the TIME penalty. It is what priced the opt-in
   default. (This entry said "the owner runs it" for a while after the owner
   had run it -- a hedge that outlives its own resolution is worse than a
   wrong number, because nobody re-checks a sentence that admits
   uncertainty.)
3. ~~**A second workload.**~~ DONE 2026-08-21: there are THREE, all swept,
   spanning per-position acceptance 0.53 to 0.98 (section 6). The serving
   default survives on all three and is now evidence-backed rather than
   two-sample. Note the third one refuted the prediction that motivated it --
   arithmetic is the EASIEST of the three, not the hardest -- and that
   sharpened the mechanism rather than the verdict: block choice turns on
   compounding and rollback cost, not on acceptance.
4. ~~**Server wiring**, which DFlash2 inherits from the MTP follow-up list.~~
   DONE 2026-08-21. `turbospark-server` takes `--speculative` and
   `--speculative-drafter` with the CLI's grammar and the CLI's meanings, and
   the policy behind both now lives in `runtime::speculation_policy` rather
   than in a CLI-private module the server could not reach. ONE thing differs
   and it is inherent: "is this run deterministic" is fixed per PROCESS on
   the CLI and per REQUEST on a server, so the install half is resolved at
   open and a sampled request falls back to the sequential loop silently.
   Most clients send a non-zero temperature, so a server started with
   `--speculative` speculates on a minority of its traffic.
5. ~~**One review finding left unfixed**: the unconditional batched-prefill
   scratch in `RealGemmaState`.~~ DONE 2026-08-21: it is a
   `BatchedPrefillScratch` allocated on first use by the chunk driver. Worth
   354 KiB on the real Gemma 4 install, so this is consistency with a rule
   rather than a memory win -- `real_mtp` and `real_dflash` are `Option`s for
   the same reason, and `families/qwen/batched_scratch.rs` states it: an
   unasked-for feature must allocate nothing so a frozen oracle row keeps
   describing the engine that shipped. Its sibling, the per-prompt-token
   `commit_and_wait` in `dflash_prime_from_capture`, was MEASURED AND CLOSED
   -- see below.
6. ~~**The M-row capture hook** in `families/qwen/batched.rs` has no test.~~
   DONE 2026-08-20. `the_batched_capture_writes_what_the_per_token_hook_writes`
   (`crates/runtime/tests/real_forward_qwen35_dflash.rs`) drafts off a
   drafter cache filled two ways -- all per-token primes, against a tail
   written by the M-row hook -- and requires the draft logits to agree. It
   is a TOLERANCE with a discriminating check beside it rather than an
   equality, because on the real install the batched and sequential kernels
   do not agree to the bit (on this fixture they happen to, at exactly
   0.000000, which is why the discriminating arm is what gives the bound
   teeth). Three mutations -- wrong aux column, wrong row stride, wrong
   capture base -- redden it and NOTHING else in the file, which is the
   point: the losslessness case and the frozen digest stay green under all
   three, exactly as the hazard predicts.
7. ~~**The probe cannot see a sampler regression.**~~ DONE 2026-08-20.
   `dflash2_accept_length_probe` argmaxed directly where the shipped loop
   calls `selection::select` for the draft chain, the acceptance comparison
   and the bonus row; it now routes all four through `select` under the same
   greedy `ShapingConfig` `real_model.rs` builds, passing the real history
   and step counter. Every deterministic column reproduced exactly
   (accepted/round 4.37 / 4.56 / 2.98 / 1.74 and rollbacks 66 / 73 / 58 / 38
   at blocks 7 / 8 / 4 / 2, and the prose divergence still at 154 tokens),
   which is what says the change was a coverage fix rather than a numerics
   one -- at temperature 0 `select` takes its argmax fast path.

### The priming cost on a long prompt, measured 2026-08-20

The review finding was that `dflash_prime_from_capture` calls
`dflash_context_write`, which ends in a blocking `commit_and_wait`, ONCE PER
PROMPT TOKEN. Every prompt measured when that was written was 22-62 tokens,
so the worry was a ~3,000-token prompt paying ~3,000 blocking GPU passes
before decode starts. **It does not happen.** `long-synthesis` (2,940 prompt
tokens) on the real install, both arms `--shaping greedy` so speculation is
the only variable:

| arm | prefill | decode | peak |
| --- | ---: | ---: | ---: |
| sequential | 159.59 s | 36.77 s | 660.4 MiB |
| speculative, block 2 | 153.11 s | 38.92 s | 873.2 MiB |

Prefill is **0.96x** -- inside this machine's run-to-run spread, and nowhere
near the feared 2x. So the per-token `commit_and_wait` does not dominate,
and BATCHING THE PRIMING WRITES IS NOT WORTH DOING. Read the ratio, not the
sign: the arms differ in two ways, not one (the speculative prefill also
pays a full-vocab head per prompt token, where the sequential path's
`produce_prefill` skips it), so a 4% difference cannot be attributed to
either term, and the honest statement is that the two roughly cancel.

Bounding arithmetic, since the null result invites suspicion that the
priming never ran: it did (the header reads `speculative=on drafter=dflash2
block=2`, and the +212.8 MiB of peak IS the DFlash2 state), so 2,939
blocking passes cost at most the observed difference plus whatever the head
cost, i.e. well under 10 ms each and plausibly 1-2.

The peak column is the other reason this measurement was worth taking: it
is what priced the opt-in default below at 213 MiB as well as at 0.88x.

**THE DEFAULT IS OPT-IN SINCE 2026-08-20, AND DETECTION IS NOT ENABLEMENT.**
`SpeculativeDrafter::Auto` reads the install's resident index and, on an
install whose only drafter is this one, resolves to `Mtp` anyway --
allocating no DFlash2 state -- while reporting a note that names
`--speculative-drafter dflash`. Enabling it by default would have been
0.88x throughput and +17.4% J/token on prose for anyone running greedy on
such an install, which is not a default anyone would choose if asked. The
MTP head keeps its `auto` because its measurement is the opposite way up.
`crates/cli`'s `resolve_drafter` owns the split and
`crates/cli/CLAUDE.md` Gotcha 10 states the contract.

**A NOTE ON WHAT A COMPARISON CAN AND CANNOT SETTLE.** Sections 1-4 are
facts about the model and both references agree on them, so a disagreement
between this port and both of them is this port's bug. Section 7's
divergences were found by READING, which is the cheap instrument
(AGENTS.md's "READ THE PRIOR ART YOUR OWN DOCS NAME" -- the MTP head's norm
convention sat one grep away and was rediscovered by two hours of bisection
instead). But reading cannot say which divergence CAUSES a measured
failure, and in the event neither of them did: the zero proposals were an
FP16 overflow, which no comparison against a BF16 reference could have
surfaced, because it is a property of this port's storage rather than of
the model. Read the references to learn what the semantics ARE; measure to
learn which of your deviations is biting.
