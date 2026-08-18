# The multi-token-prediction head: architecture and measured findings

What `Qwen/Qwen3.8-27B`'s `mtp.*` tensors are, how this port runs them, and
every number that was actually measured. All figures are from this machine
(Apple M4 Max, 36 GB, macOS 26.5.2) against `~/models/qwen38-27b-mtp.gturbo`
unless stated.

**This page holds facts. `docs/MTP_SPECULATIVE.md` holds the DECISION** --
the cost model, `c(M)`, the composed break-even table, and the record of the
two times that page reached the wrong conclusion. Nothing projected or
borrowed appears here; where a number came from a model rather than from a
run, this page does not carry it.

## The head, off real bytes

`crates/repack/tests/mtp_head_network.rs` reads the published safetensors
header, ~114 KB of ranged reads, no download.

| | |
| --- | --- |
| tensors | 15, all in `model-00018-of-00018.safetensors` |
| size at source | 849,398,784 B, BF16 throughout |
| blocks | 1 (`mtp_num_hidden_layers`), FULL attention, not gated-DeltaNet |
| embedding table | none, shared with the trunk |
| output head | none, shared with the trunk |
| size once INT4 group 64 | 238,930,944 B, i.e. 1.5% of the trunk's 15.13 GB |

Its block is a trunk full-attention layer's shape field for field, which is
what lets a draft step reuse `families/qwen/attn.rs` and `dense.rs` with no
new Metal kernel and no new dispatch shape. Verified against the INSTALLED
bytes rather than only the published header
(`mtp_head_probe.rs::the_heads_tensors_match_a_trunk_full_attention_layers`);
`hidden` 5120, 24 q heads over 4 kv at `head_dim` 256, `intermediate` 17408:

| tensor | bytes | dtype | trunk layer 3 |
| --- | ---: | ---: | --- |
| `self_attn.q_proj` | 31,457,280 | INT4 | identical |
| `self_attn.k_proj` / `v_proj` | 2,621,440 | INT4 | identical |
| `self_attn.o_proj` | 15,728,640 | INT4 | identical |
| `self_attn.q_norm` / `k_norm` | 512 | BF16 | identical |
| `mlp.gate_proj` / `up_proj` | 44,564,480 | INT4 | identical |
| `mlp.down_proj` | 44,564,480 | INT4 | identical |
| `input_layernorm`, `post_attention_layernorm` | 10,240 | BF16 | identical |

Four tensors have no trunk counterpart: `mtp.fc` (26,214,400 B INT4,
`[5120, 10240]`), `mtp.pre_fc_norm_embedding`, `mtp.pre_fc_norm_hidden` and
`mtp.norm` (10,240 B BF16 each).

`q_proj` is 2x the query width because this family packs `[query; gate]` into
it, so the head inherits the trunk's attention output gate.

## The recurrence

```text
h'  = fc([ pre_fc_norm_embedding(embed(t_i+1)), pre_fc_norm_hidden(h_i) ])
h'  = h' + attn(input_layernorm(h'))          // the head's OWN KV
h'  = h' + ffn(post_attention_layernorm(h'))
out = TRUNK_lm_head(mtp.norm(h'))
```

**Read the indices carefully.** The pair is the trunk's hidden state at `i`
beside the token occupying `i + 1`, and it predicts `i + 2`. The block sits at
position `i`. So a call takes the token the trunk just produced and drafts the
one after it, and the head's rows land contiguously from 0 with no unwritten
row, because `h_0` exists.

**`h_i` is the trunk's POST-final-norm state, not its residual stream.** It is
what the trunk's own `lm_head` consumes. The distinction is not a scale that
`pre_fc_norm_hidden` would absorb, because `model.norm` carries a learned
per-channel weight.

Every line above was confirmed by READING the reference implementation rather
than inferred: concat order (embedding first), the pairing, positions from 0,
`full_attention_interval=1` at `layer_idx=0`, and the target's `lm_head`.

## THE NORM CONVENTION, which is the finding that made it work

**The head's five whole-vector norms are CENTERED**: the checkpoint stores an
offset from unity and the effective scale is `1 + w`. **The trunk's are
plain.** One model, two conventions, which is AGENTS.md Gotcha 50 arriving on
a second family after `muse_glimmer`.

Established two ways that are decisive, and one that is only suggestive:

| evidence | reading |
| --- | --- |
| mlx-community's conversion minus this install, elementwise, all seven norms | **+1.00000**, std 0.002 |
| the ORIGINAL published shard | stores the centered form too |
| install's mtp norms, mean abs weight | 0.082, 0.166, 0.206, 0.461, **1.252** |
| install's TRUNK norms, same measure | 0.893 to 1.249 |

The first row is BF16 rounding and nothing else, and it is the proof. The
second assigns blame: the repack is correct and copies verbatim, and it is
mlx-vlm's converter that bakes the `+1` into its published weights.

**The magnitude tell in rows three and four is worth knowing AND worth not
trusting alone.** It is striking for four of the five norms, which sit near
zero where the trunk's sit near one -- and it is actively misleading for
`mtp.norm`, whose 1.252 lands inside the trunk's own 0.893 to 1.249 band. A
reader who checked only magnitudes would have cleared the one norm that is
just as centered as its four siblings. The elementwise difference is what
settles every one of them at once.

**Read plainly, this put the true next-next token at median rank 248,308 of
248,320** -- last, not random. The fix dispatches
`gpu::encode_rms_norm_bf16w_centered`, which already existed for
`muse_glimmer`. The `1 +` is applied on the FP32 accumulator and never baked
into the stored weight: these weights sit near zero and BF16's resolution near
1.0 is 2^-8, so baking would destroy a large fraction of a 0.08 offset.

| | plain | centered |
| --- | ---: | ---: |
| rank of true `t[i+2]`, of 248,320 | median 248,308 | median **0** |
| top-1 agreement with the trunk | 0/24 | **23/24** |
| pearson against the trunk's distribution | -0.28 | **+0.60** |

## How this port wires it

`crates/runtime/src/families/qwen/mtp.rs`, behind `MFERENCE_MTP_DRAFT=<depth>`.
Unset, the flow allocates nothing and encodes nothing.

**Its KV is its own one-layer cache**, built from a cloned `ArchConfig` at
`num_layers: 1` and `full_attention_layer_mask: vec![1]`. Widening the trunk's
is not available: that one's sizing is what every family's frozen memory
oracle asserts against. About 4 KiB per token here, 16 MiB at a 4,096 window.

**Three invariants, each of which failed silently before it was enforced:**

1. **The cache must be PRIMED over the prompt.**
   `encode_full_attention_block` derives its attention span from the
   `position` ARGUMENT (`position + 1`), never from the cursor, so a draft at
   decode position P off an empty head attends over P rows nobody wrote. No
   error, finite logits, plausible tokens. `mtp_prime_step` is the draft step
   without the full-vocab head, run once per prompt token.
2. **A step must be taken AT the cursor.** Past it reads unwritten rows;
   behind it silently re-drafts history. `mtp_draft_step` refuses both.
3. **The head must be REWOUND after a rejected draft**, and to a different
   target than the trunk: the trunk rolls back to where the block STARTED and
   replays, while the head goes to where the accepted prefix ENDED and
   continues, because it cannot recompute those rows (the trunk's replay has
   overwritten the `scratch.x` each one needs). Only the caller knows both,
   which is why `rollback` does not reach into the head.

**Ordering.** A draft step reads `h_i` straight out of `scratch.x`, so it must
run between the trunk's readback and the next `produce`. Outside that window
it drafts off the wrong hidden state. Reusing the trunk's scratch is safe for
exactly one reason: `scratch.x` is re-initialised from the embedding at the
top of every token.

**A round takes `depth + 1` steps for `depth` proposals.** The extra step is
taken for its KV row alone, because a round where every proposal is accepted
needs a row the proposal-producing steps do not write, which is the case a
good drafter hits most often.

## Measured results

**Weights.** The installed head against the published BF16 shard,
dequantized exactly as `dequant_int4_gemv_simd` reads it:

| tensor | pearson |
| --- | ---: |
| `mtp.fc` | 0.99239 |
| `self_attn.q_proj` | 0.99589 |
| `mlp.gate_proj` | 0.99600 |
| `mlp.down_proj` | 0.99553 |

Its 15 tensors are also distinct from each other, none zero, and none a byte
copy of the trunk layer they are shaped like.

**Independent structural agreement.** The head adds 238,930,944 bytes to the
install (866 resident entries against the headless install's 851, resident
region 15,371,732,992 against 15,132,802,048). That total is byte-for-byte
what mlx-community's separately-produced INT4 conversion reports, which is a
real agreement rather than a coincidence: same 15 tensors, same affine scheme,
same group size.

**Accept length**, real install, greedy, 256 generated tokens per arm, verified
lossless against a non-speculative reference stream on every block:

| block | accepted/round | committed/round | break-even | speedup |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 1.35 | 2.35 | 1.71 | **1.37x** |
| 4 | 1.94 | 2.94 | 2.86 | **1.03x** |
| 8 | 2.05 | 3.05 | 4.53 | 0.67x |
| 15 | 2.05 | 3.05 | 7.96 | 0.38x |

Per-position acceptance, block 8:

```text
position  0     1     2     3     4     5     6     7
accept    0.82  0.76  0.49  0.54  0.50  0.43  0.33  0.00
```

**Single-step acceptance is 0.82 and the CHAIN saturates at ~2.05.** Blocks 8
and 15 read identically, which is the same statement twice: past roughly the
sixth proposal this head contributes nothing. Committed-per-round is the
accepted prefix plus the bonus token every verify yields for free.

**Footprint.** 659.5 MiB against the headless install's 659.4, so the head's
228 MiB of resident weights are not counted. That is AGENTS.md Gotcha 40
holding a fourth time.

**The trunk is unaffected.** With drafting off, `qwen38_quality_gate`
reproduces reference-answer perplexity 4.9432 and both frozen digests exactly.

## What is still deviating

**`q_norm` and `k_norm` are centered too and are still read PLAINLY.**
`encode_rms_norm_bf16w_perhead` has no centered sibling, and the shared
attention block resolves those weights by name, so overriding them needs a
parameter on a function the trunk also calls. At 23/24 top-1 the cost is
measurably small, but the accept lengths above are a FLOOR until it lands.

Nothing else in the head is known to deviate.

## Instruments

| target | what it answers | cost |
| --- | --- | --- |
| `repack/tests/mtp_head_network.rs` | what the head is made of | header only |
| `repack/tests/mtp_quantize_network.rs` | does the quantizer survive real bytes | a few MB |
| `repack/tests/mtp_install_fidelity_network.rs` | do the INSTALLED bytes match the source | a few MB |
| `bench/tests/mtp_head_probe.rs` | what the head predicts, and its tensor shapes | ~9 s |
| `bench/tests/mtp_accept_length_probe.rs` | accept length against break-even | ~4 min |
| `scripts/mtp_bisect.py` | WHICH STAGE diverges, against the published drafter | ~20 s |

```sh
TURBOSPARK_MTP_INSTALL_DIR=~/models/qwen38-27b-mtp.gturbo \
  cargo test -p turbospark-bench --test mtp_accept_length_probe --release -- --ignored --nocapture

TURBOSPARK_MTP_INSTALL_DIR=~/models/qwen38-27b-mtp.gturbo \
  cargo test -p turbospark-bench --test mtp_head_probe --release -- --ignored --nocapture

hf download mlx-community/Qwen3.8-27B-MTP-4bit --local-dir ~/models/qwen38-mtp-ref
MFERENCE_MTP_DUMP=/tmp/mtp-dump TURBOSPARK_MTP_INSTALL_DIR=~/models/qwen38-27b-mtp.gturbo \
  cargo test -p turbospark-bench --test mtp_head_probe --release -- --ignored --nocapture dumps_one_draft
uv run --python 3.12 --with numpy scripts/mtp_bisect.py /tmp/mtp-dump ~/models/qwen38-mtp-ref
```

**The accept-length probe ASSERTS a functional drafter and fails rather than
reporting.** Its first run read 0 accepted of 7,168 proposals and would have
printed a tidy table saying "loses", which is a quotable verdict on a question
that was actually a bug. The bar separates drafting from not drafting, never
pays from loses.

**`scripts/mtp_bisect.py` carries its own safetensors reader and its own
dequantizer.** `safetensors.numpy` cannot decode BF16, and borrowing mlx's
loader would weaken the independence the comparison rests on (Gotcha 48). Its
dequantizer was validated against `mx.dequantize` at max absolute difference
9.8e-4 and pearson 0.9999987 BEFORE its verdict was believed.

**The dump is taken at position 0 against an empty head cache on purpose.** A
softmax over one key is exactly 1.0, so the block has no RoPE, q/k-norm or
history dependence and is a pure function of `concat`, which is what makes it
recomputable offline.

## Independent confirmation, and the read that should have come first

**MTPLX had already documented the norm convention, and this port found it the
expensive way.** `mtplx/mtp_patch.py` carries `_heal_raw_delta_mtp_norms`,
whose docstring reads: "Raw Qwen3.5 exports store MTP RMSNorm weights
zero-centered (delta convention); mlx-lm's trunk sanitize restores +1.0 but
the MTP tensors are loaded separately and must be restored here", and which
says such a sidecar "poisons every draft". Its `MTPContract` dataclass states
three more of this page's facts as plain fields: `hidden_variant: "post_norm"`,
`concat_order: "embedding_hidden"`, and `mtp_quant_group_size: 64` at
`mode: "affine"`. Its registry names the symptom `mtp_acceptance_collapsed`.

Its empirical detection thresholds bracket what was measured here
independently, which is what makes this confirmation rather than coincidence:

| | MTPLX threshold | measured here |
| --- | --- | --- |
| raw q/k norm mean | "near 0.75" | 0.780, 0.797 |
| healthy q/k norm mean | ">= 1.74" | 1.779, 1.791 |
| raw low-set norms | "below 0.5" | 0.082, 0.166, 0.206, 0.461 |

**It also covers what this port still does not.** Its `_RMSNORM_SUFFIXES` lists
all SEVEN norms including `q_norm`, `k_norm` and `norm.weight`, so the per-head
pair left open above is a real requirement and not optional polish. MTPLX
restores them by baking `+1.0` into the loaded weights, which is evidence that
baking into a small owned buffer is an acceptable route for those two
specifically -- their weights sit near 0.78, not near zero, so Gotcha 50's
precision objection is weak there.

**"TAKE NO MTPLX SOURCE" IS A LICENSING DECISION AND NOT AN INSTRUCTION NOT TO
READ IT.** `docs/MTP_SPECULATIVE.md` records that decision (Apache-2.0 NOTICE
obligations in a uniformly MIT repo) and it stands: no code from that project
is in this one, and none is needed, because what was required was a FACT ABOUT
THE CHECKPOINT rather than any expression of it. Conflating the two cost this
port roughly two hours of bisection to rediscover a convention that was one
grep away in the repository whose result the page's first sentence quotes.
Read the prior art; copy none of it.

## Two methodology notes worth keeping

**An A/B run on top of a dominant defect is uninformative, and it does not
announce itself.** While the norms were wrong, swapping `fc`'s concat order
end to end changed nothing, and shifting every RoPE position moved the
correlation from -0.2820 to -0.2818. Both read as "this is not the cause" and
both were really "the output is garbage either way". The reference settled
both questions by inspection in minutes, after two hypothesis-test cycles had
produced no signal. When an experiment's two arms are equally broken, its null
result is evidence about nothing.

**Reading the reference beat measuring it.** mlx-vlm's drafter
(`mlx_vlm/speculative/drafters/qwen3_5_mtp/qwen3_5_mtp.py`, and NOT mlx-lm,
which has no `qwen3_5_mtp` module) settled five design questions at zero cost
and confirmed the priming requirement independently. The same lesson is
recorded for this family's mrope question on 2026-08-14.

## Sources

- Head weights, standalone: `mlx-community/Qwen3.8-27B-MTP-4bit` (31 tensors, INT4 affine group 64)
- Reference implementation: `mlx-vlm` 0.6.14, `speculative/drafters/qwen3_5_mtp/` and `speculative/mtp.py`
- The decision, the cost model and its two reversals: `docs/MTP_SPECULATIVE.md`
- The MoE arm of the same question: `docs/SPECULATIVE_DECODING.md`
