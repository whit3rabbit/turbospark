# qwen4_exp (Qwen3.8-Flash-Next): bring-up lessons learned, Phase 0 onward

`docs/QWEN4_PHASE0.md` is fact-finding only: the checkpoint's config, tensor
layout, and two independent references cross-checked, with no weights
downloaded and no forward pass run. This page picks up where that one stops
and carries everything since: intake, the decode flow, the memory policy,
the router/shared-expert-gate dtype bug that blocked every real install
until 2026-09-04, and the first successful real-hardware decode this family
has ever produced. Read this page before touching `families/qwen4/`,
`crates/repack/src/gemma4_checkpoint/`'s safetensors write path, or
continuing this family's bring-up; read `docs/QWEN4_PHASE0.md` first if the
question is about the checkpoint's shape rather than the port's behavior.

## Timeline

| Date | Landed | Commit(s) |
|---|---|---|
| 2026-08-27/28 | Phase 0 fact-finding: config, tensor layout, two independent references cross-checked | `841ead8`, `b34ef12` |
| 2026-08-29..31 | Family/config surface, Phase 0 config gate, n-gram table classification and streaming writer, manifest wiring | `42158b1`, `48669bb`, `2c02155`, `ff74f69`, `90bf452` |
| 2026-09-01/02 | N-gram ingest + Phase 2 GDN/PLE/hyper-connection kernels; decode flow wired end to end into `RealForwardRunner` (Phase 3) | `2e458e9`, `8aee247`, `12f71a1`, `3485f1c`, `552cf63` |
| 2026-09-02/03 | Decode-capable synthetic fixture + first real-hardware attempt; Phase 4 memory policy (`ALLOWED_CACHE_SLOTS` widened to `[8,...,128]`); wired into the safetensors install path | `3d70f79`, `ab7b8a5`, `bbeb839` |
| 2026-09-04 (morning) | Real REAP-288 install OPENS for the first time (`moe_phase2_down_reduce_k8` widened to a runtime width sized to the caller's own `top_k`); decode itself still refused on the router's dtype | `25854ef` |
| 2026-09-04 (this session) | Router-dtype bug root-caused and fixed; a SECOND identical bug on the shared-expert gate found only by running the real checkpoint, and fixed; first successful real-hardware decode this family has ever produced | `612be53` (accidental partial landing), `27b666f` |

Everything through "decode itself still refused" was already accurately
recorded in `ROADMAP.md` and the handoff this session started from. This
page exists because the handoff's two open questions (root cause of the
dtype bug, and the GGUF-vs-safetensors decision) are now both closed, plus a
second bug the handoff had no way to know about yet.

## The router / shared-expert-gate dtype bug

### Symptom

`~/.turbospark/models/qwen4-reap288.gturbo` (`sh0wie/Qwen3.8-Flash-Next-REAP-288-MLX-4bit`,
68.1 GiB, 48 layers, 2560 hidden, 248320 vocab, 288 experts at top-10)
opened but refused to decode:

```
unsupported: language_model.model.layers.0.mlp.gate.weight: expected INT8
(dtype 5) 4x128, got dtype 1 with 1024 packed bytes
```

### Root cause

`crates/repack/src/gemma4_checkpoint/orchestrate.rs::read_resident_entries`
had exactly two branches for every resident tensor: already `U32`-packed on
disk (pass through verbatim, preserving whatever quantization the publisher
applied) or anything else (narrow straight to BF16, no quantization step at
all). Every MoE family that already worked here does so because its
upstream MLX conversion happens to pre-pack the router as `U32`. This
checkpoint's `mlp.gate.weight` (the MoE router) ships raw BF16 instead, so
it silently took the BF16 fallback and the runtime's INT8-only router GEMV
(`gpu::encode_router_gemv_gemma4`, no BF16 sibling exists anywhere in
`crates/gpu`) refused it at the first dispatch.

**The manifest's declared quantization was ALSO wrong, and independently.**
The real install's `manifest.json` declared `quant.router.weightBits: 4`,
which should have been impossible: `model_io::manifest::quant::validate_quant`
only accepts `weightBits: 8` for the router slot. The reason it opened
anyway is `validate_quant`'s `or_attention` escape hatch -- a slot that is
byte-for-byte identical to the `attention` slot is treated as "this
component isn't really declared" and its shape check is skipped entirely.
This checkpoint's `config.json` has no per-tensor quantization override for
the router, so `Gemma4Quant::bits_for` fell back to the model's
`default_bits` (4), which happened to equal `attention`'s bits (also 4) --
so the validator waved a genuinely wrong declaration through by
coincidence. This is why the install *opened* but failed at the runtime
dtype check rather than at manifest load.

### The fix

Two changes in `crates/repack/src/gemma4_checkpoint/`:

1. `orchestrate.rs::read_resident_entries` gained a third branch,
   `int8_force_targets(family)`, mirroring the GGUF walk's own
   `int8_transcode_targets` (`gguf_checkpoint/transcode.rs`) -- which
   already force-quantizes exactly the same two tensor names for
   `QwenGdnMoe`, for the identical reason. A tensor matching one of those
   suffixes under `Qwen4Exp` is force-quantized to INT8-affine
   (`narrow.rs::quantize_gating_matrix_int8`) regardless of what dtype it
   arrived in, instead of falling through to the BF16 narrowing.
2. `manifest_quant.rs::manifest_quant_for` forces the router slot's declared
   `weightBits` to 8 for `Qwen4Exp` unconditionally, rather than trusting
   `quant.bits_for(&router)` -- mirroring the GGUF manifest writer, whose
   router slot is also a stated fact about the bytes rather than a probe.

### A second bug, found only by running the real checkpoint

The first fix alone got decode past the router GEMV and one dispatch
further, then failed again:

```
unsupported: tensor language_model.model.layers.0.mlp.shared_expert_gate.weight:
dtype 1 has no dispatched GEMV kernel
```

`mlp.shared_expert_gate.weight` is a second small gating matrix (`[1,
hidden]`, the shared expert's sigmoid gate) that the real checkpoint
*also* ships raw, exactly like the router. This is not a coincidence: the
GGUF walk's `int8_transcode_targets` list for `QwenGdnMoe` already names
BOTH tensors together (`["mlp.gate.weight", "mlp.shared_expert_gate.weight"]`),
which is the precedent this port's own code had already established and
which the first-pass fix simply hadn't generalized to yet. `int8_force_targets`
now lists both suffixes; `quantize_gating_matrix_int8` (renamed from
`quantize_router_int8`, since it now serves two roles) is unchanged, since
its row-by-row quantize loop never assumed a router-specific row count.

No manifest change was needed for the second tensor: `shared_expert_gate`'s
bit width has no dedicated slot in `manifest.json`'s five fixed slots
(`embedding`/`attention`/`router`/`sharedExpert`/`routedExpert`), so nothing
there declares or validates it independently.

## Real-hardware verification

Both bugs were fixed, then the real 68.1 GiB checkpoint was re-streamed
twice (once per fix iteration) to confirm on real bytes rather than only on
a synthetic fixture:

```sh
turbospark-check --model qwen4-reap288 --messages-file /tmp/p.json \
  --max-new 60 --seed 1 --temperature 0.0001 --top-k 1 --max-context 2048
```

Greedy output (first ever produced by this family on real hardware):

> Coastal wetlands (such as mangroves, marshes, and tidal pools) do not
> technically "reduce flood damage" in the sense of preventing water from
> entering a property or stopping a storm from occurring. Instead, they
> **mitigate the economic and operational impact** of flooding by providing
>
> `[stop=MaxTokens prefill=22tok/4.19s new=60tok decode=5.51s tok/s=10.897]`

Sampled output (CLI defaults, `--seed 20260721`), also coherent:

> Coastal wetlands (such as mangroves, marshes, and tidal zones) act as
> natural buffers that significantly reduce the impact of flood damage
> through several interconnected mechanisms...
>
> `[stop=MaxTokens prefill=22tok/3.02s new=60tok decode=5.45s tok/s=11.014]`

**`--max-context` must not exceed 2048 on this family today.** This install
has no QSA indexer implemented, so context above `compressed_attention.index_budget`
(2048 on this checkpoint) is refused by name rather than silently degrading
(`docs/QWEN4_PHASE0.md` section 5). This is unrelated to the dtype bug and
was already known before this session.

**No memory oracle, quality gate, or throughput number exists for this
family yet.** The two runs above are a coherence smoke test on one real
install, not a frozen benchmark row. That is the natural next step, not
completed in this session.

## Lessons learned

**A fixture that pre-packs a tensor never exercises the code path that
transcodes it.** `synthetic_qwen/qwen4_decode.rs`'s original fixture built
both gating matrices via `int8_triple`, which fabricates them as
already-`U32`-packed -- so it always took the pre-existing
`pass_through_packed` branch and would have passed identically whether or
not either fix existed. A fixture has to match the checkpoint's exact
on-disk *shape* (which dtype it arrives as), not just its logical shape, to
be evidence about a write-time transcode. The fixture now builds a
`router_raw` variant that ships both gating matrices as raw BF16 matrices,
matching the real checkpoint, and that variant is what a mutation check
(reverting either half of the fix) actually reddens with the exact original
error strings.

**Real-hardware verification found a bug the fixture had not been extended
to catch.** The router-only fix passed every fixture test and the first
real re-stream got measurably further (past the router GEMV) before failing
on a different tensor. This is the concrete argument for running the real
install rather than stopping at a green fixture suite when a fix touches a
write-time transcode: the fixture is only as good as the shapes it was
built to cover, and this checkpoint had two instances of the same shape,
not one.

**A validator's "defaulted statement" escape hatch can mask a genuinely
wrong declaration, not just a legitimately absent component.**
`validate_quant`'s `or_attention` rule exists so a dense model's
router/shared-expert/routed-expert slots can validly mirror the attention
slot as "this component doesn't exist" (`crates/repack` Gotcha 8). This
checkpoint has a real router, and its slot happened to collide with
attention's width by coincidence (no override declared, same default bits)
rather than by design -- so the escape hatch fired for the wrong reason and
let a wrong `weightBits: 4` sail through. The same mechanism serves two
different truths depending on whether the model has the component at all,
and only reading the actual bytes (not the manifest) settled which one this
was.

**A concurrent session's broad `git add`/`git commit -a` can silently
absorb another session's uncommitted work into an unrelated commit.**
Commit `612be53` ("refactor(env): rename all MFERENCE_ environment
variables to TURBOSPARK_") landed the router-only half of this fix
alongside its own unrelated rename work, because this session's edits to
those files were sitting uncommitted in the same working tree when it ran.
This is AGENTS.md Gotcha 13's shape, encountered directly rather than only
read about. Both that commit and a later one (`b521830`) were already
pushed to `origin/main` by the time the entanglement was discovered, which
ruled out any rewrite: the fix was to leave published history alone and
land the remaining work (the shared-expert-gate generalization) as a new,
forward-only commit (`27b666f`) with a message that names the split
honestly, rather than attempt to un-mix a public commit.

## GGUF ingestion: still explicitly out of scope

Unchanged from the handoff's own conclusion, re-confirmed rather than
re-derived this session: `crates/repack/src/gguf_config/mod.rs` and
`gguf_checkpoint/transcode.rs` both refuse `Qwen4Exp` by name, on purpose.
GGUFs of this model exist (unsloth's and bartowski's), but nothing in this
port has mapped its n-gram shards, hyper-connection tensors, or QSA indexer
out of a GGUF file yet, so a mask derived today would be the one part of an
install that looked right while the rest went missing. This is a real,
separate multi-session bring-up, not a shortcut around anything in this
page, and should not be started opportunistically while extending the
decode flow.

## The memory oracle: this family's first frozen row (2026-09-04)

`crates/bench/tests/qwen4exp_memory_oracle.rs`, against the real
`qwen4-reap288` install. Covers `short-explanation` and `medium-review`
only -- `long-synthesis` tokenizes to 2,940 under this family's vocab, over
the 2,048-token window before a single generated token is added, so there is
no context at which it can run on this family today (`real_model_params`'s
own comment on `Qwen4Exp`). `oracle_common::run_oracle_over_cases` is the new
entry point that makes an oracle over a partial case list possible; every
other family's oracle still runs all three by construction.

One reading: `short-explanation` 62 prompt / 380 new tokens at 7.150 tok/s,
`medium-review` 426 prompt / 605 new at 7.530 tok/s, both `endOfTurn`, steady
state +0.00 MiB, peak 2,503 MiB. The peak is consistent with the arithmetic
AGENTS.md Gotcha 36 sets up for this family (288 experts, 48 layers, ~2.7648
MiB per expert blob): 16 slots is ~2,025.6 MiB of slot capacity alone,
leaving ~477 MiB for KV at 2,048 context plus the resident core and process
baseline. The tok/s is far below every other MoE family measured in this
repo (15-40 tok/s elsewhere); that is this checkpoint's shape, not a
regression to chase -- 288 experts at top-10 against a 16-slot cache means
most tokens miss and stream from disk, and nothing here has tuned
`ALLOWED_CACHE_SLOTS` for this routing profile yet (see "What's next" below).
Frozen in `crates/catalog/src/models.json` under alias `qwen4-reap288`.

## The quality gate's determinism bug: root-caused and fixed (2026-09-04)

`crates/bench/tests/qwen4exp_quality_gate.rs` used to fail at arm 3 -- the
determinism check every other family's gate treats as a formality -- after
passing its first two arms (reference-answer perplexity: 8.7224; the two
digests). Two BACK-TO-BACK warm greedy generations of the identical prompt,
on the same open runner, with `reset()` called before each (this crate's
`open_model_runner*` never enables prefix reuse, so `reset()` always runs),
produced DIFFERENT SHA-256 digests.

**Root cause: `RealQwen4State::reset` never zeroed `ple_conv_tail`.**
`families/qwen4/state.rs` rewinds the GDN chain's delta rule and conv tail
plus the PLE n-gram context, but the PLE sublayer's OWN dilated conv keeps a
separate recurrent tail (`ple_conv_tail`, `PLE_CONV_HISTORY` = 9 rows) that
`reset()` left untouched. The module's own doc comment had flagged this as
"a gap rather than a decision" and reasoned it was inert because "today
nothing [resets mid-process]" -- that reasoning was already wrong the day it
was written: `crates/bench`'s quality gate flow calls `producer.reset()`
between the perplexity pass and each digest generation, on the SAME open
runner, which is exactly a mid-process reset. A fresh process always starts
`ple_conv_tail` at the zeros `RealQwen4State::build` writes once at open, so
cross-process generation reproduced exactly throughout the investigation
below; a second within-process generation after `reset()` instead started
PLE's dilated conv from the first generation's leftover history, diverging
the wide residual stream from the very first PLE-layer token onward and
cascading into a completely different greedy digest by the time 256 tokens
had been sampled.

**Fix**: `reset()` now zeroes `ple_conv_tail` (to the buffer's own byte
length, via `gpu::write_buffer_bytes`) alongside `self.gdn.reset()` and the
n-gram context rebuild. Verified on THREE independent fresh processes: all
three agree on perplexity (8.7224) and on both digests to the last hex
character, and the gate's own within-process two-generations check (arm 3)
now passes on every run. `crates/bench/tests/qwen4exp_quality_gate.rs` now
carries a frozen `ChipQuality` row for the M4 Max.

While in this code, the SEPARATE `try_reuse_prefix` gap noted below (its
rewind guard checking `self.real_qwen.is_some()` with no matching
`self.real_qwen4.is_some()` arm) was also closed, even though it was never
the cause of this bug -- prefix reuse is disabled throughout this
investigation and in every caller today.

**What is confirmed, and what is ruled out**, from this session's
investigation:

- **Cross-process greedy generation reproduces exactly.** Three independent
  `turbospark-check` processes, same prompt, same seed, same settings,
  produced byte-identical output every time (checked to 40 generated
  tokens). So the bug is specific to repeated generation WITHIN one open
  runner, not to the forward pass itself.
- **The MoE dispatch order is NOT the cause.** `families/qwen4/moe.rs`'s
  `encode_moe_layer` was checked line-by-line against `families/qwen/moe.rs`
  (the working, frozen-gated sibling it was adapted from) for AGENTS.md
  Gotcha 27's hazard (slots dispatched by cache state rather than by router
  rank). The two are structurally identical: `ordered` is built from
  `(0..selected.len())` in both, i.e. router-rank order, never reordered by
  physical slot number.
- **Prefix-KV reuse is NOT engaged.** `RealForwardRunner::prefix_reuse_enabled`
  defaults `false` at open and nothing in `crates/bench` calls
  `set_prefix_reuse`, so `try_reuse_prefix` returns 0 unconditionally and
  `producer.reset()` runs before every generation in the gate's flow. (This
  also means the `real_qwen4` recurrent-state gap in `try_reuse_prefix` --
  its rewind guard checks `self.real_qwen.is_some()` but has no matching
  `self.real_qwen4.is_some()` arm, unlike every other family with recurrent
  state -- is a SEPARATE, real gap worth closing before this family's
  `--chat` REPL or any session-pooling caller ever enables prefix reuse for
  it, but it is not what this gate's failure is measuring.)
- **`NgramContext` is not the cause.** Plain `Vec<i64>` shift register, no
  hashing, no randomized iteration order.
- **`wide_x` (the wide residual stream) is not stale state.** It is fully
  re-written from the embedding table on every `produce()` call, at every
  one of the `hc_count` copies, unconditionally -- confirmed by reading
  `produce_real_qwen4_inner`'s first loop, not assumed.

`hc.rs` was checked and cleared: `encode_hyper_connection` reads `wide` and
writes only its own scratch buffers (`hc_normed`, `hc_low`, `hc_up`,
`hc_inject`) plus the caller-supplied `mixed_out`, none of which persist
across a `reset()`. `ple.rs` was where the state lived, but not in the
dataflow the function itself encodes -- `RealQwen4State::ple_conv_tail`, a
field `reset()` never touched (see the fix above). No GPU-side race was
needed to explain the symptom once that field was found; the encode order
in `produce.rs` was never the problem.

**Consequence for this handoff's own next item, now resolved**: a
throughput row can be written once someone wants one -- see "What's next"
below.

## MTP speculative decoding: an ingestible head does exist (2026-09-04)

The prior handoff flagged an open question ahead of any wiring work: does an
MTP-ingestible head artifact exist, published, for the REAP-288 checkpoint
specifically, given that public MLX conversions of Qwen3.8-Flash-Next
generally drop the `mtp.*` tensors? **It does.** Verified directly against
the Hugging Face API (`/api/models/...`) and the repo's raw `README.md`, not
via a summarized fetch:

`sh0wie/Qwen3.8-Flash-Next-MTP-Drafter-MLX-bf16` (same publisher as our
`qwen4-reap288` install, created 2026-08-27, last modified 2026-09-03) is a
standalone drafter: "the 31 `mtp.*` tensors of `Qwen/Qwen3.8-Flash-Next`,
taken from the official bf16 release and repacked in the standalone drafter
layout with no other change." `model.safetensors`, BF16, 2,607,150,848
parameters (~4.9 GB on disk, ~3 GB if quantized to 8-bit at load, matching
the repo's own claim that 8-bit quantization "matches the published 8-bit
head to the last bit"). `config.json` names its architecture
`qwen4_exp_mtp`. License is Qwen Community License 1.0, inherited from the
base model -- not a standard open-source license, worth knowing before
anyone scripts an automated pull of it.

**Compatibility is stated as architectural, not weight-dependent**, and the
repo's own install instructions use our exact checkpoint alias as the
worked example:

```
pmlx serve --model sh0wie/Qwen3.8-Flash-Next-REAP-288-MLX-4bit
```

("pmlx" is a third-party serving stack, `gethamster/pmlx` -- not this
project's own tooling, and not `slotstream`'s. It is the vehicle the
drafter's README happens to demonstrate against, not a dependency of the
artifact itself: the artifact is a plain safetensors file of named tensors,
attachable by anything that can read one.)

**This is a second, distinct MTP artifact from the one `slotstream` fetches**
(README.md's "Compared to slotstream" section: a 1.5 GB `mtp.safetensors`
bundled with the full 512-expert `pipenetwork` checkpoint). Both trace back
to the same 31 `mtp.*` tensors in Qwen's official bf16 release, but the
size difference (1.5 GB against this repo's 4.9 GB BF16 / ~3 GB at 8-bit) was
not reconciled here -- plausibly a different quantization width on
slotstream's side, but that is an inference, not a read fact, and should be
verified against slotstream's own artifact before being asserted anywhere.

**What the drafter's own README says about when it helps, unverified by
this port and worth re-checking before relying on it**: on the 288-expert
pack (this checkpoint's own expert count), it measured 61.9 -> 69.2 tok/s
warm-server and 51.5 -> 64.9 tok/s (+25.9%) via its own benchmark script,
both 2026-09-03, on an M4 Max 128 GB -- a different memory tier than this
machine's 36 GB, so the absolute tok/s numbers do not transfer even if the
multiplier does. That +25.9% land inside, but not matching, the user's own
"1.24-1.33x" framing for slotstream's number; the two are DIFFERENT measurements
(different artifact, different engine, different chip) and should not be
read as confirming each other. Three caveats from the same README, all
unverified here: the drafter is reported to *cost* speed rather than save it
on the full 512-expert pack (verifying against a split expert store costs
about twice what it costs against a pruned one); acceptance-per-round falls
as resident experts shrink (2.04 at 288 experts, 1.96 at 240, 1.53 at 200,
all from its own benchmark script); and it falls back to plain decoding past
~8,192 tokens of context on every pack it measured -- moot for this port
today since `qwen4_exp` is capped at 2,048 until the QSA indexer exists (see
below), but relevant the moment that cap is lifted.

**What this resolves and what it does not.** It resolves the standing
question of whether wiring MTP for this family would be blocked on someone
publishing a head -- it would not be. It does not change anything about
`families/qwen4/mod.rs`'s own scope statement, which is about this port's
code rather than about artifact availability: there is still no MTP prefix
constant and nothing under `mtp.` is read for this family (line ~154-156).

**CORRECTION to this section's own first draft: wiring is NOT a
`families/qwen/mtp.rs` reuse, and this doc already had the fact that says
so.** Item 8 of THIS document ("The MTP head") did the dataflow fact-finding
for these exact `mtp.*` tensors during Phase 0, from mlx-vlm's reference
(`mlx_vlm/speculative/drafters/qwen4_exp_mtp/`), and it says plainly: "the
head DOES carry hyper-connections... and is a QSA layer with a full
512-expert MoE, which makes it far from free -- this is not the
~1.5%-of-a-pass drafter `docs/MTP.md` records for `qwen3_5`." That sentence
was already sitting in this file when this section's first draft was
written and should have been read before writing "most likely following the
pattern in `families/qwen/mtp.rs`" -- it is not that pattern.
`families/qwen/mtp.rs` reuses the trunk's DENSE FFN and full-attention
encoders wholesale because `qwen3_5`'s dense flow has no MoE and no
hyper-connections to reuse; this family's trunk has both, and its MTP head
carries a COMPLETE decoder layer of its own: hyper-connections (both
sublayers, plus its own final `hyper_connection_mixer`), a QSA attention
layer (packed `q_proj`, per-head norms, an unused-below-budget indexer, same
shape as the trunk's mask-1 layers), and a full MoE with its OWN 512
experts, shared expert, and router -- separate from the trunk's pruned
288-expert table, not a subset or a reuse of it.

Independently confirmed against `sh0wie/Qwen3.8-Flash-Next-MTP-Drafter-MLX-bf16`'s
own safetensors header (fetched as a partial HTTP range read, ~8 MiB, not
the 4.9 GB body) rather than taken on the repo's word: all 31 tensors match
item 8's inventory name-for-name once the `mtp.` prefix is dropped
(`fc_embedding.weight [2560,2560]`, `fc_hidden.weight [2560,2560]`,
`hyper_connection_mixer.*`, `layers.0.attn_hyper_connection.*`,
`layers.0.mlp_hyper_connection.*`, `layers.0.self_attn.{q,k,v,o}_proj`
plus `q_norm`/`k_norm`, `layers.0.self_attn.indexer.*`,
`layers.0.mlp.{gate,switch_mlp.{gate,up,down}_proj,shared_expert*}`,
`pre_fc_norm_embedding`, `pre_fc_norm_hidden`), and its own `config.json`
independently states `num_experts: 512`, `indexer_budget: 2048` (same
budget as the trunk, so the same below-budget-QSA-is-dense-attention
argument the trunk's mod.rs makes applies to the head's attention too),
`mtp_num_hidden_layers: 1`, and `layer_types: ["full_attention"]` (no GDN,
no PLE in the head -- matching item 8's "does NOT carry PLE"). So the
standalone repo is a faithful, unmodified extraction of the official
checkpoint's own `mtp.*` tensors, not a re-derived or re-trained artifact --
this is worth having checked rather than assumed, since a repacked artifact
silently diverging from its stated source is exactly `docs/QWEN4_EXP.md`'s
own recurring failure class (Gotcha 62's rule, applied to a downloaded file
rather than a written claim).

**What this means for scoping the work, concretely.** Wiring this head is a
materially bigger lift than `qwen3_5`'s MTP support, in three ways none of
which `families/qwen/mtp.rs` needed to solve: (1) a repack path for a
SECOND artifact carrying its own MoE section -- `crates/repack` refuses
`Qwen4Exp` entirely today (GGUF ingestion section above), and even once the
trunk's own safetensors intake is the model, this artifact's shape (no
n-gram table, no PLE, a single QSA+MoE layer) needs its own manifest
handling, not a reuse of the trunk's; (2) a SECOND, independent expert
cache/streamer for the head's 512 experts, unrelated to the trunk's pruned
288-expert one -- this is not a shared resource the two dataflows can pool,
and doubling the resident/streaming machinery for one MoE layer is a real
memory and complexity cost the `qwen3_5` head never had to pay; (3) the
fusion is a BROADCAST ADDITION over the four hyper-connection streams
(`fused = proj_embed[.., None, :] + proj_hidden`, item 8's "The fusion, and
why `docs/MTP.md` is the wrong prior for it"), not the concatenation
`qwen3_5`'s single `fc` tensor performs -- a different failure mode from the
one Gotcha 16 warns about for that family, and `pre_fc_norm_hidden` is a
PLAIN full-width (10240) norm, a THIRD norm shape in this family's own
taxonomy (item 9) that appears nowhere else. Read `crates/runtime/CLAUDE.md`
Gotcha 16 for the GENERAL traps that recur on any MTP head (priming before
first draft, rewind on rollback, centered-vs-plain norm dispatch getting
mixed up costing 0/7168 accepted once already) -- but do not read it as a
template to copy; the shape being reused this time is `families/qwen4/`'s
own `hc.rs`/`attn.rs`/`moe.rs` functions parameterized onto a second tensor
prefix, not `families/qwen/`'s.

**Economic viability is still the OPEN question item 13.1 of this document
already flagged, and it has not been re-derived since.** "The head is a
full QSA + 512-expert MoE layer... a recurrent trunk makes a rejected
batched round expensive" (item 13). `sh0wie`'s own drafter README (this
section, above) reports it as a net win only against a PRUNED target pack
and a net LOSS against the full 512-expert one, with acceptance-per-round
falling as target residency shrinks -- none of which has been checked
against THIS engine's own batched-verify cost model (Gotcha 19 in
`crates/runtime/CLAUDE.md`: a rejected round on a recurrent trunk must
snapshot and replay, and the probability of paying that rises sharply with
block depth). Before wiring, re-derive whether a head this expensive clears
the bar on THIS engine's rollback cost, not just on `pmlx`'s.

The `qwen4-reap288` catalog row's notes previously said "no speculative
drafter exists for this checkpoint... no MTP-ingestible head in any
published MLX conversion" -- written before this repo was found, now
corrected in `crates/catalog/src/models.json`.

## QSA indexer: repack groundwork was already done, a CPU reference now exists (2026-09-04)

Before writing any code, checked what "repack groundwork" for the indexer
(item 1 of the prior handoff's "Outstanding work") would actually mean, since
`families/qwen4/mod.rs`'s own doc says "NO INDEXER CODE IN THIS PORT AT ALL".
**It turned out the repack half was already complete, from Phase 1 bring-up,
and verified against real bytes rather than assumed:**

- `crates/model-io/src/arch_baselines/qwen.rs`'s `qwen4_exp_125b_a6b()`
  already carries every `CompressedAttentionConfig` field this checkpoint's
  indexer needs (`index_n_heads: 4`, `index_kv_heads: 1`, `index_head_dim:
  128`, `index_top_k: 512`, `index_budget: 2048`, `csa_compress_rate: 4`),
  matching `config.json` field for field -- the module's own doc says so
  outright: "The indexer fields are RECORDED and not implemented."
- `crates/repack`'s `classify_for_family` already classifies
  `self_attn.indexer.{index_qk_proj,q_layernorm,k_layernorm}.weight` as
  resident for every one of the 12 QSA layers, with a comment stating the
  reason: "Carried rather than dropped: they are tiny, and an install
  missing them would need a re-stream the day QSA lands."
- **Verified this is not just a test fixture's claim**: `strings -a` against
  `~/.turbospark/models/qwen4-reap288.gturbo/model_weights.bin` finds all 36
  expected tensor names (12 layers x 3 tensors), confirming the indexer
  weights are ALREADY resident in the real install on this machine today,
  streamed there during Phase 1/3 bring-up with no further repack work
  needed. `docs/QWEN4_PHASE0.md` section 5's own inventory anticipated
  exactly this list.

So there is no repack groundwork left to do; the missing piece is entirely
on the decode/GPU side, where genuinely nothing exists: `Dsv4StateManager`
(`crates/gpu/src/dsv4_state.rs`) looked like a candidate to build on and is
not one -- it is DeepSeek-V4-Flash's own unwired memory-allocation
scaffolding (its CSA/HCA scheme, with LoRA ranks and a two-rate compress
split this checkpoint's QSA has none of), with no decode flow, no indexer
scoring, and no block-selection kernel of any kind. `docs/QWEN4_PHASE0.md`'s
own "`QsaCacheManager` decision stands" is a plan, not an implementation --
nothing under that name exists in this port.

**What this session added, following this port's own established
discipline of a CPU reference before any Metal kernel** (the precedent
`crates/compute/src/hyper_connection.rs` and `ple.rs` both set for this
same family): `crates/compute/src/qsa_indexer.rs`, covering the
BLOCK-POOLING, SCORING and SELECTION steps of section 5's pseudocode --
`pool_blocks_mean` (mean-pools consecutive raw key rows into one pooled row
per complete block, in FP32 as the spec requires, leaving the ragged tail
untouched), `score_blocks` (`relu(sum over heads of q . pooled) /
sqrt(head_dim)`, against the ONE shared pooled key every head reads since
`index_kv_heads == 1`), and `select_blocks` (top-`min(block_topk,
num_complete_blocks)` block selection plus the always-selected ragged
tail, returning a boolean mask). 9 tests, each mutation-checked individually
(reddens only its own case; `cargo test -p turbospark-compute --lib
qsa_indexer` is the target). The below-budget exactness argument
(`docs/QWEN4_PHASE0.md`'s own Q1 proof: `floor(visible/4) <= 512` selects
every block) is re-derived as a property test at the exact boundary (2,051
visible tokens) with DELIBERATELY ADVERSARIAL scores, so the test cannot
pass by the scores happening to favor the right cutoff.

**UPDATE, same day: the RoPE/norm question above is now RESOLVED, from
source rather than guessed.** Read `Blaizzy/mlx-vlm`'s actual reference
(`mlx_vlm/models/qwen4_exp/language.py` and `.../qwen3_5/language.py` at
`d68a25e71e84`, `mlx_vlm/models/rope_utils.py` at `3db7f1d3402f`) rather
than re-deriving from the pseudocode alone, the way `docs/QWEN4_PHASE0.md`'s
own PLE section had to for its hash multiplier pairing. Two facts settle it
completely:

- **The norm is [`rms_norm_centered`](../crates/compute/src/rms_norm.rs),
  applied per head.** `docs/QWEN4_PHASE0.md` item 9 already listed the
  indexer's `q_layernorm`/`k_layernorm` as CENTERED; this just connects
  that fact to the function.
- **The RoPE is `rope_neox_subdim` at `rotary_dim = 64`, `theta = 1e7` --
  the SAME two numbers the trunk's own QSA attention already dispatches --
  because the reference's `Qwen4ExpQSAIndexer` is constructed with the
  enclosing attention module's `rotary_emb` PASSED IN
  (`Qwen4ExpQSAIndexer(config, self.rotary_emb)`): it is the identical
  Python object, not merely the same convention by coincidence.** That
  object is built as `Qwen3_5RotaryEmbedding(int(head_dim *
  partial_rotary_factor), base=rope_theta, style="interleaved")`, and
  despite mlx-vlm's own confusing label for it (their "interleaved" is
  unrelated to this port's use of that word for mrope section handling),
  tracing `style="interleaved"` through `apply_multimodal_rotary_pos_emb`
  lands on `_apply_interleaved_rotary_pos_emb_axis1`, which is exactly
  `rope_neox_subdim`'s contract (pair `i` with `i + rotary_dim/2`, confined
  to the first `rotary_dim` elements) and NOT `rope_paired`'s (pair `2k`
  with `2k+1`). Independently confirmed on this port's own side: the
  trunk's already-verified quality gate (frozen perplexity 8.7224) is what
  it measured dispatching `rope_neox_subdim` for this exact checkpoint, so
  the shared object's convention is checked against real numbers here too,
  not only read off the reference.

`crates/compute/src/qsa_indexer.rs`'s module doc now states this as
resolved fact with full citations, and a new test
(`the_full_indexer_chain_composes`) proves the whole chain -- per-head
centered norm, RoPE (one call per block, since pooled keys do not share a
position), pool, score, select -- composes correctly using ONLY functions
this crate already ships, against a hand-checked (Python cross-computed)
example. **That test's first draft had a real, mutation-caught gap**: at
`rotary_dim = 2` there is only one rotation pair, so `rope_neox_subdim`'s
and `rope_paired`'s pairings coincide by construction and a
wrong-RoPE-function mutation survived silently -- AGENTS.md Gotcha 23's
self-relative-fixture shape, on a fixture rather than on a whole file.
Widening to `rotary_dim = 4` (two pairs) made the two conventions
genuinely diverge, and the same mutation now reddens exactly that one test.

No new kernel-shaped function was needed for norm or RoPE -- both already
exist and are simply called correctly now.

The `qwen4-reap288` catalog row's notes now also states this in miniature
so a reader does not have to re-derive it from the module doc.

## QSA indexer: the pooling and scoring Metal kernels (2026-09-04, later)

Two new port-local kernels, matched to `crates/compute/src/qsa_indexer.rs`'s
CPU reference: `qsa_pool_blocks_mean_fp16` (mean-pools consecutive raw key
rows into one pooled row per complete block, FP32 accumulation) and
`qsa_score_blocks_fp16` (`relu(sum over heads of q . pooled) / sqrt(D)`,
one threadgroup per block, the same two-stage SIMD-group reduction shape
`ple_gate_fp16` already uses). `crates/gpu/src/shaders/qsa_indexer.metal`
and `crates/gpu/src/qsa_indexer.rs`; parity tests in
`crates/gpu/tests/qsa_indexer_parity.rs`. Both kernels match the CPU
reference exactly (well under FP16 tolerance) and three targeted mutations
(dropping pooling's division, dropping scoring's relu clamp, and reading
the pooled key at a fixed block-0 offset regardless of which block is being
scored) each redden exactly the cases they should and nothing else.

**Block SELECTION stays host-side and gets no kernel**, matching this
port's own MoE router precedent (top-k is already a host round trip
there): `turbospark_compute::qsa_indexer::select_blocks` is the whole of
it, unchanged from the prior session.

**What this is not**: these two kernels do not touch attention itself.
Section 5's pseudocode ends with "the result is a boolean mask ANDed onto
the causal mask" -- applying that mask to real Q/K/V is a THIRD, and by far
the largest and riskiest, piece of this feature, structurally unlike
anything this port's existing `attention.metal`/`attention_decode.rs` do
today (dense causal or a contiguous sliding-window ring; a selected block
set is neither contiguous nor known until the indexer has run). Two shapes
were considered and neither is built: mirror mlx-vlm's own fused kernel
(`qsa_kernel.py`, read during this session's RoPE research -- one
threadgroup per query row, online softmax across SIMD-group-interleaved
selected blocks plus the ragged tail, sorted block indices, a GQA factor),
or gather the selected KV rows into a compact contiguous buffer first and
reuse this port's ALREADY-VERIFIED dense attention kernel over that shorter
sequence (lower-risk, reuses proven math, costs a new but much simpler
gather kernel and a host-side index-list construction from the boolean
mask). Deciding between them is real design work belonging to whoever picks
this up next, not a default to assume.

**The user split the remaining work here explicitly**: the sparse-attention
application (the gather-vs-fused-kernel design question above) went to a
separate worktree; this session continued on the indexer's OWN state and
computation, which the rest of this section covers. The two are decoupled
by design -- nothing below touches attention, `families/qwen4/attn.rs`, or
the `index_budget` refusal at `open()`.

## QSA indexer: persistent state and the composed incremental update (2026-09-05)

`crates/gpu/src/qsa_indexer_state.rs`'s `QsaIndexerCacheManager` -- the
indexer's own KV-cache-shaped state, filling in the `QsaCacheManager` plan
`docs/QWEN4_PHASE0.md` section 5 named and left unbuilt. Two buffers per
QSA layer (allocated only where `mask == 1` AND
`compressed_attention.index_budget > 0`, since `mask == 1` alone covers
every OTHER family's ordinary dense-attention layers too): the RAW
(un-normed, un-roped) key history, and the pooled-block cache built from it
incrementally.

**The raw key buffer shares its position with the main `KvCacheManager` by
DESIGN, not by convention**: it owns no cursor of its own, and every write
takes a caller-supplied `position` (`write_raw_key`, mirroring
`KvCacheManager::write_k`'s exact shape) -- section 5's own words, "must
share a position counter with the KV cache," read literally rather than as
a suggestion. Two independently-advancing cursors over one token stream is
the "upstream restore-misalignment bug" that sentence exists to warn
against, and the fix is structural: there is only ever one cursor,
`KvCacheManager`'s own, and this struct is driven by it rather than
tracking a second.

**The pooled-block cache is the opposite: it owns a real cursor**
(`pooled_block_count`, `advance_pooled_blocks`), mlx-vlm's own
"first_new_block" design -- a pooled, normed, roped block is never
recomputed, so the cursor is genuinely this struct's own state and nothing
else in the engine tracks it. The cursor refuses to move backward or past
the layer's block capacity, both guards mutation-checked individually.

`crates/gpu/src/qsa_indexer.rs` gained `encode_qsa_advance_blocks`,
composing the pooling and scoring kernels above with the ALREADY-EXISTING
`encode_rms_norm_bf16w_perhead_centered` and `encode_rope_neox_subdim` into
ONE call: pool the newly-completed blocks (batched, one dispatch), norm
them (also batched -- that kernel's own indexing already means "N
independent reductions sharing one weight," whatever it calls the axis),
then RoPE each one individually (NOT batchable: `encode_rope_neox_subdim`
takes one scalar position for the whole call, and each new block's own
first-token absolute position differs -- a real, permanent per-block
dispatch cost rather than an oversight). Verified against the SAME
composed chain `crates/compute`'s own `the_full_indexer_chain_composes`
test builds by hand from its CPU primitives, at the identical parameters.

**Two real, mutation-caught gaps came out of testing this, both worth
carrying past this feature.** First: the composed-chain test's first draft
used `key_start_position = 0` throughout, matching every other fixture in
the file, and a mutation dropping that argument from the RoPE call
survived completely silently -- the two formulas coincide at
`key_start_position == 0` by construction. AGENTS.md Gotcha 23's
self-relative-fixture shape landing on this exact argument, one session
after the CPU reference hit the identical trap over `rotary_dim`. Fixed
with a dedicated test starting at position 100. Second: the incremental
design's whole point -- that advancing block 1 alone must never touch
block 0's already-computed row -- has no way to be seen by a single
one-shot test covering every block at once; it needs two separate encoded
passes checked against each other, which
`advance_blocks_leaves_earlier_blocks_untouched_on_a_later_incremental_call`
does. Both gaps are closed now, in `crates/gpu/tests/qsa_indexer_parity.rs`
and `crates/gpu/tests/qsa_indexer_state.rs`, 15 tests total across both
files, `cargo test -p turbospark-gpu` green throughout including the
full pre-existing suite (51 test binaries, re-run twice this session to
confirm nothing else moved).

**What is still genuinely unbuilt, and deliberately not attempted here**:
whatever the attention worktree decides (gather vs. fused kernel) is the
piece that actually CONSUMES this state -- reading `raw_keys_view`,
calling `encode_qsa_advance_blocks` each time a block completes, then
using `select_blocks` and the pooled/scored result to drive real Q/K/V
attention. None of that exists yet, on purpose: it is a different session's
work by the user's own split. Also still open: lifting
`RealForwardRunner::open`'s refusal above `index_budget`, and the real
long-context hardware verification that only becomes possible once context
can actually exceed 2,048 tokens on this family -- neither is reachable
until the attention piece lands.

## What's next

1. ~~A memory oracle~~ -- **DONE** (2026-09-04, above).
2. ~~Root-cause the within-process nondeterminism~~ -- **DONE** (2026-09-04,
   above): `ple_conv_tail` was never cleared by `reset()`. Fixed, and the
   `try_reuse_prefix` `real_qwen4.is_some()` gap was closed in the same
   change.
3. ~~A throughput row~~ -- **DONE** (2026-09-04). `turbospark-bench --model`
   against the same install, two interleaved runs, both at 16 expert-cache
   slots: `short-explanation` read 7.182 and 9.035 tok/s, `medium-review`
   read 11.204 and 8.689 tok/s, peak footprint 2503.6-2509.7 MiB across both
   (agreeing with the memory oracle's own 2503-2509 MiB). `long-synthesis`
   refuses at warmup exactly as expected (2,940-token prompt over the
   2,048-token window). **The spread is wider than every other family in
   this repo's protocol table** -- roughly 25-60% case to case against the
   few-percent spreads Gotcha 15 in `crates/bench/CLAUDE.md` records
   elsewhere -- which is this family's routing profile rather than
   measurement noise: 288 experts at top-10 against a 16-slot cache means
   which experts are already resident when a case starts (left over from
   whichever case ran before it, or from nothing on a cold cache) has an
   outsized effect on that case's own hit rate, unlike a family whose
   experts mostly fit the cache regardless of history. Report a RANGE for
   this family rather than a single number, and expect a future measurement
   to land somewhere in it rather than reproducing either run exactly.
4. Whether this checkpoint's 288-expert, top-10 routing profile needs its
   own `ALLOWED_CACHE_SLOTS` tuning pass beyond Phase 4's widening -- now
   there are both a memory oracle and a protocol throughput reading to
   measure against, and the tok/s spread item 3 records is itself an
   argument for asking: a wider cache should narrow that spread as well as
   raise the floor.
5. ~~Resolve whether an MTP head artifact exists for REAP-288~~ -- **DONE**
   (2026-09-04, above): it does, `sh0wie/Qwen3.8-Flash-Next-MTP-Drafter-MLX-bf16`,
   confirmed byte-for-byte matching this document's own item 8 inventory.
   Wiring it is still unstarted, and is NOT a `families/qwen/mtp.rs` reuse
   (that assumption was written into this section's first draft and
   corrected the same session, above): the head is a complete QSA + 512-expert
   MoE decoder layer with its own hyper-connections, needing (a) a repack path
   for a second artifact with its own MoE section, (b) a second, independent
   expert cache separate from the trunk's pruned 288-expert one, and (c) a
   broadcast-addition fusion, not `qwen3_5`'s concatenation. Read this
   section's "What this means for scoping the work" paragraph before
   estimating the size of this, and re-derive item 13's economic-viability
   question (a recurrent trunk's rollback cost against a head this
   expensive) before committing to build it.
6. **QSA indexer groundwork, further along than "partially" (2026-09-04,
   both entries above).** Repack was already complete from Phase 1
   (verified against real bytes this session). A CPU reference for block
   pooling, scoring and top-k selection exists
   (`crates/compute/src/qsa_indexer.rs`), and the norm/RoPE convention that
   was open earlier the same day is now RESOLVED from source (mlx-vlm's
   indexer shares the trunk attention's own `rotary_emb` object, so it is
   `rms_norm_centered` plus `rope_neox_subdim` at `rotary_dim=64`,
   `theta=1e7` -- both already-existing functions in this crate, no new
   kernel-shaped CPU reference needed) and composed end to end in a new
   test. **The pooling and scoring Metal kernels now exist too** (same day,
   later session) -- `qsa_pool_blocks_mean_fp16` and `qsa_score_blocks_fp16`
   (`crates/gpu/src/shaders/qsa_indexer.metal`), matching the CPU reference
   exactly and mutation-checked. **The indexer's own persistent state and a
   composed incremental-update dispatch now exist too** (2026-09-05):
   `QsaIndexerCacheManager` (`crates/gpu/src/qsa_indexer_state.rs` -- the raw
   key history sharing the main KV cache's position counter, plus an
   incremental pooled-block cursor) and `encode_qsa_advance_blocks`
   (`crates/gpu/src/qsa_indexer.rs` -- pool, norm and RoPE the newly-completed
   blocks in one composed call). 15 tests across both new files, two of them
   catching real mutation-only-visible gaps (a `key_start_position` argument
   whose drop was invisible at the position-0 fixtures every other test
   used, and the incremental design's own "an earlier block must survive a
   later call" property, provable only across two separate encoded passes).
   **This work was explicitly split from the remaining piece**: the user
   took the sparse-attention APPLICATION (gathering or fusing the
   selected-block mask into real Q/K/V attention -- structurally unlike
   this port's existing dense/windowed attention kernels, and an open
   design question between a gather-then-reuse-dense-attention approach and
   a from-scratch fused sparse kernel mirroring mlx-vlm's own) to a separate
   worktree. Still open, unblocked by anything above but not started here on
   purpose: lifting the `index_budget` refusal at `open()`, wiring
   `families/qwen4/attn.rs`'s QSA-as-dense-attention path to become real
   block-sparse attention above the 2,048-token budget, and the real
   long-context hardware verification that needs both the attention piece
   and the budget lift to exist first. See `docs/QWEN4_EXP.md`'s own QSA
   sections for the fuller account.
