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

## What's next

1. A memory oracle (peak `phys_footprint` ceiling) and a quality gate
   (frozen teacher-forced perplexity plus greedy/sampled output digests),
   neither of which exists for this family yet -- the standing pattern
   every other family in `docs/BENCHMARKS.md` already follows.
2. A throughput row once the above exist, so a future kernel or dispatch
   change to this family has something to compare against.
3. Whether this checkpoint's 288-expert, top-10 routing profile needs its
   own `ALLOWED_CACHE_SLOTS` tuning pass beyond Phase 4's widening, once a
   memory oracle exists to measure against.
