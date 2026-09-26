# qwen4_exp (Qwen3.8-Flash-Next): implementation status and evidence

`docs/QWEN4_PHASE0.md` records the checkpoint contract. This page records the
implementation contract: intake, decode, memory policy, real-install
failures, and the evidence that closed them. Read it before changing
`families/qwen4/` or the safetensors write path. Read the Phase 0 page first
when the question is about checkpoint shape rather than runtime behavior.

## Current support state (2026-09-26)

Qwen4Exp is an existing family. The catalog now verifies its REAP-288 MLX
checkpoint and the pinned Swift-1.5 IQ2_XS GGUF as separate artifacts. The
new entry is `qwen4exp-swift-iq2-xs`; its quality and resource rows belong to
that GGUF only.

| Artifact | Support state | Boundary |
| --- | --- | --- |
| Qwen3.8-Flash-Next REAP-288, MLX INT4 | verified | existing Qwen4Exp quality and memory baselines |
| Swift-1.5 Qwen3.8-Flash-Next GSQ-RCO, GGUF IQ2_XS | verified | text-only, pinned GGUF and sidecars, context gates at 2,048 plus QSA probe at 4,096 |
| Swift GGUF Q2_0 | excluded | upstream labels it experimental; the exact install fails this port's generation checks |
| Swift GGUF IQ3_XXS | untested | no install or runtime evidence |

The separate 0.91 GB BF16 vision projector is not part of the text install.
The IQ2_XS quality result is this port's own regression sentinel, not a
Swift-engine parity or factual-accuracy claim. The early activation
investigation used [SlotStream](https://github.com/carloslfu/slotstream) as
its debugging reference; the detailed comparison and resulting fixes are
recorded below.

The IQ2_XS alias is already in the catalog used by the desktop welcome screen
and Model Hub. Those surfaces rank against current fit settings. Its frozen
resource row is for an M4 Max at 2,048 context and 16 slots; the fresh app
defaults to 4,096 context with automatic slots. The measured row alone does
not establish that default fit, so keep the artifact available in All Models
and let the dynamic fit ranking decide whether to show it as Recommended.

## Evidence map

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
was already known before this session. **(Superseded 2026-09-05: the indexer
is wired and the refusal is gone; see "QSA wired end to end" at the end of
this document. The paragraph stays as the record of what this session saw.)**

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

## GGUF intake: Swift GSQ-RCO Q2_0-path artifacts, quality gate did not pass

The original Qwen4Exp bring-up left GGUF intake out of scope. The request for
`ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF` scopes that source path.
This is the existing `Qwen4Exp` family, not a new decoder: the baseline and
decode flow are wired through the safetensors intake above. The GGUF registry
now admits `qwen4exp`; that means the generic decode path and type mapping are
present, not that this exact artifact has passed generation or resource gates.

The source revision is
`b22d729eae29b5796f76fb70f91aef549b9fc52c`. Its IQ2_XS recovery capsule is
14.9 MB and includes the original GGUF header, all tensor descriptors and
shard offsets, without duplicating model weights. The published tiers are two
shards each, 66.55 GB (Q2_0), 68.15 GB (IQ2_XS), and 75.97 GB (IQ3_XXS). The
Q2_0 tier was the first full-artifact target. The network install streamed the
complete payload and passed install validation. After the PLE index and Q5_K
resident-type fixes below, the task-local install was repaired to match the
corrected metadata and type tags, and all 50 manifest-listed files passed
SHA-256 verification. The full writer was not rerun after those final fixes.
The routed-header comparison below now finds that both Q2_0-path directories
have IQ2_S at layer 0 gate where the pinned Q2_0 header says Q2_0. The earlier
install and generation checks therefore do not prove a complete Q2_0 install.

The first shard header declares `general.architecture=qwen4exp` and 75 metadata
keys. The continuation shard carries split metadata and the remaining tensor
descriptors; together the headers describe 1,224 tensors. Their inspected
structural fields match the existing baseline: 48 layers, hidden width 2,560,
512 experts with top-10 routing, 24 query heads, 2 KV heads, and a
full-attention layer every four layers. The PLE table is one `IQ4_NL` tensor
shaped `[160, 320001536]`; this differs from the 128 shard layout handled by
the existing safetensors writer.

The pinned Q2_0 headers expose additional source-layout gates. All 144 routed
gate, up, and down tensors use Q2_0. Metal parity covers Q2_0 in both routed
phases and all ten selected experts. A targeted mutation changed its signed
level mapping from `(q - 1) * d` to `(q - 2) * d`; the all-Q2_0 ten-slot case
failed with a 1.019 output difference at row 0. After restoring the shader,
the all-Q2_0 case and three mixed IQ2*/Q2_0 ten-slot cases passed. Among
1,079 resident tensors, 58 Q2_0,
16 Q4_0, seven Q5_0, and one F16 tensor are converted to BF16 during repack;
the Q2_0 resident tag remains refused by the runtime. The 36 `ssm_out` tensors
have dimensions `[6144, 2560]` and use five formats: 17 IQ4_XS, three Q3_K,
11 Q4_K, three Q5_K, and two Q6_K. Their 48 value heads are 128 columns wide,
while each source block spans 256 elements. The repacker dequantizes these
projections, permutes their columns, and writes BF16. Focused tests cover the
Q2_0, Q3_K, Q4_K, Q5_K, Q6_K, IQ3_S, and IQ4_XS paths. Ordinary resident
Q5_K tensors retain their packed type; only Q5_K `ssm_out` tensors are
dequantized and permuted. IQ3_XXS, IQ4_NL, and IQ4_XS now reach their resident
Metal GEMV kernels. The PLE tensor uses IQ4_NL blocks directly. Its writer
streams whole rows in bounded 65,536-row ranges into an IQ4_NL row store, and
the Qwen4Exp PLE reader validates and dequantizes that representation. GGUF
`ple.layers` values are zero-based, so intake converts them to the internal
one-based PLE layer IDs before runtime mapping.

GGUF metadata derivation and explicit name mapping cover all 1,224 tensor
descriptors in the pinned two-shard header. The network header check confirms
their mappings, architecture fields, routed Q2_0 inventory, resident
conversion counts, and `ssm_out` type distribution. Config and transcode
behavior have focused synthetic tests. The full install gate streams the
source shards, downloads the pinned tokenizer sidecars, validates the
manifest, and checks installed-file SHA-256 values.

The install directory then treated as the corrected Q2_0 install opened on
Metal and generated tokens without runtime errors at a 2,048-token context
with 16 expert-cache slots. The required 400-token greedy prose run, using
the chat template with `--reasoning off`,
temperature zero, and top-k one, stopped on EOS after 79 tokens and was
incoherent. Repeating greedily with `--reasoning xhigh` stopped on EOS after
one token. A sampled arithmetic prompt at the CLI defaults (temperature 0.2,
top-k 64, top-p 0.95, seed 20260924) stopped on EOS after 33 tokens and was
also incoherent. The earlier short greeting and arithmetic greedy probes had
the same quality failure. These samples describe the install directory. The
later routed-header comparison shows it does not match the pinned Q2_0 gate
tensor type, so they are not quality results for either published tier.

The [pinned source model card](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF/blob/b22d729eae29b5796f76fb70f91aef549b9fc52c/README.md)
labels Q2_0 experimental and recommends IQ2_XS for a similar size. Its
reported KLD values are vendor measurements, not results from this port.
The current [upstream model card](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF)
also lists IQ3_XXS. This bring-up records IQ2_XS and Q2_0 only; it has no
IQ3_XXS install, source-fidelity, runtime, or resource result.
This intake is text-only. The model card lists a separate 0.91 GB BF16 vision
projector, which the install and runtime gates here do not include.
The exact IQ2_XS shards now have a separate full-install gate in
`qwen4exp_gguf_install_network.rs`; the pinned manifest loaded and all
installed-file SHA-256 checks passed. Its first runtime open exposed an
unimplemented IQ4_XS token-embedding lookup. A Metal selected-row kernel now
decodes the existing 136-byte blocks, with parity against the CPU decoder and
runtime row-size/bounds tests passing. The real IQ2_XS model then opened and
ran on the Apple M4 Max at 2,048 context and 16 expert-cache slots.

That run did not pass the generation gate. The greedy prose prompt at
temperature zero and top-k one stopped at end-of-turn after 55 tokens and
was incoherent. The sampled CLI-default run (temperature 0.2, top-k 64,
top-p 0.95, seed 20260924) stopped after 25 tokens and was incoherent. The
model-card sampling settings (temperature 1, top-k 20, top-p 0.95) with
`--reasoning xhigh` stopped on EOS after 56 tokens and was incoherent. A
short deterministic arithmetic prompt also returned incoherent text after 9
tokens. No run reached the required 400-token coherent response. Existing
Qwen4Exp baselines belong to REAP-288 and must not be applied to either Swift
tier.

The early-stop IQ2_XS results in this paragraph predate the SlotStream-guided
GDN normalization and PLE activation fixes below. They are historical failure
captures, not the current IQ2_XS CLI behavior. The current single-prompt
generation result is recorded in the 48-layer follow-up; it does not replace
the family quality gate or establish factual accuracy across the corpus.

Static sizing for the IQ2_XS candidate at 2,048 context and 16 expert slots
puts the routed slot capacity at 1,170,210,816 bytes (1,116 MiB): 16 slots x
48 layers x the 1,523,712-byte expert stride. The 12 full-attention layers
need 48 MiB of FP16 KV at that window. The 36 GDN layers need 110.1 MiB of
FP32 delta state and FP16 conv tails. The QSA indexer needs 7.5 MiB for raw
keys and pooled blocks, and PLE's conv tail needs 0.18 MiB. Those terms total
about 1,282 MiB before the resident weight mapping, process baseline, and
activation scratch. The 28,800,138,240-byte PLE row table is demand-paged;
its mapped file size is not the resident-memory estimate. These are shape
calculations, not a measured fit result.

One `/usr/bin/time -l` smoke at the same context and slot settings reported a
1,553.6 MiB peak physical footprint for 12 generated tokens, with no swap.
This was a single pre-fix diagnostic run, not a memory oracle or throughput
baseline. The 2026-09-26 follow-up below records a 462-token natural stop on
the chat prompt and an exploratory two-case resource reading. This supports
manual IQ2_XS use on those samples, but it does not establish factual quality
across the corpus, a frozen Swift-specific resource row, or catalog eligibility.
The header gate, install validation, and synthetic parity alone do not
establish output quality or resource fit.

On 2026-09-25, the release `real-generation-v1` benchmark was run on the
directory at the Q2_0 path on an Apple M4 Max at 2,048 context, 1,024
max-new, and 16 expert-cache slots. It used one fresh process per case,
protocol sampling, and the dirty working tree at HEAD
`ee600a8614b04c518dbcc9e3f0ff91c73cf39b3a`.
AC power was checked before and after the pair; the pre-run host reported
90.76% idle. The commands were:

```sh
cargo run --release -p turbospark-bench --bin turbospark-bench -- --model /tmp/turbospark-qwen4exp-swift-q2-0.gturbo --case short-explanation
cargo run --release -p turbospark-bench --bin turbospark-bench -- --model /tmp/turbospark-qwen4exp-swift-q2-0.gturbo --case medium-review
```

The `short-explanation` interval was 00:06:25-00:06:41 CDT: 62 prompt
tokens, 6.73 s prefill, 97 generated tokens, 9.41 s decode, 10.314 tok/s,
and 1,467.3 MiB peak footprint; it stopped at `endOfTurn`. The
`medium-review` interval was 00:07:36-00:08:22 CDT: 426 prompt tokens,
43.93 s prefill, 26 generated tokens, 2.74 s decode, 9.501 tok/s, and
1,631.7 MiB peak footprint; it stopped at `eos`. The short case is one
reading and the medium case is one reading, so these are diagnostic samples,
not a frozen row. The medium case also misses the memory oracle's required
`endOfTurn` validity condition. This protocol run is not a memory oracle or
quality gate, and the later routed-header check means these measurements
cannot be attributed to a complete pinned Q2_0 install.

Follow-up checks ruled out two vocabulary-path explanations for the bad text.
A header-only check against the pinned HF sidecars found that both GGUF tiers
embed the same 248,077 token strings by ID and the same chat template as the
base tokenizer revision `0bd4fe22431372cdad1979267d3ab45aa7e6150a`. It read
the pinned shard headers and tokenizer sidecars without downloading another
weight payload. The opt-in
`pinned_swift_ggufs_match_hf_tokenizer_sidecars` network test passed on
2026-09-25 and reported all 248,077 HF token IDs and the chat template match
for both Q2_0 and IQ2_XS. An ignored real-install probe of the sampled
`medium-review` case decoded 26 tokens before EOS; every generated ID resolved
in that tokenizer. Its test module example originally pointed the IQ2_XS
environment variable at the Q2_0 install path, so the run's tier is not
established from that command. The text was still incoherent. These checks
rule out a mismatched token-to-string sidecar and emitted IDs outside the
tokenizer, but do not establish full runtime parity with the upstream model.

### V-head ordering correction, quality still fails (2026-09-25)

The pinned [llama.cpp Qwen converter](https://github.com/ggml-org/llama.cpp/blob/035e22731a7fd70b9854b3a2d64ec68e9b1a45d3/conversion/qwen.py#L2489-L2751)
reorders V heads by reshaping `[key_head, value_within_key, width]` and
swapping the first two axes. Qwen3.8 has 16 key heads and 48 value heads, so
the ratio is 3:1. The existing two-half interleave only matches that
conversion when the ratio is 2:1. The GGUF walk now restores grouped runtime
order using the architecture's actual K/V ratio, for both row and column
axes and both raw and packed tensors. Focused tests cover the 2:1 and 3:1
layouts. The formatter check and full release repack suite passed.

The available Q2_0 install was not rewritten in place. An APFS clone at
`/tmp/turbospark-qwen4exp-swift-q2-0-vheadfix.gturbo` was changed across its
288 affected tensors (36 linear-attention layers x eight tensors), and its
`model_weights.bin` checksum was updated. The original install remains
unchanged. The clone opened on Metal, but the sampled `medium-review` probe
stopped after three tokens at `endOfTurn` with `1 `. A greedy
`short-explanation` run stopped after 11 tokens with incoherent text
(`O`, followed by `An coastal writers Wetland,,`). A sampled
`short-explanation` run at temperature 1, top-k 20, top-p 0.95, and xhigh
reasoning stopped after six tokens; its output began with U+6B65 U+9AA4
followed by `fair,/write,`. These runs used a 2,048-token context and 16
expert-cache slots on an Apple M4 Max. This targeted clone only isolates the
V-head layout change; it is not a fresh install from GGUF. The correction
fixes a concrete conversion defect but does not pass the quality gate. The
remaining generation failure is open, and neither Swift tier qualifies for
catalog inclusion or a runnable-support claim.

The clone path is the Q2_0 install path. The source header for its Q2_0-named
first shard reports `general.file_type=41`; the IQ2_XS-named source shard
reports `general.file_type=20`. Both tiers use the same routed-expert
`ggmlTypes` list (`IQ2_S`, `Q2_0`, `IQ2_XXS`, `IQ1_M`), so the installed
manifest's type inventory alone cannot distinguish these tiers. The
llama.cpp runs below read the separate IQ2_XS source shards, so they were not
using the same quantized weights as this TurboSpark clone.

The actual `turbospark-check` path was also sampled on this corrected clone
with the frozen `short-explanation` prompt, 2,048 context, 16 expert-cache
slots, temperature 0.2, top-k 64, top-p 0.95, and seed 20260721. With
`--reasoning low`, it stopped at `endOfTurn` after eight tokens and emitted
incoherent text (`P APLACEable. HTML`). With the CLI's default
`--reasoning off`, it stopped at `endOfTurn` after one token without readable
text. Both used the CLI's default 128-token prefill chunks. These are
diagnostic samples, not a quality gate, and changing reasoning effort did
not recover usable generation.

### Upstream llama.cpp diagnostic (2026-09-25)

The two pinned IQ2_XS GGUF shards were assembled at
`/tmp/turbospark-qwen4exp-swift-iq2-xs-upstream` from the complete local
range cache. Every cached range passed its stored SHA-256 check, overlaps
matched, and coverage reached both declared shard lengths. The range files
were removed only after their bytes had been durably written to the assembled
shards. This preserves the downloaded model bytes in the two GGUF files; the
IQ2_XS range cache itself is no longer available for a later install walk.

Homebrew `llama-cli` build `b11146-7fe450e19` loaded the source IQ2_XS
shards with `--jinja`, `--cpu-moe`, `--lazy-mode on`, and no GPU layers. The
diagnostic used the frozen `medium-review` prompt, the template's default
xhigh reasoning, 2,048 context, temperature 0.2, top-k 64, top-p 0.95, seed
20260722, and a 1,024-token output cap. It reached that cap without producing
a final answer. The `short-explanation` prompt at the same settings and seed
20260721 also reached the cap without a final answer. These source-level
runs show that this xhigh / 1,024-token protocol is insufficient for these
two prompts in llama.cpp; they do not explain TurboSpark's much earlier EOS
or establish how a larger budget would behave. The source GGUF did complete
a structured response to `short-explanation` under `--reasoning-effort low`
with a 1,024-token cap, and under `--reasoning off` with a 512-token cap.
Those CPU-only output samples do not pass a factual quality gate. Together
with the TurboSpark low/off results above, they show that xhigh truncation
alone does not explain TurboSpark's early EOS. The TurboSpark and llama.cpp
samplers and compute backends differ. The TurboSpark install path also fails
the pinned Q2_0 routed-type check, so these samples are not a controlled
Q2_0-versus-IQ2_XS comparison or logit-parity evidence.
The medium run used about 20 GiB RSS in sampled macOS `ps` output on a 36 GiB
host; RSS is not comparable to the TurboSpark physical-footprint oracle.
Neither engine's samples establish resource fit or throughput.

The exact frozen `short-explanation` message used for TurboSpark's bad
`--reasoning off` sample was rendered by both engines. TurboSpark's
`apply_chat_template_with_reasoning(Off)` output and llama.cpp's
`/apply-template` output were byte-identical. Tokenizing both rendered
strings produced the same 62 token IDs, matching the actual runs' 62-token
prefill. This uses llama.cpp's documented
[`/apply-template` and `/tokenize` endpoints](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)
with build `b11146-7fe450e19`. Prompt rendering and tokenization therefore do
not explain the off-mode failure for this message. This does not compare
logits, and it does not separate the differing quantizations, repack
conversion, or runtime math.

The cross-engine dump harness now accepts
`TURBOSPARK_QWEN4EXP_INSTALL_DIR`, opens this family at its 2,048-token
window, and records that window in `meta.json`. A real run against the
corrected clone of the Q2_0-path artifact completed with 62 prompt tokens and
512 answer tokens:
574 IDs produced 573 rows x 248,320 logits in float16 (271.4 MiB). It used
16 expert-cache slots and one warmup walk, and finished in 104.02 seconds.
The output hash is
`2f69b9b8e0e9a07fbc31ec873d428dbfecdf094125010c05fc10adfc797350fa`.
This verifies the TurboSpark dump path only; it does not compare against
llama.cpp.

```sh
TURBOSPARK_QWEN4EXP_INSTALL_DIR=/tmp/turbospark-qwen4exp-swift-q2-0-vheadfix.gturbo \
TURBOSPARK_LOGIT_DUMP_DIR=/tmp/kld/qwen4exp-q2-vheadfix \
  cargo test -p turbospark-bench --test logit_dump --release -- --ignored --nocapture
```

An earlier IQ2_XS install attempt from the assembled local shards ran out of
disk space while writing layer 28, with 46 GiB available before the attempt.
The partial install was removed, so the same-tier replay remained unrun at
that point. A later local-shard install and first-token comparison are
recorded below.

A full-matrix, cross-tier check now isolates the V-head conversion error in
`blk.0.ssm_out.weight`. The pinned IQ2_XS source stores this as IQ4_XS, shape
`[6144, 2560]`; all 15,728,640 values were dequantized and compared with the
layer-0 BF16 `linear_attn.out_proj.weight` in both the original artifact at
the Q2_0 path and its corrected clone. The old two-half mapping correlates at
0.99999862 with the original install, while the architecture-derived
16-key / 48-value grouped mapping correlates at 0.99999862 with the
corrected clone. The
cross-mapping correlations are 0.03637 and 0.03637, respectively; leaving
the source tiled gives 0.03610 against the original and 0.03678 against the
clone. This confirms that the old install used the wrong 2:1 mapping for the
3:1 head ratio, and that the corrected mapping matches the source matrix.
This is one resident tensor comparison across two quantization tiers, not
same-tier IQ2_XS installation, logit parity, or a generation-quality pass;
the corrected clone still fails coherence and length.

Before the fresh Q2_0 install, its source-range cache was revalidated
read-only: all 8,170 ranges matched their stored SHA-256 values, together
covered both pinned shards (39,799,117,984 and 26,750,834,816 bytes) without
gaps, and all 12 overlap comparisons were byte-identical. At that point the
cache could support an upstream Q2_0 reference without another download, but
assembling the two GGUF shards while retaining the cache required about
66.55 GB of additional logical storage, above the then-available 34 GiB.
The cache and install were later removed after the fresh fidelity gate below.

The release model probe initially refused the pinned Q2_0 file because it
treated Q4_0 and Q5_0 as runtime kernel requirements. All 16 Q4_0 and seven
Q5_0 tensors in these shards are shared-expert down-projections: the GGUF
name map places them in the resident index, and the Qwen4Exp repacker
dequantizes them to BF16. The probe now reads that per-family, per-tensor
conversion policy from repack and continues to refuse these types in routed
experts, where they would reach unsupported dispatches. The live pinned
release probe now reports `RUNNABLE` and labels both types `transcoded at
repack`. This clears the header-level install eligibility check only; it
does not establish generation quality, resource fit, a fresh install from
the corrected writer, or catalog admission.

An earlier PLE-only revision of the ignored
`qwen4exp_gguf_ple_fidelity_network` test revalidated the existing Q2_0
install's manifest against the pinned architecture, verified all 50
manifest-listed files by SHA-256, and compared the installed PLE row store
with the complete local source range cache. It read all 320,001,536 IQ4_NL
rows (28,800,138,240 bytes) in 65,536-row chunks and matched every byte
against `ngram_table/rows.bin`. That check completed in 122.29 seconds on
2026-09-25 without assembling another shard or table. This is evidence for
the PLE store only.

The expanded gate now checks routed tensor types and sizes against the pinned
GGUF header before reading payloads. It fails immediately at layer 0, expert
0, gate for both the original Q2_0-path directory and its `vheadfix` clone:
the source tensor is Q2_0 while both installed layouts record IQ2_S. Both
manifests identify the model but have no `sourceSnapshotHash`, so the routed
data cannot be tied to this pinned revision. The gate
does not compare routed payloads or rerun the PLE check after this mismatch.
Neither directory establishes source-to-install routed fidelity, real-model
quality for a published tier, or catalog eligibility. A fresh pinned install
was still needed for those results at that point.

```sh
TURBOSPARK_QWEN4EXP_GGUF_INSTALL_DIR=/tmp/turbospark-qwen4exp-swift-q2-0.gturbo \
TURBOSPARK_QWEN4EXP_SOURCE_CACHE_DIR=/tmp/turbospark-qwen4exp-swift-q2-0.source-cache \
  cargo test -p turbospark-repack --test qwen4exp_gguf_ple_fidelity_network \
  --release -- --ignored --nocapture
```

### Fresh pinned Q2_0 install and fidelity (2026-09-25)

The Q2_0 install was streamed again from pinned revision
`b22d729eae29b5796f76fb70f91aef549b9fc52c` after the final intake fixes. The
release `installs_the_real_swift_qwen38_q2_0_gguf` gate passed and reported
`verified install`. The expanded fidelity gate then passed against the pinned
source range cache: routed tensor types and sizes matched, all routed expert
payload bytes matched (31 GiB checked), and all 320,001,536 PLE rows matched
(28,800,138,240 bytes). This closes the prior source-to-install fidelity gap
for Q2_0. The install and its range cache were removed after verification to
make room for the recommended IQ2_XS tier; these checks can be repeated by
streaming the pinned Q2_0 revision again.

The exact Q2_0 install still failed real generation. On an Apple M4 Max at
2,048 context, 16 expert-cache slots, and the frozen `short-explanation`
message rendered through the chat template, greedy generation
(`--max-new 400 --reasoning off --temperature 0.0001 --top-k 1 --seed 1`)
stopped at `EndOfTurn` after one token with no readable answer. CLI-default
sampling (`temperature 0.2`, `top-k 64`, `top-p 0.95`, seed `20260721`)
stopped after three tokens and emitted `The answer`. Neither run approaches
the required 400-token coherent response. Because this install now passes
source fidelity, the quality failure cannot be attributed to the earlier
IQ2_S-versus-Q2_0 artifact mismatch. Whether the cause is Q2_0 quality or a
remaining GGUF/runtime defect is unresolved.

The existing `qwen4exp_memory_oracle` was also run against this install as a
resource diagnostic. It passed its REAP-288-calibrated 3,000 MiB ceiling and
5 tok/s floor, with a 1,583 MiB session peak, 1,578.2/1,583.7 MiB case peaks,
and no replay growth. Its cases generated only 3 and 15 tokens before
`EndOfTurn`. These readings do not establish a Swift-specific frozen resource
row, and the early stops do not clear the quality gate.

### Fresh pinned IQ2_XS install and initial mismatch (2026-09-25)

The IQ2_XS source shards assembled from the pinned revision were installed
through a test-only local-shard source. The release install gate passed all
installed-file SHA-256 checks. The expanded fidelity gate matched 35,454,976,000
routed-expert bytes and all 320,001,536 PLE rows (28,800,138,240 bytes) to the
source shards. This establishes routed and PLE source fidelity for IQ2_XS; it
does not yet compare every resident tensor after repacking.

At this initial checkpoint, before the fixes below, the exact 62-token
`short-explanation` prompt was rendered and tokenized the same way in both
engines. A direct TurboSpark logit probe ranked `O` (token 46, logit 11.671875)
first. The CPU-only llama.cpp server, reading the same IQ2_XS source shards,
ranked `Co` first with log-probability -0.0022366; its next candidate, `The`,
had log-probability -8.0657. The logit magnitudes are from different engines
and are not compared, but their top-token decisions diverged before sampling.
TurboSpark's greedy generation then stopped after 11 tokens with incoherent
text (`O`, followed by `An coastal writers Wetland,,`). This ruled out a
sampler-only explanation but did not identify which tensor or operation was
responsible.

At this initial checkpoint, the IQ2_XS install and first-token probe established
installability and a reproducible divergence, not coherent generation,
resource fit, or catalog eligibility. The follow-up below fixes that first
token mismatch and records a coherent stop. Catalog eligibility and a
Swift-specific resource row remain open. The local-shard install and first-token
probes are recorded in `crates/repack/tests/qwen4exp_gguf_install_network.rs`
and `crates/bench/tests/qwen4exp_swift_first_token_probe.rs`.

### SlotStream-guided GDN Q/K normalization check (2026-09-25)

[SlotStream](https://github.com/carloslfu/slotstream) is the debugging guide
for this mismatch. Its
[`current_backend_reference.py`](https://github.com/carloslfu/slotstream/blob/main/Tools/current_backend_reference.py)
captures early Qwen4 layer outputs, and its
[`qwen4_exp.py`](https://github.com/carloslfu/slotstream/blob/main/Tools/reference/qwen4_exp.py#L2513-L2524)
defines GDN normalization as `x / sqrt(sum(x*x) + 1e-6)`, followed by the
query scale `1 / sqrt(Dk)`. In TurboSpark's RMS-mean kernel, the equivalent
epsilon is `1e-6 / Dk`. The Qwen4 path now uses that value; the shared default
still serves the older Qwen GDN families.

The GPU parity test uses low-magnitude Q/K inputs so it distinguishes the two
epsilon conventions, checks the direct L2 equation independently, and asserts
that V is unchanged. To compare the actual IQ2_XS activation at the first GDN
layer, set `TURBOSPARK_QWEN4_GDN_NORM_CAPTURE` while running
`qwen4exp_swift_first_token_probe`. Set
`TURBOSPARK_QWEN4_GDN_NORM_CAPTURE_POSITION=61` for the final token of its
frozen 62-token prompt. Then run
`scripts/check_qwen4_gdn_norm_capture.py` on the emitted JSON. The capture is
one FP16 `conv_out` before/after pair; it validates this operation only.
SlotStream's MLX/safetensors layer captures cannot serve as direct parity
evidence for this quantized GGUF install.

### SlotStream-guided layer-boundary capture (2026-09-25)

The runtime now has an opt-in trace for the first two layers, matching
SlotStream's early-layer debugging boundary. It captures the wide residual
after any PLE update at layer entry, after the attention join, and after the
MoE join. The last stage is the same output boundary that
current_backend_reference.py writes to layer_0.bin and layer_1.bin.

    TURBOSPARK_QWEN4_LAYER_CAPTURE=/tmp/qwen4-layers.json \
    TURBOSPARK_QWEN4_LAYER_CAPTURE_LAYERS=2 \
    TURBOSPARK_QWEN4_LAYER_CAPTURE_POSITION=61 \
    TURBOSPARK_QWEN4EXP_IQ2_XS_INSTALL_DIR=/tmp/turbospark-qwen4exp-swift-iq2-xs.gturbo \
      cargo test -p turbospark-bench --test qwen4exp_swift_first_token_probe \
      --release -- --ignored --nocapture
    python3 scripts/check_qwen4_layer_boundary_capture.py /tmp/qwen4-layers.json

The capture is FP16 and writes a compact binary sidecar. The checker can
compare after_moe_join vectors with SlotStream layer outputs through
--reference-dir and --reference-position. Such a comparison is parity evidence
only when the checkpoint weights and input token IDs match. In particular,
SlotStream's default MLX model and this Swift IQ2_XS GGUF are different
artifacts; their activation deltas cannot identify a TurboSpark defect by
themselves.

The focused Metal GDN suite passed 8/8 tests. The release-profile IQ2_XS probe
captured layer 0 at position 61; the checker reported zero max error against
the SlotStream equation and zero changed V elements out of 6,144. On the
same frozen prompt, the first TurboSpark logit moved from token `O` (ID 46,
11.671875) before the epsilon fix to `In` (ID 623, 12.25) after it. CPU
llama.cpp still ranked `Co` first at this checkpoint. This confirmed that the
epsilon defect affected the output but was not the only source of the
cross-engine mismatch. The follow-up below traces all 48 layers and removes a
PLE activation defect.

### SlotStream-guided 48-layer follow-up and PLE fix (2026-09-25)

The SlotStream boundary-by-boundary method exposed a second concrete runtime
bug. `gpu::encode_gdn_conv_decode` already returns the SiLU-activated causal
convolution output. Qwen4 PLE applied another SiLU in
`crates/runtime/src/families/qwen4/ple.rs`; removing that duplicate activation
makes the PLE branch match the CPU callback equation. This fix is covered by
the release-profile real-model probe below, not by a full quality gate.

On the frozen 62-token IQ2_XS prompt, TurboSpark selects `Co` first at logit
24.09375, matching llama.cpp's top-1 token (`Co`, logit about 24.0274). Its
12-token diagnostic continuation begins `Coastal wetlands ...`. At this
checkpoint no longer post-fix response had been evaluated.

The opt-in layer trace covers all 48 layers at prompt position 61. Against
callback tensors from the same IQ2_XS GGUF, layer 1's post-PLE entry has RMSE
3.51e-5. Error grows through layer 22 (after-MoE RMSE 2.33e-4), then jumps at
layer 23's MoE join (1.22e-3). It reaches 3.68e-3 after layer 36's attention
join and 1.87e-2 after layer 46's attention join. Cosine similarity at layer
46 remains 0.99934, while the final layer 47 callback outputs were not
captured. This is a cross-engine diagnostic trace, not full-forward parity.

Across the 48 layers, 38 route lists match in order, 9 have the same set in a
different order, and layer 28 differs by one expert at the top-k boundary.
Raw router-logit captures localize both differences to near ties: layer 23
logit RMSE is 0.00617 with cosine 0.9999995 and a CPU cutoff gap of 0.00144;
layer 28 RMSE is 0.01125 with cosine 0.9999993 and a CPU cutoff gap of
0.00572. These are cross-engine numeric differences around top-k cutoffs, not
evidence of a faulty expert sorter. They can change expert order and reduction
order, but the capture has not isolated another runtime defect. The repeatable
capture and comparison commands are in
`.claude/docs/diagnostics.md`, with the callback collector in
`scripts/capture_qwen4_llama_callback.cpp` and the checker in
`scripts/check_qwen4_layer_boundary_capture.py`.

The Q2_0 artifact is a separate tier. The trace used IQ2_XS and did not reopen
or pass the Q2_0 quality gate. At this checkpoint, IQ2_XS still needed a
longer post-fix generation and resource check.

### IQ2_XS CLI and resource follow-up (2026-09-26)

The release `turbospark-check` peer and public `turbospark run` wrapper were
rebuilt against the current source. Both rendered the frozen chat prompt to
the same 62 token IDs as the direct probe and selected `Co` first. A 400-token
cap cut the answer off mid-sentence. With the same greedy shaping and a
650-token cap, the CLI reached `EndOfTurn` after 462 generated tokens at
13.274 tok/s. The readable answer addresses vegetation drag, elevation and
infiltration, two storm-related limits, and risk reduction versus complete
protection. This is a real IQ2_XS integration smoke on an M4 Max, not a
machine-scored factual-quality gate.

The repeatable CLI command was:

```sh
TURBOSPARK_DEBUG_PROMPT_IDS=1 target/release/turbospark run \
  /tmp/turbospark-qwen4exp-swift-iq2-xs.gturbo \
  --messages-file /tmp/turbospark-qwen4-swift-short-messages.json \
  --max-new 650 --max-context 2048 --temperature 0 --top-k 1 --top-p 0.95 \
  --repetition-penalty 1 --expert-cache-slots 16 --prefill-chunk 128
```

The two-case `qwen4exp_memory_oracle` also passed on IQ2_XS with its existing
REAP-288 bounds: 3,000 MiB and 5 tok/s. On M4 Max at 2,048 context and 16
slots, `short-explanation` generated 484 tokens at 13.157 tok/s with a
1,467.6 MiB case peak; `medium-review` generated 601 at 12.883 tok/s with a
1,467.7 MiB case peak. Reported session peak was 1,467 MiB and the
short-case replay changed peak by +0.00 MiB. This is one exploratory reading,
not a frozen Swift-specific resource row.

### Swift FFI integration on IQ2_XS (2026-09-26)

From `swift/TurboSpark`,
`TURBOSPARK_TEST_MODEL=/tmp/turbospark-qwen4exp-swift-iq2-xs.gturbo swift test
--filter RealModel` passed on M4 Max: 9 passed, 6 skipped, in 58.7 seconds.
The passing cases opened the install, streamed a coherent answer, reused KV
across turns, cancelled an active decode, and served generated text through
the in-process HTTP server. The coherence case stopped at its 120-token test
cap. This is Swift-to-FFI-to-Metal integration evidence, not a full-answer,
determinism, or release-quality result. The skipped cases require a separate
drafter install or a vision-capable install and image.

The follow-up `RealModelTests/testFixedSeedSamplingIsRepeatable` also passed:
two same-session, non-greedy 48-token turns with seed `20260721` produced
identical content, token counts, and stop reasons. A negative control with a
different seed on the second turn produced different text and failed the
content equality assertion as expected. This checks same-process seeded
repeatability; it is not a frozen cross-session or cross-process baseline.

At this FFI checkpoint, still open were broader Swift-specific quality
coverage, repeated resource measurements, post-fix Q2_0 generation, and any
IQ3_XXS integration. The Swift IQ2_XS closeout below freezes its quality and
memory rows and adds the pinned artifact to the catalog. Q2_0 remains
excluded and IQ3_XXS remains untested. The router near-tie drift in layers 23
and 28 remains a cross-engine numerical difference; the direct logits do not
point to a faulty sort.

### Swift IQ2_XS support closeout (2026-09-26)

The catalog alias is `qwen4exp-swift-iq2-xs`, pinned to
`ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF` revision
`b22d729eae29b5796f76fb70f91aef549b9fc52c`. Its GGUF sidecars are pinned to
`ukisai/Swift-Qwen3.8-Flash-Next` revision
`0bd4fe22431372cdad1979267d3ab45aa7e6150a`. The actual network install
test passed manifest loading and SHA-256 checks for all installed model
files. This run set `TURBOSPARK_QWEN4EXP_DISABLE_SOURCE_CACHE=1`, so it kept
one model copy on disk. It omitted Rust's `--exact` filter, which also
selected the local-shard sibling and made the aggregate command fail because
`TURBOSPARK_QWEN4EXP_IQ2_XS_SHARD_DIR` was unset. The documented command now
uses `--exact`, and the local-shard test has a distinct name to prevent the
same substring collision. Both install-test paths now fetch all six pinned
sidecars; the intended network-install test itself passed.

The earlier pinned local-shard fidelity comparison matched 35,454,976,000
routed-expert bytes and all 320,001,536 PLE rows to the source IQ2_XS shards.
The fresh network install contains all six tokenizer and chat sidecars from
the pinned base revision. Two small files, `vocab.json` and `merges.txt`, were
fetched after the already-running test binary finished because that binary
predated the test source's sidecar-list update. The local-shard path now uses
the same six-file sidecar list.

The Swift-specific quality gate runs at 2,048 context and 16 expert slots.
Two independent release processes on AC both measured reference-answer
perplexity 4.5957 and identical greedy and sampled digests. The frozen third
pass passed against those goldens. These digests are regression sentinels for
this artifact, not a comparison to the REAP-288 row or to Swift.

The Swift-specific memory oracle also runs the first two protocol cases at
2,048 context and 16 slots. Two AC calibration readings had session peaks of
1,633.8 and 1,633.7 MiB. Short-explanation decoded 484 tokens at 12.682 and
12.666 tok/s; medium-review decoded 601 at 12.367 and 12.243 tok/s. Both
stopped at `endOfTurn`; replay growth was +0.23 and +0.03 MiB. The frozen
assertion pass peaked at 1,472 MiB and passed a 2,000 MiB ceiling and 8.9
tok/s floor. `docs/BENCHMARKS.md` records the observed range and the
artifact-specific margin.

The real 4,096-context QSA probe teacher-forced the 2,940-token
`long-synthesis` prompt. Sampled positions through 2,050 were bitwise
identical between forced-dense and sparse attention. Above the selection
point at 2,051, all 21 sampled argmaxes agreed; KL(sparse || dense) averaged
0.00111 nats, with a 0.00939 maximum. This confirms the sparse path changes
selection above budget without changing the below-budget path. It is an
internal kernel comparison, not upstream parity evidence.

The previous public-CLI smoke completed a coherent 462-token response, and
the Swift FFI suite passed 9 tests with 6 environment-dependent skips. The
catalog status is now `verified` for IQ2_XS only. Q2_0 and IQ3_XXS remain
outside the supported catalog entry.

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
taxonomy (item 9) that appears nowhere else. Read `crates/runtime/AGENTS.md`
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
`crates/runtime/AGENTS.md`: a rejected round on a recurrent trunk must
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
untouched), `score_blocks` (`sum over heads of relu(q . pooled) /
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
`qsa_score_blocks_fp16` (`sum over heads of relu(q . pooled) / sqrt(D)`,
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
   few-percent spreads Gotcha 15 in `crates/bench/AGENTS.md` records
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

## QSA attention application: the indexed decode-attention kernel (2026-09-05)

The prior handoff deliberately stopped short of applying the indexer's
selected-block mask to real Q/K/V, calling it "a genuine design decision
with real correctness risk". This session made that decision and built the
kernel, and NOTHING ELSE: it is not wired into `families/qwen4/`, which
still refuses context above `index_budget` and runs dense attention below
it. It was built in a worktree at HEAD while a concurrent session was
building the indexer's cache manager and the pool/norm/RoPE composition in
the main checkout (the section above). The merge that landed both is the
first time the two halves share a tree, and nothing calls across them yet.

### The decision: gather the INDICES, not the rows

Two shapes were on the table and a third fell out of reading the existing
kernel:

- **A from-scratch fused sparse kernel** mirroring mlx-vlm's. Rejected: new
  online-softmax code, new GQA handling, new chunking, all unverified --
  and the mlx-vlm in this machine's uv cache (`mlx_vlm/models/qwen4_exp/
  language.py`) has no fused kernel at all. Its indexer returns a boolean
  mask and `Qwen4ExpAttention.__call__` ANDs it onto the causal mask and
  calls plain scaled-dot-product attention. The "fused kernel" the previous
  handoff remembered reading is not in the version installed here.
- **Gather the selected K/V rows into a compact buffer, then run the dense
  kernel.** Reuses proven math, but copies ~2 MiB of K/V per QSA layer per
  token on the real shape (2,051 rows x 2 kv heads x 256 x 2 bytes, K and
  V) and needs a gather kernel anyway.
- **Taken: an INDEXED variant of `attention_decode_partial`**
  (`crates/gpu/src/shaders/attention_indexed.metal`,
  `crates/gpu/src/attention_indexed.rs`). Same 256-thread threadgroup,
  same `block_reduce_sum`, same online-softmax recurrence and FP op order,
  same `(m, d, o)` partial layout, same `chunks_for` policy over the LIST
  length. The only change is that the loop walks `positions[i]` for `i` in
  the chunk's slice of a host-built sorted `u32` list instead of `p` in
  `[p_start, p_end)`. Pass 2 is attention.metal's own
  `attention_decode_combine`, unchanged. Cost above budget: one 8 KiB list
  upload per QSA layer per token, no row copy.

The consequence that makes this the low-risk choice is testable rather
than argued: with the identity list at equal chunk count the indexed
dispatch writes the dense kernel's exact FP32 partials and FP16 output,
and `tests/attention_indexed_parity.rs` asserts it bitwise. That equality
is what lets the dense kernel's real-model verification (every family's
frozen gate) stand in for this kernel's until the family that uses it can
be run above budget.

### What was verified, and what the verification itself taught

CPU reference `turbospark_compute::indexed_attention`: a gather followed by
`causal_attention` over the compact sequence, so the two cannot disagree on
the attention arithmetic, only on which rows enter it. Five GPU parity
cases: a non-contiguous subset, a 2-chunk subset, the real shape (`head_dim`
256, 24 q / 2 kv heads, 2,051 of 3,000 rows, 16 chunks, worst |diff| 2.4e-4
against 4e-3), the bitwise identity case, and a NaN case (every unselected
K and V row set to NaN, output required unchanged -- AGENTS.md Gotcha 59's
trap used as a stray-read detector). Three mutations, each asserted to
apply, each reddening exactly its own cases: wrong row (`i` for
`positions[i]`), dropped chunk offset, host forcing one chunk.

**The tests were wrong twice before the mutations reddened anything, and
both failures are worth carrying** (`crates/gpu/AGENTS.md`'s entry has the
numbers):

1. **Keys at magnitude 0.3 made the softmax near-uniform**, so the output
   was the mean of V over WHATEVER rows were read -- close to the same
   number for any row set -- and the wrong-row and dropped-offset mutations
   both passed within tolerance on the multi-chunk and real-shape cases. A
   parity tolerance is only meaningful if the fixture moves by much more
   than it under the mutation being guarded against.
2. **`sin(i * 1.3)` over the flat `[seq, heads, head_dim]` index is not a
   random fixture.** Each key row is a small rotation of the previous one,
   so the scores are near-periodic in position and two large row sets that
   share most of their rows sample that pattern identically (gap 0.0008 on
   the real shape, against a 0.004 tolerance). Hashed independent rows
   (splitmix64 on the index) at key magnitude 3 give O(0.1) gaps.

Both are now structural: `assert_fixture_discriminates` computes the CPU
answer over the WRONG row set (rows `0..n_sel`) and over the first chunk's
worth of the list, and requires both to differ from the right answer by
more than 10x the tolerance before any parity line is trusted. And the
NaN case compares the clean run against the CPU reference as well as
against the poisoned run, because a kernel that read only a prefix of the
list would never touch a poisoned row and would pass the clean-vs-poisoned
comparison on its own.

A third lesson is smaller: **FP16 output rounding absorbs
reassociation-level differences.** The host mutation "always dispatch one
chunk" produced the dense kernel's exact FP16 bytes against its four
chunks, so an output-only bitwise test could not see it. Comparing the
FP32 partial scratch (shared between the two dispatches, so a different
chunk count writes a different slot set and different per-slot `m`/`d`)
is what makes the bitwise claim mean "same chunking" as well as "same
math".

### What remains (Phase B, after the two halves meet)

Everything in `families/qwen4/`: reading `self_attn.indexer.*` at open in
place of the `index_budget` refusal, the `index_qk_proj` GEMV every token
with the raw key copied into the indexer cache, `encode_qsa_advance_blocks`
as blocks complete, and above budget the query norm+RoPE, the score kernel,
a mid-layer commit for the score readback, `select_blocks`, the mask turned
into a sorted position list, and `encode_attention_decode_indexed` in
place of `encode_attention_decode`. Below budget the trunk's dispatch
stream must stay byte-identical, which the frozen quality-gate digests
(perplexity 8.7224) and `the_synthetic_flows_arithmetic_is_frozen` will
say. The first real above-budget decode on this family is the greedy and
sampled smoke at `--max-context 4096` with a prompt past 2,051 tokens; a
force-dense diagnostic seam comparing logits with and without selection is
the only quantitative instrument available above budget, since no 4-bit
copy of this 125B checkpoint fits this machine for a cross-engine KL.

## QSA wired end to end (2026-09-05, Phase B)

The two halves met and were wired the same day: `families/qwen4/attn.rs`
now runs the indexer and, above the budget, sparse attention. `RealForwardRunner::open`
no longer refuses `--max-context` above `index_budget`; it requires the
three `self_attn.indexer.*` tensors per QSA layer instead (INT4 projection,
two BF16 norms, all already in the real install).

### What runs, per QSA layer, per token

1. `index_qk_proj` (640 rows: 4 query heads and one key head of 128) from
   the hyper-connection's `mixed`; the raw key head is copied into
   `QsaIndexerCacheManager`'s history at this position
   (`copy_strided_rows_fp16`, one row).
2. If `(position + 1) / 4` exceeds the layer's pooled-block cursor,
   `encode_qsa_advance_blocks` pools, centered-norms and ropes the newly
   completed blocks (one RoPE dispatch per block, at the block's first
   position) and the cursor advances.
3. The trunk's q/k/v projections, norms, RoPE and KV write, unchanged.
4. At or below 512 complete blocks (`visible <= 2051`): the dense kernel,
   unchanged. Above: the indexer query is normed and roped,
   `qsa_score_blocks_fp16` scores every complete block, the pass is
   COMMITTED AND WAITED ON (the encoder is swapped for a fresh one through
   `&mut PassEncoder`, the MoE router's own readback shape), the host runs
   `compute::select_blocks`, writes the sorted position list (at most
   2,051 entries) and dispatches `attention_decode_indexed_partial`.
5. Output gate and `o_proj`, unchanged.

`TURBOSPARK_QSA_FORCE_DENSE=1` (or `RealForwardRunner::set_qsa_force_dense`)
keeps the dense kernel above budget as a diagnostic arm.

### Verified on the synthetic fixture

The fixture now carries the three indexer tensors (3 query heads plus one
key head of 16, compress 4) and takes an `index_budget` parameter, so a
test crosses the budget at position 19 instead of 2,051
(`build_synthetic_qwen4_exp_decode_install_with_indexer_budget`).
`crates/runtime/tests/real_forward_qwen4.rs`, 17 tests green:

- `the_synthetic_flows_arithmetic_is_frozen` reproduced its frozen hash
  UNCHANGED with the indexer running every token below budget: the
  below-budget exactness claim, on the fixture.
- `sparse_and_forced_dense_agree_below_budget_and_diverge_above`: one
  fixed token sequence through both arms; bitwise equal at positions 0 to
  18, different at every position 19 to 27.
- `the_indexer_projection_moves_the_output_only_above_budget`: patching
  `index_qk_proj` moves the tiny-budget decode and leaves the default-budget
  decode bit-identical.
- `a_tiny_indexer_budget_decodes_sparsely_past_it`: greedy through every
  `visible % 4` tail phase, finite throughout.

Three mutations: "never sparse" reddens exactly the two tests built to see
it; "identity list instead of the mask" reddens those two plus the
position-buffer capacity assert; "`>=` instead of `>`" (select one block
early) SURVIVES, and legitimately so: at exactly `top_k` complete blocks
`select_blocks` keeps every block, the list is the identity, and the
indexed kernel is bit-identical to the dense one on it. The boundary is
unobservable in output and costs only the extra commit
(`crates/runtime/AGENTS.md` Gotcha 34).

### Verified on the real install

Greedy smoke, `long-synthesis` prompt (2,940 tokens, 889 of them past the
budget), `--max-context 4096 --max-new 256`: coherent and on topic
throughout, an accurate summary of the prompt's own document, stopped at
`MaxTokens` as a summary of that length should. Footer:

```
[stop=MaxTokens prefill=2940tok/506.58s new=256tok decode=39.88s tok/s=6.419]
```

Read the prefill number as the price of NO chunked prefill on this family,
not as a QSA cost: 2,940 sequential `produce` calls at 5.8 tok/s, each a
full decode step with expert streaming (48 slots, `auto`), and above the
budget each QSA layer adds one commit and wait. Chunked prefill for this
family is the item that would move it.

Sampled smoke, same prompt, CLI defaults (T 0.2, top-k 64, top-p 0.95, seed
20260721): coherent throughout, the same summary in different words, stopped
at `MaxTokens`. Footer:

```
[stop=MaxTokens prefill=2940tok/361.01s new=256tok decode=31.54s tok/s=8.117]
```

(The faster prefill is the expert cache being warm from the greedy run
that preceded it, Gotcha 20's shape; neither number is a benchmark.)

`qwen4exp_quality_gate`, at its frozen 2,048-token window: reference-answer
perplexity 8.7224, greedy digest `9f9ed49203dda1aad1b379eb503a3dd8b565d53ccc38ff417fb35e43ed9e0795`,
sampled digest `4cf6da5560936e306df09fa09bca7bd19950429e910cee2b66c93c8ed0335772`,
all three the frozen values to the last character -- with the indexer's
projection, key copy and block pooling now running on every one of those
tokens. That is the below-budget exactness claim on the real model.

`qwen4exp_memory_oracle`, same window and 16 slots: session peak 2521 MiB
against the 3000 MiB ceiling, decode 9.4 and 9.8 tok/s on the two cases
(floor 5), steady-state replay +0.02 MiB. The two readings frozen on
2026-09-04 were 2503 and 2509 MiB; the indexer's raw-key and pooled-block
caches at 2,048 context are about 7.5 MiB (12 layers x 2,048 x 256 B plus
12 x 512 x 256 B) plus their scratch, which accounts for the move.

### The force-dense probe: the one quantitative instrument above budget

No reference engine for this checkpoint fits this machine (a 4-bit MLX copy
is ~65 GB), so there is no cross-engine KL row for the sparse path. The
substitute is `crates/bench/tests/qwen4exp_qsa_probe.rs`: the same
`long-synthesis` prompt teacher-forced twice through one runner, once under
`set_qsa_force_dense(true)` and once sparse, `KL(sparse || dense)` in nats
and argmax agreement at 23 sampled positions (two below the budget, the
first eight above it, then every 64th). 943 s for both passes. Measured
2026-09-05:

```
position  2039: below budget, bitwise identical
position  2050: below budget, bitwise identical
position  2051: KL 0.00010 nats   ... 2058: 0.00002   (first eight sparse positions)
position  2499: KL 0.00882        2691: 0.09299 (max)   2755: 0.01440
21 positions above budget: KL mean 0.00577, median 0.00002, max 0.09299 nats;
argmax agreement 21/21
```

How to read it: the two below-budget positions being BITWISE identical is
the exactness claim on the real install, one layer deeper than the quality
gate (which only reaches 574 tokens). Above the budget the arms differ
(asserted), by a little: dropping the lowest-scoring blocks of a 2,500-token
prefix moves the next-token distribution by hundredths of a nat at most and
never changes the argmax in this sample, which is what a selector the
checkpoint was trained under should do. A broken kernel reads as nats, not
milli-nats. The probe asserts finiteness, below-budget identity and
above-budget difference, and REPORTS the KL rather than thresholding it
(Gotcha 38); a future reading far from these numbers is the thing to
investigate, not a test to loosen.

### Chunked prefill landed (2026-09-05)

`families/qwen4/prefill.rs`'s `prefill_chunk_real_qwen4` is the "Step 1"
shape every other family's chunk driver already has: loop the existing
per-token kernels inside a micro-batch of up to `MAX_PREFILL_BATCH` (16)
tokens, one command buffer per layer for the attention-and-router half, a
per-token routed-MoE loop pipelined the same way gemma4's is. No new kernel.
The pessimistic note this section used to carry ("would need... a position
list per QSA layer") turned out to be wrong once the driver preserved
gemma4's own per-layer commit-and-wait ordering: the QSA indexer's shared
`qsa_positions` buffer, and its mid-layer above-budget commit inside
`encode_full_attention_block`, needed zero changes.

Five buffers DID need widening to `MAX_PREFILL_BATCH` rows, four for the
by-now-familiar reasons every chunk driver's buffers widen for (`wide_x`
crosses layers; `router_logits_f32` is read back in one host round trip;
`hc_inject` and a new `moe_x` bridge the `cb1`/`"routed cb"` split for
`mlp_hc`). The fifth, PLE's `ngram_emb`, is the sharpest instance of this
whole pattern found so far, and it shipped broken on the first cut of this
driver: its upload (`gpu::write_buffer_bytes`) is a HOST write, executed the
instant the encoding function runs rather than a GPU dispatch queued for
later, so it does not respect command-buffer commit order at all. A
single-row buffer left every token but the last in a micro-batch computing
PLE from the WRONG token's n-gram embedding, silently -- caught by
`real_forward_qwen4_chunked.rs`'s `the_chunk_boundary_does_not_move_the_logits`
at chunk span 2, the first multi-token micro-batch the test tried. See
`crates/runtime/AGENTS.md` Gotcha 14's `qwen4_exp` paragraph for the full
account.

Verified byte-identical against sequential on the synthetic fixture
(`tests/real_forward_qwen4_chunked.rs`: whole-prompt and chunk-span sweep
`[1, 2, 3, 4, 7, 11]`, a span crossing the QSA sparsity boundary at position
19, and the minimal-safe-slot-count case at `2 * top_k`), and on the real
`qwen4-reap288.gturbo` install: greedy (`--temperature 0.0001 --top-k 1`) and
sampled (CLI defaults, seed `20260721`) stdout md5-identical between the
sequential and chunked paths on a 40-token prompt, 48 new tokens each. This
was a short-prompt wiring check, not a throughput measurement -- both arms'
prefill ran at ~40 tokens over 6-8s, too small a prompt to see the win a
2,940-token prefill should get from batching, and cold-cache effects
(Gotcha 20) dominate at this scale. `real_forward_qwen4.rs`'s existing 17
cases, including its frozen digest (perplexity 8.7224, both digests), all
reproduced unmoved -- the buffer widening changed sizes, not logic.

### The chunked driver's first real-install run found a crash, at the default slot count

The synthetic suite was green on seven cases and the driver still could not
prefill a 426-token prompt on `qwen4-reap288`. It panicked in
`crates/streaming/src/expert_cache.rs` with `expert cache cannot place
requested misses`, which is AGENTS.md Gotcha 64 arriving on a second family.

The arithmetic. `routed_pipeline_banks` pipelines only when
`expert_cache_slots >= 2 * top_k`. This checkpoint routes **top-10** and the
bench pins **16** slots, so `16 >= 20` fails and the loop degrades to
`banks == 1`. In that branch the loop still reserved the previous token's
slots through `RoutedSlot::protect`, leaving `16 - 10 = 6` places for a token
that can miss on all 10, and `ExpertCache::plan` asserts rather than
degrading.

The reservation was never needed there. `protect` names slots a command
buffer STILL IN FLIGHT is reading, and the `banks == 1` branch calls
`retire_routed` before planning, so nothing is in flight by the time
`protect` is consulted. The fix is an empty set at `banks == 1`, which is
what Gotcha 64 had already argued for and what `families/gptoss/prefill.rs`'s
own seam does.

**Two things about this are worth more than the fix.**

First, why gemma4 never showed it: that family routes top-8, so its default
16 slots leave exactly 8 for up to 8 misses -- it fits by one, and the
documented reproduction needed `--expert-cache-slots 8`. The bug is not
about the number 8 or the number 16, it is about `slots < 2 * top_k`, and a
family with a larger `top_k` reaches it at settings nobody would call
exotic. This one reaches it through a plain `turbospark-bench --model`.

Second, why seven green synthetic cases missed it. The suite carried a case
called `a_cache_too_small_to_pipeline_still_reproduces_the_sequential_logits`
that opens at `2 * TOP_K` -- which satisfies `>=` and therefore pipelines. It
was named for the fallback and tested the pipelined path. When the threshold
is `>=`, a fixture at exactly the threshold sits on the wrong side of it.
`the_one_bank_fallback_reproduces_the_sequential_logits` opens at `TOP_K`
instead, and reverting the fix reproduces the real install's exact panic
message on the synthetic fixture.

### The throughput measurement, attempted 2026-09-05: NOT CITABLE, and the reference arm says why

Three interleaved pairs on the real install, `medium-review` (426-token
prompt), warmup discarded, AC power, through the new `turbospark-bench
--prefill-chunk`:

| pair | sequential prefill | chunked prefill | ratio |
|---|---|---|---|
| 1 | 66.37 s | 50.07 s | 1.326x |
| 2 | 48.59 s | 47.75 s | 1.018x |
| 3 | 46.08 s | 50.79 s | 0.907x |

**Read the reference arm before reading the ratios.** The SEQUENTIAL arm,
doing byte-identical work three times, spans **1.440x** on its own
(46.08 to 66.37) and does so MONOTONICALLY, 66.37 then 48.59 then 46.08.
That is a warming trend rather than noise, and it is larger than any effect
this A/B could be looking for, so `crates/bench/AGENTS.md` Gotcha 23's gate
is not met and no magnitude here is quotable. The chunked arm is much
tighter (1.064x), which is itself a hint about the mechanism rather than
evidence for the feature.

**The ratios do not even agree on a SIGN.** Pair 1 reads 1.33x, pair 3 reads
0.91x. Dropping pair 1 as further warmup leaves sequential at a 47.3 s mean
against chunked's 49.3 s, i.e. chunked slightly SLOWER, which inverts the
first pair's conclusion. So this is not "a win we could not size", it is a
null result that cannot presently be distinguished from a small loss.

**A null result is the EXPECTED one here, and the arithmetic was available
before the run.** Chunked prefill batches COMMAND BUFFERS, not I/O. This
page's parent (`docs/BATCHED_PREFILL.md`, "The expert `pread` does not
batch") records that the expert read is the one prefill term step 1 cannot
touch, and that is exactly the term this checkpoint is dominated by: 288
experts routed top-10 against a 16-slot cache misses constantly, which is
the same reasoning that scoped the GGUF arm of step 5 out of existence at a
37.1% pread share. A family whose prefill is pread-bound has little for
command-buffer batching to win. **This remains a hypothesis rather than a
measurement**: no phase table was taken for this family, and
`TURBOSPARK_DISPATCH_PROFILE` is the wrong instrument for it (AGENTS.md
Gotcha 66). The cheap version is one `TURBOSPARK_PHASES=1` run read for its
`pread` bucket, on a short prompt so the divisor is not swamped (Gotcha 21).

**What a clean measurement would need**, and why this one could not have it:
the install is 68 GiB on a 36 GiB machine, so it can never be fully page
cached and every run re-reads expert bytes whose residency differs from the
last run's. That is the likeliest source of the monotone warming, and it is
a property of this checkpoint on this hardware rather than of the machine
being busy. More warmup runs would help and cannot fix it. The honest
options are a smaller checkpoint of the same family, or many more pairs than
three.

### P0 phase follow-up (2026-09-09): pread measured, speedup inconclusive

The pinned REAP-288 checkpoint was restored from
`sh0wie/Qwen3.8-Flash-Next-REAP-288-MLX-4bit` at
`668f31bcc56bf9400e64c9463445eee47597c2d9` (68.1 GiB installed), on the
Apple M4 Max 36 GB machine, AC power. All inference ran serially. Raw
commands, output, phase rows, binary hashes, and the harness are saved in
[the verification evidence](verification/p0-2026-09-09.json).

Both arms used the same temporary CLI binary, built from `7a0773a` plus
the recorded working Rust diff. Its only diagnostic change makes
`TURBOSPARK_PREFILL_CHUNK=0` return the sequential path; the production CLI
ignores zero, so merely setting that variable on the production binary
would not be a valid control. The chunked arm sets 128. The exact patch is
saved in the evidence; no production flag, API, or runtime implementation
was changed.

Settings were identical: context 2048, 16 slots, 48 new tokens, temperature
0.0001, top-k 1, seed 1, speculation off, KV quantization off, and
`TURBOSPARK_PHASES=1`. User-message framing of the frozen
`short-explanation` and `medium-review` prompts gives 62 and 426 tokens.
The latter is the longer prompt here; both remain below the QSA threshold.
Every run stopped at MaxTokens with exactly 48 new tokens. Output bytes
were identical within each prompt, including across sequential/chunked
arms. This is a timing experiment, not a full-answer smoke.

The phase table divides by **109 / 473 forward calls**, not by 48 decoded
tokens. Multiplying its rounded ms/token by that divisor reproduces the
total pread bucket within rounding error in every row. Differences below
use the less-rounded total bucket printed by the same table.

| Run (execution order) | Prefill s | Forward total ms | Pread total ms |
| --- | ---: | ---: | ---: |
| long seq warmup, discarded | 31.76 | 35509 | 18895.6 |
| short seq reference 1 | 6.66 | 10943 | 6320.6 |
| short seq reference 2 | 6.09 | 10175 | 5857.9 |
| short seq reference 3 | 6.07 | 10124 | 5837.6 |
| pair 1 long seq | 31.49 | 35207 | 18660.2 |
| pair 1 long chunk | 30.60 | 34267 | 18484.5 |
| pair 2 long chunk | 30.62 | 34255 | 18319.9 |
| pair 2 long seq | 30.93 | 34578 | 18376.7 |
| pair 3 long seq | 30.88 | 34706 | 18256.6 |
| pair 3 long chunk | 30.54 | 34153 | 18392.8 |
| short seq closing reference | 6.47 | 10629 | 6199.5 |

**Pread is a substantial measured cost, about half the increment.** The
long-minus-short sequential means give 24.363 s additional forward time
and 12.377 s additional pread time (50.8%). Cross-combining the observed
reference extremes gives 23.635-25.083 s additional forward time and
11.936-12.823 s pread. These are observed range bounds, not confidence
intervals. They put pread at roughly 48-54% of the increment. Equal decode
budgets do not force equal routed experts across different prompts, so the
subtraction is an estimate of added prefill cost, not a pure isolated
prefill counter. This supports the large-I/O-cost explanation, but not an
exclusive pread bottleneck or a claim that all remaining time is removable.

**The chunked speedup remains inconclusive.** Sequential prefill spans
30.88-31.49 s (0.61 s, 1.020x), decreasing monotonically again. Chunked
spans 30.54-30.62 s. All pairs favor chunked, by 0.89 / 0.31 / 0.34 s
(1.029x / 1.010x / 1.011x), but the mean difference, 0.513 s, is smaller
than the reference's own spread. This does not support a reproducible
speedup magnitude. Pread means are nearly unchanged between long arms,
18.431 s sequential and 18.399 s chunked, well inside sequential pread's
0.404 s spread. The 2048-token frozen benchmark window and all baselines
remain unchanged. After all probes, smoke, gates and phase runs finished,
logs and checkpoint metadata were saved and the task-created install was
removed. Existing user models were retained.

### Third attempt, six pairs (2026-09-18): the negative stands, the item is closed

The pinned checkpoint was re-streamed from the same revision (see
[the evidence](verification/qwen4-2026-09-18.json) for every number below,
including the pull's transport deviation) and the interleaved A/B re-run
with SIX pairs, two discarded warmups, and repeated short references on
both ends. One environmental caveat: the sequence ran on BATTERY (both
arms interleaved under the same power state, so the A/B discipline holds;
the 2026-09-09 AC capture remains the absolute reference, and these
absolute numbers read ~20% slower).

Sequential prefill: 36.81 / 40.44 / 35.15 / 37.82 / 39.33 / 37.06 s (mean
37.77, spread 5.29 s). Chunked: 35.55 / 38.12 / 35.39 / 38.07 / 38.86 /
38.13 s (mean 37.35, spread 3.47 s). Mean saving 0.42 s, inside both
spreads, and the per-pair signs FLIP 3/3. The phase buckets reproduce the
2026-09-09 attribution at the new operating point: pread is 53.3% (seq) /
53.9% (chunk) of forward total and effectively UNCHANGED between arms
(means 22.39 s vs 22.42 s), and the long-minus-short increment is 51.8%
pread. Three attempts (2026-09-05, 2026-09-09, 2026-09-18) now agree: at
this checkpoint's routing profile the prefill is pread-bound and
command-buffer batching has nothing to win. **A reproducible throughput
gain is not merely unestablished; the structural reason is measured.**
This item is closed as a measured negative.

### What remains

- **A reproducible throughput gain**: CLOSED 2026-09-18 as a measured
  negative (third attempt, above). Do not re-open without a change that
  touches the expert `pread` term itself; changing the benchmark window to
  obtain a different result remains forbidden.

  Note WHICH INSTRUMENT that comparison needs. `turbospark-bench` grew a
  `--prefill-chunk` flag on 2026-09-05 and can now drive the chunked path
  for every family, but this family's protocol window is pinned at 2,048
  (the bullet below), so `long-synthesis` does not fit it and the bench can
  only reach `short-explanation` and `medium-review`. The 506.58s baseline
  was taken through `crates/cli`, so the apples-to-apples comparison has to
  stay there; the bench flag is the right instrument for the protocol rows
  and for the energy capture, not for this particular number.
- The bench window: `QWEN4_EXP_MAX_CONTEXT` stays 2,048 so the frozen
  memory-oracle and quality-gate rows keep meaning what they say. Moving it
  to `PROTOCOL_MAX_CONTEXT` lets `long-synthesis` into the protocol and
  re-freezes every row of this family, a decision of its own.
- A GPU top-k would remove the mid-layer commit (12 per token above budget).
  **Profiled 2026-09-18, and the commit is now a MEASURED bottleneck; see
  "The QSA host top-k profiled on real hardware" below.** The build is
  justified by ROADMAP P2.1's decision rule and queued; it was not built in
  that session, so it owes a re-pull of the install.
- The chunked driver's family refusal in `turbospark-bench` (`--prefill-chunk`
  against an install whose family has no chunked driver) has never been read
  off a real run on this machine, because every install on disk answers
  `supports_chunked_prefill()` true and the one family that does not
  (`qwenGdnMoe`) has no install left here.
- `crates/runtime/AGENTS.md` Gotcha 33's one-line gap (the prefix-reuse
  recurrent-state guard is missing a `real_qwen4.is_some()` arm) is still
  open and is unrelated to chunked prefill; inert today because nothing
  wires prefix reuse to this family yet.

## Re-verification on a fresh install, and the QSA host top-k profiled (2026-09-18)

The pinned checkpoint was re-streamed from
`sh0wie/Qwen3.8-Flash-Next-REAP-288-MLX-4bit@668f31bcc56bf9400e64c9463445eee47597c2d9`
into the modality-separated store (`~/.turbospark/models/text/`) and every
frozen row plus the four blocked ROADMAP items were re-run against it. All
raw numbers, commands and the harness scripts live in
[the evidence](verification/qwen4-2026-09-18.json). No production source
changed; the two diagnostic builds used (the sequential-prefill patch and a
pull-transport experiment) were applied and reverted, and the release
binaries were rebuilt from clean sources.

**The artifact is byte-faithful, proven three independent ways.** The
quality gate reproduced perplexity 8.7224 and both frozen digests to the
last character; the memory oracle read 2517 MiB against the 2521 MiB frozen
row (both cases `endOfTurn`, 9.85/9.81 tok/s, replay +0.02 MiB); and
`kv_quant_probe`'s per-width perplexities (+0.0278 / -0.0609 / +0.0315 /
+0.7081 at 3/3.5/4/2-bit) reproduce the 2026-09-09 readings to the fourth
decimal, a deterministic quantization path over a re-streamed 68 GiB.

**The KV4 sampled finding re-confirms, at identical token counts.** The
2026-09-09 triple (KV4 sampled 542 / FP16 control 676 / KV4 greedy 499,
all `EndOfTurn` at the 4096-window CLI pair) reproduced EXACTLY, and the
KV4 sampled answer again opens by denying the prompt's premise ("Coastal
wetlands do not reduce flood damage") while the matched FP16 control keeps
it. Two installations, two source states, same shape: this remains a
sampled answer-quality concern carried per `docs/TRUBOQUANT.md`, not a
clean pass and not a proven general kernel regression.

**`kv_quant_probe`'s footprint question is still unresolved, but the
spread itself is now the phenomenon.** The repeated-off rows read 2516.7 /
2546.1 / 2597.3 / 2665.7 MiB, a 149.0 MiB spread, double the 71.6 MiB of
2026-09-09, and they trend upward across the probe's eight opens. The
quantized peaks (+81.7 to +86.9 MiB over the first baseline) sit in the
upper band but INSIDE the off range, so the probe's own rule holds: deltas
within the reference spread are unresolved. What this run adds is a
candidate cause the earlier entry said was missing: each open of the
streamed 68 GiB install touches more resident pages than the last
(phys_footprint counts the resident mapping on an MoE, AGENTS.md Gotcha
19), and the warming tracks the probe's wall clock, not the KV width. An
arm order that interleaves quantized widths BETWEEN the off rows would
separate "quantized costs footprint" from "the run warmed up"; nobody has
run it.

**The pull's transport deviation, recorded as an observation.** Three
HTTP/1.1-only production-binary pulls stalled at ~0.4 MB/s against the
current `us.aws.cdn.hf.co` backends (whose DNS answer set now includes
EC2 hosts showing millions of out-of-order packets), while single-stream
`curl` to the same IP class ran 8.5-15 MB/s and the one pull allowed to
negotiate HTTP/2 sustained 23-25 MB/s and completed in ~1h45. The
completing walk ran a diagnostic build whose ONLY delta was removing
`HttpRangeSource`'s `.http1_only()`, Gotcha 46's setting, measured when
the bridge was CloudFront-only and worth 3.4x. n=1 per arm at different
network moments, so this is NOT a refutation of Gotcha 46; it is a
re-open trigger: the next multi-GB walk that crawls should A/B the flag
before re-streaming anything else.

### The QSA host top-k profiled on real hardware (2026-09-18)

ROADMAP P2.1's decision rule, executed as written: interleaved
`TURBOSPARK_QSA_FORCE_DENSE=1` vs unset through the production binary, on
a 2,501-token prompt (450 past the 2,051 budget), `--max-context 4096`,
warmup discarded, three pairs, `TURBOSPARK_PHASES=1`. The warmup ran on
battery and was discarded; all six measured arms ran on AC.

Decode (48 tokens per arm): sparse 4.72 / 4.86 / 4.80 s (10.17 / 9.88 /
10.01 tok/s) against force-dense 4.43 / 4.46 / 4.39 s (10.83 / 10.76 /
10.93 tok/s). **Force-dense is faster in 3/3 pairs**, by 0.29 / 0.40 /
0.41 s per 48 tokens, 6.0 / 8.3 / 8.5 ms per decode token, ~7% of decode.
Prefill shows the same direction (means 211.2 s sparse vs 206.1 s dense).
Expert requests are byte-identical between arms (1,223,040 requests, 39.3%
hits), so the delta is attention-path only.

Attribution from the phase tables: the visible share lands in `cb1` wait, +4.90 / +1.29 / +2.52 ms per above-budget forward pass (498 of 2,548
passes are above budget), i.e. ~0.24-0.4 ms per QSA-layer commit-and-wait
across the 12 QSA layers, with the remainder of the arm delta in pread
wall-overlap (same bytes requested, longer wall, because the mid-layer
commits stall the encoder while the expert reads are in flight).

The reading against the rule's own bar: the MoE router's host top-k costs
0.13 ms/token and a GPU kernel there only relocated the sync (Do Not
Revisit 7). The QSA commit costs ~0.24-0.4 ms PER COMMIT, twelve times per
token, and the whole sparse apparatus is a net LOSS versus force-dense at
~2.5K context. **The rule's build branch fires unambiguously**: build the
kernel, and per the rule remove the WHOLE round trip, kernel-written
sorted position list (the selection semantics of
`compute::select_blocks`: top-`min(topk, complete)` by score with
lower-index tie-break, ragged tail always selected, positions ascending),
per-QSA-layer position buffers per the `attn.rs` safety note (the shared
buffer is only safe because of the per-layer commit the kernel deletes),
and a count-from-buffer dispatch for `attention_decode_indexed` (both
FP16 and TurboQuant variants) since the host no longer knows the list
length. The NaN guard on scores moves into the kernel or a debug readback;
`compute::select_blocks` stays as the CPU oracle for parity. Below budget
nothing changes, the frozen digests and `the_synthetic_flows_arithmetic_is_frozen` pin that, and both reproduced on this install the same day.

Not built in this session (the session ended by removing the install per
its own goal, and a numerics-critical kernel deserves better than a
session tail): the item stays open in the ROADMAP with this profile as
its completed first half, and the build owes a re-pull (~1h45 at the
h2-observed rate, or ~50 min if the transport observation above holds).
