# The `qwen3_5` vision tower

What is built, what it measures, and the traps. Facts about the CHECKPOINT
(tensor inventory, mRoPE semantics, activation magnitudes, the INT4 decision)
live in `docs/VISION_PHASE0.md` and are not repeated here; this page is about
the IMPLEMENTATION.

Status as of 2026-09-06: milestones M-V0 through M-V9 are done AND COMMITTED
(`c400329`, `2aee922`; the "not yet committed" note that stood in "What is
not built" for a week is corrected there), and the FFI and the macOS app
reach the tower too. The tower runs, agrees with mlx-vlm, and an image
reaches a generated token from the CLI, from both server endpoints, and from
`ts_generate` -- which is what the SwiftUI app and the in-process server sit
on. An image prompt also CHUNKS its prefill now, on the CLI and through the
FFI; the server's image path still takes the sequential loop.

**Updated 2026-09-06**: the tower no longer has to be bundled inside a full
trunk install to run one. "The vision memory sidecar" section below covers
the standalone `<alias>.gturbo-vision/` format, `--vision-sidecar`, and
`pull-vision`, verified end to end on real hardware; its own closing
subsection names what is still open (B2, B3, Part C, Part D).

**THE FRONT-END GAP WAS THE LAST ONE AND IT WAS INVISIBLE FROM THIS PAGE.**
Every milestone through M-V9 was true of the engine and of two front ends,
and a third read "complete" beside them while being unable to send a picture
at all: `ts_generate` took `[{role, content: String}]` with no shape an image
could travel in, `FfiChatModel::run_completion` refused one by name, and the
app's file picker declared no image type. Nothing was broken and nothing
said so. The lesson is the one `docs/BENCHMARKS.md` learned about a stale
blocker sentence: a capability is complete per CALLER, and a page that
counts milestones does not count callers.

## The pipeline, end to end

Five crates, in the order a page moves through them.

| stage | crate | what it produces |
|---|---|---|
| decode, resize, normalize, patchify | `vision-io` | `[patches, 1536]` f32 rows, merge-window order |
| position tables | `vision-io` | pos-embed index/weight table, rope frequency rows, mRoPE triples |
| ingest | `repack` | `packed_vision/` (27 blobs) + 9 `vision.*` resident tensors, all FP16 |
| kernels | `gpu` | six FP16 kernels: LayerNorm, two GELUs, 2-D rope, bidirectional attention, matmul, residual add |
| the forward pass | `runtime` | `[merged_tokens, 5120]` FP16 rows |

`crates/compute::vision` is the CPU reference every kernel is held to, and
`crates/runtime/tests/vision_tower_synthetic.rs` is what holds the whole
composition to it.

## The forward pass

Read off `mlx-vlm/models/qwen3_vl/vision.py` rather than inferred:

```text
h    = patch_embed(rows)                    # [seq,1536] x [1152,1536]^T + bias
h   += fast_pos_embed_interpolate(grid)     # bilinear over the 48x48 table
rope = rot_pos_emb(grid)                    # [seq, 36] frequency rows
for blk in 0..27:
    h = h + attn(norm1(h))                  # cu_seqlens = [0, seq]: ONE segment
    h = h + mlp(norm2(h))                   # GELU tanh
out  = merger(h)                            # norm over 1152 per row, reshape, fc1, GELU erf, fc2
```

Real shapes: depth 27, hidden 1152, 16 heads of 72, intermediate **4304**
(not 4608, which is the merger's width), out_hidden 5120, patch 16, temporal
2, merge 2, position grid 48x48, LayerNorm eps 1e-6.

## Injection: how those rows become a token (M-V5)

Two seams in `families/qwen/`, and one inherent method to set them up.

`RealForwardRunner::set_prompt_vision(embeddings, positions, prompt_len)`
takes the tower's output plus
`turbospark_vision_io::mrope_position_triples`' walk of the prompt's ids, and
builds a `vision::PromptVision`. Additive: no trait changed, because
`LogitProducer` is implemented by a scripted mock with no notion of an image.

**The embedding blit**, at the single `encode_embed_any` call in
`produce.rs`. At an image-pad position the tower's row IS the embedding, so
the table lookup is REPLACED and not blended -- the placeholder id carries no
meaning. It is a host write into `scratch.x`, the first in the repo from
outside the decode flow, and it is sound because of WHEN rather than because
of a barrier: the pass is not committed until the first router wait far below,
so a host write landing before commit is visible to every dispatch in it.

**The mRoPE dispatch**, at the single `encode_rope_neox_subdim` call in
`attn.rs`, mask-1 layers only. `position` keeps its other two jobs unchanged
-- the KV slot index and the `position + 1` attention span -- so only the
angle moves and the cache is untouched.

### The dispatch condition is a property of the DATA

`RopePosition::Triple(t, h, w)` with `t == h == w` takes the PRE-EXISTING
kernel. `get_rope_index` gives every text token of a mixed prompt exactly
that, including the text between and after images, so the divergence test IS
the "is this an image pad" test. Nothing separate has to be plumbed and
nothing can drift out of step.

The two kernels share `apply_neox_pair`, so the boundary is exact rather than
approximate. Two things are asserted on that:
`at_t_equals_h_equals_w_it_is_bit_identical_to_rope_neox_subdim` compares
`to_bits`, and `a_degenerate_position_table_is_byte_identical_to_no_table_at_all`
reaches the same claim through the real trunk.

**The degenerate arm is therefore a no-op today and the mutation that says so
survives on purpose.** Deleting it -- so every triple reaches the new kernel
-- leaves all ten synthetic cases green, the frozen digest included. It is
kept so a future change to the mRoPE kernel cannot reach a text token at all,
and so the claim rests on which function is called rather than on the shader
compiler continuing to agree.

### Decode continues at `position + rope_delta`

An image block advances the position clock by its LARGEST axis rather than by
its token count, so a prompt's positions run behind its token indices.
`rope_delta` is `max_position + 1 - len` (negative on any prompt with an
image) and a position past the prompt resolves to `(p, p, p)` with
`p = position + rope_delta`. A decode step that used its cache index would
jump.

### The map survives `reset()`, and M-V5 had that backwards

The original rule was "reset clears it", so a bulk-OCR loop could not inherit
page N's spans. `run_raw_completion` calls `producer.reset()` at ENTRY and a
caller sets the map just before that call -- so the clear landed on the map
for the very prompt about to be prefilled. **Every image run prefilled
placeholder embeddings and answered fluently about a page it had not seen**,
with the right prompt length and no error anywhere. M-V7's first end-to-end
run found it; every test until then drove `produce` directly.

The CALLER consumes the map instead, per page. A caller who forgets gets NO
injection rather than the previous page's -- vague answers instead of
confident wrong ones. `rollback` keeps it either way, since a speculative
rewind stays inside one prompt.

## Six things that are one mutation from a fluent wrong model

Each of these produces a plausible embedding rather than an error.

**The attention is unmasked over the whole page.** The reference passes
`cu_seqlens = [0, seq_len]`, one segment, so every patch attends to every
patch. Reusing a causal attention would make the top-left patch blind to
everything below and right of it -- an image that degrades rather than breaks.

**Rope reaches q and k and NOT v.** Measured cost of getting it wrong: cosine
against mlx-vlm falls from 0.99999970 to 0.805 at block 0.

**The merger norms `hidden` PER PATCH ROW, before the reshape.** The reference
builds its `PatchMerger` with `use_postshuffle_norm=False`. Norming the
4608-wide row instead reads 0.600 against 0.99999334.

**The block's GELU is tanh and the merger's is erf.** They agree to ~3e-4,
which is two orders of magnitude under what any composition test can resolve,
so **no test in this repo can see the choice** -- stated as an assertion at
both levels (`the_blocks_gelu_choice_is_invisible_at_the_parity_bound`,
`the_gelu_choice_is_invisible_at_this_bound`) rather than left implied. It is
pinned by the two per-kernel parity cases plus the one-line call site in each
stage, and by nothing else.

**Patch rows carry `(T, P_h, P_w, C)` where the reference carries
`(C, T, P_h, P_w)`.** Deliberate: it lets the repack copy
`patch_embed.proj.weight` verbatim and both GEMM operands read row-major, so
no permutation happens at repack OR per image. A mismatch keeps every shape
and reads a different image.

**`deepstack_visual_indexes` is `[]`** on all five published checkpoints, so
`deepstack_merger_list` is empty and there is nothing to build. Confirmed
off the real configs, not assumed.

## Memory

Peak vision residency is `VISION_SLOTS x block_stride` of pinned host memory
plus one page of scratch:

- **Slots: 2 x 30,490,624 = 58.2 MiB**, held for the runner's life once the
  tower opens.
- **Scratch: sized by the page**, allocated per image and dropped with the
  embedding. About 152 MB at 5,120 patches (a 1024x1280 OCR page) and about
  1.9 GB at 64,516 (the 4064x4064 extreme, essentially the largest input the
  processor accepts).

The tower opens LAZILY, on the first image. A text-only session on a vision
install pays none of the above, which is why `arch.vision.is_active()` and not
`runner.vision.is_none()` is the question "does this install have a tower".

**Row-tiling the MLP (Part B1, landed 2026-09-06) bounds this.** The largest
single allocation in that 1.9 GB was `VisionScratch::h1`, the
`[patches, 4304]` FP16 intermediate -- 555 MB at 64,516 patches.
`fc1 -> gelu -> fc2` is row-independent, so `h1` is now sized at
`min(seq, VISION_MLP_TILE_ROWS)` (2,048 by default) and the block loop runs
the same three dispatches once per tile instead of once for the whole page,
with identical arithmetic -- verified byte-identical at the shipped default
and under a forced multi-tile override on the synthetic fixture. Tiling is
unconditional rather than gated on page size: an ordinary 1024x1280 OCR page
(5,120 patches) already spans several tiles, so the saving applies to it
too, not only to the 64,516-patch extreme. Attention cannot be tiled the
same way -- it is bidirectional over the whole page -- so the lever bounds
the MLP term alone. See "The vision memory sidecar" below for what Part B
did and did not finish.

## The second slot is not load-bearing yet

v1 is synchronous: one `commit_and_wait` per stage, `depth + 2` command
buffers per image, and a plain `pread` into the next slot between them. With
that wait, ONE slot would be correct -- the host cannot overwrite a slot the
GPU is still reading, because the GPU has finished. The second slot is the
shape the later read-pool prefetch needs, and alternating `n % 2` now makes
that a one-line change.

This is measured rather than argued: the single-slot mutation survives every
case in the synthetic gate, which is the predicted result and is recorded so
nobody concludes the double buffer is doing something it is not.

## Four gates, in this order

**Stage 1, `crates/runtime/tests/vision_tower_synthetic.rs`** (1.4 s, no
network). The GPU tower against `compute::vision` reading the SAME install
bytes, plus residency, determinism, and four named refusals. Its fixture's
dimensions are mutually indivisible on purpose -- hidden 64, intermediate 96,
merger input 256 -- where the real tower has shapes that are multiples of one
another, so a transposed read changes a byte COUNT and is caught by
construction.

It CANNOT see whether the tower is the RIGHT function: untrained weights and
this repo's own reference mean a convention both sides share would pass.

**Stage 2, `crates/runtime/tests/vision_tower_parity.rs`** (`#[ignore]`d).
Against mlx-vlm on the same checkpoint, the same revision and the same patch
rows. Setup is in the test's own header. Measured 2026-08-28:

| stage | cosine |
|---|---|
| patch embed + pos | 0.99999995 |
| block 0 | 0.99999970 |
| block 26 | 0.99999383 |
| merger | **0.99999334** |

That merger figure was AT the reference's own FP16-vs-FP32 floor of
0.999993 for this tower, not above it -- with no gap left to attribute, on
the assumption that everything upstream of the merger was already exact.

**RE-MEASURED 2026-09-07 after AGENTS.md/CLAUDE.md B7** (the tower's RoPE
angle table stopped being narrowed to FP16 before the GPU dispatch -- the
angle at pair 0 equals the raw patch coordinate and this checkpoint's real
80x64 grid reaches into the tens, where FP16's step was a real, measurable
error rather than storage noise). The gap the 2026-08-28 row assumed away
was there:

| stage | cosine (2026-08-28) | cosine (2026-09-07, post-B7) |
|---|---|---|
| patch embed + pos | 0.99999995 | 0.99999995 |
| block 0 | 0.99999970 | 0.99999977 |
| block 26 | 0.99999383 | 0.99999843 |
| merger | 0.99999334 | **0.99999801** |

Patch embedding is unmoved (upstream of any rope application, as the
"four stages" mutation table below already established). Every stage from
block 0 onward moved closer to 1.0, and the merger figure is now clearly
ABOVE the 0.999993 floor rather than pinned to it -- B7 was real,
measurable numerics, not a cosmetic cleanup.

**Stage 3, `crates/runtime/tests/vision_inject_synthetic.rs`** (1.1 s, no
network). Ten cases over the same synthetic fixture, covering the two seams
M-V5 adds plus the lifetime rule M-V7 corrected. Four of them are worth naming.

Its sibling `crates/runtime/tests/vision_chunked_synthetic.rs` (2026-09-06)
holds the CHUNKED driver to the sequential one on an image prompt, over a
chunk-span sweep and two 24-token prompts -- one whose image sits inside the
first micro-batch and one whose image straddles the boundary at 16. **Its
useful finding is that an ordinary image prompt cannot separate the blit
from the angle**: deleting either reddens the same set of cases, so a red run
says "vision is broken" and not which half. Two purpose-built cases attribute
it -- a one-merged-token image, whose degenerate table the angle mutation
cannot reach, and a spans-free shifted table, which has no image row for the
blit mutation to reach.

A vision install handed no image reproduces a SEPARATELY BUILT no-tower
install's logits exactly, which says the tower's presence moves no trunk byte.
A degenerate position table reproduces the no-table run byte for byte, which
is the dispatch condition itself. And the blit REPLACES the lookup rather than
riding beside it: changing the token id at an image position must move
nothing, and changing one at a text position must move something -- the pair
is the point, since the first half alone passes against a flow ignoring token
ids and the second alone against one that never blits.

The fourth arrived with M-V7 and is the only one that drives the GENERATION
LOOP rather than `produce`:
`an_injected_map_survives_the_generation_loops_own_reset` calls
`run_raw_completion` and requires the injection to still be there afterwards.
The nine cases that preceded it could not see the `reset()` bug described
above, because not one of them called the function that calls `reset()`. Its
sibling
`only_an_explicit_clear_drops_the_injection_map` states the other half of the
contract, so the map's lifetime is pinned from both ends.

**Stage 4, `crates/bench/tests/vision_logit_dump.rs` + `scripts/kld_mlx_vlm.py`**
(`#[ignore]`d). The full model against mlx-vlm on a text+image prompt. Stage 2
stops at the merger, so this is the only instrument that reaches which rows
land at which positions, which angle each position gets, and whether the mRoPE
selector agrees with the reference's.

It replays the reference's IDS and its PIXELS, and neither is optional.
Preprocessing is held to the reference by `crates/vision-io`'s golden fixtures
and the tower by stage 2, so letting either back in would make a gap
unattributable between four candidates instead of one.

**Since M-V6 this port BUILDS the id sequence and the processor's is the
ORACLE.** The gate used to replay the reference's ids because nothing here
could produce one; it renders, encodes and splices itself now, and asserts
equality. The reference still runs first, which is only about the order of two
commands rather than about who owns the prompt: `prepare` has to write the
pixels and the question before the Rust side can read them.

Measured 2026-08-28, `mlx-community/Qwen3.8-27B-4bit` at revision `3e6447f0`,
the 1024x1280 page (grid 1x80x64, 5,120 patches, 1,280 merged tokens spliced
into a 1,302-token prompt):

| arm | mean nats | top-1 |
|---|---|---|
| **shape floor** (the reference against ITSELF, batched vs cached) | 0.1437 | 95.00% |
| **B: the reference's own merger rows** through this port | **0.1339** | **96.08%** |
| **A: this port's own tower**, the whole pipeline | 0.4214 | 89.70% |
| greedy continuation, arm A, 48 tokens each side | -- | **100%** |

**Arm B is BELOW the floor**, so the injection, the position table, the mRoPE
dispatch and the trunk have no gap left to attribute. Arm A's residual is the
TOWER's storage difference (this port FP16, the reference BF16) amplified
through 64 trunk layers -- which is why the two arms exist. A composite gap
localizes only when its stages can be substituted one at a time.

**READ THE FLOOR BEFORE READING THE HEADLINE.** 0.1437 nats at 95% top-1 is
enormous next to this family's text floor of 0.0000024, and it is not a
defect: 1,280 image positions predict near-tied continuations, so batched
versus cached flips 5% of the argmaxes inside ONE engine. An image prompt's
floor has to be measured on the image prompt; quoting the family's text row
here would understate it by five orders of magnitude.

**AND THE GREEDY ARM IS WHY 0.42 IS NOT ALARMING.** The two engines produce
the SAME transcription -- the same table rows, the same figures, the same word
sequence -- differing only by one leading whitespace token. 0.42 mean nats at
89.7% top-1 sounds like a broken model until the output is read.

The exact accounting, since "100%" over an aligned comparison needs its
denominator stated: 48 tokens generated on each side, best alignment at shift
-1, **47 compared and 47 equal, first divergence `None`**. The shift is this
port's own leading token, which a greedy continuation is free to emit.

### The position table was checked against the reference directly

`mrope_position_triples` was diffed against `get_rope_index`'s own
`position_ids` on all 1,302 positions of this prompt: zero disagreements, and
`rope_delta` reads -1240 on both sides. Comparing the INTERMEDIATE is what
made the remaining gap attributable; inferring the table's correctness from
the logits would have left it as one candidate among four.

### Two metrics failed correct implementations on the way here

Both are recorded because the pattern has now appeared three times on this
feature. The greedy comparison was POSITIONAL and called two identical
transcriptions 10.42% agreement diverging at token 0, because this port emits
a leading newline the reference does not; it aligns within a small window now
and prints the shift. And `prepare`'s first version wrote the processor's raw
`pixel_values` without permuting them to this port's `(T, P_h, P_w, C)` row
order -- 18.87 mean nats on the image positions against a near-exact 0.00026
median on the text ones, which is `crates/vision-io` Gotcha 1 arriving in a
measurement script rather than in the pipeline.

### Read the cosine, not a magnitude-relative error

The gate's first draft used `worst_absolute / rms` and FAILED a correct tower,
reading 1.30 at block 26 while its cosine read 0.999994. Block 26's absmax is
7,904 against an RMS of 117 -- a factor of 68 -- because this tower carries
outlier features concentrated in a few channels. So the worst absolute error
lands ON the outlier, where FP16's own quantum near 8,192 is 8, and comparing
it against a typical element answers a question nobody asked.

Any future instrument on this tower has to be scale-aware for the same reason.
`worst/absmax` is reported beside the cosine for exactly that.

### Four stages, because they LOCALIZE

| mutation | patch_embed | block_0 | block_26 | merger |
|---|---|---|---|---|
| none | 0.99999995 | 0.99999970 | 0.99999383 | 0.99999334 |
| rope also rotates `v` | 0.99999995 | **0.805** | **0.794** | **0.765** |
| merger norms the wide row | 0.99999995 | 0.99999970 | 0.99999383 | **0.600** |

The rope mutation leaves the patch embedding untouched because it is
upstream; the merger mutation leaves all three upstream stages bit-identical.
Neither is inferable from a merger-only comparison, and bisecting it any other
way costs a run per stage.

## Reproducing the parity run

The two probe caches live under `~/models/` (see `CLAUDE.local.md`), not
`/tmp` -- a previous handoff pointed the next session at a `/tmp` cache that
had been cleared.

```sh
# The tower, AT THE REVISION THE INSTALL WAS STREAMED FROM.
python3 scripts/fetch_vision_tower.py mlx-community/Qwen3.8-27B-4bit \
  ~/models/vision-probe-qwen38 3e6447f082e89cc7f0bc6e5441afd38dfce760ff

uv run --python 3.12 --with mlx --with numpy --with pillow --with transformers -- \
  python scripts/vision_tower_probe.py --vendor-root ../mlx-v/mlx-vlm \
    --config ~/models/vision-probe-qwen38/config.json \
    --tower-dir ~/models/vision-probe-qwen38 \
    --image ~/models/vision-probe-qwen38/imgs/page.png \
    --mode dump --out-dir /tmp/vision-dump

TURBOSPARK_QWEN38_VISION_INSTALL_DIR=~/models/qwen38-27b-vision.gturbo \
TURBOSPARK_VISION_DUMP_DIR=/tmp/vision-dump \
  cargo test -p turbospark-runtime --test vision_tower_parity --release -- \
  --ignored --nocapture
```

**The revision pin is load-bearing.** The two published towers have identical
shapes, so pairing the install with the wrong checkpoint's weights compares
two different models and fails nothing loudly.

`vision_tower.*` is NOT contiguous in this checkpoint: the fetch spans
min..max offset and pulls 4,885 MiB for 879 MiB of tensors, where Bonsai's
span is 879 for 879. That over-fetches rather than missing data, which is
correct and costs disk.

## The install

```text
~/models/qwen38-27b-vision.gturbo          15 GB
  packed_vision/layout.json                27 blocks, stride 30,490,624 bytes
  packed_vision/blobs.bin                  823,246,848 bytes
  model_weights.bin                        9 vision.* resident tensors, FP16 (tag 2)
```

**Separate from `~/models/qwen38-27b.gturbo` on purpose.** That one backs
`qwen38_memory_oracle` and `qwen38_quality_gate`'s frozen rows, and adding
~0.9 GiB of tower would force a re-freeze for a component neither gate
exercises. Do not merge them.

## FP16, and why nothing else may read these tensors

The tower is FP16 end to end. Its checkpoints ship it unquantized, the
extreme-page probe puts peak activations at 13.8% of FP16's ceiling with a
factor of 7.3 in hand, and INT4 was measured and REJECTED on OCR quality
(`docs/VISION_PHASE0.md` item 4).

`readable_resident_dtype` therefore accepts dtype tag 2 under the `vision.`
prefix and refuses it everywhere else. The scoping is what makes the exception
safe: `norm_view` and `read_bf16_host` are dtype-BLIND -- they resolve an
unquantized tensor by byte width and decode it as BF16 -- so an FP16 tensor
either of them reaches is MISREAD rather than rejected. `crates/runtime/src/
vision/weights.rs` is the only reader of `vision.` tensors, and it checks the
TAG as well as the width so it cannot quietly become a general one.

The M-V2 kernels bind `half` where every other kernel in `crates/gpu` binds
`bfloat`. The two are the same WIDTH, so binding a BF16 tensor at one of them
passes every length check and reads the bytes as a different number.

## What running it found that reading it did not

**`peek_manifest_arch` never read the tower back.** M-V3 added the vision
fields to `manifest.json` and to `arch_validation` and not to the peeker, so
every caller resolving an install's `ArchConfig` that way -- the CLI's real
generation path and the bench harness -- got `VisionConfig::NONE` from an
install carrying a tower. It opened, decoded text correctly, and refused an
image as though it were headless. Found by the parity gate on its first real
run; `the_manifest_peeker_reads_the_tower_back` now finds it offline in
milliseconds.

Its fallback is `unwrap_or(0)` and NOT the family baseline every other
extension field uses, matching `arch_validation`: an absent field means the
install declares no tower, and another family's answer about ITS tower is not
evidence.

## Building the prompt (M-V6)

Three steps, in this order, and the order is forced: the splice needs each
image's `merged_tokens`, which only preprocessing knows.

```text
preprocess(image)            -> grid, merged_tokens
apply_chat_template(msgs)    -> ONE <|image_pad|> per image
encode(rendered)             -> ids
splice_and_walk(ids, grids)  -> expanded ids + spans + position triples
```

**The expansion is a TOKEN-ID pass and not a text splice**
(`docs/VISION_PHASE0.md` item 6). The template renders exactly one
`<|image_pad|>` per image whatever its size, so the expansion cannot happen
before the template runs -- and it must not happen by editing the rendered
STRING either, because `<|image_pad|>` spelled into prose tokenizes as its
angle brackets and letters rather than as the special token. That produces a
prompt of roughly the right length carrying none of the right ids.

`add_vision_id` stays hardcoded `false`. It only controls an optional
`"Picture N: "` prefix and defaults to falsy upstream, so `false` is already
what the reference sends for the unlabeled case.

### A message carries ORDERED parts

`Message::with_parts(role, [ContentPart::Image, ContentPart::Text(q)])` is
what a multimodal turn looks like. The order is not cosmetic: the template
emits the marker run where the part sits, so moving the image after the
question moves every mRoPE position past it -- fluently.

`content_parts` is EMPTY on every text message, and then the template takes
the `content is string` branch it always took. That is what leaves every
frozen digest where it is, and
`a_text_only_message_renders_identically_through_both_constructors` pins it.

### `splice_and_walk` exists to remove a disagreement

Called separately, the splice takes merged-token COUNTS and the walk takes
GRIDS, with nothing forcing a caller to derive the first from the second.
Pass counts that do not match and both calls succeed, the placeholder run is
the wrong length, and the spans describe a picture of a different size. The
composed helper derives the counts from the grids, so they cannot disagree.
Every front end should reach for it rather than the two halves.

### The fallback renderer REFUSES an image

None of the per-dialect renderers emits a vision marker, so a multimodal
message would render as its text alone -- and then there is no placeholder to
expand, `PromptVision` gets spans that do not exist, and the model answers
about a picture it never saw. A real vision install always ships its own
template, so the refusal is what a MALFORMED install gets.

### Measured: the port's prompt IS the processor's

`vision_logit_dump.rs` renders, encodes and splices, and asserts the result
equals `header.input_ids`. On the 1024x1280 page: 23 rendered ids expand to
**1,302, byte-identical to the mlx-vlm processor's**. Everything downstream
then runs on the port's own ids, and the cross-engine numbers above did not
move by a digit -- which is what says the two sequences really are the same.

## The server (M-V8)

Both endpoints accept images, and one decoder serves them: `/v1/messages`
translates into the OpenAI request the chat route already understands, and
`anyllm_translate` maps an Anthropic `ContentBlock::Image` onto
`ChatContentPart::ImageUrl` on the way. A base64 source becomes
`data:<media_type>;base64,<data>`; a URL source passes through.

**A remote URL is REFUSED rather than fetched.** Fetching one would make the
server an HTTP client driven by request content: an SSRF surface, a timeout
budget and a redirect policy, none of which belongs in a local inference
server.

**The URL SHAPE is validated whether or not this server can serve images**,
and the payload is decoded only when it can. A remote URL is a malformed
request for this server however it is configured, so refusing it on a vision
install and accepting it on a text-only one would leave a client unable to
tell which problem it had. Splitting the check from the decode means a
text-only backend pays nothing for a multi-megabyte data URL it will discard.

**An image this server cannot serve is REPORTED**, on `x-anyllm-degradation`,
on both routes. That header existed for `/v1/messages` and its own module doc
recorded that dropped images were not among the things it knew about -- which
was the gap: a client sending a picture to a text-only install got a fluent
text answer and no signal at all. The turn still succeeds, because the text
half is answerable and refusing it would break every client that sends an
incidental image.

### The encode and the generation share ONE lock

`ChatModel::run_completion` takes the images rather than exposing a separate
"set the images" call, and that is a concurrency property rather than a style
choice. A backend serializes on its one runner per call, so `set_prompt_vision`
followed by `run_completion` would be two locks with a gap -- a second request
arriving in that gap overwrites the map, and the first generation prefills the
second's picture. Both answer fluently.

The map is cleared at the END of the generation, whether it succeeded or not,
which is the contract M-V7 established the hard way.

### Measured, both wire formats

Against the real install on the 1024x1280 page, `max_tokens: 48`,
`temperature: 0`:

```text
openai    00012 | parity streaming streaming oracle fixture tensor kernel oracle manifest ... | 8316.48
anthropic 00012 | parity streaming streaming oracle fixture tensor kernel oracle manifest ... | 8316.48
```

Identical to each other, to the CLI's transcription, and to what both engines
produced in M-V5's greedy comparison.

**The gate asserts on the OUTPUT rather than on a shape**, and that is the
lesson M-V7 paid for: M-V5's injection bug lived through two milestones
because every test drove `produce` directly, the lengths and counts all
agreed, and the model answered fluently about a page it had never seen. A
server dropping the image answers from the question alone and matches none of
the page's line numbers.

## The vision memory sidecar

Status as of 2026-09-06: Parts A1 through A6 and B1 are done, on top of M-V0
through M-V9 above. What this closes: every vision-capable install used to
bundle the tower inside a full trunk, so `~/models/qwen38-27b.gturbo` and
`~/models/qwen38-27b-vision.gturbo` are two independent 15 GB streams of the
SAME checkpoint, differing by ~0.9 GiB of tower. A sidecar is the tower
alone, installed once, attachable to any text-only trunk of the matching
family and hidden size at runtime.

### The format

`model_io::vision_sidecar` -- a `<alias>.gturbo-vision/` directory:

```text
<alias>.gturbo-vision/
  manifest.json               numLayers 0, hiddenSize = the tower's own
                               out_hidden_size, the 15 vision fields set
  model_weights.bin           the 9 vision.* resident tensors, FP16
  packed_experts/layout.json  empty (StreamingGturboWriter's REQUIRED_FILES)
  packed_vision/              layout.json + blobs.bin, write_packed_vision
                               verbatim -- unchanged from a combined install
  preprocessor_config.json    fetched from the source repo
  config.json                 fetched from the source repo
  vision_sidecar.json         the compatibility record (below)
```

**The manifest cannot say "this is a tower, not a model", and that is the
whole reason `vision_sidecar.json` exists as a separate file.**
`load_manifest` refuses an unknown `manifest.flags` key, and
`is_production_arch` keys off `(num_layers, hidden_size)` alone, so there is
no manifest-native place to stamp a "kind" marker without inventing a flag
every OTHER loader would then have to ignore. The record carries what the
manifest structurally cannot:

```json
{
  "kind": "vision-tower",
  "pairsWith": { "family": "qwen35", "hiddenSize": 5120 },
  "source": {
    "repo": "mlx-community/Qwen3.8-27B-4bit",
    "revision": "3e6447f082e89cc7f0bc6e5441afd38dfce760ff",
    "prefix": "vision_tower.",
    "file": "model-00001-of-00003.safetensors"
  },
  "towerBlocks": 27,
  "blockStride": 30490624
}
```

(the exact record for the real `qwen38-vision-tower` install pulled below --
`blockStride` matches the per-block stride the combined install already
writes, per-block-page-rounded, unchanged by this feature).

`sidecar_arch(family, hidden_size, vision)` is the ONE function both the
writer (`crates/repack::write_vision_sidecar`) and the reader
(`model_io::load_vision_sidecar`) use to build this degenerate `ArchConfig`
(`known_architecture(family)` with `num_layers: 0`, `hidden_size` set,
`full_attention_layer_mask` emptied, `vision` set), so the two sides cannot
independently drift. `is_sidecar_dir(dir)` is the whole of "is this a
sidecar" -- true iff `vision_sidecar.json` names `SIDECAR_KIND`.

A `model.visual.*`-prefixed source (an HF-native repo or a standalone
`vision.safetensors` export) canonicalizes onto the same `vision_tower.*`
naming the walk already expects (`canonicalize_vision_header`), so
`read_vision_entries` stays keyed on one prefix. A header carrying BOTH
spellings at once is refused by name, naming both prefixes -- that shape is
not a checkpoint this walk has ever seen, and silently preferring one
spelling would write an install missing half its tower or duplicating roles
under two names, neither of which fails until a dispatch four layers in.

### Attaching one at runtime

`RealForwardRunner::attach_vision_sidecar(dir)`, called AFTER `open()` and
BEFORE the tower's own lazy open (which still happens on the first image,
unchanged): it refuses a trunk that already carries its own tower, a
double-attach, and a family or hidden-size mismatch against the sidecar's
`pairsWith` record, naming both sides of whichever mismatch fired. On
success it sets the runner's own `arch.vision` from the sidecar's, so
`arch.vision.is_active()` -- the same predicate the lazy-open path already
reads -- becomes true with no other change to that path.

The tower's two GPU-bound resident reads (`packed_vision/` and the 9
`vision.*` tensors) bind against the SIDECAR's own `ResidentGpuWeights` /
`ResidentIndex` pair when one is attached, the trunk's otherwise --
`VisionTower::open_with_sidecar` shares its block-loop and merger machinery
with the combined-install `open()` through one private `build()` helper, so
the only thing that changes per arm is which resident buffer the two
GPU-encode call sites bind. `readable_resident_dtype`'s name-scoped FP16
exception (every `vision.`-prefixed tensor, and only those, may be read as
FP16) applies unchanged to the sidecar's own index -- it is a second index,
not a bypass of the first one's rule.

`RealForwardRunner::vision_dir()` returns the sidecar directory when one is
attached, the trunk's own install directory otherwise, and is what every
`preprocessor_config.json` read (CLI, FFI, server) goes through -- so
attaching a sidecar before any of those reads happen is the entire
integration; nothing downstream of that point needed to learn a sidecar
exists. `vision_is_sidecar()` is a test-only engagement accessor, following
the mapped-residency precedent: a byte-identity check alone cannot tell "the
sidecar path ran and matched" from "the sidecar path silently fell through
to the trunk's own tower", so a synthetic gate needs a text-only trunk
fixture (which structurally has no tower of its own to fall through to)
plus this accessor to close the loop.

`MfTokenizer::verify_image_markers(vision_start, image_pad)` renders a
minimal one-image turn through the checkpoint's own chat template and
requires both marker ids to resolve and to appear exactly once each -- the
multiplicity `splice_and_walk` assumes. Called right after attach, so a
mismatched sidecar/tokenizer pairing (an operator points a Qwen tower at a
Gemma trunk, say -- which the family check above would also catch, but this
is the second, independent line of defense against a checkpoint whose
`config.json` and tokenizer disagree with each other) is refused with named
ids at attach time, not deep inside the splice on the first real image.

### Ingest: `turbospark-model pull-vision`

`turbospark-model pull-vision <ALIAS>` (an alias already in `models.json`,
kind `vision-tower`) or `pull-vision --repo R[@rev] --alias NAME [--file F]
[--out DIR]` (an ad-hoc repo). `catalog::stream::fetch_prefixed_shards`
generalizes the existing MTP-shard fetcher (index-driven, or an explicit
`--file` override when the repo's index does not name a vision shard by
itself); `fetch_mtp_shards` is now a thin wrapper over it, unmoved.
`stream_vision_sidecar` reads the repo's `config.json`, derives the family
and hidden size off its text config, calls `repack::parse_vision_config`
(its first PRODUCTION caller -- previously exercised only by an ignored
network test), refuses `VisionConfig::NONE` by name, fetches the shard(s),
canonicalizes the header, and writes through
`repack::write_vision_sidecar` (also its first production caller).

**A vision-tower row bypasses `catalog::gate` outright.** That gate's MLX
check refuses a repository whose `config.json` carries no `quantization`
block, correctly, for a TRUNK -- silence there usually means "not actually
MLX-quantized". A tower repository is legitimately BF16 with no such block
(`mlx-community/Qwen3.8-27B-4bit`'s tower is exactly that), so running it
through the trunk's own gate would refuse every real tower by name. The
tower's own correctness gate is `model_io::load_vision_sidecar`, run after
the write, not a pre-flight probe.

`resolve_vision_sidecar(store, family, hidden_size)` finds exactly one
installed tower pairing with a given shape, reading each candidate's own
`vision_sidecar.json`, and refuses -- naming the candidates -- rather than
silently picking between two towers of the same architecture at different
revisions: per the revision-pin discipline this page already states for the
parity gate, two different revisions of one architecture's tower are NOT
interchangeable. **This function has no caller yet**: `--vision-sidecar
auto` (resolve by family/hidden-size through the catalog rather than an
explicit path) is unbuilt in every front end. Every `--vision-sidecar` flag
today takes an explicit path only.

The `qwen38-vision-tower` catalog row: `mlx-community/Qwen3.8-27B-4bit`
pinned at the same revision (`3e6447f0`) both `qwen38-27b.gturbo` and
`qwen38-27b-vision.gturbo` were streamed from, `download_bytes`
921,460,192 -- the exact sum of the 333 `vision_tower.*` tensors' own
`data_offsets` spans inside that revision's first shard (read off the
shard's own safetensors header, not the shard's whole published size,
since `fetch_prefixed_shards`'s per-tensor ranged reads transfer the
former), matching this page's and `CLAUDE.local.md`'s independently
recorded ~879 MiB for this exact tower.

### Verified on real hardware

`turbospark-model pull-vision qwen38-vision-tower` landed the tower at
`~/.turbospark/models/qwen38-vision-tower.gturbo-vision/`, 879 MiB on disk
(`du -sh`), `vision_sidecar.json` reading exactly the record shown above.

The headline comparison, on the same test page and prompt M-V4/M-V8 already
use (`~/models/vision-probe-qwen38/imgs/page.png`):

```sh
# Arm A: sidecar-attached text-only trunk.
turbospark-check --model ~/models/qwen38-27b.gturbo \
  --vision-sidecar ~/.turbospark/models/qwen38-vision-tower.gturbo-vision \
  --messages-file p.json --image page.png --temperature 0 --top-k 1 --max-new 128

# Arm B: the combined install (the reference).
turbospark-check --model ~/models/qwen38-27b-vision.gturbo \
  --messages-file p.json --image page.png --temperature 0 --top-k 1 --max-new 128
```

Arms A and B's stdout are SHA-256-identical past the resolved-request block
(which differs only in the `model:` path each arm names), reproducing the
same `8316.48`-line transcription this page's M-V8 section already
recorded for the combined install. The sampled arm (dropping
`--temperature 0 --top-k 1`, a fixed `--seed`) is identical too. A
text-only prompt run on `qwen38-27b.gturbo` with and without
`--vision-sidecar` attached is also byte-identical, confirming attach
perturbs nothing when no image ever arrives.

This is the direct, real-weights confirmation of what Part A1's synthetic
tests could only prove structurally (a sidecar and a combined install write
byte-identical tower bytes from the same source checkpoint): the two
install SHAPES really do produce the same model.

**Left open, precisely scoped rather than attempted and abandoned.**
`crates/runtime/tests/vision_tower_parity.rs` and
`crates/bench/tests/vision_memory_oracle.rs` both still read only
`TURBOSPARK_QWEN38_VISION_INSTALL_DIR` (the combined install); neither has a
sidecar-aware env arm, so the tower's mlx-vlm cosine and the multi-page
memory ceiling have not been re-measured specifically THROUGH a
sidecar-attached trunk. Adding one is not a trivial change to either file:
the parity gate additionally needs a `TURBOSPARK_VISION_DUMP_DIR` reference
dump and reads the install's arch via `peek_manifest_arch` rather than
through an opened runner, and the memory oracle's own
`assert_agrees_with_catalog` is deliberately kept tied to
`qwen38-27b-vision` (a decision made explicitly when this feature started,
to keep that install and its frozen rows standing). The CLI comparison
above is the evidence that a sidecar-attached run reaches the identical
bytes those two gates already certified; a session that wants the sidecar
arm measured through those specific instruments should budget it as its
own pass rather than a follow-on to this one.

### Part B, and C, landed

`VisionScratch`'s per-page buffers were the other memory cost this feature
had not closed, and all three of B's sub-parts are now built. B1
(`VISION_MLP_TILE_ROWS`, described in "Memory" above) row-tiles the MLP so
`h1` never holds more than one tile's rows -- the -555 MB term at the
64,516-patch extreme this page already named as the lever. B2 aliases the
five buffers that are never live at once under the serial encoder's
commit-order guarantee (`normed`/`attn`, `q`/`proj`, `k`/`m1`, the last pair
equal in byte count by algebraic identity rather than by luck), dropping
`VisionScratch` from seven physical `seq * hidden` allocations to five. B3
(`crates/runtime/src/vision/budget.rs`) derives `max_pixels` from the
memory guard's own budget -- a binary search against `VisionShape::
scratch_bytes` between the checkpoint's `min_pixels` and its declared
ceiling -- rather than only from the checkpoint's declared ceiling, so a
`--load-guard strict` session on a small machine gets a smaller image
than the checkpoint would otherwise hand it, with the whole subtraction
shown (`VisionBudgetTooSmall`) when even the floor does not fit.

Part C (`release_vision_tower`, exposed through `ts_session_release_vision`)
gives an idle session back the two pinned streamer slots (or the
mapped-residency mapping), the position table, and a sidecar's own
resident weights and mmap, without forgetting an attached sidecar
directory or un-declaring the install's vision capability -- the next
image reopens the tower from wherever it would have opened from before.

Part D (a Qwen3-VL Phase 0 scoping document, `docs/QWEN3VL_PHASE0.md`) is
the one sub-part that is deliberately documentation only, with no code:
fact-finding for a future bring-up, not a bring-up.

## What is not built

Nothing of M-V0 through M-V9, as of 2026-08-29: all three of M-V9's items
landed and are on `main` (see the last paragraph of this section, which
corrects what this line used to point at).

**One thing the milestone list never covered and that IS now built
(2026-09-06): an image prompt can CHUNK its prefill.** The dense qwen
chunked driver refused a live `prompt_vision` map by name -- a deliberate
first-cut scope line, since the injection in `produce.rs` was the family's
only embedding call site -- so the one family with a tower could not chunk
the prompts that need it most. A real page is over a thousand merged tokens
of a ~1,300-token prompt. `families/qwen/prefill.rs` mirrors both halves
now, the blit and the mRoPE angle, and `crates/runtime/CLAUDE.md` Gotchas
14 and 27 carry the design. `TURBOSPARK_BATCHED_GEMV` plus an image prompt
is still refused, about the ANGLE rather than the embedding.

**Still not built, and it is pre-existing rather than new**: the MTP /
DFlash2 VERIFY pass is vision-blind in both halves -- `produce_batched`
embeds every row from the table and rotates at the raw position. Past an
image prompt a decode position is `(p, p, p)` with `p = position +
rope_delta`, so the target and the verify rotate by different angles, and
the verify pass is what EMITS the accepted tokens. Unreachable today, and
by a checkpoint gap rather than a guard: no install carries both a tower
and an `mtp.*` head, so `speculation_blocker` refuses on the missing head.
Nothing refuses the COMBINATION, so one repack carrying both would make
`--image X --speculative 4` silently wrong. The fix is a fourth arm in
`produce_batched`'s existing by-name refusal set.

- **NaN-safe parity instruments.** The two remaining ungated files
  (`crates/gpu/tests/vision_block_parity.rs`, `crates/gpu/tests/rope_mrope_parity.rs`)
  now guard with `is_finite`, matching every other vision parity file. The
  gap was real rather than cosmetic: `turbospark_compute::rel_error`'s
  `max_abs_diff` folds with `f32::max`, which silently returns the
  non-NaN operand on a NaN input -- AGENTS.md Gotcha 59's "NaN reads as a
  perfect score" shape, on a second instrument.
- **The multi-page memory oracle.** `crates/bench/tests/vision_memory_oracle.rs`
  (new): four rounds over one open runner (large 1536x1536 -> medium
  1024x1280 -> small 512x640 -> large again), asserting peak
  `phys_footprint` does not grow past what the largest page establishes
  AND that each round's transcription contains a marker unique to that
  page (the M-V5 shape a memory-shape-only oracle cannot see by
  construction). Measured peaks: 785.3-871.5 MiB across two full runs;
  the REPEATED largest-page round came in BELOW the first (-69.6 MiB), so
  the flat-peak claim holds decisively rather than marginally. Ceiling
  set to 950 MiB (measured max plus ~8%).

  **The content-assertion marker took three iterations, and the mechanism
  is worth keeping.** A marker drawn from a page's THIRD (last-requested)
  line failed on a genuinely correct transcription that stopped early; a
  marker from the FIRST line's own trailing number failed the same way
  one line earlier. What reproduces reliably across every round of both
  real runs -- even under the worst observed truncation (40 generated
  tokens, every line cut mid-word) -- is a four-word phrase from the very
  OPENING of line 1. This model does not reliably complete a multi-line
  transcription request at greedy/T=0, so a content check has to anchor
  on what it reliably reaches rather than on what it was asked to finish.
- **FP16 overflow capture.** `crates/runtime/src/vision/overflow.rs`
  (new), `TURBOSPARK_VISION_OVERFLOW=/path.json`-gated on the
  `resid_capture.rs` / `ffn_hist.rs` pattern: zero cost and zero readback
  when unset (confirmed by the synthetic suite's frozen-digest test
  staying byte-identical), reads back `scratch.x` after every block,
  fails loudly and immediately on a non-finite value (AGENTS.md Gotcha
  60), warns past 50% of FP16's 65,504 ceiling.

**Verified**: both parity files' tests, the full `turbospark-runtime` /
`turbospark-gpu` / `turbospark-bench` suites, `vision_tower_parity`
against the real install (merger cosine reproduces the documented
0.99999334 exactly), and text-only greedy/sampled smoke on
`qwen38-27b.gturbo`.

**THE TWO PARAGRAPHS THAT STOOD HERE UNTIL 2026-09-06 WERE BOTH WRONG, AND
HOW THEY GOT THAT WAY IS THE POINT.** One said the whole-workspace gate and
a commit were "NOT yet done" and told the reader to re-run both "before
treating this as landed on `main`". The other said no `models.json` row
existed for `qwen38-27b-vision.gturbo`, so the oracle had no
`assert_agrees_with_catalog` sibling, "unlike every other family's oracle".

Both had been false for a week. M-V9 is commit `c400329` (2026-08-29), and
it carries exactly the five files the paragraph listed as unverified; the
catalog row and its oracle tie are `2aee922` the same day
(`crates/catalog/src/models.json`'s `qwen38-27b-vision` alias, and
`vision_memory_oracle.rs`'s `the_baselines_agree_with_the_catalogs_measured_rows`,
which is not `#[ignore]`d and needs no install). Neither sentence had to
change for it to become wrong -- only the work it described had to finish,
which is precisely the rot `docs/BENCHMARKS.md` recorded once already about
a stale blocker. A status line written in the future tense goes stale
silently and nothing goes red.

Check the claim before believing it: `git merge-base --is-ancestor c400329
HEAD` costs nothing and is what settled this.

**What was genuinely still owed, and is now done (2026-09-06).**
`crates/runtime/src/vision/overflow.rs` -- M-V9's third deliverable -- had
no test at all, and nothing in the repo sets `TURBOSPARK_VISION_OVERFLOW`
(deliberately: env is process-global and `cargo test` runs a binary's cases
in parallel). Its scan is split out of the readback as `scan_block` and
covered offline. `vision_tower_parity.rs`'s `#[ignore]` was the only bare
one under `crates/*/tests` and has a reason string. And `docs/TESTING.md`
did not contain the word "vision" at all, through nine landed milestones --
the three gated vision commands are in its ignored-tests section now.
