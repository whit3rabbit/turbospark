# The `qwen3_5` vision tower

What is built, what it measures, and the traps. Facts about the CHECKPOINT
(tensor inventory, mRoPE semantics, activation magnitudes, the INT4 decision)
live in `docs/VISION_PHASE0.md` and are not repeated here; this page is about
the IMPLEMENTATION.

Status as of 2026-08-29: milestones M-V0 through M-V8 are done. The tower
runs, agrees with mlx-vlm, and an image reaches a generated token from the
CLI and from both server endpoints. M-V9 (the multi-page memory oracle and
hardening) is what is left.

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

**The lever if the extreme page ever matters is row-tiling the MLP.** The
largest single allocation in that 1.9 GB is `VisionScratch::h1`, the
`[patches, 4304]` FP16 intermediate, which is 555 MB at 64,516 patches.
`fc1 -> gelu -> fc2` is row-independent, so a fixed row tile caps that term at
the tile's size with identical arithmetic. Attention cannot be tiled the same
way -- it is bidirectional over the whole page -- so the lever bounds the MLP
term alone.

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

**That merger figure is AT the reference's own FP16-vs-FP32 floor of
0.999993** for this tower, not above it. There is no gap left to attribute.

**Stage 3, `crates/runtime/tests/vision_inject_synthetic.rs`** (1.1 s, no
network). Ten cases over the same synthetic fixture, covering the two seams
M-V5 adds plus the lifetime rule M-V7 corrected. Four of them are worth naming.

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

## What is not built

M-V9. See the milestone plan; in one line each:
- **M-V9** the memory oracle's multi-page loop, and hardening. The CLI's
  `--image-batch` already walks pages over one open runner with the tower's
  scratch dropped per page; what M-V9 adds is the oracle ASSERTING that the
  peak is flat across them, plus NaN-safe parity instruments and an FP16
  overflow capture.
