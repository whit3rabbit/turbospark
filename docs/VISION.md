# The `qwen3_5` vision tower

What is built, what it measures, and the traps. Facts about the CHECKPOINT
(tensor inventory, mRoPE semantics, activation magnitudes, the INT4 decision)
live in `docs/VISION_PHASE0.md` and are not repeated here; this page is about
the IMPLEMENTATION.

Status as of 2026-08-28: milestones M-V0 through M-V4 are done. The tower runs
and agrees with mlx-vlm. **Nothing consumes its output yet** -- injecting the
rows into the trunk is M-V5, and until that lands no image can reach a
generated token.

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

**The lever if the extreme page ever matters is row-tiling the MLP.**
`fc1 -> gelu -> fc2` is row-independent, so a fixed row tile caps the 555 MB
term at that size with identical arithmetic. Attention cannot be tiled the
same way -- it is bidirectional over the whole page -- so that lever bounds
the MLP term alone.

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

## Two gates, in this order

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

## What is not built

M-V5 through M-V9. See the milestone plan; in one line each:

- **M-V5** injection: blit the merger rows into the residual stream at
  image-pad positions, plus the trunk's mRoPE. **This is what makes an image
  reach a token.**
- **M-V6** tokenizer and template: the post-tokenization splice that expands
  one `<|image_pad|>` into `merged_tokens` copies.
- **M-V7** CLI: `--image`, and image parts in `--messages-file`.
- **M-V8** server: stop dropping image parts on both endpoints.
- **M-V9** the memory oracle's multi-page loop, and hardening.
