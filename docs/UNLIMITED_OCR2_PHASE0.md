# Unlimited-OCR-2

Baidu's document parser, in `mlx-community/Unlimited-OCR-2-4bit` @
`ac04b808479927cc42ea82e06f55ebecd2b0a623`. The upstream reference is mlx-vlm
0.7.11, `models/unlimited_ocr/`, merged as Blaizzy/mlx-vlm#2601 and #2603. The
family is `unlimited_ocr2`. It is an OCR model only: a text-only prompt runs, but
what comes out is not meaningful (a repeated word on the one prompt tried).

```bash
turbospark-model pull --repo mlx-community/Unlimited-OCR-2-4bit@ac04b808479927cc42ea82e06f55ebecd2b0a623 --alias unlimited-ocr-2-4bit
```

```bash
turbospark-ocr --model unlimited-ocr-2-4bit page.png
```

v1 shares `model_type: "unlimited-ocr"` and is refused at the door, naming the
Core ML profile that runs it: its R-SWA ring is a different cache.

## One decoder layer

`crates/runtime/src/families/unlimited_ocr2/state.rs` has the pseudocode. What
separates it from `deepseek2`, whose FFN half it shares:

| | `unlimited_ocr2` | `deepseek2` |
|---|---|---|
| Attention | plain q/k/v over a full KV, 10 heads of 128 | MLA, compressed cache |
| Keys read | 512 at most: 12 routed blocks of 32, plus a 128-token window | all |
| Rope | page2d: `(token in page, page mod 2)` | yarn by position |
| Router | BF16 gate through a GEMV | INT8 |

Prefill is the decode path, one token at a time. That equals upstream's chunks
because a query only routes to blocks ending `local_blocks` before its own, which
are complete under either driver.

## Where the routing lives

On the host, in `compute::dash`, at a per-layer commit. A layer commits twice per
token: once to read the normed row and route, once to read the expert gate. A
router kernel would remove one of those waits and is the obvious next speedup. It
would need its own parity against this file.

## Six things that are one mutation from a fluent wrong model

Items 1 to 3 and 6 were checked by breaking them and watching the test fail, and
item 4 by the mlx-vlm vectors. For item 5 the detector has a unit test, and the
render itself was checked on the real CLI.

1. **The bias is added after the attention scale.** Upstream's additive mask
   lands on the scaled scores.
2. **SAM's relative-position term is not scaled at all.** It is `q` dotted with
   the tables, added to `scale * q.k`.
3. **CLIP's activation is `x * sigmoid(1.702 x)`.** Neither GELU the other towers
   use.
4. **Trailing prompt text is on no page.** It routes to any block, and only a
   generated `<PAGE>` restricts it. Candidates apply inside a page's image span.
5. **A template that prints `content` raw prints a part list as JSON.** This
   checkpoint's does. The prompt then reads `[{"type": "image"}, ...]` and the
   model hallucinates, with no error and the right image rows injected.
6. **Stale block summaries are invisible to a same-runner replay.** They keep the
   low indices, so `reset` is only tested against a fresh runner.

## Ties, and why a box can differ by one

Logits are FP16, which steps by 0.25 at this magnitude, so a box's digits can
tie exactly. mlx-vlm's logits differ a little anyway, being BF16 arithmetic, so it
sees the tie differently. Both `turbospark-ocr` and the sampler take the lowest
index, so the two paths here are byte-identical to each other. An earlier reader
kept the last index on a tie and matched this page's mlx-vlm output exactly, which
was luck of direction and not agreement, so that match is not evidence of anything.

## Install layout

| File | Holds | Read |
|---|---|---|
| `model_weights.bin` | trunk resident weights, 298 MiB | at open, pinned |
| `packed_experts/` | 11 routed layers, 64 experts of 1.9 MiB | streamed |
| `uocr2_tower.bin`, `.json` | SAM, CLIP, projector, FP16, 803 MiB | on the first image |
| `unlimited_ocr2_dash.json` | `sparse_attention_config`, unmodified | at open |

The tower has its own file because its kernels read FP16 and the resident walk
narrows to BF16, and so a text prompt never pays for it. The projector is the one
quantized tower tensor, and it is dequantized at repack. The router constants are
in a sidecar because they are per-layer values one family reads, and `ArchConfig`
is a struct literal in every baseline.

## Page layout

One page is gundam mode: a 1024 global view plus up to 16 640 crops. Several pages
are one 1024 view each, which is what the model card and the fork's README
describe. **mlx-vlm 0.7.11's `process_one` tiles every page unless told otherwise**
(`cropping = True` when unset), so a comparison against it must pass
`cropping=False` for several pages.

Prompt positions: `<bos>`, then per page `<PAGE>` and its image tokens, then the
text, which is `Multi page parsing.` in every released example. Fused rows go local
tiles first, then the global view, then the view separator, with a newline row per
grid row.

## What was measured

Rendered test pages, on one Apple M4 Max 36 GB, on a machine not confirmed quiet.
These are single indicative readings, and nothing here is frozen in
`docs/BENCHMARKS.md`.

| Check | Result |
|---|---|
| `compute::dash` against mlx-vlm vectors | exact key sets, bias within 2e-4 |
| Preprocessing against PIL, after upstream's BF16 rounding | worst pixel error 0 |
| Tower stages against mlx-vlm, 1024 view | cosine: SAM 0.99983, CLIP 0.99940, projector 0.99974 |
| Tower stages against mlx-vlm, 640 view | cosine: SAM 0.99996, CLIP 0.99972, projector 0.99987 |
| Whole page's fused rows | cosine 0.99964, worst row 0.967 |
| Decoder on mlx-vlm's rows, teacher forced, 171 steps | 2 argmax flips, both near-ties (reference margin 0.25) |
| Top-5 logits on the same run | mean error 0.087 on logits near 25 to 31, worst 1.19 |
| Whole pipeline, free-running, one page | text equal to mlx-vlm, stops at EOS, 2 of 20 box coordinates differ by 1 |
| Whole pipeline, free-running, two pages | text equal, 2 box coordinates differ, largest by 1 |
| `turbospark-ocr`, one page | 23.7 to 27.8 s total, 5.4 GB peak |
| Same page twice, and at 16 and 64 expert slots | byte-identical each way |
| Page encode, 13 views | 4.4 s |
| Prefill, 1,519 tokens | 14 to 21 s, with the head on every token in the test harness |
| Decode | about 85 tok/s |

For scale, upstream's own #2605 reports 130 tok/s decode and 660 tok/s prefill on
an M2 Pro at 7.03 GB peak. The prefill gap is the largest cost here.

**Not measured:** a real document, a dense or tabular page, sampled decoding, a
memory oracle, a quality gate, KL against upstream, an FP32 tower. Upstream
documents an MXFP4 trunk needing an FP32 tower. This checkpoint is affine 4-bit,
which upstream reports passing with stored BF16, and this port runs the tower in
FP16. No gate here distinguishes FP16 from BF16 on a hard page.

## Reproducing

```bash
python scripts/unlimited_ocr2_dash_oracle.py crates/compute/tests/generated/dash_oracle.rs
```

```bash
python scripts/unlimited_ocr2_decoder_dump.py ~/models/Unlimited-OCR-2-4bit /path/to/dump
```

```bash
TURBOSPARK_UOCR2_INSTALL_DIR=~/.turbospark/models/text/unlimited-ocr-2-4bit.gturbo TURBOSPARK_UOCR2_DUMP=/path/to/dump cargo test -p turbospark-runtime --release --test unlimited_ocr2_real --test unlimited_ocr2_tower_real -- --ignored --nocapture --test-threads=1
```

## Deliberately not done

- Chunked prefill, a router kernel, and mapped expert residency.
- Steering (the flow has no hook, and `family_dispatches_steering` says so).
- Speculation: no published conversion ships a drafter.
- Prefix reuse and rollback. Router state cannot be rebuilt from token ids, so
  every generation prefills.
- Stitching `cross_*` and `append_*` blocks across pages in `uocr2_plain_text`.
  Upstream's converter does that.
- The app's model catalog UI, beyond what the catalog row gives it.
