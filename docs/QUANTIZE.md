# GGUF quantizer

For every supported type, the architectures that run and the Unsloth Dynamic
files, read [GGUF.md](GGUF.md) first.

`turbospark_repack::gguf_quantize` rewrites a float GGUF under a recipe that
picks a target type per tensor. It is separate from the MLX `--quantize 4|8`
install flag (`quantize_on_install.rs`), which targets MLX affine weights.

## Scope and honesty

- This is a generic recipe-driven quantizer. It is NOT a reimplementation of
  Unsloth "Dynamic" (UD) quants. Unsloth publishes the names and benchmark
  results of its 3.0 and 3.1 methods, not the per-tensor recipe or imatrix
  data, so no preset here claims to reproduce a UD file.
- The ggml types the Unsloth llama.cpp fork adds (IQ1_XS, IQ1_XXS, IQ1_XXXS,
  type ids 64 to 66, branch `iq1-narrow-grids-b10775`) ARE produced and read: each is
  IQ1_S's ternary codebook behind a narrower index (46, 42 and 38 bytes per 256
  weights). Their block layouts, grids and quantizer search were read from the fork's
  source, and the decoder and quantizer are held to a build of that fork
  (`crates/compute/tests/quant_gguf_ggml_golden.rs`, `gi3_64.bin` to `gi3_66.bin`).
  A stock llama.cpp refuses these files by design (the fork keeps ggml's reserved
  ids 42 to 63 free), so a file using them runs in turbospark and in the fork only.
  The UD-IQ1_S files of Qwen3.8-27B, Qwen3.8-Flash-Next and DeepSeek-V4.1-Flash-1017
  that were header-checked do not use them.
- Phases A to C produce `f32`, `f16`, `bf16`, `q1_0`, `q2_0`, `q4_0`, `q4_1`, `q5_0`, `q5_1`,
  `q8_0`, `tq1_0`, `tq2_0`, `mxfp4`, `nvfp4`, `q2_k`, `q3_k`, `q4_k`, `q5_k`, `q6_k`, `iq1_s`,
  `iq1_m`, `iq1_xs`, `iq1_xxs`, `iq1_xxxs`, `iq2_xxs`, `iq2_xs`, `iq2_s`, `iq3_xxs`, `iq3_s`,
  `iq4_nl` and `iq4_xs` from F32/F16/BF16 sources, with optional importance-matrix weighting (the K, IQ and legacy
  `q4_*`/`q5_*` types use it, the rest ignore it, as in ggml). The 31 types are complete,
  and every one is accepted by turbospark's install gate
  (`model_io::EXECUTABLE_GGUF_TYPES`) in every role, routed experts included. A published
  mix can be followed with `--recipe-from` (below). MTP-module handling is not implemented
  (the UD-IQ1_S headers checked had no MTP tensors).
- `q1_0`, `q2_0`, `tq1_0`, `tq2_0`, `mxfp4` and `nvfp4` were reverse-checked against ggml's
  own bytes (`tests/fixtures/gguf_quantize/g3_*.bin` on the adversarial `in3.f32`), and
  three of their rules are ggml's observed behavior rather than anything a spec states:
  rounding ties go away from zero for `q2_0`, `tq1_0` and `tq2_0`; an `nvfp4` scale byte
  takes the higher code on a tie and jumps from 240 straight to 448 once `amax / 6` reaches
  248; a zero `q1_0` element is a set bit. `q8_1` and `q8_K` are ggml's activation formats and
  never appear in a model file, so they are named and refused, not written.
- Low-bit types need an imatrix, as in ggml and llama.cpp: `iq2_xxs`,
  `iq2_xs` and `iq1_s` always (ggml asserts), and `iq3_xxs`, `iq2_s` and
  `iq1_m` on every tensor except `token_embd` and `output.weight` (llama.cpp's
  policy). A missing entry is refused by tensor name.

## Recipe

```json
{
  "default": "q8_0",
  "fallback": "f16",
  "rules": [
    { "match": "blk.*.ffn_*_exps.weight", "type": "q4_0" },
    { "match": "blk.*.attn_output.weight", "type": "q5_0", "layers": [0, 3] },
    { "match": "token_embd.weight", "type": "keep" }
  ]
}
```

- First matching rule wins. `*` matches any run of characters, dots included.
- `type` is a target type or `keep` (copy bytes through).
- `layers` is an inclusive range on the `blk.<N>.` prefix.
- 1-D tensors are always copied. A tensor whose row length (`dims[0]`) is not a
  whole number of target blocks uses `fallback` (default `f16`).
- Unknown JSON fields are errors, so a typo does not silently fall back to the
  default.
- A source tensor that is already quantized is copied when it keeps its type,
  and when only the default applies. An explicit rule asking to change it is
  refused by tensor name.

## Following a published mix

Unsloth's "UD" files pick a type per tensor, and the choice varies layer by
layer. In the Qwen3.8-27B UD-IQ1_S header that was checked, `ffn_down` alone
mixes IQ1_S (46 layers), IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S and Q2_K, and `attn_v`
runs from IQ2_XS to IQ4_XS. No role-level preset can reproduce that, and
this crate does not ship invented "ud-*" presets. Instead it reads the type
map out of a published GGUF and applies it to your own float copy:

```sh
cargo run -p turbospark-repack --release --example gguf_quantize -- \
    --recipe-from Qwen3.8-27B-UD-IQ1_S.gguf --imatrix imatrix.gguf \
    qwen3.8-27b-f16.gguf out.gguf
# or write the recipe out to inspect or edit it:
#   ... --recipe-from <ref.gguf> --dump-recipe recipe.json
```

This copies the *type assignment* only. The values come from your source and
your imatrix (Unsloth says it publishes its imatrix). Your source must use the
same tensor names and shapes (a `convert_hf_to_gguf.py` output of the same
model). A source tensor the reference does not name is kept as it is, and a
reference tensor of a type this crate cannot produce (Q8_1, or a type id from a
fork it has not read) is an error naming it. It reproduces the *types* of the file you
point it at, not its quality: nothing here has been compared against a UD file's
perplexity or KL divergence.

## Importance matrix

`--imatrix <file>` reads a llama.cpp imatrix in either format (legacy `.dat`,
or GGUF with `<tensor>.in_sum2` and `<tensor>.counts`). Semantics follow
llama.cpp: entries are matched by exact tensor name; legacy values are divided
by `ncall`, GGUF values by the per-expert count (a zero count becomes 1.0); an
entry must hold `ne0 * ne2 * ne3` values or the run is refused (except for
`token_embd`, which is skipped); expert tensors use one slice of `ne0` values
per expert. A tensor with no entry is quantized without weights and listed in
the report. The reader loads the whole file into memory.

The imatrix changes the output of every type here except `f32`/`f16`/`bf16`
and `q8_0`, which ignore it, exactly as ggml does.

## From Swift and the C ABI

The app-facing surface is five `ts_quantize_*` entry points
(`crates/ffi/src/api/quantize.rs`, declared in `crates/ffi/include/turbospark.h`)
and the Swift `GGUFQuantizer` over them (see
[SWIFT_BINDINGS.md](SWIFT_BINDINGS.md#quantizing-a-gguf)):

| Entry point | Does |
| --- | --- |
| `ts_quantize_gguf` | the run; options JSON takes `recipe` or `recipeFromGguf` (exactly one), `imatrix`, `threads`, `iq1TieOrder`, `provenance`, `hf` |
| `ts_quantize_cancel` | stops the run in flight; 0 when none |
| `ts_quantize_recipe_from_gguf_json` | `--dump-recipe` as a call |
| `ts_quantize_types_json` | the producible types and whether each is executable here |
| `ts_quantize_check_updates_json` | the provenance freshness check; blocks on the network |

One run at a time per process (`TS_ERR_BUSY`). A cancelled or failed run
leaves no destination and no `.partial` file. These entry points are
macOS-only; elsewhere they return `TS_ERR_UNSUPPORTED`. The model-free checks
are `crates/ffi/tests/quantize_surface.rs` and
`swift/TurboSpark/Tests/TurboSparkTests/GGUFQuantizerTests.swift`. Neither runs a
real quantization, and the Swift stream's mid-run cancellation has not been
exercised against a real model.

## Run

```sh
cargo run -p turbospark-repack --release --example gguf_quantize -- \
    --recipe recipe.json in-f16.gguf out.gguf
```

The output is written to `out.gguf.partial` and renamed on success. Existing
outputs are not overwritten.

## What is verified

`cargo test -p turbospark-repack --lib gguf_quantize`:

- **Byte parity with ggml, all 22 types.** `tests/fixtures/gguf_quantize/`
  holds the exact output of ggml's `ggml_quantize_chunk` (Homebrew llama.cpp
  build 12359, commit b1b770d90) on two inputs, with and without an imatrix
  (types ggml refuses without one have no imatrix-free file). The second
  input is adversarial: heavy outliers, zeroed and nearly constant
  sub-blocks, an all-negative sub-block, tiny- and large-magnitude blocks and
  near-zero importance columns. Every case matches byte for byte. The
  file-level path (reader, writer, expert slicing) is checked against the
  same goldens.
- **Bit-exactness depends on fused multiply-add, and it is per loop.** The
  Homebrew ggml build contracts `a * b + c` into fused multiply-adds (Apple
  clang's default) in most places, and the encoders use `f32::mul_add` at those
  sites, following clang's rule (the left product of `a*b + c*d` is fused). But
  some reduction loops were vectorized and are not fused: the `sumqx`/`sumq2`
  accumulations in the IQ2 and IQ3 encoders, and the distance accumulation in
  the 4-value IQ3 neighbour search, were found only by comparing against
  `llama-quantize` on real weights (below); the synthetic goldens cannot tell
  those apart. A `llama-quantize` built differently (no FMA, another compiler,
  CUDA) can differ from this crate in near-tie blocks. This crate is
  deterministic on every platform because `mul_add` is exactly rounded.
- **IQ1 tie order.** ggml sorts each IQ1 block with libc `qsort`, which is
  not stable, so the order of exactly equal values depends on the platform.
  Real weights converted from BF16 have many ties, so it matters. Measured on
  Qwen3-1.7B: this crate's default (`--iq1-tie-order stable`, ties keep index
  order) agrees with Unsloth's published IQ1_M tensors on 99.8 to 99.9% of
  blocks, against 98% for the Apple order, so Unsloth's build behaves like a
  stable sort (consistent with Linux glibc). `--iq1-tie-order apple` is a port
  of Apple's libc `qsort`, checked against the system `qsort` on 400 tied arrays,
  and makes every IQ1 tensor byte-identical to Homebrew `llama-quantize` on
  macOS (310 / 310 on both IQ1 mixes, against 0 / 90 IQ1_M tensors with the
  stable order). Choose the setting that matches the build you want to follow.
- One input pair is still a thin basis for "all inputs": parity on other data
  is likely but not proven.
- Each of Q4_0, Q5_0, Q8_0 round-trips through the independent `compute`
  dequantizers within one quantization step.
- A hand-derived Q4_0 block fixture (a mutation of the rounding offset fails it).
- An end-to-end rewrite parsed back with the production header parser: types,
  metadata (including an array), alignment, and tensor data.
- Refusal and keep paths for already-quantized sources.

### Real model (Qwen3-0.6B)

`unsloth/Qwen3-0.6B-GGUF`: the BF16 GGUF as source, `imatrix_unsloth.dat`, and
the type map of each published UD file via `--recipe-from`.

| mix | vs Homebrew `llama-quantize` (same source, imatrix, types) | vs Unsloth's published file |
|---|---|---|
| UD-Q2_K_XL (9 types) | 310 / 310 tensors byte-identical | 126 / 310 identical, others differ by a few bytes |
| UD-IQ2_XXS (5 types) | 310 / 310 tensors byte-identical | 276 / 310 identical, others differ by a few bytes |

KL divergence against the BF16 model (60 chunks of 512 tokens of this repo's
docs, `llama-perplexity`), mine versus Unsloth's file:

| model, mix | Mean KLD, Unsloth | Mean KLD, this crate | PPL, Unsloth | PPL, this crate |
|---|---|---|---|---|
| 0.6B, UD-Q2_K_XL | 0.638 +/- 0.005 | 0.637 +/- 0.005 | 58.02 | 58.06 |
| 0.6B, UD-IQ2_XXS | 2.045 +/- 0.014 | 2.045 +/- 0.014 | 202.158 | 202.160 |
| 1.7B, UD-IQ1_S | 1.934 +/- 0.013 | 1.934 +/- 0.013 | 132.836 | 132.830 |
| 1.7B, UD-IQ1_M | 1.645 +/- 0.012 | 1.645 +/- 0.012 | 102.191 | 102.185 |

(BF16 PPL is 40.57 for 0.6B and 30.92 for 1.7B. The absolute losses are large
because these are 1 and 2-bit mixes of small models.) Qwen3-1.7B adds the two
IQ1 mixes: 310 / 310 tensors byte-identical to Homebrew `llama-quantize` with
`--iq1-tie-order apple`, and 99.8 to 99.9% block agreement with Unsloth's files
with the default order.

The residual gap to Unsloth's files is not this crate: `llama-quantize` on
the same machine gives exactly the same 126 and 276 identical tensors on the
0.6B mixes. It comes from their build environment, llama.cpp version or
source.

Speed: rows are encoded on all cores (`--threads N`, default all; the output
bytes do not depend on the thread count, which a test checks). On a 14 core
Mac, Qwen3-0.6B takes 6.1 s (Q2_K_XL) and 11.7 s (IQ2_XXS) against 41 s and 87 s
single threaded, and Qwen3-1.7B takes about 21 s per IQ1 mix, level with
`llama-quantize` (7 s / 11 s on 0.6B, 23 s / 25 s on 1.7B).

An `#[ignore]`d test (`dbg_real_rows`) compares rows extracted from a real
model against ggml bytes; it is how the fusion and tie-order differences above
were found.

Not verified: other model families and sizes, the Unsloth fork's IQ1_XS
family, behavior and speed on a multi-GB model, the GGUF-format imatrix on a
real file (the legacy `.dat` format was read from Unsloth's real
`imatrix_unsloth.dat`; the GGUF format is checked only against hand-built
files), and whether turbospark can execute every type it can now write.
