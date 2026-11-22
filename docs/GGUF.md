# GGUF support

What TurboSpark reads, runs and writes in GGUF, including Unsloth's "Dynamic"
(UD) files and the extra types of Unsloth's llama.cpp fork. For the quantizer
itself (recipes, imatrix, verification against ggml) read
[QUANTIZE.md](QUANTIZE.md). For how an install is built from a GGUF read
`.claude/docs/modules/repack.md`.

Everything here is checked against ggml's own bytes (decoders and quantizers)
and against Q8_0 twins through whole models (kernels). Two things are NOT
checked and are called out again at the end: no real file of the newer types
has been run against llama.cpp, and the new kernels are not timed.

## Block types

Every quantized ggml block type that can appear in a model file is supported,
in every role: resident tensors (attention, FFN, embedding, head) and routed
MoE experts, both phases. "Quantizer" means `turbospark_repack::gguf_quantize`
can write it. A **dtype tag** is the byte a `.gturbo` resident index stores for
the type.

| Type | ggml id | Elements per block | Bytes per block | Bits per weight | Tag | Quantizer | Imatrix |
|---|---|---|---|---|---|---|---|
| Q1_0 | 41 | 128 | 18 | 1.125 | 29 | yes | no |
| Q2_0 | 42 | 64 | 18 | 2.25 | 25 | yes | no |
| TQ1_0 | 34 | 256 | 54 | 1.6875 | 31 | yes | no |
| TQ2_0 | 35 | 256 | 66 | 2.0625 | 30 | yes | no |
| MXFP4 | 39 | 32 | 17 | 4.25 | 14 | yes | no |
| NVFP4 | 40 | 64 | 36 | 4.5 | 32 | yes | no |
| Q4_0 | 2 | 32 | 18 | 4.5 | 9 | yes | optional |
| Q4_1 | 3 | 32 | 20 | 5 | 27 | yes | optional |
| Q5_0 | 6 | 32 | 22 | 5.5 | 26 | yes | optional |
| Q5_1 | 7 | 32 | 24 | 6 | 28 | yes | optional |
| Q8_0 | 8 | 32 | 34 | 8.5 | 6 | yes | no |
| Q2_K | 10 | 256 | 84 | 2.625 | 17 | yes | optional |
| Q3_K | 11 | 256 | 110 | 3.4375 | 24 | yes | optional |
| Q4_K | 12 | 256 | 144 | 4.5 | 7 | yes | optional |
| Q5_K | 13 | 256 | 176 | 5.5 | 13 | yes | optional |
| Q6_K | 14 | 256 | 210 | 6.5625 | 8 | yes | optional |
| IQ1_S | 19 | 256 | 50 | 1.5625 | 20 | yes | required |
| IQ1_M | 29 | 256 | 56 | 1.75 | 23 | yes | optional (policy: required) |
| IQ2_XXS | 16 | 256 | 66 | 2.0625 | 18 | yes | required |
| IQ2_XS | 17 | 256 | 74 | 2.3125 | 19 | yes | required |
| IQ2_S | 22 | 256 | 82 | 2.5625 | 22 | yes | optional (policy: required) |
| IQ3_XXS | 18 | 256 | 98 | 3.0625 | 10 | yes | optional (policy: required) |
| IQ3_S | 21 | 256 | 110 | 3.4375 | 21 | yes | optional |
| IQ4_NL | 20 | 32 | 18 | 4.5 | 11 | yes | optional |
| IQ4_XS | 23 | 256 | 136 | 4.25 | 12 | yes | optional |
| IQ1_XS (Unsloth fork) | 64 | 256 | 46 | 1.4375 | 33 | yes | required |
| IQ1_XXS (Unsloth fork) | 65 | 256 | 42 | 1.3125 | 34 | yes | required |
| IQ1_XXXS (Unsloth fork) | 66 | 256 | 38 | 1.1875 | 35 | yes | required |

Float types F32, F16 and BF16 are read and written too. They are not block
types, so the install gate does not list them (`executable` is false for them
in the quantizer's type list) and they are transcoded to BF16 at install.

"Required" is ggml's hard rule: its encoder asserts without an importance
matrix. "Policy" is llama.cpp's stricter rule, which TurboSpark follows: these
types are refused without one on every tensor except `token_embd` and
`output.weight`.

A tensor row has to be a whole number of blocks. The quantizer writes a row that
does not tile as `fallback` (default F16) instead.

### Not supported, by design

- **Q8_1 and Q8_K** are ggml's activation formats. They never appear in a model
  file, so they are named in errors and refused, not implemented.
- **Any other ggml type id** is refused by name at parse time, with the type
  and tensor in the message. A guessed block size would silently misalign every
  tensor after the first one of that type, so a type is added only with a size
  read from ggml itself.

### How each type was checked

- **Decoders** (`crates/compute`): bit-exact against ggml's `to_float`. Q1_0,
  TQ1_0, TQ2_0 and NVFP4 have checked-in goldens
  (`crates/compute/tests/fixtures/ggml/`). The Unsloth types are held to a build
  of the fork itself.
- **Kernels** (`crates/gpu`): Metal parity per type against the CPU decoders,
  for the resident GEMV, the embedding lookup and both routed phases, plus
  mutation checks (a changed shift, offset, table or stride must fail).
- **Whole models** (`crates/runtime`): a synthetic install of each type is run
  and compared with the same weights decoded and stored as Q8_0. This is how a
  kernel layout bug that still produces finite logits gets caught.
- **Quantizers** (`crates/repack`): byte-identical to ggml's own output on an
  adversarial input. Rules ggml shows but does not state were measured, not
  assumed: Q2_0, TQ1_0 and TQ2_0 round ties away from zero, a zero Q1_0 element
  is a set bit, and the NVFP4 scale search takes the higher code on a tie and
  jumps from 240 to 448 once `amax / 6` reaches 248.

## Which models run

A GGUF is matched by its `general.architecture`. The registry is
`crates/repack/src/arch_registry.rs` (`SUPPORTED_GGUF`), which is the
authoritative list. At the time of writing:

| `general.architecture` | Family |
|---|---|
| `gemma4` | Gemma 4 |
| `llama` | Llama flow (Mixtral-style MoE, and families that share it) |
| `granite` | Llama flow |
| `qwen35`, `qwen35moe` | Qwen 3.5 dense, Qwen 3.5/3.6 GDN MoE |
| `qwen3`, `qwen3moe`, `qwen3vl`, `qwen3vlmoe` | Qwen 3 dense, MoE and VL text trunks |
| `qwen2` | Qwen 2 dense |
| `qwen4exp` | Qwen 3.8 Flash-Next |
| `gpt-oss` | gpt-oss |
| `spark2_5` | Spark 2.5 |
| `minimax-m2` | MiniMax-M2 |
| `deepseek2` | DeepSeek 2 |
| `lfm3` | LFM3 |
| `muse-glimmer` | Muse Glimmer |

Inkling 2 (`inkling2_loop`, `inkling2`) is not in `SUPPORTED_GGUF`: the
generic walk refuses it by name. It installs through its own intake,
`write_inkling2_gguf_install` (`crates/repack/src/inkling2_gguf.rs`), because
its experts need the Inkling mode of the routed pairs and a per-expert outer
scale record (`docs/INKLING2_PHASE0.md`, GGUF intake). It is reached from
`turbospark model pull-local <file.gguf> --tokenizer-from DIR`, from
`pull --repo` (the probe reports an Inkling GGUF as runnable and the install
downloads the file whole), and from the app's Discover > Install from Files.
A GGUF carries no `tokenizer.json`, so the tokenizer comes from
`thinkingmachines/Inkling-2-Mini` unless another source is named.

Match is by exact string, never by prefix: `qwen35` and `qwen35moe` are
different families and a prefix match would run a dense file through a router.
An architecture the registry only recognizes is refused at every call site.
Vision towers are not in a GGUF (llama.cpp keeps them in a separate `mmproj`
file), so a VL GGUF installs its text trunk.

The quantizer refuses a split GGUF (`-0000N-of-0000M.gguf`) and discovery skips
one. Merge it first with `llama-gguf-split`.

## Unsloth Dynamic (UD) files

Unsloth publishes GGUFs named like `UD-Q2_K_XL`, `UD-Q3_K_M`, `UD-IQ1_S` and
`UD-IQ2_XXS`. "Dynamic" means the type is chosen per tensor and varies layer by
layer, instead of one type for a whole role. What was measured here:

- In the Qwen3.8-27B UD-IQ1_S header, `ffn_down` alone mixes IQ1_S (46 layers),
  IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S and Q2_K, and `attn_v` runs from IQ2_XS to
  IQ4_XS. No role-level rule reproduces that.
- The UD-IQ1_S files of Qwen3.8-27B, Qwen3.8-Flash-Next and
  DeepSeek-V4.1-Flash-1017 were header-checked. They use standard ggml types
  and do not use the fork's IQ1_XS family. Q2_0 and MXFP4 appear in the
  DeepSeek and Flash-Next files.
- `unsloth/gemma-4-26B-A4B-it-GGUF` `UD-Q3_K_M` is a pinned catalog install. It
  is mixed along two axes: IQ3_XXS gate/up over IQ4_NL down inside one expert,
  and a layer 29 that is IQ4_XS over Q8_0. It installs by streaming one layer at
  a time.
- A real Unsloth Qwen3-0.6B UD-IQ2_XXS keeps `token_embd` at Q5_K. That file
  loads and generates here once the Q5_K embedding lookup exists. At that
  quant its output is degraded in llama.cpp too, so no quality claim is made.

### What TurboSpark does and does not do with UD

- **It runs them.** Every type a UD file uses, in every role, has a kernel.
- **It can follow their type map.** `--recipe-from <ref.gguf>` reads the
  per-tensor types out of a published file and applies them to your own float
  copy. It copies the type assignment only. The values come from your source and
  your imatrix.
- **It does not reproduce UD.** Unsloth publishes the names and benchmark
  results of its methods, not the per-tensor recipe or the imatrix data, so no
  `ud-*` preset exists here and none should be invented. Following a mix gives
  the same types, not the same file.
- **It matches `llama-quantize`, not Unsloth's bytes.** On Qwen3-0.6B, 310 of
  310 tensors are byte-identical to Homebrew `llama-quantize` for the UD-Q2_K_XL
  and UD-IQ2_XXS type maps. Against Unsloth's published files fewer tensors are
  identical (126 and 276 of 310), and `llama-quantize` on the same machine gives
  the same numbers, so the gap is their build or llama.cpp version. KL
  divergence against BF16 agrees with Unsloth's files to within error on
  0.6B UD-Q2_K_XL (0.637 against 0.638), 0.6B UD-IQ2_XXS (2.045), and 1.7B
  UD-IQ1_S (1.934) and UD-IQ1_M (1.645). Details and the IQ1 tie-order note are
  in [QUANTIZE.md](QUANTIZE.md).

### The fork types (ids 64 to 66)

Unsloth's llama.cpp fork (branch `iq1-narrow-grids-b10775`) adds IQ1_XS,
IQ1_XXS and IQ1_XXXS. Each is IQ1_S's 8-dimensional ternary codebook behind a
narrower index (10, 9 and 8 bits instead of 11), so a block is 46, 42 and 38
bytes per 256 weights. The fork keeps ggml's reserved ids 42 to 63 free and
starts its own types at 64, so a stock llama.cpp refuses these files instead of
misreading them. TurboSpark reads, runs and writes them. A file that uses them
runs in TurboSpark and in the fork only, which is why the quantizer's type list
carries `readableByStockLlamaCpp` (false for these three) for a UI to warn on.

The grid tables are generated from the fork's `ggml-common.h` by
`scripts/gen_iq1_narrow_grids.py`, and both the decoder and the quantizer are
held to a build of the fork.

## Using it

```sh
# probe a file header-only (types, shares, whether it would run)
turbospark-model probe <ref or file>

# install a local GGUF through the same walk a pull runs
cargo run --release -p turbospark-repack --example gguf_local_install -- \
    <model.gguf> <new install dir> <sidecar dir> [model id]

# write a GGUF from a float one, following a published mix
cargo run -p turbospark-repack --release --example gguf_quantize -- \
    --recipe-from Qwen3.8-27B-UD-IQ1_S.gguf --imatrix imatrix.gguf \
    qwen3.8-27b-f16.gguf out.gguf
```

From Swift, `GGUFQuantizer` wraps the quantizer (`supportedTypes()` lists the 31
writable types with `executable` and `readableByStockLlamaCpp`). See
[SWIFT_BINDINGS.md](SWIFT_BINDINGS.md).

## Known limits

- **A K-quant `ssm_out` is re-encoded, not copied.** Qwen's gated-DeltaNet
  families (`qwen35`, `qwen35moe`) de-interleave the V heads on the COLUMNS of
  `linear_attn.out_proj` (GGUF `ssm_out`). A standard llama.cpp Q4_K_M stores
  that tensor at Q4_K or Q6_K, whose 256-element block is wider than a
  128-column head, so the permutation cannot be a byte move. The walk
  dequantizes that one tensor, restores the order in f32 and stores Q8_0
  (`gdn_out_proj_needs_q8_0_requant`), counted in `lossy_narrowing`, and the
  probe says so as a warning. It is not byte-identical to the file, and the
  install is a little larger (about 0.8 GB on a 27B). Checked on a real Q4_K_M
  of `Qwen3.8-27B-pi-adapted` (coherent chat decode); Q3_K, Q5_K, Q6_K and IQ
  `ssm_out` take the same path with no real-file run. A Q8_0 `ssm_out` is a
  pure byte move and is untouched.
- **The multi-token-prediction head is dropped, and speculation is off for a
  K-quant trunk.** A GGUF that declares `nextn_predict_layers` carries the head
  as one more `blk.N.` block; the walk excludes it from the trunk. Ingesting it
  would not enable speculation on its own: the batched verify has a GEMM only
  for INT4-affine and the 1-/2-bit affine pair, so a Q4_K/Q6_K/Q8_0 trunk
  decodes normally with speculation off. See [MTP.md](MTP.md).
- **A GGUF with no tokenizer needs `--sidecar-repo`.** The probe reports
  RUNNABLE only with a repository that supplies `tokenizer.json` and the chat
  template (for example the matching MLX repo).

- **No real-file comparison for the newer types.** No real Q4_0, Q1_0, TQ, NVFP4
  or fork-type file has been run side by side with llama.cpp. The evidence is
  ggml-derived goldens and Q8_0 oracles on synthetic installs. Those catch
  layout, sign, scale and routing errors. They do not catch a flaw in how a
  real model uses a type.
- **No performance measurement** of the new kernels. They are element-addressed
  and correct first. Do not quote a speed for them.
- **Batched prefill** is gated on the affine expert layout, so the new routed
  layouts install on the per-token path.
- **Q2_0 routed phase 2** serves a `top_k` of 10 (hand-written kernel) or at most
  8 (generated kernel). Other values are refused at open.
- **Quality of UD mixes at runtime** has not been compared against llama.cpp
  perplexity on a real UD file here. The KL and perplexity numbers above compare
  this crate's quantizer with Unsloth's files, not this engine's forward pass.

## Where things live

| What | Where |
|---|---|
| Type names and block sizes | `crates/repack/src/gguf_header/ggml.rs` |
| Install gate and dtype tags | `crates/model-io/src/manifest/quant.rs`, `crates/repack/src/resident_writer.rs` |
| CPU decoders | `crates/compute/src/quant_gguf/`, `quant_gguf_iq.rs`, `quant_gguf_iq1_narrow.rs` |
| Resident kernels | `crates/gpu/src/shaders/dequant_*.metal` |
| Routed kernels | `crates/gpu/src/shaders/moe_gguf.metal`, `moe_gguf_ext.metal` |
| Quantizer | `crates/repack/src/gguf_quantize/` |
| Architecture registry | `crates/repack/src/arch_registry.rs` |
| Golden fixtures | `crates/compute/tests/fixtures/ggml/`, `crates/repack/tests/fixtures/gguf_quantize/` |
| Fixture generators | `scripts/ggml_dequant_fixture.c`, `scripts/ggml_quantize_fixture.c` |
