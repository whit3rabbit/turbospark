---
title: "Quantization Formats and Kernel Coverage"
description: "The twelve quantization layouts this engine executes (four MLX affine widths, eight GGUF block types), their block structure, and the per-role kernel matrix that decides which type can run where."
diataxisType: "reference"
---

This engine executes twelve quantization layouts across two container
families. The MLX **affine** family stores a packed plane beside per-group
scale and bias companions (`w = q * scale + bias`); the **GGUF block** family
stores self-contained interleaved blocks whose scales sit inside the block.
A thirteenth GGUF type, Q4_0, is parsed and installable but has no kernel of
any kind and is refused.

What a type can do is decided per ROLE, not per type: a resident matrix GEMV,
an embedding-table lookup, a routed-expert phase 1 (gate/up) and a routed
phase 2 (down projection) are four separate kernels, and a type may have any
subset of them. The reference decoders and constants live in
`crates/compute` (`quant.rs`, `quant_1bit.rs`, `quant_2bit.rs`,
`quant_gguf/`, `quant_gguf_iq.rs`, `quant_gguf_mxfp4.rs`); the executable-set
gate lives in `crates/model-io/src/manifest/quant.rs`. Signatures are on the
Rust API pages listed at the bottom.

## Format index

| Family | Type | Width | Block / group | Block bytes | Roles covered |
|---|---|---|---|---|---|
| MLX affine | INT4 | 4-bit | group 64 | varies | GEMV, embed, routed 1+2 |
| MLX affine | INT8 | 8-bit | group 64 | varies | GEMV |
| MLX affine | 1-bit | 1-bit | group 128 | varies | GEMV, embed |
| MLX affine | 2-bit | 2-bit | group 128 | varies | GEMV, embed |
| GGUF | Q8_0 | 8-bit | 32 elements | 34 | GEMV, embed, routed 1+2 |
| GGUF | Q4_K | 4-bit | 256 elements | 144 | GEMV, embed, routed 1+2 |
| GGUF | Q5_K | 5-bit | 256 elements | 176 | GEMV |
| GGUF | Q6_K | 6-bit | 256 elements | 210 | GEMV, embed, routed 2 |
| GGUF | IQ3_XXS | 3.06-bit | 256 elements | 98 | GEMV, routed 1 |
| GGUF | IQ4_NL | 4-bit | 32 elements | 18 | GEMV, routed 2 |
| GGUF | IQ4_XS | 4-bit | 256 elements | 136 | GEMV, routed 1 |
| GGUF | MXFP4 | 4-bit | 32 elements | 17 | routed 1+2 |
| GGUF | Q4_0 | 4-bit | 32 elements | 18 | none (refused) |

## MLX affine family

Three planar regions per tensor: a packed plane, a per-group `scales` plane,
a per-group `biases` plane. An element decodes as
`w = q * scale + bias` with unsigned `q`. The group size, the companion
dtype and the bit width travel together and are validated as ONE shape, not
three independent axes (`validate_quant` in
`crates/model-io/src/manifest/quant.rs`).

### INT4 affine (group 64, BF16 companions)

- Source: `crates/compute/src/quant.rs`.
- Layout: `n / 2` packed bytes, two nibbles per byte, low nibble = even
  index, high nibble = odd index. One BF16 scale and one BF16 bias per
  group of 64 elements (`GROUP_SIZE`). BF16 companions are carried as raw
  `u16` bit patterns (top 16 bits of the FP32 word).
- Quantizer rule: min/max affine, `scale = (max - min) / 15`,
  `bias = min`, `q in [0, 15]`, rounded through BF16 before the quantize
  pass.
- Roles: resident GEMV, embedding lookup (with optional post-lookup scale
  such as Gemma's `sqrt(hidden)`), and the routed-expert pair.

### INT8 affine (group 64, BF16 companions)

- Source: `crates/compute/src/quant.rs`.
- Layout: one unsigned byte per element, one BF16 scale and one BF16 bias
  per group of 64.
- Quantizer rule: min/max affine with `q in [0, 255]`.
- Roles: resident GEMV (attention, router and shared-expert matrices). No
  GPU embedding lookup exists and the manifest's `embedding` slot admits
  4-bit affine only, so an INT8 embedding table is refused rather than
  decoded.

### 1-bit affine (group 128, FP16 companions)

- Source: `crates/compute/src/quant_1bit.rs`.
- Layout: `n / 8` packed bytes; element `i` is bit `i % 8` of byte
  `i / 8`, least significant bit first. One FP16 scale and bias per group.
  The group size is a PARAMETER (`BONSAI_GROUP_SIZE = 128` is what the real
  checkpoint declares in `config.json`, not a constant of the format).
- Companions are FP16, not BF16: the two are the same width and share no
  exponent field, so a misread passes every length check and decodes
  scales as ~1e-16. This is the axis the one-shape validation exists to
  catch.
- Real checkpoint property (measured, never assumed): every group has
  `bias == -scale / 2`, giving symmetric binary weights
  (`is_symmetric` measures it; `dequant_int1_gemv_symmetric` is the
  `+/-1` accumulate-then-scale form and panics on a row that does not
  qualify).
- Roles: resident GEMV and embedding lookup. No routed-expert pair: the
  one real 1-bit checkpoint is dense.

### 2-bit affine (group 128, FP16 companions)

- Source: `crates/compute/src/quant_2bit.rs`.
- Layout: `n / 4` packed bytes; element `i` is the 2-bit field at bit
  `2 * (i % 4)` of byte `i / 4`, least significant field first
  (`INT2_ELEMENTS_PER_BYTE = 4`). One FP16 scale and bias per group
  (`TERNARY_GROUP_SIZE = 128`, a checkpoint declaration).
- Real checkpoint property (measured, never assumed): every group has
  `bias == -scale` and level 3 never occurs, so the three levels in use
  are `-scale`, `0`, `+scale` (a ternary grid in a 2-bit container).
  `is_ternary_symmetric` and `uses_fourth_level` measure both, and they
  are independent: the container permits a fourth level at `+2s`.
- Roles: resident GEMV and embedding lookup. No routed-expert pair.

### The one-shape rule

`validate_quant` accepts exactly three affine shapes, each as a single
conjunction:

| Shape | Bits | Companions | Group |
|---|---|---|---|
| INT4/INT8 affine | 4 or 8 | BF16 | 64 |
| 1-bit affine | 1 | FP16 | 128 |
| 2-bit affine | 2 | FP16 | 128 |

Widening one axis (1-bit at group 64, 4-bit with FP16 companions) accepts
combinations no kernel implements. The affine arm additionally admits
`weight_bits == 2` with BF16 at group 64 on the `routedExpert` slot alone,
for the DeepSeek-V4-Flash dynamic-quant checkpoint; that shape and the
2-bit FP16 shape above share nothing but their width. Both sub-4-bit shapes
are accepted on all five slots because both published checkpoints are dense
and their empty slots are defaulted copies of the attention slot.

## GGUF block family

A GGUF block is self-contained and interleaved: the scale sits immediately
before the weights it scales, so a row is one contiguous byte run with no
companion arrays. Source: `crates/compute/src/quant_gguf/` (K-quants and
Q8_0), `quant_gguf_iq.rs` (IQ codebook types), `quant_gguf_mxfp4.rs`.
Block constants are cross-checked against the GGUF header reader in
`crates/repack` rather than imported (compute must not depend on repack).

### Q8_0

- 32 elements in 34 bytes: one f16 scale `d`, then 32 signed bytes.
- Symmetric: `w = q * d`, no bias, zero exactly representable.
- Roles: resident GEMV, embedding lookup, routed phase 1 and phase 2.

### Q4_K

- 256 elements in 144 bytes: two f16 super-scales (`d`, `dmin`), 12 bytes
  packing eight sub-blocks' 6-bit scales and 6-bit mins (sub-blocks 4..8
  split across two bytes), then 128 nibble-packed bytes. Eight sub-blocks
  of 32 elements.
- Asymmetric with unsigned quants: `w = d * sc * q - dmin * m`.
- Roles: resident GEMV, embedding lookup, routed phase 1 and phase 2.

### Q5_K

- 256 elements in 176 bytes: Q4_K's 144 bytes plus a separate 32-byte
  high-bit run `qh`, indexed by the element's position within a sub-block
  at a bit that advances with the 64-element group. Reuses Q4_K's 6-bit
  scale/min unpacker byte for byte.
- Roles: resident GEMV only (its real use is Mixtral 8x7B's
  `attn_output`). No embedding lookup, no routed phases.

### Q6_K

- 256 elements in 210 bytes: one f16 super-scale `d`, sixteen signed int8
  sub-block scales stored as plain bytes, a `ql` run of four low bits and
  a `qh` run of two high bits per element. Sixteen sub-blocks of 16
  elements.
- Symmetric with a fixed offset: `(q - 32) * d * sc`; there is no
  per-sub-block min.
- Roles: resident GEMV, embedding lookup, routed phase 2. No routed phase
  1 (its real use is Qwen 3.6 Q4_K_M's single `output.weight` tensor).

### IQ3_XXS

- 256 elements in 98 bytes: one f16 scale `d`, 64 grid-index bytes, then
  eight u32 words each packing four 7-bit sign indices and one 4-bit
  sub-block scale.
- Codebook type: `w = db * grid[q] * sign` with
  `db = d * (0.5 + (aux >> 28)) * 0.5`. The grid holds magnitudes only;
  the eighth sign bit is a PARITY bit over the seven stored ones
  (`iq3xxs_signs`).
- Roles: resident GEMV, routed phase 1. No phase 2.

### IQ4_NL

- 32 elements in 18 bytes: one f16 scale `d`, then 16 nibble-packed
  indices into the fixed 16-entry `IQ4NL_VALUES` table.
- Codebook type: `w = d * table[q]`. The table is non-linear; an affine
  `q - 8` read gets the sign right and the tails wrong. The two nibbles of
  a byte serve elements SIXTEEN apart (low nibbles fill `0..16`, high
  nibbles `16..32`).
- Roles: resident GEMV, routed phase 2. No phase 1.

### IQ4_XS

- 256 elements in 136 bytes: one f16 `d`, a u16 of high scale bits, four
  bytes of low scale nibbles, then 128 nibble-packed indices. Eight
  sub-blocks of 32 elements.
- IQ4_NL's table under a two-level scale: `dl = d * (ls - 32)` with `ls`
  assembled from a 4-bit nibble of `scales_l` and 2 bits of `scales_h`.
  The scale is BIASED by 32 and can be negative; dropping the bias mirrors
  whole sub-blocks.
- Roles: resident GEMV, routed phase 1. No phase 2.

### MXFP4

- 32 elements in 17 bytes: one E8M0 exponent byte (OCP Microscaling), then
  16 nibble-packed indices into the fixed `MXFP4_VALUES` codebook
  (`{0, 1, 2, 3, 4, 6, 8, 12}` and their negatives; index 8 is a second,
  POSITIVE zero).
- Scale is a bare exponent, always an exact power of two:
  `mxfp4_scale(e) = 2^(e - 128)` for `e >= 2`, with `e = 0` and `e = 1`
  built from a shifted subnormal pattern. Nibbles are split-half: byte `j`
  serves elements `j` and `j + 16`.
- Roles: routed phase 1 and phase 2 ONLY. No resident GEMV of any kind:
  the one real checkpoint carrying it (gpt-oss) puts MXFP4 in
  `ffn_{gate,up,down}_exps` and keeps attention, `token_embd` and `output`
  at Q8_0.

### Q4_0

- Parsed by the GGUF reader and installable, and the ONLY tag with no
  kernel of any kind in this port. A slot declaring it is refused with
  `GGUF block type q4_0 has no kernel in this port`.

## Per-role support matrix

| Type | Matrix GEMV | Embedding lookup | Routed phase 1 (gate/up) | Routed phase 2 (down) |
|---|---|---|---|---|
| INT4 affine | yes | yes | yes | yes |
| INT8 affine | yes | no | no | no |
| 1-bit affine | yes | yes | no | no |
| 2-bit affine | yes | yes | no | no |
| Q8_0 | yes | yes | yes | yes |
| Q4_K | yes | yes | yes | yes |
| Q5_K | yes | no | no | no |
| Q6_K | yes | yes | no | yes |
| IQ3_XXS | yes | no | yes | no |
| IQ4_NL | yes | no | no | yes |
| IQ4_XS | yes | no | yes | no |
| MXFP4 | no | no | yes | yes |
| Q4_0 | no | no | no | no |

Phase 1 computes `acts[slot * f_dim + f] = activation(gate_f(x)) * up_f(x)`
per routed slot; phase 2 maps the activations back through the down
projection weighted by the routing weights and adds the residual.

Each type's coverage is what its real files ask for, no more. A type used in
a role it has no kernel for passes the executable-set gate and fails at the
dispatch site, by tensor name. Examples of the deliberate asymmetry: Q6_K
has phase 2 but no phase 1; IQ3_XXS and IQ4_XS have phase 1 but no phase 2;
IQ4_NL the reverse; MXFP4 has both routed phases and no resident GEMV.

## Per-tensor mixing and phase resolution

- The quantization type is resolved PER TENSOR from the resident index's
  dtype tag, never from the checkpoint's dominant type. A mixed checkpoint
  executes exactly as published: Qwen 3.6's Q4_K_M carries exactly one Q6_K
  tensor (`output.weight`) beside Q4_K; gpt-oss carries MXFP4 experts with
  Q8_0 attention, embedding and output; the mixed sub-4-bit install puts
  IQ3_XXS + IQ4_NL experts on 29 layers and IQ4_XS + Q8_0 on the
  thirtieth.
- For ROUTED EXPERTS the type is resolved per LAYER and per PHASE
  (gate/up versus down), so one layer can mix types across its own three
  expert sub-tensors. The per-layer expert stride is likewise per layer,
  which is what lets a mixed-width install pad each layer to its own blob
  size instead of the model-wide maximum.
- A manifest slot declares its block types with `ggmlType` (the dominant
  one) plus an optional `ggmlTypes` array; EVERY declared member is
  checked, because checking only the dominant one would pass an install on
  the strength of its majority and fail at a dispatch thirty layers in.

## Validation and refusal behaviour

Two gates, twins and not copies:

1. **Manifest `ggmlType` validation**
   (`crates/model-io/src/manifest/quant.rs::validate_quant`). Reads the
   manifest's per-SLOT types and asks whether the type has the kernels its
   slot needs. Accepts the three affine shapes above (per-slot bit lists:
   embedding 4, attention 4, router 8, sharedExpert 4 or 8, routedExpert 2
   or 4, each plus the 1-bit and 2-bit conjunctions on every slot) and any
   GGUF slot whose declared types are all in `EXECUTABLE_GGUF_TYPES`:
   `q8_0, q4_k, q5_k, q6_k, iq3_xxs, iq4_nl, iq4_xs, mxfp4`. This list
   INCLUDES `mxfp4`. Affine refusals name the observed
   `(bits, scale/bias dtype, group)` triple, because the FP16-versus-BF16
   failure is invisible to every length check; GGUF refusals name the
   offending block type.

2. **Resident-index dtype-tag backstop**
   (`crates/runtime`, `EXECUTABLE_GGUF_DTYPES` over `GGUF_BLOCK_DTYPES`).
   Reads the dtype TAG of each resident tensor in `model_weights.bin` at
   open and asks whether that tensor can be dispatched. It believes the
   bytes over the manifest, catching a hand-edited or hand-made install.
   This list deliberately OMITS `mxfp4` (tag 14): MXFP4 is executable as a
   routed slot type but has no resident GEMV, so a resident tensor tagged
   MXFP4 is something no walk has ever written and is refused by name. It
   also omits `q4_0` (tag 9), which is in the nine-tag block list purely so
   the reader's and writer's lists stay structurally parallel.

So an install with MXFP4 attention passes the manifest gate and is stopped
by the backstop. In every refusal case the tensor is NAMED (or the tag
number, at open) rather than decoded as another type: a dtype tag this
crate has no reader for is refused with its number in the message, where a
permissive default would dispatch it as something else. A type in a role
with no kernel (Q6_K phase 1, an IQ embedding table, a 1-bit routed expert)
is refused at the dispatch site with the tensor named, which is a worse
error message than the gate's but never a wrong answer.

## See also

- [Model families](/reference/model-families/): which decode flow consumes
  these types, and the per-family installs.
- [The `.gturbo` format](/reference/gturbo-format/): how the manifest's
  quant slots and the resident index's dtype tags are written.
- [Rust API: core primitives](/reference/rust-core-primitives/): the
  `turbospark-compute` reference kernel signatures.
- [Rust API: model data path](/reference/rust-model-data-path/): the
  `turbospark-model-io` manifest validation API.
